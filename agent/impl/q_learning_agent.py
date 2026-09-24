import math
import os
import random
from collections import defaultdict, deque
from datetime import datetime

from action.impl.restart_action import RestartAction
from agent.agent import Agent
from config.settings import settings
from exceptions import NoActionsException
from observation.observer import Observer
from state.impl.action_execute_failed_state import ActionExecuteFailedState
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.impl.out_of_domain_state import OutOfDomainState
from state.impl.same_url_state import SameUrlState
from state.impl.tag_sequence_state import TagSequenceState
from utils import safe_console_text
from utils import read_reward_clip_abs, clip_reward_value
from fairness import is_fair_mode


class QLearningAgent(Agent):
    def __init__(self, params):
        self.AGENT_TYPE = os.environ.get("WEBTEST_QLEARNING_AGENT_TYPE", params["agent_type"])
        self.state_mode = os.environ.get("WEBTEST_QLEARNING_STATE_MODE", "tagseq").strip().lower()
        self.reward_mode = os.environ.get("WEBTEST_QLEARNING_REWARD_MODE", "").strip().lower()
        allow_webexplor_tagseq = (
            os.environ.get("WEBTEST_ALLOW_TAG_SEQUENCE_IN_FAIR_WEBEXPLOR", "").strip().lower()
            in ("1", "true", "yes", "on")
            and self.state_mode == "tagseq"
        )
        if is_fair_mode() and self.state_mode != "actionset" and not allow_webexplor_tagseq:
            self.state_mode = "actionset"
        self.ALPHA = params["alpha"]
        self.GAMMA = params["gamma"]
        self.EPSILON = params["epsilon"]
        self.INITIAL_Q_VALUE = params["initial_q_value"]
        self.R_REWARD = params["r_reward"]
        self.R_PENALTY = params["r_penalty"]
        self.MAX_SIM_LINE = params["max_sim_line"]
        self.state_repr_list = list()
        self.action_list = list()
        self.q_table = dict()
        self.action_count = dict()  # key: index(action), value: visited times
        self.url_count = dict()  # key: string(url), value: visited times
        self.state_count = dict()
        self.trans_count = defaultdict(int)
        self.previous_state = None
        self.previous_action = None
        self.previous_web_state = None
        self.previous_action_obj = None
        self.stop_update = False
        self._latest_state_max_sim = 0.0
        self._latest_state_is_new = False
        self._obs_step_counter = 0
        self._obs_state_history = []
        self.reward_clip_abs = read_reward_clip_abs(5.0)
        self.start_time = datetime.now()
        self.state_action_count = defaultdict(int)
        self.initial_epsilon = float(self.EPSILON)
        self.max_transitions = self._read_env_positive_int("WEBTEST_MAX_TRANSITIONS")
        self.webqt_mode = self.reward_mode == "webqt"
        self.policy_mode = os.environ.get("WEBTEST_QLEARNING_POLICY_MODE", "epsilon").strip().lower()
        self.gumbel_tau = max(
            1e-6,
            float(os.environ.get("WEBTEST_QLEARNING_GUMBEL_TAU", "1.0")),
        )
        self.enable_dfa_guidance = (
            os.environ.get("WEBTEST_QLEARNING_ENABLE_DFA", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        raw_dfa_stuck_steps = os.environ.get("WEBTEST_QLEARNING_DFA_STUCK_STEPS", "").strip()
        try:
            self.dfa_stuck_steps = (
                max(1, int(raw_dfa_stuck_steps)) if raw_dfa_stuck_steps else None
            )
        except ValueError:
            self.dfa_stuck_steps = None
        self.dfa_stuck_seconds = max(
            1.0,
            float(
                os.environ.get(
                    "WEBTEST_QLEARNING_DFA_STUCK_SECONDS",
                    os.environ.get("WEBEXPLOR_DFA_STUCK_SECONDS", "120"),
                )
            ),
        )
        self.dfa_initial_state = None
        self.dfa_edges = {}
        self.dfa_adj = defaultdict(dict)
        self.guided_action_queue = deque()
        self.steps_since_new_state = 0
        self.last_new_state_time = datetime.now()
        self.last_dfa_guidance_time = None
        self._last_guided_target = None
        self.webqt_w_loc = float(os.environ.get("WEBQT_W_LOC", "10.0"))
        self.webqt_w_attention = float(os.environ.get("WEBQT_W_ATTENTION", "50.0"))
        self.webqt_w_freq = float(os.environ.get("WEBQT_W_FREQ", "5.0"))
        self.webqt_w_explore = float(os.environ.get("WEBQT_W_EXPLORE", "5.0"))
        self.webqt_min_epsilon = float(os.environ.get("WEBQT_MIN_EPSILON", "0.1"))
        self.webqt_decay_epsilon = os.environ.get("WEBQT_DECAY_EPSILON", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.webqt_stagnation_window = max(
            1, int(os.environ.get("WEBQT_STAGNATION_WINDOW", "50"))
        )
        self.webqt_no_new_state_streak = 0
        if self.webqt_mode:
            self.AGENT_TYPE = "W"
            self.state_mode = "actionset"
            self.MAX_SIM_LINE = float(
                os.environ.get("WEBQT_STATE_SIM_THRESHOLD", "0.85")
            )
        _mg_cap = None if os.environ.get("OBSERVATION_MG_FULL_HISTORY", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ) else 50
        self.observer = Observer(
            agent_name="WebQT-QLearning" if self.webqt_mode else "QLearningAgent",
            log_dir="observation_logs",
            marginal_gain_sample_cap=_mg_cap,
        )

    @staticmethod
    def _read_env_positive_int(name: str):
        raw = os.environ.get(name, "").strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        return value if value > 0 else None

    def state_abstraction(self, state):
        actions = state.get_action_list()
        for a in actions:
            self._register_action(a)

        action_index_set = set()
        for a in actions:
            action_index_set.add(self.get_action_index(a))
        action_index_list = list(action_index_set)
        action_index_list.sort()
        action_index_list_str = [str(x) for x in action_index_list]
        state_representation = ','.join(action_index_list_str)
        return state_representation

    def _register_action(self, action):
        if isinstance(action, RestartAction):
            if len(self.action_list) == 0:
                self.action_list.append(action)
                self.action_count[0] = 0
            return 0
        if action not in self.action_list:
            self.action_list.append(action)
            self.action_count[self.action_list.index(action)] = 0
        action_index = self.action_list.index(action)
        self.action_count.setdefault(action_index, 0)
        return action_index

    @staticmethod
    def _action_signature(action):
        if action is None:
            return ("none",)
        return (
            type(action).__name__,
            getattr(getattr(action, "locator", None), "value", ""),
            getattr(action, "location", ""),
            getattr(action, "text", ""),
            getattr(action, "action_type", ""),
            getattr(action, "addition_info", ""),
        )

    @staticmethod
    def _levenshtein_distance(lhs: str, rhs: str) -> int:
        if lhs == rhs:
            return 0
        if not lhs:
            return len(rhs)
        if not rhs:
            return len(lhs)
        previous = list(range(len(rhs) + 1))
        for i, ch_l in enumerate(lhs, start=1):
            current = [i]
            for j, ch_r in enumerate(rhs, start=1):
                insert_cost = current[j - 1] + 1
                delete_cost = previous[j] + 1
                replace_cost = previous[j - 1] + (0 if ch_l == ch_r else 1)
                current.append(min(insert_cost, delete_cost, replace_cost))
            previous = current
        return previous[-1]

    def _compute_webqt_locality(self, previous_action_obj, next_action) -> float:
        if previous_action_obj is None or next_action is None or isinstance(next_action, RestartAction):
            return 0.0

        prev_rect = getattr(previous_action_obj, "rect", None)
        next_rect = getattr(next_action, "rect", None)
        if isinstance(prev_rect, dict) and isinstance(next_rect, dict):
            prev_w = max(1.0, float(prev_rect.get("width", 1.0)))
            prev_h = max(1.0, float(prev_rect.get("height", 1.0)))
            next_w = max(1.0, float(next_rect.get("width", 1.0)))
            next_h = max(1.0, float(next_rect.get("height", 1.0)))
            prev_cx = float(prev_rect.get("x", 0.0)) + prev_w / 2.0
            prev_cy = float(prev_rect.get("y", 0.0)) + prev_h / 2.0
            next_cx = float(next_rect.get("x", 0.0)) + next_w / 2.0
            next_cy = float(next_rect.get("y", 0.0)) + next_h / 2.0
            dist = math.dist((prev_cx, prev_cy), (next_cx, next_cy))
            prev_size = prev_w + prev_h
            next_size = next_w + next_h
        else:
            prev_loc = getattr(previous_action_obj, "location", "")
            next_loc = getattr(next_action, "location", "")
            prev_text = getattr(previous_action_obj, "text", "")
            next_text = getattr(next_action, "text", "")
            dist = float(self._levenshtein_distance(prev_loc, next_loc))
            prev_size = max(1.0, float(len(prev_text) + len(prev_loc)))
            next_size = max(1.0, float(len(next_text) + len(next_loc)))

        dist = max(1.0, dist)
        locality = math.sqrt(prev_size * next_size) / dist
        return min(1.0, float(locality))

    def _compute_webqt_attention(self, previous_web_state, current_web_state, next_action) -> float:
        if (
            not isinstance(previous_web_state, ActionSetWithExecutionTimesState)
            or not isinstance(current_web_state, ActionSetWithExecutionTimesState)
            or next_action is None
            or isinstance(next_action, RestartAction)
        ):
            return 0.0

        previous_signatures = {
            self._action_signature(action)
            for action in previous_web_state.get_action_list()
        }
        new_actions = [
            action
            for action in current_web_state.get_action_list()
            if self._action_signature(action) not in previous_signatures
        ]
        if not new_actions:
            return 0.0
        next_signature = self._action_signature(next_action)
        if all(self._action_signature(action) != next_signature for action in new_actions):
            return 0.0
        return 1.0 / float(len(new_actions))

    def _compute_webqt_explore(self, state_index: int) -> float:
        state_actions = self.q_table.get(state_index, {})
        total_actions = len(state_actions)
        if total_actions <= 0:
            return 0.0
        unexplored_actions = sum(
            1
            for action_index in state_actions.keys()
            if self.state_action_count[(state_index, action_index)] == 0
        )
        return float(unexplored_actions) / float(total_actions)

    def _current_epsilon(self) -> float:
        if not self.webqt_mode or not self.webqt_decay_epsilon or not self.max_transitions:
            return float(self.EPSILON)
        progress = min(1.0, float(self._obs_step_counter) / float(self.max_transitions))
        decayed = self.initial_epsilon - (
            (self.initial_epsilon - self.webqt_min_epsilon) * progress
        )
        return max(self.webqt_min_epsilon, float(decayed))

    def _pick_webqt_escape_action(self, actions, state_index):
        if len(actions) <= 1:
            return RestartAction(settings.entry_url), 0.0, True

        q_values = {
            action: self.q_table[state_index][self.get_action_index(action)]
            for action in actions
            if not isinstance(action, RestartAction)
        }
        if not q_values:
            return RestartAction(settings.entry_url), 0.0, True

        best_q = max(q_values.values())
        non_best_actions = [action for action, value in q_values.items() if value < best_q]
        candidate_actions = non_best_actions if non_best_actions else list(q_values.keys())
        return random.choice(candidate_actions), float(best_q), False

    def _pick_gumbel_action(self, actions, state_index):
        candidate_actions = [
            action for action in actions if not isinstance(action, RestartAction)
        ]
        if not candidate_actions:
            return RestartAction(settings.entry_url), 0.0, True

        best_action = candidate_actions[0]
        best_score = float("-inf")
        max_q = float("-inf")
        for action in candidate_actions:
            action_index = self.get_action_index(action)
            q_value = float(self.q_table[state_index][action_index])
            max_q = max(max_q, q_value)
            u = min(max(random.random(), 1e-12), 1.0 - 1e-12)
            gumbel_noise = -math.log(-math.log(u))
            # Sample from softmax(Q / tau) via the Gumbel-max trick.
            # Dividing both Q and noise by tau leaves argmax unchanged.
            score = (q_value / self.gumbel_tau) + gumbel_noise
            if score > best_score:
                best_score = score
                best_action = action
        return best_action, max_q, False

    def _transition_count(self, source_state, action_index, target_state):
        key = "{}-{}-{}".format(source_state, action_index, target_state)
        return max(1, int(self.trans_count.get(key, 1)))

    def _transition_curiosity(self, source_state, action_index, target_state):
        return 1.0 / math.sqrt(self._transition_count(source_state, action_index, target_state))

    @staticmethod
    def is_dfa_control_reset(action):
        return isinstance(action, RestartAction) and bool(
            getattr(action, "_webexplor_dfa_control_reset", False)
        )

    @staticmethod
    def _make_dfa_control_reset():
        action = RestartAction(settings.entry_url)
        setattr(action, "_webexplor_dfa_control_reset", True)
        return action

    def _record_dfa_transition(self, previous_state, previous_action, current_state):
        if not self.enable_dfa_guidance:
            return
        if previous_state is None or previous_action is None:
            return
        if previous_state <= 2 or current_state <= 2:
            return
        if previous_state == current_state:
            return
        self.dfa_edges[(previous_state, previous_action)] = current_state
        self.dfa_adj[previous_state][previous_action] = current_state

    def _find_shortest_action_trace(self, target_state):
        if self.dfa_initial_state is None:
            return None
        if self.dfa_initial_state == target_state:
            return []

        queue = deque([(self.dfa_initial_state, [])])
        seen = {self.dfa_initial_state}
        while queue:
            state_index, action_trace = queue.popleft()
            for action_index, next_state in self.dfa_adj.get(state_index, {}).items():
                if next_state in seen:
                    continue
                next_trace = action_trace + [action_index]
                if next_state == target_state:
                    return next_trace
                seen.add(next_state)
                queue.append((next_state, next_trace))
        return None

    def _select_dfa_trace(self):
        best = None
        best_curiosity = float("-inf")
        for (source_state, action_index), target_state in self.dfa_edges.items():
            if source_state <= 2 or target_state <= 2:
                continue
            if source_state == target_state:
                continue
            trace = self._find_shortest_action_trace(source_state)
            if trace is None:
                continue
            curiosity = self._transition_curiosity(source_state, action_index, target_state)
            if curiosity > best_curiosity:
                best_curiosity = curiosity
                best = (trace + [action_index], source_state, action_index, target_state)
        return best

    def _dequeue_guided_action(self, actions):
        while self.guided_action_queue:
            action_index = self.guided_action_queue.popleft()
            if action_index < 0 or action_index >= len(self.action_list):
                continue
            guided_action = self.action_list[action_index]
            if guided_action in actions:
                return guided_action
            self.guided_action_queue.clear()
            now = datetime.now()
            self.last_new_state_time = now
            self.last_dfa_guidance_time = now
            return None
        return None

    def _select_forced_action(self, actions, state_index):
        if not self.enable_dfa_guidance:
            return None
        if state_index > 2 and self.dfa_initial_state is None:
            self.dfa_initial_state = state_index

        guided_action = self._dequeue_guided_action(actions)
        if guided_action is not None:
            action_index = self.get_action_index(guided_action)
            max_q = self.q_table[state_index].get(action_index, 0.0)
            return guided_action, float(max_q), False

        if self._latest_state_is_new:
            self.steps_since_new_state = 0
            self.last_new_state_time = datetime.now()
            return None
        self.steps_since_new_state += 1
        now = datetime.now()
        elapsed_without_new_state = (now - self.last_new_state_time).total_seconds()
        if self.dfa_stuck_steps is not None:
            if self.steps_since_new_state < self.dfa_stuck_steps:
                return None
        else:
            elapsed_since_guidance = (
                float("inf")
                if self.last_dfa_guidance_time is None
                else (now - self.last_dfa_guidance_time).total_seconds()
            )
            if (
                elapsed_without_new_state < self.dfa_stuck_seconds
                or elapsed_since_guidance < self.dfa_stuck_seconds
            ):
                return None

        selected = self._select_dfa_trace()
        if selected is None:
            return None

        trace, source_state, action_index, target_state = selected
        self.guided_action_queue = deque(trace)
        self.steps_since_new_state = 0
        self.last_new_state_time = now
        self.last_dfa_guidance_time = now
        self._last_guided_target = (source_state, action_index, target_state)
        print(
            "[QLearning-DFA] stuck_seconds={:.1f} stuck_steps={} target=({}, {}, {}) replay_len={}".format(
                elapsed_without_new_state,
                self.dfa_stuck_steps if self.dfa_stuck_steps is not None else "time",
                source_state,
                action_index,
                target_state,
                len(trace),
            )
        )
        return self._make_dfa_control_reset(), 0.0, True

    def get_state_index(self, state, html):
        if len(self.state_repr_list) == 0:
            self.state_repr_list.append(OutOfDomainState("111"))
            self.state_repr_list.append(ActionExecuteFailedState("111"))
            self.state_repr_list.append(SameUrlState("111"))
            self.q_table[0] = dict()
            self.q_table[1] = dict()
            self.q_table[2] = dict()
            self.q_table[0][0] = -9999
            self.q_table[1][0] = -99
            self.q_table[2][0] = -99
        if len(self.action_list) == 0:
            self.action_list.append(RestartAction("111"))
            self.action_count[0] = 0
        if isinstance(state, OutOfDomainState):
            self._latest_state_is_new = False
            self._latest_state_max_sim = 1.0
            return 0
        if isinstance(state, ActionExecuteFailedState):
            self._latest_state_is_new = False
            self._latest_state_max_sim = 1.0
            return 1
        if isinstance(state, SameUrlState):
            self._latest_state_is_new = False
            self._latest_state_max_sim = 1.0
            return 2

        if self.AGENT_TYPE == "Q":
            state_instance = self.state_abstraction(state)
            self._latest_state_is_new = False
            if state_instance not in self.state_repr_list:
                self.state_repr_list.append(state_instance)
                s_idx = self.state_repr_list.index(state_instance)
                self._latest_state_is_new = True
                action_value = dict()
                actions = state.get_action_list()
                for action in actions:
                    a_idx = self._register_action(action)
                    action_value[a_idx] = self.INITIAL_Q_VALUE
                self.q_table[s_idx] = action_value
            s_idx = self.state_repr_list.index(state_instance)
            return s_idx
        elif self.AGENT_TYPE == "W":
            use_actionset = (
                self.state_mode == "actionset"
                and isinstance(state, ActionSetWithExecutionTimesState)
            )
            state_url = getattr(state, "raw_url", None) or getattr(state, "url", "")
            new_state = state if use_actionset else TagSequenceState(html, url=state_url)
            max_sim = -1
            max_state = None
            for temp_state in self.state_repr_list:
                if isinstance(temp_state, OutOfDomainState) or isinstance(temp_state, SameUrlState) or isinstance(
                        temp_state, ActionExecuteFailedState):
                    continue
                if use_actionset and (not isinstance(temp_state, ActionSetWithExecutionTimesState)):
                    continue
                if (not use_actionset) and (not isinstance(temp_state, TagSequenceState)):
                    continue
                new_sim = new_state.similarity(temp_state)
                if new_sim > max_sim:
                    max_sim = new_sim
                    max_state = temp_state
            self._latest_state_max_sim = max(0.0, max_sim)
            if max_sim > self.MAX_SIM_LINE:
                s_idx = self.state_repr_list.index(max_state)
                self._latest_state_is_new = False
                actions = state.get_action_list()
                for action in actions:
                    a_idx = self._register_action(action)
                    if a_idx not in self.q_table[s_idx]:
                        self.q_table[s_idx][a_idx] = self.INITIAL_Q_VALUE
            else:
                self.state_repr_list.append(new_state)
                s_idx = self.state_repr_list.index(new_state)
                self._latest_state_is_new = True
                action_value = dict()
                actions = state.get_action_list()
                for action in actions:
                    a_idx = self._register_action(action)
                    action_value[a_idx] = self.INITIAL_Q_VALUE
                self.q_table[s_idx] = action_value
            return s_idx

    def get_reward(self, state_index, web_state=None, next_action=None):
        if self.reward_mode == "webqt":
            if self.previous_state is None or self.previous_action is None:
                return 0.0

            state_action_key = (self.previous_state, self.previous_action)
            r_explore = self._compute_webqt_explore(self.previous_state)
            self.state_action_count[state_action_key] += 1

            if not isinstance(web_state, ActionSetWithExecutionTimesState):
                return float(self.R_PENALTY)

            r_loc = self._compute_webqt_locality(self.previous_action_obj, next_action)
            r_attention = self._compute_webqt_attention(
                self.previous_web_state, web_state, next_action
            )
            r_freq = 1.0 / math.sqrt(self.state_action_count[state_action_key])
            reward = (
                self.webqt_w_loc * r_loc
                + self.webqt_w_attention * r_attention
                + self.webqt_w_freq * r_freq
                + self.webqt_w_explore * r_explore
            )
            return float(reward)

        if self.reward_mode in ("pure", "vanilla", "flat"):
            if isinstance(
                web_state,
                (OutOfDomainState, ActionExecuteFailedState, SameUrlState),
            ):
                return float(self.R_PENALTY)
            return float(self.R_REWARD)

        if self.reward_mode == "state_only":
            return max(0.0, 1.0 - self._latest_state_max_sim)
        if self.reward_mode == "state_binary_1":
            return 1.0 if self._latest_state_is_new else 0.0
        if self.reward_mode == "state_binary_100":
            return 100.0 if self._latest_state_is_new else 0.0
        if self.reward_mode == "state_count":
            self.state_count[state_index] = self.state_count.get(state_index, 0) + 1
            return 1.0 / math.sqrt(self.state_count[state_index])
        if self.AGENT_TYPE == "W":
            s = "{}-{}-{}".format(self.previous_state, self.previous_action, state_index)
            self.trans_count[s] += 1
            reward = 1 / math.sqrt(self.trans_count[s])
            return reward
        elif self.AGENT_TYPE == "Q":
            action_count = self.action_count[self.previous_action]
            if action_count == 1:
                reward = 500
            else:
                reward = 1 / action_count
            return reward

    def update(self, web_state_index, web_state, next_action=None):
        ps_q_values = self.q_table[self.previous_state]
        cs_q_values = self.q_table[web_state_index]
        reward = self.get_reward(web_state_index, web_state, next_action)
        reward = clip_reward_value(reward, self.reward_clip_abs)
        q_predict = ps_q_values[self.previous_action]
        if self.AGENT_TYPE == "Q":
            action_len=1
            if isinstance(web_state, ActionExecuteFailedState):
                action_list = web_state.get_action_list()
                action_len = len(action_list)
            gamma = 0.9 * math.exp(-0.1 * (abs(action_len) - 1))
        else:
            gamma = self.GAMMA
        # Treat failure / out-of-domain / same-url states as terminal to avoid
        # bootstrapping from sentinel Q values (e.g. -9999), which would
        # otherwise dominate updates and collapse the policy.
        is_terminal = isinstance(
            web_state,
            (OutOfDomainState, ActionExecuteFailedState, SameUrlState),
        )
        if is_terminal:
            q_target = reward
        else:
            q_target = reward + gamma * max(cs_q_values.values())
        print("Updated Q Value:", self.q_table[self.previous_state][self.previous_action], "->",
              q_predict + self.ALPHA * (q_target - q_predict))
        self.q_table[self.previous_state][self.previous_action] = q_predict + self.ALPHA * (q_target - q_predict)
        self._record_dfa_transition(self.previous_state, self.previous_action, web_state_index)
        return float(reward)

    def get_action_index(self, action):
        if isinstance(action, RestartAction):
            return 0
        return self._register_action(action)


    def get_action(self, web_state, html):
        actions = web_state.get_action_list()
        if len(actions) == 0:
            raise NoActionsException("The state does not have any actions")

        chosen_action = None
        stop_update = False
        step_reward = 0.0

        self._obs_state_history.append(web_state)

        state_index = self.get_state_index(web_state, html)
        if self.webqt_mode:
            if self._latest_state_is_new:
                self.webqt_no_new_state_streak = 0
            else:
                self.webqt_no_new_state_streak += 1

        current_epsilon = self._current_epsilon()
        forced_action = self._select_forced_action(actions, state_index)
        if forced_action is not None:
            chosen_action, max_val, stop_update = forced_action
        elif self.policy_mode == "gumbel":
            chosen_action, max_val, stop_update = self._pick_gumbel_action(
                actions, state_index
            )
        elif self.webqt_mode and self.webqt_no_new_state_streak >= self.webqt_stagnation_window:
            chosen_action, max_val, stop_update = self._pick_webqt_escape_action(
                actions, state_index
            )
        elif random.uniform(0, 1) < current_epsilon:
            max_val = max(self.q_table[state_index].values())
            chosen_action = random.choice(actions)
        else:
            chosen_action = actions[0]
            max_val = self.q_table[state_index][self.get_action_index(chosen_action)]
            for temp_action in actions:
                if isinstance(temp_action, RestartAction):
                    chosen_action = temp_action
                    stop_update = True
                    break
                if self.q_table[state_index][self.get_action_index(temp_action)] > max_val:
                    max_val = self.q_table[state_index][self.get_action_index(temp_action)]
                    chosen_action = temp_action

        if self.is_dfa_control_reset(chosen_action):
            self.stop_update = True
            print(
                "max_q_value: ",
                max_val,
                "  chosen_action: ",
                safe_console_text(chosen_action),
                "  qlearning_dfa_control_reset: True",
            )
            return chosen_action

        self.action_count[self.get_action_index(chosen_action)] += 1
        if (self.previous_state is not None and self.previous_state > 2 and self.previous_action is not None and
                not self.stop_update):
            step_reward = self.update(state_index, web_state, chosen_action)
        self.previous_state = state_index
        self.previous_action = self.get_action_index(chosen_action)
        self.previous_web_state = web_state
        self.previous_action_obj = chosen_action
        self.stop_update = stop_update

        self._obs_step_counter += 1
        elapsed_obs = (datetime.now() - self.start_time).total_seconds()
        history_for_mg = (
            list(self._obs_state_history[:-1]) if len(self._obs_state_history) > 1 else []
        )
        action_idx = self.get_action_index(chosen_action)
        marg_count = max(1, self.action_count.get(action_idx, 1))
        marg_reward = 1.0 / marg_count
        self.observer.record_step(
            step=self._obs_step_counter,
            web_state=web_state if isinstance(web_state, ActionSetWithExecutionTimesState) else web_state,
            chosen_action=chosen_action,
            actual_reward=step_reward,
            max_q_value=float(max_val),
            history_states=history_for_mg,
            elapsed_seconds=elapsed_obs,
            marg_style_reward=marg_reward,
            action_global_count=marg_count,
        )

        print("max_q_value: ", max_val, "  chosen_action: ", safe_console_text(chosen_action))
        return chosen_action
