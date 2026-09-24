"""
WebExplor baseline under the shared environment contract.

The paper version combines URL-gated HTML tag-sequence state abstraction,
transition-count curiosity, Gumbel-Softmax action selection, and high-level DFA
guidance that triggers after a time plateau with no newly discovered state.
"""

import os
import random
import re
from collections import defaultdict, deque
from datetime import datetime
from math import sqrt

from action.impl.click_action import ClickAction
from action.impl.random_input_action import RandomInputAction
from action.impl.random_select_action import RandomSelectAction
from action.impl.restart_action import RestartAction
from exceptions import NoActionsException
from config.settings import settings
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from utils import safe_console_text

from agent.impl.q_learning_agent import QLearningAgent


class WebExplorAgent(QLearningAgent):
    def __init__(self, params):
        reward_mode = os.environ.get("WEBTEST_WEBEXPLOR_REWARD_MODE", "").strip()
        if reward_mode and not os.environ.get("WEBTEST_QLEARNING_REWARD_MODE", "").strip():
            os.environ["WEBTEST_QLEARNING_REWARD_MODE"] = reward_mode

        state_mode = os.environ.get("WEBTEST_WEBEXPLOR_STATE_MODE", "tagseq").strip().lower()
        if state_mode == "tagseq":
            os.environ["WEBTEST_ALLOW_TAG_SEQUENCE_IN_FAIR_WEBEXPLOR"] = "1"
            if not os.environ.get("WEBTEST_TAG_SEQUENCE_REQUIRE_SAME_URL", "").strip():
                os.environ["WEBTEST_TAG_SEQUENCE_REQUIRE_SAME_URL"] = "1"
        if not os.environ.get("WEBTEST_QLEARNING_STATE_MODE", "").strip():
            os.environ["WEBTEST_QLEARNING_STATE_MODE"] = state_mode
        if not os.environ.get("WEBTEST_QLEARNING_AGENT_TYPE", "").strip():
            os.environ["WEBTEST_QLEARNING_AGENT_TYPE"] = "W"
        if not os.environ.get("WEBTEST_QLEARNING_POLICY_MODE", "").strip():
            os.environ["WEBTEST_QLEARNING_POLICY_MODE"] = os.environ.get(
                "WEBTEST_WEBEXPLOR_POLICY_MODE",
                "gumbel",
            )
        if not os.environ.get("WEBTEST_QLEARNING_GUMBEL_TAU", "").strip():
            os.environ["WEBTEST_QLEARNING_GUMBEL_TAU"] = os.environ.get(
                "WEBEXPLOR_GUMBEL_TAU",
                "1.0",
            )

        q_params = {
            "agent_type": "W",
            "alpha": float(os.environ.get("WEBEXPLOR_ALPHA", "1.0")),
            "gamma": float(os.environ.get("WEBEXPLOR_GAMMA", "0.95")),
            "epsilon": float(os.environ.get("WEBEXPLOR_EPSILON", "0.0")),
            "initial_q_value": float(os.environ.get("WEBEXPLOR_INITIAL_Q_VALUE", "0.0")),
            "r_reward": float(os.environ.get("WEBEXPLOR_R_REWARD", "1.0")),
            "r_penalty": float(os.environ.get("WEBEXPLOR_R_PENALTY", "-1.0")),
            "max_sim_line": float(os.environ.get("WEBEXPLOR_STATE_SIM_THRESHOLD", "0.8")),
        }
        super().__init__(q_params)
        self.observer.agent_name = "WebExplor-tabular"
        self._stable_action_key_to_index = {}

        enable_dfa = os.environ.get("WEBEXPLOR_ENABLE_DFA", "1").strip().lower()
        self.enable_dfa_guidance = enable_dfa in ("1", "true", "yes", "on")
        self.dfa_stuck_seconds = max(
            1.0,
            float(os.environ.get("WEBEXPLOR_DFA_STUCK_SECONDS", "120")),
        )
        raw_stuck_steps = os.environ.get("WEBEXPLOR_DFA_STUCK_STEPS", "").strip()
        try:
            self.dfa_stuck_steps = int(raw_stuck_steps) if raw_stuck_steps else None
        except ValueError:
            self.dfa_stuck_steps = None
        self.dfa_initial_state = None
        self.dfa_edges = {}
        self.dfa_adj = defaultdict(dict)
        self.guided_action_queue = deque()
        self.steps_since_new_state = 0
        self.last_new_state_time = datetime.now()
        self.last_dfa_guidance_time = None
        self._last_guided_target = None

    @staticmethod
    def _normalize_replay_location(value):
        text = str(value or "").strip().lower()
        text = ActionSetWithExecutionTimesState._UUID_RE.sub("{id}", text)
        text = ActionSetWithExecutionTimesState._LONG_TOKEN_RE.sub("{token}", text)
        return re.sub(r"\s+", "", text)

    @staticmethod
    def _webexplor_action_signature(action):
        """Stable-enough executable action identity for DFA replay.

        The paper-level DFA stores abstract transitions, but this implementation
        replays concrete WebAction objects.  Dynamic pages can refresh hrefs or
        session-specific URL hosts while exposing the same executable element;
        this signature keeps locator position and action kind concrete, while
        normalizing dynamic target URLs to route identity.
        """
        if isinstance(action, RestartAction):
            return ("RestartAction",)

        locator = getattr(getattr(action, "locator", None), "value", "")
        location = WebExplorAgent._normalize_replay_location(getattr(action, "location", ""))
        text = ActionSetWithExecutionTimesState._normalize_token_text(
            getattr(action, "text", "")
        )

        if isinstance(action, ClickAction):
            action_type = ActionSetWithExecutionTimesState._normalize_token_text(
                getattr(action, "action_type", "default") or "default"
            )
            route_hint = ActionSetWithExecutionTimesState._functional_action_route_hint(action)
            if route_hint:
                info = route_hint
            else:
                info = ActionSetWithExecutionTimesState._normalize_action_info(
                    action_type,
                    getattr(action, "addition_info", ""),
                )
            return (
                "ClickAction",
                locator,
                location,
                text,
                action_type,
                info,
            )

        if isinstance(action, RandomInputAction):
            return ("RandomInputAction", locator, location, text)
        if isinstance(action, RandomSelectAction):
            return ("RandomSelectAction", locator, location, text)
        return (type(action).__name__, locator, location, text)

    @staticmethod
    def is_dfa_control_reset(action):
        return isinstance(action, RestartAction) and bool(
            getattr(action, "_webexplor_dfa_control_reset", False)
        )

    def _make_dfa_control_reset(self):
        """Paper-faithful environment reset for DFA trace replay.

        WebExplor's Algorithm 1 resets the environment before replaying a DFA
        trace, but that reset is not one of the learned/test actions.  Mark it
        so the runner can execute it without recording a transition or Q update.
        """
        action = RestartAction(settings.entry_url)
        setattr(action, "_webexplor_dfa_control_reset", True)
        return action

    def _match_guided_action(self, guided_action, actions):
        if guided_action in actions:
            return guided_action

        guided_signature = self._webexplor_action_signature(guided_action)
        matches = [
            action
            for action in actions
            if self._webexplor_action_signature(action) == guided_signature
        ]
        if len(matches) == 1:
            print(
                "[WebExplor-DFA] replay_signature_match action={}".format(
                    safe_console_text(matches[0])
                )
            )
            return matches[0]
        if len(matches) > 1:
            print(
                "[WebExplor-DFA] replay_signature_ambiguous matches={} signature={}".format(
                    len(matches),
                    safe_console_text(guided_signature),
                )
            )
        return None

    def _register_action(self, action):
        if isinstance(action, RestartAction):
            if len(self.action_list) == 0:
                self.action_list.append(action)
                self.action_count[0] = 0
            return 0

        action_key = self._webexplor_action_signature(action)
        existing_index = self._stable_action_key_to_index.get(action_key)
        if existing_index is not None:
            self.action_count.setdefault(existing_index, 0)
            return existing_index

        action_index = len(self.action_list)
        self.action_list.append(action)
        self.action_count[action_index] = 0
        self._stable_action_key_to_index[action_key] = action_index
        return action_index

    def get_action_index(self, action):
        if isinstance(action, RestartAction):
            return 0
        return self._register_action(action)

    def _transition_count(self, source_state, action_index, target_state):
        key = "{}-{}-{}".format(source_state, action_index, target_state)
        return max(1, int(self.trans_count.get(key, 1)))

    def _transition_curiosity(self, source_state, action_index, target_state):
        return 1.0 / sqrt(self._transition_count(source_state, action_index, target_state))

    def _record_dfa_transition(self, previous_state, previous_action, current_state):
        if not self.enable_dfa_guidance:
            return None
        if previous_state is None or previous_action is None:
            return None
        if previous_state <= 2 or current_state <= 2:
            return None
        if previous_state == current_state:
            return None

        self.dfa_edges[(previous_state, previous_action)] = current_state
        self.dfa_adj[previous_state][previous_action] = current_state
        return None

    def update(self, web_state_index, web_state, next_action=None):
        return super().update(web_state_index, web_state, next_action)

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
            matched_action = self._match_guided_action(guided_action, actions)
            if matched_action is not None:
                return matched_action

            # If the exact action is unavailable, the replayed trace no longer
            # matches the live page.  Fall back to the learned policy.  The
            # signature fallback above keeps dynamic URL/session changes from
            # disabling DFA guidance unnecessarily.
            self.guided_action_queue.clear()
            # A failed replay is still a guidance attempt.  Cool down the
            # time-based stuck detector so DFA does not immediately emit an
            # infinite RestartAction loop on the next decision.
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
        if self.dfa_stuck_steps is not None and self.steps_since_new_state < self.dfa_stuck_steps:
            return None
        if self.dfa_stuck_steps is None:
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
            "[WebExplor-DFA] stuck_seconds={:.1f} stuck_steps={} target=({}, {}, {}) replay_len={}".format(
                elapsed_without_new_state,
                self.steps_since_new_state,
                source_state,
                action_index,
                target_state,
                len(trace),
            )
        )
        return self._make_dfa_control_reset(), 0.0, True

    def get_action(self, web_state, html):
        actions = web_state.get_action_list()
        if len(actions) == 0:
            raise NoActionsException("The state does not have any actions")

        chosen_action = None
        stop_update = False
        step_reward = 0.0

        self._obs_state_history.append(web_state)

        state_index = self.get_state_index(web_state, html)
        current_epsilon = self._current_epsilon()
        forced_action = self._select_forced_action(actions, state_index)
        if forced_action is not None:
            chosen_action, max_val, stop_update = forced_action
        elif self.policy_mode == "gumbel":
            chosen_action, max_val, stop_update = self._pick_gumbel_action(
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
                "  webexplor_dfa_control_reset: True",
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
