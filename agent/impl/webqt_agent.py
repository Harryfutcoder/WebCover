"""
WebQT-style baseline: weighted multi-indicator hand-crafted reward.

This is a reproducible approximation aligned with WebQT's design spirit:
    r = w_loc * r_loc + w_attention * r_attention + w_freq * r_freq + w_explore * r_explore

All components are in [0, 1], then combined by fixed weights.
"""

import math
import os
from collections import defaultdict

from action.impl.random_input_action import RandomInputAction
from action.impl.random_select_action import RandomSelectAction
from action.impl.restart_action import RestartAction
from observation.observer import Observer
from state.impl.action_set_with_execution_times_state import (
    ActionSetWithExecutionTimesState,
)

from agent.impl.drl_agent import DRLagent


class WebQTAgent(DRLagent):
    def __init__(self, params):
        super().__init__(params)
        self.w_loc = float(os.environ.get("WEBQT_W_LOC", params.get("w_loc", 10.0)))
        self.w_attention = float(
            os.environ.get("WEBQT_W_ATTENTION", params.get("w_attention", 50.0))
        )
        self.w_freq = float(os.environ.get("WEBQT_W_FREQ", params.get("w_freq", 5.0)))
        self.w_explore = float(
            os.environ.get("WEBQT_W_EXPLORE", params.get("w_explore", 5.0))
        )
        self.transition_counts = defaultdict(lambda: 1)
        self.action_type_counts = defaultdict(int)
        self.observer = Observer(
            agent_name="WebQT-style",
            log_dir="observation_logs",
            marginal_gain_sample_cap=None
            if os.environ.get("OBSERVATION_MG_FULL_HISTORY", "").strip().lower()
            in ("1", "true", "yes")
            else 50,
        )

    @staticmethod
    def _action_type(action) -> str:
        if action is None:
            return "none"
        if isinstance(action, RestartAction):
            return "restart"
        if isinstance(action, RandomInputAction):
            return "input"
        if isinstance(action, RandomSelectAction):
            return "select"
        return getattr(action, "action_type", "default")

    def get_reward(self, web_state):
        if not isinstance(web_state, ActionSetWithExecutionTimesState):
            return -1.0

        # r_loc: state novelty (global)
        max_sim, _ = self._max_state_similarity(web_state)
        r_loc = max(0.0, 1.0 - max_sim)

        # r_attention: action-type novelty for previous action
        r_attention = 0.0
        if self.previous_action is not None:
            action_type = self._action_type(self.previous_action)
            self.action_type_counts[action_type] += 1
            r_attention = 1.0 / math.sqrt(self.action_type_counts[action_type])

        # r_freq: action execution frequency shaping
        r_freq = 0.0
        if (
            self.previous_state is not None
            and isinstance(self.previous_state, ActionSetWithExecutionTimesState)
            and self.previous_action is not None
            and self.previous_action in self.action_list
        ):
            action_idx = self.action_list.index(self.previous_action)
            execution_time = max(1, self.action_count[action_idx])
            r_freq = 1.0 / math.sqrt(execution_time)

        # r_explore: transition curiosity
        r_explore = 0.0
        if (
            self.previous_state is not None
            and isinstance(self.previous_state, ActionSetWithExecutionTimesState)
            and self.previous_action is not None
            and self.previous_state in self.state_list
            and web_state in self.state_list
            and self.previous_action in self.action_list
        ):
            prev_idx = self.state_list.index(self.previous_state)
            cur_idx = self.state_list.index(web_state)
            act_idx = self.action_list.index(self.previous_action)
            key = (prev_idx, act_idx, cur_idx)
            n = self.transition_counts[key]
            self.transition_counts[key] = n + 1
            r_explore = 1.0 / math.sqrt(n)

        reward = (
            self.w_loc * r_loc
            + self.w_attention * r_attention
            + self.w_freq * r_freq
            + self.w_explore * r_explore
        )
        return float(reward)

