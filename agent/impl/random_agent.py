import random
import os
from collections import defaultdict
from datetime import datetime

from action.web_action import WebAction
from agent.agent import Agent
from exceptions import NoActionsException
from observation.observer import Observer
from state.web_state import WebState


class RandomAgent(Agent):
    def __init__(self, params=None):
        super().__init__()
        self._obs_step_counter = 0
        self._obs_state_history = []
        self._action_counts = defaultdict(int)
        self.start_time = datetime.now()
        _mg_cap = None if os.environ.get("OBSERVATION_MG_FULL_HISTORY", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ) else 50
        self.observer = Observer(
            agent_name="RandomAgent",
            log_dir="observation_logs",
            marginal_gain_sample_cap=_mg_cap,
        )

    def get_action(self, web_state: WebState, html: str) -> WebAction:
        actions = web_state.get_action_list()
        if len(actions) == 0:
            raise NoActionsException("The state does not have any actions")

        chosen_action = random.choice(actions)

        self._obs_state_history.append(web_state)
        self._obs_step_counter += 1
        self._action_counts[chosen_action] += 1
        elapsed_obs = (datetime.now() - self.start_time).total_seconds()

        history_for_mg = (
            list(self._obs_state_history[:-1]) if len(self._obs_state_history) > 1 else []
        )
        marg_count = max(1, self._action_counts[chosen_action])
        marg_reward = 1.0 / marg_count

        self.observer.record_step(
            step=self._obs_step_counter,
            web_state=web_state,
            chosen_action=chosen_action,
            actual_reward=0.0,
            max_q_value=0.0,
            history_states=history_for_mg,
            elapsed_seconds=elapsed_obs,
            marg_style_reward=marg_reward,
            action_global_count=marg_count,
        )

        return chosen_action
