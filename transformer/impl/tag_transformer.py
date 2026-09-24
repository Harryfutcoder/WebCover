from collections import defaultdict

import numpy as np
import torch
from torch import Tensor

from action.impl.click_action import ClickAction
from action.impl.random_input_action import RandomInputAction
from action.impl.random_select_action import RandomSelectAction
from action.web_action import WebAction
from transformer.utils.generator import embedding, load_embedding_model
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.web_state import WebState
from transformer.utils.state_analysis import get_state_embedding
from transformer.transformer import Transformer

class TagTransformer(Transformer):
    def __init__(self):
        self.wv_from_bin = load_embedding_model()
        self.state_tensor_table = defaultdict(Tensor)
        self.action_tensor_table = {}

    def action_to_tensor(self, state: WebState, action: WebAction, execution_time=-1):
        action_data, execution_histogram = state.get_action_detailed_data()
        details = action_data.get(action) if hasattr(action_data, 'get') else action_data[action]
        if details is None:
            # ActionExecuteFailedState returns None for details; use safe defaults.
            if execution_time == -1:
                execution_time = 0
            child_state = None
        else:
            if execution_time == -1:
                execution_time = details['execution_time']
            child_state = details['child_state']

        child_array = [0] * 10
        if isinstance(child_state, ActionSetWithExecutionTimesState):
            child_array = child_state.action_execution_time_histogram
        child_key = tuple(int(value) for value in child_array)
        text = str(getattr(action, "text", "") or "")
        action_key = (type(action).__name__, text, int(execution_time or 0), child_key)
        cached = self.action_tensor_table.get(action_key)
        if cached is not None:
            return cached

        if (isinstance(action, ClickAction) or isinstance(action, RandomSelectAction) or
                isinstance(action, RandomInputAction)):

            try:
                embedding_result = embedding(text, execution_time, child_array, self.wv_from_bin)
            except Exception:
                # Keep training/testing alive even if NLP preprocessing fails.
                text_similar = 0.0
                embedding_result = np.concatenate(
                    (np.array([text_similar]), np.array([execution_time]), np.array(child_array))
                )
        else:
            text_similar = 0.0
            embedding_result = np.concatenate(
                (np.array([text_similar]), np.array([execution_time]), np.array(child_array)))

        tensor = torch.tensor(embedding_result)
        self.action_tensor_table[action_key] = tensor
        return tensor

    def state_to_tensor(self, state: WebState, html: str):
        if state not in self.state_tensor_table:
            embedding_result = get_state_embedding(html)
            tensor = torch.tensor(embedding_result)
            self.state_tensor_table[state] = tensor
        else:
            tensor = self.state_tensor_table[state]
        return tensor
