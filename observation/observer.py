"""
运行时观测记录器：不修改训练，仅记录子模边际增益 vs Bellman 奖励等。
"""

import json
import os
import random
from collections import defaultdict
from datetime import datetime
from typing import List, Optional

from action.impl.restart_action import RestartAction
from state.impl.action_execute_failed_state import ActionExecuteFailedState
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.impl.out_of_domain_state import OutOfDomainState
from state.impl.same_url_state import SameUrlState


class Observer:
    """
    记录：
    1. 真实子模边际增益 vs 实际 Bellman 奖励
    2. 状态访问次数
    3. 新状态发现速度
    4. 同一状态多次访问时的 Q 值
    """

    def __init__(
        self,
        agent_name: str = "DRLagent",
        log_dir: str = "observation_logs",
        marginal_gain_sample_cap: Optional[int] = 50,
    ):
        """
        marginal_gain_sample_cap:
            - 正整数：历史状态随机采样的上限（长运行降耗时）
            - None：全量历史计算 true_marginal_gain（短测/要精确相关系数时用）
        环境变量 OBSERVATION_MG_FULL_HISTORY=1 等价于 marginal_gain_sample_cap=None（在 DRLagent 里读取）。
        """
        self.agent_name = agent_name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.step_log: List[dict] = []
        self.state_visit_count = defaultdict(int)
        self.state_first_seen_step = {}
        self.state_q_values_history = defaultdict(list)

        self.total_steps = 0
        self.unique_states_over_time = []
        self.start_time = datetime.now()
        self._marginal_gain_sample_cap = marginal_gain_sample_cap

    def _get_state_id(self, web_state) -> str:
        def _action_signature(action) -> str:
            action_class = type(action).__name__
            locator = getattr(getattr(action, "locator", None), "value", "")
            location = getattr(action, "location", "")
            text = getattr(action, "text", "")
            if action_class == "ClickAction":
                action_type = getattr(action, "action_type", "")
                addition_info = getattr(action, "addition_info", "")
                return (
                    f"{action_class}|locator={locator}|location={location}|text={text}"
                    f"|type={action_type}|info={addition_info}"
                )
            return f"{action_class}|locator={locator}|location={location}|text={text}"

        if isinstance(web_state, ActionSetWithExecutionTimesState):
            return (
                f"url={web_state.url}|actions="
                f"{','.join(sorted(_action_signature(a) for a in web_state.get_action_list()))}"
            )
        if isinstance(web_state, OutOfDomainState):
            return "OUT_OF_DOMAIN"
        if isinstance(web_state, ActionExecuteFailedState):
            return "ACTION_FAILED"
        if isinstance(web_state, SameUrlState):
            return "SAME_URL"
        return f"OTHER_{type(web_state).__name__}"

    def compute_true_marginal_gain(self, web_state, history_states: list) -> float:
        """
        Δ(s | H) = 1 - max_{h ∈ H} similarity(s, h)，H 为历史中已出现的有效状态。
        历史过长且 _marginal_gain_sample_cap 为正整数时随机采样；为 None 则全量。
        """
        if not isinstance(web_state, ActionSetWithExecutionTimesState):
            return 0.0

        valid_history = [
            s
            for s in history_states
            if isinstance(s, ActionSetWithExecutionTimesState) and s is not web_state
        ]
        if not valid_history:
            return 1.0

        cap = self._marginal_gain_sample_cap
        if cap is not None and len(valid_history) > cap:
            valid_history = random.sample(valid_history, cap)

        try:
            max_sim = max(web_state.similarity(s) for s in valid_history)
        except Exception:
            max_sim = 0.0

        return max(0.0, 1.0 - max_sim)

    def record_step(
        self,
        step: int,
        web_state,
        chosen_action,
        actual_reward: float,
        max_q_value: float,
        history_states: list,
        elapsed_seconds: float,
        marg_style_reward: float = None,
        action_global_count: int = None,
    ):
        # --- 先分类并过滤：避免异常状态或仅 Restart（max_q=0）污染 step_log / Q 轨迹 ---
        if isinstance(web_state, OutOfDomainState):
            state_type = "out_of_domain"
        elif isinstance(web_state, ActionExecuteFailedState):
            state_type = "action_failed"
        elif isinstance(web_state, SameUrlState):
            state_type = "same_url"
        elif isinstance(web_state, ActionSetWithExecutionTimesState):
            state_type = "valid"
        else:
            state_type = "other"

        if max_q_value == 0.0 and state_type != "valid":
            return
        if max_q_value == 0.0 and isinstance(chosen_action, RestartAction):
            return

        self.total_steps = step
        true_mg = self.compute_true_marginal_gain(web_state, history_states)
        state_id = self._get_state_id(web_state)

        is_new_state = self.state_visit_count[state_id] == 0
        self.state_visit_count[state_id] += 1
        visit_count = self.state_visit_count[state_id]

        if is_new_state:
            self.state_first_seen_step[state_id] = step

        self.state_q_values_history[state_id].append((step, float(max_q_value)))

        unique_count = len(self.state_visit_count)
        self.unique_states_over_time.append((step, unique_count, elapsed_seconds))

        record = {
            "step": step,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "state_id": state_id,
            "state_type": state_type,
            "is_new_state": is_new_state,
            "visit_count": visit_count,
            "true_marginal_gain": round(true_mg, 4),
            "actual_reward": round(float(actual_reward), 4),
            "max_q_value": round(float(max_q_value), 4),
            "unique_states_so_far": unique_count,
            "reward_mg_gap": round(float(actual_reward) - true_mg, 4),
        }
        if marg_style_reward is not None:
            record["marg_style_reward"] = round(marg_style_reward, 4)
            record["action_global_count"] = action_global_count or 0
            record["marg_mg_gap"] = round(marg_style_reward - true_mg, 4)
        self.step_log.append(record)

    def save_logs(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        step_log_path = os.path.join(self.log_dir, f"step_log_{timestamp}.json")
        with open(step_log_path, "w", encoding="utf-8") as f:
            json.dump(self.step_log, f, ensure_ascii=False, indent=2)

        freq_path = os.path.join(self.log_dir, f"visit_frequency_{timestamp}.json")
        freq_data = {
            "state_visit_counts": dict(self.state_visit_count),
            "unique_states_over_time": self.unique_states_over_time,
        }
        with open(freq_path, "w", encoding="utf-8") as f:
            json.dump(freq_data, f, ensure_ascii=False, indent=2)

        q_history = {
            k: v for k, v in self.state_q_values_history.items() if len(v) >= 3
        }
        q_path = os.path.join(self.log_dir, f"q_value_history_{timestamp}.json")
        with open(q_path, "w", encoding="utf-8") as f:
            json.dump(q_history, f, ensure_ascii=False, indent=2)

        print(f"[Observer] Logs saved under {self.log_dir}/")
        print(f"  Step log: {step_log_path}")
        print(f"  Frequency: {freq_path}")
        print(f"  Q history: {q_path}")

        return timestamp
