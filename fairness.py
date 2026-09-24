import os
from collections import Counter
from typing import Dict, Iterable, List, Tuple

from state.impl.action_execute_failed_state import ActionExecuteFailedState
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.impl.out_of_domain_state import OutOfDomainState
from state.impl.same_url_state import SameUrlState

REQUIRED_STATE_MODULE = "state.impl.action_set_with_execution_times_state"
REQUIRED_STATE_CLASS = "ActionSetWithExecutionTimesState"
REQUIRED_ACTION_DETECTOR_MODULE = "action.detector.combination_detector"
REQUIRED_ACTION_DETECTOR_CLASS = "CombinationDetector"


def is_fair_mode() -> bool:
    return os.environ.get("WEBTEST_FAIR_MODE", "0").strip().lower() in ("1", "true", "yes", "on")


def apply_fair_env_defaults(agent_module: str, agent_class: str) -> Dict[str, str]:
    if not is_fair_mode():
        return {}

    defaults: Dict[str, str] = {
        "WEBTEST_URL_NORMALIZE_MODE": "path_only",
        "WEBTEST_RESTART_POLICY": "entry",
        "WEBTEST_REWARD_CLIP_ABS": "5.0",
    }
    if agent_module == "agent.impl.q_learning_agent" and agent_class == "QLearningAgent":
        defaults["WEBTEST_QLEARNING_STATE_MODE"] = "actionset"
        defaults["WEBTEST_QLEARNING_AGENT_TYPE"] = "W"
    applied: Dict[str, str] = {}
    for key, value in defaults.items():
        if not os.environ.get(key, "").strip():
            os.environ[key] = value
            applied[key] = value
    return applied


def validate_fair_contract(settings_obj, agent_module: str, agent_class: str) -> List[str]:
    if not is_fair_mode():
        return []

    errors: List[str] = []
    state_cfg = getattr(settings_obj, "state", {}) or {}
    detector_cfg = getattr(settings_obj, "action_detector", {}) or {}

    if state_cfg.get("module") != REQUIRED_STATE_MODULE or state_cfg.get("class") != REQUIRED_STATE_CLASS:
        errors.append(
            f"state must be {REQUIRED_STATE_MODULE}.{REQUIRED_STATE_CLASS}, got "
            f"{state_cfg.get('module')}.{state_cfg.get('class')}"
        )
    if (
        detector_cfg.get("module") != REQUIRED_ACTION_DETECTOR_MODULE
        or detector_cfg.get("class") != REQUIRED_ACTION_DETECTOR_CLASS
    ):
        errors.append(
            f"action_detector must be {REQUIRED_ACTION_DETECTOR_MODULE}.{REQUIRED_ACTION_DETECTOR_CLASS}, got "
            f"{detector_cfg.get('module')}.{detector_cfg.get('class')}"
        )

    if os.environ.get("WEBTEST_URL_NORMALIZE_MODE", "").strip().lower() != "path_only":
        errors.append("WEBTEST_URL_NORMALIZE_MODE must be path_only in fair mode")
    if os.environ.get("WEBTEST_RESTART_POLICY", "").strip().lower() != "entry":
        errors.append("WEBTEST_RESTART_POLICY must be entry in fair mode")
    if not os.environ.get("WEBTEST_EFFECTIVE_SEED", "").strip():
        errors.append("WEBTEST_EFFECTIVE_SEED must be initialized in fair mode")
    if not os.environ.get("WEBTEST_MAX_TRANSITIONS", "").strip():
        errors.append("WEBTEST_MAX_TRANSITIONS must be set in fair mode")

    if agent_module == "agent.impl.q_learning_agent" and agent_class == "QLearningAgent":
        if os.environ.get("WEBTEST_QLEARNING_STATE_MODE", "").strip().lower() != "actionset":
            errors.append("QLearning fair mode requires WEBTEST_QLEARNING_STATE_MODE=actionset")

    return errors


def summarize_transitions(
    transitions: Iterable[Tuple[object, object, object]],
) -> Dict[str, object]:
    state_types = Counter()
    action_types = Counter()
    terminal_types = Counter()
    schema_error_count = 0

    for prev_state, action, next_state in transitions:
        if next_state is None:
            schema_error_count += 1
            continue
        next_name = type(next_state).__name__
        state_types[next_name] += 1
        if action is not None:
            action_types[type(action).__name__] += 1
        if isinstance(next_state, (OutOfDomainState, SameUrlState, ActionExecuteFailedState)):
            terminal_types[next_name] += 1
        elif not isinstance(next_state, ActionSetWithExecutionTimesState):
            schema_error_count += 1
        if prev_state is not None and not isinstance(
            prev_state,
            (ActionSetWithExecutionTimesState, OutOfDomainState, SameUrlState, ActionExecuteFailedState),
        ):
            schema_error_count += 1

    return {
        "total_transitions": int(sum(state_types.values())),
        "state_types": dict(state_types),
        "action_types": dict(action_types),
        "terminal_state_types": dict(terminal_types),
        "schema_error_count": int(schema_error_count),
    }
