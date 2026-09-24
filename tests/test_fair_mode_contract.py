import os
from types import SimpleNamespace

import pytest

from agent.impl.q_learning_agent import QLearningAgent
from fairness import apply_fair_env_defaults, summarize_transitions, validate_fair_contract
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.impl.same_url_state import SameUrlState
from state.impl.tag_sequence_state import TagSequenceState


def test_apply_fair_env_defaults_for_qlearning(monkeypatch):
    monkeypatch.setenv("WEBTEST_FAIR_MODE", "1")
    monkeypatch.delenv("WEBTEST_QLEARNING_STATE_MODE", raising=False)
    applied = apply_fair_env_defaults("agent.impl.q_learning_agent", "QLearningAgent")
    assert applied["WEBTEST_QLEARNING_STATE_MODE"] == "actionset"
    assert os.environ["WEBTEST_QLEARNING_AGENT_TYPE"] == "W"


def test_validate_fair_contract_passes_with_required_settings(monkeypatch):
    monkeypatch.setenv("WEBTEST_FAIR_MODE", "1")
    monkeypatch.setenv("WEBTEST_URL_NORMALIZE_MODE", "path_only")
    monkeypatch.setenv("WEBTEST_RESTART_POLICY", "entry")
    monkeypatch.setenv("WEBTEST_EFFECTIVE_SEED", "123")
    monkeypatch.setenv("WEBTEST_MAX_TRANSITIONS", "100")
    monkeypatch.setenv("WEBTEST_QLEARNING_STATE_MODE", "actionset")
    settings_stub = SimpleNamespace(
        state={"module": "state.impl.action_set_with_execution_times_state", "class": "ActionSetWithExecutionTimesState"},
        action_detector={"module": "action.detector.combination_detector", "class": "CombinationDetector"},
    )
    errors = validate_fair_contract(settings_stub, "agent.impl.q_learning_agent", "QLearningAgent")
    assert errors == []


def test_qlearning_reward_is_clipped_in_update(monkeypatch):
    monkeypatch.setenv("WEBTEST_FAIR_MODE", "1")
    monkeypatch.setenv("WEBTEST_QLEARNING_STATE_MODE", "actionset")
    monkeypatch.setenv("WEBTEST_QLEARNING_AGENT_TYPE", "W")
    monkeypatch.setenv("WEBTEST_REWARD_CLIP_ABS", "0.2")
    params = {
        "agent_type": "W",
        "alpha": 0.1,
        "gamma": 0.5,
        "epsilon": 0.5,
        "initial_q_value": 10.0,
        "r_reward": 1.0,
        "r_penalty": -1.0,
        "max_sim_line": 0.8,
    }
    agent = QLearningAgent(params)
    agent.previous_state = 3
    agent.previous_action = 1
    agent.q_table = {3: {1: 0.0}, 4: {0: 0.0}}

    # Default W-type transition reward starts at 1.0 for the first transition;
    # clip threshold is 0.2, so the update reward should be clipped to 0.2.
    reward = agent.update(4, SameUrlState("https://github.com"))
    assert reward == 0.2
    assert agent.q_table[3][1] == pytest.approx(0.02, abs=1e-9)


def test_tag_sequence_state_rejected_in_fair_mode(monkeypatch):
    monkeypatch.setenv("WEBTEST_FAIR_MODE", "1")
    with pytest.raises(RuntimeError):
        TagSequenceState("<html><body></body></html>")


def test_summarize_transitions_flags_schema_error():
    normal = ActionSetWithExecutionTimesState([], "https://github.com")
    report = summarize_transitions(
        [
            (normal, None, normal),
            (normal, None, object()),
        ]
    )
    assert report["total_transitions"] == 2
    assert report["schema_error_count"] >= 1
