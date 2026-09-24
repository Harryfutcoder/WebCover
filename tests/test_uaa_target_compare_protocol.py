from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / 'rebuttal/RQ2/uaa_target_compare_20260924/run_serial.py').is_file():
    pytest.skip('Local campaign records are not part of the source snapshot', allow_module_level=True)
SPEC = importlib.util.spec_from_file_location(
    'target_compare', ROOT / 'rebuttal/RQ2/uaa_target_compare_20260924/run_serial.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_followup_plan_only_changes_target_and_distribution():
    plan = MODULE.build_plan()
    MODULE.validate_plan(plan)
    assert len(plan['references']) == 10
    assert len(plan['runs']) == 5


@pytest.mark.parametrize('key,value', [
    ('WEBTEST_A2C_ENTROPY_COEF', '0.001'),
    ('WEBTEST_GRAPH_RESIDUAL_AUX_COEF', '0.4'),
    ('WEBTEST_GRAPH_RESIDUAL_AUX_TIE_BREAK_COEF', '0.05'),
])
def test_followup_rejects_other_training_changes(key, value):
    plan = deepcopy(MODULE.build_plan())
    plan['runs'][0]['environment'][key] = value
    with pytest.raises(AssertionError):
        MODULE.validate_plan(plan)


def test_runtime_checker_rejects_uniform_and_changed_hyperparameter():
    record = MODULE.read_json(
        MODULE.REFERENCE / 'runs/seeduaapaperonpetclinic1_20260924/run.json')
    uniform = record['verified_runtime_config']
    with pytest.raises(RuntimeError, match='graph_residual_aux_target'):
        MODULE.config_line(uniform, 'max_only')
    maximal = uniform.replace('graph_residual_aux_target=uncovered', 'graph_residual_aux_target=graph')
    maximal = maximal.replace('graph_residual_aux_distribution=uniform', 'graph_residual_aux_distribution=max_only')
    assert MODULE.config_line(maximal, 'max_only') == maximal
    with pytest.raises(RuntimeError, match='entropy_coef'):
        MODULE.config_line(maximal.replace('entropy_coef=0.010', 'entropy_coef=0.001'), 'max_only')


def test_actual_targets_differ_even_with_same_uncovered_actions():
    from action.element_locator import ElementLocator
    from action.impl.click_action import ClickAction
    from agent.impl.subweb_frontier_a2c_agent import SubWebFrontierA2CAgent
    from agent.impl.subweb_frontier_coverage import FrontierCoverageTracker
    from agent.impl.subweb_padded_actions import PaddedActionBatch
    from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
    new = ClickAction(ElementLocator.XPATH, '//a[1]', 'New', 'redirect', '/owners/new')
    seen = ClickAction(ElementLocator.XPATH, '//a[2]', 'Seen', 'redirect', '/vets.html')
    state = ActionSetWithExecutionTimesState([new, seen], 'http://localhost/owners/find')
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.coverage_tracker = FrontierCoverageTracker('frontier_uniform')
    agent.coverage_tracker.covered_nodes.add('node:/vets.html|state=already-seen|n=4')
    agent.use_action_coverage_features = False
    agent.graph_residual_aux_target_power = 1.0
    agent.graph_residual_aux_tie_break_coef = 0.0
    batch = PaddedActionBatch(
        actions_full=[new, seen], actions_policy=[new, seen], action_mat=torch.zeros((2, 3)),
        action_mask=torch.tensor([True, True]), original_action_count=2, kept_action_count=2,
        truncated_count=0, truncated_by_type={}, action_set_context=torch.zeros(0),
        structural_action_set_context=torch.zeros(0))
    agent.graph_residual_aux_target_mode = 'uncovered'
    agent.graph_residual_aux_distribution = 'uniform'
    uniform = agent._graph_residual_aux_target(batch, state)
    agent.graph_residual_aux_target_mode = 'graph'
    agent.graph_residual_aux_distribution = 'max_only'
    maximal = agent._graph_residual_aux_target(batch, state)
    assert torch.equal(uniform, torch.tensor([.5, .5]))
    assert torch.equal(maximal, torch.tensor([1., 0.]))
    assert not agent.coverage_tracker.covered_edges
