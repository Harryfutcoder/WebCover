from types import SimpleNamespace

import torch

from action.element_locator import ElementLocator
from action.impl.click_action import ClickAction
from agent.impl.subweb_frontier_a2c_agent import SubWebFrontierA2CAgent
from agent.impl.subweb_frontier_coverage import FrontierCoverageTracker, extract_graph_node_id, graph_edge_prefix
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from web_test.webtest_single_agent import Webtest


def fixture():
    actions = [
        ClickAction(ElementLocator.XPATH, "//a[1]", "New", "redirect", "/owners/new"),
        ClickAction(ElementLocator.XPATH, "//a[2]", "Seen target", "redirect", "/vets.html"),
        ClickAction(ElementLocator.XPATH, "//a[3]", "Home", "redirect", "/"),
    ]
    state = ActionSetWithExecutionTimesState(actions, "http://localhost:8081/owners/find")
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.coverage_tracker = FrontierCoverageTracker("frontier_uniform")
    agent.coverage_tracker.covered_nodes.add("node:/vets.html|state=seen")
    agent.graph_residual_aux_target_mode = "uncovered"
    # This legacy estimate path must never be called by paper UAA.
    agent._graph_residual_scores = lambda *args: (_ for _ in ()).throw(AssertionError("Residual score used"))
    batch = SimpleNamespace(actions_policy=actions, action_mask=torch.tensor([True, True, True, False]))
    return agent, state, batch


def cover(agent, state, action):
    prefix = graph_edge_prefix(state.url, action, source_node_id=extract_graph_node_id(state))
    agent.coverage_tracker.covered_edges.add(prefix + "target_node=observed")


def test_uncovered_uaa_uniform_ignores_target_gain_home_weight_and_padding():
    agent, state, batch = fixture()
    before = batch.action_mask.clone()
    target = agent._graph_residual_aux_target(batch, state)
    assert torch.allclose(target, torch.tensor([1 / 3, 1 / 3, 1 / 3, 0.0]))
    assert torch.equal(before, batch.action_mask)
    assert agent.coverage_tracker.cumulative_F == 0.0


def test_uncovered_uaa_paper_example_and_no_new_reward_memory():
    agent, state, batch = fixture()
    cover(agent, state, batch.actions_policy[0])
    before = set(agent.coverage_tracker.covered_edges)
    assert torch.equal(agent._graph_residual_aux_target(batch, state), torch.tensor([0.0, .5, .5, 0.0]))
    assert before == agent.coverage_tracker.covered_edges


def test_uncovered_uaa_all_resolved_is_inactive():
    agent, state, batch = fixture()
    for action in batch.actions_policy:
        cover(agent, state, action)
    assert torch.equal(agent._graph_residual_aux_target(batch, state), torch.zeros(4))
    assert agent.last_graph_residual_aux_target_debug["target_source"] == "zero"


def test_uncovered_uaa_is_source_specific_and_respects_mask():
    agent, state, batch = fixture()
    other_state = ActionSetWithExecutionTimesState(batch.actions_policy, "http://localhost:8081/")
    cover(agent, other_state, batch.actions_policy[0])
    batch.action_mask[1] = False
    assert torch.equal(agent._graph_residual_aux_target(batch, state), torch.tensor([.5, 0.0, .5, 0.0]))
    assert torch.equal(agent._graph_residual_aux_target(batch, None), torch.zeros(4))


def browser_error(url="http://localhost:8081/oups", status=500):
    return {"level": "SEVERE", "source": "network", "timestamp": 1,
            "message": f"{url} - Failed to load resource: the server responded with a status of {status} ()"}


def test_expected_500_guard_is_exact_endpoint_and_status_only():
    web = Webtest.__new__(Webtest)
    web.expected_http_500_urls = frozenset({"http://localhost:8081/oups"})
    assert web._is_expected_http_500(browser_error())
    assert not web._is_expected_http_500(browser_error(status=503))
    for url in ("http://localhost:8081/owners", "http://localhost:8081/oups/other",
                "http://other.example/oups", "http://localhost:8081/oups?new=1"):
        assert not web._is_expected_http_500(browser_error(url))
    entry = browser_error()
    entry["source"] = "javascript"
    assert not web._is_expected_http_500(entry)


def test_expected_error_logging_retained_and_real_errors_still_count(tmp_path, monkeypatch):
    from web_test import webtest_single_agent as module
    monkeypatch.setattr(module, "settings", SimpleNamespace(output_path=str(tmp_path)))
    web = Webtest.__new__(Webtest)
    web.expected_http_500_urls = frozenset({"http://localhost:8081/oups"})
    web.filter_known_noise = False
    web.http_5xx_streak = 0
    web.driver = SimpleNamespace(get_log=lambda kind: [browser_error()])
    for _ in range(25):
        assert web.trace_error() is True
    assert web.http_5xx_streak == 0
    assert len((tmp_path / "bug.log").read_text().splitlines()) == 25
    web.driver.get_log = lambda kind: [browser_error(), browser_error("http://localhost:8081/owners")]
    web.trace_error()
    assert web.http_5xx_streak == 1


def test_expected_error_guard_without_opt_in_unchanged():
    web = Webtest.__new__(Webtest)
    assert not web._is_expected_http_500(browser_error())
