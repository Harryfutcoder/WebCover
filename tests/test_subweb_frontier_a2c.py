import math
import os
from collections import defaultdict, deque
from types import SimpleNamespace

import torch

from action.element_locator import ElementLocator
from action.impl.click_action import ClickAction
from action.impl.random_input_action import RandomInputAction
from agent.impl.subweb_action_context_features import (
    action_role_feature_rows,
    action_role_features,
    context_feature_dim,
)
from agent.impl.subweb_frontier_coverage import (
    FrontierCoverageTracker,
    abstract_action_signature,
    extract_arrival_units,
    extract_frontier_units,
    extract_graph_edge_units,
    extract_graph_node_id,
    extract_graph_node_units,
    graph_edge_prefix,
)
from agent.impl.subweb_action_coverage_features import (
    FEATURE_NAMES as ACTION_COVERAGE_FEATURE_NAMES,
    SubWebActionCoverageFeatures,
    action_coverage_feature_dim,
)
from agent.impl.subweb_padded_actions import (
    PaddedActionBatch,
    action_kind,
    action_signature,
    build_padded_action_batch,
    deterministic_rank_actions,
)
from agent.impl.subweb_structural_action_features import (
    structural_action_feature_rows,
    structural_action_feature_names,
    structural_action_features,
    structural_feature_dim,
)
from agent.impl.subweb_frontier_a2c_agent import SubWebFrontierA2CAgent
from agent.impl.subweb_policy_diagnostics import summarize_policy_distribution
from agent.impl.subweb_recent_action_summary import (
    action_bucket_from_transition,
    recent_action_summary,
    recent_action_summary_dim,
)
from model.subweb_frontier_a2c import CandidateQHead, FrontierCritic, MaskedCandidateActor
from observation.observer import Observer
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from web_test.webtest_single_agent import Webtest


class _TinyTransformer:
    def state_to_tensor(self, web_state, html):
        return torch.zeros(40)

    def action_to_tensor(self, web_state, action):
        return torch.zeros(12)


def _state(url, actions):
    return ActionSetWithExecutionTimesState(actions, url)


def _restore_env(name, old_value):
    if old_value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old_value


def test_graph_form_value_buckets_are_opt_in():
    old_value = os.environ.get("WEBTEST_GRAPH_FORM_VALUE_BUCKETS")
    try:
        os.environ.pop("WEBTEST_GRAPH_FORM_VALUE_BUCKETS", None)
        assert Webtest._graph_form_value_buckets_enabled() is False
        os.environ["WEBTEST_GRAPH_FORM_VALUE_BUCKETS"] = "1"
        assert Webtest._graph_form_value_buckets_enabled() is True
    finally:
        _restore_env("WEBTEST_GRAPH_FORM_VALUE_BUCKETS", old_value)


def test_recent_action_summary_encodes_site_agnostic_workflow_history():
    input_transition = {
        "raw_reward": 0.0,
        "exploration": {
            "action_signature": "family=form_field|kind=input:text|intent=other|target=|rel=none|field=name"
        },
    }
    submit_transition = {
        "raw_reward": 1.0,
        "exploration": {
            "action_signature": "family=form_submit|kind=submit|intent=save|target=|rel=none|field="
        },
    }
    reset_transition = {
        "raw_reward": 0.0,
        "exploration": {
            "action_signature": "family=boilerplate|kind=link|intent=home|target=/|rel=parent|field="
        },
    }

    assert action_bucket_from_transition(input_transition) == "input"
    assert action_bucket_from_transition(submit_transition) == "submit"
    assert action_bucket_from_transition(reset_transition) == "reset_or_nav"

    summary = recent_action_summary([
        ("input", False),
        ("submit", True),
    ])

    assert summary.numel() == recent_action_summary_dim()
    assert summary[2].item() == 1.0
    assert summary[-2].item() == 1.0
    assert summary[-1].item() == 0.5


def test_masked_actor_assigns_zero_probability_to_padding():
    actor = MaskedCandidateActor(input_dim=3, hidden_dim=8)
    action_mat = torch.randn(4, 3)
    action_mask = torch.tensor([True, True, False, False])

    dist = actor.get_action_dist(action_mat, action_mask)

    assert dist.probs[2].item() == 0.0
    assert dist.probs[3].item() == 0.0
    assert abs(float(dist.probs[:2].sum().item()) - 1.0) < 1e-6


def test_graph_node_id_separates_form_progress_without_changing_state_equality():
    old_form = os.environ.get("WEBTEST_GRAPH_FORM_STATE")
    old_mode = os.environ.get("WEBTEST_GRAPH_NODE_MODE")
    os.environ["WEBTEST_GRAPH_FORM_STATE"] = "1"
    os.environ["WEBTEST_GRAPH_NODE_MODE"] = "observer_state"
    try:
        input_action = RandomInputAction(ElementLocator.XPATH, "//input[@name='name']", "Name")
        submit_action = ClickAction(ElementLocator.XPATH, "//button", "Save", "submit", "1")
        empty_state = _state("http://localhost/owners/new", [input_action, submit_action])
        filled_state = _state("http://localhost/owners/new", [input_action, submit_action])
        empty_state.graph_form_state_id = "fields=1|filled=0|required=1|required_filled=0|select_changed=0|sig=empty"
        filled_state.graph_form_state_id = "fields=1|filled=1|required=1|required_filled=1|select_changed=0|sig=filled"

        assert empty_state == filled_state
        assert extract_graph_node_id(empty_state) != extract_graph_node_id(filled_state)
        assert "|form=" in extract_graph_node_id(filled_state)
    finally:
        _restore_env("WEBTEST_GRAPH_FORM_STATE", old_form)
        _restore_env("WEBTEST_GRAPH_NODE_MODE", old_mode)


def test_graph_node_id_ignores_form_progress_by_default():
    old_form = os.environ.get("WEBTEST_GRAPH_FORM_STATE")
    old_mode = os.environ.get("WEBTEST_GRAPH_NODE_MODE")
    os.environ.pop("WEBTEST_GRAPH_FORM_STATE", None)
    os.environ.pop("WEBTEST_GRAPH_NODE_MODE", None)
    try:
        input_action = RandomInputAction(ElementLocator.XPATH, "//input[@name='name']", "Name")
        submit_action = ClickAction(ElementLocator.XPATH, "//button", "Save", "submit", "1")
        empty_state = _state("http://localhost/owners/new", [input_action, submit_action])
        filled_state = _state("http://localhost/owners/new", [input_action, submit_action])
        empty_state.graph_form_state_id = "fields=1|filled=0|required=1|required_filled=0|select_changed=0|sig=empty"
        filled_state.graph_form_state_id = "fields=1|filled=1|required=1|required_filled=1|select_changed=0|sig=filled"

        assert extract_graph_node_id(empty_state) == extract_graph_node_id(filled_state)
        assert "|form=" not in extract_graph_node_id(filled_state)
    finally:
        _restore_env("WEBTEST_GRAPH_FORM_STATE", old_form)
        _restore_env("WEBTEST_GRAPH_NODE_MODE", old_mode)


def test_repeated_form_input_does_not_create_new_graph_edge_by_default():
    old_form = os.environ.get("WEBTEST_GRAPH_FORM_STATE")
    old_mode = os.environ.get("WEBTEST_GRAPH_NODE_MODE")
    os.environ.pop("WEBTEST_GRAPH_FORM_STATE", None)
    os.environ.pop("WEBTEST_GRAPH_NODE_MODE", None)
    try:
        input_action = RandomInputAction(ElementLocator.XPATH, "//input[@name='name']", "Name")
        source = _state("http://localhost/new", [input_action])
        target_a = _state("http://localhost/new", [input_action])
        target_b = _state("http://localhost/new", [input_action])
        source.graph_form_state_id = "fields=1|filled=0|sig=empty"
        target_a.graph_form_state_id = "fields=1|filled=1|sig=a"
        target_b.graph_form_state_id = "fields=1|filled=1|sig=b"
        source_node = extract_graph_node_id(source)

        tracker = FrontierCoverageTracker("frontier_uniform")
        tracker.seed_initial_state(source)
        first = tracker.compute_strict_marginal_reward(
            target_a,
            chosen_action=input_action,
            source_url="http://localhost/new",
            source_node_id=source_node,
        )
        second = tracker.compute_strict_marginal_reward(
            target_b,
            chosen_action=input_action,
            source_url="http://localhost/new",
            source_node_id=source_node,
        )

        assert first.node_gain == 0
        assert first.edge_gain == 1
        assert second.reward == 0.0
        assert second.node_gain == 0
        assert second.edge_gain == 0
    finally:
        _restore_env("WEBTEST_GRAPH_FORM_STATE", old_form)
        _restore_env("WEBTEST_GRAPH_NODE_MODE", old_mode)


def _reward_variant_agent(mode):
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.reward_mode = mode
    agent.count_decay_scale = 1.0
    agent.count_decay_power = 0.5
    agent.webqt_w_loc = 10.0
    agent.webqt_w_attention = 50.0
    agent.webqt_w_freq = 5.0
    agent.webqt_w_explore = 5.0
    agent.webrled_w_global = 0.6
    agent.webrled_w_episode = 0.3
    agent.webrled_w_transition = 0.1
    agent.qexplore_valid_reward_scale = 1.0
    agent.qexplore_invalid_reward = -1.0
    agent.qexplore_initial_q = 500.0
    agent.reward_transition_counts = defaultdict(int)
    agent.reward_action_counts = defaultdict(int)
    agent.reward_qexplore_action_counts = defaultdict(int)
    agent.reward_action_type_counts = defaultdict(int)
    agent.reward_node_visit_counts = defaultdict(int)
    agent.reward_episode_nodes = set()
    return agent


def test_reward_variant_marginal_keeps_delta_f_reward():
    action = ClickAction(ElementLocator.XPATH, "//a[@href='/alpha']", "Alpha", "redirect", "/alpha")
    state = _state("http://localhost/alpha", [action])
    agent = _reward_variant_agent("marginal")
    agent.pending_step = {
        "source_node_id": "node:source",
        "source_url": "http://localhost/",
        "action_signature": "action:alpha",
        "chosen_action": action,
    }
    reward_info = SimpleNamespace(reward=3.0, node_gain=1, edge_gain=1)

    reward, details = agent._compute_training_reward(state, reward_info)

    assert reward == 3.0
    assert details["mode"] == "marginal"


def test_reward_variant_count_decay_uses_same_transition_key():
    action = ClickAction(ElementLocator.XPATH, "//a[@href='/alpha']", "Alpha", "redirect", "/alpha")
    state = _state("http://localhost/alpha", [action])
    agent = _reward_variant_agent("count_decay")
    agent.pending_step = {
        "source_node_id": "node:source",
        "source_url": "http://localhost/",
        "action_signature": "action:alpha",
        "chosen_action": action,
    }
    reward_info = SimpleNamespace(reward=0.0, node_gain=0, edge_gain=0)

    first, first_details = agent._compute_training_reward(state, reward_info)
    second, second_details = agent._compute_training_reward(state, reward_info)

    assert first == 1.0
    assert math.isclose(second, 1.0 / math.sqrt(2.0))
    assert first_details["transition_count"] == 1
    assert second_details["transition_count"] == 2


def test_reward_variant_qexplore_uses_source_code_global_action_count():
    action = ClickAction(ElementLocator.XPATH, "//a[@href='/alpha']", "Alpha", "redirect", "/alpha")
    state = _state("http://localhost/alpha", [action])
    agent = _reward_variant_agent("qexplore")
    agent.pending_step = {
        "source_node_id": "node:source",
        "source_url": "http://localhost/",
        "action_signature": "action:alpha",
        "chosen_action": action,
    }
    reward_info = SimpleNamespace(reward=2.0, node_gain=1, edge_gain=1)

    first, first_details = agent._compute_training_reward(state, reward_info)
    second, second_details = agent._compute_training_reward(state, reward_info)

    assert first == 1.0
    assert second == 0.5
    assert first_details["mode"] == "qexplore"
    assert first_details["action_count"] == 1
    assert second_details["action_count"] == 2
    assert "source_code_reward" in first_details["note"]


def test_strict_frontier_reward_is_monotone_and_zero_on_revisit():
    action = ClickAction(ElementLocator.XPATH, "//button", "Save", "submit", "")
    state = _state("http://localhost:8081/owners/new", [action])
    tracker = FrontierCoverageTracker("frontier_uniform")

    first = tracker.compute_strict_marginal_reward(state)
    second = tracker.compute_strict_marginal_reward(state)

    assert first.reward > 0.0
    assert second.reward == 0.0
    assert second.cumulative_F == first.cumulative_F


def test_initial_seed_covers_only_graph_node_without_action_reward():
    login = ClickAction(ElementLocator.XPATH, "//form/button", "Login", "submit", "")
    state = _state("http://localhost/login", [login])
    tracker = FrontierCoverageTracker("frontier_uniform")

    seed = tracker.seed_initial_state(state)
    node_units = extract_graph_node_units(state)

    assert seed.reward == 0.0
    assert seed.node_gain == 1
    assert seed.edge_gain == 0
    assert seed.cumulative_F == 1.0
    assert seed.new_unit_set == node_units
    assert all(unit.startswith("node:") for unit in seed.units)
    assert not any(unit.startswith("edge:") for unit in tracker.covered_units)
    assert not any(unit.startswith("action:") for unit in tracker.covered_units)


def test_agent_state_graph_node_uses_assigned_state_list_index():
    old = os.environ.get("WEBTEST_GRAPH_NODE_MODE")
    os.environ["WEBTEST_GRAPH_NODE_MODE"] = "agent_state"
    try:
        action = ClickAction(ElementLocator.XPATH, "//a", "Owners", "redirect", "/owners")
        state_a = _state("http://localhost/", [action])
        state_b = _state("http://localhost/", [action])
        state_c = _state("http://localhost/owners", [action])
        state_a.graph_state_index = 0
        state_b.graph_state_index = 0
        state_c.graph_state_index = 1

        assert extract_graph_node_id(state_a) == extract_graph_node_id(state_b)
        assert extract_graph_node_id(state_a).endswith("|state_idx=0")
        assert extract_graph_node_id(state_c).endswith("|state_idx=1")
        assert extract_graph_node_id(state_a) != extract_graph_node_id(state_c)
    finally:
        _restore_env("WEBTEST_GRAPH_NODE_MODE", old)


def test_a2c_graph_state_identity_matches_state_list_equality():
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.graph_state_list = []

    action = ClickAction(ElementLocator.XPATH, "//a", "Owners", "redirect", "/owners")
    state_a = _state("http://localhost/", [action])
    state_b = _state("http://localhost/", [action])
    state_c = _state("http://localhost/owners", [action])

    agent._ensure_graph_state_identity(state_a)
    agent._ensure_graph_state_identity(state_b)
    agent._ensure_graph_state_identity(state_c)

    assert state_a == state_b
    assert state_a.graph_state_index == 0
    assert state_b.graph_state_index == 0
    assert state_c.graph_state_index == 1
    assert len(agent.graph_state_list) == 2


def test_strict_reward_agent_state_node_zero_on_equal_revisit():
    old = os.environ.get("WEBTEST_GRAPH_NODE_MODE")
    os.environ["WEBTEST_GRAPH_NODE_MODE"] = "agent_state"
    try:
        agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
        agent.graph_state_list = []
        action = ClickAction(ElementLocator.XPATH, "//a", "Target", "redirect", "/target")
        back = ClickAction(ElementLocator.XPATH, "//a", "Back", "redirect", "/source")
        source = _state("http://localhost/source", [action])
        target_a = _state("http://localhost/target", [back])
        target_b = _state("http://localhost/target", [back])

        agent._ensure_graph_state_identity(source)
        agent._ensure_graph_state_identity(target_a)
        agent._ensure_graph_state_identity(target_b)

        source_node = extract_graph_node_id(source)
        tracker = FrontierCoverageTracker("frontier_uniform")
        tracker.seed_initial_state(source)
        first = tracker.compute_strict_marginal_reward(
            target_a,
            chosen_action=action,
            source_url="http://localhost/source",
            source_node_id=source_node,
        )
        second = tracker.compute_strict_marginal_reward(
            target_b,
            chosen_action=action,
            source_url="http://localhost/source",
            source_node_id=source_node,
        )

        assert first.node_gain == 1
        assert first.edge_gain == 1
        assert second.reward == 0.0
        assert second.node_gain == 0
        assert second.edge_gain == 0
        assert tracker.cumulative_F == first.cumulative_F
    finally:
        _restore_env("WEBTEST_GRAPH_NODE_MODE", old)


def test_default_graph_node_matches_observer_state_identity():
    save = ClickAction(ElementLocator.XPATH, "//button", "Save", "submit", "")
    edit = ClickAction(ElementLocator.XPATH, "//a", "Edit", "redirect", "/owners/1/edit")
    state_a = _state("http://localhost/owners/1", [save])
    state_b = _state("http://localhost/owners/1", [save, edit])

    old = os.environ.pop("WEBTEST_GRAPH_NODE_MODE", None)
    try:
        nodes_a = extract_graph_node_units(state_a)
        nodes_b = extract_graph_node_units(state_b)
        node_id_a = extract_graph_node_id(state_a)
        node_id_b = extract_graph_node_id(state_b)
    finally:
        if old is not None:
            os.environ["WEBTEST_GRAPH_NODE_MODE"] = old

    assert len(nodes_a) == 1
    assert len(nodes_b) == 1
    assert nodes_a.isdisjoint(nodes_b)
    assert node_id_a.startswith("/owners/{id}|state=")
    assert node_id_b.startswith("/owners/{id}|state=")


def test_default_graph_node_uses_observer_state_equivalence_class():
    first = ClickAction(ElementLocator.XPATH, "//table/tr[1]/td/a", "Jane Doe", "redirect", "/owners/1")
    second = ClickAction(ElementLocator.XPATH, "//table/tr[2]/td/a", "Sam Smith", "redirect", "/owners/2")
    same_as_first = ClickAction(ElementLocator.XPATH, "//table/tr[1]/td/a", "Jane Doe", "redirect", "/owners/1")
    state_a = _state("http://localhost/owners", [first])
    state_b = _state("http://localhost/owners", [second])
    state_c = _state("http://localhost/owners", [same_as_first])

    old = os.environ.pop("WEBTEST_GRAPH_NODE_MODE", None)
    try:
        observer = Observer("unit", log_dir="observation_logs")
        assert observer._get_state_id(state_a) != observer._get_state_id(state_b)
        assert observer._get_state_id(state_a) == observer._get_state_id(state_c)
        assert extract_graph_node_id(state_a) != extract_graph_node_id(state_b)
        assert extract_graph_node_id(state_a) == extract_graph_node_id(state_c)
    finally:
        if old is not None:
            os.environ["WEBTEST_GRAPH_NODE_MODE"] = old


def test_route_actionset_graph_node_mode_remains_available_as_variant():
    save = ClickAction(ElementLocator.XPATH, "//button", "Save", "submit", "")
    edit = ClickAction(ElementLocator.XPATH, "//a", "Edit", "redirect", "/owners/1/edit")
    state_a = _state("http://localhost/owners/1", [save])
    state_b = _state("http://localhost/owners/1", [save, edit])

    old = os.environ.get("WEBTEST_GRAPH_NODE_MODE")
    os.environ["WEBTEST_GRAPH_NODE_MODE"] = "route_actionset"
    try:
        node_id_a = extract_graph_node_id(state_a)
        node_id_b = extract_graph_node_id(state_b)
    finally:
        if old is None:
            os.environ.pop("WEBTEST_GRAPH_NODE_MODE", None)
        else:
            os.environ["WEBTEST_GRAPH_NODE_MODE"] = old

    assert node_id_a.startswith("/owners/{id}|actions=")
    assert node_id_b.startswith("/owners/{id}|actions=")
    assert node_id_a != node_id_b


def test_route_actionset_graph_nodes_collapse_row_link_variants_while_observer_state_separates_them():
    row_a = ClickAction(ElementLocator.XPATH, "//table/tr[1]/td/a", "Jane Doe", "redirect", "/owners/1")
    row_b = ClickAction(ElementLocator.XPATH, "//table/tr[2]/td/a", "Sam Smith", "redirect", "/owners/2")
    state_a = _state("http://localhost/owners", [row_a])
    state_b = _state("http://localhost/owners", [row_b])

    old = os.environ.get("WEBTEST_GRAPH_NODE_MODE")
    try:
        os.environ["WEBTEST_GRAPH_NODE_MODE"] = "route_actionset"
        route_actionset_node_a = extract_graph_node_id(state_a)
        route_actionset_node_b = extract_graph_node_id(state_b)
        os.environ["WEBTEST_GRAPH_NODE_MODE"] = "observer_state"
        observer_node_a = extract_graph_node_id(state_a)
        observer_node_b = extract_graph_node_id(state_b)
    finally:
        if old is None:
            os.environ.pop("WEBTEST_GRAPH_NODE_MODE", None)
        else:
            os.environ["WEBTEST_GRAPH_NODE_MODE"] = old

    assert route_actionset_node_a == route_actionset_node_b
    assert observer_node_a != observer_node_b


def test_template_navigation_edge_is_source_specific_by_default():
    home = ClickAction(
        ElementLocator.XPATH,
        "/html/body/nav/div/ul/li[1]/a",
        "HOME",
        "redirect",
        "http://localhost/",
    )
    owners = _state("http://localhost/owners", [home])
    vets = _state("http://localhost/vets.html", [home])
    root = _state("http://localhost/", [home])
    tracker = FrontierCoverageTracker("frontier_uniform")
    tracker.seed_initial_state(owners)

    first = tracker.compute_strict_marginal_reward(
        root,
        chosen_action=home,
        source_url="http://localhost/owners",
        source_node_id=extract_graph_node_id(owners),
    )
    second = tracker.compute_strict_marginal_reward(
        root,
        chosen_action=home,
        source_url="http://localhost/vets.html",
        source_node_id=extract_graph_node_id(vets),
    )

    assert first.edge_gain == 1
    assert second.edge_gain == 1
    assert not any(edge.startswith("edge:template_reset|") for edge in tracker.covered_edges)
    assert any(extract_graph_node_id(owners) in edge for edge in tracker.covered_edges)
    assert any(extract_graph_node_id(vets) in edge for edge in tracker.covered_edges)


def test_template_navigation_global_scope_is_explicit_variant():
    home = ClickAction(
        ElementLocator.XPATH,
        "/html/body/nav/div/ul/li[1]/a",
        "HOME",
        "redirect",
        "http://localhost/",
    )
    owners = _state("http://localhost/owners", [home])
    vets = _state("http://localhost/vets.html", [home])
    root = _state("http://localhost/", [home])
    old_scope = os.environ.get("WEBTEST_GRAPH_TEMPLATE_EDGE_SCOPE")
    try:
        os.environ["WEBTEST_GRAPH_TEMPLATE_EDGE_SCOPE"] = "global"
        tracker = FrontierCoverageTracker("frontier_uniform")
        tracker.seed_initial_state(owners)
        first = tracker.compute_strict_marginal_reward(
            root,
            chosen_action=home,
            source_url="http://localhost/owners",
            source_node_id=extract_graph_node_id(owners),
        )
        second = tracker.compute_strict_marginal_reward(
            root,
            chosen_action=home,
            source_url="http://localhost/vets.html",
            source_node_id=extract_graph_node_id(vets),
        )
    finally:
        _restore_env("WEBTEST_GRAPH_TEMPLATE_EDGE_SCOPE", old_scope)

    assert first.edge_gain == 1
    assert second.edge_gain == 0
    assert any(edge.startswith("edge:template_reset|") for edge in tracker.covered_edges)


def test_graph_node_is_one_unit_not_one_unit_per_visible_action():
    save = ClickAction(ElementLocator.XPATH, "//button", "Save", "submit", "")
    edit = ClickAction(ElementLocator.XPATH, "//a", "Edit", "redirect", "/owners/1/edit")
    delete = ClickAction(ElementLocator.XPATH, "//a", "Delete", "redirect", "/owners/1/delete")
    state = _state("http://localhost/owners/1", [save, edit, delete])
    tracker = FrontierCoverageTracker("frontier_uniform")

    seed = tracker.seed_initial_state(state)

    assert len(seed.units) == 1
    assert seed.node_gain == 1
    assert seed.edge_gain == 0
    assert seed.reward == 0.0


def test_visible_but_unexecuted_action_does_not_cover_graph_edge():
    login = ClickAction(ElementLocator.XPATH, "//form/button", "Login", "submit", "")
    help_link = ClickAction(ElementLocator.XPATH, "//a", "Help", "redirect", "/help")
    source = _state("http://localhost/login", [login, help_link])
    target = _state("http://localhost/dashboard", [])
    tracker = FrontierCoverageTracker("frontier_uniform")
    source_node = extract_graph_node_id(source)

    tracker.seed_initial_state(source)
    reward = tracker.compute_strict_marginal_reward(
        target,
        chosen_action=login,
        source_url="http://localhost/login",
        source_node_id=source_node,
    )

    help_edge_units = extract_graph_edge_units(
        "http://localhost/login",
        help_link,
        _state("http://localhost/help", []),
        source_node_id=source_node,
    )

    assert reward.edge_gain == 1
    assert all(unit not in tracker.covered_edges for unit in help_edge_units)
    assert all("target_context=/help" not in unit for unit in tracker.covered_edges)


def test_frontier_units_keep_same_action_context_specific():
    login_a = ClickAction(ElementLocator.XPATH, "//form/button", "Login", "submit", "")
    login_b = ClickAction(ElementLocator.XPATH, "//form/button", "Login", "submit", "")
    state_a = _state("http://localhost/login", [login_a])
    state_b = _state("http://localhost/admin/login", [login_b])

    units_a = extract_frontier_units(state_a)
    units_b = extract_frontier_units(state_b)

    assert any(unit.startswith("frontier:/login|") and "login" in unit for unit in units_a)
    assert any(unit.startswith("frontier:/admin/login|") and "login" in unit for unit in units_b)
    assert units_a.isdisjoint(units_b)


def test_arrival_units_do_not_cover_visible_actions():
    login = ClickAction(ElementLocator.XPATH, "//form/button", "Login", "submit", "")
    help_link = ClickAction(ElementLocator.XPATH, "//a", "Help", "redirect", "/help")
    state = _state("http://localhost/login", [login, help_link])

    arrival_units = extract_arrival_units(state)
    legacy_frontier_units = extract_frontier_units(state)

    assert "context:/login" in arrival_units
    assert any(unit.startswith("pageinfo:/login|") for unit in arrival_units)
    assert not any(unit.startswith("frontier:/login|") for unit in arrival_units)
    assert not any(unit.startswith("action:/login|") for unit in arrival_units)
    assert any(unit.startswith("frontier:/login|") for unit in legacy_frontier_units)


def test_executed_state_action_edge_is_covered_once():
    submit = ClickAction(ElementLocator.XPATH, "//form/button", "Find Owner", "submit", "")
    source = _state("http://localhost/owners/find", [submit])
    owner_link = ClickAction(ElementLocator.XPATH, "//td/a", "Jane Doe", "redirect", "/owners/1")
    target = _state("http://localhost/owners", [owner_link])
    tracker = FrontierCoverageTracker("frontier_uniform")
    tracker.seed_initial_state(source)

    first = tracker.compute_strict_marginal_reward(
        target,
        chosen_action=submit,
        source_url="http://localhost/owners/find",
    )
    second = tracker.compute_strict_marginal_reward(
        target,
        chosen_action=submit,
        source_url="http://localhost/owners/find",
    )

    assert first.reward > 0.0
    assert first.node_gain == 1
    assert first.edge_gain == 1
    assert any(unit.startswith("edge:/owners/find|") for unit in first.new_unit_set)
    assert second.reward == 0.0


def test_graph_reward_is_node_plus_alpha_edge():
    submit = ClickAction(ElementLocator.XPATH, "//form/button", "Find Owner", "submit", "")
    target = _state("http://localhost/owners", [])
    old_alpha = os.environ.pop("WEBTEST_GRAPH_EDGE_ALPHA", None)
    try:
        tracker = FrontierCoverageTracker("frontier_uniform")
        tracker.seed_initial_state(_state("http://localhost/owners/find", [submit]))

        reward = tracker.compute_strict_marginal_reward(
            target,
            chosen_action=submit,
            source_url="http://localhost/owners/find",
        )
    finally:
        if old_alpha is not None:
            os.environ["WEBTEST_GRAPH_EDGE_ALPHA"] = old_alpha

    assert tracker.graph_edge_alpha == 1.0
    assert tracker.graph_edge_weight_mode == "home_zero"
    assert reward.node_gain == 1
    assert reward.edge_gain == 1
    assert reward.reward == 1.0 + tracker.graph_edge_alpha
    assert reward.cumulative_F == 2.0 + tracker.graph_edge_alpha


def test_unknown_edge_weight_modes_collapse_to_home_zero():
    nav = ClickAction(ElementLocator.XPATH, "/html/body/nav/a", "Home", "redirect", "/")
    source = _state("http://localhost/source", [nav])
    target = _state("http://localhost/", [])
    old_mode = os.environ.get("WEBTEST_GRAPH_EDGE_WEIGHT_MODE")
    os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = "webtest_experimental"
    try:
        tracker = FrontierCoverageTracker("frontier_uniform")
        tracker.seed_initial_state(source)
        reward = tracker.compute_strict_marginal_reward(
            target,
            chosen_action=nav,
            source_url="http://localhost/source",
            source_node_id=extract_graph_node_id(source),
        )
    finally:
        if old_mode is None:
            os.environ.pop("WEBTEST_GRAPH_EDGE_WEIGHT_MODE", None)
        else:
            os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = old_mode

    assert tracker.graph_edge_weight_mode == "home_zero"
    assert reward.node_gain == 1
    assert reward.edge_gain == 1
    assert reward.node_weight_gain == 1.0
    assert reward.edge_weight_gain == 0.0
    assert abs(reward.reward - 1.0) < 1e-9


def test_uniform_edge_weight_mode_keeps_home_edges_weighted_for_diagnostics():
    nav = ClickAction(ElementLocator.XPATH, "/html/body/nav/a", "Home", "redirect", "/")
    source = _state("http://localhost/source", [nav])
    target = _state("http://localhost/", [])
    old_mode = os.environ.get("WEBTEST_GRAPH_EDGE_WEIGHT_MODE")
    os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = "uniform"
    try:
        tracker = FrontierCoverageTracker("frontier_uniform")
        tracker.seed_initial_state(source)
        reward = tracker.compute_strict_marginal_reward(
            target,
            chosen_action=nav,
            source_url="http://localhost/source",
            source_node_id=extract_graph_node_id(source),
        )
    finally:
        if old_mode is None:
            os.environ.pop("WEBTEST_GRAPH_EDGE_WEIGHT_MODE", None)
        else:
            os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = old_mode

    assert tracker.graph_edge_weight_mode == "uniform"
    assert reward.node_gain == 1
    assert reward.edge_gain == 1
    assert reward.node_weight_gain == 1.0
    assert reward.edge_weight_gain == 1.0
    assert abs(reward.reward - 2.0) < 1e-9


def test_source_degree_capped_edge_weight_bounds_each_source_frontier():
    actions = [
        ClickAction(ElementLocator.XPATH, "//a[1]", "A", "redirect", "/a"),
        ClickAction(ElementLocator.XPATH, "//a[2]", "B", "redirect", "/b"),
        ClickAction(ElementLocator.XPATH, "//button", "Toggle", "default", "button"),
    ]
    source = _state("http://localhost/source", actions)
    target_a = _state("http://localhost/a", [])
    target_b = _state("http://localhost/b", [])
    old_mode = os.environ.get("WEBTEST_GRAPH_EDGE_WEIGHT_MODE")
    os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = "source_degree_capped"
    try:
        tracker = FrontierCoverageTracker("frontier_uniform")
        tracker.seed_initial_state(source)
        source_node = extract_graph_node_id(source)
        tracker.register_graph_source_degree(source_node, len(actions))
        reward_a = tracker.compute_strict_marginal_reward(
            target_a,
            chosen_action=actions[0],
            source_url="http://localhost/source",
            source_node_id=source_node,
            source_action_count=len(actions),
        )
        reward_b = tracker.compute_strict_marginal_reward(
            target_b,
            chosen_action=actions[1],
            source_url="http://localhost/source",
            source_node_id=source_node,
            source_action_count=len(actions),
        )
    finally:
        if old_mode is None:
            os.environ.pop("WEBTEST_GRAPH_EDGE_WEIGHT_MODE", None)
        else:
            os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = old_mode

    assert tracker.graph_edge_weight_mode == "source_degree"
    assert abs(reward_a.edge_weight_gain - (1.0 / 3.0)) < 1e-9
    assert abs(reward_b.edge_weight_gain - (1.0 / 3.0)) < 1e-9
    assert abs(reward_a.reward - (1.0 + 1.0 / 3.0)) < 1e-9
    assert abs(reward_b.reward - (1.0 + 1.0 / 3.0)) < 1e-9
    assert abs(tracker.cumulative_F - (3.0 + 2.0 / 3.0)) < 1e-9


def test_source_degree_alias_edge_weight_bounds_each_source_frontier():
    actions = [
        ClickAction(ElementLocator.XPATH, "//a[1]", "A", "redirect", "/a"),
        ClickAction(ElementLocator.XPATH, "//a[2]", "B", "redirect", "/b"),
        ClickAction(ElementLocator.XPATH, "//button", "Toggle", "default", "button"),
    ]
    source = _state("http://localhost/source", actions)
    target = _state("http://localhost/a", [])
    old_mode = os.environ.get("WEBTEST_GRAPH_EDGE_WEIGHT_MODE")
    os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = "source_degree"
    try:
        tracker = FrontierCoverageTracker("frontier_uniform")
        tracker.seed_initial_state(source)
        source_node = extract_graph_node_id(source)
        reward = tracker.compute_strict_marginal_reward(
            target,
            chosen_action=actions[0],
            source_url="http://localhost/source",
            source_node_id=source_node,
            source_action_count=len(actions),
        )
    finally:
        if old_mode is None:
            os.environ.pop("WEBTEST_GRAPH_EDGE_WEIGHT_MODE", None)
        else:
            os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = old_mode

    assert tracker.graph_edge_weight_mode == "source_degree"
    assert abs(reward.edge_weight_gain - (1.0 / 3.0)) < 1e-9
    assert abs(reward.reward - (1.0 + 1.0 / 3.0)) < 1e-9


def test_action_coverage_features_expose_weighted_graph_marginal():
    actions = [
        ClickAction(
            ElementLocator.XPATH,
            "//a[1]",
            "Home",
            "redirect",
            "http://localhost/",
        ),
        ClickAction(
            ElementLocator.XPATH,
            "//a[2]",
            "Find Owners",
            "redirect",
            "http://localhost/owners/find",
        ),
    ]
    source = _state("http://localhost/source", actions)
    target = _state("http://localhost/owners/find", [])
    old_mode = os.environ.get("WEBTEST_GRAPH_EDGE_WEIGHT_MODE")
    os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = "home_zero"
    try:
        tracker = FrontierCoverageTracker("frontier_uniform")
        tracker.seed_initial_state(source)
        provider = SubWebActionCoverageFeatures(tracker)
        source_node = extract_graph_node_id(source)
        home_before = provider.features_for(actions[0], source, actions, source_node_id=source_node)
        forward_before = provider.features_for(actions[1], source, actions, source_node_id=source_node)
        tracker.compute_strict_marginal_reward(
            target,
            chosen_action=actions[1],
            source_url="http://localhost/source",
            source_node_id=source_node,
        )
        forward_after = provider.features_for(actions[1], source, actions, source_node_id=source_node)
    finally:
        if old_mode is None:
            os.environ.pop("WEBTEST_GRAPH_EDGE_WEIGHT_MODE", None)
        else:
            os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = old_mode

    edge_idx = ACTION_COVERAGE_FEATURE_NAMES.index("estimated_edge_marginal_norm")
    node_idx = ACTION_COVERAGE_FEATURE_NAMES.index("estimated_target_node_marginal_norm")
    total_idx = ACTION_COVERAGE_FEATURE_NAMES.index("estimated_graph_marginal_norm")

    assert home_before[edge_idx].item() == 0.0
    assert forward_before[edge_idx].item() == 1.0
    assert forward_before[node_idx].item() == 1.0
    assert forward_before[total_idx].item() == 1.0
    assert forward_after[edge_idx].item() == 0.0
    assert forward_after[node_idx].item() == 0.0
    assert forward_after[total_idx].item() == 0.0


def test_action_coverage_feature_rows_match_per_action_features():
    actions = [
        ClickAction(ElementLocator.XPATH, "//a[1]", "Home", "redirect", "http://localhost/"),
        ClickAction(
            ElementLocator.XPATH,
            "//a[2]",
            "Find Owners",
            "redirect",
            "http://localhost/owners/find",
        ),
        ClickAction(ElementLocator.XPATH, "//button", "Find Owner", "submit", ""),
    ]
    source = _state("http://localhost/source", actions)
    tracker = FrontierCoverageTracker("frontier_uniform")
    tracker.seed_initial_state(source)
    source_node = extract_graph_node_id(source)

    per_action_provider = SubWebActionCoverageFeatures(tracker)
    per_action_rows = [
        per_action_provider.features_for(action, source, actions, source_node_id=source_node)
        for action in actions
    ]

    batch_provider = SubWebActionCoverageFeatures(tracker)
    batch_rows = batch_provider.feature_rows_for_actions(
        source,
        actions,
        source_node_id=source_node,
    )

    assert len(batch_rows) == len(per_action_rows)
    for expected, actual in zip(per_action_rows, batch_rows):
        assert torch.allclose(actual, expected)


def test_action_coverage_features_use_empirical_target_for_estimated_marginal():
    submit = ClickAction(ElementLocator.XPATH, "//form/button", "Find Owner", "submit", "")
    source = _state("http://localhost/owners/find", [submit])
    target = _state("http://localhost/owners", [])
    tracker = FrontierCoverageTracker("frontier_uniform")
    tracker.seed_initial_state(source)
    provider = SubWebActionCoverageFeatures(tracker)
    source_node = extract_graph_node_id(source)

    provider.update(
        source_node,
        "http://localhost/owners/find",
        submit,
        reward=0.0,
        target_state=target,
    )
    features = provider.features_for(submit, source, [submit], source_node_id=source_node)

    node_idx = ACTION_COVERAGE_FEATURE_NAMES.index("estimated_target_node_marginal_norm")
    total_idx = ACTION_COVERAGE_FEATURE_NAMES.index("estimated_graph_marginal_norm")
    assert features[node_idx].item() == 1.0
    assert features[total_idx].item() > 0.0


def test_legacy_webtest_edge_weight_mode_collapses_to_home_zero():
    nav = ClickAction(ElementLocator.XPATH, "/html/body/nav/div/ul/li/a", "Home", "redirect", "/")
    source = _state("http://localhost/owners/find", [nav])
    target = _state("http://localhost/", [])
    old_mode = os.environ.get("WEBTEST_GRAPH_EDGE_WEIGHT_MODE")
    os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = "webtest"
    try:
        tracker = FrontierCoverageTracker("frontier_uniform")
        tracker.seed_initial_state(source)
        reward = tracker.compute_strict_marginal_reward(
            target,
            chosen_action=nav,
            source_url="http://localhost/owners/find",
        )
    finally:
        if old_mode is None:
            os.environ.pop("WEBTEST_GRAPH_EDGE_WEIGHT_MODE", None)
        else:
            os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = old_mode

    assert tracker.graph_edge_weight_mode == "home_zero"
    assert reward.node_gain == 1
    assert reward.edge_gain == 1
    assert reward.node_weight_gain == 1.0
    assert reward.edge_weight_gain == 0.0
    assert abs(reward.reward - 1.0) < 1e-9


def test_home_zero_edge_weight_records_home_edge_without_rewarding_it():
    nav = ClickAction(ElementLocator.XPATH, "/html/body/nav/div/ul/li/a", "Home", "redirect", "/")
    source = _state("http://localhost/owners/find", [nav])
    target = _state("http://localhost/", [])
    old_mode = os.environ.get("WEBTEST_GRAPH_EDGE_WEIGHT_MODE")
    os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = "home_zero"
    try:
        tracker = FrontierCoverageTracker("frontier_uniform")
        tracker.seed_initial_state(source)
        reward = tracker.compute_strict_marginal_reward(
            target,
            chosen_action=nav,
            source_url="http://localhost/owners/find",
        )
    finally:
        if old_mode is None:
            os.environ.pop("WEBTEST_GRAPH_EDGE_WEIGHT_MODE", None)
        else:
            os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = old_mode

    assert tracker.graph_edge_weight_mode == "home_zero"
    assert reward.node_gain == 1
    assert reward.edge_gain == 1
    assert reward.node_weight_gain == 1.0
    assert reward.edge_weight_gain == 0.0
    assert abs(reward.reward - 1.0) < 1e-9
    assert any("target=/" in edge and "target_context=/" in edge for edge in reward.units if edge.startswith("edge:"))


def test_home_zero_edge_weight_keeps_non_home_edges_uniform():
    nav = ClickAction(ElementLocator.XPATH, "/html/body/nav/div/ul/li/a", "Find Owners", "redirect", "/owners/find")
    source = _state("http://localhost/", [nav])
    target = _state("http://localhost/owners/find", [])
    old_mode = os.environ.get("WEBTEST_GRAPH_EDGE_WEIGHT_MODE")
    os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = "home_zero"
    try:
        tracker = FrontierCoverageTracker("frontier_uniform")
        tracker.seed_initial_state(source)
        reward = tracker.compute_strict_marginal_reward(
            target,
            chosen_action=nav,
            source_url="http://localhost/",
        )
    finally:
        if old_mode is None:
            os.environ.pop("WEBTEST_GRAPH_EDGE_WEIGHT_MODE", None)
        else:
            os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = old_mode

    assert tracker.graph_edge_weight_mode == "home_zero"
    assert reward.node_gain == 1
    assert reward.edge_gain == 1
    assert reward.node_weight_gain == 1.0
    assert reward.edge_weight_gain == 1.0
    assert abs(reward.reward - 2.0) < 1e-9


def test_home_zero_keeps_spa_root_state_change_edges_weighted():
    button = ClickAction(ElementLocator.XPATH, "//button", "Continue", "default", "button")
    source = _state("http://localhost/", [button])
    target = _state("http://localhost/", [button])
    source.graph_form_state_id = "spa=root|modal=closed"
    target.graph_form_state_id = "spa=root|modal=open"
    old_values = {
        key: os.environ.get(key)
        for key in ("WEBTEST_GRAPH_EDGE_WEIGHT_MODE", "WEBTEST_GRAPH_FORM_STATE")
    }
    os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = "home_zero"
    os.environ["WEBTEST_GRAPH_FORM_STATE"] = "1"
    try:
        tracker = FrontierCoverageTracker("frontier_uniform")
        tracker.seed_initial_state(source)
        reward = tracker.compute_strict_marginal_reward(
            target,
            chosen_action=button,
            source_url="http://localhost/",
            source_node_id=extract_graph_node_id(source),
        )
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert any("target_context=/" in edge for edge in reward.units if edge.startswith("edge:"))
    assert reward.node_gain == 1
    assert reward.edge_gain == 1
    assert reward.edge_weight_gain == 1.0
    assert abs(reward.reward - 2.0) < 1e-9


def test_home_zero_keeps_spa_hash_route_navigation_weighted():
    nav = ClickAction(ElementLocator.XPATH, "//a", "Collections", "redirect", "http://localhost/")
    source = _state("http://localhost/#/home", [nav])
    target = _state("http://localhost/#/admin/collections", [])
    old_mode = os.environ.get("WEBTEST_GRAPH_EDGE_WEIGHT_MODE")
    os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = "home_zero"
    try:
        tracker = FrontierCoverageTracker("frontier_uniform")
        tracker.seed_initial_state(source)
        reward = tracker.compute_strict_marginal_reward(
            target,
            chosen_action=nav,
            source_url="http://localhost/#/home",
            source_node_id=extract_graph_node_id(source),
        )
    finally:
        if old_mode is None:
            os.environ.pop("WEBTEST_GRAPH_EDGE_WEIGHT_MODE", None)
        else:
            os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = old_mode

    assert any("target_context=/#/admin/collections" in edge for edge in reward.units if edge.startswith("edge:"))
    assert reward.node_gain == 1
    assert reward.edge_gain == 1
    assert reward.edge_weight_gain == 1.0
    assert abs(reward.reward - 2.0) < 1e-9


def test_submit_same_route_edge_uses_uniform_non_home_weight():
    submit = ClickAction(ElementLocator.XPATH, "//form/button", "Add Pet", "submit", "3")
    source = _state("http://localhost/owners/1/pets/new", [submit])
    target = _state("http://localhost/owners/1/pets/new", [submit])
    source.graph_form_state_id = "fields=3|filled=1|required=3|required_filled=1|select_changed=0|sig=a"
    target.graph_form_state_id = "fields=3|filled=2|required=3|required_filled=2|select_changed=0|sig=b"
    old_values = {
        key: os.environ.get(key)
        for key in ("WEBTEST_GRAPH_EDGE_WEIGHT_MODE", "WEBTEST_GRAPH_FORM_STATE")
    }
    os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = "home_zero"
    os.environ["WEBTEST_GRAPH_FORM_STATE"] = "1"
    try:
        tracker = FrontierCoverageTracker("frontier_uniform")
        tracker.seed_initial_state(source)
        reward = tracker.compute_strict_marginal_reward(
            target,
            chosen_action=submit,
            source_url="http://localhost/owners/1/pets/new",
            source_node_id=extract_graph_node_id(source),
        )
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert any("|outcome=same_route" in edge for edge in reward.units if edge.startswith("edge:"))
    assert reward.node_gain == 1
    assert reward.edge_gain == 1
    assert abs(reward.edge_weight_gain - 1.0) < 1e-9


def test_submit_route_change_edge_keeps_primary_outcome_weight():
    submit = ClickAction(ElementLocator.XPATH, "//form/button", "Add Pet", "submit", "3")
    source = _state("http://localhost/owners/1/pets/new", [submit])
    target = _state("http://localhost/owners/1", [])
    old_values = {key: os.environ.get(key) for key in ("WEBTEST_GRAPH_EDGE_WEIGHT_MODE",)}
    os.environ["WEBTEST_GRAPH_EDGE_WEIGHT_MODE"] = "home_zero"
    try:
        tracker = FrontierCoverageTracker("frontier_uniform")
        tracker.seed_initial_state(source)
        reward = tracker.compute_strict_marginal_reward(
            target,
            chosen_action=submit,
            source_url="http://localhost/owners/1/pets/new",
            source_node_id=extract_graph_node_id(source),
        )
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert any("|outcome=route_change" in edge for edge in reward.units if edge.startswith("edge:"))
    assert reward.node_gain == 1
    assert reward.edge_gain == 1
    assert abs(reward.edge_weight_gain - 1.0) < 1e-9


def test_click_form_control_is_form_field_not_plain_secondary_click():
    action = ClickAction(
        ElementLocator.XPATH,
        "/html/body/div/form/label/input",
        "",
        "default",
        "input",
    )
    state = _state("http://localhost/#/admin/department-edit", [action])

    signature = abstract_action_signature(action, "/#/admin/department-edit")
    role = action_role_features(action, state, [action])
    structural = structural_action_features(action, state, [action])
    structural_names = structural_action_feature_names()

    assert "family=form_field" in signature
    assert "field=" in signature
    assert not signature.endswith("field=")
    assert role[6].item() == 1.0
    assert structural[structural_names.index("likely_self_loop")].item() == 0.0


def test_distinct_click_form_controls_are_distinct_graph_edges():
    first = ClickAction(
        ElementLocator.XPATH,
        "/html/body/div/form/label[1]/input",
        "",
        "default",
        "input",
    )
    second = ClickAction(
        ElementLocator.XPATH,
        "/html/body/div/form/label[2]/input",
        "",
        "default",
        "input",
    )
    state = _state("http://localhost/#/admin/department-edit", [first, second])
    source_node = extract_graph_node_id(state)

    sig_first = abstract_action_signature(first, "/#/admin/department-edit")
    sig_second = abstract_action_signature(second, "/#/admin/department-edit")

    tracker = FrontierCoverageTracker("frontier_uniform")
    tracker.seed_initial_state(state)
    reward_first = tracker.compute_strict_marginal_reward(
        state,
        chosen_action=first,
        source_url="http://localhost/#/admin/department-edit",
        source_node_id=source_node,
    )
    reward_second = tracker.compute_strict_marginal_reward(
        state,
        chosen_action=second,
        source_url="http://localhost/#/admin/department-edit",
        source_node_id=source_node,
    )

    assert sig_first != sig_second
    assert reward_first.node_gain == 0
    assert reward_first.edge_gain == 1
    assert reward_second.node_gain == 0
    assert reward_second.edge_gain == 1


def test_template_nav_is_kept_as_graph_edge_by_default():
    nav = ClickAction(ElementLocator.XPATH, "/html/body/nav/div/ul/li/a", "Home", "redirect", "/")
    target = _state("http://localhost/", [])

    old = os.environ.pop("WEBTEST_GRAPH_FILTER_TEMPLATE_EDGES", None)
    try:
        units = extract_graph_edge_units(
            "http://localhost/owners/find",
            nav,
            target,
        )
    finally:
        if old is not None:
            os.environ["WEBTEST_GRAPH_FILTER_TEMPLATE_EDGES"] = old

    assert not any(unit.startswith("edge:template_reset|") for unit in units)
    assert any(unit.startswith("edge:/owners/find|") for unit in units)
    assert any("|target_node=/" in unit for unit in units)
    assert any("|target_context=/" in unit for unit in units)


def test_template_nav_can_use_global_edge_scope_as_explicit_variant():
    nav = ClickAction(ElementLocator.XPATH, "/html/body/nav/div/ul/li/a", "Home", "redirect", "/")
    target = _state("http://localhost/", [])

    old_filter = os.environ.pop("WEBTEST_GRAPH_FILTER_TEMPLATE_EDGES", None)
    old_scope = os.environ.get("WEBTEST_GRAPH_TEMPLATE_EDGE_SCOPE")
    try:
        os.environ["WEBTEST_GRAPH_TEMPLATE_EDGE_SCOPE"] = "global"
        units = extract_graph_edge_units(
            "http://localhost/owners/find",
            nav,
            target,
        )
    finally:
        _restore_env("WEBTEST_GRAPH_FILTER_TEMPLATE_EDGES", old_filter)
        _restore_env("WEBTEST_GRAPH_TEMPLATE_EDGE_SCOPE", old_scope)

    assert any(unit.startswith("edge:template_reset|") for unit in units)
    assert any("|target_context=/" in unit for unit in units)


def test_template_nav_can_be_filtered_as_explicit_variant():
    nav = ClickAction(ElementLocator.XPATH, "/html/body/nav/div/ul/li/a", "Home", "redirect", "/")
    target = _state("http://localhost/", [])

    old = os.environ.get("WEBTEST_GRAPH_FILTER_TEMPLATE_EDGES")
    os.environ["WEBTEST_GRAPH_FILTER_TEMPLATE_EDGES"] = "1"
    try:
        units = extract_graph_edge_units(
            "http://localhost/owners/find",
            nav,
            target,
        )
    finally:
        if old is None:
            os.environ.pop("WEBTEST_GRAPH_FILTER_TEMPLATE_EDGES", None)
        else:
            os.environ["WEBTEST_GRAPH_FILTER_TEMPLATE_EDGES"] = old

    assert units == set()


def test_local_workflow_link_still_gets_transition_coverage_with_template_filter_default():
    add_owner = ClickAction(ElementLocator.XPATH, "/html/body/div/div/a", "Add Owner", "redirect", "/owners/new")
    target = _state("http://localhost/owners/new", [])

    units = extract_graph_edge_units(
        "http://localhost/owners/find",
        add_owner,
        target,
    )

    assert any(unit.startswith("edge:/owners/find|") for unit in units)


def test_abstract_action_units_collapse_repeated_row_links():
    row_a = ClickAction(ElementLocator.XPATH, "//table/tr[1]/td/a", "Jane Doe", "redirect", "/owners/1")
    row_b = ClickAction(ElementLocator.XPATH, "//table/tr[2]/td/a", "Sam Smith", "redirect", "/owners/2")
    state = _state("http://localhost/owners", [row_a, row_b])

    action_units = {
        unit
        for unit in extract_frontier_units(state)
        if unit.startswith("action:/owners|") and "target=/owners/{id}" in unit
    }

    assert len(action_units) == 1


def test_action_ranking_prioritizes_form_actions_before_navigation():
    nav = ClickAction(ElementLocator.XPATH, "//nav/a", "Home", "redirect", "/")
    submit = ClickAction(ElementLocator.XPATH, "//form/button", "Save", "submit", "")
    input_action = RandomInputAction(ElementLocator.XPATH, "//form/input", "Name")
    actions = [nav, submit, input_action]
    state = SimpleNamespace(url="http://localhost/app", action_dict={a: {"execution_time": 0} for a in actions})

    ranked = deterministic_rank_actions(actions, state)

    assert ranked[0] == input_action
    assert ranked[1] == submit
    assert ranked[-1] == nav


def test_padded_batch_truncates_and_logs_policy_counts():
    actions = [
        RandomInputAction(ElementLocator.XPATH, "//form/input", "Name"),
        ClickAction(ElementLocator.XPATH, "//form/button", "Save", "submit", ""),
        ClickAction(ElementLocator.XPATH, "//nav/a", "Home", "redirect", "/"),
    ]
    state = _state("http://localhost/app", actions)

    batch = build_padded_action_batch(
        web_state=state,
        html="<html></html>",
        transformer=_TinyTransformer(),
        max_actions=2,
        input_dim=81 + structural_feature_dim(),
    )

    assert batch.action_mat.shape == (2, 81 + structural_feature_dim())
    assert batch.action_mask.tolist() == [True, True]
    assert batch.original_action_count == 3
    assert batch.kept_action_count == 2
    assert batch.truncated_count == 1
    assert batch.truncated_by_type == {"redirect": 1}


def test_action_coverage_features_are_appended_to_action_tensor():
    action = ClickAction(ElementLocator.XPATH, "/html/body/div/a", "Add Pet", "redirect", "/owners/1/pets/new")
    state = _state("http://localhost/owners/1", [action])
    tracker = FrontierCoverageTracker("frontier_uniform")
    tracker.seed_initial_state(state)
    provider = SubWebActionCoverageFeatures(tracker)

    batch = build_padded_action_batch(
        web_state=state,
        html="<html></html>",
        transformer=_TinyTransformer(),
        max_actions=2,
        input_dim=81 + structural_feature_dim() + action_coverage_feature_dim(),
        coverage_feature_provider=provider,
        source_node_id=extract_graph_node_id(state),
    )

    assert batch.action_mat.shape == (
        2,
        81 + structural_feature_dim() + action_coverage_feature_dim(),
    )
    appended = batch.action_mat[0, -action_coverage_feature_dim():]
    unseen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("edge_unseen_from_current_node")
    assert appended[unseen_idx].item() == 1.0


def test_action_coverage_features_do_not_mark_seen_home_zero_edge_as_residual():
    home = ClickAction(ElementLocator.XPATH, "//nav/a", "Home", "redirect", "/")
    local = ClickAction(ElementLocator.XPATH, "//main/a", "New Transaction", "redirect", "/transactions/new")
    state = _state("http://localhost/new", [home, local])
    tracker = FrontierCoverageTracker("frontier_uniform")
    tracker.seed_initial_state(state)
    tracker.covered_nodes.add("node:/")
    provider = SubWebActionCoverageFeatures(tracker)
    source_node = extract_graph_node_id(state)

    home_features = provider.features_for(home, state, state.get_action_list(), source_node_id=source_node)
    local_features = provider.features_for(local, state, state.get_action_list(), source_node_id=source_node)

    unseen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("edge_unseen_from_current_node")
    edge_marginal_idx = ACTION_COVERAGE_FEATURE_NAMES.index("estimated_edge_marginal_norm")
    node_marginal_idx = ACTION_COVERAGE_FEATURE_NAMES.index("estimated_target_node_marginal_norm")
    graph_marginal_idx = ACTION_COVERAGE_FEATURE_NAMES.index("estimated_graph_marginal_norm")
    untried_ratio_idx = ACTION_COVERAGE_FEATURE_NAMES.index("context_untried_action_ratio")
    residual_count_idx = ACTION_COVERAGE_FEATURE_NAMES.index("source_residual_count_log")

    assert home_features[unseen_idx].item() == 0.0
    assert home_features[edge_marginal_idx].item() == 0.0
    assert home_features[node_marginal_idx].item() == 0.0
    assert home_features[graph_marginal_idx].item() == 0.0
    assert local_features[unseen_idx].item() == 1.0
    assert local_features[graph_marginal_idx].item() > 0.0
    assert abs(home_features[untried_ratio_idx].item() - 0.5) < 1e-6
    assert abs(home_features[residual_count_idx].item() - (math.log1p(1.0) / math.log(32.0))) < 1e-6


def test_action_coverage_features_track_zero_gain_without_changing_logits():
    home = ClickAction(ElementLocator.XPATH, "/html/body/nav/a", "Home", "redirect", "/")
    add_pet = ClickAction(ElementLocator.XPATH, "/html/body/div/a", "Add Pet", "redirect", "/owners/1/pets/new")
    state = _state("http://localhost/owners/1", [home, add_pet])
    tracker = FrontierCoverageTracker("frontier_uniform")
    tracker.seed_initial_state(state)
    provider = SubWebActionCoverageFeatures(tracker)
    source_node = extract_graph_node_id(state)
    home_prefix = graph_edge_prefix("http://localhost/owners/1", home, source_node_id=source_node)
    tracker.covered_edges.add(f"{home_prefix}target_node={source_node}")
    provider.update(source_node, "http://localhost/owners/1", home, reward=0.0)

    batch = build_padded_action_batch(
        web_state=state,
        html="<html></html>",
        transformer=_TinyTransformer(),
        max_actions=4,
        input_dim=81 + structural_feature_dim() + action_coverage_feature_dim(),
        coverage_feature_provider=provider,
        source_node_id=source_node,
    )
    home_idx = batch.actions_policy.index(home)
    home_features = batch.action_mat[home_idx, -action_coverage_feature_dim():]
    edge_seen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("edge_seen_from_current_node")
    zero_idx = ACTION_COVERAGE_FEATURE_NAMES.index("zero_gain_count_log")
    last_zero_idx = ACTION_COVERAGE_FEATURE_NAMES.index("last_action_zero_gain")

    assert home_features[edge_seen_idx].item() == 1.0
    assert home_features[zero_idx].item() > 0.0
    assert home_features[last_zero_idx].item() == 1.0


def test_action_coverage_features_include_local_graph_progress():
    tracker = FrontierCoverageTracker("frontier_uniform")
    provider = SubWebActionCoverageFeatures(tracker)
    source = _state("http://localhost/owners/find", [
        ClickAction(ElementLocator.XPATH, "//button", "Find Owner", "submit", ""),
        ClickAction(ElementLocator.XPATH, "//a", "Home", "redirect", "/"),
    ])
    submit = next(action for action in source.get_action_list() if getattr(action, "text", "") == "Find Owner")
    target = _state("http://localhost/owners", [
        ClickAction(ElementLocator.XPATH, "//td/a", "Jane Doe", "redirect", "/owners/1"),
    ])
    source_node = extract_graph_node_id(source)
    tracker.seed_initial_state(source)

    before = provider.features_for(submit, source, source.get_action_list(), source_node_id=source_node)
    tracker.compute_strict_marginal_reward(
        target,
        chosen_action=submit,
        source_url="http://localhost/owners/find",
        source_node_id=source_node,
    )
    provider.update(source_node, "http://localhost/owners/find", submit, 1.25, target_state=target)
    after = provider.features_for(submit, source, source.get_action_list(), source_node_id=source_node)

    source_visit_idx = ACTION_COVERAGE_FEATURE_NAMES.index("source_visit_count_log")
    source_gain_idx = ACTION_COVERAGE_FEATURE_NAMES.index("source_gain_rate")
    source_out_seen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("source_out_seen_ratio")
    target_in_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_in_count_log")
    source_residual_idx = ACTION_COVERAGE_FEATURE_NAMES.index("source_residual_count_log")
    source_out_degree_idx = ACTION_COVERAGE_FEATURE_NAMES.index("source_out_degree_log")
    global_seen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("global_edge_seen_ratio")
    global_residual_idx = ACTION_COVERAGE_FEATURE_NAMES.index("global_edge_residual_count_log")

    assert before[source_visit_idx].item() == 0.0
    assert after[source_visit_idx].item() > 0.0
    assert after[source_gain_idx].item() == 1.0
    assert after[source_out_seen_idx].item() > before[source_out_seen_idx].item()
    assert after[target_in_idx].item() > 0.0
    assert before[source_residual_idx].item() > after[source_residual_idx].item()
    assert before[source_out_degree_idx].item() == after[source_out_degree_idx].item()
    assert after[global_seen_idx].item() > before[global_seen_idx].item()
    assert after[global_residual_idx].item() < before[global_residual_idx].item()


def test_action_coverage_features_include_target_frontier_potential():
    tracker = FrontierCoverageTracker("frontier_uniform")
    provider = SubWebActionCoverageFeatures(tracker)
    target_action_a = ClickAction(
        ElementLocator.XPATH,
        "//main/a[1]",
        "Add Pet",
        "redirect",
        "http://localhost/owners/1/pets/new",
    )
    target_action_b = ClickAction(
        ElementLocator.XPATH,
        "//main/a[2]",
        "Edit Owner",
        "redirect",
        "http://localhost/owners/1/edit",
    )
    target_state = _state("http://localhost/owners/1", [target_action_a, target_action_b])
    target_node = extract_graph_node_id(target_state)

    tracker.seed_initial_state(target_state)
    provider.features_for(target_action_a, target_state, target_state.get_action_list(), source_node_id=target_node)
    tracker.compute_strict_marginal_reward(
        _state("http://localhost/owners/1/pets/new", []),
        chosen_action=target_action_a,
        source_url="http://localhost/owners/1",
        source_node_id=target_node,
    )
    provider.update(target_node, "http://localhost/owners/1", target_action_a, reward=1.0)

    bridge_action = ClickAction(
        ElementLocator.XPATH,
        "//td/a",
        "Jane Doe",
        "redirect",
        "http://localhost/owners/1",
    )
    source_state = _state("http://localhost/owners", [bridge_action])
    source_node = extract_graph_node_id(source_state)
    features = provider.features_for(
        bridge_action,
        source_state,
        source_state.get_action_list(),
        source_node_id=source_node,
    )

    target_ratio_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_ratio")
    target_count_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_count_log")

    assert features[target_ratio_idx].item() == 0.5
    assert features[target_count_idx].item() > 0.0


def test_action_coverage_features_do_not_treat_self_loop_form_as_target_frontier():
    input_action = RandomInputAction(ElementLocator.XPATH, "//form/input[1]", "Name")
    submit = ClickAction(ElementLocator.XPATH, "//form/button", "Add Pet", "submit", "3")
    leave = ClickAction(ElementLocator.XPATH, "//nav/a", "Find Owners", "redirect", "/owners/find")
    state = _state("http://localhost/owners/1/pets/new", [input_action, submit, leave])
    tracker = FrontierCoverageTracker("frontier_uniform")
    tracker.seed_initial_state(state)
    provider = SubWebActionCoverageFeatures(tracker)
    source_node = extract_graph_node_id(state)

    provider.features_for(input_action, state, state.get_action_list(), source_node_id=source_node)
    provider.update(
        source_node,
        "http://localhost/owners/1/pets/new",
        input_action,
        reward=0.0,
        target_state=state,
    )
    features = provider.features_for(
        input_action,
        state,
        state.get_action_list(),
        source_node_id=source_node,
    )

    target_ratio_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_ratio")
    target_count_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_count_log")

    assert features[target_ratio_idx].item() == 0.0
    assert features[target_count_idx].item() == 0.0


def test_action_coverage_features_do_not_treat_same_route_link_as_target_frontier():
    same_route = ClickAction(ElementLocator.XPATH, "//nav/a", "New Event", "redirect", "/new")
    submit = ClickAction(ElementLocator.XPATH, "//form/button", "Create", "submit", "2")
    state = _state("http://localhost/new", [same_route, submit])
    tracker = FrontierCoverageTracker("frontier_uniform")
    tracker.seed_initial_state(state)
    provider = SubWebActionCoverageFeatures(tracker)
    source_node = extract_graph_node_id(state)

    provider.features_for(submit, state, state.get_action_list(), source_node_id=source_node)
    features = provider.features_for(
        same_route,
        state,
        state.get_action_list(),
        source_node_id=source_node,
    )

    target_ratio_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_ratio")
    target_count_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_count_log")

    assert features[target_ratio_idx].item() == 0.0
    assert features[target_count_idx].item() == 0.0


def test_action_coverage_features_keep_exact_sibling_links_distinct():
    owner_a = ClickAction(
        ElementLocator.XPATH,
        "/html/body/table/tbody/tr[1]/td[1]/a",
        "Alice Smith",
        "redirect",
        "http://localhost/owners/1",
    )
    owner_b = ClickAction(
        ElementLocator.XPATH,
        "/html/body/table/tbody/tr[2]/td[1]/a",
        "Bob Jones",
        "redirect",
        "http://localhost/owners/2",
    )
    state = _state("http://localhost/owners", [owner_a, owner_b])
    tracker = FrontierCoverageTracker("frontier_uniform")
    tracker.seed_initial_state(state)
    provider = SubWebActionCoverageFeatures(tracker)
    source_node = extract_graph_node_id(state)

    provider.update(source_node, "http://localhost/owners", owner_a, reward=1.0)
    batch = build_padded_action_batch(
        web_state=state,
        html="<html></html>",
        transformer=_TinyTransformer(),
        max_actions=4,
        input_dim=81 + structural_feature_dim() + action_coverage_feature_dim(),
        coverage_feature_provider=provider,
        source_node_id=source_node,
    )
    owner_b_idx = batch.actions_policy.index(owner_b)
    owner_b_features = batch.action_mat[owner_b_idx, -action_coverage_feature_dim():]

    edge_unseen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("edge_unseen_from_current_node")
    action_count_idx = ACTION_COVERAGE_FEATURE_NAMES.index("action_count_log")
    zero_gain_idx = ACTION_COVERAGE_FEATURE_NAMES.index("zero_gain_count_log")

    assert owner_b_features[edge_unseen_idx].item() == 1.0
    assert owner_b_features[action_count_idx].item() == 0.0
    assert owner_b_features[zero_gain_idx].item() == 0.0


def test_structural_features_mark_generic_child_workflow_route():
    action = ClickAction(
        ElementLocator.XPATH,
        "//main/a",
        "Create Task",
        "redirect",
        "/projects/1/tasks/new",
    )
    nav = ClickAction(ElementLocator.XPATH, "//nav/a", "Home", "redirect", "/")
    state = _state("http://localhost/projects/1", [nav, action])
    names = structural_action_feature_names()

    features = structural_action_features(action, state, [nav, action])

    assert features[names.index("child_descendant_route")].item() == 1.0
    assert features[names.index("create_new_add_word")].item() == 1.0
    assert features[names.index("local_non_nav_route_action")].item() == 1.0
    assert features[names.index("likely_global_or_boilerplate")].item() == 0.0


def test_batched_action_feature_rows_match_per_action_features():
    actions = [
        ClickAction(ElementLocator.XPATH, f"//main/a[{idx}]", f"Run {idx}", "redirect", f"/run/{idx}")
        for idx in range(30)
    ]
    actions += [
        ClickAction(ElementLocator.XPATH, "//nav/a", "Home", "redirect", "/"),
        ClickAction(ElementLocator.XPATH, "//main/a", "Create Task", "redirect", "/projects/1/tasks/new"),
    ]
    state = _state("http://localhost/projects/1", actions)

    role_rows = action_role_feature_rows(state, actions)
    structural_rows = structural_action_feature_rows(state, actions)

    assert len(role_rows) == len(actions)
    assert len(structural_rows) == len(actions)
    for action, role, structural in zip(actions, role_rows, structural_rows):
        assert torch.allclose(role, action_role_features(action, state, actions))
        assert torch.allclose(structural, structural_action_features(action, state, actions))


def test_padded_action_batch_fast_feature_path_preserves_rank_order():
    actions = [
        ClickAction(ElementLocator.XPATH, "//nav/a", "Home", "redirect", "/"),
        ClickAction(ElementLocator.XPATH, "//main/a", "Create Task", "redirect", "/projects/1/tasks/new"),
        ClickAction(ElementLocator.XPATH, "//form/button", "Save", "submit", ""),
    ]
    state = _state("http://localhost/projects/1", actions)
    ranked = deterministic_rank_actions(actions, state)

    batch = build_padded_action_batch(
        web_state=state,
        html="<html></html>",
        transformer=_TinyTransformer(),
        max_actions=8,
        input_dim=81 + structural_feature_dim(),
    )

    assert batch.actions_policy == ranked
    assert batch.action_mask[: len(actions)].all().item() is True
    assert batch.action_mask[len(actions) :].any().item() is False


def test_a2c_rollout_uses_clipped_training_reward_but_keeps_raw_gain():
    action = ClickAction(ElementLocator.XPATH, "//form/button", "Save", "submit", "")
    state = _state("http://localhost:8081/owners/new", [action])
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.coverage_tracker = FrontierCoverageTracker("frontier_uniform")
    agent.rollout_buffer = []
    agent.pending_step = {
        "step": 7,
        "action_mat": torch.zeros((2, 3)),
        "action_mask": torch.tensor([True, False]),
        "coverage_summary": torch.zeros(1),
        "critic_input": torch.zeros(4),
        "chosen_idx": torch.tensor(0, dtype=torch.long),
        "chosen_action": action,
        "source_url": "http://localhost:8081/owners/new",
        "action_type": "submit",
        "action_signature": "submit:save",
    }
    agent.last_reward = 0.0
    agent.last_train_reward = 0.0
    agent.last_finalized_transition = None
    agent.recent_rewards = []
    agent.zero_gain_streak = 0
    agent.reward_clip_abs = 1.0
    agent.action_coverage_features = SubWebActionCoverageFeatures(agent.coverage_tracker)

    train_reward = agent._finalize_pending(state)

    assert agent.last_reward > 1.0
    assert train_reward == 1.0
    assert agent.last_train_reward == 1.0
    assert agent.rollout_buffer[0]["reward"].item() == 1.0
    assert agent.last_finalized_transition["raw_reward"] == agent.last_reward
    assert agent.last_finalized_transition["train_reward"] == 1.0
    assert agent.last_finalized_transition["clipped"] is True


def test_graph_residual_aux_target_is_zero_without_graph_or_coverage_support():
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.use_action_coverage_features = False
    agent.graph_residual_aux_target_mode = "coverage"
    agent.graph_residual_aux_target_power = 1.0
    agent.graph_residual_aux_distribution = "max_only"
    agent.graph_residual_aux_tie_break_coef = 0.05

    batch = PaddedActionBatch(
        actions_full=[],
        actions_policy=[],
        action_mat=torch.eye(2),
        action_mask=torch.tensor([True, True]),
        original_action_count=2,
        kept_action_count=2,
        truncated_count=0,
        truncated_by_type={},
        action_set_context=torch.zeros(0),
        structural_action_set_context=torch.zeros(0),
    )

    target = agent._graph_residual_aux_target(batch)

    assert float(target.sum().item()) == 0.0
    assert agent.last_graph_residual_aux_target_debug["target_source"] == "zero"


def test_graph_residual_aux_target_does_not_create_teacher_from_coverage_only_mass():
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.graph_residual_aux_target_mode = "coverage"
    agent.use_action_coverage_features = True
    agent.action_coverage_feature_dim = action_coverage_feature_dim()

    features = torch.zeros((3, action_coverage_feature_dim()))
    unseen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("edge_unseen_from_current_node")
    target_ratio_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_ratio")
    target_count_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_count_log")
    features[1, unseen_idx] = 1.0
    features[1, target_ratio_idx] = 1.0
    features[1, target_count_idx] = 1.0
    features[2, unseen_idx] = 1.0

    batch = PaddedActionBatch(
        actions_full=[],
        actions_policy=[],
        action_mat=features,
        action_mask=torch.tensor([True, True, True]),
        original_action_count=3,
        kept_action_count=3,
        truncated_count=0,
        truncated_by_type={},
        action_set_context=torch.zeros(1),
        structural_action_set_context=torch.zeros(1),
    )

    target = agent._graph_residual_aux_target(batch)

    assert float(target.sum().item()) == 0.0
    assert agent.last_graph_residual_aux_target_debug["target_source"] == "zero"
    assert agent.last_graph_residual_aux_target_debug["coverage_score_sum"] > 0.0


def test_graph_residual_aux_target_prefers_graph_uncovered_edge_prefixes_first():
    unseen_action = ClickAction(ElementLocator.XPATH, "//main/a[1]", "Add Owner", "redirect", "/owners/new")
    seen_action = ClickAction(ElementLocator.XPATH, "//main/a[2]", "Home", "redirect", "/")
    state = _state("http://localhost/owners/find", [unseen_action, seen_action])
    source_node = extract_graph_node_id(state)
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.coverage_tracker = FrontierCoverageTracker("frontier_uniform")
    agent.use_action_coverage_features = True
    agent.action_coverage_feature_dim = action_coverage_feature_dim()

    batch = PaddedActionBatch(
        actions_full=[unseen_action, seen_action],
        actions_policy=[unseen_action, seen_action],
        action_mat=torch.zeros((2, 3 + action_coverage_feature_dim())),
        action_mask=torch.tensor([True, True]),
        original_action_count=2,
        kept_action_count=2,
        truncated_count=0,
        truncated_by_type={},
        action_set_context=torch.zeros(0),
        structural_action_set_context=torch.zeros(0),
    )
    seen_prefix = graph_edge_prefix("http://localhost/owners/find", seen_action, source_node_id=source_node)
    agent.coverage_tracker.covered_edges.add(f"{seen_prefix}target_node=/|target_context=/|outcome=route_change")

    target = agent._graph_residual_aux_target(batch, state)

    assert target[0].item() == 1.0
    assert target[1].item() == 0.0


def test_graph_residual_aux_target_includes_graph_node_gain_for_unseen_targets():
    new_target = ClickAction(ElementLocator.XPATH, "//main/a[1]", "Add Owner", "redirect", "/owners/new")
    seen_target = ClickAction(ElementLocator.XPATH, "//main/a[2]", "Veterinarians", "redirect", "/vets.html")
    state = _state("http://localhost/owners/find", [new_target, seen_target])
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    # This test exercises the explicit legacy proportional residual variant.
    agent.graph_residual_aux_target_mode = "graph"
    agent.graph_residual_aux_distribution = "proportional"
    agent.coverage_tracker = FrontierCoverageTracker("frontier_uniform")
    agent.coverage_tracker.covered_nodes.add("node:/vets.html|state=already-seen|n=4")
    agent.use_action_coverage_features = True
    agent.action_coverage_feature_dim = action_coverage_feature_dim()

    batch = PaddedActionBatch(
        actions_full=[new_target, seen_target],
        actions_policy=[new_target, seen_target],
        action_mat=torch.zeros((2, 3 + action_coverage_feature_dim())),
        action_mask=torch.tensor([True, True]),
        original_action_count=2,
        kept_action_count=2,
        truncated_count=0,
        truncated_by_type={},
        action_set_context=torch.zeros(0),
        structural_action_set_context=torch.zeros(0),
    )

    target = agent._graph_residual_aux_target(batch, state)

    assert target[0].item() > target[1].item() > 0.0
    assert abs(float(target.sum().item()) - 1.0) < 1e-6


def test_graph_residual_aux_target_uses_coverage_features_to_break_uncovered_edge_ties():
    first = RandomInputAction(ElementLocator.XPATH, "//form/input[1]", "First Name")
    second = RandomInputAction(ElementLocator.XPATH, "//form/input[2]", "Last Name")
    state = _state("http://localhost/owners/new", [first, second])
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.graph_residual_aux_target_mode = "coverage"
    agent.graph_residual_aux_distribution = "proportional"
    agent.graph_residual_aux_tie_break_coef = 0.05
    agent.coverage_tracker = FrontierCoverageTracker("frontier_uniform")
    agent.use_action_coverage_features = True
    agent.action_coverage_feature_dim = action_coverage_feature_dim()

    action_mat = torch.zeros((2, 3 + action_coverage_feature_dim()))
    feature_start = action_mat.shape[1] - action_coverage_feature_dim()
    target_ratio_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_ratio")
    target_count_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_count_log")
    edge_unseen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("edge_unseen_from_current_node")
    action_mat[:, feature_start + edge_unseen_idx] = 1.0
    action_mat[1, feature_start + target_ratio_idx] = 1.0
    action_mat[1, feature_start + target_count_idx] = 1.0

    batch = PaddedActionBatch(
        actions_full=[first, second],
        actions_policy=[first, second],
        action_mat=action_mat,
        action_mask=torch.tensor([True, True]),
        original_action_count=2,
        kept_action_count=2,
        truncated_count=0,
        truncated_by_type={},
        action_set_context=torch.zeros(0),
        structural_action_set_context=torch.zeros(0),
    )

    target = agent._graph_residual_aux_target(batch, state)

    assert target[1].item() > target[0].item() > 0.0
    assert abs(float(target.sum().item()) - 1.0) < 1e-6


def test_graph_residual_aux_target_tiebreak_does_not_override_larger_graph_gain():
    new_target = ClickAction(ElementLocator.XPATH, "//main/a[1]", "Add Owner", "redirect", "/owners/new")
    seen_target = ClickAction(ElementLocator.XPATH, "//main/a[2]", "Veterinarians", "redirect", "/vets.html")
    state = _state("http://localhost/owners/find", [new_target, seen_target])
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.graph_residual_aux_target_mode = "coverage"
    agent.graph_residual_aux_distribution = "proportional"
    agent.coverage_tracker = FrontierCoverageTracker("frontier_uniform")
    agent.coverage_tracker.covered_nodes.add("node:/vets.html|state=already-seen|n=4")
    agent.use_action_coverage_features = True
    agent.action_coverage_feature_dim = action_coverage_feature_dim()
    agent.graph_residual_aux_tie_break_coef = 0.05

    action_mat = torch.zeros((2, 3 + action_coverage_feature_dim()))
    feature_start = action_mat.shape[1] - action_coverage_feature_dim()
    target_seen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_route_seen")
    target_ratio_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_ratio")
    target_count_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_count_log")
    action_mat[1, feature_start + target_seen_idx] = 1.0
    action_mat[1, feature_start + target_ratio_idx] = 1.0
    action_mat[1, feature_start + target_count_idx] = 1.0

    batch = PaddedActionBatch(
        actions_full=[new_target, seen_target],
        actions_policy=[new_target, seen_target],
        action_mat=action_mat,
        action_mask=torch.tensor([True, True]),
        original_action_count=2,
        kept_action_count=2,
        truncated_count=0,
        truncated_by_type={},
        action_set_context=torch.zeros(0),
        structural_action_set_context=torch.zeros(0),
    )

    target = agent._graph_residual_aux_target(batch, state)

    assert target[0].item() > target[1].item() > 0.0
    assert agent.last_graph_residual_aux_target_debug["target_source"] == "graph+coverage"


def test_graph_residual_aux_target_does_not_keep_seen_bridge_without_graph_support():
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.graph_residual_aux_target_mode = "coverage"
    agent.graph_residual_aux_distribution = "proportional"
    agent.coverage_tracker = FrontierCoverageTracker("frontier_uniform")
    agent.use_action_coverage_features = True
    agent.action_coverage_feature_dim = action_coverage_feature_dim()

    features = torch.zeros((2, action_coverage_feature_dim()))
    edge_seen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("edge_seen_from_current_node")
    target_seen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_route_seen")
    target_ratio_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_ratio")
    target_count_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_count_log")
    features[0, edge_seen_idx] = 1.0
    features[0, target_seen_idx] = 1.0
    features[0, target_ratio_idx] = 1.0
    features[0, target_count_idx] = 1.0

    batch = PaddedActionBatch(
        actions_full=[],
        actions_policy=[],
        action_mat=features,
        action_mask=torch.tensor([True, True]),
        original_action_count=2,
        kept_action_count=2,
        truncated_count=0,
        truncated_by_type={},
        action_set_context=torch.zeros(0),
        structural_action_set_context=torch.zeros(0),
    )

    target = agent._graph_residual_aux_target(batch)

    assert float(target.sum().item()) == 0.0
    assert agent.last_graph_residual_aux_target_debug["target_source"] == "zero"
    assert agent.last_graph_residual_aux_target_debug["coverage_score_sum"] > 0.0


def test_graph_residual_aux_target_power_sharpens_without_changing_support_or_mask():
    actions = [
        ClickAction(ElementLocator.XPATH, "//main/a[1]", "A", "redirect", "/a"),
        ClickAction(ElementLocator.XPATH, "//main/a[2]", "B", "redirect", "/b"),
        ClickAction(ElementLocator.XPATH, "//main/a[3]", "C", "redirect", "/c"),
        ClickAction(ElementLocator.XPATH, "//main/a[4]", "D", "redirect", "/d"),
    ]
    state = _state("http://localhost/source", actions)
    source_node = extract_graph_node_id(state)
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.graph_residual_aux_distribution = "proportional"
    agent.graph_residual_aux_tie_break_coef = 0.05
    agent.coverage_tracker = FrontierCoverageTracker("frontier_uniform")
    covered_prefix = graph_edge_prefix("http://localhost/source", actions[2], source_node_id=source_node)
    agent.coverage_tracker.covered_edges.add(
        f"{covered_prefix}target_node=/c|target_context=/c|outcome=route_change"
    )
    agent.graph_residual_aux_target_mode = "coverage"
    agent.use_action_coverage_features = True
    agent.action_coverage_feature_dim = action_coverage_feature_dim()

    features = torch.zeros((4, 3 + action_coverage_feature_dim()))
    feature_start = features.shape[1] - action_coverage_feature_dim()
    unseen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("edge_unseen_from_current_node")
    target_ratio_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_ratio")
    target_count_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_count_log")
    features[0, feature_start + unseen_idx] = 1.0
    features[1, feature_start + unseen_idx] = 1.0
    features[1, feature_start + target_ratio_idx] = 1.0
    features[1, feature_start + target_count_idx] = 1.0
    features[3, feature_start + unseen_idx] = 1.0

    batch = PaddedActionBatch(
        actions_full=actions,
        actions_policy=actions,
        action_mat=features,
        action_mask=torch.tensor([True, True, True, False]),
        original_action_count=3,
        kept_action_count=3,
        truncated_count=0,
        truncated_by_type={},
        action_set_context=torch.zeros(0),
        structural_action_set_context=torch.zeros(0),
    )

    agent.graph_residual_aux_target_power = 1.0
    base_target = agent._graph_residual_aux_target(batch, state)
    agent.graph_residual_aux_target_power = 2.0
    sharp_target = agent._graph_residual_aux_target(batch, state)

    assert sharp_target[1].item() > base_target[1].item()
    assert sharp_target[0].item() < base_target[0].item()
    assert sharp_target[2].item() == 0.0
    assert sharp_target[3].item() == 0.0
    assert abs(float(sharp_target.sum().item()) - 1.0) < 1e-6
    assert agent.last_graph_residual_aux_target_debug["target_source"] == "graph+coverage"
    assert agent.last_graph_residual_aux_target_debug["target_power"] == 2.0
    assert agent.last_graph_residual_aux_target_debug["post_power_top_idx"] == 1


def test_graph_residual_aux_target_max_only_keeps_only_max_estimated_graph_gain():
    new_target = ClickAction(ElementLocator.XPATH, "//main/a[1]", "Add Owner", "redirect", "/owners/new")
    seen_target = ClickAction(ElementLocator.XPATH, "//main/a[2]", "Veterinarians", "redirect", "/vets.html")
    state = _state("http://localhost/owners/find", [new_target, seen_target])
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.coverage_tracker = FrontierCoverageTracker("frontier_uniform")
    agent.coverage_tracker.covered_nodes.add("node:/vets.html|state=already-seen|n=4")
    agent.use_action_coverage_features = True
    agent.action_coverage_feature_dim = action_coverage_feature_dim()
    agent.graph_residual_aux_target_power = 1.0
    agent.graph_residual_aux_distribution = "max_only"
    agent.graph_residual_aux_tie_break_coef = 0.05

    batch = PaddedActionBatch(
        actions_full=[new_target, seen_target],
        actions_policy=[new_target, seen_target],
        action_mat=torch.zeros((2, 3 + action_coverage_feature_dim())),
        action_mask=torch.tensor([True, True]),
        original_action_count=2,
        kept_action_count=2,
        truncated_count=0,
        truncated_by_type={},
        action_set_context=torch.zeros(0),
        structural_action_set_context=torch.zeros(0),
    )

    target = agent._graph_residual_aux_target(batch, state)

    assert target[0].item() == 1.0
    assert target[1].item() == 0.0
    assert agent.last_graph_residual_aux_target_debug["target_distribution"] == "max_only"


def test_graph_residual_aux_env_defaults_to_paper_uniform_uncovered_target():
    old_graph_aux = os.environ.get("WEBTEST_GRAPH_RESIDUAL_AUX")
    old_coef = os.environ.get("WEBTEST_GRAPH_RESIDUAL_AUX_COEF")
    old_target = os.environ.get("WEBTEST_GRAPH_RESIDUAL_AUX_TARGET")
    old_distribution = os.environ.get("WEBTEST_GRAPH_RESIDUAL_AUX_DISTRIBUTION")
    old_tie = os.environ.get("WEBTEST_GRAPH_RESIDUAL_AUX_TIE_BREAK_COEF")
    old_diag = os.environ.get("WEBTEST_POLICY_DIAGNOSTICS")
    try:
        os.environ["WEBTEST_GRAPH_RESIDUAL_AUX"] = "1"
        os.environ.pop("WEBTEST_GRAPH_RESIDUAL_AUX_COEF", None)
        os.environ.pop("WEBTEST_GRAPH_RESIDUAL_AUX_TARGET", None)
        os.environ.pop("WEBTEST_GRAPH_RESIDUAL_AUX_DISTRIBUTION", None)
        os.environ.pop("WEBTEST_GRAPH_RESIDUAL_AUX_TIE_BREAK_COEF", None)
        os.environ["WEBTEST_POLICY_DIAGNOSTICS"] = "0"
        agent = SubWebFrontierA2CAgent({
            "transformer_module": "tests.test_subweb_frontier_a2c",
            "transformer_class": "_TinyTransformer",
            "hidden_dim": 8,
            "input_dim": 52,
            "rollout_len": 2,
            "max_actions": 4,
        })

        assert agent.graph_residual_aux is True
        assert abs(agent.graph_residual_aux_coef - 0.2) < 1e-9
        assert agent.graph_residual_aux_target_mode == "uncovered"
        assert agent.graph_residual_aux_distribution == "uniform"
        assert abs(agent.graph_residual_aux_tie_break_coef) < 1e-12
    finally:
        _restore_env("WEBTEST_GRAPH_RESIDUAL_AUX", old_graph_aux)
        _restore_env("WEBTEST_GRAPH_RESIDUAL_AUX_COEF", old_coef)
        _restore_env("WEBTEST_GRAPH_RESIDUAL_AUX_TARGET", old_target)
        _restore_env("WEBTEST_GRAPH_RESIDUAL_AUX_DISTRIBUTION", old_distribution)
        _restore_env("WEBTEST_GRAPH_RESIDUAL_AUX_TIE_BREAK_COEF", old_tie)
        _restore_env("WEBTEST_POLICY_DIAGNOSTICS", old_diag)


def test_graph_residual_exact_target_ignores_coverage_feature_tiebreaks():
    new_target = ClickAction(ElementLocator.XPATH, "//main/a[1]", "Add Owner", "redirect", "/owners/new")
    seen_target = ClickAction(ElementLocator.XPATH, "//main/a[2]", "Veterinarians", "redirect", "/vets.html")
    state = _state("http://localhost/owners/find", [new_target, seen_target])
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.coverage_tracker = FrontierCoverageTracker("frontier_uniform")
    agent.coverage_tracker.covered_nodes.add("node:/vets.html|state=already-seen|n=4")
    agent.graph_residual_aux_target_mode = "graph"
    agent.graph_residual_aux_distribution = "max_only"
    agent.graph_residual_aux_target_power = 1.0
    agent.graph_residual_aux_tie_break_coef = 0.0
    agent.use_action_coverage_features = True
    agent.action_coverage_feature_dim = action_coverage_feature_dim()

    action_mat = torch.zeros((2, 3 + action_coverage_feature_dim()))
    feature_start = action_mat.shape[1] - action_coverage_feature_dim()
    target_seen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_route_seen")
    target_ratio_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_ratio")
    target_count_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_count_log")
    action_mat[1, feature_start + target_seen_idx] = 1.0
    action_mat[1, feature_start + target_ratio_idx] = 10.0
    action_mat[1, feature_start + target_count_idx] = 10.0

    batch = PaddedActionBatch(
        actions_full=[new_target, seen_target],
        actions_policy=[new_target, seen_target],
        action_mat=action_mat,
        action_mask=torch.tensor([True, True]),
        original_action_count=2,
        kept_action_count=2,
        truncated_count=0,
        truncated_by_type={},
        action_set_context=torch.zeros(0),
        structural_action_set_context=torch.zeros(0),
    )

    target = agent._graph_residual_aux_target(batch, state)

    assert target[0].item() == 1.0
    assert target[1].item() == 0.0
    assert agent.last_graph_residual_aux_target_debug["target_source"] == "graph"
    assert agent.last_graph_residual_aux_target_debug["coverage_score_sum"] == 0.0


def test_graph_residual_exact_target_respects_action_mask():
    valid = ClickAction(ElementLocator.XPATH, "//main/a[1]", "A", "redirect", "/a")
    masked = ClickAction(ElementLocator.XPATH, "//main/a[2]", "B", "redirect", "/b")
    state = _state("http://localhost/source", [valid, masked])
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.coverage_tracker = FrontierCoverageTracker("frontier_uniform")
    agent.graph_residual_aux_target_mode = "graph"
    agent.graph_residual_aux_distribution = "max_only"
    agent.graph_residual_aux_target_power = 1.0
    agent.graph_residual_aux_tie_break_coef = 0.0
    agent.use_action_coverage_features = False

    batch = PaddedActionBatch(
        actions_full=[valid, masked],
        actions_policy=[valid, masked],
        action_mat=torch.zeros((2, 3)),
        action_mask=torch.tensor([True, False]),
        original_action_count=2,
        kept_action_count=1,
        truncated_count=1,
        truncated_by_type={"redirect": 1},
        action_set_context=torch.zeros(0),
        structural_action_set_context=torch.zeros(0),
    )

    target = agent._graph_residual_aux_target(batch, state)

    assert target[0].item() == 1.0
    assert target[1].item() == 0.0
    assert abs(float(target.sum().item()) - 1.0) < 1e-6


def test_graph_residual_target_is_zero_when_all_graph_residual_is_exhausted():
    action = ClickAction(ElementLocator.XPATH, "//main/a[1]", "Add Owner", "redirect", "/owners/new")
    state = _state("http://localhost/owners/find", [action])
    source_node = extract_graph_node_id(state)
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.coverage_tracker = FrontierCoverageTracker("frontier_uniform")
    covered_prefix = graph_edge_prefix("http://localhost/owners/find", action, source_node_id=source_node)
    agent.coverage_tracker.covered_edges.add(
        f"{covered_prefix}target_node=/owners/new|target_context=/owners/new|outcome=route_change"
    )
    agent.graph_residual_aux_target_mode = "graph"
    agent.graph_residual_aux_distribution = "max_only"
    agent.graph_residual_aux_target_power = 1.0
    agent.graph_residual_aux_tie_break_coef = 0.0
    agent.use_action_coverage_features = True
    agent.action_coverage_feature_dim = action_coverage_feature_dim()

    action_mat = torch.zeros((1, 3 + action_coverage_feature_dim()))
    feature_start = action_mat.shape[1] - action_coverage_feature_dim()
    unseen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("edge_unseen_from_current_node")
    target_ratio_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_ratio")
    action_mat[0, feature_start + unseen_idx] = 1.0
    action_mat[0, feature_start + target_ratio_idx] = 10.0

    batch = PaddedActionBatch(
        actions_full=[action],
        actions_policy=[action],
        action_mat=action_mat,
        action_mask=torch.tensor([True]),
        original_action_count=1,
        kept_action_count=1,
        truncated_count=0,
        truncated_by_type={},
        action_set_context=torch.zeros(0),
        structural_action_set_context=torch.zeros(0),
    )

    target = agent._graph_residual_aux_target(batch, state)

    assert float(target.sum().item()) == 0.0
    assert agent.last_graph_residual_aux_target_debug["target_source"] == "zero"
    assert agent.last_graph_residual_aux_target_debug["graph_score_sum"] == 0.0
    assert agent.last_graph_residual_aux_target_debug["coverage_score_sum"] == 0.0


def test_graph_bridge_target_uses_known_residual_frontier_when_source_exhausted():
    bridge = ClickAction(ElementLocator.XPATH, "//nav/a[1]", "Owners", "redirect", "/owners")
    local = RandomInputAction(ElementLocator.XPATH, "//form/input[1]", "Name")
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.coverage_tracker = FrontierCoverageTracker("frontier_uniform")
    agent.graph_residual_aux_target_mode = "graph_bridge"
    agent.graph_residual_aux_distribution = "max_only"
    agent.graph_residual_aux_target_power = 1.0
    agent.graph_residual_aux_tie_break_coef = 0.0
    agent.use_action_coverage_features = True
    agent.action_coverage_feature_dim = action_coverage_feature_dim()

    action_mat = torch.zeros((2, 3 + action_coverage_feature_dim()))
    feature_start = action_mat.shape[1] - action_coverage_feature_dim()
    edge_seen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("edge_seen_from_current_node")
    target_seen_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_route_seen")
    target_ratio_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_ratio")
    target_count_idx = ACTION_COVERAGE_FEATURE_NAMES.index("target_known_frontier_count_log")
    action_mat[:, feature_start + edge_seen_idx] = 1.0
    action_mat[0, feature_start + target_seen_idx] = 1.0
    action_mat[0, feature_start + target_ratio_idx] = 1.0
    action_mat[0, feature_start + target_count_idx] = 1.0

    batch = PaddedActionBatch(
        actions_full=[bridge, local],
        actions_policy=[bridge, local],
        action_mat=action_mat,
        action_mask=torch.tensor([True, True]),
        original_action_count=2,
        kept_action_count=2,
        truncated_count=0,
        truncated_by_type={},
        action_set_context=torch.zeros(0),
        structural_action_set_context=torch.zeros(0),
    )

    target = agent._graph_residual_aux_target(batch)

    assert target[0].item() == 1.0
    assert target[1].item() == 0.0
    assert agent.last_graph_residual_aux_target_debug["target_source"] == "graph_bridge"
    assert agent.last_graph_residual_aux_target_debug["graph_score_sum"] == 0.0
    assert agent.last_graph_residual_aux_target_debug["coverage_score_sum"] > 0.0


def test_graph_residual_aux_does_not_change_hard_rollout_reward():
    action = ClickAction(ElementLocator.XPATH, "//main/a[1]", "Add Owner", "redirect", "/owners/new")
    next_state = _state("http://localhost/owners/new", [])
    source_state = _state("http://localhost/owners/find", [action])
    source_node = extract_graph_node_id(source_state)
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.coverage_tracker = FrontierCoverageTracker("frontier_uniform")
    agent.action_coverage_features = SubWebActionCoverageFeatures(agent.coverage_tracker)
    agent.recent_rewards = deque(maxlen=16)
    agent.recent_action_history = deque(maxlen=8)
    agent.rollout_buffer = []
    agent.graph_state_list = []
    agent.zero_gain_streak = 0
    agent.reward_clip_abs = None
    agent.last_reward = 0.0
    agent.last_train_reward = 0.0
    agent.pending_step = {
        "step": 3,
        "action_mat": torch.tensor([[1.0, 0.0, 0.0]]),
        "action_mask": torch.tensor([True]),
        "coverage_summary": torch.zeros(1),
        "critic_input": torch.zeros(4),
        "policy_temperature": 1.0,
        "chosen_idx": torch.tensor(0, dtype=torch.long),
        "chosen_action": action,
        "source_url": "http://localhost/owners/find",
        "source_node_id": source_node,
        "action_type": action_kind(action),
        "action_signature": action_signature(action),
        "graph_residual_aux_target": torch.tensor([1.0]),
    }

    reward = agent._finalize_pending(next_state)

    assert reward == agent.last_finalized_transition["raw_reward"]
    assert reward == agent.last_finalized_transition["train_reward"]
    assert reward == agent.rollout_buffer[0]["reward"].item()
    assert reward == 2.0
    assert agent.rollout_buffer[0]["graph_residual_aux_target"].tolist() == [1.0]


def test_a2c_critic_input_uses_masked_mean_and_max_without_padding():
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.input_dim = 3
    agent.coverage_summary_dim = 2
    agent.critic_pooling = "mean_max"
    batch = PaddedActionBatch(
        actions_full=[],
        actions_policy=[],
        action_mat=torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [3.0, 0.0, 4.0],
                [99.0, 99.0, 99.0],
            ]
        ),
        action_mask=torch.tensor([True, True, False]),
        original_action_count=2,
        kept_action_count=2,
        truncated_count=0,
        truncated_by_type={},
        action_set_context=torch.zeros(0),
        structural_action_set_context=torch.zeros(0),
    )

    critic_input = agent._critic_input(batch, torch.tensor([0.5, 0.25]))

    assert agent.critic_input_dim == 8
    assert torch.allclose(
        critic_input,
        torch.tensor([2.0, 1.0, 2.0, 3.0, 2.0, 4.0, 0.5, 0.25]),
    )


def test_a2c_critic_input_defaults_to_masked_mean_pooling():
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.input_dim = 3
    agent.coverage_summary_dim = 2
    agent.critic_pooling = "mean"
    batch = PaddedActionBatch(
        actions_full=[],
        actions_policy=[],
        action_mat=torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [3.0, 0.0, 4.0],
                [99.0, 99.0, 99.0],
            ]
        ),
        action_mask=torch.tensor([True, True, False]),
        original_action_count=2,
        kept_action_count=2,
        truncated_count=0,
        truncated_by_type={},
        action_set_context=torch.zeros(0),
        structural_action_set_context=torch.zeros(0),
    )

    critic_input = agent._critic_input(batch, torch.tensor([0.5, 0.25]))

    assert agent.critic_input_dim == 5
    assert torch.allclose(critic_input, torch.tensor([2.0, 1.0, 2.0, 0.5, 0.25]))


def test_a2c_td_targets_do_not_smear_late_reward_over_all_prior_actions():
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.gamma = 1.0
    rewards = torch.tensor([0.0, 10.0])
    values = torch.tensor([0.0, 0.0])

    td_targets = agent._one_step_td_targets(rewards, values, torch.tensor(0.0))

    assert torch.allclose(td_targets, torch.tensor([0.0, 10.0]))


def test_a2c_td_targets_bootstrap_only_the_rollout_tail():
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.gamma = 1.0
    rewards = torch.tensor([0.0, 0.0])
    values = torch.tensor([0.0, 0.0])

    td_targets = agent._one_step_td_targets(rewards, values, torch.tensor(7.0))

    assert torch.allclose(td_targets, torch.tensor([0.0, 7.0]))


def test_a2c_gae_softens_delayed_reward_credit_smearing():
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.gamma = 1.0
    agent.gae_lambda = 0.5
    rewards = torch.tensor([0.0, 0.0, 10.0])
    values = torch.tensor([0.0, 0.0, 0.0])

    targets, advantages = agent._gae_targets(rewards, values, torch.tensor(0.0))

    assert torch.allclose(advantages, torch.tensor([2.5, 5.0, 10.0]))
    assert torch.allclose(targets, advantages)


def test_a2c_reads_global_gamma_alias_and_gae_lambda():
    old_gamma = os.environ.get("WEBTEST_GAMMA")
    old_a2c_gamma = os.environ.get("WEBTEST_A2C_GAMMA")
    old_lambda = os.environ.get("WEBTEST_A2C_GAE_LAMBDA")
    old_diag = os.environ.get("WEBTEST_POLICY_DIAGNOSTICS")
    try:
        os.environ["WEBTEST_GAMMA"] = "0.75"
        os.environ.pop("WEBTEST_A2C_GAMMA", None)
        os.environ["WEBTEST_A2C_GAE_LAMBDA"] = "0.42"
        os.environ["WEBTEST_POLICY_DIAGNOSTICS"] = "0"
        agent = SubWebFrontierA2CAgent({
            "transformer_module": "tests.test_subweb_frontier_a2c",
            "transformer_class": "_TinyTransformer",
            "hidden_dim": 8,
            "input_dim": 52,
            "rollout_len": 2,
            "max_actions": 4,
        })

        assert abs(agent.gamma - 0.75) < 1e-9
        assert abs(agent.gae_lambda - 0.42) < 1e-9
    finally:
        _restore_env("WEBTEST_GAMMA", old_gamma)
        _restore_env("WEBTEST_A2C_GAMMA", old_a2c_gamma)
        _restore_env("WEBTEST_A2C_GAE_LAMBDA", old_lambda)
        _restore_env("WEBTEST_POLICY_DIAGNOSTICS", old_diag)


def test_a2c_defaults_to_strict_marginal_without_auxiliary():
    old_aux = os.environ.get("WEBTEST_GRAPH_RESIDUAL_AUX")
    old_coef = os.environ.get("WEBTEST_GRAPH_RESIDUAL_AUX_COEF")
    old_diag = os.environ.get("WEBTEST_POLICY_DIAGNOSTICS")
    try:
        os.environ.pop("WEBTEST_GRAPH_RESIDUAL_AUX", None)
        os.environ.pop("WEBTEST_GRAPH_RESIDUAL_AUX_COEF", None)
        os.environ["WEBTEST_POLICY_DIAGNOSTICS"] = "0"
        agent = SubWebFrontierA2CAgent({
            "transformer_module": "tests.test_subweb_frontier_a2c",
            "transformer_class": "_TinyTransformer",
            "hidden_dim": 8,
            "input_dim": 52,
            "rollout_len": 2,
            "max_actions": 4,
        })

        assert agent.graph_residual_aux is False
        assert abs(agent.graph_residual_aux_coef) < 1e-9
    finally:
        _restore_env("WEBTEST_GRAPH_RESIDUAL_AUX", old_aux)
        _restore_env("WEBTEST_GRAPH_RESIDUAL_AUX_COEF", old_coef)
        _restore_env("WEBTEST_POLICY_DIAGNOSTICS", old_diag)


def test_single_step_rollout_defaults_do_not_disable_actor_updates():
    old_rollout = os.environ.get("WEBTEST_ROLLOUT_LEN")
    old_a2c_min = os.environ.get("WEBTEST_A2C_MIN_ACTOR_UPDATE_STEPS")
    old_diag = os.environ.get("WEBTEST_POLICY_DIAGNOSTICS")
    try:
        os.environ.pop("WEBTEST_A2C_MIN_ACTOR_UPDATE_STEPS", None)
        os.environ["WEBTEST_ROLLOUT_LEN"] = "1"
        os.environ["WEBTEST_POLICY_DIAGNOSTICS"] = "0"

        a2c = SubWebFrontierA2CAgent({
            "transformer_module": "tests.test_subweb_frontier_a2c",
            "transformer_class": "_TinyTransformer",
            "hidden_dim": 8,
            "input_dim": 52,
            "max_actions": 4,
        })

        assert a2c.rollout_len == 1
        assert a2c.min_actor_update_steps == 1
    finally:
        _restore_env("WEBTEST_ROLLOUT_LEN", old_rollout)
        _restore_env("WEBTEST_A2C_MIN_ACTOR_UPDATE_STEPS", old_a2c_min)
        _restore_env("WEBTEST_POLICY_DIAGNOSTICS", old_diag)


def test_a2c_optimizer_can_use_separate_actor_and_critic_learning_rates():
    old_actor_lr = os.environ.get("WEBTEST_A2C_ACTOR_LR")
    old_critic_lr = os.environ.get("WEBTEST_A2C_CRITIC_LR")
    old_diag = os.environ.get("WEBTEST_POLICY_DIAGNOSTICS")
    try:
        os.environ["WEBTEST_A2C_ACTOR_LR"] = "0.003"
        os.environ["WEBTEST_A2C_CRITIC_LR"] = "0.0005"
        os.environ["WEBTEST_POLICY_DIAGNOSTICS"] = "0"
        agent = SubWebFrontierA2CAgent({
            "transformer_module": "tests.test_subweb_frontier_a2c",
            "transformer_class": "_TinyTransformer",
            "input_dim": 52 + context_feature_dim() + structural_feature_dim() + action_coverage_feature_dim(),
            "hidden_dim": 8,
            "max_actions": 4,
            "rollout_len": 2,
        })

        lrs = sorted(group["lr"] for group in agent.optimizer.param_groups)
        assert lrs == [0.0005, 0.003]
    finally:
        _restore_env("WEBTEST_A2C_ACTOR_LR", old_actor_lr)
        _restore_env("WEBTEST_A2C_CRITIC_LR", old_critic_lr)
        _restore_env("WEBTEST_POLICY_DIAGNOSTICS", old_diag)


def test_a2c_policy_temperature_preserves_mask_and_sharpens_distribution():
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.policy_temperature = 1.0
    logits = torch.tensor([1.0, 0.0, -1.0e9])
    base = agent._policy_distribution(logits, temperature=1.0)
    sharp = agent._policy_distribution(logits, temperature=0.5)

    assert base.probs[2].item() == 0.0
    assert sharp.probs[2].item() == 0.0
    assert sharp.probs[0].item() > base.probs[0].item()


def test_a2c_gae_propagates_delayed_graph_gain_to_bridge_actions():
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.gamma = 1.0
    agent.gae_lambda = 0.8
    rewards = torch.tensor([0.0, 0.0, 2.0])
    values = torch.zeros(3)
    bootstrap = torch.tensor(0.0)

    targets, advantages = agent._gae_targets(rewards, values, bootstrap)

    assert torch.allclose(advantages, torch.tensor([1.28, 1.6, 2.0]), atol=1e-6)
    assert torch.allclose(targets, advantages, atol=1e-6)
    assert advantages[0].item() > 0.0
    assert advantages[1].item() > 0.0


def _a2c_update_test_agent():
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.actor = MaskedCandidateActor(input_dim=3, hidden_dim=8, context_dim=1)
    agent.critic = FrontierCritic(input_dim=4, hidden_dim=8)
    agent.optimizer = torch.optim.Adam(
        [
            {"params": agent.actor.parameters(), "lr": 0.01},
            {"params": agent.critic.parameters(), "lr": 0.01},
        ]
    )
    agent.gamma = 1.0
    agent.gae_lambda = 0.8
    agent.advantage_estimator = "n_step"
    agent.entropy_coef = 0.0
    agent.policy_temperature = 1.0
    agent.value_coef = 0.5
    agent.max_grad_norm = 0.5
    agent.separate_grad_clip = True
    agent.update_epochs = 1
    agent.normalize_advantages = True
    agent.advantage_norm_min_std = 0.0
    agent.min_actor_update_steps = 2
    agent.total_updates = 0
    agent.graph_residual_aux = False
    agent.graph_residual_aux_coef = 0.0
    agent.coverage_tracker = SimpleNamespace(
        cumulative_F=0.0,
        covered_units=set(),
        covered_nodes=set(),
        covered_edges=set(),
        graph_node_weight=1.0,
        graph_edge_alpha=1.0,
        graph_edge_weight_mode="home_zero",
    )
    agent.graph_state_list = []
    return agent


def _a2c_rollout_step(reward: float, chosen_idx: int = 0):
    return {
        "action_mat": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "action_mask": torch.tensor([True, True]),
        "coverage_summary": torch.zeros(1),
        "critic_input": torch.zeros(4),
        "chosen_idx": torch.tensor(chosen_idx, dtype=torch.long),
        "reward": torch.tensor(float(reward)),
        "policy_temperature": 1.0,
    }


def _qlearning_update_test_agent():
    agent = SubWebFrontierA2CAgent.__new__(SubWebFrontierA2CAgent)
    agent.optimizer_mode = "qlearning"
    agent.actor = MaskedCandidateActor(input_dim=3, hidden_dim=8, context_dim=1)
    agent.candidate_q_head = CandidateQHead(agent.actor.embedding_dim, hidden_dim=8)
    agent.optimizer = torch.optim.Adam(
        list(agent.actor.parameters()) + list(agent.candidate_q_head.parameters()),
        lr=0.01,
    )
    agent.gamma = 0.5
    agent.max_grad_norm = 0.5
    agent.qlearning_epsilon = 0.0
    agent.qlearning_epsilon_final = 0.0
    agent.qlearning_epsilon_decay_steps = 1
    agent.qlearning_target_sync_interval = 99
    agent.candidate_q_aux = False
    agent.candidate_q_aux_coef = 0.0
    agent.total_steps = 0
    agent.total_updates = 0
    agent.coverage_tracker = SimpleNamespace(
        cumulative_F=0.0,
        covered_units=set(),
        covered_nodes=set(),
        covered_edges=set(),
        graph_node_weight=1.0,
        graph_edge_alpha=1.0,
        graph_edge_weight_mode="home_zero",
    )
    agent.graph_state_list = []
    return agent


def test_qlearning_switch_creates_primary_candidate_q_head_without_aux():
    old_optimizer = os.environ.get("WEBTEST_SUBWEB_OPTIMIZER")
    old_candidate_aux = os.environ.get("WEBTEST_CANDIDATE_Q_AUX")
    old_diag = os.environ.get("WEBTEST_POLICY_DIAGNOSTICS")
    try:
        os.environ["WEBTEST_SUBWEB_OPTIMIZER"] = "qlearning"
        os.environ["WEBTEST_CANDIDATE_Q_AUX"] = "0"
        os.environ["WEBTEST_POLICY_DIAGNOSTICS"] = "0"
        agent = SubWebFrontierA2CAgent({
            "transformer_module": "tests.test_subweb_frontier_a2c",
            "transformer_class": "_TinyTransformer",
            "hidden_dim": 8,
            "input_dim": 52,
            "rollout_len": 2,
            "max_actions": 4,
        })

        assert agent.optimizer_mode == "qlearning"
        assert agent._qlearning_enabled() is True
        assert agent.candidate_q_aux is False
        assert agent.candidate_q_head is not None
        assert agent._candidate_q_enabled() is True
        assert agent.target_actor is not None
        assert agent.target_candidate_q_head is not None
    finally:
        _restore_env("WEBTEST_SUBWEB_OPTIMIZER", old_optimizer)
        _restore_env("WEBTEST_CANDIDATE_Q_AUX", old_candidate_aux)
        _restore_env("WEBTEST_POLICY_DIAGNOSTICS", old_diag)


def test_qlearning_action_choice_respects_mask():
    agent = _qlearning_update_test_agent()
    agent._candidate_q_values = lambda action_mat, action_mask, coverage_summary: torch.tensor([1.0, 100.0, 0.0])
    action_mat = torch.zeros((3, 3))
    action_mask = torch.tensor([True, False, True])

    chosen_idx, _q_values, probs, epsilon, explore = agent._choose_qlearning_action(
        action_mat,
        action_mask,
        torch.zeros(1),
    )

    assert int(chosen_idx.item()) == 0
    assert probs[1].item() == 0.0
    assert epsilon == 0.0
    assert explore is False


def test_qlearning_update_bootstraps_from_masked_next_candidate_q():
    agent = _qlearning_update_test_agent()
    agent.rollout_buffer = [_a2c_rollout_step(1.0, chosen_idx=0)]
    next_batch = PaddedActionBatch(
        actions_full=[],
        actions_policy=[],
        action_mat=torch.zeros((3, 3)),
        action_mask=torch.tensor([True, True, False]),
        original_action_count=2,
        kept_action_count=2,
        truncated_count=0,
        truncated_by_type={},
        action_set_context=torch.zeros(0),
        structural_action_set_context=torch.zeros(0),
    )
    agent._target_candidate_q_values = (
        lambda action_mat, action_mask, coverage_summary: torch.tensor([2.0, 5.0, 100.0])
    )

    updated = agent._update_qlearning_if_ready(
        next_batch=next_batch,
        next_coverage_summary=torch.zeros(1),
        force=True,
    )

    assert updated is True
    assert agent.total_updates == 1
    assert agent.rollout_buffer == []
    assert math.isclose(agent.last_qlearning_update["bootstrap_next_max_q"], 5.0)
    assert math.isclose(agent.last_qlearning_update["target"], 3.5)


def test_a2c_all_zero_terminal_rollout_does_not_update_actor_from_critic_noise():
    torch.manual_seed(17)
    agent = _a2c_update_test_agent()
    agent.rollout_buffer = [
        _a2c_rollout_step(0.0, chosen_idx=0),
        _a2c_rollout_step(0.0, chosen_idx=1),
    ]
    actor_before = [param.detach().clone() for param in agent.actor.parameters()]
    critic_before = [param.detach().clone() for param in agent.critic.parameters()]

    updated = agent._update_if_ready(bootstrap_value=torch.tensor(0.0), force=True)

    actor_after = list(agent.actor.parameters())
    critic_after = list(agent.critic.parameters())
    assert updated is True
    assert agent.total_updates == 1
    assert agent.rollout_buffer == []
    assert all(torch.allclose(old, new.detach()) for old, new in zip(actor_before, actor_after))
    assert any(not torch.allclose(old, new.detach()) for old, new in zip(critic_before, critic_after))


def test_a2c_gae_bootstrap_without_policy_signal_updates_critic_not_actor():
    torch.manual_seed(18)
    agent = _a2c_update_test_agent()
    agent.advantage_estimator = "gae"
    with torch.no_grad():
        agent.critic.net[-1].weight.zero_()
        agent.critic.net[-1].bias.zero_()
    agent.rollout_buffer = [
        _a2c_rollout_step(0.0, chosen_idx=0),
        _a2c_rollout_step(0.0, chosen_idx=1),
    ]
    actor_before = [param.detach().clone() for param in agent.actor.parameters()]
    critic_before = [param.detach().clone() for param in agent.critic.parameters()]

    updated = agent._update_if_ready(bootstrap_value=torch.tensor(1.0), force=True)

    actor_after = list(agent.actor.parameters())
    critic_after = list(agent.critic.parameters())
    assert updated is True
    assert agent.total_updates == 1
    assert agent.rollout_buffer == []
    assert all(torch.allclose(old, new.detach()) for old, new in zip(actor_before, actor_after))
    assert any(not torch.allclose(old, new.detach()) for old, new in zip(critic_before, critic_after))


def test_a2c_graph_residual_aux_updates_actor_even_on_zero_reward_plateau():
    torch.manual_seed(181)
    agent = _a2c_update_test_agent()
    agent.advantage_estimator = "gae"
    agent.graph_residual_aux = True
    agent.graph_residual_aux_coef = 0.5
    agent.rollout_buffer = [
        {
            **_a2c_rollout_step(0.0, chosen_idx=0),
            "graph_residual_aux_target": torch.tensor([0.0, 1.0]),
        },
        {
            **_a2c_rollout_step(0.0, chosen_idx=1),
            "graph_residual_aux_target": torch.tensor([0.0, 1.0]),
        },
    ]
    actor_before = [param.detach().clone() for param in agent.actor.parameters()]

    updated = agent._update_if_ready(bootstrap_value=torch.tensor(0.0), force=True)

    actor_after = list(agent.actor.parameters())
    assert updated is True
    assert agent.total_updates == 1
    assert agent.rollout_buffer == []
    assert any(not torch.allclose(old, new.detach()) for old, new in zip(actor_before, actor_after))


def test_a2c_actor_update_still_applies_when_rollout_has_coverage_signal():
    torch.manual_seed(19)
    agent = _a2c_update_test_agent()
    agent.rollout_buffer = [
        _a2c_rollout_step(1.0, chosen_idx=0),
        _a2c_rollout_step(0.0, chosen_idx=1),
    ]
    actor_before = [param.detach().clone() for param in agent.actor.parameters()]

    updated = agent._update_if_ready(bootstrap_value=torch.tensor(0.0), force=True)

    actor_after = list(agent.actor.parameters())
    assert updated is True
    assert agent.total_updates == 1
    assert agent.rollout_buffer == []
    assert any(not torch.allclose(old, new.detach()) for old, new in zip(actor_before, actor_after))


def test_a2c_batch_mean_policy_baseline_updates_actor_from_coverage_signal():
    torch.manual_seed(23)
    agent = _a2c_update_test_agent()
    agent.policy_baseline_mode = "batch_mean"
    agent.rollout_buffer = [
        _a2c_rollout_step(1.0, chosen_idx=0),
        _a2c_rollout_step(0.0, chosen_idx=1),
    ]
    actor_before = [param.detach().clone() for param in agent.actor.parameters()]

    updated = agent._update_if_ready(bootstrap_value=torch.tensor(0.0), force=True)

    actor_after = list(agent.actor.parameters())
    assert updated is True
    assert agent.total_updates == 1
    assert agent.rollout_buffer == []
    assert any(not torch.allclose(old, new.detach()) for old, new in zip(actor_before, actor_after))


def test_policy_diagnostics_report_category_probability_mass():
    nav = ClickAction(ElementLocator.XPATH, "//nav/a", "Home", "redirect", "/")
    submit = ClickAction(ElementLocator.XPATH, "//form/button", "Search", "submit", "")
    local = ClickAction(
        ElementLocator.XPATH,
        "//main/a",
        "Create Task",
        "redirect",
        "/projects/1/tasks/new",
    )
    actions = [nav, submit, local]
    state = _state("http://localhost/projects/1", actions)
    batch = build_padded_action_batch(
        web_state=state,
        html="<html></html>",
        transformer=_TinyTransformer(),
        max_actions=4,
        input_dim=81 + structural_feature_dim(),
    )
    probs = torch.zeros(4)
    chosen_idx = 0
    for idx, action in enumerate(batch.actions_policy):
        if action.text == "Home":
            probs[idx] = 0.2
        elif action.text == "Search":
            probs[idx] = 0.3
        elif action.text == "Create Task":
            probs[idx] = 0.5
            chosen_idx = idx

    summary = summarize_policy_distribution(
        web_state=state,
        batch=batch,
        probs=probs,
        chosen_idx=chosen_idx,
    )

    assert summary["category_mass"]["global_nav"] == 0.2
    assert summary["category_mass"]["form"] == 0.3
    assert summary["category_mass"]["local_workflow"] == 0.5
    assert summary["primary_mass"]["global_nav"] == 0.2
    assert summary["primary_mass"]["submit"] == 0.3
    assert summary["primary_mass"]["local_workflow"] == 0.5
    assert summary["chosen"]["primary"] == "local_workflow"
    assert summary["mask_valid"] == 3
