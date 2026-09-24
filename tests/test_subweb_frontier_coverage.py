from action.element_locator import ElementLocator
from action.impl.click_action import ClickAction
from action.impl.random_input_action import RandomInputAction
from action.impl.random_select_action import RandomSelectAction
from agent.impl.subweb_frontier_coverage import (
    FrontierCoverageTracker,
    extract_frontier_units,
    unit_class,
)
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState


def _state(url, actions):
    return ActionSetWithExecutionTimesState(actions, url)


def test_extract_frontier_units_keeps_route_and_functional_form_units():
    actions = [
        RandomInputAction(ElementLocator.XPATH, "//form/input[@name='name']", "Name"),
        RandomSelectAction(ElementLocator.XPATH, "//form/select", "Type"),
        ClickAction(ElementLocator.XPATH, "//form/button", "Save", "submit", ""),
        ClickAction(ElementLocator.XPATH, "//nav/a", "Home", "redirect", "/"),
    ]
    units = extract_frontier_units(_state("http://localhost:8081/owners/123/pets/new", actions))

    assert "context:/owners/{id}/pets/new" in units
    assert any("family=form_field" in unit and "kind=input:text" in unit for unit in units)
    assert any("family=form_field" in unit and "kind=select" in unit for unit in units)
    assert any("family=form_submit" in unit and "name=save" in unit for unit in units)
    assert not any("name=home" in unit for unit in units)


def test_unit_class_is_coarse_and_site_independent():
    assert unit_class("context:/owners/{id}") == "context"
    assert unit_class("frontier:/x|family=form_field|kind=input:text|name=name") == "field:input:text"
    assert unit_class("frontier:/x|family=form_submit|kind=submit|name=save") == "submit:save"
    assert unit_class("frontier:/x|family=main_action|kind=click|name=send invitations") == "main-action:send"


def test_frontier_uniform_rewards_only_new_units():
    action = ClickAction(ElementLocator.XPATH, "//button", "Save", "submit", "")
    state = _state("http://localhost:8081/owners/new", [action])
    tracker = FrontierCoverageTracker("frontier_uniform")

    first = tracker.compute_reward(state, chosen_action=action, source_url="http://localhost:8081/owners")
    second = tracker.compute_reward(state, chosen_action=action, source_url="http://localhost:8081/owners")

    assert first > 0.0
    assert second == 0.0
    assert tracker.summary()["covered_units"] >= 2


def test_frontier_density_and_edge_modes_are_bounded():
    action = ClickAction(ElementLocator.XPATH, "//button", "Create", "submit", "")
    state1 = _state("http://localhost:4200/new", [action])
    state2 = _state("http://localhost:4200/event/123", [action])

    static_tracker = FrontierCoverageTracker("frontier_value_density_static")
    dynamic_tracker = FrontierCoverageTracker("frontier_value_density_dynamic")
    edge_tracker = FrontierCoverageTracker("frontier_uniform_edge_bonus")

    assert 0.0 <= static_tracker.compute_reward(state1, action, "http://localhost:4200") <= 1.0
    assert 0.0 <= dynamic_tracker.compute_reward(state1, action, "http://localhost:4200") <= 1.0
    edge_reward = edge_tracker.compute_reward(state2, action, "http://localhost:4200/new")
    assert 0.0 <= edge_reward <= 1.25
    assert edge_tracker.summary()["unique_edges"] == 1
