from types import SimpleNamespace

import torch

from action.element_locator import ElementLocator
from action.impl.click_action import ClickAction
from action.impl.random_input_action import RandomInputAction
from action.impl.random_select_action import RandomSelectAction
from agent.impl.subweb_action_context_features import (
    action_role_features,
    action_set_context_features,
    augment_action_tensors,
    context_feature_dim,
)


def _state(actions):
    return SimpleNamespace(
        url="http://localhost:8081/owners",
        action_dict={action: {"execution_time": 0} for action in actions},
    )


def test_redirect_click_role_feature():
    action = ClickAction(ElementLocator.XPATH, "/html/body/a", "Home", "redirect", "/")
    role = action_role_features(action, _state([action]), [action])
    assert role[0].item() == 1.0
    assert role[1].item() == 1.0
    assert role[9].item() == 1.0


def test_submit_click_role_feature():
    action = ClickAction(ElementLocator.XPATH, "/html/body/button", "Save", "submit", "")
    role = action_role_features(action, _state([action]), [action])
    assert role[2].item() == 1.0
    assert role[6].item() == 1.0


def test_input_and_select_are_form_related():
    input_action = RandomInputAction(ElementLocator.XPATH, "//input", "")
    select_action = RandomSelectAction(ElementLocator.XPATH, "//select", "")
    actions = [input_action, select_action]
    state = _state(actions)
    input_role = action_role_features(input_action, state, actions)
    select_role = action_role_features(select_action, state, actions)
    assert input_role[3].item() == 1.0
    assert input_role[6].item() == 1.0
    assert select_role[4].item() == 1.0
    assert select_role[6].item() == 1.0


def test_global_nav_and_boilerplate_text():
    action = ClickAction(ElementLocator.XPATH, "//nav/a", "Logout", "redirect", "/logout")
    role = action_role_features(action, _state([action]), [action])
    assert role[7].item() == 1.0
    assert role[8].item() == 1.0


def test_page_context_has_input_and_submit_and_fractions():
    input_action = RandomInputAction(ElementLocator.XPATH, "//input", "")
    submit_action = ClickAction(ElementLocator.XPATH, "//button", "Find Owner", "submit", "")
    redirect_action = ClickAction(ElementLocator.XPATH, "//a", "Home", "redirect", "/")
    actions = [input_action, submit_action, redirect_action]
    ctx = action_set_context_features(_state(actions), actions)
    assert ctx[12].item() == 1.0
    assert ctx[13].item() == 1.0
    assert ctx[14].item() == 1.0
    assert abs(ctx[6].item() - (1.0 / 3.0)) < 1e-6
    assert abs(ctx[7].item() - (1.0 / 3.0)) < 1e-6
    assert abs(ctx[8].item() - (1.0 / 3.0)) < 1e-6


def test_feature_dim_and_augmentation_are_stable():
    action = ClickAction(ElementLocator.XPATH, "//button", "Save", "submit", "")
    base = [torch.zeros(52)]
    augmented = augment_action_tensors(base, _state([action]), [action])
    assert context_feature_dim() == 29
    assert augmented[0].shape == (52 + context_feature_dim(),)
