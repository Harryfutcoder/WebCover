import os
import sys

sys.argv = [sys.argv[0]]

from action.detector.click_action_detector import ClickActionDetector
from action.detector.random_input_action_detector import RandomInputActionDetector
from action.detector.random_select_action_detector import RandomSelectActionDetector
from action.element_locator import ElementLocator
from action.impl.click_action import ClickAction
from action.impl.random_input_action import RandomInputAction
from agent.impl.q_learning_agent import QLearningAgent
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from web_test.webtest_single_agent import Webtest


class _FakeElement:
    def __init__(self, attrs=None, tag_name="a"):
        self._attrs = attrs or {}
        self.tag_name = tag_name
        self.rect = self._attrs.get("rect", {})

    def get_attribute(self, name):
        return self._attrs.get(name)

    def find_element(self, *_args, **_kwargs):
        return _FakeElement(tag_name="body")

    def find_elements(self, *_args, **_kwargs):
        return []


class _FakeDriver:
    def __init__(self, current_url, script_result, elements_by_xpath):
        self.current_url = current_url
        self._script_result = script_result
        self._elements = elements_by_xpath

    def execute_script(self, *_args, **_kwargs):
        return self._script_result

    def find_element(self, _by, xpath):
        return self._elements[xpath]


def _click(text, href=None, element_type=None):
    attrs = {}
    if href is not None:
        attrs["href"] = href
    if element_type is not None:
        attrs["type"] = element_type
    return _FakeElement(attrs=attrs, tag_name="a")


def test_smart_input_value_uses_field_semantics():
    action = RandomInputAction(ElementLocator.XPATH, "//input", "")

    phone = _FakeElement(attrs={"type": "tel", "name": "telephone"})
    birth_date = _FakeElement(attrs={"type": "date", "name": "birthDate"})
    first_name = _FakeElement(attrs={"type": "text", "name": "firstName"})

    assert action._smart_input_value(phone) == "1234567890"
    assert action._smart_input_value(birth_date) == "2020-01-01"
    assert action._smart_input_value(first_name) == "Alex"


def test_smart_input_value_respects_maxlength():
    action = RandomInputAction(ElementLocator.XPATH, "//input", "")
    field = _FakeElement(attrs={"type": "text", "name": "firstName", "maxlength": "3"})

    assert action._smart_input_value(field) == "Ale"


def test_url_normalization_strips_auth_query_and_tracking(monkeypatch):
    monkeypatch.setenv("WEBTEST_URL_NORMALIZE_MODE", "strip_tracking")
    login = "https://github.com/login?return_to=%2Fabc&utm_source=x"
    normal = "https://github.com/search?q=abc&utm_source=x&ref_page=1"

    assert Webtest._normalize_url_for_state(login) == "https://github.com/login"
    assert Webtest._normalize_url_for_state(normal) == "https://github.com/search?q=abc"


def test_url_normalization_can_preserve_spa_hash_route(monkeypatch):
    monkeypatch.setenv("WEBTEST_URL_NORMALIZE_MODE", "strip_tracking")
    monkeypatch.setenv("WEBTEST_URL_PRESERVE_HASH_ROUTE", "1")

    url = "http://localhost:3001/#/admin/users?tab=active"

    assert Webtest._normalize_url_for_state(url) == url
    assert ClickActionDetector._normalize_redirect_url(url) == url


def test_state_similarity_is_symmetric_and_bounded():
    a1 = ClickAction(ElementLocator.XPATH, "/x/a", "A", "redirect", "https://github.com/a")
    a2 = ClickAction(ElementLocator.XPATH, "/x/b", "B", "default", "button")
    b1 = ClickAction(ElementLocator.XPATH, "/x/a", "A", "redirect", "https://github.com/a")

    s1 = ActionSetWithExecutionTimesState([a1, a2], "https://github.com/a")
    s2 = ActionSetWithExecutionTimesState([b1], "https://github.com/b")

    sim_12 = s1.similarity(s2)
    sim_21 = s2.similarity(s1)

    assert 0.0 <= sim_12 <= 1.0
    assert 0.0 <= sim_21 <= 1.0
    assert abs(sim_12 - sim_21) < 1e-9


def test_state_similarity_downweights_same_actions_on_different_urls(monkeypatch):
    monkeypatch.setenv("WEBTEST_STATE_SIMILARITY_MODE", "url_action")
    monkeypatch.setenv("WEBTEST_STATE_URL_SIM_WEIGHT", "0.35")
    monkeypatch.setenv("WEBTEST_CROSS_URL_SIM_CAP", "0.65")

    action = ClickAction(ElementLocator.XPATH, "/nav/details", "Details", "redirect", "https://example.test/details")
    s1 = ActionSetWithExecutionTimesState([action], "https://example.test/owners/3")
    s2 = ActionSetWithExecutionTimesState([action], "https://example.test/owners/7")

    sim = s1.similarity(s2)

    assert 0.0 <= sim < 1.0
    assert abs(sim - 0.65) < 1e-9


def test_state_similarity_preserves_spa_hash_routes(monkeypatch):
    monkeypatch.setenv("WEBTEST_STATE_SIMILARITY_MODE", "url_action")
    monkeypatch.setenv("WEBTEST_STATE_URL_SIM_WEIGHT", "0.35")
    monkeypatch.setenv("WEBTEST_CROSS_URL_SIM_CAP", "0.65")
    monkeypatch.setenv("WEBTEST_SIM_PRESERVE_HASH_ROUTE", "1")

    action = ClickAction(ElementLocator.XPATH, "/nav/admin", "Admin", "redirect", "http://localhost:3001/#/admin")
    s1 = ActionSetWithExecutionTimesState([action], "http://localhost:3001/#/admin")
    s2 = ActionSetWithExecutionTimesState([action], "http://localhost:3001/#/calendar")

    sim = s1.similarity(s2)

    assert 0.0 <= sim < 1.0
    assert abs(sim - 0.65) < 1e-9


def test_redirect_url_tokens_are_preserved_in_action_similarity(monkeypatch):
    monkeypatch.setenv("WEBTEST_STATE_SIMILARITY_MODE", "url_action")
    monkeypatch.setenv("WEBTEST_STATE_URL_SIM_WEIGHT", "0.35")
    monkeypatch.setenv("WEBTEST_SIM_PRESERVE_URL_TOKENS", "1")

    a1 = ClickAction(
        ElementLocator.XPATH,
        "/owners/action",
        "Add Pet",
        "redirect",
        "http://localhost:8081/owners/3/pets/new",
    )
    a2 = ClickAction(
        ElementLocator.XPATH,
        "/owners/action",
        "Add Pet",
        "redirect",
        "http://localhost:8081/owners/7/pets/new",
    )
    s1 = ActionSetWithExecutionTimesState([a1], "http://localhost:8081/owners/find")
    s2 = ActionSetWithExecutionTimesState([a2], "http://localhost:8081/owners/find")

    sim = s1.similarity(s2)

    assert 0.0 <= sim < 1.0
    # On the same URL, affordances determine similarity; sharing a page does
    # not add a constant URL bonus to disjoint redirect actions.
    assert sim == 0.0
    identical = ActionSetWithExecutionTimesState([a1], "http://localhost:8081/owners/find")
    assert s1.similarity(identical) == 1.0


def test_cgfs_same_route_same_affordances_is_one(monkeypatch):
    monkeypatch.setenv("WEBTEST_STATE_SIMILARITY_MODE", "cgfs_lite")

    action = ClickAction(ElementLocator.XPATH, "/html/body/a", "Edit Owner", "redirect", "/owners/123/edit")
    s1 = ActionSetWithExecutionTimesState([action], "http://localhost:8081/owners/123/edit")
    s2 = ActionSetWithExecutionTimesState([action], "http://localhost:8081/owners/123/edit")

    assert abs(s1.similarity(s2) - 1.0) < 1e-9


def test_cgfs_same_route_disjoint_affordances_is_zero(monkeypatch):
    monkeypatch.setenv("WEBTEST_STATE_SIMILARITY_MODE", "cgfs_lite")

    s1 = ActionSetWithExecutionTimesState(
        [ClickAction(ElementLocator.XPATH, "/html/body/a[1]", "Edit Owner", "redirect", "/owners/123/edit")],
        "http://localhost:8081/owners/123/edit",
    )
    s2 = ActionSetWithExecutionTimesState(
        [ClickAction(ElementLocator.XPATH, "/html/body/button[9]", "Delete Pet", "submit", "button")],
        "http://localhost:8081/owners/123/edit",
    )

    assert abs(s1.similarity(s2) - 0.0) < 1e-9


def test_cgfs_different_route_same_affordances_uses_discount(monkeypatch):
    monkeypatch.setenv("WEBTEST_STATE_SIMILARITY_MODE", "cgfs_lite")
    monkeypatch.setenv("WEBTEST_CGFS_CROSS_ROUTE_DISCOUNT", "0.25")

    action = ClickAction(ElementLocator.XPATH, "/html/body/a", "Save", "submit", "button")
    s1 = ActionSetWithExecutionTimesState([action], "http://localhost:8081/owners/123/edit")
    s2 = ActionSetWithExecutionTimesState([action], "http://localhost:8081/pets/123/edit")

    assert abs(s1.similarity(s2) - 0.25) < 1e-9


def test_cgfs_different_hash_routes_are_different_context(monkeypatch):
    monkeypatch.setenv("WEBTEST_STATE_SIMILARITY_MODE", "cgfs_lite")
    monkeypatch.setenv("WEBTEST_CGFS_CROSS_ROUTE_DISCOUNT", "0.25")

    action = ClickAction(ElementLocator.XPATH, "/html/body/a", "Open", "redirect", "#/editor")
    s1 = ActionSetWithExecutionTimesState([action], "http://localhost:3001/#/editor/abc123")
    s2 = ActionSetWithExecutionTimesState([action], "http://localhost:3001/#/article/abc123ef")

    assert ActionSetWithExecutionTimesState._normalize_route_identity("http://localhost:3001/#/editor/abc123") == "/#/editor/{id}"
    assert abs(s1.similarity(s2) - 0.25) < 1e-9


def test_cgfs_empty_affordance_sets_are_zero(monkeypatch):
    monkeypatch.setenv("WEBTEST_STATE_SIMILARITY_MODE", "cgfs_lite")

    s1 = ActionSetWithExecutionTimesState([], "http://localhost:8081/owners")
    s2 = ActionSetWithExecutionTimesState([], "http://localhost:8081/owners")

    assert abs(s1.similarity(s2) - 0.0) < 1e-9


def test_cgfs_normalizes_dynamic_route_ids(monkeypatch):
    monkeypatch.setenv("WEBTEST_STATE_SIMILARITY_MODE", "cgfs_lite")
    monkeypatch.setenv("WEBTEST_CGFS_CROSS_ROUTE_DISCOUNT", "0.25")

    action = ClickAction(ElementLocator.XPATH, "/html/body/a", "Edit Owner", "redirect", "/owners/123/edit")
    s1 = ActionSetWithExecutionTimesState([action], "http://localhost:8081/owners/123/edit")
    s2 = ActionSetWithExecutionTimesState([action], "http://localhost:8081/owners/456/edit")

    assert (
        ActionSetWithExecutionTimesState._normalize_route_identity("http://localhost:8081/owners/123/edit")
        == ActionSetWithExecutionTimesState._normalize_route_identity("http://localhost:8081/owners/456/edit")
    )
    assert abs(s1.similarity(s2) - 1.0) < 1e-9


def test_cgfs_normalizes_uuid_route_ids(monkeypatch):
    monkeypatch.setenv("WEBTEST_STATE_SIMILARITY_MODE", "cgfs_lite")

    action = ClickAction(ElementLocator.XPATH, "/html/body/a", "Edit Item", "redirect", "/items/id/edit")
    url = "http://localhost:8081/items/550e8400-e29b-41d4-a716-446655440000/edit"
    s1 = ActionSetWithExecutionTimesState([action], url)
    s2 = ActionSetWithExecutionTimesState([action], "http://localhost:8081/items/11111111-2222-4333-8444-555555555555/edit")

    assert ActionSetWithExecutionTimesState._normalize_route_identity(url) == "/items/{id}/edit"
    assert abs(s1.similarity(s2) - 1.0) < 1e-9


def test_cgfs_drops_tracking_query_params(monkeypatch):
    monkeypatch.setenv("WEBTEST_STATE_SIMILARITY_MODE", "cgfs_lite")

    action = ClickAction(ElementLocator.XPATH, "/html/body/a", "Owner", "redirect", "/owners/1")
    s1 = ActionSetWithExecutionTimesState([action], "http://localhost:8081/owners/1?utm_source=x")
    s2 = ActionSetWithExecutionTimesState([action], "http://localhost:8081/owners/1?utm_source=y")

    assert (
        ActionSetWithExecutionTimesState._normalize_route_identity("http://localhost:8081/owners/1?utm_source=x")
        == ActionSetWithExecutionTimesState._normalize_route_identity("http://localhost:8081/owners/1?utm_source=y")
    )
    assert abs(s1.similarity(s2) - 1.0) < 1e-9


def test_cgfs_functional_ignores_boilerplate_when_form_schema_matches(monkeypatch):
    monkeypatch.setenv("WEBTEST_STATE_SIMILARITY_MODE", "cgfs_functional")

    field = RandomInputAction(ElementLocator.XPATH, "/html/body/main/form/input[1]", "Name")
    home = ClickAction(ElementLocator.XPATH, "/html/body/nav/a[1]", "Home", "redirect", "/")
    error = ClickAction(ElementLocator.XPATH, "/html/body/nav/a[2]", "Error", "redirect", "/error")
    s1 = ActionSetWithExecutionTimesState([home, field], "http://localhost:8081/owners/new")
    s2 = ActionSetWithExecutionTimesState([error, field], "http://localhost:8081/owners/new")

    assert abs(s1.similarity(s2) - 1.0) < 1e-9


def test_cgfs_functional_discounts_same_form_schema_across_routes(monkeypatch):
    monkeypatch.setenv("WEBTEST_STATE_SIMILARITY_MODE", "cgfs_functional")
    monkeypatch.setenv("WEBTEST_CGFS_CROSS_ROUTE_DISCOUNT", "0.25")

    field = RandomInputAction(ElementLocator.XPATH, "/html/body/main/form/input[1]", "Date")
    s1 = ActionSetWithExecutionTimesState([field], "http://localhost:8081/owners/123/pets/new")
    s2 = ActionSetWithExecutionTimesState([field], "http://localhost:8081/owners/123/pets/456/visits/new")

    assert abs(s1.similarity(s2) - 0.25) < 1e-9


def test_cgfs_functional_keeps_pure_boilerplate_from_becoming_repeat_novelty(monkeypatch):
    monkeypatch.setenv("WEBTEST_STATE_SIMILARITY_MODE", "cgfs_functional")

    home = ClickAction(ElementLocator.XPATH, "/html/body/nav/a[1]", "Home", "redirect", "/")
    error = ClickAction(ElementLocator.XPATH, "/html/body/nav/a[2]", "Error", "redirect", "/error")
    s1 = ActionSetWithExecutionTimesState([home], "http://localhost:8081/error")
    s2 = ActionSetWithExecutionTimesState([error], "http://localhost:8081/error")

    assert abs(s1.similarity(s2) - 1.0) < 1e-9


def test_cgfs_functional_retains_workflow_links(monkeypatch):
    monkeypatch.setenv("WEBTEST_STATE_SIMILARITY_MODE", "cgfs_functional")

    add_pet = ClickAction(
        ElementLocator.XPATH,
        "/html/body/main/table/tr/td/a[1]",
        "Add Pet",
        "redirect",
        "/owners/123/pets/new",
    )
    add_visit = ClickAction(
        ElementLocator.XPATH,
        "/html/body/main/table/tr/td/a[2]",
        "Add Visit",
        "redirect",
        "/owners/123/pets/456/visits/new",
    )
    s1 = ActionSetWithExecutionTimesState([add_pet], "http://localhost:8081/owners/123")
    s2 = ActionSetWithExecutionTimesState([add_visit], "http://localhost:8081/owners/123")

    assert abs(s1.similarity(s2) - 0.0) < 1e-9


def test_click_detector_blocks_auth_actions_but_keeps_exit_links(monkeypatch):
    monkeypatch.setenv("WEBTEST_BLOCK_AUTH_ACTIONS", "1")
    detector = ClickActionDetector()

    script_result = [
        {"visible": True, "xpath": "/auth/link", "text": "Pricing"},
        {"visible": True, "xpath": "/auth/submit", "text": "Continue with Google"},
        {"visible": True, "xpath": "/auth/signin", "text": "Sign in"},
    ]
    elements = {
        "/auth/link": _click(text="Pricing", href="https://github.com/pricing"),
        "/auth/submit": _click(text="Continue with Google", href=None, element_type="submit"),
        "/auth/signin": _click(text="Sign in", href="https://github.com/login?return_to=%2Ffoo"),
    }
    driver = _FakeDriver(
        current_url="https://github.com/login?return_to=%2Ffoo",
        script_result=script_result,
        elements_by_xpath=elements,
    )

    actions = detector.get_actions(driver)
    assert len(actions) == 1
    assert actions[0].text == "Pricing"


def test_input_select_detectors_block_auth_page(monkeypatch):
    monkeypatch.setenv("WEBTEST_BLOCK_AUTH_ACTIONS", "1")
    input_detector = RandomInputActionDetector()
    select_detector = RandomSelectActionDetector()

    script_result = [{"visible": True, "xpath": "/x", "text": "Email"}]
    driver = _FakeDriver(
        current_url="https://github.com/signup",
        script_result=script_result,
        elements_by_xpath={"/x": _FakeElement()},
    )

    assert input_detector.get_actions(driver) == []
    assert select_detector.get_actions(driver) == []


def test_click_detector_can_disable_auth_filter(monkeypatch):
    monkeypatch.setenv("WEBTEST_BLOCK_AUTH_ACTIONS", "0")
    detector = ClickActionDetector()

    script_result = [{"visible": True, "xpath": "/x", "text": "Sign in"}]
    elements = {"/x": _click(text="Sign in", href="https://github.com/login")}
    driver = _FakeDriver(
        current_url="https://github.com/",
        script_result=script_result,
        elements_by_xpath=elements,
    )

    actions = detector.get_actions(driver)
    assert len(actions) == 1


def test_qlearning_w_actionset_mode_merges_similar_states(monkeypatch):
    monkeypatch.setenv("WEBTEST_QLEARNING_AGENT_TYPE", "W")
    monkeypatch.setenv("WEBTEST_QLEARNING_STATE_MODE", "actionset")
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
    a = ClickAction(ElementLocator.XPATH, "/x/a", "A", "redirect", "https://github.com/features")
    s1 = ActionSetWithExecutionTimesState([a], "https://github.com/features")
    s2 = ActionSetWithExecutionTimesState([a], "https://github.com/features")

    idx1 = agent.get_state_index(s1, "<html></html>")
    idx2 = agent.get_state_index(s2, "<html></html>")
    assert idx1 == idx2
