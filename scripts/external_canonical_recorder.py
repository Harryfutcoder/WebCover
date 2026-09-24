#!/usr/bin/env python
"""Write WebTest-compatible coverage snapshots for external baselines.

External tools such as official WebRLED and QExplore keep their own policy,
state abstraction, and reward logic.  This recorder is an observability side
channel: it converts each observed action set and executed action into the same
canonical WebTest snapshot schema used by DataCollector, so table metrics can
be computed with the same parser.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from utils import normalize_url_for_state

try:
    from action.element_locator import ElementLocator
    from action.impl.click_action import ClickAction
    from action.impl.random_input_action import RandomInputAction
    from action.impl.random_select_action import RandomSelectAction
    from action.impl.restart_action import RestartAction
    from action.web_action import WebAction
    from state.impl.action_execute_failed_state import ActionExecuteFailedState
    from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
    from state.impl.out_of_domain_state import OutOfDomainState
    from state.impl.same_url_state import SameUrlState
    from state.web_state import WebState
except Exception:
    class _Locator:
        value = "xpath"

        def __str__(self) -> str:
            return "ElementLocator.XPATH"

    class ElementLocator:
        XPATH = _Locator()

    class WebAction:
        pass

    class ClickAction(WebAction):
        def __init__(self, locator: _Locator, location: str, text: str, action_type: str, addition_info: str) -> None:
            self.locator = locator
            self.location = location
            self.text = text
            self.action_type = action_type
            self.addition_info = addition_info

        def __eq__(self, other: object) -> bool:
            return isinstance(other, ClickAction) and (
                self.locator,
                self.location,
                self.text,
                self.action_type,
                self.addition_info,
            ) == (
                other.locator,
                other.location,
                other.text,
                other.action_type,
                other.addition_info,
            )

        def __hash__(self) -> int:
            return hash((self.locator, self.location, self.text, self.action_type, self.addition_info))

        def __lt__(self, other: object) -> bool:
            return str(self) < str(other)

        def __str__(self) -> str:
            return (
                f"ClickAction(locator={self.locator}, location={self.location}, text={self.text}, "
                f"action_type={self.action_type}, addition_info={self.addition_info})"
            )

    class RandomInputAction(WebAction):
        def __init__(self, locator: _Locator, location: str, text: str) -> None:
            self.locator = locator
            self.location = location
            self.text = text

        def __eq__(self, other: object) -> bool:
            return isinstance(other, RandomInputAction) and (
                self.locator,
                self.location,
                self.text,
            ) == (other.locator, other.location, other.text)

        def __hash__(self) -> int:
            return hash((self.locator, self.location, self.text))

        def __lt__(self, other: object) -> bool:
            return str(self) < str(other)

        def __str__(self) -> str:
            return f"RandomInputAction(locator={self.locator}, location={self.location}, text={self.text})"

    class RandomSelectAction(RandomInputAction):
        def __str__(self) -> str:
            return f"RandomSelectAction(locator={self.locator}, location={self.location}, text={self.text})"

    class RestartAction(WebAction):
        pass

    class WebState:
        pass

    class _NeverState(WebState):
        pass

    ActionExecuteFailedState = _NeverState
    OutOfDomainState = _NeverState
    SameUrlState = _NeverState

    class ActionSetWithExecutionTimesState(WebState):
        def __init__(self, actions: list[WebAction], url: str) -> None:
            self.action_dict = {key: {"execution_time": 0, "child_state": None} for key in actions}
            self.action_execution_time_histogram = [0] * 10
            self.action_execution_time_histogram[0] = len(self.action_dict)
            self.url = url

        def get_action_list(self) -> list[WebAction]:
            return sorted(self.action_dict.keys())

        def get_action_detailed_data(self) -> tuple[dict[WebAction, Any], Any]:
            return self.action_dict, self.action_execution_time_histogram

        def update_action_execution_time(self, action: WebAction) -> None:
            if action not in self.action_dict:
                raise KeyError("action not in state")
            self.action_dict[action]["execution_time"] += 1
            execution_time = self.action_dict[action]["execution_time"]
            if execution_time < 10:
                self.action_execution_time_histogram[execution_time - 1] -= 1
                self.action_execution_time_histogram[execution_time] += 1

        def update_transition_information(self, action: WebAction, new_state: WebState) -> None:
            if action not in self.action_dict:
                raise KeyError("action not in state")
            self.action_dict[action]["child_state"] = new_state

        def __eq__(self, other: object) -> bool:
            return isinstance(other, ActionSetWithExecutionTimesState) and (
                self.action_dict.keys() == other.action_dict.keys()
                and self.url == other.url
            )

        def __hash__(self) -> int:
            return hash(self.url) + sum(hash(action) for action in self.action_dict.keys())

        def __lt__(self, other: object) -> bool:
            return hash(self) < hash(other) if isinstance(other, ActionSetWithExecutionTimesState) else str(type(self)) < str(type(other))

        def __str__(self) -> str:
            return (
                f"ActionSetWithExecutionTimesState(action_number={len(self.action_dict)}, "
                f"action_execution_time_histogram={self.action_execution_time_histogram}, url={self.url})"
            )


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def recorder_from_env(
    default_algorithm: str,
    default_site: str = "",
    default_seed: str = "",
    default_profile: str = "",
) -> "ExternalCanonicalRecorder | None":
    if not _env_enabled("WEBTEST_EXTERNAL_CANONICAL", "0"):
        return None
    algorithm = os.environ.get("WEBTEST_EXTERNAL_CANONICAL_ALGORITHM", default_algorithm).strip() or default_algorithm
    site = os.environ.get("WEBTEST_EXTERNAL_CANONICAL_SITE", default_site).strip() or default_site
    seed = os.environ.get("WEBTEST_EXTERNAL_CANONICAL_SEED", default_seed).strip() or default_seed
    profile = os.environ.get("WEBTEST_EXTERNAL_CANONICAL_PROFILE", default_profile).strip() or default_profile
    return ExternalCanonicalRecorder(site=site, algorithm=algorithm, seed=seed, profile=profile)


class ExternalCanonicalRecorder:
    DEFAULT_PROFILES = {
        "webrled": "drl-1agent-observation",
        "webrled-official": "drl-1agent-observation",
        "qexplore": "qexplore-1agent",
    }
    _ATTR_RE_TEMPLATE = r"""{name}\s*=\s*["']([^"']*)["']"""

    def __init__(self, site: str, algorithm: str, seed: str, profile: str = "") -> None:
        self.site = self._clean_run_part(site or "unknown")
        self.algorithm = self._clean_run_part(algorithm or "external")
        self.seed = self._clean_run_part(seed or "seedexternal")
        self.profile = self._clean_run_part(profile or self.DEFAULT_PROFILES.get(self.algorithm, f"{self.algorithm}-1agent"))

        repo_root = Path(os.environ.get("WEBTEST_REPO_ROOT", "") or Path.cwd()).resolve()
        output_root = Path(
            os.environ.get("WEBTEST_EXTERNAL_CANONICAL_OUTPUT_ROOT", "")
            or (repo_root / "webtest_output" / "result")
        )
        self.run_name = f"{self.site}-{self.profile}-{self.algorithm}-{self.seed}"
        self.output_path = output_root / self.run_name
        self.output_data_path = self.output_path / "output_data"
        self.output_data_path.mkdir(parents=True, exist_ok=True)

        self.lock = threading.RLock()
        self.action_dict: dict[WebAction, int] = {}
        self.state_dict: dict[WebState, int] = {}
        self.url_count_dict: dict[str, int] = {}
        self.transition_record_list: list[tuple[WebState | None, WebAction | None, WebState]] = []
        self._finalized = False
        atexit.register(self.finalize)

    @staticmethod
    def _clean_run_part(value: str) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"[^a-z0-9_.-]+", "_", text)
        return text.strip("_") or "unknown"

    @staticmethod
    def _clean_text(value: Any, max_len: int = 160) -> str:
        text = " ".join(str(value or "").split())
        return text[:max_len]

    @classmethod
    def _html_attr(cls, html: Any, name: str) -> str:
        text = str(html or "")
        pattern = cls._ATTR_RE_TEMPLATE.format(name=re.escape(name))
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _xpath_location(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "external://unknown"
        if text.startswith(("/", "(", ".")):
            return text
        return f"/{text}"

    def _action_from_webrled(self, action_info: dict[str, Any]) -> WebAction:
        html = action_info.get("outerHTML") or action_info.get("shtml") or ""
        tag = str(action_info.get("tagName") or self._html_attr(html, "tag") or "element").strip().lower()
        action_type_num = str(action_info.get("actiontype") or "").strip()
        xpath = self._xpath_location(action_info.get("xpath"))
        text = self._clean_text(
            action_info.get("innerText")
            or action_info.get("value")
            or self._html_attr(html, "aria-label")
            or self._html_attr(html, "title")
            or self._html_attr(html, "name")
            or tag
        )
        input_type = (self._html_attr(html, "type") or "").strip().lower()
        href = self._html_attr(html, "href")
        form_action = self._html_attr(html, "action")

        if action_type_num == "2" or tag in {"input", "textarea"}:
            if input_type in {"button", "submit", "radio", "checkbox", "image", "reset"}:
                return ClickAction(ElementLocator.XPATH, xpath, text, "submit" if input_type == "submit" else "default", tag)
            return RandomInputAction(ElementLocator.XPATH, xpath, text or input_type or tag)
        if action_type_num == "3" or tag == "select":
            return RandomSelectAction(ElementLocator.XPATH, xpath, text or tag)
        if action_type_num == "4" or tag == "form":
            return ClickAction(ElementLocator.XPATH, xpath, text or "submit", "submit", form_action or tag)
        if tag == "a" and href:
            return ClickAction(ElementLocator.XPATH, xpath, text or href, "redirect", href)
        return ClickAction(ElementLocator.XPATH, xpath, text or tag, "default", href or form_action or tag)

    def _action_from_qexplore(self, raw: str) -> WebAction:
        parts = str(raw or "").split("!@!")
        while len(parts) < 4:
            parts.append("nan")
        tag, name, value, href = [part.strip() for part in parts[:4]]
        tag_l = (tag or "element").lower()
        name = "" if name == "nan" else name
        value = "" if value == "nan" else value
        href = "" if href == "nan" else href
        label = self._clean_text(name or value or href or tag_l)
        loc = f"qexplore://{tag_l}/{name or '_'}|{value or '_'}|{href or '_'}"
        if tag_l == "input":
            input_type = value.lower()
            if input_type in {"submit", "button", "radio", "checkbox", "image", "reset"}:
                return ClickAction(ElementLocator.XPATH, loc, label or tag_l, "submit" if input_type == "submit" else "default", tag_l)
            return RandomInputAction(ElementLocator.XPATH, loc, label or tag_l)
        if tag_l == "select":
            return RandomSelectAction(ElementLocator.XPATH, loc, label or tag_l)
        if tag_l == "a" and href:
            return ClickAction(ElementLocator.XPATH, loc, label or href, "redirect", href)
        action_type = "submit" if tag_l in {"submit", "form"} else "default"
        return ClickAction(ElementLocator.XPATH, loc, label or tag_l, action_type, href or tag_l)

    def _to_web_action(self, raw: Any) -> WebAction:
        if isinstance(raw, WebAction):
            return raw
        if isinstance(raw, dict):
            return self._action_from_webrled(raw)
        return self._action_from_qexplore(str(raw or ""))

    def _to_web_actions(self, raw_actions: Iterable[Any] | None) -> list[WebAction]:
        seen: set[WebAction] = set()
        actions: list[WebAction] = []
        for raw in raw_actions or []:
            try:
                action = self._to_web_action(raw)
            except Exception:
                continue
            if action not in seen:
                seen.add(action)
                actions.append(action)
        return actions

    def _build_state(self, raw_url: str, raw_actions: Iterable[Any] | None) -> ActionSetWithExecutionTimesState:
        normalized_url = normalize_url_for_state(raw_url or "")
        state = ActionSetWithExecutionTimesState(self._to_web_actions(raw_actions), normalized_url)
        try:
            state.raw_url = raw_url or normalized_url
        except Exception:
            pass
        return state

    def _add_or_get_state(self, state: WebState, increment: bool = True) -> WebState:
        for existing in self.state_dict.keys():
            if existing == state:
                if increment:
                    self.state_dict[existing] += 1
                return existing
        self.state_dict[state] = 1 if increment else 0
        return state

    def observe_state(self, raw_url: str, raw_actions: Iterable[Any] | None) -> WebState:
        with self.lock:
            state = self._add_or_get_state(self._build_state(raw_url, raw_actions), increment=True)
            for action in state.get_action_list():
                self.action_dict.setdefault(action, 0)
            if raw_url:
                norm = normalize_url_for_state(raw_url)
                self.url_count_dict[norm] = self.url_count_dict.get(norm, 0) + 1
            self.save_data()
            return state

    def record_transition(
        self,
        source_url: str,
        source_actions: Iterable[Any] | None,
        selected_action: Any,
        target_url: str,
        target_actions: Iterable[Any] | None,
    ) -> None:
        with self.lock:
            selected = self._to_web_action(selected_action) if selected_action is not None else None
            source_raw_actions = list(source_actions or [])
            if selected is not None:
                source_raw_actions.append(selected)
            source_state = self._add_or_get_state(self._build_state(source_url, source_raw_actions), increment=True)
            target_state = self._add_or_get_state(self._build_state(target_url, target_actions), increment=True)

            for action in source_state.get_action_list():
                self.action_dict.setdefault(action, 0)
            for action in target_state.get_action_list():
                self.action_dict.setdefault(action, 0)

            if selected is not None:
                self.action_dict[selected] = self.action_dict.get(selected, 0) + 1
                try:
                    source_state.update_action_execution_time(selected)
                except Exception:
                    pass
                try:
                    source_state.update_transition_information(selected, target_state)
                except Exception:
                    pass

            if target_url:
                norm = normalize_url_for_state(target_url)
                self.url_count_dict[norm] = self.url_count_dict.get(norm, 0) + 1
            self.transition_record_list.append((source_state, selected, target_state))
            self.save_data()

    def _snapshot_payload(self) -> dict[str, Any]:
        action_list = sorted(self.action_dict.keys())
        action_list_with_execution_time = [(str(key), self.action_dict[key]) for key in action_list]
        state_list = sorted(self.state_dict.keys())
        state_dict_list: list[dict[str, Any]] = []
        for state in state_list:
            if (
                not isinstance(state, ActionExecuteFailedState)
                and not isinstance(state, OutOfDomainState)
                and not isinstance(state, SameUrlState)
            ):
                action_index_list = [
                    action_list.index(action)
                    for action in state.get_action_list()
                    if action in action_list
                ]
                state_dict: dict[str, Any] = {
                    "info": str(state),
                    "action_list": action_index_list,
                    "visited_time": self.state_dict[state],
                }
                if isinstance(state, ActionSetWithExecutionTimesState):
                    detailed_data_index_dict: dict[str, Any] = {}
                    for action, data_dict in state.get_action_detailed_data()[0].items():
                        action_key: int | str
                        if isinstance(action, RestartAction) or action not in action_list:
                            action_key = str(action)
                        else:
                            action_key = action_list.index(action)
                        child_state = data_dict.get("child_state")
                        detailed_data_index_dict[str(action_key)] = {
                            "execution_time": data_dict.get("execution_time", 0),
                            "child_state": state_list.index(child_state) if child_state in state_list else None,
                        }
                    state_dict["detailed_data"] = detailed_data_index_dict
                state_dict_list.append(state_dict)
            else:
                state_dict_list.append(
                    {
                        "info": str(state),
                        "action_list": list(map(str, state.get_action_list())),
                        "visited_time": self.state_dict[state],
                    }
                )

        transition_tuple_list = []
        for source_state, action, target_state in self.transition_record_list:
            if action is None:
                action_id = None
            elif isinstance(action, RestartAction) or action not in action_list:
                action_id = str(action)
            else:
                action_id = action_list.index(action)
            transition_tuple_list.append(
                (
                    state_list.index(source_state) if source_state in state_list else None,
                    action_id,
                    state_list.index(target_state),
                )
            )
        return {
            "action_list": action_list_with_execution_time,
            "state_list": state_dict_list,
            "url_count": dict(self.url_count_dict),
            "transition_list": transition_tuple_list,
            "canonical_metric_source": {
                "schema_version": 1,
                "algorithm": self.algorithm,
                "site": self.site,
                "seed": self.seed,
                "profile": self.profile,
                "note": "External baseline policy preserved; WebTest-compatible accounting side channel.",
            },
        }

    def save_data(self, finish: bool = False) -> None:
        payload = self._snapshot_payload()
        stamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        if finish:
            with (self.output_data_path / f"{stamp}-finish.json").open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4, sort_keys=True, ensure_ascii=True)
        tmp = self.output_data_path / "newest.json.tmp"
        newest = self.output_data_path / "newest.json"
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, sort_keys=True, ensure_ascii=True)
        tmp.replace(newest)

    def finalize(self) -> None:
        with self.lock:
            if self._finalized:
                return
            self._finalized = True
            self.save_data(finish=True)
