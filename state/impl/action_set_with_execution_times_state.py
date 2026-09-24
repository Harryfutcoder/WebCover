import math
import os
import re
from collections import Counter
from collections import defaultdict
from typing import List, Dict, Union, Any, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from action.impl.click_action import ClickAction
from action.impl.random_input_action import RandomInputAction
from action.impl.random_select_action import RandomSelectAction
from action.web_action import WebAction
from exceptions import WebtestException
from state.web_state import WebState
from utils import normalize_url_for_state


class ActionSetWithExecutionTimesState(WebState):
    _BUCKET_WEIGHTS = {
        "redirect": 0.75,
        "submit": 0.25,
        "default": 0.10,
    }
    _UUID_RE = re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
        re.IGNORECASE,
    )
    _EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE)
    _DATE_RE = re.compile(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b")
    _LONG_TOKEN_RE = re.compile(r"\b(?=[a-z0-9_-]{16,}\b)(?=[a-z0-9_-]*[a-z])(?=[a-z0-9_-]*\d)[a-z0-9_-]+\b", re.IGNORECASE)
    _NUMBER_RE = re.compile(r"(?<![a-z])[-+]?\d+(?:\.\d+)?(?![a-z])", re.IGNORECASE)
    _URL_LIKE_RE = re.compile(r"^(?:https?://|/)", re.IGNORECASE)
    _ROUTE_UUID_RE = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )
    _ROUTE_HEX_RE = re.compile(r"^(?:(?=[0-9a-f]{6,}$)(?=.*\d)[0-9a-f]+|0x[0-9a-f]+)$", re.IGNORECASE)
    _ROUTE_MIXED_ID_RE = re.compile(
        r"^(?=[a-z0-9_-]{16,}$)(?=[a-z0-9_-]*[a-z])(?=[a-z0-9_-]*\d)[a-z0-9_-]+$",
        re.IGNORECASE,
    )
    _FUNCTIONAL_BOILERPLATE_LABELS = {
        "",
        "home",
        "root",
        "overview",
        "back",
        "cancel",
        "close",
        "error",
        "errors",
        "logout",
        "log out",
        "login",
        "log in",
        "sign in",
        "sign out",
        "menu",
        "about",
        "help",
        "docs",
        "documentation",
        "features",
        "pricing",
        "veterinarians",
        "owners",
        "pets",
    }
    _FUNCTIONAL_MAIN_KEYWORDS = {
        "add",
        "admin",
        "article",
        "create",
        "delete",
        "edit",
        "event",
        "find",
        "invite",
        "invitation",
        "new",
        "owner",
        "participant",
        "pet",
        "publish",
        "remove",
        "save",
        "search",
        "send",
        "settings",
        "share",
        "submit",
        "update",
        "visit",
    }
    _FUNCTIONAL_STRUCTURE_KEYWORDS = {
        "dialog",
        "modal",
        "row",
        "table",
        "tab",
        "view",
        "filter",
    }

    def __init__(self, actions: List[WebAction], url: str) -> None:
        self.action_dict: Dict[WebAction, Dict[str, Union[int, WebState]]] = {
            key: {'execution_time': 0, 'child_state': None} for key in actions}
        self.action_execution_time_histogram: List[int] = [0] * 10
        self.action_execution_time_histogram[0] = len(self.action_dict)
        self.url: str = url
        self.sim_dic = defaultdict(float)

        self.global_url_list = set()  # 用来保存所有网页中可能出现的 URL
        self.global_form_list = set()  # 用来保存可能的form的操作次数
        self.global_tag_list = set()  # 用来保存可能的action的type

    def get_action_list(self) -> List[WebAction]:
        action_list = list(self.action_dict.keys())
        action_list.sort()
        return action_list

    def get_action_detailed_data(self) -> Tuple[Dict[WebAction, Any], Any]:
        return self.action_dict, self.action_execution_time_histogram

    def update_action_execution_time(self, action: WebAction) -> None:
        if action in self.action_dict:
            self.action_dict[action]['execution_time'] += 1
            execution_time = self.action_dict[action]['execution_time']
            if execution_time < 10:
                self.action_execution_time_histogram[execution_time - 1] -= 1
                self.action_execution_time_histogram[execution_time] += 1
        else:
            raise WebtestException("The action is not exist in the state")

    def update_transition_information(self, action: WebAction, new_state: 'WebState') -> None:
        if action in self.action_dict:
            self.action_dict[action]['child_state'] = new_state
        else:
            raise WebtestException("The action is not exist in the state")

    # def similarity(self, other: 'WebState') -> float:
    #     if not isinstance(other, ActionSetWithExecutionTimesState):
    #         return 0
    #     if not self.sim_dic.__contains__(other):
    #         action_list_self = set(self.get_action_list())
    #         action_list_other = set(other.get_action_list())
    #
    #         if len(action_list_self) == 0 and len(action_list_other) == 0:
    #             return 1.0
    #         intersection = action_list_self.intersection(action_list_other)
    #         union = action_list_self.union(action_list_other)
    #         self.sim_dic[other] = len(intersection) / len(union)
    #         other.sim_dic[other] = len(intersection) / len(union)
    #     return self.sim_dic[other]

    def cosine_similarity(self, X: List[int], Y: List[int]) -> float:
        dot_product = sum(x * y for x, y in zip(X, Y))
        magnitude_X = math.sqrt(sum(x ** 2 for x in X))
        magnitude_Y = math.sqrt(sum(y ** 2 for y in Y))

        if magnitude_X == 0 or magnitude_Y == 0:
            return 0
        return float(dot_product) / (magnitude_X * magnitude_Y)

    @classmethod
    def _normalize_token_text(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        text = cls._UUID_RE.sub("<uuid>", text)
        text = cls._EMAIL_RE.sub("<email>", text)
        text = cls._DATE_RE.sub("<date>", text)
        text = cls._LONG_TOKEN_RE.sub("<token>", text)
        text = cls._NUMBER_RE.sub("<num>", text)
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _read_similarity_float(name: str, default: float) -> float:
        raw = os.environ.get(name, "")
        if not str(raw).strip():
            value = default
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = default
        return max(0.0, min(1.0, value))

    @classmethod
    def _normalize_url_identity(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            normalized = normalize_url_for_state(text).strip().lower()
        except Exception:
            normalized = text.lower()

        preserve_hash_route = os.environ.get(
            "WEBTEST_SIM_PRESERVE_HASH_ROUTE",
            "1",
        ).strip().lower() not in ("0", "false", "no", "off")
        if not preserve_hash_route:
            return normalized

        try:
            fragment = urlsplit(text).fragment.strip().lower()
        except Exception:
            fragment = ""
        if not fragment:
            return normalized

        fragment = fragment[1:] if fragment.startswith("!") else fragment
        fragment = re.sub(r"\s+", "", fragment).rstrip("/")
        if not fragment:
            return normalized

        parsed = urlsplit(normalized)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, fragment))

    @classmethod
    def _normalize_route_segment(cls, segment: str) -> str:
        token = str(segment or "").strip().lower()
        if not token:
            return token
        if (
            token.isdigit()
            or cls._ROUTE_UUID_RE.match(token)
            or cls._ROUTE_HEX_RE.match(token)
            or cls._ROUTE_MIXED_ID_RE.match(token)
        ):
            return "{id}"
        token = cls._UUID_RE.sub("{id}", token)
        token = cls._LONG_TOKEN_RE.sub("{id}", token)
        return cls._NUMBER_RE.sub("{id}", token)

    @classmethod
    def _normalize_route_path(cls, path: str) -> str:
        raw_path = str(path or "/").strip().lower() or "/"
        raw_path = re.sub(r"/+", "/", raw_path)
        prefix = "/" if raw_path.startswith("/") else ""
        suffix = "/" if raw_path.endswith("/") and raw_path != "/" else ""
        parts = [
            cls._normalize_route_segment(part)
            for part in raw_path.strip("/").split("/")
            if part
        ]
        if not parts:
            return "/"
        return prefix + "/".join(parts) + suffix

    @classmethod
    def _normalize_route_query(cls, query: str) -> str:
        kept = []
        for key, value in parse_qsl(str(query or ""), keep_blank_values=True):
            key_l = key.strip().lower()
            if key_l in {"view", "tab"}:
                kept.append((key_l, cls._normalize_route_segment(value)))
        return urlencode(sorted(kept), doseq=True)

    @classmethod
    def _normalize_route_identity(cls, value: Any) -> str:
        """
        CGFS-lite route abstraction R(s): normalized path plus SPA hash route.

        Query parameters are dropped except stable view/tab selectors. Dynamic
        route IDs are masked so entity pages share a route pattern.
        """
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            parsed = urlsplit(text)
        except Exception:
            parsed = urlsplit(text.split("?", 1)[0])

        path = cls._normalize_route_path(parsed.path or "/")
        query = cls._normalize_route_query(parsed.query)

        fragment = (parsed.fragment or "").strip()
        if fragment.startswith("!"):
            fragment = fragment[1:]
        fragment_route = ""
        if fragment:
            # SPA fragments often look like /#/route?tab=x. Treat that route as
            # first-class context rather than as browser-only noise.
            frag_parsed = urlsplit(fragment)
            frag_path = cls._normalize_route_path(frag_parsed.path or fragment)
            frag_query = cls._normalize_route_query(frag_parsed.query)
            fragment_route = f"#{frag_path}"
            if frag_query:
                fragment_route += f"?{frag_query}"

        route = path
        if query:
            route += f"?{query}"
        if fragment_route:
            route += fragment_route
        return route

    @classmethod
    def _normalize_action_info(cls, action_type: str, value: Any) -> str:
        text = str(value or "").strip()
        preserve_url_tokens = os.environ.get(
            "WEBTEST_SIM_PRESERVE_URL_TOKENS",
            "1",
        ).strip().lower() not in ("0", "false", "no", "off")
        if preserve_url_tokens and cls._URL_LIKE_RE.search(text):
            return cls._normalize_url_identity(text)
        return cls._normalize_token_text(value)

    @classmethod
    def _locator_tail(cls, action: WebAction) -> str:
        locator = getattr(action, "locator", None)
        locator_name = getattr(locator, "value", str(locator or ""))
        location = cls._normalize_token_text(getattr(action, "location", ""))
        parts = [part for part in re.split(r"[/>\s]+", location) if part]
        tail = "/".join(parts[-3:])
        return f"{locator_name}:{tail}" if tail else locator_name

    @classmethod
    def _bucket_for_action_type(cls, action_type: str) -> str:
        return action_type if action_type in cls._BUCKET_WEIGHTS else "default"

    @classmethod
    def _similarity_mode(cls) -> str:
        return os.environ.get("WEBTEST_STATE_SIMILARITY_MODE", "legacy").strip().lower()

    @classmethod
    def _canonical_action_signature(cls, action: WebAction, action_type: str, addition_info: Any = "") -> str:
        return "|".join(
            (
                f"type={cls._normalize_token_text(action_type)}",
                f"info={cls._normalize_action_info(action_type, addition_info)}",
                f"text={cls._normalize_token_text(getattr(action, 'text', ''))}",
                f"loc={cls._locator_tail(action)}",
            )
        )

    @classmethod
    def _cgfs_action_kind(cls, action: WebAction) -> str:
        if isinstance(action, ClickAction):
            action_type = cls._normalize_token_text(getattr(action, "action_type", "click") or "click")
            if action_type == "redirect":
                return "link"
            if action_type == "submit":
                return "submit"
            return "click"
        if isinstance(action, RandomInputAction):
            label = cls._normalize_token_text(getattr(action, "text", "")) or "text"
            if "date" in label:
                return "input:date"
            if "number" in label or "count" in label or "amount" in label:
                return "input:number"
            return "input:text"
        if isinstance(action, RandomSelectAction):
            return "select"
        return cls._normalize_token_text(type(action).__name__) or "action"

    @classmethod
    def _cgfs_action_label(cls, action: WebAction) -> str:
        label = getattr(action, "text", "") or ""
        if isinstance(action, ClickAction) and not str(label).strip():
            label = getattr(action, "addition_info", "") or ""
        label = cls._normalize_token_text(label)
        if len(label) > 64:
            label = label[:64]
        return label

    @classmethod
    def _cgfs_affordance_signature(cls, action: WebAction) -> str:
        return "|".join(
            (
                f"kind={cls._cgfs_action_kind(action)}",
                f"name={cls._cgfs_action_label(action)}",
                f"loc={cls._locator_tail(action)}",
            )
        )

    def _cgfs_affordance_set(self) -> set:
        return {self._cgfs_affordance_signature(action) for action in self.get_action_list()}

    @classmethod
    def _functional_words(cls, value: Any) -> set:
        text = cls._normalize_token_text(value)
        return {part for part in re.split(r"[^a-z0-9{}<>]+", text) if part}

    @classmethod
    def _functional_action_route_hint(cls, action: WebAction) -> str:
        raw = getattr(action, "addition_info", "") if isinstance(action, ClickAction) else ""
        if not raw:
            return ""
        raw_text = str(raw).strip()
        if not cls._URL_LIKE_RE.search(raw_text) and not raw_text.startswith("#"):
            return ""
        return cls._normalize_route_identity(raw_text)

    @classmethod
    def _is_click_form_control(cls, action: WebAction) -> bool:
        if not isinstance(action, ClickAction):
            return False
        action_type = str(getattr(action, "action_type", "") or "").strip().lower()
        if action_type in {"redirect", "submit"}:
            return False
        info = cls._normalize_token_text(getattr(action, "addition_info", ""))
        loc = cls._normalize_token_text(getattr(action, "location", ""))
        text = cls._normalize_token_text(getattr(action, "text", ""))
        blob = f" {info} {loc} {text} "
        return any(
            f" {token} " in blob
            for token in ("input", "select", "textarea", "checkbox", "radio", "option", "label")
        )

    @classmethod
    def _functional_action_signature(cls, action: WebAction, current_route: str) -> Union[str, None]:
        kind = cls._cgfs_action_kind(action)
        label = cls._cgfs_action_label(action)
        loc = cls._locator_tail(action)
        route_hint = cls._functional_action_route_hint(action)
        words = set(cls._functional_words(label))
        words.update(cls._functional_words(route_hint))
        words.update(cls._functional_words(loc))

        if isinstance(action, (RandomInputAction, RandomSelectAction)):
            family = "form_field"
        elif cls._is_click_form_control(action):
            family = "form_field"
        elif kind == "submit":
            family = "form_submit"
        elif words & cls._FUNCTIONAL_STRUCTURE_KEYWORDS:
            family = "structure"
        elif words & cls._FUNCTIONAL_MAIN_KEYWORDS:
            family = "main_action"
        elif kind == "link" and route_hint and route_hint != current_route and label not in cls._FUNCTIONAL_BOILERPLATE_LABELS:
            family = "workflow_link"
        elif kind in {"link", "click"} and label in cls._FUNCTIONAL_BOILERPLATE_LABELS:
            return None
        else:
            family = "main_action" if kind not in {"link", "click"} else "secondary_action"

        name = label or route_hint or loc
        if len(name) > 64:
            name = name[:64]
        return "|".join(
            (
                f"family={family}",
                f"kind={kind}",
                f"name={name}",
                f"route={route_hint}",
                f"loc={loc}",
            )
        )

    def _cgfs_functional_set(self) -> set:
        current_route = self._normalize_route_identity(getattr(self, "raw_url", self.url))
        elements = {
            signature
            for action in self.get_action_list()
            for signature in [self._functional_action_signature(action, current_route)]
            if signature
        }
        if elements:
            return elements

        fallback_actions = self._cgfs_affordance_set()
        if fallback_actions:
            return {f"fallback|route={current_route}|action_count={len(fallback_actions)}"}
        return set()

    @staticmethod
    def _jaccard_similarity(lhs: set, rhs: set) -> float:
        union = lhs | rhs
        if not union:
            return 0.0
        return float(len(lhs & rhs) / len(union))

    def _cgfs_route_gated_similarity(self, other: "ActionSetWithExecutionTimesState", lhs_items: set, rhs_items: set) -> float:
        content_similarity = self._jaccard_similarity(lhs_items, rhs_items)
        lhs_url = getattr(self, "raw_url", self.url)
        rhs_url = getattr(other, "raw_url", other.url)
        lhs_route = self._normalize_route_identity(lhs_url)
        rhs_route = self._normalize_route_identity(rhs_url)
        if lhs_route == rhs_route:
            return float(content_similarity)

        discount = self._read_similarity_float("WEBTEST_CGFS_CROSS_ROUTE_DISCOUNT", 0.25)
        return float(discount * content_similarity)

    def convert_action_to_legacy_vector(self, actions: List[WebAction]) -> Dict[str, Counter]:
        action_counts: Dict[str, Counter] = defaultdict(Counter)
        for action in actions:
            if isinstance(action, ClickAction):
                action_type = getattr(action, "action_type", "default") or "default"
                addition_info = getattr(action, "addition_info", "")
                bucket = self._bucket_for_action_type(action_type)
                action_counts[bucket][str(addition_info)] += 1
            elif isinstance(action, RandomInputAction):
                action_counts["default"]["random_input"] += 1
            elif isinstance(action, RandomSelectAction):
                action_counts["default"]["random_select"] += 1
        return dict(action_counts)

    def convert_action_to_vector(self, actions: List[WebAction]) -> Dict[str, Counter]:
        """
        Build sparse counters by action type.

        This keeps similarity symmetric and independent from call order.
        """
        action_counts: Dict[str, Counter] = defaultdict(Counter)
        for action in actions:
            if isinstance(action, ClickAction):
                action_type = getattr(action, "action_type", "default") or "default"
                addition_info = getattr(action, "addition_info", "")
                bucket = self._bucket_for_action_type(action_type)
                token = self._canonical_action_signature(action, action_type, addition_info)
                action_counts[bucket][token] += 1
            elif isinstance(action, RandomInputAction):
                token = self._canonical_action_signature(action, "random_input")
                action_counts["default"][token] += 1
            elif isinstance(action, RandomSelectAction):
                token = self._canonical_action_signature(action, "random_select")
                action_counts["default"][token] += 1
        return dict(action_counts)

    @staticmethod
    def _cosine_from_counters(lhs: Counter, rhs: Counter) -> float:
        if not lhs or not rhs:
            return 0.0
        keys = set(lhs.keys()) | set(rhs.keys())
        dot_product = sum(lhs[k] * rhs[k] for k in keys)
        magnitude_lhs = math.sqrt(sum(lhs[k] ** 2 for k in keys))
        magnitude_rhs = math.sqrt(sum(rhs[k] ** 2 for k in keys))
        if magnitude_lhs == 0 or magnitude_rhs == 0:
            return 0.0
        return float(dot_product) / (magnitude_lhs * magnitude_rhs)

    @classmethod
    def _url_similarity(cls, lhs_url: str, rhs_url: str) -> float:
        lhs = cls._normalize_url_identity(lhs_url)
        rhs = cls._normalize_url_identity(rhs_url)
        if not lhs or not rhs:
            return 0.0
        return 1.0 if lhs == rhs else 0.0

    @classmethod
    def _combine_url_action_similarity(cls, url_similarity: float, action_similarity: float) -> float:
        # Within the same route/page identity, action affordances define the
        # state novelty. URL only disambiguates otherwise similar affordances
        # across different routes/entities.
        if url_similarity >= 1.0:
            return max(0.0, min(1.0, action_similarity))
        url_weight = cls._read_similarity_float("WEBTEST_STATE_URL_SIM_WEIGHT", 0.35)
        combined = (url_weight * url_similarity) + ((1.0 - url_weight) * action_similarity)
        if url_similarity < 1.0:
            cap = cls._read_similarity_float("WEBTEST_CROSS_URL_SIM_CAP", 1.0 - url_weight)
            combined = min(combined, cap)
        return max(0.0, min(1.0, combined))

    def _action_set_similarity(self, other: "ActionSetWithExecutionTimesState") -> float:
        vector_self = self.convert_action_to_vector(self.get_action_list())
        vector_other = other.convert_action_to_vector(other.get_action_list())
        weighted_sum = 0.0
        active_weight_sum = 0.0

        for bucket, weight in self._BUCKET_WEIGHTS.items():
            lhs = vector_self.get(bucket, Counter())
            rhs = vector_other.get(bucket, Counter())
            if lhs or rhs:
                weighted_sum += weight * self._cosine_from_counters(lhs, rhs)
                active_weight_sum += weight

        return float(weighted_sum / active_weight_sum) if active_weight_sum > 0 else 1.0

    def _legacy_action_set_similarity(self, other: "ActionSetWithExecutionTimesState") -> float:
        vector_self = self.convert_action_to_legacy_vector(self.get_action_list())
        vector_other = other.convert_action_to_legacy_vector(other.get_action_list())
        weighted_sum = 0.0
        active_weight_sum = 0.0

        for bucket, weight in self._BUCKET_WEIGHTS.items():
            lhs = vector_self.get(bucket, Counter())
            rhs = vector_other.get(bucket, Counter())
            if lhs and rhs:
                weighted_sum += weight * self._cosine_from_counters(lhs, rhs)
                active_weight_sum += weight

        return float(weighted_sum / active_weight_sum) if active_weight_sum > 0 else 0.0

    def _cgfs_lite_similarity(self, other: "ActionSetWithExecutionTimesState") -> float:
        lhs_actions = self._cgfs_affordance_set()
        rhs_actions = other._cgfs_affordance_set()
        return self._cgfs_route_gated_similarity(other, lhs_actions, rhs_actions)

    def _cgfs_functional_similarity(self, other: "ActionSetWithExecutionTimesState") -> float:
        lhs_actions = self._cgfs_functional_set()
        rhs_actions = other._cgfs_functional_set()
        return self._cgfs_route_gated_similarity(other, lhs_actions, rhs_actions)

    def similarity(self, other: WebState) -> float:
        if not isinstance(other, ActionSetWithExecutionTimesState):
            return 0
        if not self.sim_dic.__contains__(other):
            mode = self._similarity_mode()
            if mode in ("cgfs_functional", "cgfs-functional", "cgfs_func"):
                overall_similarity = self._cgfs_functional_similarity(other)
            elif mode in ("cgfs_lite", "cgfs-lite", "cgfs"):
                overall_similarity = self._cgfs_lite_similarity(other)
            elif mode in ("url_action", "url-aware", "final"):
                action_similarity = self._action_set_similarity(other)
                lhs_url = getattr(self, "raw_url", self.url)
                rhs_url = getattr(other, "raw_url", other.url)
                url_similarity = self._url_similarity(lhs_url, rhs_url)
                overall_similarity = self._combine_url_action_similarity(
                    url_similarity,
                    action_similarity,
                )
            elif mode in ("action_affordance", "canonical_action"):
                overall_similarity = self._action_set_similarity(other)
            else:
                overall_similarity = self._legacy_action_set_similarity(other)
            self.sim_dic[other] = overall_similarity
            other.sim_dic[self] = overall_similarity
        return self.sim_dic[other]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ActionSetWithExecutionTimesState):
            return (self.action_dict.keys() == other.action_dict.keys()) and (self.url == other.url)
        return False

    def __hash__(self) -> int:
        hash_value = 0
        for action in self.action_dict.keys():
            hash_value += hash(action)
        hash_value += hash(self.url)
        return hash_value

    def __lt__(self, other: object) -> bool:
        if isinstance(other, ActionSetWithExecutionTimesState):
            return hash(self) < hash(other)
        else:
            return type(self).__name__ < type(other).__name__

    def __str__(self) -> str:
        return f"ActionSetWithExecutionTimesState(action_number={len(self.action_dict)}, action_execution_time_histogram={self.action_execution_time_histogram}, url={self.url})"
