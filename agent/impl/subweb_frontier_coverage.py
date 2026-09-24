"""Frontier coverage objectives for WebCover.

This module implements Web analogues of the SubRL weighted coverage form:
F(tau) = sum_{u in union D(s)} w(u).

The helper is intentionally independent from policy code. It only reads the
reached Web state and maintains coverage memory for the current run.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from action.impl.click_action import ClickAction
from action.impl.random_input_action import RandomInputAction
from action.impl.random_select_action import RandomSelectAction
from action.impl.restart_action import RestartAction
from action.web_action import WebAction
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.web_state import WebState


_INTENT_WORDS = (
    "save",
    "create",
    "add",
    "find",
    "search",
    "submit",
    "update",
    "delete",
    "remove",
    "edit",
    "share",
    "send",
    "invite",
    "confirm",
    "next",
)


def _read_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _read_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def graph_node_mode_from_env() -> str:
    return os.environ.get("WEBTEST_GRAPH_NODE_MODE", "agent_state").strip().lower() or "agent_state"


def graph_edge_label_mode_from_env() -> str:
    return (
        os.environ.get("WEBTEST_GRAPH_EDGE_LABEL_MODE", "observer_action")
        .strip()
        .lower()
        .replace("-", "_")
        or "observer_action"
    )


def _canonical_intent(text: str) -> str:
    words = set(re.split(r"[^a-z0-9{}<>]+", str(text or "").lower()))
    for word in _INTENT_WORDS:
        if word in words:
            return word
    return "other"


def _parse_signature(signature: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for part in str(signature or "").split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed[key] = value
    return parsed


def _route_for_state(web_state: ActionSetWithExecutionTimesState) -> str:
    raw_url = getattr(web_state, "raw_url", getattr(web_state, "url", ""))
    return ActionSetWithExecutionTimesState._normalize_route_identity(raw_url)


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "no", "off"}


def _count_bucket(value: int) -> str:
    value = max(0, int(value))
    if value <= 2:
        return str(value)
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    if value <= 20:
        return "11-20"
    return "20+"


def _route_segments(route: str) -> List[str]:
    route = str(route or "").split("?", 1)[0].split("#", 1)[0].strip("/")
    return [part for part in route.split("/") if part]


def _route_relation(source_route: str, target_route: str) -> str:
    if not target_route:
        return "none"
    if source_route == target_route:
        return "same"
    source_parts = _route_segments(source_route)
    target_parts = _route_segments(target_route)
    if len(target_parts) > len(source_parts) and target_parts[: len(source_parts)] == source_parts:
        return "child"
    if len(target_parts) < len(source_parts) and source_parts[: len(target_parts)] == target_parts:
        return "parent"
    if (
        len(target_parts) == len(source_parts)
        and target_parts
        and source_parts
        and target_parts[:-1] == source_parts[:-1]
    ):
        return "sibling"
    return "other"


def _functional_family(action: WebAction, route: str) -> str:
    signature = ActionSetWithExecutionTimesState._functional_action_signature(action, route)
    if not signature:
        return "boilerplate"
    parsed = _parse_signature(signature)
    return parsed.get("family", "") or "unknown"


def _abstract_action_signature(action: WebAction, current_route: str) -> str:
    """Context-agnostic action identity used for graph/action coverage.

    The existing frontier signature keeps text/location detail. This signature
    intentionally collapses repeated row links and repeated navigation by using
    target route relation and generic intent rather than raw text/DOM position.
    """
    kind = ActionSetWithExecutionTimesState._cgfs_action_kind(action)
    label = ActionSetWithExecutionTimesState._cgfs_action_label(action)
    route_hint = ActionSetWithExecutionTimesState._functional_action_route_hint(action)
    family = _functional_family(action, current_route)
    relation = _route_relation(current_route, route_hint)
    intent = _canonical_intent(" ".join([label, route_hint]))
    field = ""
    if isinstance(action, (RandomInputAction, RandomSelectAction)):
        field = label or ActionSetWithExecutionTimesState._locator_tail(action)
        if len(field) > 48:
            field = field[:48]
    elif isinstance(action, ClickAction) and family == "form_field":
        # Clickable form controls (labels, checkboxes, custom selects) are
        # separate executable affordances. Use a stable DOM-derived key so the
        # graph objective does not collapse all same-page controls into one edge.
        raw_location = str(getattr(action, "location", "") or "")
        locator_tail = ActionSetWithExecutionTimesState._locator_tail(action)
        base = label or ActionSetWithExecutionTimesState._cgfs_action_label(action) or locator_tail
        digest = _stable_short_hash(raw_location or locator_tail or base, length=12)
        field = f"{base[:32]}#{digest}" if base else digest
    return "|".join(
        (
            f"family={family}",
            f"kind={kind}",
            f"intent={intent}",
            f"target={route_hint}",
            f"rel={relation}",
            f"field={field}",
        )
    )


def abstract_action_signature(action: WebAction, current_route: str) -> str:
    """Public wrapper for the action identity used by graph coverage units."""
    return _abstract_action_signature(action, current_route)


def _graph_edge_action_signature(action: WebAction, current_route: str) -> str:
    """Action label for strict graph edges.

    The main graph objective treats an edge as an executed concrete action slot
    from the current node to the observed target node. The abstract label is
    still available as an explicit variant for more aggressive generalization.
    """
    abstract = _abstract_action_signature(action, current_route)
    mode = graph_edge_label_mode_from_env()
    if mode in {"abstract", "abstract_action", "functional"}:
        return abstract
    exact = _observer_like_action_signature(action)
    digest = _stable_short_hash(exact, length=20)
    return f"exact={digest}|{abstract}"


def _is_template_navigation_or_error_action(action: WebAction, current_route: str) -> bool:
    """Return True for repeated site chrome links that should not get edge credit.

    Arrival at the target page can still create coverage reward. This only
    suppresses extra executed-action/edge units for global navigation, logo,
    breadcrumb-like home links, and known error links.
    """
    if not isinstance(action, ClickAction):
        return False

    kind = ActionSetWithExecutionTimesState._cgfs_action_kind(action)
    action_type = str(getattr(action, "action_type", "") or "").lower()
    if kind != "link" and action_type != "redirect":
        return False

    location = str(getattr(action, "location", "") or "").lower()
    text = str(getattr(action, "text", "") or "").strip().lower()
    route_hint = ActionSetWithExecutionTimesState._functional_action_route_hint(action)

    if "error" in text or route_hint == "/oups":
        return True
    if any(token in location for token in ("/nav", "/header", "/footer", "navbar", "breadcrumb")):
        return True
    if route_hint == current_route and _canonical_intent(text) == "other":
        return True
    if route_hint == "/" and text in {"", "home"}:
        return True
    return False


def is_template_navigation_or_error_action(action: WebAction, current_route: str) -> bool:
    """Public predicate for actions excluded from functional transition units."""
    return _is_template_navigation_or_error_action(action, current_route)


def template_graph_edge_scope(action: WebAction, current_route: str, target_route: str = "") -> str:
    """Return a run-global template-edge scope for site chrome actions.

    Template navigation is still credited only after execution. By default,
    template links are ordinary source-node-specific graph edges, matching the
    clean online graph objective E={(s_t,a_t,s_{t+1})}. Run-global template
    scoping is kept only as an explicit variant.
    """
    scope_mode = os.environ.get("WEBTEST_GRAPH_TEMPLATE_EDGE_SCOPE", "source").strip().lower()
    if scope_mode not in {"global", "template", "run_global"}:
        return ""
    if not _is_template_navigation_or_error_action(action, current_route):
        return ""
    route_hint = ActionSetWithExecutionTimesState._functional_action_route_hint(action)
    text = str(getattr(action, "text", "") or "").strip().lower()
    target = target_route or route_hint
    if target == "/oups" or route_hint == "/oups" or "error" in text:
        return "template_error"
    if target == "/" or route_hint == "/":
        return "template_reset"
    return "template_nav"


def _action_signature(action: WebAction) -> str:
    if isinstance(action, ClickAction):
        action_type = getattr(action, "action_type", "default") or "default"
        addition_info = getattr(action, "addition_info", "")
        return ActionSetWithExecutionTimesState._canonical_action_signature(
            action,
            action_type,
            addition_info,
        )
    if isinstance(action, RandomInputAction):
        return ActionSetWithExecutionTimesState._canonical_action_signature(action, "random_input")
    if isinstance(action, RandomSelectAction):
        return ActionSetWithExecutionTimesState._canonical_action_signature(action, "random_select")
    return ActionSetWithExecutionTimesState._canonical_action_signature(action, type(action).__name__)


def _pageinfo_unit(route: str, action_signatures: Iterable[str]) -> str:
    classes = sorted({unit_class(f"frontier:{route}|{sig}") for sig in action_signatures})
    class_text = ",".join(classes[:12]) if classes else "none"
    family_counts = Counter(cls.split(":", 1)[0] for cls in classes)
    return "|".join(
        (
            f"pageinfo:{route}",
            f"classes={class_text}",
            f"n_classes={_count_bucket(len(classes))}",
            f"has_field={1 if family_counts.get('field', 0) else 0}",
            f"has_submit={1 if family_counts.get('submit', 0) else 0}",
            f"has_main={1 if family_counts.get('main-action', 0) else 0}",
            f"has_structure={1 if family_counts.get('structure', 0) or family_counts.get('dialog', 0) or family_counts.get('tab', 0) else 0}",
        )
    )


def _stable_short_hash(text: str, length: int = 16) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:length]


def _observer_like_action_signature(action: WebAction) -> str:
    """Mirror the Observer state-id action signature for reward/eval alignment."""
    action_class = type(action).__name__
    locator = getattr(getattr(action, "locator", None), "value", "")
    location = getattr(action, "location", "")
    text = getattr(action, "text", "")
    if action_class == "ClickAction":
        action_type = getattr(action, "action_type", "")
        addition_info = getattr(action, "addition_info", "")
        return (
            f"{action_class}|locator={locator}|location={location}|text={text}"
            f"|type={action_type}|info={addition_info}"
        )
    return f"{action_class}|locator={locator}|location={location}|text={text}"


def _observer_like_state_id(web_state: ActionSetWithExecutionTimesState) -> str:
    """State abstraction equivalent to observation.Observer._get_state_id for valid states."""
    action_text = ",".join(sorted(_observer_like_action_signature(a) for a in web_state.get_action_list()))
    return f"url={web_state.url}|actions={action_text}"


def _graph_form_state_id(web_state: ActionSetWithExecutionTimesState) -> str:
    if not _read_bool_env("WEBTEST_GRAPH_FORM_STATE", False):
        return ""
    return str(getattr(web_state, "graph_form_state_id", "") or "").strip()


def extract_graph_node_id(web_state: WebState) -> str:
    """Return the strict graph node identity for a Web state.

    The default node abstraction is ``agent_state``: SubWeb agents annotate
    each observed state with the same equality boundary used by WebTest's
    ``state_dict`` / ``newest.json`` unique-state list. If this extractor is
    called outside a running agent, it falls back to the observer-style state
    identity so tests and legacy diagnostics still get a stable node id.
    Coarser route-actionset nodes and form-state-sensitive nodes remain
    available as explicit variants.
    This is intentionally one node unit per observed state abstraction, not one
    unit per visible action. Concrete action coverage is still credited only
    after execution through labeled graph edges.
    """
    if not isinstance(web_state, ActionSetWithExecutionTimesState):
        return ""

    route = _route_for_state(web_state)
    mode = graph_node_mode_from_env()
    if mode in {"route", "route_only", "context"}:
        return route
    if mode in {"agent_state", "env_state", "webtest_state", "state_list"}:
        # Main SubWeb node identity: match WebTest's result snapshot boundary.
        # Agents annotate each observed state with the index obtained by
        # ``web_state in state_list`` / ``state_list.index(web_state)``, the same
        # equality relation used by DataCollector for ``newest.json``.
        state_index = getattr(web_state, "graph_state_index", None)
        if state_index is not None:
            try:
                return f"{route}|state_idx={int(state_index)}"
            except (TypeError, ValueError):
                return f"{route}|state_idx={state_index}"
        # Unit tests and legacy callers may call the extractor outside a running
        # SubWeb agent. Fall back to the observer-style identity rather than
        # returning an empty node.
        mode = "observer_state"
    if mode in {"observer", "observer_state", "state", "state_id", "raw_actionset"}:
        state_id = _observer_like_state_id(web_state)
        form_state = _graph_form_state_id(web_state)
        if form_state:
            state_id = f"{state_id}|form_state={form_state}"
        digest = _stable_short_hash(state_id, length=24)
        action_count = len(web_state.get_action_list())
        form_suffix = f"|form={_stable_short_hash(form_state, length=8)}" if form_state else ""
        return f"{route}|state={digest}|n={action_count}{form_suffix}"

    signatures = [
        _abstract_action_signature(action, route)
        for action in web_state.get_action_list()
    ]
    counts = Counter(signatures)
    fingerprint_text = "\n".join(f"{sig}\t{count}" for sig, count in sorted(counts.items()))
    form_state = _graph_form_state_id(web_state)
    if form_state:
        fingerprint_text = f"{fingerprint_text}\nform_state\t{form_state}"
    digest = _stable_short_hash(fingerprint_text or "<empty>")
    form_suffix = f"|form={_stable_short_hash(form_state, length=8)}" if form_state else ""
    return f"{route}|actions={digest}|n={len(signatures)}{form_suffix}"


def _extract_state_units(web_state: WebState, include_exposed_action_units: bool) -> Set[str]:
    """Extract route/page units, optionally including actions visible on arrival."""
    if not isinstance(web_state, ActionSetWithExecutionTimesState):
        return set()

    route = _route_for_state(web_state)
    units = {f"context:{route}"}

    functional_signatures: List[str] = []
    for action in web_state.get_action_list():
        signature = ActionSetWithExecutionTimesState._functional_action_signature(action, route)
        if signature:
            functional_signatures.append(signature)
            if include_exposed_action_units:
                units.add(f"frontier:{route}|{signature}")
        if include_exposed_action_units and _read_bool_env("WEBTEST_FRONTIER_GRAPH_UNITS", True):
            units.add(f"action:{route}|{_abstract_action_signature(action, route)}")

    if _read_bool_env("WEBTEST_FRONTIER_GRAPH_UNITS", True):
        units.add(_pageinfo_unit(route, functional_signatures))

    return units


def extract_arrival_units(web_state: WebState) -> Set[str]:
    """Extract units covered by observing a state before executing any action.

    Web testing should not treat every available action on a page as covered
    merely because the page was reached. Action/edge coverage is added by
    extract_transition_units after an action is actually executed.
    """
    return _extract_state_units(web_state, include_exposed_action_units=False)


def extract_graph_node_units(web_state: WebState) -> Set[str]:
    """Extract graph nodes covered by observing a Web state.

    This is the strict online graph objective node set V. The default node
    matches the Observer state identity, so exact action-signature states on
    the same route count as different graph nodes. Visible actions are not
    separate reward units; executing an action is credited by graph edges.
    """
    node_id = extract_graph_node_id(web_state)
    if not node_id:
        return set()
    return {f"node:{node_id}"}


def extract_frontier_units(web_state: WebState) -> Set[str]:
    """Extract legacy D(s): route context plus visible functional frontier units."""
    return _extract_state_units(web_state, include_exposed_action_units=True)


def extract_transition_units(
    source_url: Optional[str],
    chosen_action: Optional[WebAction],
    target_state: WebState,
) -> Set[str]:
    """Extract executed state-action and topology-edge units."""
    if not _read_bool_env("WEBTEST_FRONTIER_GRAPH_UNITS", True):
        return set()
    if (
        chosen_action is None
        or isinstance(chosen_action, RestartAction)
        or not isinstance(target_state, ActionSetWithExecutionTimesState)
    ):
        return set()

    source_route = ActionSetWithExecutionTimesState._normalize_route_identity(source_url or "")
    target_route = _route_for_state(target_state)
    if _read_bool_env("WEBTEST_FILTER_TEMPLATE_TRANSITION_UNITS", False):
        if _is_template_navigation_or_error_action(chosen_action, source_route):
            return set()
    action_sig = _graph_edge_action_signature(chosen_action, source_route)
    return {
        f"exec_action:{source_route}|{action_sig}",
        f"edge:{source_route}|{action_sig}|target_context={target_route}",
    }


def graph_edge_prefix(
    source_url: Optional[str],
    chosen_action: Optional[WebAction],
    source_node_id: Optional[str] = None,
) -> str:
    """Prefix for a candidate executed graph edge before its target is known.

    The strict reward can only create the full edge after executing the action
    and observing the target node. Optimizer-side residual signals use this
    source-node plus graph-edge action label prefix so their notion of
    "untried" stays tied to the same graph edge namespace as the hard
    F=|V|+|E| objective.
    """
    if chosen_action is None or isinstance(chosen_action, RestartAction):
        return ""
    source_route = ActionSetWithExecutionTimesState._normalize_route_identity(source_url or "")
    source_node = str(source_node_id or "").strip() or source_route
    action_sig = _graph_edge_action_signature(chosen_action, source_route)
    target_hint = ActionSetWithExecutionTimesState._functional_action_route_hint(chosen_action)
    target_route = ActionSetWithExecutionTimesState._normalize_route_identity(target_hint or "")
    template_scope = template_graph_edge_scope(chosen_action, source_route, target_route)
    edge_source = template_scope or source_node
    return f"edge:{edge_source}|{action_sig}|"


def extract_graph_edge_units(
    source_url: Optional[str],
    chosen_action: Optional[WebAction],
    target_state: WebState,
    source_node_id: Optional[str] = None,
) -> Set[str]:
    """Extract executed labeled graph edges covered by taking an action.

    The Web graph edge is labeled by the abstract action because many useful
    Web actions are self-loops at the route level, for example form input.
    """
    if not _read_bool_env("WEBTEST_FRONTIER_GRAPH_UNITS", True):
        return set()
    if (
        chosen_action is None
        or isinstance(chosen_action, RestartAction)
        or not isinstance(target_state, ActionSetWithExecutionTimesState)
    ):
        return set()

    source_route = ActionSetWithExecutionTimesState._normalize_route_identity(source_url or "")
    source_node = str(source_node_id or "").strip() or source_route
    target_node = extract_graph_node_id(target_state) or _route_for_state(target_state)
    target_route = _route_for_state(target_state)
    if _read_bool_env("WEBTEST_GRAPH_FILTER_TEMPLATE_EDGES", False):
        if _is_template_navigation_or_error_action(chosen_action, source_route):
            return set()
    prefix = graph_edge_prefix(source_url, chosen_action, source_node_id=source_node_id)
    if not prefix:
        return set()
    if target_node == source_node:
        outcome = "same_node"
    elif target_route == source_route:
        outcome = "same_route"
    else:
        outcome = "route_change"
    return {f"{prefix}target_node={target_node}|target_context={target_route}|outcome={outcome}"}


def unit_class(unit: str) -> str:
    """Coarse, site-independent class used by density weighting."""
    text = str(unit or "")
    if text.startswith("node:"):
        return "context"
    if text.startswith("context:"):
        return "context"
    if text.startswith("pageinfo:"):
        return "pageinfo"
    if text.startswith("edge:"):
        parsed = _parse_signature(text)
        return f"edge:{parsed.get('intent', 'other')}"
    if text.startswith("exec_action:") or text.startswith("action:"):
        parsed = _parse_signature(text)
        kind = parsed.get("kind", "")
        intent = parsed.get("intent", "")
        if kind.startswith("input"):
            return f"field:{kind}"
        if kind == "select":
            return "field:select"
        if kind == "submit":
            return f"submit:{intent or 'other'}"
        return f"action:{intent or kind or 'other'}"

    parsed = _parse_signature(text)
    family = parsed.get("family", "")
    kind = parsed.get("kind", "")
    name = parsed.get("name", "")

    if family == "form_field":
        return f"field:{kind or 'unknown'}"
    if family == "form_submit":
        return f"submit:{_canonical_intent(name)}"
    if family == "structure":
        if "dialog" in name:
            return "dialog"
        if "tab" in name:
            return "tab"
        if "table" in name or "row" in name:
            return "row-action:other"
        return "structure"
    if family in {"workflow_link", "main_action", "secondary_action"}:
        return f"main-action:{_canonical_intent(name)}"
    if family:
        return family
    return "unknown"


@dataclass
class FrontierStepStats:
    mode: str
    reward: float
    raw_gain: float
    norm_gain: float
    new_units: int
    total_units: int
    edge_gain: int = 0


@dataclass
class StrictFrontierReward:
    """Strict marginal coverage reward used by online actor-critic agents."""

    reward: float
    new_units: int
    total_units: int
    cumulative_F: float
    units: Set[str]
    new_unit_set: Set[str]
    node_gain: int = 0
    edge_gain: int = 0
    node_weight_gain: float = 0.0
    edge_weight_gain: float = 0.0
    total_nodes: int = 0
    total_edges: int = 0


class FrontierCoverageTracker:
    """Run-scoped coverage memory and weighting for frontier coverage modes."""

    VALID_MODES = {
        "frontier_uniform",
        "frontier_value_density_static",
        "frontier_value_density_dynamic",
        "frontier_uniform_edge_bonus",
    }

    def __init__(self, mode: str):
        normalized = str(mode or "").strip().lower().replace("-", "_")
        if normalized not in self.VALID_MODES:
            raise ValueError(f"Unsupported WEBTEST_SUBWEB_COVERAGE_MODE={mode!r}")
        self.mode = normalized
        self.covered_units: Set[str] = set()
        self.covered_nodes: Set[str] = set()
        self.covered_edges: Set[str] = set()
        self.cumulative_F = 0.0
        self.seen_edges: Set[Tuple[str, str, str]] = set()
        self.class_frequencies: Counter = Counter()
        self.class_values: Dict[str, float] = defaultdict(float)
        self.pending_density: List[Dict[str, Any]] = []
        self.step_index = 0
        self.stats: List[FrontierStepStats] = []
        self.new_units_by_class: Counter = Counter()
        self.lambda_ = _read_float_env("WEBTEST_FRONTIER_DENSITY_LAMBDA", 0.5)
        self.weight_min = _read_float_env("WEBTEST_FRONTIER_WEIGHT_MIN", 0.5)
        self.weight_max = _read_float_env("WEBTEST_FRONTIER_WEIGHT_MAX", 2.0)
        self.ema_alpha = _read_float_env("WEBTEST_FRONTIER_DENSITY_EMA_ALPHA", 0.1)
        self.payoff_horizon = max(1, _read_int_env("WEBTEST_FRONTIER_DENSITY_HORIZON", 3))
        self.edge_beta = _read_float_env("WEBTEST_FRONTIER_EDGE_BETA", 0.25)
        # Main SubWeb objective is the clean graph coverage analogue:
        # F(tau)=|covered graph nodes|+alpha*sum_e w(e).
        # The default weight is intentionally simple and site-agnostic:
        # w(e)=1 for every executed edge except return-to-home/root edges,
        # which are still recorded as covered but contribute w(e)=0.
        self.graph_edge_alpha = _read_float_env("WEBTEST_GRAPH_EDGE_ALPHA", 1.0)
        self.graph_node_weight = _read_float_env("WEBTEST_GRAPH_NODE_WEIGHT", 1.0)
        requested_edge_mode = (
            os.environ.get("WEBTEST_GRAPH_EDGE_WEIGHT_MODE", "home_zero")
            .strip()
            .lower()
            .replace("-", "_")
            or "home_zero"
        )
        if requested_edge_mode == "source_degree_capped":
            requested_edge_mode = "source_degree"
        if requested_edge_mode not in {"uniform", "home_zero", "source_degree"}:
            # Keep legacy/experimental values harmless unless explicitly using
            # an explicitly supported graph edge objective below.
            requested_edge_mode = "home_zero"
        self.graph_edge_weight_mode = requested_edge_mode
        self.graph_source_out_degrees: Dict[str, int] = {}

    @staticmethod
    def _edge_source_key(edge_unit: str) -> str:
        text = str(edge_unit or "")
        if not text.startswith("edge:"):
            return ""
        body = text[len("edge:") :]
        markers = ("|exact=", "|family=")
        marker_positions = [body.find(marker) for marker in markers if body.find(marker) >= 0]
        if marker_positions:
            return body[: min(marker_positions)]
        return body.split("|target_node=", 1)[0].split("|target_context=", 1)[0]

    def register_graph_source_degree(self, source_node_id: str, action_count: int) -> None:
        """Record the full detected out-degree for source-normalized edge weights."""
        source = str(source_node_id or "").strip()
        if not source:
            return
        degree = max(1, int(action_count or 0))
        previous = self.graph_source_out_degrees.get(source, 0)
        if degree > previous:
            self.graph_source_out_degrees[source] = degree

    def _refresh_graph_value(self) -> None:
        edge_weight = sum(self._graph_edge_weight(edge) for edge in self.covered_edges)
        self.cumulative_F = (
            self.graph_node_weight * float(len(self.covered_nodes))
            + self.graph_edge_alpha * float(edge_weight)
        )

    def _graph_edge_weight(self, edge_unit: str) -> float:
        """Fixed graph edge weight for the clean Web coverage objective."""
        text = str(edge_unit or "")
        if self.graph_edge_weight_mode == "uniform":
            return 1.0
        parsed = _parse_signature(text)
        target = parsed.get("target", "")
        target_context = parsed.get("target_context", "")
        is_explicit_home_edge = target == "/" and target_context in {"", "/"}
        if text.startswith("edge:template_reset|") or is_explicit_home_edge:
            return 0.0
        if self.graph_edge_weight_mode == "source_degree":
            source = self._edge_source_key(text)
            degree = max(1, int(self.graph_source_out_degrees.get(source, 1)))
            return 1.0 / float(degree)
        return 1.0

    def seed_initial_state(self, web_state: WebState) -> StrictFrontierReward:
        """Seed the graph objective with the initial observed node only."""
        node_units = extract_graph_node_units(web_state)
        new_nodes = node_units - self.covered_nodes
        self.covered_nodes.update(node_units)
        self.covered_units.update(node_units)
        for unit in node_units:
            self.class_frequencies[unit_class(unit)] += 1
        for unit in new_nodes:
            self.new_units_by_class[unit_class(unit)] += 1
        self._refresh_graph_value()
        return StrictFrontierReward(
            reward=0.0,
            new_units=len(new_nodes),
            total_units=len(node_units),
            cumulative_F=self.cumulative_F,
            units=set(node_units),
            new_unit_set=set(new_nodes),
            node_gain=len(new_nodes),
            edge_gain=0,
            node_weight_gain=self.graph_node_weight * float(len(new_nodes)),
            edge_weight_gain=0.0,
            total_nodes=len(self.covered_nodes),
            total_edges=len(self.covered_edges),
        )

    def _unit_weight(self, unit: str) -> float:
        if self.mode == "frontier_value_density_static":
            cls = unit_class(unit)
            return 1.0 / math.sqrt(float(self.class_frequencies[cls]) + 1.0)
        if self.mode == "frontier_value_density_dynamic":
            cls = unit_class(unit)
            value = 1.0 + self.lambda_ * self.class_values[cls]
            return max(self.weight_min, min(self.weight_max, value))
        return 1.0

    def _update_dynamic_payoffs(self, uniform_norm_gain: float) -> None:
        if self.mode != "frontier_value_density_dynamic":
            return
        remaining = []
        for pending in self.pending_density:
            pending["payoff"] += float(uniform_norm_gain)
            if pending["due_step"] <= self.step_index:
                for cls in pending["classes"]:
                    old = self.class_values[cls]
                    self.class_values[cls] = (1.0 - self.ema_alpha) * old + self.ema_alpha * pending["payoff"]
            else:
                remaining.append(pending)
        self.pending_density = remaining

    def _register_new_dynamic_units(self, new_units: Iterable[str]) -> None:
        if self.mode != "frontier_value_density_dynamic":
            return
        classes = sorted({unit_class(unit) for unit in new_units})
        if not classes:
            return
        self.pending_density.append({
            "classes": classes,
            "due_step": self.step_index + self.payoff_horizon,
            "payoff": 0.0,
        })

    def _edge_gain(
        self,
        source_url: Optional[str],
        action: Optional[WebAction],
        target_route: str,
    ) -> int:
        if self.mode != "frontier_uniform_edge_bonus" or action is None:
            return 0
        source_route = ActionSetWithExecutionTimesState._normalize_route_identity(source_url or "")
        edge = (source_route, _action_signature(action), target_route)
        if edge in self.seen_edges:
            return 0
        self.seen_edges.add(edge)
        return 1

    def compute_reward(
        self,
        web_state: WebState,
        chosen_action: Optional[WebAction] = None,
        source_url: Optional[str] = None,
    ) -> float:
        self.step_index += 1

        units = extract_frontier_units(web_state)
        new_units = units - self.covered_units
        weights = {unit: self._unit_weight(unit) for unit in units}
        raw_gain = sum(weights[unit] for unit in new_units)
        total_weight = sum(weights.values())
        norm_gain = raw_gain / max(1.0, total_weight)

        edge_gain = 0
        if isinstance(web_state, ActionSetWithExecutionTimesState):
            edge_gain = self._edge_gain(source_url, chosen_action, _route_for_state(web_state))

        reward = norm_gain
        if self.mode == "frontier_uniform_edge_bonus":
            reward += self.edge_beta * float(edge_gain)

        self._update_dynamic_payoffs(norm_gain)
        self._register_new_dynamic_units(new_units)
        self.covered_units.update(units)
        for unit in units:
            self.class_frequencies[unit_class(unit)] += 1
        for unit in new_units:
            self.new_units_by_class[unit_class(unit)] += 1

        self.stats.append(FrontierStepStats(
            mode=self.mode,
            reward=float(reward),
            raw_gain=float(raw_gain),
            norm_gain=float(norm_gain),
            new_units=len(new_units),
            total_units=len(units),
            edge_gain=edge_gain,
        ))
        return float(reward)

    def compute_strict_marginal_reward(
        self,
        web_state: WebState,
        chosen_action: Optional[WebAction] = None,
        source_url: Optional[str] = None,
        source_node_id: Optional[str] = None,
        source_action_count: Optional[int] = None,
    ) -> StrictFrontierReward:
        """Compute strict marginal Web graph coverage and update run coverage.

        The strict objective is F(tau)=|V(tau)|+alpha*|E(tau)|. Observing a
        state covers only its abstract graph node. Executing an action covers
        one labeled graph edge from the source node to the observed target node.
        This avoids counting every available action on a page as explored
        before it has actually been executed.
        """
        self.step_index += 1
        if source_action_count is not None:
            source_key = str(source_node_id or "").strip()
            if not source_key:
                source_key = ActionSetWithExecutionTimesState._normalize_route_identity(source_url or "")
            self.register_graph_source_degree(source_key, int(source_action_count))

        node_units = extract_graph_node_units(web_state)
        edge_units = extract_graph_edge_units(
            source_url,
            chosen_action,
            web_state,
            source_node_id=source_node_id,
        )
        new_nodes = node_units - self.covered_nodes
        new_edges = edge_units - self.covered_edges
        units = set(node_units) | set(edge_units)
        new_units = set(new_nodes) | set(new_edges)
        node_gain = len(new_nodes)
        edge_gain = len(new_edges)
        node_weight_gain = self.graph_node_weight * float(node_gain)
        edge_weight_gain = sum(self._graph_edge_weight(edge) for edge in new_edges)
        reward = node_weight_gain + self.graph_edge_alpha * float(edge_weight_gain)

        self.covered_nodes.update(node_units)
        self.covered_edges.update(edge_units)
        self.covered_units.update(units)
        self._refresh_graph_value()
        for unit in units:
            self.class_frequencies[unit_class(unit)] += 1
        for unit in new_units:
            self.new_units_by_class[unit_class(unit)] += 1

        total_units = len(units)
        norm_gain = reward / max(1.0, 1.0 + self.graph_edge_alpha)
        self.stats.append(FrontierStepStats(
            mode=f"{self.mode}_strict",
            reward=reward,
            raw_gain=reward,
            norm_gain=norm_gain,
            new_units=len(new_units),
            total_units=total_units,
            edge_gain=edge_gain,
        ))
        return StrictFrontierReward(
            reward=reward,
            new_units=len(new_units),
            total_units=total_units,
            cumulative_F=self.cumulative_F,
            units=set(units),
            new_unit_set=set(new_units),
            node_gain=node_gain,
            edge_gain=edge_gain,
            node_weight_gain=node_weight_gain,
            edge_weight_gain=edge_weight_gain,
            total_nodes=len(self.covered_nodes),
            total_edges=len(self.covered_edges),
        )

    def summary(self) -> Dict[str, Any]:
        if not self.stats:
            return {
                "mode": self.mode,
                "steps": 0,
                "covered_units": len(self.covered_units),
                "covered_nodes": len(self.covered_nodes),
                "covered_edges": len(self.covered_edges),
                "cumulative_F": self.cumulative_F,
                "graph_node_mode": graph_node_mode_from_env(),
                "graph_node_weight": self.graph_node_weight,
                "graph_edge_alpha": self.graph_edge_alpha,
                "graph_edge_label_mode": graph_edge_label_mode_from_env(),
                "graph_edge_weight_mode": self.graph_edge_weight_mode,
                "mean_reward": 0.0,
                "mean_raw_gain": 0.0,
                "mean_norm_gain": 0.0,
                "unique_edges": len(self.seen_edges),
            }
        return {
            "mode": self.mode,
            "steps": len(self.stats),
            "covered_units": len(self.covered_units),
            "covered_nodes": len(self.covered_nodes),
            "covered_edges": len(self.covered_edges),
            "cumulative_F": self.cumulative_F,
            "graph_node_mode": graph_node_mode_from_env(),
            "graph_node_weight": self.graph_node_weight,
            "graph_edge_alpha": self.graph_edge_alpha,
            "graph_edge_label_mode": graph_edge_label_mode_from_env(),
            "graph_edge_weight_mode": self.graph_edge_weight_mode,
            "mean_reward": sum(s.reward for s in self.stats) / len(self.stats),
            "mean_raw_gain": sum(s.raw_gain for s in self.stats) / len(self.stats),
            "mean_norm_gain": sum(s.norm_gain for s in self.stats) / len(self.stats),
            "unique_edges": len(self.seen_edges),
            "top_new_unit_classes": dict(self.new_units_by_class.most_common(8)),
        }


def make_frontier_tracker_from_env() -> Optional[FrontierCoverageTracker]:
    mode = os.environ.get("WEBTEST_SUBWEB_COVERAGE_MODE", "").strip().lower()
    if not mode or mode in {"0", "off", "none", "state_similarity", "similarity"}:
        return None
    return FrontierCoverageTracker(mode)
