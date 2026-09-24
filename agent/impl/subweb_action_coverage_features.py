"""Action-level coverage residual features for SubWeb agents.

These features expose the current graph-coverage state to the policy without
directly modifying logits. The reward objective remains strict node/edge
coverage; the actor learns from advantages which residual features are useful.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

import torch

from action.web_action import WebAction
from agent.impl.subweb_frontier_coverage import (
    graph_edge_prefix,
    is_template_navigation_or_error_action,
)
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.web_state import WebState


FEATURE_NAMES = (
    "edge_seen_from_current_node",
    "edge_unseen_from_current_node",
    "action_count_log",
    "zero_gain_count_log",
    "last_action_zero_gain",
    "target_route_seen",
    "template_edge_filtered",
    "context_untried_action_ratio",
    "source_out_seen_ratio",
    "source_visit_count_log",
    "source_gain_rate",
    "source_zero_gain_streak_log",
    "target_in_count_log",
    "target_gain_rate",
    "target_known_frontier_ratio",
    "target_known_frontier_count_log",
    "source_residual_count_log",
    "source_out_degree_log",
    "global_edge_seen_ratio",
    "global_edge_residual_count_log",
    "estimated_edge_marginal_norm",
    "estimated_target_node_marginal_norm",
    "estimated_graph_marginal_norm",
)


def action_coverage_feature_dim() -> int:
    return len(FEATURE_NAMES)


def _context_from_state(web_state: WebState) -> str:
    url = getattr(web_state, "raw_url", getattr(web_state, "url", "")) or ""
    return ActionSetWithExecutionTimesState._normalize_route_identity(str(url))


def _context_from_url(url: str) -> str:
    return ActionSetWithExecutionTimesState._normalize_route_identity(str(url or ""))


def _target_context_for(action: WebAction) -> str:
    try:
        route_hint = ActionSetWithExecutionTimesState._functional_action_route_hint(action)
    except Exception:
        route_hint = ""
    return _context_from_url(route_hint)


def _log_count(value: int) -> float:
    return min(1.0, math.log1p(float(max(0, value))) / math.log(32.0))


def _raw_text(value: object, limit: int = 256) -> str:
    text = str(value or "").strip().lower()
    return text[:limit]


def _stable_short_hash(text: str, length: int = 16) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:length]


def _exact_action_signature(action: WebAction) -> str:
    """Run-local executable identity for residual coverage features.

    The graph reward intentionally uses an abstract action signature so that
    policies can generalize across repeated Web structures. For residual
    action-memory features we also need the concrete candidate identity:
    otherwise clicking one row link can make all sibling row links look tried.
    """
    parts = [
        type(action).__name__,
        _raw_text(getattr(action, "action_type", "")),
        _raw_text(getattr(action, "addition_info", "")),
        _raw_text(getattr(action, "location", "")),
        _raw_text(getattr(action, "text", "")),
    ]
    return _stable_short_hash("|".join(parts), length=20)


class SubWebActionCoverageFeatures:
    """Run-scoped state-action coverage statistics used as learnable inputs."""

    def __init__(self, coverage_tracker) -> None:
        self.coverage_tracker = coverage_tracker
        self.action_counts: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        self.zero_gain_counts: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        self.last_zero_gain: DefaultDict[Tuple[str, str], bool] = defaultdict(bool)
        self.source_visit_counts: DefaultDict[str, int] = defaultdict(int)
        self.source_gain_counts: DefaultDict[str, int] = defaultdict(int)
        self.source_zero_gain_streaks: DefaultDict[str, int] = defaultdict(int)
        self.target_in_counts: DefaultDict[str, int] = defaultdict(int)
        self.target_gain_counts: DefaultDict[str, int] = defaultdict(int)
        self.context_action_signatures: DefaultDict[str, Set[str]] = defaultdict(set)
        self.context_action_counts: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        self.source_action_signatures: DefaultDict[str, Set[str]] = defaultdict(set)
        self.action_target_contexts: Dict[Tuple[str, str], str] = {}

    def key_for(self, source_node_id: str, source_url: str, action: WebAction) -> Tuple[str, str]:
        context = _context_from_url(source_url)
        source_node = str(source_node_id or "").strip() or context
        return source_node, _exact_action_signature(action)

    def _edge_seen(self, source_node_id: str, source_url: str, action: WebAction) -> bool:
        key = self.key_for(source_node_id, source_url, action)
        return self.action_counts[key] > 0

    def _target_seen(self, action: WebAction) -> bool:
        return self._target_context_seen(_target_context_for(action))

    def _target_context_seen(self, target_context: str) -> bool:
        if not target_context:
            return False
        prefix = f"node:{target_context}|"
        exact = f"node:{target_context}"
        return any(
            str(node) == exact or str(node).startswith(prefix)
            for node in getattr(self.coverage_tracker, "covered_nodes", set())
        )

    def _estimated_graph_marginal(
        self,
        source_node_id: str,
        source_url: str,
        action: WebAction,
        target_context: str = "",
    ) -> Tuple[float, float, float]:
        """Estimate the current one-step marginal F for this visible action.

        This mirrors the graph residual auxiliary target and exposes the same
        objective-local signal as learnable input. It does not add reward,
        filter actions, or modify logits directly.
        """
        source_node = str(source_node_id or "").strip()
        prefix = graph_edge_prefix(source_url, action, source_node_id=source_node)
        covered_edges = set(getattr(self.coverage_tracker, "covered_edges", set()))
        edge_alpha = float(getattr(self.coverage_tracker, "graph_edge_alpha", 1.0))
        node_weight = float(getattr(self.coverage_tracker, "graph_node_weight", 1.0))

        edge_gain = 0.0
        if prefix and not any(str(edge).startswith(prefix) for edge in covered_edges):
            edge_gain = edge_alpha * float(self.coverage_tracker._graph_edge_weight(prefix))

        node_gain = 0.0
        resolved_target_context = str(target_context or "").strip() or _target_context_for(action)
        if resolved_target_context and not self._target_context_seen(resolved_target_context):
            node_gain = node_weight

        edge_norm = 0.0 if edge_alpha <= 1e-8 else max(0.0, min(1.0, edge_gain / edge_alpha))
        node_norm = 0.0 if node_weight <= 1e-8 else max(0.0, min(1.0, node_gain / node_weight))
        denom = max(1e-8, edge_alpha + node_weight)
        total_norm = max(0.0, min(1.0, (edge_gain + node_gain) / denom))
        return edge_norm, node_norm, total_norm

    def _untried_ratio(
        self,
        source_node_id: str,
        source_url: str,
        actions_full: Iterable[WebAction],
    ) -> float:
        actions = list(actions_full)
        if not actions:
            return 0.0
        untried = 0
        for action in actions:
            _, _, graph_marginal_norm = self._estimated_graph_marginal(
                source_node_id,
                source_url,
                action,
            )
            if graph_marginal_norm > 1e-8:
                untried += 1
        return float(untried) / float(len(actions))

    def _observe_context_actions(self, context: str, actions_full: Iterable[WebAction]) -> None:
        for action in actions_full:
            self.context_action_signatures[context].add(_exact_action_signature(action))

    def _observe_source_actions(self, source_key: str, actions_full: Iterable[WebAction]) -> None:
        for action in actions_full:
            self.source_action_signatures[source_key].add(_exact_action_signature(action))

    def _source_untried_count(
        self,
        source_node_id: str,
        source_url: str,
        actions_full: Iterable[WebAction],
    ) -> int:
        count = 0
        for action in actions_full:
            _, _, graph_marginal_norm = self._estimated_graph_marginal(
                source_node_id,
                source_url,
                action,
            )
            if graph_marginal_norm > 1e-8:
                count += 1
        return count

    def _global_edge_progress(self) -> Tuple[float, int]:
        known = 0
        seen = 0
        for source_key, signatures in self.source_action_signatures.items():
            for signature in signatures:
                known += 1
                if self.action_counts[(source_key, signature)] > 0:
                    seen += 1
        if known <= 0:
            return 0.0, 0
        residual = max(0, known - seen)
        return float(seen) / float(known), residual

    def _target_frontier_count(self, target_context: str) -> int:
        signatures = self.context_action_signatures.get(target_context, set())
        if not signatures:
            return 0
        return sum(
            1
            for signature in signatures
            if self.context_action_counts[(target_context, signature)] == 0
        )

    def _target_frontier_ratio(self, target_context: str) -> float:
        signatures = self.context_action_signatures.get(target_context, set())
        if not signatures:
            return 0.0
        return float(self._target_frontier_count(target_context)) / float(len(signatures))

    @staticmethod
    def _frontier_target_context(action: WebAction, context: str, recorded_target: str = "") -> str:
        """Known non-self target used for residual frontier-at-target features.

        Target-frontier features are meant to describe travel to another known
        graph context with remaining actions. Form inputs/selects and same-page
        submit retries should not inherit the current page's residual frontier;
        otherwise a spent self-loop can look valuable simply because the source
        page still has unrelated actions.
        """
        explicit_target = _target_context_for(action)
        if explicit_target and explicit_target != context:
            return explicit_target
        recorded = str(recorded_target or "").strip()
        if recorded and recorded != context:
            return recorded
        return ""

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return max(0.0, min(1.0, float(numerator) / float(denominator)))

    def features_for(
        self,
        action: WebAction,
        web_state: WebState,
        actions_full: Iterable[WebAction],
        source_node_id: Optional[str] = None,
    ) -> torch.Tensor:
        source_url = str(getattr(web_state, "raw_url", getattr(web_state, "url", "")) or "")
        node_id = str(source_node_id or "").strip()
        if not node_id and isinstance(web_state, ActionSetWithExecutionTimesState):
            try:
                from agent.impl.subweb_frontier_coverage import extract_graph_node_id

                node_id = extract_graph_node_id(web_state)
            except Exception:
                node_id = _context_from_state(web_state)

        key = self.key_for(node_id, source_url, action)
        context = _context_from_url(source_url)
        self._observe_context_actions(context, actions_full)
        source_key = str(node_id or context)
        self._observe_source_actions(source_key, actions_full)
        actions = list(actions_full)
        edge_seen = self._edge_seen(node_id, source_url, action)
        untried_count = self._source_untried_count(node_id, source_url, actions)
        untried_ratio = float(untried_count) / float(max(1, len(actions)))
        global_seen_ratio, global_residual_count = self._global_edge_progress()
        target_context = self.action_target_contexts.get(key) or _target_context_for(action) or context
        frontier_target_context = self._frontier_target_context(
            action,
            context,
            recorded_target=self.action_target_contexts.get(key, ""),
        )
        target_frontier_count = (
            self._target_frontier_count(frontier_target_context) if frontier_target_context else 0
        )
        edge_marginal_norm, target_node_marginal_norm, graph_marginal_norm = self._estimated_graph_marginal(
            node_id,
            source_url,
            action,
            target_context=frontier_target_context,
        )
        source_visits = self.source_visit_counts[source_key]
        target_visits = self.target_in_counts[target_context]
        values: List[float] = [
            1.0 if edge_seen else 0.0,
            1.0 if (not edge_seen and edge_marginal_norm > 1e-8) else 0.0,
            _log_count(self.action_counts[key]),
            _log_count(self.zero_gain_counts[key]),
            1.0 if self.last_zero_gain[key] else 0.0,
            1.0 if self._target_seen(action) else 0.0,
            1.0 if is_template_navigation_or_error_action(action, context) else 0.0,
            untried_ratio,
            1.0 - untried_ratio,
            _log_count(source_visits),
            self._rate(self.source_gain_counts[source_key], source_visits),
            _log_count(self.source_zero_gain_streaks[source_key]),
            _log_count(target_visits),
            self._rate(self.target_gain_counts[target_context], target_visits),
            self._target_frontier_ratio(frontier_target_context) if frontier_target_context else 0.0,
            _log_count(target_frontier_count),
            _log_count(untried_count),
            _log_count(len(actions)),
            global_seen_ratio,
            _log_count(global_residual_count),
            edge_marginal_norm,
            target_node_marginal_norm,
            graph_marginal_norm,
        ]
        return torch.tensor(values, dtype=torch.float32)

    def feature_rows_for_actions(
        self,
        web_state: WebState,
        actions_full: Iterable[WebAction],
        source_node_id: Optional[str] = None,
    ) -> List[torch.Tensor]:
        """Batch equivalent of ``features_for`` for a single visible action set.

        ``features_for`` is intentionally easy to reason about, but calling it
        once per candidate repeats source-level scans such as residual out-edge
        counts. Large Web pages can expose hundreds of actions, so compute the
        source/global terms once and then materialize per-action rows.
        """
        actions = list(actions_full)
        source_url = str(getattr(web_state, "raw_url", getattr(web_state, "url", "")) or "")
        node_id = str(source_node_id or "").strip()
        if not node_id and isinstance(web_state, ActionSetWithExecutionTimesState):
            try:
                from agent.impl.subweb_frontier_coverage import extract_graph_node_id

                node_id = extract_graph_node_id(web_state)
            except Exception:
                node_id = _context_from_state(web_state)

        context = _context_from_url(source_url)
        source_key = str(node_id or context)
        self._observe_context_actions(context, actions)
        self._observe_source_actions(source_key, actions)

        untried_count = 0
        for candidate in actions:
            _, _, graph_marginal_norm = self._estimated_graph_marginal(
                node_id,
                source_url,
                candidate,
            )
            if graph_marginal_norm > 1e-8:
                untried_count += 1
        untried_ratio = float(untried_count) / float(max(1, len(actions)))
        global_seen_ratio, global_residual_count = self._global_edge_progress()
        source_visits = self.source_visit_counts[source_key]
        source_gain_rate = self._rate(self.source_gain_counts[source_key], source_visits)
        source_zero_gain_streak_log = _log_count(self.source_zero_gain_streaks[source_key])
        source_out_seen_ratio = 1.0 - untried_ratio
        source_residual_count_log = _log_count(untried_count)
        source_out_degree_log = _log_count(len(actions))
        global_edge_residual_count_log = _log_count(global_residual_count)

        rows: List[torch.Tensor] = []
        for action in actions:
            key = self.key_for(node_id, source_url, action)
            edge_seen = self._edge_seen(node_id, source_url, action)
            target_context = self.action_target_contexts.get(key) or _target_context_for(action) or context
            frontier_target_context = self._frontier_target_context(
                action,
                context,
                recorded_target=self.action_target_contexts.get(key, ""),
            )
            target_frontier_count = (
                self._target_frontier_count(frontier_target_context) if frontier_target_context else 0
            )
            edge_marginal_norm, target_node_marginal_norm, graph_marginal_norm = (
                self._estimated_graph_marginal(
                    node_id,
                    source_url,
                    action,
                    target_context=frontier_target_context,
                )
            )
            target_visits = self.target_in_counts[target_context]
            values: List[float] = [
                1.0 if edge_seen else 0.0,
                1.0 if (not edge_seen and edge_marginal_norm > 1e-8) else 0.0,
                _log_count(self.action_counts[key]),
                _log_count(self.zero_gain_counts[key]),
                1.0 if self.last_zero_gain[key] else 0.0,
                1.0 if self._target_seen(action) else 0.0,
                1.0 if is_template_navigation_or_error_action(action, context) else 0.0,
                untried_ratio,
                source_out_seen_ratio,
                _log_count(source_visits),
                source_gain_rate,
                source_zero_gain_streak_log,
                _log_count(target_visits),
                self._rate(self.target_gain_counts[target_context], target_visits),
                self._target_frontier_ratio(frontier_target_context) if frontier_target_context else 0.0,
                _log_count(target_frontier_count),
                source_residual_count_log,
                source_out_degree_log,
                global_seen_ratio,
                global_edge_residual_count_log,
                edge_marginal_norm,
                target_node_marginal_norm,
                graph_marginal_norm,
            ]
            rows.append(torch.tensor(values, dtype=torch.float32))
        return rows

    def update(
        self,
        source_node_id: str,
        source_url: str,
        action: Optional[WebAction],
        reward: float,
        target_state: Optional[WebState] = None,
    ) -> Dict[str, object]:
        if action is None:
            return {}
        key = self.key_for(source_node_id, source_url, action)
        context = _context_from_url(source_url)
        action_signature = key[1]
        self.context_action_signatures[context].add(action_signature)
        source_key = str(source_node_id or "").strip() or context
        self.source_action_signatures[source_key].add(action_signature)
        if target_state is not None:
            target_context = _context_from_state(target_state)
        else:
            target_context = _target_context_for(action) or context
        self.action_target_contexts[key] = target_context
        before = {
            "coverage_action_count_before": self.action_counts[key],
            "coverage_zero_gain_count_before": self.zero_gain_counts[key],
            "coverage_source_visit_count_before": self.source_visit_counts[source_key],
            "coverage_source_zero_gain_streak_before": self.source_zero_gain_streaks[source_key],
            "coverage_target_in_count_before": self.target_in_counts[target_context],
        }
        self.action_counts[key] += 1
        if float(reward) <= 0.0:
            self.zero_gain_counts[key] += 1
            self.last_zero_gain[key] = True
            self.source_zero_gain_streaks[source_key] += 1
        else:
            self.zero_gain_counts[key] = 0
            self.last_zero_gain[key] = False
            self.source_gain_counts[source_key] += 1
            self.source_zero_gain_streaks[source_key] = 0
            self.target_gain_counts[target_context] += 1
        self.source_visit_counts[source_key] += 1
        self.target_in_counts[target_context] += 1
        self.context_action_counts[(context, action_signature)] += 1
        before.update(
            {
                "coverage_action_count_after": self.action_counts[key],
                "coverage_zero_gain_count_after": self.zero_gain_counts[key],
                "coverage_last_zero_gain": self.last_zero_gain[key],
                "coverage_source_visit_count_after": self.source_visit_counts[source_key],
                "coverage_source_zero_gain_streak_after": self.source_zero_gain_streaks[source_key],
                "coverage_target_context": target_context,
                "coverage_target_in_count_after": self.target_in_counts[target_context],
            }
        )
        return before
