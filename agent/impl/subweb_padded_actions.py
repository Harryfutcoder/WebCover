"""Padded candidate-action utilities for SubWeb frontier agents."""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from action.impl.click_action import ClickAction
from action.impl.random_input_action import RandomInputAction
from action.impl.random_select_action import RandomSelectAction
from action.web_action import WebAction
from agent.impl.subweb_action_context_features import (
    action_role_feature_rows,
    action_role_features,
    action_set_context_features_from_rows,
)
from agent.impl.subweb_structural_action_features import (
    augment_with_structural_features,
    structural_action_feature_names,
    structural_action_feature_rows,
    structural_action_set_features_from_rows,
    structural_action_features,
)
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState

logger = logging.getLogger(__name__)

_BOILERPLATE_WORDS = {
    "about",
    "back",
    "cancel",
    "help",
    "home",
    "log in",
    "log out",
    "login",
    "logout",
    "menu",
    "overview",
    "root",
    "veterinarians",
}

_TOKEN_RE = re.compile(r"[\W_]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


@dataclass
class PaddedActionBatch:
    actions_full: List[WebAction]
    actions_policy: List[WebAction]
    action_mat: torch.Tensor
    action_mask: torch.Tensor
    original_action_count: int
    kept_action_count: int
    truncated_count: int
    truncated_by_type: Dict[str, int]
    action_set_context: torch.Tensor
    structural_action_set_context: torch.Tensor
    build_elapsed_ms: float = 0.0
    rank_elapsed_ms: float = 0.0
    context_elapsed_ms: float = 0.0
    tensor_elapsed_ms: float = 0.0
    role_features_policy: Optional[List[torch.Tensor]] = None
    structural_features_policy: Optional[List[torch.Tensor]] = None
    coverage_features_policy: Optional[List[torch.Tensor]] = None


def _normalize_text(value) -> str:
    text = str(value or "").strip().lower()
    text = _TOKEN_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def action_kind(action: WebAction) -> str:
    if isinstance(action, RandomInputAction):
        return "input"
    if isinstance(action, RandomSelectAction):
        return "select"
    if isinstance(action, ClickAction):
        action_type = str(getattr(action, "action_type", "") or "").lower()
        if action_type == "redirect":
            return "redirect"
        if action_type == "submit":
            return "submit"
        return "click"
    return "default"


def _action_signature(action: WebAction) -> str:
    if isinstance(action, ClickAction):
        return ActionSetWithExecutionTimesState._canonical_action_signature(
            action,
            getattr(action, "action_type", "default") or "default",
            getattr(action, "addition_info", ""),
        )
    if isinstance(action, RandomInputAction):
        return ActionSetWithExecutionTimesState._canonical_action_signature(action, "random_input")
    if isinstance(action, RandomSelectAction):
        return ActionSetWithExecutionTimesState._canonical_action_signature(action, "random_select")
    return str(action)


def action_signature(action: WebAction) -> str:
    return _action_signature(action)


def _is_boilerplate(action: WebAction) -> bool:
    text = _normalize_text(
        f"{getattr(action, 'text', '')} {getattr(action, 'addition_info', '')} {getattr(action, 'location', '')}"
    )
    padded = f" {text} "
    return any(f" {word} " in padded for word in _BOILERPLATE_WORDS)


def _action_priority_from_structural(
    action: WebAction,
    web_state,
    all_actions: Sequence[WebAction],
    structural: torch.Tensor,
    role: torch.Tensor | None = None,
) -> Tuple[int, str]:
    if role is None:
        role = action_role_features(action, web_state, list(all_actions))
    structural_names = structural_action_feature_names()
    is_submit = role[2].item() > 0.5
    is_input = role[3].item() > 0.5
    is_select = role[4].item() > 0.5
    is_form = role[6].item() > 0.5
    is_nav = role[7].item() > 0.5 or role[8].item() > 0.5 or role[9].item() > 0.5 or _is_boilerplate(action)
    is_local_workflow = structural[structural_names.index("local_non_nav_route_action")].item() > 0.5
    is_error = structural[structural_names.index("error_target_or_text")].item() > 0.5
    kind = action_kind(action)

    if is_input or is_select:
        bucket = 0
    elif is_submit:
        bucket = 1
    elif is_form:
        bucket = 2
    elif is_local_workflow:
        bucket = 3
    elif is_error:
        bucket = 7
    elif kind in {"click", "default"}:
        bucket = 4
    elif kind == "redirect" and not is_nav:
        bucket = 5
    elif is_nav:
        bucket = 8
    else:
        bucket = 6
    return bucket, _action_signature(action)


def action_priority(action: WebAction, web_state, all_actions: Sequence[WebAction]) -> Tuple[int, str]:
    actions = list(all_actions)
    structural = structural_action_features(action, web_state, actions)
    return _action_priority_from_structural(action, web_state, actions, structural)


def deterministic_rank_actions(actions: Sequence[WebAction], web_state) -> List[WebAction]:
    actions = list(actions)
    structural_rows = structural_action_feature_rows(web_state, actions)
    role_rows = action_role_feature_rows(web_state, actions)
    structural_by_id = {id(action): row for action, row in zip(actions, structural_rows)}
    role_by_id = {id(action): row for action, row in zip(actions, role_rows)}
    return sorted(
        actions,
        key=lambda action: _action_priority_from_structural(
            action,
            web_state,
            actions,
            structural_by_id[id(action)],
            role=role_by_id[id(action)],
        ),
    )


def _pad_action_tensors(action_tensors: Sequence[torch.Tensor], max_actions: int, input_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    action_mat = torch.zeros((max_actions, input_dim), dtype=torch.float32)
    action_mask = torch.zeros((max_actions,), dtype=torch.bool)
    for idx, tensor in enumerate(action_tensors[:max_actions]):
        action_mat[idx] = tensor.float()
        action_mask[idx] = True
    return action_mat, action_mask


def build_padded_action_batch(
    web_state,
    html: str,
    transformer,
    max_actions: int,
    input_dim: int,
    include_structural_features: bool = True,
    coverage_feature_provider: Optional[Any] = None,
    source_node_id: Optional[str] = None,
) -> PaddedActionBatch:
    build_start = time.perf_counter()
    actions_full = list(web_state.get_action_list())
    rank_start = time.perf_counter()
    structural_rows = structural_action_feature_rows(web_state, actions_full)
    role_rows = action_role_feature_rows(web_state, actions_full)
    structural_by_id = {id(action): row.float() for action, row in zip(actions_full, structural_rows)}
    role_by_id = {id(action): row.float() for action, row in zip(actions_full, role_rows)}
    actions_ranked = sorted(
        actions_full,
        key=lambda action: _action_priority_from_structural(
            action,
            web_state,
            actions_full,
            structural_by_id[id(action)],
            role=role_by_id[id(action)],
        ),
    )
    rank_elapsed_ms = (time.perf_counter() - rank_start) * 1000.0
    actions_policy = actions_ranked[:max_actions]

    context_start = time.perf_counter()
    state_tensor = transformer.state_to_tensor(web_state, html).float()
    ctx = action_set_context_features_from_rows(web_state, actions_full, role_rows).float()
    if include_structural_features:
        structural_ctx = structural_action_set_features_from_rows(structural_rows).float()
    else:
        structural_ctx = torch.zeros(0)
    context_elapsed_ms = (time.perf_counter() - context_start) * 1000.0

    tensor_start = time.perf_counter()
    coverage_by_id: Dict[int, torch.Tensor] = {}
    if coverage_feature_provider is not None:
        feature_rows_for_actions = getattr(coverage_feature_provider, "feature_rows_for_actions", None)
        if callable(feature_rows_for_actions):
            coverage_rows = feature_rows_for_actions(
                web_state,
                actions_full,
                source_node_id=source_node_id,
            )
            coverage_by_id = {
                id(action): row.float() for action, row in zip(actions_full, coverage_rows)
            }
    tensors: List[torch.Tensor] = []
    role_features_policy: List[torch.Tensor] = []
    structural_features_policy: List[torch.Tensor] = []
    coverage_features_policy: List[torch.Tensor] = []
    for action in actions_policy:
        action_tensor = transformer.action_to_tensor(web_state, action).float()
        role = role_by_id[id(action)]
        role_features_policy.append(role)
        base = torch.cat([state_tensor, action_tensor, role, ctx]).float()
        if include_structural_features:
            structural_features = structural_by_id.get(id(action))
            if structural_features is not None:
                structural_features_policy.append(structural_features)
            base = augment_with_structural_features(
                base,
                action,
                web_state,
                actions_full,
                structural_ctx,
                structural_features=structural_features,
            )
        if coverage_feature_provider is not None:
            coverage_features = coverage_by_id.get(id(action))
            if coverage_features is None:
                coverage_features = coverage_feature_provider.features_for(
                    action,
                    web_state,
                    actions_full,
                    source_node_id=source_node_id,
                ).float()
            coverage_features_policy.append(coverage_features)
            base = torch.cat([base, coverage_features]).float()
        tensors.append(base.float())

    action_mat, action_mask = _pad_action_tensors(tensors, max_actions, input_dim)
    tensor_elapsed_ms = (time.perf_counter() - tensor_start) * 1000.0
    truncated = actions_ranked[max_actions:]
    truncated_by_type = dict(Counter(action_kind(action) for action in truncated))
    if truncated:
        logger.info(
            "SubWeb action truncation: original=%d kept=%d truncated=%d by_type=%s",
            len(actions_full),
            len(actions_policy),
            len(truncated),
            truncated_by_type,
        )

    return PaddedActionBatch(
        actions_full=actions_full,
        actions_policy=actions_policy,
        action_mat=action_mat,
        action_mask=action_mask,
        original_action_count=len(actions_full),
        kept_action_count=len(actions_policy),
        truncated_count=len(truncated),
        truncated_by_type=truncated_by_type,
        action_set_context=ctx,
        structural_action_set_context=structural_ctx,
        build_elapsed_ms=(time.perf_counter() - build_start) * 1000.0,
        rank_elapsed_ms=rank_elapsed_ms,
        context_elapsed_ms=context_elapsed_ms,
        tensor_elapsed_ms=tensor_elapsed_ms,
        role_features_policy=role_features_policy,
        structural_features_policy=structural_features_policy if include_structural_features else None,
        coverage_features_policy=coverage_features_policy if coverage_feature_provider is not None else None,
    )
