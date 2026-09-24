"""Site-agnostic structural action features for SubWeb frontier agents.

These features describe how an available action relates to the current web
context. They intentionally avoid benchmark-specific nouns and do not read
future rewards or outcomes.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Sequence, Tuple
from urllib.parse import urlsplit

import torch

from action.impl.click_action import ClickAction
from action.impl.random_input_action import RandomInputAction
from action.impl.random_select_action import RandomSelectAction
from action.web_action import WebAction
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState


_TOKEN_RE = re.compile(r"[\W_]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_URL_LIKE_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)

_BOILERPLATE_WORDS = {
    "about",
    "back",
    "cancel",
    "docs",
    "documentation",
    "help",
    "home",
    "log in",
    "log out",
    "login",
    "logout",
    "menu",
    "overview",
    "profile",
    "root",
    "settings",
}

_CREATE_WORDS = {"add", "create", "new"}
_EDIT_WORDS = {"edit", "update", "save"}
_SEARCH_WORDS = {"find", "filter", "search"}
_DELETE_WORDS = {"delete", "destroy", "remove"}
_NEXT_WORDS = {"confirm", "continue", "done", "next", "ok", "submit"}
_ERROR_WORDS = {"500", "error", "exception", "oops", "oups"}
_AUTH_WORDS = {"log in", "log out", "login", "logout", "sign in", "sign out"}

_ACTION_FEATURE_NAMES: Tuple[str, ...] = (
    "has_target_route",
    "same_route",
    "parent_route",
    "child_descendant_route",
    "sibling_route",
    "external_target",
    "error_target_or_text",
    "auth_or_logout",
    "root_or_home_target",
    "target_depth_norm",
    "current_depth_norm",
    "positive_depth_delta_norm",
    "negative_depth_delta_norm",
    "same_target_count_norm",
    "same_signature_count_norm",
    "create_new_add_word",
    "edit_update_save_word",
    "search_find_word",
    "delete_remove_word",
    "next_continue_confirm_word",
    "local_non_nav_route_action",
    "likely_self_loop",
    "likely_global_or_boilerplate",
    "has_url_target",
)

_SET_FEATURE_NAMES: Tuple[str, ...] = (
    "target_route_count_norm",
    "frac_same_route",
    "frac_parent_route",
    "frac_child_descendant_route",
    "frac_sibling_route",
    "frac_external_target",
    "frac_error_target",
    "frac_create_new_add",
    "frac_edit_update_save",
    "frac_search_find",
    "has_local_workflow_action",
    "frac_local_workflow_action",
)


def structural_action_feature_names() -> Tuple[str, ...]:
    return _ACTION_FEATURE_NAMES


def structural_action_set_feature_names() -> Tuple[str, ...]:
    return _SET_FEATURE_NAMES


def structural_feature_dim() -> int:
    return len(_ACTION_FEATURE_NAMES) + len(_SET_FEATURE_NAMES)


def _norm_count(value: int, cap: int = 20) -> float:
    return min(max(int(value), 0), cap) / float(cap)


def _normalize_text(value) -> str:
    text = str(value or "").strip().lower()
    text = _TOKEN_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def _text_words(value) -> set:
    return {part for part in _normalize_text(value).split(" ") if part}


def _contains_any(text: str, words: Iterable[str]) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    padded = f" {normalized} "
    for word in words:
        word_norm = _normalize_text(word)
        if word_norm and f" {word_norm} " in padded:
            return True
    return False


def _action_text_blob(action: WebAction) -> str:
    return " ".join(
        str(part or "")
        for part in (
            getattr(action, "text", ""),
            getattr(action, "addition_info", ""),
            getattr(action, "location", ""),
        )
    )


def _current_raw_url(web_state) -> str:
    return str(getattr(web_state, "raw_url", getattr(web_state, "url", "")) or "")


def _current_route(web_state) -> str:
    return ActionSetWithExecutionTimesState._normalize_route_identity(_current_raw_url(web_state))


def _route_segments(route: str) -> List[str]:
    route = str(route or "").split("?", 1)[0].split("#", 1)[0].strip("/")
    return [part for part in route.split("/") if part]


def _target_text(action: WebAction) -> str:
    if not isinstance(action, ClickAction):
        return ""
    return str(getattr(action, "addition_info", "") or "").strip()


def _target_route(action: WebAction) -> str:
    target = _target_text(action)
    if not target:
        return ""
    if target.startswith("#"):
        return ActionSetWithExecutionTimesState._normalize_route_identity(target)
    if target.startswith("/") or _URL_LIKE_RE.search(target):
        return ActionSetWithExecutionTimesState._normalize_route_identity(target)
    return ""


def _is_external_target(action: WebAction, web_state) -> bool:
    target = _target_text(action)
    if not _URL_LIKE_RE.search(target):
        return False
    try:
        target_host = urlsplit(target).netloc.lower()
        current_host = urlsplit(_current_raw_url(web_state)).netloc.lower()
        return bool(target_host and current_host and target_host != current_host)
    except Exception:
        return False


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


def _target_key(action: WebAction) -> str:
    return _target_route(action) or _normalize_text(_target_text(action))


def _relation_flags(action: WebAction, web_state) -> dict:
    current = _current_route(web_state)
    target = _target_route(action)
    current_parts = _route_segments(current)
    target_parts = _route_segments(target)
    has_target = bool(target)
    same = has_target and target == current
    parent = has_target and len(target_parts) < len(current_parts) and current_parts[: len(target_parts)] == target_parts
    child = has_target and len(target_parts) > len(current_parts) and target_parts[: len(current_parts)] == current_parts
    sibling = (
        has_target
        and len(target_parts) == len(current_parts)
        and len(target_parts) > 0
        and target_parts[:-1] == current_parts[:-1]
        and target_parts[-1:] != current_parts[-1:]
    )
    root_or_home = target in {"", "/", "#/"} or _contains_any(_action_text_blob(action), {"home", "root"})
    return {
        "current": current,
        "target": target,
        "has_target": has_target,
        "same": same,
        "parent": parent,
        "child": child,
        "sibling": sibling,
        "root_or_home": root_or_home,
        "current_depth": len(current_parts),
        "target_depth": len(target_parts),
        "depth_delta": len(target_parts) - len(current_parts),
    }


def _generic_word_flags(action: WebAction) -> dict:
    blob = _action_text_blob(action)
    return {
        "create": _contains_any(blob, _CREATE_WORDS),
        "edit": _contains_any(blob, _EDIT_WORDS),
        "search": _contains_any(blob, _SEARCH_WORDS),
        "delete": _contains_any(blob, _DELETE_WORDS),
        "next": _contains_any(blob, _NEXT_WORDS),
        "error": _contains_any(blob, _ERROR_WORDS),
        "auth": _contains_any(blob, _AUTH_WORDS),
        "boilerplate": _contains_any(blob, _BOILERPLATE_WORDS),
    }


def _is_form_action(action: WebAction) -> bool:
    if isinstance(action, (RandomInputAction, RandomSelectAction)):
        return True
    if isinstance(action, ClickAction):
        action_type = str(getattr(action, "action_type", "") or "").strip().lower()
        if action_type == "submit":
            return True
        if action_type == "redirect":
            return False
        blob = _normalize_text(
            " ".join(
                str(part or "")
                for part in (
                    getattr(action, "addition_info", ""),
                    getattr(action, "location", ""),
                    getattr(action, "text", ""),
                )
            )
        )
        padded = f" {blob} "
        return any(
            f" {token} " in padded
            for token in ("input", "select", "textarea", "checkbox", "radio", "option", "label")
        )
    return False


def _same_target_count(action: WebAction, all_actions: Sequence[WebAction]) -> int:
    key = _target_key(action)
    if not key:
        return 0
    return sum(1 for item in all_actions if _target_key(item) == key)


def _same_signature_count(action: WebAction, all_actions: Sequence[WebAction]) -> int:
    signature = _action_signature(action)
    return sum(1 for item in all_actions if _action_signature(item) == signature)


def _is_local_workflow_action(
    action: WebAction,
    web_state,
    all_actions: Sequence[WebAction],
    same_target_count: int | None = None,
) -> bool:
    relation = _relation_flags(action, web_state)
    words = _generic_word_flags(action)
    is_global = words["boilerplate"] or relation["root_or_home"] or words["auth"]
    if _is_external_target(action, web_state) or words["error"] or is_global:
        return False
    if _is_form_action(action):
        return True
    if relation["child"] and (words["create"] or words["edit"] or words["search"] or words["next"]):
        return True
    if relation["same"] and (words["edit"] or words["search"] or words["next"]):
        return True
    if relation["sibling"] and (words["create"] or words["edit"] or words["search"]):
        return True
    if relation["has_target"] and not relation["parent"] and not is_global:
        if same_target_count is None:
            same_target_count = _same_target_count(action, all_actions)
        return same_target_count <= 2
    return False


def structural_action_features(
    action: WebAction,
    web_state,
    all_actions: Sequence[WebAction],
    same_target_count: int | None = None,
    same_signature_count: int | None = None,
) -> torch.Tensor:
    relation = _relation_flags(action, web_state)
    words = _generic_word_flags(action)
    external = _is_external_target(action, web_state)
    if same_target_count is None:
        same_target_count = _same_target_count(action, all_actions)
    if same_signature_count is None:
        same_signature_count = _same_signature_count(action, all_actions)
    depth_delta = relation["depth_delta"]
    local_workflow = _is_local_workflow_action(
        action,
        web_state,
        all_actions,
        same_target_count=same_target_count,
    )
    self_loop = relation["same"] or (not relation["has_target"] and not _is_form_action(action))

    values = [
        float(relation["has_target"]),
        float(relation["same"]),
        float(relation["parent"]),
        float(relation["child"]),
        float(relation["sibling"]),
        float(external),
        float(words["error"]),
        float(words["auth"]),
        float(relation["root_or_home"]),
        _norm_count(relation["target_depth"], cap=10),
        _norm_count(relation["current_depth"], cap=10),
        _norm_count(max(depth_delta, 0), cap=5),
        _norm_count(max(-depth_delta, 0), cap=5),
        _norm_count(same_target_count),
        _norm_count(same_signature_count),
        float(words["create"]),
        float(words["edit"]),
        float(words["search"]),
        float(words["delete"]),
        float(words["next"]),
        float(local_workflow),
        float(self_loop),
        float(words["boilerplate"] or relation["root_or_home"] or words["auth"]),
        float(bool(_target_text(action))),
    ]
    return torch.tensor(values, dtype=torch.float32)


def structural_action_feature_rows(web_state, all_actions: Sequence[WebAction]) -> List[torch.Tensor]:
    actions = list(all_actions)
    target_counts = Counter(_target_key(action) for action in actions if _target_key(action))
    signature_counts = Counter(_action_signature(action) for action in actions)
    return [
        structural_action_features(
            action,
            web_state,
            actions,
            same_target_count=target_counts.get(_target_key(action), 0),
            same_signature_count=signature_counts.get(_action_signature(action), 0),
        )
        for action in actions
    ]


def structural_action_set_features_from_rows(rows: Sequence[torch.Tensor]) -> torch.Tensor:
    denom = max(1, len(rows))
    if not rows:
        return torch.zeros((len(_SET_FEATURE_NAMES),), dtype=torch.float32)
    stacked = torch.stack([row.float() for row in rows])
    has_target_count = int(stacked[:, _ACTION_FEATURE_NAMES.index("has_target_route")].sum().item())
    local_idx = _ACTION_FEATURE_NAMES.index("local_non_nav_route_action")
    values = [
        _norm_count(has_target_count),
        float(stacked[:, _ACTION_FEATURE_NAMES.index("same_route")].mean().item()),
        float(stacked[:, _ACTION_FEATURE_NAMES.index("parent_route")].mean().item()),
        float(stacked[:, _ACTION_FEATURE_NAMES.index("child_descendant_route")].mean().item()),
        float(stacked[:, _ACTION_FEATURE_NAMES.index("sibling_route")].mean().item()),
        float(stacked[:, _ACTION_FEATURE_NAMES.index("external_target")].mean().item()),
        float(stacked[:, _ACTION_FEATURE_NAMES.index("error_target_or_text")].mean().item()),
        float(stacked[:, _ACTION_FEATURE_NAMES.index("create_new_add_word")].mean().item()),
        float(stacked[:, _ACTION_FEATURE_NAMES.index("edit_update_save_word")].mean().item()),
        float(stacked[:, _ACTION_FEATURE_NAMES.index("search_find_word")].mean().item()),
        float(stacked[:, local_idx].sum().item() > 0),
        float(stacked[:, local_idx].sum().item() / denom),
    ]
    return torch.tensor(values, dtype=torch.float32)


def structural_action_set_features(web_state, all_actions: Sequence[WebAction]) -> torch.Tensor:
    rows = structural_action_feature_rows(web_state, all_actions)
    return structural_action_set_features_from_rows(rows)


def structural_action_rows_and_set_features(
    web_state,
    all_actions: Sequence[WebAction],
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    actions = list(all_actions)
    rows = structural_action_feature_rows(web_state, actions)
    return rows, structural_action_set_features_from_rows(rows)


def _legacy_structural_action_set_features(web_state, all_actions: Sequence[WebAction]) -> torch.Tensor:
    actions = list(all_actions)
    denom = max(1, len(actions))
    rows = [structural_action_features(action, web_state, actions) for action in actions]
    if not rows:
        return torch.zeros((len(_SET_FEATURE_NAMES),), dtype=torch.float32)
    stacked = torch.stack(rows)
    has_target_count = int(stacked[:, _ACTION_FEATURE_NAMES.index("has_target_route")].sum().item())
    local_idx = _ACTION_FEATURE_NAMES.index("local_non_nav_route_action")
    values = [
        _norm_count(has_target_count),
        float(stacked[:, _ACTION_FEATURE_NAMES.index("same_route")].mean().item()),
        float(stacked[:, _ACTION_FEATURE_NAMES.index("parent_route")].mean().item()),
        float(stacked[:, _ACTION_FEATURE_NAMES.index("child_descendant_route")].mean().item()),
        float(stacked[:, _ACTION_FEATURE_NAMES.index("sibling_route")].mean().item()),
        float(stacked[:, _ACTION_FEATURE_NAMES.index("external_target")].mean().item()),
        float(stacked[:, _ACTION_FEATURE_NAMES.index("error_target_or_text")].mean().item()),
        float(stacked[:, _ACTION_FEATURE_NAMES.index("create_new_add_word")].mean().item()),
        float(stacked[:, _ACTION_FEATURE_NAMES.index("edit_update_save_word")].mean().item()),
        float(stacked[:, _ACTION_FEATURE_NAMES.index("search_find_word")].mean().item()),
        float(stacked[:, local_idx].sum().item() > 0),
        float(stacked[:, local_idx].sum().item() / denom),
    ]
    return torch.tensor(values, dtype=torch.float32)


def augment_with_structural_features(
    base_tensor: torch.Tensor,
    action: WebAction,
    web_state,
    all_actions: Sequence[WebAction],
    action_set_structural_context: torch.Tensor,
    structural_features: torch.Tensor | None = None,
) -> torch.Tensor:
    structural = (
        structural_features.float()
        if structural_features is not None
        else structural_action_features(action, web_state, all_actions)
    )
    return torch.cat([base_tensor.float(), structural, action_set_structural_context.float()]).float()
