"""Action-set context features for WebCover policy inputs.

These features only change policy parameterization. They do not read rewards,
outcomes, future states, or marginal gains.
"""

import re
from collections import Counter
from typing import Iterable, List
from urllib.parse import urlsplit

import torch

from action.impl.click_action import ClickAction
from action.impl.random_input_action import RandomInputAction
from action.impl.random_select_action import RandomSelectAction
from action.web_action import WebAction


_SUBMIT_WORDS = {
    "add",
    "create",
    "confirm",
    "done",
    "next",
    "ok",
    "save",
    "submit",
    "update",
}

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
    "root",
}

_GLOBAL_NAV_WORDS = _BOILERPLATE_WORDS | {
    "docs",
    "documentation",
    "features",
    "overview",
    "pricing",
    "veterinarians",
}

_DIALOG_WORDS = {
    "dialog",
    "modal",
    "popup",
    "tab",
}

_TOKEN_RE = re.compile(r"[\W_]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")

_ROLE_FEATURE_DIM = 12
_CONTEXT_FEATURE_DIM = 17


def context_feature_dim() -> int:
    return _ROLE_FEATURE_DIM + _CONTEXT_FEATURE_DIM


def _norm_count(value: int, cap: int = 20) -> float:
    return min(max(int(value), 0), cap) / float(cap)


def _normalize_text(value) -> str:
    text = str(value or "").strip().lower()
    text = _TOKEN_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def _action_text(action: WebAction) -> str:
    parts = [
        getattr(action, "text", ""),
        getattr(action, "addition_info", ""),
    ]
    return _normalize_text(" ".join(str(p or "") for p in parts))


def _action_location(action: WebAction) -> str:
    return _normalize_text(getattr(action, "location", ""))


def _contains_any(text: str, words: Iterable[str]) -> bool:
    if not text:
        return False
    padded = f" {text} "
    for word in words:
        word_norm = _normalize_text(word)
        if word_norm and f" {word_norm} " in padded:
            return True
    return False


def _action_kind(action: WebAction) -> str:
    if isinstance(action, RandomInputAction):
        return "input"
    if isinstance(action, RandomSelectAction):
        return "select"
    if isinstance(action, ClickAction):
        action_type = str(getattr(action, "action_type", "") or "").lower()
        text = _action_text(action)
        if action_type == "redirect":
            return "redirect"
        if action_type == "submit" or _contains_any(text, _SUBMIT_WORDS):
            return "submit"
        return "click"
    return "default"


def _execution_count(action: WebAction, web_state) -> int:
    action_dict = getattr(web_state, "action_dict", {}) or {}
    data = action_dict.get(action, {}) if hasattr(action_dict, "get") else {}
    try:
        return int(data.get("execution_time", 0))
    except Exception:
        return 0


def _same_type_count(action: WebAction, all_actions: List[WebAction]) -> int:
    kind = _action_kind(action)
    return sum(1 for item in all_actions if _action_kind(item) == kind)


def _page_root(url: str) -> str:
    try:
        split = urlsplit(str(url or ""))
        return f"{split.scheme}://{split.netloc}/" if split.scheme and split.netloc else "/"
    except Exception:
        return "/"


def _is_empty_or_root_link(action: WebAction, web_state) -> bool:
    if not isinstance(action, ClickAction):
        return False
    if str(getattr(action, "action_type", "") or "").lower() != "redirect":
        return False
    target = str(getattr(action, "addition_info", "") or "").strip()
    if target in {"", "#", "/", "./"}:
        return True
    current_root = _page_root(getattr(web_state, "url", ""))
    try:
        split = urlsplit(target)
        if split.scheme or split.netloc:
            normalized = f"{split.scheme}://{split.netloc}{split.path or '/'}"
            return normalized.rstrip("/") == current_root.rstrip("/")
        return (split.path or "/").rstrip("/") in {"", "/"}
    except Exception:
        return False


def _is_global_nav(action: WebAction) -> bool:
    text = _action_text(action)
    location = _action_location(action)
    if _contains_any(text, _GLOBAL_NAV_WORDS):
        return True
    return _contains_any(location, {"header", "footer", "sidebar", "nav", "navigation", "menu", "global"})


def _is_dialog_like(action: WebAction) -> bool:
    return _contains_any(_action_text(action), _DIALOG_WORDS) or _contains_any(_action_location(action), _DIALOG_WORDS)


def _is_click_form_control(action: WebAction) -> bool:
    if not isinstance(action, ClickAction):
        return False
    action_type = str(getattr(action, "action_type", "") or "").strip().lower()
    if action_type in {"redirect", "submit"}:
        return False
    text = _normalize_text(
        " ".join(
            str(part or "")
            for part in (
                getattr(action, "addition_info", ""),
                getattr(action, "location", ""),
                getattr(action, "text", ""),
            )
        )
    )
    padded = f" {text} "
    return any(
        f" {token} " in padded
        for token in ("input", "select", "textarea", "checkbox", "radio", "option", "label")
    )


def action_role_features(
    action: WebAction,
    web_state,
    all_actions: List[WebAction],
    kind: str | None = None,
    same_type_count: int | None = None,
) -> torch.Tensor:
    kind = kind or _action_kind(action)
    is_click = isinstance(action, ClickAction)
    is_redirect = kind == "redirect"
    is_submit = kind == "submit"
    is_input = kind == "input"
    is_select = kind == "select"
    is_default = kind in {"click", "default"}
    is_form_control_click = _is_click_form_control(action)
    is_form_related = is_input or is_select or is_submit or is_form_control_click
    is_global_nav = _is_global_nav(action)
    is_boilerplate = _contains_any(_action_text(action), _BOILERPLATE_WORDS)
    is_empty_root = _is_empty_or_root_link(action, web_state)
    execution_count_norm = _norm_count(_execution_count(action, web_state))
    if same_type_count is None:
        same_type_count = _same_type_count(action, all_actions)
    same_type_count_norm = _norm_count(same_type_count)
    return torch.tensor(
        [
            float(is_click),
            float(is_redirect),
            float(is_submit),
            float(is_input),
            float(is_select),
            float(is_default),
            float(is_form_related),
            float(is_global_nav),
            float(is_boilerplate),
            float(is_empty_root),
            execution_count_norm,
            same_type_count_norm,
        ],
        dtype=torch.float32,
    )


def action_role_feature_rows(web_state, all_actions: List[WebAction]) -> List[torch.Tensor]:
    actions = list(all_actions)
    kinds = [_action_kind(action) for action in actions]
    kind_counts = Counter(kinds)
    return [
        action_role_features(
            action,
            web_state,
            actions,
            kind=kind,
            same_type_count=kind_counts[kind],
        )
        for action, kind in zip(actions, kinds)
    ]


def action_set_context_features_from_rows(
    web_state,
    all_actions: List[WebAction],
    role_rows: List[torch.Tensor],
) -> torch.Tensor:
    n_actions = len(all_actions)
    kinds = Counter(_action_kind(action) for action in all_actions)
    n_redirect = kinds["redirect"]
    n_submit = kinds["submit"]
    n_input = kinds["input"]
    n_select = kinds["select"]
    n_default = kinds["default"] + kinds["click"]
    n_form_related = n_submit + n_input + n_select
    n_global_nav = int(sum(float(row[7].item()) for row in role_rows))
    n_dialog = sum(1 for action in all_actions if _is_dialog_like(action))
    denom = max(1, n_actions)
    return torch.tensor(
        [
            _norm_count(n_actions),
            _norm_count(n_redirect),
            _norm_count(n_submit),
            _norm_count(n_input),
            _norm_count(n_select),
            _norm_count(n_default),
            n_redirect / denom,
            n_submit / denom,
            n_input / denom,
            n_select / denom,
            n_form_related / denom,
            n_global_nav / denom,
            float(n_input > 0),
            float(n_submit > 0),
            float(n_input > 0 and n_submit > 0),
            float(n_select > 0),
            float(n_dialog > 0),
        ],
        dtype=torch.float32,
    )


def action_set_context_features(web_state, all_actions: List[WebAction]) -> torch.Tensor:
    role_rows = action_role_feature_rows(web_state, all_actions)
    return action_set_context_features_from_rows(web_state, all_actions, role_rows)


def augment_action_tensors(base_tensors: List[torch.Tensor], web_state, all_actions: List[WebAction]) -> List[torch.Tensor]:
    if len(base_tensors) != len(all_actions):
        raise ValueError("base_tensors and all_actions must have the same length")
    ctx = action_set_context_features(web_state, all_actions)
    role_rows = action_role_feature_rows(web_state, all_actions)
    augmented = []
    for base, role in zip(base_tensors, role_rows):
        base = base.float()
        augmented.append(torch.cat([base, role, ctx]).float())
    return augmented
