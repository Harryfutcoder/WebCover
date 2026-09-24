"""Recent action-history features for SubWeb frontier agents.

These are input features only. They do not change rewards, masks, action
ranking, or optimizer behavior. The goal is to expose generic workflow
sequence context, for example input/select followed by submit, to the actor and
critic under the same strict marginal graph objective.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Deque, Dict, Iterable, Tuple

import torch


RECENT_ACTION_BUCKETS = (
    "input",
    "select",
    "submit",
    "form_field",
    "local_workflow",
    "reset_or_nav",
    "error",
    "other_redirect",
    "other_click",
)

_FIELD_RE = re.compile(r"(?:^|\|)([^=|]+)=([^|]*)")


def recent_action_summary_dim() -> int:
    return 2 * len(RECENT_ACTION_BUCKETS) + 2


def _parse_signature(signature: str) -> Dict[str, str]:
    return {match.group(1): match.group(2) for match in _FIELD_RE.finditer(str(signature or ""))}


def action_bucket_from_transition(transition: Dict[str, Any]) -> str:
    exploration = transition.get("exploration") or {}
    signature = str(exploration.get("action_signature") or transition.get("action_signature") or "")
    parsed = _parse_signature(signature)
    family = parsed.get("family", "")
    kind = parsed.get("kind", "") or str(transition.get("action_type") or "")
    target = parsed.get("target", "")
    intent = parsed.get("intent", "")

    if kind.startswith("input"):
        return "input"
    if kind == "select":
        return "select"
    if kind == "submit" or family == "form_submit":
        return "submit"
    if family == "form_field":
        return "form_field"
    if family in {"workflow_link", "main_action"}:
        return "local_workflow"
    if target == "/oups" or "error" in target or intent == "error":
        return "error"
    if family == "boilerplate" or target == "/" or intent in {"home", "back"}:
        return "reset_or_nav"
    if kind in {"link", "redirect"}:
        return "other_redirect"
    return "other_click"


def recent_action_summary(history: Iterable[Tuple[str, bool]]) -> torch.Tensor:
    items = list(history)
    last_bucket = items[-1][0] if items else ""
    last_positive = bool(items[-1][1]) if items else False
    counts = Counter(bucket for bucket, _positive in items)
    positives = sum(1 for _bucket, positive in items if positive)
    denom = float(max(1, len(items)))
    values = []
    for bucket in RECENT_ACTION_BUCKETS:
        values.append(1.0 if bucket == last_bucket else 0.0)
    for bucket in RECENT_ACTION_BUCKETS:
        values.append(float(counts[bucket]) / denom)
    values.append(1.0 if last_positive else 0.0)
    values.append(float(positives) / denom)
    return torch.tensor(values, dtype=torch.float32)


def append_recent_action(
    history: Deque[Tuple[str, bool]],
    transition: Dict[str, Any],
) -> None:
    reward = 0.0
    try:
        reward = float(transition.get("raw_reward", 0.0) or 0.0)
    except Exception:
        reward = 0.0
    history.append((action_bucket_from_transition(transition), reward > 0.0))
