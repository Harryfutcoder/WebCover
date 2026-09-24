"""Policy-distribution diagnostics for SubWeb frontier agents.

This module observes masked action distributions. It does not change rewards,
sampling, action detection, or optimizer behavior.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import torch

from action.impl.click_action import ClickAction
from action.impl.random_input_action import RandomInputAction
from action.impl.random_select_action import RandomSelectAction
from agent.impl.subweb_action_context_features import action_role_features
from agent.impl.subweb_padded_actions import PaddedActionBatch, action_kind, action_signature
from agent.impl.subweb_structural_action_features import structural_action_feature_names, structural_action_features
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState


_MASS_KEYS = (
    "global_nav",
    "form",
    "local_workflow",
    "error",
    "redirect",
    "input",
    "submit",
    "select",
    "click",
    "other",
)


def _read_bool_env(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "no", "off"}


def _route_for_state(web_state) -> str:
    raw_url = getattr(web_state, "raw_url", getattr(web_state, "url", ""))
    return ActionSetWithExecutionTimesState._normalize_route_identity(raw_url)


def _structural_flag(features: torch.Tensor, names, key: str) -> bool:
    try:
        return bool(features[names.index(key)].item() > 0.5)
    except Exception:
        return False


def action_diagnostic_tags(
    action,
    web_state,
    all_actions,
    role: Optional[torch.Tensor] = None,
    structural: Optional[torch.Tensor] = None,
) -> List[str]:
    tags = set()
    kind = action_kind(action)
    if role is None:
        role = action_role_features(action, web_state, list(all_actions))
    if structural is None:
        structural = structural_action_features(action, web_state, list(all_actions))
    structural_names = structural_action_feature_names()

    if role[7].item() > 0.5 or role[8].item() > 0.5 or role[9].item() > 0.5:
        tags.add("global_nav")
    if role[6].item() > 0.5:
        tags.add("form")
    if _structural_flag(structural, structural_names, "local_non_nav_route_action"):
        tags.add("local_workflow")
    if _structural_flag(structural, structural_names, "error_target_or_text"):
        tags.add("error")
    if kind == "redirect":
        tags.add("redirect")
    if kind == "input":
        tags.add("input")
    if kind == "submit":
        tags.add("submit")
    if kind == "select":
        tags.add("select")
    if kind in {"click", "default"} and not tags:
        tags.add("click")
    if not tags:
        tags.add("other")
    return sorted(tags)


def primary_action_category(
    action,
    web_state,
    all_actions,
    role: Optional[torch.Tensor] = None,
    structural: Optional[torch.Tensor] = None,
) -> str:
    tags = set(action_diagnostic_tags(action, web_state, all_actions, role=role, structural=structural))
    for key in ("error", "input", "select", "submit", "form", "global_nav", "local_workflow", "redirect", "click"):
        if key in tags:
            return key
    return "other"


def summarize_policy_distribution(
    *,
    web_state,
    batch: PaddedActionBatch,
    probs: torch.Tensor,
    chosen_idx: int,
    top_k: int = 5,
) -> Dict[str, object]:
    probs_cpu = probs.detach().float().cpu()
    valid_count = int(batch.action_mask.sum().item())
    actions = list(batch.actions_policy)
    all_actions = list(batch.actions_full)
    role_rows = getattr(batch, "role_features_policy", None) or []
    structural_rows = getattr(batch, "structural_features_policy", None) or []
    mass = {key: 0.0 for key in _MASS_KEYS}
    counts = {key: 0 for key in _MASS_KEYS}
    primary_mass = {key: 0.0 for key in _MASS_KEYS}
    primary_counts = {key: 0 for key in _MASS_KEYS}
    action_rows = []

    for idx, action in enumerate(actions):
        if idx >= len(probs_cpu) or not bool(batch.action_mask[idx].item()):
            continue
        prob = float(probs_cpu[idx].item())
        role = role_rows[idx] if idx < len(role_rows) else None
        structural = structural_rows[idx] if idx < len(structural_rows) else None
        tags = action_diagnostic_tags(
            action,
            web_state,
            all_actions,
            role=role,
            structural=structural,
        )
        primary = primary_action_category(
            action,
            web_state,
            all_actions,
            role=role,
            structural=structural,
        )
        for tag in tags:
            if tag in mass:
                mass[tag] += prob
                counts[tag] += 1
        if primary in primary_mass:
            primary_mass[primary] += prob
            primary_counts[primary] += 1
        action_rows.append({
            "idx": idx,
            "prob": prob,
            "kind": action_kind(action),
            "primary": primary,
            "tags": tags,
            "signature": action_signature(action),
            "text": str(getattr(action, "text", "") or "")[:120],
            "target": str(getattr(action, "addition_info", "") or "")[:160] if isinstance(action, ClickAction) else "",
        })

    top_actions = sorted(action_rows, key=lambda row: row["prob"], reverse=True)[:top_k]
    chosen_row = next((row for row in action_rows if row["idx"] == int(chosen_idx)), None)

    return {
        "route": _route_for_state(web_state),
        "action_count_full": int(batch.original_action_count),
        "action_count_policy": int(batch.kept_action_count),
        "mask_valid": valid_count,
        "truncated_count": int(batch.truncated_count),
        "truncated_by_type": dict(batch.truncated_by_type),
        "category_mass": {key: round(float(value), 6) for key, value in mass.items()},
        "category_counts": counts,
        "primary_mass": {key: round(float(value), 6) for key, value in primary_mass.items()},
        "primary_counts": primary_counts,
        "chosen": chosen_row,
        "top_actions": top_actions,
    }


def format_policy_brief(summary: Dict[str, object]) -> str:
    mass = summary.get("category_mass", {}) or {}
    counts = summary.get("category_counts", {}) or {}
    primary_mass = summary.get("primary_mass", {}) or {}
    chosen = summary.get("chosen", {}) or {}
    top = summary.get("top_actions", []) or []
    top_bits = []
    for row in top[:3]:
        sig = str(row.get("signature", ""))
        sig = sig[:48] + "..." if len(sig) > 48 else sig
        top_bits.append(f"{row.get('prob', 0.0):.3f}:{row.get('primary')}:{sig}")
    return (
        "primary_mass="
        f"nav:{primary_mass.get('global_nav', 0.0):.3f},"
        f"form:{primary_mass.get('form', 0.0):.3f},"
        f"local:{primary_mass.get('local_workflow', 0.0):.3f},"
        f"err:{primary_mass.get('error', 0.0):.3f},"
        f"redir:{primary_mass.get('redirect', 0.0):.3f} "
        "tag_mass="
        f"nav:{mass.get('global_nav', 0.0):.3f},"
        f"form:{mass.get('form', 0.0):.3f},"
        f"local:{mass.get('local_workflow', 0.0):.3f},"
        f"err:{mass.get('error', 0.0):.3f},"
        f"redir:{mass.get('redirect', 0.0):.3f} "
        "counts="
        f"nav:{counts.get('global_nav', 0)},"
        f"form:{counts.get('form', 0)},"
        f"local:{counts.get('local_workflow', 0)},"
        f"err:{counts.get('error', 0)} "
        f"chosen={chosen.get('primary', 'none')} top=[{'; '.join(top_bits)}]"
    )


class SubWebPolicyDiagnostics:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.enabled = _read_bool_env("WEBTEST_SUBWEB_POLICY_DIAGNOSTICS", True)
        self.console_enabled = _read_bool_env("WEBTEST_SUBWEB_POLICY_DIAGNOSTICS_CONSOLE", True)
        self.path: Optional[str] = None
        if self.enabled:
            os.makedirs("observation_logs", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = agent_name.lower().replace(" ", "_").replace("-", "_")
            self.path = os.path.join("observation_logs", f"policy_diagnostics_{safe_name}_{timestamp}.jsonl")

    def record_step(
        self,
        *,
        step: int,
        web_state,
        batch: PaddedActionBatch,
        probs: torch.Tensor,
        chosen_idx: int,
        entropy: float,
        max_prob: float,
        finalized_transition: Optional[Dict[str, object]],
        F: int,
        zero_gain_streak: int,
        value: Optional[float] = None,
        planner_score_parts: Optional[Dict[str, object]] = None,
        extra_diagnostics: Optional[Dict[str, object]] = None,
    ) -> Optional[Dict[str, object]]:
        if not self.enabled:
            return None
        summary = summarize_policy_distribution(
            web_state=web_state,
            batch=batch,
            probs=probs,
            chosen_idx=chosen_idx,
        )
        transition = finalized_transition or {}
        reward_t = transition.get("realized_delta", transition.get("raw_reward", None))
        record = {
            "agent": self.agent_name,
            "step": int(step),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "entropy": float(entropy),
            "max_prob": float(max_prob),
            "value": None if value is None else float(value),
            "F": float(F),
            "reward_t": None if reward_t is None else float(reward_t),
            "raw_reward_t": None if transition.get("raw_reward", None) is None else float(transition.get("raw_reward")),
            "realized_delta": None if transition.get("realized_delta", None) is None else float(transition.get("realized_delta")),
            "zero_gain_streak": int(zero_gain_streak),
            "prev_transition": transition,
            "planner_score_parts": planner_score_parts or {},
            **summary,
        }
        if extra_diagnostics:
            record.update(extra_diagnostics)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record
