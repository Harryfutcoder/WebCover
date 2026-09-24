"""Learning-signal diagnostics for SubWeb graph-coverage agents.

These helpers are observability only. They do not change rewards, losses, or
optimizer behavior; they make it easier to see whether hard graph marginal
coverage rewards are producing usable return/advantage gradients.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

import torch


def _flat(tensor: Optional[torch.Tensor]) -> torch.Tensor:
    if tensor is None:
        return torch.zeros(0, dtype=torch.float32)
    return tensor.detach().float().view(-1).cpu()


def _std(values: torch.Tensor) -> float:
    if values.numel() <= 1:
        return 0.0
    return float(values.std(unbiased=False).item())


def _value_hist(values: torch.Tensor, limit: int = 8) -> str:
    if values.numel() == 0:
        return "none"
    counter = Counter(round(float(item), 4) for item in values.tolist())
    items = sorted(counter.items(), key=lambda pair: (pair[0], pair[1]))
    shown = items[:limit]
    text = ",".join(f"{value:g}:{count}" for value, count in shown)
    if len(items) > limit:
        text += ",..."
    return text


def _sign_counts(values: torch.Tensor, eps: float = 1e-8) -> tuple[int, int, int]:
    if values.numel() == 0:
        return 0, 0, 0
    pos = int((values > eps).sum().item())
    neg = int((values < -eps).sum().item())
    zero = int((values.abs() <= eps).sum().item())
    return pos, neg, zero


def _abs_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.abs().mean().item())


def parameter_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Return total L2 norm of existing gradients without mutating them."""

    total_sq = 0.0
    for param in parameters:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        grad_norm = float(grad.detach().float().norm(2).item())
        total_sq += grad_norm * grad_norm
    return total_sq ** 0.5


def format_learning_signal_diagnostics(
    rewards: torch.Tensor,
    returns: torch.Tensor,
    raw_advantages: Optional[torch.Tensor] = None,
    policy_advantages: Optional[torch.Tensor] = None,
    grad_norm: Optional[float] = None,
    max_grad_norm: Optional[float] = None,
    actor_grad_norm: Optional[float] = None,
    critic_grad_norm: Optional[float] = None,
) -> str:
    """Return compact one-line diagnostics for an update log."""

    rewards_f = _flat(rewards)
    returns_f = _flat(returns)
    raw_adv_f = _flat(raw_advantages)
    policy_adv_f = _flat(policy_advantages)

    reward_pos, _, reward_zero = _sign_counts(rewards_f)
    return_pos, return_neg, return_zero = _sign_counts(returns_f)
    raw_pos, raw_neg, raw_zero = _sign_counts(raw_adv_f)
    policy_pos, policy_neg, policy_zero = _sign_counts(policy_adv_f)

    parts = [
        f"reward_min={float(rewards_f.min().item()) if rewards_f.numel() else 0.0:.4f}",
        f"reward_max={float(rewards_f.max().item()) if rewards_f.numel() else 0.0:.4f}",
        f"reward_mean={float(rewards_f.mean().item()) if rewards_f.numel() else 0.0:.4f}",
        f"reward_std={_std(rewards_f):.4f}",
        f"reward_positive_count={reward_pos}",
        f"reward_zero_count={reward_zero}",
        f"reward_values={_value_hist(rewards_f)}",
        f"return_min={float(returns_f.min().item()) if returns_f.numel() else 0.0:.4f}",
        f"return_max={float(returns_f.max().item()) if returns_f.numel() else 0.0:.4f}",
        f"return_positive_count={return_pos}",
        f"return_negative_count={return_neg}",
        f"return_zero_count={return_zero}",
        f"raw_adv_abs_mean={_abs_mean(raw_adv_f):.4f}",
        f"raw_adv_pos_count={raw_pos}",
        f"raw_adv_neg_count={raw_neg}",
        f"raw_adv_zero_count={raw_zero}",
        f"policy_signal_abs_mean={_abs_mean(policy_adv_f):.4f}",
        f"policy_signal_pos_count={policy_pos}",
        f"policy_signal_neg_count={policy_neg}",
        f"policy_signal_zero_count={policy_zero}",
    ]
    if grad_norm is not None:
        grad = float(grad_norm)
        parts.append(f"grad_norm_preclip={grad:.4f}")
        if max_grad_norm is not None:
            max_norm = float(max_grad_norm)
            parts.append(f"grad_clip_active={1 if grad > max_norm else 0}")
    if actor_grad_norm is not None:
        actor_grad = float(actor_grad_norm)
        parts.append(f"actor_grad_norm_preclip={actor_grad:.4f}")
        if max_grad_norm is not None:
            parts.append(f"actor_grad_clip_active={1 if actor_grad > float(max_grad_norm) else 0}")
    if critic_grad_norm is not None:
        critic_grad = float(critic_grad_norm)
        parts.append(f"critic_grad_norm_preclip={critic_grad:.4f}")
        if max_grad_norm is not None:
            parts.append(f"critic_grad_clip_active={1 if critic_grad > float(max_grad_norm) else 0}")
        if actor_grad_norm is not None:
            denom = max(abs(float(actor_grad_norm)), 1e-8)
            parts.append(f"critic_actor_grad_ratio={float(critic_grad_norm) / denom:.4f}")
    return " ".join(parts)
