"""
Generate motivation figures and alignment statistics from observation logs.

Paper-aligned metrics:
  - Corr(R, MG): Pearson correlation between normalized reward and true MG
  - HighR-LowMG: reward > P90 and MG < P10 ratio
  - Bootstrap 95% CI for correlation / HighR-LowMG / MAE
  - Coverage efficiency: UniqueStates / TotalSteps * 100
"""

import json
import math
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats as scipy_stats  # type: ignore
except Exception:  # pragma: no cover - scipy may be absent in some environments
    scipy_stats = None


def _find_matching_triplet(log_dir: str) -> Tuple[str, str, str, str]:
    """Find the latest matching step/frequency/q-history log files."""
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(log_dir)

    step_files = [
        f for f in os.listdir(log_dir) if f.startswith("step_log_") and f.endswith(".json")
    ]
    if not step_files:
        raise FileNotFoundError(f"No step_log_*.json in {log_dir}")

    step_files.sort(reverse=True)
    latest = step_files[0]
    suffix = latest[len("step_log_") : -len(".json")]

    step_path = os.path.join(log_dir, latest)
    freq_path = os.path.join(log_dir, f"visit_frequency_{suffix}.json")
    q_path = os.path.join(log_dir, f"q_value_history_{suffix}.json")

    if not os.path.isfile(freq_path):
        raise FileNotFoundError(freq_path)
    if not os.path.isfile(q_path):
        raise FileNotFoundError(q_path)

    return step_path, freq_path, q_path, suffix


def load_logs(log_dir: str, timestamp: Optional[str] = None):
    """Load observation logs for a given timestamp or the latest triplet."""
    if timestamp:
        step_path = os.path.join(log_dir, f"step_log_{timestamp}.json")
        freq_path = os.path.join(log_dir, f"visit_frequency_{timestamp}.json")
        q_path = os.path.join(log_dir, f"q_value_history_{timestamp}.json")
        for p in (step_path, freq_path, q_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
    else:
        step_path, freq_path, q_path, _ = _find_matching_triplet(log_dir)

    with open(step_path, "r", encoding="utf-8") as f:
        step_log = json.load(f)
    with open(freq_path, "r", encoding="utf-8") as f:
        freq_data = json.load(f)
    with open(q_path, "r", encoding="utf-8") as f:
        q_history = json.load(f)

    return step_log, freq_data, q_history


def _normalize_01(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0,1]; return 0.5 when array is constant."""
    if values.size == 0:
        return values
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if v_max <= v_min + 1e-12:
        return np.full_like(values, 0.5, dtype=np.float64)
    return (values - v_min) / (v_max - v_min)


def _pearson_with_p(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Return Pearson r and p-value.
    If scipy is unavailable, compute r and use a normal approximation for p.
    """
    if len(x) < 3 or len(y) < 3:
        return 0.0, float("nan")
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0, float("nan")

    if scipy_stats is not None:
        try:
            r, p = scipy_stats.pearsonr(x, y)
            if np.isnan(r):
                return 0.0, float("nan")
            return float(r), float(p)
        except Exception:
            pass

    # Fallback: numpy correlation + Fisher z normal approximation.
    try:
        r = float(np.corrcoef(x, y)[0, 1])
    except Exception:
        return 0.0, float("nan")
    if np.isnan(r):
        return 0.0, float("nan")

    n = len(x)
    if n <= 3 or abs(r) >= 1.0:
        return max(min(r, 1.0), -1.0), 0.0

    z = 0.5 * math.log((1.0 + r) / (1.0 - r))
    z_score = abs(z) * math.sqrt(max(n - 3, 1))
    # Two-tailed p from normal approximation.
    p = math.erfc(z_score / math.sqrt(2.0))
    return r, p


def _bootstrap_ci(values: np.ndarray, fn, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """Generic bootstrap CI on index-resampled arrays."""
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    outs = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = values[idx]
        v = fn(boot)
        if v is not None and not np.isnan(v):
            outs.append(float(v))
    if not outs:
        return float("nan"), float("nan")
    lo, hi = np.percentile(np.array(outs), [2.5, 97.5])
    return float(lo), float(hi)


def compute_alignment_metrics(
    step_log: List[dict],
    reward_key: str = "actual_reward",
    high_q: float = 90.0,
    low_q: float = 10.0,
    n_boot: int = 1000,
) -> Dict[str, float]:
    """
    Compute paper-style reward/goal alignment metrics on valid steps only.

    Definitions:
      MG = true_marginal_gain
      R  = min-max normalized reward signal
      HighR-LowMG = P(R > P90 and MG < P10)
    """
    valid = [r for r in step_log if r.get("state_type") == "valid" and reward_key in r]
    if len(valid) == 0:
        return {
            "n_valid": 0,
            "corr": 0.0,
            "p_value": float("nan"),
            "corr_ci_low": float("nan"),
            "corr_ci_high": float("nan"),
            "highr_lowmg_ratio": 0.0,
            "highr_lowmg_count": 0,
            "highr_lowmg_ci_low": float("nan"),
            "highr_lowmg_ci_high": float("nan"),
            "mae_norm_reward_mg": float("nan"),
            "mae_ci_low": float("nan"),
            "mae_ci_high": float("nan"),
            "reward_high_threshold": float("nan"),
            "mg_low_threshold": float("nan"),
        }

    mg = np.array([float(r.get("true_marginal_gain", 0.0)) for r in valid], dtype=np.float64)
    reward = np.array([float(r.get(reward_key, 0.0)) for r in valid], dtype=np.float64)
    reward_norm = _normalize_01(reward)

    corr, p_value = _pearson_with_p(reward_norm, mg)

    reward_high_threshold = float(np.percentile(reward_norm, high_q))
    mg_low_threshold = float(np.percentile(mg, low_q))
    highr_lowmg_mask = (reward_norm > reward_high_threshold) & (mg < mg_low_threshold)
    highr_lowmg_ratio = float(np.mean(highr_lowmg_mask))
    highr_lowmg_count = int(np.sum(highr_lowmg_mask))

    mae = float(np.mean(np.abs(reward_norm - mg)))

    # Bootstrap CIs
    pairs = np.column_stack([reward_norm, mg, highr_lowmg_mask.astype(np.float64)])

    def _boot_corr(boot_pairs: np.ndarray) -> float:
        x = boot_pairs[:, 0]
        y = boot_pairs[:, 1]
        r, _ = _pearson_with_p(x, y)
        return r

    def _boot_highr_lowmg(boot_pairs: np.ndarray) -> float:
        return float(np.mean(boot_pairs[:, 2]))

    def _boot_mae(boot_pairs: np.ndarray) -> float:
        return float(np.mean(np.abs(boot_pairs[:, 0] - boot_pairs[:, 1])))

    corr_ci_low, corr_ci_high = _bootstrap_ci(pairs, _boot_corr, n_boot=n_boot)
    hl_ci_low, hl_ci_high = _bootstrap_ci(pairs, _boot_highr_lowmg, n_boot=n_boot)
    mae_ci_low, mae_ci_high = _bootstrap_ci(pairs, _boot_mae, n_boot=n_boot)

    return {
        "n_valid": int(len(valid)),
        "corr": float(corr),
        "p_value": float(p_value),
        "corr_ci_low": float(corr_ci_low),
        "corr_ci_high": float(corr_ci_high),
        "highr_lowmg_ratio": float(highr_lowmg_ratio),
        "highr_lowmg_count": int(highr_lowmg_count),
        "highr_lowmg_ci_low": float(hl_ci_low),
        "highr_lowmg_ci_high": float(hl_ci_high),
        "mae_norm_reward_mg": float(mae),
        "mae_ci_low": float(mae_ci_low),
        "mae_ci_high": float(mae_ci_high),
        "reward_high_threshold": float(reward_high_threshold),
        "mg_low_threshold": float(mg_low_threshold),
    }


def plot_figure1_new_state_discovery_rate(step_log: List[dict], ax, window: int = 50):
    steps = [r["step"] for r in step_log]
    is_new = [1 if r.get("is_new_state", False) else 0 for r in step_log]

    discovery_rate = []
    for i in range(len(is_new)):
        start = max(0, i - window + 1)
        rate = sum(is_new[start : i + 1]) / (i - start + 1)
        discovery_rate.append(rate)

    ax.plot(steps, discovery_rate, color="#E74C3C", linewidth=1.5, label="New state discovery rate")
    ax.fill_between(steps, discovery_rate, alpha=0.15, color="#E74C3C")
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("New State Discovery Rate\n(sliding window)", fontsize=11)
    ax.set_title("(a) New State Discovery Rate Over Time", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    for i, r in enumerate(discovery_rate):
        if r < 0.1 and steps[i] > window:
            ax.axvline(x=steps[i], color="gray", linestyle="--", alpha=0.7)
            ax.text(
                steps[i],
                max(discovery_rate) * 0.85,
                f"Saturation\n@step {steps[i]}",
                fontsize=8,
                color="gray",
                ha="right",
            )
            break


def plot_figure2_visit_frequency_distribution(freq_data: dict, ax):
    counts = list(freq_data.get("state_visit_counts", {}).values())
    if not counts:
        ax.text(0.5, 0.5, "No frequency data", transform=ax.transAxes, ha="center")
        return

    counter = Counter(counts)
    visit_counts = sorted(counter.keys())
    frequencies = [counter[v] for v in visit_counts]
    max_bins = min(20, len(visit_counts))

    ax.bar(
        visit_counts[:max_bins],
        frequencies[:max_bins],
        color="#3498DB",
        alpha=0.8,
        edgecolor="white",
    )
    ax.set_xlabel("Visit Count per State", fontsize=11)
    ax.set_ylabel("Number of States", fontsize=11)
    ax.set_title("(b) State Visit Frequency Distribution", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    total_states = len(counts)
    visited_once = sum(1 for c in counts if c == 1)
    visited_many = sum(1 for c in counts if c >= 10)
    ax.text(
        0.55,
        0.85,
        f"Total states: {total_states}\n"
        f"Visited only once: {visited_once} ({100 * visited_once / max(total_states, 1):.1f}%)\n"
        f"Visited 10+ times: {visited_many} ({100 * visited_many / max(total_states, 1):.1f}%)",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


def _plot_reward_scatter(
    valid: List[dict],
    reward_key: str,
    color: str,
    marker: str,
    alpha: float,
    size: int,
    label_prefix: str,
    ax,
):
    mg = np.array([float(r.get("true_marginal_gain", 0.0)) for r in valid], dtype=np.float64)
    reward = np.array([float(r.get(reward_key, 0.0)) for r in valid], dtype=np.float64)
    reward_norm = _normalize_01(reward)

    m = compute_alignment_metrics(valid, reward_key=reward_key, n_boot=300)
    label = f"{label_prefix} (r={m['corr']:.3f})"
    ax.scatter(mg, reward_norm, c=color, alpha=alpha, s=size, marker=marker, label=label)
    return mg, reward_norm, m


def plot_figure3_reward_vs_marginal_gain(step_log: List[dict], ax):
    """
    Plot reward-vs-MG with paper-style HighR-LowMG coloring:
      high reward threshold = P90(R_norm)
      low MG threshold      = P10(MG)
    """
    from matplotlib.patches import Patch

    valid = [r for r in step_log if r.get("state_type") == "valid"]
    if not valid:
        ax.text(0.5, 0.5, "No valid state records", transform=ax.transAxes, ha="center")
        return

    mg, reward_norm, main_m = _plot_reward_scatter(
        valid=valid,
        reward_key="actual_reward",
        color="#808B96",
        marker="o",
        alpha=0.20,
        size=10,
        label_prefix="Actual reward",
        ax=ax,
    )

    high_thr = main_m["reward_high_threshold"]
    low_thr = main_m["mg_low_threshold"]
    highr_lowmg = (reward_norm > high_thr) & (mg < low_thr)
    lowr_highmg = (reward_norm < float(np.percentile(reward_norm, 10))) & (
        mg > float(np.percentile(mg, 90))
    )
    aligned = ~(highr_lowmg | lowr_highmg)

    ax.scatter(mg[aligned], reward_norm[aligned], c="#95A5A6", alpha=0.35, s=10)
    ax.scatter(mg[highr_lowmg], reward_norm[highr_lowmg], c="#E74C3C", alpha=0.55, s=12)
    ax.scatter(mg[lowr_highmg], reward_norm[lowr_highmg], c="#2ECC71", alpha=0.55, s=12)

    has_marg = "marg_style_reward" in valid[0]
    if has_marg:
        _plot_reward_scatter(
            valid=valid,
            reward_key="marg_style_reward",
            color="#3498DB",
            marker="^",
            alpha=0.25,
            size=10,
            label_prefix="1/Count(a) proxy",
            ax=ax,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1, label="y=x alignment")

    legend_elements = [
        Patch(facecolor="#E74C3C", alpha=0.7, label=f"HighR-LowMG ({main_m['highr_lowmg_ratio']:.1%})"),
        Patch(facecolor="#2ECC71", alpha=0.7, label="LowR-HighMG"),
        Patch(facecolor="#95A5A6", alpha=0.7, label="Aligned"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")
    ax.set_xlabel("True Marginal Gain Δ(s|H)", fontsize=11)
    ax.set_ylabel("Reward Signal (normalized)", fontsize=11)
    ax.set_title(
        (
            "(c) Reward vs. True Marginal Gain\n"
            f"Corr={main_m['corr']:.3f}, p={main_m['p_value']:.2e}, "
            f"HighR-LowMG={main_m['highr_lowmg_ratio']:.1%}"
        ),
        fontsize=10.5,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)


def plot_figure4_q_value_history_for_revisited_states(q_history: dict, ax):
    sorted_states = sorted(q_history.items(), key=lambda x: len(x[1]), reverse=True)
    top_states = sorted_states[: min(5, len(sorted_states))]

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
    has_data = False
    for i, (state_id, history) in enumerate(top_states):
        if len(history) < 3:
            continue
        has_data = True
        visit_indices = list(range(1, len(history) + 1))
        q_vals = [h[1] for h in history]
        short_id = state_id[:40] + "..." if len(state_id) > 40 else state_id
        ax.plot(
            visit_indices,
            q_vals,
            marker="o",
            markersize=4,
            color=colors[i % len(colors)],
            linewidth=1.5,
            label=f"State {i + 1}: {short_id}",
            alpha=0.8,
        )

    if not has_data:
        ax.text(
            0.5,
            0.5,
            "No states visited 3+ times\n(try longer run)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        return

    all_q = [h[1] for hist in [v for _, v in top_states] for h in hist]
    peak_q = max(all_q) if all_q else 0
    if peak_q > 100:
        ax.annotate(
            f"Peak Q >= {peak_q:,.0f}",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            color="#E74C3C",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", alpha=0.8),
        )

    ax.set_xlabel("Visit Count (nth visit to this state)", fontsize=11)
    ax.set_ylabel("Q Value", fontsize=11)
    ax.set_title(
        "(d) Q Value Divergence Across Re-visits\n"
        "(Bellman TD instability under non-stationary reward)",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)


def generate_motivation_figures(log_dir: str, output_path: str = "motivation_figures.pdf"):
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    step_log, freq_data, q_history = load_logs(log_dir)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_figure1_new_state_discovery_rate(step_log, ax1)
    plot_figure2_visit_frequency_distribution(freq_data, ax2)
    plot_figure3_reward_vs_marginal_gain(step_log, ax3)
    plot_figure4_q_value_history_for_revisited_states(q_history, ax4)

    fig.suptitle(
        "Motivation: Reward-Objective Misalignment in RL Web Testing",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"[Analyzer] Motivation figures saved to {output_path}")
    plt.close(fig)


def print_summary_statistics(log_dir: str):
    step_log, freq_data, _q_history = load_logs(log_dir)

    counts = list(freq_data.get("state_visit_counts", {}).values())
    total_unique_states = len(counts)
    total_steps = len(step_log)
    valid_steps = [r for r in step_log if r.get("state_type") == "valid"]

    m_actual = compute_alignment_metrics(step_log, reward_key="actual_reward")
    has_marg = len(valid_steps) > 0 and "marg_style_reward" in valid_steps[0]
    m_marg = (
        compute_alignment_metrics(step_log, reward_key="marg_style_reward")
        if has_marg
        else None
    )

    states_per_100 = (100.0 * total_unique_states / total_steps) if total_steps > 0 else 0.0

    print("\n" + "=" * 78)
    print("MOTIVATION EXPERIMENT SUMMARY (PAPER-ALIGNED)")
    print("=" * 78)
    print(f"Total steps:                           {total_steps}")
    print(f"Valid steps (state_type=valid):       {len(valid_steps)}")
    print(f"Total unique states (IDs):            {total_unique_states}")
    print(f"Coverage efficiency (states/100step): {states_per_100:.2f}")
    print(
        f"States visited only once:             {sum(1 for c in counts if c == 1)} "
        f"({100 * sum(1 for c in counts if c == 1) / max(total_unique_states, 1):.1f}%)"
    )
    print(
        f"States visited 10+ times:             {sum(1 for c in counts if c >= 10)} "
        f"({100 * sum(1 for c in counts if c >= 10) / max(total_unique_states, 1):.1f}%)"
    )
    print(f"Max visit count (single state):       {max(counts) if counts else 0}")
    print("-" * 78)
    print("Actual Reward vs MG")
    print(
        f"  Corr(R,MG):                         {m_actual['corr']:.3f} "
        f"(p={m_actual['p_value']:.2e})"
    )
    print(
        f"  Corr 95% CI:                        [{m_actual['corr_ci_low']:.3f}, "
        f"{m_actual['corr_ci_high']:.3f}]"
    )
    print(
        f"  HighR-LowMG (R>P90 & MG<P10):      {m_actual['highr_lowmg_count']} / "
        f"{m_actual['n_valid']} ({100 * m_actual['highr_lowmg_ratio']:.2f}%)"
    )
    print(
        f"  HighR-LowMG 95% CI:                 [{100 * m_actual['highr_lowmg_ci_low']:.2f}%, "
        f"{100 * m_actual['highr_lowmg_ci_high']:.2f}%]"
    )
    print(
        f"  MAE(|R_norm-MG|):                   {m_actual['mae_norm_reward_mg']:.4f} "
        f"(95% CI [{m_actual['mae_ci_low']:.4f}, {m_actual['mae_ci_high']:.4f}])"
    )

    if m_marg is not None:
        print("-" * 78)
        print("1/Count(a) Proxy Reward vs MG")
        print(
            f"  Corr(R,MG):                         {m_marg['corr']:.3f} "
            f"(p={m_marg['p_value']:.2e})"
        )
        print(
            f"  Corr 95% CI:                        [{m_marg['corr_ci_low']:.3f}, "
            f"{m_marg['corr_ci_high']:.3f}]"
        )
        print(
            f"  HighR-LowMG (R>P90 & MG<P10):      {m_marg['highr_lowmg_count']} / "
            f"{m_marg['n_valid']} ({100 * m_marg['highr_lowmg_ratio']:.2f}%)"
        )

    print("=" * 78)
