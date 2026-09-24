"""
Unified run summarizer for webtest_output/result.

Goals:
  1) Use one consistent metric source for all baselines (output_data/newest.json).
  2) Keep optional motivation-summary fields from root logs for reference.
  3) Report per-run and aggregated metrics with the same denominator.

Usage:
  python summarize_runs.py
  python summarize_runs.py --site github
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Optional, Tuple


RESULT_BASE = os.path.join("webtest_output", "result")

SITE_ORDER = [
    # Open-source subjects first, then commercial/hosted subjects.
    "4gaboards",
    "agilefant",
    "gadael",
    "petclinic",
    "realworld",
    "splittypie",
    "timeoff",
    "github",
    "nextcloud",
    "odoo",
]
SITE_RANK = {site: i for i, site in enumerate(SITE_ORDER)}

KNOWN_BASELINES = [
    "subweb-frontier-a2c",
    "webrled-official",
    "webqt",
    "webexplor",
    "qexplore",
    "random",
]


def _read_text_auto(path: str) -> str:
    for enc in ("utf-16", "utf-8", "utf-8-sig", "gbk"):
        try:
            return open(path, "r", encoding=enc, errors="replace").read()
        except Exception:
            continue
    return open(path, "r", encoding="latin-1", errors="replace").read()


def _normalise_message(msg: str) -> str:
    msg = re.sub(r"https?://\S+", "<URL>", msg)
    msg = re.sub(r"\d+:\d+", "<LOC>", msg)
    msg = re.sub(r"0x[0-9a-fA-F]+", "<HEX>", msg)
    msg = re.sub(r"\s+", " ", msg).strip()
    return msg


def _fault_counts(bug_log_path: str) -> Tuple[int, int, int]:
    if not os.path.isfile(bug_log_path):
        return 0, 0, 0
    severe = Counter()
    warning = Counter()
    with open(bug_log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            lvl = entry.get("level", "")
            msg = entry.get("message", "")
            if lvl not in ("SEVERE", "WARNING") or not msg:
                continue
            msg = _normalise_message(msg)
            if lvl == "SEVERE":
                severe[msg] += 1
            else:
                warning[msg] += 1
    return len(severe), len(warning), len(severe) + len(warning)


def _parse_run_name(run_name: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse run directory name:
      <site>-<profile>-<baseline>-<seed>
    where baseline may contain hyphens.
    """
    m = re.search(r"-(seed\d+)$", run_name)
    if not m:
        return None
    seed = m.group(1)
    prefix = run_name[: -len(seed) - 1]

    for baseline in sorted(KNOWN_BASELINES, key=len, reverse=True):
        suffix = f"-{baseline}"
        if not prefix.endswith(suffix):
            continue
        site_and_profile = prefix[: -len(suffix)]
        site, sep, profile = site_and_profile.partition("-")
        if not sep or not site or not profile:
            continue
        return site, profile, baseline, seed
    return None


def _parse_summary_from_root_log(profile: str, baseline: str, seed: str) -> Tuple[Optional[int], Optional[int]]:
    root_log = f"{profile}_{baseline}_{seed}.log"
    if not os.path.isfile(root_log):
        return None, None
    text = _read_text_auto(root_log)
    m_steps = re.search(r"Total steps:\s+(\d+)", text)
    m_states = re.search(r"Total unique states \(IDs\):\s+(\d+)", text)
    steps = int(m_steps.group(1)) if m_steps else None
    states = int(m_states.group(1)) if m_states else None
    return steps, states


def _is_valid_state_entry(state_entry) -> bool:
    if not isinstance(state_entry, dict):
        return False
    info = str(state_entry.get("info", ""))
    return info.startswith("ActionSetWithExecutionTimesState(")


def _load_newest_metrics(run_dir: str, step_cutoff: Optional[int] = None) -> Dict[str, Optional[float]]:
    newest = os.path.join(RESULT_BASE, run_dir, "output_data", "newest.json")
    if not os.path.isfile(newest):
        return {
            "transitions": None,
            "transitions_used": None,
            "state_count": None,
            "valid_state_count": None,
            "state_count_cutoff": None,
            "valid_state_count_cutoff": None,
            "special_state_count": None,
            "url_keys": None,
            "total_url_visits": None,
            "auth_url_visits": None,
            "auth_ratio": None,
        }

    data = json.load(open(newest, "r", encoding="utf-8"))
    transition_list = data.get("transition_list", [])
    state_list = data.get("state_list", [])
    url_count = data.get("url_count", {}) or {}

    valid_state_count = 0
    for s in state_list:
        if _is_valid_state_entry(s):
            valid_state_count += 1

    transitions_total = len(transition_list)
    transitions_used = transitions_total
    if step_cutoff is not None and step_cutoff > 0:
        transitions_used = min(transitions_total, int(step_cutoff))

    state_indices_cutoff = set()
    for t in transition_list[:transitions_used]:
        if not isinstance(t, (list, tuple)) or len(t) < 3:
            continue
        prev_idx, _act_idx, new_idx = t
        for idx in (prev_idx, new_idx):
            if isinstance(idx, int) and 0 <= idx < len(state_list):
                state_indices_cutoff.add(idx)

    state_count_cutoff = float(len(state_indices_cutoff))
    valid_state_count_cutoff = float(
        sum(1 for idx in state_indices_cutoff if _is_valid_state_entry(state_list[idx]))
    )

    total_url_visits = sum(url_count.values())
    auth_url_visits = sum(
        v for k, v in url_count.items()
        if "/login" in k or "/signup" in k or "/password_reset" in k
    )
    auth_ratio = (auth_url_visits / total_url_visits) if total_url_visits else 0.0

    return {
        "transitions": float(transitions_total),
        "transitions_used": float(transitions_used),
        "state_count": float(len(state_list)),
        "valid_state_count": float(valid_state_count),
        "state_count_cutoff": state_count_cutoff,
        "valid_state_count_cutoff": valid_state_count_cutoff,
        "special_state_count": float(len(state_list) - valid_state_count),
        "url_keys": float(len(url_count)),
        "total_url_visits": float(total_url_visits),
        "auth_url_visits": float(auth_url_visits),
        "auth_ratio": float(auth_ratio),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="", help="Optional site filter, e.g., github")
    parser.add_argument(
        "--step-cutoff",
        type=int,
        default=0,
        help="Optional fixed transition cutoff K for fair per-run coverage comparison.",
    )
    parser.add_argument(
        "--align-min-transitions",
        action="store_true",
        help="Use min(transitions) across selected runs as cutoff K.",
    )
    args = parser.parse_args()
    site_filter = args.site.strip().lower()
    requested_cutoff = args.step_cutoff if args.step_cutoff and args.step_cutoff > 0 else None

    if not os.path.isdir(RESULT_BASE):
        raise SystemExit(f"Missing {RESULT_BASE!r}")

    rows = []
    for name in sorted(os.listdir(RESULT_BASE)):
        run_path = os.path.join(RESULT_BASE, name)
        if not os.path.isdir(run_path):
            continue
        parsed = _parse_run_name(name)
        if not parsed:
            continue
        site, profile, baseline, seed = parsed
        if site_filter and site != site_filter:
            continue

        newest = _load_newest_metrics(name, step_cutoff=None)
        bug_log = os.path.join(run_path, "bug.log")
        sev_u, warn_u, fault_u = _fault_counts(bug_log)
        summary_steps, summary_unique_states = _parse_summary_from_root_log(profile, baseline, seed)

        rows.append({
            "run": name,
            "site": site,
            "profile": profile,
            "baseline": baseline,
            "seed": seed,
            "summary_steps": summary_steps,
            "summary_unique_states": summary_unique_states,
            "uniq_severe": sev_u,
            "uniq_warning": warn_u,
            "uniq_faults": fault_u,
            **newest,
        })

    effective_cutoff = requested_cutoff
    if args.align_min_transitions:
        transition_candidates = [
            int(r["transitions"]) for r in rows if r.get("transitions") is not None
        ]
        if transition_candidates:
            min_cutoff = min(transition_candidates)
            effective_cutoff = min_cutoff if effective_cutoff is None else min(effective_cutoff, min_cutoff)

    if effective_cutoff is not None:
        for r in rows:
            refreshed = _load_newest_metrics(r["run"], step_cutoff=effective_cutoff)
            for k, v in refreshed.items():
                r[k] = v

    if not rows:
        raise SystemExit("No matching run directories found.")

    header = (
        f"{'Run':<58} {'Base':<14} {'Seed':<6} {'Trans':>7} {'ValidSt':>8} "
        f"{'URLs':>6} {'Auth%':>7} {'Faults':>7} {'SumSt':>7} {'SumUniq':>8}"
    )
    if effective_cutoff is not None:
        header += f" {'K':>6} {'Trans@K':>8} {'Valid@K':>8}"
    print(header)
    print("-" * 142)
    for r in rows:
        trans = "NA" if r["transitions"] is None else f"{int(r['transitions'])}"
        valid_states = "NA" if r["valid_state_count"] is None else f"{int(r['valid_state_count'])}"
        url_keys = "NA" if r["url_keys"] is None else f"{int(r['url_keys'])}"
        auth = "NA" if r["auth_ratio"] is None else f"{r['auth_ratio'] * 100:.1f}"
        sum_steps = "NA" if r["summary_steps"] is None else str(r["summary_steps"])
        sum_uniq = "NA" if r["summary_unique_states"] is None else str(r["summary_unique_states"])
        line = (
            f"{r['run']:<58} {r['baseline']:<14} {r['seed']:<6} {trans:>7} {valid_states:>8} "
            f"{url_keys:>6} {auth:>7} {r['uniq_faults']:>7} {sum_steps:>7} {sum_uniq:>8}"
        )
        if effective_cutoff is not None:
            trans_used = "NA" if r.get("transitions_used") is None else f"{int(r['transitions_used'])}"
            valid_k = "NA" if r.get("valid_state_count_cutoff") is None else f"{int(r['valid_state_count_cutoff'])}"
            line += f" {int(effective_cutoff):>6} {trans_used:>8} {valid_k:>8}"
        print(line)

    agg: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in rows:
        key = (r["site"], r["baseline"])
        agg[key]["n"] += 1
        for metric in (
            "transitions",
            "valid_state_count",
            "transitions_used",
            "valid_state_count_cutoff",
            "state_count",
            "state_count_cutoff",
            "url_keys",
            "auth_ratio",
            "uniq_faults",
            "summary_steps",
            "summary_unique_states",
        ):
            v = r.get(metric)
            if v is not None:
                agg[key][metric] += float(v)
                agg[key][f"{metric}_n"] += 1

    print("\nAggregated (mean on available runs):")
    agg_header = (
        f"{'Site':<12} {'Baseline':<14} {'n':>3} {'Trans':>9} {'ValidSt':>9} "
        f"{'States':>9} {'URLs':>8} {'Auth%':>7} {'Faults':>8} {'SumSt':>8} {'SumUniq':>8}"
    )
    if effective_cutoff is not None:
        agg_header += f" {'K':>6} {'Trans@K':>9} {'Valid@K':>9}"
    print(agg_header)
    print("-" * 140)
    def _agg_sort_key(key: Tuple[str, str]) -> Tuple[int, str, str]:
        site, baseline = key
        return SITE_RANK.get(site, len(SITE_RANK)), site, baseline

    for (site, baseline) in sorted(agg.keys(), key=_agg_sort_key):
        a = agg[(site, baseline)]

        def avg(metric: str) -> float:
            denom = a.get(f"{metric}_n", 0.0)
            return (a.get(metric, 0.0) / denom) if denom else 0.0

        line = (
            f"{site:<12} {baseline:<14} {int(a['n']):>3} "
            f"{avg('transitions'):>9.1f} {avg('valid_state_count'):>9.1f} {avg('state_count'):>9.1f} "
            f"{avg('url_keys'):>8.1f} {avg('auth_ratio') * 100:>7.1f} {avg('uniq_faults'):>8.1f} "
            f"{avg('summary_steps'):>8.1f} {avg('summary_unique_states'):>8.1f}"
        )
        if effective_cutoff is not None:
            line += (
                f" {int(effective_cutoff):>6} "
                f"{avg('transitions_used'):>9.1f} {avg('valid_state_count_cutoff'):>9.1f}"
            )
        print(line)


if __name__ == "__main__":
    main()
