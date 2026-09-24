"""
Diagnose environment-side exploration noise from root logs + newest.json.

Focus:
  - failure exception types
  - failed action categories / top failed actions
  - restart concentration
  - state concentration (max visited state, top visited states)
  - per-URL state variants (same URL split into multiple action-set states)

Usage:
  python diagnose_site_environment.py --sites 4gaboards
  python diagnose_site_environment.py --sites 4gaboards,odoo --baselines subweb-frontier-a2c,webexplor
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = pathlib.Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "webtest_output" / "result"

KNOWN_BASELINES = [
    "subweb-frontier-a2c",
    "webrled-official",
    "webqt",
    "webexplor",
    "qexplore",
    "random",
]

TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} ")
SEED_RE = re.compile(r"_(seed\d+)$")
PAT_CHOSEN = re.compile(r"Chosen action: (.+)$")
PAT_FAIL = re.compile(r"Execute action failed \(([^)]+)\)")
PAT_OOD = re.compile(r"Out of domain")
PAT_SAME = re.compile(r"Same url too many times")
PAT_RESTART = re.compile(r"Chosen action: RestartAction")
PAT_TOTAL_STEPS = re.compile(r"Total steps:\s+(\d+)")
PAT_UNIQUE = re.compile(r"Total unique states \(IDs\):\s+(\d+)")
PAT_COVERAGE = re.compile(r"Coverage efficiency \(states/100step\):\s+([0-9.]+)")
PAT_ONCE = re.compile(r"States visited only once:\s+(\d+) \(([0-9.]+)%\)")
PAT_GE10 = re.compile(r"States visited 10\+ times:\s+(\d+) \(([0-9.]+)%\)")
PAT_MAXV = re.compile(r"Max visit count \(single state\):\s+(\d+)")
PAT_VALID_URL = re.compile(r"url=([^,)]+)")


@dataclass
class RunDiagnostics:
    site: str
    profile: str
    baseline: str
    seed: str
    log_path: pathlib.Path
    result_dir: pathlib.Path
    total_steps: Optional[int]
    total_unique_states: Optional[int]
    coverage_efficiency: Optional[float]
    only_once_pct: Optional[float]
    ge10_pct: Optional[float]
    max_visit_count: Optional[int]
    out_of_domain_count: int
    same_url_count: int
    restart_count: int
    fail_count: int
    exception_types: Counter
    failed_action_types: Counter
    top_failed_actions: List[Tuple[str, int]]
    state_variant_count: Optional[int]
    unique_url_count: Optional[int]
    top_url_variants: List[Tuple[str, int, int]]
    top_visited_states: List[Tuple[str, int]]


def parse_root_log_name(stem: str) -> Optional[Tuple[str, str, str, str]]:
    m = SEED_RE.search(stem)
    if not m:
        return None
    seed = m.group(1)
    prefix = stem[: m.start()]
    for baseline in sorted(KNOWN_BASELINES, key=len, reverse=True):
        suffix = "_" + baseline
        if not prefix.endswith(suffix):
            continue
        profile = prefix[: -len(suffix)]
        site, sep, _ = profile.partition("-")
        if not sep:
            continue
        return site, profile, baseline, seed
    return None


def load_log_records(path: pathlib.Path) -> List[str]:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16", errors="ignore").splitlines()
    else:
        try:
            text = raw.decode("utf-8", errors="ignore").splitlines()
        except Exception:
            text = raw.decode("latin-1", errors="ignore").splitlines()
    records: List[str] = []
    current = ""
    for line in text:
        if TIMESTAMP_RE.match(line):
            if current:
                records.append(current)
            current = line
        else:
            current += line
    if current:
        records.append(current)
    return records


def action_bucket(action_repr: str) -> str:
    if "RandomInputAction" in action_repr:
        return "RandomInputAction"
    if "RandomSelectAction" in action_repr:
        return "RandomSelectAction"
    if "RestartAction" in action_repr:
        return "RestartAction"
    if "ClickAction" in action_repr:
        return "ClickAction"
    return "Other"


def parse_newest_json(result_dir: pathlib.Path) -> Tuple[Optional[int], Optional[int], List[Tuple[str, int, int]], List[Tuple[str, int]]]:
    newest = result_dir / "output_data" / "newest.json"
    if not newest.exists():
        return None, None, [], []
    obj = json.loads(newest.read_text(encoding="utf-8"))

    state_entries: List[Tuple[str, int]] = []
    if isinstance(obj.get("state_list"), list):
        for item in obj["state_list"]:
            if not isinstance(item, dict):
                continue
            info = str(item.get("info", ""))
            visited = int(item.get("visited_time", 0))
            state_entries.append((info, visited))
    elif isinstance(obj.get("state_dict"), dict):
        for info, visited in obj["state_dict"].items():
            try:
                visited_int = int(visited)
            except Exception:
                visited_int = 0
            state_entries.append((str(info), visited_int))

    by_url_variants: Dict[str, int] = defaultdict(int)
    by_url_visits: Dict[str, int] = defaultdict(int)
    top_states: List[Tuple[str, int]] = []
    for info, visited in state_entries:
        top_states.append((info, visited))
        if "ActionSetWithExecutionTimesState(" not in info:
            continue
        m = PAT_VALID_URL.search(info)
        if not m:
            continue
        url = m.group(1)
        by_url_variants[url] += 1
        by_url_visits[url] += visited

    top_url_variants = sorted(
        ((url, variants, by_url_visits[url]) for url, variants in by_url_variants.items()),
        key=lambda x: (-x[1], -x[2], x[0]),
    )[:10]
    top_states_sorted = sorted(top_states, key=lambda x: (-x[1], x[0]))[:10]
    return len(by_url_variants), sum(by_url_variants.values()), top_url_variants, top_states_sorted


def parse_run(log_path: pathlib.Path) -> Optional[RunDiagnostics]:
    parsed = parse_root_log_name(log_path.stem)
    if not parsed:
        return None
    site, profile, baseline, seed = parsed
    result_dir = RESULT_ROOT / f"{profile}-{baseline}-{seed}"
    records = load_log_records(log_path)

    chosen_action: Optional[str] = None
    exception_types: Counter = Counter()
    failed_action_types: Counter = Counter()
    failed_actions: Counter = Counter()

    total_steps = None
    total_unique = None
    coverage = None
    only_once_pct = None
    ge10_pct = None
    max_visit_count = None
    out_of_domain = 0
    same_url = 0
    restart = 0
    fail = 0

    for rec in records:
        m = PAT_CHOSEN.search(rec)
        if m:
            chosen_action = m.group(1).strip()
        m = PAT_FAIL.search(rec)
        if m:
            fail += 1
            exception_types[m.group(1)] += 1
            if chosen_action:
                failed_action_types[action_bucket(chosen_action)] += 1
                failed_actions[chosen_action] += 1
        if PAT_OOD.search(rec):
            out_of_domain += 1
        if PAT_SAME.search(rec):
            same_url += 1
        if PAT_RESTART.search(rec):
            restart += 1

        if total_steps is None:
            m = PAT_TOTAL_STEPS.search(rec)
            if m:
                total_steps = int(m.group(1))
        if total_unique is None:
            m = PAT_UNIQUE.search(rec)
            if m:
                total_unique = int(m.group(1))
        if coverage is None:
            m = PAT_COVERAGE.search(rec)
            if m:
                coverage = float(m.group(1))
        if only_once_pct is None:
            m = PAT_ONCE.search(rec)
            if m:
                only_once_pct = float(m.group(2))
        if ge10_pct is None:
            m = PAT_GE10.search(rec)
            if m:
                ge10_pct = float(m.group(2))
        if max_visit_count is None:
            m = PAT_MAXV.search(rec)
            if m:
                max_visit_count = int(m.group(1))

    unique_url_count, state_variant_count, top_url_variants, top_visited_states = parse_newest_json(result_dir)
    return RunDiagnostics(
        site=site,
        profile=profile,
        baseline=baseline,
        seed=seed,
        log_path=log_path,
        result_dir=result_dir,
        total_steps=total_steps,
        total_unique_states=total_unique,
        coverage_efficiency=coverage,
        only_once_pct=only_once_pct,
        ge10_pct=ge10_pct,
        max_visit_count=max_visit_count,
        out_of_domain_count=out_of_domain,
        same_url_count=same_url,
        restart_count=restart,
        fail_count=fail,
        exception_types=exception_types,
        failed_action_types=failed_action_types,
        top_failed_actions=failed_actions.most_common(10),
        state_variant_count=state_variant_count,
        unique_url_count=unique_url_count,
        top_url_variants=top_url_variants,
        top_visited_states=top_visited_states,
    )


def iter_logs() -> Iterable[pathlib.Path]:
    for path in sorted(ROOT.glob("*.log")):
        if path.name.endswith(".backup") or ".backup_" in path.name:
            continue
        yield path


def fmt_num(value: Optional[float], ndigits: int = 1) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{ndigits}f}"


def print_group(label: str, runs: Sequence[RunDiagnostics]) -> None:
    print(f"\n=== {label} ===")
    print(
        "baseline\tseed\tsteps\tuniq\tcov\tmaxv\tmax/steps\tFAIL\tRESTART\tOOD\t1x%\t10+%\turls\tvariants"
    )
    for run in runs:
        max_share = (
            (run.max_visit_count / run.total_steps)
            if run.max_visit_count is not None and run.total_steps
            else None
        )
        print(
            "\t".join(
                [
                    run.baseline,
                    run.seed,
                    fmt_num(run.total_steps, 0),
                    fmt_num(run.total_unique_states, 0),
                    fmt_num(run.coverage_efficiency, 2),
                    fmt_num(run.max_visit_count, 0),
                    fmt_num(max_share, 3),
                    str(run.fail_count),
                    str(run.restart_count),
                    str(run.out_of_domain_count),
                    fmt_num(run.only_once_pct, 1),
                    fmt_num(run.ge10_pct, 1),
                    fmt_num(run.unique_url_count, 0),
                    fmt_num(run.state_variant_count, 0),
                ]
            )
        )
        if run.exception_types:
            print(f"  exceptions: {run.exception_types.most_common(5)}")
        if run.failed_action_types:
            print(f"  failed action types: {run.failed_action_types.most_common()}")
        if run.top_failed_actions:
            print("  top failed actions:")
            for action, count in run.top_failed_actions[:5]:
                print(f"    {count} x {action[:260]}")
        if run.top_url_variants:
            print("  top url variants:")
            for url, variants, visits in run.top_url_variants[:5]:
                print(f"    variants={variants:>2} visits={visits:>4} url={url}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", default="", help="Comma-separated site filter")
    parser.add_argument("--baselines", default="", help="Comma-separated baseline filter")
    args = parser.parse_args()

    site_filter = {x.strip() for x in args.sites.split(",") if x.strip()}
    baseline_filter = {x.strip() for x in args.baselines.split(",") if x.strip()}

    runs = [r for r in (parse_run(p) for p in iter_logs()) if r is not None]
    if site_filter:
        runs = [r for r in runs if r.site in site_filter]
    if baseline_filter:
        runs = [r for r in runs if r.baseline in baseline_filter]

    by_site: Dict[str, List[RunDiagnostics]] = defaultdict(list)
    for run in runs:
        by_site[run.site].append(run)

    for site in sorted(by_site):
        print_group(site, sorted(by_site[site], key=lambda r: (r.baseline, r.seed)))

        by_baseline: Dict[str, List[RunDiagnostics]] = defaultdict(list)
        for run in by_site[site]:
            by_baseline[run.baseline].append(run)
        print(f"\n--- {site} aggregated ---")
        print("baseline\truns\tmean_steps\tmean_uniq\tmean_cov\tmean_maxv\tmean_max/steps\tmean_FAIL\tmean_RESTART")
        for baseline in sorted(by_baseline):
            grp = by_baseline[baseline]
            mean_steps = mean(r.total_steps for r in grp if r.total_steps is not None)
            mean_uniq = mean(r.total_unique_states for r in grp if r.total_unique_states is not None)
            mean_cov = mean(r.coverage_efficiency for r in grp if r.coverage_efficiency is not None)
            mean_maxv = mean(r.max_visit_count for r in grp if r.max_visit_count is not None)
            max_shares = [
                r.max_visit_count / r.total_steps
                for r in grp
                if r.max_visit_count is not None and r.total_steps
            ]
            mean_fail = mean(r.fail_count for r in grp)
            mean_restart = mean(r.restart_count for r in grp)
            print(
                "\t".join(
                    [
                        baseline,
                        str(len(grp)),
                        fmt_num(mean_steps, 1),
                        fmt_num(mean_uniq, 1),
                        fmt_num(mean_cov, 2),
                        fmt_num(mean_maxv, 1),
                        fmt_num(mean(max_shares) if max_shares else None, 3),
                        fmt_num(mean_fail, 1),
                        fmt_num(mean_restart, 1),
                    ]
                )
            )


if __name__ == "__main__":
    main()
