#!/usr/bin/env python
"""Monitor WebRLED official and local QExplore runs.

This is intentionally read-only. It does not start, stop, or mutate runs.
It reports whether each run has state/action metrics, coverage fields, fresh
logs, and obvious runtime failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULT_ROOT = Path("webtest_output/result")
CANONICAL_PROFILES = {
    "webrled-official": "drl-1agent-observation",
    "qexplore": "qexplore-1agent",
}


def _split_filter(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _age_min(path: Path) -> float:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return (datetime.now(timezone.utc) - mtime).total_seconds() / 60.0
    except Exception:
        return -1.0


def _tail_text(path: Path, limit: int = 20000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except Exception:
        return ""


def _metric_int(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return str(int(value))
    except Exception:
        return str(value)


def _as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(str(value).strip().rstrip("%")))
    except Exception:
        return None


def _clean_run_part(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in text)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _canonical_metrics(site: str, algorithm: str, seed: str) -> dict[str, str]:
    profile = CANONICAL_PROFILES.get(algorithm, f"{algorithm}-1agent")
    site_key = _clean_run_part(site)
    seed_key = _clean_run_part(seed)
    run_dir = RESULT_ROOT / f"{site_key}-{profile}-{algorithm}-{seed_key}"
    snapshot = run_dir / "output_data" / "newest.json"
    if not snapshot.exists():
        return {}
    data = _read_json(snapshot)
    transitions = [tuple(item) for item in data.get("transition_list", [])]
    action_list = data.get("action_list", [])
    executed_actions = 0
    for item in action_list:
        try:
            if isinstance(item, list) and len(item) >= 2 and int(item[1] or 0) > 0:
                executed_actions += 1
        except Exception:
            continue
    return {
        "unique_states": str(len(data.get("state_list", []))),
        "total_actions": str(len(transitions)),
        "unique_actions": str(executed_actions),
        "unique_edges": str(len(set(transitions))),
        "unique_urls": str(len(data.get("url_count", {}))),
        "canonical_run_dir": str(run_dir),
    }


def _qexplore_metrics(run_dir: Path) -> dict[str, str]:
    qmap_path = run_dir / "Q.map"
    if not qmap_path.exists():
        return {
            "unique_states": "",
            "total_actions": "",
            "unique_actions": "",
            "unique_edges": "",
            "unique_urls": "",
        }
    qmap = _read_json(qmap_path)
    states: set[str] = set()
    actions: set[str] = set()
    edges: set[tuple[str, str, str]] = set()
    urls: set[str] = set()
    total_actions = 0
    if isinstance(qmap, dict):
        for src, state in qmap.items():
            states.add(str(src))
            if isinstance(state, dict):
                if state.get("url"):
                    urls.add(str(state.get("url")))
                for edge in state.get("edges") or []:
                    if not isinstance(edge, dict):
                        continue
                    action = str(edge.get("action", ""))
                    dst = str(edge.get("state", ""))
                    total_actions += 1
                    if dst:
                        states.add(dst)
                    if action:
                        actions.add(action)
                    edges.add((str(src), action, dst))
    return {
        "unique_states": str(len(states)) if states else "",
        "total_actions": str(total_actions) if total_actions else "",
        "unique_actions": str(len(actions)) if actions else "",
        "unique_edges": str(len(edges)) if edges else "",
        "unique_urls": str(len(urls)) if urls else "",
    }


def _coverage_fields(path: Path) -> dict[str, str]:
    if not path.exists():
        return {"branch_coverage": "", "line_coverage": ""}
    try:
        data = _read_json(path)
        return {
            "branch_coverage": str(data.get("branch_coverage", "")),
            "line_coverage": str(data.get("line_coverage", "")),
        }
    except Exception:
        return {"branch_coverage": "", "line_coverage": ""}


def _webexplor_rows(root: Path, stale_minutes: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for app_dir in sorted(p for p in root.glob("run-*/*") if p.is_dir()):
        site = app_dir.name
        run_id = app_dir.parent.name
        metrics_path = app_dir / "metrics.json"
        log_path = app_dir / f"{site}.log"
        base = {
            "algorithm": "webrled-official",
            "site": site,
            "seed": "",
            "status": "no_metrics",
            "age_min": "",
            "unique_states": "",
            "total_actions": "",
            "unique_actions": "",
            "unique_edges": "",
            "unique_urls": "",
            "branch_coverage": "",
            "line_coverage": "",
            "run_id": run_id,
            "run_dir": str(app_dir),
            "issues": "",
        }
        issues: list[str] = []
        latest_path = metrics_path if metrics_path.exists() else log_path
        if latest_path.exists():
            base["age_min"] = f"{_age_min(latest_path):.1f}"
        if metrics_path.exists():
            try:
                metrics = _read_json(metrics_path)
                done = _as_int(metrics.get("done"))
                age = _age_min(metrics_path)
                base.update(
                    seed=str(metrics.get("seed") or ""),
                    unique_states=_metric_int(metrics.get("unique_states")),
                    total_actions=_metric_int(metrics.get("total_actions")),
                    unique_actions=_metric_int(metrics.get("unique_actions")),
                    unique_urls=_metric_int(metrics.get("unique_urls")),
                    branch_coverage=str(metrics.get("branch_coverage", "")),
                    line_coverage=str(metrics.get("line_coverage", "")),
                )
                canonical = _canonical_metrics(base["site"], "webrled-official", base["seed"])
                if canonical:
                    base.update(canonical)
                    issues.append("metrics=canonical_newest")
                if done == 1:
                    base["status"] = "finalized"
                elif age <= stale_minutes:
                    base["status"] = "active"
                else:
                    base["status"] = "stale"
                for field in ["unique_states", "total_actions", "unique_actions"]:
                    if not base[field]:
                        issues.append(f"missing_{field}")
                unique_states = _as_int(base["unique_states"])
                total_actions = _as_int(base["total_actions"])
                if unique_states is not None and total_actions is not None:
                    if unique_states <= 1 and total_actions >= 50:
                        issues.append("low_state_loop")
                    elif unique_states <= 3 and total_actions >= 200:
                        issues.append("low_coverage_loop")
            except Exception as exc:
                base["status"] = "metrics_parse_error"
                issues.append(str(exc).splitlines()[0][:80])
        if log_path.exists():
            tail = _tail_text(log_path)
            if "Traceback (most recent call last):" in tail:
                if base["status"] in {"finalized", "active", "stale"}:
                    base["status"] = "runtime_error"
                issues.append("traceback")
            if "JSONDecodeError" in tail:
                issues.append("coverage_json_decode")
            if "HARD TIMEOUT" in tail:
                issues.append("hard_timeout")
        if not metrics_path.exists() and log_path.exists():
            age = _age_min(log_path)
            base["status"] = "active_no_metrics" if age <= stale_minutes else "stale_no_metrics"
            tail = _tail_text(log_path)
            if "Traceback (most recent call last):" in tail:
                issues.append("traceback")
            if "HARD TIMEOUT" in tail:
                issues.append("hard_timeout")
        elif not metrics_path.exists():
            base["status"] = "empty_run_dir"
            issues.append("missing_log")
        base["issues"] = ";".join(issues)
        rows.append(base)
    return rows


def _qexplore_rows(root: Path, stale_minutes: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for run_dir in sorted(p for p in root.glob("*/*") if p.is_dir()):
        metadata_path = run_dir / "run_metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = _read_json(metadata_path)
            except Exception:
                metadata = {}
        site = str(metadata.get("site") or run_dir.parent.name)
        seed = str(metadata.get("seed") or run_dir.name)
        qmap_path = run_dir / "Q.map"
        log_path = run_dir / "qexplore.log"
        coverage_path = run_dir / "qexplore_coverage.json"
        finish_path = run_dir / "qexplore_finish.json"
        latest_path = qmap_path if qmap_path.exists() else log_path
        row = {
            "algorithm": "qexplore",
            "site": site,
            "seed": seed,
            "status": "missing_q_map",
            "age_min": f"{_age_min(latest_path):.1f}" if latest_path.exists() else "",
            "unique_states": "",
            "total_actions": "",
            "unique_actions": "",
            "unique_edges": "",
            "unique_urls": "",
            "branch_coverage": "",
            "line_coverage": "",
            "run_id": seed,
            "run_dir": str(run_dir),
            "issues": "",
        }
        issues: list[str] = []
        finish: dict[str, Any] = {}
        if finish_path.exists():
            try:
                finish = _read_json(finish_path)
            except Exception as exc:
                issues.append(f"finish_parse:{str(exc).splitlines()[0][:60]}")
        if qmap_path.exists():
            try:
                row.update(_qexplore_metrics(run_dir))
                canonical = _canonical_metrics(site, "qexplore", seed)
                if canonical:
                    row.update(canonical)
                    issues.append("metrics=canonical_newest")
                if finish:
                    exit_code = _as_int(finish.get("exit_code"))
                    row["status"] = "finalized" if exit_code == 0 else "runtime_error"
                    if exit_code not in (None, 0):
                        issues.append(f"exit_code={exit_code}")
                elif _age_min(qmap_path) <= stale_minutes:
                    row["status"] = "active"
                else:
                    row["status"] = "stale"
                if not (row["unique_states"] or row["total_actions"]):
                    row["status"] = "empty_q_map"
            except Exception as exc:
                row["status"] = "qmap_parse_error"
                issues.append(str(exc).splitlines()[0][:80])
        elif log_path.exists():
            age = _age_min(log_path)
            if finish:
                exit_code = _as_int(finish.get("exit_code"))
                row["status"] = "finalized_no_qmap" if exit_code == 0 else "runtime_error"
                if exit_code not in (None, 0):
                    issues.append(f"exit_code={exit_code}")
            else:
                row["status"] = "active_no_qmap" if age <= stale_minutes else "stale_no_qmap"
        else:
            row["status"] = "missing_log"
            issues.append("missing_log")
        row.update(_coverage_fields(coverage_path))
        tail = _tail_text(log_path)
        if "Traceback (most recent call last):" in tail:
            row["status"] = "runtime_error"
            issues.append("traceback")
        unique_states = _as_int(row["unique_states"])
        total_actions = _as_int(row["total_actions"])
        if unique_states is not None and total_actions is not None:
            if unique_states <= 1 and total_actions >= 50:
                issues.append("low_state_loop")
            elif unique_states <= 3 and total_actions >= 200:
                issues.append("low_coverage_loop")
        for field in ["unique_states", "total_actions", "unique_actions"]:
            if row["status"] in {"finalized", "runtime_error"} and not row[field]:
                issues.append(f"missing_{field}")
        row["issues"] = ";".join(sorted(set(issues)))
        rows.append(row)
    return rows


def _filter_rows(rows: list[dict[str, Any]], sites: list[str], seeds: list[str]) -> list[dict[str, Any]]:
    site_set = {x.lower() for x in sites}
    seed_set = {x.lower() for x in seeds}
    out: list[dict[str, Any]] = []
    for row in rows:
        if site_set and str(row.get("site", "")).lower() not in site_set:
            continue
        if seed_set and str(row.get("seed", "")).lower() not in seed_set:
            # Keep active WebRLED rows without metrics because their seed is not
            # recoverable until metrics.json is written. Do not keep old stale
            # unseeded logs from historical runs; they make focused seed
            # monitors noisy and misleading.
            if not (
                row.get("algorithm") == "webrled-official"
                and not row.get("seed")
                and str(row.get("status")) == "active_no_metrics"
            ):
                continue
        out.append(row)
    return out


def _add_missing(rows: list[dict[str, Any]], algorithms: list[str], sites: list[str], seeds: list[str]) -> list[dict[str, Any]]:
    if not sites or not seeds:
        return rows
    seen = {
        (str(r.get("algorithm")), str(r.get("site")).lower(), str(r.get("seed")).lower())
        for r in rows
        if r.get("seed")
    }
    out = list(rows)
    for algorithm in algorithms:
        for site in sites:
            for seed in seeds:
                key = (algorithm, site.lower(), seed.lower())
                if key in seen:
                    continue
                out.append(
                    {
                        "algorithm": algorithm,
                        "site": site,
                        "seed": seed,
                        "status": "missing",
                        "age_min": "",
                        "unique_states": "",
                        "total_actions": "",
                        "unique_actions": "",
                        "unique_edges": "",
                        "unique_urls": "",
                        "branch_coverage": "",
                        "line_coverage": "",
                        "run_id": "",
                        "run_dir": "",
                        "issues": "",
                    }
                )
    return out


def _print_table(rows: list[dict[str, Any]], limit: int) -> None:
    fields = [
        "algorithm",
        "site",
        "seed",
        "status",
        "age_min",
        "unique_states",
        "total_actions",
        "unique_actions",
        "branch_coverage",
        "line_coverage",
        "issues",
    ]
    widths = {field: len(field) for field in fields}
    for row in rows[:limit]:
        for field in fields:
            widths[field] = max(widths[field], min(32, len(str(row.get(field, "")))))
    print("  ".join(field.ljust(widths[field]) for field in fields))
    print("  ".join("-" * widths[field] for field in fields))
    for row in rows[:limit]:
        print(
            "  ".join(
                str(row.get(field, ""))[: widths[field]].ljust(widths[field])
                for field in fields
            )
        )
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more rows")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "algorithm",
        "site",
        "seed",
        "status",
        "age_min",
        "unique_states",
        "total_actions",
        "unique_actions",
        "unique_edges",
        "unique_urls",
        "branch_coverage",
        "line_coverage",
        "run_id",
        "run_dir",
        "issues",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--only", choices=["all", "webrled", "qexplore"], default="all")
    parser.add_argument("--stale-minutes", type=float, default=30.0)
    parser.add_argument("--webrled-root", default="external/webrled_official_src/src/run")
    parser.add_argument("--qexplore-root", default="external/QExplore/runs")
    parser.add_argument("--include-missing", action="store_true")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()

    sites = _split_filter(args.sites)
    seeds = _split_filter(args.seeds)
    rows: list[dict[str, Any]] = []
    algorithms: list[str] = []
    if args.only in {"all", "webrled"}:
        algorithms.append("webrled-official")
        rows.extend(_webexplor_rows(Path(args.webrled_root), args.stale_minutes))
    if args.only in {"all", "qexplore"}:
        algorithms.append("qexplore")
        rows.extend(_qexplore_rows(Path(args.qexplore_root), args.stale_minutes))
    rows = _filter_rows(rows, sites, seeds)
    if args.include_missing:
        rows = _add_missing(rows, algorithms, sites, seeds)
    status_order = {
        "runtime_error": 0,
        "stale_no_metrics": 1,
        "stale_no_qmap": 1,
        "active_no_metrics": 2,
        "active_no_qmap": 2,
        "missing": 3,
        "empty_q_map": 4,
        "finalized": 9,
    }
    rows.sort(
        key=lambda r: (
            status_order.get(str(r.get("status")), 5),
            str(r.get("algorithm")),
            str(r.get("site")),
            str(r.get("seed")),
            str(r.get("run_id")),
        )
    )
    if args.output_csv:
        _write_csv(rows, Path(args.output_csv))
        print(f"[external-monitor] wrote {len(rows)} rows to {args.output_csv}")
    _print_table(rows, args.limit)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("status"))] = counts.get(str(row.get("status")), 0) + 1
    print("summary " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    unhealthy = sum(v for k, v in counts.items() if k.startswith("stale") or k in {"runtime_error", "qmap_parse_error", "metrics_parse_error"})
    return 2 if unhealthy else 0


if __name__ == "__main__":
    raise SystemExit(main())
