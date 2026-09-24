#!/usr/bin/env python
"""Summarize local external-baseline runs without changing the algorithms.

The script understands:
- WebRLED official runs instrumented with metrics.json by run_webrled_official.py.
- Older WebRLED runs, via a conservative log fallback.
- Local QExplore runs, via Q.map and run_metadata.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

if __package__:
    from .benchmark_reporting_order import external_row_sort_key
else:
    from benchmark_reporting_order import external_row_sort_key

RESULT_ROOT = Path("webtest_output/result")
CANONICAL_PROFILES = {
    "webrled-official": "drl-1agent-observation",
    "qexplore": "qexplore-1agent",
}


STEP_RE = re.compile(r"\[Episode\s+\d+\]\s+Step:\s+(\d+)")
URL_RE = re.compile(r"Page url='([^']+)'")
COV_RE = re.compile(r"branch_coverage:\s+([0-9.]+)\s+line_coverage:\s+([0-9.]+)")


def _split_filter(raw: str) -> set[str]:
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value == "" or value is None:
            return default
        return int(value)
    except Exception:
        return default


def _clean_run_part(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    return text.strip("_")


def _canonical_metrics(site: str, algorithm: str, seed: str) -> dict[str, Any]:
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
        "unique_states": len(data.get("state_list", [])),
        "total_actions": len(transitions),
        "unique_actions": executed_actions,
        "unique_edges": len(set(transitions)),
        "unique_urls": len(data.get("url_count", {})),
        "metric_source": "canonical_newest",
        "run_dir": str(run_dir),
    }


def _qexplore_rows(root: Path, sites: set[str], seeds: set[str]) -> list[dict[str, Any]]:
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
        if sites and site.lower() not in sites:
            continue
        if seeds and seed.lower() not in seeds:
            continue
        qmap_path = run_dir / "Q.map"
        coverage_path = run_dir / "qexplore_coverage.json"
        log_path = run_dir / "qexplore.log"
        row: dict[str, Any] = {
            "algorithm": "qexplore",
            "site": site,
            "seed": seed,
            "status": "missing_q_map",
            "unique_states": "",
            "total_actions": "",
            "unique_actions": "",
            "unique_edges": "",
            "unique_urls": "",
            "branch_coverage": "",
            "line_coverage": "",
            "metric_source": "qmap",
            "run_dir": str(run_dir),
        }
        if qmap_path.exists():
            try:
                qmap = _read_json(qmap_path)
                action_total = 0
                action_sigs: set[str] = set()
                edge_sigs: set[tuple[str, str, str]] = set()
                state_sigs: set[str] = set()
                urls: set[str] = set()
                if isinstance(qmap, dict):
                    for src, state in qmap.items():
                        state_sigs.add(str(src))
                        if isinstance(state, dict):
                            if state.get("url"):
                                urls.add(str(state.get("url")))
                            for edge in state.get("edges") or []:
                                if not isinstance(edge, dict):
                                    continue
                                action = str(edge.get("action", ""))
                                dst = str(edge.get("state", ""))
                                action_total += 1
                                if dst:
                                    state_sigs.add(dst)
                                if action:
                                    action_sigs.add(action)
                                edge_sigs.add((str(src), action, dst))
                    row.update(
                        status="ok" if state_sigs or action_total else "empty_q_map",
                        unique_states=len(state_sigs),
                        total_actions=action_total,
                        unique_actions=len(action_sigs),
                        unique_edges=len(edge_sigs),
                        unique_urls=len(urls),
                    )
            except Exception as exc:
                row.update(status="parse_error", metric_source=f"qmap_error:{exc}")
        if coverage_path.exists():
            try:
                coverage = _read_json(coverage_path)
                row.update(
                    branch_coverage=coverage.get("branch_coverage", ""),
                    line_coverage=coverage.get("line_coverage", ""),
                )
                if row.get("metric_source") == "qmap":
                    row["metric_source"] = "qmap+coverage_json"
            except Exception as exc:
                if row.get("status") == "ok":
                    row["status"] = "coverage_parse_error"
                row["metric_source"] = f"{row.get('metric_source')};coverage_error:{exc}"
        canonical = _canonical_metrics(site, "qexplore", seed)
        if canonical:
            row.update(canonical)
            if row.get("status") in {"missing_q_map", "empty_q_map"}:
                row["status"] = "ok"
        if log_path.exists():
            try:
                log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
                if "Traceback (most recent call last):" in log_tail:
                    row["status"] = "runtime_error"
                    row["metric_source"] = f"{row.get('metric_source')};traceback"
            except Exception as exc:
                row["metric_source"] = f"{row.get('metric_source')};log_check_error:{exc}"
        rows.append(row)
    return rows


def _parse_webrled_log(log_path: Path) -> dict[str, Any]:
    total_steps = 0
    action_sigs: set[str] = set()
    urls: set[str] = set()
    branch = ""
    line = ""
    try:
        for raw in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if STEP_RE.search(raw):
                total_steps += 1
            if "Clicked:" in raw or "Typed" in raw or "Select" in raw or "form_action_normal" in raw:
                action_sigs.add(raw.strip())
            m_url = URL_RE.search(raw)
            if m_url:
                urls.add(m_url.group(1))
            m_cov = COV_RE.search(raw)
            if m_cov:
                branch, line = m_cov.group(1), m_cov.group(2)
    except Exception:
        pass
    return {
        "total_actions": total_steps,
        "unique_actions": len(action_sigs) if action_sigs else "",
        "unique_urls": len(urls) if urls else "",
        "branch_coverage": branch,
        "line_coverage": line,
    }


def _webrled_rows(root: Path, sites: set[str], seeds: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for app_dir in sorted(p for p in root.glob("run-*/*") if p.is_dir()):
        metrics_path = app_dir / "metrics.json"
        log_path = app_dir / f"{app_dir.name}.log"
        site = app_dir.name
        seed = ""
        row: dict[str, Any] = {
            "algorithm": "webrled-official",
            "site": site,
            "seed": seed,
            "status": "no_metrics_json",
            "unique_states": "",
            "total_actions": "",
            "unique_actions": "",
            "unique_edges": "",
            "unique_urls": "",
            "branch_coverage": "",
            "line_coverage": "",
            "metric_source": "fallback_log_no_state_metrics",
            "run_dir": str(app_dir),
        }
        if metrics_path.exists():
            try:
                metrics = _read_json(metrics_path)
                site = str(metrics.get("app_name") or site)
                seed = str(metrics.get("seed") or "")
                row.update(
                    site=site,
                    seed=seed,
                    status="ok",
                    unique_states=_safe_int(metrics.get("unique_states"), ""),
                    total_actions=_safe_int(metrics.get("total_actions"), ""),
                    unique_actions=_safe_int(metrics.get("unique_actions"), ""),
                    unique_edges="",
                    unique_urls=_safe_int(metrics.get("unique_urls"), ""),
                    branch_coverage=metrics.get("branch_coverage", ""),
                    line_coverage=metrics.get("line_coverage", ""),
                    metric_source="metrics_json",
                )
                canonical = _canonical_metrics(site, "webrled-official", seed)
                if canonical:
                    row.update(canonical)
            except Exception as exc:
                row.update(status="metrics_parse_error", metric_source=f"metrics_error:{exc}")
        elif log_path.exists():
            row.update(_parse_webrled_log(log_path))
        if sites and str(row["site"]).lower() not in sites:
            continue
        if seeds and str(row["seed"]).lower() not in seeds:
            continue
        rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "algorithm",
        "site",
        "seed",
        "status",
        "unique_states",
        "total_actions",
        "unique_actions",
        "unique_edges",
        "unique_urls",
        "branch_coverage",
        "line_coverage",
        "metric_source",
        "run_dir",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _print_table(rows: list[dict[str, Any]], limit: int) -> None:
    print(
        "algorithm,site,seed,status,unique_states,total_actions,unique_actions,"
        "unique_edges,unique_urls,branch_coverage,line_coverage,source"
    )
    for row in rows[:limit]:
        print(
            ",".join(
                str(row.get(k, ""))
                for k in [
                    "algorithm",
                    "site",
                    "seed",
                    "status",
                    "unique_states",
                    "total_actions",
                    "unique_actions",
                    "unique_edges",
                    "unique_urls",
                    "branch_coverage",
                    "line_coverage",
                    "metric_source",
                ]
            )
        )
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more rows")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qexplore-root", default="external/QExplore/runs")
    parser.add_argument("--webrled-root", default="external/webrled_official_src/src/run")
    parser.add_argument("--sites", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--only", choices=["all", "qexplore", "webrled"], default="all")
    parser.add_argument("--output-csv", default="analysis/external_baselines/external_baseline_metrics.csv")
    parser.add_argument("--print-table", action="store_true")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    sites = _split_filter(args.sites)
    seeds = _split_filter(args.seeds)
    rows: list[dict[str, Any]] = []
    if args.only in {"all", "qexplore"}:
        rows.extend(_qexplore_rows(Path(args.qexplore_root), sites, seeds))
    if args.only in {"all", "webrled"}:
        rows.extend(_webrled_rows(Path(args.webrled_root), sites, seeds))
    rows.sort(key=external_row_sort_key)
    output = Path(args.output_csv)
    _write_csv(rows, output)
    print(f"[external-metrics] wrote {len(rows)} rows to {output}")
    if args.print_table:
        _print_table(rows, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
