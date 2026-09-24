#!/usr/bin/env python
"""Monitor generic WebTest code-coverage logs.

This reads logs produced by the main WebTest framework when
``WEBTEST_CODE_COVERAGE=1`` is enabled. It complements
``monitor_external_baseline_runs.py`` for external WebRLED/QExplore runs.
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COV_RE = re.compile(
    r"CodeCoverage\s+reason=(?P<reason>\S+)\s+"
    r"branch_coverage=(?P<branch>\S*)\s+"
    r"line_coverage=(?P<line>\S*)\s+"
    r"statement_coverage=(?P<statement>\S*)\s+"
    r"function_coverage=(?P<function>\S*)\s+"
    r"url=(?P<url>\S+)"
)


def _split_filter(raw: str) -> set[str]:
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def _age_min(path: Path) -> float:
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 60.0


def _tail_text(path: Path, limit: int = 200000) -> str:
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def _parse_log_name(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    site = ""
    algorithm = ""
    seed = ""
    if "_" in stem:
        left, right = stem.split("_", 1)
        site = left.split("-")[0]
        if "_" in right:
            algorithm, seed = right.rsplit("_", 1)
        elif "-" in right:
            parts = right.split("-")
            seed_idx = next((i for i, p in enumerate(parts) if p.lower().startswith("seed")), -1)
            if seed_idx > 0:
                algorithm = "-".join(parts[:seed_idx])
                seed = "-".join(parts[seed_idx:])
            else:
                algorithm = right
        else:
            algorithm = right
    return site, algorithm, seed


def parse_log(path: Path, stale_minutes: float) -> dict[str, Any] | None:
    text = _tail_text(path)
    matches = list(COV_RE.finditer(text))
    if not matches and "CodeCoverage" not in text:
        return None
    site, algorithm, seed = _parse_log_name(path)
    age = _age_min(path)
    status = "active" if age <= stale_minutes else "stale"
    issues: list[str] = []
    if "Traceback (most recent call last):" in text:
        status = "runtime_error"
        issues.append("traceback")
    if "CodeCoverage fetch failed" in text:
        issues.append("coverage_fetch_failed")
    if "Abort run early" in text:
        issues.append("abort_run_early")
    row: dict[str, Any] = {
        "site": site,
        "algorithm": algorithm,
        "seed": seed,
        "status": status,
        "age_min": f"{age:.1f}",
        "coverage_events": len(matches),
        "branch_coverage": "",
        "line_coverage": "",
        "statement_coverage": "",
        "function_coverage": "",
        "last_reason": "",
        "coverage_url": "",
        "log": str(path),
        "issues": ";".join(sorted(set(issues))),
    }
    if matches:
        last = matches[-1]
        row.update(
            branch_coverage=last.group("branch"),
            line_coverage=last.group("line"),
            statement_coverage=last.group("statement"),
            function_coverage=last.group("function"),
            last_reason=last.group("reason"),
            coverage_url=last.group("url"),
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="workspace root")
    parser.add_argument("--glob", default="*.log", help="log glob relative to root")
    parser.add_argument("--sites", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--stale-minutes", type=float, default=30.0)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--max-files", type=int, default=200, help="only inspect the newest N matching logs")
    args = parser.parse_args()

    root = Path(args.root)
    sites = _split_filter(args.sites)
    seeds = _split_filter(args.seeds)
    rows: list[dict[str, Any]] = []
    candidates = sorted(root.glob(args.glob), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates[: max(1, args.max_files)]:
        if not path.is_file():
            continue
        row = parse_log(path, args.stale_minutes)
        if row is None:
            continue
        if sites and str(row["site"]).lower() not in sites:
            continue
        if seeds and str(row["seed"]).lower() not in seeds:
            continue
        rows.append(row)

    fields = [
        "site",
        "algorithm",
        "seed",
        "status",
        "age_min",
        "coverage_events",
        "branch_coverage",
        "line_coverage",
        "statement_coverage",
        "function_coverage",
        "last_reason",
        "coverage_url",
        "issues",
        "log",
    ]
    if args.output_csv:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fields})
        print(f"[codecov-log-monitor] wrote {len(rows)} rows to {output}")

    if not rows:
        print("No generic WebTest CodeCoverage logs found.")
        return 0

    try:
        from tabulate import tabulate

        print(tabulate([{k: r.get(k, "") for k in fields[:-1]} for r in rows[: args.limit]], headers="keys"))
    except Exception:
        for row in rows[: args.limit]:
            print(",".join(str(row.get(k, "")) for k in fields[:-1]))
    return 0 if all(r.get("status") == "active" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
