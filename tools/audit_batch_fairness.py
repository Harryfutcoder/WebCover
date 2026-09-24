import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


STATE_URL_RE = re.compile(r"url=([^\)]+)")


def parse_run_name(name: str, sites: set[str], baselines: set[str]):
    for site in sorted(sites, key=len, reverse=True):
        if not name.startswith(site + "-"):
            continue
        for baseline in sorted(baselines, key=len, reverse=True):
            suffix = f"-{baseline}-"
            if suffix not in name:
                continue
            seed_match = re.search(r"-(seed\d+)$", name)
            if not seed_match:
                continue
            return site, baseline, seed_match.group(1)
    return None


def summarize_run(result_dir: Path):
    newest = result_dir / "output_data" / "newest.json"
    if not newest.exists():
        return None
    data = json.loads(newest.read_text(encoding="utf-8"))

    valid_unique = 0
    urls = set()
    for state in data.get("state_list") or []:
        info = str(state.get("info", ""))
        if info.startswith("ActionSetWithExecutionTimesState"):
            valid_unique += 1
            match = STATE_URL_RE.search(info)
            if match:
                urls.add(match.group(1))

    return {
        "valid_unique": valid_unique,
        "unique_urls": len(urls),
        "transitions": len(data.get("transition_list") or []),
    }


def is_cliff(values: list[int], min_high: int, low_median: int, ratio: float) -> bool:
    if len(values) < 3:
        return False
    med = statistics.median(values)
    high = max(values)
    return med <= low_median and high >= min_high and high >= ratio * max(med, 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-level fairness audit for gated/outlier exploration patterns."
    )
    parser.add_argument("--result-base", default="webtest_output/result", type=Path)
    parser.add_argument("--sites", required=True, help="Comma-separated site names.")
    parser.add_argument("--baselines", required=True, help="Comma-separated baseline names.")
    parser.add_argument("--min-high", type=int, default=30)
    parser.add_argument("--low-median", type=int, default=10)
    parser.add_argument("--ratio", type=float, default=3.0)
    parser.add_argument("--fail-on-cliff", action="store_true")
    args = parser.parse_args()

    sites = {x.strip().lower() for x in args.sites.split(",") if x.strip()}
    baselines = {x.strip().lower() for x in args.baselines.split(",") if x.strip()}
    grouped = defaultdict(list)

    for result_dir in args.result_base.iterdir():
        if not result_dir.is_dir():
            continue
        parsed = parse_run_name(result_dir.name, sites, baselines)
        if not parsed:
            continue
        summary = summarize_run(result_dir)
        if not summary:
            continue
        site, baseline, seed = parsed
        grouped[(site, baseline)].append({"seed": seed, **summary})

    exit_code = 0
    for (site, baseline), runs in sorted(grouped.items()):
        runs = sorted(runs, key=lambda r: r["seed"])
        values = [r["valid_unique"] for r in runs]
        med = statistics.median(values)
        mean = statistics.mean(values)
        high = max(values)
        low = min(values)
        cliff = is_cliff(values, args.min_high, args.low_median, args.ratio)
        status = "WARN" if cliff else "OK"
        print(
            f"[BatchFairness][{status}] site={site} baseline={baseline} "
            f"n={len(runs)} meanU={mean:.1f} medianU={med:.1f} minU={low} maxU={high}"
        )
        if cliff:
            print(
                "  suspicious gated/outlier pattern: most seeds have low coverage, "
                "but at least one seed jumps much higher."
            )
            print("  per-seed:", ", ".join(f"{r['seed']}={r['valid_unique']}" for r in runs))
            if args.fail_on_cliff:
                exit_code = 4

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
