"""
Extract and count unique faults (JS errors) from bug.log files.

Usage:
    python extract_faults.py                          # scan all result dirs
    python extract_faults.py  path/to/result_dir      # single result dir
    python extract_faults.py  path/to/result_dir  --detail   # show each error
"""

import ast
import os
import re
import sys
from collections import Counter, defaultdict


def _normalise_message(msg: str) -> str:
    """Strip URLs, line numbers, and hex ids to group equivalent errors."""
    msg = re.sub(r"https?://\S+", "<URL>", msg)
    msg = re.sub(r"\d+:\d+", "<LOC>", msg)
    msg = re.sub(r"0x[0-9a-fA-F]+", "<HEX>", msg)
    msg = re.sub(r"\s+", " ", msg).strip()
    return msg


def parse_bug_log(bug_log_path: str):
    """Yield (level, normalised_message) from a bug.log file."""
    if not os.path.isfile(bug_log_path):
        return
    with open(bug_log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            level = entry.get("level", "")
            message = entry.get("message", "")
            if level in ("SEVERE", "WARNING") and message:
                yield level, _normalise_message(message)


def analyse_dir(result_dir: str, detail: bool = False):
    bug_path = os.path.join(result_dir, "bug.log")
    if not os.path.isfile(bug_path):
        return None

    severe = Counter()
    warning = Counter()
    for level, msg in parse_bug_log(bug_path):
        if level == "SEVERE":
            severe[msg] += 1
        else:
            warning[msg] += 1

    name = os.path.basename(result_dir)
    total_severe = sum(severe.values())
    total_warning = sum(warning.values())
    unique_severe = len(severe)
    unique_warning = len(warning)

    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"  SEVERE: {unique_severe} unique / {total_severe} total")
    print(f"  WARNING: {unique_warning} unique / {total_warning} total")
    print(f"  TOTAL unique faults: {unique_severe + unique_warning}")
    print(f"{'=' * 70}")

    if detail:
        if severe:
            print("  [SEVERE errors]")
            for msg, cnt in severe.most_common(20):
                short = msg[:100] + ("..." if len(msg) > 100 else "")
                print(f"    x{cnt:>4}  {short}")
        if warning:
            print("  [WARNINGs]")
            for msg, cnt in warning.most_common(10):
                short = msg[:100] + ("..." if len(msg) > 100 else "")
                print(f"    x{cnt:>4}  {short}")

    return {
        "name": name,
        "unique_severe": unique_severe,
        "total_severe": total_severe,
        "unique_warning": unique_warning,
        "total_warning": total_warning,
    }


def main():
    detail = "--detail" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if args:
        dirs = args
    else:
        base = os.path.join("webtest_output", "result")
        if not os.path.isdir(base):
            print(f"Default result dir not found: {base}")
            sys.exit(1)
        dirs = sorted(
            os.path.join(base, d) for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
        )

    results = []
    for d in dirs:
        r = analyse_dir(d, detail=detail)
        if r is not None:
            results.append(r)

    if results:
        print("\n\n===  SUMMARY TABLE  ===")
        print(f"{'Experiment':<60} {'Uniq_SEVERE':>12} {'Uniq_WARN':>10} {'Uniq_TOTAL':>11}")
        print("-" * 95)
        for r in results:
            total = r["unique_severe"] + r["unique_warning"]
            print(
                f"{r['name']:<60} {r['unique_severe']:>12} "
                f"{r['unique_warning']:>10} {total:>11}"
            )


if __name__ == "__main__":
    main()
