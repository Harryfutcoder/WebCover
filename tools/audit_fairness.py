import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

import yaml


URL_RE = re.compile(r"https?://[^\s,\)]+")
STATE_URL_RE = re.compile(r"url=([^\)]+)")


AUTO_SITE_DEFAULTS = {
    "odoo": {"domains": ["localhost:8069", "localhost"]},
    "discourse": {"domains": ["localhost:4203", "localhost"]},
    "nextcloud": {"domains": ["localhost:8082", "localhost"]},
    "4gaboards": {"domains": ["localhost:3000", "localhost"]},
    "agilefant": {"domains": ["localhost:8084", "localhost"]},
}


def load_allowed_domains(settings_path: Path, profile: str, site: str) -> list[str]:
    settings_data = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    profiles = settings_data.get("profiles", {}) if isinstance(settings_data, dict) else {}

    domains = []
    if profile in profiles:
        domains = list(profiles[profile].get("domains") or [])
    elif site in AUTO_SITE_DEFAULTS:
        domains = list(AUTO_SITE_DEFAULTS[site]["domains"])

    env_key = site.upper().replace("-", "_")
    override = os.environ.get(f"WEBTEST_SITE_{env_key}_DOMAINS", "").strip()
    if override:
        domains = [x.strip() for x in override.split(",") if x.strip()]

    return [str(x).strip().lower() for x in domains if str(x).strip()]


def is_allowed_url(raw_url: str, allowed_domains: list[str]) -> bool:
    try:
        parsed = urlsplit(raw_url)
    except Exception:
        return True

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return True

    netloc = (parsed.netloc or "").lower()
    host = (parsed.hostname or "").lower()
    if not netloc:
        return True

    for allowed in allowed_domains:
        if netloc == allowed or netloc.endswith("." + allowed):
            return True
        if host and (host == allowed or host.endswith("." + allowed)):
            return True
    return False


def load_newest(result_dir: Path) -> dict:
    newest_path = result_dir / "output_data" / "newest.json"
    if not newest_path.exists():
        raise FileNotFoundError(f"missing result file: {newest_path}")
    return json.loads(newest_path.read_text(encoding="utf-8"))


def summarize_result(data: dict, allowed_domains: list[str]) -> dict:
    external_exec = 0
    total_exec = 0
    external_urls = {}
    for item in data.get("action_list") or []:
        if not isinstance(item, list) or len(item) < 2:
            continue
        action_text = str(item[0])
        try:
            count = int(item[1] or 0)
        except Exception:
            count = 0
        total_exec += count

        bad_urls = [
            url for url in URL_RE.findall(action_text)
            if not is_allowed_url(url, allowed_domains)
        ]
        if bad_urls:
            external_exec += count
            for url in bad_urls:
                external_urls[url] = external_urls.get(url, 0) + count

    valid_unique = 0
    urls = set()
    for state in data.get("state_list") or []:
        info = str(state.get("info", ""))
        if info.startswith("ActionSetWithExecutionTimesState"):
            valid_unique += 1
            match = STATE_URL_RE.search(info)
            if match:
                urls.add(match.group(1))

    transitions = len(data.get("transition_list") or [])
    external_pct = (100.0 * external_exec / total_exec) if total_exec else 0.0
    return {
        "total_action_exec": total_exec,
        "external_action_exec": external_exec,
        "external_action_pct": external_pct,
        "external_urls": external_urls,
        "valid_unique_states": valid_unique,
        "unique_urls": len(urls),
        "transitions": transitions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a webtest run for fair in-domain action-space behavior.")
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--settings", default="settings.yaml", type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--site", default="")
    parser.add_argument("--max-offdomain-exec", type=int, default=0)
    parser.add_argument("--min-unique-states", type=int, default=10)
    parser.add_argument("--min-transitions-for-low-unique", type=int, default=200)
    parser.add_argument("--strict-low-unique", action="store_true")
    args = parser.parse_args()

    site = args.site.strip().lower() or args.profile.split("-", 1)[0].lower()
    allowed_domains = load_allowed_domains(args.settings, args.profile, site)
    data = load_newest(args.result_dir)
    summary = summarize_result(data, allowed_domains)

    print(
        "[FairnessAudit] "
        f"profile={args.profile} site={site} "
        f"allowed_domains={allowed_domains} "
        f"valid_unique={summary['valid_unique_states']} "
        f"unique_urls={summary['unique_urls']} "
        f"transitions={summary['transitions']} "
        f"offdomain_exec={summary['external_action_exec']}/{summary['total_action_exec']} "
        f"({summary['external_action_pct']:.2f}%)"
    )

    exit_code = 0
    if summary["external_action_exec"] > args.max_offdomain_exec:
        print("[FairnessAudit][FAIL] Off-domain actions were exposed/executed.")
        for url, count in sorted(summary["external_urls"].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  offdomain count={count}: {url}")
        exit_code = 2

    low_unique = (
        summary["transitions"] >= args.min_transitions_for_low_unique
        and summary["valid_unique_states"] < args.min_unique_states
    )
    if low_unique:
        level = "FAIL" if args.strict_low_unique else "WARN"
        print(
            f"[FairnessAudit][{level}] Low unique-state coverage after a long run: "
            f"{summary['valid_unique_states']} < {args.min_unique_states}"
        )
        if args.strict_low_unique and exit_code == 0:
            exit_code = 3

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
