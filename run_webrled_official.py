import argparse
import importlib.util
import os
import site
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


SITE_DEFAULTS = {
    "github": ("https://github.com", "https://github.com"),
    "realworld": ("https://demo.realworld.show", "https://demo.realworld.show"),
    "odoo": ("https://runbot.odoo.com/", "https://runbot.odoo.com"),
    "discourse": ("http://localhost:4203", "http://localhost:4203"),
    "nextcloud": ("http://localhost:8082/apps/files/", "http://localhost:8082"),
    "4gaboards": ("http://localhost:3000", "http://localhost:3000"),
    "agilefant": ("http://localhost:8084/agilefant", "http://localhost:8084/agilefant"),
    "gadael": ("http://localhost:3001", "http://localhost:3001"),
    "timeoff": ("http://localhost:3002", "http://localhost:3002"),
    "petclinic": ("http://localhost:8081", "http://localhost:8081"),
    "splittypie": ("http://localhost:4200", "http://localhost:4200"),
}

COVERAGE_DEFAULTS = {
    # Instrumented first-benchmark apps use the external express-istanbul
    # listener described by the WebRLED artifact. Only one such app should run
    # with coverage on this shared port at a time unless the caller overrides it.
    "retroboard": "http://localhost:6969",
    "dimeshift": "http://localhost:6969",
    "phoenix": "http://localhost:6969",
    "pagekit": "http://localhost:6969",
    "parabank": "http://localhost:6969",
    "agilefant": "http://localhost:6969",
    # Current local Docker deployment maps these container-side coverage
    # services to distinct host ports, so they can coexist.
    "gadael": "http://localhost:6970",
    "timeoff": "http://localhost:6971",
}

POSTRUN_COVERAGE_SITES = {
    "4gaboards",
    "realworld",
}

APPNAME_MAP = {
    "4gaboards": "4gaBoards",
    "agilefant": "agilefant",
    "gadael": "gadael",
    "realworld": "realworld",
    "petclinic": "petclinic",
    "splittypie": "splittypie",
    "timeoff": "timeoff",
    "nextcloud": "nextcloud",
    "github": "github",
    "odoo": "odoo",
}

REQUIRED_MODULES = (
    "bs4",
    "gensim",
    "joblib",
    "lz4",
    "numpy",
    "playwright",
    "requests",
    "sentence_transformers",
    "sklearn",
    "torch",
    "transformers",
)
OPTIONAL_MODULES = (
    # Listed in the upstream requirements, but not imported by the local
    # official source path used in our wrapper. Keep it optional so we do not
    # pollute PYTHONPATH with a foreign Anaconda site-packages just to satisfy
    # check-only validation.
    "regex",
)
REQUIRED_MODEL_FILES = (
    "src/models/all-MiniLM-L6-v2/config.json",
    "src/models/all-MiniLM-L6-v2/pytorch_model.bin",
    "src/models/markuplm-base/pytorch_model.bin",
    "src/models/webembed/content_tags_model_train_setsize300epoch30.doc2vec.model",
    "src/models/webembed/content_tags_model_train_setsize300epoch30.doc2vec.model.dv.vectors.npy",
    "src/models/webembed/content_tags_model_train_setsize300epoch30.doc2vec.model.syn1neg.npy",
    "src/models/webembed/content_tags_model_train_setsize300epoch30.doc2vec.model.wv.vectors.npy",
)

# The SVM comparator is optional: environment.py falls back to threshold comparison
# when this file is absent or incompatible with the installed sklearn.
OPTIONAL_MODEL_FILES = (
    "src/models/webembed/webEmbed_content_tag_size300_epoch30_fix_seed_classifier.pkl",
)


def _optional_user_site() -> str:
    try:
        user_site = site.getusersitepackages()
    except Exception:
        return ""
    if user_site and os.path.isdir(user_site):
        return user_site
    return ""


def _optional_compat_paths() -> list[str]:
    paths: list[str] = []
    raw_extra = os.environ.get("WEBTEST_WEBRLED_EXTRA_PYTHONPATH", "").strip()
    if raw_extra:
        for part in raw_extra.split(os.pathsep):
            if part and os.path.isdir(part):
                paths.append(part)

    # Keep this opt-in only. Adding the base Anaconda site-packages to
    # PYTHONPATH on Windows makes it take precedence over the selected env and
    # can load incompatible numpy/sklearn DLLs during real browser runs.
    return paths


def _prepend_sys_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _extract_seed(seed_text: str, session: str) -> int:
    if seed_text:
        digits = "".join(ch for ch in seed_text if ch.isdigit())
        if digits:
            return int(digits)
    if session:
        digits = "".join(ch for ch in session if ch.isdigit())
        if digits:
            return int(digits)
    return 1


def _resolve_site_target(site: str, url_override: str = "", domain_override: str = "") -> tuple[str, str]:
    site_key = site.lower().strip()
    env_key = site_key.upper().replace("-", "_")

    default_url, default_domain = SITE_DEFAULTS.get(site_key, ("https://github.com", "github.com"))
    entry_url = (
        url_override.strip()
        or os.environ.get(f"WEBTEST_SITE_{env_key}_ENTRY_URL", "").strip()
        or default_url
    )

    resolved_domain_override = (
        domain_override.strip()
        or os.environ.get(f"WEBTEST_SITE_{env_key}_DOMAINS", "").strip()
    )
    if resolved_domain_override:
        domain = resolved_domain_override.split(",")[0].strip()
    else:
        domain = default_domain
        if url_override.strip() or f"WEBTEST_SITE_{env_key}_ENTRY_URL" in os.environ:
            parsed = urlparse(entry_url)
            if parsed.scheme and parsed.netloc:
                path = parsed.path.rstrip("/")
                domain = f"{parsed.scheme}://{parsed.netloc}{path}"
            else:
                domain = entry_url.rstrip("/")

    parsed_domain = urlparse(domain if "://" in domain else f"//{domain}")
    if parsed_domain.netloc.startswith("www."):
        if "://" in domain:
            domain = domain.replace(f"://www.{parsed_domain.netloc[4:]}", f"://{parsed_domain.netloc[4:]}", 1)
        else:
            domain = domain[4:]
    return entry_url, domain


def _coverage_requested(cli_flag: bool) -> bool:
    raw = os.environ.get("WEBTEST_WEBRLED_COVERAGE", "").strip().lower()
    return cli_flag or raw in {"1", "true", "yes", "on", "coverage"}


def _resolve_coverage_url(site: str) -> str:
    site_key = site.lower().strip()
    env_key = site_key.upper().replace("-", "_")
    return (
        os.environ.get(f"WEBTEST_SITE_{env_key}_COVERAGE_URL", "").strip()
        or os.environ.get("WEBTEST_COVERAGE_BASE_URL", "").strip()
        or COVERAGE_DEFAULTS.get(site_key, "")
    )


def _check_prerequisites(official_root: Path) -> list[str]:
    errors: list[str] = []
    main_py = official_root / "src" / "main.py"
    if not main_py.exists():
        errors.append(f"missing official entry: {main_py}")

    for mod in REQUIRED_MODULES:
        if importlib.util.find_spec(mod) is None:
            errors.append(f"missing python module: {mod}")

    for mod in OPTIONAL_MODULES:
        if importlib.util.find_spec(mod) is None:
            print(f"[WebRLED official] WARNING: optional python module missing: {mod}")

    for rel in REQUIRED_MODEL_FILES:
        p = official_root / rel
        if not p.exists():
            errors.append(f"missing model asset: {p}")
        elif p.stat().st_size == 0:
            errors.append(f"model asset is empty (0 bytes) placeholder — download the real file: {p}")

    for rel in OPTIONAL_MODEL_FILES:
        p = official_root / rel
        if not p.exists() or p.stat().st_size == 0:
            print(f"[WebRLED official] WARNING: optional model asset missing/empty (will use fallback): {rel}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official WebRLED package for one site/seed.")
    parser.add_argument("--site", required=True, help="site key, e.g. github/realworld/odoo/...")
    parser.add_argument("--session", default="", help="experiment session label")
    parser.add_argument("--seed", default="", help="seed text, e.g. seed1")
    parser.add_argument(
        "--time-limit",
        type=int,
        default=0,
        help="seconds, 0 means read WEBTEST_ALIVE_TIME_OVERRIDE or fallback to 10800",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="enable WebRLED code-coverage collection for sites with an instrumented deployment",
    )
    parser.add_argument("--url", default="", help="override the site entry URL")
    parser.add_argument("--domain", default="", help="override the site domain/scope")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate wrapper prerequisites and resolved target without starting the browser",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    official_root = repo_root / "external" / "webrled_official_src"
    user_site = _optional_user_site()
    _prepend_sys_path(user_site)
    compat_paths = _optional_compat_paths()
    for compat_path in reversed(compat_paths):
        _prepend_sys_path(compat_path)
    errs = _check_prerequisites(official_root)
    if errs:
        print("[WebRLED official] prerequisites not satisfied:")
        for e in errs:
            print(f"  - {e}")
        print(f"[WebRLED official] current python: {sys.executable}")
        print("Hint: set WEBTEST_WEBRLED_PYTHON to the Python that already has playwright/torch/sentence_transformers/transformers,")
        print("      or install official deps and extract full webrled.zip (including src/models).")
        return 2

    site_key = args.site.lower().strip()
    appname = APPNAME_MAP.get(site_key, "other")
    url, domain = _resolve_site_target(site_key, args.url, args.domain)
    seed_value = _extract_seed(args.seed, args.session)
    if args.check_only:
        print(f"[WebRLED official] check_ok site={site_key} appname={appname} seed={seed_value}")
        print(f"[WebRLED official] url={url} domain={domain}")
        return 0

    env = os.environ.copy()
    py_path = str(official_root)
    python_path_parts = [py_path, str(repo_root)]
    if user_site:
        python_path_parts.append(user_site)
    python_path_parts.extend(compat_paths)
    if env.get("PYTHONPATH", "").strip():
        python_path_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path_parts)

    if args.time_limit > 0:
        env["WEBRLED_TIME_LIMIT"] = str(args.time_limit)
    else:
        env["WEBRLED_TIME_LIMIT"] = os.environ.get("WEBTEST_ALIVE_TIME_OVERRIDE", "10800")
    env["WEBRLED_SEED"] = str(seed_value)
    env["WEBRLED_SEED_TEXT"] = args.seed or args.session or str(seed_value)
    env["WEBRLED_SESSION"] = args.session or ""
    env["WEBTEST_FORCE_HEADFUL"] = os.environ.get("WEBTEST_FORCE_HEADFUL", "1")
    env["WEBTEST_REPO_ROOT"] = str(repo_root)
    env["WEBTEST_EXTERNAL_CANONICAL"] = os.environ.get("WEBTEST_EXTERNAL_CANONICAL", "1")
    env["WEBTEST_EXTERNAL_CANONICAL_SITE"] = site_key
    env["WEBTEST_EXTERNAL_CANONICAL_ALGORITHM"] = "webrled-official"
    env["WEBTEST_EXTERNAL_CANONICAL_SEED"] = args.seed or args.session or str(seed_value)
    env["WEBTEST_EXTERNAL_CANONICAL_PROFILE"] = "drl-1agent-observation"
    enable_coverage = _coverage_requested(args.coverage)
    postrun_coverage = enable_coverage and site_key in POSTRUN_COVERAGE_SITES
    if postrun_coverage:
        print(
            f"[WebRLED official] coverage_mode=postrun_nyc site={site_key}; "
            "not passing --coverage to the live runner. Collect coverage after "
            "the run with nyc report inside the app container."
        )
        enable_coverage = False
    coverage_url = _resolve_coverage_url(site_key) if enable_coverage else ""
    if enable_coverage and not coverage_url:
        print(
            f"[WebRLED official] WARNING: coverage requested for site={site_key}, "
            "but no live coverage endpoint is configured; continuing without "
            "--coverage. Set WEBTEST_SITE_<SITE>_COVERAGE_URL after starting "
            "the instrumented coverage listener."
        )
        enable_coverage = False
    if enable_coverage:
        env_key = site_key.upper().replace("-", "_")
        env.setdefault(f"WEBTEST_SITE_{env_key}_COVERAGE_URL", coverage_url)
        env.setdefault("WEBTEST_COVERAGE_BASE_URL", coverage_url)

    # Use a bootstrap script that injects a minimal datasets stub before any imports,
    # bypassing Anaconda's broken datasets 2.12.0 (s3fs->aiobotocore->urllib3 conflict).
    bootstrap = str(official_root / "_bootstrap.py")
    cmd = [
        sys.executable,
        bootstrap,
        "--appname",
        appname,
        "--url",
        url,
        "--domain",
        domain,
    ]
    if enable_coverage:
        cmd.append("--coverage")
    print(f"[WebRLED official] site={site_key} appname={appname} seed={seed_value}")
    print(f"[WebRLED official] url={url} domain={domain} time_limit={env['WEBRLED_TIME_LIMIT']}")
    if enable_coverage:
        print(f"[WebRLED official] coverage_url={coverage_url}")
    time_limit = int(env["WEBRLED_TIME_LIMIT"])
    grace = int(os.environ.get("WEBTEST_WEBRLED_HARD_TIMEOUT_GRACE", "300"))
    hard_timeout = max(1, time_limit + grace)
    print(f"[WebRLED official] hard_timeout={hard_timeout}s (wrapper-level)")
    proc = subprocess.Popen(cmd, cwd=str(official_root), env=env)
    try:
        return proc.wait(timeout=hard_timeout)
    except subprocess.TimeoutExpired:
        print(
            f"[WebRLED official] HARD TIMEOUT after {hard_timeout}s; "
            "terminating official child process tree."
        )
        if os.name == "nt":
            killed = subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if killed.returncode != 0:
                detail = (killed.stderr or killed.stdout or "").strip()
                print(f"[WebRLED official] WARNING: taskkill failed rc={killed.returncode}: {detail}")
        else:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=15)
        except Exception:
            print("[WebRLED official] WARNING: official child process did not exit after kill request.")
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
