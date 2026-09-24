import hashlib
import logging
import os
import random
import re
import time

from selenium.webdriver.chrome.options import Options

from data_collector.data_collector_single_agent import DataCollector
from config.log_config import LogConfig
from config.settings import settings
from fairness import validate_fair_contract, is_fair_mode
from web_test.webtest_single_agent import Webtest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(LogConfig.get_file_handler())


def _inject_headless_stability_args(chrome_options: Options) -> None:
    """Add conservative flags to reduce headless Chrome instability on Windows."""
    args = chrome_options.arguments
    headless = any(a == "--headless" or a.startswith("--headless=") for a in args)
    if not headless:
        return

    for i, a in enumerate(args):
        if a == "--headless":
            args[i] = "--headless=new"
            break

    for flag in (
        "--disable-webgl",
        "--disable-webgl2",
        "--disable-3d-apis",
        "--remote-debugging-port=0",
    ):
        if flag not in args:
            chrome_options.add_argument(flag)


def configure_chrome_options() -> Options:
    chrome_options = Options()

    force_system_chrome = os.environ.get("WEBTEST_FORCE_SYSTEM_CHROME", "0") == "1"
    bp = settings.browser_path
    if (not force_system_chrome) and bp and os.path.isfile(bp):
        chrome_options.binary_location = bp
    else:
        logger.warning("browser_path is not an existing file (%r), using system default Chrome", bp)

    browser_args = list(settings.browser_arguments)
    # Optional runtime override for anti-bot-sensitive sites (e.g., toppr).
    if os.environ.get("WEBTEST_FORCE_HEADFUL", "0") == "1":
        browser_args = [a for a in browser_args if not a.startswith("--headless")]
    for argument in browser_args:
        chrome_options.add_argument(argument)

    # Always pin Chrome profile data to a writable workspace path.
    # Use per-run subdirs to avoid stale lock files between sessions.
    data_dir = settings.browser_data_path
    if data_dir:
        run_key = f"{settings.profile}-{settings.session}"
        safe_run_key = re.sub(r"[^A-Za-z0-9._-]+", "_", run_key)
        data_dir = os.path.join(data_dir, safe_run_key)
        os.makedirs(data_dir, exist_ok=True)
        for stale_name in ("SingletonLock", "SingletonCookie", "SingletonSocket", "DevToolsActivePort"):
            stale_path = os.path.join(data_dir, stale_name)
            if os.path.exists(stale_path):
                try:
                    if os.path.isdir(stale_path):
                        continue
                    os.remove(stale_path)
                except OSError:
                    pass
        chrome_options.add_argument(f"--user-data-dir={data_dir}")
        chrome_options.add_argument("--no-first-run")
        chrome_options.add_argument("--no-default-browser-check")

        # Keep crash data inside workspace to avoid permission-related startup failures
        # on restricted Windows environments.
        crash_dir = os.path.join(settings.output_path, "chrome_crash")
        os.makedirs(crash_dir, exist_ok=True)
        chrome_options.add_argument(f"--crash-dumps-dir={crash_dir}")
        chrome_options.add_argument("--disable-crash-reporter")

    ua = os.environ.get("WEBTEST_USER_AGENT", "").strip()
    if ua:
        chrome_options.add_argument(f"--user-agent={ua}")
    if os.environ.get("WEBTEST_RELAX_HTTPS", "0") == "1":
        for flag in (
            "--ignore-certificate-errors",
            "--ignore-ssl-errors",
            "--allow-running-insecure-content",
            "--test-type",
        ):
            chrome_options.add_argument(flag)

    chrome_options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": "/dev/null",
            "download.prompt_for_download": False,
            "download.directory_upgrade": False,
            "credentials_enable_service": False,
            "profile.password_manager_enabled": False,
            "profile.default_content_setting_values.notifications": 2,
            "profile.default_content_setting_values.popups": 2,
            "protocol_handler.excluded_schemes": {
                "vscode": True,
                "x-github-client": True,
                "github-windows": True,
                "github-mac": True,
                "mailto": True,
                "tel": True,
                "sms": True,
                "intent": True,
            },
        },
    )

    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--noerrdialogs")
    chrome_options.add_argument("--disable-features=ExternalProtocolDialog")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    _inject_headless_stability_args(chrome_options)
    return chrome_options



def configure_global_seed() -> int:
    """Set one deterministic seed for random/numpy/torch for this run."""
    seed_env = os.environ.get("WEBTEST_RUN_SEED", "").strip()
    session = getattr(settings, "session", "") or ""

    seed = None
    if seed_env:
        try:
            seed = int(seed_env)
        except ValueError:
            seed = None

    if seed is None and session:
        m = re.fullmatch(r"seed(\d+)", session.lower())
        if m:
            seed = int(m.group(1))

    if seed is None:
        profile = getattr(settings, "profile", "") or ""
        key = f"{profile}:{session}"
        seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)

    random.seed(seed)

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    os.environ["WEBTEST_EFFECTIVE_SEED"] = str(seed)
    return seed


def _read_env_positive_int(name: str):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _get_transition_count(webtest) -> int:
    getter = getattr(webtest, "get_transition_count", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            pass

    records = getattr(webtest, "transition_record_list", None)
    if records is not None:
        try:
            return int(len(records))
        except Exception:
            pass

    return 0


def main() -> None:
    os.makedirs(settings.output_path, exist_ok=True)
    os.makedirs(settings.model_path, exist_ok=True)
    seed = configure_global_seed()
    logger.info("Global seed initialized: %s (session=%s)", seed, settings.session)
    contract_errors = validate_fair_contract(
        settings,
        getattr(settings, "agent_module", settings.agent["module"]),
        getattr(settings, "agent_class", settings.agent["class"]),
    )
    if contract_errors:
        raise ValueError("Fair mode contract violations: " + "; ".join(contract_errors))

    # Prepend bundled Chrome folder only when not forcing system Chrome.
    force_system_chrome = os.environ.get("WEBTEST_FORCE_SYSTEM_CHROME", "0") == "1"
    browser_dir = os.path.dirname(settings.browser_path) if settings.browser_path else ""
    if (not force_system_chrome) and browser_dir and os.path.isdir(browser_dir):
        os.environ["PATH"] = browser_dir + os.pathsep + os.environ["PATH"]
    chrome_options = configure_chrome_options()

    if settings.agent_num != 1:
        raise ValueError("Multi-agent execution has been removed; use a single-agent profile.")

    webtest = Webtest(chrome_options)
    data_collector = DataCollector(webtest)

    data_collector.start()
    webtest.start()

    # Stop when either time budget is reached or web thread exits early (fail-fast path).
    deadline = time.time() + settings.alive_time
    max_transitions = _read_env_positive_int("WEBTEST_MAX_TRANSITIONS")
    if max_transitions is not None:
        logger.info("Stop condition enabled: WEBTEST_MAX_TRANSITIONS=%d", max_transitions)
    while time.time() < deadline:
        if not webtest.is_alive():
            break
        if max_transitions is not None:
            current_transitions = _get_transition_count(webtest)
            if current_transitions >= max_transitions:
                logger.info(
                    "Stop condition reached: transitions=%d (limit=%d)",
                    current_transitions,
                    max_transitions,
                )
                break
        time.sleep(2)

    webtest.stop()
    data_collector.stop()
    data_collector.join()
    if settings.agent_num == 1 and is_fair_mode():
        schema_errors = 0
        getter = getattr(webtest, "get_fair_schema_error_count", None)
        if callable(getter):
            schema_errors = getter()
        if schema_errors > 0:
            raise RuntimeError(f"Fair mode transition schema errors detected: {schema_errors}")

    if settings.agent_num == 1:
        try:
            agent = getattr(webtest, "agent", None)
            if agent is not None and hasattr(agent, "observer"):
                obs = agent.observer
                if obs.step_log:
                    obs.save_logs()
                    from observation.analyzer import generate_motivation_figures, print_summary_statistics

                    print_summary_statistics("observation_logs")
                    fig_path = os.path.join(settings.output_path, "motivation_figures.pdf")
                    generate_motivation_figures("observation_logs", output_path=fig_path)
        except Exception as e:
            logger.warning("Observation export skipped: %s", e)


if __name__ == "__main__":
    main()
