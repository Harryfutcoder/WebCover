import logging
import hashlib
import json
import os
import os.path
import random
import re
import threading
import time
from datetime import datetime
from typing import Tuple, Optional, List, Dict
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait

from web_test.selenium_util import make_chrome_service

from action.impl.restart_action import RestartAction
from action.web_action import WebAction
from action.web_action_detector import WebActionDetector
from agent.agent import Agent
from config.log_config import LogConfig
from config.settings import settings
from fairness import is_fair_mode, summarize_transitions
from exceptions import NoActionsException
from state.impl.action_execute_failed_state import ActionExecuteFailedState
from state.impl.out_of_domain_state import OutOfDomainState
from state.impl.same_url_state import SameUrlState
from state.web_state import WebState
from utils import instantiate_class_by_module_and_class_name, get_class_by_module_and_class_name, \
    instantiate_class_by_module_and_class_name_and_params, normalize_url_for_state

logger = logging.getLogger(__name__)
logger.addHandler(LogConfig.get_file_handler())


class Webtest(threading.Thread):
    def __init__(self, chrome_options: Options) -> None:
        super().__init__()
        self.chrome_options = chrome_options
        self.driver = self._create_driver_with_fallback(self.chrome_options)
        logger.info("Webdriver created successfully")
        self.action_detector_class: type = get_class_by_module_and_class_name(settings.action_detector["module"],
                                                                              settings.action_detector["class"])
        if settings.action_detector["class"] == "CombinationDetector":
            detectors: List[WebActionDetector] = []
            for detector_name in settings.action_detector["detectors"]:
                detectors.append(
                    instantiate_class_by_module_and_class_name(detector_name["module"], detector_name["class"]))
            self.action_detector: WebActionDetector = self.action_detector_class(detectors)
        else:
            self.action_detector: WebActionDetector = self.action_detector_class()
        self.state_class: type = get_class_by_module_and_class_name(settings.state["module"], settings.state["class"])
        cfg_agent_module = getattr(settings, "agent_module", None)
        cfg_agent_class = getattr(settings, "agent_class", None)
        agent_module = cfg_agent_module if cfg_agent_module is not None else settings.agent["module"]
        agent_class = cfg_agent_class if cfg_agent_class is not None else settings.agent["class"]
        self.agent: Agent = instantiate_class_by_module_and_class_name_and_params(
            agent_module,
            agent_class,
            settings.agent["params"],
        )
        self.prev_state: Optional[WebState] = None
        self.current_state: Optional[WebState] = None
        self.action_dict: Dict[WebAction, int] = {}
        self.state_dict: Dict[WebState, int] = {}
        self.url_count_dict: Dict[str, int] = {}
        self.transition_record_list: List[Tuple[Optional[WebState], WebAction, WebState]] = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.screen_shot_count = 0
        self.screen_shot_record_round = 50
        self.same_url_count = 0
        self.current_url = "111"
        self.restart_interval = settings.restart_interval
        self.empty_action_streak = 0
        self.max_empty_action_streak = int(os.environ.get("WEBTEST_MAX_EMPTY_ACTION_ROUNDS", "120"))
        self.action_detect_retry = int(os.environ.get("WEBTEST_ACTION_DETECT_RETRY", "3"))
        self.action_detect_retry_wait = float(os.environ.get("WEBTEST_ACTION_DETECT_RETRY_WAIT", "1.2"))
        self.browser_error_streak = 0
        self.max_browser_error_streak = int(os.environ.get("WEBTEST_MAX_BROWSER_ERROR_ROUNDS", "25"))
        self.http_5xx_streak = 0
        self.max_http_5xx_streak = int(os.environ.get("WEBTEST_MAX_HTTP_5XX_ROUNDS", "20"))
        expected_http_500_urls = json.loads(os.environ.get("WEBTEST_EXPECTED_HTTP_500_URLS", "[]"))
        if (not isinstance(expected_http_500_urls, list)
                or not all(isinstance(url, str) and urlparse(url).scheme in {"http", "https"}
                           and urlparse(url).netloc for url in expected_http_500_urls)):
            raise ValueError("WEBTEST_EXPECTED_HTTP_500_URLS must be a JSON list of exact HTTP(S) URLs")
        self.expected_http_500_urls = frozenset(expected_http_500_urls)
        logger.info("HTTP 500 guard exemptions (logs retained): %s", sorted(self.expected_http_500_urls))
        self.http_5xx_as_failed_state = os.environ.get(
            "WEBTEST_HTTP_5XX_AS_FAILED_STATE", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        self.filter_known_noise = os.environ.get("WEBTEST_FILTER_KNOWN_NOISE", "1") == "1"
        self.restart_policy = os.environ.get("WEBTEST_RESTART_POLICY", "entry").strip().lower()
        self.seen_restart_action_enabled = (
            os.environ.get("WEBTEST_SEEN_RESTART_ACTION", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        if is_fair_mode():
            self.seen_restart_action_enabled = False
        self.seen_restart_action_min_steps = max(
            0,
            int(os.environ.get("WEBTEST_SEEN_RESTART_ACTION_MIN_STEPS", "50")),
        )
        self.seen_restart_action_interval = max(
            1,
            int(os.environ.get("WEBTEST_SEEN_RESTART_ACTION_INTERVAL", "25")),
        )
        self._known_noise_patterns = [
            "failed to execute 'matches' on 'element': '[open]:not(:modal)' is not a valid selector",
            "deprecated api for given entry type",
            "a matching frame for #repo-content-turbo-frame was missing from the response",
            "error: not connected to alive",
        ]
        self._fair_schema_error_count = 0
        self.code_coverage_enabled = os.environ.get("WEBTEST_CODE_COVERAGE", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.code_coverage_url = self._resolve_code_coverage_url() if self.code_coverage_enabled else ""
        self.code_coverage_interval = max(1, int(os.environ.get("WEBTEST_CODE_COVERAGE_INTERVAL", "1")))
        self.code_coverage_timeout = max(1, int(os.environ.get("WEBTEST_CODE_COVERAGE_TIMEOUT", "5")))
        self.code_coverage_sync_browser = os.environ.get(
            "WEBTEST_CODE_COVERAGE_SYNC_BROWSER",
            "1",
        ).strip().lower() in ("1", "true", "yes", "on")
        self.code_coverage_step = 0
        self.last_code_coverage: Dict[str, str] = {}
        if self.code_coverage_enabled:
            if self.code_coverage_url:
                logger.info(
                    "CodeCoverage enabled url=%s interval=%d timeout=%d sync_browser=%s",
                    self.code_coverage_url,
                    self.code_coverage_interval,
                    self.code_coverage_timeout,
                    self.code_coverage_sync_browser,
                )
            else:
                logger.warning("CodeCoverage requested but no coverage URL could be resolved.")

    def _resolve_code_coverage_url(self) -> str:
        explicit = os.environ.get("WEBTEST_CODE_COVERAGE_URL", "").strip()
        if explicit:
            return explicit.rstrip("/")
        entry = getattr(settings, "entry_url", "") or ""
        env_key = ""
        if "localhost:8084" in entry:
            env_key = "AGILEFANT"
            default_url = "http://localhost:6969"
        elif "localhost:3001" in entry:
            env_key = "GADAEL"
            default_url = "http://localhost:6970"
        elif "localhost:3002" in entry:
            env_key = "TIMEOFF"
            default_url = "http://localhost:6971"
        elif "localhost:4005" in entry or "localhost:4200" in entry:
            env_key = "SPLITTYPIE"
            default_url = "http://localhost:6972"
        elif "localhost:4002" in entry or "localhost:8081" in entry:
            env_key = "PETCLINIC"
            default_url = "http://localhost:6973"
        else:
            default_url = ""
        if env_key:
            site_url = os.environ.get(f"WEBTEST_SITE_{env_key}_COVERAGE_URL", "").strip()
            if site_url:
                return site_url.rstrip("/")
        return default_url.rstrip("/")

    @staticmethod
    def _parse_coverage_response(raw: str) -> Dict[str, str]:
        text = (raw or "").strip()
        if not text:
            return {}
        if text.startswith("{"):
            data = json.loads(text)
            return {
                "branch_coverage": str(data.get("branch_coverage", "")),
                "line_coverage": str(data.get("line_coverage", "")),
                "line_coverage_info": str(data.get("line_coverage_info", "")),
            }
        metrics: Dict[str, str] = {}
        for value, label in re.findall(
            r'<span class="strong">\s*([^<]+?)\s*</span>\s*<span class="quiet">\s*([^<]+?)\s*</span>',
            text,
            flags=re.I,
        ):
            key = label.strip().lower()
            if key.startswith("branch"):
                metrics["branch_coverage"] = value.strip()
            elif key.startswith("line"):
                metrics["line_coverage"] = value.strip()
            elif key.startswith("statement"):
                metrics["statement_coverage"] = value.strip()
            elif key.startswith("function"):
                metrics["function_coverage"] = value.strip()
        return metrics

    def _sync_browser_istanbul_coverage(self, reason: str) -> None:
        if not self.code_coverage_sync_browser:
            return
        driver = getattr(self, "driver", None)
        if driver is None:
            return
        try:
            coverage = driver.execute_script("return window.__coverage__ || null;")
        except Exception as exc:
            logger.debug("CodeCoverage browser_sync skipped reason=%s err=%s", reason, exc)
            return
        if not isinstance(coverage, dict) or not coverage:
            return
        post_url = self.code_coverage_url.rstrip("/") + "/coverage/client"
        try:
            payload = json.dumps(coverage).encode("utf-8")
            request = Request(
                post_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=self.code_coverage_timeout) as response:
                response.read()
            logger.debug(
                "CodeCoverage browser_sync reason=%s files=%d url=%s",
                reason,
                len(coverage),
                post_url,
            )
        except Exception as exc:
            logger.warning("CodeCoverage browser_sync failed reason=%s url=%s err=%s", reason, post_url, exc)

    def _record_code_coverage(self, reason: str = "step") -> None:
        if not self.code_coverage_enabled or not self.code_coverage_url:
            return
        self.code_coverage_step += 1
        if reason == "step" and self.code_coverage_step % self.code_coverage_interval != 0:
            return
        url = self.code_coverage_url.rstrip("/") + "/coverage"
        try:
            self._sync_browser_istanbul_coverage(reason)
            with urlopen(url, timeout=self.code_coverage_timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
            metrics = self._parse_coverage_response(raw)
            if not metrics:
                logger.warning("CodeCoverage empty_or_unparsed url=%s reason=%s", url, reason)
                return
            self.last_code_coverage = metrics
            logger.info(
                "CodeCoverage reason=%s branch_coverage=%s line_coverage=%s "
                "statement_coverage=%s function_coverage=%s url=%s",
                reason,
                metrics.get("branch_coverage", ""),
                metrics.get("line_coverage", ""),
                metrics.get("statement_coverage", ""),
                metrics.get("function_coverage", ""),
                url,
            )
        except Exception as exc:
            logger.warning("CodeCoverage fetch failed reason=%s url=%s err=%s", reason, url, exc)

    @staticmethod
    def _clone_options(options: Options) -> Options:
        new_options = Options()
        if getattr(options, "binary_location", None):
            new_options.binary_location = options.binary_location
        for arg in getattr(options, "arguments", []):
            new_options.add_argument(arg)
        for k, v in getattr(options, "experimental_options", {}).items():
            new_options.add_experimental_option(k, v)
        return new_options

    @staticmethod
    def _has_headless_flag(options: Options) -> bool:
        return any(str(arg).startswith("--headless") for arg in getattr(options, "arguments", []))

    @staticmethod
    def _ensure_headless_new(options: Options) -> Options:
        cloned = Webtest._clone_options(options)
        if not Webtest._has_headless_flag(cloned):
            cloned.add_argument("--headless=new")
        return cloned

    @staticmethod
    def _replace_headless_new_with_legacy(options: Options) -> Options:
        cloned = Webtest._clone_options(options)
        args = []
        replaced = False
        for arg in cloned.arguments:
            if arg == "--headless=new":
                args.append("--headless")
                replaced = True
            else:
                args.append(arg)
        if not replaced:
            return cloned
        patched = Options()
        if getattr(cloned, "binary_location", None):
            patched.binary_location = cloned.binary_location
        for arg in args:
            patched.add_argument(arg)
        for k, v in getattr(cloned, "experimental_options", {}).items():
            patched.add_experimental_option(k, v)
        return patched

    @staticmethod
    def _remove_headless_flags(options: Options) -> Options:
        cloned = Webtest._clone_options(options)
        args = [a for a in cloned.arguments if not a.startswith("--headless")]
        patched = Options()
        if getattr(cloned, "binary_location", None):
            patched.binary_location = cloned.binary_location
        for arg in args:
            patched.add_argument(arg)
        for k, v in getattr(cloned, "experimental_options", {}).items():
            patched.add_experimental_option(k, v)
        return patched

    @staticmethod
    def _clear_binary_location(options: Options) -> Options:
        cloned = Webtest._clone_options(options)
        patched = Options()
        for arg in cloned.arguments:
            patched.add_argument(arg)
        for k, v in getattr(cloned, "experimental_options", {}).items():
            patched.add_experimental_option(k, v)
        return patched

    @staticmethod
    def _strip_browser_dir_from_path() -> None:
        browser_path = getattr(settings, "browser_path", "") or ""
        browser_dir = os.path.dirname(browser_path) if browser_path else ""
        if not browser_dir:
            return
        current_path = os.environ.get("PATH", "")
        if not current_path:
            return
        normalized_target = os.path.normcase(os.path.normpath(browser_dir))
        filtered_parts = []
        for part in current_path.split(os.pathsep):
            p = part.strip()
            if not p:
                continue
            if os.path.normcase(os.path.normpath(p)) == normalized_target:
                continue
            filtered_parts.append(part)
        os.environ["PATH"] = os.pathsep.join(filtered_parts)

    @staticmethod
    def _guess_system_chrome_binary() -> Optional[str]:
        candidates = []
        for root in (
            os.environ.get("PROGRAMFILES", ""),
            os.environ.get("PROGRAMFILES(X86)", ""),
            os.environ.get("LOCALAPPDATA", ""),
        ):
            if not root:
                continue
            candidates.append(os.path.join(root, "Google", "Chrome", "Application", "chrome.exe"))
            candidates.append(os.path.join(root, "Google", "Chrome Beta", "Application", "chrome.exe"))
            candidates.append(os.path.join(root, "Chromium", "Application", "chrome.exe"))
        for path in candidates:
            if path and os.path.isfile(path):
                return path
        return None

    def _create_driver_with_fallback(self, base_options: Options):
        try:
            return webdriver.Chrome(options=base_options, service=make_chrome_service())
        except WebDriverException as e:
            msg = str(e).lower()
            if "devtoolsactiveport" not in msg and "chrome failed to start" not in msg:
                raise

            legacy_source = base_options
            if not self._has_headless_flag(base_options):
                logger.warning(
                    "Chrome failed in headful mode; retrying with headless=new. Error=%s",
                    e,
                )
                legacy_source = self._ensure_headless_new(base_options)
                try:
                    return webdriver.Chrome(options=legacy_source, service=make_chrome_service())
                except WebDriverException as e_headless:
                    logger.warning(
                        "Chrome failed with headless=new; retrying with legacy headless flag. Error=%s",
                        e_headless,
                    )
            else:
                logger.warning(
                    "Chrome failed with current options; retrying with legacy headless flag. Error=%s",
                    e,
                )

            legacy_options = self._replace_headless_new_with_legacy(legacy_source)
            try:
                return webdriver.Chrome(options=legacy_options, service=make_chrome_service())
            except WebDriverException as e2:
                logger.warning(
                    "Chrome failed with legacy headless; retrying in headful mode. Error=%s",
                    e2,
                )
                headful_options = self._remove_headless_flags(base_options)
                try:
                    return webdriver.Chrome(options=headful_options, service=make_chrome_service())
                except WebDriverException as e3:
                    if getattr(base_options, "binary_location", None):
                        logger.warning(
                            "Bundled Chrome failed to start; retrying with system Chrome. Error=%s",
                            e3,
                        )
                        self._strip_browser_dir_from_path()
                        system_options = self._clear_binary_location(base_options)
                        guessed_binary = self._guess_system_chrome_binary()
                        if guessed_binary:
                            system_options.binary_location = guessed_binary
                        if not self._has_headless_flag(system_options):
                            system_options = self._ensure_headless_new(system_options)
                        try:
                            return webdriver.Chrome(options=system_options, service=make_chrome_service())
                        except WebDriverException as e4:
                            logger.warning(
                                "System Chrome fallback also failed. Error=%s",
                                e4,
                            )
                            raise e3
                    raise

    def _open_entry_url_with_retry(self, retries: int = 3, wait_seconds: float = 2.0) -> bool:
        for attempt in range(1, retries + 1):
            try:
                self.driver.get(settings.entry_url)
                self._fix_gadael_base_url()
                return True
            except WebDriverException as e:
                if self._is_fatal_network_error(e):
                    logger.error(
                        "Abort run early: fatal network error while opening %s: %s",
                        settings.entry_url,
                        e,
                    )
                    return False
                logger.warning(
                    "Failed to open entry URL %s (attempt %d/%d): %s",
                    settings.entry_url,
                    attempt,
                    retries,
                    e,
                )
                if attempt < retries:
                    self.stop_event.wait(wait_seconds)
        return False

    @staticmethod
    def _is_gadael_site() -> bool:
        profile = (getattr(settings, "profile", "") or "").lower()
        entry = (getattr(settings, "entry_url", "") or "").lower()
        return "gadael" in profile or "localhost:3001" in entry

    def _fix_gadael_base_url(self, route_path: Optional[str] = None, reload_route: bool = False) -> bool:
        """Patch Gadael's hard-coded public base URL from localhost:3000 to the served origin."""
        if not self._is_gadael_site():
            return False
        try:
            return bool(self.driver.execute_script(
                """
                const routePath = arguments[0];
                const reloadRoute = arguments[1];
                const fixed = window.location.origin + '/';
                const base = document.querySelector('base');
                if (base) {
                  base.setAttribute('href', fixed);
                  base.setAttribute('ng-href', fixed);
                }
                if (window.requirejs) {
                  window.requirejs.config({ baseUrl: fixed + 'js/' });
                  if (reloadRoute) {
                    [
                      'controllers/home',
                      'controllers/login/index',
                      'controllers/login/createfirstadmin'
                    ].forEach((moduleName) => {
                      try { window.requirejs.undef(moduleName); } catch (e) {}
                    });
                  }
                }
                const injector = window.angular
                  && window.angular.element(document.body).injector
                  && window.angular.element(document.body).injector();
                if (injector) {
                  const root = injector.get('$rootScope');
                  root.baseUrl = fixed;
                  if (routePath || reloadRoute) {
                    const location = injector.get('$location');
                    const route = injector.get('$route');
                    if (routePath) {
                      location.path(routePath);
                    }
                    try { root.$apply(); } catch (e) {}
                    if (reloadRoute) {
                      setTimeout(() => {
                        try {
                          route.reload();
                          root.$apply();
                        } catch (e) {}
                      }, 0);
                    }
                  }
                }
                return true;
                """,
                route_path,
                bool(reload_route),
            ))
        except Exception as e:
            logger.debug("Gadael base-url patch skipped: %s", e)
            return False

    def _gadael_runtime_status(self) -> Dict[str, object]:
        if not self._is_gadael_site():
            return {}
        try:
            status = self.driver.execute_script(
                """
                const visible = (el) => {
                  if (!el) return false;
                  const style = window.getComputedStyle(el);
                  const rect = el.getBoundingClientRect();
                  return style
                    && style.visibility !== 'hidden'
                    && style.display !== 'none'
                    && rect.width > 0
                    && rect.height > 0;
                };
                const actions = Array.from(document.querySelectorAll(
                  'a,button,input[type="button"],input[type="submit"],input[type="checkbox"],input[type="radio"],summary'
                )).filter(visible);
                let authenticated = false;
                let menuReady = false;
                try {
                  const injector = window.angular
                    && window.angular.element(document.body).injector
                    && window.angular.element(document.body).injector();
                  if (injector) {
                    const root = injector.get('$rootScope');
                    authenticated = !!(root.sessionUser && root.sessionUser.isAuthenticated);
                    menuReady = !!(root.menu && (root.menu.admin || root.menu.user || root.menu.account));
                  }
                } catch (e) {}
                const text = ((document.body && document.body.innerText) || '').trim().toLowerCase();
                const blankRootLinks = actions.filter((el) => {
                  const href = (el.href || '').split('#')[0].replace(/\\/$/, '/');
                  const label = ((el.innerText || el.value || '') + '').trim().toLowerCase();
                  return href === (window.location.origin + '/') && (label === '' || label === 'loading...');
                }).length;
                return {
                  url: window.location.href,
                  text: text.slice(0, 200),
                  actionCount: actions.length,
                  blankRootLinks,
                  authenticated,
                  menuReady,
                  loading: text.includes('loading') || text === ''
                };
                """
            )
            return dict(status or {})
        except Exception as e:
            logger.debug("Gadael runtime status unavailable: %s", e)
            return {}

    def _wait_for_gadael_ready(self, timeout: float = 12.0) -> bool:
        if not self._is_gadael_site():
            return True
        deadline = time.time() + timeout
        last_status: Dict[str, object] = {}
        while time.time() < deadline:
            self._fix_gadael_base_url()
            last_status = self._gadael_runtime_status()
            action_count = int(last_status.get("actionCount") or 0)
            blank_root_links = int(last_status.get("blankRootLinks") or 0)
            authenticated = bool(last_status.get("authenticated"))
            menu_ready = bool(last_status.get("menuReady"))
            loading = bool(last_status.get("loading"))
            if authenticated and not loading and (menu_ready or action_count > max(1, blank_root_links)):
                return True
            self.stop_event.wait(0.5)
        logger.debug("Gadael readiness wait ended without ready status: %s", last_status)
        return False

    def _looks_like_gadael_placeholder_actions(self, action_list: List[WebAction]) -> bool:
        if not self._is_gadael_site():
            return False
        if len(action_list) == 1:
            action = action_list[0]
            text = (getattr(action, "text", "") or "").strip().lower()
            action_type = (getattr(action, "action_type", "") or "").strip().lower()
            addition_info = str(getattr(action, "addition_info", "") or "")
            location = str(getattr(action, "location", "") or "")
            root_url = "http://localhost:3001/"
            if (
                action_type == "redirect"
                and text in ("", "loading...")
                and self._normalize_url_for_state(addition_info) == root_url
                and location.endswith("/a")
            ):
                return True
        if len(action_list) == 0:
            status = self._gadael_runtime_status()
            url = str(status.get("url") or "")
            loading = bool(status.get("loading"))
            action_count = int(status.get("actionCount") or 0)
            blank_root_links = int(status.get("blankRootLinks") or 0)
            if "localhost:3001" in url and loading and action_count <= max(1, blank_root_links):
                return True
        return False

    def _recover_gadael_placeholder_state(self) -> bool:
        if not self._is_gadael_site():
            return False
        logger.warning("Gadael placeholder/root-only state detected; refreshing authenticated #/home route.")
        try:
            self._fix_gadael_base_url("/home", reload_route=True)
            if self._wait_for_gadael_ready(timeout=6):
                return True
            self.driver.get("http://localhost:3001/#/home")
            self.stop_event.wait(3)
            self._fix_gadael_base_url("/home", reload_route=True)
            if self._wait_for_gadael_ready(timeout=8):
                return True
            self._pre_login()
            return self._wait_for_gadael_ready(timeout=10)
        except Exception as e:
            logger.warning("Gadael placeholder recovery failed: %s", e)
            return False

    @staticmethod
    def _is_fatal_network_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        markers = (
            "err_connection_refused",
            "err_connection_timed_out",
            "err_timed_out",
            "err_name_not_resolved",
            "err_internet_disconnected",
            "err_address_unreachable",
            "err_proxy_connection_failed",
            "err_tunnel_connection_failed",
            "net::err_cert_authority_invalid",
            "connection refused",
            "dns probe finished nxdomain",
        )
        return any(m in msg for m in markers)

    @staticmethod
    def _normalize_url_for_state(raw_url: str) -> str:
        return normalize_url_for_state(raw_url)

    @staticmethod
    def _graph_form_state_enabled() -> bool:
        raw = os.environ.get("WEBTEST_GRAPH_FORM_STATE", "1").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    @staticmethod
    def _graph_form_value_buckets_enabled() -> bool:
        raw = os.environ.get("WEBTEST_GRAPH_FORM_VALUE_BUCKETS", "0").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _bucket_count(value: int) -> str:
        value = max(0, int(value))
        if value <= 2:
            return str(value)
        if value <= 5:
            return "3-5"
        if value <= 10:
            return "6-10"
        return "10+"

    def _capture_graph_form_state_id(self) -> str:
        """Summarize visible form progress without recording concrete values."""
        if not self._graph_form_state_enabled():
            return ""
        try:
            fields = self.driver.execute_script(
                """
                const blockedTypes = new Set(['button', 'hidden', 'image', 'reset', 'submit']);
                const isVisible = (el) => {
                  const style = window.getComputedStyle(el);
                  const rect = el.getBoundingClientRect();
                  return style.display !== 'none' &&
                         style.visibility !== 'hidden' &&
                         (rect.width > 0 || rect.height > 0 || el.getClientRects().length > 0);
                };
                const shortPath = (el) => {
                  const parts = [];
                  let cur = el;
                  while (cur && cur.nodeType === 1 && parts.length < 5) {
                    const tag = cur.tagName.toLowerCase();
                    let idx = 0;
                    let sib = cur;
                    while ((sib = sib.previousElementSibling)) {
                      if (sib.tagName === cur.tagName) idx += 1;
                    }
                    parts.unshift(`${tag}:${idx}`);
                    cur = cur.parentElement;
                  }
                  return parts.join('/');
                };
                return Array.from(document.querySelectorAll('input, textarea, select'))
                  .filter((el) => {
                    const tag = el.tagName.toLowerCase();
                    const type = (el.getAttribute('type') || tag).toLowerCase();
                    return !blockedTypes.has(type) &&
                           !el.disabled &&
                           !el.readOnly &&
                           isVisible(el);
                  })
                  .slice(0, 64)
                  .map((el) => {
                    const tag = el.tagName.toLowerCase();
                    const type = (el.getAttribute('type') || tag).toLowerCase();
                    const rawValue = (el.value || '').trim();
                    const filled = type === 'checkbox' || type === 'radio' ? !!el.checked : rawValue.length > 0;
                    const selectChanged = tag === 'select' && el.selectedIndex > 0;
                    const key = [
                      tag,
                      type,
                      (el.getAttribute('name') || '').toLowerCase(),
                      (el.id || '').toLowerCase(),
                      (el.getAttribute('placeholder') || '').toLowerCase(),
                      (el.getAttribute('aria-label') || '').toLowerCase(),
                      shortPath(el)
                    ].join('|');
                    return {
                      key,
                      filled,
                      required: !!el.required,
                      selectChanged,
                      lenBucket: rawValue.length === 0 ? 0 : (rawValue.length <= 2 ? 1 : (rawValue.length <= 10 ? 2 : 3))
                    };
                  });
                """
            )
        except Exception:
            return ""
        if not fields:
            return ""

        include_value_buckets = self._graph_form_value_buckets_enabled()
        lines = []
        filled_count = 0
        required_count = 0
        required_filled_count = 0
        select_changed_count = 0
        for field in fields:
            key = str(field.get("key", ""))
            filled = 1 if field.get("filled") else 0
            required = 1 if field.get("required") else 0
            select_changed = 1 if field.get("selectChanged") else 0
            len_bucket = int(field.get("lenBucket") or 0)
            filled_count += filled
            required_count += required
            required_filled_count += 1 if required and filled else 0
            select_changed_count += select_changed
            line = f"{key}|filled={filled}|required={required}|select={select_changed}"
            if include_value_buckets:
                line = f"{line}|len={len_bucket}"
            lines.append(line)
        digest = hashlib.sha1("\n".join(sorted(lines)).encode("utf-8", errors="ignore")).hexdigest()[:16]
        return (
            f"fields={self._bucket_count(len(fields))}"
            f"|filled={self._bucket_count(filled_count)}"
            f"|required={self._bucket_count(required_count)}"
            f"|required_filled={self._bucket_count(required_filled_count)}"
            f"|select_changed={self._bucket_count(select_changed_count)}"
            f"|sig={digest}"
        )

    @staticmethod
    def _copy_dynamic_state_observation(target_state: WebState, source_state: WebState) -> WebState:
        for attr in ("graph_form_state_id",):
            try:
                setattr(target_state, attr, getattr(source_state, attr, ""))
            except Exception:
                pass
        return target_state

    def _build_state_for_url(self, action_list, normalized_url: str, raw_url: str):
        state = self.state_class(action_list, normalized_url)
        try:
            state.raw_url = raw_url
        except Exception:
            pass
        try:
            state.graph_form_state_id = self._capture_graph_form_state_id()
        except Exception:
            state.graph_form_state_id = ""
        return state

    def _pick_restart_url(self) -> str:
        """
        Choose restart URL with explicit policy.

        Default "entry" avoids hidden exploration bias from restarting to
        least-visited URLs.
        """
        entry_url = settings.entry_url
        if not self.url_count_dict:
            return entry_url

        if self.restart_policy == "least_visited":
            return min(self.url_count_dict, key=self.url_count_dict.get)
        if self.restart_policy == "most_visited":
            return max(self.url_count_dict, key=self.url_count_dict.get)
        if self.restart_policy == "random_seen":
            return random.choice(list(self.url_count_dict.keys()))
        return entry_url

    def _maybe_add_seen_restart_action(self, action_list: List[WebAction], current_url: str) -> List[WebAction]:
        """Optionally expose a zero-reward control action back to a seen URL.

        This is disabled by default and by fair mode. It is intended for
        controlled hard-exploration variants where RestartAction is treated like
        reset or home: a control transition that may recover access to a
        previously discovered context, but should not create graph-edge coverage
        reward.
        """
        if not self.seen_restart_action_enabled:
            return action_list
        step_count = self.get_transition_count()
        if step_count < self.seen_restart_action_min_steps:
            return action_list
        if step_count % self.seen_restart_action_interval != 0:
            return action_list
        if len(self.url_count_dict) <= 1:
            return action_list

        restart_url = self._pick_restart_url()
        if not restart_url:
            return action_list
        current_norm = self._normalize_url_for_state(current_url)
        if restart_url in {current_url, current_norm}:
            return action_list

        restart_action = RestartAction(restart_url)
        if restart_action in action_list:
            return action_list
        return list(action_list) + [restart_action]

    def _is_timeoff_site(self) -> bool:
        return "localhost:3002" in (getattr(settings, "entry_url", "") or "").lower()

    def _reset_agent_context_after_forced_restart(self) -> None:
        """
        Break cross-episode links after webdriver/process-level recovery.

        Without this reset, the next state after a forced browser restart can be
        incorrectly treated as a direct successor of a pre-crash action.
        """
        # Learning agents may maintain per-episode trajectories. Try to close/clear the
        # current episode first, then clear generic transition memory.
        if hasattr(self.agent, "_flush_episode"):
            try:
                self.agent._flush_episode()
            except Exception as e:
                logger.warning("Skip agent episode flush after forced restart: %s", e)

        for attr in ("previous_state", "previous_action", "previous_html", "pending_step"):
            if hasattr(self.agent, attr):
                try:
                    setattr(self.agent, attr, None)
                except Exception:
                    pass

        for attr in ("stop_update", "stop_update_round"):
            if hasattr(self.agent, attr):
                try:
                    setattr(self.agent, attr, False)
                except Exception:
                    pass

        if hasattr(self.agent, "_reset_rnn_hidden"):
            try:
                self.agent._reset_rnn_hidden()
            except Exception as e:
                logger.warning("Skip RNN hidden reset after forced restart: %s", e)

    def _looks_like_auth_wall(self) -> bool:
        auth_prefixes = (
            "/login",
            "/signup",
            "/sign-in",
            "/signin",
            "/register",
            "/password_reset",
            "/session",
            "/sessions",
            "/oauth",
            "/auth",
        )
        auth_markers = (
            "log in",
            "login",
            "sign in",
            "signin",
            "register",
            "sign up",
            "forgot password",
            "reset password",
        )

        current_url = (getattr(self.driver, "current_url", "") or "").lower()
        try:
            parsed = urlparse(current_url)
            path = (parsed.path or "").lower()
        except Exception:
            path = current_url
        if any(path.startswith(prefix) for prefix in auth_prefixes):
            return True

        try:
            if not self.driver.find_elements(By.XPATH, "//input[@type='password']"):
                return False
            page_source = (self.driver.page_source or "").lower()
            return any(marker in page_source for marker in auth_markers)
        except Exception:
            return False

    def _pre_login(self) -> bool:
        """Auto-login or auto-register for sites that require authentication."""
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException, NoSuchElementException

        entry = settings.entry_url.lower()

        def _as_xpath_list(xpaths):
            if isinstance(xpaths, str):
                return [xpaths]
            return list(xpaths)

        def _wait_and_fill(xpaths, value, timeout=10):
            for xpath in _as_xpath_list(xpaths):
                try:
                    el = WebDriverWait(self.driver, timeout).until(
                        EC.presence_of_element_located((By.XPATH, xpath))
                    )
                    el.clear()
                    el.send_keys(value)
                    return True
                except (TimeoutException, NoSuchElementException):
                    continue
            return False

        def _wait_and_click(xpaths, timeout=10):
            for xpath in _as_xpath_list(xpaths):
                try:
                    el = WebDriverWait(self.driver, timeout).until(
                        EC.element_to_be_clickable((By.XPATH, xpath))
                    )
                    try:
                        el.click()
                    except WebDriverException:
                        self.driver.execute_script("arguments[0].click();", el)
                    return True
                except (TimeoutException, NoSuchElementException, WebDriverException):
                    continue
            return False

        def _submit_form_fallback():
            script = """
                const selectors = [
                  'button[type="submit"]',
                  'input[type="submit"]',
                  'button[data-login-form-submit]',
                  'form button',
                  'form input[type="submit"]'
                ];
                for (const selector of selectors) {
                  const node = document.querySelector(selector);
                  if (node) {
                    node.click();
                    return true;
                  }
                }
                const form = document.querySelector('form');
                if (form) {
                  if (typeof form.requestSubmit === 'function') form.requestSubmit();
                  else form.submit();
                  return true;
                }
                return false;
            """
            try:
                return bool(self.driver.execute_script(script))
            except Exception:
                return False

        def _wait_until_not_auth(timeout=12):
            end_time = time.time() + timeout
            while time.time() < end_time:
                if not self._looks_like_auth_wall():
                    return True
                self.stop_event.wait(0.8)
            return not self._looks_like_auth_wall()

        def _return_to_entry(url=None):
            self.driver.get(url or settings.entry_url)
            self.stop_event.wait(2)
            self._fix_gadael_base_url()

        def _env_enabled(name: str, default: str = "1") -> bool:
            return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")

        def _bootstrap_splittypie_event() -> bool:
            base_url = settings.entry_url.rstrip("/")
            self.driver.get(base_url + "/new")
            self.stop_event.wait(3)

            event_name = f"WebTest {os.environ.get('WEBTEST_RUN_SEED', 'seed')}"
            created = bool(self.driver.execute_script(
                """
                const eventName = arguments[0];
                const visible = (el) => {
                  const rect = el.getBoundingClientRect();
                  const style = window.getComputedStyle(el);
                  return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
                };
                const fire = (el, type) => el.dispatchEvent(new Event(type, { bubbles: true }));
                const setValue = (el, value) => {
                  el.scrollIntoView({ block: 'center', inline: 'center' });
                  el.focus();
                  el.value = value;
                  fire(el, 'input');
                  fire(el, 'change');
                  el.blur();
                };

                const textInputs = Array.from(document.querySelectorAll('input'))
                  .filter(visible)
                  .filter((el) => {
                    const type = (el.getAttribute('type') || 'text').toLowerCase();
                    return ['text', 'search', 'email', 'url', 'number', ''].includes(type);
                  });
                const values = [eventName, 'Alice', 'Bob', 'Carol', 'Dave'];
                textInputs.forEach((el, i) => setValue(el, values[i] || `User${i + 1}`));

                Array.from(document.querySelectorAll('select')).filter(visible).forEach((el) => {
                  if (el.options && el.options.length > 1) {
                    el.selectedIndex = 1;
                    fire(el, 'change');
                  }
                });

                const buttons = Array.from(document.querySelectorAll('button, input[type="submit"]')).filter(visible);
                const createButton = buttons.find((el) => {
                  const label = ((el.innerText || el.value || '') + '').trim().toLowerCase();
                  return label.includes('create') && !label.includes('cancel');
                });
                if (createButton) {
                  createButton.scrollIntoView({ block: 'center', inline: 'center' });
                  createButton.click();
                  return true;
                }
                const form = document.querySelector('form');
                if (form) {
                  if (typeof form.requestSubmit === 'function') form.requestSubmit();
                  else form.submit();
                  return true;
                }
                return false;
                """,
                event_name,
            ))
            if not created:
                return False

            end_time = time.time() + 12
            while time.time() < end_time:
                current_url = (self.driver.current_url or "").rstrip("/")
                if current_url and current_url != base_url and current_url != base_url + "/new":
                    settings.entry_url = self.driver.current_url
                    return True
                self.stop_event.wait(0.8)
            return False

        login_button_xpaths = (
            "//button[@type='submit']",
            "//input[@type='submit']",
            "//*[self::button or self::a][contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'log in')]",
            "//*[self::button or self::a][contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'login')]",
            "//*[self::button or self::a][contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign in')]",
        )
        register_button_xpaths = (
            "//button[@type='submit']",
            "//input[@type='submit']",
            "//*[self::button or self::a][contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'register')]",
            "//*[self::button or self::a][contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign up')]",
        )

        if "localhost:4200" in entry and _env_enabled("WEBTEST_SPLITTYPIE_BOOTSTRAP_EVENT", "1"):
            try:
                if _bootstrap_splittypie_event():
                    logger.info(
                        "pre_login: splittypie event bootstrap complete, run entry_url=%s",
                        settings.entry_url,
                    )
                else:
                    _return_to_entry()
                    logger.warning("pre_login: splittypie event bootstrap failed, url=%s", self.driver.current_url)
            except Exception as e:
                logger.warning("pre_login: splittypie event bootstrap failed: %s", e)

        elif "localhost:8069" in entry:
            try:
                base_url = "http://localhost:8069"
                database = os.environ.get("WEBTEST_ODOO_DB", "")
                login = os.environ["WEBTEST_ODOO_LOGIN"]
                password = os.environ["WEBTEST_ODOO_PASSWORD"]
                db_query = f"?db={quote(database, safe='')}" if database else ""
                login_url = base_url + "/web/login" + db_query
                app_url = base_url + "/web" + db_query

                self.driver.get(login_url)
                self.stop_event.wait(4)
                _wait_and_fill((
                    "//*[@id='login']",
                    "//input[@name='login']",
                    "//input[@type='email']",
                    "//input[@type='text']",
                ), login, timeout=10)
                _wait_and_fill((
                    "//*[@id='password']",
                    "//input[@name='password']",
                    "//input[@type='password']",
                ), password, timeout=10)
                if not _wait_and_click((
                    "//button[@type='submit']",
                    "//input[@type='submit']",
                    "//button[contains(@class,'btn-primary')]",
                ) + login_button_xpaths, timeout=10):
                    _submit_form_fallback()
                self.stop_event.wait(6)

                if _wait_until_not_auth(timeout=12):
                    settings.entry_url = app_url
                    _return_to_entry(app_url)
                    if self._looks_like_auth_wall():
                        logger.warning("pre_login: odoo app auth wall still active, url=%s", self.driver.current_url)
                        return False
                    logger.info(
                        "pre_login: odoo login complete, db=%s login=%s url=%s",
                        database,
                        login,
                        self.driver.current_url,
                    )
                    return True
                logger.warning("pre_login: odoo auth wall still active, url=%s", self.driver.current_url)
                return False
            except Exception as e:
                logger.warning("pre_login: odoo failed: %s", e)
                return False

        elif "localhost:3000" in entry:
            try:
                # Step 1: try login with demo account
                self.driver.get("http://localhost:3000/login")
                self.stop_event.wait(4)
                _wait_and_fill((
                    "//input[@name='emailOrUsername']",
                    "//input[@name='username']",
                    "//input[@name='email']",
                    "//input[@type='email']",
                ), os.environ["WEBTEST_4GABOARDS_EMAIL"])
                _wait_and_fill((
                    "//input[@name='password']",
                    "//input[@type='password']",
                ), os.environ["WEBTEST_4GABOARDS_PASSWORD"])
                if not _wait_and_click(login_button_xpaths):
                    _submit_form_fallback()
                self.stop_event.wait(5)
                if not _wait_until_not_auth(timeout=10):
                    # Step 2: login failed — register the demo account
                    logger.info("pre_login: 4gaboards login failed, attempting registration")
                    self.driver.get("http://localhost:3000/register")
                    self.stop_event.wait(4)
                    _wait_and_fill((
                        "//input[@name='username']",
                        "//input[@placeholder='Username']",
                    ), os.environ["WEBTEST_4GABOARDS_USERNAME"])
                    _wait_and_fill((
                        "//input[@name='email']",
                        "//input[@type='email']",
                        "//input[@placeholder='Email']",
                    ), os.environ["WEBTEST_4GABOARDS_EMAIL"])
                    _wait_and_fill((
                        "//input[@name='password']",
                        "//input[@type='password']",
                        "//input[@placeholder='Password']",
                    ), os.environ["WEBTEST_4GABOARDS_PASSWORD"])
                    # accept policy checkbox if present
                    try:
                        cb = self.driver.find_element(By.XPATH,
                            "//input[@type='checkbox' and (@name='policy' or contains(@id,'policy') or contains(@id,'terms'))]")
                        if not cb.is_selected():
                            cb.click()
                    except Exception:
                        pass
                    if not _wait_and_click(register_button_xpaths):
                        _submit_form_fallback()
                    self.stop_event.wait(6)
                    # after registration Taiga may auto-login or redirect to login
                    if self._looks_like_auth_wall():
                        self.driver.get("http://localhost:3000/login")
                        self.stop_event.wait(3)
                        _wait_and_fill((
                            "//input[@name='emailOrUsername']",
                            "//input[@name='username']",
                            "//input[@name='email']",
                        ), os.environ["WEBTEST_4GABOARDS_EMAIL"])
                        _wait_and_fill((
                            "//input[@name='password']",
                            "//input[@type='password']",
                        ), os.environ["WEBTEST_4GABOARDS_PASSWORD"])
                        if not _wait_and_click(login_button_xpaths):
                            _submit_form_fallback()
                        self.stop_event.wait(5)
                if _wait_until_not_auth(timeout=10):
                    _return_to_entry()
                    logger.info("pre_login: 4gaboards login complete, url=%s", self.driver.current_url)
                else:
                    logger.warning("pre_login: 4gaboards auth wall still active, url=%s", self.driver.current_url)
            except Exception as e:
                logger.warning("pre_login: 4gaboards failed: %s", e)

        elif "localhost:8084" in entry:
            try:
                base_url = "http://localhost:8084/agilefant"
                landing = os.environ.get("WEBTEST_AGILEFANT_LANDING", "dailyWork.action").lstrip("/")
                app_url = base_url + "/" + landing
                credential_pairs = []

                def _add_credential(username, password):
                    pair = ((username or "").strip(), (password or "").strip())
                    if pair[0] and pair[1] and pair not in credential_pairs:
                        credential_pairs.append(pair)

                _add_credential(
                    os.environ.get("WEBTEST_AGILEFANT_USERNAME", ""),
                    os.environ.get("WEBTEST_AGILEFANT_PASSWORD", ""),
                )

                self.driver.get(app_url)
                self.stop_event.wait(3)
                if _wait_until_not_auth(timeout=5):
                    settings.entry_url = app_url
                    _return_to_entry(app_url)
                    logger.info("pre_login: agilefant existing session url=%s", self.driver.current_url)
                    return True

                last_error = ""
                for username, password in credential_pairs:
                    try:
                        self.driver.get(base_url + "/login.jsp")
                        self.stop_event.wait(2)
                        if not _wait_and_fill(('//*[@id="username"]', "//input[@name='j_username']"), username):
                            last_error = f"user={username} username field missing"
                            continue
                        if not _wait_and_fill(('//input[@name="j_password"]', "//input[@type='password']"), password):
                            last_error = f"user={username} password field missing"
                            continue
                        if not _wait_and_click(("//input[@type='submit']",) + login_button_xpaths):
                            _submit_form_fallback()
                        self.stop_event.wait(4)
                        self.driver.get(app_url)
                        self.stop_event.wait(3)
                        if _wait_until_not_auth(timeout=10):
                            settings.entry_url = app_url
                            _return_to_entry(app_url)
                            logger.info(
                                "pre_login: agilefant login complete, user=%s url=%s",
                                username,
                                self.driver.current_url,
                            )
                            return True
                        last_error = f"user={username} url={self.driver.current_url}"
                    except Exception as cred_error:
                        last_error = f"user={username} err={cred_error}"
                logger.warning("pre_login: agilefant auth wall still active: %s", last_error)
                return False
            except Exception as e:
                logger.warning("pre_login: agilefant failed: %s", e)
                return False

        elif "localhost:3002" in entry:
            try:
                base_url = "http://localhost:3002"
                calendar_url = base_url + "/calendar/"

                def _timeoff_login() -> None:
                    self.driver.get(base_url + "/login")
                    self.stop_event.wait(3)
                    _wait_and_fill(('//*[@id="email_inp"]', "//input[@name='username']"), os.environ["WEBTEST_TIMEOFF_EMAIL"])
                    _wait_and_fill(('//*[@id="pass_inp"]', "//input[@name='password']"), os.environ["WEBTEST_TIMEOFF_PASSWORD"])
                    if not _wait_and_click(('//*[@id="submit_login"]',) + login_button_xpaths):
                        _submit_form_fallback()
                    self.stop_event.wait(4)

                _timeoff_login()
                if not _wait_until_not_auth(timeout=8):
                    self.driver.get(base_url + "/register")
                    self.stop_event.wait(3)
                    _wait_and_fill('//*[@id="company_name_inp"]', "TestCompany")
                    _wait_and_fill('//*[@id="name_inp"]', "Secret")
                    _wait_and_fill('//*[@id="lastname_inp"]', "User")
                    _wait_and_fill('//*[@id="email_inp"]', os.environ["WEBTEST_TIMEOFF_EMAIL"])
                    _wait_and_fill('//*[@id="pass_inp"]', os.environ["WEBTEST_TIMEOFF_PASSWORD"])
                    _wait_and_fill('//*[@id="confirm_pass_inp"]', os.environ["WEBTEST_TIMEOFF_PASSWORD"])
                    if not _wait_and_click(register_button_xpaths):
                        _submit_form_fallback()
                    self.stop_event.wait(5)
                    if self._looks_like_auth_wall():
                        _timeoff_login()
                if _wait_until_not_auth(timeout=10):
                    settings.entry_url = calendar_url
                    _return_to_entry(calendar_url)
                    if self._looks_like_auth_wall():
                        _timeoff_login()
                        settings.entry_url = calendar_url
                        _return_to_entry(calendar_url)
                    if self._looks_like_auth_wall():
                        logger.warning("pre_login: timeoff calendar auth wall still active, url=%s", self.driver.current_url)
                        return False
                    logger.info("pre_login: timeoff login complete, url=%s", self.driver.current_url)
                    return True
                else:
                    logger.warning("pre_login: timeoff auth wall still active, url=%s", self.driver.current_url)
                    return False
            except Exception as e:
                logger.warning("pre_login: timeoff failed: %s", e)
                return False

        elif "localhost:3001" in entry:
            try:
                base_url = "http://localhost:3001"
                email = os.environ["WEBTEST_GADAEL_EMAIL"]
                password = os.environ["WEBTEST_GADAEL_PASSWORD"]

                def _gadael_logged_in() -> bool:
                    try:
                        return bool(self.driver.execute_script(
                            """
                            const injector = window.angular
                              && window.angular.element(document.body).injector
                              && window.angular.element(document.body).injector();
                            if (!injector) return false;
                            const root = injector.get('$rootScope');
                            return !!(root.sessionUser && root.sessionUser.isAuthenticated);
                            """
                        ))
                    except Exception:
                        return False

                def _goto_gadael_route(route_path: str) -> None:
                    self.driver.get(base_url + "/#" + route_path)
                    self.stop_event.wait(2)
                    self._fix_gadael_base_url(route_path, reload_route=True)
                    self.stop_event.wait(4)

                _goto_gadael_route("/login/createfirstadmin")
                if _wait_and_fill('//*[@id="user_firstname"]', "Secret", timeout=4):
                    _wait_and_fill('//*[@id="user_lastname"]', "User", timeout=4)
                    _wait_and_fill('//*[@id="user_email"]', email, timeout=4)
                    _wait_and_fill('//*[@id="user_password"]', password, timeout=4)
                    _wait_and_fill('//*[@id="user_password2"]', password, timeout=4)
                    if not _wait_and_click((
                        "//*[@id='user_password2']/ancestor::form//button[contains(@class,'btn-primary')]",
                        "//button[contains(@class,'btn-primary') and contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'save')]",
                    ), timeout=6):
                        _submit_form_fallback()
                    self.stop_event.wait(6)
                    self._fix_gadael_base_url("/home", reload_route=True)
                    self.stop_event.wait(2)

                if not _gadael_logged_in():
                    _goto_gadael_route("/login")
                    _wait_and_fill((
                        "//input[@name='username']",
                        "//input[@ng-model='username']",
                        "//input[@type='text']",
                    ), email, timeout=8)
                    _wait_and_fill((
                        "//input[@name='password']",
                        "//input[@type='password']",
                    ), password, timeout=8)
                    if not _wait_and_click((
                        "//button[@type='submit']",
                        "//button[contains(@class,'btn-login')]",
                    ), timeout=8):
                        _submit_form_fallback()
                    self.stop_event.wait(6)
                    self._fix_gadael_base_url("/home", reload_route=True)
                    self.stop_event.wait(2)

                settings.entry_url = base_url + "/#/home"
                _return_to_entry(settings.entry_url)
                self._fix_gadael_base_url("/home", reload_route=True)
                self.stop_event.wait(2)
                try:
                    self.driver.get_log("browser")
                except Exception:
                    pass
                logger.info(
                    "pre_login: gadael init complete, logged_in=%s, url=%s",
                    _gadael_logged_in(),
                    self.driver.current_url,
                )
            except Exception as e:
                logger.warning("pre_login: gadael failed: %s", e)

        elif "realworld.show" in entry or "localhost:4203" in entry:
            try:
                base_url = "https://demo.realworld.show" if "realworld.show" in entry else "http://localhost:4203"
                self.driver.get(base_url + "/#/register")
                self.stop_event.wait(8)
                _wait_and_fill("//input[@placeholder='Username']", os.environ["WEBTEST_REALWORLD_USERNAME"], timeout=12)
                _wait_and_fill("//input[@placeholder='Email']", os.environ["WEBTEST_REALWORLD_EMAIL"], timeout=8)
                _wait_and_fill("//input[@placeholder='Password']", os.environ["WEBTEST_REALWORLD_PASSWORD"], timeout=8)
                _wait_and_click(register_button_xpaths, timeout=8)
                self.stop_event.wait(4)
                if self._looks_like_auth_wall():
                    self.driver.get(base_url + "/#/login")
                    self.stop_event.wait(6)
                    _wait_and_fill("//input[@placeholder='Email']", os.environ["WEBTEST_REALWORLD_EMAIL"], timeout=12)
                    _wait_and_fill("//input[@placeholder='Password']", os.environ["WEBTEST_REALWORLD_PASSWORD"], timeout=8)
                    _wait_and_click(login_button_xpaths, timeout=8)
                    self.stop_event.wait(4)
                if _wait_until_not_auth(timeout=8):
                    _return_to_entry()
                    logger.info("pre_login: realworld login complete, url=%s", self.driver.current_url)
                else:
                    logger.warning("pre_login: realworld auth wall still active, url=%s", self.driver.current_url)
            except Exception as e:
                logger.warning("pre_login: realworld failed: %s", e)

        elif "localhost:8082" in entry:
            try:
                self.driver.get("http://localhost:8082/login")
                self.stop_event.wait(3)
                _wait_and_fill(('//*[@id="user"]', "//input[@name='user']"), os.environ["WEBTEST_NEXTCLOUD_USERNAME"], timeout=14)
                _wait_and_fill(('//*[@id="password"]', "//input[@name='password']"), os.environ["WEBTEST_NEXTCLOUD_PASSWORD"], timeout=14)
                if not _wait_and_click(("//button[@type='submit']", "//input[@type='submit']", "//*[@data-login-form-submit]"), timeout=14):
                    _submit_form_fallback()
                self.stop_event.wait(5)
                if _wait_until_not_auth(timeout=12):
                    _return_to_entry("http://localhost:8082/apps/files/")
                    logger.info("pre_login: nextcloud login complete, url=%s", self.driver.current_url)
                else:
                    logger.warning("pre_login: nextcloud auth wall still active, url=%s", self.driver.current_url)
            except Exception as e:
                logger.warning("pre_login: nextcloud failed: %s", e)

        return True

    def run(self):
        logger.info("Execution start")
        if not self._open_entry_url_with_retry():
            logger.error("Abort run early: cannot open entry URL %s", settings.entry_url)
            self.stop_event.set()
            self.driver.quit()
            return
        self.stop_event.wait(10)
        if not self._pre_login() and self._is_timeoff_site():
            logger.error("Abort run early: timeoff pre_login failed at %s", self.driver.current_url)
            self.stop_event.set()
            self.driver.quit()
            return
        if self._is_gadael_site():
            self._wait_for_gadael_ready(timeout=12)
        try:
            html = self.init_state()
        except Exception:
            logger.exception("Failed to initialize state at entry URL: %s", settings.entry_url)
            self.stop_event.set()
            self.driver.quit()
            return
        self._record_code_coverage(reason="initial")
        self.driver.set_page_load_timeout(settings.page_load_timeout)
        continuous_restart_count = 0
        while not self.stop_event.is_set():
            chosen_action = None
            try:
                chosen_action = self.agent.get_action(self.current_state, html)
                if isinstance(chosen_action, RestartAction) and getattr(
                    chosen_action,
                    "_webexplor_dfa_control_reset",
                    False,
                ):
                    logger.info(
                        "WebExplor DFA control reset to %s; reset is not recorded as an action transition",
                        chosen_action.url,
                    )
                    continuous_restart_count = 0
                    chosen_action.execute(self.driver)
                    WebDriverWait(self.driver, settings.page_load_timeout).until(
                        lambda x: x.execute_script('return document.readyState') == 'complete'
                    )
                    self.stop_event.wait(2)
                    try:
                        html = self.init_state()
                    except Exception:
                        logger.exception(
                            "Failed to reinitialize state after WebExplor DFA control reset"
                        )
                        self.stop_event.set()
                    continue
                logger.info(f"Chosen action: {chosen_action}")
                if not isinstance(chosen_action, RestartAction):
                    self.action_dict[chosen_action] = self.action_dict.get(chosen_action, 0) + 1
                    continuous_restart_count = 0
                else:
                    continuous_restart_count += 1
                if continuous_restart_count >= settings.continuous_restart_threshold:
                    self.restart_webdriver()
                    continuous_restart_count = 0
                    if self.stop_event.is_set():
                        continue
                chosen_action.execute(self.driver)
                WebDriverWait(self.driver, settings.page_load_timeout).until(
                    lambda x: x.execute_script('return document.readyState') == 'complete'
                )
                self.stop_event.wait(2)
                check_result, domain = self.check_domain()
                if check_result and self.same_url_count < self.restart_interval:
                    if self.is_browser_error_page():
                        self.browser_error_streak += 1
                        logger.warning(
                            "Browser error page detected on %s (streak=%d)",
                            self.driver.current_url,
                            self.browser_error_streak,
                        )
                        if self.browser_error_streak >= self.max_browser_error_streak:
                            logger.error(
                                "Abort run early: browser error page persisted for %d rounds",
                                self.browser_error_streak,
                            )
                            self.stop_event.set()
                            continue
                    else:
                        self.browser_error_streak = 0

                    normalized_url = self._normalize_url_for_state(self.driver.current_url)
                    with self.lock:
                        if normalized_url in self.url_count_dict:
                            self.url_count_dict[normalized_url] += 1
                        else:
                            self.url_count_dict[normalized_url] = 1
                    has_http_5xx = self.trace_error()
                    if self.http_5xx_streak >= self.max_http_5xx_streak:
                        logger.error(
                            "Abort run early: persistent HTTP 5xx errors (streak=%d) on %s",
                            self.http_5xx_streak,
                            self.driver.current_url,
                        )
                        self.stop_event.set()
                        continue
                    if self.http_5xx_as_failed_state and has_http_5xx:
                        restart_url = self._pick_restart_url()
                        self.url_count_dict[restart_url] = self.url_count_dict.get(restart_url, 0) + 1
                        new_state = ActionExecuteFailedState(restart_url)
                        self.transit(chosen_action, new_state)
                        if settings.enable_screen_shot:
                            self.save_screen_shot()
                        logger.warning(
                            "HTTP 5xx treated as failed transition, New state: %s",
                            new_state,
                        )
                        continue
                    action_list = self.detect_actions_with_retry()
                    if len(action_list) == 0:
                        if self._is_timeoff_site():
                            if self._pre_login():
                                action_list = self.detect_actions_with_retry()
                            if len(action_list) == 0:
                                logger.error(
                                    "Abort run early: timeoff has no actions after pre_login recovery at %s",
                                    self.driver.current_url,
                                )
                                self.stop_event.set()
                                continue
                        if action_list:
                            self.empty_action_streak = 0
                        else:
                            self.empty_action_streak += 1
                            logger.warning(
                                "No actionable elements detected on %s (streak=%d)",
                                self.driver.current_url,
                                self.empty_action_streak,
                            )
                            if self.empty_action_streak >= self.max_empty_action_streak:
                                logger.error(
                                    "Abort run early: no actions for %d consecutive rounds on %s",
                                    self.empty_action_streak,
                                    self.driver.current_url,
                                )
                                self.stop_event.set()
                                continue
                    else:
                        self.empty_action_streak = 0
                    action_list = self._maybe_add_seen_restart_action(action_list, self.driver.current_url)
                    with self.lock:
                        for action in action_list:
                            self.action_dict.setdefault(action, 0)
                    if normalized_url == self.current_url:
                        self.same_url_count += 1
                    else:
                        self.same_url_count = 0
                        self.current_url = normalized_url
                    fresh_state = self._build_state_for_url(action_list, normalized_url, self.driver.current_url)
                    new_state = self._copy_dynamic_state_observation(
                        self.get_state(fresh_state),
                        fresh_state,
                    )
                    html = self.driver.page_source
                    self.transit(chosen_action, new_state)
                    if settings.enable_screen_shot:
                        self.save_screen_shot()
                    logger.info(f"Execute action success, New state: {new_state}")
                    self._record_code_coverage(reason="step")
                elif not check_result:
                    self.browser_error_streak = 0
                    self.http_5xx_streak = 0
                    restart_url = self._pick_restart_url()
                    self.url_count_dict[restart_url] = self.url_count_dict.get(restart_url, 0) + 1
                    # 手动+1避免卡死
                    new_state = OutOfDomainState(restart_url)
                    self.transit(chosen_action, new_state)
                    if settings.enable_screen_shot:
                        self.save_screen_shot()
                    logger.warning(f"Out of domain, New state: {new_state}")
                else:
                    self.browser_error_streak = 0
                    self.http_5xx_streak = 0
                    self.same_url_count = 0
                    restart_url = self._pick_restart_url()
                    self.url_count_dict[restart_url] = self.url_count_dict.get(restart_url, 0) + 1
                    # 手动+1避免卡死
                    new_state = SameUrlState(restart_url)
                    self.transit(chosen_action, new_state)
                    if settings.enable_screen_shot:
                        self.save_screen_shot()
                    logger.warning(f"Same url too many times, New state: {new_state}")
            except (NoActionsException, WebDriverException) as e:
                restart_url = self._pick_restart_url()
                new_state = ActionExecuteFailedState(restart_url)
                self.transit(chosen_action, new_state)
                # 如果重启仍然失败，多增加一些计数，避免多次在同一个url重启
                if chosen_action is not None and isinstance(chosen_action, RestartAction):
                    self.url_count_dict[restart_url] = self.url_count_dict.get(restart_url, 0) + 1
                if isinstance(e, NoActionsException):
                    logger.warning(f"Choose action failed, no actions, New state: {new_state}")
                    if self._is_timeoff_site() and self._pre_login():
                        try:
                            html = self.init_state()
                            self.empty_action_streak = 0
                            continue
                        except Exception:
                            logger.exception("Failed to recover timeoff after no-actions state")
                else:
                    logger.warning(f"Execute action failed ({type(e).__name__}), New state: {new_state}")
                if isinstance(e, WebDriverException) and self._is_fatal_network_error(e):
                    logger.error(
                        "Abort run early: fatal network error on target service (%s): %s",
                        settings.entry_url,
                        e,
                    )
                    self.stop_event.set()
            except (Exception, KeyboardInterrupt) as e:
                logger.exception(f"Error happened")
                self.restart_webdriver()
                continuous_restart_count = 0
                if not self._open_entry_url_with_retry():
                    logger.error("Abort run early: cannot re-open entry URL %s", settings.entry_url)
                    self.stop_event.set()
                    continue
                self.stop_event.wait(2)
                try:
                    html = self.init_state()
                except Exception:
                    logger.exception("Failed to re-initialize state after webdriver restart")
                    self.stop_event.set()
                    continue
                self.empty_action_streak = 0
                self.browser_error_streak = 0
                self.http_5xx_streak = 0
            if len(self.driver.window_handles) > 1:
                self.driver.switch_to.window(self.driver.window_handles[1])
                self.close_other_windows()

        if hasattr(self.agent, "finalize_run"):
            try:
                self.agent.finalize_run(self.current_state, reason="run_end")
            except Exception as e:
                logger.warning("Skip agent finalization at run end: %s", e)

        self._record_code_coverage(reason="final")
        self.driver.quit()
        if is_fair_mode():
            report = summarize_transitions(self.transition_record_list)
            self._fair_schema_error_count = int(report.get("schema_error_count", 0))
            logger.info("Fairness transition summary: %s", report)

    def init_state(self):
        action_list = self.detect_actions_with_retry()
        action_list = self._maybe_add_seen_restart_action(action_list, self.driver.current_url)
        normalized_url = self._normalize_url_for_state(self.driver.current_url)
        fresh_state = self._build_state_for_url(action_list, normalized_url, self.driver.current_url)
        self.current_state = self._copy_dynamic_state_observation(
            self.get_state(fresh_state),
            fresh_state,
        )
        with self.lock:
            for action in action_list:
                self.action_dict.setdefault(action, 0)
            self.state_dict[self.current_state] = 1
        html: str = self.driver.page_source
        with self.lock:
            self.url_count_dict[normalized_url] = 1
        self.trace_error()
        logger.info(f"Initial state: {self.current_state}")
        return html

    def is_browser_error_page(self) -> bool:
        cur = (self.driver.current_url or "").lower()
        if cur.startswith("chrome-error://"):
            return True
        src = (self.driver.page_source or "").lower()
        title = (self.driver.title or "").lower()
        if "chrome-error://" in src:
            return True
        # Chrome interstitial (privacy / cert / network) often keeps target URL but renders internal error DOM.
        if self.driver.find_elements(By.ID, "main-message"):
            return True
        if self.driver.find_elements(By.ID, "details-button"):
            return True
        if self.driver.find_elements(By.ID, "proceed-link"):
            return True
        if self.driver.find_elements(By.ID, "error-code"):
            return True
        if "this site can" in title and "reached" in title:
            return True
        if "err_" in src and "chrome-error" in src:
            return True
        if "your connection is not private" in title or "privacy error" in title:
            return True
        if "您的连接不是私密连接" in src or "隐私错误" in src:
            return True
        if "返回安全连接" in src and "高级" in src:
            return True
        return False

    def detect_actions_with_retry(self):
        action_list = []
        for i in range(max(1, self.action_detect_retry)):
            self._fix_gadael_base_url()
            action_list = self.action_detector.get_actions(self.driver)
            if self._looks_like_gadael_placeholder_actions(action_list):
                if self._recover_gadael_placeholder_state():
                    action_list = self.action_detector.get_actions(self.driver)
                else:
                    action_list = []
            if action_list:
                return action_list
            if i < self.action_detect_retry - 1:
                self.stop_event.wait(self.action_detect_retry_wait)
        return action_list

    def _is_expected_http_500(self, entry):
        message = str(entry.get("message", ""))
        resource_url = message.split(None, 1)[0] if message.strip() else ""
        return (
            entry.get("source") == "network"
            and "status of 500" in message.lower()
            and resource_url in getattr(self, "expected_http_500_urls", frozenset())
        )

    def trace_error(self):
        logs = self.driver.get_log("browser")
        has_5xx = False
        has_unexpected_5xx = False
        dedup = set()
        with open(os.path.join(settings.output_path, "bug.log"), "a", encoding="utf-8") as f:
            for log in logs:
                msg = str(log.get("message", "")).lower()
                level = log.get("level", "")
                if (level == "WARNING") or (level == "SEVERE"):
                    is_known_noise = self.filter_known_noise and any(
                        p in msg for p in self._known_noise_patterns
                    )
                    if not is_known_noise:
                        dedup_key = (level, msg)
                        if dedup_key not in dedup:
                            dedup.add(dedup_key)
                            logger.info(f"Detect browser error: {log}")
                            f.write(str(log) + "\n")
                if (
                    "status of 500" in msg
                    or "status of 502" in msg
                    or "status of 503" in msg
                    or "status of 504" in msg
                    or "http error 5" in msg
                ):
                    has_5xx = True
                    if not self._is_expected_http_500(log):
                        has_unexpected_5xx = True
        if has_unexpected_5xx:
            self.http_5xx_streak += 1
            logger.warning("HTTP 5xx detected (streak=%d)", self.http_5xx_streak)
        else:
            self.http_5xx_streak = 0
        return has_5xx

    def add_new_state_to_list(self, new_state: WebState) -> WebState:
        for state in self.state_dict.keys():
            if state == new_state:
                with self.lock:
                    self.state_dict[state] += 1
                return state
        with self.lock:
            self.state_dict[new_state] = 1
        return new_state

    def transit(self, chosen_action: WebAction, new_state: WebState) -> None:
        new_state = self.add_new_state_to_list(new_state)
        if chosen_action is not None and self.current_state is not None:
            try:
                self.current_state.update_action_execution_time(chosen_action)
            except Exception as e:
                logger.warning(
                    "Skip action execution-time update: action not in current state (action=%s, state=%s, err=%s)",
                    chosen_action,
                    self.current_state,
                    e,
                )
            try:
                self.current_state.update_transition_information(chosen_action, new_state)
            except Exception as e:
                logger.warning(
                    "Skip transition info update: action not in current state (action=%s, state=%s, err=%s)",
                    chosen_action,
                    self.current_state,
                    e,
                )
        self.prev_state = self.current_state
        self.current_state = new_state
        with self.lock:
            self.transition_record_list.append((self.prev_state, chosen_action, self.current_state))

    def check_domain(self) -> Tuple[bool, str]:
        current_url = self.driver.current_url
        domain = urlparse(current_url).netloc.lower()
        for allowed in settings.domains:
            if domain == allowed or domain.endswith("." + allowed):
                return True, domain
        return False, domain

    def close_other_windows(self) -> None:
        current_window = self.driver.current_window_handle
        for handle in self.driver.window_handles:
            if handle != current_window:
                self.driver.switch_to.window(handle)
                self.driver.close()
        self.driver.switch_to.window(current_window)

    def restart_webdriver(self) -> None:
        self.driver.quit()
        self.driver = self._create_driver_with_fallback(self.chrome_options)
        self.driver.set_page_load_timeout(settings.page_load_timeout)
        # Forced browser restart breaks transition continuity.
        # Always cut agent short-term memory here to avoid false transitions.
        self._reset_agent_context_after_forced_restart()
        if not self._pre_login() and self._is_timeoff_site():
            logger.error("Webdriver restart failed timeoff pre_login at %s", self.driver.current_url)
            self.stop_event.set()
        logger.info("Webdriver restart successfully")

    def stop(self):
        self.stop_event.set()

    def get_transition_count(self) -> int:
        with self.lock:
            return len(self.transition_record_list)

    def get_fair_schema_error_count(self) -> int:
        return int(self._fair_schema_error_count)

    def save_screen_shot(self):
        if self.screen_shot_count % self.screen_shot_record_round != 0:
            self.screen_shot_count += 1
            return

        original_window_size = None
        try:
            # Save the original window size
            original_window_size = self.driver.get_window_size()

            # Execute JavaScript to get the full page height
            js = "return Math.max( document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight);"
            scroll_height = int(self.driver.execute_script(js))
            if scroll_height <= 0:
                scroll_height = original_window_size["height"]

            # Set the window size to the full page height and take the screenshot
            self.driver.set_window_size(original_window_size['width'], scroll_height)
            screenshot = self.driver.get_screenshot_as_png()

            folder_path = os.path.join(settings.output_path, "ScreenShots")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_path = os.path.join(folder_path,
                                     "screenShot-" + f"{self.screen_shot_count}" + "---" + datetime.now().strftime(
                                         "%Y-%m-%d_%H_%M_%S") + ".png")
            with open(file_path, "wb") as file:
                file.write(screenshot)
        except Exception as e:
            logger.warning("Skip screenshot at step=%d due to %s: %s", self.screen_shot_count, type(e).__name__, e)
        finally:
            if original_window_size is not None:
                try:
                    self.driver.set_window_size(original_window_size['width'], original_window_size['height'])
                except Exception:
                    pass
            self.screen_shot_count += 1

    def get_state(self, web_state: WebState) -> WebState:
        with self.lock:
            if web_state not in self.state_dict:
                self.state_dict[web_state] = 0
                return web_state

        for state in self.state_dict.keys():
            if state == web_state:
                return state
