import json
import os
import re
import subprocess
import time
from urllib.parse import urljoin, urlparse

import requests

from src import settings
from src.webenv.utils import init_browser, write_file
from playwright._impl._errors import Error, TargetClosedError


class WebConfig:
    def __init__(self, app_name: str, url: str, domain: str, flag_coverage: bool):
        # config (config means fixed)
        ## web app
        self.app_name = app_name
        self.online = True
        self.url = url
        self.domain = domain
        self.domain_scopes = self._parse_domain_scopes(domain)
        ## cov
        self.flag_coverage = flag_coverage
        env_key = self._env_key(app_name)
        env_cov_type = os.environ.get(f"WEBTEST_SITE_{env_key}_COV_TYPE", "").strip()
        env_benchmark = os.environ.get(f"WEBTEST_SITE_{env_key}_BENCHMARK", "").strip()
        self.cov_type = (env_cov_type or settings.WEB_INFO[app_name]['cov_type']) if flag_coverage else 'error'
        self.benchmark = (env_benchmark or settings.WEB_INFO[app_name]['benchmark']) if flag_coverage else 'complex'
        self.coverage_base_url = self._resolve_coverage_base_url(app_name).rstrip('/')
        if flag_coverage and self.cov_type in ('nyc', 'jacoco') and not self.coverage_base_url:
            raise ValueError(
                f"Coverage requested for {app_name}, but no live coverage endpoint is configured. "
                f"Set WEBTEST_SITE_{self._env_key(app_name)}_COVERAGE_URL after starting the "
                "instrumented listener, or run without --coverage for post-run coverage sites."
            )
        if self.cov_type == 'nyc':
            self.update_cov_and_errors = self.update_cov_and_errors_nyc
        elif self.cov_type == 'jacoco':
            self.update_cov_and_errors = self.update_cov_and_errors_jacoco
        else:
            assert self.cov_type == 'error'
            self.update_cov_and_errors = self.update_errors
        ## driver
        self.width, self.height = 1200, 1200
        self.action_timeout = 500
        ## file
        # self.errors_file = os.path.join(settings.APP_DIR_PATH, 'errors.txt')
        # runtime
        ## cov
        self.old_branch_coverage, self.old_line_coverage = 0, 0
        self.branch_coverage, self.line_coverage = 0, 0
        self.errors = set()
        self.errors_detail = []
        self.errors_buffer = []
        self.log_set = set()
        self._auth_recovery_active = False
        ## driver
        self.pw, self.browser, self.context, self.page = init_browser(self.width, self.height)
        self.context.on("page", lambda page: self.handle_new_page(page))
        self.page.on("console", lambda msg: self.handle_errors(msg))  # console error
        self.page.on("pageerror", lambda exc: self.handle_page_error(exc))
        self.reset_page()

    @staticmethod
    def _env_key(app_name: str) -> str:
        return (app_name or "").upper().replace("-", "_")

    @classmethod
    def _resolve_coverage_base_url(cls, app_name: str) -> str:
        env_key = cls._env_key(app_name)
        return (
            os.environ.get(f"WEBTEST_SITE_{env_key}_COVERAGE_URL", "").strip()
            or os.environ.get("WEBTEST_COVERAGE_BASE_URL", "").strip()
            or settings.WEB_INFO.get(app_name, {}).get('coverage_url', '')
        )

    @staticmethod
    def _parse_domain_scopes(domain: str) -> list[str]:
        scopes = [part.strip() for part in (domain or "").split(",") if part.strip()]
        return scopes or [domain]

    @staticmethod
    def _strip_www(host: str) -> str:
        host = (host or "").lower()
        return host[4:] if host.startswith("www.") else host

    def is_url_in_scope(self, url: str | None = None) -> bool:
        current_url = url or self.page.url or ""
        if not current_url or current_url == "about:blank":
            return False
        parsed_current = urlparse(current_url)
        current_host = self._strip_www(parsed_current.netloc)
        for scope in self.domain_scopes:
            if scope and scope in current_url:
                return True
            parsed_scope = urlparse(scope if "://" in scope else f"//{scope}")
            scope_host = self._strip_www(parsed_scope.netloc)
            scope_path = parsed_scope.path.rstrip("/")
            if scope_host and current_host == scope_host:
                if not scope_path or parsed_current.path.rstrip("/").startswith(scope_path):
                    return True
        return False

    def _auth_path_prefixes(self) -> tuple[str, ...]:
        raw = os.environ.get(
            "WEBTEST_AUTH_PATH_PREFIXES",
            "/login,/logout,/signup,/sign-in,/sign-out,/signin,/signout,/register,"
            "/password_reset,/session,/sessions,/oauth,/auth",
        )
        return tuple(prefix.strip().lower() for prefix in raw.split(",") if prefix.strip())

    def _is_auth_url(self, url: str) -> bool:
        parsed = urlparse(url)
        candidates = [parsed.path.lower()]
        if parsed.fragment:
            fragment = parsed.fragment.lower()
            candidates.append(fragment if fragment.startswith("/") else f"/{fragment}")
        for candidate in candidates:
            for prefix in self._auth_path_prefixes():
                normalized_prefix = prefix.rstrip("/")
                if candidate == normalized_prefix or candidate.startswith(normalized_prefix + "/"):
                    return True
        return False

    def _is_auth_or_logout_action(self, action_info: dict) -> bool:
        html = (action_info.get("outerHTML") or "").lower()
        label = (action_info.get("innerText") or "").lower()
        auth_terms = (
            "logout", "log out", "sign out", "signout",
            "login", "log in", "sign in", "signin",
            "signup", "sign up", "register",
        )
        if any(term in html or term in label for term in auth_terms):
            return True
        for attr_value in re.findall(r"""(?:href|action)=["']([^"']+)["']""", html, flags=re.I):
            absolute_url = urljoin(self.page.url or self.url, attr_value)
            if self._is_auth_url(absolute_url):
                return True
        return False

    def _pw_fill(self, selectors, value, timeout=10000):
        """Try multiple CSS/XPath selectors, fill the first visible one."""
        if isinstance(selectors, str):
            selectors = [selectors]
        for sel in selectors:
            try:
                el = self.page.wait_for_selector(sel, timeout=timeout)
                if el:
                    el.fill(value)
                    return True
            except Exception:
                continue
        return False

    def _pw_click(self, selectors, timeout=10000):
        if isinstance(selectors, str):
            selectors = [selectors]
        for sel in selectors:
            try:
                el = self.page.wait_for_selector(sel, timeout=timeout)
                if el:
                    el.click()
                    return True
            except Exception:
                continue
        return False

    def _pw_current_url_has(self, *fragments):
        url = self.page.url or ""
        return any(f in url for f in fragments)

    def _normalized_app_name(self) -> str:
        return (self.app_name or "").strip().lower()

    def _origin(self) -> str:
        parsed = urlparse(self.url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
        return self.url.rstrip("/")

    def _app_base_url(self, marker: str) -> str:
        parsed = urlparse(self.url)
        if not (parsed.scheme and parsed.netloc):
            return self.url.rstrip("/")
        path_parts = [part for part in parsed.path.split("/") if part]
        marker = marker.strip("/")
        if marker in path_parts:
            idx = path_parts.index(marker)
            base_path = "/" + "/".join(path_parts[:idx + 1])
        else:
            base_path = "/" + marker
        return f"{parsed.scheme}://{parsed.netloc}{base_path}".rstrip("/")

    def _goto(self, url: str, wait_ms: int = 3000):
        self.page.goto(url, wait_until="domcontentloaded")
        self.page.wait_for_timeout(wait_ms)

    def _wait_until(self, predicate, timeout_ms: int = 10000, interval_ms: int = 300) -> bool:
        deadline = time.time() + (timeout_ms / 1000.0)
        while time.time() < deadline:
            try:
                if predicate():
                    return True
            except Exception:
                pass
            self.page.wait_for_timeout(interval_ms)
        try:
            return bool(predicate())
        except Exception:
            return False

    def _submit_form_fallback(self) -> bool:
        try:
            return bool(self.page.evaluate(
                """
                () => {
                  const form = document.querySelector('form');
                  if (!form) return false;
                  if (typeof form.requestSubmit === 'function') {
                    form.requestSubmit();
                  } else {
                    form.submit();
                  }
                  return true;
                }
                """
            ))
        except Exception:
            return False

    def _looks_like_auth_wall(self) -> bool:
        url = (self.page.url or "").lower()
        url_markers = ("login", "sign-in", "signin", "register", "signup", "auth", "session")
        if any(marker in url for marker in url_markers):
            return True
        try:
            has_password = self.page.locator("input[type='password']").count() > 0
        except Exception:
            has_password = False
        if not has_password:
            return False
        try:
            body_text = (self.page.text_content("body") or "").lower()
        except Exception:
            body_text = ""
        text_markers = ("log in", "login", "sign in", "register", "sign up", "forgot password")
        return any(marker in body_text for marker in text_markers)

    def _wait_until_not_auth(self, timeout_ms: int = 10000) -> bool:
        return self._wait_until(lambda: not self._looks_like_auth_wall(), timeout_ms=timeout_ms)

    def ensure_authenticated(self) -> bool:
        app_name = self._normalized_app_name()
        if app_name not in {
            '4gaboards', 'agilefant', 'gadael', 'nextcloud',
            'realworld', 'splittypie', 'timeoff'
        }:
            return False
        if self._auth_recovery_active or not self._looks_like_auth_wall():
            return False
        self._auth_recovery_active = True
        try:
            settings.logger.info(
                f"{self.app_name} auth wall detected at {self.page.url}; running login bootstrap"
            )
            self.first_init(app_name)
            return not self._looks_like_auth_wall()
        except Exception as e:
            settings.logger.info(f"{self.app_name} auth recovery failed: {e}")
            return False
        finally:
            self._auth_recovery_active = False

    def _fix_gadael_base_url(self, route_path=None, reload_route: bool = False) -> bool:
        if self._normalized_app_name() != "gadael":
            return False
        try:
            return bool(self.page.evaluate(
                """
                ({ routePath, reloadRoute }) => {
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
                }
                """,
                {"routePath": route_path, "reloadRoute": bool(reload_route)},
            ))
        except Exception as e:
            settings.logger.debug(f"Gadael base-url patch skipped: {e}")
            return False

    def first_init(self, app_name):
        app_name = self._normalized_app_name()

        if app_name in ('github', 'odoo', 'petclinic'):
            try:
                self._goto(self.url, wait_ms=2000)
                settings.logger.info(f"{self.app_name} Init Successfully")
            except Exception as e:
                settings.logger.info(f"{self.app_name} Init Failed: {e}")

        elif app_name == 'splittypie':
            try:
                base_url = self._origin().rstrip("/")
                self._goto(base_url + "/new", wait_ms=3000)
                event_name = f"WebRLED {os.environ.get('WEBRLED_SEED', 'seed')}"
                created = bool(self.page.evaluate(
                    """
                    (eventName) => {
                      const visible = (el) => {
                        const rect = el.getBoundingClientRect();
                        const style = window.getComputedStyle(el);
                        return rect.width > 0 && rect.height > 0
                          && style.visibility !== 'hidden'
                          && style.display !== 'none';
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
                    }
                    """,
                    event_name,
                ))
                if not created:
                    raise RuntimeError("event bootstrap submission did not trigger")
                if not self._wait_until(
                    lambda: (self.page.url or "").rstrip("/") not in {base_url, base_url + "/new"},
                    timeout_ms=12000,
                    interval_ms=500,
                ):
                    raise RuntimeError("event page was not created in time")
                new_url = (self.page.url or "").rstrip("/")
                if new_url:
                    self.url = new_url
                self._goto(self.url, wait_ms=2000)
                settings.logger.info("Splittypie Init Successfully")
            except Exception as e:
                settings.logger.info(f"Splittypie Init Failed: {e}")

        elif app_name == '4gaboards':
            try:
                base_url = self._origin().rstrip("/")
                self._goto(base_url + "/login", wait_ms=4000)
                self._pw_fill(["input[name='emailOrUsername']", "input[name='email']"], os.environ["WEBTEST_4GABOARDS_EMAIL"])
                self._pw_fill(["input[name='password']", "input[type='password']"], os.environ["WEBTEST_4GABOARDS_PASSWORD"])
                if not self._pw_click(["button[type='submit']", "xpath=//button[contains(text(),'Log in') or contains(text(),'Sign in')]"]):
                    self._submit_form_fallback()
                self.page.wait_for_timeout(5000)
                if not self._wait_until_not_auth(timeout_ms=8000):
                    self._goto(base_url + "/register", wait_ms=4000)
                    self._pw_fill(["input[name='username']", "input[placeholder='Username']"], os.environ["WEBTEST_4GABOARDS_USERNAME"])
                    self._pw_fill(["input[name='email']", "input[type='email']"], os.environ["WEBTEST_4GABOARDS_EMAIL"])
                    self._pw_fill(["input[name='password']", "input[type='password']"], os.environ["WEBTEST_4GABOARDS_PASSWORD"])
                    try:
                        cb = self.page.query_selector("input[type='checkbox']")
                        if cb and not cb.is_checked():
                            cb.click()
                    except Exception:
                        pass
                    if not self._pw_click(["button[type='submit']", "xpath=//button[contains(text(),'Register') or contains(text(),'Sign up')]"]):
                        self._submit_form_fallback()
                    self.page.wait_for_timeout(6000)
                    if self._looks_like_auth_wall():
                        self._goto(base_url + "/login", wait_ms=3000)
                        self._pw_fill(["input[name='emailOrUsername']", "input[name='email']"], os.environ["WEBTEST_4GABOARDS_EMAIL"])
                        self._pw_fill(["input[name='password']", "input[type='password']"], os.environ["WEBTEST_4GABOARDS_PASSWORD"])
                        if not self._pw_click(["button[type='submit']"]):
                            self._submit_form_fallback()
                        self.page.wait_for_timeout(5000)
                self._goto(self.url, wait_ms=2000)
                settings.logger.info("4gaBoards Init Successfully")
            except Exception as e:
                settings.logger.info(f"4gaBoards Init Failed: {e}")

        elif app_name == 'agilefant':
            try:
                base_url = self._app_base_url("agilefant")
                primary_username = os.environ.get(
                    "WEBTEST_AGILEFANT_USERNAME",
                    settings.WEB_INFO.get("agilefant", {}).get("username", ""),
                ).strip()
                primary_password = os.environ.get(
                    "WEBTEST_AGILEFANT_PASSWORD",
                    settings.WEB_INFO.get("agilefant", {}).get("password", ""),
                ).strip()
                landing = os.environ.get(
                    "WEBTEST_WEBRLED_AGILEFANT_LANDING",
                    "dailyWork.action",
                ).lstrip("/")
                self.url = base_url + "/" + landing

                def _agilefant_logged_in():
                    if self._looks_like_auth_wall():
                        return False
                    current_url = (self.page.url or "").lower()
                    if "login.jsp" in current_url or "j_spring_security_check" in current_url:
                        return False
                    try:
                        body_text = (self.page.text_content("body") or "").lower()
                    except Exception:
                        body_text = ""
                    return (
                        "logout" in body_text
                        or "daily work" in body_text
                        or "my work" in body_text
                        or "create new" in body_text
                    )

                credential_pairs = []

                def _add_credential(username, password):
                    pair = ((username or "").strip(), (password or "").strip())
                    if pair[0] and pair[1] and pair not in credential_pairs:
                        credential_pairs.append(pair)

                _add_credential(primary_username, primary_password)

                # First try the landing page. Some Agilefant deployments retain a
                # valid server-side session, and this avoids unnecessary login churn.
                self._goto(self.url, wait_ms=2000)
                active_username = "existing-session"
                if not _agilefant_logged_in():
                    last_error = ""
                    for username, password in credential_pairs:
                        try:
                            self._goto(base_url + "/login.jsp", wait_ms=2000)
                            if not self._pw_fill(["#username", "input[name='j_username']"], username):
                                raise RuntimeError("username field not found")
                            if not self._pw_fill(["input[name='j_password']", "input[type='password']"], password):
                                raise RuntimeError("password field not found")
                            if not self._pw_click(["input[type='submit']", "button[type='submit']"]):
                                self._submit_form_fallback()
                            self.page.wait_for_timeout(3000)
                            self._goto(self.url, wait_ms=3000)
                            if _agilefant_logged_in():
                                active_username = username
                                break
                            last_error = f"user={username} current_url={self.page.url}"
                        except Exception as cred_error:
                            last_error = f"user={username} err={cred_error}"
                    else:
                        raise RuntimeError(
                            f"login did not reach Agilefant app pages; last_attempt={last_error}"
                        )

                if not _agilefant_logged_in():
                    raise RuntimeError(
                        f"login did not reach Agilefant app pages for user={active_username}; current_url={self.page.url}"
                    )
                settings.logger.info(f"Agilefant Init Successfully: user={active_username}, url={self.page.url}")
            except Exception as e:
                settings.logger.info(f"Agilefant Init Failed: {e}")

        elif app_name == 'timeoff':
            try:
                base_url = self._origin().rstrip("/")
                self._goto(base_url + "/login", wait_ms=3000)
                self._pw_fill(["#email_inp", "input[name='username']"], os.environ["WEBTEST_TIMEOFF_EMAIL"])
                self._pw_fill(["#pass_inp", "input[type='password']"], os.environ["WEBTEST_TIMEOFF_PASSWORD"])
                if not self._pw_click(["#submit_login", "button[type='submit']", "input[type='submit']"]):
                    self._submit_form_fallback()
                self.page.wait_for_timeout(4000)
                if not self._wait_until_not_auth(timeout_ms=7000):
                    self._goto(base_url + "/register", wait_ms=3000)
                    self._pw_fill(["#company_name_inp"], "TestCompany")
                    self._pw_fill(["#name_inp"], "Secret")
                    self._pw_fill(["#lastname_inp"], "User")
                    self._pw_fill(["#email_inp"], os.environ["WEBTEST_TIMEOFF_EMAIL"])
                    self._pw_fill(["#pass_inp"], os.environ["WEBTEST_TIMEOFF_PASSWORD"])
                    self._pw_fill(["#confirm_pass_inp"], os.environ["WEBTEST_TIMEOFF_PASSWORD"])
                    if not self._pw_click(["button[type='submit']", "input[type='submit']"]):
                        self._submit_form_fallback()
                    self.page.wait_for_timeout(5000)
                self.url = base_url + "/calendar/"
                self._goto(self.url, wait_ms=2000)
                settings.logger.info("Timeoff Init Successfully")
            except Exception as e:
                settings.logger.info(f"Timeoff Init Failed: {e}")

        elif app_name == 'gadael':
            try:
                base_url = self._origin().rstrip("/")
                email = os.environ["WEBTEST_GADAEL_EMAIL"]
                password = os.environ["WEBTEST_GADAEL_PASSWORD"]

                def _gadael_logged_in():
                    try:
                        return bool(self.page.evaluate(
                            """
                            () => {
                              const injector = window.angular
                                && window.angular.element(document.body).injector
                                && window.angular.element(document.body).injector();
                              if (!injector) return false;
                              const root = injector.get('$rootScope');
                              return !!(root.sessionUser && root.sessionUser.isAuthenticated);
                            }
                            """
                        ))
                    except Exception:
                        return False

                def _goto_gadael_route(route_path: str):
                    self._goto(base_url + "/#" + route_path, wait_ms=2000)
                    self._fix_gadael_base_url(route_path, reload_route=True)
                    self.page.wait_for_timeout(4000)

                _goto_gadael_route("/login/createfirstadmin")
                if self._pw_fill(["#user_firstname"], "Secret", timeout=4000):
                    self._pw_fill(["#user_lastname"], "User", timeout=4000)
                    self._pw_fill(["#user_email"], email, timeout=4000)
                    self._pw_fill(["#user_password"], password, timeout=4000)
                    self._pw_fill(["#user_password2"], password, timeout=4000)
                    if not self._pw_click([
                        "xpath=//*[@id='user_password2']/ancestor::form//button[contains(@class,'btn-primary')]",
                        "xpath=//button[contains(@class,'btn-primary') and contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'save')]",
                    ], timeout=6000):
                        self._submit_form_fallback()
                    self.page.wait_for_timeout(6000)
                    self._fix_gadael_base_url("/home", reload_route=True)
                    self.page.wait_for_timeout(2000)

                if not _gadael_logged_in():
                    _goto_gadael_route("/login")
                    self._pw_fill([
                        "input[name='username']",
                        "input[ng-model='username']",
                        "input[type='text']",
                    ], email, timeout=8000)
                    self._pw_fill([
                        "input[name='password']",
                        "input[type='password']",
                    ], password, timeout=8000)
                    if not self._pw_click([
                        "button[type='submit']",
                        "button.btn-login",
                    ], timeout=8000):
                        self._submit_form_fallback()
                    self.page.wait_for_timeout(6000)
                    self._fix_gadael_base_url("/home", reload_route=True)
                    self.page.wait_for_timeout(2000)

                self.url = base_url + "/#/home"
                self._goto(self.url, wait_ms=2000)
                self._fix_gadael_base_url("/home", reload_route=True)
                self.page.wait_for_timeout(2000)
                settings.logger.info(f"Gadael Init Successfully: logged_in={_gadael_logged_in()}")
            except Exception as e:
                settings.logger.info(f"Gadael Init Failed: {e}")

        elif app_name == 'nextcloud':
            try:
                base_url = self._origin().rstrip("/")
                self._goto(base_url + "/login", wait_ms=4000)
                self._pw_fill(["#user", "input[name='user']"], os.environ["WEBTEST_NEXTCLOUD_USERNAME"])
                self._pw_fill(["#password", "input[name='password']"], os.environ["WEBTEST_NEXTCLOUD_PASSWORD"])
                if not self._pw_click(["button[type='submit']", "input[type='submit']", "[data-login-form-submit]"]):
                    self._submit_form_fallback()
                self.page.wait_for_timeout(5000)
                self.url = base_url + "/apps/files/"
                self._goto(self.url, wait_ms=2000)
                settings.logger.info("Nextcloud Init Successfully")
            except Exception as e:
                settings.logger.info(f"Nextcloud Init Failed: {e}")

        elif app_name == 'realworld':
            try:
                base_url = self.url.rstrip("/").split("#")[0]
                self._goto(base_url + "/#/register", wait_ms=8000)
                self._pw_fill(["input[placeholder='Username']"], os.environ["WEBTEST_REALWORLD_USERNAME"])
                self._pw_fill(["input[placeholder='Email']"], os.environ["WEBTEST_REALWORLD_EMAIL"])
                self._pw_fill(["input[placeholder='Password']"], os.environ["WEBTEST_REALWORLD_PASSWORD"])
                if not self._pw_click(["button[type='submit']", "xpath=//button[contains(text(),'Sign up')]"]):
                    self._submit_form_fallback()
                self.page.wait_for_timeout(5000)
                if self._looks_like_auth_wall():
                    self._goto(base_url + "/#/login", wait_ms=5000)
                    self._pw_fill(["input[placeholder='Email']"], os.environ["WEBTEST_REALWORLD_EMAIL"])
                    self._pw_fill(["input[placeholder='Password']"], os.environ["WEBTEST_REALWORLD_PASSWORD"])
                    if not self._pw_click(["button[type='submit']"]):
                        self._submit_form_fallback()
                    self.page.wait_for_timeout(4000)
                self._goto(self.url, wait_ms=2000)
                settings.logger.info("Realworld Init Successfully")
            except Exception as e:
                settings.logger.info(f"Realworld Init Failed: {e}")

    @staticmethod
    def extract_domain(url):
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    def __del__(self):
        try:
            self.context.close()
        except Exception:
            pass
        try:
            self.browser.close()
        except Exception:
            pass
        try:
            self.pw.stop()
        except Exception:
            pass

    def reset_page(self):
        settings.logger.debug('Reset Page.')
        i = 0
        max_retries = int(os.environ.get("WEBTEST_WEBRLED_RESET_MAX_RETRIES", "5"))
        while i < max_retries:
            try:
                self.page.goto(self.url, wait_until="domcontentloaded", timeout=30000)
                if self._normalized_app_name() == "gadael":
                    # Gadael is an AngularJS hash-route SPA. A plain page.goto()
                    # can leave the URL at /#/home while Angular still exposes
                    # stale route DOM, which makes WebRLED see an empty action
                    # space and enter reset loops. Keep the official policy
                    # unchanged; only force the SPA route to settle after reset.
                    self._fix_gadael_base_url("/home", reload_route=True)
                    self.page.wait_for_timeout(2000)
                else:
                    self._fix_gadael_base_url()
                self.ensure_authenticated()
                break
            except Error as e:
                msg = str(e).replace("\n", '\\n')
                settings.logger.debug('Error: Reset Page. ' + msg)
                i += 1
                self.page.wait_for_timeout(min(2000 + i * 5000, 30000))
        else:
            raise RuntimeError(
                f"Reset Page failed after {max_retries} retries for {self.app_name}: {self.url}"
            )

    def new_page(self):
        while True:
            try:
                settings.logger.debug('New Page')
                page = self.context.new_page()
                page.goto(self.url)
                page.wait_for_timeout(timeout=10000)
                break
            except TargetClosedError as te:
                self.new_context()

    def new_context(self):
        while True:
            try:
                settings.logger.error('New Context')
                context = self.browser.new_context(viewport={'width': self.width, 'height': self.height})
                context.on("page", lambda page: self.handle_new_page(page))
                old_context = self.context
                self.context = context
                old_context.close()
                break
            except TargetClosedError as te:
                settings.logger.error('New Context')
                self.new_browser()

    def new_browser(self):
        headless = os.environ.get("WEBTEST_FORCE_HEADFUL", "0") != "1"
        browser = self.pw.chromium.launch(headless=headless)
        old_browser = self.browser
        self.browser = browser
        old_browser.close()

    def handle_errors(self, msg):
        if msg.type == "error":
            self.errors_buffer.append(msg)
        log_info = str(msg.type) + str(msg)
        if log_info not in self.log_set:
            self.log_set.add(log_info)
            settings.logger.debug(f'Level: {msg.type}, full console log is: {msg}, Current console log is: {msg.text}')

    def handle_page_error(self, exc):
        log_info = f"PageError: {exc}"
        self.errors_buffer.append(log_info)

    def manage_browser_pages(self):
        settings.logger.debug('manage_browser_pages')
        pages = self.context.pages
        self.page = pages[-1]
        if len(pages) > 1:
            for page in pages[:-1]:
                page.close()

    def handle_new_page(self, page):
        if page.url != ':':
            settings.logger.debug('handle_new_page')
            # self.context.pages.append(page)
            self.change_page(page)
            # self.page.wait_for_timeout(2000)  # other
            # self.page.wait_for_load_state('networkidle')
            settings.logger.debug('handle_new_page Page: {}'.format(str(self.page)))
            settings.logger.debug('handle_new_page -')

    def change_page(self, page):
        self.page = page
        self.page.on("dialog", lambda dialog: dialog.dismiss())
        self.page.on("console", lambda msg: self.handle_errors(msg))  # console error
        self.page.on("pageerror", lambda exc: self.handle_page_error(exc))

    # Coverage and error
    @property
    def error_num(self):
        return len(self.errors)

    ## nyc
    def update_cov_and_errors_nyc(self) -> bool:
        is_cov_changed, is_errors_changed = False, False

        self.old_branch_coverage, self.old_line_coverage = self.branch_coverage, self.line_coverage
        while True:
            try:
                self.branch_coverage, self.line_coverage = self.get_cov_nyc()
                break
            except Exception as e:
                settings.logger.debug(e)
        if self.old_branch_coverage != self.branch_coverage or self.old_line_coverage != self.line_coverage:
            is_cov_changed = True
        is_errors_changed = self.update_errors()
        return is_cov_changed

    def update_cov_nyc(self):
        headers = {'Content-Type': 'application/json', }
        index = 0
        while True:
            try:
                index += 1
                cov_raw = self.page.evaluate('(() => { return window.__coverage__; })()')
                response = requests.post(f"{self.coverage_base_url}/coverage/client", data=json.dumps(cov_raw),
                                         headers=headers)

                if response.status_code == 200:
                    break
                time.sleep(1)
                if index > 10:
                    self.reset_page()
                # if cov_raw is None:
                #     self.reset_page()
            except Exception as e:
                settings.logger.debug('Get cov nyc failed.')
                time.sleep(1)

    def get_cov_nyc(self):
        if self.benchmark == 'simple':
            self.update_cov_nyc()
            url = f'{self.coverage_base_url}/coverage'
        else:
            url = f'{self.coverage_base_url}/'
        # get coverage from web
        html = requests.get(url).text
        # 正则表达式，匹配覆盖率百分比、类型（Statements, Branches, Functions, Lines）、计数
        pattern = r'<span class="strong">([\d.]+%) </span>\s*<span class="quiet">(\w+)</span>\s*<span class=\'fraction\'>(\d+/\d+)</span>'
        # 使用findall提取所有匹配的数据
        matches = re.findall(pattern, html)
        # 格式化输出
        branch_coverage, line_coverage = None, None
        for match in matches:
            percentage, label, fraction = match
            # 打印结果，按照指定格式输出
            settings.logger.info(
                f"Current coverage is: {label}: {percentage},{fraction}"
            )
            if label == 'Branches':
                branch_coverage = percentage
            elif label == 'Lines':
                line_coverage = percentage
        assert branch_coverage is not None and line_coverage is not None
        return branch_coverage, line_coverage

    ## error
    def reset_errors(self) -> None:
        self.errors.clear()
        self.errors_detail.clear()
        self.errors_buffer.clear()

    def update_errors(self) -> bool:
        is_errors_changed = False
        write_buffer = []
        url = self.page.url
        for error in self.errors_buffer:
            if isinstance(error, str):
                error_info = error
            else:
                error_info = f"{error.text}"
            if error_info not in self.errors:
                is_errors_changed = True
                self.errors.add(error_info)
                settings.logger.info('Current page URL is: {}, Console Error Log: {}'.format(url, error_info))
                write_buffer.append(error_info)
        self.errors_buffer.clear()
        # if is_errors_changed:
        #     write_file(self.errors_file, write_buffer)
        return is_errors_changed

    ## jacoco
    def get_cov_jacoco(self):
        url = f'{self.coverage_base_url}/coverage'
        last_error = None
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                coverage_dict = response.json()
                break
            except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                settings.logger.warning(
                    f"Coverage JSON read failed for {url} "
                    f"(attempt {attempt + 1}/3): {exc}"
                )
                time.sleep(1)
        else:
            raise last_error
        branch_coverage = coverage_dict['branch_coverage']
        line_coverage = coverage_dict['line_coverage']
        settings.logger.info(f"Current coverage is: Branches: {branch_coverage}%,")
        settings.logger.info(f"Current coverage is: Lines: {line_coverage}%,")
        if 'line_coverage_info' in coverage_dict:
            line_coverage_info = coverage_dict['line_coverage_info']
            settings.logger.info(f"Current coverage info is: Lines_info: {line_coverage_info}")
        return branch_coverage, line_coverage

    def update_cov_and_errors_jacoco(self):
        is_cov_changed, is_errors_changed = False, False
        self.old_branch_coverage, self.old_line_coverage = self.branch_coverage, self.line_coverage
        self.branch_coverage, self.line_coverage = self.get_cov_jacoco()
        if self.old_branch_coverage != self.branch_coverage or self.old_line_coverage != self.line_coverage:
            is_cov_changed = True
        is_errors_changed = self.update_errors()
        return is_cov_changed
