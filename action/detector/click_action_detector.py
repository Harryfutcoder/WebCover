import os
from typing import List
from urllib.parse import urlsplit

from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By

from action.element_locator import ElementLocator
from action.element_text_detect_mode import ElementTextDetectMode
from action.impl.click_action import ClickAction
from action.web_action import WebAction
from action.web_action_detector import WebActionDetector
from config.settings import settings
from utils import normalize_url_for_state


class ClickActionDetector(WebActionDetector):
    def __init__(self):
        self.js_file_path = os.path.join(settings.resources_path, "js", "action_detector.js")
        self.selectors = [
            "a",
            "button",
            "input[type=\"button\"]",
            "input[type=\"submit\"]",
            "input[type=\"checkbox\"]",
            "input[type=\"radio\"]",
            "input[type=\"image\"]",
            "summary",
        ]
        self.blocked_external_schemes = {
            "vscode",
            "x-github-client",
            "github-windows",
            "github-mac",
            "mailto",
            "tel",
            "sms",
            "intent",
        }
        self.blocked_action_text_markers = (
            "open in visual studio code",
            "open in github desktop",
            "open with github desktop",
        )
        self.block_auth_actions = os.environ.get("WEBTEST_BLOCK_AUTH_ACTIONS", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.block_offdomain_actions = os.environ.get("WEBTEST_BLOCK_OFFDOMAIN_ACTIONS", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.skip_file_upload_submits = os.environ.get("WEBTEST_SKIP_FILE_UPLOAD_SUBMITS", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        raw_prefixes = os.environ.get(
            "WEBTEST_AUTH_PATH_PREFIXES",
            "/login,/logout,/signup,/sign-in,/sign-out,/signin,/signout,/register,/password_reset,/session,/sessions,/oauth,/auth",
        )
        self.auth_path_prefixes = tuple(
            p.strip().lower() for p in raw_prefixes.split(",") if p.strip()
        )
        self.blocked_auth_text_markers = (
            "sign in",
            "signin",
            "log in",
            "login",
            "sign up",
            "signup",
            "create account",
            "forgot password",
            "reset password",
            "passkey",
            "continue with google",
            "continue with apple",
            "continue with github",
            "continue with microsoft",
            "log out",
            "logout",
            "sign out",
            "signout",
        )

    @staticmethod
    def _normalize_redirect_url(raw_url: str) -> str:
        return normalize_url_for_state(raw_url)

    def _contains_external_scheme(self, raw_text: str) -> bool:
        txt = (raw_text or "").lower()
        if not txt:
            return False
        return any(f"{scheme}:" in txt for scheme in self.blocked_external_schemes)

    def _is_auth_url(self, raw_url: str) -> bool:
        if not raw_url:
            return False
        try:
            parsed = urlsplit(raw_url)
            path = (parsed.path or "").lower()
            return any(path.startswith(prefix) for prefix in self.auth_path_prefixes)
        except Exception:
            lower = raw_url.lower()
            return any(prefix in lower for prefix in self.auth_path_prefixes)

    def _is_allowed_domain_url(self, raw_url: str) -> bool:
        if not raw_url:
            return True
        try:
            parsed = urlsplit(raw_url)
        except Exception:
            return True

        scheme = (parsed.scheme or "").lower()
        if scheme not in ("http", "https"):
            return True

        netloc = (parsed.netloc or "").lower()
        if not netloc:
            return True

        allowed_domains = getattr(settings, "domains", []) or []
        for allowed in allowed_domains:
            allowed = str(allowed or "").strip().lower()
            if not allowed:
                continue
            if netloc == allowed or netloc.endswith("." + allowed):
                return True

            host = (parsed.hostname or "").lower()
            # For localhost benchmark apps the port identifies the site.  A
            # bare "localhost" allowlist entry must not make every local port
            # clickable, otherwise one app can silently jump into another.
            allowed_is_local_host = allowed in ("localhost", "127.0.0.1", "::1")
            if allowed_is_local_host and parsed.port is not None:
                continue
            if host and (host == allowed or host.endswith("." + allowed)):
                return True

        return False

    def _contains_auth_marker(self, raw_text: str) -> bool:
        txt = (raw_text or "").lower()
        if not txt:
            return False
        return any(marker in txt for marker in self.blocked_auth_text_markers)

    def _is_gadael_empty_root_link(self, raw_url: str, raw_text: str) -> bool:
        """Gadael exposes an empty logo link to / while the Angular app is loading."""
        entry = (getattr(settings, "entry_url", "") or "").lower()
        if "localhost:3001" not in entry:
            return False
        text = (raw_text or "").strip().lower()
        if text not in ("", "loading..."):
            return False
        try:
            parsed = urlsplit(raw_url or "")
        except Exception:
            return False
        if (parsed.scheme or "").lower() not in ("http", "https"):
            return False
        netloc = (parsed.netloc or "").lower()
        path = (parsed.path or "/").rstrip("/") or "/"
        fragment = (parsed.fragment or "").strip()
        return netloc == "localhost:3001" and path == "/" and fragment == ""

    def _is_blank_same_url_redirect(self, driver: WebDriver, raw_url: str, raw_text: str) -> bool:
        """Filter uninformative blank/loading links that only navigate to the current normalized URL."""
        text = (raw_text or "").strip().lower()
        if text not in ("", "loading..."):
            return False
        if not raw_url:
            return False
        try:
            parsed = urlsplit(raw_url)
        except Exception:
            return False
        if (parsed.scheme or "").lower() not in ("http", "https"):
            return False
        return self._normalize_redirect_url(raw_url) == normalize_url_for_state(getattr(driver, "current_url", ""))

    @staticmethod
    def _form_has_file_input(form_element) -> bool:
        try:
            file_inputs = form_element.find_elements(
                By.XPATH,
                ".//input[translate(@type, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='file']",
            )
        except Exception:
            return False
        return bool(file_inputs)

    def get_actions(self, driver: WebDriver) -> List[WebAction]:
        web_action_list = []
        with open(self.js_file_path, "r") as js_file:
            js_code = js_file.read()
        result = driver.execute_script(js_code, self.selectors, ElementTextDetectMode.INNER_TEXT.value)
        on_auth_page = self.block_auth_actions and self._is_auth_url(getattr(driver, "current_url", ""))
        for web_action_info in result:
            if not web_action_info["visible"]:
                continue
            action_text = (web_action_info.get("text") or "").strip().lower()
            if any(marker in action_text for marker in self.blocked_action_text_markers):
                continue
            if self.block_auth_actions and self._contains_auth_marker(action_text):
                continue

            web_element = driver.find_element(By.XPATH, web_action_info["xpath"])

            href_value = web_element.get_attribute("href")
            scheme = ""
            if href_value:
                if self._is_blank_same_url_redirect(driver, href_value, web_action_info.get("text") or ""):
                    continue
                if self._is_gadael_empty_root_link(href_value, web_action_info.get("text") or ""):
                    continue
                try:
                    scheme = (urlsplit(href_value).scheme or "").lower()
                except Exception:
                    scheme = ""
                if scheme in self.blocked_external_schemes:
                    continue
                if self.block_offdomain_actions and not self._is_allowed_domain_url(href_value):
                    continue
                if self.block_auth_actions and self._is_auth_url(href_value):
                    continue

            onclick_value = web_element.get_attribute("onclick")
            if self._contains_external_scheme(onclick_value):
                continue

            data_action = web_element.get_attribute("data-action")
            if self._contains_external_scheme(data_action):
                continue

            element_type = web_element.get_attribute("type")
            if self.block_auth_actions and on_auth_page and not href_value:
                # Avoid repeatedly interacting with auth forms; keep only explicit links that can leave the wall.
                continue

            click_type = "default"
            addition_info = ""
            if href_value and (scheme in ("", "http", "https")):
                click_type = "redirect"
                addition_info = self._normalize_redirect_url(href_value)
            elif element_type and element_type.lower() == "submit":
                click_type = "submit"
                parent_element = web_element
                while parent_element is not None:
                    parent_element = parent_element.find_element(By.XPATH, "..")
                    if parent_element.tag_name.lower() == "form":
                        break
                    if parent_element.tag_name.lower() == "body":
                        parent_element = None
                        break
                if parent_element:
                    if self.skip_file_upload_submits and self._form_has_file_input(parent_element):
                        continue
                    addition_info = len(parent_element.find_elements(By.XPATH, "./*"))
                else:
                    addition_info = 0
            else:
                addition_info = web_element.tag_name.lower()

            action = ClickAction(
                ElementLocator.XPATH,
                web_action_info["xpath"],
                web_action_info["text"],
                click_type,
                addition_info,
            )
            action.rect = dict(web_element.rect or {})
            web_action_list.append(action)
        return web_action_list
