import os
from typing import List
from urllib.parse import urlsplit

from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException

from action.element_locator import ElementLocator
from action.element_text_detect_mode import ElementTextDetectMode
from action.impl.random_select_action import RandomSelectAction
from action.web_action import WebAction
from action.web_action_detector import WebActionDetector
from config.settings import settings


class RandomSelectActionDetector(WebActionDetector):
    def __init__(self):
        self.js_file_path = os.path.join(settings.resources_path, "js", "action_detector.js")
        self.selectors = ["select"]
        self.block_auth_actions = os.environ.get("WEBTEST_BLOCK_AUTH_ACTIONS", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        raw_prefixes = os.environ.get(
            "WEBTEST_AUTH_PATH_PREFIXES",
            "/login,/signup,/sign-in,/signin,/register,/password_reset,/session,/sessions,/oauth,/auth",
        )
        self.auth_path_prefixes = tuple(
            p.strip().lower() for p in raw_prefixes.split(",") if p.strip()
        )

    def _on_auth_page(self, driver: WebDriver) -> bool:
        if not self.block_auth_actions:
            return False
        try:
            path = (urlsplit(getattr(driver, "current_url", "")).path or "").lower()
        except Exception:
            path = (getattr(driver, "current_url", "") or "").lower()
        return any(path.startswith(prefix) for prefix in self.auth_path_prefixes)

    @staticmethod
    def _has_meaningful_options(web_element) -> bool:
        try:
            if web_element.get_attribute("disabled"):
                return False
            if not (web_element.is_displayed() and web_element.is_enabled()):
                return False
            options = web_element.find_elements(By.TAG_NAME, "option")
            usable = [
                option for option in options
                if option.is_enabled() and not option.get_attribute("disabled")
            ]
            return any(not option.is_selected() for option in usable)
        except WebDriverException:
            return False

    def get_actions(self, driver: WebDriver) -> List[WebAction]:
        if self._on_auth_page(driver):
            return []
        web_action_list = []
        with open(self.js_file_path, 'r') as js_file:
            js_code = js_file.read()
        result = driver.execute_script(js_code, self.selectors, ElementTextDetectMode.LABEL.value)
        for web_action_info in result:
            if web_action_info["visible"]:
                try:
                    web_element = driver.find_element(By.XPATH, web_action_info["xpath"])
                except WebDriverException:
                    continue
                if not self._has_meaningful_options(web_element):
                    continue
                action = RandomSelectAction(
                    ElementLocator.XPATH,
                    web_action_info["xpath"],
                    web_action_info["text"],
                )
                action.rect = dict(web_element.rect or {})
                web_action_list.append(action)
        return web_action_list
