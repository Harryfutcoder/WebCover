import time

from selenium.common.exceptions import (
    ElementClickInterceptedException,
    ElementNotInteractableException,
    JavascriptException,
    StaleElementReferenceException,
)
from selenium.webdriver.chrome.webdriver import WebDriver

from action.element_locator import ElementLocator
from action.web_action import WebAction


class ClickAction(WebAction):
    def __init__(self, locator: ElementLocator, location: str, text: str, action_type: str, addition_info: str ) -> None:
        super().__init__()
        self.locator = locator
        self.location = location
        self.text = text
        self.action_type = action_type
        self.addition_info = addition_info

    def execute(self, driver: WebDriver) -> None:
        # Be resilient to transient overlays/animations that can intercept clicks.
        last_error = None
        for attempt in range(3):
            web_element = self.locator.locate(driver, self.location)
            try:
                driver.execute_script(
                    "arguments[0].scrollIntoView({block:'center', inline:'center'});",
                    web_element,
                )
            except Exception:
                pass

            try:
                web_element.click()
                return
            except ElementClickInterceptedException as e:
                last_error = e
                # First retry: tiny wait for overlay/menu animation.
                if attempt == 0:
                    time.sleep(0.2)
                    continue
                # Second retry: JS click as fallback when native click is blocked.
                try:
                    driver.execute_script("arguments[0].click();", web_element)
                    return
                except JavascriptException as js_e:
                    last_error = js_e
                    time.sleep(0.1)
            except (StaleElementReferenceException, ElementNotInteractableException) as e:
                last_error = e
                time.sleep(0.1)

        if last_error is not None:
            raise last_error

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ClickAction):
            return (
                (self.locator == other.locator)
                and (self.location == other.location)
                and (self.text == other.text)
                and (self.action_type == other.action_type)
                and (self.addition_info == other.addition_info)
            )
        return False

    def __hash__(self) -> int:
        return hash((self.locator, self.location, self.text, self.action_type, self.addition_info))

    def __lt__(self, other: object) -> bool:
        if isinstance(other, ClickAction):
            return (
                self.locator.value
                + self.location
                + self.text
                + str(self.action_type)
                + str(self.addition_info)
            ) < (
                other.locator.value
                + other.location
                + other.text
                + str(other.action_type)
                + str(other.addition_info)
            )
        else:
            return type(self).__name__ < type(other).__name__

    def __str__(self) -> str:
        return (
            f"ClickAction(locator={self.locator}, location={self.location}, text={self.text}, "
            f"action_type={self.action_type}, addition_info={self.addition_info})"
        )
