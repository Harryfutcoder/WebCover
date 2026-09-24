import random
import string
import os

from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.support.ui import WebDriverWait

from action.element_locator import ElementLocator
from action.web_action import WebAction


class RandomInputAction(WebAction):
    NON_TEXT_INPUT_TYPES = {
        "button",
        "checkbox",
        "color",
        "file",
        "hidden",
        "image",
        "radio",
        "range",
        "reset",
        "submit",
    }

    def __init__(self, locator: ElementLocator, location: str, text: str) -> None:
        super().__init__()
        self.locator = locator
        self.location = location
        self.text = text
        self.max_input_length = 10

    def execute(self, driver: WebDriver) -> None:
        web_element = WebDriverWait(driver, 1.0).until(
            lambda d: self._locate_interactable(d)
        )
        if self._smart_inputs_enabled():
            input_str = self._smart_input_value(web_element)
        else:
            input_str = self._random_input_value()
        web_element.clear()
        web_element.send_keys(input_str)

    @staticmethod
    def _smart_inputs_enabled() -> bool:
        return os.environ.get("WEBTEST_SMART_INPUTS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _random_input_value(self) -> str:
        input_length = random.randint(1, self.max_input_length)
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(input_length))

    def _smart_input_value(self, web_element) -> str:
        input_type = (web_element.get_attribute("type") or "text").strip().lower()
        hint_parts = [
            input_type,
            web_element.get_attribute("name") or "",
            web_element.get_attribute("id") or "",
            web_element.get_attribute("placeholder") or "",
            web_element.get_attribute("aria-label") or "",
            web_element.get_attribute("autocomplete") or "",
            self.text or "",
            self.location or "",
        ]
        hint = " ".join(str(part).lower() for part in hint_parts if part)

        if input_type == "datetime-local":
            value = "2020-01-01T12:30"
        elif input_type == "date" or "date" in hint or "birth" in hint:
            value = "2020-01-01"
        elif input_type == "time" or "time" in hint:
            value = "12:30"
        elif input_type == "month":
            value = "2020-01"
        elif input_type == "week":
            value = "2020-W01"
        elif input_type in {"number", "range"} or any(
            token in hint for token in ("number", "amount", "price", "age", "count", "quantity")
        ):
            value = "1"
        elif input_type in {"tel", "phone"} or any(
            token in hint for token in ("phone", "telephone", "mobile", "tel")
        ):
            value = "1234567890"
        elif input_type == "email" or "email" in hint:
            value = "test@example.com"
        elif "password" in hint:
            value = "TestPass123!"
        elif any(token in hint for token in ("zip", "postal")):
            value = "12345"
        elif any(token in hint for token in ("city", "town")):
            value = "Springfield"
        elif any(token in hint for token in ("address", "street")):
            value = "123 Main St"
        elif any(token in hint for token in ("first", "given")):
            value = "Alex"
        elif any(token in hint for token in ("last", "family", "surname")):
            value = "Smith"
        elif "name" in hint:
            value = "Test"
        elif input_type == "url" or "url" in hint or "website" in hint:
            value = "https://example.com"
        elif input_type == "search" or "search" in hint:
            value = "a"
        else:
            value = "TestValue"

        max_length = web_element.get_attribute("maxlength")
        try:
            max_len = int(max_length) if max_length not in (None, "", "-1") else 0
        except ValueError:
            max_len = 0
        if max_len > 0:
            value = value[:max_len]
        return value or "x"

    def _locate_interactable(self, driver: WebDriver):
        web_element = self.locator.locate(driver, self.location)
        input_type = (web_element.get_attribute("type") or "text").strip().lower()
        if input_type in self.NON_TEXT_INPUT_TYPES:
            return False
        if web_element.get_attribute("disabled") or web_element.get_attribute("readonly"):
            return False
        if not (web_element.is_displayed() and web_element.is_enabled()):
            return False
        return web_element

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RandomInputAction):
            return (self.locator == other.locator) and (self.location == other.location) and (self.text == other.text)
        return False

    def __hash__(self) -> int:
        return hash((self.locator, self.location, self.text))

    def __lt__(self, other: object) -> bool:
        if isinstance(other, RandomInputAction):
            return (self.locator.value + self.location + self.text) < (other.locator.value + other.location + other.text)
        else:
            return type(self).__name__ < type(other).__name__

    def __str__(self) -> str:
        return f'RandomInputAction(locator={self.locator}, location={self.location}, text={self.text})'
