import random

from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait

from action.element_locator import ElementLocator
from action.web_action import WebAction


class RandomSelectAction(WebAction):
    def __init__(self, locator: ElementLocator, location: str, text: str) -> None:
        super().__init__()
        self.locator = locator
        self.location = location
        self.text = text

    def execute(self, driver: WebDriver) -> None:
        web_element = WebDriverWait(driver, 1.0).until(
            lambda d: self._locate_interactable(d)
        )
        select = Select(web_element)
        options = [
            option for option in select.options
            if (
                option.is_enabled()
                and not option.get_attribute("disabled")
                and not option.is_selected()
            )
        ]
        if not options:
            raise ValueError("RandomSelectAction requires an enabled unselected option")
        option = random.choice(options)
        value = option.get_attribute('value')
        if value is None:
            option.click()
        else:
            select.select_by_value(value)

    def _locate_interactable(self, driver: WebDriver):
        web_element = self.locator.locate(driver, self.location)
        if web_element.get_attribute("disabled"):
            return False
        if not (web_element.is_displayed() and web_element.is_enabled()):
            return False
        return web_element

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RandomSelectAction):
            return (self.locator == other.locator) and (self.location == other.location) and (self.text == other.text)
        return False

    def __hash__(self) -> int:
        return hash((self.locator, self.location, self.text))

    def __lt__(self, other: object) -> bool:
        if isinstance(other, RandomSelectAction):
            return (self.locator.value + self.location + self.text) < (other.locator.value + other.location + other.text)
        else:
            return type(self).__name__ < type(other).__name__

    def __str__(self) -> str:
        return f'RandomSelectAction(locator={self.locator}, location={self.location}, text={self.text})'
