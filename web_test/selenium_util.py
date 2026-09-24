"""Chrome WebDriver Service: use configured chromedriver if present, else Selenium Manager."""
import logging
import os
import subprocess
import sys

from selenium.webdriver.chrome.service import Service

from config.settings import settings

logger = logging.getLogger(__name__)

try:
    from subprocess import CREATE_NO_WINDOW
except ImportError:
    CREATE_NO_WINDOW = 0


def make_chrome_service() -> Service:
    path = getattr(settings, "driver_path", None) or ""
    path = path.strip()
    # Keep chromedriver from opening a console / logging to the parent terminal on Windows.
    log_output = subprocess.DEVNULL
    if path and os.path.isfile(path):
        svc = Service(executable_path=path, log_output=log_output)
    else:
        if path:
            logger.warning(
                "chromedriver not found at %s — using Selenium Manager (auto download)",
                path,
            )
        svc = Service(log_output=log_output)
    if sys.platform == "win32":
        svc.creation_flags = CREATE_NO_WINDOW
    return svc
