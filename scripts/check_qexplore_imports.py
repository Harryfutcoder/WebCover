#!/usr/bin/env python
"""Check QExplore runtime prerequisites in the selected Python environment."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
QEXPLORE_SOURCE = REPO_ROOT / "external" / "QExplore" / "Qexplore"
if QEXPLORE_SOURCE.exists():
    sys.path.insert(0, str(QEXPLORE_SOURCE))

WEBTEST_SITE_PACKAGES = Path.home() / ".conda" / "envs" / "webtest-python" / "Lib" / "site-packages"
if WEBTEST_SITE_PACKAGES.exists() and str(WEBTEST_SITE_PACKAGES) not in sys.path:
    # Lets base Anaconda reuse the Selenium installation from the main browser
    # env without forcing all QExplore dependencies into that env.
    sys.path.append(str(WEBTEST_SITE_PACKAGES))

RUNTIME_IMPORTS = {
    # Qexplore.py imports apted at module import time, even though it is mostly
    # used by the post-processing helpers.
    "apted": "apted",
    "beautifulsoup4": "bs4",
    "cryptohash": "cryptohash",
    "exrex": "exrex",
    "gensim": "gensim",
    "js-regex": "js_regex",
    "matplotlib": "matplotlib",
    "nltk": "nltk",
    "num2words": "num2words",
    "numpy": "numpy",
    "pandas": "pandas",
    "pyenchant": "enchant",
    "prettytable": "prettytable",
    "requests": "requests",
    "scikit-learn": "sklearn",
    "selenium": "selenium",
    "sister": "sister",
    "tqdm": "tqdm",
    "wordninja": "wordninja",
    "IPython": "IPython",
}

CLI_GUI_IMPORTS = {
    # The wrapper runs with --ignore-gooey, but the original source still
    # imports Gooey at module load time unless a local compatibility shim exists.
    "gooey": "gooey",
}


def _check_imports(title: str, modules: dict[str, str]) -> list[str]:
    print(f"== {title} ==")
    missing: list[str] = []
    for package, module in modules.items():
        ok = importlib.util.find_spec(module) is not None
        print(f"{package}: {'OK' if ok else 'MISSING'}")
        if not ok:
            missing.append(package)
    return missing


def _check_browser(skip_browser: bool) -> list[str]:
    if skip_browser:
        return []
    print("== browser ==")
    missing: list[str] = []
    browser = os.environ.get("QEXPLORE_BROWSER", "chrome").strip().lower()
    print(f"browser: {browser}")
    if browser == "chrome":
        chrome_override = os.environ.get("QEXPLORE_CHROMEDRIVER", "").strip()
        chrome_binary_override = os.environ.get("QEXPLORE_CHROME_BINARY", "").strip()
        default_chrome_binary = Path(r"C:\Users\SUST\Desktop\webTest\webTest\chrome-win\chrome.exe")
        default_chromedriver = Path(r"C:\Users\SUST\Desktop\webTest\webTest\chromedriver.exe")
        chrome_binary = (
            chrome_binary_override
            if chrome_binary_override and Path(chrome_binary_override).exists()
            else str(default_chrome_binary)
            if default_chrome_binary.exists()
            else shutil.which("chrome")
        )
        chromedriver = (
            chrome_override
            if chrome_override and Path(chrome_override).exists()
            else str(default_chromedriver)
            if default_chromedriver.exists()
            else shutil.which("chromedriver")
        )
        print(f"chrome_binary: {chrome_binary or 'MISSING'}")
        print(f"chromedriver: {chromedriver or 'MISSING'}")
        if chrome_binary_override and not Path(chrome_binary_override).exists():
            print(f"chrome_binary_override_missing={chrome_binary_override}")
        if chrome_override and not Path(chrome_override).exists():
            print(f"chromedriver_override_missing={chrome_override}")
        if not chrome_binary:
            missing.append("chrome_binary")
        if not chromedriver:
            missing.append("chromedriver")
        return missing

    firefox_override = os.environ.get("QEXPLORE_FIREFOX_BINARY", "").strip()
    gecko_override = os.environ.get("QEXPLORE_GECKODRIVER", "").strip()
    firefox = firefox_override if firefox_override and Path(firefox_override).exists() else shutil.which("firefox")
    gecko = gecko_override if gecko_override and Path(gecko_override).exists() else shutil.which("geckodriver")
    print(f"firefox: {firefox or 'MISSING'}")
    print(f"geckodriver: {gecko or 'MISSING'}")
    if firefox_override and not Path(firefox_override).exists():
        print(f"firefox_override_missing={firefox_override}")
    if gecko_override and not Path(gecko_override).exists():
        print(f"geckodriver_override_missing={gecko_override}")
    if not firefox:
        missing.append("firefox")
    if not gecko:
        missing.append("geckodriver")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-browser", action="store_true")
    parser.add_argument(
        "--allow-missing-gooey",
        action="store_true",
        help="treat Gooey as optional only if a CLI shim is intentionally provided",
    )
    args = parser.parse_args()

    print(f"python={sys.executable}")
    print(f"version={sys.version.split()[0]}")
    print(f"qexplore_source={QEXPLORE_SOURCE}")
    missing = []
    missing.extend(_check_imports("runtime imports", RUNTIME_IMPORTS))
    gui_missing = _check_imports("cli/gui import", CLI_GUI_IMPORTS)
    if gui_missing and not args.allow_missing_gooey:
        missing.extend(gui_missing)
    missing.extend(_check_browser(args.skip_browser))
    if missing:
        print("missing=" + ",".join(missing))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
