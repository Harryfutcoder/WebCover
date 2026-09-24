"""
Bootstrap script for the local official WebRLED source copy.

Injects a tiny datasets stub before imports so Anaconda's broken datasets
dependency chain is never loaded, then delegates to src/main.py.
"""
import importlib.util
import importlib.metadata as _metadata
import json
import os
import runpy
import sys
import time
import types

def _import_module_from_fallback_path(module_name, fallback_path):
    """Import one missing module from a fallback path without polluting sys.path."""
    if module_name in sys.modules:
        return
    if not fallback_path or not os.path.isdir(fallback_path):
        return
    inserted = False
    try:
        if fallback_path not in sys.path:
            sys.path.insert(0, fallback_path)
            inserted = True
        __import__(module_name)
    except Exception:
        sys.modules.pop(module_name, None)
    finally:
        if inserted:
            try:
                sys.path.remove(fallback_path)
            except ValueError:
                pass


_stub = types.ModuleType("datasets")
_stub.__version__ = "2.12.0"
_stub.__spec__ = importlib.util.spec_from_loader("datasets", loader=None)
_C = type("_C", (), {})
_stub.Dataset = _C
_stub.DatasetDict = _C
_stub.IterableDataset = _C
_stub.IterableDatasetDict = _C
_stub.Value = _C
_stub.Features = _C
_stub.ClassLabel = _C
sys.modules["datasets"] = _stub

# Compatibility shim for newer torch versions: WebRLED's action discriminator
# builds NumPy labels as int32 on Windows, while CrossEntropyLoss requires int64.
# Keep the official model/reward logic unchanged and only coerce target dtype.
try:
    import torch

    _orig_ce_forward = torch.nn.CrossEntropyLoss.forward

    def _ce_forward_long_target(self, input, target):
        if hasattr(target, "long") and target.dtype is not torch.long:
            target = target.long()
        return _orig_ce_forward(self, input, target)

    torch.nn.CrossEntropyLoss.forward = _ce_forward_long_target
except Exception:
    pass

_regex_fallback = os.environ.get(
    "WEBTEST_WEBRLED_REGEX_FALLBACK_PATH",
    r"C:\ProgramData\anaconda3\Lib\site-packages",
)
_import_module_from_fallback_path("regex", _regex_fallback)
if "regex" in sys.modules:
    _orig_metadata_version = _metadata.version

    def _metadata_version_with_regex(name):
        if str(name).lower() == "regex":
            return getattr(sys.modules["regex"], "__version__", "2024.5.15")
        return _orig_metadata_version(name)

    _metadata.version = _metadata_version_with_regex

_cwd = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(_cwd, "src")
_repo_root = os.path.abspath(os.path.join(_cwd, os.pardir, os.pardir))
for _p in (_src, _cwd, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import requests
    import webenv.config as _webrled_config

    def _safe_get_cov_jacoco(self):
        """Tolerate transient empty/non-JSON coverage responses.

        Some local coverage listeners briefly return an empty body while the
        target web app is busy or resetting. The official WebRLED code treats
        that as a fatal JSONDecodeError. Retrying keeps the algorithm unchanged
        and only makes the measurement hook robust.
        """
        url = f"{self.coverage_base_url}/coverage"
        retries = int(os.environ.get("WEBTEST_WEBRLED_COVERAGE_RETRIES", "3"))
        delay = float(os.environ.get("WEBTEST_WEBRLED_COVERAGE_RETRY_DELAY", "0.5"))
        last_error = None
        for attempt in range(max(1, retries)):
            try:
                resp = requests.get(url, timeout=10)
                text = (resp.text or "").strip()
                if not text:
                    raise ValueError(f"empty coverage response status={resp.status_code}")
                coverage_dict = json.loads(text)
                branch_coverage = coverage_dict["branch_coverage"]
                line_coverage = coverage_dict["line_coverage"]
                _webrled_config.settings.logger.info(f"Current coverage is: Branches: {branch_coverage}%,")
                _webrled_config.settings.logger.info(f"Current coverage is: Lines: {line_coverage}%,")
                if "line_coverage_info" in coverage_dict:
                    line_coverage_info = coverage_dict["line_coverage_info"]
                    _webrled_config.settings.logger.info(
                        f"Current coverage info is: Lines_info: {line_coverage_info}"
                    )
                return branch_coverage, line_coverage
            except Exception as exc:
                last_error = exc
                if attempt + 1 < max(1, retries):
                    time.sleep(delay)
        _webrled_config.settings.logger.warning(
            "Coverage endpoint read failed after %s attempts: %s; keeping previous coverage values",
            max(1, retries),
            last_error,
        )
        return getattr(self, "branch_coverage", 0), getattr(self, "line_coverage", 0)

    _webrled_config.WebConfig.get_cov_jacoco = _safe_get_cov_jacoco
except Exception:
    pass

runpy.run_path(os.path.join(_cwd, "src", "main.py"), run_name="__main__")
