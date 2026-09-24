"""Initialize the legacy CLI singleton without consuming pytest's arguments."""
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = tempfile.TemporaryDirectory(prefix='webcover-tests-')
arguments = sys.argv[:]
try:
    sys.argv = [arguments[0], '--settings', str(ROOT / 'settings.yaml'),
                '--output', OUTPUT.name, '--model_path', OUTPUT.name, '--session', 'unit-tests']
    from config import LogConfig
finally:
    sys.argv = arguments


def pytest_sessionfinish(session, exitstatus):
    LogConfig.get_file_handler().close()
    OUTPUT.cleanup()
