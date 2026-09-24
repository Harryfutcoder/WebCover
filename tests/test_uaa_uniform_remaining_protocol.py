import importlib.util
from copy import deepcopy
from pathlib import Path
import unittest
import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'rebuttal/RQ2/uaa_uniform_remaining_20260924/run_serial.py'
if not PATH.is_file():
    pytest.skip('Local campaign records are not part of the source snapshot', allow_module_level=True)
SPEC = importlib.util.spec_from_file_location('uaa_remaining', PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class ContinuationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.plan = RUNNER.read(PATH.parent / 'run_plan.json')

    def test_30_matched_rows_preserve_original_settings(self):
        RUNNER.validate_plan(self.plan)
        self.assertEqual(self.plan['runs'][0]['site'], 'nextcloud')
        self.assertEqual(self.plan['runs'][-1]['site'], 'splittypie')

    def test_failed_retry_preserves_numeric_seed(self):
        retries = [r for r in self.plan['runs'] if 'previous_failed_attempt' in r]
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0]['site'], 'splittypie')
        self.assertIn('_retry_', retries[0]['seed'])
        self.assertEqual(retries[0]['runner_seed_input'], '120260924')

    def test_reject_changed_training_settings(self):
        plan = deepcopy(self.plan)
        plan['runs'][0]['environment']['WEBTEST_A2C_LR'] = '0.002'
        with self.assertRaises(AssertionError):
            RUNNER.validate_plan(plan)

    def test_reject_max_only_target(self):
        plan = deepcopy(self.plan)
        plan['runs'][0]['environment']['WEBTEST_GRAPH_RESIDUAL_AUX_DISTRIBUTION'] = 'max_only'
        with self.assertRaises(AssertionError):
            RUNNER.validate_plan(plan)

    def test_reject_seed_mismatch(self):
        plan = deepcopy(self.plan)
        plan['runs'][0]['seed'] += '99'
        with self.assertRaises(AssertionError):
            RUNNER.validate_plan(plan)

    def test_original_source_still_frozen(self):
        self.assertEqual(len(RUNNER.verify_reference()), 68)


if __name__ == '__main__':
    unittest.main()
