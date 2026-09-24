import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('paper_entry', ROOT / 'scripts/run_webcover_paper.py')
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)
CAMPAIGN = ROOT / 'rebuttal/RQ2/uaa_paper_uniform_20260924/run_plan.json'


class PaperEntryTests(unittest.TestCase):
    def test_all_four_sites_have_explicit_uniform_gae64(self):
        for site in ('petclinic', 'splittypie', 'nextcloud', 'realworld'):
            env = entry.make_plan(site, 'seed1', inherited={})['environment']
            self.assertEqual(env['WEBTEST_ROLLOUT_LEN'], '64')
            self.assertEqual(env['WEBTEST_A2C_ADVANTAGE_ESTIMATOR'], 'gae')
            self.assertEqual(env['WEBTEST_GRAPH_RESIDUAL_AUX_TARGET'], 'uncovered')
            self.assertEqual(env['WEBTEST_GRAPH_RESIDUAL_AUX_DISTRIBUTION'], 'uniform')

    def test_only_auxiliary_loss_flag_differs(self):
        for site in ('petclinic', 'splittypie', 'nextcloud', 'realworld'):
            on = entry.make_plan(site, 'seed1', 'on', {})['environment']
            off = entry.make_plan(site, 'seed1', 'off', {})['environment']
            self.assertEqual({k for k in on.keys() | off.keys() if on.get(k) != off.get(k)},
                             {'WEBTEST_GRAPH_RESIDUAL_AUX'})

    def test_inherited_experiment_overrides_are_not_applied(self):
        inherited = {'PATH': 'test-path', 'SystemRoot': 'Windows',
                     'WEBTEST_GRAPH_RESIDUAL_AUX_DISTRIBUTION': 'max_only',
                     'WEBTEST_A2C_ADVANTAGE_ESTIMATOR': 'n_step',
                     'WEBTEST_ARBITRARY_OLD_SWITCH': '1', 'WEBQT_OLD_SWITCH': '1'}
        plan = entry.make_plan('petclinic', 'seed1', inherited=inherited)
        env = entry.child_environment(plan, inherited)
        self.assertEqual(env['WEBTEST_GRAPH_RESIDUAL_AUX_DISTRIBUTION'], 'uniform')
        self.assertEqual(env['WEBTEST_A2C_ADVANTAGE_ESTIMATOR'], 'gae')
        self.assertEqual(env['SystemRoot'], 'Windows')
        self.assertNotIn('WEBTEST_ARBITRARY_OLD_SWITCH', env)
        self.assertNotIn('WEBQT_OLD_SWITCH', env)
        self.assertIn('WEBTEST_GRAPH_RESIDUAL_AUX_DISTRIBUTION', plan['ignored_inherited_experiment_keys'])
        self.assertEqual(inherited['WEBTEST_GRAPH_RESIDUAL_AUX_DISTRIBUTION'], 'max_only')

    def test_credentials_forwarded_without_recording_values_in_plan(self):
        inherited = {'WEBTEST_NEXTCLOUD_USERNAME': 'fixture-user',
                     'WEBTEST_NEXTCLOUD_PASSWORD': 'fixture-secret-not-a-real-password',
                     'WEBTEST_ARBITRARY_PASSWORD': 'not-allowed'}
        plan = entry.make_plan('nextcloud', 'seed1', inherited=inherited)
        env = entry.child_environment(plan, inherited)
        self.assertEqual(env['WEBTEST_NEXTCLOUD_PASSWORD'], inherited['WEBTEST_NEXTCLOUD_PASSWORD'])
        self.assertNotIn('WEBTEST_ARBITRARY_PASSWORD', env)
        self.assertNotIn(inherited['WEBTEST_NEXTCLOUD_PASSWORD'], json.dumps(plan))

    def test_paired_seed_numeric_input_is_preserved(self):
        plan = entry.make_plan('petclinic', 'seeduaapaperonpetclinic1_20260924', inherited={})
        self.assertEqual(plan['environment']['PYTHONHASHSEED'], '120260924')

    def test_bad_seed_labels_are_rejected(self):
        for seed in ('../seed1', 'seed', 'seed4294967296', 'seed1;whoami'):
            with self.assertRaises(ValueError):
                entry.make_plan('petclinic', seed, inherited={})

    def test_expected_error_exemption_is_petclinic_only(self):
        for site in ('petclinic', 'splittypie', 'nextcloud', 'realworld'):
            env = entry.make_plan(site, 'seed1', inherited={})['environment']
            expected = ['http://localhost:8081/oups'] if site == 'petclinic' else []
            self.assertEqual(json.loads(env['WEBTEST_EXPECTED_HTTP_500_URLS']), expected)

    @unittest.skipUnless(CAMPAIGN.exists(), 'Local frozen experiment plan is not bundled')
    def test_profile_matches_all_40_frozen_run_environments(self):
        for row in json.loads(CAMPAIGN.read_text())['runs']:
            new = entry.make_plan(row['site'], row['seed'], row['arm'], {})['environment']
            self.assertEqual({key: new[key] for key in row['environment']}, row['environment'])

    def test_busy_check_fails_closed(self):
        with patch.object(entry.subprocess, 'run') as run:
            run.return_value.returncode = 2
            run.return_value.stdout = 'active'
            run.return_value.stderr = ''
            with self.assertRaises(RuntimeError):
                entry.ensure_idle()

    def test_runtime_header_is_required(self):
        with self.assertRaises(RuntimeError):
            entry.verify_runtime('request only', entry.make_plan('petclinic', 'seed1', inherited={}))

    @unittest.skipUnless(CAMPAIGN.exists(), 'Local runtime evidence is not bundled')
    def test_actual_uniform_runtime_and_mismatch_detection(self):
        run = CAMPAIGN.parent / 'runs/seeduaapaperonpetclinic1_20260924/run.json'
        if not run.exists():
            self.skipTest('No completed runtime evidence yet')
        header = json.loads(run.read_text())['verified_runtime_config']
        plan = entry.make_plan('petclinic', 'seed1', inherited={})
        self.assertEqual(entry.verify_runtime(header, plan), header)
        for altered in (header.replace('distribution=uniform', 'distribution=max_only'),
                        header.replace('advantage_estimator=gae', 'advantage_estimator=n_step'),
                        header.replace('gae_lambda=0.980', 'gae_lambda=nan'),
                        header + '\nAbort run early: example'):
            with self.assertRaises(RuntimeError):
                entry.verify_runtime(altered, plan)


if __name__ == '__main__':
    unittest.main()
