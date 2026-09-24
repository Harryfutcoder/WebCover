import gzip
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_webcover_reproduction as doctor
from scripts import run_webcover_paper as entry


class ReproductionChecksTests(unittest.TestCase):
    def test_embedding_header_and_raw_vector_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'glove.txt'
            for text, expected in [('10 200\n', 200), ('0 200\n', 0),
                                   ('word ' + ' '.join(['0.25'] * 200), 200),
                                   ('word 0.25 1.0\n', 2), ('\n', 0)]:
                path.write_text(text, encoding='utf-8')
                self.assertEqual(doctor.embedding_dimension(path), expected)

    def test_embedding_gzip_is_supported_and_corruption_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'glove.gz'
            with gzip.open(path, 'wt', encoding='utf-8') as stream:
                stream.write('3 200\n')
            self.assertEqual(doctor.embedding_dimension(path), 200)
            path.write_bytes(b'not gzip')
            with self.assertRaises(OSError):
                doctor.embedding_dimension(path)

    def test_embedding_non_numeric_vector_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'glove.txt'
            path.write_text('word 1.0 invalid', encoding='utf-8')
            with self.assertRaises(ValueError):
                doctor.embedding_dimension(path)

    def test_browser_profile_order_and_missing_site(self):
        direct, template, fallback = {'id': 1}, {'id': 2}, {'id': 3}
        profiles = {'petclinic-subweb-frontier-a2c-1agent': direct,
                    'petclinic-qlearning-1agent': template,
                    'realworld-subweb-frontier-a2c-1agent': fallback}
        settings = {'profiles': profiles}
        self.assertIs(doctor.resolve_browser_profile(settings, 'petclinic'), direct)
        del profiles['petclinic-subweb-frontier-a2c-1agent']
        self.assertIs(doctor.resolve_browser_profile(settings, 'petclinic'), template)
        self.assertIs(doctor.resolve_browser_profile(settings, 'nextcloud'), fallback)
        with self.assertRaises(ValueError):
            doctor.resolve_browser_profile(settings, 'splittypie')

    def test_custom_asset_and_isolated_child_environment(self):
        asset = Path('test-assets/glove.gz').resolve()
        inherited = {'PYTHONPATH': 'untrusted-packages', 'PYTHONHOME': 'old-python',
                     'PYTHONNOUSERSITE': '0', 'PATH': 'existing-path',
                     'WEBTEST_ALLOW_GENSIM_DOWNLOAD': '1'}
        plan = entry.make_plan('petclinic', 'seed1', inherited=inherited, glove=asset)
        env = entry.child_environment(plan, inherited)
        self.assertNotIn('PYTHONPATH', env)
        self.assertNotIn('PYTHONHOME', env)
        self.assertEqual(env['PYTHONNOUSERSITE'], '1')
        self.assertEqual(env['WEBTEST_GLOVE_PATH'], str(asset))
        self.assertEqual(env['WEBTEST_ALLOW_GENSIM_DOWNLOAD'], '0')
        self.assertEqual(inherited['PYTHONPATH'], 'untrusted-packages')

    def test_empty_embedding_logs_cannot_validate(self):
        plan = entry.make_plan('petclinic', 'seed1', inherited={})
        for text in ('WARNING: using empty embedding model', 'Loaded vocab size 0'):
            with self.assertRaisesRegex(RuntimeError, 'Empty text embeddings'):
                entry.verify_runtime(text, plan)

    def test_failed_preflight_never_launches_runner(self):
        plan = entry.make_plan('petclinic', 'seed1', inherited={})
        with patch.object(entry, 'ensure_idle'), \
                patch.object(doctor, 'check_environment', return_value={
                    'ok': False, 'checks': [{'name': 'glove', 'ok': False}]}), \
                patch.object(entry.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'preflight failed before launch'):
                entry.execute(plan)
            launch.assert_not_called()

    def test_runtime_contract_does_not_depend_on_private_campaign_files(self):
        header = ('[config] rollout_len=64 advantage_estimator=gae policy_baseline=critic '
                  'reward_mode=marginal optimizer_mode=a2c update_epochs=3 '
                  'graph_residual_aux_target=uncovered graph_residual_aux_distribution=uniform '
                  'graph_residual_aux=1 structural_features=1 action_coverage_features=1 '
                  'candidate_q_aux=0 max_actions=512 policy_input_mode=augmented '
                  'normalize_advantages=1 separate_grad_clip=1 reward_clip_abs=disabled '
                  'graph_filter_template_edges=0 graph_edge_weight_mode=home_zero '
                  'graph_node_mode=agent_state graph_edge_label_mode=observer_action '
                  'gae_lambda=0.980 gamma=1.0 graph_residual_aux_coef=0.2 value_coef=0.5 '
                  'max_grad_norm=0.5 graph_node_weight=1.0 graph_edge_alpha=1.0 '
                  'lr=0.001 entropy_coef=0.01 graph_residual_aux_target_power=1.0 '
                  'graph_residual_aux_tie_break_coef=0.0')
        plan = entry.make_plan('petclinic', 'seed1', inherited={})
        self.assertEqual(entry.verify_runtime(header, plan), header)
        for before, after in [('max_actions=512', 'max_actions=64'),
                              ('gae_lambda=0.980', 'gae_lambda=nan'),
                              ('separate_grad_clip=1', 'separate_grad_clip=0'),
                              ('graph_residual_aux=1', 'graph_residual_aux=0'),
                              ('graph_node_weight=1.0', 'graph_node_weight=0.0')]:
            with self.assertRaises(RuntimeError):
                entry.verify_runtime(header.replace(before, after), plan)


if __name__ == '__main__':
    unittest.main()
