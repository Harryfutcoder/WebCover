"""Explicit paper-UAA entry; historical experiment records are never rewritten."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROFILE = ROOT / 'profiles/webcover_paper_uaa.json'
PREFIXES = ('WEBTEST_', 'WEBEXPLOR_', 'WEBQT_')
CREDENTIAL_KEYS = frozenset(
    f'WEBTEST_{site}_{field}'
    for site, fields in {
        'NEXTCLOUD': ('USERNAME', 'PASSWORD'),
        'REALWORLD': ('USERNAME', 'EMAIL', 'PASSWORD'),
        'AGILEFANT': ('USERNAME', 'PASSWORD'),
        '4GABOARDS': ('USERNAME', 'EMAIL', 'PASSWORD'),
        'TIMEOFF': ('EMAIL', 'PASSWORD'),
        'GADAEL': ('EMAIL', 'PASSWORD'),
        'ODOO': ('DB', 'LOGIN', 'PASSWORD'),
        'PAGEKIT': ('USERNAME', 'PASSWORD'),
    }.items()
    for field in fields
)


def make_plan(site, seed, uaa='on', inherited=None, glove=None):
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    if site not in profile['sites'] or uaa not in ('on', 'off'):
        raise ValueError('Unsupported site or UAA arm')
    seed = seed.lower()
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', seed):
        raise ValueError('Seed labels may contain letters, digits, underscores and hyphens only')
    numeric = re.sub(r'[^0-9]', '', seed)
    if not numeric or int(numeric) >= 2**32:
        raise ValueError('Seed label must yield a reproducible unsigned 32-bit numeric seed')
    environment = dict(profile['environment'])
    required = {
        'WEBTEST_ROLLOUT_LEN': '64', 'WEBTEST_A2C_ADVANTAGE_ESTIMATOR': 'gae',
        'WEBTEST_A2C_GAE_LAMBDA': '0.98', 'WEBTEST_SUBWEB_REWARD_MODE': 'marginal',
        'WEBTEST_SUBWEB_OPTIMIZER': 'a2c', 'WEBTEST_GRAPH_RESIDUAL_AUX_TARGET': 'uncovered',
        'WEBTEST_GRAPH_RESIDUAL_AUX_DISTRIBUTION': 'uniform',
        'WEBTEST_GRAPH_RESIDUAL_AUX_TARGET_POWER': '1.0',
        'WEBTEST_GRAPH_RESIDUAL_AUX_TIE_BREAK_COEF': '0.0',
    }
    if any(environment.get(k) != v for k, v in required.items()):
        raise ValueError('Profile does not match the paper-uniform UAA entry contract')
    deployment = profile['sites'][site]
    environment.update({
        'WEBTEST_GRAPH_RESIDUAL_AUX': '1' if uaa == 'on' else '0',
        f'WEBTEST_SITE_{site.upper()}_ENTRY_URL': deployment['url'],
        f'WEBTEST_SITE_{site.upper()}_DOMAINS': urlsplit(deployment['url']).netloc,
        'WEBTEST_EXPECTED_HTTP_500_URLS': json.dumps(deployment['expected_http_500_urls'], separators=(',', ':')),
        'WEBTEST_MAIN_PYTHON': sys.executable,
        'WEBTEST_GLOBAL_BROWSER_CLEANUP': '0',
        'PYTHONHASHSEED': numeric,
        'PYTHONUNBUFFERED': '1',
        'PYTHONNOUSERSITE': '1',
        'WEBTEST_GLOVE_PATH': str(Path(glove or Path.home() / 'gensim-data/glove-wiki-gigaword-200/glove-wiki-gigaword-200.gz').expanduser().resolve()),
        'WEBTEST_ALLOW_GENSIM_DOWNLOAD': '0',
    })
    inherited = os.environ if inherited is None else inherited
    return {
        'profile_id': profile['profile_id'], 'site': site, 'seed': seed, 'uaa': uaa,
        'environment': environment,
        'ignored_inherited_experiment_keys': sorted(k for k in inherited if k.startswith(PREFIXES)),
        'result_dir': str(ROOT / 'webtest_output/result' / f'{site}-subweb-frontier-a2c-1agent-subweb-frontier-a2c-{seed}'),
        'profile_sha256': hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
        'entry_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reset_policy': 'No implicit server reset. Controlled ablations must use the frozen serial campaign supervisor.',
    }


def child_environment(plan, inherited=None):
    inherited = os.environ if inherited is None else inherited
    env = {k: v for k, v in inherited.items() if not k.startswith(PREFIXES)}
    env.update({k: v for k, v in inherited.items() if k in CREDENTIAL_KEYS})
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env.update(plan['environment'])
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    return env


def verify_runtime(log, plan):
    if 'using empty embedding model' in log or re.search(r'Loaded vocab size\s+0\b', log):
        raise RuntimeError('Empty text embeddings; not a valid reproduction of the configured method')
    headers = [line for line in log.splitlines() if '[config]' in line and 'rollout_len=' in line]
    if not headers:
        raise RuntimeError('Missing actual agent configuration; requested settings alone are not evidence')
    fields = dict(re.findall(r'([a-zA-Z_][a-zA-Z_0-9]*)=([^\s]+)', headers[0]))
    expected = {'rollout_len': '64', 'advantage_estimator': 'gae', 'policy_baseline': 'critic',
                'reward_mode': 'marginal', 'optimizer_mode': 'a2c', 'update_epochs': '3',
                'graph_residual_aux_target': 'uncovered', 'graph_residual_aux_distribution': 'uniform',
                'graph_residual_aux': plan['environment']['WEBTEST_GRAPH_RESIDUAL_AUX'],
                'structural_features': '1', 'action_coverage_features': '1', 'candidate_q_aux': '0',
                'max_actions': '512', 'policy_input_mode': 'augmented',
                'normalize_advantages': '1', 'separate_grad_clip': '1',
                'reward_clip_abs': 'disabled', 'graph_filter_template_edges': '0',
                'graph_edge_weight_mode': 'home_zero', 'graph_node_mode': 'agent_state',
                'graph_edge_label_mode': 'observer_action'}
    for key, value in expected.items():
        if fields.get(key) != value:
            raise RuntimeError(f'Actual runtime configuration mismatch: {key}')
    for key, value in {'gae_lambda': .98, 'gamma': 1., 'graph_residual_aux_coef': .2,
                       'value_coef': .5, 'max_grad_norm': .5,
                       'graph_node_weight': 1., 'graph_edge_alpha': 1.,
                       'lr': .001, 'entropy_coef': .01, 'graph_residual_aux_target_power': 1.,
                       'graph_residual_aux_tie_break_coef': 0.}.items():
        if key not in fields or not abs(float(fields[key]) - value) <= 1e-7:
            raise RuntimeError(f'Actual runtime configuration mismatch: {key}')
    if 'Abort run early:' in log or 'Traceback (most recent call last)' in log:
        raise RuntimeError('Run contains an early-abort or traceback; inspect raw logs')
    return headers[0]


def ensure_idle():
    command = r"""$ErrorActionPreference='Stop';
    $busy = @(Get-CimInstance Win32_Process | Where-Object {
      $_.ProcessId -ne $PID -and $_.Name -match '^(python(w)?|powershell|pwsh)\.exe$' -and
      $_.CommandLine -match '(run_serial\.py|run_experiments\.ps1|\bmain\.py\b)'
    });
    if ($busy.Count) { $busy | Select-Object ProcessId,Name | ConvertTo-Json -Compress; exit 2 }
    """
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
                            capture_output=True, text=True, timeout=30,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError('Another experiment is active, or the busy check failed. '
                           'No run was started. ' + result.stdout.strip() + result.stderr.strip())


def execute(plan):
    # This launcher must not contend with the frozen serial ablation campaign.
    ensure_idle()
    from scripts.check_webcover_reproduction import check_environment
    preflight = check_environment(plan['site'], plan['environment']['WEBTEST_GLOVE_PATH'])
    if not preflight['ok']:
        failures = [c for c in preflight['checks'] if not c['ok']]
        raise RuntimeError('Reproduction preflight failed before launch: ' + json.dumps(failures))
    import msvcrt
    lock_path = ROOT / 'webtest_output/paper_entry.lock'
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        result_dir = Path(plan['result_dir'])
        record = ROOT / 'rebuttal/reproduction_checks' / f"{plan['site']}-{plan['seed']}"
        if result_dir.exists() or record.exists():
            raise RuntimeError('Existing result or run record; refusing overwrite. Choose a fresh label.')
        record.mkdir(parents=True)
        (record / 'requested.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
        (record / 'preflight.json').write_text(json.dumps(preflight, indent=2), encoding='utf-8')
        args = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                '-File', str(ROOT / 'run_experiments.ps1'), '-Sites', plan['site'],
                '-Baselines', 'subweb-frontier-a2c', '-Seeds', plan['seed'],
                '-ForceSystemChrome', '0', '-ForceHeadful', '0', '-UpdateEpochs', '3']
        status = {'status': 'running', 'started_unix': time.time()}
        process = None
        try:
            with (record / 'stdout.log').open('wb') as out, (record / 'stderr.log').open('wb') as err:
                process = subprocess.Popen(args, cwd=ROOT, env=child_environment(plan), stdout=out, stderr=err,
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                status['pid'] = process.pid
                (record / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
                exit_code = process.wait(timeout=4500)
            finishes = list((result_dir / 'output_data').glob('*-finish.json'))
            if exit_code or len(finishes) != 1:
                raise RuntimeError(f'Runner exit={exit_code}; finish JSON count={len(finishes)}. Inspect raw logs.')
            runner_log = ROOT / f"{plan['site']}-subweb-frontier-a2c-1agent_subweb-frontier-a2c_{plan['seed']}.log"
            raw = runner_log.read_bytes()
            log = raw.decode('utf-16' if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig', 'replace')
            status['verified_runtime_config'] = verify_runtime(log, plan)
            data = json.loads(finishes[0].read_text())
            interactions = len(data['transition_list'])
            if interactions == 0 or (time.time() - status['started_unix'] < 3500 and interactions < 2200):
                raise RuntimeError('Empty or early finish; do not count as a full-budget experiment.')
            status.update(status='complete', finish_json=str(finishes[0]), interactions=interactions)
        except BaseException as exc:
            if process is not None and process.poll() is None:
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               capture_output=True, timeout=30,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            status.update(status='failed_or_interrupted', error=str(exc))
            raise
        finally:
            status['finished_unix'] = time.time()
            (record / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        print(f'Finished with verified runtime config. Raw evidence: {record}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', choices=['petclinic', 'splittypie', 'nextcloud', 'realworld'], required=True)
    parser.add_argument('--seed', required=True)
    parser.add_argument('--uaa', choices=['on', 'off'], default='on')
    parser.add_argument('--glove', type=Path, help='Path to the 200-dimensional GloVe text/gzip asset; no silent empty fallback')
    parser.add_argument('--check', action='store_true', help='Print resolved plan only; no processes, resets or output files')
    args = parser.parse_args()
    plan = make_plan(args.site, args.seed, args.uaa, glove=args.glove)
    if args.check:
        print(json.dumps(plan, indent=2))
    else:
        execute(plan)


if __name__ == '__main__':
    main()
