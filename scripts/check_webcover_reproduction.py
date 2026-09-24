"""Read-only reproduction checks. Never downloads assets or launches a browser."""
import argparse
import gzip
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    'torch': ('torch', '2.6.0'), 'numpy': ('numpy', '1.26.4'),
    'selenium': ('selenium', '4.22.0'), 'PyYAML': ('yaml', '6.0.3'),
    'urllib3': ('urllib3', '2.4.0'), 'scikit-learn': ('sklearn', '1.5.2'),
    'beautifulsoup4': ('bs4', '4.12.3'), 'gensim': ('gensim', '4.3.3'),
    'scipy': ('scipy', '1.13.1'), 'translate': ('translate', '3.6.1'),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def embedding_dimension(path):
    opener = gzip.open if str(path).lower().endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        parts = f.readline().split()
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return int(parts[1]) if int(parts[0]) > 0 else 0
    if len(parts) > 1:
        for value in parts[1:]:
            float(value)
        return len(parts) - 1
    return 0


def resolve_browser_profile(settings, site):
    profiles = settings['profiles']
    name = site + '-subweb-frontier-a2c-1agent'
    if name in profiles:
        return profiles[name]
    # Mirror the existing site's browser-template selection, not its algorithm.
    template = site + '-qlearning-1agent'
    if template in profiles:
        return profiles[template]
    if site == 'nextcloud':
        return profiles['realworld-subweb-frontier-a2c-1agent']
    raise ValueError(f'No supported browser profile for {site}')


def check_environment(site, glove_path, root=ROOT):
    root = Path(root)
    checks, versions = [], {}

    def check(name, ok, detail):
        checks.append({'name': name, 'ok': bool(ok), 'detail': str(detail)})

    check('platform', sys.platform == 'win32', 'This launch path is Windows/PowerShell only')
    check('python', sys.version_info[:2] == (3, 11), sys.version)
    check('isolated_environment', sys.prefix != sys.base_prefix,
          'Use a fresh venv, not a mixed Conda/user-site interpreter')
    check('powershell', shutil.which('powershell.exe') is not None, shutil.which('powershell.exe'))
    for distribution, (module, expected) in MODULES.items():
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, '__version__', None) or importlib.metadata.version(distribution)
            origin = Path(imported.__file__).resolve()
            versions[distribution] = {'version': str(version), 'loaded_from': str(origin)}
            check('dependency:' + distribution, str(version).split('+')[0] == expected,
                  f'imported={version}, expected={expected}')
            check('module_origin:' + distribution, origin.is_relative_to(Path(sys.prefix).resolve()), origin)
        except Exception as exc:
            check('dependency:' + distribution, False, exc)
    assets = {}
    try:
        import yaml
        profile = resolve_browser_profile(yaml.safe_load((root / 'settings.yaml').read_text(encoding='utf-8')), site)
        for key in ('browser_path', 'driver_path'):
            path = Path(profile[key]).expanduser()
            if not path.is_absolute():
                path = root / path
            check(key, path.is_file(), path)
            if path.is_file():
                assets[key] = {'path': str(path.resolve()), 'sha256': sha256(path)}
        if sys.platform == 'win32' and 'browser_path' in assets and 'driver_path' in assets:
            quoted = assets['browser_path']['path'].replace("'", "''")
            chrome = subprocess.check_output(
                ['powershell.exe', '-NoProfile', '-Command', f"(Get-Item -LiteralPath '{quoted}').VersionInfo.ProductVersion"],
                text=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW).strip()
            driver = subprocess.check_output([assets['driver_path']['path'], '--version'],
                text=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW).strip()
            c, d = re.search(r'\d+\.\d+\.\d+\.\d+', chrome), re.search(r'\d+\.\d+\.\d+\.\d+', driver)
            check('browser_driver_major_match', c and d and c[0].split('.')[0] == d[0].split('.')[0],
                  f'Chrome={chrome}; driver={driver}')
            assets['browser_path']['version'] = chrome
            assets['driver_path']['version'] = driver
        resources = Path(profile['resources_path']).expanduser()
        if not resources.is_absolute():
            resources = root / resources
        check('detector_resource', (resources / 'js/action_detector.js').is_file(), resources)
    except Exception as exc:
        check('browser_profile', False, exc)
    glove = Path(glove_path).expanduser()
    try:
        dimension = embedding_dimension(glove)
        check('glove_dimension', dimension == 200, f'{glove}: dimensions={dimension}')
        assets['glove'] = {'path': str(glove.resolve()), 'sha256': sha256(glove), 'dimensions': dimension}
    except Exception as exc:
        check('glove', False, exc)
    return {'ok': all(c['ok'] for c in checks), 'site': site,
            'python_executable': sys.executable, 'checks': checks, 'versions': versions, 'assets': assets,
            'scope': 'Read-only dependencies/assets check; not a server reset, browser session, or performance validation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', choices=['petclinic', 'splittypie', 'nextcloud', 'realworld'], required=True)
    parser.add_argument('--glove', type=Path, default=Path.home() / 'gensim-data/glove-wiki-gigaword-200/glove-wiki-gigaword-200.gz')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = check_environment(args.site, args.glove)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()
