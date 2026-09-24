# Reproducing the current WebCover entry

This guide covers `profiles/webcover_paper_uaa.json` and
`scripts/run_webcover_paper.py`, not arbitrary historical scripts or baselines.
The supported launch path is Windows x64, Python 3.11, and Windows PowerShell.
Linux, a fresh-machine benchmark deployment, and a full run in the new isolated
environment have not yet been validated. Passing unit tests is not an empirical
coverage or performance guarantee.

## 1. Isolated dependencies

Run from the repository root. Use a new virtual environment rather than an
existing Conda environment with packages shadowed by a user-site installation.

```powershell
py -3.11 -m venv .venv-repro
& .\.venv-repro\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
& .\.venv-repro\Scripts\python.exe -m pip install -r requirements-reproduction.lock.txt
& .\.venv-repro\Scripts\python.exe -m pip check
& .\.venv-repro\Scripts\python.exe -m pytest -q
```

`requirements-reproduction.txt` lists the direct dependencies. The lock file
includes the transitive versions installed in the validation environment; it is
not a reconstruction of every historical environment. The launcher checks
actual imported module versions and locations, not only package metadata.
Evidence-integration tests and campaign-protocol modules skip when local campaign records are absent;
the method and entry unit tests must still pass.

## 2. Browser and embedding assets

Chrome, ChromeDriver and GloVe vectors are not bundled. In **your own checkout**,
set `browser_path` and `driver_path` in `settings.yaml` to real absolute paths
for a compatible browser/driver pair; use forward slashes in YAML paths. Do not
edit settings belonging to an in-progress frozen campaign.

Browser template selection is the site's `*-subweb-frontier-a2c-1agent` profile,
then its `*-qlearning-1agent` profile; the existing Nextcloud bootstrap can fall
back to `realworld-subweb-frontier-a2c-1agent`. Keep the other profile settings
unchanged. These templates select browser assets, not a different algorithm.
The checker prints the resolved files and hashes before any run.

Local reference assets were Chrome `98.0.4758.0` and ChromeDriver
`98.0.4758.102`. They are legacy software, not a recommendation for general
browsing. Use isolated test systems. Matching major versions is only a driver
compatibility check; different browser builds may change exploration results.
The reference executable hashes are in the local readiness preflight report.

Provide `glove-wiki-gigaword-200` as a text/gzip file with 200-dimensional vectors.
The default location is
`~/gensim-data/glove-wiki-gigaword-200/glove-wiki-gigaword-200.gz`; an existing
asset elsewhere can be selected with `--glove`. Provision it before the timed
run using your approved asset source. Runtime downloads are disabled. The
checker examines its first vector/header and records its SHA-256; a run whose
loader falls back to empty embeddings is invalid, even if it writes a finish
JSON. A header check alone is not a full asset-content validation.

## 3. Prepare a disposable benchmark

| Site | Current entry | Reset requirement |
|---|---|---|
| PetClinic | `http://localhost:8081` | Reset the dedicated H2 test container before each run |
| SplittyPie | `http://localhost:4200` | Fresh browser/event; backend equivalence is not guaranteed |
| Nextcloud | `http://localhost:8082/apps/files/` | Reset a dedicated test database/files snapshot and supply test login |
| RealWorld | `https://demo.realworld.show` | Existing login flow; the shared public backend is not resettable |

The convenience entry does **not** create/reset containers, provision logins,
or guarantee that the target is the intended app. Use only authorized test
deployments. Container image names using `latest` are not immutable versions.
Local image IDs, ports, reset procedures and limitations are recorded under
`rebuttal/RQ2/uaa_paper_uniform_20260924/`; this evidence is not yet a portable
public deployment recipe. Never publish private database/files snapshots or
credentials. Public redistribution and fresh-machine deployment remain release
checks, not completed claims.

PetClinic deliberately exposes `/oups`. Its exact HTTP 500 is retained in
`bug.log` but exempted from the fatal-error streak in both UAA arms. It must not
be counted as a newly discovered defect. This does not exempt other errors.

## 4. Check, then run

```powershell
& .\.venv-repro\Scripts\python.exe .\scripts\check_webcover_reproduction.py --site petclinic
& .\.venv-repro\Scripts\python.exe .\scripts\run_webcover_paper.py --site petclinic --seed seedpaperon1 --uaa on --check
```

The first command checks dependencies and local assets without launching a
browser or accessing the site. The second prints the resolved run plan only;
`--check` is not an environment/endpoint validation. For a non-default embedding
path, pass the same `--glove "D:/assets/glove-wiki-gigaword-200.gz"` to both.

After the site is prepared and **no other experiment is running**:

```powershell
& .\.venv-repro\Scripts\python.exe .\scripts\run_webcover_paper.py --site petclinic --seed seedpaperon1 --uaa on
```

The entry rejects active experiment processes, missing/wrong assets and
dependencies, inherited experiment overrides, and pre-existing run output.
It removes `PYTHONPATH`/`PYTHONHOME` overrides and disables user-site imports in
the child. Use a fresh label for a new attempt; do not overwrite failures.
Training uses the digits in the seed label as the numeric RNG input. For a
paired loss ablation, `seedpaperon1` and `seedpaperoff1` both map to seed 1.
Run them serially and independently reset the benchmark between arms.

The explicit configuration is GAE rollout 64, lambda 0.98, gamma 1, marginal
reward with `home_zero` weighting, A2C with three update epochs, and uniform UAA
over unresolved valid source-action prefixes. `--uaa off` disables only that
auxiliary loss; structural/action-coverage inputs stay on. Budget is 3600
seconds **or** 2200 recorded interactions, whichever stops the run first.
Three update epochs alone do not turn this loss into PPO.

## 5. Validate and retain evidence

For each attempt, keep both `webtest_output/result/<run>/` and
`rebuttal/reproduction_checks/<site>-<seed>/`, plus the root runner log.
The latter directory contains requested settings, dependency/asset checks,
stdout/stderr and completion status. A successful process exit or a JSON file
alone is insufficient: the entry checks the actual logged agent settings,
nonempty/full-budget output and early-abort indicators before marking complete.

Report `states`, unique `(src, action, dst)` transitions, and
`interactions = len(transition_list)` with realized budgets. For repeated runs,
retain every run and summarize run-level mean/sample SD. Inspect failures and
reset evidence; do not replace weak seeds with better historical runs.

Historical non-uniform auxiliary configurations are kept as historical evidence,
not rebranded as this entry. Legacy `run_experiments.ps1` defaults alone are not
equivalent to the explicit profile. Before publication, ensure required new
files are actually tracked, exclude private records, and validate a fresh clone
plus a full benchmark run. This working directory is not proof that those
release steps have been completed.
