# WebCover

Working source snapshot for WebCover and its web-testing baseline integrations.
This first upload intentionally preserves a broad set of code and utilities for
subsequent cleanup. It is **not yet a finalized, independently reproduced paper
artifact**. Historical utilities are not all part of the paper configuration.

## Contents

- `agent/`, `model/`, `state/`, `action/`, `transformer/`: policies, coverage-aware
  observations, action representations, and shared dependencies.
- `main.py`, `web_test/`, `config/`, `settings.yaml`: execution harness and site
  initialization. Configure deployments and browser paths before running.
- `observation/`, `data_collector/`, `fairness.py`: common observation, collection,
  and measurement support.
- `profiles/webcover_paper_uaa.json`, `scripts/run_webcover_paper.py`: explicit
  GAE64 / uniform uncovered-action auxiliary profile and guarded entry point.
- `agent/impl/webexplor_agent.py`, `state/impl/tag_sequence_state.py`,
  `scripts/start_webexplor_paper_3seed_1h.ps1`: the retained local WebExplor
  reproduction, not a claim of unmodified upstream WebExplor code.
- `external/QExplore/`, `external/webrled_official_src/`, WebQT integration,
  `run_webrled_official.py`: baseline source snapshots and local adapters.
- `scripts/`, `tools/`, root PowerShell runners: broad experiment, coverage, and
  diagnostic utilities, including legacy site-specific helpers.
- `tests/`, dependency files, `docs/REPRODUCING.md`: tests and setup notes.

## Configuration and Validation

Use a separate Python environment. See `requirements-reproduction.txt` and its
lock file; external baselines also have their own dependencies. Browser binaries,
drivers, pretrained weights/embeddings, and benchmark containers are not bundled.
The placeholder browser paths in `settings.yaml` must be configured locally.

The explicit entry currently names PetClinic, Nextcloud, Realworld, and SplittyPie.
Inspect a plan without launching a browser:

```powershell
python scripts/run_webcover_paper.py --site petclinic --seed seed123 --check
python -m pytest -q
```

Deployment credentials are supplied locally through `WEBTEST_<SITE>_*`
environment variables; see `docs/LOCAL_CONFIGURATION.md`. Never commit real
credentials. Local campaign-protocol tests skip when their private run plans are
absent. Passing unit tests is not a substitute for end-to-end benchmark validation.
Some retained helpers still assume the original Windows deployment and need
portability review. In particular, SplittyPie deployment readiness must be checked
before treating any run as an experimental result.

The explicit profile fixes rollout length 64, GAE lambda 0.98, marginal reward,
A2C optimization, and uniform UAA. Generic and historical entry points can have
different defaults; do not treat their outputs as interchangeable. Historical
target modes are preserved for inspection, not silently relabeled as uniform.

## Data and Scope

Raw results, browser logs, credentials, private campaign records, the paper PDF,
and large downloaded assets are not in this source snapshot. Existing experiment
data remain in their original local directories. No active experiment was moved
to this checkout. Unique transitions use `(source, action, destination)`;
interactions count all recorded transition entries. Coverage and browser errors
are distinct measurements; an error log entry is not a confirmed defect.

See `THIRD_PARTY_NOTICES.md` and the retained licenses before redistribution.

