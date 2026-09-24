# Local QExplore Setup

This folder is an isolated clone of `https://github.com/salmansherin/QExplore`.
It is not merged into the main web-testing repo.

## What Was Configured

- Site targets live in `configs/sites.local.json`.
- Runs are launched through `scripts/run_qexplore_site.ps1`.
- Each run gets its own working directory under `runs/<site>/<seed>/`, so the original QExplore outputs `Q_Result`, `Q-table`, and `Q.map` do not overwrite other seeds.
- The wrapper runs QExplore from a per-run directory and sets `PYTHONPATH` to the cloned `Qexplore` source folder.
- The current local default is base `python` plus repo-local compatibility shims, Chrome, and `C:\Users\SUST\Desktop\webTest\webTest\chromedriver.exe`.
- Coverage can be collected in the same run with `-Coverage`; it writes `qexplore_coverage.json` and does not alter Q-learning rewards.

## Dry Run

```powershell
cd C:\Users\SUST\artifacts\external\QExplore

powershell -ExecutionPolicy Bypass -File .\scripts\run_qexplore_site.ps1 `
  -Site splittypie `
  -Seed seedqexploresmoke1 `
  -ActivityTime 60 `
  -DryRun
```

## Preflight

Before launching a real run, check the selected Python and browser runtime:

```powershell
cd C:\Users\SUST\artifacts
python .\scripts\check_qexplore_imports.py
```

The wrapper runs this same preflight automatically before any non-dry-run
browser session. It checks QExplore's imported Python packages plus
Chrome/chromedriver.

This clone provides small local shims for old upstream dependencies that are
not required for the core command-line experiment path. The shims are
compatibility glue, not new exploration logic.

```powershell
cd C:\Users\SUST\artifacts
powershell -ExecutionPolicy Bypass -File .\scripts\setup_qexplore_env.ps1 `
  -RepoRoot C:\Users\SUST\artifacts `
  -InstallBrowser `
  -PrintOnly
```

Override the browser only if needed:

```powershell
$env:QEXPLORE_CHROME_BINARY = "C:\Path\To\chrome.exe"
$env:QEXPLORE_CHROMEDRIVER = "C:\Path\To\chromedriver.exe"
```

## Example 1h Run

Run this only after the currently active experiments finish, because QExplore opens its own browser.

```powershell
cd C:\Users\SUST\artifacts\external\QExplore

powershell -ExecutionPolicy Bypass -File .\scripts\run_qexplore_site.ps1 `
  -Site splittypie `
  -Seed seedqexplore1_20260621 `
  -ActivityTime 3600 `
  -Depth 100
```

## Batch 3-Seed Run

```powershell
cd C:\Users\SUST\artifacts\external\QExplore

powershell -ExecutionPolicy Bypass -File .\scripts\start_qexplore_3seed_1h.ps1 `
  -Sites "splittypie,timeoff,gadael" `
  -Seeds "seedqexplore1_20260621,seedqexplore2_20260621,seedqexplore3_20260621" `
  -ActivityTime 3600
```

## Coverage Smoke

```powershell
cd C:\Users\SUST\artifacts
powershell -ExecutionPolicy Bypass -File .\scripts\start_qexplore_batch.ps1 `
  -Sites "agilefant" `
  -Seeds "seedqexplorecovsmoke2_20260622" `
  -ActivityTime 20 `
  -Depth 10 `
  -Coverage `
  -Force
```

Verified local output:

```text
qexplore,agilefant,seedqexplorecovsmoke2_20260622,ok,
unique_states=1,total_actions=12,unique_actions=6,unique_edges=6,
branch_coverage=31,line_coverage=45,source=qmap+coverage_json
```

## Important Caveats

QExplore is a research prototype. The local clone has a compatibility pass for
new Selenium/Chrome APIs, UTF-8 output, deterministic fallback form inputs,
and per-run metric capture.

The main repo's pre-login/profile machinery is not imported here. Sites such as Nextcloud, TimeOff, Gadael, and Agilefant may need credentials or site-specific bootstrap before QExplore results are comparable.
