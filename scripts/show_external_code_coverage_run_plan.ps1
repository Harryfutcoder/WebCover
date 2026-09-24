param(
  [string]$Seeds = "seedextcov1_20260622,seedextcov2_20260622,seedextcov3_20260622",
  [int]$TimeLimit = 3600,
  [int]$ForceHeadful = 0,
  [int]$QExploreDepth = 100,
  [switch]$IncludePostRun4ga,
  [switch]$IncludeDryRun
)

$ErrorActionPreference = "Stop"

function Write-Section {
  param([string]$Title)
  Write-Host ""
  Write-Host ("# " + $Title)
}

function Write-Block {
  param([string[]]$Lines)
  Write-Host '```powershell'
  foreach ($line in $Lines) {
    Write-Host $line
  }
  Write-Host '```'
}

$repo = "C:\Users\SUST\artifacts"
$javaLiveSites = "agilefant,gadael,timeoff"
$instrumentedSites = "splittypie,petclinic"
$strictCoverageSites = "$javaLiveSites,$instrumentedSites"

Write-Host "# External State/Action + Code Coverage Run Plan"
Write-Host ""
Write-Host "Strict live coverage sites: $strictCoverageSites"
Write-Host "Post-run coverage only: 4gaboards"
Write-Host "No current server-side coverage: github,odoo,realworld,nextcloud"
Write-Host "Seeds: $Seeds"
Write-Host "TimeLimit/ActivityTime: $TimeLimit"
Write-Host ""
Write-Host "Both WebRLED and QExplore summaries report unique_states, total_actions, unique_actions, and coverage columns when coverage is available."

Write-Section "Preflight"
Write-Block @(
  "cd $repo",
  "powershell -ExecutionPolicy Bypass -File .\scripts\check_external_baseline_prereqs.ps1 -DetailedCoverage",
  "python .\scripts\check_qexplore_imports.py"
)

Write-Section "Setup: SplittyPie And PetClinic Coverage Listeners"
Write-Block @(
  "cd $repo",
  "powershell -ExecutionPolicy Bypass -File .\scripts\start_istanbul_coverage_listener_background.ps1 ``",
  "  -Port 6972 ``",
  "  -OutDir analysis\webrled_code_coverage\istanbul_listener_splittypie ``",
  "  -Name splittypie_6972 ``",
  "  -Force",
  "",
  "powershell -ExecutionPolicy Bypass -File .\scripts\start_istanbul_coverage_listener_background.ps1 ``",
  "  -Port 6973 ``",
  "  -OutDir analysis\webrled_code_coverage\istanbul_listener_petclinic ``",
  "  -Name petclinic_6973 ``",
  "  -Force"
)

Write-Section "Setup: Instrumented SplittyPie And PetClinic Apps"
Write-Block @(
  "cd $repo",
  "powershell -ExecutionPolicy Bypass -File .\scripts\start_webrled_instrumented_first_benchmark.ps1 ``",
  "  -Site splittypie ``",
  "  -CoverageBaseUrl http://localhost:6972",
  "",
  "powershell -ExecutionPolicy Bypass -File .\scripts\start_webrled_instrumented_first_benchmark.ps1 ``",
  "  -Site petclinic ``",
  "  -CoverageBaseUrl http://localhost:6973"
)

Write-Section "Window 1: WebRLED Live Java Coverage"
$lines = @(
  "cd $repo",
  "powershell -ExecutionPolicy Bypass -File .\scripts\start_webrled_official_batch.ps1 ``",
  "  -Sites `"$javaLiveSites`" ``",
  "  -Seeds `"$Seeds`" ``",
  "  -TimeLimit $TimeLimit ``",
  "  -ForceHeadful $ForceHeadful ``",
  "  -GlobalBrowserCleanup 0 ``",
  "  -Coverage"
)
if ($IncludeDryRun) {
  $lines[$lines.Count - 1] = $lines[$lines.Count - 1] + " ``"
  $lines += "  -DryRun"
}
Write-Block $lines

Write-Section "Window 2: QExplore Live Java Coverage"
$lines = @(
  "cd $repo",
  "powershell -ExecutionPolicy Bypass -File .\scripts\start_qexplore_batch.ps1 ``",
  "  -Sites `"$javaLiveSites`" ``",
  "  -Seeds `"$Seeds`" ``",
  "  -ActivityTime $TimeLimit ``",
  "  -Depth $QExploreDepth ``",
  "  -Coverage"
)
if ($IncludeDryRun) {
  $lines[$lines.Count - 1] = $lines[$lines.Count - 1] + " ``"
  $lines += "  -DryRun"
}
Write-Block $lines

Write-Section "Window 3: WebRLED Instrumented First-Benchmark Coverage"
$lines = @(
  "cd $repo",
  "`$env:WEBTEST_SITE_SPLITTYPIE_COVERAGE_URL = `"http://localhost:6972`"",
  "`$env:WEBTEST_SITE_SPLITTYPIE_ENTRY_URL = `"http://localhost:4005`"",
  "`$env:WEBTEST_SITE_SPLITTYPIE_DOMAINS = `"http://localhost:4005`"",
  "`$env:WEBTEST_SITE_PETCLINIC_COVERAGE_URL = `"http://localhost:6973`"",
  "`$env:WEBTEST_SITE_PETCLINIC_ENTRY_URL = `"http://localhost:4002`"",
  "`$env:WEBTEST_SITE_PETCLINIC_DOMAINS = `"http://localhost:4002`"",
  "powershell -ExecutionPolicy Bypass -File .\scripts\start_webrled_official_batch.ps1 ``",
  "  -Sites `"$instrumentedSites`" ``",
  "  -Seeds `"$Seeds`" ``",
  "  -TimeLimit $TimeLimit ``",
  "  -ForceHeadful $ForceHeadful ``",
  "  -GlobalBrowserCleanup 0 ``",
  "  -Coverage"
)
if ($IncludeDryRun) {
  $lines[$lines.Count - 1] = $lines[$lines.Count - 1] + " ``"
  $lines += "  -DryRun"
}
Write-Block $lines

Write-Section "Window 4: QExplore Instrumented First-Benchmark Coverage"
$lines = @(
  "cd $repo",
  "powershell -ExecutionPolicy Bypass -File .\scripts\start_qexplore_batch.ps1 ``",
  "  -Sites `"$instrumentedSites`" ``",
  "  -Seeds `"$Seeds`" ``",
  "  -ActivityTime $TimeLimit ``",
  "  -Depth $QExploreDepth ``",
  "  -Coverage"
)
if ($IncludeDryRun) {
  $lines[$lines.Count - 1] = $lines[$lines.Count - 1] + " ``"
  $lines += "  -DryRun"
}
Write-Block $lines

if ($IncludePostRun4ga) {
  Write-Section "Optional Window 5: 4gaboards Post-Run Coverage"
  Write-Host "4gaboards coverage is container-level post-run nyc, not a live per-run hook. Keep the container isolated while this window runs."
  $lines = @(
    "cd $repo",
    "powershell -ExecutionPolicy Bypass -File .\scripts\start_webrled_official_batch.ps1 ``",
    "  -Sites `"4gaboards`" ``",
    "  -Seeds `"$Seeds`" ``",
    "  -TimeLimit $TimeLimit ``",
    "  -ForceHeadful $ForceHeadful ``",
    "  -GlobalBrowserCleanup 0",
    "",
    "powershell -ExecutionPolicy Bypass -File .\scripts\collect_webrled_postrun_nyc_coverage.ps1 -Site 4gaboards"
  )
  if ($IncludeDryRun) {
    $lines[6] = $lines[6] + " ``"
    $lines = $lines[0..6] + @("  -DryRun") + $lines[7..($lines.Count - 1)]
  }
  Write-Block $lines

  Write-Section "Optional Window 6: QExplore 4gaboards State/Action Counts"
  Write-Host "Run this only if you want QExplore states/actions on 4gaboards. Coverage should still be collected as post-run container coverage, so avoid concurrent traffic."
  $lines = @(
    "cd $repo",
    "powershell -ExecutionPolicy Bypass -File .\scripts\start_qexplore_batch.ps1 ``",
    "  -Sites `"4gaboards`" ``",
    "  -Seeds `"$Seeds`" ``",
    "  -ActivityTime $TimeLimit ``",
    "  -Depth $QExploreDepth",
    "",
    "powershell -ExecutionPolicy Bypass -File .\scripts\collect_webrled_postrun_nyc_coverage.ps1 -Site 4gaboards"
  )
  if ($IncludeDryRun) {
    $lines[5] = $lines[5] + " ``"
    $lines = $lines[0..5] + @("  -DryRun") + $lines[6..($lines.Count - 1)]
  }
  Write-Block $lines
}

Write-Section "Summaries"
Write-Block @(
  "cd $repo",
  "python .\scripts\summarize_external_baseline_metrics.py --only webrled --sites `"$strictCoverageSites`" --seeds `"$Seeds`" --print-table",
  "python .\scripts\summarize_external_baseline_metrics.py --only qexplore --sites `"$strictCoverageSites`" --seeds `"$Seeds`" --print-table",
  "python .\scripts\summarize_external_baseline_metrics.py --sites `"$strictCoverageSites`" --seeds `"$Seeds`" --print-table"
)

Write-Section "Monitor"
Write-Block @(
  "cd $repo",
  "python .\scripts\monitor_external_baseline_runs.py ``",
  "  --sites `"$strictCoverageSites`" ``",
  "  --seeds `"$Seeds`" ``",
  "  --include-missing ``",
  "  --stale-minutes 30 ``",
  "  --output-csv analysis\external_baselines\external_coverage_monitor_latest.csv"
)
