param(
  [string]$Sites = "agilefant,gadael,timeoff,splittypie,petclinic",
  [string]$Seeds = "seedextcov1_20260622,seedextcov2_20260622,seedextcov3_20260622,seedextcov4_20260622,seedextcov5_20260622",
  [ValidateSet("all", "webrled", "qexplore")]
  [string]$Only = "all",
  [int]$StaleMinutes = 30,
  [switch]$IncludeMissing,
  [switch]$SkipCoverageEndpointCheck
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Write-Host "[codecov-watch] sites=$Sites"
Write-Host "[codecov-watch] seeds=$Seeds"
Write-Host "[codecov-watch] only=$Only stale_minutes=$StaleMinutes"

if (-not $SkipCoverageEndpointCheck) {
  Write-Host ""
  Write-Host "== Coverage endpoint readiness =="
  powershell -ExecutionPolicy Bypass -File .\scripts\check_webrled_code_coverage_setup.ps1 -Detailed
}

Write-Host ""
Write-Host "== External WebRLED/QExplore states-actions-codecov monitor =="
$csv = "analysis\external_baselines\external_codecov_monitor_latest.csv"
$argsList = @(
  ".\scripts\monitor_external_baseline_runs.py",
  "--stale-minutes", [string]$StaleMinutes,
  "--only", $Only,
  "--sites", $Sites,
  "--seeds", $Seeds,
  "--output-csv", $csv,
  "--limit", "200"
)
if ($IncludeMissing) {
  $argsList += "--include-missing"
}
python @argsList
Write-Host ""
Write-Host "[codecov-watch] wrote $csv"

Write-Host ""
Write-Host "== Generic WebTest code coverage log monitor =="
$genericCsv = "analysis\external_baselines\generic_codecov_monitor_latest.csv"
python .\scripts\monitor_code_coverage_logs.py `
  --stale-minutes $StaleMinutes `
  --sites $Sites `
  --seeds $Seeds `
  --output-csv $genericCsv `
  --max-files 200 `
  --limit 80
Write-Host ""
Write-Host "[codecov-watch] wrote $genericCsv"
