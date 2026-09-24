param(
  [string]$Sites = "odoo,4gaboards,github,petclinic,agilefant,splittypie,realworld,nextcloud,gadael,timeoff",
  [string]$Seeds = "seedqexplore1,seedqexplore2,seedqexplore3",
  [int]$ActivityTime = 3600,
  [int]$Depth = 100,
  [string]$Python = "",
  [switch]$Coverage,
  [switch]$DryRun,
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$qexploreScript = Join-Path $repo "external\QExplore\scripts\start_qexplore_3seed_1h.ps1"

if (-not $Python) {
  if ($env:QEXPLORE_PYTHON) {
    $Python = $env:QEXPLORE_PYTHON
  } else {
    $Python = "python"
  }
}

Write-Host "[qexplore-batch] python=$Python"

& $qexploreScript `
  -Sites $Sites `
  -Seeds $Seeds `
  -ActivityTime $ActivityTime `
  -Depth $Depth `
  -Python $Python `
  -Coverage:$Coverage `
  -DryRun:$DryRun `
  -Force:$Force

if (-not $DryRun) {
  & $Python (Join-Path $repo "scripts\summarize_external_baseline_metrics.py") `
    --only qexplore `
    --sites $Sites `
    --seeds $Seeds `
    --output-csv (Join-Path $repo "analysis\external_baselines\qexplore_latest.csv") `
    --print-table
}
