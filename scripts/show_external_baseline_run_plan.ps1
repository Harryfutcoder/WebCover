param(
  [string]$Sites = "odoo,4gaboards,github,petclinic,agilefant,splittypie,realworld,nextcloud,gadael,timeoff",
  [string]$Seeds = "seedexternal1_20260622,seedexternal2_20260622,seedexternal3_20260622",
  [int]$TimeLimit = 3600,
  [int]$ForceHeadful = 0,
  [int]$Windows = 3,
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

function Split-IntoGroups {
  param([string[]]$Items, [int]$GroupCount)
  if ($GroupCount -lt 1) { $GroupCount = 1 }
  $groups = @()
  for ($i = 0; $i -lt $GroupCount; $i++) {
    $groups += ,(New-Object System.Collections.Generic.List[string])
  }
  for ($i = 0; $i -lt $Items.Count; $i++) {
    $groups[$i % $GroupCount].Add($Items[$i])
  }
  return $groups
}

$repo = "C:\Users\SUST\artifacts"
$siteList = $Sites.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
if (-not $siteList) {
  throw "No sites provided."
}

$groups = Split-IntoGroups -Items $siteList -GroupCount $Windows
$dry = if ($IncludeDryRun) { " ``" } else { "" }

Write-Host "# External Baseline Run Plan"
Write-Host ""
Write-Host "Sites: $($siteList -join ',')"
Write-Host "Seeds: $Seeds"
Write-Host "TimeLimit: $TimeLimit"
Write-Host "Windows per algorithm: $Windows"

Write-Section "Preflight"
Write-Block @(
  "cd $repo",
  "C:\Users\SUST\.conda\envs\webtest-python\python.exe .\run_webrled_official.py --site github --session preflight --seed preflight --check-only",
  "python .\scripts\check_qexplore_imports.py",
  "# If QExplore preflight fails, check QEXPLORE_PYTHON, QEXPLORE_CHROME_BINARY, and QEXPLORE_CHROMEDRIVER.",
  "powershell -ExecutionPolicy Bypass -File .\scripts\check_webrled_code_coverage_setup.ps1 -Detailed"
)

Write-Section "WebRLED Official Windows"
for ($i = 0; $i -lt $groups.Count; $i++) {
  if ($groups[$i].Count -eq 0) { continue }
  $sitesForWindow = ($groups[$i] -join ",")
  Write-Host ""
  Write-Host "## WebRLED Window $($i + 1)"
  $lines = @(
    "cd $repo",
    "powershell -ExecutionPolicy Bypass -File .\scripts\start_webrled_official_batch.ps1 ``",
    "  -Sites `"$sitesForWindow`" ``",
    "  -Seeds `"$Seeds`" ``",
    "  -TimeLimit $TimeLimit ``",
    "  -ForceHeadful $ForceHeadful ``",
    "  -GlobalBrowserCleanup 0"
  )
  if ($IncludeDryRun) {
    $lines[$lines.Count - 1] = $lines[$lines.Count - 1] + " ``"
    $lines += "  -DryRun"
  }
  Write-Block $lines
}

Write-Section "QExplore Windows"
Write-Host "Run these only after QExplore preflight passes. The local runner defaults to base Python + Chrome/chromedriver."
for ($i = 0; $i -lt $groups.Count; $i++) {
  if ($groups[$i].Count -eq 0) { continue }
  $sitesForWindow = ($groups[$i] -join ",")
  Write-Host ""
  Write-Host "## QExplore Window $($i + 1)"
  $lines = @(
    "cd $repo",
    "powershell -ExecutionPolicy Bypass -File .\scripts\start_qexplore_batch.ps1 ``",
    "  -Sites `"$sitesForWindow`" ``",
    "  -Seeds `"$Seeds`" ``",
    "  -ActivityTime $TimeLimit ``",
    "  -Depth 100"
  )
  if ($IncludeDryRun) {
    $lines[$lines.Count - 1] = $lines[$lines.Count - 1] + " ``"
    $lines += "  -DryRun"
  }
  Write-Block $lines
}

Write-Section "Summaries"
Write-Block @(
  "cd $repo",
  "python .\scripts\summarize_external_baseline_metrics.py --only webrled --seeds `"$Seeds`" --print-table",
  "python .\scripts\summarize_external_baseline_metrics.py --only qexplore --seeds `"$Seeds`" --print-table",
  "python .\scripts\summarize_external_baseline_metrics.py --seeds `"$Seeds`" --print-table"
)

Write-Section "Monitor"
Write-Block @(
  "cd $repo",
  "python .\scripts\monitor_external_baseline_runs.py ``",
  "  --sites `"$($siteList -join ',')`" ``",
  "  --seeds `"$Seeds`" ``",
  "  --include-missing ``",
  "  --stale-minutes 30 ``",
  "  --output-csv analysis\external_baselines\external_monitor_latest.csv"
)
