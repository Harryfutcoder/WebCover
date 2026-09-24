param(
  [string]$WebRLEDPython = "",
  [string]$QExplorePython = "",
  [switch]$SkipDocker
)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$failures = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]

function Add-Check {
  param([string]$Name, [bool]$Passed, [string]$Detail = "")
  $prefix = if ($Passed) { "[OK]" } else { "[FAIL]" }
  if ($Detail) {
    Write-Host "$prefix $Name - $Detail"
  } else {
    Write-Host "$prefix $Name"
  }
  if (-not $Passed) {
    $script:failures.Add($Name) | Out-Null
  }
}

function Add-Warn {
  param([string]$Name, [string]$Detail = "")
  if ($Detail) {
    Write-Host "[WARN] $Name - $Detail"
  } else {
    Write-Host "[WARN] $Name"
  }
  $script:warnings.Add($Name) | Out-Null
}

function Test-File {
  param([string]$Name, [string]$Path)
  Add-Check -Name $Name -Passed (Test-Path -LiteralPath $Path) -Detail $Path
}

function Test-MetricRows {
  param(
    [string]$Name,
    [string]$Only,
    [string]$Sites,
    [string]$Seeds,
    [int]$ExpectedMinRows
  )
  $tmp = Join-Path $env:TEMP ("external_metric_verify_{0}_{1}.csv" -f $Only, [Guid]::NewGuid().ToString("N"))
  & python (Join-Path $repo "scripts\summarize_external_baseline_metrics.py") `
    --only $Only `
    --sites $Sites `
    --seeds $Seeds `
    --output-csv $tmp | Out-Host
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $tmp)) {
    Add-Check -Name $Name -Passed $false -Detail "summarizer failed"
    return
  }
  $rows = @(Import-Csv -LiteralPath $tmp)
  $okRows = @(
    $rows | Where-Object {
      $_.status -eq "ok" -and
      $_.unique_states -ne "" -and
      $_.total_actions -ne "" -and
      $_.unique_actions -ne "" -and
      $_.branch_coverage -ne "" -and
      $_.line_coverage -ne ""
    }
  )
  Add-Check -Name $Name -Passed ($okRows.Count -ge $ExpectedMinRows) -Detail ("ok_rows={0}/{1}" -f $okRows.Count, $rows.Count)
}

Write-Host "== External Baseline Setup Verifier =="
Write-Host "repo=$repo"
Write-Host ""

Write-Host "== Entrypoints =="
Test-File -Name "WebRLED wrapper" -Path (Join-Path $repo "run_webrled_official.py")
Test-File -Name "WebRLED official source main" -Path (Join-Path $repo "external\webrled_official_src\src\main.py")
Test-File -Name "WebRLED official bootstrap" -Path (Join-Path $repo "external\webrled_official_src\_bootstrap.py")
Test-File -Name "QExplore source" -Path (Join-Path $repo "external\QExplore\Qexplore\Qexplore.py")
Test-File -Name "QExplore batch wrapper" -Path (Join-Path $repo "scripts\start_qexplore_batch.ps1")
Test-File -Name "Combined all-site run plan" -Path (Join-Path $repo "scripts\show_external_baseline_run_plan.ps1")
Test-File -Name "Combined coverage plan" -Path (Join-Path $repo "scripts\show_external_code_coverage_run_plan.ps1")
Test-File -Name "All-site direct window launcher" -Path (Join-Path $repo "scripts\start_external_baseline_window.ps1")
Test-File -Name "Coverage direct window launcher" -Path (Join-Path $repo "scripts\start_external_coverage_window.ps1")
Test-File -Name "Coverage environment setup launcher" -Path (Join-Path $repo "scripts\prepare_external_coverage_environment.ps1")
Test-File -Name "External baseline monitor" -Path (Join-Path $repo "scripts\monitor_external_baseline_runs.py")
Test-File -Name "Metrics summarizer" -Path (Join-Path $repo "scripts\summarize_external_baseline_metrics.py")
Write-Host ""

Write-Host "== Runtime Preflight =="
$preflightArgs = @(
  "-ExecutionPolicy", "Bypass",
  "-File", (Join-Path $repo "scripts\check_external_baseline_prereqs.ps1"),
  "-DetailedCoverage"
)
if ($SkipDocker) { $preflightArgs += "-SkipDocker" }
if ($WebRLEDPython) { $preflightArgs += @("-WebRLEDPython", $WebRLEDPython) }
if ($QExplorePython) { $preflightArgs += @("-QExplorePython", $QExplorePython) }
& powershell @preflightArgs
Add-Check -Name "WebRLED/QExplore preflight command" -Passed ($LASTEXITCODE -eq 0) -Detail "exit=$LASTEXITCODE"
Write-Host ""

Write-Host "== Coverage Readiness =="
$coverageArgs = @(
  "-ExecutionPolicy", "Bypass",
  "-File", (Join-Path $repo "scripts\check_webrled_code_coverage_setup.ps1"),
  "-Detailed"
)
if ($SkipDocker) { $coverageArgs += "-SkipDocker" }
$coverageText = (& powershell @coverageArgs) | Out-String
Write-Host $coverageText
$strictSites = @("agilefant", "gadael", "timeoff", "splittypie", "petclinic")
foreach ($site in $strictSites) {
  $sitePattern = "site\s+:\s+$site"
  $chunk = ($coverageText -split "(\r?\n){2,}") | Where-Object { $_ -match $sitePattern } | Select-Object -First 1
  $expected = if ($site -in @("splittypie", "petclinic") -and $SkipDocker) {
    "READY_LIVE_UNVERIFIED_DOCKER|READY_LIVE"
  } elseif ($site -in @("splittypie", "petclinic")) {
    "READY_LIVE"
  } else {
    "OK"
  }
  Add-Check -Name "coverage readiness $site" -Passed ($chunk -match "status\s+:\s+($expected)") -Detail ("expected={0}" -f $expected)
}
$chunk4ga = ($coverageText -split "(\r?\n){2,}") | Where-Object { $_ -match "site\s+:\s+4gaboards" } | Select-Object -First 1
if ($SkipDocker) {
  Add-Warn -Name "4gaboards post-run coverage not fully verified" -Detail "Docker skipped"
} else {
  Add-Check -Name "coverage readiness 4gaboards post-run" -Passed ($chunk4ga -match "status\s+:\s+READY_POSTRUN") -Detail "expected=READY_POSTRUN"
}
Write-Host ""

Write-Host "== Existing Smoke Metrics =="
Test-MetricRows `
  -Name "WebRLED smoke has states/actions/coverage" `
  -Only "webrled" `
  -Sites "agilefant" `
  -Seeds "seedcodecovsmoke_manual5_20260622" `
  -ExpectedMinRows 1
Test-MetricRows `
  -Name "QExplore smoke has states/actions/coverage" `
  -Only "qexplore" `
  -Sites "agilefant,splittypie" `
  -Seeds "seedqexplorecovsmoke3_20260622,seedqexploresplitcovsmoke_20260622" `
  -ExpectedMinRows 2
Write-Host ""

Write-Host "== All-Site Multi-Window Plan Dry Run =="
$allSitePlanPath = Join-Path $env:TEMP ("external_allsite_plan_verify_{0}.txt" -f [Guid]::NewGuid().ToString("N"))
& powershell -ExecutionPolicy Bypass -File (Join-Path $repo "scripts\show_external_baseline_run_plan.ps1") -IncludeDryRun > $allSitePlanPath
$allSitePlan = Get-Content -LiteralPath $allSitePlanPath -Raw
Add-Check -Name "all-site plan includes WebRLED windows" -Passed ($allSitePlan -match "WebRLED Official Windows")
Add-Check -Name "all-site plan includes QExplore windows" -Passed ($allSitePlan -match "QExplore Windows")
Add-Check -Name "all-site plan includes monitor command" -Passed ($allSitePlan -match "monitor_external_baseline_runs.py")
Add-Check -Name "all-site plan includes dry-run flag" -Passed ($allSitePlan -match "-DryRun")
Write-Host "all_site_plan=$allSitePlanPath"
Write-Host ""

Write-Host "== Direct Launcher Dry Runs =="
& powershell -ExecutionPolicy Bypass -File (Join-Path $repo "scripts\start_external_baseline_window.ps1") `
  -Algorithm webrled `
  -Window 1 `
  -Seeds "seedverifydryrun" `
  -TimeLimit 60 `
  -DryRun | Out-Host
Add-Check -Name "direct WebRLED all-site window dry-run" -Passed ($LASTEXITCODE -eq 0) -Detail "exit=$LASTEXITCODE"
& powershell -ExecutionPolicy Bypass -File (Join-Path $repo "scripts\start_external_baseline_window.ps1") `
  -Algorithm qexplore `
  -Window 2 `
  -Seeds "seedverifydryrun" `
  -TimeLimit 60 `
  -DryRun | Out-Host
Add-Check -Name "direct QExplore all-site window dry-run" -Passed ($LASTEXITCODE -eq 0) -Detail "exit=$LASTEXITCODE"
& powershell -ExecutionPolicy Bypass -File (Join-Path $repo "scripts\prepare_external_coverage_environment.ps1") `
  -DryRun | Out-Host
Add-Check -Name "direct coverage environment setup dry-run" -Passed ($LASTEXITCODE -eq 0) -Detail "exit=$LASTEXITCODE"
foreach ($coverageWindow in @(1, 2, 3, 4)) {
  & powershell -ExecutionPolicy Bypass -File (Join-Path $repo "scripts\start_external_coverage_window.ps1") `
    -Window $coverageWindow `
    -Seeds "seedverifydryrun" `
    -TimeLimit 60 `
    -DryRun | Out-Host
  Add-Check -Name "direct coverage window $coverageWindow dry-run" -Passed ($LASTEXITCODE -eq 0) -Detail "exit=$LASTEXITCODE"
}
Write-Host ""

Write-Host "== Coverage Multi-Window Plan Dry Run =="
$planPath = Join-Path $env:TEMP ("external_cov_plan_verify_{0}.txt" -f [Guid]::NewGuid().ToString("N"))
& powershell -ExecutionPolicy Bypass -File (Join-Path $repo "scripts\show_external_code_coverage_run_plan.ps1") -IncludeDryRun > $planPath
$plan = Get-Content -LiteralPath $planPath -Raw
Add-Check -Name "plan includes WebRLED Java coverage window" -Passed ($plan -match "Window 1: WebRLED Live Java Coverage")
Add-Check -Name "plan includes QExplore Java coverage window" -Passed ($plan -match "Window 2: QExplore Live Java Coverage")
Add-Check -Name "plan includes WebRLED instrumented window" -Passed ($plan -match "Window 3: WebRLED Instrumented")
Add-Check -Name "plan includes QExplore instrumented window" -Passed ($plan -match "Window 4: QExplore Instrumented")
Add-Check -Name "plan excludes unavailable server-side coverage sites" -Passed ($plan -match "No current server-side coverage: github,odoo,realworld,nextcloud")
Add-Check -Name "plan includes monitor command" -Passed ($plan -match "monitor_external_baseline_runs.py")
Write-Host "plan=$planPath"
Write-Host ""

Write-Host "== Smoke Plan Dry Run =="
$smokePlanPath = Join-Path $env:TEMP ("external_smoke_plan_verify_{0}.txt" -f [Guid]::NewGuid().ToString("N"))
& powershell -ExecutionPolicy Bypass -File (Join-Path $repo "scripts\show_external_coverage_smoke_plan.ps1") -IncludeDryRun > $smokePlanPath
$smokePlan = Get-Content -LiteralPath $smokePlanPath -Raw
Add-Check -Name "smoke plan includes WebRLED smoke" -Passed ($smokePlan -match "Smoke A: WebRLED Official")
Add-Check -Name "smoke plan includes QExplore smoke" -Passed ($smokePlan -match "Smoke B: QExplore")
Add-Check -Name "smoke plan summarizes states/actions/coverage" -Passed ($smokePlan -match "unique_states, total_actions, unique_actions, branch_coverage, line_coverage")
Write-Host "smoke_plan=$smokePlanPath"
Write-Host ""

Write-Host "== Summary =="
Write-Host ("failures={0} warnings={1}" -f $failures.Count, $warnings.Count)
if ($failures.Count -gt 0) {
  foreach ($failure in $failures) {
    Write-Host "  FAIL: $failure"
  }
  exit 1
}
if ($warnings.Count -gt 0) {
  foreach ($warning in $warnings) {
    Write-Host "  WARN: $warning"
  }
}
exit 0
