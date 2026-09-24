param(
  [ValidateRange(1, 3)]
  [int]$Window = 1,
  [string]$Seeds = "seedwebrledmain1_20260622,seedwebrledmain2_20260622,seedwebrledmain3_20260622,seedwebrledmain4_20260622,seedwebrledmain5_20260622",
  [int]$TimeLimit = 3600,
  [int]$ForceHeadful = 0,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$windowSites = @{
  1 = "github,odoo,nextcloud"
  2 = "4gaboards,realworld,petclinic"
  3 = "agilefant,gadael,timeoff,splittypie"
}

$sites = $windowSites[$Window]
if ([string]::IsNullOrWhiteSpace($sites)) {
  throw "No WebRLED main-table sites configured for window $Window."
}

Write-Host "[webrled-main-table] window=$Window"
Write-Host "[webrled-main-table] sites=$sites"
Write-Host "[webrled-main-table] seeds=$Seeds"
Write-Host "[webrled-main-table] time_limit=$TimeLimit force_headful=$ForceHeadful"
Write-Host "[webrled-main-table] coverage=disabled canonical_metrics=enabled"

& (Join-Path $repo "scripts\start_webrled_official_batch.ps1") `
  -Sites $sites `
  -Seeds $Seeds `
  -TimeLimit $TimeLimit `
  -ForceHeadful $ForceHeadful `
  -GlobalBrowserCleanup 0 `
  -DryRun:$DryRun
