param(
  [string]$Seeds = "seedwebrledmain1_20260622,seedwebrledmain2_20260622,seedwebrledmain3_20260622,seedwebrledmain4_20260622,seedwebrledmain5_20260622",
  [int]$TimeLimit = 3600,
  [int]$ForceHeadful = 0,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$logDir = Join-Path $repo "analysis\external_baselines\main_table_webrled_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

$campaign = [ordered]@{
  started_at = (Get-Date).ToString("o")
  algorithm = "webrled-official"
  seeds = $Seeds
  time_limit = $TimeLimit
  force_headful = $ForceHeadful
  coverage = $false
  canonical_metrics = $true
  windows = @(
    @{ window = 1; sites = "github,odoo,nextcloud" },
    @{ window = 2; sites = "4gaboards,realworld,petclinic" },
    @{ window = 3; sites = "agilefant,gadael,timeoff,splittypie" }
  )
}
$campaign | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $logDir "current_campaign.json")

for ($window = 1; $window -le 3; $window++) {
  $log = Join-Path $logDir ("window{0}_{1}.log" -f $window, $stamp)
  $cmd = @"
Set-Location '$repo'
& '$repo\scripts\start_webrled_main_table_window.ps1' -Window $window -Seeds '$Seeds' -TimeLimit $TimeLimit -ForceHeadful $ForceHeadful$(if ($DryRun) { " -DryRun" } else { "" }) *> '$log'
"@
  Write-Host "[webrled-main-table] start window=$window log=$log"
  $p = Start-Process -FilePath powershell.exe `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $cmd) `
    -WindowStyle Hidden `
    -PassThru
  Write-Host "[webrled-main-table] pid=$($p.Id)"
}
