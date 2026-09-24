param(
  [string]$Sites = "nextcloud,gadael,petclinic,splittypie",
  [string]$Seeds = "seedrq3qexplore1_20260629,seedrq3qexplore2_20260629,seedrq3qexplore3_20260629",
  [int]$ForceHeadful = 0,
  [int]$MaxTransitions = 2200,
  [int]$UpdateEpochs = 3,
  [switch]$SkipBusy = $true,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Test-SiteBusy([string]$Site) {
  $sitePattern = [regex]::Escape($Site)
  $busy = Get-CimInstance Win32_Process |
    Where-Object {
      $_.CommandLine -match $sitePattern -and
      $_.CommandLine -match "run_experiments|main.py|start_fixed_env_site_algorithms|webqt|subweb-frontier|qlearning-1agent"
    } |
    Select-Object -First 1
  return $null -ne $busy
}

$siteList = @()
foreach ($part in ($Sites -split ",")) {
  $site = $part.Trim()
  if ($site) {
    $siteList += $site
  }
}

if ($siteList.Count -eq 0) {
  throw "No sites provided."
}

$env:WEBTEST_SUBWEB_REWARD_MODE = "qexplore"
$env:WEBTEST_SUBWEB_OPTIMIZER = "a2c"
$env:WEBTEST_SUBWEB_QEXPLORE_VALID_REWARD_SCALE = "1.0"
$env:WEBTEST_SUBWEB_QEXPLORE_INVALID_REWARD = "-1.0"

Write-Host "[rq3-qexplore-reward] sites=$($siteList -join ',') seeds=$Seeds max_transitions=$MaxTransitions reward=qexplore optimizer=a2c skip_busy=$([bool]$SkipBusy)"
Write-Host "[rq3-qexplore-reward] source-compatible valid reward: 1/action_count; invalid reward clipped-safe -1.0"

foreach ($site in $siteList) {
  if ($SkipBusy -and (Test-SiteBusy $site)) {
    Write-Host "[rq3-qexplore-reward] SKIP busy site=$site"
    continue
  }

  Write-Host "[rq3-qexplore-reward] START site=$site"
  $params = @{
    Sites = $site
    Baselines = "subweb-frontier-a2c"
    Seeds = $Seeds
    ForceHeadful = $ForceHeadful
    MaxTransitions = $MaxTransitions
    UpdateEpochs = $UpdateEpochs
  }
  if ($DryRun) {
    $params["DryRun"] = $true
  }
  & "$root\run_experiments.ps1" @params
  if ((-not $DryRun) -and $LASTEXITCODE -ne 0) {
    throw "[rq3-qexplore-reward] site=$site failed with exit=$LASTEXITCODE"
  }
  Write-Host "[rq3-qexplore-reward] END site=$site"
}
