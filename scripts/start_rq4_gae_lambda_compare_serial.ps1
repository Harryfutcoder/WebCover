param(
  [string]$Sites = "nextcloud,gadael,splittypie,petclinic",
  [string]$Lambdas = "0.32,0.64",
  [string]$Seeds = "seedrq4gae1_20260630,seedrq4gae2_20260630,seedrq4gae3_20260630",
  [int]$ForceHeadful = 0,
  [int]$MaxTransitions = 2200,
  [int]$UpdateEpochs = 3,
  [int]$RolloutLen = 32,
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
      $_.CommandLine -match "run_experiments|main.py|start_fixed_env_site_algorithms|webrled|webqt|subweb-frontier|qlearning-1agent"
    } |
    Select-Object -First 1
  return $null -ne $busy
}

$siteList = @($Sites -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$lambdaList = @($Lambdas -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })

if ($siteList.Count -eq 0) {
  throw "No sites provided."
}
if ($lambdaList.Count -eq 0) {
  throw "No GAE lambdas provided."
}

Write-Host "[rq4-gae-compare] sites=$($siteList -join ',') lambdas=$($lambdaList -join ',') seeds=$Seeds max_transitions=$MaxTransitions rollout_len=$RolloutLen skip_busy=$([bool]$SkipBusy)"
Write-Host "[rq4-gae-compare] config=WebCover A2C except rollout/lambda vary as requested; default 0.98 main-run data remains the reference."

foreach ($site in $siteList) {
  if ($SkipBusy -and (Test-SiteBusy $site)) {
    Write-Host "[rq4-gae-compare] SKIP busy site=$site"
    continue
  }

  foreach ($lambda in $lambdaList) {
    Write-Host "[rq4-gae-compare] START site=$site gae_lambda=$lambda"
    $params = @{
      Sites = $site
      Baselines = "subweb-frontier-a2c"
      Seeds = $Seeds
      ForceHeadful = $ForceHeadful
      MaxTransitions = $MaxTransitions
      UpdateEpochs = $UpdateEpochs
      RolloutLen = $RolloutLen
      GaeLambda = [double]$lambda
    }
    if ($DryRun) {
      $params["DryRun"] = $true
    }
    & "$root\run_experiments.ps1" @params
    if ((-not $DryRun) -and $LASTEXITCODE -ne 0) {
      throw "[rq4-gae-compare] site=$site gae_lambda=$lambda failed with exit=$LASTEXITCODE"
    }
    Write-Host "[rq4-gae-compare] END site=$site gae_lambda=$lambda"
  }
}
