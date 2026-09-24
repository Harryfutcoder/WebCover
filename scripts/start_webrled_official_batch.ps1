param(
  [string]$Sites = "odoo,4gaboards,github,petclinic,agilefant,splittypie,realworld,nextcloud,gadael,timeoff",
  [string]$Seeds = "seedwebrled1,seedwebrled2,seedwebrled3",
  [int]$TimeLimit = 3600,
  [int]$ForceHeadful = 0,
  [int]$GlobalBrowserCleanup = 0,
  [int]$HardTimeoutGrace = 300,
  [switch]$SkipTimeOffPreflight,
  [switch]$Coverage,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$previousAlive = $env:WEBTEST_ALIVE_TIME_OVERRIDE
$previousImpl = $env:WEBTEST_WEBRLED_IMPL
$previousCoverage = $env:WEBTEST_WEBRLED_COVERAGE
$previousCleanup = $env:WEBTEST_GLOBAL_BROWSER_CLEANUP
$previousHardTimeoutGrace = $env:WEBTEST_WEBRLED_HARD_TIMEOUT_GRACE

try {
  $env:WEBTEST_ALIVE_TIME_OVERRIDE = [string]$TimeLimit
  $env:WEBTEST_WEBRLED_IMPL = "official"
  $env:WEBTEST_GLOBAL_BROWSER_CLEANUP = [string]$GlobalBrowserCleanup
  $env:WEBTEST_WEBRLED_HARD_TIMEOUT_GRACE = [string]$HardTimeoutGrace
  if ($Coverage) {
    $env:WEBTEST_WEBRLED_COVERAGE = "1"
  } else {
    $env:WEBTEST_WEBRLED_COVERAGE = "0"
  }

  Write-Host "[webrled-batch] sites=$Sites"
  Write-Host "[webrled-batch] seeds=$Seeds"
  Write-Host "[webrled-batch] time_limit=$TimeLimit cleanup=$GlobalBrowserCleanup coverage=$([bool]$Coverage) hard_timeout_grace=$HardTimeoutGrace"

  $siteList = @($Sites -split "," | ForEach-Object { $_.Trim().ToLowerInvariant() } | Where-Object { $_ })
  if (($siteList -contains "timeoff") -and (-not $SkipTimeOffPreflight)) {
    Write-Host "[webrled-batch] timeoff preflight: ensure upload directory and app health"
    if ($DryRun) {
      Write-Host "[webrled-batch] dry-run: skip timeoff docker preflight"
    } else {
      $timeoffPreflight = 'cd /usr/local/timeoff && mkdir -p coverage_data && chmod 777 coverage_data && (pgrep -f "node bin/wwww" >/dev/null || (nohup npm start > /tmp/timeoff_app_recover.log 2>&1 & echo $! > /tmp/timeoff_manual.pid))'
      & docker exec timeoff-container sh -lc $timeoffPreflight

      $timeoffHealthy = $false
      for ($i = 0; $i -lt 10; $i++) {
        try {
          $resp = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "http://localhost:3002/login"
          if ($resp.StatusCode -eq 200) {
            $timeoffHealthy = $true
            break
          }
        } catch {
          Start-Sleep -Seconds 2
        }
      }
      if (-not $timeoffHealthy) {
        throw "TimeOff preflight failed: http://localhost:3002/login did not return 200"
      }
    }
  }
  if (($siteList -contains "timeoff") -and $SkipTimeOffPreflight) {
    Write-Host "[webrled-batch] timeoff preflight skipped by request"
  }

  & (Join-Path $repo "run_experiments.ps1") `
    -Sites $Sites `
    -Baselines "webrled-official" `
    -Seeds $Seeds `
    -ForceHeadful $ForceHeadful `
    -DryRun:$DryRun

  if (-not $DryRun) {
    & python (Join-Path $repo "scripts\summarize_external_baseline_metrics.py") `
      --only webrled `
      --sites $Sites `
      --seeds $Seeds `
      --output-csv (Join-Path $repo "analysis\external_baselines\webrled_official_latest.csv") `
      --print-table
  }
} finally {
  $env:WEBTEST_ALIVE_TIME_OVERRIDE = $previousAlive
  $env:WEBTEST_WEBRLED_IMPL = $previousImpl
  $env:WEBTEST_WEBRLED_COVERAGE = $previousCoverage
  $env:WEBTEST_GLOBAL_BROWSER_CLEANUP = $previousCleanup
  $env:WEBTEST_WEBRLED_HARD_TIMEOUT_GRACE = $previousHardTimeoutGrace
}
