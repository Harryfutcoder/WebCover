<#
.SYNOPSIS
  Fast preflight for single-agent baselines before long experiment runs.

.DESCRIPTION
  Runs short smoke sessions for:
    webexplor, webqt, webrled-official, subweb-frontier-a2c, random

  It validates each run with:
    - process exit code == 0
    - "Total steps" extracted from summary > 0
    - at least one "Chosen action" in runtime log

  Example:
    powershell -ExecutionPolicy Bypass -File .\run_preflight_algorithms.ps1 -AliveTime 180 -ForceHeadful 0
#>

[CmdletBinding()]
param(
    [ValidateSet("github", "petclinic", "splittypie", "realworld", "odoo", "discourse", "nextcloud", "4gaboards", "agilefant", "gadael", "timeoff")]
    [string]$Site = "github",
    [int]$AliveTime = 180,
    [int]$PageLoadTimeout = 45,
    [int]$RecordInterval = 30,
    [string]$Seed = "seed1",
    [ValidateSet("0","1")]
    [string]$ForceSystemChrome = "0",
    [ValidateSet("0","1")]
    [string]$ForceHeadful = "0"
)

$ErrorActionPreference = "Continue"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

Set-Location -LiteralPath $PSScriptRoot

function Use-OfficialWebRLED {
    return $true
}

$webrledOfficialPython = $env:WEBTEST_WEBRLED_PYTHON
if ([string]::IsNullOrWhiteSpace($webrledOfficialPython)) {
    $webrledOfficialPython = "python"
}

$vendorPath = Join-Path $PSScriptRoot "_vendor"
if (Test-Path $vendorPath) {
    if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
        $env:PYTHONPATH = $vendorPath
    } else {
        $env:PYTHONPATH = "$vendorPath;$($env:PYTHONPATH)"
    }
}
$rewardClipAbsOverride = $env:WEBTEST_REWARD_CLIP_ABS

function Get-LocalSitePort([string]$SiteName) {
    switch ($SiteName) {
        "petclinic" { return 8081 }
        "splittypie" { return 4200 }
        "odoo" { return 8069 }
        "discourse" { return 4203 }
        "nextcloud" { return 8082 }
        "4gaboards" { return 3000 }
        "agilefant" { return 8084 }
        "gadael" { return 3001 }
        "timeoff" { return 3002 }
        default { return $null }
    }
}

$requiredPort = Get-LocalSitePort $Site
if ($null -ne $requiredPort) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $iar = $client.BeginConnect("127.0.0.1", $requiredPort, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(1500, $false)
        if (-not ($ok -and $client.Connected)) {
            throw "$Site preflight requires localhost:$requiredPort"
        }
        $client.EndConnect($iar) | Out-Null
    } catch {
        throw "$Site preflight requires localhost:$requiredPort but it is not reachable."
    } finally {
        $client.Close()
    }
}

$runs = @(
    @{
        Name = "webexplor"
        Profile = "$Site-webexplor-1agent"
        Env = @{}
    },
    @{
        Name = "webqt"
        Profile = "$Site-qlearning-1agent"
        Env = @{
            WEBTEST_QLEARNING_AGENT_TYPE = "W"
            WEBTEST_QLEARNING_STATE_MODE = "actionset"
            WEBTEST_QLEARNING_REWARD_MODE = "webqt"
            WEBQT_W_LOC = "10"
            WEBQT_W_ATTENTION = "50"
            WEBQT_W_FREQ = "5"
            WEBQT_W_EXPLORE = "5"
            WEBQT_STATE_SIM_THRESHOLD = "0.85"
            WEBQT_STAGNATION_WINDOW = "50"
            WEBQT_DECAY_EPSILON = "1"
            WEBQT_MIN_EPSILON = "0.1"
            WEBTEST_REWARD_CLIP_ABS = "0"
        }
    },
    @{
        Name = "webrled-official"
        Profile = "$Site-drl-1agent-observation"
        Env = @{}
    },
    @{
        Name = "subweb-frontier-a2c"
        Profile = "$Site-subweb-frontier-a2c-1agent"
        Env = @{
            WEBTEST_FRONTIER_COVERAGE = "1"
            WEBTEST_PAD_ACTIONS = "1"
        }
    },
    @{
        Name = "random"
        Profile = "$Site-random-1agent"
        Env = @{}
    }
)

$rootLogDir = Join-Path $PSScriptRoot "preflight_logs"
if (-not (Test-Path $rootLogDir)) {
    New-Item -ItemType Directory -Path $rootLogDir | Out-Null
}

$results = New-Object System.Collections.Generic.List[object]
$index = 0
$total = $runs.Count

foreach ($run in $runs) {
    $index++
    $name = $run.Name
    $profile = $run.Profile
    $session = "preflight-$name-$Seed"
    $log = Join-Path $rootLogDir ("{0}_{1}_{2}.log" -f $Site, $name, (Get-Date -Format "yyyyMMdd_HHmmss"))

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[$index / $total] preflight=$name profile=$profile session=$session"
    Write-Host "  Log: $log"
    Write-Host "============================================================"

    # Common env
    $env:OBSERVATION_MG_FULL_HISTORY = "1"
    $env:WEBTEST_FAIR_MODE = "1"
    $env:WEBTEST_EXPERIMENT_TYPE = "fair-benchmark"
    $env:WEBTEST_RUN_SEED = ($Seed -replace '[^0-9]', '')
    if ([string]::IsNullOrWhiteSpace($env:WEBTEST_RUN_SEED)) { $env:WEBTEST_RUN_SEED = "1" }
    $env:WEBTEST_FORCE_SYSTEM_CHROME = $ForceSystemChrome
    $env:WEBTEST_FORCE_HEADFUL = $ForceHeadful
    $env:WEBTEST_ENABLE_SCREENSHOT = "0"
    $env:WEBTEST_ENABLE_TRANSLATE_API = "0"
    $env:WEBTEST_RELAX_HTTPS = "0"
    $env:WEBTEST_BLOCK_AUTH_ACTIONS = "1"
    $env:WEBTEST_AUTH_PATH_PREFIXES = "/login,/signup,/sign-in,/signin,/register,/password_reset,/session,/sessions,/oauth,/auth"
    $env:WEBTEST_MAX_EMPTY_ACTION_ROUNDS = "40"
    $env:WEBTEST_MAX_BROWSER_ERROR_ROUNDS = "15"
    $env:WEBTEST_MAX_HTTP_5XX_ROUNDS = "12"
    $env:WEBTEST_FILTER_KNOWN_NOISE = "1"
    $env:WEBTEST_ACTION_DETECT_RETRY = "2"
    $env:WEBTEST_ACTION_DETECT_RETRY_WAIT = "0.8"
    $env:WEBTEST_URL_NORMALIZE_MODE = "path_only"
    $env:WEBTEST_RESTART_POLICY = "entry"
    if ([string]::IsNullOrWhiteSpace($env:WEBTEST_MAX_TRANSITIONS)) {
        $env:WEBTEST_MAX_TRANSITIONS = "120"
    }
    if ([string]::IsNullOrWhiteSpace($env:WEBTEST_REPLAY_BUFFER_CAPACITY)) {
        $env:WEBTEST_REPLAY_BUFFER_CAPACITY = "10000"
    }
    if ([string]::IsNullOrWhiteSpace($rewardClipAbsOverride)) {
        $env:WEBTEST_REWARD_CLIP_ABS = "5.0"
    } else {
        $env:WEBTEST_REWARD_CLIP_ABS = $rewardClipAbsOverride
    }
    $env:WEBTEST_ALIVE_TIME_OVERRIDE = [string]$AliveTime
    $env:WEBTEST_PAGE_LOAD_TIMEOUT_OVERRIDE = [string]$PageLoadTimeout
    $env:WEBTEST_RECORD_INTERVAL_OVERRIDE = [string]$RecordInterval

    # Clear previous algorithm-specific env
    Remove-Item Env:WEBTEST_DRL_REWARD_FUNCTION -ErrorAction SilentlyContinue
    Remove-Item Env:WEBTEST_WEBEXPLOR_REWARD_MODE -ErrorAction SilentlyContinue
    Remove-Item Env:WEBTEST_QLEARNING_REWARD_MODE -ErrorAction SilentlyContinue
    Remove-Item Env:WEBTEST_QLEARNING_AGENT_TYPE -ErrorAction SilentlyContinue
    Remove-Item Env:WEBTEST_QLEARNING_STATE_MODE -ErrorAction SilentlyContinue
    Remove-Item Env:WEBTEST_QLEARNING_POLICY_MODE -ErrorAction SilentlyContinue
    Remove-Item Env:WEBTEST_QLEARNING_GUMBEL_TAU -ErrorAction SilentlyContinue
    Remove-Item Env:WEBQT_W_LOC -ErrorAction SilentlyContinue
    Remove-Item Env:WEBQT_W_ATTENTION -ErrorAction SilentlyContinue
    Remove-Item Env:WEBQT_W_FREQ -ErrorAction SilentlyContinue
    Remove-Item Env:WEBQT_W_EXPLORE -ErrorAction SilentlyContinue
    Remove-Item Env:WEBQT_STATE_SIM_THRESHOLD -ErrorAction SilentlyContinue
    Remove-Item Env:WEBQT_STAGNATION_WINDOW -ErrorAction SilentlyContinue
    Remove-Item Env:WEBQT_DECAY_EPSILON -ErrorAction SilentlyContinue
    Remove-Item Env:WEBQT_MIN_EPSILON -ErrorAction SilentlyContinue
    Remove-Item Env:WEBTEST_FRONTIER_COVERAGE -ErrorAction SilentlyContinue
    Remove-Item Env:WEBTEST_PAD_ACTIONS -ErrorAction SilentlyContinue
    Remove-Item Env:WEBTEST_AGENT_MODULE -ErrorAction SilentlyContinue
    Remove-Item Env:WEBTEST_AGENT_CLASS -ErrorAction SilentlyContinue

    $useOfficialWebRLED = ($name -eq "webrled-official" -and (Use-OfficialWebRLED))
    if (-not $useOfficialWebRLED) {
        foreach ($k in $run.Env.Keys) {
            Set-Item -Path "Env:$k" -Value ([string]$run.Env[$k])
        }
    }

    if ($useOfficialWebRLED) {
        & $webrledOfficialPython run_webrled_official.py --site $Site --session $session --seed $Seed --time-limit $AliveTime 2>&1 | Tee-Object -FilePath $log
    } else {
        python main.py --profile $profile --session $session 2>&1 | Tee-Object -FilePath $log
    }
    $exitCode = $LASTEXITCODE

    if ($useOfficialWebRLED) {
        $chosenCount = (Select-String -Path $log -Pattern "Clicked:|Typed \\[|Select:" | Measure-Object).Count
    } else {
        $chosenCount = (Select-String -Path $log -Pattern "Chosen action:" -SimpleMatch | Measure-Object).Count
    }
    $stepsLine = Select-String -Path $log -Pattern "Total steps:" -SimpleMatch | Select-Object -Last 1
    $totalSteps = 0
    if ($useOfficialWebRLED) {
        $stepLines = Select-String -Path $log -Pattern "Step:\s+(\d+)"
        if ($stepLines) {
            $lastMatch = $stepLines | Select-Object -Last 1
            $m = [regex]::Match($lastMatch.Line, "Step:\s+(\d+)")
            if ($m.Success) { $totalSteps = [int]$m.Groups[1].Value }
        }
    } else {
        if ($stepsLine) {
            $m = [regex]::Match($stepsLine.Line, "Total steps:\s+(\d+)")
            if ($m.Success) { $totalSteps = [int]$m.Groups[1].Value }
        }
    }

    $pass = ($exitCode -eq 0) -and ($chosenCount -gt 0) -and ($totalSteps -gt 0)
    $results.Add([PSCustomObject]@{
        baseline = $name
        profile = $profile
        exit_code = $exitCode
        chosen_actions = $chosenCount
        total_steps = $totalSteps
        pass = $pass
        log = $log
    }) | Out-Null
}

Write-Host ""
Write-Host "================ PRE-FLIGHT SUMMARY ================"
$results | Format-Table -AutoSize

$failed = @($results | Where-Object { -not $_.pass })
if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "Preflight failed for $($failed.Count) baselines." -ForegroundColor Red
    foreach ($f in $failed) {
        Write-Host "  - $($f.baseline): exit=$($f.exit_code), chosen=$($f.chosen_actions), steps=$($f.total_steps), log=$($f.log)"
    }
    exit 1
}

Write-Host ""
Write-Host "All preflight baselines passed."
exit 0
