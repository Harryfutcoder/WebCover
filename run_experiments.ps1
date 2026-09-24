<#
.SYNOPSIS
    Parameterized experiment runner for cross-site paper evaluation.

.DESCRIPTION
    Example:
      powershell -ExecutionPolicy Bypass -File run_experiments.ps1 `
        -Sites github,toppr,vuestic `
        -Baselines webexplor,webqt,webrled-official,subweb-frontier-a2c `
        -Seeds seed1,seed2,seed3

    Supported baselines:
      subweb-frontier-a2c, webexplor, webqt, webrled-official

    Use -DryRun to print selected runs without executing them.
#>

[CmdletBinding()]
param(
    [string[]]$Sites = @("github", "vuestic", "smadex", "eatingwell", "petclinic"),
    [string[]]$Baselines = @("webexplor", "webqt", "webrled-official", "subweb-frontier-a2c"),
[string[]]$Seeds = @("seed1", "seed2", "seed3", "seed4", "seed5"),
[switch]$IncludeRandom,
[ValidateSet("0","1")]
[string]$ForceSystemChrome = "0",
[ValidateSet("0","1")]
[string]$ForceHeadful = "0",
[ValidateRange(1,20)]
[int]$UpdateEpochs = 3,
[switch]$DryRun
)

$ErrorActionPreference = "Continue"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

# Always run relative to this script so commands like "python main.py" are stable
# regardless of where PowerShell was launched from.
Set-Location -LiteralPath $PSScriptRoot

$subwebMaxActionsOverride = $env:WEBTEST_MAX_ACTIONS
$subwebRolloutLenOverride = $env:WEBTEST_ROLLOUT_LEN
$subwebGammaOverride = $env:WEBTEST_A2C_GAMMA
$subwebAdvantageEstimatorOverride = $env:WEBTEST_A2C_ADVANTAGE_ESTIMATOR
$subwebGaeLambdaOverride = $env:WEBTEST_A2C_GAE_LAMBDA
$subwebUpdateEpochsOverride = $env:WEBTEST_A2C_UPDATE_EPOCHS
$subwebEntropyCoefOverride = $env:WEBTEST_A2C_ENTROPY_COEF
$subwebValueCoefOverride = $env:WEBTEST_A2C_VALUE_COEF
$subwebMaxGradNormOverride = $env:WEBTEST_A2C_MAX_GRAD_NORM
$subwebSeparateGradClipOverride = $env:WEBTEST_A2C_SEPARATE_GRAD_CLIP
$subwebLrOverride = $env:WEBTEST_A2C_LR
$subwebGraphEdgeWeightModeOverride = $env:WEBTEST_GRAPH_EDGE_WEIGHT_MODE
$subwebPolicyInputModeOverride = $env:WEBTEST_SUBWEB_POLICY_INPUT_MODE
$subwebGraphResidualAuxOverride = $env:WEBTEST_GRAPH_RESIDUAL_AUX
$subwebGraphResidualAuxCoefOverride = $env:WEBTEST_GRAPH_RESIDUAL_AUX_COEF
$subwebGraphResidualAuxTargetOverride = $env:WEBTEST_GRAPH_RESIDUAL_AUX_TARGET
$subwebGraphResidualAuxTargetPowerOverride = $env:WEBTEST_GRAPH_RESIDUAL_AUX_TARGET_POWER
$subwebGraphResidualAuxDistributionOverride = $env:WEBTEST_GRAPH_RESIDUAL_AUX_DISTRIBUTION
$subwebGraphResidualAuxTieBreakCoefOverride = $env:WEBTEST_GRAPH_RESIDUAL_AUX_TIE_BREAK_COEF
$webrledOfficialPython = $env:WEBTEST_WEBRLED_PYTHON
$webrledOfficialUsePyLauncher = $false
$webrledOfficialPyVersion = $env:WEBTEST_WEBRLED_PY_LAUNCHER_VERSION
if ([string]::IsNullOrWhiteSpace($webrledOfficialPyVersion)) {
    $webrledOfficialPyVersion = "3.12"
}
$webrledOfficialSitePackages = $env:WEBTEST_WEBRLED_SITE_PACKAGES
if ([string]::IsNullOrWhiteSpace($webrledOfficialPython)) {
    $webtestPython = Join-Path $env:USERPROFILE ".conda\envs\webtest-python\python.exe"
    if (Test-Path $webtestPython) {
        # This env has the WebRLED-compatible Python deps installed. Extra
        # site-packages are opt-in via WEBTEST_WEBRLED_SITE_PACKAGES to avoid
        # shadowing conda packages with stale local virtualenv metadata.
        $webrledOfficialPython = $webtestPython
    }
    if ([string]::IsNullOrWhiteSpace($webrledOfficialPython)) {
        $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
        if (Test-Path $venvPython) {
            try {
                & $venvPython -c "import sys" *> $null
                if ($LASTEXITCODE -eq 0) {
                    $webrledOfficialPython = $venvPython
                }
            } catch {}
        }
    }
    if ([string]::IsNullOrWhiteSpace($webrledOfficialPython)) {
        try {
            & py "-$webrledOfficialPyVersion" -c "import sys" *> $null
            if ($LASTEXITCODE -eq 0) {
                $webrledOfficialUsePyLauncher = $true
            }
        } catch {}
    }
    if ((-not $webrledOfficialUsePyLauncher) -and [string]::IsNullOrWhiteSpace($webrledOfficialPython)) {
        $webrledOfficialPython = "python"
    }
}

$mainPython = $env:WEBTEST_MAIN_PYTHON
if ([string]::IsNullOrWhiteSpace($mainPython)) {
    $webtestPython = Join-Path $env:USERPROFILE ".conda\envs\webtest-python\python.exe"
    if (Test-Path $webtestPython) {
        $mainPython = $webtestPython
    } else {
        $mainPython = "python"
    }
}

$vendorPath = Join-Path $PSScriptRoot "_vendor"
if (Test-Path $vendorPath) {
    if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
        $env:PYTHONPATH = $vendorPath
    } else {
        $env:PYTHONPATH = "$vendorPath;$($env:PYTHONPATH)"
    }
    Write-Host "Using local vendor deps: $vendorPath"
}
$rewardClipAbsOverride = $env:WEBTEST_REWARD_CLIP_ABS
$profileSuffix = @{
    "webexplor"      = "webexplor-1agent"
    "webqt"          = "qlearning-1agent"
    "webrled-official" = "drl-1agent-observation"
    "subweb-frontier-a2c" = "subweb-frontier-a2c-1agent"
    "random"         = "random-1agent"
}

$fairBaselines = @(
    "webexplor", "webqt", "webrled-official", "subweb-frontier-a2c",
    "random"
)

function Normalize-List([string[]]$items) {
    $result = New-Object System.Collections.Generic.List[string]
    foreach ($item in $items) {
        foreach ($part in ($item -split ",")) {
            $v = $part
            if ($null -eq $v) { continue }
            $v = $v.Trim()
            # Be tolerant to copy/paste artifacts from markdown or multiline PS snippets.
            # Examples:
            #   `github
            #   "github"
            #   'github'
            $v = $v -replace '^[`"''\s]+|[`"''\s]+$', ''
            $v = $v.ToLower().Trim()
            if ($v -ne "") {
                $result.Add($v)
            }
        }
    }
    return @($result)
}

$normalizedSites = Normalize-List $Sites
$normalizedBaselines = Normalize-List $Baselines
$normalizedSeeds = Normalize-List $Seeds

function Use-OfficialWebRLED([string]$BaselineName) {
    return ($BaselineName -eq "webrled-official")
}

function Test-LocalPort([int]$Port, [int]$TimeoutMs = 1500) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $iar = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
        if ($ok -and $client.Connected) {
            $client.EndConnect($iar) | Out-Null
            return $true
        }
        return $false
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Test-GadaelCommonBaseUrl {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:3001/rest/common" -TimeoutSec 8
        return ($response.Content -like '*"baseUrl":"http://localhost:3001/"*')
    } catch {
        return $false
    }
}

function Test-TimeoffLoginPage {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:3002/login" -TimeoutSec 8
        return (($response.Content -like '*TimeOff.Management*') -and ($response.Content -like '*email_inp*'))
    } catch {
        return $false
    }
}

function Ensure-TimeoffReady {
    if (Test-TimeoffLoginPage) {
        return $true
    }

    $dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $dockerCmd) {
        Write-Warning "timeoff login page is not healthy and Docker is unavailable for auto-repair."
        return $false
    }

    $containerNames = @("timeoff-container", "timeoff-flask-container")
    foreach ($containerName in $containerNames) {
        try {
            docker exec -d --workdir /usr/local/timeoff $containerName npm start | Out-Null
            Start-Sleep -Seconds 8
            if (Test-TimeoffLoginPage) {
                Write-Host "  timeoff repaired: npm start is serving localhost:3002/login"
                return $true
            }
        } catch {
            continue
        }
    }

    Write-Warning "timeoff still did not return a valid login page after repair."
    return $false
}

function Ensure-GadaelBootstrapState {
    if ($env:WEBTEST_SKIP_GADAEL_DOCKER_BOOTSTRAP -eq "1") {
        Write-Host "  gadael Docker bootstrap-state check skipped by WEBTEST_SKIP_GADAEL_DOCKER_BOOTSTRAP=1"
        return $true
    }

    $dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $dockerCmd) {
        return $true
    }

    try {
        $status = docker exec gadael-container mongo localhost/gadael --quiet --eval "var activeAdmin = db.users.find({isActive:true}).toArray().filter(function(u){ return u.roles && u.roles.admin; }).length; var users = db.users.count(); print(activeAdmin + ',' + users);"
        $lastLine = ($status | Where-Object { $_ -match '^\d+,\d+$' } | Select-Object -Last 1)
        if ([string]::IsNullOrWhiteSpace($lastLine)) {
            return $true
        }
        $parts = $lastLine.Split(',')
        $activeAdminCount = [int]$parts[0]
        $userCount = [int]$parts[1]
        if ($activeAdminCount -gt 0 -or $userCount -eq 0) {
            return $true
        }

        Write-Warning "gadael has users but no active admin; clearing stale users/sessions so the shared bootstrap can create a valid admin."
        docker exec gadael-container mongo localhost/gadael --quiet --eval "db.users.remove({}); db.admins.remove({}); db.sessions.remove({}); db.loginattempts.remove({});" | Out-Null
        return $true
    } catch {
        Write-Warning "gadael bootstrap-state check failed: $($_.Exception.Message)"
        return $true
    }
}

function Ensure-GadaelReady {
    if (Test-GadaelCommonBaseUrl) {
        Ensure-GadaelBootstrapState | Out-Null
        return $true
    }

    $dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $dockerCmd) {
        Write-Warning "gadael is not returning baseUrl=http://localhost:3001/ and Docker is unavailable for auto-repair."
        return $false
    }

    Write-Warning "gadael baseUrl/service is not ready; repairing container config.url and starting node app."
    try {
        docker exec gadael-container perl -0pi -e 's#http://localhost:3000/#http://localhost:3001/#g' /usr/local/gadael/config.js | Out-Null
        docker exec gadael-container sh -lc "pkill -f 'node app.js 3000 gadael' || true" | Out-Null
        docker exec -d gadael-container sh -lc "cd /usr/local/gadael && node app.js 3000 gadael > /var/log/gadael_app.log 2>&1" | Out-Null
        Start-Sleep -Seconds 8
    } catch {
        Write-Warning "gadael auto-repair failed: $($_.Exception.Message)"
        return $false
    }

    if (Test-GadaelCommonBaseUrl) {
        Write-Host "  gadael repaired: rest/common baseUrl=http://localhost:3001/"
        Ensure-GadaelBootstrapState | Out-Null
        return $true
    }

    Write-Warning "gadael still did not return baseUrl=http://localhost:3001/ after repair."
    return $false
}

function Ensure-SplittypiePort4200 {
    if (Test-LocalPort -Port 4200) {
        return $true
    }
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $docker) {
        Write-Warning "Docker not found, cannot auto-start splittypie."
        return $false
    }

    try {
        docker start splittypie | Out-Null
        Start-Sleep -Seconds 3
        if (Test-LocalPort -Port 4200) {
            Write-Host "  splittypie container started on localhost:4200"
            return $true
        }
    } catch {
    }

    try {
        $running = docker ps --filter "name=^splittypie$" --format "{{.Names}}" 2>$null
        if (-not [string]::IsNullOrWhiteSpace($running)) {
            return (Test-LocalPort -Port 4200)
        }

        $existing = docker ps -a --filter "name=^splittypie$" --format "{{.Names}}" 2>$null
        if (-not [string]::IsNullOrWhiteSpace($existing)) {
            docker rm splittypie | Out-Null
        }

        $imageId = docker image inspect pako88/splittypie --format "{{.Id}}" 2>$null
        if ([string]::IsNullOrWhiteSpace($imageId)) {
            Write-Host "  Pulling splittypie image: pako88/splittypie"
            docker pull pako88/splittypie | Out-Null
        }

        docker run -d --name splittypie -p 4200:49153 -p 4201:5100 pako88/splittypie | Out-Null
        Start-Sleep -Seconds 5
        if (Test-LocalPort -Port 4200) {
            Write-Host "  splittypie container created on localhost:4200"
            return $true
        }
    } catch {
        Write-Warning "Auto-start splittypie via Docker failed: $($_.Exception.Message)"
    }
    return $false
}

function Get-LocalSitePort([string]$SiteName) {
    # Resolve from explicit env overrides first.
    $envKey = $SiteName.ToUpper().Replace("-", "_")
    $entryOverride = [Environment]::GetEnvironmentVariable("WEBTEST_SITE_${envKey}_ENTRY_URL", "Process")
    if (-not [string]::IsNullOrWhiteSpace($entryOverride)) {
        try {
            $u = [Uri]$entryOverride
            if ($u.Host -eq "localhost" -or $u.Host -eq "127.0.0.1") {
                return [int]$u.Port
            }
            return $null
        } catch {
            return $null
        }
    }

    switch ($SiteName) {
        "petclinic" { return 8081 }
        "splittypie" { return 4200 }

        "odoo"      { return 8069 }
        "discourse" { return 4203 }
        "nextcloud" { return 8082 }
        "4gaboards" { return 3000 }
        "agilefant" { return 8084 }
        "gadael"    { return 3001 }
        "timeoff"   { return 3002 }
        default { return $null }
    }
}

function Set-SiteTargetEnv([string]$SiteName) {
    $targets = @{
        "github"     = @("https://github.com", "github.com")
        "odoo"       = @("http://localhost:8069", "localhost:8069")
        "4gaboards"  = @("http://localhost:3000", "localhost:3000")
        "agilefant"  = @("http://localhost:8084/agilefant", "localhost:8084")
        "splittypie" = @("http://localhost:4200", "localhost:4200")
        "petclinic"  = @("http://localhost:8081", "localhost:8081")
        "gadael"     = @("http://localhost:3001", "localhost:3001")
        "timeoff"    = @("http://localhost:3002", "localhost:3002")
        "nextcloud"  = @("http://localhost:8082/apps/files/", "localhost:8082")
        "realworld"  = @("https://demo.realworld.show", "demo.realworld.show")
    }
    if (-not $targets.ContainsKey($SiteName)) {
        return
    }

    $envKey = $SiteName.ToUpper().Replace("-", "_")
    $entryName = "WEBTEST_SITE_${envKey}_ENTRY_URL"
    $domainsName = "WEBTEST_SITE_${envKey}_DOMAINS"
    $entryOverride = [Environment]::GetEnvironmentVariable($entryName, "Process")
    $domainsOverride = [Environment]::GetEnvironmentVariable($domainsName, "Process")
    if ([string]::IsNullOrWhiteSpace($entryOverride)) {
        [Environment]::SetEnvironmentVariable($entryName, $targets[$SiteName][0], "Process")
    }
    if ([string]::IsNullOrWhiteSpace($domainsOverride)) {
        [Environment]::SetEnvironmentVariable($domainsName, $targets[$SiteName][1], "Process")
    }
}

# Preflight for local websites to avoid wasting long runs on obvious setup issues.
# Do not hard-fail the whole batch: mark unavailable local sites and skip their runs.
$unavailableSites = New-Object System.Collections.Generic.HashSet[string]
if (-not $DryRun) {
    if ($normalizedSites -contains "splittypie") {
        if (-not (Ensure-SplittypiePort4200)) {
            Write-Warning "splittypie requires http://localhost:4200 but it is unavailable. splittypie runs will be skipped."
            [void]$unavailableSites.Add("splittypie")
        }
    }
    if ($normalizedSites -contains "petclinic") {
        if (-not (Test-LocalPort -Port 8081)) {
            Write-Warning "petclinic requires http://localhost:8081 but port 8081 is closed. petclinic runs will be skipped."
            [void]$unavailableSites.Add("petclinic")
        }
    }
    if ($normalizedSites -contains "gadael") {
        if (-not (Ensure-GadaelReady)) {
            Write-Warning "gadael requires a healthy http://localhost:3001/rest/common response. gadael runs will be skipped."
            [void]$unavailableSites.Add("gadael")
        }
    }
    if ($normalizedSites -contains "timeoff") {
        if (-not (Ensure-TimeoffReady)) {
            Write-Warning "timeoff requires a healthy http://localhost:3002/login response. timeoff runs will be skipped."
            [void]$unavailableSites.Add("timeoff")
        }
    }
    foreach ($site in $normalizedSites) {
        if ($unavailableSites.Contains($site)) {
            continue
        }
        $requiredPort = Get-LocalSitePort $site
        if ($null -eq $requiredPort) {
            continue
        }
        if (-not (Test-LocalPort -Port $requiredPort)) {
            Write-Warning "$site requires localhost:$requiredPort but this port is closed. $site runs will be skipped."
            [void]$unavailableSites.Add($site)
        }
    }
} else {
    Write-Host "DryRun: skip local site health checks and Docker auto-repair."
}

if ($IncludeRandom -and ($normalizedBaselines -notcontains "random")) {
    $normalizedBaselines += "random"
}

$selectedRuns = New-Object System.Collections.Generic.List[object]
foreach ($site in $normalizedSites) {
    if ($unavailableSites.Contains($site)) {
        continue
    }
    foreach ($baseline in $normalizedBaselines) {
        if (-not $profileSuffix.ContainsKey($baseline)) {
            Write-Warning "Unknown baseline '$baseline', skip."
            continue
        }
        $selectedRuns.Add([PSCustomObject]@{
            Site     = $site
            Baseline = $baseline
            Profile  = "$site-$($profileSuffix[$baseline])"
        })
    }
}

if ($selectedRuns.Count -eq 0) {
    throw "No valid (site, baseline) combinations to run."
}

$total = $selectedRuns.Count * $normalizedSeeds.Count
$current = 0
$failedRuns = New-Object System.Collections.Generic.List[object]

Write-Host "Selected Sites: $($normalizedSites -join ', ')"
Write-Host "Selected Baselines: $($normalizedBaselines -join ', ')"
Write-Host "Selected Seeds: $($normalizedSeeds -join ', ')"
Write-Host "Total runs: $total"

foreach ($run in $selectedRuns) {
    $profile = $run.Profile
    $baseline = $run.Baseline
    foreach ($seed in $normalizedSeeds) {
        $current++
        $session = "$baseline-$seed"
        $logfile = "${profile}_${baseline}_${seed}.log"

        Write-Host ""
        Write-Host "============================================================"
        Write-Host "[$current / $total] profile=$profile  baseline=$baseline  session=$session"
        Write-Host "  Log: $logfile"
        Write-Host "  Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        Write-Host "============================================================"

        if ($DryRun) {
            continue
        }

        if ($run.Site -eq "splittypie" -and -not (Test-LocalPort -Port 4200)) {
            Write-Warning "Skip run: splittypie requires http://localhost:4200 but port 4200 is closed."
            $failedRuns.Add([PSCustomObject]@{
                Profile  = $profile
                Baseline = $baseline
                Seed     = $seed
                Log      = $logfile
                ExitCode = 111
            })
            continue
        }
        if ($run.Site -eq "petclinic" -and -not (Test-LocalPort -Port 8081)) {
            Write-Warning "Skip run: petclinic requires http://localhost:8081 but port 8081 is closed."
            $failedRuns.Add([PSCustomObject]@{
                Profile  = $profile
                Baseline = $baseline
                Seed     = $seed
                Log      = $logfile
                ExitCode = 111
            })
            continue
        }

        $env:OBSERVATION_MG_FULL_HISTORY = "1"
        if ([string]::IsNullOrWhiteSpace($env:WEBTEST_ALIVE_TIME_OVERRIDE)) {
            $env:WEBTEST_ALIVE_TIME_OVERRIDE = "3600"
        }
        if ($fairBaselines -contains $baseline) {
            $env:WEBTEST_FAIR_MODE = "1"
            $env:WEBTEST_EXPERIMENT_TYPE = "fair-benchmark"
        } else {
            $env:WEBTEST_FAIR_MODE = "0"
            $env:WEBTEST_EXPERIMENT_TYPE = "benchmark"
        }
        $env:WEBTEST_FORCE_SYSTEM_CHROME = $ForceSystemChrome
        $env:WEBTEST_FORCE_HEADFUL = $ForceHeadful
        if ([string]::IsNullOrWhiteSpace($env:WEBTEST_ENABLE_SCREENSHOT)) {
            $env:WEBTEST_ENABLE_SCREENSHOT = "0"
        }
        $env:WEBTEST_ENABLE_TRANSLATE_API = "0"
        $env:WEBTEST_RELAX_HTTPS = "0"
        $env:WEBTEST_BLOCK_AUTH_ACTIONS = "1"
        $env:WEBTEST_BLOCK_OFFDOMAIN_ACTIONS = "1"
        Set-SiteTargetEnv $run.Site
        if ($run.Site -eq "splittypie") {
            $env:WEBTEST_SPLITTYPIE_BOOTSTRAP_EVENT = "1"
        } else {
            Remove-Item Env:WEBTEST_SPLITTYPIE_BOOTSTRAP_EVENT -ErrorAction SilentlyContinue
        }
        $env:WEBTEST_AUTH_PATH_PREFIXES = "/login,/logout,/signup,/sign-in,/sign-out,/signin,/signout,/register,/password_reset,/session,/sessions,/oauth,/auth"
        $env:WEBTEST_MAX_EMPTY_ACTION_ROUNDS = "120"
        $env:WEBTEST_MAX_BROWSER_ERROR_ROUNDS = "25"
        $env:WEBTEST_MAX_HTTP_5XX_ROUNDS = "20"
        $env:WEBTEST_FILTER_KNOWN_NOISE = "1"
        $env:WEBTEST_ACTION_DETECT_RETRY = "3"
        $env:WEBTEST_ACTION_DETECT_RETRY_WAIT = "1.2"
        if ([string]::IsNullOrWhiteSpace($env:PYTHONIOENCODING)) {
            $env:PYTHONIOENCODING = "utf-8"
        }
        $env:WEBTEST_URL_NORMALIZE_MODE = "path_only"
        if ([string]::IsNullOrWhiteSpace($env:WEBTEST_RESTART_POLICY_OVERRIDE)) {
            $env:WEBTEST_RESTART_POLICY = "entry"
        } else {
            $env:WEBTEST_RESTART_POLICY = $env:WEBTEST_RESTART_POLICY_OVERRIDE
        }
        if ([string]::IsNullOrWhiteSpace($env:WEBTEST_MAX_TRANSITIONS)) {
            $env:WEBTEST_MAX_TRANSITIONS = "2200"
        }
        if ([string]::IsNullOrWhiteSpace($env:WEBTEST_REPLAY_BUFFER_CAPACITY)) {
            $env:WEBTEST_REPLAY_BUFFER_CAPACITY = "10000"
        }
        if ([string]::IsNullOrWhiteSpace($rewardClipAbsOverride)) {
            $env:WEBTEST_REWARD_CLIP_ABS = "5.0"
        } else {
            $env:WEBTEST_REWARD_CLIP_ABS = $rewardClipAbsOverride
        }
        Remove-Item Env:WEBTEST_DRL_REWARD_FUNCTION -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_WEBEXPLOR_REWARD_MODE -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_QLEARNING_REWARD_MODE -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_QLEARNING_AGENT_TYPE -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_QLEARNING_STATE_MODE -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_QLEARNING_POLICY_MODE -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_QLEARNING_GUMBEL_TAU -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_QLEARNING_ENABLE_DFA -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_QLEARNING_DFA_STUCK_STEPS -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_QLEARNING_DFA_STUCK_SECONDS -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_ALLOW_TAG_SEQUENCE_IN_FAIR_WEBEXPLOR -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_TAG_SEQUENCE_REQUIRE_SAME_URL -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_URL_PRESERVE_HASH_ROUTE -ErrorAction SilentlyContinue
        Remove-Item Env:WEBEXPLOR_DFA_STUCK_SECONDS -ErrorAction SilentlyContinue
        Remove-Item Env:WEBEXPLOR_DFA_STUCK_STEPS -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_SMART_INPUTS -ErrorAction SilentlyContinue
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
        Remove-Item Env:WEBTEST_MAX_ACTIONS -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_ROLLOUT_LEN -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_A2C_GAMMA -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_A2C_ADVANTAGE_ESTIMATOR -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_A2C_GAE_LAMBDA -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_A2C_UPDATE_EPOCHS -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_A2C_ENTROPY_COEF -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_A2C_VALUE_COEF -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_A2C_MAX_GRAD_NORM -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_A2C_SEPARATE_GRAD_CLIP -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_A2C_LR -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_SUBWEB_POLICY_INPUT_MODE -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_FRONTIER_DENSITY_LAMBDA -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_FRONTIER_WEIGHT_MIN -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_FRONTIER_WEIGHT_MAX -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_FRONTIER_DENSITY_EMA_ALPHA -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_FRONTIER_DENSITY_HORIZON -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_GRAPH_EDGE_WEIGHT_MODE -ErrorAction SilentlyContinue
        $subwebCleanEnvNames = @(
            "WEBTEST_GRAPH_RESIDUAL_AUX",
            "WEBTEST_GRAPH_RESIDUAL_AUX_COEF",
            "WEBTEST_GRAPH_RESIDUAL_AUX_TARGET",
            "WEBTEST_GRAPH_RESIDUAL_AUX_TARGET_POWER",
            "WEBTEST_GRAPH_RESIDUAL_AUX_DISTRIBUTION",
            "WEBTEST_GRAPH_RESIDUAL_AUX_TIE_BREAK_COEF",
            "WEBTEST_GRAPH_FILTER_TEMPLATE_EDGES"
        )
        foreach ($name in $subwebCleanEnvNames) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        }
        Remove-Item Env:WEBTEST_FRONTIER_EDGE_BETA -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_STATE_SIMILARITY_MODE -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_STATE_URL_SIM_WEIGHT -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_CROSS_URL_SIM_CAP -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_CGFS_CROSS_ROUTE_DISCOUNT -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_SIM_PRESERVE_URL_TOKENS -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_SIM_PRESERVE_HASH_ROUTE -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_AGENT_MODULE -ErrorAction SilentlyContinue
        Remove-Item Env:WEBTEST_AGENT_CLASS -ErrorAction SilentlyContinue
        $numericSeed = ($seed -replace '[^0-9]', '')
        if ([string]::IsNullOrWhiteSpace($numericSeed)) {
            $env:WEBTEST_RUN_SEED = $seed
        } else {
            $env:WEBTEST_RUN_SEED = $numericSeed
        }
        Remove-Item Env:WEBTEST_USER_AGENT -ErrorAction SilentlyContinue

        if ($baseline -eq "webexplor") {
            $env:WEBTEST_WEBEXPLOR_STATE_MODE = "tagseq"
            $env:WEBTEST_ALLOW_TAG_SEQUENCE_IN_FAIR_WEBEXPLOR = "1"
            $env:WEBTEST_TAG_SEQUENCE_REQUIRE_SAME_URL = "1"
            $env:WEBTEST_URL_PRESERVE_HASH_ROUTE = "1"
            $env:WEBEXPLOR_DFA_STUCK_SECONDS = "120"
            $env:WEBTEST_SMART_INPUTS = "1"
            if ($run.Site -eq "odoo") {
                # Use the public Odoo runbot target rather than the sparse local demo DB.
                $env:WEBTEST_SITE_ODOO_ENTRY_URL = "https://runbot.odoo.com/"
                $env:WEBTEST_SITE_ODOO_DOMAINS = "odoo.com"
            }
            if ($run.Site -eq "splittypie") {
                $webExplorSplittyPieBootstrap = $env:WEBTEST_WEBEXPLOR_SPLITTYPIE_BOOTSTRAP_EVENT
                if ([string]::IsNullOrWhiteSpace($webExplorSplittyPieBootstrap)) {
                    $webExplorSplittyPieBootstrap = "0"
                }
                $env:WEBTEST_SPLITTYPIE_BOOTSTRAP_EVENT = $webExplorSplittyPieBootstrap
            }
            Write-Host "  WebExplor mode: paper-style URL-gated tag-sequence + 120s DFA guidance + W3C smart inputs"
        }
        if ($baseline -eq "webqt") {
            $env:WEBTEST_QLEARNING_AGENT_TYPE = "W"
            $env:WEBTEST_QLEARNING_STATE_MODE = "actionset"
            $env:WEBTEST_QLEARNING_REWARD_MODE = "webqt"
            $env:WEBQT_W_LOC = "10"
            $env:WEBQT_W_ATTENTION = "50"
            $env:WEBQT_W_FREQ = "5"
            $env:WEBQT_W_EXPLORE = "5"
            $env:WEBQT_STATE_SIM_THRESHOLD = "0.85"
            $env:WEBQT_STAGNATION_WINDOW = "50"
            $env:WEBQT_DECAY_EPSILON = "1"
            $env:WEBQT_MIN_EPSILON = "0.1"
            $env:WEBTEST_REWARD_CLIP_ABS = "0"
            Write-Host "  Baseline mode: WebQT-style tabular Q-learning"
        }
        if (Use-OfficialWebRLED $baseline) {
            Write-Host "  Baseline mode: Official WebRLED package"
        }
        if ($baseline -eq "subweb-frontier-a2c") {
            $env:WEBTEST_FRONTIER_COVERAGE = "1"
            $env:WEBTEST_PAD_ACTIONS = "1"
            if (-not [string]::IsNullOrWhiteSpace($subwebGraphEdgeWeightModeOverride)) {
                $env:WEBTEST_GRAPH_EDGE_WEIGHT_MODE = $subwebGraphEdgeWeightModeOverride
            }
            if (-not [string]::IsNullOrWhiteSpace($subwebPolicyInputModeOverride)) {
                $env:WEBTEST_SUBWEB_POLICY_INPUT_MODE = $subwebPolicyInputModeOverride
            } else {
                $env:WEBTEST_SUBWEB_POLICY_INPUT_MODE = "augmented"
            }
            if (-not [string]::IsNullOrWhiteSpace($subwebGraphResidualAuxOverride)) {
                $env:WEBTEST_GRAPH_RESIDUAL_AUX = $subwebGraphResidualAuxOverride
                $graphResidualAuxEnabled = $subwebGraphResidualAuxOverride -match '^(1|true|yes|on)$'
                Write-Host "  WebCover residual auxiliary override=$subwebGraphResidualAuxOverride enabled=$([int]$graphResidualAuxEnabled)"
            } elseif ([string]::IsNullOrWhiteSpace($env:WEBTEST_GRAPH_RESIDUAL_AUX)) {
                $env:WEBTEST_GRAPH_RESIDUAL_AUX = "1"
            }
            if (-not [string]::IsNullOrWhiteSpace($subwebGraphResidualAuxCoefOverride)) {
                $env:WEBTEST_GRAPH_RESIDUAL_AUX_COEF = $subwebGraphResidualAuxCoefOverride
            }
            if (-not [string]::IsNullOrWhiteSpace($subwebGraphResidualAuxTargetOverride)) {
                $env:WEBTEST_GRAPH_RESIDUAL_AUX_TARGET = $subwebGraphResidualAuxTargetOverride
            }
            if (-not [string]::IsNullOrWhiteSpace($subwebGraphResidualAuxTargetPowerOverride)) {
                $env:WEBTEST_GRAPH_RESIDUAL_AUX_TARGET_POWER = $subwebGraphResidualAuxTargetPowerOverride
            }
            if (-not [string]::IsNullOrWhiteSpace($subwebGraphResidualAuxDistributionOverride)) {
                $env:WEBTEST_GRAPH_RESIDUAL_AUX_DISTRIBUTION = $subwebGraphResidualAuxDistributionOverride
            }
            if (-not [string]::IsNullOrWhiteSpace($subwebGraphResidualAuxTieBreakCoefOverride)) {
                $env:WEBTEST_GRAPH_RESIDUAL_AUX_TIE_BREAK_COEF = $subwebGraphResidualAuxTieBreakCoefOverride
            }
            $subwebPolicyInputLabel = $env:WEBTEST_SUBWEB_POLICY_INPUT_MODE
            $subwebOptimizerLabel = $env:WEBTEST_SUBWEB_OPTIMIZER
            if ([string]::IsNullOrWhiteSpace($subwebOptimizerLabel)) {
                $subwebOptimizerLabel = "a2c"
            }
            if ([string]::IsNullOrWhiteSpace($rewardClipAbsOverride)) {
                $env:WEBTEST_REWARD_CLIP_ABS = "0"
            }
            if (-not [string]::IsNullOrWhiteSpace($subwebMaxActionsOverride)) {
                $env:WEBTEST_MAX_ACTIONS = $subwebMaxActionsOverride
            } elseif ([string]::IsNullOrWhiteSpace($env:WEBTEST_MAX_ACTIONS)) {
                $env:WEBTEST_MAX_ACTIONS = "512"
            }
            if (-not [string]::IsNullOrWhiteSpace($subwebRolloutLenOverride)) {
                $env:WEBTEST_ROLLOUT_LEN = $subwebRolloutLenOverride
            } elseif ([string]::IsNullOrWhiteSpace($env:WEBTEST_ROLLOUT_LEN)) {
                $env:WEBTEST_ROLLOUT_LEN = "32"
            }
                if (-not [string]::IsNullOrWhiteSpace($subwebAdvantageEstimatorOverride)) {
                    $env:WEBTEST_A2C_ADVANTAGE_ESTIMATOR = $subwebAdvantageEstimatorOverride
                } elseif ([string]::IsNullOrWhiteSpace($env:WEBTEST_A2C_ADVANTAGE_ESTIMATOR)) {
                    $env:WEBTEST_A2C_ADVANTAGE_ESTIMATOR = "n_step"
                }
                if (-not [string]::IsNullOrWhiteSpace($subwebGaeLambdaOverride)) {
                    $env:WEBTEST_A2C_GAE_LAMBDA = $subwebGaeLambdaOverride
                }
                if (-not [string]::IsNullOrWhiteSpace($subwebUpdateEpochsOverride)) {
                    $env:WEBTEST_A2C_UPDATE_EPOCHS = $subwebUpdateEpochsOverride
                } elseif ([string]::IsNullOrWhiteSpace($env:WEBTEST_A2C_UPDATE_EPOCHS)) {
                    $env:WEBTEST_A2C_UPDATE_EPOCHS = [string]$UpdateEpochs
                }
                if (-not [string]::IsNullOrWhiteSpace($subwebGammaOverride)) {
                    $env:WEBTEST_A2C_GAMMA = $subwebGammaOverride
                } elseif ([string]::IsNullOrWhiteSpace($env:WEBTEST_A2C_GAMMA)) {
                    $env:WEBTEST_A2C_GAMMA = "1.0"
                }
                if (-not [string]::IsNullOrWhiteSpace($subwebEntropyCoefOverride)) {
                    $env:WEBTEST_A2C_ENTROPY_COEF = $subwebEntropyCoefOverride
                }
                if (-not [string]::IsNullOrWhiteSpace($subwebValueCoefOverride)) {
                    $env:WEBTEST_A2C_VALUE_COEF = $subwebValueCoefOverride
                }
                if (-not [string]::IsNullOrWhiteSpace($subwebMaxGradNormOverride)) {
                    $env:WEBTEST_A2C_MAX_GRAD_NORM = $subwebMaxGradNormOverride
                }
                if (-not [string]::IsNullOrWhiteSpace($subwebSeparateGradClipOverride)) {
                    $env:WEBTEST_A2C_SEPARATE_GRAD_CLIP = $subwebSeparateGradClipOverride
                } elseif ([string]::IsNullOrWhiteSpace($env:WEBTEST_A2C_SEPARATE_GRAD_CLIP)) {
                    $env:WEBTEST_A2C_SEPARATE_GRAD_CLIP = "1"
                }
                if (-not [string]::IsNullOrWhiteSpace($subwebLrOverride)) {
                    $env:WEBTEST_A2C_LR = $subwebLrOverride
                }
                if ($subwebOptimizerLabel -eq "qlearning") {
                    Write-Host "  WebCover mode: graph coverage |V| + alpha|E| + Markov coverage-augmented candidate-action Q-learning (input=$subwebPolicyInputLabel)"
                } else {
                    Write-Host "  WebCover mode: graph coverage |V| + alpha|E| + Markov padded-action online A2C (input=$subwebPolicyInputLabel)"
                }
        }

        if ($profile -like "toppr-*") {
            # toppr tends to return anti-bot/empty pages in headless mode.
            $env:WEBTEST_FORCE_HEADFUL = "1"
            $env:WEBTEST_MAX_EMPTY_ACTION_ROUNDS = "35"
            $env:WEBTEST_MAX_BROWSER_ERROR_ROUNDS = "15"
            $env:WEBTEST_MAX_HTTP_5XX_ROUNDS = "6"
            $env:WEBTEST_RELAX_HTTPS = "1"
            $env:WEBTEST_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
            Write-Host "  Toppr mode: headful + anti-bot UA + fast fail on empty actions"
        }

        if (Use-OfficialWebRLED $baseline) {
            # Ensure public-URL sites are passed correctly to run_webrled_official.py,
            # which has its own SITE_DEFAULTS dict and does not read settings.yaml.
            $prevOdooEntryUrl = $env:WEBTEST_SITE_ODOO_ENTRY_URL
            $prevOdooDomains = $env:WEBTEST_SITE_ODOO_DOMAINS
            $env:WEBTEST_SITE_ODOO_ENTRY_URL = "https://runbot.odoo.com/"
            $env:WEBTEST_SITE_GITHUB_DOMAINS = "https://github.com"
            $env:WEBTEST_SITE_ODOO_DOMAINS = "https://runbot.odoo.com"
            $prevPythonPath = $env:PYTHONPATH
            if (-not [string]::IsNullOrWhiteSpace($webrledOfficialSitePackages)) {
                if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
                    $env:PYTHONPATH = $webrledOfficialSitePackages
                } elseif ($env:PYTHONPATH -notlike "*$webrledOfficialSitePackages*") {
                    $env:PYTHONPATH = "$webrledOfficialSitePackages;$($env:PYTHONPATH)"
                }
            }
            if ($webrledOfficialUsePyLauncher) {
                Write-Host "  WebRLED official python: py -$webrledOfficialPyVersion"
                if (-not [string]::IsNullOrWhiteSpace($webrledOfficialSitePackages)) {
                    Write-Host "  WebRLED official site-packages: $webrledOfficialSitePackages"
                }
                # Python loggers write normal progress to stderr. Let cmd.exe merge
                # streams first so PowerShell does not turn stderr into NativeCommandError.
                $nativeCommand = "py -$webrledOfficialPyVersion run_webrled_official.py --site `"$($run.Site)`" --session `"$session`" --seed `"$seed`" 2>&1"
                & cmd.exe /d /c $nativeCommand | Tee-Object -FilePath $logfile
            } else {
                Write-Host "  WebRLED official python: $webrledOfficialPython"
                if (-not [string]::IsNullOrWhiteSpace($webrledOfficialSitePackages)) {
                    Write-Host "  WebRLED official site-packages: $webrledOfficialSitePackages"
                }
                $nativeCommand = "`"$webrledOfficialPython`" run_webrled_official.py --site `"$($run.Site)`" --session `"$session`" --seed `"$seed`" 2>&1"
                & cmd.exe /d /c $nativeCommand | Tee-Object -FilePath $logfile
            }
            if ($null -eq $prevPythonPath) {
                Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
            } else {
                $env:PYTHONPATH = $prevPythonPath
            }
            if ($null -eq $prevOdooEntryUrl) {
                Remove-Item Env:WEBTEST_SITE_ODOO_ENTRY_URL -ErrorAction SilentlyContinue
            } else {
                $env:WEBTEST_SITE_ODOO_ENTRY_URL = $prevOdooEntryUrl
            }
            if ($null -eq $prevOdooDomains) {
                Remove-Item Env:WEBTEST_SITE_ODOO_DOMAINS -ErrorAction SilentlyContinue
            } else {
                $env:WEBTEST_SITE_ODOO_DOMAINS = $prevOdooDomains
            }
        } else {
            Write-Host "  Main python: $mainPython"
            $nativeCommand = "`"$mainPython`" main.py --profile `"$profile`" --session `"$session`" 2>&1"
            & cmd.exe /d /c $nativeCommand | Tee-Object -FilePath $logfile
        }
        $mainExitCode = $LASTEXITCODE

        # --- IMPORTANT CLEANUP ---
        # Kill dangling chromedriver/chrome instances to prevent OOM / port
        # conflicts across long serial batches. Disable this in parallel mode,
        # otherwise one runner may kill another runner's active browser.
        $globalBrowserCleanup = $env:WEBTEST_GLOBAL_BROWSER_CLEANUP
        if ([string]::IsNullOrWhiteSpace($globalBrowserCleanup)) {
            $globalBrowserCleanup = "1"
        }
        if ($globalBrowserCleanup -eq "0") {
            Write-Host "  Parallel mode: skip global ChromeDriver/Chrome cleanup."
        } else {
            Write-Host "  Cleaning up ChromeDriver and Chrome zombies..."
            Stop-Process -Name "chromedriver" -Force -ErrorAction SilentlyContinue
            Stop-Process -Name "chrome" -Force -ErrorAction SilentlyContinue
        }

        $auditExitCode = 0
        if (-not (Use-OfficialWebRLED $baseline)) {
            $resultDir = Join-Path $PSScriptRoot ("webtest_output\result\{0}-{1}" -f $profile, $session)
            if (Test-Path $resultDir) {
                python tools\audit_fairness.py --result-dir $resultDir --profile $profile --site $($run.Site) --settings settings.yaml --max-offdomain-exec 0 --min-unique-states 10 --min-transitions-for-low-unique 200
                $auditExitCode = $LASTEXITCODE
            } else {
                Write-Warning "Fairness audit skipped: result directory not found: $resultDir"
            }
        }

        $runExitCode = $mainExitCode
        Write-Host "  Finish: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        Write-Host "  Exit code: $runExitCode"
        if ($runExitCode -ne 0) {
            $failedRuns.Add([PSCustomObject]@{
                Profile  = $profile
                Baseline = $baseline
                Seed     = $seed
                Log      = $logfile
                ExitCode = $runExitCode
            })
            Write-Warning "Run failed: profile=$profile baseline=$baseline seed=$seed (exit=$runExitCode), log=$logfile"

            if (Test-Path $logfile) {
                $hasDevToolsPortCrash = Select-String -Path $logfile -Pattern "DevToolsActivePort file doesn't exist" -Quiet
                $hasNoChromeBinary = Select-String -Path $logfile -Pattern "cannot find Chrome binary" -Quiet
                if ($hasDevToolsPortCrash -or $hasNoChromeBinary) {
                    Write-Host "  Hint: browser bootstrap failed (Chrome/Driver startup issue, not RL logic)." -ForegroundColor Yellow
                    Write-Host "  Suggestion 1: install a system Chrome and rerun with -ForceSystemChrome 1" -ForegroundColor Yellow
                    Write-Host "  Suggestion 2: update bundled chrome/chromedriver to a matching modern major version" -ForegroundColor Yellow
                }
            }
        }
        if ($auditExitCode -ne 0) {
            $failedRuns.Add([PSCustomObject]@{
                Profile  = $profile
                Baseline = $baseline
                Seed     = $seed
                Log      = $logfile
                ExitCode = "audit-$auditExitCode"
            })
            Write-Warning "Fairness audit failed: profile=$profile baseline=$baseline seed=$seed (audit=$auditExitCode), log=$logfile"
        }
    }
}

Write-Host ""
if ($DryRun) {
    Write-Host "DryRun complete."
} else {
    if ($failedRuns.Count -eq 0) {
        Write-Host "All experiments complete. Total runs: $total"
    } else {
        Write-Host "Experiments finished with failures."
        Write-Host "Failed runs: $($failedRuns.Count) / $total"
        foreach ($f in $failedRuns) {
            Write-Host "  - profile=$($f.Profile), baseline=$($f.Baseline), seed=$($f.Seed), exit=$($f.ExitCode), log=$($f.Log)"
        }
        exit 1
    }
}
