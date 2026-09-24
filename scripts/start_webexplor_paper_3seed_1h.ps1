param(
  [string]$Sites = "odoo,4gaboards,github,petclinic,agilefant,splittypie,realworld,nextcloud,gadael,timeoff",
  [string]$Seeds = "seedwebexplorpaperfinal1_20260621,seedwebexplorpaperfinal2_20260621,seedwebexplorpaperfinal3_20260621",
  [int]$ForceHeadful = 0,
  [int]$MaxTransitions = 2200,
  [ValidateSet("webexplor")]
  [string]$Baseline = "webexplor",
  [ValidateSet("0","1")]
  [string]$GlobalBrowserCleanup = "0",
  [ValidateSet("auto","0","1")]
  [string]$SplittyPieBootstrapEvent = "auto",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$campaignDir = Join-Path $root "analysis\webexplor_paper"
$campaignFile = Join-Path $campaignDir "current_campaign_start.txt"

Set-Location $root
New-Item -ItemType Directory -Force -Path $campaignDir | Out-Null

$stamp = Get-Date -Format "o"
if (-not $DryRun) {
  Set-Content -Path $campaignFile -Value $stamp -Encoding UTF8
}

# Paper-faithful defaults. run_experiments.ps1 also sets these for the
# webexplor profile; keeping them here makes the campaign contract explicit.
$env:WEBTEST_MAX_TRANSITIONS = "$MaxTransitions"
$env:WEBTEST_ALIVE_TIME_OVERRIDE = "3600"
$env:WEBTEST_WEBEXPLOR_STATE_MODE = "tagseq"
$env:WEBTEST_ALLOW_TAG_SEQUENCE_IN_FAIR_WEBEXPLOR = "1"
$env:WEBTEST_TAG_SEQUENCE_REQUIRE_SAME_URL = "1"
$env:WEBTEST_URL_PRESERVE_HASH_ROUTE = "1"
$env:WEBEXPLOR_DFA_STUCK_SECONDS = "120"
$env:WEBTEST_SMART_INPUTS = "1"
$env:WEBTEST_GLOBAL_BROWSER_CLEANUP = "$GlobalBrowserCleanup"
if ($SplittyPieBootstrapEvent -eq "auto") {
  $effectiveSplittyPieBootstrapEvent = "0"
} else {
  $effectiveSplittyPieBootstrapEvent = "$SplittyPieBootstrapEvent"
}
$env:WEBTEST_WEBEXPLOR_SPLITTYPIE_BOOTSTRAP_EVENT = "$effectiveSplittyPieBootstrapEvent"
$env:WEBTEST_SPLITTYPIE_BOOTSTRAP_EVENT = "$effectiveSplittyPieBootstrapEvent"
$env:WEBTEST_SITE_ODOO_ENTRY_URL = "https://runbot.odoo.com/"
$env:WEBTEST_SITE_ODOO_DOMAINS = "odoo.com"
Remove-Item Env:WEBEXPLOR_DFA_STUCK_STEPS -ErrorAction SilentlyContinue

Write-Host "[webexplor-3seed] campaign_start=$stamp"
Write-Host "[webexplor-3seed] campaign_file=$campaignFile"
Write-Host "[webexplor-3seed] sites=$Sites"
Write-Host "[webexplor-3seed] baseline=$Baseline seeds=$Seeds alive_time=3600s max_transitions=$MaxTransitions"
Write-Host "[webexplor-3seed] config=URL-gated tagseq threshold=0.8 preserve_hash_route=1 curiosity=transition_count gumbel_tau=1 DFA_stuck_seconds=120 smart_inputs=W3C"
Write-Host "[webexplor-3seed] odoo_target=https://runbot.odoo.com/ domains=odoo.com"
Write-Host "[webexplor-3seed] splittypie_bootstrap_event=$effectiveSplittyPieBootstrapEvent (requested=$SplittyPieBootstrapEvent; 0=faithful WebExplor cold-start, 1=match WebQT site setup)"
Write-Host "[webexplor-3seed] global_browser_cleanup=$GlobalBrowserCleanup (0 is required for concurrent windows)"
if ($DryRun) {
  Write-Host "[webexplor-3seed] DryRun: campaign start file was not modified."
}

$runParams = @{
  Sites = $Sites
  Baselines = $Baseline
  Seeds = $Seeds
  ForceHeadful = "$ForceHeadful"
}
if ($DryRun) {
  $runParams["DryRun"] = $true
}

& .\run_experiments.ps1 @runParams
