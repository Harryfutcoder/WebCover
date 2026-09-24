param(
  [ValidateSet("webrled", "qexplore")]
  [string]$Algorithm = "webrled",
  [ValidateRange(1, 3)]
  [int]$Window = 1,
  [string]$Sites = "odoo,4gaboards,github,petclinic,agilefant,splittypie,realworld,nextcloud,gadael,timeoff",
  [string]$Seeds = "seedexternal1_20260622,seedexternal2_20260622,seedexternal3_20260622",
  [int]$TimeLimit = 3600,
  [int]$ForceHeadful = 0,
  [int]$QExploreDepth = 100,
  [switch]$DryRun,
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Split-IntoGroups {
  param([string[]]$Items, [int]$GroupCount)
  $groups = @()
  for ($i = 0; $i -lt $GroupCount; $i++) {
    $groups += ,(New-Object System.Collections.Generic.List[string])
  }
  for ($i = 0; $i -lt $Items.Count; $i++) {
    $groups[$i % $GroupCount].Add($Items[$i])
  }
  return $groups
}

$siteList = $Sites.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
if (-not $siteList) {
  throw "No sites provided."
}

$groups = Split-IntoGroups -Items $siteList -GroupCount 3
$selectedSites = ($groups[$Window - 1] -join ",")
if (-not $selectedSites) {
  throw "Window $Window has no assigned sites."
}

Write-Host "[external-baseline-window] algorithm=$Algorithm window=$Window sites=$selectedSites"
Write-Host "[external-baseline-window] seeds=$Seeds time_limit=$TimeLimit dry_run=$([bool]$DryRun)"

if ($Algorithm -eq "webrled") {
  & (Join-Path $repo "scripts\start_webrled_official_batch.ps1") `
    -Sites $selectedSites `
    -Seeds $Seeds `
    -TimeLimit $TimeLimit `
    -ForceHeadful $ForceHeadful `
    -GlobalBrowserCleanup 0 `
    -DryRun:$DryRun
} else {
  $qexploreForce = [bool]$Force -or [bool]$DryRun
  & (Join-Path $repo "scripts\start_qexplore_batch.ps1") `
    -Sites $selectedSites `
    -Seeds $Seeds `
    -ActivityTime $TimeLimit `
    -Depth $QExploreDepth `
    -DryRun:$DryRun `
    -Force:$qexploreForce
}
