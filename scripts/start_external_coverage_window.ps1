param(
  [ValidateRange(1, 4)]
  [int]$Window = 1,
  [string]$Seeds = "seedextcov1_20260622,seedextcov2_20260622,seedextcov3_20260622",
  [int]$TimeLimit = 3600,
  [int]$ForceHeadful = 0,
  [int]$QExploreDepth = 100,
  [switch]$DryRun,
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$javaLiveSites = "agilefant,gadael,timeoff"
$instrumentedSites = "splittypie,petclinic"
$qexploreForce = [bool]$Force -or [bool]$DryRun

Write-Host "[external-coverage-window] window=$Window seeds=$Seeds time_limit=$TimeLimit dry_run=$([bool]$DryRun)"

switch ($Window) {
  1 {
    Write-Host "[external-coverage-window] WebRLED live Java coverage sites=$javaLiveSites"
    & .\scripts\start_webrled_official_batch.ps1 `
      -Sites $javaLiveSites `
      -Seeds $Seeds `
      -TimeLimit $TimeLimit `
      -ForceHeadful $ForceHeadful `
      -GlobalBrowserCleanup 0 `
      -Coverage `
      -DryRun:$DryRun
  }
  2 {
    Write-Host "[external-coverage-window] QExplore live Java coverage sites=$javaLiveSites"
    & .\scripts\start_qexplore_batch.ps1 `
      -Sites $javaLiveSites `
      -Seeds $Seeds `
      -ActivityTime $TimeLimit `
      -Depth $QExploreDepth `
      -Coverage `
      -DryRun:$DryRun `
      -Force:$qexploreForce
  }
  3 {
    Write-Host "[external-coverage-window] WebRLED instrumented coverage sites=$instrumentedSites"
    $env:WEBTEST_SITE_SPLITTYPIE_COVERAGE_URL = "http://localhost:6972"
    $env:WEBTEST_SITE_SPLITTYPIE_ENTRY_URL = "http://localhost:4005"
    $env:WEBTEST_SITE_SPLITTYPIE_DOMAINS = "http://localhost:4005"
    $env:WEBTEST_SITE_PETCLINIC_COVERAGE_URL = "http://localhost:6973"
    $env:WEBTEST_SITE_PETCLINIC_ENTRY_URL = "http://localhost:4002"
    $env:WEBTEST_SITE_PETCLINIC_DOMAINS = "http://localhost:4002"
    & .\scripts\start_webrled_official_batch.ps1 `
      -Sites $instrumentedSites `
      -Seeds $Seeds `
      -TimeLimit $TimeLimit `
      -ForceHeadful $ForceHeadful `
      -GlobalBrowserCleanup 0 `
      -Coverage `
      -DryRun:$DryRun
  }
  4 {
    Write-Host "[external-coverage-window] QExplore instrumented coverage sites=$instrumentedSites"
    & .\scripts\start_qexplore_batch.ps1 `
      -Sites $instrumentedSites `
      -Seeds $Seeds `
      -ActivityTime $TimeLimit `
      -Depth $QExploreDepth `
      -Coverage `
      -DryRun:$DryRun `
      -Force:$qexploreForce
  }
}
