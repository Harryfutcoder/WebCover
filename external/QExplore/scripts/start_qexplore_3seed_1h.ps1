param(
  [string]$Sites = "odoo,4gaboards,github,petclinic,agilefant,splittypie,realworld,nextcloud,gadael,timeoff",
  [string]$Seeds = "seedqexplore1,seedqexplore2,seedqexplore3",
  [int]$ActivityTime = 3600,
  [int]$Depth = 100,
  [string]$Python = "python",
  [switch]$Coverage,
  [switch]$DryRun,
  [switch]$Force
)

$ErrorActionPreference = "Stop"

$siteList = $Sites.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$seedList = $Seeds.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }

foreach ($site in $siteList) {
  foreach ($seed in $seedList) {
    & (Join-Path $PSScriptRoot "run_qexplore_site.ps1") `
      -Site $site `
      -Seed $seed `
      -ActivityTime $ActivityTime `
      -Depth $Depth `
      -Python $Python `
      -Coverage:$Coverage `
      -DryRun:$DryRun `
      -Force:$Force
  }
}
