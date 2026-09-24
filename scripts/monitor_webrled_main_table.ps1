param(
  [string]$Sites = "4gaboards,agilefant,gadael,petclinic,realworld,splittypie,timeoff,github,nextcloud,odoo",
  [string]$Seeds = "seedwebrledmain1_20260622,seedwebrledmain2_20260622,seedwebrledmain3_20260622,seedwebrledmain4_20260622,seedwebrledmain5_20260622",
  [int]$StaleMinutes = 30,
  [int]$Limit = 120,
  [switch]$IncludeMissing,
  [string]$OutputCsv = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$argsList = @(
  "scripts\monitor_external_baseline_runs.py",
  "--only", "webrled",
  "--sites", $Sites,
  "--seeds", $Seeds,
  "--stale-minutes", [string]$StaleMinutes,
  "--limit", [string]$Limit
)

if ($IncludeMissing) {
  $argsList += "--include-missing"
}
if (-not [string]::IsNullOrWhiteSpace($OutputCsv)) {
  $argsList += @("--output-csv", $OutputCsv)
}

& python @argsList
