param(
  [Parameter(Mandatory=$true)]
  [ValidateSet("4gaboards", "realworld")]
  [string]$Site,

  [string]$ContainerName = "",
  [string]$WorkDir = "",
  [string]$ReportPath = "",
  [string]$OutDir = "analysis/webrled_code_coverage/postrun"
)

$ErrorActionPreference = "Stop"

if (-not $ContainerName) {
  if ($Site -eq "4gaboards") { $ContainerName = "4gaboards-app-1" }
  if ($Site -eq "realworld") { $ContainerName = "realworld-app-1" }
}
if (-not $WorkDir) {
  if ($Site -eq "4gaboards") { $WorkDir = "/app" }
  if ($Site -eq "realworld") { $WorkDir = "/app" }
}
if (-not $ReportPath) {
  if ($Site -eq "4gaboards") { $ReportPath = "$WorkDir/server/coverage" }
  if ($Site -eq "realworld") { $ReportPath = "$WorkDir/coverage" }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$localDir = Join-Path $OutDir "${Site}_${stamp}"
New-Item -ItemType Directory -Force -Path $localDir | Out-Null

Write-Host "[postrun-nyc] site=$Site container=$ContainerName workdir=$WorkDir"
Write-Host "[postrun-nyc] running nyc report inside container"

$cmd = "cd '$WorkDir' && if ! command -v nyc >/dev/null 2>&1; then echo 'nyc_not_found'; exit 12; fi && if [ ! -d .nyc_output ]; then echo '.nyc_output_not_found'; exit 13; fi && nyc report --reporter=html && test -f '$ReportPath/index.html'"
docker exec $ContainerName sh -lc $cmd

Write-Host "[postrun-nyc] copying $ContainerName`:$ReportPath -> $localDir"
docker cp "$ContainerName`:$ReportPath" $localDir

$index = Join-Path $localDir "coverage/index.html"
if (-not (Test-Path $index)) {
  $index = Join-Path $localDir "index.html"
}

Write-Host "[postrun-nyc] done: $index"
