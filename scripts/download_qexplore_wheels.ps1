param(
  [string]$RepoRoot = "",
  [string]$Python = "python",
  [string]$Wheelhouse = "analysis\external_baselines\qexplore_wheelhouse",
  [switch]$FullUpstreamDeps
)

$ErrorActionPreference = "Stop"

if ($RepoRoot) {
  $repo = (Resolve-Path -LiteralPath $RepoRoot).Path
} else {
  $repo = Split-Path -Parent $PSScriptRoot
}

$reqName = if ($FullUpstreamDeps) { "requirements.local.txt" } else { "requirements.runtime.txt" }
$req = Join-Path $repo "external\QExplore\$reqName"
if (-not (Test-Path -LiteralPath $req)) {
  throw "Missing QExplore requirements file: $req"
}

if ([System.IO.Path]::IsPathRooted($Wheelhouse)) {
  $wheelhousePath = $Wheelhouse
} else {
  $wheelhousePath = Join-Path $repo $Wheelhouse
}
New-Item -ItemType Directory -Force -Path $wheelhousePath | Out-Null

Write-Host "[qexplore-wheels] requirements=$req"
Write-Host "[qexplore-wheels] wheelhouse=$wheelhousePath"
Write-Host "[qexplore-wheels] python=$Python"

& $Python -m pip download -r $req -d $wheelhousePath
if ($LASTEXITCODE -ne 0) {
  throw "pip download failed with exit=$LASTEXITCODE"
}

Write-Host "[qexplore-wheels] OK. Offline install command:"
Write-Host "powershell -ExecutionPolicy Bypass -File .\scripts\setup_qexplore_env.ps1 -RepoRoot $repo -Wheelhouse `"$wheelhousePath`" -NoIndex -InstallBrowser"
