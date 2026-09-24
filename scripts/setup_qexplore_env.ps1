param(
  [string]$EnvName = "qexplore37",
  [string]$PythonVersion = "3.7",
  [string]$CondaExe = "conda",
  [switch]$PrintOnly,
  [switch]$ForceRecreate,
  [switch]$FullUpstreamDeps,
  [switch]$InstallBrowser,
  [string]$Wheelhouse = "",
  [switch]$NoIndex,
  [string]$RepoRoot = ""
)

$ErrorActionPreference = "Stop"
if ($RepoRoot) {
  $repo = (Resolve-Path -LiteralPath $RepoRoot).Path
} else {
  $repo = Split-Path -Parent $PSScriptRoot
  if (-not (Test-Path -LiteralPath (Join-Path $repo "external\QExplore"))) {
    $repo = (Resolve-Path -LiteralPath ".").Path
  }
}
$reqName = if ($FullUpstreamDeps) { "requirements.local.txt" } else { "requirements.runtime.txt" }
$req = Join-Path $repo "external\QExplore\$reqName"

if (-not (Test-Path -LiteralPath $req)) {
  throw "Missing QExplore requirements file: $req"
}

function Test-CondaEnvExists {
  param([string]$Name)
  try {
    $rows = & $CondaExe env list 2>$null
    if ($LASTEXITCODE -ne 0) { return $false }
    foreach ($row in $rows) {
      $text = [string]$row
      if ($text -match "^\s*$([regex]::Escape($Name))\s" -or $text -match "[\\/]$([regex]::Escape($Name))\s*$") {
        return $true
      }
    }
  } catch {
    return $false
  }
  return $false
}

$envExists = Test-CondaEnvExists -Name $EnvName
$commands = New-Object System.Collections.Generic.List[string]
$pipInstall = "$CondaExe run -n $EnvName python -m pip install"
if ($Wheelhouse) {
  $wheelhousePath = (Resolve-Path -LiteralPath $Wheelhouse).Path
  if ($NoIndex) {
    $pipInstall = "$pipInstall --no-index"
  }
  $pipInstall = "$pipInstall --find-links `"$wheelhousePath`""
}

if ($ForceRecreate) {
  $commands.Add("$CondaExe env remove -y -n $EnvName")
  $commands.Add("$CondaExe create -y -n $EnvName python=$PythonVersion pip")
} elseif (-not $envExists) {
  $commands.Add("$CondaExe create -y -n $EnvName python=$PythonVersion pip")
} else {
  Write-Host "[qexplore-env] Existing conda env detected: $EnvName; skipping create."
}
$commands.Add("$pipInstall -r `"$req`"")
if ($InstallBrowser) {
  $commands.Add("$CondaExe install -y -n $EnvName -c conda-forge firefox geckodriver")
}
$commands.Add("$CondaExe run -n $EnvName python `"$repo\scripts\check_qexplore_imports.py`"")

Write-Host "[qexplore-env] QExplore's recorded environment uses Python 3.7.4 and Selenium 3.141."
Write-Host "[qexplore-env] Optional legacy env prepared. The current local wrapper defaults to Chrome/chromedriver; set QEXPLORE_BROWSER=firefox only if you intentionally use Firefox."
Write-Host "[qexplore-env] requirements=$req"
Write-Host "[qexplore-env] FullUpstreamDeps=$([bool]$FullUpstreamDeps) InstallBrowser=$([bool]$InstallBrowser) Wheelhouse=$Wheelhouse NoIndex=$([bool]$NoIndex)"
Write-Host "[qexplore-env] Planned commands:"
$commands | ForEach-Object { Write-Host "  $_" }

if ($PrintOnly) {
  exit 0
}

foreach ($cmd in $commands) {
  Write-Host "[qexplore-env] RUN $cmd"
  & cmd.exe /d /c $cmd
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit=$LASTEXITCODE : $cmd"
  }
}

Write-Host "[qexplore-env] OK. Use this Python for QExplore:"
Write-Host "  $CondaExe run -n $EnvName python"
