param(
  [string]$WebRLEDPython = "",
  [string]$QExplorePython = "",
  [string]$WebRLEDSite = "github",
  [switch]$DetailedCoverage,
  [switch]$SkipDocker
)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot

if (-not $WebRLEDPython) {
  if ($env:WEBTEST_WEBRLED_PYTHON) {
    $WebRLEDPython = $env:WEBTEST_WEBRLED_PYTHON
  } else {
    $candidate = Join-Path $env:USERPROFILE ".conda\envs\webtest-python\python.exe"
    if (Test-Path $candidate) {
      $WebRLEDPython = $candidate
    } else {
      $WebRLEDPython = "python"
    }
  }
}

if (-not $QExplorePython) {
  if ($env:QEXPLORE_PYTHON) {
    $QExplorePython = $env:QEXPLORE_PYTHON
  } else {
    # The local QExplore runner is validated against the base Python plus the
    # repo-local compatibility shims and Chrome driver. A partial qexplore37 env
    # may exist on this machine, but it is not the default runtime.
    $QExplorePython = "python"
  }
}

Write-Host "== WebRLED official =="
Write-Host "python=$WebRLEDPython"
& $WebRLEDPython (Join-Path $repo "run_webrled_official.py") `
  --site $WebRLEDSite `
  --session prereq_check `
  --seed prereq_check `
  --check-only
$webrledExit = $LASTEXITCODE
if ($webrledExit -eq 0) {
  Write-Host "[OK] WebRLED official prerequisites passed"
} else {
  Write-Host "[WARN] WebRLED official prerequisites failed with exit=$webrledExit"
}

Write-Host ""
Write-Host "== QExplore =="
Write-Host "python=$QExplorePython"
& $QExplorePython (Join-Path $repo "scripts\check_qexplore_imports.py")
$qexploreExit = $LASTEXITCODE
if ($qexploreExit -eq 0) {
  Write-Host "[OK] QExplore core prerequisites passed"
} else {
  Write-Host "[WARN] QExplore core prerequisites failed with exit=$qexploreExit"
  Write-Host "      Current default is base Python with repo-local QExplore shims and Chrome."
  Write-Host "      Override with -QExplorePython or QEXPLORE_PYTHON only if you intentionally use another env."
}

Write-Host ""
Write-Host "== Code coverage endpoints =="
$coverageArgs = @(
  "-ExecutionPolicy", "Bypass",
  "-File", (Join-Path $repo "scripts\check_webrled_code_coverage_setup.ps1")
)
if ($DetailedCoverage) { $coverageArgs += "-Detailed" }
if ($SkipDocker) { $coverageArgs += "-SkipDocker" }
& powershell @coverageArgs

if ($webrledExit -ne 0 -or $qexploreExit -ne 0) {
  exit 1
}
exit 0
