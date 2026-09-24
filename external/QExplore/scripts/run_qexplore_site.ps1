param(
  [Parameter(Mandatory = $true)]
  [string]$Site,

  [string]$Seed = "seedqexplore1",
  [int]$ActivityTime = 3600,
  [int]$Depth = 100,
  [double]$ActionWait = 0.5,
  [string]$Config = "",
  [string]$Python = "python",
  [switch]$Coverage,
  [switch]$DryRun,
  [switch]$Force
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$sourceDir = Join-Path $root "Qexplore"
$repo = Split-Path -Parent (Split-Path -Parent $root)
$preflight = Join-Path $repo "scripts\check_qexplore_imports.py"
$seleniumFallback = Join-Path $env:USERPROFILE ".conda\envs\webtest-python\Lib\site-packages"
$chromedriverDefault = "C:\Users\SUST\Desktop\webTest\webTest\chromedriver.exe"
if (-not $Config) {
  $Config = Join-Path $root "configs\sites.local.json"
}

if (-not (Test-Path -LiteralPath $Config)) {
  throw "QExplore site config not found: $Config"
}

$configData = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
$siteConfig = $configData.sites | Where-Object { $_.site -eq $Site } | Select-Object -First 1
if (-not $siteConfig) {
  $known = ($configData.sites | ForEach-Object { $_.site }) -join ","
  throw "Unknown site '$Site'. Known sites: $known"
}

$siteKey = $Site.ToLowerInvariant()
$envKey = $siteKey.ToUpperInvariant().Replace("-", "_")
$effectiveUrl = [string]$siteConfig.url
$effectiveBaseUrl = [string]$siteConfig.baseurl
$coverageUrl = ""
$coverageKind = ""

if ($Coverage) {
  $coverageDefaults = @{
    "agilefant" = @{ Url = "http://localhost:6969"; Kind = "jacoco" }
    "gadael" = @{ Url = "http://localhost:6970"; Kind = "jacoco" }
    "timeoff" = @{ Url = "http://localhost:6971"; Kind = "jacoco" }
    "splittypie" = @{ Url = "http://localhost:6972"; Kind = "nyc"; EntryUrl = "http://localhost:4005"; BaseUrl = "localhost:4005" }
    "petclinic" = @{ Url = "http://localhost:6973"; Kind = "nyc"; EntryUrl = "http://localhost:4002"; BaseUrl = "localhost:4002" }
  }
  $coverageUrl = [Environment]::GetEnvironmentVariable("WEBTEST_SITE_${envKey}_COVERAGE_URL")
  if (-not $coverageUrl -and $coverageDefaults.ContainsKey($siteKey)) {
    $coverageUrl = $coverageDefaults[$siteKey].Url
  }
  if ($coverageDefaults.ContainsKey($siteKey)) {
    $coverageKind = $coverageDefaults[$siteKey].Kind
    if ($coverageDefaults[$siteKey].ContainsKey("EntryUrl")) {
      $entryOverride = [Environment]::GetEnvironmentVariable("WEBTEST_SITE_${envKey}_ENTRY_URL")
      $domainOverride = [Environment]::GetEnvironmentVariable("WEBTEST_SITE_${envKey}_DOMAINS")
      $effectiveUrl = if ($entryOverride) { $entryOverride } else { $coverageDefaults[$siteKey].EntryUrl }
      $effectiveBaseUrl = if ($domainOverride) { $domainOverride } else { $coverageDefaults[$siteKey].BaseUrl }
    }
  }
  if (-not $coverageUrl) {
    Write-Host "[qexplore] WARNING coverage requested but no live coverage endpoint configured for site=$Site; running without live coverage hook."
  }
}

$runDir = Join-Path $root ("runs\{0}\{1}" -f $Site, $Seed)
if ((Test-Path -LiteralPath $runDir) -and -not $Force) {
  throw "Run directory already exists: $runDir. Use -Force to overwrite the wrapper working files."
}

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$graphViewDest = Join-Path $runDir "graphView"
if (Test-Path -LiteralPath $graphViewDest) {
  $resolvedRunDir = [System.IO.Path]::GetFullPath($runDir)
  $resolvedGraphViewDest = [System.IO.Path]::GetFullPath($graphViewDest)
  if (-not $resolvedGraphViewDest.StartsWith($resolvedRunDir, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to clean graphView outside runDir: $resolvedGraphViewDest"
  }
  Remove-Item -LiteralPath $graphViewDest -Recurse -Force
}
Copy-Item -Path (Join-Path $sourceDir "graphView") -Destination $graphViewDest -Recurse -Force

$metadata = [ordered]@{
  site = $Site
  seed = $Seed
  url = $effectiveUrl
  baseurl = $effectiveBaseUrl
  login_urls = $siteConfig.login_urls
  activity_time = $ActivityTime
  depth = $Depth
  action_wait = $ActionWait
  coverage = [bool]$Coverage
  coverage_url = $coverageUrl
  coverage_kind = $coverageKind
  created_at = (Get-Date -Format o)
  notes = $siteConfig.notes
}
$metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runDir "run_metadata.json") -Encoding UTF8
$startMarker = [ordered]@{
  site = $Site
  seed = $Seed
  started_at = (Get-Date -Format o)
  status = "running"
}
$startMarker | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runDir "qexplore_start.json") -Encoding UTF8

$loginArg = [string]$siteConfig.login_urls
if ([string]::IsNullOrWhiteSpace($loginArg)) {
  # QExplore requires --login_urls because the original main() calls split().
  $loginArg = "__none__"
}

$argsList = @(
  (Join-Path $sourceDir "Qexplore.py"),
  "--ignore-gooey",
  "--url", [string]$effectiveUrl,
  "--login_urls", $loginArg,
  "--baseurl", [string]$effectiveBaseUrl,
  "--action_wait", [string]$ActionWait,
  "--timebound",
  "--activity_time", [string]$ActivityTime,
  "--depth", [string]$Depth
)

if (-not [string]::IsNullOrWhiteSpace([string]$siteConfig.username)) {
  $argsList += @("--username", [string]$siteConfig.username)
}
if (-not [string]::IsNullOrWhiteSpace([string]$siteConfig.password)) {
  $argsList += @("--password", [string]$siteConfig.password)
}

$commandPreview = @($Python) + $argsList
Set-Content -LiteralPath (Join-Path $runDir "command.txt") -Encoding UTF8 -Value ($commandPreview -join " ")

Write-Host "[qexplore] site=$Site seed=$Seed seconds=$ActivityTime depth=$Depth"
Write-Host "[qexplore] url=$effectiveUrl baseurl=$effectiveBaseUrl"
if ($Coverage) {
  Write-Host "[qexplore] coverage=$coverageUrl kind=$coverageKind"
}
Write-Host "[qexplore] runDir=$runDir"
Write-Host "[qexplore] command=$($commandPreview -join ' ')"

if ($DryRun) {
  Write-Host "[qexplore] dry-run only; not starting browser."
  exit 0
}

$previousPythonPath = $env:PYTHONPATH
$previousBrowser = $env:QEXPLORE_BROWSER
$previousChromeDriver = $env:QEXPLORE_CHROMEDRIVER
$previousCoverageUrl = $env:QEXPLORE_COVERAGE_URL
$previousCoverageKind = $env:QEXPLORE_COVERAGE_KIND
$previousPythonWarnings = $env:PYTHONWARNINGS
$previousRepoRoot = $env:WEBTEST_REPO_ROOT
$previousCanonical = $env:WEBTEST_EXTERNAL_CANONICAL
$previousCanonicalSite = $env:WEBTEST_EXTERNAL_CANONICAL_SITE
$previousCanonicalAlgorithm = $env:WEBTEST_EXTERNAL_CANONICAL_ALGORITHM
$previousCanonicalSeed = $env:WEBTEST_EXTERNAL_CANONICAL_SEED
$previousCanonicalProfile = $env:WEBTEST_EXTERNAL_CANONICAL_PROFILE
$pathParts = @($sourceDir, $repo)
if ((Test-Path -LiteralPath $seleniumFallback) -and $Python -eq "python") {
  $pathParts += $seleniumFallback
}
if ($previousPythonPath) {
  $pathParts += $previousPythonPath
}
$env:PYTHONPATH = ($pathParts -join ";")
if (-not $env:QEXPLORE_BROWSER) {
  $env:QEXPLORE_BROWSER = "chrome"
}
if (-not $env:QEXPLORE_CHROMEDRIVER -and (Test-Path -LiteralPath $chromedriverDefault)) {
  $env:QEXPLORE_CHROMEDRIVER = $chromedriverDefault
}
if (-not $env:PYTHONWARNINGS) {
  # Paramiko may emit a Blowfish deprecation warning through stderr during
  # imports in the base Anaconda environment. It is unrelated to QExplore.
  $env:PYTHONWARNINGS = "ignore:Blowfish has been deprecated"
}

try {
  if (Test-Path -LiteralPath $preflight) {
    Write-Host "[qexplore] preflight=$preflight"
    & $Python $preflight
    if ($LASTEXITCODE -ne 0) {
      throw "QExplore runtime preflight failed for Python=$Python. Install missing dependencies or provide browser driver paths."
    }
  }
  if ($Coverage -and $coverageUrl) {
    $env:QEXPLORE_COVERAGE_URL = $coverageUrl
    $env:QEXPLORE_COVERAGE_KIND = $coverageKind
  }
  $env:WEBTEST_REPO_ROOT = $repo
  $env:WEBTEST_EXTERNAL_CANONICAL = "1"
  $env:WEBTEST_EXTERNAL_CANONICAL_SITE = $Site
  $env:WEBTEST_EXTERNAL_CANONICAL_ALGORITHM = "qexplore"
  $env:WEBTEST_EXTERNAL_CANONICAL_SEED = $Seed
  $env:WEBTEST_EXTERNAL_CANONICAL_PROFILE = "qexplore-1agent"
  Push-Location -LiteralPath $runDir
  $oldErrorActionPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    & $Python @argsList 2>&1 | Tee-Object -FilePath "qexplore.log"
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $oldErrorActionPreference
    Pop-Location
  }
  $finishMarker = [ordered]@{
    site = $Site
    seed = $Seed
    finished_at = (Get-Date -Format o)
    status = if ($exitCode -eq 0) { "ok" } else { "failed" }
    exit_code = $exitCode
  }
  $finishMarker | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runDir "qexplore_finish.json") -Encoding UTF8
  exit $exitCode
} finally {
  $env:PYTHONPATH = $previousPythonPath
  $env:QEXPLORE_BROWSER = $previousBrowser
  $env:QEXPLORE_CHROMEDRIVER = $previousChromeDriver
  $env:QEXPLORE_COVERAGE_URL = $previousCoverageUrl
  $env:QEXPLORE_COVERAGE_KIND = $previousCoverageKind
  $env:PYTHONWARNINGS = $previousPythonWarnings
  $env:WEBTEST_REPO_ROOT = $previousRepoRoot
  $env:WEBTEST_EXTERNAL_CANONICAL = $previousCanonical
  $env:WEBTEST_EXTERNAL_CANONICAL_SITE = $previousCanonicalSite
  $env:WEBTEST_EXTERNAL_CANONICAL_ALGORITHM = $previousCanonicalAlgorithm
  $env:WEBTEST_EXTERNAL_CANONICAL_SEED = $previousCanonicalSeed
  $env:WEBTEST_EXTERNAL_CANONICAL_PROFILE = $previousCanonicalProfile
}
