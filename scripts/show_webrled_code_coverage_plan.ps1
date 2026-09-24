param(
  [string]$Seeds = "seedcodecov1_20260622,seedcodecov2_20260622,seedcodecov3_20260622",
  [int]$TimeLimit = 3600,
  [int]$ForceHeadful = 0,
  [switch]$IncludeRealWorld
)

$ErrorActionPreference = "Stop"

function Write-Section {
  param([string]$Title)
  Write-Host ""
  Write-Host ("# " + $Title)
}

function Write-Block {
  param([string[]]$Lines)
  Write-Host '```powershell'
  foreach ($line in $Lines) {
    Write-Host $line
  }
  Write-Host '```'
}

$seedList = $Seeds
$repo = "C:\Users\SUST\artifacts"

Write-Host "# WebRLED Code Coverage Plan"
Write-Host ""
Write-Host "Stable coverage set: 4gaboards, agilefant, gadael, timeoff, petclinic, splittypie"
Write-Host "RealWorld is optional because the current config points to https://demo.realworld.show and no local realworld container is running."
Write-Host "Seeds: $seedList"
Write-Host "TimeLimit: $TimeLimit"

Write-Section "Preflight"
Write-Block @(
  "cd $repo",
  "powershell -ExecutionPolicy Bypass -File .\scripts\check_webrled_code_coverage_setup.ps1",
  "docker images --format `"{{.Repository}}:{{.Tag}}`" | Select-String `"dockercontainervm/(splittypie|petclinic)`""
)

Write-Section "Window 1: Live Coverage Sites"
Write-Host "These already expose /coverage on the current containers."
Write-Block @(
  "cd $repo",
  "powershell -ExecutionPolicy Bypass -File .\scripts\start_webrled_official_batch.ps1 ``",
  "  -Sites `"agilefant,gadael,timeoff`" ``",
  "  -Seeds `"$seedList`" ``",
  "  -TimeLimit $TimeLimit ``",
  "  -ForceHeadful $ForceHeadful ``",
  "  -GlobalBrowserCleanup 0 ``",
  "  -Coverage"
)

Write-Section "Window 2: 4gaboards Post-Run NYC"
Write-Host "Run WebRLED normally, then collect nyc coverage from the app container after the run finishes."
Write-Block @(
  "cd $repo",
  "powershell -ExecutionPolicy Bypass -File .\scripts\start_webrled_official_batch.ps1 ``",
  "  -Sites `"4gaboards`" ``",
  "  -Seeds `"$seedList`" ``",
  "  -TimeLimit $TimeLimit ``",
  "  -ForceHeadful $ForceHeadful ``",
  "  -GlobalBrowserCleanup 0",
  "",
  "powershell -ExecutionPolicy Bypass -File .\scripts\collect_webrled_postrun_nyc_coverage.ps1 -Site 4gaboards"
)

Write-Section "Setup: SplittyPie Listener"
Write-Host "Use a dedicated background listener so SplittyPie coverage does not merge with PetClinic."
Write-Block @(
  "cd $repo",
  "powershell -ExecutionPolicy Bypass -File .\scripts\start_istanbul_coverage_listener_background.ps1 ``",
  "  -Port 6972 ``",
  "  -OutDir analysis\webrled_code_coverage\istanbul_listener_splittypie ``",
  "  -Name splittypie_6972 ``",
  "  -Force"
)

Write-Section "Setup: SplittyPie Instrumented App"
Write-Host "Requires dockercontainervm/splittypie:latest. If missing, run: docker pull dockercontainervm/splittypie:latest"
Write-Block @(
  "cd $repo",
  "powershell -ExecutionPolicy Bypass -File .\scripts\start_webrled_instrumented_first_benchmark.ps1 ``",
  "  -Site splittypie ``",
  "  -CoverageBaseUrl http://localhost:6972"
)

Write-Section "Window 3: SplittyPie WebRLED Coverage"
Write-Block @(
  "cd $repo",
  "`$env:WEBTEST_SITE_SPLITTYPIE_COVERAGE_URL = `"http://localhost:6972`"",
  "`$env:WEBTEST_SITE_SPLITTYPIE_ENTRY_URL = `"http://localhost:4005`"",
  "`$env:WEBTEST_SITE_SPLITTYPIE_DOMAINS = `"http://localhost:4005`"",
  "powershell -ExecutionPolicy Bypass -File .\scripts\start_webrled_official_batch.ps1 ``",
  "  -Sites `"splittypie`" ``",
  "  -Seeds `"$seedList`" ``",
  "  -TimeLimit $TimeLimit ``",
  "  -ForceHeadful $ForceHeadful ``",
  "  -GlobalBrowserCleanup 0 ``",
  "  -Coverage"
)

Write-Section "Setup: PetClinic Listener"
Write-Host "Use a separate background listener port/output directory from SplittyPie."
Write-Block @(
  "cd $repo",
  "powershell -ExecutionPolicy Bypass -File .\scripts\start_istanbul_coverage_listener_background.ps1 ``",
  "  -Port 6973 ``",
  "  -OutDir analysis\webrled_code_coverage\istanbul_listener_petclinic ``",
  "  -Name petclinic_6973 ``",
  "  -Force"
)

Write-Section "Setup: PetClinic Instrumented App"
Write-Host "Requires dockercontainervm/petclinic:latest. If missing, run: docker pull dockercontainervm/petclinic:latest"
Write-Block @(
  "cd $repo",
  "powershell -ExecutionPolicy Bypass -File .\scripts\start_webrled_instrumented_first_benchmark.ps1 ``",
  "  -Site petclinic ``",
  "  -CoverageBaseUrl http://localhost:6973"
)

Write-Section "Window 4: PetClinic WebRLED Coverage"
Write-Block @(
  "cd $repo",
  "`$env:WEBTEST_SITE_PETCLINIC_COVERAGE_URL = `"http://localhost:6973`"",
  "`$env:WEBTEST_SITE_PETCLINIC_ENTRY_URL = `"http://localhost:4002`"",
  "`$env:WEBTEST_SITE_PETCLINIC_DOMAINS = `"http://localhost:4002`"",
  "powershell -ExecutionPolicy Bypass -File .\scripts\start_webrled_official_batch.ps1 ``",
  "  -Sites `"petclinic`" ``",
  "  -Seeds `"$seedList`" ``",
  "  -TimeLimit $TimeLimit ``",
  "  -ForceHeadful $ForceHeadful ``",
  "  -GlobalBrowserCleanup 0 ``",
  "  -Coverage"
)

if ($IncludeRealWorld) {
  Write-Section "Optional: RealWorld"
  Write-Host "Only use this after deploying a local instrumented RealWorld container and overriding the app URL/domain."
  Write-Block @(
    "cd $repo",
    "# TODO: start local RealWorld container and update URL/domain.",
    "powershell -ExecutionPolicy Bypass -File .\scripts\start_webrled_official_batch.ps1 ``",
    "  -Sites `"realworld`" ``",
    "  -Seeds `"$seedList`" ``",
    "  -TimeLimit $TimeLimit ``",
    "  -ForceHeadful $ForceHeadful ``",
    "  -GlobalBrowserCleanup 0",
    "",
    "powershell -ExecutionPolicy Bypass -File .\scripts\collect_webrled_postrun_nyc_coverage.ps1 -Site realworld -ContainerName realworld-app-1"
  )
}
