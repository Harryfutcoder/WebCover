param(
  [switch]$DryRun,
  [switch]$SkipSplittyPie,
  [switch]$SkipPetClinic
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Invoke-Or-Print {
  param([string]$Label, [scriptblock]$Command, [string]$Preview)
  Write-Host ""
  Write-Host "[coverage-setup] $Label"
  if ($DryRun) {
    Write-Host $Preview
  } else {
    & $Command
  }
}

if (-not $SkipSplittyPie) {
  Invoke-Or-Print `
    -Label "start SplittyPie Istanbul listener on 6972" `
    -Preview "powershell -ExecutionPolicy Bypass -File .\scripts\start_istanbul_coverage_listener_background.ps1 -Port 6972 -OutDir analysis\webrled_code_coverage\istanbul_listener_splittypie -Name splittypie_6972 -Force" `
    -Command {
      powershell -ExecutionPolicy Bypass -File .\scripts\start_istanbul_coverage_listener_background.ps1 `
        -Port 6972 `
        -OutDir analysis\webrled_code_coverage\istanbul_listener_splittypie `
        -Name splittypie_6972 `
        -Force
    }

  Invoke-Or-Print `
    -Label "start instrumented SplittyPie app on 4005" `
    -Preview "powershell -ExecutionPolicy Bypass -File .\scripts\start_webrled_instrumented_first_benchmark.ps1 -Site splittypie -CoverageBaseUrl http://localhost:6972" `
    -Command {
      powershell -ExecutionPolicy Bypass -File .\scripts\start_webrled_instrumented_first_benchmark.ps1 `
        -Site splittypie `
        -CoverageBaseUrl http://localhost:6972
    }
}

if (-not $SkipPetClinic) {
  Invoke-Or-Print `
    -Label "start PetClinic Istanbul listener on 6973" `
    -Preview "powershell -ExecutionPolicy Bypass -File .\scripts\start_istanbul_coverage_listener_background.ps1 -Port 6973 -OutDir analysis\webrled_code_coverage\istanbul_listener_petclinic -Name petclinic_6973 -Force" `
    -Command {
      powershell -ExecutionPolicy Bypass -File .\scripts\start_istanbul_coverage_listener_background.ps1 `
        -Port 6973 `
        -OutDir analysis\webrled_code_coverage\istanbul_listener_petclinic `
        -Name petclinic_6973 `
        -Force
    }

  Invoke-Or-Print `
    -Label "start instrumented PetClinic app on 4002" `
    -Preview "powershell -ExecutionPolicy Bypass -File .\scripts\start_webrled_instrumented_first_benchmark.ps1 -Site petclinic -CoverageBaseUrl http://localhost:6973" `
    -Command {
      powershell -ExecutionPolicy Bypass -File .\scripts\start_webrled_instrumented_first_benchmark.ps1 `
        -Site petclinic `
        -CoverageBaseUrl http://localhost:6973
    }
}

if (-not $DryRun) {
  powershell -ExecutionPolicy Bypass -File .\scripts\check_webrled_code_coverage_setup.ps1 -Detailed
}
