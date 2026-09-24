param(
  [int]$Port = 6972,
  [string]$OutDir = "analysis\webrled_code_coverage\istanbul_listener",
  [string]$Python = "python",
  [string]$LogFile = "",
  [switch]$PrintOnly
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$script = Join-Path $repo "scripts\istanbul_coverage_server.py"
$resolvedOut = Join-Path $repo $OutDir
$cmd = @($Python, $script, "--port", [string]$Port, "--out-dir", $resolvedOut)
if ($LogFile) {
  $resolvedLog = Join-Path $repo $LogFile
  $cmd += @("--log-file", $resolvedLog)
}

Write-Host "[istanbul-listener] command=$($cmd -join ' ')"
Write-Host "[istanbul-listener] endpoint=http://localhost:$Port/coverage"
Write-Host "[istanbul-listener] output=$resolvedOut"
if ($LogFile) {
  Write-Host "[istanbul-listener] log=$resolvedLog"
}

if ($PrintOnly) {
  exit 0
}

if ($LogFile) {
  & $Python $script --port $Port --out-dir $resolvedOut --log-file $resolvedLog
} else {
  & $Python $script --port $Port --out-dir $resolvedOut
}
