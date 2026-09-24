param(
  [int]$Port = 6972,
  [string]$OutDir = "analysis\webrled_code_coverage\istanbul_listener",
  [string]$Python = "python",
  [string]$Name = "",
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$script = Join-Path $repo "scripts\istanbul_coverage_server.py"
$resolvedOut = if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $repo $OutDir }
$logDir = Join-Path $repo "analysis\webrled_code_coverage\listener_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Quote-Arg {
  param([string]$Value)
  '"' + ($Value -replace '"', '\"') + '"'
}

if (-not $Name) {
  $Name = "listener_$Port"
}

$pidFile = Join-Path $logDir "$Name.pid"
$outLog = Join-Path $logDir "$Name.out.log"
$errLog = Join-Path $logDir "$Name.err.log"
$serverLog = Join-Path $logDir "$Name.server.log"

if ((Test-Path -LiteralPath $pidFile) -and -not $Force) {
  $oldPid = (Get-Content -LiteralPath $pidFile -Raw).Trim()
  if ($oldPid) {
    $existing = Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue
    if ($existing) {
      Write-Host "[istanbul-listener-bg] already-running name=$Name pid=$oldPid endpoint=http://localhost:$Port/coverage"
      exit 0
    }
  }
}

$pythonCommand = Get-Command $Python -ErrorAction Stop
$pythonExe = $pythonCommand.Source
$argsList = @(
  $script,
  "--port", [string]$Port,
  "--out-dir", $resolvedOut,
  "--log-file", $serverLog
)

$psi = [System.Diagnostics.ProcessStartInfo]::new()
$psi.FileName = $pythonExe
$psi.Arguments = ($argsList | ForEach-Object { Quote-Arg ([string]$_) }) -join " "
$psi.WorkingDirectory = $repo
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$proc = [System.Diagnostics.Process]::new()
$proc.StartInfo = $psi
[void]$proc.Start()

$stdoutTask = $proc.StandardOutput.ReadToEndAsync()
$stderrTask = $proc.StandardError.ReadToEndAsync()
Start-Sleep -Milliseconds 300
if ($proc.HasExited) {
  Set-Content -LiteralPath $outLog -Encoding UTF8 -Value $stdoutTask.Result
  Set-Content -LiteralPath $errLog -Encoding UTF8 -Value $stderrTask.Result
  throw "Listener exited immediately with code=$($proc.ExitCode). See $outLog and $errLog"
}

Set-Content -LiteralPath $pidFile -Encoding ASCII -Value ([string]$proc.Id)
Write-Host "[istanbul-listener-bg] started name=$Name pid=$($proc.Id) endpoint=http://localhost:$Port/coverage out=$resolvedOut"
