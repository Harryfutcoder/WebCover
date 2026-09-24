param(
  [string]$Sites = "agilefant,gadael,timeoff,4gaboards,realworld,petclinic,splittypie,nextcloud,github,odoo",
  [int]$TimeoutSec = 3,
  [switch]$SkipDocker,
  [switch]$Detailed
)

$ErrorActionPreference = "Continue"

function Get-DockerAccessStatus {
  if ($SkipDocker) {
    return [pscustomobject]@{
      status = "SKIPPED"
      detail = "Docker checks skipped by -SkipDocker"
    }
  }
  try {
    $output = & docker ps --format "{{.Names}}" 2>&1
    if ($LASTEXITCODE -eq 0) {
      return [pscustomobject]@{
        status = "OK"
        detail = "Docker CLI/API reachable"
      }
    }
    return [pscustomobject]@{
      status = "ERR"
      detail = (((($output | Out-String).Trim()) -split "`r?`n")[0] -replace "^docker(\.exe)?\s*:\s*", "")
    }
  } catch {
    return [pscustomobject]@{
      status = "ERR"
      detail = $_.Exception.Message
    }
  }
}

$script:DockerAccess = Get-DockerAccessStatus

function Invoke-DockerSafe {
  param([string[]]$DockerArgs)
  if ($script:DockerAccess.status -ne "OK") { return $null }
  try {
    $output = & docker @DockerArgs 2>$null
    if ($LASTEXITCODE -ne 0) { return $null }
    return $output
  } catch {
    return $null
  }
}

function Test-DockerImage {
  param([string]$Image)
  $out = Invoke-DockerSafe -DockerArgs @("image", "inspect", $Image, "--format", "{{.Id}}")
  return [bool]$out
}

function Get-DockerContainerStatus {
  param([string]$ContainerName)
  $rows = Invoke-DockerSafe -DockerArgs @("ps", "-a", "--format", "{{.Names}}|{{.Status}}")
  if (-not $rows) { return "" }
  foreach ($row in $rows) {
    $parts = [string]$row -split "\|", 2
    if ($parts.Count -ge 2 -and $parts[0] -eq $ContainerName) {
      return $parts[1]
    }
  }
  return ""
}

function Test-CoverageEndpoint {
  param([string]$Site, [string]$BaseUrl)
  $coverageUrl = $BaseUrl.TrimEnd("/") + "/coverage"
  try {
    $response = Invoke-WebRequest -UseBasicParsing -Uri $coverageUrl -TimeoutSec $TimeoutSec
    $sample = ""
    if ($null -ne $response.Content) {
      $sample = $response.Content.Substring(0, [Math]::Min(90, $response.Content.Length)) -replace "`r|`n", " "
    }
    [pscustomobject]@{
      site = $Site
      mode = "live"
      endpoint = $coverageUrl
      status = "OK"
      detail = "$($response.StatusCode) $sample"
    }
  } catch {
    [pscustomobject]@{
      site = $Site
      mode = "live"
      endpoint = $coverageUrl
      status = "ERR"
      detail = $_.Exception.Message
    }
  }
}

$liveDefaults = @{
  "agilefant" = "http://localhost:6969"
  "gadael" = "http://localhost:6970"
  "timeoff" = "http://localhost:6971"
}

$postRunSites = @{
  "4gaboards" = @{
    ContainerName = "4gaboards-app-1"
    Detail = "post-run nyc report; expected report server/coverage/index.html"
  }
  "realworld" = @{
    ContainerName = "realworld-app-1"
    Detail = "post-run nyc report; expected report coverage/index.html"
  }
}

$externalInstrumentationSites = @{
  "petclinic" = @{
    Image = "dockercontainervm/petclinic:latest"
    ContainerName = "webrled-petclinic-instrumented"
    AppUrl = "http://localhost:4002"
    DefaultCoverageUrl = "http://localhost:6973"
    Detail = "requires WebRLED instrumented image plus Istanbul listener"
  }
  "splittypie" = @{
    Image = "dockercontainervm/splittypie:latest"
    ContainerName = "webrled-splittypie-instrumented"
    AppUrl = "http://localhost:4005"
    DefaultCoverageUrl = "http://localhost:6972"
    Detail = "requires WebRLED instrumented image plus Istanbul listener"
  }
}

$unavailableSites = @{
  "github" = "remote third-party site; server-side code coverage unavailable"
  "odoo" = "remote/runbot-style deployment; server-side code coverage unavailable"
  "nextcloud" = "current local deployment is not instrumented"
}

$rows = New-Object System.Collections.Generic.List[object]
foreach ($rawSite in $Sites.Split(",")) {
  $site = $rawSite.Trim().ToLowerInvariant()
  if (-not $site) { continue }

  if ($liveDefaults.ContainsKey($site)) {
    $rows.Add((Test-CoverageEndpoint -Site $site -BaseUrl $liveDefaults[$site]))
    continue
  }

  $envName = "WEBTEST_SITE_{0}_COVERAGE_URL" -f ($site.ToUpperInvariant().Replace("-", "_"))
  $envValue = [Environment]::GetEnvironmentVariable($envName)
  if ($externalInstrumentationSites.ContainsKey($site)) {
    $info = $externalInstrumentationSites[$site]
    $image = $info.Image
    $containerStatus = ""
    $imageReady = $false
    if ($script:DockerAccess.status -eq "OK") {
      $containerStatus = Get-DockerContainerStatus -ContainerName $info.ContainerName
      $imageReady = Test-DockerImage -Image $image
    }
    $defaultCoverageUrl = $info.DefaultCoverageUrl
    if ($envValue) {
      $probe = Test-CoverageEndpoint -Site $site -BaseUrl $envValue
      if ($probe.status -eq "OK" -and $script:DockerAccess.status -ne "OK") {
        $probe.status = "READY_LIVE_UNVERIFIED_DOCKER"
      } elseif ($probe.status -eq "OK" -and $containerStatus -notlike "Up*") {
        $probe.status = "NEEDS_APP"
      } elseif ($probe.status -eq "OK") {
        $probe.status = "READY_LIVE"
      }
      $probe.detail = "$($probe.detail); image=$image image_ready=$imageReady container=$($info.ContainerName) status=$containerStatus app_url=$($info.AppUrl); docker_status=$($script:DockerAccess.status); docker_detail=$($script:DockerAccess.detail)"
      $rows.Add($probe)
    } elseif ($script:DockerAccess.status -ne "OK") {
      $probe = Test-CoverageEndpoint -Site $site -BaseUrl $defaultCoverageUrl
      $probe.mode = "external_instrumentation_required"
      if ($probe.status -eq "OK") {
        $probe.status = "READY_LIVE_UNVERIFIED_DOCKER"
        $probe.detail = "$($probe.detail); default listener OK but Docker state not inspected; docker_status=$($script:DockerAccess.status); docker_detail=$($script:DockerAccess.detail); set $envName=$defaultCoverageUrl before running"
      } else {
        $probe.status = $(if ($script:DockerAccess.status -eq "SKIPPED") { "SKIP_DOCKER_CHECK" } else { "DOCKER_UNAVAILABLE" })
        $probe.detail = "$($info.Detail); cannot inspect image/container; docker_status=$($script:DockerAccess.status); docker_detail=$($script:DockerAccess.detail); expected image=$image container=$($info.ContainerName) app_url=$($info.AppUrl); start listener $defaultCoverageUrl; listener_probe=$($probe.detail)"
      }
      $rows.Add($probe)
    } elseif ($imageReady) {
      $probe = Test-CoverageEndpoint -Site $site -BaseUrl $defaultCoverageUrl
      $probe.mode = "external_instrumentation_required"
      if ($probe.status -eq "OK" -and $containerStatus -like "Up*") {
        $probe.status = "READY_LIVE"
        $probe.detail = "$($probe.detail); image=$image ready; set $envName=$defaultCoverageUrl or use wrapper defaults; container=$($info.ContainerName) status=$containerStatus app_url=$($info.AppUrl)"
      } elseif ($probe.status -eq "OK") {
        $probe.status = "NEEDS_APP"
        $probe.detail = "$($probe.detail); image=$image ready; listener OK; start container=$($info.ContainerName) app_url=$($info.AppUrl)"
      } else {
        $probe.status = "NEEDS_LISTENER"
        $probe.detail = "$($info.Detail); image=$image ready; set $envName=$defaultCoverageUrl; container=$($info.ContainerName) status=$containerStatus app_url=$($info.AppUrl); listener_probe=$($probe.detail)"
      }
      $rows.Add($probe)
    } else {
      $rows.Add([pscustomobject]@{
        site = $site
        mode = "external_instrumentation_required"
        endpoint = "-"
        status = "MISSING_IMAGE"
        detail = "$($info.Detail); missing image=$image; pull/load image, start listener $defaultCoverageUrl, then set $envName"
      })
    }
    continue
  }

  if ($postRunSites.ContainsKey($site)) {
    $info = $postRunSites[$site]
    $containerStatus = ""
    $ready = $false
    if ($script:DockerAccess.status -eq "OK") {
      $containerStatus = Get-DockerContainerStatus -ContainerName $info.ContainerName
      $ready = [bool]$containerStatus
    }
    $rows.Add([pscustomobject]@{
      site = $site
      mode = "postrun_nyc"
      endpoint = "-"
      status = $(if ($ready) { "READY_POSTRUN" } elseif ($script:DockerAccess.status -eq "OK") { "MISSING_CONTAINER" } elseif ($script:DockerAccess.status -eq "SKIPPED") { "SKIP_DOCKER_CHECK" } else { "DOCKER_UNAVAILABLE" })
      detail = "$($info.Detail); container=$($info.ContainerName) status=$containerStatus; docker_status=$($script:DockerAccess.status); docker_detail=$($script:DockerAccess.detail)"
    })
    continue
  }

  if ($unavailableSites.ContainsKey($site)) {
    $rows.Add([pscustomobject]@{
      site = $site
      mode = "unavailable"
      endpoint = "-"
      status = "SKIP"
      detail = $unavailableSites[$site]
    })
    continue
  }

  $rows.Add([pscustomobject]@{
    site = $site
    mode = "unknown"
    endpoint = "-"
    status = "TODO"
    detail = "No coverage mode configured"
  })
}

if ($Detailed) {
  $rows | Format-List
} else {
  $rows | Format-Table -AutoSize
}
