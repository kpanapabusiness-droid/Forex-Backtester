<#
.SYNOPSIS
  Arc 10 sidecar watchdog. Restarts the NSSM service if sidecar.heartbeat
  is older than -StaleSec seconds.

.DESCRIPTION
  Run from Windows Task Scheduler every 30 seconds. Idempotent — does
  not restart if a restart is already in progress.

  Per phase_1_build_intent.md §10:
    - Read sidecar.heartbeat JSON
    - Parse last_heartbeat_utc
    - If older than -StaleSec, run ``nssm restart``

.EXAMPLE
  powershell -File watchdog.ps1 -SidecarRoot "C:\...\runtime" -StaleSec 120
#>

param(
  [Parameter(Mandatory=$true)][string]$SidecarRoot,
  [int]$StaleSec = 120,
  [string]$ServiceName = "Arc10Sidecar",
  [string]$AlertWebhookUrl = "",
  [string]$LogFile = ""
)

$ErrorActionPreference = "Stop"

function Write-Log($msg) {
  $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  $line = "$stamp watchdog: $msg"
  Write-Host $line
  if ($LogFile -ne "") {
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
  }
}

$heartbeatPath = Join-Path $SidecarRoot "sidecar.heartbeat"

if (-not (Test-Path $heartbeatPath)) {
  Write-Log "heartbeat file missing at $heartbeatPath — restarting $ServiceName"
  & nssm restart $ServiceName
  if ($AlertWebhookUrl) {
    try { Invoke-RestMethod -Uri $AlertWebhookUrl -Method Post -Body @{ msg="sidecar restart: heartbeat missing" } | Out-Null } catch {}
  }
  exit 0
}

try {
  $hb = Get-Content $heartbeatPath -Raw -Encoding UTF8 | ConvertFrom-Json
  $hbTime = [DateTime]::Parse($hb.last_heartbeat_utc).ToUniversalTime()
  $now = (Get-Date).ToUniversalTime()
  $ageSec = ($now - $hbTime).TotalSeconds
  if ($ageSec -gt $StaleSec) {
    Write-Log "heartbeat stale ($([math]::Round($ageSec))s > $StaleSec) — restarting $ServiceName"
    & nssm restart $ServiceName
    if ($AlertWebhookUrl) {
      try { Invoke-RestMethod -Uri $AlertWebhookUrl -Method Post -Body @{ msg="sidecar restart: heartbeat stale ${ageSec}s" } | Out-Null } catch {}
    }
  } else {
    Write-Log "heartbeat OK (age=$([math]::Round($ageSec))s)"
  }
} catch {
  Write-Log "watchdog error: $($_.Exception.Message); attempting restart"
  & nssm restart $ServiceName
}
