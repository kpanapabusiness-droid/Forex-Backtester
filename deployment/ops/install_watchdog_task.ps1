<#
.SYNOPSIS
  Install the Arc 10 watchdog as a Windows Scheduled Task that fires
  every 30 seconds.

.DESCRIPTION
  The watchdog itself is watchdog.ps1; this script registers it with
  Task Scheduler. Run as administrator.
#>

param(
  [Parameter(Mandatory=$true)][string]$SidecarRoot,
  [string]$WatchdogScript = (Join-Path $PSScriptRoot "watchdog.ps1"),
  [string]$TaskName = "Arc10SidecarWatchdog",
  [int]$IntervalSec = 30
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $WatchdogScript)) {
  throw "watchdog.ps1 not found at $WatchdogScript"
}

# Compose the action.
$action = New-ScheduledTaskAction `
  -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$WatchdogScript`" -SidecarRoot `"$SidecarRoot`""

# 30-second trigger via a repeating-once-every-30s Daily start.
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date)
$trigger.Repetition = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Seconds $IntervalSec) -RepetitionDuration (New-TimeSpan -Days 9999) | Select-Object -ExpandProperty Repetition

$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -RunLevel Highest
$settings  = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask -TaskName $TaskName `
  -Action $action `
  -Trigger $trigger `
  -Principal $principal `
  -Settings $settings `
  -Description "Arc 10 sidecar watchdog — restart on stale heartbeat"

Write-Host "Task $TaskName installed; fires every $IntervalSec seconds."
