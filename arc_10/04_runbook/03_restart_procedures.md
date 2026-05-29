# Restart Procedures

> **Purpose:** Safely restart individual components without affecting open trades.

## Restart hierarchy

From least invasive to most:

1. Restart ONE sidecar service (other broker unaffected)
2. Restart BOTH sidecar services
3. Restart ONE MT5 (and its sidecar)
4. Restart whole VPS

## What happens to open positions during a restart

**Broker holds positions and SL orders regardless of EA state.** When an EA or sidecar restarts:

- Open positions stay open at broker
- Initial SL stays in place at broker
- Trailing SL stays in place (last value before restart)
- TP1 partial-close logic resumes (EA reconstructs from broker state)
- Bar counter (`bar_ord`) reconstructs from entry time
- `peak_high_bid` reconstructs as max of entry and current — conservative (may re-anchor lower than pre-restart peak, but trail SL still ratchets correctly going forward)

Net: a restart is safe for open positions.

## Restart ONE sidecar service

Use when:
- Sidecar logs show repeated errors
- Heartbeat is stale and you suspect sidecar is hung
- After config changes (rare)

```powershell
# 5ers
& "C:\Tools\nssm.exe" restart Arc10Sidecar5ers
Start-Sleep -Seconds 10
& "C:\Tools\nssm.exe" status Arc10Sidecar5ers
# Should show SERVICE_RUNNING

Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\logs\sidecar.stderr.log" -Tail 10
# Should show "sidecar starting — config_hash=4467366b... restart_count=N"
```

For FundedNext substitute `Arc10SidecarFundedNext` and `Arc10_FundedNext`.

**What happens during this restart (~10-30 seconds):**
- Watchdog may also kick in if heartbeat goes stale during restart (harmless, double-restart)
- EA sees heartbeat stale temporarily, blocks new entries
- Existing positions continue to manage normally
- Once sidecar boots and cycles, heartbeat refreshes, EA resumes

## Restart BOTH sidecars

Use when:
- VPS-wide issue suspected (e.g. Windows update affecting Python or MT5 lib)
- After git pull that includes sidecar changes

```powershell
# Restart both at once
& "C:\Tools\nssm.exe" restart Arc10Sidecar5ers
& "C:\Tools\nssm.exe" restart Arc10SidecarFundedNext
Start-Sleep -Seconds 15

# Verify both running
Get-Service Arc10Sidecar5ers, Arc10SidecarFundedNext | Format-Table Name, Status
```

Same impact as single-sidecar restart, but doubled.

## Restart ONE MT5 terminal

Use when:
- MT5 broker disconnected and won't reconnect
- MT5 frozen / not responding
- After manual EA changes

**Important:** when MT5 restarts, its EA also restarts (EA is a chart-attached process). The sidecar continues running, but won't be able to fetch data while MT5 is down.

```powershell
# Stop sidecar first (to avoid spamming errors during MT5 restart)
& "C:\Tools\nssm.exe" stop Arc10Sidecar5ers

# Now restart 5ers MT5
Get-Process terminal64 | Where-Object { $_.Path -like "*Five Percent*" } | Stop-Process -Force

# Wait a few seconds
Start-Sleep -Seconds 5

# Launch MT5 fresh
Start-Process "C:\Program Files\Five Percent Online MetaTrader 5\terminal64.exe"
```

In the MT5 GUI:
1. If login is remembered, MT5 auto-logs in
2. If not, manually log in (File → Login)
3. Verify connection (green icon bottom-right)
4. Verify EA is still attached to chart (smiley face top-right) — if not, re-attach

Then restart the sidecar:

```powershell
& "C:\Tools\nssm.exe" start Arc10Sidecar5ers
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\logs\sidecar.stderr.log" -Tail 10
```

For FundedNext, substitute paths.

## Restart whole VPS

Use when:
- Weekly maintenance (preferred during weekend market closure)
- Windows updates require reboot
- VPS performance degraded

**Always restart during weekend market closure if possible.** Avoid restarting during live trading hours.

```powershell
# Pre-flight check
Get-Process terminal64 | Select-Object Id, Path
# Note both MT5s are running
# Optionally close them manually first (gracefully): File → Exit in each

# Trigger reboot
shutdown /r /t 0
```

RDP disconnects. Reconnect after ~3-5 minutes.

**Post-reboot verification:**

```powershell
# Services should auto-start
Get-Service Arc10Sidecar5ers, Arc10SidecarFundedNext | Format-Table Name, Status

# Watchdogs should be ready
Get-ScheduledTask -TaskName "Arc10WatchDog_*" | Format-Table TaskName, State

# MT5s should be running (if startup shortcuts were set up — see 02_weekly_check.md)
Get-Process terminal64 | Select-Object Id, Path
```

If MT5s didn't auto-start, launch them manually. If they keep not auto-starting, add to Windows startup folder.

## When NOT to restart

- Don't restart during live trading hours unless absolutely necessary
- Don't restart immediately after a trade enters (give the EA a minute to log the entry event)
- Don't restart if a trade is about to hit TP1 or SL (wait a few minutes for the event to log, then restart)

## What to do if a restart fails

### Sidecar service won't start

```powershell
# Check NSSM config
& "C:\Tools\nssm.exe" get Arc10Sidecar5ers Application
& "C:\Tools\nssm.exe" get Arc10Sidecar5ers AppParameters
& "C:\Tools\nssm.exe" get Arc10Sidecar5ers AppDirectory

# Check stderr for last error
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\logs\sidecar.stderr.log" -Tail 50

# Try running the command manually
cd C:\Forex-Backtester
python -m deployment.sidecar `
  --winning-config "configs\l_arc_10_v3.0.2_utc_rerun\winning_config.yaml" `
  --sidecar-root "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers" `
  --mt5-path "C:\Program Files\Five Percent Online MetaTrader 5\terminal64.exe" `
  --quick-test --log-level INFO
```

Common causes:
- MT5 not running → start MT5 first
- MT5 not logged in → log in
- Wrong --mt5-path → check exact path
- Config hash mismatch → check winning_config.yaml against stored hash

### MT5 won't launch

- Check if another terminal64 process is hanging: `Get-Process terminal64`
- Force-kill stuck processes: `Stop-Process -Name terminal64 -Force`
- Re-launch from start menu or shortcut

### VPS won't boot

- Use Contabo control panel → console access (VNC)
- Last resort: restore from snapshot (Contabo offers periodic snapshots)
- If catastrophic: redeploy from scratch using `04_vps_setup_guide.md`

## Confirming healthy post-restart state

```powershell
# All systems check
Write-Host "--- Services ---"
Get-Service Arc10Sidecar5ers, Arc10SidecarFundedNext | Format-Table Name, Status

Write-Host "`n--- MT5 Processes ---"
Get-Process terminal64 | Select-Object Id, Path

Write-Host "`n--- Watchdogs ---"
Get-ScheduledTask -TaskName "Arc10WatchDog_*" | Format-Table TaskName, State

Write-Host "`n--- 5ers heartbeat ---"
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\sidecar.heartbeat" -Raw

Write-Host "`n--- FundedNext heartbeat ---"
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext\sidecar.heartbeat" -Raw
```

If everything shows Running/Ready and heartbeats are recent, you're back to healthy state.
