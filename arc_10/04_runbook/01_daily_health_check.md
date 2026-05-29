# Daily Health Check (30 seconds)

> **Purpose:** Quick check that everything is operating normally.
> **Frequency:** Once a day, or any time you log into the VPS.
> **What it does NOT do:** Check if trades are profitable (that's weekly review).

## The check

RDP into the VPS. Open Administrator PowerShell. Run:

```powershell
# Services running?
Get-Service Arc10Sidecar5ers, Arc10SidecarFundedNext | Format-Table Name, Status, StartType -AutoSize

# Both MT5s running?
Get-Process terminal64 | Select-Object Id, Path

# Watchdogs ready?
Get-ScheduledTask -TaskName "Arc10WatchDog_*" | Format-Table TaskName, State

# Heartbeats fresh? (Should be < 4-5 hours old)
Write-Host "--- 5ers heartbeat ---"
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\sidecar.heartbeat" -Raw -ErrorAction SilentlyContinue
Write-Host "--- FundedNext heartbeat ---"
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext\sidecar.heartbeat" -Raw -ErrorAction SilentlyContinue

# Recent logs (last 10 lines each)
Write-Host "--- 5ers latest log ---"
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\logs\sidecar.stderr.log" -Tail 10 -ErrorAction SilentlyContinue
Write-Host "--- FundedNext latest log ---"
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext\logs\sidecar.stderr.log" -Tail 10 -ErrorAction SilentlyContinue
```

## What "healthy" looks like

| Check | Expected |
|---|---|
| Services | Both `Running`, `Automatic` |
| MT5 processes | 2 visible, one from each install path |
| Watchdog tasks | Both `Ready` |
| Heartbeats | `last_heartbeat_utc` within last 4 hours (UTC) or 5 hours (EET, accounting for DST) |
| Logs | No tracebacks; latest line shows successful cycle completion or "sleeping until next H4" |

## What's NOT a problem

- `restart_count` > 1 in heartbeat: services have restarted during deployment (normal early on)
- Heartbeat older than expected by minutes (broker disconnect, sidecar retrying)
- "fetch failed" lines for 1-2 specific pairs once in a while (occasional symbol freeze)

## What IS a problem

- Service status `Stopped` or `Paused` → restart needed
- One or both MT5 not running → restart MT5
- Heartbeat older than 5h on either sidecar with no recent log activity → investigate
- Repeated tracebacks → check error message, may need bug fix
- "anchor probe failed" → MT5 broker disconnected at boot or config mismatch
- Multiple consecutive cycles with NO pairs processed → MT5 lost connection

If anything looks bad, refer to `04_incident_response.md`.

## In MT5 (visual check, optional)

If you want to look at the MT5 windows:

- 5ers MT5: bottom-right shows balance/equity + green connection icon
- FundedNext MT5: same
- Each chart with EA attached shows smiley face icon (top-right)
- Experts tab at bottom shows recent `[ARC10] ...` log lines (no errors)
- No red "X" icons or warning popups

## What to do if everything looks good

Disconnect RDP and go about your day. The system runs itself.

## Trades vs health

Health check ≠ performance check. A healthy system might:
- Have made trades in the last 24h (normal)
- Have made no trades in the last 24h (also normal — Arc 10 averages ~2-3 trades/week per pair, or ~80/year × 28 pairs = ~225 trades total, so quiet days happen)
- Be in drawdown (some folds in backtest had multi-week drawdowns)

Don't react to short-term performance via daily check. Performance review is a weekly job (`02_weekly_check.md`).

## Bonus: equity over time

In each MT5: Account History tab → see balance/equity progression. Useful for noting if anything went weird, but don't make trading decisions from this.

## 30-second elevator version

If you only have 30 seconds:

```powershell
Get-Service Arc10Sidecar5ers, Arc10SidecarFundedNext | Format-Table Name, Status
Get-Process terminal64 | Measure-Object | Select-Object -ExpandProperty Count  # should be 2
```

If both services are `Running` and you see 2 terminal64 processes → close PowerShell and disconnect. You're good.
