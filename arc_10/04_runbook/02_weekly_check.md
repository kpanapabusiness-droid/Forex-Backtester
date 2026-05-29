# Weekly Check (Sundays, ~30 min)

> **Purpose:** Reconcile trades, look for drift vs backtest, restart VPS during weekend market closure.
> **When:** Sunday during market closure (before Sunday 22:00 UTC reopen).
> **Tool:** Manual for now. Will be automated via `scripts/weekly_reconciliation.py` (TBD — build after 4 weeks of live data).

## What the weekly check covers

1. **System health refresh** (5 min) — confirm 7 days of clean operation
2. **Trade reconciliation** (10 min) — does every broker trade match a logged signal? Does every logged signal match a broker trade?
3. **Performance review** (5 min) — what's the running P&L, are trades behaving like backtest expectations?
4. **VPS restart** (10 min, optional) — full VPS reboot during market closure to flush memory leaks, apply Windows updates

## Step 1 — System health refresh

RDP into VPS. Run all checks from `01_daily_health_check.md`. Plus:

```powershell
# Look at NSSM service uptime
& "C:\Tools\nssm.exe" status Arc10Sidecar5ers
& "C:\Tools\nssm.exe" status Arc10SidecarFundedNext

# How many restart events in the past week?
Get-EventLog -LogName Application -Source nssm -After (Get-Date).AddDays(-7) -ErrorAction SilentlyContinue | Format-Table TimeGenerated, EntryType, Message -AutoSize

# Disk space (should be plenty)
Get-PSDrive C | Select-Object Used, Free
```

Expected:
- Both services running with high uptime (no recent restarts in normal week)
- Disk space > 20GB free

## Step 2 — Trade reconciliation

For each broker, pull last 7 days of trades from MT5 + compare to logs.

### Manual approach (until script exists)

In each MT5:
1. Account History (Ctrl+Shift+H) → "Last Week" (or custom range)
2. Right-click → Save as Report → save as CSV

Now you have broker's trade history.

Cross-check against EA's trade log:

```powershell
# 5ers EA trade log
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\trade_log.csv" | Where-Object { $_ -match "entry|exit" } | Select-Object -Last 50

# FundedNext EA trade log
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext\trade_log.csv" | Where-Object { $_ -match "entry|exit" } | Select-Object -Last 50
```

Cross-reference:
- Every `entry` event in trade_log → should have matching broker trade with same ticket
- Every closed broker trade → should have matching `exit` event with reason

**Phantom trades** (broker trade with no entry log) → manual intervention or bug
**Lost signals** (entry log with no broker trade) → execution failure

### What you're looking for

- [ ] Trade count matches between EA log and broker
- [ ] All exit reasons are expected (`trail_stop`, `initial_sl_hit`, `time_exit` — NOT `external_close`)
- [ ] No DD halt events in `trade_log.csv`
- [ ] No `news_discard` or `news_delay` events that suggest a calendar issue

### Signal envelopes accounting

```powershell
# How many signals were generated this week?
Get-ChildItem "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\signals_processed\" | Where-Object { $_.LastWriteTime -gt (Get-Date).AddDays(-7) } | Measure-Object | Select-Object Count
Get-ChildItem "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\signals_failed\" | Where-Object { $_.LastWriteTime -gt (Get-Date).AddDays(-7) } | Measure-Object | Select-Object Count

# Same for FundedNext
```

Expected: ~5-15 signals/week per broker. `signals_failed` count should be 0 in normal operation. Any failed signal needs investigation.

## Step 3 — Performance review

Each MT5 → "Account" tab or Account History → Reports section.

| Metric | Expected (1 week) | Red flag |
|---|---|---|
| Trades | 5-15 per broker | 0 trades (signal generation issue) |
| Win rate | ~35-45% (long-only, asymmetric R) | <25% sustained |
| Net P&L | varies (single week high variance) | -3% in single week (rare, but possible) |
| Daily DD reached | usually <2% peak intraday | approaching 4% (close to halt) |
| Total DD | first 2 weeks: 0-3% | >7% (close to halt) |

**Note:** Single weeks have very high variance. Don't over-react. Pattern-match across 4+ weeks.

## Step 4 — VPS restart (optional, recommended weekly)

Sunday during market closure is the ideal time to restart the VPS. Reasons:
- Flush any Windows memory leaks accumulated over the week
- Apply pending Windows updates that need a reboot
- Clear any zombie processes
- Test that services + EAs auto-restart cleanly

### Pre-restart check

```powershell
# Confirm no positions are open right now (markets are closed, but double-check)
# In each MT5: Trade tab → should be empty

# Confirm market is closed
# Friday 22:00 UTC through Sunday 22:00 UTC = forex closed
```

If positions are open during weekend — yes they CAN be open across weekend, but verify no open positions before restart to be safe.

### Restart

```powershell
# Triggering a reboot
shutdown /r /t 0
```

VPS reboots. RDP session ends. Reconnect after ~3-5 minutes.

### Post-restart verification

After VPS comes back:

```powershell
# Both services should be auto-started
Get-Service Arc10Sidecar5ers, Arc10SidecarFundedNext | Format-Table Name, Status

# Both MT5s should be running (they were configured to auto-start at boot — if not, manually launch)
Get-Process terminal64 | Select-Object Id, Path

# Watchdogs should be Ready
Get-ScheduledTask -TaskName "Arc10WatchDog_*" | Format-Table TaskName, State
```

**If MT5s did NOT auto-start:** they're GUI apps, not services. Add them to Windows startup folder:

```powershell
# Shortcut paths
$startup = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup"

$shell = New-Object -ComObject WScript.Shell

# 5ers shortcut
$shortcut5 = $shell.CreateShortcut("$startup\5ers MT5.lnk")
$shortcut5.TargetPath = "C:\Program Files\Five Percent Online MetaTrader 5\terminal64.exe"
$shortcut5.WorkingDirectory = "C:\Program Files\Five Percent Online MetaTrader 5"
$shortcut5.Save()

# FundedNext shortcut
$shortcutFN = $shell.CreateShortcut("$startup\FundedNext MT5.lnk")
$shortcutFN.TargetPath = "C:\Program Files\FundedNext MT5 Terminal\terminal64.exe"
$shortcutFN.WorkingDirectory = "C:\Program Files\FundedNext MT5 Terminal"
$shortcutFN.Save()
```

Now they'll auto-start on every VPS boot.

## Step 5 — Log review for anything weird

```powershell
# Check for errors over the past week
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\logs\sidecar.stderr.log" | Where-Object { $_ -match "ERROR|Traceback|failed" } | Select-Object -Last 30
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext\logs\sidecar.stderr.log" | Where-Object { $_ -match "ERROR|Traceback|failed" } | Select-Object -Last 30
```

Notable issues to look for:
- Repeated symbol fetch failures
- "MT5 not initialized" errors
- "anchor probe failed" — convention drift?
- Python tracebacks

If you see anything weird, save the log lines and investigate (see `04_incident_response.md`).

## What the automated weekly script will do (future)

`scripts/weekly_reconciliation.py` (TBD):

1. Connect to both MT5s via Python `MetaTrader5` lib
2. Pull broker trade history for past 7 days
3. Read trade_log.csv from both sidecar roots
4. Read signals_processed/ and signals_failed/ contents
5. Cross-reference:
   - Every broker trade → matching `entry` event in trade_log
   - Every signal envelope → either matching `entry` event OR documented `rejected` reason
   - Every exit event → matching broker trade close
6. Compute live R-multiples vs backtest expectations
7. Detect anomalies (phantom trades, lost signals, drift)
8. Output: `results/weekly_reports/YYYY_MM_DD_weekly_report.md`

Build this AFTER 4 weeks of live data, so edge cases are known.

## Why weekend market closure is the right time

- Markets closed Friday 22:00 UTC through Sunday 22:00 UTC
- No new signals fire during this period (sidecar sleeps)
- No open positions can change P&L
- Safe to restart, update, modify configuration
- Watchdog will tolerate up to 4h 10min of staleness without action

If you must do maintenance during market hours, see `03_restart_procedures.md` for safe sequencing.
