# Incident Response

> **Purpose:** What to do when something goes wrong.
> **Organized by:** symptom → cause → fix.

## Quick triage

| You see this | Severity | Go to |
|---|---|---|
| Service stopped | Medium | §1 |
| Heartbeat stale > 6h | Medium | §2 |
| Anchor probe fails | High | §3 |
| Repeated tracebacks in logs | Medium | §4 |
| Phantom trades on broker | High | §5 |
| Lost signals (envelope but no trade) | High | §6 |
| Single bad trade (loss > expected) | Low | §7 |
| Account approaching DD halt | High | §8 |
| MT5 disconnected | Medium | §9 |
| News URL returns 401/403 | Low | §10 |

## §1 — Sidecar service stopped

**Symptom:** `Get-Service Arc10Sidecar*` shows `Stopped` or `Paused`.

**First check:** look at stderr log for the reason.

```powershell
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\logs\sidecar.stderr.log" -Tail 50
```

**Common causes:**
- Python script crashed with unhandled exception → see traceback in log
- NSSM auto-restart limit hit → check exit code in log
- Windows shut down the service → check Windows event log

**Fix:** restart the service per `03_restart_procedures.md`. If it won't start, run manually with `--quick-test` to see the error in real time.

## §2 — Heartbeat stale > 6 hours

**Symptom:** Heartbeat `last_heartbeat_utc` is more than ~6 hours old (one full H4 + significant buffer).

**Possible causes:**
- Sidecar hung but service shows running (Python deadlock — rare but possible)
- MT5 disconnected for extended period, sidecar still cycling but writing partial state
- VPS clock skew (timezone confusion)

**Diagnosis:**

```powershell
# Is sidecar actually doing anything?
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\logs\sidecar.stderr.log" -Tail 30

# Is the process actually alive?
Get-Process python | Select-Object Id, StartTime, CPU
```

If process exists but log hasn't been written in hours → sidecar is hung. Watchdog should have restarted by now (15000s threshold). If watchdog also didn't fire, check:

```powershell
Get-ScheduledTask Arc10WatchDog_5ers | Get-ScheduledTaskInfo
# Look at LastRunTime, LastTaskResult
```

**Fix:** manually restart sidecar service. If recurrent, the threshold (StaleSec) may need adjustment or sidecar has a bug.

## §3 — Anchor probe fails on boot

**Symptom:** Sidecar log shows:
```
ERROR ... H4 alignment probe failed: bars not on expected anchor convention
```

**This is a HIGH severity issue.** Convention mismatch = silent wrong signals.

**Possible causes:**
- Broker changed bar anchor convention (very rare, but possible)
- Sidecar pointed at wrong MT5 (operator error in `--mt5-path`)
- Wrong `winning_config.yaml` for this broker

**Fix:**

1. **Stop the sidecar immediately** so it doesn't fire wrong signals:
   ```powershell
   & "C:\Tools\nssm.exe" stop Arc10Sidecar5ers
   ```

2. Verify which MT5 the sidecar is talking to:
   ```powershell
   & "C:\Tools\nssm.exe" get Arc10Sidecar5ers AppParameters
   ```
   Confirm `--mt5-path` points to the expected broker's terminal64.exe.

3. Verify the config matches the convention:
   ```powershell
   Get-Content "C:\Forex-Backtester\configs\l_arc_10_v3.0.2_utc_rerun\winning_config.yaml" | Select-String -Pattern "boundary_convention"
   # Should match expected convention for this broker (utc for 5ers, 5ers_eet for FundedNext)
   ```

4. Run anchor probe manually:
   ```powershell
   cd C:\Forex-Backtester
   # Manually invoke probe to see which convention the broker's data actually matches
   python scripts/fundednext_anchor_check.py  # or equivalent
   ```

5. If broker convention has actually changed, **stop and investigate before trading more**. Strategy needs re-validation under new convention.

## §4 — Repeated tracebacks in logs

**Symptom:** Multiple `Traceback` lines in `sidecar.stderr.log`.

**Common patterns:**

### MT5 connection lost mid-cycle

```
ConnectionError: terminal not initialized
```

**Fix:** restart MT5 (often it auto-recovers; if not, manual restart). Sidecar will resume on next cycle.

### Symbol fetch failure

```
copy_rates_from_pos returned None for GBPNZD
```

**Possible causes:** symbol temporarily unavailable, broker maintenance window, symbol removed from broker offering.

**Fix:** check if symbol is still in Market Watch. If pair was removed, the strategy needs re-validation on the reduced symbol set (high impact — flag immediately).

### News calendar fetch failure

```
HTTPError: 404 fetching ForexFactory calendar
```

**Possible cause:** ForexFactory changed URL or temporarily down.

**Fix:** verify URL still works in browser. If permanently changed, update `News_Calendar_URL` in EA inputs on both MT5s. Until fixed, EA falls back to "no news data" mode (allows all signals through; safer than rejecting).

### Memory leak

```
MemoryError: ...
```

**Fix:** restart sidecar. If recurrent, restart whole VPS. If still recurrent, this is a real bug — log details and investigate in code.

## §5 — Phantom trades on broker

**Symptom:** Broker history shows trade with the correct magic number, but no `entry` event in `trade_log.csv`.

**This is HIGH severity.** Implies either operator intervention or an EA bug.

**Possible causes:**
- Manual trade placed in MT5 with same magic number (you accidentally clicked Buy?)
- EA placed trade but failed to write log (rare — file permissions issue?)
- Position opened during EA restart, log entry lost

**Fix:**

1. **Don't close the position manually** unless it's clearly wrong (could mess up state).
2. Investigate: was it placed at a real signal time? Is the position size correct (matches Risk_Per_Trade)?
3. If trade matches a recent envelope in `signals_processed/`, the log write failed — log it manually for record-keeping.
4. If trade matches no signal envelope, it's manual or anomalous. Decide whether to close (probably yes — system did not authorize).

## §6 — Lost signals

**Symptom:** Envelope in `signals_processed/` (or `signals_failed/`) but no corresponding broker trade.

**Possible causes:**
- Envelope rejected by EA (check `signals_failed/`)
- EA rejected for staleness, hash mismatch, news, or DD halt
- Order placement failed (broker rejected lot size, spread too wide, etc.)

**Diagnosis:**

```powershell
# Find the envelope
ls "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\signals_failed\" -ErrorAction SilentlyContinue
ls "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\signals_processed\" -ErrorAction SilentlyContinue

# Check EA log in MT5 Experts tab — look for events around the signal timestamp
# Should show why the envelope was rejected (news_discard, hash_mismatch, stale, etc.)
```

**Fix:** root cause depends on rejection reason. Common cases:
- Stale envelope → sidecar cycle ran late, signal timestamp too old. Acceptable miss; review sidecar performance.
- Hash mismatch → operator error (wrong config). Update EA input.
- News discard → expected behavior; check news filter calibration.
- DD halt → expected behavior; account approaching limits.

## §7 — Single bad trade

**Symptom:** A single trade lost more than expected (e.g. -2R instead of -1R).

**Possible causes:**
- Slippage on SL fill (broker filled at worse price than SL level)
- Gap-through-SL (price gapped over the SL during news or market close)
- Multi-trade simultaneous loss (correlation event)

**Fix:**
- For single bad fill: log it, monitor for pattern. Single events happen.
- For gap-through: this is the unmodeled tail risk. Acceptable.
- For correlation: this is normal — Arc 10 doesn't manage correlation; expects DD halt to catch large losses.

If pattern (multiple unexpected -2R+ losses in a week): investigate broker fill quality, consider escalating to broker support.

## §8 — Account approaching DD halt

**Symptom:** Daily DD nearing 3.5% or total DD nearing 7%.

**This is HIGH severity.** Approach to halt means a stretch of losing trades.

**What the system does automatically:** at 3.5% daily (or 7% total), no new entries. At 4.5% (or 8%), force close all.

**What you should NOT do:**
- Don't close open positions manually unless you're absolutely sure they shouldn't continue
- Don't restart EA expecting to "reset" anything
- Don't increase risk to "make it back"

**What you SHOULD do:**
- Verify the system is operating correctly (check logs, no bugs)
- Note which folds/conditions historically had similar drawdowns (look at WFO data)
- Wait. The system halts entries and waits for daily DD to reset (UTC or EET midnight depending on broker).

**If the system actually halts (4.5% or 8% close-all):**
- Trading stops automatically
- Re-enabling trading requires verification that the system is healthy
- Consider reducing risk for the next week (manual reduction of Risk_Per_Trade)

## §9 — MT5 disconnected

**Symptom:** MT5 bottom-right shows red icon, "No connection" text, or stale prices.

**Possible causes:**
- Broker server maintenance
- Broker connection issue
- VPS network problem
- MT5 internal hang

**Diagnosis:** check if internet connectivity works on VPS:

```powershell
Test-NetConnection -ComputerName 8.8.8.8 -Port 53
Test-NetConnection -ComputerName -Port 443 # to broker host (find in MT5 properties)
```

**Fix:**
- Wait 5-10 minutes (usually auto-reconnects)
- If still disconnected: File → Login (re-trigger connection)
- If still disconnected: restart MT5
- If still disconnected: investigate VPS network or contact broker

## §10 — News URL returns 401/403/404

**Symptom:** Sidecar log shows news fetch errors.

**Possible cause:** ForexFactory changed URL or blocked the user-agent.

**Fix:**
- Check the URL in a browser: https://nfs.faireconomy.media/ff_calendar_thisweek.xml
- If URL changed: update `News_Calendar_URL` in EA inputs on both MT5s
- If access blocked: investigate user-agent or find alternative news source

**While broken:** the EA's news filter defaults to "no news data" → allows all signals. This is the safer default than "all signals blocked", but it means trades fire through news events. Monitor for unexpected losses during news windows.

## When to escalate vs. when to wait

**Wait it out:**
- Single bad trade
- Daily DD up to 2%
- One symbol fetch failure
- Brief broker disconnect (<10 min)
- One night of weird behavior

**Investigate same day:**
- Repeated tracebacks
- Heartbeat stale 6+ hours despite watchdog
- Phantom trade
- Total DD up to 6%

**Escalate immediately (consider halting):**
- Anchor probe fails (convention drift)
- Multiple phantom trades
- Total DD 7%+ (system halts entries automatically)
- Total DD 8%+ (system force-closes; investigate before re-enabling)
- Trade behavior significantly diverges from backtest expectations across 2+ weeks
