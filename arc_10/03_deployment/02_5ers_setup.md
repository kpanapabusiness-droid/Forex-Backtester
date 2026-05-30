# 5ers Deployment

> **Purpose:** Document the exact configuration of the 5ers UTC deployment.
> **Status:** Live on demo $10k account, will continue post-Challenge.
> **Convention:** UTC.

## Account

- **Broker:** Five Percent Online (5ers)
- **Server:** `FivePercentOnline-Real`
- **Account size:** $10k (demo for now; live after Challenge if applicable)
- **Hard limits:** 10% max DD, 5% daily DD

## Files & paths (on VPS)

| Item | Path |
|---|---|
| MT5 install | `C:\Program Files\Five Percent Online MetaTrader 5\` |
| MT5 terminal64.exe | `C:\Program Files\Five Percent Online MetaTrader 5\terminal64.exe` |
| Terminal data folder | `C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\10CE948A1DFC9A8C27E56E827008EBD4\` |
| EA experts folder | `...\10CE948A1DFC9A8C27E56E827008EBD4\MQL5\Experts\Arc10_Sidecar\` |
| Sidecar root | `C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\` |
| Sidecar config | `C:\Forex-Backtester\configs\l_arc_10_v3.0.2_utc_rerun\winning_config.yaml` |

## Service & supervision

- **NSSM service name:** `Arc10Sidecar5ers`
- **Service start type:** Automatic (auto-starts on VPS reboot)
- **Watchdog task:** `Arc10WatchDog_5ers` (Task Scheduler, every 5 minutes)
- **Logs:** `C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\logs\sidecar.stderr.log`

## Sidecar command-line (exact)

```
C:\Python311\python.exe -m deployment.sidecar
  --winning-config "C:\Forex-Backtester\configs\l_arc_10_v3.0.2_utc_rerun\winning_config.yaml"
  --sidecar-root "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers"
  --mt5-path "C:\Program Files\Five Percent Online MetaTrader 5\terminal64.exe"
  --log-level INFO
```

(NSSM stores these; AppDirectory is `C:\Forex-Backtester`.)

## EA Input parameters

Attached to **ONE chart** on the 5ers MT5 (any pair; EURUSD H4 is the convention).

| Input | Value |
|---|---|
| Risk_Per_Trade | `0.0040` |
| **Initial_Equity_Floor** | _operator-set; the broker's **static starting balance** (per-account, not committed here)_ |
| Total_DD_Halt_Pct | `0.07` |
| Total_DD_CloseAll_Pct | `0.08` |
| Daily_DD_Halt_Pct | `0.035` |
| Daily_DD_CloseAll_Pct | `0.045` |
| Time_Exit_Bars | `240` |
| SL_ATR_Multiplier_Expected | `3.5` |
| Sidecar_Inbox_Dir | `Arc10_5ers\signals_out` |
| Sidecar_Processed_Dir | `Arc10_5ers\signals_processed` |
| Sidecar_Failed_Dir | `Arc10_5ers\signals_failed` |
| Sidecar_Heartbeat_Path | `Arc10_5ers\sidecar.heartbeat` |
| Ea_Heartbeat_Path | `Arc10_5ers\ea.heartbeat` |
| Ea_Positions_Path | `Arc10_5ers\ea_positions.json` |
| Trade_Log_Path | `Arc10_5ers\trade_log.csv` |
| Sidecar_Heartbeat_Max_Age_Sec | `600` |
| **Expected_Config_Hash** | `4467366b9537871fe9019af45cf26f54e042358f211ff58774497b00c840821e` |
| News_Calendar_URL | `https://nfs.faireconomy.media/ff_calendar_thisweek.xml` |
| Enable_News_Filter | `true` |
| News_Window_Sec | `120` |
| News_Delay_Buffer_Sec | `5` |
| News_Delay_Max_Sec | `3600` |
| News_Refresh_Sec | `14400` |
| **Magic_Number** | `1010202601` |
| Signal_Poll_Min_Interval_Sec | `5` |

Plus on the Common tab: ✅ "Allow Algo Trading".

> **`Initial_Equity_Floor` (OPEN-001, `b386287`).** The total-DD floor is now a solely operator-set input — there is **no live-equity capture**. Set it to the broker's static starting balance and confirm the journal shows `equity init: floor=<value> source=input`. If it is left unset / `< 5000`, the EA **fails loud**: refuses to trade, fires `Alert()`, and journals `FLOOR_FAIL`. The value survives terminal restart via the MT5 saved profile; a broker scale-up is handled by editing this input and reattaching (manual). See [`../05_history/07_open_issue_dd_restart_rebaselining.md`](../05_history/07_open_issue_dd_restart_rebaselining.md).

## News filter URL whitelisted in MT5

Tools → Options → Expert Advisors:
- ✅ Allow WebRequest for listed URL
- URL: `https://nfs.faireconomy.media`

## Risk per trade rationale

0.40% locked per cost sweep (`02_validation/05_cost_sweep.md`):
- 5ers carries swap, which is dominant cost vector under UTC
- At r_base 0.5%, central case (1.5× spread, swap-ON) lands at 10.47% worst-fold DD — breaches 10% hard limit
- Linear scaling to 0.40% brings central worst-fold DD to 8.38% — 1.6pp margin

Expected live (after haircuts): worst-fold DD ~9.5%, worst-fold ratio ~1.8, holdout ROI ~27%.

## Connection details (operational)

- **Server timezone:** EET (+2 winter / +3 summer) — chart timestamps are EET
- **Bar anchor convention:** UTC (despite EET clock) — verified by panel-diff
- **Sidecar wakes at:** 00, 04, 08, 12, 16, 20 UTC (+10s buffer)
- **Daily DD reset:** UTC midnight (00:00 UTC)
- **Weekend hold:** allowed (no specific 5ers prohibition for funded — verify if rules change)

## Per-trade economics

| Item | Value |
|---|---|
| Risk per trade ($) | $40 (0.40% of $10k) |
| SL distance | 3.5 × ATR_at_signal |
| Commission | $4 per lot round-turn |
| Swap | Real (charged) |
| Average expected R per trade | ~0.5-0.7R (cost sweep central case) |
| Expected trades per year | ~225 (3,152 trades / 14 years = 225) |
| Expected annual ROI (cost-adjusted) | ~27% (after haircuts) |

## Verification checklist (post-deploy)

After deploying or restarting:

```powershell
# Service status
& "C:\Tools\nssm.exe" status Arc10Sidecar5ers
# → SERVICE_RUNNING

# Watchdog status
Get-ScheduledTask Arc10WatchDog_5ers | Select TaskName, State
# → State: Ready

# Heartbeat freshness
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\sidecar.heartbeat" -Raw
# → last_heartbeat_utc within last 4 hours
# → pairs_processed_last_loop lists all 28 pairs

# Latest logs
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_5ers\logs\sidecar.stderr.log" -Tail 20
# → "sidecar starting — config_hash=4467366b... restart_count=N pairs=28"
# → No tracebacks, no "fetch failed" lines
```

In MT5 (visual check):
- Bottom-right shows green connection icon, balance/equity numbers
- Chart with EA attached shows smiley face icon (top-right)
- Experts tab in Toolbox shows `[ARC10] EA init magic=1010202601 sidecar_inbox=Common\Files\Arc10_5ers\signals_out`
- Once first H4 cycle completes: `[ARC10] sidecar-heartbeat stale=false`

## What changes if 5ers becomes the primary deployment

Currently 5ers is the secondary/fallback. If FundedNext path breaks and 5ers becomes primary:

1. Increase capital if 5ers offers it
2. Risk stays at 0.40% (locked by cost sweep)
3. No code or config changes
4. EA continues to operate the same way against the same MT5 install
