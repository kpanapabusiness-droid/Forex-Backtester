# FundedNext Deployment

> **Purpose:** Document the exact configuration of the FundedNext EET deployment.
> **Status:** Live on demo $100k account, ready for $100k Challenge.
> **Convention:** EET (`5ers_eet`).

## Account

- **Broker:** FundedNext
- **Server:** FundedNext-Demo (will be different post-Challenge — check FundedNext panel for exact server name)
- **Account size:** $100k (demo) → $100k Challenge → $100k funded (post-Challenge pass)
- **Hard limits:** 10% max DD, 5% daily DD
- **Special add-on:** Swap-free enabled (zero swap cost on all positions, both Challenge and funded)
- **Weekend holds:** Allowed (confirmed by FundedNext)
- **News trading:** Allowed (confirmed; we keep news filter ON for consistency with 5ers)

## Files & paths (on VPS)

| Item | Path |
|---|---|
| MT5 install | `C:\Program Files\FundedNext MT5 Terminal\` |
| MT5 terminal64.exe | `C:\Program Files\FundedNext MT5 Terminal\terminal64.exe` |
| Terminal data folder | `C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\` |
| EA experts folder | `...\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts\Arc10_Sidecar\` |
| Sidecar root | `C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext\` |
| Sidecar config | `C:\Forex-Backtester\configs\l_arc_10_v3.0.2\winning_config.yaml` |

## Service & supervision

- **NSSM service name:** `Arc10SidecarFundedNext`
- **Service start type:** Automatic
- **Watchdog task:** `Arc10WatchDog_FundedNext`
- **Logs:** `C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext\logs\sidecar.stderr.log`

## Sidecar command-line (exact)

```
C:\Python311\python.exe -m deployment.sidecar
  --winning-config "C:\Forex-Backtester\configs\l_arc_10_v3.0.2\winning_config.yaml"
  --sidecar-root "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext"
  --mt5-path "C:\Program Files\FundedNext MT5 Terminal\terminal64.exe"
  --log-level INFO
```

## EA Input parameters

Attached to **ONE chart** on the FundedNext MT5.

| Input | Value |
|---|---|
| Risk_Per_Trade | `0.0050` |
| Total_DD_Halt_Pct | `0.07` |
| Total_DD_CloseAll_Pct | `0.08` |
| Daily_DD_Halt_Pct | `0.035` |
| Daily_DD_CloseAll_Pct | `0.045` |
| Time_Exit_Bars | `240` |
| SL_ATR_Multiplier_Expected | `3.5` |
| Sidecar_Inbox_Dir | `Arc10_FundedNext\signals_out` |
| Sidecar_Processed_Dir | `Arc10_FundedNext\signals_processed` |
| Sidecar_Failed_Dir | `Arc10_FundedNext\signals_failed` |
| Sidecar_Heartbeat_Path | `Arc10_FundedNext\sidecar.heartbeat` |
| Ea_Heartbeat_Path | `Arc10_FundedNext\ea.heartbeat` |
| Ea_Positions_Path | `Arc10_FundedNext\ea_positions.json` |
| Trade_Log_Path | `Arc10_FundedNext\trade_log.csv` |
| Sidecar_Heartbeat_Max_Age_Sec | `600` |
| **Expected_Config_Hash** | `75d03904457580b77be63639a281dc550ca584fc5603cfb946102b836d41cf87` |
| News_Calendar_URL | `https://nfs.faireconomy.media/ff_calendar_thisweek.xml` |
| Enable_News_Filter | `true` |
| News_Window_Sec | `120` |
| News_Delay_Buffer_Sec | `5` |
| News_Delay_Max_Sec | `3600` |
| News_Refresh_Sec | `14400` |
| **Magic_Number** | `1010202602` |
| Signal_Poll_Min_Interval_Sec | `5` |

Plus on the Common tab: ✅ "Allow Algo Trading".

## News filter URL whitelisted in MT5

Tools → Options → Expert Advisors:
- ✅ Allow WebRequest for listed URL
- URL: `https://nfs.faireconomy.media`

## Risk per trade rationale

0.50% locked per cost sweep (`02_validation/05_cost_sweep.md`):
- FundedNext is swap-free → strict improvement vs UTC (no dominant swap cost)
- At r_base 0.5%, central case (1.5× spread, swap-OFF, 0.5 slip) lands at 7.80% worst-fold DD — within target
- Adjacent cells recommend 0.49% (adverse) to 0.53% (optimistic) — band spans r_base
- Deploy at 0.50%, accept central case with 2.2pp margin to 10% hard limit

Expected live (after haircuts): worst-fold DD ~8.8%, worst-fold ratio ~3.4, holdout ROI ~44%.

## Connection details (operational)

- **Server timezone:** EET (+2 winter / +3 summer)
- **Bar anchor convention:** EET — verified by panel-diff
- **Sidecar wakes at:** EET-anchored boundaries = UTC 22/02/06/10/14/18 (winter) or UTC 21/01/05/09/13/17 (summer)
- **Daily DD reset:** EET midnight (broker-local)
- **DST handling:** Sidecar handles EET ↔ EEST transitions via `zoneinfo.ZoneInfo("Europe/Athens")`. No operator action needed.

## Per-trade economics

| Item | Value |
|---|---|
| Risk per trade ($) | $500 (0.50% of $100k) |
| SL distance | 3.5 × ATR_at_signal |
| Commission | $5 per lot round-turn |
| Swap | None (swap-free add-on) |
| Average expected R per trade | ~0.7-0.9R (cost sweep central case) |
| Expected trades per year | ~225 |
| Expected annual ROI (cost-adjusted) | ~44% (after haircuts) |

## Challenge phase rules (verify current rules before buying)

**Phase 1 (Challenge):**
- Target: typically +8% profit
- Max DD: 10% account, 5% daily
- Min trading days: typically 5 days
- Time limit: typically 60+ days

**Phase 2 (Verification):**
- Target: typically +5% profit
- Same DD limits
- Same min days

**Funded:**
- 80% profit split
- Continues to be on swap-free
- Can scale account at performance milestones

**Always verify current rules with FundedNext support before purchasing.** Rules change.

## Transition from demo → Challenge → funded

**No system changes required.** Same MT5 install, same sidecar service, same EA.

1. Buy Challenge → FundedNext emails account credentials
2. In FundedNext MT5: File → Login to Trade Account → enter Challenge credentials
3. MT5 reconnects, EA detects new balance on next init
4. Trade as normal

If Challenge passes → repeat for verification phase, then for funded account. Each transition is a re-login, not a re-deploy.

## Verification checklist (post-deploy)

```powershell
# Service status
& "C:\Tools\nssm.exe" status Arc10SidecarFundedNext
# → SERVICE_RUNNING

# Watchdog status
Get-ScheduledTask Arc10WatchDog_FundedNext | Select TaskName, State
# → Ready

# Heartbeat freshness
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext\sidecar.heartbeat" -Raw
# → last_heartbeat_utc within last 4 hours (or 5 hours during EEST)
# → 28 pairs listed

# Logs
Get-Content "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10_FundedNext\logs\sidecar.stderr.log" -Tail 20
# → "sidecar starting — config_hash=75d03904... pairs=28"
```

In MT5:
- Bottom-right green connection, balance/equity
- Chart shows EA smiley face
- Experts tab: `[ARC10] EA init magic=1010202602 sidecar_inbox=Common\Files\Arc10_FundedNext\signals_out`

## Risk ramp for first weeks live

Don't deploy at full 0.50% on day one. Ramp:

- **Week 1:** Risk_Per_Trade = `0.0020` (0.20%, $200/trade)
- **Week 2:** if Week 1 clean → `0.0030` (0.30%)
- **Week 3+:** if Week 2 clean → `0.0050` (0.50%, target)

If anything anomalous happens in any week, hold or scale back. Don't ramp up under pressure.

## What changes between demo and live Challenge

| Item | Demo | Challenge | Funded |
|---|---|---|---|
| Account credentials | Demo login | Challenge login | Funded login |
| Sidecar | Same | Same | Same |
| EA | Same | Same | Same |
| Risk | 0.20-0.50% (ramping) | 0.50% | 0.50% |
| Hard DD limit | 10% | 10% | 10% |
| Profit target | N/A | +8% (Phase 1) | None |
| Profit split | N/A | None (challenge) | 80% |

## How to verify swap-free is actually applied

After first overnight position holds, check the trade in MT5's history:

- View → Reports → click trade
- Check "Swap" column: should be `0.00` for ALL closed trades that held overnight

If swap is NOT 0 on swap-eligible trades, swap-free add-on is not active. Contact FundedNext support immediately. **Don't continue trading at 0.50% risk if swap is being applied** — cost sweep economics break.
