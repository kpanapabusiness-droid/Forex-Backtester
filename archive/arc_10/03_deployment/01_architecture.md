# Deployment Architecture

> **Audience:** Anyone who needs to understand how the live system is wired up.

## High-level diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Contabo VPS (Frankfurt, Win Server 2022)            │
│                                                                      │
│  ┌────────────────────────┐         ┌────────────────────────┐     │
│  │   5ers MT5 Terminal     │         │  FundedNext MT5 Terminal│     │
│  │  (logged into demo)     │         │  (logged into demo)     │     │
│  │                         │         │                         │     │
│  │  • 28 pairs in Market   │         │  • 28 pairs in Market   │     │
│  │    Watch                │         │    Watch                │     │
│  │  • EA on EURUSD chart   │         │  • EA on EURUSD chart   │     │
│  │  • Magic 1010202601     │         │  • Magic 1010202602     │     │
│  │  • config_hash 4467...  │         │  • config_hash 75d0...  │     │
│  └───────────┬─────────────┘         └───────────┬─────────────┘     │
│              │ FILE_COMMON                       │ FILE_COMMON       │
│              ▼                                   ▼                   │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  C:\Users\Administrator\AppData\Roaming\MetaQuotes\         │    │
│  │  Terminal\Common\Files\                                     │    │
│  │  ├── Arc10_5ers\        (5ers paths)                        │    │
│  │  │   ├── signals_out\      ← sidecar writes here            │    │
│  │  │   ├── signals_processed\                                 │    │
│  │  │   ├── signals_failed\                                    │    │
│  │  │   ├── sidecar.heartbeat                                  │    │
│  │  │   ├── sidecar_state.json                                 │    │
│  │  │   ├── ea.heartbeat                                       │    │
│  │  │   ├── ea_positions.json                                  │    │
│  │  │   ├── trade_log.csv                                      │    │
│  │  │   └── logs\                                              │    │
│  │  └── Arc10_FundedNext\  (FN paths, same structure)          │    │
│  └────────────────────────────────────────────────────────────┘    │
│              ▲                                   ▲                   │
│              │ writes signals                    │                   │
│  ┌───────────┴─────────────┐         ┌───────────┴─────────────┐     │
│  │  Arc10Sidecar5ers       │         │  Arc10SidecarFundedNext │     │
│  │  (NSSM service)         │         │  (NSSM service)         │     │
│  │                         │         │                         │     │
│  │  python -m sidecar      │         │  python -m sidecar      │     │
│  │  --config UTC.yaml      │         │  --config EET.yaml      │     │
│  │  --mt5-path "5ers"      │         │  --mt5-path "FN"        │     │
│  │  --sidecar-root "..."   │         │  --sidecar-root "..."   │     │
│  │                         │         │                         │     │
│  │  Wakes every 4h UTC     │         │  Wakes every 4h EET     │     │
│  │  Fetches 28 pairs       │         │  Fetches 28 pairs       │     │
│  │  Emits signal envelopes │         │  Emits signal envelopes │     │
│  └─────────────────────────┘         └─────────────────────────┘     │
│              ▲                                   ▲                   │
│              │ supervises                        │ supervises        │
│  ┌───────────┴─────────────┐         ┌───────────┴─────────────┐     │
│  │  Arc10WatchDog_5ers     │         │  Arc10WatchDog_FundedNext│    │
│  │  (Task Scheduler, 5min) │         │  (Task Scheduler, 5min) │     │
│  │                         │         │                         │     │
│  │  • Check heartbeat fresh│         │  • Same                 │     │
│  │  • Restart service if   │         │                         │     │
│  │    stale > 4h10min      │         │                         │     │
│  └─────────────────────────┘         └─────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

## The four moving parts

### 1. MT5 Terminals

Each broker has its own MT5 terminal installation. Both running 24/7 on the VPS, logged into their respective accounts, with all 28 Arc 10 pairs in Market Watch.

**Critical:** must stay logged in. If MT5 logs out, the sidecar can't fetch data and the EA can't place trades. NSSM doesn't supervise MT5 — it's a GUI app, not a service.

### 2. Sidecar Services

Each broker has its own NSSM-managed Python sidecar. Each runs `python -m deployment.sidecar` with broker-specific arguments:

- `--winning-config` points to that broker's config file (UTC or EET)
- `--sidecar-root` points to that broker's `Arc10_*\` subfolder
- `--mt5-path` points to that broker's `terminal64.exe`

Both services share the SAME repo (`C:\Forex-Backtester`) and SAME Python interpreter (`C:\Python311\python.exe`). Code is identical; only runtime configuration differs.

### 3. EAs

Each MT5 has the Arc10_DLR_Sidecar_EA attached to ONE chart. Same compiled .ex5 on both terminals.

Input parameters differ between brokers:
- Risk_Per_Trade: 0.0040 (5ers) vs 0.0050 (FundedNext)
- Magic_Number: 1010202601 vs 1010202602
- Expected_Config_Hash: 4467366b...840821e vs 75d03904...41cf87
- Sidecar_Inbox_Dir / etc.: Arc10_5ers vs Arc10_FundedNext

The EA polls its sidecar's `signals_out/` folder via FILE_COMMON. Validates each envelope's config_hash matches Expected_Config_Hash before processing.

### 4. Watchdogs

Each sidecar has a Task Scheduler entry checking heartbeat freshness every 5 minutes. If `sidecar.heartbeat` is older than 4h 10min (one H4 cycle + 10min buffer), watchdog runs `nssm restart Arc10Sidecar<broker>`.

Plus NSSM's own restart-on-crash supervision. Two layers of failure recovery.

## File flow per trade

1. **H4 boundary closes** (e.g. 12:00 UTC for UTC convention)
2. **Sidecar wakes** at +10s buffer
3. **Sidecar fetches** H4 + D1 panels for 28 pairs from MT5
4. **Sidecar computes** signals via `signals.lchar_dlr_long.compute_signal()`
5. **For any pair that fires:** sidecar writes `Arc10_5ers/signals_out/EURUSD_2026-05-29T12_00_00Z.json` (or similar)
6. **Sidecar writes** `sidecar.heartbeat` with timestamp
7. **EA detects** the envelope on next OnTick (every ~few seconds during market hours)
8. **EA validates** envelope: config_hash matches, schema valid, sidecar heartbeat fresh
9. **EA places** market buy order with SL set
10. **Broker fills** the order, returns ticket
11. **EA writes** `trade_log.csv` entry row
12. **EA moves** envelope to `signals_processed/`

Future events on the same trade (partial close at TP1, trail modifications, exit) write additional rows to `trade_log.csv`.

## Independence between the two brokers

The two deployments are **fully independent**:

- Different MT5 terminals → different broker connections, different price feeds
- Different `Arc10_*\` subfolders → no shared files between sidecars
- Different config_hash → hash mismatch would catch cross-contamination
- Different magic numbers → broker can distinguish trades from each EA
- Different NSSM services → one can be restarted/upgraded without affecting the other
- Different watchdogs → independent failure detection

If 5ers MT5 disconnects, FundedNext keeps trading. If FundedNext sidecar crashes, 5ers sidecar keeps cycling. No shared failure points except:
- The VPS itself (single point of failure for the whole system)
- The OS (Windows update could affect both)
- The repo (a bad git pull would affect both if it broke shared code)

## VPS specs

| Resource | Spec | Used | Margin |
|---|---|---|---|
| CPU | 4 cores | ~20% idle, peaks at H4 closes | 80% margin |
| RAM | 8 GB | ~3 GB used by MT5×2 + sidecars×2 + Windows | 5 GB margin |
| Disk | 150 GB SSD | <2 GB used | massive margin |
| Network | 200 Mbit/s | minimal sustained, peaks during fetch | huge margin |
| Location | Frankfurt (EU) | ~25ms RTT to FundedNext server (London) | acceptable |

Cost: ~$14.59/month. No upgrade needed for current deployment.

## What's NOT on the VPS

- The repo's research code (backtester, WFO, signal discovery). Only deployment code is needed live.
- Strategy research data (`results/l_arc_10_*` folders). Production sidecar reads only `winning_config.yaml`.
- ForexFactory news calendar download cache. Downloaded fresh by sidecar every 4 hours.
- Trade analytics / reporting. Generated from `trade_log.csv` files post-hoc.

Keeping the VPS lean reduces attack surface and noise. Research work happens on local; only deployment artifacts live on VPS.

## Source artifacts

| Artifact | Path |
|---|---|
| Sidecar entry point | `deployment/sidecar/__main__.py` |
| Sidecar main loop | `deployment/sidecar/sidecar.py` |
| Boundary convention logic | `deployment/sidecar/boundary.py` |
| EA source | `deployment/ea/Arc10_DLR_Sidecar_EA.mq5` + `include/*.mqh` |
| Watchdog script | `deployment/ops/watchdog.ps1` |
| Watchdog install script (reference) | `deployment/ops/install_watchdog_task.ps1` |
| NSSM | `C:\Tools\nssm.exe` on VPS |
