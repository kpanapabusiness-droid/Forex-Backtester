# Sidecar + Thin EA — signal-agnostic deployment template

> **Status:** deployment/runbook MACHINERY ONLY — a signal-agnostic template. **Not wired to any
> live system; deployable-system count = 0.** The Arc-10 (DLR) system this was first built for is
> KILLED (gate-fidelity defect — see [../docs/ARC_10_GATE_FIDELITY_DEFECT.md](../docs/ARC_10_GATE_FIDELITY_DEFECT.md));
> its result artefacts are archived under `archive/`. Do NOT trade any system live off this template
> without a fresh PASS-DEPLOYABLE verdict from the sole gate engine (`MultiPairBacktester`).

---

## Overview

Two-process deployment:

```
┌─────────────────────────────────┐         ┌──────────────────────────┐
│  Python sidecar (Windows VPS)   │         │  MQL5 EA (same VPS)      │
│  ───────────────────────────    │         │  ────────────────────    │
│  • Wake at every UTC H4 close   │  files  │  • Poll signals_out/     │
│  • mt5.copy_rates_from_pos      │ ──────► │  • Validate config_hash  │
│  • signals.lchar_dlr_long       │         │  • Place entries         │
│  • Emit envelope JSON           │         │  • Manage exits          │
│  • Write sidecar.heartbeat      │ ──────► │  • Trade log + telemetry │
└─────────────────────────────────┘         └──────────────────────────┘
```

The sidecar owns **all** signal computation (signal logic stays in
Python where the WFO verdict was earned). The EA is intentionally thin
— no signal logic, no SL/exit policy parameters, no risk math outside
of lot sizing.

---

## Directory layout

```
deployment/
├── sidecar/                                # Python sidecar package
│   ├── __init__.py
│   ├── __main__.py                         # CLI entry — `python -m deployment.sidecar ...`
│   ├── config.py                           # winning_config + canonical config_hash
│   ├── heartbeat.py                        # sidecar.heartbeat atomic write
│   ├── mt5_data_fetcher.py                 # MT5 library wrapper + backoff
│   ├── sidecar.py                          # main loop + UTC H4 scheduler
│   ├── signal_emitter.py                   # v1.0.0 envelope schema + atomic emit
│   ├── signal_runner.py                    # signals.lchar_dlr_long wrapper
│   └── state_manager.py                    # sidecar_state.json + corruption recovery
│
├── ea/                                     # MQL5 thin EA
│   ├── Arc10_DLR_Sidecar_EA.mq5            # main EA
│   └── include/
│       ├── SignalPoller.mqh                # signals_out poll + JSON parse + move
│       ├── PositionManager.mqh             # per-position state + lot sizing + entry
│       ├── ExitPolicyEngine.mqh            # TP1 intra-tick + bar-close trail + queued exit
│       ├── NewsFilter.mqh                  # FF XML pull + ±120s blackout
│       ├── EquityGuards.mqh                # EET-day rollover + DD thresholds
│       ├── TradeLogger.mqh                 # 24-col atomic-append CSV
│       ├── RecoveryManager.mqh             # OnInit broker-position reconstruction
│       └── HeartbeatWriter.mqh             # ea.heartbeat + sidecar staleness check
│
├── ops/                                    # Windows service ops
│   ├── nssm_sidecar.bat                    # NSSM service install
│   ├── watchdog.ps1                        # heartbeat watchdog (Task Scheduler)
│   ├── install_watchdog_task.ps1           # registers watchdog with Task Scheduler
│   └── uninstall.bat                       # NSSM service remove
│
└── README.md                               # this file
```

Runtime IPC layout (created at deploy time under `<sidecar_root>/`,
which lives at `<APPDATA>\MetaQuotes\Terminal\Common\Files\Arc10\` —
see §1 "Set runtime directory"):

```
<sidecar_root>/                             # = Terminal\Common\Files\Arc10\
├── signals_out/                            # sidecar writes; EA polls (FILE_COMMON)
│   └── <pair>_<bar_iso>.json
├── signals_processed/                      # EA moves here after successful entry
├── signals_failed/                         # EA moves here on validation/news/equity reject
├── sidecar.heartbeat                       # sidecar writes; EA + watchdog read (FILE_COMMON)
├── ea.heartbeat                            # EA writes (FILE_COMMON)
├── sidecar_state.json                      # sidecar internal
├── ea_positions.json                       # EA internal (FILE_COMMON)
├── trade_log.csv                           # EA appends (FILE_COMMON)
└── logs/                                   # NSSM-captured stdout/stderr
```

---

## Setup (Windows VPS)

### 0. Prereqs

- Windows 10/11 or Windows Server 2019+
- Python 3.12+ with `pip install -r requirements-dev.txt` plus `pip install MetaTrader5`
- MetaTrader 5 terminal installed and logged into 5ers
- [NSSM](https://nssm.cc) on `PATH`
- The MT5 terminal must allow **DLL imports** for the `MetaTrader5` Python library to attach
- The MT5 terminal must whitelist `https://nfs.faireconomy.media/` under **Tools → Options → Expert Advisors → "Allow WebRequest for the following URLs"** (only required if `Enable_News_Filter=true`)

### 1. Set runtime directory

The EA uses `FILE_COMMON` for **all** file IO, so the sidecar root
must live under MT5's shared common folder:

```
<APPDATA>\MetaQuotes\Terminal\Common\Files\Arc10\
```

On the standard Windows install this is
`C:\Users\<you>\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Arc10\`.
This path is shared between the live terminal, all Strategy Tester
agents, and the Python sidecar — no directory junctions or path
juggling required. It also survives Strategy Tester's per-agent
`MQL5\Files\` sandbox wipe, so the EA-vs-sidecar handshake works
identically in tester and production.

```powershell
$env:ARC10_SIDECAR_ROOT = "$env:APPDATA\MetaQuotes\Terminal\Common\Files\Arc10"
New-Item -ItemType Directory -Force -Path "$env:ARC10_SIDECAR_ROOT\signals_out"
New-Item -ItemType Directory -Force -Path "$env:ARC10_SIDECAR_ROOT\signals_processed"
New-Item -ItemType Directory -Force -Path "$env:ARC10_SIDECAR_ROOT\signals_failed"
New-Item -ItemType Directory -Force -Path "$env:ARC10_SIDECAR_ROOT\logs"
```

The EA's input defaults (`Sidecar_Inbox_Dir = "Arc10\signals_out"`,
etc.) resolve directly to subdirectories under this root — no further
configuration needed.

> **VPS deployment note:** the live VPS runbook (Contabo) must use the
> same Common\\Files path. If a prior deployment used a custom
> `sidecar_root` outside Common\\Files (e.g. `C:\Arc10\runtime`),
> migrate it: stop the NSSM service, copy state files to the new path,
> update `deployment/ops/nssm_sidecar.bat` and the watchdog config,
> restart. The EA must be re-deployed from this branch (its file IO
> changed) for the new path to be read.

### 2. Compute the canonical config_hash

The sidecar prints the hash at startup; for the EA's `Expected_Config_Hash`:

```powershell
py -c @"
from pathlib import Path
from deployment.sidecar.config import compute_config_hash, load_winning_config
print(compute_config_hash(load_winning_config(
  'configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml'
)))
"@
```

Copy the printed sha256 into the EA's `Expected_Config_Hash` input.

> **Multi-broker VPS:** when two MT5 terminals run on the same host, each
> sidecar must attach to a specific terminal (not the first one that answers).
> See [§ Multi-broker deployment (single VPS)](#multi-broker-deployment-single-vps).

### 3. Install sidecar as NSSM service

Edit `deployment/ops/nssm_sidecar.bat` to set `PYTHON_EXE`, `REPO_ROOT`,
and `SIDECAR_ROOT` to your actual paths. Run as administrator:

```cmd
deployment\ops\nssm_sidecar.bat
```

Then start:

```cmd
net start Arc10Sidecar
```

Verify:

```powershell
Get-Content "$env:ARC10_SIDECAR_ROOT\logs\sidecar.stdout.log" -Tail 50
```

You should see `sidecar starting — config_hash=...` followed by
`waiting Ns for next H4 close at ...`.

### 4. Install watchdog

```powershell
# As administrator
.\deployment\ops\install_watchdog_task.ps1 -SidecarRoot "$env:ARC10_SIDECAR_ROOT"
```

The watchdog fires every 30s; restarts the NSSM service if
`sidecar.heartbeat` is missing or older than 120s.

### 5. Compile and attach the EA

1. Open MetaEditor (F4 inside MT5).
2. Copy `deployment/ea/Arc10_DLR_Sidecar_EA.mq5` and the `deployment/ea/include/` directory into `MQL5/Experts/Arc10_Sidecar/` (preserving structure).
3. Compile. Resolve any include-path errors by editing the `#include "include/*.mqh"` directives if you placed the includes elsewhere.
4. **Subscribe all 28 traded pairs in Market Watch** (right-click Market Watch → "Show All" or add each pair manually). The EA needs SymbolInfo access + iTime/iHigh/iClose data for every pair it trades. `ArcPlaceEntry` calls `SymbolSelect(pair, true)` defensively, but pre-subscribing avoids first-entry latency. The 28 pairs are listed in `configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml`.
5. **Attach the EA to a SINGLE chart** — recommended EURUSD H4 (highest tick frequency = most responsive intra-tick TP1 polling for all positions). The EA is symbol-agnostic: it processes envelopes for any pair the sidecar emits, places trades on `envelope.pair` via broker routing, and tracks per-position H4 bar rollover independently per pair (single-chart-multi-pair topology, matching KH-24 ops model). The chart symbol is irrelevant to position management beyond OnTick wake-up frequency.
6. In the EA input dialog, set:
   - `Risk_Per_Trade` = `0.0043` (default; UTC r_safe)
   - `Sidecar_Inbox_Dir` = `Arc10\signals_out` (default — resolves to
     `<APPDATA>\MetaQuotes\Terminal\Common\Files\Arc10\signals_out\`
     via `FILE_COMMON`)
   - `Expected_Config_Hash` = the sha256 from step 2
   - `Magic_Number` = `1010202601` (or chosen value not already in use)
   - `Enable_News_Filter` = `true` (only if you've whitelisted the FF URL)

> **Topology note:** the EA previously documented "28 charts, one per pair" deployment. That was incorrect — the code is pair-agnostic and 28 parallel instances would each try to enter every signal, producing duplicate orders. Single-chart-EURUSD is the canonical topology.

### 6. Verify

After the first UTC H4 close (00/04/08/12/16/20 UTC), the sidecar log
should show `emitted signal EURUSD-2026-...` if a signal fires. The
EA's Experts journal should show `[ARC10] entry placed` shortly after.
`<sidecar_root>/trade_log.csv` should grow with one row per event.

---

## Multi-broker deployment (single VPS)

To run Arc 10 on two brokers simultaneously from one Windows VPS (e.g. 5ers
under the `utc` convention and FundedNext under `5ers_eet`), run **two
independent MT5 terminals + two independent sidecars**, fully isolated from
each other. The two brokers use different bar-grid conventions, so they are
driven by separate `winning_config.yaml` files and separate sidecar roots —
nothing is shared.

The single hard problem this solves: `mt5.initialize()` with no path attaches
to **whichever terminal answers first**, which is non-deterministic when two
terminals are running. The `--mt5-path` argument pins each sidecar to its
broker's terminal.

### 1. Install each MT5 to a distinct path, portable mode

Install the two terminals to separate directories so their data never collides:

```
C:\MT5_5ers\          # 5ers terminal install
C:\MT5_FundedNext\    # FundedNext terminal install
```

Launch each with the **`/portable`** flag so all data (config, logs, and the
`Common\Files` sandbox) lives **inside the install directory** instead of the
shared `%APPDATA%\MetaQuotes\Terminal\Common\Files\`:

```cmd
C:\MT5_5ers\terminal64.exe /portable
C:\MT5_FundedNext\terminal64.exe /portable
```

The decisive property of `/portable`: it relocates `Common\Files` into the
install dir, giving each broker its **own** common sandbox:

```
C:\MT5_5ers\MetaTrader 5\Common\Files\Arc10\            # 5ers sidecar root
C:\MT5_FundedNext\MetaTrader 5\Common\Files\Arc10\      # FundedNext sidecar root
```

Without `/portable`, both terminals would share the single
`%APPDATA%\...\Common\Files\Arc10\`, and the two sidecars + two EAs would read
and overwrite each other's `signals_out/`, heartbeats, state, and trade logs.
With `/portable`, there is **one `Common\Files` per MT5**, no shared sandbox.

> Confirm the exact `Common\Files` location per terminal: in MetaEditor (F4)
> run `Print(TerminalInfoString(TERMINAL_COMMONDATA_PATH))`, or check
> `File → Open Data Folder` in the terminal. Under `/portable` it resolves
> under the install directory; the `\Common\Files\Arc10\` suffix is where each
> sidecar root lives.

### 2. Log each terminal in once, leave the Windows session running

Launch each terminal interactively once, log into its broker account, enable
AutoTrading + DLL imports (Tools → Options → Expert Advisors), then leave the
Windows session logged in. The sidecars do **not** re-authenticate — they only
*attach* to an already-logged-in terminal. (`--mt5-login` / `--mt5-password` /
`--mt5-server` exist as future-proofing for sidecar-side re-auth but are not
needed in this topology.)

### 3. Create each sidecar's runtime directories

```powershell
foreach ($root in @(
  "C:\MT5_5ers\MetaTrader 5\Common\Files\Arc10",
  "C:\MT5_FundedNext\MetaTrader 5\Common\Files\Arc10"
)) {
  New-Item -ItemType Directory -Force -Path "$root\signals_out"
  New-Item -ItemType Directory -Force -Path "$root\signals_processed"
  New-Item -ItemType Directory -Force -Path "$root\signals_failed"
  New-Item -ItemType Directory -Force -Path "$root\logs"
}
```

### 4. Launch each sidecar pinned to its terminal

Pass `--mt5-path` (the absolute path to that broker's `terminal64.exe`) plus
the broker-appropriate `--winning-config` and `--sidecar-root`:

```powershell
# 5ers (UTC convention)
py -m deployment.sidecar `
  --winning-config "configs\l_arc_10_v3.0.2_utc_rerun\winning_config.yaml" `
  --sidecar-root  "C:\MT5_5ers\MetaTrader 5\Common\Files\Arc10" `
  --mt5-path      "C:\MT5_5ers\terminal64.exe"

# FundedNext (EET convention)
py -m deployment.sidecar `
  --winning-config "configs\l_arc_10_v3.0.2_eet\winning_config.yaml" `
  --sidecar-root  "C:\MT5_FundedNext\MetaTrader 5\Common\Files\Arc10" `
  --mt5-path      "C:\MT5_FundedNext\terminal64.exe"
```

The convention (`utc` vs `5ers_eet`) is read from each `winning_config.yaml`'s
`boundary_convention` key — the sidecar wakes, fetches, normalises broker time,
and probes the bar grid accordingly. (Swap the EET winning-config path for
whichever EET-convention config you deploy; the FundedNext panel-diff was
validated at tag `arc-10-eet-parity-validated`.)

Without `--mt5-path` the sidecar falls back to the legacy default attach
(first terminal to answer) — fine for a single-broker host, non-deterministic
with two terminals running. **Always pass `--mt5-path` on a multi-broker VPS.**

For NSSM, install **two services** (e.g. `Arc10Sidecar5ers`,
`Arc10SidecarFundedNext`), each with its own `--mt5-path`, `--winning-config`,
and `--sidecar-root` in the launch arguments, and a watchdog per service
pointed at that service's `sidecar.heartbeat`.

### 5. Attach each EA to its own terminal

Compile + attach the EA inside **each** terminal separately (step 5 of the
single-broker setup, run once per terminal). Each EA's `Sidecar_Inbox_Dir`
resolves to *its own* terminal's `Common\Files\Arc10\signals_out` via
`FILE_COMMON`, so the two EAs never see each other's signals. Use a distinct
`Magic_Number` per terminal. Set `Expected_Config_Hash` from that broker's
winning-config (the UTC and EET configs hash differently because
`boundary_convention` is in the hashed subset).

### FundedNext symbol mapping

FundedNext exposes all 28 traded pairs as **plain symbols with no suffix**
(`EURUSD`, `GBPUSD`, … — confirmed in the `arc-10-eet-parity-validated`
panel-diff). This is exactly the sidecar's **default identity mapping**:
`SidecarConfig.mt5_symbol_for(pair)` returns the canonical pair unchanged when
no `mt5_symbol_map` is configured. **No `mt5_symbol_map` is required for
FundedNext** — omit it from the sidecar config entirely.

Only configure `mt5_symbol_map` (see [sidecar-config-schema](#sidecar-config-schema))
for a broker that appends a suffix (e.g. `EURUSD.r`, `EURUSD.raw`). Neither
5ers nor FundedNext does, so both run with identity mapping.

---

## trade-log-schema

`<sidecar_root>/trade_log.csv` — 24 columns, header on first line:

```
timestamp_utc,event,signal_id,pair,ticket,direction,entry_price,sl_initial,
sl_distance,r_atr,initial_lots,current_lots,peak_high_bid,trail_sl,
tp1_fired,tp1_bar_ord,bar_ord,partial_price,fill_price,reason,
daily_dd_pct,total_dd_pct,equity,note
```

`event` enum: `entry`, `entry_failed`, `partial_close`, `trail_modify`,
`exit`, `news_delay`, `news_discard`, `equity_block`,
`recovery_reconstructed`. One row per event.

`signal_id` shape: `<pair>-<signal_bar_close_iso8601>`, e.g.
`EURUSD-2026-05-27T16:00:00Z`.

---

## sidecar-config-schema

Optional `sidecar.yaml` (passed via `--sidecar-config`):

```yaml
# Pairs to trade (override winning_config's pairs list).
pairs: [EURUSD, GBPUSD]

# Map canonical pair to broker symbol (some brokers append .raw / .r / etc).
# OPTIONAL — omit entirely for identity mapping. 5ers and FundedNext both use
# plain unsuffixed symbols, so neither needs this block.
mt5_symbol_map:
  EURUSD: EURUSD
  GBPUSD: GBPUSD

# Seconds to wait after H4 close before fetching bars (broker bar
# publication can lag). Default: 10.
bar_publish_buffer_sec: 10

# MT5 reconnect backoff config.
mt5_reconnect_initial_sec: 1
mt5_reconnect_max_sec: 60
mt5_reconnect_alert_after: 3

# Bar history depths.
h4_history_bars: 300
d1_history_bars: 100

# Optional alert webhook (Slack / Discord / etc.).
alert_webhook_url: "https://hooks.slack.com/services/..."
```

---

## Troubleshooting

### `MT5 initialize failed`
- Is the MT5 terminal running and logged in?
- Has the `MetaTrader5` Python package been installed in the same Python interpreter NSSM is using? Check `deployment\runtime\logs\sidecar.stderr.log`.
- Is the terminal allowed to run autotrading? (Toolbar button → "AutoTrading")

### `H4 anchor probe failed — broker is emitting non-UTC-anchored bars`
- The sidecar refuses to start if `mt5.copy_rates_from_pos(EURUSD, H4, ...)` returns bars at non-UTC anchors. This is intentional — running on EET-anchored bars would mean trading on a different bar series than the v3.0.2 UTC verdict was earned on.
- Check: 5ers MT5 should publish UTC H4 bars natively (per Phase 0 ESCALATION).
- If your broker doesn't, this dispatch's design is wrong for that broker — escalate.

### EA: `signal parse failed: config_hash_mismatch`
- The `Expected_Config_Hash` input doesn't match the sidecar's emit.
- Re-run the sha256 print step (above) and update the EA input.

### EA: `signal parse failed: schema_version_mismatch`
- The sidecar's `SCHEMA_VERSION` and the EA's `ARC10_SIGNAL_SCHEMA_VERSION` are out of sync. Both must be `1.0.0` for this dispatch.

### EA: `lot_size_zero`
- Account equity × `Risk_Per_Trade` is too small to satisfy `SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN)`. Increase risk or wait for equity to rebuild.

### Watchdog cycles the service indefinitely
- Inspect `sidecar.stderr.log` for the underlying error. Common causes: MT5 not initialised, winning_config.yaml path wrong, sidecar_state.json corrupted (delete it to reset).

### Sidecar starts then immediately exits with `state file: schema_version=...`
- Corrupted or out-of-version `sidecar_state.json`. Delete the file — sidecar will fresh-initialise on next start.

---

## Out of scope for Phase 1

- Live trading approval (mastermind / user portfolio decision)
- Phase 2 parity validation (see separate dispatch — Dispatch C)
- Modifications to `signals/`, `core/sim/exit_policies/`, or
  `configs/l_arc_10_v3.0.2*` (frozen v3.0.2 anchor)
- Modifications to `reference/arc_10_ea/` on `live/arc_10_dlr_ea`
  (prior 1:1 port, preserved as reference)

---

## Audit trail

- Build dispatch: `DISPATCH_B_SIDECAR_BUILD.md`
- Intent doc:    [phase_1_build_intent.md](../phase_1_build_intent.md)
- UTC verdict:   [results/l_arc_10_v3_0_2_utc_rerun/COMPARISON_REPORT.md](../results/l_arc_10_v3_0_2_utc_rerun/COMPARISON_REPORT.md)
- Phase branch:  `phase/arc_10_sidecar_build`
- Prior 1:1 EA:  `live/arc_10_dlr_ea` (frozen reference)
- Unit tests:    `tests/sidecar/` (46 pass + 1 skip), `tests/ea/` (16 pass)
- ST scenarios:  [tests/ea/scenarios/scenarios.json](../tests/ea/scenarios/scenarios.json)
