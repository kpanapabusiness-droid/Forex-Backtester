# Sidecar + EA Build Guide

> **Purpose:** Generic how-to for building a Python sidecar + MQL5 EA for a new trading system, abstracted from Arc 10's specific implementation.
> **Audience:** future-you (or a CC dispatch) building a follow-on system from a validated signal.
> **Prerequisites:** validated signal logic in Python lab. Cost sweep done. WFO PASS-DEPLOYABLE.

## Architecture overview

The deployment pattern is **Path B — Python sidecar + thin MQL5 EA**:

```
MT5 Terminal (logged into broker)
    ↓ Python MetaTrader5 lib
Python Sidecar (runs as Windows service via NSSM)
    ↓ writes envelope JSON files
Common\Files\<broker>\signals_out\
    ↓ reads via FILE_COMMON
MQL5 EA (attached to one chart in MT5)
    ↓ places orders
Broker
```

**Why Path B over alternatives:**
- Strategy logic stays in Python (the lab language)
- EA stays thin (~25K lines MQL5 total)
- Lab ↔ sidecar parity provable via byte-identical signal output
- EA only does execution mechanics (validated separately via ST scenarios)
- Sidecar can be restarted/upgraded without touching the EA

**Alternatives we considered and rejected:**
- **Pure MQL5 EA** — would require porting Python signal logic to MQL5. Path that destroyed multiple early arcs (GPT-4 hallucinated conversions). Permanently excluded.
- **Direct Python → broker via FIX/API** — most prop brokers don't offer it. Tied to MT5 for broker compatibility.
- **One process for both** — couples lifecycle concerns. Sidecar dies → EA dies. Separation gives independence.

## The four components

### 1. The lab (Python signal logic)

**Stays where it is.** The validated signal lives in `signals/<signal_name>.py` (or similar). For Arc 10: `signals/lchar_dlr_long.py`.

**Contract:** the signal exports a `compute_signal(h4_panel, d1_panel, config) -> SignalDecision` function. Pure function, no side effects, deterministic output for given input. This is what gets reused in the sidecar.

**Why this matters:** the sidecar imports and calls THIS function. There's no separate "live version" — same code, same behavior, byte-identical output.

### 2. The sidecar (Python service)

**Location:** `deployment/sidecar/` in repo.

**Responsibilities:**
- Wake on the configured H4 boundary schedule
- Fetch H4 + D1 panels from MT5 for the pair universe
- Call the lab's `compute_signal()` for each pair
- Write signal envelopes (JSON) to `signals_out/`
- Write heartbeat
- Persist state across restarts
- Probe MT5 alignment at boot
- Handle DST and convention conversions

**Key modules** (Arc 10 pattern, adapt for new system):
- `__main__.py` — entry point, arg parsing
- `sidecar.py` — main loop
- `signal_runner.py` — wraps lab signal call
- `signal_emitter.py` — writes envelopes
- `mt5_data_fetcher.py` — MT5 data layer
- `boundary.py` — convention logic (UTC vs EET vs other)
- `heartbeat.py` — heartbeat writer
- `h4_schedule.py` — boundary computation

### 3. The EA (MQL5)

**Location:** `deployment/ea/` in repo.

**Responsibilities:**
- Poll `signals_out/` for envelopes
- Validate envelope schema + config_hash + freshness
- Compute position size from risk + SL distance
- Place market order with SL set on broker
- Monitor positions per H4 bar
- Execute partial close at TP1
- Trail SL per the exit policy
- Time-exit at configured bar count
- Enforce equity DD halts
- Reconstruct state on restart

**Key MQL5 files** (Arc 10 pattern, adapt for new system):
- `<Name>_EA.mq5` — entry point + main loop (OnInit, OnTick, OnDeinit)
- `include/SignalPoller.mqh` — envelope polling + validation
- `include/PositionManager.mqh` — entry placement + position tracking
- `include/ExitPolicyEngine.mqh` — exit logic (TP1, trail, time, etc.)
- `include/EquityGuards.mqh` — DD halt enforcement
- `include/RecoveryManager.mqh` — state reconstruction on restart
- `include/HeartbeatWriter.mqh` — EA-side heartbeat (for health check)
- `include/TradeLogger.mqh` — trade_log.csv writes
- `include/NewsFilter.mqh` — news event filtering

### 4. Ops infrastructure

- **NSSM service** wrapping the Python sidecar
- **Task Scheduler watchdog** restarting sidecar on heartbeat staleness
- **Windows startup shortcuts** for MT5 terminals (so they auto-start on VPS reboot)

## Build sequence (for a new system)

### Phase 0 — Strategy validation (prerequisite)

Before you build anything in this guide:
- Signal logic validated in lab
- WFO PASS-DEPLOYABLE
- Cost sweep deployable
- Exit policy locked
- Pair universe locked
- Conventions locked (UTC, EET, other)

If any of those is open, **do not start building the sidecar/EA**. Premature build = wasted work when the strategy changes.

### Phase 1 — Sidecar build (~1 week)

Reference dispatch: see git history for `DISPATCH_B_SIDECAR_BUILD.md` from Arc 10. Adapt for new system.

**Deliverables:**

1. **`winning_config.yaml`** — the locked strategy config. Includes:
   - `boundary_convention` (utc / 5ers_eet / other)
   - `pairs` (list of 28 or whatever)
   - `signal_params` (whatever the lab uses)
   - `exit_policy` parameters
   - `risk_params` (Risk_Per_Trade, DD halts)
   - `news_filter` settings

2. **`config_hash`** — SHA-256 of the canonical-serialized YAML. Computed at sidecar boot, logged on every cycle, baked into envelopes.

3. **Sidecar core loop:**
   - Computes next H4 close based on `boundary_convention`
   - Sleeps until target + 10s buffer
   - Fetches H4 + D1 panels via `MetaTrader5.copy_rates_from_pos()`
   - For each pair: calls `lab.compute_signal()`
   - For each fire: writes envelope to `signals_out/<PAIR>_<iso_ts>.json`
   - Writes heartbeat
   - Persists state
   - Sleeps until next cycle

4. **CLI:**
   ```
   python -m deployment.sidecar
     --winning-config <path>
     --sidecar-root <path>
     --mt5-path <terminal64 path>
     --log-level INFO
     [--quick-test]   # bypass H4 boundary wait for smoke tests
     [--iterations N] # run N cycles then exit
   ```

5. **Unit tests:**
   - Boundary computation per convention
   - Envelope schema generation
   - Heartbeat write
   - Anchor probe
   - State persistence + reload
   - DST transitions

**Gates before Phase 2:**
- Sidecar boots, anchor probe passes
- All 28 pairs fetched in single cycle
- Envelope written matches schema
- Heartbeat written
- State persistence survives restart

### Phase 2 — EA build (~1 week)

Reference: Arc 10 EA at `deployment/ea/`.

**Deliverables:**

1. **Main EA file** (`.mq5`):
   - OnInit: read inputs, init equity floor, reconstruct state from positions
   - OnTick: poll signals, manage positions, write heartbeat, check DD
   - OnDeinit: clean exit, write final state

2. **Include modules** (`.mqh`):
   - `SignalPoller` — reads `signals_out/`, validates, hands off to PositionManager
   - `PositionManager` — places entries, tracks positions per pair (last_processed_h4_bar)
   - `ExitPolicyEngine` — TP1 partial close, trail, time exit
   - `EquityGuards` — daily + total DD halt enforcement
   - `RecoveryManager` — reconstruct positions from broker on restart
   - `NewsFilter` — defer/discard entries near scheduled news
   - `HeartbeatWriter` — EA-side heartbeat
   - `TradeLogger` — trade_log.csv emission

3. **Inputs** (configurable per broker):
   - `Risk_Per_Trade` (per-broker)
   - DD halt thresholds (universal)
   - Time_Exit_Bars (universal)
   - SL_ATR_Multiplier_Expected (universal — must match config YAML)
   - Sidecar path inputs (per-broker subfolder)
   - `Magic_Number` (per-broker)
   - `Expected_Config_Hash` (per-broker)
   - News filter settings (universal)

4. **ST scenario tests:**
   - Scenarios.json defining 10-12 test cases
   - fake_sidecar.py emitting synthetic envelopes
   - Run each scenario in MT5 Strategy Tester
   - Verify trade_log.csv matches expected events

**Gates before Phase 3:**
- All ST scenarios pass (or have documented valid-skip / defer-to-live reasoning)
- EA attaches cleanly to MT5
- No memory leaks in extended Strategy Tester run

### Phase 3 — Parity validation (~2 days)

**Goal:** prove the sidecar's signal output is byte-identical to the lab's signal output on the same historical panels.

**Method:**
1. Fetch historical H4 + D1 panels from broker via Python MetaTrader5 lib
2. Aggregate / align to the configured convention
3. For each candidate bar (lab signal ∪ lab prefilter_pass ∪ random sample): run sidecar's `run_signal()` and lab's `compute_signal()` on the same panel
4. Compare fire/no-fire decision + audit field values
5. Acceptance gates:
   - ≥99.5% byte-identical signal emission
   - 0 true logic divergence
   - Audit field deltas ≤ 1e-9
   - Pool x-check exact on all pairs

**Outputs:**
- `parity_report.md` per convention
- `divergence_ledger.parquet` documenting any tolerated residuals

**Gates before Phase 4:**
- Parity gate passes
- Any divergence rows are timing-only / live-harmless

### Phase 4 — VPS deployment

Follow `arc_10/03_deployment/04_vps_setup_guide.md` — the steps are generic enough to reuse. Substitute the system-specific paths, magic numbers, config hashes.

## Patterns that work

### Convention-aware sidecar via single module

Don't fork the sidecar for different conventions. Put all convention logic in one module (Arc 10: `boundary.py`). All other sidecar layers delegate to it. Adding a new convention = one new branch in `boundary.py`, not a new sidecar codebase.

### Config hash baked into envelope

The envelope carries the config_hash that produced it. EA validates against `Expected_Config_Hash`. Catches misconfiguration at envelope-receive time, not after a wrong trade fires.

### Same compiled .ex5 across brokers

Don't compile per-broker. Same binary, different inputs. Reduces drift risk.

### FILE_COMMON path resolution

Even in portable mode, MT5's `FILE_COMMON` resolves to user-wide AppData. For multi-broker on one Windows user, use broker-specific subfolders (`Arc10_5ers\`, `Arc10_FundedNext\`). Zero EA code changes; all input parameter strings.

### NSSM for service supervision

PowerShell can't supervise a long-running Python process reliably. NSSM handles restart-on-crash, log capture, service start/stop properly.

### Watchdog at 5min checking heartbeat 4h10min threshold

Tight enough to catch real issues. Tolerant enough not to spuriously restart during normal H4 gaps.

## Patterns that DON'T work

### Indicator-based entries (NNFX-style)

Tested. Fails. Indicator parameters are too flexible; overfit to fold-specific noise. Use structural signals instead.

### GPT-4 for MQL→Python conversions

Permanently excluded. Hallucinates function signatures. Aider permanently excluded for same reason.

### Volume features

Don't port across brokers. Microstructure metric, not strategy edge.

### Same-day D1 in features

Lookahead. D1 isn't closed yet at H4 bar close. Always use yesterday's D1, never today's.

### Whole-pool MFE/MAE optimization

Overfits. Use cross-fold stability instead.

### Magnitude clustering

Identifies winners post-hoc but can't predict them ex-ante. Use path-shape clustering instead.

### Bundled changes per phase

Destroys interpretability. One change per phase. Pre-commit the gate.

## Quick reference — what each file does

| File | What |
|---|---|
| `signals/<name>.py` | Lab signal logic (pure function) |
| `configs/<name>/winning_config.yaml` | Locked strategy config |
| `deployment/sidecar/__main__.py` | Sidecar CLI entry |
| `deployment/sidecar/sidecar.py` | Main loop |
| `deployment/sidecar/boundary.py` | Convention logic |
| `deployment/sidecar/signal_runner.py` | Wraps lab signal call |
| `deployment/sidecar/signal_emitter.py` | Envelope writer |
| `deployment/sidecar/mt5_data_fetcher.py` | MT5 data layer |
| `deployment/sidecar/heartbeat.py` | Heartbeat writer |
| `deployment/sidecar/h4_schedule.py` | Boundary computation |
| `deployment/ea/<Name>_EA.mq5` | EA entry point |
| `deployment/ea/include/SignalPoller.mqh` | Envelope polling |
| `deployment/ea/include/PositionManager.mqh` | Position management |
| `deployment/ea/include/ExitPolicyEngine.mqh` | Exit logic |
| `deployment/ea/include/EquityGuards.mqh` | DD halts |
| `deployment/ea/include/RecoveryManager.mqh` | Restart recovery |
| `deployment/ea/include/NewsFilter.mqh` | News filtering |
| `deployment/ea/include/HeartbeatWriter.mqh` | EA-side heartbeat |
| `deployment/ea/include/TradeLogger.mqh` | trade_log.csv writer |
| `deployment/ops/watchdog.ps1` | Heartbeat watchdog |
| `deployment/ops/install_watchdog_task.ps1` | Task Scheduler installer |

## When to use this guide

- Building a new strategy from a validated signal
- Porting Arc 10 to a fundamentally different broker (different MT5 platform / different data feed)
- Migrating from one prop firm to another with structural differences

## When NOT to use this guide

- Tweaking Arc 10 parameters → just change the YAML + recompute hash
- Adding a new pair → just add to the YAML + re-run parity
- New convention → add a branch in `boundary.py`, don't rebuild

## Estimated build time (for a new system)

If you have a validated signal already in the lab:

- Phase 1 (sidecar): ~1 week
- Phase 2 (EA): ~1 week
- Phase 3 (parity): ~2 days
- Phase 4 (deployment): ~3 hours focused work + soak

**Total: ~2.5 weeks from validated lab signal to live VPS deployment.**

Most of the time is in the EA (MQL5 is fiddly) and parity validation (must be byte-identical, no shortcuts).

## What this guide doesn't cover

- Strategy validation (that's the L_ARC_PROTOCOL methodology — see `L_PROTOCOL.md`)
- Cost sweep methodology (see `arc_10/02_validation/05_cost_sweep.md` and the cost-sweep dispatch pattern)
- WFO methodology (see `BACKTESTER_ARCHITECTURE.md` and L_PROTOCOL)
- Broker selection (see `arc_10/03_deployment/06_broker_specs.md`)

These are upstream concerns. This guide picks up AFTER strategy is validated and starts at "build the sidecar+EA."
