# Phase 1 — Python Sidecar + MQL5 EA Build Intent

> **Dispatch:** Dispatch B v2 Phase 1 — Python Sidecar + MQL5 EA Build (UTC)
> **Phase branch (target):** `phase/arc_10_sidecar_build` (work proceeds on worktree branch `claude/naughty-engelbart-94fe71`; pushed under the phase name at PR time)
> **Anchor:** UTC rerun merge — commit `9722bc2`, `origin/main`
> **Date:** 2026-05-27
> **Status:** intent doc — no code written
>
> Per dispatch "Read-first (mandatory)": all six items read; this doc captures the resulting design and the read-first surprises that must be resolved before §1/§2 code begins. **No sidecar, EA, test, or ops-script code until this intent is committed and reviewed.**

---

## §0 Executive summary

Build a UTC-native deployment stack with two pieces:

1. **Python sidecar** (≈300 LOC) running on the Contabo VPS. Connects to 5ers MT5 via the `MetaTrader5` library, fetches H4 + D1 bars at every UTC H4 close, invokes `signals.lchar_dlr_long.compute_signal` byte-identically to the UTC rerun, emits signal JSON to a shared `signals_out/` directory.
2. **Thin MQL5 EA** (≈250 LOC) running on the same 5ers MT5 instance. Polls `signals_out/`, validates config hash, applies the news filter + equity guards, places entries, manages exits per `sl_partial_close_1r_runner_trail` semantics, writes audit telemetry, recovers from restart.

UTC-native means we deploy at the convention the broker actually emits (UTC H4 closes at 00/04/08/12/16/20 UTC). The validation gate is the v3.0.2 UTC rerun PASS-DEPLOYABLE verdict at **r_safe = 0.4336% per trade**.

This phase produces the build only; Phase 2 (separate dispatch) measures parity against the UTC-rerun ledger.

---

## §1 Read-first surprises (must read before §2/§3)

The dispatch references a `docs/ARC_10_DEPLOYMENT_PLAN.md` and §§4.1, 4.3, 4.4, 4.5, 4.8, 4.9 within it. **This document does not exist in the repo.** Cross-checking the dispatch's section references against the canonical sources:

| Dispatch ref | What dispatch implies it covers | What actually exists |
|---|---|---|
| §4 "sidecar spec" | Sidecar architecture | Closure §4 is `deployment_spec` — engine deployment params, not sidecar |
| §4.1 "Signal JSON schema" | JSON schema spec | Closure §4.1 is "Pair set" |
| §4.3 "exit policy semantics" | Exit-policy reference | Closure §4.3 is "Boundary convention" |
| §4.4 "Python-vs-live divergence — EA does NOT replicate SL suppression" | Divergence enumeration | Closure §4.4 is "Filter chain" — and no canonical code mentions "SL suppression" |
| §4.5 "restart recovery" | Recovery spec | Closure §4.5 is "Entry mechanics" |
| §4.8 "trade log schema" | Telemetry spec | Closure §4.8 is "Time exit" |
| §4.9 "directory layout" | IPC layout | Closure §4.9 is "Exposure" |

**Resolution.** The dispatch's "deployment plan" is a planned-but-unwritten document. This intent doc derives the equivalent design from the actual sources:

- [results/l_arc_10_v3.0.2/ARC_CLOSURE.md](results/l_arc_10_v3.0.2/ARC_CLOSURE.md) §4 (canonical engine deployment_spec, §§4.1–4.11)
- [results/l_arc_10_v3_0_2_utc_rerun/COMPARISON_REPORT.md](results/l_arc_10_v3_0_2_utc_rerun/COMPARISON_REPORT.md) (UTC verdict + `r_safe = 0.4336%`)
- [signals/lchar_dlr_long.py](signals/lchar_dlr_long.py) (canonical signal module — defines the function signature the sidecar invokes)
- [core/sim/exit_policies/sl_partial_close_1r_runner_trail.py](core/sim/exit_policies/sl_partial_close_1r_runner_trail.py) (canonical exit semantics)
- [configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml](configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml) (UTC-deployment locked parameters)

The "SL suppression" reference in dispatch §"Read-first" item 1 has no analogue in the canonical exit policy. Working interpretation: the dispatch author meant the Python sim's idealised exit-timing semantics (close-at-bar-close fill) that the live EA cannot replicate at the same precision. See §6 below — this is one of three Python-vs-live divergences we will document explicitly rather than hide.

### §1.1 Prior Arc 10 EA already exists on `live/arc_10_dlr_ea`

Substantial prior implementation lives at `reference/arc_10_ea/` on branch `live/arc_10_dlr_ea`:

- `Arc10_DLR_EA.mq5` (~647 LOC, "1:1 Python port")
- 7 includes: `Arc10_State.mqh`, `Arc10_ATR.mqh`, `Arc10_Signal.mqh`, `Arc10_Risk.mqh`, `Arc10_Exit.mqh`, `Arc10_News.mqh`, `Arc10_Telemetry.mqh`
- Parity tooling (`parity_diff.py`, EURUSD 2023 ledger comparisons)
- `SOURCE_CONTRACT.md` — Python source verbatim snapshot

The history shows: Dispatch 01 (source contract) → Dispatch 02 (1:1 port) → multiple parity-failure diagnostics → Dispatch 04 (signal-set divergence audit) → Dispatch 05a (M1 history pull) → Phase 0 §1 ESCALATION (5ers MT5 publishes UTC bars, lab v3.0.2 used EET — different bar series). The escalation drove the UTC rerun (PR #218) and the present dispatch's sidecar redesign.

**Implication for this build.** Don't copy `Arc10_DLR_EA.mq5` or `Arc10_Signal.mqh` — the whole point of Phase 1 is to remove signal logic from MQL5. But the following prior-EA includes are conceptually reusable for the **thin EA** we are building:

| Prior include | Reusable concept |
|---|---|
| `Arc10_Exit.mqh` | Per-position state machine: peak-high-bid ratchet at bar close, `bar_ordinal > tp1_bar_ordinal` constraint, intra-bar TP1 partial via `PositionClosePartial`, time exit |
| `Arc10_News.mqh` | FF weekly XML pull URL (`https://nfs.faireconomy.media/ff_calendar_thisweek.xml`), tester-mode bypass, retry policy, WebRequest whitelist failure handling |
| `Arc10_Risk.mqh` | Reset-floor sizing under EET day boundary (but dispatch §2.9 says daily-DD tracking IS still EET — see §6.2 below) |
| `Arc10_State.mqh` | Per-symbol state persistence pattern, magic-number filtering |
| `Arc10_Telemetry.mqh` | 43-col ledger schema (extends what dispatch §2.10 sketches) |

We will **adapt** these (re-architect into the dispatch's `include/` layout, strip signal logic), not copy them wholesale.

### §1.2 MQL5 reference article (https://www.mql5.com/en/articles/19065) is a different architecture

The dispatch says "architectural reference pattern only". On read, the article describes an **HTTP polling** architecture (MQL5 `WebRequest` POSTs OHLC payloads to a Flask `/analyze` endpoint, receives JSON signal responses) — not the file-based IPC the dispatch specifies (`signals_out/` directory polled by EA). It has no heartbeat, no atomic file moves, no restart recovery — none of the dispatch's resilience patterns.

**Treat as inspiration for the JSON payload shape only.** The dispatch's IPC design (sidecar writes JSON files, EA polls directory, atomic moves to `signals_processed/` / `signals_failed/`) is the load-bearing architecture and is unrelated to the article.

### §1.3 Risk parameter convention

- Dispatch §"Read-first" item 2: "0.43% per trade under UTC, NOT 0.5%."
- Dispatch §2.3: "defaults to **0.0043** in the EA's input parameters."
- COMPARISON_REPORT.md §4: canonical `r_safe = 0.4336%` (k_safe = 0.8673, scales DOWN from r_base = 0.5%).
- COMPARISON_REPORT.md §7.5: "the appropriate per-trade risk is **0.43%** (r_safe) or **0.54%** (r_hard)."

**Resolution.** EA `Risk_Per_Trade` input parameter default = **0.0043** (rounded; matches dispatch). Sidecar `winning_config_subset` written into `config_hash` will use the canonical `0.004336` to keep determinism tight. Telemetry records the EA's actual sized risk per trade vs the canonical r_safe so any rounding drift is auditable.

### §1.4 Boundary-convention asymmetry (deliberate)

- Lab WFO verdict (UTC rerun): `boundary_convention="utc"`, H4 bars at 00/04/08/12/16/20 UTC, native to 5ers MT5.
- Daily-DD measurement boundary: Amendment 6 specifies **5ers EET broker trading day**.
- ARC_CLOSURE.md §4.3 declares "5ers_eet end-to-end" — that's the **EET-canonical** closure; under the UTC rerun verdict the engine convention is UTC but Amendment 6 still applies to daily-DD bucketing because the prop-firm rules are EET-anchored.

**Resolution.** EA equity-guards (dispatch §2.9) track daily P&L on the EET broker day (matches Amendment 6 + dispatch). All other timing (H4 bar polling, signal evaluation, sidecar heartbeat, news filter) is UTC.

---

## §2 Sidecar module structure

Path: `deployment/sidecar/` (created in §1 of the build). Target ~300 LOC across the modules below; LOC budgets are guidance, not gates.

| File | Responsibility | LOC budget |
|---|---|---|
| `__main__.py` | Entry point: parses CLI args, validates config, hands off to `sidecar.main()` | ~20 |
| `sidecar.py` | Main loop per dispatch §1.3: `init_mt5 → load_state → loop{wait_close, fetch_panels, run_signal, emit, heartbeat, save_state}` | ~80 |
| `config.py` | Loads `configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml` + sidecar-specific config (`PAIRS`, `BAR_PUBLISH_BUFFER_SECONDS`, `OUT_DIR`, alert webhook), computes deterministic `config_hash` (sha256 over the canonical relevant subset) | ~50 |
| `mt5_data_fetcher.py` | Wraps `MetaTrader5.copy_rates_from_pos(...)` for H4 + D1; returns DataFrames with the column schema `signals.lchar_dlr_long.compute_signal` expects (`date`, `open`, `high`, `low`, `close`) — `date` must be UTC datetime64[ns] (naive after the MT5 server-time → UTC adjustment) | ~60 |
| `signal_runner.py` | Thin wrapper that invokes `signals.lchar_dlr_long.compute_signal(df_4h, df_d1)` unchanged; extracts the latest-bar row; builds the canonical signal dict (NOT JSON serialisation — that lives in emitter) | ~40 |
| `signal_emitter.py` | Schema validation (jsonschema), atomic file write (write to `signals_out/.tmp_<uuid>`, fsync, rename to `signals_out/<signal_id>.json`), filename deterministic from signal | ~40 |
| `heartbeat.py` | Writes `sidecar.heartbeat` per dispatch §1.8 at end of every successful loop (success = all pairs processed without uncaught exception, even if signals = ∅) | ~20 |
| `state_manager.py` | Reads/writes `sidecar_state.json` (last processed bar per pair, last loop completion UTC, restart count); per dispatch §1.7 recovery semantics | ~40 |

**File layout finalised.** Tests at `tests/sidecar/` per dispatch §1.2.

### §2.1 UTC H4 close schedule (dispatch §1.4)

`compute_next_utc_h4_close()` returns `next H4 anchor ≥ now_utc + small_epsilon`, where anchors are `{00, 04, 08, 12, 16, 20}` UTC. Hard-coded in code; validation step at sidecar startup queries `mt5.copy_rates_from_pos(EURUSD, mt5.TIMEFRAME_H4, 0, 24)` and asserts every returned bar's `time` modulo 14400s is 0 (no DST drift, no broker offset). On mismatch, sidecar aborts with explicit error — not "silently work on whatever 5ers returns".

`BAR_PUBLISH_BUFFER_SECONDS = 10` (dispatch default). Configurable.

### §2.2 Critical invariant (dispatch §1.6)

`signal_runner.run_signal(df_4h, df_d1, pair, config)` invokes `signals.lchar_dlr_long.compute_signal(df_4h, df_d1)` with byte-identical DataFrame schema to the WFO orchestrator:

- Columns: `date`, `open`, `high`, `low`, `close` (others optional; only these read by `compute_signal`).
- `date` is `datetime64[ns]` UTC-naive (matches `data/cache/utc/<PAIR>.parquet` convention).
- Floats are `float64` (matches `.astype(float)` in `compute_signal`).
- Sort order: chronological ascending, reset_index applied before invocation.
- D1 panel must include enough history for the most-recent identifiable `L_1` lookup (right-edge offset = 4, structure lookback = 30, freshness max = 20 → safest is fetching 100 D1 bars; matches dispatch §1.3).
- H4 panel: ATR(14) Wilder warmup needs ≥14 bars; signal evaluation reads last bar only; dispatch §1.3 fetches 300 (overkill, safe).

Unit test §4 below validates byte-identical output on a UTC cache slice.

### §2.3 Failure modes (dispatch §1.7)

- MT5 connection lost: exponential backoff 1s/2s/4s/.../60s, alert after 3 consecutive failures (alert mechanism = webhook URL in config, or stderr log to NSSM-captured log if no webhook).
- Bar fetch incomplete (e.g., MT5 returns fewer than `count` bars or last bar timestamp doesn't match expected): skip that pair this cycle, log, continue.
- Sidecar crash: NSSM restarts. On restart, `state_manager` reads `sidecar_state.json` for last processed bar per pair. Skipped bars stay skipped (no backfill — dispatch §1.7 explicit). Log the gap.
- Stale state file (next-bar-anchor in past by more than 24h on restart): log warning, jump to current-cycle scheduling, leave state file untouched until next successful save.

---

## §3 EA module structure

Path: `deployment/ea/`. Target ~250 LOC.

| File | Responsibility | LOC budget |
|---|---|---|
| `Arc10_DLR_Sidecar_EA.mq5` | OnInit/OnDeinit/OnTick + on-new-H4-bar dispatch; loads includes; defines `input` parameters | ~120 |
| `include/SignalPoller.mqh` | Directory polling of `signals_out/`, schema validation, `config_hash` check, atomic file move to `signals_processed/` or `signals_failed/` | ~50 |
| `include/PositionManager.mqh` | Order placement (entry at market on next H4 bar after signal arrival; matches Python `entry_long: open_ask`), per-symbol state, `ea_positions.json` persistence | ~40 |
| `include/ExitPolicyEngine.mqh` | TP1 partial intra-bar via `PositionClosePartial`, peak-high-bid ratchet on bar close, trail SL detection at bar close, queue exit at next bar open. Adapted from prior `Arc10_Exit.mqh` (see §6 divergence notes) | ~60 |
| `include/NewsFilter.mqh` | FF weekly XML pull (URL: `https://nfs.faireconomy.media/ff_calendar_thisweek.xml`), red-impact filter, ±2-min blackout per dispatch §2.8, tester-mode bypass. Adapted from prior `Arc10_News.mqh` | ~50 |
| `include/EquityGuards.mqh` | 5ers EET broker day P&L tracking (snapshot equity at EET 00:00), DD thresholds 3.5%/4.5%/7%/8% per dispatch §2.9 | ~40 |
| `include/TradeLogger.mqh` | Atomic-append CSV `trade_log.csv` — schema per dispatch §2.10 / closure-style 43-col superset adapted from prior `Arc10_Telemetry.mqh` | ~30 |
| `include/RecoveryManager.mqh` | OnInit restart recovery per dispatch §2.7 and §4 below | ~40 |
| `include/HeartbeatWriter.mqh` | EA's own heartbeat (`ea.heartbeat`), sidecar-heartbeat staleness check (dispatch §2.2) | ~20 |

LOC will likely overrun the dispatch's 250 target by ~30% once recovery + telemetry are honest. Acceptable.

### §3.1 EA input parameters (defaults)

```
input double  Risk_Per_Trade            = 0.0043;           // 0.43% UTC r_safe
input double  Total_DD_Halt_Pct         = 0.07;             // dispatch §2.9
input double  Total_DD_CloseAll_Pct     = 0.08;
input double  Daily_DD_Halt_Pct         = 0.035;
input double  Daily_DD_CloseAll_Pct     = 0.045;
input int     Time_Exit_Bars            = 240;
input double  SL_ATR_Multiplier         = 3.5;
input string  Signal_Inbox_Path         = "signals_out";
input string  Signal_Processed_Path     = "signals_processed";
input string  Signal_Failed_Path        = "signals_failed";
input int     Sidecar_Heartbeat_Max_Age_Sec = 600;          // 10 min — looser than 2-min watchdog
input string  Expected_Config_Hash      = "<filled at deploy>";
input string  News_Calendar_URL         = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml";
input bool    Enable_News_Filter        = true;
input int     News_Window_Sec           = 120;              // ±2 min
input int     News_Delay_Buffer_Sec     = 5;                // dispatch §2.8 "+5s buffer"
input int     News_Delay_Max_Sec        = 3600;             // discard if delay >1h past signal bar
input long    Magic_Number              = 1010202601;       // arc10 v3.0.2 = 0x... (filled at deploy)
```

---

## §4 Signal JSON schema (final form)

Validated against `signals.lchar_dlr_long.compute_signal` output columns. Sidecar emits one JSON file per signal bar that fires (`signal == True` at the latest fetched H4 bar). Filename: `<pair>_<signal_bar_close_iso8601_utc>.json`, e.g. `EURUSD_2026-05-27T12_00_00Z.json` (colons replaced with underscores for filesystem safety).

```json
{
  "schema_version": "1.0.0",
  "signal_id": "EURUSD-2026-05-27T12:00:00Z",
  "config_hash": "<sha256_hex>",
  "emitted_at_utc": "2026-05-27T12:00:10.234Z",

  "pair": "EURUSD",
  "direction": "long",
  "signal_bar_close_utc": "2026-05-27T12:00:00Z",
  "entry_bar_open_utc": "2026-05-27T16:00:00Z",

  "signal_bar_close_price_mid": 1.08423,

  "sl": {
    "atr_period": 14,
    "atr_multiplier": 3.5,
    "atr14_at_signal_bar": 0.00214,
    "sl_distance_price": 0.00749,
    "anchor": "entry_price",
    "reference": "entry_price"
  },

  "exit_policy": {
    "name": "sl_partial_close_1r_runner_trail",
    "partial_close_at_r": 1.0,
    "partial_close_fraction": 0.5,
    "runner_trail_atr_below_peak": 1.0,
    "time_exit_bars": 240
  },

  "audit": {
    "L1_value": 1.07688,
    "L0_value": 1.07412,
    "L1_age_d1_bars": 6,
    "L0_age_d1_bars": 18,
    "L1_to_atr_proximity": 0.13,
    "reject_buffer_atr": 0.42,
    "upper_fraction": 0.71,
    "d_t_idx": 4321,
    "d_for_l1_search_max": 4317
  }
}
```

### §4.1 Field-by-field provenance

| Field | Source | Notes |
|---|---|---|
| `schema_version` | sidecar constant | `1.0.0` for Phase 1; bumped on breaking change. EA validates major version |
| `signal_id` | `<pair>-<signal_bar_close_iso8601>` | deterministic; collision-free assuming pair+bar uniqueness |
| `config_hash` | sidecar boot-time sha256 over canonical subset of winning_config.yaml | EA rejects on mismatch (dispatch §2.3). Subset: `risk_per_trade`, `sl.atr_multiplier`, `exit_policy.*`, `time_exit_bars`, `pairs`, `boundary_convention` |
| `emitted_at_utc` | sidecar wall-clock at emit time | informational; not used for any decision |
| `pair`, `direction` | sidecar config | `direction` always `"long"` (arc 10 is long-only) |
| `signal_bar_close_utc` | H4 panel's last bar `date` | identity to compute_signal input |
| `entry_bar_open_utc` | `signal_bar_close_utc + 4h` | matches closure §4.5 "Bar N+1 open after signal on bar N close" |
| `signal_bar_close_price_mid` | H4 panel `close` of last bar | audit field; EA uses ask at entry-bar open for actual fill |
| `sl.atr14_at_signal_bar` | `compute_signal` output `atr14` at last bar | Wilder ATR(14); R unit |
| `sl.sl_distance_price` | `3.5 * atr14_at_signal_bar` | redundant but EA double-checks |
| `audit.*` | `compute_signal` output columns at last bar | for parity verification in Phase 2 |

### §4.2 What the EA does NOT receive from sidecar (intentional)

- **Entry price.** EA fills at `open_ask` of the next H4 bar — only the broker tape knows the actual fill.
- **Lot size.** EA computes from `Risk_Per_Trade × account_equity / sl_distance_price / pip_value`. Sidecar doesn't know account equity.
- **Stop-loss price.** EA computes `entry_fill_price - sl_distance_price` after fill.
- **R-multiple at entry.** Derived from the SL distance — same R as the sl_distance_price provides.

### §4.3 What the EA validates on read

1. `schema_version` major == 1.
2. `config_hash` == `Expected_Config_Hash` input parameter.
3. `pair` matches at least one symbol the EA is configured to trade.
4. `entry_bar_open_utc` is in the future (or within a small grace window) — stale signals (sidecar published well after the entry bar opened) are moved to `signals_failed/` with reason `stale_signal`.
5. `signal_id` not already in `ea_positions.json` (idempotent — sidecar restart can't double-fire).

On any validation failure: move to `signals_failed/` with reason field, log_event. On success: move to `signals_processed/` after entry-order placed (or after news-delay-discard / equity-guard-block).

---

## §5 Unit test plan

### §5.1 Sidecar tests (`tests/sidecar/`)

Per dispatch §4.1, with concrete specifications:

1. **`test_signal_runner_byte_identity`** — Load 10 historical signal events from `results/l_arc_10_v3_0_2_utc_rerun/trade_ledger_utc.parquet`; for each, slice the corresponding UTC H4 + D1 panel from `data/cache/utc/<PAIR>.parquet` (windowed to end at the signal bar); invoke `signal_runner.run_signal`; assert returned dict matches the trade ledger's signal-time fields byte-identically (timestamp UTC, pair, atr14, L1_value, L0_value).
2. **`test_signal_emitter_schema`** — jsonschema validation, atomic write semantics (verify `signals_out/.tmp_*` then rename pattern; corrupted partial-write must not be visible to a directory poll).
3. **`test_signal_emitter_filename_determinism`** — Same input → same filename.
4. **`test_state_manager_roundtrip`** — write → read → write → read; assert byte-stability of serialised JSON (sorted keys, deterministic ordering).
5. **`test_state_manager_corruption_recovery`** — corrupted `sidecar_state.json` (truncated, bad JSON, missing required keys): sidecar should refuse to start with explicit error, NOT silently default-initialise (per dispatch §1.7 explicit "log gap").
6. **`test_mt5_data_fetcher_mock`** — mock `MetaTrader5.copy_rates_from_pos`, assert call signature (`PAIR_STRING, mt5.TIMEFRAME_H4, 0, 300` for H4; `..., TIMEFRAME_D1, 0, 100` for D1), assert returned DataFrame has expected columns + dtypes + ordering.
7. **`test_config_hash_stability`** — same winning_config → same hash; whitespace/comment change in YAML → same hash (parsed-content-based, not raw-file).
8. **`test_compute_next_utc_h4_close`** — boundary cases (exactly on an anchor, 1s before, 1s after, day-rollover, year-rollover).

### §5.2 EA tests (MT5 Strategy Tester)

12 scenarios per dispatch §4.2. Each test is a fake-signal harness — a Python helper writes synthetic signal JSON into the ST sandbox's `signals_out/`, ST runs the EA against `_FXT` tick history for that pair, parity checker compares the EA's resulting trade log against the expected outcome.

The 12 scenarios are listed in the dispatch. We commit to building all 12; the build closure will include a status table per scenario.

### §5.3 Determinism

Sidecar produces byte-identical signal output across runs on the same historical panel slice (`random_state=42` not used in DLR — purely deterministic anyway, but locked in `config.py` for hygiene; sorted file emission order; sorted dict keys in JSON; `lineterminator='\n'` for any CSV the sidecar writes).

EA: per-symbol arrays are iterated by sorted symbol order; `trade_log.csv` is append-only; `ea_positions.json` is written via atomic rename. Two parallel runs of ST on the same fixture must produce identical trade logs.

---

## §6 Python-vs-EA divergences (explicit enumeration)

The dispatch flags one divergence ("SL suppression") that has no Python analogue. We instead enumerate the three real divergences the build must document for Phase 2 parity measurement.

### §6.1 Trail-exit fill timing

| Layer | Trail-exit fires when | Fill recorded at |
|---|---|---|
| Python canonical (`sl_partial_close_1r_runner_trail.py`) | `close_bid <= peak - r_atr` AND `bar_ordinal > tp1_bar_ordinal` | `at_close` — engine treats the close price of the trail bar as fill |
| Prior live EA (`reference/arc_10_ea/Arc10_Includes/Arc10_Exit.mqh`) | Same detection condition on bar close | Market order at OnBar wake time — typically the next bar's first tick after the close |
| **This phase's EA (dispatch §2.5)** | Same detection condition on bar close | **Next bar's open** — explicitly queued |

The dispatch picks "next bar open". This is the most realistic of the three (EA cannot transact at the closed-bar's close price — by the time the bar has closed, the market has moved on). Phase 2 will measure the slippage from this in R-units.

### §6.2 TP1 partial fill price

| Layer | Trigger | Fill price |
|---|---|---|
| Python canonical | `bar.high_bid >= entry + r_atr` (intra-bar) | `entry + r_atr` (level fill — idealised) |
| EA | `Bid >= entry + sl_distance_price` (intra-tick) | actual fill returned by `trade.PositionClosePartial` |

The Python canonical idealises the fill at the TP1 level. The EA gets whatever the broker fills (usually within a tick of the level, but not guaranteed). Phase 2 measures this drift.

### §6.3 Original SL during runner phase

Closure §4.7 step 3: "Runner (remaining 50%) is trailed: stop tracks `H4 peak-close − 1.0R`. Updates on bar close only."

Read literally, this implies the original SL is REPLACED by the trail. The canonical code is more subtle — `apply_to_order` returns `{}` (no broker SL modification), so the original SL set by the engine's intra-bar SL check stays in place during stage 2. The trail level is in EA memory only; the EA decides to close when the trail is hit.

**For the EA:** original SL stays attached to the broker position throughout both stages. The trail is EA-side state. If the trail level moves above the original SL price (which it always does once the position is profitable enough to even consider trailing), the EA also calls `OrderModify` to move the broker SL up to the trail level so the broker stops the position if the EA crashes between bar closes. This matches prior `Arc10_Exit.mqh` behaviour. The "SL suppression" the dispatch §"Read-first" obliquely references is the canonical Python NOT actively modifying the broker SL during runner phase — the EA improves on this by ratcheting the broker SL up to the trail level.

This is a Python-vs-live improvement, not a divergence — Phase 2 will record it as such.

---

## §7 EA restart recovery algorithm

Per dispatch §2.7, with concrete steps. Fires in `OnInit` after sidecar-heartbeat check.

```
on_init():
  log("EA init — magic=" + Magic_Number)
  load_cached_state("ea_positions.json")       # map: ticket → cached_pos_state
  broker_positions = positions_filtered_by_magic(Magic_Number)

  for broker_pos in broker_positions:
    if broker_pos.ticket in cached_state:
      reconcile_cached_with_broker(cached_state[broker_pos.ticket], broker_pos)
      log_event("recovery_reconciled", ticket=broker_pos.ticket)
    else:
      reconstruct_position_state(broker_pos)
      log_event("recovery_reconstructed", ticket=broker_pos.ticket)

  for cached_ticket in cached_state.keys() - broker_positions.tickets:
    mark_closed_external(cached_state[cached_ticket])
    log_event("recovery_closed_during_downtime",
              ticket=cached_ticket, last_known_state=cached_state[cached_ticket])

  save_state_atomic("ea_positions.json")
```

### §7.1 `reconstruct_position_state`

For each broker position with no cached state:

1. **Entry bar time** = round_down_to_h4_utc(broker_pos.position_open_time) where open_time is the MT5 server time converted to UTC then floored to the nearest 4-hour anchor (00/04/08/12/16/20).
2. **Bar ordinal** = floor((now_utc - entry_bar_time_utc) / 4h).
3. **Fetch H4 bars** from `entry_bar_time` to `now` (inclusive of both) via `CopyRates(symbol, PERIOD_H4, entry_bar_time, now)`.
4. **peak_high_bid** = max over the fetched bars' `high` values (which on 5ers MT5 are bid-anchored).
5. **Check broker deal history** (`HistoryDealsTotal` filtered by symbol + magic + ticket-id-parent-of-position) for any deal of type `DEAL_TYPE_PARTIAL` OR (`DEAL_TYPE_SELL` AND `volume < initial_position_volume`) → if found, set `tp1_fired = true`, `tp1_bar_ordinal` from the deal's time, `partial_close_price` from the deal's price.
6. **Compute trail level** if `tp1_fired`: `trail_level = peak_high_bid - r_atr` where `r_atr = 1R in price units = sl_distance_price = sl_atr_mult × ATR_at_entry`. Per `core/sim/exit_policies/_base.py:95` (canonical Python definition: `ExitPolicyContext.r_atr = sl_atr_mult * atr_at_entry`). If recovering from broker state with no cached signal, back-compute via `r_atr = entry_price - sl_price` (direct sl_distance — do NOT divide by sl_atr_mult). The earlier draft of this doc had `r_atr = (entry_price - sl_price) / 3.5` which treated r_atr as 1×ATR instead of 1R; that formula was the source of the EA bug fixed in commit [post-9351]. Correct: TP1 fires at `entry + r_atr` = `entry + sl_distance` (= entry + 3.5×ATR at the current config's sl_atr_mult=3.5); trail level is `peak - r_atr` = `peak - sl_distance`.
7. **Modify broker SL** to `max(broker_current_sl, trail_level)` so a crash doesn't expose the position above the trail.
8. **Time exit check** — if `bar_ordinal >= Time_Exit_Bars`, queue immediate close.

### §7.2 Edge cases

- Broker position with magic match but no `signals_processed/<id>.json`: log `recovery_orphan_position`, reconstruct per §7.1 anyway (the position was opened by this EA on a prior incarnation; the signal file was hand-deleted or lost).
- Cached state ticket no longer in broker: log `recovery_closed_during_downtime` with the cached state's last-known fields (mfe, mae, bar_ordinal). This is informational; nothing to do.
- Sidecar heartbeat stale on init: EA still recovers existing positions and continues managing them; only new entries are blocked (dispatch §2.2). Recovery is independent of sidecar state.

---

## §8 News filter integration

### §8.1 Feed choice

**Feed:** ForexFactory weekly XML — `https://nfs.faireconomy.media/ff_calendar_thisweek.xml`.

**Rationale:**
- Same feed used by prior `reference/arc_10_ea/Arc10_Includes/Arc10_News.mqh` — proven against this broker / VPS combo. Avoid introducing a new feed in Phase 1.
- Free, no API key, no rate limits, weekly schedule is enough granularity for ±2-min blackout windows.
- ForexFactory red-folder events are the prop-firm-relevant signal; CPI/NFP/FOMC are all marked red.
- XML schema is stable (used unchanged for 5+ years).

**Alternative considered:** the ForexFactory JSON Calendar API requires an account and API key, and its weekly endpoint has a stricter rate limit. Not worth the friction for what is a redundant data source.

### §8.2 Refresh schedule

Dispatch §2.8: "Pull every 4 hours". Prior EA: "Refresh: once daily". We adopt the dispatch's 4-hour cadence because there is a real failure mode where overnight calendar adjustments (e.g., FOMC schedule change on a Wednesday) need to be picked up before the Asia session.

Implementation: track `last_news_pull_time_utc`; on every OnTick, if `now_utc - last_pull > 4h`, attempt pull. On WebRequest failure, retry up to 3 times in the current 4-hour window with ≥30s spacing (matches prior EA pattern); after 3 failures, back off until next 4-hour window; log alert.

### §8.3 Blackout policy

Per dispatch §2.8:

- For each event in the cached calendar with `impact == "High"`:
  - If event's currency is in `pair`'s base OR quote currencies, AND the signal's `entry_bar_open_utc` falls within `[event_time - 120s, event_time + 120s]`:
    - **Delay** entry to `event_time + 120s + 5s buffer`.
    - If `delay_target > signal.entry_bar_open_utc + 1h`: **discard** signal (`reason="news_delay_exceeds_max"`, move to `signals_failed/`).

### §8.4 Tester-mode bypass

`Enable_News_Filter` defaults to `true` in production. In MT5 Strategy Tester (`MQLInfoInteger(MQL_TESTER) != 0`), the filter is wholly disabled — no WebRequest attempts, blackout check returns `false`. This matches the prior EA and matches Python sim behaviour (sim has no news filter; the filter is a venue-compliance layer for live, not a strategy component).

---

## §9 IPC directory layout

Per dispatch §3, with explicit paths. All directories live under a single `sidecar_root/` that both processes can reach (on the Contabo VPS, this is the MT5 `MQL5/Files/` directory or a junction to it; on dev, a temp dir under `tests/sidecar/fixtures/`):

```
<sidecar_root>/
  signals_out/             # sidecar writes; EA polls
    <pair>_<bar_iso>.json
  signals_processed/       # EA moves here after successful entry
    <pair>_<bar_iso>.json
  signals_failed/          # EA moves here on validation/news/equity-guard reject
    <pair>_<bar_iso>.json    # body unchanged; sidecar/operator inspect via filename + log_event
  sidecar.heartbeat        # sidecar writes; EA reads
  ea.heartbeat             # EA writes; watchdog reads
  sidecar_state.json       # sidecar internal
  ea_positions.json        # EA internal
  trade_log.csv            # EA appends
```

**Atomic moves only.** All file moves use same-volume rename (`os.rename` Python / `FileMove` MQL5) so the move is atomic at the filesystem level. No cross-volume copies (those are not atomic).

**Polling cadence (EA).** Once per OnTick after the H4-bar-close event. Lower-bound by a 5-second poll interval flag to avoid pathological tick-rate file-system thrashing in Strategy Tester.

---

## §10 NSSM + Task Scheduler (dispatch §5)

`deployment/ops/`:

- `nssm_sidecar.bat` — `nssm install Arc10Sidecar "<python_exe>" "-m deployment.sidecar <args>"`; sets AppDirectory, AppStdout, AppStderr to log files in `deployment/sidecar/logs/`; auto-restart on crash with 5s delay.
- `watchdog.ps1` — every 30s: read `sidecar.heartbeat`, parse `last_heartbeat_utc`, if older than 2 min trigger `nssm restart Arc10Sidecar` and post alert to webhook. Idempotent — does not restart if a restart is already in progress.
- `uninstall.bat` — `nssm stop Arc10Sidecar`, `nssm remove Arc10Sidecar confirm`.

We will NOT install or run any of these in Phase 1 — they ship as artefacts for the user / Phase 2 to invoke. Live deployment is out of scope per dispatch §7.

---

## §11 Open questions / assumptions to flag

These are decisions we have made for the build but want the user to confirm before code begins (or to confirm by silence):

1. **Trail exit timing = next bar open (not at-close fill).** Per §6.1 above and the dispatch's explicit `queue_exit_at_next_bar_open` instruction. This is a deliberate Python-vs-live divergence; Phase 2 will measure the slippage. ✓ matches dispatch.
2. **Risk default = 0.0043 (not 0.004336).** EA input is `0.0043` for human-readability; sidecar `config_hash` uses canonical `0.004336`. EA telemetry logs both so audit can recover. ✓ matches dispatch.
3. **News refresh = every 4 hours (not daily).** Per dispatch §2.8 against prior EA's daily. ✓ matches dispatch.
4. **Reuse from prior EA = exit/news/state/telemetry includes only.** Signal logic stripped. Need user sign-off that re-using these copies into `deployment/ea/include/` (rather than referencing across branches) is acceptable.
5. **`reference/arc_10_ea/` left untouched.** Dispatch §7 says "no modification of files under `signals/`, `core/sim/exit_policies/`, or `configs/l_arc_10_v3.0.2*`" — does not list `reference/arc_10_ea/`. We propose: leave `reference/arc_10_ea/` as a frozen reference; new EA lives at `deployment/ea/` per dispatch §2.1.
6. **Sidecar's H4 close validation aborts on broker DST drift.** If 5ers MT5 starts emitting bars at non-UTC anchors (e.g., a server config change or DST-affected feed), the sidecar aborts at startup rather than silently producing signals on the wrong timestamps. The watchdog will then fast-cycle restart it — that's a feature, not a bug; it makes the failure mode loud.
7. **EA magic number.** Need a value not already in use on the live 5ers MT5 instance. We will pick `1010202601` (arc 10, version 02, 2026, increment 01) but defer to the user if there's an existing convention.
8. **MT5 Strategy Tester sidecar harness.** Since the real sidecar requires Windows MT5 connection, ST tests for the EA need a Python helper that writes synthetic signals into the ST sandbox's `signals_out/` during the ST run. We will build this as `tests/ea/fake_sidecar.py`. Confirmable: ST sandboxes file IO under `MQL5/Tester/Common/Files/` per MT5 docs — verify at first scenario test.

---

## §12 Out of scope (dispatch §7, restated)

- Live deployment, 5ers Demo connection.
- Modifications under `signals/`, `core/sim/exit_policies/`, `configs/l_arc_10_v3.0.2*`.
- Phase 2 parity validation (separate dispatch).
- Signal-related logic in EA (the whole point of the redesign).
- Modifications to `reference/arc_10_ea/` (frozen reference).

---

## §13 Deliverables checklist (dispatch §6, this phase only)

- [x] `phase_1_build_intent.md` (this file)
- [ ] `deployment/sidecar/` complete per §2 above
- [ ] `deployment/ea/Arc10_DLR_Sidecar_EA.mq5` + 8 includes per §3 above
- [ ] `tests/sidecar/` all passing per §5.1
- [ ] 12 ST scenario tests all passing per §5.2
- [ ] `deployment/ops/` scripts complete per §10
- [ ] `deployment/README.md` — setup, directory layout, troubleshooting
- [ ] PR against `main` with diff summary + test outputs

---

End of intent doc. Code begins after user review.
