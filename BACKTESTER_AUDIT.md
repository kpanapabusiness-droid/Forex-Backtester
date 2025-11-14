# Forex Backtester — READ-ONLY Audit Report
**Version:** v2.0.0 (as of codebase inspection)  
**Date:** 2024  
**Scope:** Complete entry/exit/confirmation logic, config schema, accounting, outputs

---

## A) DECISION PIPELINE NARRATIVE

### 1. Entry/Exit Decision Flow

#### 1.1 Signal Generation (`core/signal_logic.py:158-553`)

**Input:** DataFrame with indicator columns (`c1_signal`, `c2_signal`, `baseline_signal`, `volume_signal`, `baseline` price series, `atr`)

**Output:** DataFrame with `entry_signal` ({-1,0,+1}) and `exit_signal`/`exit_signal_final` ({0,1})

**Entry Signal Logic:**

1. **C1 Detection** (`core/signal_logic.py:132-156`): Auto-detects C1 column by searching for `c1_signal`, `c1`, `signal_c1`, or columns with values in {-1,0,+1}

2. **Catalyst Detection** (`core/signal_logic.py:340-380`):
   - **C1 Catalyst**: C1 signal != 0 on bar `i`
   - **Baseline Catalyst** (if `rules.allow_baseline_as_catalyst=true`): Baseline cross detected AND last C1 signal within `bridge_too_far_days` (default 7) (`core/signal_logic.py:347-380`)
     - Bridge-Too-Far check: `bars_since_c1 >= bridge_too_far_days` → block entry (`core/signal_logic.py:354-360`)
     - Baseline cross detection: price crosses baseline from wrong side to right side (`core/signal_logic.py:363-379`)

3. **Filter Validation** (`core/signal_logic.py:36-98`):
   - **C2**: Must match C1 direction (`core/signal_logic.py:56-58`)
   - **Volume**: Must be +1 (`core/signal_logic.py:61-63`)
   - **Baseline Direction**: Price must be on correct side of baseline (`core/signal_logic.py:65-78`)
   - **Pullback Rule** (if enabled): Price must not be > 1×ATR from baseline on both current AND previous bar (`core/signal_logic.py:80-96`)

4. **One-Candle Rule** (`core/signal_logic.py:429-472`):
   - **Definition**: If filters fail on bar `i` (C1 signal present), store as pending
   - **Recovery**: On bar `i+1`, re-check failed filters; if all pass, emit entry signal on `i+1`
   - **NOT applied** to baseline-catalyst entries (`core/signal_logic.py:429`)
   - **Implementation**: Uses `pending_entry`, `pending_direction`, `pending_failed_filters` state (`core/signal_logic.py:242-245`)

5. **Pullback Rule** (`core/signal_logic.py:80-96`):
   - **Definition**: If price > 1×ATR from baseline on signal bar AND previous bar, block entry
   - **Recovery**: Wait one bar; if price returns to ≤ 1×ATR, enter (handled via One-Candle mechanism)
   - **Mutually exclusive** with One-Candle Rule (`core/signal_logic.py:207-208`)

6. **Entry Timing** (`core/backtester.py:937-947`):
   - **Signal bar**: `entry_signal` on bar `i` means entry at **close of bar `i`** (`c_i`)
   - **No off-by-one delay**: Entry happens on the same bar as signal
   - **Duplicate-open guard**: Prevents multiple entries on same bar/side (`core/backtester.py:692-693, 941-945`)

#### 1.2 Exit Signal Logic (`core/signal_logic.py:483-523`)

**Exit Conditions (priority order):**

1. **C1 Reversal** (`exit.exit_on_c1_reversal=true`, default): C1 signal flips opposite to position (`core/signal_logic.py:489-496`)
2. **Baseline Cross** (`exit.exit_on_baseline_cross=true`): Price crosses baseline against position (`core/signal_logic.py:498-505`)
3. **Stop Loss Hit**: Price touches `current_sl` (`core/signal_logic.py:507-513`)

**Exit Signal Output:**
- `exit_signal_final=1` when exit triggered
- Legacy `exit_signal` also set for backward compatibility (`core/signal_logic.py:531-536`)

#### 1.3 Trade Execution (`core/backtester.py:589-1049`)

**Entry Execution** (`core/backtester.py:937-1036`):
- **Entry Price**: Close of signal bar (`c_i`) (`core/backtester.py:947`)
- **ATR at Entry**: ATR value from signal bar (`atr_i`) (`core/backtester.py:949`)
- **TP1/SL Calculation**: Using entry ATR (immutable) (`core/backtester.py:985-986`)
  - TP1 = entry_price ± (1.0 × ATR_entry)
  - SL = entry_price ∓ (1.5 × ATR_entry)
- **Immutable Fields Set**: `tp1_at_entry_price`, `sl_at_entry_price` set at entry, never mutated (`core/backtester.py:1010-1011`)

**Position Management** (`core/backtester.py:724-842`):

1. **Trailing Stop Activation** (`core/backtester.py:735-741`):
   - Activates when close-based move from entry ≥ 2×ATR (entry ATR, immutable)
   - Formula: `move_cl = direction_int * (close_i - entry_price)`; `threshold = 2.0 * atr_entry`
   - Initial TS level: `close_i ∓ (1.5 × atr_entry)` (`core/backtester.py:650-655`)

2. **Trailing Stop Ratchet** (`core/backtester.py:743-746`):
   - Updates TS level each bar: `trail_level = close_i ∓ (1.5 × atr_entry)`
   - Uses `better_stop()` to only move in favorable direction (`core/backtester.py:657-662`)

3. **Effective Stop Calculation** (`core/backtester.py:748-753`):
   - Priority: `sl_price → breakeven (if TP1 hit) → trailing_stop`
   - `better_stop()` selects most favorable level for position

4. **Exit Priority** (`core/backtester.py:760-842`):

   **Pre-TP1:**
   - Priority 1: System exits (C1 reversal) → exit at entry price (scratch) (`core/backtester.py:825-836`)
   - Priority 2: Hard stop loss → exit at stop level (`core/backtester.py:838-842`)
   - TP1 check happens first; if TP1 hits, SL moves to BE immediately (`core/backtester.py:761-778`)

   **Post-TP1:**
   - Priority 1: Trailing stop → exit at TS level (`core/backtester.py:787-799`)
   - Priority 2: C1 reversal → exit at close price (`core/backtester.py:801-812`)
   - Priority 3: Breakeven → exit at entry price (`core/backtester.py:814-822`)

5. **Intrabar Priority** (`core/backtester.py:151-159, 612-614, 762-773`):
   - Configurable: `execution.intrabar_priority` ∈ {"tp_first", "sl_first", "best", "worst"}
   - Default: "tp_first" → check TP1 before SL when both possible on same bar

6. **W/L/S Classification** (`core/backtester.py:875-884`):
   - **WIN**: `tp1_hit = True` (regardless of runner outcome)
   - **LOSS**: `tp1_hit = False` AND `exit_reason = "stoploss"`
   - **SCRATCH**: `tp1_hit = False` AND `exit_reason != "stoploss"`

7. **Stop Recording** (`core/backtester.py:899-904, core/backtester_helpers.py:336-380`):
   - `sl_at_exit_price`: Stop level in effect at exit (BE, TS, or original SL)
   - Set via `finalize_trade_row()` with `current_stop_price_at_exit` parameter
   - Immutable entry fields (`tp1_at_entry_price`, `sl_at_entry_price`) never mutated

#### 1.4 Re-entry Rules (`core/signal_logic.py:316-332, core/backtester.py:906-911`)

**Continuation Trades** (`engine.allow_continuation=true`):
- Allowed after exit if baseline not crossed since original entry
- C1 must flip back to original direction
- Volume and baseline distance checks **skipped** (`core/signal_logic.py:326`)
- Thread ID tracks related trades (`core/backtester.py:695-697, 1032-1036`)

**Duplicate-Open Guard** (`core/backtester.py:692-693, 941-945`):
- Prevents multiple entries on same bar/side/instrument
- Policy: `engine.duplicate_open_policy` ∈ {"block", "allow"}
- Default: "block"

---

### 2. Risk, Cost, and Accounting

#### 2.1 Position Sizing (`core/backtester.py:970-982`)

**Formula:**
```
risk_money = equity_balance × risk_pct_eff
sl_dist_pips = sl_atr_mult × atr_pips
lots_total = risk_money / (sl_dist_pips × pip_value_per_lot)
```

**Parameters:**
- `risk.risk_per_trade` (default 0.02 = 2%) (`core/backtester.py:621`)
- `entry.sl_atr` (default 1.5) (`core/backtester.py:608`)
- `risk_pct_eff`: May be reduced by DBCVIX filter (`core/backtester.py:961-968`)
- Minimum lot safeguard: 0.01 if calculated ≤ 0 (`core/backtester.py:978-979`)

**Lot Split:**
- `lots_half = lots_total / 2.0` (for TP1 partial exit)
- `lots_runner = lots_total - lots_half` (remaining position)

#### 2.2 Spreads/Slippage/Commission (`core/backtester.py:512-538, 546-581`)

**Spread Model** (`core/backtester.py:512-538`):
- **PnL-only**: Spreads affect PnL calculation, NOT entry/exit decisions or trade counts
- **Resolution order**:
  1. Per-bar override: `row["spread_pips"]` if present
  2. Per-pair: `spreads.per_pair[pair]`
  3. ATR-multiplier: `spreads.atr_mult × (atr_pips)` if `mode="atr_mult"`
  4. Default: `spreads.default_pips`

**PnL Calculation with Spreads** (`core/backtester.py:546-581`):
- Entry fill: `entry_mid ± (spread_pips × pip_size) / 2`
- TP1 fill: `tp1_mid ∓ (spread_pips × pip_size) / 2`
- Exit fill: `exit_mid ∓ (spread_pips × pip_size) / 2`
- If TP1 hit: PnL = (pips_half × pip_value × lots_half) + (pips_runner × pip_value × lots_runner)
- If no TP1: PnL = (pips_full × pip_value × lots_total)

**Slippage** (`core/backtester.py:664-672, 463-483`):
- Applied only to trailing stop fills (via `fills.slippage.pips`)
- Exit price adjusted: `final_stop_price ∓ slippage_pips × pip_size`

**Invariant**: Spreads/slippage do NOT change trade counts or exit reasons (tested in `tests/test_audit_immutability.py:266-484`)

#### 2.3 Equity Curve (`core/backtester.py:924-932, 1271-1289`)

**Equity Tracking:**
- Enabled by `tracking.in_sim_equity` (default true)
- Snapshot per bar: `equity = starting_balance + cumulative_realized_pnl` (`core/backtester.py:928-932`)
- Written to `results/{output_dir}/equity_curve.csv` with columns: `date`, `equity`, `peak`, `drawdown` (`core/backtester.py:1281-1287`)

**Drawdown Calculation:**
- `peak = equity.cummax()`
- `drawdown = equity - peak`

#### 2.4 Audit Invariants

**Immutable Fields** (set at entry, never mutated):
- `tp1_at_entry_price` (`core/backtester.py:1010`)
- `sl_at_entry_price` (`core/backtester.py:1011`)

**Dynamic Fields** (updated during position):
- `current_sl`: Current stop level (BE/TS/SL)
- `ts_active`: Trailing stop active flag
- `ts_level`: Current trailing stop level

**Finalized at Exit:**
- `sl_at_exit_price`: Stop level in effect at exit (`core/backtester_helpers.py:336-380`)

---

### 3. Config (YAML) Contract

**Schema Location:** `validators_config.py:144-242`

**Required Keys:**
- `pairs`: List[str] (must contain "_", e.g., "EUR_USD")
- `indicators`: Indicators object (requires `c1` name)
- `rules`: Rules object
- `exit`: Exit object
- `tracking`: Tracking object

**Optional Keys with Defaults:**

| Key Path | Type | Default | Module | Source |
|----------|------|---------|--------|--------|
| `strategy_version` | str | `"forex_backtester_v1.9.7"` | engine | `validators_config.py:146` |
| `timeframe` | Literal | `"D"` | engine | `validators_config.py:150` |
| `data_dir` | Optional[str] | `None` | engine | `validators_config.py:151` |
| `indicator_params` | Dict | `{}` | indicators | `validators_config.py:155` |
| `engine.allow_continuation` | bool | `True` | engine | `validators_config.py:68` |
| `engine.duplicate_open_policy` | Literal | `"block"` | engine | `validators_config.py:69` |
| `continuation.allow_continuation` | bool | `False` | engine | `validators_config.py:73` |
| `continuation.skip_volume_check` | bool | `False` | engine | `validators_config.py:74` |
| `continuation.skip_pullback_check` | bool | `False` | engine | `validators_config.py:75` |
| `continuation.block_if_crossed_baseline_since_entry` | bool | `False` | engine | `validators_config.py:76` |
| `indicators.use_c2` | bool | `False` | c2 | `validators_config.py:40` |
| `indicators.use_baseline` | bool | `False` | baseline | `validators_config.py:41` |
| `indicators.use_volume` | bool | `False` | volume | `validators_config.py:42` |
| `indicators.use_exit` | bool | `False` | exit | `validators_config.py:43` |
| `indicators.c2` | Optional[str] | `None` | c2 | `validators_config.py:46` |
| `indicators.baseline` | Optional[str] | `None` | baseline | `validators_config.py:47` |
| `indicators.volume` | Optional[str] | `None` | volume | `validators_config.py:48` |
| `indicators.exit` | Optional[str] | `None` | exit | `validators_config.py:49` |
| `rules.one_candle_rule` | bool | `False` | entry | `validators_config.py:53` |
| `rules.pullback_rule` | bool | `False` | entry | `validators_config.py:54` |
| `rules.bridge_too_far_days` | int | `7` | entry | `validators_config.py:55` |
| `rules.allow_baseline_as_catalyst` | bool | `False` | entry | `validators_config.py:56` |
| `exit.use_trailing_stop` | bool | `True` | exit | `validators_config.py:60` |
| `exit.move_to_breakeven_after_atr` | bool | `True` | exit | `validators_config.py:61` |
| `exit.exit_on_c1_reversal` | bool | `True` | exit | `validators_config.py:62` |
| `exit.exit_on_baseline_cross` | bool | `False` | exit | `validators_config.py:63` |
| `exit.exit_on_exit_signal` | bool | `False` | exit | `validators_config.py:64` |
| `entry.sl_atr` | float | `1.5` | risk | `core/backtester.py:608` |
| `entry.tp1_atr` | float | `1.0` | risk | `core/backtester.py:609` |
| `entry.trail_after_atr` | float | `2.0` | exit | `core/backtester.py:610` |
| `entry.ts_atr` | float | `1.5` | exit | `core/backtester.py:611` |
| `execution.intrabar_priority` | str | `"tp_first"` | execution | `core/backtester.py:613` |
| `spreads.enabled` | bool | `False` | costs | `validators_config.py:30` |
| `spreads.default_pips` | float | `0.0` | costs | `validators_config.py:31` |
| `spreads.per_pair` | Dict | `{}` | costs | `validators_config.py:32` |
| `spreads.mode` | Literal | `"fixed"` | costs | `validators_config.py:33` |
| `spreads.atr_mult` | float | `0.0` | costs | `validators_config.py:34` |
| `filters.dbcvix.enabled` | bool | `False` | risk | `validators_config.py:103` |
| `filters.dbcvix.mode` | Literal | `"reduce"` | risk | `validators_config.py:104` |
| `filters.dbcvix.threshold` | Optional[float] | `None` | risk | `validators_config.py:105` |
| `filters.dbcvix.reduce_risk_to` | float | `1.0` | risk | `validators_config.py:106` |
| `filters.dbcvix.source` | Literal | `"synthetic"` | risk | `validators_config.py:107` |
| `filters.dbcvix.csv_path` | str | `"data/external/dbcvix_synth.csv"` | risk | `validators_config.py:108` |
| `risk.starting_balance` | float | `10000.0` | risk | `validators_config.py:121` |
| `risk.risk_per_trade_pct` | float | `2.0` | risk | `validators_config.py:122` |
| `risk.risk_per_trade` | float | `0.02` | risk | `core/backtester.py:621` (uses `risk_per_trade_pct/100` or direct) |
| `risk.account_ccy` | str | `"AUD"` | risk | `core/backtester.py:620` |
| `risk.fx_quotes` | Dict | `{}` | risk | `core/backtester.py:622` |
| `cache.enabled` | bool | `True` | cache | `validators_config.py:88` |
| `cache.dir` | str | `"cache"` | cache | `validators_config.py:89` |
| `cache.format` | Literal | `"parquet"` | cache | `validators_config.py:90` |
| `cache.scope_key` | Optional[str] | `None` | cache | `validators_config.py:91` |
| `cache.roles` | Optional[List] | `None` | cache | `validators_config.py:93` |
| `validation.enabled` | bool | `True` | validation | `validators_config.py:97` |
| `validation.fail_fast` | bool | `True` | validation | `validators_config.py:98` |
| `validation.strict_contract` | bool | `False` | validation | `validators_config.py:99` |
| `output.results_dir` | str | `"results"` | output | `validators_config.py:117` |
| `outputs.dir` | str | `"results"` | output | `core/backtester.py:1067` (new format) |
| `outputs.write_trades_csv` | bool | `True` | output | `core/backtester.py:1790` |
| `tracking.track_win_loss_scratch` | bool | `True` | tracking | `validators_config.py:80` |
| `tracking.track_roi` | bool | `True` | tracking | `validators_config.py:81` |
| `tracking.track_drawdown` | bool | `True` | tracking | `validators_config.py:82` |
| `tracking.in_sim_equity` | bool | `True` | tracking | `validators_config.py:83` |
| `tracking.verbose_logs` | bool | `False` | tracking | `validators_config.py:84` |
| `date_range.start` | Optional[str] | `None` | engine | `validators_config.py:126` |
| `date_range.end` | Optional[str] | `None` | engine | `validators_config.py:127` |
| `date_from` | Optional[str] | `None` | engine | `core/backtester.py:1148` (legacy) |
| `date_to` | Optional[str] | `None` | engine | `core/backtester.py:1149` (legacy) |
| `walk_forward.start` | Optional[str] | `None` | wfo | `validators_config.py:132` |
| `walk_forward.end` | Optional[str] | `None` | wfo | `validators_config.py:133` |
| `walk_forward.train_years` | int | `3` | wfo | `validators_config.py:134` |
| `walk_forward.test_years` | int | `1` | wfo | `validators_config.py:135` |
| `walk_forward.step_years` | int | `1` | wfo | `validators_config.py:136` |
| `monte_carlo.enabled` | bool | `False` | mc | `validators_config.py:10` |
| `monte_carlo.iterations` | int | `300` | mc | `validators_config.py:11` |
| `monte_carlo.horizon` | str | `"oos"` | mc | `validators_config.py:12` |

**Config Loading** (`validators_config.py:213-232`):
- Loads via `load_and_validate_config(path)` → `validate_config(dict)` → Pydantic `Config` model
- Raises `ValueError` with multi-line errors on validation failure

**Derived/Implicit Defaults:**
- `entry.sl_atr`: Defaults to 1.5 if not in config (`core/backtester.py:608`)
- `entry.tp1_atr`: Defaults to 1.0 (`core/backtester.py:609`)
- `entry.trail_after_atr`: Defaults to 2.0 (`core/backtester.py:610`)
- `entry.ts_atr`: Defaults to 1.5 (`core/backtester.py:611`)
- `risk.risk_per_trade`: Uses `risk_per_trade_pct / 100.0` if `risk_per_trade` not present (`core/backtester.py:621`)

---

### 4. Outputs & Paths

**Output Directory Resolution** (`core/backtester.py:1064-1071`):
- Priority: `results_dir` param → `outputs.dir` → `output.results_dir` → `"results"`
- Created via `ensure_results_dir()` (`core/backtester.py:1071`)

**Files Written:**

1. **`trades.csv`** (`core/backtester.py:1769-1888`):
   - Columns: `TRADES_COLS` (see `core/backtester.py:70-110`)
   - Written via `write_trades_csv_with_diagnostics()` with atomic write (temp → rename)
   - Gated by `outputs.write_trades_csv` flag (default true)

2. **`summary.txt`** (`core/backtester.py:1325-1384`):
   - Human-readable summary with trade counts, win rates, ROI
   - Generated via `summarize_results()` (`core/utils.py:464-560`) or fallback

3. **`equity_curve.csv`** (`core/backtester.py:1286-1387`):
   - Columns: `date`, `equity`, `peak`, `drawdown`
   - Written only if `tracking.in_sim_equity=true`
   - Generated from per-bar equity snapshots (`core/backtester.py:924-932`)

**Cache Keys** (`indicators_cache.py` referenced, `core/backtester.py:405-407`):
- Key parts: `(pair, timeframe, role, name, params_hash, data_hash, scope_key)`
- Format: `cache.{format}` (parquet/feather)
- Path: `{cache.dir}/{parts_path}/{key}.{format}`

**Cache Control:**
- Env var: `FB_NO_CACHE=1` disables caching
- Config: `cache.enabled=false` disables caching
- Both checked (`core/backtester.py:360`)

---

### 5. Ambiguities or Conflicts

1. **Risk Config Key Mismatch**:
   - Schema defines `risk.risk_per_trade_pct` (float, 2.0) (`validators_config.py:122`)
   - Code reads `risk.risk_per_trade` (decimal, 0.02) (`core/backtester.py:621`)
   - **Resolution**: Code accepts both; if `risk_per_trade` present, use it; else use `risk_per_trade_pct / 100.0`

2. **Output Directory Keys**:
   - Legacy: `output.results_dir` (`validators_config.py:117`)
   - New: `outputs.dir` (`core/backtester.py:1067`)
   - **Resolution**: Code checks both; `outputs.dir` takes precedence

3. **Date Range Keys**:
   - Legacy: `date_from`, `date_to` (top-level) (`core/backtester.py:1148-1149`)
   - New: `date_range.start`, `date_range.end` (`validators_config.py:125-127`)
   - **Resolution**: Code checks both; `date_range` takes precedence if present

4. **Continuation Config Duplication**:
   - `engine.allow_continuation` (default true) (`validators_config.py:68`)
   - `continuation.allow_continuation` (default false) (`validators_config.py:73`)
   - **Resolution**: `engine.allow_continuation` is primary; `continuation.*` sub-keys are for continuation-specific settings

5. **Exit Signal Column Naming**:
   - `exit_signal_final` (new) vs `exit_signal` (legacy) (`core/signal_logic.py:179, 186-187`)
   - **Resolution**: Both set; `exit_signal_final` is canonical

6. **DBCVIX Config Path Resolution** (`core/backtester.py:191-226`):
   - Checks multiple nested paths: `filters.dbcvix`, `risk_filters.dbcvix`, `dbcvix`, `rules.risk_filters.dbcvix`, `risk.dbcvix`
   - **Resolution**: First dict found is used; normalized to standard shape

7. **Data Dependency**:
   - If `spread_pips` column present in data, it overrides config spreads (`core/backtester.py:517-521`)
   - If `spread_col` missing, falls back to config

---

## B) MACHINE-READABLE SCHEMA (JSON)

```json
{
  "schema_version": "2.0.0",
  "keys": [
    {
      "key": "pairs",
      "type": "List[str]",
      "default": null,
      "required": true,
      "module": "engine",
      "source_file": "validators_config.py",
      "source_lines": "149"
    },
    {
      "key": "strategy_version",
      "type": "str",
      "default": "forex_backtester_v1.9.7",
      "required": false,
      "module": "engine",
      "source_file": "validators_config.py",
      "source_lines": "146"
    },
    {
      "key": "timeframe",
      "type": "Literal['D', 'daily', 'H1', 'H4']",
      "default": "D",
      "required": false,
      "module": "engine",
      "source_file": "validators_config.py",
      "source_lines": "150"
    },
    {
      "key": "data_dir",
      "type": "Optional[str]",
      "default": null,
      "required": false,
      "module": "engine",
      "source_file": "validators_config.py",
      "source_lines": "151"
    },
    {
      "key": "indicators.c1",
      "type": "str",
      "default": null,
      "required": true,
      "module": "c1",
      "source_file": "validators_config.py",
      "source_lines": "39"
    },
    {
      "key": "indicators.use_c2",
      "type": "bool",
      "default": false,
      "required": false,
      "module": "c2",
      "source_file": "validators_config.py",
      "source_lines": "40"
    },
    {
      "key": "indicators.use_baseline",
      "type": "bool",
      "default": false,
      "required": false,
      "module": "baseline",
      "source_file": "validators_config.py",
      "source_lines": "41"
    },
    {
      "key": "indicators.use_volume",
      "type": "bool",
      "default": false,
      "required": false,
      "module": "volume",
      "source_file": "validators_config.py",
      "source_lines": "42"
    },
    {
      "key": "indicators.use_exit",
      "type": "bool",
      "default": false,
      "required": false,
      "module": "exit",
      "source_file": "validators_config.py",
      "source_lines": "43"
    },
    {
      "key": "indicators.c2",
      "type": "Optional[str]",
      "default": null,
      "required": false,
      "module": "c2",
      "source_file": "validators_config.py",
      "source_lines": "46"
    },
    {
      "key": "indicators.baseline",
      "type": "Optional[str]",
      "default": null,
      "required": false,
      "module": "baseline",
      "source_file": "validators_config.py",
      "source_lines": "47"
    },
    {
      "key": "indicators.volume",
      "type": "Optional[str]",
      "default": null,
      "required": false,
      "module": "volume",
      "source_file": "validators_config.py",
      "source_lines": "48"
    },
    {
      "key": "indicators.exit",
      "type": "Optional[str]",
      "default": null,
      "required": false,
      "module": "exit",
      "source_file": "validators_config.py",
      "source_lines": "49"
    },
    {
      "key": "indicator_params",
      "type": "Dict[str, Dict[str, Any]]",
      "default": "{}",
      "required": false,
      "module": "indicators",
      "source_file": "validators_config.py",
      "source_lines": "155"
    },
    {
      "key": "rules.one_candle_rule",
      "type": "bool",
      "default": false,
      "required": false,
      "module": "entry",
      "source_file": "validators_config.py",
      "source_lines": "53"
    },
    {
      "key": "rules.pullback_rule",
      "type": "bool",
      "default": false,
      "required": false,
      "module": "entry",
      "source_file": "validators_config.py",
      "source_lines": "54"
    },
    {
      "key": "rules.bridge_too_far_days",
      "type": "int",
      "default": 7,
      "required": false,
      "module": "entry",
      "source_file": "validators_config.py",
      "source_lines": "55"
    },
    {
      "key": "rules.allow_baseline_as_catalyst",
      "type": "bool",
      "default": false,
      "required": false,
      "module": "entry",
      "source_file": "validators_config.py",
      "source_lines": "56"
    },
    {
      "key": "exit.use_trailing_stop",
      "type": "bool",
      "default": true,
      "required": false,
      "module": "exit",
      "source_file": "validators_config.py",
      "source_lines": "60"
    },
    {
      "key": "exit.move_to_breakeven_after_atr",
      "type": "bool",
      "default": true,
      "required": false,
      "module": "exit",
      "source_file": "validators_config.py",
      "source_lines": "61"
    },
    {
      "key": "exit.exit_on_c1_reversal",
      "type": "bool",
      "default": true,
      "required": false,
      "module": "exit",
      "source_file": "validators_config.py",
      "source_lines": "62"
    },
    {
      "key": "exit.exit_on_baseline_cross",
      "type": "bool",
      "default": false,
      "required": false,
      "module": "exit",
      "source_file": "validators_config.py",
      "source_lines": "63"
    },
    {
      "key": "exit.exit_on_exit_signal",
      "type": "bool",
      "default": false,
      "required": false,
      "module": "exit",
      "source_file": "validators_config.py",
      "source_lines": "64"
    },
    {
      "key": "entry.sl_atr",
      "type": "float",
      "default": 1.5,
      "required": false,
      "module": "risk",
      "source_file": "core/backtester.py",
      "source_lines": "608"
    },
    {
      "key": "entry.tp1_atr",
      "type": "float",
      "default": 1.0,
      "required": false,
      "module": "risk",
      "source_file": "core/backtester.py",
      "source_lines": "609"
    },
    {
      "key": "entry.trail_after_atr",
      "type": "float",
      "default": 2.0,
      "required": false,
      "module": "exit",
      "source_file": "core/backtester.py",
      "source_lines": "610"
    },
    {
      "key": "entry.ts_atr",
      "type": "float",
      "default": 1.5,
      "required": false,
      "module": "exit",
      "source_file": "core/backtester.py",
      "source_lines": "611"
    },
    {
      "key": "execution.intrabar_priority",
      "type": "str",
      "default": "tp_first",
      "required": false,
      "module": "execution",
      "source_file": "core/backtester.py",
      "source_lines": "613"
    },
    {
      "key": "engine.allow_continuation",
      "type": "bool",
      "default": true,
      "required": false,
      "module": "engine",
      "source_file": "validators_config.py",
      "source_lines": "68"
    },
    {
      "key": "engine.duplicate_open_policy",
      "type": "Literal['block', 'allow']",
      "default": "block",
      "required": false,
      "module": "engine",
      "source_file": "validators_config.py",
      "source_lines": "69"
    },
    {
      "key": "continuation.allow_continuation",
      "type": "bool",
      "default": false,
      "required": false,
      "module": "engine",
      "source_file": "validators_config.py",
      "source_lines": "73"
    },
    {
      "key": "continuation.skip_volume_check",
      "type": "bool",
      "default": false,
      "required": false,
      "module": "engine",
      "source_file": "validators_config.py",
      "source_lines": "74"
    },
    {
      "key": "continuation.skip_pullback_check",
      "type": "bool",
      "default": false,
      "required": false,
      "module": "engine",
      "source_file": "validators_config.py",
      "source_lines": "75"
    },
    {
      "key": "continuation.block_if_crossed_baseline_since_entry",
      "type": "bool",
      "default": false,
      "required": false,
      "module": "engine",
      "source_file": "validators_config.py",
      "source_lines": "76"
    },
    {
      "key": "spreads.enabled",
      "type": "bool",
      "default": false,
      "required": false,
      "module": "costs",
      "source_file": "validators_config.py",
      "source_lines": "30"
    },
    {
      "key": "spreads.default_pips",
      "type": "float",
      "default": 0.0,
      "required": false,
      "module": "costs",
      "source_file": "validators_config.py",
      "source_lines": "31"
    },
    {
      "key": "spreads.per_pair",
      "type": "Dict[Pair, float]",
      "default": "{}",
      "required": false,
      "module": "costs",
      "source_file": "validators_config.py",
      "source_lines": "32"
    },
    {
      "key": "spreads.mode",
      "type": "Literal['fixed', 'atr_mult']",
      "default": "fixed",
      "required": false,
      "module": "costs",
      "source_file": "validators_config.py",
      "source_lines": "33"
    },
    {
      "key": "spreads.atr_mult",
      "type": "float",
      "default": 0.0,
      "required": false,
      "module": "costs",
      "source_file": "validators_config.py",
      "source_lines": "34"
    },
    {
      "key": "filters.dbcvix.enabled",
      "type": "bool",
      "default": false,
      "required": false,
      "module": "risk",
      "source_file": "validators_config.py",
      "source_lines": "103"
    },
    {
      "key": "filters.dbcvix.mode",
      "type": "Literal['reduce', 'block']",
      "default": "reduce",
      "required": false,
      "module": "risk",
      "source_file": "validators_config.py",
      "source_lines": "104"
    },
    {
      "key": "filters.dbcvix.threshold",
      "type": "Optional[float]",
      "default": null,
      "required": false,
      "module": "risk",
      "source_file": "validators_config.py",
      "source_lines": "105"
    },
    {
      "key": "filters.dbcvix.reduce_risk_to",
      "type": "float",
      "default": 1.0,
      "required": false,
      "module": "risk",
      "source_file": "validators_config.py",
      "source_lines": "106"
    },
    {
      "key": "filters.dbcvix.source",
      "type": "Literal['synthetic', 'manual_csv', 'refinitiv', 'bloomberg']",
      "default": "synthetic",
      "required": false,
      "module": "risk",
      "source_file": "validators_config.py",
      "source_lines": "107"
    },
    {
      "key": "filters.dbcvix.csv_path",
      "type": "str",
      "default": "data/external/dbcvix_synth.csv",
      "required": false,
      "module": "risk",
      "source_file": "validators_config.py",
      "source_lines": "108"
    },
    {
      "key": "risk.starting_balance",
      "type": "float",
      "default": 10000.0,
      "required": false,
      "module": "risk",
      "source_file": "validators_config.py",
      "source_lines": "121"
    },
    {
      "key": "risk.risk_per_trade_pct",
      "type": "float",
      "default": 2.0,
      "required": false,
      "module": "risk",
      "source_file": "validators_config.py",
      "source_lines": "122"
    },
    {
      "key": "risk.risk_per_trade",
      "type": "float",
      "default": 0.02,
      "required": false,
      "module": "risk",
      "source_file": "core/backtester.py",
      "source_lines": "621"
    },
    {
      "key": "risk.account_ccy",
      "type": "str",
      "default": "AUD",
      "required": false,
      "module": "risk",
      "source_file": "core/backtester.py",
      "source_lines": "620"
    },
    {
      "key": "risk.fx_quotes",
      "type": "Dict",
      "default": "{}",
      "required": false,
      "module": "risk",
      "source_file": "core/backtester.py",
      "source_lines": "622"
    },
    {
      "key": "cache.enabled",
      "type": "bool",
      "default": true,
      "required": false,
      "module": "cache",
      "source_file": "validators_config.py",
      "source_lines": "88"
    },
    {
      "key": "cache.dir",
      "type": "str",
      "default": "cache",
      "required": false,
      "module": "cache",
      "source_file": "validators_config.py",
      "source_lines": "89"
    },
    {
      "key": "cache.format",
      "type": "Literal['parquet', 'feather']",
      "default": "parquet",
      "required": false,
      "module": "cache",
      "source_file": "validators_config.py",
      "source_lines": "90"
    },
    {
      "key": "cache.scope_key",
      "type": "Optional[str]",
      "default": null,
      "required": false,
      "module": "cache",
      "source_file": "validators_config.py",
      "source_lines": "91"
    },
    {
      "key": "cache.roles",
      "type": "Optional[List[Literal['c1', 'c2', 'baseline', 'volume', 'exit']]]",
      "default": null,
      "required": false,
      "module": "cache",
      "source_file": "validators_config.py",
      "source_lines": "93"
    },
    {
      "key": "validation.enabled",
      "type": "bool",
      "default": true,
      "required": false,
      "module": "validation",
      "source_file": "validators_config.py",
      "source_lines": "97"
    },
    {
      "key": "validation.fail_fast",
      "type": "bool",
      "default": true,
      "required": false,
      "module": "validation",
      "source_file": "validators_config.py",
      "source_lines": "98"
    },
    {
      "key": "validation.strict_contract",
      "type": "bool",
      "default": false,
      "required": false,
      "module": "validation",
      "source_file": "validators_config.py",
      "source_lines": "99"
    },
    {
      "key": "output.results_dir",
      "type": "str",
      "default": "results",
      "required": false,
      "module": "output",
      "source_file": "validators_config.py",
      "source_lines": "117"
    },
    {
      "key": "outputs.dir",
      "type": "str",
      "default": "results",
      "required": false,
      "module": "output",
      "source_file": "core/backtester.py",
      "source_lines": "1067"
    },
    {
      "key": "outputs.write_trades_csv",
      "type": "bool",
      "default": true,
      "required": false,
      "module": "output",
      "source_file": "core/backtester.py",
      "source_lines": "1790"
    },
    {
      "key": "tracking.track_win_loss_scratch",
      "type": "bool",
      "default": true,
      "required": false,
      "module": "tracking",
      "source_file": "validators_config.py",
      "source_lines": "80"
    },
    {
      "key": "tracking.track_roi",
      "type": "bool",
      "default": true,
      "required": false,
      "module": "tracking",
      "source_file": "validators_config.py",
      "source_lines": "81"
    },
    {
      "key": "tracking.track_drawdown",
      "type": "bool",
      "default": true,
      "required": false,
      "module": "tracking",
      "source_file": "validators_config.py",
      "source_lines": "82"
    },
    {
      "key": "tracking.in_sim_equity",
      "type": "bool",
      "default": true,
      "required": false,
      "module": "tracking",
      "source_file": "validators_config.py",
      "source_lines": "83"
    },
    {
      "key": "tracking.verbose_logs",
      "type": "bool",
      "default": false,
      "required": false,
      "module": "tracking",
      "source_file": "validators_config.py",
      "source_lines": "84"
    },
    {
      "key": "date_range.start",
      "type": "Optional[str]",
      "default": null,
      "required": false,
      "module": "engine",
      "source_file": "validators_config.py",
      "source_lines": "126"
    },
    {
      "key": "date_range.end",
      "type": "Optional[str]",
      "default": null,
      "required": false,
      "module": "engine",
      "source_file": "validators_config.py",
      "source_lines": "127"
    },
    {
      "key": "date_from",
      "type": "Optional[str]",
      "default": null,
      "required": false,
      "module": "engine",
      "source_file": "core/backtester.py",
      "source_lines": "1148"
    },
    {
      "key": "date_to",
      "type": "Optional[str]",
      "default": null,
      "required": false,
      "module": "engine",
      "source_file": "core/backtester.py",
      "source_lines": "1149"
    },
    {
      "key": "walk_forward.start",
      "type": "Optional[str]",
      "default": null,
      "required": false,
      "module": "wfo",
      "source_file": "validators_config.py",
      "source_lines": "132"
    },
    {
      "key": "walk_forward.end",
      "type": "Optional[str]",
      "default": null,
      "required": false,
      "module": "wfo",
      "source_file": "validators_config.py",
      "source_lines": "133"
    },
    {
      "key": "walk_forward.train_years",
      "type": "int",
      "default": 3,
      "required": false,
      "module": "wfo",
      "source_file": "validators_config.py",
      "source_lines": "134"
    },
    {
      "key": "walk_forward.test_years",
      "type": "int",
      "default": 1,
      "required": false,
      "module": "wfo",
      "source_file": "validators_config.py",
      "source_lines": "135"
    },
    {
      "key": "walk_forward.step_years",
      "type": "int",
      "default": 1,
      "required": false,
      "module": "wfo",
      "source_file": "validators_config.py",
      "source_lines": "136"
    }
  ]
}
```

---

## C) YAML TEMPLATE

```yaml
# Forex Backtester Configuration Template v2.0.0
# Generated from codebase audit

strategy_version: "forex_backtester_v1.9.7"

# Required: pairs list
pairs:
  - "EUR_USD"
  - "USD_JPY"

timeframe: "D"

data_dir: "data/daily"

# Required: indicators configuration
indicators:
  c1: "fisher"  # Required: C1 indicator name
  use_c2: false
  c2: null
  use_baseline: false
  baseline: null
  use_volume: false
  volume: null
  use_exit: false
  exit: null

indicator_params: {}

# Required: rules configuration
rules:
  one_candle_rule: false
  pullback_rule: false
  bridge_too_far_days: 7
  allow_baseline_as_catalyst: false

entry:
  sl_atr: 1.5
  tp1_atr: 1.0
  trail_after_atr: 2.0
  ts_atr: 1.5

execution:
  intrabar_priority: "tp_first"

engine:
  allow_continuation: true
  duplicate_open_policy: "block"

continuation:
  allow_continuation: false
  skip_volume_check: false
  skip_pullback_check: false
  block_if_crossed_baseline_since_entry: false

# Required: exit configuration
exit:
  use_trailing_stop: true
  move_to_breakeven_after_atr: true
  exit_on_c1_reversal: true
  exit_on_baseline_cross: false
  exit_on_exit_signal: false

spreads:
  enabled: false
  default_pips: 0.0
  per_pair: {}
  mode: "fixed"
  atr_mult: 0.0

filters:
  dbcvix:
    enabled: false
    mode: "reduce"
    threshold: null
    reduce_risk_to: 1.0
    source: "synthetic"
    csv_path: "data/external/dbcvix_synth.csv"

risk:
  starting_balance: 10000.0
  risk_per_trade_pct: 2.0
  risk_per_trade: 0.02
  account_ccy: "AUD"
  fx_quotes: {}

cache:
  enabled: true
  dir: "cache"
  format: "parquet"
  scope_key: null
  roles: null

validation:
  enabled: true
  fail_fast: true
  strict_contract: false

output:
  results_dir: "results"

outputs:
  dir: "results"
  write_trades_csv: true

# Required: tracking configuration
tracking:
  track_win_loss_scratch: true
  track_roi: true
  track_drawdown: true
  in_sim_equity: true
  verbose_logs: false

date_range:
  start: null
  end: null

date_from: null
date_to: null

walk_forward:
  start: null
  end: null
  train_years: 3
  test_years: 1
  step_years: 1

monte_carlo:
  enabled: false
  iterations: 300
  horizon: "oos"
  use_daily_returns: null
  modes: null
  auto_after_wfo: false
  save_mode_specific: true
  save_main_as_latest: true
  rng_seed: null
```

---

## D) VALIDATION TEST CHECKLIST

### Entry Logic Tests

- [ ] **One-Candle Rule**: C1 signal on bar i with failed filters → pending → bar i+1 filters pass → entry on i+1 (`tests/test_apply_signal_logic.py:272-293`)
- [ ] **Pullback Rule**: Price > 1×ATR from baseline on signal bar AND previous bar → block entry
- [ ] **One-Candle + Pullback Mutually Exclusive**: Config validation rejects both enabled (`core/signal_logic.py:207-208`)
- [ ] **Baseline Catalyst**: Baseline cross triggers entry if last C1 within bridge_too_far_days (`core/signal_logic.py:347-380`)
- [ ] **Bridge-Too-Far**: Baseline catalyst blocked if last C1 ≥ bridge_too_far_days ago (`core/signal_logic.py:354-360`)
- [ ] **Entry Timing**: entry_signal on bar i → entry at close of bar i (`core/backtester.py:947`)
- [ ] **Duplicate-Open Guard**: Prevents multiple entries on same bar/side (`core/backtester.py:941-945`)

### Exit Logic Tests

- [ ] **TP1 Partial Exit**: TP1 hit → 50% lots closed, SL moves to BE (`core/backtester.py:763-778`)
- [ ] **Trailing Stop Activation**: Activates after +2×ATR move from entry (close-based) (`core/backtester.py:735-741`)
- [ ] **Trailing Stop Ratchet**: TS level updates each bar, only moves favorably (`core/backtester.py:743-746`)
- [ ] **Pre-TP1 Exit Priority**: C1 reversal > hard SL (`core/backtester.py:824-842`)
- [ ] **Post-TP1 Exit Priority**: TS > C1 reversal > BE (`core/backtester.py:784-822`)
- [ ] **Pre-TP1 C1 Reversal**: Exit at entry price (scratch) (`core/backtester.py:835`)
- [ ] **Post-TP1 C1 Reversal**: Exit at close price (`core/backtester.py:811`)
- [ ] **Intrabar Priority**: TP1 vs SL checked in order specified by `intrabar_priority` (`core/backtester.py:762-773`)

### W/L/S Classification Tests

- [ ] **WIN**: tp1_hit=True → WIN regardless of runner outcome (`core/backtester.py:876-878`)
- [ ] **LOSS**: tp1_hit=False AND exit_reason="stoploss" → LOSS (`core/backtester.py:881-882`)
- [ ] **SCRATCH**: tp1_hit=False AND exit_reason!="stoploss" → SCRATCH (`core/backtester.py:884`)

### Risk & Accounting Tests

- [ ] **Position Sizing**: lots_total = risk_money / (sl_dist_pips × pip_value_per_lot) (`core/backtester.py:970-982`)
- [ ] **DBCVIX Risk Reduction**: risk_pct_eff reduced when DBCVIX > threshold (`core/backtester.py:961-968`)
- [ ] **DBCVIX Block Mode**: risk_pct_eff=0 → entry blocked (`core/backtester.py:967-968`)
- [ ] **Spreads Invariant**: Spreads change PnL only, not trade counts or exit reasons (`tests/test_audit_immutability.py:266-484`)
- [ ] **Spread Resolution**: Per-bar → per-pair → ATR-mult → default (`core/backtester.py:512-538`)

### Audit Invariants Tests

- [ ] **Immutable Entry Fields**: tp1_at_entry_price, sl_at_entry_price never mutated after entry (`core/backtester.py:1010-1011`)
- [ ] **Stop Recording**: sl_at_exit_price set correctly for BE/TS/hard stops (`core/backtester_helpers.py:336-380`)
- [ ] **Equity Tracking**: equity = starting_balance + cumulative_realized_pnl (`core/backtester.py:928-932`)

### Continuation Tests

- [ ] **Continuation Entry**: Allowed after exit if baseline not crossed (`core/signal_logic.py:316-332`)
- [ ] **Continuation Skip Checks**: Volume and pullback checks skipped for continuation (`core/signal_logic.py:326`)

### Config Tests

- [ ] **Required Keys**: Validation fails if pairs, indicators.c1, rules, exit, tracking missing
- [ ] **Date Range**: date_range.start/end or date_from/date_to both supported (`core/backtester.py:1148-1149`)
- [ ] **Output Dir**: outputs.dir or output.results_dir both supported (`core/backtester.py:1067`)

---

**End of Audit Report**

