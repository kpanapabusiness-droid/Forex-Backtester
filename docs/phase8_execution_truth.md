### Phase 8 – Execution Truth (Scratch, TP1→BE, Trailing, Live Actions)

This document states the **single source of truth** for how the current engine behaves, and what a human operator must do in MT5 to mirror the backtests. It is entirely descriptive – **no engine logic has been changed**.

---

### 1. Scratch and System Exit Behaviour

- **Code path**
  - File: `core/backtester.py`
  - Function: `simulate_pair_trades`
  - Region: open-position management loop

```736:896:core/backtester.py
if open_tr is not None:
    d = int(open_tr["direction_int"])
    entry_px = float(open_tr["entry_price"])
    ...
    tp1_done = bool(open_tr.get("tp1_hit", False))
    ...
    has_system_exit = exit_sig != 0
    ...
    if tp1_done:
        ...
        # 2. C1 reversal / exit indicator / baseline cross (post-TP1)
        if (not closed_this_bar) and has_system_exit:
            ...
            reason = "c1_reversal" or "exit_indicator" or "baseline_cross"
            exit_px = next_open_px  # Phase 8: execute at NEXT bar open
            closed_this_bar = True
        ...
    else:
        # Pre-TP1: System exits (C1 reversal) take priority over hard stops
        if (not closed_this_bar) and has_system_exit:
            ...
            reason = "c1_reversal" or "exit_indicator" or "baseline_cross"
            exit_px = next_open_px  # Phase 8: execute at NEXT bar open
            closed_this_bar = True
        ...
    ...
    if closed_this_bar:
        ...
        if reason == "trailing_stop":
            ...
        else:
            # All non-TS exits (both pre- and post-TP1) use the resolved price above
            open_tr["exit_price"] = float(exit_px)
        ...
        if bool(open_tr.get("tp1_hit", False)):
            open_tr["win"], open_tr["loss"], open_tr["scratch"] = True, False, False
        else:
            if reason == "stoploss":
                open_tr["win"], open_tr["loss"], open_tr["scratch"] = False, True, False
            else:
                open_tr["win"], open_tr["loss"], open_tr["scratch"] = False, False, True
```

- **Conditions for a scratch exit (pre‑TP1)**
  - A trade is open (`open_tr is not None`).
  - `tp1_hit` is **False** (`tp1_done == False`).
  - The engine sees a **system exit** on the current bar (`exit_sig != 0`), coming from:
    - C1 reversal (default),
    - exit indicator (`exit.exit_on_exit_signal == True`),
    - or baseline cross (`exit.exit_on_baseline_cross == True`).
  - This path runs **before** any hard stop (`stoploss`) intrabar checks.

- **System exit execution price (Phase 8 rule)**
  - When a **system exit** is triggered on bar \( i \) (signal computed on the close of bar \( i \)):
    - If a **next bar exists** (\( i+1 \) in the data):
      - The accounting exit price is **the open of bar \( i+1 \)**:
        - `exit_price = open[i+1]`.
    - If bar \( i \) is the **last bar** (no \( i+1 \)):
      - The accounting exit price falls back to the **close of bar \( i \)**:
        - `exit_price = close[i]`.
  - This applies **both**:
    - **Pre‑TP1** (scratch exits), and
    - **Post‑TP1** (system exits after TP1 has hit).

- **Spread interaction**
  - `exit_price` (the accounting price) is mid‑price at entry.
  - PnL is then computed by `compute_trade_pnl_money`, which adjusts entry/exit fills by half‑spread:

```552:587:core/backtester.py
def compute_trade_pnl_money(tr: Dict[str, Any], pair: str, pip_value_1lot: float) -> float:
    dir_int = int(tr["direction_int"])
    entry_mid = float(tr["entry_price"])
    exit_mid = float(tr["exit_price"])
    ...
    sp_pips = float(tr.get("spread_pips_used", 0.0))
    ps = pip_size_for_pair(pair)
    sp_price = sp_pips * ps
    if dir_int > 0:  # long
        entry_fill = entry_mid + sp_price / 2.0
        ...
        exit_fill = exit_mid - sp_price / 2.0
    else:  # short
        entry_fill = entry_mid - sp_price / 2.0
        ...
        exit_fill = exit_mid + sp_price / 2.0
```

  - **Result pre‑TP1**: a system exit produces a **scratch** where:
    - Accounting exit is at **next‑bar open** (or last‑bar close).
    - Realistic spread is still paid/received via the PnL model.
    - Classification: `scratch=True, win=False, loss=False`.

- **Pre‑TP1 vs post‑TP1**
  - **Pre‑TP1**:
    - System exits use **next‑bar open** as accounting exit (`exit_price = open[i+1]` or `close[i]` if last bar).
    - Any non‑SL, non‑TP1 exit remains a **SCRATCH**.
  - **Post‑TP1**:
    - System exits (C1 reversal / exit indicator / baseline cross) also execute at the **next‑bar open** (with last‑bar fallback).
    - The trade remains a **WIN** because `tp1_hit == True`, regardless of runner outcome.

---

### 2. TP1 → Breakeven Behaviour

- **TP1 definition**
  - File: `core/backtester.py`
  - Function: `simulate_pair_trades`

```614:617:core/backtester.py
SL_ATR_MULT = float(overrides.get("sl_atr_mult", entry_cfg.get("sl_atr", 1.5)))
TP1_ATR_MULT = float(overrides.get("tp1_atr_mult", entry_cfg.get("tp1_atr", 1.0)))
TRAIL_AFTER_ATR = float(overrides.get("trail_after_atr", entry_cfg.get("trail_after_atr", 2.0)))
TS_ATR_MULT = float(overrides.get("ts_atr_mult", entry_cfg.get("ts_atr", 1.5)))
...
tp1_px = entry_px + d_int * (TP1_ATR_MULT * atr_entry)
sl_px = entry_px - d_int * (SL_ATR_MULT * atr_entry)
```

- **Exactly what TP1 is**
  - ATR used: **ATR at entry bar**, from column `atr` on that bar.
  - TP1 distance: `TP1_ATR_MULT × ATR_entry`, default **1.0×ATR** unless `entry.tp1_atr` is set in YAML.
  - TP1 price:
    - Long: `entry_price + 1.0×ATR_entry`.
    - Short: `entry_price − 1.0×ATR_entry`.

- **How TP1 is detected**

```773:781:core/backtester.py
if not tp1_done:
    for ev in order:
        if ev == "tp" and hit_level(d, h_i, l_i, tp1_px, "tp"):
            tp1_done = True
            be_price = entry_px
            sl_px = be_price
            open_tr["tp1_hit"] = True
            open_tr["breakeven_after_tp1"] = True
```

  - TP1 is checked **intrabar**, using the bar’s **high/low**:
    - Helper `hit_level` compares `high`/`low` vs `tp1_px` in a direction‑aware way.
  - Priority between TP1 and SL (pre‑TP1) is controlled by `intrabar_priority` (default `"tp_first"`).

- **Breakeven behaviour after TP1**
  - On TP1 hit **in the same bar**:
    - `tp1_hit` = `True`.
    - `breakeven_after_tp1` = `True`.
    - `sl_px` is immediately set to **exactly** `entry_price` (breakeven).
  - Effective stop is then recomputed, and from that point on the **worst‑case** stop is breakeven (unless a trailing stop ratchets higher).

---

### 3. Trailing Stop Behaviour

- **Activation condition**

```747:753:core/backtester.py
if not ts_active and math.isfinite(atr_entry) and atr_entry > 0:
    move_cl = signed_move_from_entry(d, c_i, entry_px)
    threshold = TRAIL_AFTER_ATR * atr_entry
    if move_cl >= threshold:
        ts_active = True
        ts_level = trail_level_from_close(d, c_i, atr_entry)
```

  - Trigger: signed move from entry to current **close** ≥ `TRAIL_AFTER_ATR × ATR_entry`.
  - Default: `TRAIL_AFTER_ATR = 2.0`, so trailing starts after a **+2×ATR** favorable move from entry.

- **Trailing distance**

```662:667:core/backtester.py
def trail_level_from_close(direction_int: int, close_px: float, atr_entry: float) -> float:
    return (
        close_px - TS_ATR_MULT * atr_entry
        if direction_int > 0
        else close_px + TS_ATR_MULT * atr_entry
    )
```

  - Trailing stop is placed:
    - Long: `close − TS_ATR_MULT × ATR_entry`.
    - Short: `close + TS_ATR_MULT × ATR_entry`.
  - Default `TS_ATR_MULT = 1.5`, so the stop trails **1.5×ATR_entry** behind price.
  - **ATR is always the ATR at entry**, not a rolling ATR.

- **Ratchet behaviour**

```755:758:core/backtester.py
if ts_active and math.isfinite(atr_entry) and atr_entry > 0:
    cand = trail_level_from_close(d, c_i, atr_entry)
    ts_level = better_stop(d, ts_level, cand)
```

  - Each new bar recomputes a candidate trailing level from the latest close and uses `better_stop` to only move the stop in the **favourable** direction (never back towards entry).

- **Priority after TP1**

```796:835:core/backtester.py
if tp1_done:
    # 1. Trailing stop (highest priority post-TP1)
    if (
        (not closed_this_bar)
        and ts_active
        and hit_level(d, h_i, l_i, effective_stop, "sl")
    ):
        ...
        reason = "trailing_stop"
        exit_px = effective_stop
        closed_this_bar = True
    # 2. C1 / exit indicator / baseline exits at close
    if (not closed_this_bar) and has_system_exit:
        ...
        exit_px = c_i  # Post-TP1: execute at close price
        closed_this_bar = True
    # 3. Breakeven / stoploss using effective_stop
    if (not closed_this_bar) and hit_level(...):
        reason = "breakeven_after_tp1" or "stoploss"
        exit_px = effective_stop
        closed_this_bar = True
```

  - **Post‑TP1 exit priority**:
    1. Trailing stop (intrabar, at the effective stop level).
    2. System exits (C1 reversal, exit indicator, baseline cross) which, after Phase 8, **execute at next‑bar open** for pricing.
    3. Breakeven/stoploss at `effective_stop` (intrabar).

- **Trailing stop fill + audit**

```458:479:core/backtester.py
def _apply_trailing_stop_fill(row: dict, *, final_stop_price: float, is_long: bool, pair: str, cfg: dict) -> dict:
    ps = pip_size_for_pair(pair)
    fills = (cfg or {}).get("fills") or {}
    sl = fills.get("slippage") or {}
    slip_pips = float(sl.get("pips", 0.0)) if sl.get("enabled", False) else 0.0
    slip_px = slip_pips * ps
    exit_px = final_stop_price - slip_px if is_long else final_stop_price + slip_px
    row["exit_reason"] = "trailing_stop"
    row["exit_price"] = float(exit_px)
    row["sl_at_exit_price"] = float(final_stop_price)
    row["ts_active"] = True
    row["ts_level"] = float(final_stop_price)
    row["slippage_pips"] = float(slip_pips)
    return row
```

  - Trailing stop exits always record:
    - `exit_reason="trailing_stop"`,
    - `exit_price` at stop ± optional slippage,
    - `sl_at_exit_price` at the **stop level** in mid‑price units,
    - `ts_active=True`, `ts_level=stop`.

---

### 4. YAML Exit Flags – Implemented or Not?

- **Flags in question**
  - `exit.use_trailing_stop`
  - `exit.move_to_breakeven_after_atr`

- **Where they live**
  - Schema: `validators_config.py`:

```63:68:validators_config.py
class Exit(BaseModel):
    use_trailing_stop: bool = True
    move_to_breakeven_after_atr: bool = True
    exit_on_c1_reversal: bool = True
    exit_on_baseline_cross: bool = False
    exit_on_exit_signal: bool = False
```

  - Configs: e.g. `configs/v1_system.yaml`, `configs/phase6_spread_full.yaml`, `configs/phase6_1_spread_full.yaml` set these booleans.

- **Where they are used**
  - `exit.exit_on_c1_reversal`, `exit.exit_on_baseline_cross`, `exit.exit_on_exit_signal` **are** read in `simulate_pair_trades`:

```611:612:core/backtester.py
exit_cfg = cfg.get("exit") or {}
...
if tp1_done:
    ...
    if (not closed_this_bar) and has_system_exit:
        if exit_cfg.get("exit_on_exit_signal", False):
            reason = "exit_indicator"
        elif exit_cfg.get("exit_on_c1_reversal", True):
            reason = "c1_reversal"
        elif exit_cfg.get("exit_on_baseline_cross", False):
            reason = "baseline_cross"
```

  - **However**, there is **no reference** in the simulator code to:
    - `exit.use_trailing_stop`
    - `exit.move_to_breakeven_after_atr`
  - A repo‑wide search (`Grep` over the repo) shows those names appear only in:
    - Config files (`configs/*.yaml`),
    - The schema (`validators_config.py`),
    - Some tests and research configs.

- **Conclusion**
  - **Trailing stop and TP1→BE are always on** in the engine, controlled only by the ATR multipliers:
    - `entry.sl_atr`, `entry.tp1_atr`, `entry.trail_after_atr`, `entry.ts_atr`.
  - YAML booleans:
    - `exit.use_trailing_stop` → **NOT IMPLEMENTED** in simulator logic.
    - `exit.move_to_breakeven_after_atr` → **NOT IMPLEMENTED** in simulator logic.

---

### 5. What the Human Operator Must Do in MT5

These are the **live actions** required to match the Phase 6/6.1 engine:

- **Entry**
  - Enter **once per signal**, at/near the **daily close** that generated the signal.
  - Place **two orders**:
    - TP1 order (half size) with TP at `entry ± 1.0×ATR_entry`.
    - Runner order (half size) with **no TP**.

- **Initial SL**
  - For both TP1 and runner legs:
    - Long: SL at `entry − 1.5×ATR_entry`.
    - Short: SL at `entry + 1.5×ATR_entry`.

- **After TP1 is hit**
  - As soon as TP1 is filled:
    - Move the **runner SL to exactly the entry price** (breakeven).
    - The TP1 leg is considered a **WIN** permanently, regardless of what happens next.

- **Trailing stop**
  - Once price has moved at least **+2.0×ATR_entry** from entry (based on closes):
    - Activate a trailing SL on the **runner**:
      - Long: `SL = close − 1.5×ATR_entry`.
      - Short: `SL = close + 1.5×ATR_entry`.
    - On each new daily close, ratchet the SL further in your favour if this formula produces a **better** stop; never move it backwards.

- **Scratch / system exits**
  - If, **before TP1 and before SL**, the system generates an exit (C1 flips, exit indicator, or baseline exit if enabled):
    - Close **the whole position at the NEXT daily open** in live trading, to match the engine’s “next‑open” accounting.
    - Treat this as a **SCRATCH** in your own stats (win=False, loss=False, spread + gap PnL from entry to next open).
  - If a system exit occurs **after TP1**:
    - Also close at the **next daily open**.
    - The trade is still a **WIN** in stats because TP1 was hit earlier.

- **Never do manually**
  - Do **not**:
    - Adjust SL or TP for discretionary reasons beyond:
      - TP1→BE,
      - Trailing 1.5×ATR_entry after 2×ATR_entry move.
    - Change per‑trade risk away from what the config specifies (e.g. 0.25% for Phase 6.1).
    - Apply any other trailing or partial‑take rules not described here.

