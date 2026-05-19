# Exit Indicator Framework - Implementation Summary

## Overview

Exit indicator infrastructure has been set up to enable testing of C1-only vs C1+Exit strategies via Walk-Forward Optimization (WFO). The framework ensures that exit indicators **do not affect entry logic** and only influence when already-open trades are closed.

## What Was Implemented

### 1. Exit Indicator Signal Integration

**File: `core/signal_logic.py`**
- Added check for exit indicator signals when `exit_on_exit_signal=True`
- Exit indicator is checked **only when a position is open** (does not affect entries)
- Priority order: C1 reversal > Exit indicator > Baseline cross > Stop loss
- Sets `exit_reason="exit_indicator"` when exit indicator triggers

**Key Changes:**
```python
# Exit indicator signal (second priority, only if exit indicator is enabled)
if not exit_triggered and exit_on_exit_signal and has_exit_signal:
    exit_signal_val = pd.to_numeric(out.loc[i, "exit_signal"], errors="coerce")
    if not pd.isna(exit_signal_val) and int(exit_signal_val) == 1:
        exit_triggered = True
        exit_reason = "exit_indicator"
```

### 2. Exit Indicator Contract

**File: `indicators/exit_funcs.py`**
- Exit functions must write {0,1} to `exit_signal` column
- `exit_none()`: No-op exit (always returns 0) for parity testing
- `exit_twiggs_money_flow()`: Example exit indicator implementation

**Contract:**
```python
def exit_<name>(df: pd.DataFrame, *, signal_col: str = "exit_signal", **kwargs) -> pd.DataFrame:
    """
    Writes df[signal_col] ∈ {0,1} where 1 = "exit now"
    Returns modified df
    """
```

### 3. Config Support

**Existing Config Keys (already supported):**
- `indicators.use_exit`: bool (default: False) - Enable exit indicator
- `indicators.exit`: Optional[str] - Exit indicator name (e.g., "exit_twiggs_money_flow")
- `exit.exit_on_exit_signal`: bool (default: False) - Enable exit indicator signals

**Example Config:**
```yaml
indicators:
  c1: "c1_twiggs_money_flow"
  use_exit: true
  exit: "exit_twiggs_money_flow"

exit:
  exit_on_c1_reversal: true
  exit_on_exit_signal: true  # Enable exit indicator
```

### 4. Exit Reason Tracking

**File: `core/backtester.py`**
- `exit_reason` field in trades.csv includes "exit_indicator"
- Other exit reasons: "c1_reversal", "baseline_cross", "stoploss", "trailing_stop", "breakeven_after_tp1"

**Exit Reason Values:**
- `"exit_indicator"`: Trade closed by exit indicator signal
- `"c1_reversal"`: Trade closed by C1 signal reversal
- `"baseline_cross"`: Trade closed by baseline cross
- `"stoploss"`: Trade closed by stop loss hit
- `"trailing_stop"`: Trade closed by trailing stop
- `"breakeven_after_tp1"`: Trade closed at breakeven after TP1

### 5. Tests

**File: `tests/test_signal_exits.py`**
- `test_exit_indicator_signal()`: Verifies exit indicator triggers exits
- `test_exit_indicator_priority_after_c1_reversal()`: Verifies C1 reversal takes priority
- `test_exit_indicator_does_not_affect_entries()`: Verifies exits don't prevent entries

## Usage

### C1-Only Configuration (Baseline)

```yaml
indicators:
  c1: "c1_twiggs_money_flow"
  use_exit: false

exit:
  exit_on_c1_reversal: true
  exit_on_exit_signal: false
```

### C1 + Exit Configuration

```yaml
indicators:
  c1: "c1_twiggs_money_flow"
  use_exit: true
  exit: "exit_twiggs_money_flow"

exit:
  exit_on_c1_reversal: true
  exit_on_exit_signal: true
```

### WFO Configuration Example

```yaml
# configs/wfo_c1_with_exit.yaml
pairs:
  - AUD_USD
  - EUR_USD
  - GBP_USD
  - USD_CHF

timeframe: "D"

indicators:
  c1: "c1_twiggs_money_flow"
  use_exit: true
  exit: "exit_twiggs_money_flow"

exit:
  exit_on_c1_reversal: true
  exit_on_exit_signal: true

walk_forward:
  enabled: true
  run_name: "wfo_c1_twiggs_with_exit"
  train_years: 2
  test_months: 6
  step_months: 6
```

**Run WFO:**
```bash
python scripts/walk_forward.py -c configs/wfo_c1_with_exit.yaml
```

## Exit Indicator Priority

When multiple exit conditions are present, priority order is:

1. **C1 Reversal** (if `exit_on_c1_reversal=True`)
2. **Exit Indicator** (if `exit_on_exit_signal=True`)
3. **Baseline Cross** (if `exit_on_baseline_cross=True`)
4. **Stop Loss** (always checked)

## Key Guarantees

✅ **Exit indicators do NOT affect entries**
- Entry logic (C1 + NNFX classification + baseline/volume/C2) is unchanged
- Exit signals are only checked when a position is already open

✅ **Exit reasons are tracked**
- `trades.csv` includes `exit_reason` field
- Can count trades by exit type: SL, TP1/TP2, scratch, exit_indicator

✅ **WFO compatible**
- Configs pass through correctly
- Same structure as C1-only runs
- Results in `results/wfo_<run_name>/`

✅ **Backward compatible**
- When `use_exit=false` or `exit_on_exit_signal=false`, behavior is unchanged
- C1-only runs work exactly as before

## Metrics for Evaluation

When comparing C1-only vs C1+Exit, track:

1. **Exit Reason Distribution**
   - Count of trades by `exit_reason`
   - % of trades exiting via stop_loss (should decrease)
   - % of trades exiting via TP1/TP2 (should not decrease significantly)

2. **Performance Metrics**
   - ROI (should improve or at least not degrade)
   - Max drawdown (should improve)
   - Expectancy (should improve)
   - Win rate
   - Scratch rate

3. **Runner Preservation**
   - Verify TP2 hits are not eliminated
   - Check that big winners survive

## Next Steps

1. **Add specific exit indicators** (e.g., Chandelier Stop, STC)
2. **Run WFO comparisons** between C1-only and C1+Exit
3. **Analyze results** using exit_reason distribution and performance metrics
4. **Optimize exit indicator parameters** if needed

## Files Modified

- `core/signal_logic.py`: Added exit indicator signal checking
- `tests/test_signal_exits.py`: Added exit indicator tests

## Files Already Supporting Exits

- `indicators/exit_funcs.py`: Exit indicator functions
- `core/backtester.py`: Exit reason tracking
- `validators_config.py`: Config schema
- `core/backtester_helpers.py`: Indicator loading

