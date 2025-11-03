# MT5 Fisher Transform Indicator - Python Parity Guide

## Overview

This document describes the MT5 implementation of the Fisher Transform indicator (`C1_Fisher_Transform.mq5`) and how it matches the Python `c1_fisher_transform` function for EURUSD D1 parity verification.

## Installation Steps

1. **Locate MT5 Data Folder**
   - In MT5: `File` → `Open Data Folder`
   - Navigate to `MQL5/Indicators/`

2. **Copy Indicator File**
   - Copy `C1_Fisher_Transform.mq5` to the `MQL5/Indicators/` directory

3. **Compile Indicator**
   - Open MetaEditor (F4 in MT5)
   - Open `C1_Fisher_Transform.mq5`
   - Press `F7` to compile
   - Verify: Should show "0 error(s), 0 warning(s)"

4. **Attach to Chart**
   - In MT5, open EURUSD D1 chart
   - `Insert` → `Indicators` → `Custom` → `C1_Fisher_Transform`
   - Set parameters:
     - `Length = 10`
     - `EmaSpan = 3`
     - `ShiftToEntryBar = false` (or `true` for entry bar visualization)

## How to Read the Indicator

The indicator displays three plots in a separate window:

1. **Fisher Line (Blue)**
   - The raw Fisher Transform value
   - Oscillates around zero
   - Formula: `0.5 * log((1 + x) / (1 - x))` where x is clipped EMA of normalized price

2. **Signal+ Histogram (Green)**
   - Shows positive Fisher values (fisher > 0 → signal = +1)
   - Green bars indicate bullish confirmation

3. **Signal- Histogram (Red)**
   - Shows negative Fisher values (fisher < 0 → signal = -1)
   - Red bars indicate bearish confirmation

4. **Zero Baseline (Gray Dot)**
   - Level line at 0.0 for reference
   - Helps identify signal crossovers

## Non-Repainting Guarantees

The indicator is **non-repainting** by design:

- **Series Buffers**: All buffers use `ArraySetAsSeries(..., true)`
- **Iteration Direction**: Processes bars from oldest → newest (i decreases toward 0)
- **Completed Bars Only**: Skips index 0 (forming bar), processes index 1+ (completed bars)
- **Rolling Window**: Uses bars [i..i+Length-1] where i+1 is older in series mode
- **EMA State**: Uses previous-in-time EMA from XBuffer[i+1], never looks ahead

## Spot-Check Instructions Against Python

To verify parity with Python `c1_fisher_transform`:

1. **Export Same Data from MT5**
   - Right-click EURUSD D1 chart → `Save As` → CSV
   - Ensure same date range as Python backtest

2. **Compare Signal Crossovers**
   - In MT5: Look for sign changes in Fisher (blue line crosses zero)
   - In Python: Check `c1_signal` column in processed DataFrame
   - Signal flips should occur on the same bars

3. **Match Execution Timing**
   - Python uses "next-bar entry" logic (signal on bar i, entry on bar i+1)
   - With `ShiftToEntryBar=false`: Fisher/signal plotted on signal bar
   - With `ShiftToEntryBar=true`: Fisher/signal plotted on entry bar (visual shift only)

4. **Compare trades.csv**
   - Open `results/trades.csv` from Python backtest
   - Find entry dates/times
   - In MT5, verify Fisher signal sign matches on the **previous bar** (signal bar)
   - Example: If Python entered long on 2024-01-15, check that MT5 shows signal +1 on 2024-01-14 (previous bar)

## Formula Verification

The MT5 indicator matches Python exactly:

```python
# Python (c1_fisher_transform)
high_roll = df["high"].rolling(length).max()
low_roll = df["low"].rolling(length).min()
value = 2 * ((close - low_roll) / (high_roll - low_roll + 1e-10) - 0.5)
x = value.ewm(span=3, adjust=False).mean()
x = np.clip(x, -0.999, 0.999)
fisher = 0.5 * np.log((1 + x) / (1 - x))
signal = sign(fisher)  # +1 if >0, -1 if <0, 0 otherwise
```

MT5 equivalent:
1. Rolling max/min over `Length` bars using manual scan (matches Python rolling)
2. Normalized value calculation with EPS = 1e-10
3. EMA with `alpha = 2/(EmaSpan+1)` (matches pandas ewm with adjust=False)
4. Clip to [-0.999, 0.999]
5. Fisher transform using `MathLog`
6. Signal: +1 (green) if >0, -1 (red) if <0, 0 otherwise

## Troubleshooting

### Issue: All values show as zero in Data Window

**Symptoms**: Indicator compiles but Fisher, Signal+/- all show 0.0

**Causes**:
1. **EMA state buffer not initialized**: Check that `XBuffer` is set as series
2. **Rolling window out of bounds**: Ensure sufficient historical bars (need Length + EmaSpan + buffer)
3. **Array indexing error**: Verify all buffers use `ArraySetAsSeries(..., true)`

**Fix**:
- Verify compilation: Should show 0 errors
- Check Data Window: Right-click chart → `Indicators` → `C1_Fisher_Transform` → `Properties` → `Data` tab
- Ensure EURUSD D1 has enough history (≥50 bars recommended)

### Issue: Signals misaligned with Python

**Symptoms**: MT5 signals occur on different bars than Python

**Causes**:
1. **Different data source**: MT5 broker data vs Python CSV may differ
2. **Timezone mismatch**: MT5 server time vs Python processing time
3. **Rolling window direction**: Incorrect series indexing

**Fix**:
- Export same MT5 CSV that Python uses (use `scripts/export_mt5_daily.py`)
- Compare bar-by-bar: MT5 bar i should match Python row i (assuming same CSV)
- Verify rolling window: MT5 uses bars [i..i+Length-1] where i+1 is older

### Issue: Fisher values don't match Python exactly

**Symptoms**: Signals match but Fisher values are slightly different

**Causes**:
1. **Floating-point precision**: MT5 vs Python may differ in rounding
2. **EMA initialization**: First value handling may differ

**Expected**: Small differences (<0.001) are acceptable. Signal signs (+1/-1/0) must match exactly.

### Issue: Indicator doesn't appear on chart

**Symptoms**: Compiled successfully but not in indicator list

**Fix**:
1. Restart MT5 after compilation
2. Refresh indicator list: `Insert` → `Indicators` → `Custom` (should see C1_Fisher_Transform)
3. Check compilation log: MetaEditor → `View` → `Toolbox` → `Errors` tab

## Parameters

| Parameter | Default | Python Equivalent | Description |
|-----------|---------|-------------------|-------------|
| `Length` | 10 | `length=10` | Rolling window length for high/low max/min |
| `EmaSpan` | 3 | `span=3` | EMA smoothing span (matches pandas ewm) |
| `ShiftToEntryBar` | false | N/A | Visual shift +1 bar for entry bar plotting (cosmetic only) |

## Visual Settings

For better visibility on light/dark themes:

- **Fisher Line**: Blue, solid, width 2
- **Signal+**: Green histogram, width 3
- **Signal-**: Red histogram, width 3
- **Zero Level**: Gray dot line

Customize in indicator properties: Right-click indicator → `Properties` → `Colors` tab.

## Next Steps

After verifying parity:

1. Document any discrepancies (if signal signs don't match)
2. Update this doc with broker-specific notes if needed
3. Add to CI/CD if automated parity checks are implemented

---

**Last Updated**: 2024 (MT5 Fisher Transform v1.01)

