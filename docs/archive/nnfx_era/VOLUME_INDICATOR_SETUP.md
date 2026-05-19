# Volume Indicator Setup Documentation

## Summary

This document describes the volume indicator infrastructure and confirms the filter-only contract is properly enforced.

## Volume Indicator Location

**File**: `indicators/volume_funcs.py`

All volume indicators are implemented as functions with the `volume_` prefix in this module.

## Available Volume Indicators

Discovered via dynamic inspection of `indicators.volume_funcs`:

1. **volume_adx** - Passes (1) when Wilder ADX(length) >= min_adx, else 0
2. **volume_volatility_ratio** - Passes (1) when current ATR(length) >= threshold * EMA(ATR, smooth), else 0

## Configuration Keys

Volume indicators are controlled via the following config keys:

- `indicators.use_volume` (bool, default: False) - Enable/disable volume filtering
- `indicators.volume` (str, optional) - Name of volume indicator to use (e.g., "adx", "volatility_ratio")

## Volume Contract

### Signal Column
- **Column name**: `volume_signal`
- **Domain**: `{0, 1}` only
- **Semantics**:
  - `1` = Allow trade entry
  - `0` = Block trade entry

### Filter-Only Behavior

Volume indicators **DO NOT**:
- Set trade direction (no ±1 semantics)
- Modify SL/TP/TS
- Directly close trades (that's exit's job)

Volume indicators **ONLY**:
- Control whether a candidate entry from C1 (and later C2/Baseline) is allowed or skipped
- Act as a binary filter: pass (1) or block (0)

## Implementation Verification

### Signal Logic Integration

In `core/signal_logic.py` (lines 60-63):

```python
# Volume must be pass (=1)
if use_volume:
    if df.loc[i, "volume_signal"] != 1:
        failed.append("volume")
```

This confirms volume is checked as a filter only - if `volume_signal != 1`, the entry is blocked.

### Validation

In `validators_util.py` (lines 51-54):

```python
# Volume pass must be {0 or 1} if present
if "volume_signal" in df.columns and not _in_allowed_set_after_warmup(
    df["volume_signal"], warmup, {0, 1}
):
    raise ValueError("volume_signal must be in {0,1} after warmup.")
```

This enforces the {0,1} domain contract.

## Conclusion

The volume indicator contract is **already correctly implemented**:
- Volume indicators write to `volume_signal` with domain {0,1}
- Volume is checked as a filter only in signal logic
- Volume does not affect trade direction, SL/TP, or exits
- The contract is validated by the validators

No changes to core behavior are needed.

















