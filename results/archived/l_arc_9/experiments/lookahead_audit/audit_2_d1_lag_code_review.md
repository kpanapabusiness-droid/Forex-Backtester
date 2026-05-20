# Audit 2 — D1 lag code review

## Arc 9 D1 lag implementation

Source: `scripts/l_arc_9/experiments/pipeline_e_retry.py:_attach_d1_features`

```python
def _attach_d1_features(
    trades: pd.DataFrame, data_d1_dir: Path, atr_4h_by_tid: Dict[int, float],
) -> pd.DataFrame:
    """For each trade, merge_asof against the PRIOR D1 bar of its pair.
```

## Reference engine pattern (KH-24 backtester)

Source: `scripts/phase_kgl_v2_4h_wfo.py:_precompute_d1_exit_arrays` (line ~900)

```python
dates_4h_norm = pd.to_datetime(df_4h["date"]).dt.normalize() - pd.Timedelta(days=1)
df_4h_dates = pd.DataFrame({
    "date": dates_4h_norm,
    "_idx": np.arange(n, dtype=int),
})
...
merged = pd.merge_asof(
    df_4h_dates.sort_values('date'),
    df_d1,
    on='date',
    direction='backward',
)
```

## Pattern equivalence

Both implementations perform:
1. Normalize 4H signal-bar timestamp to its calendar date.
2. Subtract `pd.Timedelta(days=1)` to produce `lookup_date` = signal-day minus 1.
3. `pd.merge_asof(direction='backward')` on the lookup_date against the D1 date series.

Result: each 4H signal bar receives the D1 bar with the LATEST date ≤ (signal_date − 1 day).
- Mid-week signal: signal_date − 1 = yesterday → D1 bar = yesterday's D1 close.
- Monday signal: signal_date − 1 = Sunday (no D1 bar) → merge_asof finds Friday's D1 close. Days_lag = 3.
- 00:00 UTC signal: normalize() yields the SAME signal_date that 04:00 UTC has;
  the −1 day subtraction shifts uniformly. No 'same-day D1' branch exists in the code.

**Arc 9 pattern verbatim-equivalent to KH-24 engine pattern: YES.**

Missing pieces in Arc 9 code (should be empty): []
Missing pieces in reference engine code (should be empty): []
