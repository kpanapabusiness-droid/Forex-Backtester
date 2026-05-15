# v1.3 calibration — schema audit on KH-24 + Arc 1 + Arc 2 trade-paths

> Date: 2026-05-15
> Author: Claude Code (Opus 4.7)

This audit confirms the three trade-path artefacts are compatible with
the v1.3 capturability metric battery, identifies the schema
differences, and pins the normalisation decisions the loader applies.

## §1 Mapping table

| Dataset | File path | Per-bar? | Forward-after-exit? | R-unit convention | Compatible |
|---|---|---|---|---|---|
| KH-24 | `results/kh24/trades_paths.csv` | yes | yes (to bar_offset = 240) | ATR-units (signal-bar ATR(14)_4H) — divided by SL_MULT=2.0 by the loader to land in SL-distance R-units | reference |
| Arc 1 | `results/l_arc_1/step2_descriptive/trade_paths.csv` | partial (running MFE/MAE + cum logret, NO per-bar OHLC) | yes (to t = 480) | fwd_mfe_atr / fwd_mae_atr in ATR-units (signal-bar ATR(14)_1H) — divided by 2.0 → R-units | partial |
| Arc 2 | `results/l_arc_2/step2_descriptive/trade_paths.csv` | yes (OHLC + running MFE/MAE) | yes (to bar_offset = 479) | mfe/mae in ATR-units (1H ATR) — divided by 2.0 → R-units; high/low/close derived from raw OHLC | yes |

## §2 Per-dataset row counts

| Dataset | Trades | Per-bar rows | Bars/trade (mean) | bar_offset range |
|---|---|---|---|---|
| KH-24 | 553 | 132,423 | 239.5 | 0..240 |
| Arc 1 | 45,673 | 21,923,040 | 480.0 | 0..479 |
| Arc 2 | 3,993 | 1,904,282 | 477.0 | 0..479 |

Arc 1 has 145,283 NaN rows at the tail of late trades whose forward
window ran past the dataset end (619 unique trades). Pandas
aggregations skip NaN so no special handling beyond documentation.

Arc 2 has 8 trades flagged with `data_end_flag = True` (forward window
truncated by end-of-data).

## §3 t / bar_offset alignment

| Dataset | Column name | Starts at | Aligned with KH-24? | Loader fix |
|---|---|---|---|---|
| KH-24 | `bar_offset` | 0 (entry bar) | reference | none |
| Arc 1 | `t` | 1 (first bar after signal, = entry bar) | offset by 1 | subtract 1 → bar_offset 0..479 |
| Arc 2 | `bar_offset` | 0 (entry bar) | yes | none |

Arc 1's `t=1` corresponds to KH-24's `bar_offset=0` (both are the entry
bar). Confirmed by sample inspection of Arc 1 trade 0: `fwd_logret_cum`
at t=1 is ~2.6e-5 (sub-pip), `fwd_mfe_atr` already 0.15 — consistent
with a bar's worth of intra-bar excursion measured from entry.

## §4 is_held / forward-window flag

| Dataset | Native column | Loader output |
|---|---|---|
| KH-24 | `is_held` (0/1) | passed through |
| Arc 1 | (none) | synthesised: `bar_offset <= bars_held`, where `bars_held = (exit_bar_ts - entry_bar_ts) / 1H` from `signals_features.csv`. Never-exited trades clamped so `is_held=1` across their full emitted window. |
| Arc 2 | `is_held_bar` (bool) | cast to int8 |

## §5 high_r / low_r availability

| Dataset | High/low per bar | Loader output |
|---|---|---|
| KH-24 | derived from OHLC during the backtester hold loop (signed) | passed through (with /SL_MULT normalisation) |
| Arc 1 | **NOT EMITTED** by step 2 (only running MFE/MAE) | `high_r` / `low_r` set to NaN; TP/SL detection uses running MFE/MAE (exact at the bar each running max first crosses the threshold); trail-exit detection falls back to `close_r`-based logic |
| Arc 2 | full OHLC per bar | derived: `(high - entry_open) / sl_distance` and the same for low/close, signed |

## §6 close_r derivation

| Dataset | Source | Conversion |
|---|---|---|
| KH-24 | `close_r` column in trades_paths.csv (ATR-units signed) | divided by SL_MULT=2.0 → R-units |
| Arc 1 | `fwd_logret_cum` + `signal_bar_close` + `atr_at_signal_1h` from `signals_features.csv` | `close_r = entry_px × (exp(fwd_logret_cum) − 1) / sl_distance`, where `entry_px ≈ signal_bar_close`, `sl_distance = 2 × atr_at_signal_1h`. Sub-bar approximation — `entry_px` is the *signal* bar close, not the *entry* bar open. For long-only 1H FX this is one bar of slippage, which is small relative to ATR; documented in `loader_decisions.md`. |
| Arc 2 | `close` column (raw price) + entry_px from bar_offset=0 row | `(close - entry_open) / sl_distance` |

## §7 Verdict per dataset

| Dataset | Verdict |
|---|---|
| KH-24 | **Compatible (reference).** All metrics computable. |
| Arc 1 | **Partial.** TP/SL/MFE-lock simulations computable exactly from running MFE/MAE. Close-based trail computable from derived close_r (approximate; documented). Intrabar high/low-based trail NOT computable — flagged unavailable. Conditional predictivity (Axis 2e) uses derived close_r. All other metrics computable. |
| Arc 2 | **Compatible.** All metrics computable. |

Neither arc is fully incompatible → proceed with all three datasets,
flag the Arc 1 intrabar-trail unavailability in the calibration
report's §5.

## §8 Cluster assignments

| Dataset | File | Trades | K3_kmeans distribution |
|---|---|---|---|
| KH-24 | (none — not in L-arc registry) | — | — |
| Arc 1 | `results/l_arc_1/step3_extractability/cluster_assignments.csv` | 45,673 | {0: 14882, 1: 16230, 2: 14561} |
| Arc 2 | `results/l_arc_2/step3_extractability/cluster_assignments.csv` | 3,993 | {-2: 112, 0: 693, 1: 2334, 2: 854} |

KH-24 has no L-arc cluster assignments — per-cluster metrics will be
emitted for Arc 1 and Arc 2 only.

Arc 2 includes a `-2` cluster (HDBSCAN noise / unassigned trades, 112
trades = 2.8%); standard K3_kmeans clusters are {0, 1, 2}.
