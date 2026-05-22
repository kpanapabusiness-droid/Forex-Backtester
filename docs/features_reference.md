# Features Reference (v3.0 Step 1)

Single source of truth for every Step-1 feature emitted by 
`core.features.pipeline.compute_feature_matrix`. Each row is one column 
in the output matrix. The `lineage` column drives the Step 6 producer 
audit per L_PROTOCOL §2 Step 6.

## Lineage tag semantics

- **clean** — verified-by-construction: uses only data from bars closed 
  strictly before the signal bar; no hidden lookback or future-leaking 
  aggregation. Lookahead spot-check passes by construction (asserted in 
  `tests/test_features_pipeline.py::test_lookahead_spotcheck_5_random_trades`).
- **suspect** — plausible lookahead risk; needs the Step 6 producer 
  audit before any deployment. Cross-pair features default here 
  because cross-panel alignment is non-trivial.
- **unverified** — no causal review yet. `clean` requires the 
  lookahead spot-check; this tag is for new features awaiting that work.

## Pipeline contract

- Every producer returns a Series aligned to `pair_df.index`.
- Producers that need cross-pair state declare `needs_panel=True`; when 
  the pipeline is called without a Panel those columns emit all-NaN.
- Output columns are sorted alphabetically — output ordering is 
  deterministic across imports.

## Inventory (27 features across 7 classes)

### cross_pair

| Name | Lineage | Needs panel | Description |
|---|---|---:|---|
| `dollar_bloc_state` | suspect | True | Average signed 1-bar mid-return of dollar-bloc currencies (USD, CAD, AUD, NZD) vs the non-bloc complement. Proxy for risk-on/off bloc moves. |
| `eur_strength_index` | suspect | True | Average signed 1-bar mid-return of EUR across all EUR-bearing pairs at the prior bar. +ve = EUR strengthening. SUSPECT pending Step 6 audit. |
| `signal_density_28` | suspect | True | Count of pairs whose prior-bar mid-range exceeds that pair's trailing-100 mean range. 0..28. Coarse universe-volatility proxy. |
| `usd_strength_index` | suspect | True | Average signed 1-bar mid-return of USD across all USD-bearing pairs at the prior bar. +ve = USD strengthening. SUSPECT pending Step 6 audit of the cross-pair alignment + ffill semantics. |

### distance

| Name | Lineage | Needs panel | Description |
|---|---|---:|---|
| `distance_to_round_number` | clean | False | Minimum pip distance from prior mid-close to the nearest of three round-number grids (0.0050 / 0.0100 / 0.0500 price units). |
| `prior_session_high_distance` | clean | False | Distance from the prior-bar mid-close to the prior-calendar-day's bid-side session high. Positive ⇒ above prior day's high. |
| `prior_session_low_distance` | clean | False | Distance from the prior-bar mid-close to the prior-calendar-day's ask-side session low. Positive ⇒ above prior day's low. |

### multi_tf

| Name | Lineage | Needs panel | Description |
|---|---|---:|---|
| `d1_atr_percentile_100` | clean | True | Trailing-100 percentile rank of D1 Wilder ATR(14). Computed on the lag-1 D1 series so it's strictly prior at the signal bar. |
| `d1_close_slope_magnitude` | clean | True | Absolute D1 close-on-close change (lag-1). Pair with d1_close_slope_sign for directional + magnitude features. |
| `d1_close_slope_sign` | clean | True | Sign of the prior-D1 close-on-close slope. -1 / 0 / +1. Uses the L_PROTOCOL §1 one-day-lag rule (D1 bar from calendar day T-1 visible at T). |
| `w1_close_slope_sign` | clean | True | Sign of the prior-W1 close-on-close slope. -1 / 0 / +1. Strictly prior W1 bar (no exact-match alignment — week N's bar isn't visible until week N+1 starts). |

### price_geometry

| Name | Lineage | Needs panel | Description |
|---|---|---:|---|
| `atr_14` | clean | False | Wilder ATR(14) on mid-OHLC, shifted 1 to use bars closed strictly before signal. |
| `kijun_26_distance` | clean | False | Distance from prior mid-close to prior Kijun-sen(26). Positive = price above Kijun. Shifted 1 — uses strictly prior bars only. |
| `range_close_ratio` | clean | False | Prior-bar (high - low) / close. Bar-1 volatility proxy. |
| `swing_high_distance_14` | clean | False | Distance from prior mid-close to the 14-bar trailing swing high (also on prior bars). Positive — swing high is above current close. |
| `swing_low_distance_14` | clean | False | Distance from prior mid-close to the 14-bar trailing swing low. Positive — current close is above the swing low. |

### session

| Name | Lineage | Needs panel | Description |
|---|---|---:|---|
| `day_of_week` | clean | False | UTC weekday (Mon=0..Sun=6). |
| `hour_of_day` | clean | False | UTC hour of the bar's left edge (0-23). |
| `session_dead` | clean | False | Dead-zone flag — UTC 21-07 (wraps midnight). |
| `session_ldn_ny_overlap` | clean | False | London/NY overlap flag (UTC 12-16). |
| `session_london` | clean | False | London session flag (UTC 07-16). |
| `session_ny` | clean | False | NY session flag (UTC 12-21). |
| `session_tokyo` | clean | False | Tokyo session flag (UTC 00-09). |

### spread_regime

| Name | Lineage | Needs panel | Description |
|---|---|---:|---|
| `spread_percentile_100` | clean | False | Trailing-100 percentile rank of the prior-bar spread. 0..1. |
| `spread_vs_trailing_100` | clean | False | Prior-bar spread divided by mean of the trailing 100-bar spread. >1 ⇒ widening spread regime. |

### vol_regime

| Name | Lineage | Needs panel | Description |
|---|---|---:|---|
| `atr_percentile_100` | clean | False | Trailing-100 percentile rank of Wilder ATR(14). 0..1 (1 = highest vol in the trailing window). Uses strictly prior bars only. |
| `atr_vs_trailing_100` | clean | False | Wilder ATR(14) at signal bar (computed on prior data) divided by the mean of the trailing 100-bar ATR series. >1 ⇒ vol expansion. |

## Causal audit hooks

Step 6 (lazy, deployment-only per L_PROTOCOL §2 Step 6) reads the lineage 
tags above and runs a producer-level trace for each feature in the 
winning candidate. The audit verifies:

1. The producer reads only OHLC / spread / panel columns it declares 
   in `FeatureSpec.inputs`.
2. Every input series is shifted to bars closed strictly before the 
   signal-bar open (`shift(1)` or merge_asof(direction="backward", 
   allow_exact_matches=False) for cross-TF alignment).
3. End-to-end check: regenerate a small random sample of feature values 
   from raw OHLC, byte-compare to the cached pool values.

A `suspect` feature that survives the audit is promoted to `clean` 
in this file as part of the closure doc PR. A `suspect` feature 
flagged by the audit gets the candidate downgraded or killed per 
L_PROTOCOL §2 Step 6.

