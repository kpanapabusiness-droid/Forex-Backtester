# l_arc_8_v3.0.2 — Step 1 Integrity Report

_Generated: 2026-05-25T07:57:32.248307+00:00Z_

**Boundary convention:** `5ers_eet` (locked end-to-end per corrected dispatch §0.5).

## Pool size

- **Total trades:** 6,612
- **Coverage:** 2010-01-12 22:00:00+00:00 → 2025-12-31 14:00:00+00:00

### Arc 8 v3.0 (UTC) reference — informational only

- **Arc 8 v3.0 (UTC) n:** 6,757
- **Arc 8 v3.0.2 (5ers_eet) n:** 6,612
- **Delta:** -2.1%

Note: UTC vs 5ers_eet pools are not directly comparable (H4 boundary shifts the signal evaluation grid). Delta is reported for awareness only, not as a calibration target.

### Per-pair trade counts

| Pair | Trades |
|---|---:|
| AUDCAD | 255 |
| AUDCHF | 233 |
| AUDJPY | 270 |
| AUDNZD | 232 |
| AUDUSD | 239 |
| CADCHF | 231 |
| CADJPY | 246 |
| CHFJPY | 259 |
| EURAUD | 220 |
| EURCAD | 215 |
| EURCHF | 195 |
| EURGBP | 218 |
| EURJPY | 243 |
| EURNZD | 238 |
| EURUSD | 247 |
| GBPAUD | 237 |
| GBPCAD | 225 |
| GBPCHF | 215 |
| GBPJPY | 254 |
| GBPNZD | 233 |
| GBPUSD | 240 |
| NZDCAD | 230 |
| NZDCHF | 233 |
| NZDJPY | 263 |
| NZDUSD | 249 |
| USDCAD | 225 |
| USDCHF | 211 |
| USDJPY | 256 |

### Per-pair signal-stage diagnostics

| Pair | Bars | Trend OK | Pullback OK | Resume OK | Signals (spaced) |
|---|---:|---:|---:|---:|---:|
| AUDCAD | 25,405 | 3,547 | 2,471 | 512 | 255 |
| AUDCHF | 25,407 | 3,226 | 2,073 | 437 | 233 |
| AUDJPY | 25,403 | 3,902 | 2,457 | 516 | 270 |
| AUDNZD | 25,406 | 3,072 | 2,129 | 402 | 232 |
| AUDUSD | 25,401 | 3,292 | 2,239 | 462 | 239 |
| CADCHF | 25,400 | 3,052 | 2,121 | 431 | 231 |
| CADJPY | 25,403 | 3,558 | 2,360 | 480 | 246 |
| CHFJPY | 25,400 | 3,536 | 2,372 | 488 | 259 |
| EURAUD | 25,405 | 3,199 | 2,276 | 443 | 220 |
| EURCAD | 25,406 | 3,112 | 2,126 | 409 | 215 |
| EURCHF | 25,401 | 2,690 | 1,883 | 377 | 195 |
| EURGBP | 25,399 | 3,194 | 2,245 | 428 | 218 |
| EURJPY | 25,409 | 3,422 | 2,176 | 438 | 243 |
| EURNZD | 25,399 | 3,301 | 2,233 | 433 | 238 |
| EURUSD | 25,405 | 3,270 | 2,302 | 478 | 247 |
| GBPAUD | 25,319 | 3,427 | 2,497 | 474 | 237 |
| GBPCAD | 25,388 | 3,306 | 2,180 | 442 | 225 |
| GBPCHF | 25,402 | 3,208 | 2,134 | 420 | 215 |
| GBPJPY | 25,401 | 3,744 | 2,359 | 511 | 254 |
| GBPNZD | 25,383 | 3,023 | 2,061 | 433 | 233 |
| GBPUSD | 25,397 | 3,182 | 2,149 | 444 | 240 |
| NZDCAD | 25,402 | 3,322 | 2,362 | 431 | 230 |
| NZDCHF | 25,400 | 3,240 | 2,189 | 434 | 233 |
| NZDJPY | 25,403 | 4,092 | 2,527 | 517 | 263 |
| NZDUSD | 25,398 | 3,513 | 2,333 | 477 | 249 |
| USDCAD | 25,401 | 3,119 | 2,122 | 424 | 225 |
| USDCHF | 25,403 | 3,164 | 2,041 | 410 | 211 |
| USDJPY | 25,401 | 3,544 | 2,389 | 488 | 256 |

## Multi-TF feature NaN coverage (Arc 8 original closure §2 caveat 2 check + W1 fix verification)

| Feature | NaN rate |
|---|---:|
| `d1_atr_percentile_100` | 2.06% |
| `d1_close_slope_magnitude` | 0.00% |
| `d1_close_slope_sign` | 0.00% |
| `w1_close_slope_sign` | 0.03% |

**✓ multi_tf features computed cleanly — Arc 8 v3 caveat 2 closed under 5ers_eet.**

## Spread data-quality flag rate per pair

Per L_PROTOCOL §1: no fallback. Reported as `1 - frac(quality == ok)`.

| Pair | Flag rate |
|---|---:|
| AUDCAD | 0.0000% |
| AUDCHF | 0.0000% |
| AUDJPY | 0.0000% |
| AUDNZD | 0.0000% |
| AUDUSD | 0.0000% |
| CADCHF | 0.0000% |
| CADJPY | 0.0000% |
| CHFJPY | 0.0000% |
| EURAUD | 0.0000% |
| EURCAD | 0.0000% |
| EURCHF | 0.0000% |
| EURGBP | 0.0000% |
| EURJPY | 0.0000% |
| EURNZD | 0.0000% |
| EURUSD | 0.0000% |
| GBPAUD | 0.0000% |
| GBPCAD | 0.0000% |
| GBPCHF | 0.0000% |
| GBPJPY | 0.0000% |
| GBPNZD | 0.0000% |
| GBPUSD | 0.0000% |
| NZDCAD | 0.0000% |
| NZDCHF | 0.0000% |
| NZDJPY | 0.0000% |
| NZDUSD | 0.0000% |
| USDCAD | 0.0000% |
| USDCHF | 0.0000% |
| USDJPY | 0.0000% |

## Lookahead spot-check + right-edge swing audit

- **Result:** PASS
- **Samples:** 5 (pass=5 / fail=0)

Per-sample detail:

```
[
  {
    "pair": "NZDCAD",
    "signal_time": "2021-04-06 13:00:00+00:00",
    "re_fires": true,
    "most_recent_sh_bar": 17855.0,
    "right_edge_lag_bars": 5.0,
    "right_edge_ok": true
  },
  {
    "pair": "EURJPY",
    "signal_time": "2017-03-03 10:00:00+00:00",
    "re_fires": true,
    "most_recent_sh_bar": 11369.0,
    "right_edge_lag_bars": 7.0,
    "right_edge_ok": true
  },
  {
    "pair": "GBPJPY",
    "signal_time": "2017-05-05 09:00:00+00:00",
    "re_fires": true,
    "most_recent_sh_bar": 11639.0,
    "right_edge_lag_bars": 6.0,
    "right_edge_ok": true
  },
  {
    "pair": "AUDJPY",
    "signal_time": "2016-03-15 22:00:00+00:00",
    "re_fires": true,
    "most_recent_sh_bar": 9817.0,
    "right_edge_lag_bars": 11.0,
    "right_edge_ok": true
  },
  {
    "pair": "EURJPY",
    "signal_time": "2013-09-12 21:00:00+00:00",
    "re_fires": true,
    "most_recent_sh_bar": 5882.0,
    "right_edge_lag_bars": 11.0,
    "right_edge_ok": true
  }
]
```

## D1-lag NaN perturbation

- **Result:** PASS
- PR-HHHL signal trigger is computed entirely from H4 bid OHLC + mid ATR. No D1 series is read inside evaluate_pullback_resume_hhhl_signal. NaN-perturbation of D1 inputs cannot change signal evaluation — test trivially passes by construction.

## KH-24 co-fire rate (informational)

- **Overall:** 0 co-fires across 6,612 Arc 8 signals (rate 0.0000%).

| Pair | Arc 8 signals | KH-24 co-fires | Rate |
|---|---:|---:|---:|
| AUDCAD | 255 | 0 | 0.0000% |
| AUDCHF | 233 | 0 | 0.0000% |
| AUDJPY | 270 | 0 | 0.0000% |
| AUDNZD | 232 | 0 | 0.0000% |
| AUDUSD | 239 | 0 | 0.0000% |
| CADCHF | 231 | 0 | 0.0000% |
| CADJPY | 246 | 0 | 0.0000% |
| CHFJPY | 259 | 0 | 0.0000% |
| EURAUD | 220 | 0 | 0.0000% |
| EURCAD | 215 | 0 | 0.0000% |
| EURCHF | 195 | 0 | 0.0000% |
| EURGBP | 218 | 0 | 0.0000% |
| EURJPY | 243 | 0 | 0.0000% |
| EURNZD | 238 | 0 | 0.0000% |
| EURUSD | 247 | 0 | 0.0000% |
| GBPAUD | 237 | 0 | 0.0000% |
| GBPCAD | 225 | 0 | 0.0000% |
| GBPCHF | 215 | 0 | 0.0000% |
| GBPJPY | 254 | 0 | 0.0000% |
| GBPNZD | 233 | 0 | 0.0000% |
| GBPUSD | 240 | 0 | 0.0000% |
| NZDCAD | 230 | 0 | 0.0000% |
| NZDCHF | 233 | 0 | 0.0000% |
| NZDJPY | 263 | 0 | 0.0000% |
| NZDUSD | 249 | 0 | 0.0000% |
| USDCAD | 225 | 0 | 0.0000% |
| USDCHF | 211 | 0 | 0.0000% |
| USDJPY | 256 | 0 | 0.0000% |

## Determinism — pool content sha256

`d3543b1a996743d4b289cc1b0ee09becdd90e450c72cd65ed5fdccf3a7cc25d5`

Stable sha (sorted rows by `(pair, signal_time)`, sorted columns) — re-run from same seed reproduces this value.

## Feature catalogue (v3.0)

27 features × 6,612 trades. Lineage summary:

| Class | clean | suspect | unverified |
|---|---:|---:|---:|
| cross_pair | 0 | 4 | 0 |
| distance | 3 | 0 | 0 |
| multi_tf | 4 | 0 | 0 |
| price_geometry | 5 | 0 | 0 |
| session | 7 | 0 | 0 |
| spread_regime | 2 | 0 | 0 |
| vol_regime | 2 | 0 | 0 |
