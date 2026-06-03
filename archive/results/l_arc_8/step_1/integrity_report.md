# Arc 8 — Step 1 Integrity Report

_Generated: 2026-05-22T07:10:31.051734+00:00Z_

## Pool size

- **Total trades:** 6,757
- **Coverage:** 2010-01-11 08:00:00+00:00 → 2026-04-10 12:00:00+00:00

### Per-pair trade counts

| Pair | Trades |
|---|---:|
| AUDCAD | 261 |
| AUDCHF | 247 |
| AUDJPY | 280 |
| AUDNZD | 231 |
| AUDUSD | 247 |
| CADCHF | 234 |
| CADJPY | 252 |
| CHFJPY | 255 |
| EURAUD | 232 |
| EURCAD | 223 |
| EURCHF | 209 |
| EURGBP | 215 |
| EURJPY | 242 |
| EURNZD | 239 |
| EURUSD | 249 |
| GBPAUD | 240 |
| GBPCAD | 216 |
| GBPCHF | 231 |
| GBPJPY | 255 |
| GBPNZD | 234 |
| GBPUSD | 233 |
| NZDCAD | 241 |
| NZDCHF | 239 |
| NZDJPY | 271 |
| NZDUSD | 262 |
| USDCAD | 227 |
| USDCHF | 234 |
| USDJPY | 258 |

### Per-pair signal-stage diagnostics

| Pair | Bars | Trend OK | Pullback OK | Resume OK | Signals (spaced) |
|---|---:|---:|---:|---:|---:|
| AUDCAD | 26,164 | 3,603 | 2,464 | 503 | 261 |
| AUDCHF | 26,169 | 3,352 | 2,208 | 461 | 247 |
| AUDJPY | 26,168 | 4,086 | 2,500 | 531 | 280 |
| AUDNZD | 26,168 | 3,235 | 2,186 | 417 | 231 |
| AUDUSD | 26,163 | 3,406 | 2,342 | 461 | 247 |
| CADCHF | 26,165 | 3,263 | 2,240 | 454 | 234 |
| CADJPY | 26,168 | 3,651 | 2,419 | 485 | 252 |
| CHFJPY | 26,167 | 3,489 | 2,296 | 468 | 255 |
| EURAUD | 26,164 | 3,219 | 2,237 | 428 | 232 |
| EURCAD | 26,168 | 3,261 | 2,217 | 447 | 223 |
| EURCHF | 26,170 | 2,838 | 1,980 | 382 | 209 |
| EURGBP | 26,166 | 3,187 | 2,251 | 411 | 215 |
| EURJPY | 26,161 | 3,319 | 2,057 | 431 | 242 |
| EURNZD | 26,166 | 3,532 | 2,391 | 448 | 239 |
| EURUSD | 26,170 | 3,241 | 2,283 | 483 | 249 |
| GBPAUD | 26,076 | 3,429 | 2,503 | 479 | 240 |
| GBPCAD | 26,147 | 3,199 | 2,128 | 412 | 216 |
| GBPCHF | 26,160 | 3,346 | 2,255 | 442 | 231 |
| GBPJPY | 26,165 | 3,717 | 2,359 | 513 | 255 |
| GBPNZD | 26,154 | 3,163 | 2,188 | 456 | 234 |
| GBPUSD | 26,169 | 3,321 | 2,217 | 440 | 233 |
| NZDCAD | 26,164 | 3,347 | 2,341 | 468 | 241 |
| NZDCHF | 26,168 | 3,459 | 2,320 | 460 | 239 |
| NZDJPY | 26,164 | 4,142 | 2,632 | 534 | 271 |
| NZDUSD | 26,162 | 3,473 | 2,374 | 515 | 262 |
| USDCAD | 26,164 | 3,346 | 2,331 | 481 | 227 |
| USDCHF | 26,165 | 3,448 | 2,200 | 457 | 234 |
| USDJPY | 26,165 | 3,503 | 2,341 | 492 | 258 |

## Spread data-quality flag rate per pair

Per L_PROTOCOL §1 (chat F4): no fallback. Reported as `1 - frac(quality == ok)`.

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
    "signal_time": "2023-10-16 00:00:00+00:00",
    "re_fires": true,
    "most_recent_sh_bar": 22143.0,
    "right_edge_lag_bars": 19.0,
    "right_edge_ok": true
  },
  {
    "pair": "EURJPY",
    "signal_time": "2015-01-22 00:00:00+00:00",
    "re_fires": true,
    "most_recent_sh_bar": 8116.0,
    "right_edge_lag_bars": 9.0,
    "right_edge_ok": true
  },
  {
    "pair": "GBPJPY",
    "signal_time": "2018-06-25 12:00:00+00:00",
    "re_fires": true,
    "most_recent_sh_bar": 13622.0,
    "right_edge_lag_bars": 8.0,
    "right_edge_ok": true
  },
  {
    "pair": "AUDJPY",
    "signal_time": "2015-02-20 08:00:00+00:00",
    "re_fires": true,
    "most_recent_sh_bar": 8240.0,
    "right_edge_lag_bars": 16.0,
    "right_edge_ok": true
  },
  {
    "pair": "EURJPY",
    "signal_time": "2012-08-08 16:00:00+00:00",
    "re_fires": true,
    "most_recent_sh_bar": 4191.0,
    "right_edge_lag_bars": 6.0,
    "right_edge_ok": true
  }
]
```

## D1-lag NaN perturbation

- **Result:** PASS
- PR-HHHL signal trigger is computed entirely from H4 bid OHLC + mid ATR. No D1 series is read inside evaluate_pullback_resume_hhhl_signal. NaN-perturbation of D1 inputs cannot change signal evaluation — test trivially passes by construction. Cross-asset / D1 features that enter the Step 1 FEATURE matrix (e.g. d1_close_slope_sign) carry their own lag-1 handling and are covered by tests/test_features_multi_tf.py.

## KH-24 co-fire rate (informational)

- **Overall:** 0 co-fires across 6,757 Arc 8 signals (rate 0.0000%).

| Pair | Arc 8 signals | KH-24 co-fires | Rate |
|---|---:|---:|---:|
| AUDCAD | 261 | 0 | 0.0000% |
| AUDCHF | 247 | 0 | 0.0000% |
| AUDJPY | 280 | 0 | 0.0000% |
| AUDNZD | 231 | 0 | 0.0000% |
| AUDUSD | 247 | 0 | 0.0000% |
| CADCHF | 234 | 0 | 0.0000% |
| CADJPY | 252 | 0 | 0.0000% |
| CHFJPY | 255 | 0 | 0.0000% |
| EURAUD | 232 | 0 | 0.0000% |
| EURCAD | 223 | 0 | 0.0000% |
| EURCHF | 209 | 0 | 0.0000% |
| EURGBP | 215 | 0 | 0.0000% |
| EURJPY | 242 | 0 | 0.0000% |
| EURNZD | 239 | 0 | 0.0000% |
| EURUSD | 249 | 0 | 0.0000% |
| GBPAUD | 240 | 0 | 0.0000% |
| GBPCAD | 216 | 0 | 0.0000% |
| GBPCHF | 231 | 0 | 0.0000% |
| GBPJPY | 255 | 0 | 0.0000% |
| GBPNZD | 234 | 0 | 0.0000% |
| GBPUSD | 233 | 0 | 0.0000% |
| NZDCAD | 241 | 0 | 0.0000% |
| NZDCHF | 239 | 0 | 0.0000% |
| NZDJPY | 271 | 0 | 0.0000% |
| NZDUSD | 262 | 0 | 0.0000% |
| USDCAD | 227 | 0 | 0.0000% |
| USDCHF | 234 | 0 | 0.0000% |
| USDJPY | 258 | 0 | 0.0000% |

## Determinism — pool content sha256

`bbaa0d42afed6c01d9c969984cf8f9212f1c2a6443ab9da28a77fc5e0252c618`

Stable sha (sorted rows by `(pair, signal_time)`, sorted columns) — re-run from same seed reproduces this value.

## Feature catalogue (v3.0)

27 features × 6,757 trades. Lineage summary:

| Class | clean | suspect | unverified |
|---|---:|---:|---:|
| cross_pair | 0 | 4 | 0 |
| distance | 3 | 0 | 0 |
| multi_tf | 4 | 0 | 0 |
| price_geometry | 5 | 0 | 0 |
| session | 7 | 0 | 0 |
| spread_regime | 2 | 0 | 0 |
| vol_regime | 2 | 0 | 0 |

Cross-asset features (DXY, US10Y) listed in L_PROTOCOL §2 Step 1 are NOT in the v3.0 27-feature catalogue per chat F7 — running as-is.
