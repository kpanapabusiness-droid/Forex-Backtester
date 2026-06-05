# arc 1013 — IS + OOS results (failed-breakdown reclaim long)

All scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first, take-the-loss), judged by
`judge_all_folds_positive`. Committed config: K=40, shadow≥1.25, `sl_plus_trailing_atr`, SL=2.0.
fold_id→OOS-year anchor: fold 6 = 2015 (arc 2006).

## Per-fold IS (build_v3_folds, is_days≥365 → 10 folds)

| fold | year | ROI | maxDD | n |
|---|---|---|---|---|
| 2 | 2011 | +7.55% | 2.02% | 20 |
| 3 | 2012 | +3.05% | 2.92% | 17 |
| 4 | 2013 | +0.91% | 4.39% | 24 |
| 5 | 2014 | +0.19% | 4.28% | 28 |
| 6 | 2015 | **+3.17%** | 2.02% | 16 |
| 7 | 2016 | **+2.55%** | 3.79% | 21 |
| 8 | 2017 | +1.23% | 3.78% | 22 |
| 9 | 2018 | **−4.20%** | 4.87% | 18 |
| 10 | 2019 | +0.05% | 3.52% | 24 |
| 11 | 2020 | **+4.03%** | 1.98% | 18 |

**IS: mean +1.854%, 9/10 positive (only 2018), worst −4.20%, maxDD 4.87%, minN 16.** NOT all-folds-positive.

## Per-fold OOS (build_oos_year_folds(2021)) — ONE-SHOT frozen, not tuned

| year | ROI | maxDD | n |
|---|---|---|---|
| 2021 | +1.63% | 2.10% | 23 |
| 2022 | −1.39% | 2.89% | 17 |
| 2023 | +1.10% | 4.20% | 19 |
| 2024 | +7.61% | 3.29% | 28 |
| 2025 | −2.76% | 3.76% | 14 |
| 2026 | −0.57% | 2.63% | 5 (partial year) |

**OOS: mean +0.936%, 3/6 positive, worst −2.76%, maxDD 4.20%.** Mean-POSITIVE forward (edge persists).

## §5f exit menu (K=40, shadow=1.25, SL=2.0) — every exit mean-POSITIVE

| exit | AFP | mean | worst | neg | maxDD | minN |
|---|---|---|---|---|---|---|
| sl_only | N | +1.43% | −4.63% | 2/10 | 6.40% | 16 |
| sl_plus_tp_2r | N | +1.37% | −4.63% | 2/10 | 5.52% | 16 |
| sl_plus_tp_3r | N | +1.57% | −4.63% | 2/10 | 5.54% | 16 |
| **sl_plus_trailing_atr** | N | **+1.85%** | −4.20% | 1/10 | 4.87% | 16 |
| sl_plus_trailing_swing | N | +1.61% | −4.85% | 3/10 | 5.74% | 16 |
| sl_partial_close_1r_runner_trail | N | +1.37% | −2.88% | 2/10 | 4.05% | 26 |
| sl_only+time4 | N | +0.83% | −1.24% | 3/10 | 3.44% | 17 |
| sl_only+time6 | N | +1.38% | −2.62% | 2/10 | 4.33% | 16 |
| sl_only+time8 | N | +1.59% | −2.68% | 2/10 | 3.88% | 16 |
| sl_only+time12 | N | +1.66% | −2.56% | 3/10 | 4.46% | 16 |

## Fair same-exit NULL (random entry, matched fire-rate, same exit/pairs) — DECISIVE

| | mean | neg | worst |
|---|---|---|---|
| REAL (K40 sh1.25) | **+1.854%** | 1/10 | −4.20% |
| NULL seed 42 | −1.048% | 7/10 | −3.99% |
| NULL seed 7 | −0.999% | 5/10 | −5.59% |
| NULL seed 123 | −1.275% | 7/10 | −6.14% |

**REAL +1.854% vs NULL avg −1.107% → excess +2.96pp.** Null is firmly NEGATIVE — the structure is the
entire edge.

## Structure control (the mechanism test — gross observation, IS)

| rejection magnitude | (A) AT swept swing low | (B) elsewhere (control) |
|---|---|---|
| shadow≥1.0 | n=507 cap 0.5227 drift +0.049 | n=1577 cap 0.4807 drift −0.049 |
| shadow≥1.25 | n=237 cap 0.5865 drift +0.146 | n=634 cap 0.4905 drift −0.217 |
| shadow≥1.5 | n=131 cap 0.6107 drift +0.203 | n=296 cap 0.5068 drift −0.176 |

Same-magnitude wick AT a swept swing low captures 0.52→0.61; elsewhere a coin-flip with negative drift.
Structure is load-bearing; excess grows with rejection size.

## K/shadow robustness (exit=sl_plus_trailing_atr, SL=2.0)

| K | shadow | mean | worst | neg | minN |
|---|---|---|---|---|---|
| 40 | 1.00 | +0.20% | −7.32% | 4/10 | 29 |
| 40 | 1.25 | **+1.85%** | −4.20% | 1/10 | 16 |
| 40 | 1.50 | +1.50% | −1.05% | 2/10 | 6 |
| 60 | 1.00 | +0.83% | −4.18% | 5/10 | 27 |
| 60 | 1.25 | +1.62% | −2.68% | 2/10 | 14 |
| 60 | 1.50 | +1.47% | −0.96% | 2/10 | 4 |

shadow≥1.25 is the load-bearing rejection gate (1.0 dilutes to 4–5/10 neg); robust across K.

## Leave-one-pair-out (committed config, IS)

| dropped | mean | worst | neg |
|---|---|---|---|
| none (all 7) | +1.854% | −4.20% | 1/10 |
| EURUSD | +1.175% | −3.19% | 4/10 |
| GBPUSD | +1.710% | −3.24% | 3/10 |
| AUDUSD | +1.616% | −3.39% | 2/10 |
| NZDUSD | +1.505% | −3.68% | 2/10 |
| USDCAD | +1.414% | −4.50% | 2/10 |
| USDCHF | +1.759% | −2.03% | 3/10 |
| USDJPY | +1.925% | −3.70% | 1/10 |

Mean positive dropping ANY pair — broad-based (EURUSD strongest contributor, USDJPY mild drag).
