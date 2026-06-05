# Arc 1019 — IS results (month-end reversion SHORT). OOS NOT touched (holdout preserved, §4).

All numbers scored by `MultiPairBacktester` (FundedNext costs ON, SL-first take-the-loss) via the canonical
`ArcFoldRunner` over `build_v3_folds` IS folds (fold_id 2–11 = OOS years 2011–2020; fold 6 = 2015, fold 9 =
2018). Pool n=116 (floor PASS), gross mean_final_r +0.1713.

## Observation (gross, characterization only — `_disco_work/arc1019_observe_monthend_short.py`)

| UP move into month-end | n | mean | median | frac short-pos |
|---|---|---|---|---|
| ≥ +1.0 ATR | 122 | +0.075 | +0.096 | 0.566 |
| +0.5..+1.0 | 147 | +0.070 | +0.050 | 0.531 |

**Month-end-vs-random-day control (≥+1 ATR up move):** MONTH-END +0.075 (median +0.096) vs RANDOM −0.015
(median −0.009) → **+0.089 ATR month-end excess** (timing load-bearing, not generic reversion).

**Honest +1R-before-SL short capture (month-end ≥+1 ATR):** pooled **0.5508** (n=118) — first corpus short
> 0.50. Per-year: 2015 cap 0.545, **2018 cap 0.818**.

**Per-year gross short drift (≥+1 ATR month-end):** 2015 +0.464 (med +0.273, frac .727), **2018 +0.365**
(med +0.122, frac .727); positive years 2010/11/13/15/17/18/19, negative 2014/16/20 (2020 mean −0.20 but
median +0.10 = a thin down-tail). **Per-pair:** EURUSD +0.137, GBPUSD +0.286, USDJPY +0.146, USDCAD +0.096,
USDCHF +0.008; AUDUSD −0.100, NZDUSD −0.274 (the opposite pairs from the long `me`).

## Honest-engine WFO — exit sweep (`_disco_work/arc1019_wfo.py`)

| exit | folds_pos | mean | worst | 2015 | 2018 | AFP |
|---|---|---|---|---|---|---|
| sl_only + time-exit 2-bar | 5/10 | +0.043% | −0.93% | +0.61 | +0.14 | N |
| sl_only + time-exit 3-bar | 6/10 | +0.102% | −1.52% | +1.84 | +0.14 | N |
| sl_only + time-exit 5-bar | 6/10 | +0.218% | −1.32% | +1.85 | +0.54 | N |
| sl_plus_tp_2r | 5/10 | +0.362% | −1.49% | +1.50 | −0.43 | N |
| **sl_partial_close_1r_runner_trail** | **7/10** | **+0.683%** | **−0.91%** | **+0.40** | **+0.86** | N |
| FAIR NULL (random entry, sl_only 2-bar) | 2/10 | −0.761% | −2.35% | −1.13 | +0.51 | N |

**Real beats the fair same-exit null by +0.80pp** (sl_only te2: +0.043% vs −0.761%; 5/10 vs 2/10).

Per-fold (partial-runner, 7 pairs): `2011 +3.39 · 2012 +1.69 · 2013 −0.90 · 2014 +0.98 · 2015 +0.40 ·
2016 −0.91 · 2017 −0.68 · 2018 +0.86 · 2019 +1.29 · 2020 +0.71`.

## Robustness (`_disco_work/arc1019_robust.py`, partial-runner exit)

**Threshold sweep:**

| thr | folds_pos | mean | worst | 2015 | 2018 |
|---|---|---|---|---|---|
| 0.75 | 8/10 | +0.839% | −1.96% | +0.54 | +1.72 |
| 1.0 | 7/10 | +0.683% | −0.91% | +0.40 | +0.86 |
| 1.25 | 8/10 | +0.753% | −0.81% | −0.76 | +1.21 |
| 1.5 | 6/10 | +0.312% | −0.80% | −0.76 | +0.68 |

→ **2018 positive at every threshold; 2015 positive only at thr ≤1.0 (fragile).**

**Leave-one-pair-out (thr 1.0):** 2018 positive in EVERY drop (+0.59 to +1.99). drop-EURUSD 9/10 +0.82%;
**drop-GBPUSD 5/10 −0.016% (2015 −0.16)** — GBPUSD is load-bearing for 2015 (not 2018); all other drops
6–8/10 positive. → **2018 robustly broad; 2015 leans on GBPUSD.**

## Disposition
NOT all-folds-positive (best 7/10) → FAIL sole judge → **PORTFOLIO**. Mean-positive every exit, beats null
+0.80pp, mechanism-controlled, **robustly 2018-positive**, 2015 fragile. OOS preserved.
