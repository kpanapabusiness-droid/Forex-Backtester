# Arc 0 — raw run outputs (audit trail)

Verbatim console output from the scratch drivers (`_arc0_work/`, not committed), honest engine,
FundedNext costs ON, H4 5ers_eet, 8 pairs, data from the recovered backup corpus. Date 2026-06-04.

---

## (c) build_pool.py — population + clustering + capturability

```
panel pairs: ['AUDUSD','EURGBP','EURUSD','GBPUSD','NZDUSD','USDCAD','USDCHF','USDJPY']  timestamps: 25862
pool_sha256: 1b653aaaa3a62c1f5846c2d1165bd43df716d57d26a7fd6b0c963982c43ebb4e
TOTAL IS trades: 4985
per-pair: AUDUSD 601, EURGBP 621, EURUSD 612, GBPUSD 659, NZDUSD 606, USDCAD 667, USDCHF 619, USDJPY 600
POOL FLOOR: n=4985 -> PASS (>=50)
exit_reason: hard_sl 3968, time_exit 1017
final_r: mean -0.045 median -1.000 p10 -1.000 p90 +3.172
mfe_r:   mean +1.961 median +0.999 p10 +0.088 p90 +5.166
mae_r:   mean -1.157 median -1.131 p10 -1.608 p90 -0.504
win rate (final_r>0) = 0.195
frac reach >=1R MFE = 0.500 ; >=2R = 0.327 ; >=3R = 0.233
frac reached +1R BEFORE SL (honest) = 0.498

STEP 2 clustering: k_selected=4  silhouette {2:0.398,3:0.382,4:0.477,5:0.438,6:0.421}
  cluster 0 n=733  bimodal        mfe_p50_proxy=3.10
  cluster 1 n=1742 unclassified   mfe_p50_proxy=0.68
  cluster 2 n=1428 unclassified   mfe_p50_proxy=-0.12
  cluster 3 n=1081 monotonic_down mfe_p50_proxy=-0.59
STEP 3 capturability:
  cluster 0: candidate=True  composite=1.651 reach_1r=1.000 mfe_p50=5.916 wrong_way_pp=0.003
  cluster 1: candidate=True  composite=0.942 reach_1r=0.963 mfe_p50=1.969 wrong_way_pp=0.038
  cluster 2: candidate=False composite=0.192 reach_1r=0.055 mfe_p50=0.510 wrong_way_pp=0.958
  cluster 3: candidate=False composite=0.025 reach_1r=0.004 mfe_p50=0.100 wrong_way_pp=0.997
```

## (d) cheap_kills.py — oracle ceiling + raw triage (3 representative folds)

```
best candidate cluster = 0 [is_candidate max composite]
ORACLE (cluster 0 only, perfect hindsight, SL+trail, costed):
  fold 4 (OOS 2013): roi +0.3820 dd 0.0113 n 39 ratio +33.79
  fold 7 (OOS 2016): roi +0.2617 dd 0.0111 n 27 ratio +23.50
  fold 10 (OOS 2019): roi +0.1274 dd 0.0227 n 21 ratio +5.62
  oracle worst-fold ROI = +0.1274
RAW (all signals, SL+trail, costed):
  fold 4 (OOS 2013): roi +0.0785 dd 0.0664 n 156 ratio +1.18
  fold 7 (OOS 2016): roi -0.0296 dd 0.0960 n 156 ratio -0.31
  fold 10 (OOS 2019): roi -0.1403 dd 0.1512 n 154 ratio -0.93
  raw worst -0.1403 mean -0.0305
VERDICT: ceiling strong (+12.7%), raw not deeply negative -> PROCEED to diagnose
```

## (e) diagnose.py — entry-feature separation (good={0,1} vs bad={2,3})

```
trades 4984  good 2475  bad 2509  good_fraction 0.497
univariate rank-AUC: atr_pct 0.527, rsi14 0.514, ext_above_sma50 0.507, dist_sma200 0.503,
                     trend_struct 0.501, sma50_slope20 0.491, pullback_depth 0.488
structural filter lift (good_frac vs base 0.497): trend_struct==1 0.495; rsi<40 0.333(n15);
  rsi>50 0.502; ext>1 0.496; slope>0.5 0.492; trend&slope 0.494; trend&rsi>45&slope 0.497
=> NO structural filter lifts the good-fraction. Selection not extractable from entry structure.
```

## (f)+(g) wfo_validate.py — full IS WFO + per-year OOS (honest engine, costs ON)

```
IS (OOS years 2011-2020, 10 folds):
  partial_1r_runner_trail: per-fold roi
    2011 -0.0526 | 2012 -0.0139 | 2013 +0.0569 | 2014 -0.1408 | 2015 +0.0285
    2016 -0.0108 | 2017 +0.0120 | 2018 -0.1129 | 2019 -0.0969 | 2020 -0.1417
    -> worst -0.1417  n_negative 7/10  ALL-FOLDS-POSITIVE False
  fixed_tp_2r:  worst -0.2007  n_negative 7/10  False
  sl_only:      worst -0.2636  n_negative 4/10  False (thin n, some daily breaches)

OOS (per-year 2021-2026, measured not tuned):
  partial_1r_runner_trail: 2021 -0.0592 | 2022 -0.0270 | 2023 -0.0443 | 2024 -0.1351 |
    2025 -0.1425 | 2026 +0.0070  -> worst -0.1425  n_negative 5/6  False
  fixed_tp_2r:  worst -0.1790  n_negative 6/6  False
  sl_only:      worst -0.1563  n_negative 3/6  False

DISCOVERY JUDGE: NO config achieves IS all-folds-positive => FAIL.
```

## null_compare.py — council-mandated random-entry baseline (partial exit, 10 IS folds)

```
REAL signal (7446 fires): per-fold roi
  [-0.0526,-0.0139,0.0569,-0.1408,0.0285,-0.0108,0.0120,-0.1129,-0.0969,-0.1417]
  mean -0.0472  worst -0.1417  n_neg 7/10
NULL random entry (~7446 fires):
  seed  42: mean -0.0835 worst -0.2243 n_neg 9/10
  seed   7: mean -0.0837 worst -0.1240 n_neg 10/10
  seed 123: mean -0.0941 worst -0.2240 n_neg 9/10
SOUNDNESS READ: real mean -0.0472 vs null mean -0.0871 ; real worst -0.1417 vs null worst -0.1908
  -> real signal BEATS random entry => residual edge exists, but insufficient to pass the judge.
```
