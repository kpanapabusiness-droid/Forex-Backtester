# Phase KG-L-V2 Report: D1 ATR Cap + signal_flip Exit Removed

> Generated: 2026-04-20
> Base: KG-L-V1 (KG-J + trail fix + classification fix + cond_9)
>
> **V2 additional change — signal_flip exit REMOVED:**
>   KG-L-V1 exits: signal_flip | kijun_d1 | trailing_stop | stoploss
>   KG-L-V2 exits:              kijun_d1 | trailing_stop | stoploss
>   signal_flip is not computed and not checked anywhere in V2.
>
> Inherited from V1 (all three fixes present in V2):
>   Fix 1: Trail activation close-based (not intrabar high)
>   Fix 2: Classification from r_multiple (WIN/LOSS/SCRATCH)
>   Change 3: cond_9 — D1 close <= D1 Kijun(26) + 1.0 × D1 ATR(14)
>
> Timeframe: 4H | Pairs: 28 | WFO: 7 anchored-expanding folds
> Gate: worst-fold OOS ROI > 0% AND max DD < 8%

---

## 1. Gate Result

| Criterion | Threshold | Actual | Result |
|-----------|-----------|--------|--------|
| Worst-fold ROI | > 0.0% | 1.92% | **PASS** |
| Worst-fold DD | < 8.0% | 6.37% | **PASS** |
| **Overall Gate** | both | — | **PASS** |

---

## 2. Full Fold Table

| Fold | OOS Start | OOS End | Trades | KGJ | Δ | c9_blk | ROI% | MaxDD% | WinRate | ExpR | AvgBars | Gate |
|------|-----------|---------|--------|-----|---|--------|------|--------|---------|------|---------|------|
| 1 | 2020-10-01 | 2021-07-01 | 41 | 50 | -9 | 100 | 13.35% | 6.37% | 0.4390 | 0.3413 | 14.2 | PASS |
| 2 | 2021-07-01 | 2022-04-01 | 36 | 47 | -11 | 86 | 9.63% | 4.45% | 0.5833 | 0.2784 | 11.6 | PASS |
| 3 | 2022-04-01 | 2023-01-01 | 25 | 48 | -23 | 106 | 11.90% | 4.43% | 0.5600 | 0.4791 | 24.5 | PASS |
| 4 | 2023-01-01 | 2023-10-01 | 32 | 40 | -8 | 94 | 3.32% | 3.80% | 0.4688 | 0.1178 | 18.3 | PASS |
| 5 | 2023-10-01 | 2024-07-01 | 23 | 34 | -11 | 106 | 6.23% | 3.09% | 0.5217 | 0.2833 | 18.1 | PASS |
| 6 | 2024-07-01 | 2025-04-01 | 30 | 27 | +3 | 78 | 3.24% | 5.03% | 0.4333 | 0.1403 | 14.5 | PASS |
| 7 | 2025-04-01 | 2026-01-01 | 27 | 41 | -14 | 83 | 1.92% | 4.06% | 0.5185 | 0.0817 | 14.6 | PASS |

**Worst-fold ROI:** 1.92% | **Median-fold ROI:** 6.23% | **Best-fold ROI:** 13.35%
**Worst-fold DD:** 6.37% | **Median-fold DD:** 4.43%
**Gate: PASS**

---

## 3. Trade Count Delta vs KG-J

| Fold | KGJ | KGL-V2 | Δ | cond_9 blocked OOS |
|------|-----|--------|---|---------------------|
| 1 | 50 | 41 | -9 | 100 |
| 2 | 47 | 36 | -11 | 86 |
| 3 | 48 | 25 | -23 | 106 |
| 4 | 40 | 32 | -8 | 94 |
| 5 | 34 | 23 | -11 | 106 |
| 6 | 27 | 30 | +3 | 78 |
| 7 | 41 | 27 | -14 | 83 |
| **Total** | **287** | **214** | **-73** | **653** |

Total cond_9 blocked (all-time): 1840
Total cond_9 blocked (OOS only): 653

---

## 4. Exit Reason Breakdown (OOS)

OOS total: 214
Note: signal_flip is disabled in V2 — kijun_d1 absorbs those exits.

| Exit Reason | Count | % of OOS | Avg R | Avg Bars |
|-------------|-------|----------|-------|----------|
| kh13_early         |     0 |    0.0% |       — |        — |
| kh14_bar6          |     0 |    0.0% |       — |        — |
| kijun_d1           |    66 |   30.8% | -0.1737 |      7.7 |
| trailing_stop      |    90 |   42.1% |  1.3573 |     27.2 |
| stoploss           |    58 |   27.1% | -1.0000 |      8.3 |

---

## 5. KG-J → KG-L-V2 Comparison (fold by fold)

### ROI%

| Fold | OOS Period | KG-J | KG-L-V2 | Delta |
|------|------------|------|---------|-------|
| 1 | 2020-10-01 – 2021-07-01 | 12.57% | 13.35% | +0.78% |
| 2 | 2021-07-01 – 2022-04-01 | 3.65% | 9.63% | +5.98% |
| 3 | 2022-04-01 – 2023-01-01 | 24.46% | 11.90% | -12.56% |
| 4 | 2023-01-01 – 2023-10-01 | -1.13% | 3.32% | +4.45% |
| 5 | 2023-10-01 – 2024-07-01 | 9.69% | 6.23% | -3.46% |
| 6 | 2024-07-01 – 2025-04-01 | 0.62% | 3.24% | +2.62% |
| 7 | 2025-04-01 – 2026-01-01 | 3.91% | 1.92% | -1.99% |

### MaxDD%

| Fold | OOS Period | KG-J | KG-L-V2 | Delta |
|------|------------|------|---------|-------|
| 1 | 2020-10-01 – 2021-07-01 | 9.71% | 6.37% | -3.34% |
| 2 | 2021-07-01 – 2022-04-01 | 9.75% | 4.45% | -5.30% |
| 3 | 2022-04-01 – 2023-01-01 | 3.97% | 4.43% | +0.46% |
| 4 | 2023-01-01 – 2023-10-01 | 6.28% | 3.80% | -2.48% |
| 5 | 2023-10-01 – 2024-07-01 | 6.32% | 3.09% | -3.23% |
| 6 | 2024-07-01 – 2025-04-01 | 7.75% | 5.03% | -2.72% |
| 7 | 2025-04-01 – 2026-01-01 | 5.13% | 4.06% | -1.07% |

---

## 6. SL Distance Sanity Check

Original trades (553):  min sl_distance_atr=2.000000  max=2.000000  all exactly 2.0: **YES — OK**

---

## 7. Trades Blocked by Exposure Cap

Total blocked (all time): 9
Total blocked (OOS only): 7

## 8. Per-Pair OOS Performance

| Pair | Trades | ROI% | WinRate | ExpR | AvgBars |
|------|--------|------|---------|------|---------|
| AUD_CAD | 5 | 7.11% | 0.8000 | 1.4388 | 22.4 |
| AUD_CHF | 10 | -2.77% | 0.4000 | -0.2603 | 10.7 |
| AUD_JPY | 9 | 6.40% | 0.5556 | 0.7192 | 24.8 |
| AUD_NZD | 6 | -0.22% | 0.5000 | -0.0132 | 30.5 |
| AUD_USD | 5 | 0.46% | 0.4000 | 0.1033 | 16.0 |
| CAD_CHF | 14 | -3.36% | 0.4286 | -0.2264 | 10.8 |
| CAD_JPY | 10 | -1.98% | 0.4000 | -0.1945 | 10.6 |
| CHF_JPY | 3 | -1.29% | 0.3333 | -0.4218 | 8.3 |
| EUR_AUD | 9 | 3.15% | 0.4444 | 0.3555 | 17.4 |
| EUR_CAD | 5 | -1.31% | 0.4000 | -0.2496 | 20.0 |
| EUR_CHF | 10 | 13.28% | 0.5000 | 1.3838 | 32.3 |
| EUR_GBP | 14 | -0.28% | 0.4286 | 0.0057 | 9.5 |
| EUR_JPY | 7 | -0.91% | 0.2857 | -0.1182 | 18.3 |
| EUR_NZD | 7 | 4.79% | 0.5714 | 0.6941 | 14.0 |
| EUR_USD | 6 | -0.29% | 0.5000 | -0.0375 | 9.0 |
| GBP_AUD | 11 | 0.85% | 0.5455 | 0.0872 | 13.5 |
| GBP_CAD | 7 | 0.45% | 0.5714 | 0.0781 | 9.7 |
| GBP_CHF | 5 | 0.63% | 0.6000 | 0.1476 | 13.0 |
| GBP_JPY | 4 | 1.57% | 0.5000 | 0.4021 | 21.8 |
| GBP_NZD | 8 | 0.32% | 0.5000 | 0.0750 | 7.9 |
| GBP_USD | 6 | 0.01% | 0.5000 | 0.0124 | 14.8 |
| NZD_CAD | 8 | 2.38% | 0.6250 | 0.3087 | 21.4 |
| NZD_CHF | 5 | -0.65% | 0.4000 | -0.1057 | 11.6 |
| NZD_JPY | 4 | 2.14% | 0.7500 | 0.5428 | 13.8 |
| NZD_USD | 9 | 8.08% | 0.8889 | 0.9016 | 21.9 |
| USD_CAD | 7 | 0.42% | 0.4286 | 0.0650 | 7.4 |
| USD_CHF | 11 | 1.09% | 0.4545 | 0.1066 | 12.4 |
| USD_JPY | 9 | 9.51% | 0.4444 | 1.0575 | 30.4 |

---

## 9. KH-16 Re-entry Analysis (OOS)

Re-entry disabled (USE_REENTRY=False).

---

## 10. Methodology Notes

**signal_flip exit — DISABLED:**
- V1 exits: signal_flip | kijun_d1 | trailing_stop | stoploss
- V2 exits:              kijun_d1 | trailing_stop | stoploss
- signal_flip is not computed on any bar and never checked in the loop.
- Effect: trades that would have exited on signal_flip in V1 now remain
  open until kijun_d1, trailing_stop, or stoploss fires.
- Expected: longer avg_bars_held, potentially higher or lower R per trade
  depending on whether the signal_flip correctly predicted direction.

| Exit | Condition |
|------|-----------|
| kijun_d1 | D1 close < D1 Kijun(26), asof 4H bar date |
| trailing_stop | close ≤ best_close − 1.5 × ATR, gated on trail_active |
| stoploss | intrabar low ≤ entry_px − 2.0 × ATR |
| trail activation | bar close ≥ entry_px + 2.0 × ATR (CLOSE-BASED) |
