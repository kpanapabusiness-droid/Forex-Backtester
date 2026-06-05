# arc 1011 — per-fold IS results + soundness (PORTFOLIO component)

> Source: [`../../arcs/arc_1011_month_end_reversion_long.md`](../../arcs/arc_1011_month_end_reversion_long.md).
> Honest engine (`MultiPairBacktester`), FundedNext costs ON, SL-first. Produced live (NOT transcribed).

## Verdict
**FAIL the sole judge** (not all-folds-positive on IS) → **disposition PORTFOLIO** (mean-positive net of costs,
beats fair null, mechanism-controlled, robust — but fold-fragile 7/10). **OOS NOT touched** (holdout preserved).

## IS — 10 folds (2010–2020), sl_only + 2-bar time exit, thr=1.0
Per-fold ROI %, in fold order:

```
+0.40, +0.29, +0.96, -0.23, -1.14, -0.51, +0.34, +0.90, +1.16, +0.15
```

- all-folds-positive: **NO** — 7/10 positive (3 negative)
- worst fold ROI: **-1.14%**   ·   best: **+1.16%**
- mean fold ROI: **+0.23%**

## Exit sweep (full IS WFO, all canonical exits)
| exit | folds_pos | mean | worst |
|---|---|---|---|
| sl_only, 2-bar time exit | **7/10** | **+0.23%** | −1.14% |
| sl_only, 3-bar time exit | 6/10 | +0.50% | −0.79% |
| sl_only, 5-bar time exit | 4/10 | −0.03% | −1.38% |
| sl_partial_close_1r_runner_trail, 5-bar | 3/10 | −0.08% | −1.21% |
| sl_plus_tp_2r, 5-bar | 4/10 | −0.04% | −1.35% |

Mechanism-aligned short holds (2–3 bar) carry it; longer holds and TP/trail overlays dilute the reversion.
No exit reaches all-folds-positive (so no exit-fishing claim; OOS not touched).

## Null comparison (soundness control — fair, same-exit)
| | mean fold ROI | neg folds | worst |
|---|---|---|---|
| REAL signal (IS, sl_only 2-bar) | **+0.23%** | 3/10 | −1.14% |
| RANDOM entry (matched fire-rate, SAME exit/SL/universe) | −0.33% | 6/10 | −1.54% |

→ real **BEATS** the fair null by **+0.56pp** — the month-end-timing-specific excess.

## Mechanism control (the decisive test — month-end vs generic reversion)
Generic short-term reversion is dead (arcs 3000/3001): a big down move on a random day continues down. The
same ≤ −1 ATR 2-day down move, month-end vs random non-month-end day (gross fwd2, ATR units):

| ≤ −1 ATR down move | n | fwd2 (ATR) | frac_pos |
|---|---|---|---|
| MONTH-END | 125 | **+0.186** | 0.616 |
| RANDOM DAY | 2783 | **−0.063** | 0.480 |

**Month-end EXCESS = +0.249 ATR.** The timing is causal — this is a calendar-flow effect, not generic reversion.

## Robustness (arc-1009 audit discipline)
- **Threshold-robust:** thr 0.75/1.0/1.25/1.5 → mean +0.18/+0.23/+0.27/+0.19%, 6–7/10 positive. (More robust
  than the gap-fill, which lived only at 0.5 ATR.)
- **Leave-one-pair-out (sl_only 2-bar, thr 1.0):** all 7 drops stay positive (mean +0.15 to +0.28%, 6–7/10).
  NOT single-pair-driven (holds even dropping AUDUSD, the strongest single pair).
- **Per-pair (gross, observation):** AUDUSD/NZDUSD/GBPUSD/USDCHF strong (+0.32 to +0.47 ATR fwd2);
  EURUSD/USDJPY weak-negative → a future combination should weight toward the risk/commodity + CHF majors.

## Pool
121 IS trades · gross mean final_r +0.0635 (positive before costs / fold-fragility).
