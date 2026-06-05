# PORTFOLIO CANDIDATE — arc 2013: Weekend UP-gap Short (JPY crosses) — the corpus's FIRST short component

> **Disposition: PORTFOLIO** (protocol §11) — mean-positive net of costs but **NOT**
> all-folds-positive. A **CANDIDATE component, NOT deployable solo**, never a survivor. It is a
> decorrelated input to a future portfolio-combination arc — itself gated by all-folds-positive WFO
> on the *combined* book. It never auto-deploys.
>
> **First-of-kind:** this is the **first SHORT component** in the programme, and the **first
> end-to-end short engine run** (arcs 1014/2009/2011/3010 were short OBSERVATIONS only; the canonical
> short pool/engine path is now exercised — it builds sign-correctly, no canonical-core change). Scored
> solely by `MultiPairBacktester` (FundedNext costs ON, SL-first take-the-loss).

## Why PORTFOLIO (not PASS, not KILL)
- **Mean-positive net of costs:** IS mean fold ROI **+0.745%** (best overshoot exit `sl_plus_trailing_atr`,
  thr up-gap≥1.0). Positive across the WHOLE mechanism-aligned overshoot-exit family
  (trailing_atr +0.745%, tp_3r +0.510%, partial_runner +0.324%, tp_2r +0.015%) — NOT exit-fishing; the
  overshoot exits are pre-specified by the 1006/1007 gap-overshoot mechanism. Non-overshoot exits
  (`sl_only` −1.97%, `trailing_swing` −1.43%) are negative, as the mechanism predicts.
- **Beats its fair same-side null by a large margin:** a random JPY-cross weekly-open SHORT at the same
  fire-count LOSES (5-seed mean −0.99% under trailing_atr, −1.17% under tp_3r — the JPY-basket drifts UP
  against shorts) → **lift +1.74% / +1.68%**, much cleaner than the 1006 long's +0.36pp null margin
  (arc 1009). The up-gap TIMING carries real information.
- **But NOT all-folds-positive:** IS 5/10 folds negative (worst −3.18%) → fails the sole discovery judge
  → not a survivor (no `passed/`, council not reached). Thin (~14 trades/yr; some folds 1–2 trades — the
  arc-2001 thinness caveat).
- **Not KILL:** genuinely net-positive AND beats its null. You cannot diversify a net-negative component
  positive (arcs 3000/3001), so net-negative = KILL — this clears that bar.

## Why this is the most portfolio-relevant component since 1013 — the 2018 leg
The combination route (arcs 2006/2008/3009/1015) is **provably blocked by 2015 & 2018**: all three
existing components are LONG flow/fade-reversion edges that bleed strong-USD/risk-off trend years. 1015's
4th-component spec: **"positive in BOTH 2015 & 2018 → structurally a SHORT/USD-trend leg."** This component
delivers the **2018 half strongly**: IS 2018 fold **+5.08%** (and 2019 **+5.92%**) — where gap-fill
(−6.79) and fbr (−4.20) both deep-bleed and only month-end (+0.90) held. It bleeds a **different** set
(2013/14/16) than the long fades (2015/16/18/20) → genuinely regime-complementary on the binding 2018
fold. (It does NOT cover 2015 — fold 2015 +/−? is −0.57% on n=2, noise — so the 2015 half of the spec is
still open.) **Flagged for a future 4-way combination arc** (gap-fill 1006 + month-end 1011 + fbr 1013 +
up-gap-short 2013): 2018 may now be satisfiable by up-gap-short + month-end, freeing fbr weight for 2015.

## Mechanism (the *because*)
Weekend illiquidity reprices the weekly open away from the prior Friday close; large gaps revert (fill)
toward it — symmetric in gap size (arc 2001). The UP-gap fill = SHORT the gap, betting on reversion DOWN.
On JPY CROSSES (not majors — majors' up-gap CONTINUES up, drift −0.355 short = dead, the 1006 cross>major
asymmetry mirrored). The edge is the fill-OVERSHOOT (price runs past the prior close), best captured by a
let-the-down-move-run trailing exit (mirrors arc 1007: capping at the fill target is worse).

## Contents
| File | What |
|---|---|
| [`config.yaml`](./config.yaml) | Exact signal + engine config + repro command. |
| [`is_oos_results.md`](./is_oos_results.md) | Per-fold IS series (by year), exit-menu sweep, fair-null comparison, gross pool stats. OOS PRESERVED. |

## OOS status — PRESERVED (deliberately not measured)
IS is NOT all-folds-positive (won't PASS) and the exit was IS-selected, so per the arc-2001/1011
discipline the 2021+ holdout is **kept pristine**. The 4-way combination arc is the proper
all-folds-positive gate that will touch OOS on the *combined* book.
