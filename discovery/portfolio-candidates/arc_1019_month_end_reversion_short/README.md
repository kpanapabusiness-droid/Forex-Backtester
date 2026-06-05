# PORTFOLIO CANDIDATE — arc 1019: Month-End Reversion SHORT (USD majors)

> **Disposition: PORTFOLIO** (protocol §11) — mean-positive net of costs but **NOT** all-folds-positive.
> A **CANDIDATE component, NOT deployable solo**, never a survivor. A decorrelated input to the gated
> 4-way portfolio-combination arc (1020) — itself gated by all-folds-positive WFO on the *combined* book.
> Never auto-deploys.
>
> **Provenance:** produced live by arc 1019 (NOT transcribed) — full record
> [`../../arcs/arc_1019_month_end_reversion_short.md`](../../arcs/arc_1019_month_end_reversion_short.md) and
> Tier-1 ledger row 1019 in [`../../DISCOVERY_LOG.md`](../../DISCOVERY_LOG.md). Scored solely by
> `MultiPairBacktester` (FundedNext costs ON, SL-first). The signal is the registered
> [`MonthEndReversionShortSignal`](../../tools/month_end_signals.py) → turnkey-reproducible (see `config.yaml`).

## Why this matters — the FIRST robustly-2018-positive component
The deployment route (a decorrelated long+short book) was **provably blocked** (arcs 1015/2008/3009; 0/5151
convex weightings all-folds-positive) by two strong-USD folds, **2015 and 2018**. ~12 routes to a
2018-positive leg died (structural/trend/flow/vol shorts, relative-value, deep-continuation long, USD-neutral
gap-fill, weekly trend, end-of-week squaring). **This is the first construction that is robustly positive in
2018** — under every exit (except tp_2r), every threshold (0.75–1.5), and every leave-one-pair-out drop —
with a fragile-but-present 2015 tilt. It directly attacks the binding wall the 3-way book could not clear.

## Why PORTFOLIO (not PASS, not KILL)
- **Mean-positive net of costs under EVERY exit:** IS mean fold ROI +0.04% (sl_only/2-bar) → **+0.68%**
  (sl_partial_close_1r_runner_trail, 7/10 folds, worst −0.91%); positive across thresholds 0.75–1.5.
- **Beats the fair same-exit null by +0.80pp** (real sl_only-2bar +0.043% vs random-entry −0.761%, matched
  fire-rate, same exit/SL/universe; 5/10 vs 2/10). Corroborated by the gross **month-end-vs-random-day
  control: +0.089 ATR excess** (a big up move into month-end reverts; the SAME move on a random day does not
  → the month-end *timing* is the mechanism, mirroring the long `me`).
- **Mechanism-controlled & not a thin-tail mirage:** observation median ≈ mean (+0.096 vs +0.075 at ≥+1 ATR),
  honest short capture 0.5508 > 0.50 (the first corpus short to clear 0.50). 2018 capture 0.818.
- **But NOT all-folds-positive:** best 7/10 (negatives 2013/2016/2017) → fails the sole discovery judge →
  not a survivor (no `passed/` record; council not reached). OOS deliberately NOT touched.
- **Not KILL:** genuinely net-positive (you cannot diversify a net-negative component positive, arcs
  3000/3001) → it lands here.

## The honest caveat
The **2018 leg is the robust one** (positive across every exit/threshold/pair-drop). The **2015 leg is
fragile** — it lives at threshold ≤1.0 and leans on GBPUSD (drop-GBPUSD flips 2015 to −0.16 and the mean to
~0). For the 4-way book this is acceptable because 2015 is already strongly handled by fbr (+3.17); this
component's unique value is the robust 2018 contributor that only `me` weakly provided.

## Contents
| File | What |
|---|---|
| [`config.yaml`](./config.yaml) | Exact signal + engine config — turnkey-reproducible (registered signal). |
| [`is_oos_results.md`](./is_oos_results.md) | Per-fold IS series, exit sweep, null, control, robustness. |
| [`correlation_profile.md`](./correlation_profile.md) | Correlation + per-fold vectors vs the candidate set; regime-complementarity on 2015/2018. |

## The 4-way combination is the gated next arc (1020)
Four net-positive components now exist (gap 1006, me 1011, fbr 1013, this). For the first time one is
robustly 2018-positive. The combination is its OWN gated arc (all-folds-positive WFO on the co-simulated /
`combine_fold_roi` combined book) — **arc 1020**, NOT claimed by this record.
