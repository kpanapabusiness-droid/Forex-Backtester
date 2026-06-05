# arc 3022 — cost-robustness (cost cushion) of the 4-component book

**Chat:** 3000s · **Date:** 2026-06-05 · **Verdict:** DIAGNOSTIC → **KILL** (no new component; all 4 components UNCHANGED, PORTFOLIO; book stays strict-gate FAIL)
**Disposition:** KILL · **passed:** N · **Component touched:** none (cost-axis book characterization)

> The whole corpus's "sub-cost" kills and the book's "mean-positive **net of costs**" pass are measured
> at ONE cost point — FundedNext default. Nobody has stress-tested the deployability conclusion against
> that single assumption. This maps the book on the cost axis (the last unmapped one, after noise
> 2016/2017/1023, time 2021, densification 3021).
> **Result: the book's mean-positive edge is COST-ROBUST, not knife-edge — break-even at ~3.3× FundedNext
> cost; FundedNext eats ~30% of gross, 70% survives. AND the AFP failure is NOT a cost artifact — the
> book fails the per-year gate even COST-FREE (κ=0, 2/10 neg).** Cost is not the wall; the per-year gate
> vs thin components is. Removes the "is the deploy case a cost artifact?" objection to path-A (as arc
> 2021 removed the temporal-decay objection). Path-A remains the only deployability lever.

---

## Log reading (step a — FRESH EYES, honest-era only)

Pulled main; no `discovery/STOP`. Resumed 3000s after my arc 3021 (highest in-range 3021 → 3022).
Converged corpus state (~57 arcs, honest-era): closed-ground shallow directional space (long AND short,
all TF H1/H4/D1/W1); **4 net-positive PORTFOLIO components** — gap-fill 1006 (JPY-cross H4), month-end-long
1011 (USD-major D1), failed-breakdown-reclaim `fbr` 1013 (USD-major H4, the crown jewel +1.854%/9-of-10),
month-end-short 1019 (USD-major D1, the first robustly-+2018 leg). The 4-way book is a sound ~3-independent-
bet (ENB 3.32, arc 2019), mean-positive (t=2.66, P(mean<0)=0.004, arc 1023; risk-parity +0.589%, arc 2019),
temporally robust (arc 2021) PORTFOLIO whose all-folds-positive (AFP) failure is purely the per-year gate
sitting **below the legs' noise floor** (2016/2017). The edge frontier is closed on every documented lever;
**path-B (densification) is now PROVABLY closed** (my arc 3021: at ρ=0.1 P(AFP) plateaus ~0.33, never
reaches 0.9 at any N) → the sole deployability lever is the operator's **path-A (gate-governance call).**

The one quantity the whole corpus treats as fixed but never checks: **the cost assumption.** Every kill
("sub-cost") and the book's pass ("mean-positive net of costs") is at FundedNext default (1.5× spread,
$5/lot RT, 0.5 pip/fill, no swaps). The book's *net* edge (+0.589%) is small vs gross — so the cushion
could be thin. Untested. That gap is this arc.

## Idea + why (an adversarial stress on the cost axis — try to BREAK the deploy case)

A book that is mean +0.589% net of FundedNext but would go NEGATIVE at modestly higher *realistic* live
cost is NOT deployable under ANY gate (path-A is then moot). Conversely, if the edge survives well past
the modeled cost, the deployability conclusion is robust to the one assumption nobody has swept. Same
posture as arc 2021 (stress the time axis) — conservative bias = try to break the mean.

**Mechanism note (why this is a clean, faithful test).** The engine is GROSS (SL-honest take-the-loss);
broker cost is netted post-hoc by debiting each closed position's cost from the equity curve
(`build_fold_stats_from_run` → `apply_cost_model`). In the A1 architecture the entry/SL/exit geometry is
gross-price-based, so the trade SET is invariant to cost — which means I can run each component's gross
fold ONCE and re-net the SAME `RunResult` at a sweep of cost multipliers κ (no engine re-run). This is the
faithful "same strategy, different broker cost" sensitivity, and the conservative one: a cost-aware live
trader who *skipped* marginal trades at high cost could only do better, so the cost-passive break-even is
a LOWER bound on the real cushion.

## Method (CALLS canonical; the only experiment piece is a scaled cost vector)

Driver: `_disco3_work/arc3022_cost_robustness.py`. Reproduce all 4 components via the registered EXPERIMENT
signals + `make_time_exit_predicate` + `combine_fold_roi`, scored through the canonical apparatus
(`ArcFoldRunner` → `A1Architecture` → `MultiPairBacktester`), at risk_pct 0.005 (0.5%, the deployable
linear band, arc 1024). Exact committed configs (arc 3009/2015): gap `WeekendGapFillLongSignal(0.5,
gap_hours=36)` 5 JPY crosses H4, 24-bar time exit, `sl_only`, trail off; me_long `MonthEndReversionLong
(1.0, into=2)` 7 majors D1, 2-bar time exit, `sl_only`, trail off; fbr `FailedBreakdownReclaim(K40,
shadow1.25)` 7 majors H4, `sl_plus_trailing_atr`, trail ON (the native double-trail, arc 3009); me_short
`MonthEndReversionShort(1.0, into=2)` 7 majors D1, `sl_partial_close_1r_runner_trail`. **A1Config exposure
caps left at DEFAULT (per-currency 2, per-pair 1)** — reproducing live caught that overriding them to None
inflated the JPY-cross / USD-major clustered fires (gap +1.23% vs +0.685%); the default per-currency cap is
load-bearing for the headline (Arc-10 reproduce-don't-transcribe discipline working again, cf. arc 3009's
trail nuance).

**The one experiment tool (BUILT):** `discovery/tools/cost_scaling.scaled_fundednext(κ)` — returns a
`CostModel` equal to κ × the FundedNext cost VECTOR (spread_mult = 1+0.5κ, commission $5κ/lot, slippage
0.5κ pip/fill). κ=1 = FundedNext exactly; κ=0 = cost-free (≡ `CostModel.zero`); κ>1 = harsher. The cost
MATH stays canonical (`apply_cost_model`); only the parameter vector is scaled. Sweep κ ∈ {0, 0.5, 1, 1.5,
2, 3, 4, 5}; re-net each component's gross per-fold `RunResult` at each κ via the canonical
`build_fold_stats_from_run(..., cost_model=scaled_fundednext(κ))`.

## Reproduction gate (Arc-10 discipline — all 4 BYTE-EXACT at κ=1)

```
gap      : -0.07, +8.23, -2.06, +2.94, -4.19, +3.20, +0.53, -6.79, +7.45, -2.39   mean +0.685%  ✓ (=3009)
me_long  : +0.40, +0.29, +0.96, -0.23, -1.14, -0.51, +0.34, +0.90, +1.16, +0.15   mean +0.232%  ✓ (=3009)
fbr      : +7.55, +3.05, +0.91, +0.19, +3.17, +2.55, +1.23, -4.20, +0.05, +4.03   mean +1.854%  ✓ (=3009)
me_short : +3.39, +1.69, -0.90, +0.98, +0.40, -0.91, -0.68, +0.86, +1.29, +0.71   mean +0.683%  ✓ (2015:2018 +0.86 / 2015 +0.40)
```
Risk-parity book at κ=1 = **+0.5889%** = arc 2019's +0.589% to the basis point. Full corpus reproduction.

## Results — cost cushion

**Per-component mean ROI % vs κ (provably linear — additive cost re-netted on a fixed gross trade set):**
```
component   k=0    k=0.5  k=1(FN) k=1.5  k=2    k=3    k=4    k=5
gap        +1.308 +0.996 +0.685  +0.373 +0.061 -0.563 -1.189 -1.816
me_long    +0.400 +0.316 +0.232  +0.148 +0.064 -0.104 -0.272 -0.440
fbr        +2.507 +2.180 +1.854  +1.527 +1.200 +0.544 -0.114 -0.773
me_short   +0.824 +0.753 +0.683  +0.612 +0.541 +0.400 +0.259 +0.118
```
**Per-component cost cushion (drag = fraction of gross eaten by FundedNext; break-even κ where mean→0):**
| component | gross | net(FN) | cost-drag %of-gross | break-even κ |
|---|---|---|---|---|
| gap | +1.308% | +0.685% | **47.6%** | **2.10** |
| me_long | +0.400% | +0.232% | 42.0% | 2.38 |
| fbr | +2.507% | +1.854% | 26.0% | 3.84 |
| me_short | +0.824% | +0.683% | **17.1%** | **5.84** |

**Book (risk-parity, the deployable allocation; weights 0.077/0.523/0.121/0.280 frozen at κ=1):**
| κ | book mean % | worst fold % | n_neg |
|---|---|---|---|
| 0 (gross) | +0.843 | −0.265 | 2/10 |
| 1 (FundedNext) | **+0.589** | −0.422 | 2/10 |
| 2 | +0.335 | −0.598 | 5/10 |
| 3 | +0.080 | −0.882 | 6/10 |
| 4 | −0.174 | −1.167 | 7/10 |
| 5 | −0.429 | −1.451 | 7/10 |

**Break-even κ = 3.32** (risk-parity); equal-weight book gives 3.17 (consistent). Cost eats **30.1%** of
the book's gross edge; **70% survives**; the book stays positive out to ~3× FundedNext cost.

## Reading the result

1. **The book's mean-positive edge is COST-ROBUST, not knife-edge.** It would take ~3.3× the modeled
   FundedNext cost to erase it — far outside any plausible live-vs-modeled cost gap (FundedNext is already
   the conservative bound: 1.5× spread, full-size slippage). The +0.589% deploy case does NOT hinge on a
   fragile cost model. This **removes a potential objection to path-A** (the deploy case is not a
   cost-modeling artifact) — the cost-axis analog of arc 2021's "not a temporal-decay artifact."

2. **The AFP gate failure is NOT a cost artifact (independent confirmation of arcs 2016/2017).** Even
   COST-FREE (κ=0) the risk-parity book is 2/10 negative — NOT all-folds-positive. Raising cost makes the
   gate monotonically harder (n_neg 2→7 across κ 0→5) but the failure already exists at ZERO cost. So the
   per-year-gate failure is structural (thin-component noise floor + `fbr`'s intrinsic −2018), nothing to
   do with the cost assumption. **Cost is not the wall; the per-year gate vs thin components is** — exactly
   the corpus's converged diagnosis, now confirmed from the cost angle.

3. **Per-component cost-sensitivity is mechanism-meaningful.** Drag fraction tracks gross-edge-per-trade ×
   trade-frequency × spread: `me_short` (D1, infrequent, strong overshoot edge) has the fattest cushion
   (17% drag, break-even 5.8); `gap` (H4, frequent, JPY-cross wider spread, thinnest gross-per-trade) is
   the cost-FRAGILE leg (47.6% drag, break-even 2.1) — consistent with gap being the most fold-fragile
   component (arc 1009: "lives at 0.5 ATR, ~0 by 1.25"). `fbr` (strongest, +2.5% gross) sits at 3.84.

4. **The principled (risk-parity) weighting happens to also be the cost-robust one.** Risk-parity
   down-weights the cost-fragile high-vol `gap` to 0.077 and up-weights the cost-robust low-vol `me_long`/
   `me_short` (0.523/0.280) → the BOOK cushion (3.32) is healthier than its weakest leg (gap 2.10). No
   conflict between variance-optimal and cost-optimal allocation here.

## Verdict — DIAGNOSTIC → KILL (no new component; book stays strict-gate FAIL)

No edge found or claimed; this is a cost-axis characterization of the existing book (like the noise-floor
2016/2017/1023, time 2021, and densification 3021 diagnostics). The 4 components are UNCHANGED — reproduced
byte-exact, retain PORTFOLIO. The strict per-year AFP gate STAYS FAIL (this does NOT loosen it; it shows
the failure is cost-independent). No OOS spent (IS book characterization; the book fails IS AFP → OOS
unearned, §5g). No council (a measurement characterizing a measurement, cf. 1023/3021; §7 council is for
idea-fork/diagnosis/survivor, not a diagnostic).

**Net for the operator's path-A call:** the deploy case now has three robustness legs — mean significantly
positive (t=2.66, arc 1023), temporally stable (arc 2021), and **cost-robust (break-even ~3.3× FundedNext,
this arc)** — and the AFP failure is confirmed cost-independent. Path-A (gate-governance) is unblocked of
the cost objection; path-B stays closed (arc 3021).

## Threads / lessons

1. **NEW lesson — separate cost-robustness from gate-satisfiability.** A thin-component book can be
   simultaneously (a) cost-robust in its MEAN (survives 3× cost) and (b) cost-SENSITIVE in its per-year
   gate (n_neg climbs with cost). The two are different questions: the mean cushion measures whether the
   edge is real net of plausible cost; the gate sensitivity is moot once you know the gate already fails
   cost-free. Report both; don't conflate "survives cost" with "passes the gate."
2. **Re-netting a fixed gross run at κ is the cheap, faithful cost-sensitivity instrument** — one engine
   run per component, re-net for free at any cost vector (the trade set is gross-geometry-invariant in A1).
   The cost-passive break-even is a conservative LOWER bound (a cost-aware trader skipping marginal trades
   does better). Reusable for any future component or book.
3. **Default exposure caps are load-bearing for reproduction** — overriding per-currency to None inflated
   the clustered JPY-cross/USD-major fires (gap +1.23% vs +0.685%). Reproduce with A1Config DEFAULTS unless
   an arc documents otherwise; the per-currency=2 cap is part of the committed components (arc 1017's
   currency-cap effect, here as a reproduction requirement).
4. **The gap-fill is the corpus's most cost-fragile component** (break-even 2.1× vs the book's 3.3×) —
   relevant if anyone revisits it: it is the leg most exposed to a live cost realization above the model,
   and risk-parity's down-weighting of it is doing real cost-defense work, not just variance-defense.

## Tooling

BUILT: `discovery/tools/cost_scaling.scaled_fundednext(κ)` (registered). Reused the 4 registered signals +
`make_time_exit_predicate` + `combine_fold_roi` (`fit_weights`, `combine_fold_rois`), all CALLED; scoring
canonical (`ArcFoldRunner` → `A1Architecture` → `MultiPairBacktester`; cost netted via the canonical
`build_fold_stats_from_run` with the scaled `cost_model`). No canonical change; no FLAG.
