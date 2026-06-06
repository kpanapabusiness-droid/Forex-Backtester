# arc 1043 — Honest-exit book-mean SIGNIFICANCE (resolving arc-1042 F3: does the t=2.66 deploy pillar survive?)

> chat: 1000s | range 1000-1999 | timestamp: 2026-06-06
> disposition: **DIAGNOSTIC → KILL** (no new component; resolves arc-1042 F3 with a computed number)
> components UNCHANGED (all 4 PORTFOLIO). OOS NOT touched (IS book mean only).

## (a) Log read / synthesis
Resuming directly on the thread I opened in arc 1042 (last 1000s arc). State unchanged: 4 PORTFOLIO
components, mean-positive 4-way book NOT all-folds-positive, edge-hunt structurally closed, §11
complete, sole lever = operator path-A. The path-A deploy case rests on FOUR characterization pillars:
**mean-positive + statistically significant** (arc 1023 t=2.66 / 2019 P(mean<0)=0.004), **temporally
stable** (2021), **cost-robust** (3022 κ=3.32), **~3 independent bets** (2019 ENB 3.32). Arcs
2040/2041 + my 1042 showed the committed component exits are §5f-forbidden full-sample picks; the
honest §5f book deploy mean is ~half (+0.27% RP) — and I FLAGGED (1042 F3) that arc-1023's t=2.66
significance, computed on the COMMITTED-exit series, likely does NOT survive honest exits (estimated
t≈1.2), owing an exact recompute. Independent concurrent arc 2042 (2000s) converged on the component
verdicts (gap-neg/me_long-robust).

## (b) Idea (the because)
The significance pillar is the single strongest quantitative argument in the entire deploy case: a
thin book that merely "looks positive" is not deployable, but one whose mean is **statistically
distinguishable from zero** (t=2.66, P(mean<0)=0.004) is a real positive-expectancy edge that fails
only the per-year gate (arcs 1023/2019/3021's framing of path-A). If that significance was itself an
artifact of full-sample exit-fishing (the thing 1042 proved inflates the point estimate ~2×), then
the deploy case is materially weaker than advertised. F3 is therefore the highest-value follow-up:
**recompute the book-mean significance on the HONEST §5f exit series, by the SAME method arc 1023 used
(fold-bootstrap of the 10 IS fold ROIs, seed 42), and compare to the committed t.**

## (c)/(f) Method
- 100% canonical scoring (`ArcFoldRunner`→A1→`MultiPairBacktester`, FundedNext); honest exits via
  BUILT `nested_exit_selection` (3 metrics); book combination via BUILT `combine_fold_roi`.
- Fold-bootstrap (arc-1023/2019 method): resample the 10 IS fold ROIs with replacement, N=10000,
  seed 42 → mean, 95% CI, P(mean<0); report the implied t = mean/(sd/√10) alongside.
- Compute for: COMMITTED-exit book (reproduce arc-1023 significance) + HONEST-exit book under all 3
  selection metrics × {committed RP weights, honest-refit RP weights}.
- Driver: `_disco_work/arc1043_honest_book_significance.py`. No gate reimplemented; OOS untouched
  (this is the IS book-mean significance only).

## Results

**Committed-exit book — reproduces the significance pillar.**
| book | mean | sd | t | 95% CI | P(mean<0) | neg-folds |
|---|---|---|---|---|---|---|
| committed RP | +0.594% | 0.868 | **+2.16** | [+0.11, +1.14] | **0.006** | 2/10 |
| committed EQUAL | +0.921% | 1.761 | +1.65 | [−0.14, +1.94] | 0.043 | 3/10 |

My committed-RP t=+2.16 / P(mean<0)=0.006 / CI [+0.11,+1.14] qualitatively REPRODUCES arc-1023's
t=2.66 / P=0 / CI [+0.22,+1.09] (the modest t gap is the trail-OFF fbr +2.084% used here vs 1023's
trail-ON double-trail +1.854%, plus fold-bootstrap nuance; both are clearly significant, t>2, CI
strictly >0). Anchor holds.

**Honest-§5f book — significance LOST across EVERY metric & weighting.**
| honest book | mean | t | 95% CI | P(mean<0) |
|---|---|---|---|---|
| mean_roi RP(commit-wts) | +0.407% | +1.21 | [−0.21, +1.05] | 0.108 |
| mean_roi RP(refit-wts) | +0.156% | +0.83 | [−0.19, +0.52] | 0.200 |
| afp_then_mean RP(commit-wts) | +0.291% | +1.15 | [−0.18, +0.76] | 0.113 |
| afp_then_mean RP(refit-wts) | +0.347% | +1.38 | [−0.13, +0.82] | 0.075 |
| worst_then_mean RP(commit-wts) | +0.267% | +1.38 | [−0.09, +0.63] | 0.076 |
| **worst_then_mean RP(refit-wts)** | +0.285% | **+1.52** | [−0.06, +0.63] | **0.054** |

⇒ **EVERY honest configuration is non-significant** — t ∈ [+0.83, +1.52] (all < 2), P(mean<0) ∈
[0.054, 0.20] (all > 0.05), **every 95% CI spans zero.** The single best honest case (worst_then_mean,
refit-wts) is t=1.52 / P=0.054 — borderline, still not significant. The honest §5f exit correction
**roughly HALVES the t-stat (committed +2.16 → honest ~+1.3) at constant n=10** — so the loss of
significance is specifically the exit correction, not a power change.

**Honest caveat (low power + method choice).** n=10 folds is low power, so "non-significant" partly
reflects 10-fold resolution — BUT the committed series CLEARS it at the SAME n=10 (t=2.16), so the
apples-to-apples statement is exact: honest exits remove the significance the committed exits had. A
per-TRADE bootstrap (arc 2016's more-powerful method) MIGHT tighten the honest CI; the fold-level
statement (the resolution arc 1023's t=2.66 deploy claim was argued at) stands — a per-trade honest
re-bootstrap is a possible follow-up, but it would have to overcome a point estimate that is itself
~half the committed.

## Verdict
**DIAGNOSTIC → KILL** (no new component; components UNCHANGED, all 4 PORTFOLIO). **arc-1042 F3
RESOLVED with a computed number: the book-mean SIGNIFICANCE does NOT survive honest §5f exits.** The
committed +0.59%/t=2.16 (≈arc-1023's t=2.66) was substantially an exit-selection artifact — under
honest no-lookahead exits the book mean is a positive POINT estimate (+0.27%) but is **statistically
indistinguishable from zero** (t≈1.2–1.5, every CI spans 0).

**Net for the operator's path-A:** of the four deploy-case pillars, the strongest one — "mean-positive
**AND statistically significant**" (arc 1023/2019) — is **the one that fails the honest-exit
correction.** The remaining pillars (temporally-stable 2021, cost-robust 3022, ~3-bet 2019) were also
computed on committed-exit series and are now suspect by the same mechanism (FLAG, below). The honest
deploy case is: a thin book with a positive but **not-significant** ~+0.27%/yr mean, carried by the
one exit-robust leg (me_long), still never AFP. This is a materially weaker deploy case than the
committed characterization advertised.

**FLAGS (docs only):**
- **F1 (extends 1042-F3 → resolved):** the path-A "significant mean-positive" pillar is an
  exit-artifact — honest book mean t≈1.3, CI spans zero, P(mean<0)≈0.05–0.11.
- **F2 (new, owed):** arcs 2021 (temporal stability), 3022 (cost-robustness κ=3.32) and 2019 (ENB,
  P(mean<0)=0.004) were ALL computed on the COMMITTED-exit component series — by the same mechanism
  that halved the mean, their conclusions may shift under honest exits. A clean honest-exit re-run of
  the book-characterization suite (temporal / cost / ENB) is the owed follow-up before path-A leans on
  any of them.

## (i) New lesson
The §5f exit-honesty correction does not merely halve a thin book's deploy MEAN (arc 1042) — it
removes the mean's STATISTICAL SIGNIFICANCE (fold-bootstrap t ≈2.16 → ≈1.3, CI flips from strictly-
positive to spanning zero), because full-sample-best exits inflate BOTH the point estimate AND the
t-stat. A thin mean-positive book's "statistically-significant" deploy claim (the load-bearing
quantitative pillar) must be re-bootstrapped under honest §5f nested-WFO exits before it can support a
deployment decision — and here it does not survive. Corollary: every OTHER book-characterization
statistic computed on the committed-exit series (temporal stability, cost cushion, ENB, noise floor)
inherits the same exit-optimism and is owed an honest-exit recheck.

## (k) Re-orient
Detail persisted (this doc + log). Tools REUSED (`nested_exit_selection`, `combine_fold_roi`); no new
BUILT tool. Driver `_disco_work/arc1043_honest_book_significance.py`. No canonical change, no council,
OOS untouched. Next: resume at arc 1044.
