# arc 1021 — Broad-universe month-end-SHORT (§5f best-version) + 4-way re-combination

**Chat:** 1000s · **Date:** 2026-06-05 · **Disposition:** KILL (the broad-universe "all-folds-positive
4-way book" is an in-sample-selection ARTIFACT — partial-runner fat-tail × IS-optimized weights; collapses
under honest sl_only exits and under non-optimized weights). Components UNCHANGED (narrow me-short retains
PORTFOLIO per 1019). · **Council:** none (the robustness check disqualified it before survivor status; council
is for IS+OOS survivors). **A deliberate Arc-10-defense near-miss catch.**

## Idea + why (the §5f best-version question arc 1019 left open)

The 4-way book (arc 1020, and independently 2000s arc 2015) is blocked at **2015 & 2016**, both marginal;
me-short's 2015 (+0.40) was shown to be **a single GBPUSD trade** (arc 2015/2000s; my arc-1019 LOO agreed).
arc 1019 claimed me-short on **7 USD majors** but never established the mechanism's best universe (cf. gap-fill,
found JPY-cross-specific). **Question:** does the month-end-short flow generalize to a broader universe, giving
a *robust, multi-pair* 2015 (and ideally 2016) that unblocks the book? Same proven mechanism, broader universe
= legitimate best-version development (developed entirely on IS; OOS untouched).

## Observation — the flow exists across groups, with complementary fold coverage

`_disco_work/arc1021_obs_universe.py` (D1, month-end up-move ≥+1 ATR → short reversion, gross):

| group | month-end mean / median | random-day | excess | 2015 | 2016 |
|---|---|---|---|---|---|
| USD majors (7) | +0.075 / +0.096 | −0.015 | +0.089 | **+0.464** | −0.069 |
| JPY crosses (5) | +0.026 / +0.048 | −0.033 | +0.059 | −0.115 (n5) | **+0.327** (n13) |
| EUR/GBP crosses (3) | +0.067 / +0.191 | −0.030 | +0.097 | **+0.405** (n6) | −0.462 (n7) |

Each group beats its random-day control (mechanism real in each). The lead: **JPY crosses carry 2016**
(+0.327) where USD majors are negative, and the EUR/GBP crosses add an *independent* 2015 (+0.405, not
GBPUSD). A broad universe pooled ≈ +0.32 (2015) / +0.01 (2016) — looked like it might cover BOTH binding folds.

## Engine WFO — the apparent breakthrough (and why it is NOT real)

Broad universe = 15 pairs (7 USD majors + 5 JPY crosses + EURGBP/EURAUD/GBPAUD). `_disco_work/
arc1021_wfo_broad.py` + `arc1021_robust_check.py`. Scored solely by `MultiPairBacktester`, FundedNext ON.

**Under the partial-runner exit it LOOKED like the corpus's first deployable book:**
- broad me-short: **9/10, mean +1.998%**, 2015 +1.69, 2018 +2.32 (only 2016 −1.18); beats fair null +2.10pp.
- 4-way (gap+me-long+fbr+broad-me-short): **25 convex weightings all-folds-positive**, best max-min
  (gap 0.00 / me-long 0.08 / fbr 0.32 / **me-short 0.60**) worst-fold **+0.069%**, mean +1.81%, 0/10 neg.

**The robustness check disqualified it — two layers of in-sample selection (the Arc-10 failure mode):**

1. **Exit-dependent (fat-tail).** The 9/10 / +1.998% rests on the **partial-runner's fat right tail** (a few
   big runner trades — folds swing 2017 −0.68 narrow → +5.00 broad; the null swings +4.46/−4.52 too). Under
   honest **sl_only** the broad me-short is only **4/10 (te2) / 6/10 (te3) / 7/10 (te5)**, **median −0.21 / +0.75 /
   +1.28**, and **2018 goes NEGATIVE** under te2 (−0.88) & te3 (−1.09). The edge is not a robust per-trade
   reversion — it is the runner exit harvesting occasional continuation, not the month-end reversion thesis.
2. **Weight-overfit.** Even under partial-runner, the all-folds-positive book exists ONLY at the **IS-fold-
   optimized max-min weighting**; the principled **risk-parity** gives **8/10 (worst −0.193%)** and **equal
   7/10 (worst −1.942%)**. The 25 "all-folds-positive" weightings are an IS-optimized region, not a robust book.
3. **2016 robustly negative.** me-short (broad) is negative in 2016 under **every** exit (−1.18 to −1.96) — the
   broad universe does NOT fix 2016. The 4-way is **0 all-folds-positive under every sl_only exit** (risk-parity
   worst −0.31% to −0.85%) — no better than the narrow arc-1020 book (worst −0.11%).

So the apparent "first all-folds-positive book" is the conjunction of (a) the fat-tail partial-runner exit +
(b) hand-optimized IS weights — an internally-consistent IS result resting on in-sample selection, exactly the
Arc-10 class of false positive. **KILL the deployable-book claim.** OOS NEVER touched (the disqualifying check
was all IS — exit + weight variation — so the holdout stays pristine; this is not an OOS failure, it is an
IS-robustness failure caught before the holdout was spent).

## Verdict

**KILL.** The broad-universe extension does not produce a robust deployable book; the all-folds-positive result
is partial-runner fat-tail × IS-weight-overfit. **Components UNCHANGED** — the narrow (7-USD-major) me-short
retains PORTFOLIO (arc 1019); the narrow 4-way (arc 1020, worst −0.11%, blocked 2015/2016) remains the honest
frontier. No new `portfolio-candidates/` entry.

## What is genuinely true (the honest carry-forward)

1. **The month-end-short flow generalizes** (every group beats its random-day control) and the broad universe
   **robustifies me-short's 2015** — under honest sl_only te3/te5 it is +0.92 / +3.33, multi-pair (EUR/GBP
   crosses + USDCAD/USDJPY), **no longer the single GBPUSD trade** that arc 1020/2015 flagged. That part is real.
2. **But it is a trade-off, not a clean win:** the broad universe makes **2018 exit-fragile** (positive only
   under partial-runner / te5, negative under te2/te3) and **does not fix 2016** (negative under every exit).
   So even at the component level, broadening trades 2015-robustness for 2018-fragility — not a strict improvement.
3. **2016 is now the hardest residual fold** — negative in me-long, me-short (narrow AND broad, every exit), and
   gap; positive only in fbr & gap, which 2018 caps. 2016 (Brexit / US-election) is a genuine non-reversion gap.

## The Arc-10 lesson (the high-value output of this arc)

**A near-miss caught by discipline.** A combined book that is all-folds-positive on IS can rest entirely on
in-sample selection — here the fat-tail exit choice AND the weight optimization. The defenses that caught it:
(i) re-score under a **non-fat-tail exit** (sl_only) — a fat-tail edge that vanishes under sl_only is the runner
harvesting variance, not the thesis; (ii) check **non-optimized weights** (equal / risk-parity), not just the
max-min point — an all-folds-positive book that needs hand-tuned weights is overfit; (iii) the binding folds
were razor-thin (+0.07%), which alone should trigger maximal scrutiny. **Re-usable rule:** before promoting any
combined book, require all-folds-positive under (sl_only OR a non-runner exit) AND under risk-parity (not just
the IS-optimal weighting); a result that needs the fat-tail exit and the optimized weights is an Arc-10 false
positive. This would have been a fabricated "first deployable system" had the partial-runner+max-min number
been taken at face value.

## Threads

1. The route's honest state is UNCHANGED from arc 1020: narrow 4-way blocked at 2015 & 2016 (worst −0.11%).
   The broad universe is not a clean fix.
2. **2016 is the precise residual target** (negative in all reversion legs under every exit; non-reversion,
   Brexit/US-election regime). The 5th-component spec sharpens to: **positive in 2016 without dragging 2018**,
   structurally non-reversion. 2015 may be partially addressable by a broader me-short universe under an
   honest exit, but only if 2018/2016 are not sacrificed — not demonstrated here.
3. me-short's exit is load-bearing AND fragile: the runner harvests fat tails (inflates IS), the honest
   sl_only is thinner. A future combination must freeze me-short on an honest (non-runner) exit.

## FLAGS (code not merged)

- **None requiring the canonical core.** No new tool (reused `MonthEndReversionShortSignal`,
  `make_time_exit_predicate`, `build_null_signal_evaluation`, `combine_fold_roi` — all CALLED).
- Carries the standing `A1Config.time_exit_bars`-unwired flag and the gap-reproduction nit (arc 1020;
  immaterial — gap gets 0 weight). Drivers scratch `_disco_work/arc1021_*.py`.
