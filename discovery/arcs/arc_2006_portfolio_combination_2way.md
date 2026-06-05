# arc 2006 — 2-way PORTFOLIO combination WFO (gap-fill 1006 + month-end 1011)

**Chat:** 2000s · **Date:** 2026-06-05 · **Disposition:** KILL (combined 2-way book not deployable;
the two components retain their PORTFOLIO status, unchanged) · **Council:** none (not a PASS survivor,
not an idea-fork/diagnosis — a direct quantitative combination test with a decisive arithmetic result).

## (a) Log read / synthesis (fresh eyes, honest-era only)
Pulled main; read DISCOVERY_PROTOCOL, full DISCOVERY_LOG (both tiers, through arc 2005 / 1011),
LESSONS, TOOL_REGISTRY. State after 23 arcs / 3 chats:
- **Directional prediction is comprehensively closed** (all entry constructions, H1/H4/D1, 28 pairs,
  capture + drift lenses, all exits/SL, even stop-removed 3004). Cost side of EDGE<COST also closed
  (3007 spread-timing backfires; 2005 trailing-cheapness non-durable OOS). Stop is a real ~5pp drag,
  not the wall. Novel structural mechanisms keep dissolving on honest data (gotobi, round-numbers,
  triangulation, breakout-retest).
- **Exactly TWO live net-positive long-only components, both PORTFOLIO:** (1006/1009) weekend gap-fill
  long on JPY crosses; (1011) month-end reversion long on USD majors. Both are *discrete
  non-informational flow-event over-extension → reversion* edges. corr ≈ +0.117.
- **The explicit open thread (arc 1011):** "the combined-book all-folds-positive WFO (co-simulated,
  risk-weighted) is the gated next step, NOT claimed." Nobody had actually RUN the honest co-simulated
  risk-parity combination — the "naive 2-way add is 6/10" was a back-of-envelope fold-ROI add.

## (b) Idea + why
The PORTFOLIO route's whole premise is: combine decorrelated mean-positive components into an
all-folds-positive book (§6/§11 — the combined book is its own all-folds-positive gate). We have
exactly the 2 components the route was built on. The single most decision-relevant unrun experiment is
the honest 2-way combination with the **risk-parity** lever arc 1011 flagged (the gap-fill's ±8% fold
swings dominate the month-end's ±1%; equal capital weight lets gap-fill variance run the book). Outcome
is decisive either way: a PASS candidate (→ council → `passed/`), or a rigorous "2 is not enough → 3rd
needed, here is its exact spec." This is the literal next step the route has been building toward, not a
new fishing expedition on dead ground.

## (c)/(d)/(e) Method (measurement CALLED, combination is the one BUILT tool)
No new pool/cluster/oracle needed — both components are already characterized (1006/1009/1011). The arc
is a **validation/combination** step:
1. Reproduce each component via its **REGISTERED** signal over the SAME canonical IS fold set
   (`build_v3_folds`, `is_days>=365` → 10 expanding folds, OOS years 2011–2020):
   - **A — gap-fill:** `WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36)` on 5 JPY crosses
     (EURJPY,GBPJPY,AUDJPY,CADJPY,CHFJPY), H4, 24-bar `make_time_exit_predicate`, `sl_only`, SL 2·ATR.
   - **B — month-end:** `MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2)` on 7 USD majors,
     D1, 2-bar `make_time_exit_predicate`, `sl_only`, SL 2·ATR.
   Both scored solely by `MultiPairBacktester` via `A1Architecture`+`ArcFoldRunner`+`run_config_over_folds`,
   FundedNext costs ON, SL-first.
2. Combine the per-fold ROI with the BUILT `discovery/tools/combine_fold_roi.py` under **equal** and
   **risk-parity** (inverse-fold-vol) weights; weights FIT ON IS, frozen. Apply all-folds-positive to
   the combined book.

**Reproduction check (Arc-10 discipline — don't trust transcribed numbers):** both components reproduce
EXACTLY via the registered tools — gap-fill IS mean **+0.685%**, month-end **+0.232%**, and
corr(A,B) = **+0.117** — matching the arc 1009 / 1011 records to the basis point and confirming the two
per-fold series are fold-aligned (same `build_v3_folds` date windows; fold *j* is the same calendar year
for an H4 and a D1 signal).

## (g) Results — combined book (IS, 10 folds, OOS years 2011–2020)

Per-fold ROI %, fold order (ids 2–11 → OOS years 2011–2020):
```
gap-fill (A):  -0.07, +8.23, -2.06, +2.94, -4.19, +3.20, +0.53, -6.79, +7.45, -2.39   mean +0.685%  5/10 neg
month-end(B):  +0.40, +0.29, +0.96, -0.23, -1.14, -0.51, +0.34, +0.90, +1.16, +0.15   mean +0.232%  3/10 neg
```

| weighting | weights (gap / me) | combined per-fold ROI % | mean | neg | all-folds-pos |
|---|---|---|---|---|---|
| equal | 0.500 / 0.500 | +0.16,+4.26,−0.55,+1.36,−2.66,+1.35,+0.43,−2.94,+4.30,−1.12 | +0.459% | 4/10 | **NO** |
| risk-parity | 0.128 / 0.872 | +0.34,+1.31,+0.57,+0.17,**−1.53**,−0.03,+0.36,−0.08,+1.96,−0.17 | +0.290% | 4/10 | **NO** |

- **Risk-parity helps variance, not the gate.** It down-weights the high-vol gap-fill 0.500→0.128, so
  the worst fold improves materially (equal −2.94% → risk-parity −1.53%; gap-fill *solo* worst −6.79%) —
  diversification + risk-weighting does exactly what it should to *variance* — but it is **still 4/10
  negative**.
- **The blocker is a mutually-negative fold.** Fold 6 (**OOS year 2015**) is negative for BOTH
  components (gap-fill −4.19%, month-end −1.14%). **No convex combination — and no honest single-engine
  co-simulation — can make 2015 positive:** book P&L is additive, so two books that both lose money over
  the same window sum to a loss over that window regardless of weighting. The mutually-negative-fold
  argument is **combination-method-invariant** (it does not depend on the linear-combination
  approximation; see the LIMITATION note below).
- **The negatives share a regime, not just a fold.** Under risk-parity the 2-way book is negative in
  **2015, 2016, 2018, 2020** — the choppy / risk-event years (2015 = CHF de-peg + risk-off). Both
  components are *reversion / flow* edges, so they fail in the *same* tail regime. corr +0.117 says
  "decorrelated on average," but they are **tail-correlated** in exactly the years a reversion book
  bleeds. Average-correlation decorrelation is necessary but **not sufficient** for fold-complementarity.

**IS is not all-folds-positive → OOS deliberately NOT touched** (§4 + holdout-preservation; no
IS-clearing book, so the 2021+ holdout stays pristine for a future combined book that *does* clear IS).

## Verdict: FAIL the sole judge (combined 2-way book) → KILL
The 2-way combined book of the programme's only two net-positive components is not all-folds-positive
under any weighting, and is *provably* blocked by ≥1 mutually-negative fold. It is therefore not a
deployable book and does not graduate. **The two components are unchanged** — they retain their
PORTFOLIO status (this arc did not re-test or weaken them; it reproduced them exactly). No new
`portfolio-candidates/` entry is created (no new decorrelated component was discovered — that would
double-count 1006/1011). The portfolio **thread stays ACTIVE**: a 3rd component is now a hard
requirement, with a precise spec.

## The 3rd-component spec (the actionable deliverable)
To make a 3-way risk-parity book all-folds-positive, the 3rd component must be net-positive on the folds
where the 2-way book is negative — **OOS years 2015, 2016, 2018, 2020** — and *critically* on **2015**
(fold 6), the year BOTH existing edges lose. Per-fold 2-way deficits to overcome (risk-parity):
2015 −1.53%, 2016 −0.03%, 2018 −0.08%, 2020 −0.17%. The binding one is 2015 by a wide margin.

**The deeper requirement (the real lesson): regime-orthogonality, not just low average correlation.**
Both live edges are flow-reversion and fail together in choppy/risk-off years. A 3rd *reversion-flavored*
flow edge (the natural next hunt under the winning template) would very likely **inherit the same 2015-type
tail** and not fix fold 6. The 3rd component should be selected for *complementary regime exposure* —
something that is positive precisely when reversion bleeds (a trend/vol-expansion harvest in risk-off
years) — but the directional/trend menu is comprehensively dead long-only (arcs 0–3006). This sharpens
the standing **arc-3004 escalation**: the regime-orthogonal leg the portfolio route needs is structurally
a *short / second-leg / relative-value* construction (positive in the risk-off regime via the short side),
which is operator-gated (shorts PR #273, FLAG-1). Pre-shorts, the only long-only candidates for the 3rd
slot are flow events with a *different regime signature* than weekend-gap / month-end reversion.

## Threads / lessons
1. **Two near-zero-correlated mean-positive components are NOT enough for an all-folds-positive book** if
   they share even one mutually-negative fold (here 2015) — and these two do. The PORTFOLIO route needs
   **≥3** components, *selected for fold-complementarity*, not merely low average correlation.
2. **Average-correlation decorrelation ≠ fold-complementarity (tail-correlation is the real test).**
   corr +0.117 looked great, but both edges are reversion and bleed in the *same* risk-off years
   (2015/16/18/20). A portfolio of same-mechanism-family edges is tail-correlated; the diversifying leg
   must be a *different mechanism family / regime*. **Re-usable selection criterion for any future
   combination arc: rank a candidate 3rd by its ROI on the existing book's NEGATIVE folds, not by its
   standalone mean or its average correlation.**
3. **Risk-parity is the right weighting (it cut worst-fold −6.79%→−1.53%) but cannot manufacture a gate
   pass** — it is a variance lever, not an edge lever. The all-folds-positive gate is robust to
   re-weighting; only a genuinely complementary component moves it.
4. **The mutually-negative-fold test is a cheap, combination-method-invariant pre-screen** for ANY
   portfolio-combination arc: if the components share a fold where all are negative, stop — no weighting
   or co-simulation can pass the gate; you need a component positive on that fold first.
5. Reinforces the **arc-3004 escalation**: the regime-orthogonal leg the long-only portfolio route needs
   to clear the gate is most naturally a short/second-leg edge (positive in risk-off) → shorts/second-leg
   unlock (FLAG-1) stays the highest-leverage operator move.

## Tooling
Built + registered `discovery/tools/combine_fold_roi.py` (`fit_weights`, `combine_fold_rois`,
`rois_from_fold_stats`, `CombinedBook`) — EXPERIMENT tool: linear (equal / inverse-vol risk-parity)
combination of canonical per-fold `FoldStats` ROI into a combined-book series for the all-folds-positive
judge. Never realizes P&L (the engine did, upstream). Reusable by every future portfolio-combination arc.

**LIMITATION (documented in the tool, not hidden):** this is a per-fold *linear* combination of two
independently-simulated books, not a single co-simulated equity curve. It is a faithful first-order
risk-parity book here because the components trade **disjoint universes** (JPY crosses vs USD majors) and
**disjoint event timing** (weekend gaps vs month-end last day) → simultaneous-position margin/daily-DD
interaction is negligible. The KILL verdict does not rest on this approximation: the mutually-negative
fold (2015) blocks any honest combination, linear or co-simulated, because P&L is additive.

## FLAGS (code not merged)
None new. Carries FLAG-1 (long-only blocks the UP-gap short side / the regime-orthogonal short leg the
portfolio route needs — operator/human-gated) + the standing `A1Config.time_exit_bars`-unwired flag
(worked around via the BUILT `make_time_exit_predicate`). Driver scratch
`_disco2000_work/arc2006_combo.py` (reproducible: `PYTHONPATH=. py _disco2000_work/arc2006_combo.py`,
`histdata_root=C:\Users\panap\histdata_backup`).
