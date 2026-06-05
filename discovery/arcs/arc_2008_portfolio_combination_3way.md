# arc 2008 — 3-way PORTFOLIO combination WFO (gap-fill 1006 + month-end 1011 + failed-breakdown 1013)

**Chat:** 2000s · **Date:** 2026-06-05 · **Disposition:** KILL (combined 3-way book not deployable;
the three components retain their PORTFOLIO status, unchanged) · **Council:** none (not a PASS survivor,
not an idea-fork/diagnosis — a direct quantitative combination test with a decisive, provable result).

## (a) Log read / synthesis (fresh eyes, honest-era only)
Pulled main; read DISCOVERY_PROTOCOL, full DISCOVERY_LOG (both tiers, through arc 2007 / 1013), LESSONS,
TOOL_REGISTRY. State after 26 arcs / 3 chats:
- **The shallow single-condition directional space is comprehensively closed** (all entry constructions,
  H1/H4/D1, 28 pairs, capture + drift lenses, every exit/SL, even stop-removed 3004; both EDGE and COST
  sides; novel structural mechanisms — gotobi/round-numbers/triangulation/breakout-retest — keep
  dissolving on honest data). The DEEP multi-factor lane is now also tested (arc 2007 KILL loses-to-null;
  arc 1013 PORTFOLIO).
- **Exactly THREE live net-positive long-only components, all PORTFOLIO:** (1006/1009) weekend gap-fill
  long, JPY crosses H4; (1011/1012) month-end reversion long, USD majors D1; **(1013) failed-breakdown
  reclaim (stop-run reversal) long, USD majors H4 — the NEW addition since arc 2006, a structurally
  different mechanism family (structural stop-run reversal vs the two flow-event reversions) and the
  strongest/cleanest directional long in the corpus (IS +1.854%, beats a NEGATIVE null +2.96pp, OOS
  mean-positive).**
- **arc 2006 ran the 2-way (gap-fill + month-end) → KILL:** blocked by a mutually-negative fold (2015,
  both lose) + tail-correlation; verdict = need **≥3 components selected for fold-complementarity**
  (ROI on the book's NEGATIVE folds), not average correlation. The 2-way book bled in **2015/16/18/20**.
- **arc 1013's explicit high-value steer for the 2000s range:** run the **3-way combination WFO**
  (1006 + 1011 + 1013), ranking by ROI on the 2-way book's negative folds — 1013 is positive in
  **2015/16/20** (three of the four bleed years), sharing only 2018, "so a 3-way should cut the
  mutually-negative folds from four to ≈one (2018)."

## (b) Idea + why
This is the route's literal gated next step: arc 2006 proved 2 components are not enough and specced
exactly what a 3rd must do; arc 1013 then DELIVERED a 3rd that is mean-positive, OOS-surviving, of a NEW
mechanism family, and positive in 3 of the 4 folds the 2-way book bled. The single most decision-relevant
unrun experiment is the honest 3-way risk-parity combination, gated all-folds-positive on the combined
book (§6/§11). Decisive either way: a PASS candidate (→ council → `passed/` — the first deployable), or a
rigorous "3 is still not enough, here is the binding fold and the 4th's exact spec."

## (c)/(d)/(e) Method (measurement CALLED, combination is the one BUILT tool)
No new pool/cluster/oracle — all three components are already characterized (1006/1009, 1011/1012, 1013).
The arc is a **validation/combination** step:
1. Reproduce each component LIVE via its **REGISTERED** signal over the SAME canonical IS fold set
   (`build_v3_folds`, `is_days>=365` → 10 expanding folds, OOS years 2011–2020), scored solely by
   `MultiPairBacktester` via `A1Architecture`+`ArcFoldRunner`+`run_config_over_folds`, FundedNext ON,
   SL-first:
   - **A — gap-fill:** `WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36)` on 5 JPY crosses
     (EURJPY,GBPJPY,AUDJPY,CADJPY,CHFJPY), H4, 24-bar `make_time_exit_predicate`, `sl_only`, SL 2·ATR,
     **trail OFF**.
   - **B — month-end:** `MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2)` on 7 USD majors,
     D1, 2-bar `make_time_exit_predicate`, `sl_only`, SL 2·ATR, **trail OFF**.
   - **C — failed-breakdown:** `FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25)`
     on 7 USD majors, H4, `sl_plus_trailing_atr`, SL 2·ATR.
2. Combine per-fold ROI with the BUILT `discovery/tools/combine_fold_roi.py` under **equal** and
   **risk-parity** (inverse-fold-vol) weights, weights FIT ON IS and frozen. Apply all-folds-positive.
3. **Combination-method-invariance check** (Arc-10 discipline — don't rest a KILL on the linear
   approximation): grid the full convex simplex (step 0.01, 5,151 weightings) and ask whether ANY
   weighting makes all 10 folds positive.

**Reproduction check (Arc-10 discipline — don't trust transcribed numbers): all three reproduce EXACTLY
(≤0.005pp per fold)** — gap-fill IS mean **+0.685%**, month-end **+0.232%**, failed-breakdown **+1.854%**,
matching the 1009 / 1011 / 1013 records to the basis point. (A first pass mismatched gap-fill by 3.8pp
per fold at the SAME total trade count 260 — traced to `A1Config.trail_enabled` defaulting True, adding a
trailing stop to the 24-bar-hold gap-fill; the committed config is pure `sl_only`+time-exit, trail OFF.
B/C were unaffected — B's 2-bar hold never reaches trail activation; C's trail comes from its
`sl_plus_trailing_atr` policy. **Methodological catch worth keeping: `sl_only` ≠ no-trail unless you also
set `trail_enabled=False`.**) corr(gap,me)=**+0.117** (matches arc 2006 → fold-alignment re-confirmed),
corr(gap,fbr)=**+0.189**, corr(me,fbr)=**−0.366**.

## (g) Results — combined book (IS, 10 folds, OOS years 2011–2020)

Per-fold ROI %, fold order (ids 2–11 → OOS years 2011–2020):
```
gap-fill (A):  -0.07, +8.23, -2.06, +2.94, -4.19, +3.20, +0.53, -6.79, +7.45, -2.39   mean +0.685%  5/10 neg
month-end(B):  +0.40, +0.29, +0.96, -0.23, -1.14, -0.51, +0.34, +0.90, +1.16, +0.15   mean +0.232%  3/10 neg
failed-bd(C):  +7.55, +3.05, +0.91, +0.19, +3.17, +2.55, +1.23, -4.20, +0.05, +4.03   mean +1.854%  1/10 neg
```

| weighting | weights (gap / me / fbr) | combined per-fold ROI % | mean | neg | all-folds-pos |
|---|---|---|---|---|---|
| equal | 0.333 / 0.333 / 0.333 | +2.63,+3.86,**−0.07**,+0.97,**−0.72**,+1.75,+0.70,**−3.36**,+2.89,+0.60 | +0.924% | 3/10 | **NO** |
| risk-parity | 0.106 / 0.726 / 0.168 | +1.55,+1.60,+0.63,+0.18,**−0.74**,+0.40,+0.51,**−0.77**,+1.64,+0.53 | +0.552% | 2/10 | **NO** |

- **C did exactly what arc 1013 predicted — and it is not enough.** The new component CUT the 2-way
  book's four bleed folds (2015/16/18/20) to **two**: it fixed **2016** (C +2.55) and **2020** (C +4.03)
  outright, but **2015 and 2018 survive**.
- **2018 is the binding fold.** A gap-fill **−6.79**, B month-end +0.90, **C failed-breakdown −4.20**.
  BOTH the flow-reversion (gap-fill) AND the structural stop-run reversal (C) lose; only the small
  month-end is positive. 2018 is a persistent strong-USD trend year — arc 1013 named it as C's single
  weak fold ("breakdowns are real, not swept"). **C shares the 2-way book's 2018 weakness rather than
  fixing it** — exactly the "shares only 2018" arc-1013 flagged, and the year no weighting can rescue.
- **2015 is a near-miss C nearly rescues but can't quite.** A −4.19, B −1.14, **C +3.17**: C's strong
  positive almost covers A+B's −5.33 deficit (equal-weight −0.72%), but under risk-parity C is weighted
  only 0.168 (it is the 2nd-highest-vol component) so its rescue is muted (−0.74%).
- **PROVABLY blocked — combination-method-invariant.** Gridding the full convex simplex (5,151
  weightings): **0 are all-folds-positive.** The best-possible (max-min) weighting is gap 0.00 / me 0.78 /
  fbr 0.22, and it STILL leaves min-fold **−0.222%** (2014/2015/2018 negative). The infeasibility is a
  hard 2015-vs-2018 conflict: 2018 > 0 needs heavy month-end weight (its only positive leg: 0.90 w_B >
  6.79 w_A + 4.20 w_C ⇒ at w_A=0, w_B > 4.67 w_C); 2015 > 0 needs heavy failed-breakdown weight and is
  *hurt* by month-end (3.17 w_C > 4.19 w_A + 1.14 w_B ⇒ at w_A=0, w_B < 2.78 w_C). `4.67 w_C < w_B < 2.78
  w_C` is empty. No weighting — and (P&L being additive) no honest single-engine co-simulation — passes
  both folds. The KILL does not rest on the linear-combination approximation.

**IS is not all-folds-positive → OOS deliberately NOT touched** (§4 + holdout preservation). The 2021+
holdout stays pristine for a future combined book that *does* clear IS. (C's solo OOS was already measured
once in arc 1013; A's in arc 1006; B's was preserved — but the COMBINED-book OOS is not measured here.)

## Verdict: FAIL the sole judge (combined 3-way book) → KILL
The 3-way combined book of the programme's three net-positive components is not all-folds-positive under
any weighting, and is *provably* blocked by a 2015-vs-2018 infeasibility (verified by full simplex grid,
not just the two named weightings). It does not graduate. **The three components are unchanged** — each
reproduced EXACTLY (not re-tested or weakened); they retain PORTFOLIO status. No new
`portfolio-candidates/` entry (it would double-count 1006/1011/1013). The portfolio **thread stays
ACTIVE**: a 4th component is now a hard requirement, with a sharply narrowed spec.

## The 4th-component spec (the actionable deliverable)
To make the book all-folds-positive the 4th component must be net-positive on the surviving negative
folds — **OOS years 2015 and 2018** — and *critically* **2018**, the binding fold where BOTH a
flow-reversion (gap-fill) AND the new structural stop-run-reversal family (failed-breakdown) lose.
Per-fold 3-way deficits to overcome (risk-parity): 2015 −0.74%, 2018 −0.77% (near-equal — both bind).

**The deeper requirement (sharper than arc 2006's): the 4th leg must be positive in a persistent
strong-trend / strong-USD regime, which all three current edges are structurally SHORT.** All three
components are fade/reversion-flavored relative to a real directional trend: the gap-fill fades a weekend
gap, the month-end fades a rebalancing over-extension, and the failed-breakdown fades a breakdown (bets
it was a liquidity sweep, not real). In 2018 — USD trending up, EURUSD/etc. breaking down *for real* —
all three "fade" bets lose together. **Average correlation hid this** (corr gap-fbr +0.189, me-fbr −0.366
look beautifully decorrelated) but the 2018 TAIL is shared by the two large components. The diversifying
4th leg must be a **trend-continuation / momentum** edge positive precisely when trends persist — and the
long-only directional/trend menu that would win 2018 is comprehensively dead (arcs 0–3006). This sharpens
the standing **arc-3004 escalation** with a concrete binding fold: the regime-orthogonal leg the portfolio
route needs is structurally a **short / second-leg / relative-value / trend-continuation** construction,
which is the operator's shorts/second-leg unlock (FLAG-1) — now the single highest-leverage move, with a
named target year (2018) and the arc-2007 climax-sweep SHORT as a concrete candidate (it is the *short*
leg of the very 2018 breakdowns that beat the long failed-breakdown).

## Threads / lessons
1. **Three near-decorrelated mean-positive components — including a NEW mechanism family — are still NOT
   enough** when two of them share one un-rescuable tail fold (2018). The PORTFOLIO route's "≥3
   components" (arc 2006) was necessary but not sufficient; the real requirement is **fold-complementarity
   on the binding fold**, and 2018 remains uncovered.
2. **Average-correlation decorrelation STILL ≠ fold-complementarity — even across mechanism families.**
   corr(gap,fbr)=+0.189 (and me,fbr=−0.366) looked like genuine diversification, but the gap-fill and
   failed-breakdown bleed together in 2018. The arc-2006 tail-correlation lesson generalizes: **all
   long-only "fade/reversion" edges are implicitly short-trend and tail-correlate in a strong-trend year**,
   regardless of their surface mechanism. The 4th must be long-trend, not another fade.
3. **The simplex-feasibility grid is the rigorous combination-invariant test** (extends arc 2006's
   single-mutually-negative-fold pre-screen): when no single fold is mutually-negative but the book still
   fails, grid the convex hull — if 0/N weightings pass and the max-min is ≤ 0, the KILL is proven without
   touching OOS or trusting the linear approximation. Re-usable for every future combination arc.
4. **`sl_only` is not trail-free** — `A1Config.trail_enabled` defaults True and silently added a trailing
   stop to the 24-bar gap-fill (3.8pp/fold error at identical entries). Always set `trail_enabled=False`
   for a pure `sl_only`+time-exit reproduction. (Not a code flag — a config-usage note; B/C unaffected.)
5. **Reinforces + sharpens the arc-3004 escalation:** the portfolio route is now blocked on a single,
   named, regime fold (2018) that requires a long-trend / short / second-leg leg the long-only apparatus
   cannot express → shorts/second-leg unlock (FLAG-1) is the highest-leverage operator move, and arc
   2007's climax-sweep SHORT is the concrete first target (it is the short side of the 2018 breakdowns).

## Tooling
No new tool. Reused the BUILT `combine_fold_roi` (arc 2006) + the three REGISTERED signals
(`WeekendGapFillLongSignal`, `MonthEndReversionLongSignal`, `FailedBreakdownReclaimLongSignal`) + the
BUILT `make_time_exit_predicate`. The simplex-feasibility grid is a one-off arithmetic check over the
reproduced per-fold vectors (no signal/engine), kept in scratch.

## FLAGS (code not merged)
None new. Carries FLAG-1 (long-only blocks the trend-continuation / short / second-leg regime-orthogonal
leg the portfolio route now demonstrably needs for fold 2018 — operator/human-gated) + the standing
`A1Config.time_exit_bars`-unwired flag (worked around via the BUILT `make_time_exit_predicate`). Drivers
scratch `_disco2000_work/arc2008_combo3.py` (reproducible: `PYTHONPATH=. py _disco2000_work/arc2008_combo3.py`,
`histdata_root=C:\Users\panap\histdata_backup`), `arc2008_feasibility.py` (simplex grid).
