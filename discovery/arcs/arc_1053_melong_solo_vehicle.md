# arc 1053 — Deployment-vehicle feasibility of me_long-SOLO (the actual honest deploy object)

**Chat:** 1000s · **Date:** 2026-06-06 · **Disposition:** DIAGNOSTIC → KILL (no new component) · OOS-preserving (geometry of already-scored curves; nothing tuned) · no null/council · components UNCHANGED.

Full driver: [`../_disco1_work/arc1053_melong_solo_vehicle.py`](../_disco1_work/arc1053_melong_solo_vehicle.py). BUILT tools only (canonical `cosim_book_fold` for the contiguous single-component curve + BUILT `equity_risk_profile.compute_risk_profile` + `propfirm_feasibility`); NO new canonical code, no engine reimplementation.

## Why this arc (the named gap)
arc 1046 established that under honest frozen §5f exits, the 4-component book collapses to **me_long-SOLO out-of-sample** — fbr goes mean-negative OOS under its own frozen exit, so adding it (or gap/me_short) introduces negative folds me_long alone does not have. me_long (committed = honest exit per 1042/2042: `sl_only` / 2-bar D1 / SL2.0, USD majors) is the only leg positive under BOTH its IS-robust exit AND OOS (+0.33–0.49%/yr, Sharpe ~0.5, 5/6 OOS years positive). **It is the real honest deploy object.** Yet every deployment-vehicle profile — arc 2033 (prop-firm/Sharpe), arc 1033 (Calmar/underwater), arc 2045 (honest 2-way geometry) — was computed on the 4-way or 2-way BOOK, **never on me_long-solo.** The operator's path-A deploy call is about *this exact object*; its standalone vehicle number has never existed. This fills that gap, nothing more.

## Method
me_long scored per-year (canonical A1 + `MultiPairBacktester`) over IS folds 2011-2020 + `build_oos_year_folds(2021)` 2021-2026; trades re-id'd per fold and concatenated into ONE contiguous curve via canonical `cosim_book_fold` (single component, weight 1.0, exposure cap ON). Risk geometry + Sharpe + prop-firm `T_min` read off the IS / OOS / full-contiguous windows. OOS measured ONCE as characterization (arc-1042/1046 precedent — 1046 already measured it; nothing tuned, holdout not spent §4).

**Anchor:** IS per-year ROI reproduces the committed arc-1020 me_long record EXACTLY (max dev **0.004pp**); IS mean +0.232% (7/10), OOS mean +0.487% (5/6, only 2021 −0.63 neg) — matches 1046.

## Result — the honest solo object is vehicle-WORSE than the book it was preferred over
| window | ann ret | maxDD | **Calmar** | underwater | Sharpe |
|---|---|---|---|---|---|
| IS 2011-2020 | +0.191% | 3.00% | **0.064** | 1951d (99%) | 0.175 |
| OOS 2021-2026 | +0.532% | 1.73% | 0.308 | 700d (97%) | 0.387 |
| FULL 2011-2026 | +0.319% | 3.00% | **0.106** | 1951d (99%) | 0.266 |

Prop-firm-challenge `T_min` (years to pass at the DD limit by linear leverage, Calmar-bound): **4.7–26 yr** across the 5 representative configs (full-contiguous Calmar 0.11; 7.5yr on the most lenient 8%/10% config) — versus the weeks-to-months a challenge expects. Decisively vehicle-INFEASIBLE.

**The key finding:** me_long-solo's IS Calmar **0.064** (full 0.106) is *materially WORSE* than the 4-way book's ~0.245 (arc 1033) and the honest 2-way's 0.245 (arc 2045). Pruning all the way to the single OOS-robust leg **removes the drawdown-smoothing diversification** the multi-leg book had — me_long alone is the lumpiest single curve (99% of the IS decade underwater, deepest single curve in the corpus). So:
- arc 2045: dropping the 2 net-subtracting legs (4→2) **improves Sharpe/significance** (removes mean-noise) but leaves Calmar ~unchanged (~0.245).
- arc 1053: dropping to solo (2→1) **degrades Calmar** to 0.06–0.11 — the diversification (even from the drag legs) was buying drawdown-duration smoothing at the cost of mean/Sharpe.

## Verdict + diagnosis
**DIAGNOSTIC → KILL** (no new component; me_long-solo's standalone vehicle profile computed). The honest deploy object is vehicle-infeasible on BOTH framings: as a per-fold mean-gate book it fails AFP (2018/2015 negative) and is vehicle-marginal (Calmar ~0.245); as the OOS-robust me_long-SOLO it is mean-positive both IS+OOS but vehicle-WORSE (Calmar 0.06–0.11, T_min 4.7–26yr). **There is no "clean solo" escape** — the leg that survives honest exits + OOS is precisely the one with the least drawdown smoothing.

**NEW lesson — a thin reversion book has THREE corners, none deployable, and pruning trades between them:** (1) the **4-way** maximizes diversification-smoothing (Calmar ~0.245) but the drag legs kill mean/Sharpe and OOS; (2) the **2-way {me_long+fbr}** maximizes Sharpe/significance (drops mean-noise) but fbr dies OOS and Calmar is unchanged; (3) **me_long-solo** is the only IS+OOS-robust mean (the deploy object per 1046) but has the WORST Calmar (no diversification left). The mean/Sharpe-vs-Calmar split (arc 2033/2045) is a genuine 3-way frontier — pruning to the robust leg moves you OFF the diversification corner. Confirms arc 1024's risk-invariance closure from the other side: leverage cannot rescue any corner (Calmar fixed under scaling), and neither can leg-selection — the corners trade quality axes against each other but every corner is vehicle-infeasible. **The operator's path-A call now has the complete object map: there is no leg-subset of the corpus that is BOTH AFP/OOS-robust AND vehicle-feasible.**

Operative frontier unchanged: lever = operator path-A gate-governance call (1032); honest deploy object = me_long-solo (1046, now vehicle-profiled); deployable-system count = 0. Driver committed at `discovery/_disco1_work/arc1053_melong_solo_vehicle.py` (BUILT tools only; single-use diagnostic, no new BUILT registry entry).
