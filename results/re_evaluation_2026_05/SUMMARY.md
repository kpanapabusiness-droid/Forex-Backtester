# Amendment 3 Re-evaluation Summary

> **Generated:** 2026-05-22
> **Dispatch:** Re-evaluate pre-Amendment-3 arcs (8, 10, 11) under the risk-normalised §3 gates.
> **Scope:** analysis + closure-doc + tracker update only. No engine re-runs.
> **L_PROTOCOL version at re-evaluation:** v3.0 + Amendment 3 (locked 2026-05-22, commit `8a9723e`).
> **Template version:** v1.0 (closures pre-date v1.1; v1.1 renamed-field equivalence applied — see `docs/templates/ARC_CLOSURE_TEMPLATE.md` §"Schema versioning").

---

## Re-evaluation table

| Arc | Original verdict | Re-evaluated verdict | Status | Key reason |
|---|---|---|---|---|
| l_arc_8 | FAIL | FAIL | definitive | Worst-fold ROI/DD ratio = 0.88 < 2.0 (invariant under linear scaling); separately, `r_safe = 2.018% > 2.0%` ceiling — but chat directive sets ratio as primary failure mode. |
| l_arc_10 | PASS-VIABLE | **PASS-DEPLOYABLE-PROVISIONAL** | provisional (missing chained DD + per-day series; Step 6 PASS already on same A1 winner — carries forward) | Worst-fold DD 9.22% at `r_base` scales DOWN to 8% at `r_safe = 0.434%`; ratio invariant 2.87 ≥ 2.0; ROI scales to +22.98%; holdout proxy clears. Upgrade is structurally sound conditional on chained DD ≤ 11.52% at `r_base` (linearly scales to ≤ 10% at `r_safe`). |
| l_arc_11 | FAIL | FAIL | definitive (six+ independent failures; missing data doesn't affect verdict) | `r_safe = 0.104%` below 0.15% floor → `step5_not_scalable` (per Amendment 3 §3 priority order). Also: ratio −0.63 < 2.0; ROI sign-negative at any scaling; 4 negative folds (DEPLOYABLE requires 0, VIABLE allows ≤1); min trades/fold 15 < 25 floor. |

---

## Scaling derivation (single table for cross-arc comparison)

| Arc | `worst_fold_dd_base` | `worst_fold_roi_base` | `k_safe` | `r_safe` | `k_hard` | `r_hard` | `ratio_§3` | `scalable_to_safe` | `scalable_to_hard` |
|---|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|
| l_arc_8  |  1.9826% |  +1.7486% | 4.0351 | **2.0176%** | 5.0439 | 2.5219% | **0.882** | ✗ (above ceiling) | ✗ |
| l_arc_10 |  9.2200% | +26.4900% | 0.8677 |   0.4339% | 1.0846 | 0.5423% | **2.873** | ✓ | ✓ |
| l_arc_11 | 38.3600% | −24.0273% | 0.2085 | **0.1043%** | 0.2607 | 0.1303% | **−0.626** | ✗ (below floor) | ✗ |

> `ratio_§3` = `worst_fold_roi_base / worst_fold_dd_base` per L_PROTOCOL §3 amended definition. Differs from each closure's reported `worst_fold_ratio` (engine convention) for all three arcs — see "Cross-arc observations" below.

---

## Cross-arc observations

### Verdict flips and what drove them

- **Arc 10: PASS-VIABLE → PASS-DEPLOYABLE-PROVISIONAL** is the only verdict change. Mechanism: Amendment 3's risk-normalisation lets a strategy with worst-fold DD above 8% (the prior DEPLOYABLE bound) qualify for DEPLOYABLE by *scaling down risk-per-trade* so the worst-fold DD lands at exactly 8%. The ratio is invariant under linear scaling and was already comfortably above 2.0; ROI scales proportionally and remains positive. Step 6 was already PASS on the same A1 Top-1 winner — carries forward without re-run.

- **Arc 8 and Arc 11: FAIL stands** but via different binding constraints than the original closures named:
  - Arc 8 original: `entry_feature_auc_ceiling` (Step 4 diagnostic). Amendment 3 binding: ratio 0.88 < 2.0 (per chat directive); secondarily `r_safe = 2.018%` exceeds `r_max = 2.0%` ceiling.
  - Arc 11 original: `step5_dd_above_gate` (now deprecated in Amendment 3 taxonomy, retained for historical). Amendment 3 binding: `step5_not_scalable` (`r_safe = 0.104%` below 0.15% floor) per priority order.

### Binding constraints across arcs

- **Scalability bounds bite on both sides of the DD distribution.** Arc 8 hits the *ceiling* (`r_safe > 2.0%`) because worst-fold DD 1.98% is too small to fill the 8% allowance at allowed leverage. Arc 11 hits the *floor* (`r_safe < 0.15%`) because worst-fold DD 38.36% is too large to compress to 8% within allowed per-trade risk. Both are first documented instances of their respective scalability failure modes. The Amendment 3 §3 scalability bounds are doing real work — they distinguish "operationally deployable" from "mathematically positive-expectancy" in a way the prior protocol's DD ≤ 8% bound did not.
- **Ratio invariance is decisive when ROI scales toward zero.** Arc 8's ratio 0.88 fails at base, fails at `r_safe`, fails at any positive scaling — the strategy simply doesn't have a 2:1 ROI/DD pre-leverage. Linear scaling can only convert one currency into another (more ROI for more DD, or less for less); it can't change the ratio.
- **Step 6 audit carries forward across the amendment.** Arc 10's Step 6 PASS was scoped to the A1 Top-1 winner. Since Amendment 3 only changes Step 5 gates and the winning architecture / config doesn't change, the audit carries forward without re-run. This pattern will hold for any future PASS-VIABLE → PASS-DEPLOYABLE upgrade where the same candidate is the winner under both gate sets.

### Missing-data concentration

| Constraint | l_arc_8 | l_arc_10 | l_arc_11 | Notes |
|---|:-:|:-:|:-:|---|
| `chained_max_dd_base_pct` | MISSING | MISSING | MISSING | Universally missing — none of the pre-Amendment-3 v3.0 closures measured chained max DD. Engine artefact `step_5/per_day_max_dd_base.parquet` per §"Daily DD measurement" was an Amendment-3 introduction. |
| Per-day max-DD series | MISSING | MISSING | MISSING | Same reason. |
| Per-fold daily-breach count at `r_base` | PRESENT (all 0) | MISSING (no per_fold_metrics.csv emitted for Arc 10) | PRESENT (non-zero on folds 6/9/10/11) | Arc 10's gap is the only "missing per-fold breakdown" case — engine emitted aggregated wfo_results.csv only. |
| Per-fold n_trades | PRESENT | MISSING (per-fold breakdown not separately saved; total only) | PRESENT | |
| Step 6 audit | not run | PASS | not run | Step 6 lazy — only fires on PASS-VIABLE/DEPLOYABLE Top-1. Arcs 8 and 11 didn't produce one. |

Missing data does NOT affect Arc 8 or Arc 11 verdicts (multiple definitive failures are independent of the gaps). It DOES affect Arc 10's verdict definitiveness — hence the `-PROVISIONAL` suffix.

### Engine vs §3 ratio convention divergence (all three arcs)

Every closure's `§1 tracker_payload.best_architecture.worst_fold_ratio` differs from the §3 mathematical definition `worst_fold_roi_base / worst_fold_dd_base`. Inspection of `step_5/per_fold_metrics.csv` for Arc 8 config 30 (the winner) shows the engine's `worst_fold_ratio: 1.7487` is the per-fold `roi_dd_ratio` of fold 9 (worst-ROI fold — its OWN ROI/DD), NOT the cross-fold `worst_roi / worst_dd`. The convention difference is consistent across arcs:

| Arc | Engine `worst_fold_ratio` | §3 math (`worst_roi/worst_dd`) | Gate (≥ 2.0) outcome |
|---|---:|---:|---|
| l_arc_8  |  1.749 |  0.882 | FAIL either way |
| l_arc_10 |  5.419 |  2.873 | PASS either way |
| l_arc_11 | −0.769 | −0.626 | FAIL either way |

Same FAIL/PASS outcome on all three under both readings — but the *numerical margins* differ enough that any future amendment specifying tighter ratio thresholds (or any reporting framework comparing arcs by `worst_fold_ratio`) needs to lock the convention. Recommended: a v3.1 amendment that pins the ratio definition to the §3 math, OR explicitly documents the engine's "ratio inside the worst-ROI fold" convention. Tagged in all three closure §10 blocks as `engine_ratio_vs_amendment_ratio_divergence`.

---

## Recommendations

### Immediate (this dispatch unlocks)

- **Arc 10 verdict effectively upgraded to DEPLOYABLE-PROVISIONAL.** Deployment readiness now hinges on the two PROVISIONAL constraints (chained max DD + per-day breaches). A single engine re-run of the A1 winning config (`sl_3.5x_partial_close_1r_runner_trail_unlimited`) with continuous-equity + per-day max-DD emission would upgrade to DEPLOYABLE-DEFINITIVE. Scope: ~12 fold-runs (11 IS + 1 holdout). Low compute. Recommended as next-up.

### Engine work surfaced

- **Emit `step_5/per_day_max_dd_base.parquet` for all future Step 5 runs** per Amendment 3 §"Daily DD measurement" spec. Columns: `date`, `pair_set`, `day_max_dd_base_pct`, `n_trades_open_start_of_day`. Without this, every future PASS-VIABLE/DEPLOYABLE candidate will land as `-PROVISIONAL` on constraint #6.
- **Emit `chained_max_dd_base_pct` in WFO results CSV.** Continuous-equity tracking across folds. Required to evaluate constraint #7 definitively.
- **Lock the `worst_fold_ratio` reporting convention.** Either rename the engine's column to `worst_roi_fold_ratio` (to make the convention explicit) or change the computation to match §3's `worst_fold_roi / worst_fold_dd`. Either fix would make the tracker's `Worst-fold ratio` column directly comparable to the §3 gate.

### Protocol work surfaced

- **A5 follow-up flag (already deferred in Amendment 3 §3).** Arc 8 surfaces a concrete A5 candidate: low-DD (worst 1.98%), 11/11 sign-consistent, 0 daily breaches, KH-24-orthogonal (co-fire 0.0%). The A5 portfolio-DD combination spec for VIABLE-tier components is the blocker. Recommended: open the A5 spec amendment once a second portfolio-eligible candidate emerges (or once chat decides Arc 8 alone justifies opening the spec).

### Tracker work surfaced (out-of-scope for this dispatch)

- **Backfill ARC_TRACKER.md sections for arcs 8 and 10.** Under this dispatch only the "Closed arcs summary" table received rows for Arc 8 and Arc 10 (needed to populate the new `re_evaluated_verdict` column). The other tracker sections — per-feature contribution, per-architecture win rate, per-archetype recurrence, per-failure-mode count, cross-arc cluster registry, cost-decomposition registry, cross-arc tag registry — still reflect only Arc 11. A separate dispatch should run the full `docs/templates/ARC_CLOSURE_TEMPLATE.md` §4 mapping (steps A through K) for Arc 8 and Arc 10's closure docs.
- **Decide failure-mode counting convention under re-evaluation.** Should `per_failure_mode_count` reflect the original `primary_failure_mode` (audit trail), the re-evaluated one (current methodology), or both? Currently Arc 11's row shows `step5_dd_above_gate: 1` (original) — which is deprecated in Amendment 3 taxonomy. The amended primary is `step5_not_scalable`. Same question latently applies to Arc 8 (`entry_feature_auc_ceiling` → `step5_ratio_below_gate_after_scaling`).

---

## Per-arc closure pointers

- [results/l_arc_8/ARC_CLOSURE.md](../l_arc_8/ARC_CLOSURE.md) — §10 added.
- [results/l_arc_10/ARC_CLOSURE.md](../l_arc_10/ARC_CLOSURE.md) — §10 added.
- [results/l_arc_11/ARC_CLOSURE.md](../l_arc_11/ARC_CLOSURE.md) — §10 added.
- [results/re_evaluation_2026_05/re_evaluation_intent.md](re_evaluation_intent.md) — pre-edit intent doc, captures the read-first findings and the five chat-question answers that drove the §10 blocks.
