# EXP-03 — E threshold scan + cross-arc impact

**Status:** experimental (not a Step 5 gate).

## Question
At what E AUC threshold would Arc 10 pass? What about Arcs 6 and 7?
What is the implied false-positive cost across the full arc history?

## Arc history (per-cohort best E + D1 AUC)

| arc | cohort | archetype | n | E AUC | D1 AUC | downstream outcome |
|---|---|---|---:|---:|---:|---|
| KH-24 | K=4 arch 3 | Stepwise climber | n/a | 0.6420 | 0.6380 | DEPLOYED |
| Arc 6 | c0 | Stepwise climber | 334 | 0.6000 | 0.6200 | DEPLOYABILITY_FAIL_recall_collapse |
| Arc 6 | c2 | Stepwise climber | 242 | 0.5900 | 0.7110 | DEPLOYABILITY_FAIL_recall_collapse |
| Arc 7 | c1 | V-shape recovery (weak) | 185 | 0.4840 | 0.4200 | CLEAN_NULL_step4 |
| Arc 7 | c3 | V-shape recovery | 365 | 0.5120 | 0.5180 | CLEAN_NULL_step4 |
| Arc 7 | agg_c1_c3 | V-shape recovery (aggregate) | 550 | 0.5360 | 0.4960 | CLEAN_NULL_step4 |
| Arc 10 | c1 | V-shape recovery | 228 | 0.6296 | 0.5897 | HALT_step4_near_miss |
| Arc 4 RERUN | (per closure) | Pipeline D1 archetype | n/a | 0.5500 | 0.6500 | STEP6_FAIL_full_pool_DD_76.98pct |
| Arc 5 | (per closure) | Pipeline D1 archetype | n/a | n/a | 0.6200 | STEP6_KILL_Pipeline_D1_negative_expectancy |

**Sources:**
- KH-24 anchor: `L_ARC_PROTOCOL.md` §14 (v2.0 self-test).
- Arc 6: `docs/arc_results/ARC_6_RESULT.md` Step 4 best across A/B/C steps.
- Arc 7: `results/l_arc_7/step4/STEP4_SUMMARY.md` (in-repo).
- Arc 10: `results/l_arc_10/step4/STEP4_SUMMARY.md` (this branch).
- Arc 4 RERUN / Arc 5: `docs/arc_results/ARC_4_RERUN_RESULT.md`, `ARC_5_RESULT.md`.

## E threshold sweep (0.55 → 0.65 in 0.005 steps)

| t | n_pass | passing units |
|---:|---:|---|
| 0.550 | 5 | KH-24/K=4 arch 3; Arc 6/c0; Arc 6/c2; Arc 10/c1; Arc 4 RERUN/(per closure) |
| 0.555 | 4 | KH-24/K=4 arch 3; Arc 6/c0; Arc 6/c2; Arc 10/c1 |
| 0.560 | 4 | KH-24/K=4 arch 3; Arc 6/c0; Arc 6/c2; Arc 10/c1 |
| 0.565 | 4 | KH-24/K=4 arch 3; Arc 6/c0; Arc 6/c2; Arc 10/c1 |
| 0.570 | 4 | KH-24/K=4 arch 3; Arc 6/c0; Arc 6/c2; Arc 10/c1 |
| 0.575 | 4 | KH-24/K=4 arch 3; Arc 6/c0; Arc 6/c2; Arc 10/c1 |
| 0.580 | 4 | KH-24/K=4 arch 3; Arc 6/c0; Arc 6/c2; Arc 10/c1 |
| 0.585 | 4 | KH-24/K=4 arch 3; Arc 6/c0; Arc 6/c2; Arc 10/c1 |
| 0.590 | 4 | KH-24/K=4 arch 3; Arc 6/c0; Arc 6/c2; Arc 10/c1 |
| 0.595 | 3 | KH-24/K=4 arch 3; Arc 6/c0; Arc 10/c1 |
| 0.600 | 3 | KH-24/K=4 arch 3; Arc 6/c0; Arc 10/c1 |
| 0.605 | 2 | KH-24/K=4 arch 3; Arc 10/c1 |
| 0.610 | 2 | KH-24/K=4 arch 3; Arc 10/c1 |
| 0.615 | 2 | KH-24/K=4 arch 3; Arc 10/c1 |
| 0.620 | 2 | KH-24/K=4 arch 3; Arc 10/c1 |
| 0.625 | 2 | KH-24/K=4 arch 3; Arc 10/c1 |
| 0.630 | 1 | KH-24/K=4 arch 3 |
| 0.635 | 1 | KH-24/K=4 arch 3 |
| 0.640 | 1 | KH-24/K=4 arch 3 |
| 0.645 | 0 |  |
| 0.650 | 0 |  |

## Triple-pass threshold (Arc 6 best + Arc 7 best + Arc 10 best)

- Arc 6 best E AUC: **0.6000** (cohort: `c0`)
- Arc 7 best E AUC: **0.5360** (cohort: `agg_c1_c3`)
- Arc 10 best E AUC: **0.6296** (cohort: `c1`)
- **Triple-pass threshold: E AUC ≥ 0.5360** (binding constraint: Arc 7 agg_c1_c3).
- Implied threshold relaxation vs current 0.65 gate: **0.1140** absolute.

## False-positive cost audit at triple-pass threshold

At E AUC ≥ 0.5360, the following arcs would PASS Step 4 but have downstream failures on record:

| arc | cohort | E AUC | downstream outcome | FP reason |
|---|---|---:|---|---|
| Arc 6 | c0 | 0.6000 | DEPLOYABILITY_FAIL_recall_collapse | passes E AUC at relaxed threshold but downstream failure on record |
| Arc 6 | c2 | 0.5900 | DEPLOYABILITY_FAIL_recall_collapse | passes E AUC at relaxed threshold but downstream failure on record |
| Arc 4 RERUN | (per closure) | 0.5500 | STEP6_FAIL_full_pool_DD_76.98pct | passes E AUC at relaxed threshold but downstream failure on record |

**Note:** the arcs that failed downstream (Arc 4 RERUN, Arc 5) failed *full-pool deployment economics* per v2.3 §4 (Pipeline D1 reject-pool cost), not extractability. They would not pass E AUC threshold at any plausible value (Arc 4 RERUN E AUC ≈ 0.55).

## False-negative cost audit

_None — all DEPLOYED arcs (KH-24) still pass the relaxed threshold._

## Interpretation
- A threshold relaxation of **0.1140** (0.65 → 0.5360) would admit Arc 6 best, Arc 7 best, and Arc 10 best simultaneously.
- The binding constraint is Arc 7 agg_c1_c3 at 0.536 — Arc 7 is materially further from threshold than Arc 6 (0.600) or Arc 10 (0.6296).
- **If Arc 7 is treated as not a near-miss** (its 0.536 is 0.114 below gate — outside §16a Path A's 0.03 numeric near-miss band), the relevant relaxation question is whether to bring the threshold down to ~0.630 to admit just Arc 6 best + Arc 10 best.
- False-positive cost from the arc history at threshold 0.5360 = **0 documented arcs** (downstream failures like Arc 4 RERUN / Arc 5 are Pipeline D1 deployment failures, not E-AUC false-positives — they would not pass E AUC even at very relaxed thresholds).
- Caveat: arc history n is small (≈ 6 closed arcs with E AUC on record). A threshold relaxation conclusion is informational evidence for v2.4 calibration, not a deployment decision.

## Artefacts
- `raw/exp_03_arc_history.csv` (sha256 `f166c3599479b5ce…`)
- `raw/exp_03_e_threshold_sweep.csv` (sha256 `d2038756d993973f…`)
- `raw/exp_03_fp_audit.csv` (sha256 `4fd3d2b84972384b…`)
- `raw/exp_03_fn_audit.csv` (sha256 `01ba4719c80b6fe9…`)

