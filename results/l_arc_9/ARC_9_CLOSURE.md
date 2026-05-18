# Arc 9 - Closure Doc

## Disposition

**STEP_4_KILL** under L_ARC_PROTOCOL v2.3 (base v2.1.2 + v2.2 + v2.3 amendments) §16a.

## Signal

| Field | Value |
|---|---|
| Name | `signal_inside_bar_break_trend_long_v0.1` |
| Source spec | `docs/signal_spec_inside_bar_break_trend_long_v0.1.md` (provided by dispatcher) |
| Family | Trend continuation (compression-and-break) |
| Direction | Long only |
| Signal TF | 4H |
| Pair set | 28 FX (KH-24 set) |
| Data window | 2020-10-01 → 2026-01-31 |
| Pool size | 2,153 trades |

## Step results

| Step | Result | Killer |
|---|---|---|
| 1 — Plumbing | PASS | - |
| 2 — Clustering (K=3, sil 0.4247) | PASS | - |
| 3 — Capturability (cluster_0_individual survives, SL=2.0×ATR) | PASS (1 archetype) | - |
| 4 — Extractability | **FAIL** | Pipeline E RF AUC 0.511 << 0.65; Pipeline D1 reaches AUC 0.626 at t=1 but threshold sweep recall maxes at 0.003 (gate 0.60), per v2.2 §3 no max-F1 fallback |

## §16a HALT vs KILL evaluation

| Criterion | Status |
|---|---|
| Single criterion fail (exactly one gate fails at the killing step) | **FAIL** — Pipeline E AUC AND Pipeline D1 threshold-sweep recall both fail (two failure modes within the cohort) |
| Cohort viability (size_fraction ≥ 0.10) | PASS (0.170) |
| Path A (numeric near-miss, margin < 0.03) | **FAIL** — E AUC margin 0.139; D1 recall margin 0.597. Both far above 0.03. |
| Path B (categorical with strong magnitude, fwd_mfe_h240_p50 ≥ 3.0R) | N/A — failing criteria are numeric, not categorical |

**Disposition: KILL.** No `## Cross-arc calibration candidate` section per §16a KILL convention.

## Closure summary

The signal generates a pool comfortably within the spec's prior (2,153 / target 1,500-2,500) and Step 1 passes all sub-gates with margin. Path-shape clustering at K=3 produces one capturable cluster (cluster_0, n=365, 17% of pool, mono 0.534, ttp_rel 0.771, lp 31.44): forward MFE p50 6.18R, frac_reach_1R 1.000, frac_wrong_way_pre_peak 0.000, final_r_mean +4.423R t=+21.81 at SL=2.0×ATR. Capturability is clean and strong.

Extractability fails. Pipeline E entry-time RF AUC is 0.511 — at chance — across the 16-feature set (8 protocol-base + 8 arc-specific compression/break geometry). The signal-spec's hypothesis that compression geometry features carry entry-time edge is empirically refuted. Pipeline D1 with path-so-far features at bar offset t reaches the AUC floor at every t∈{1..5} (best t=10 at AUC 0.692 but exclusion 32.7% violates ≤30%); the smallest-t selection yields t=1, AUC 0.626. The classifier discriminates, but its probability outputs cluster below 0.40 — the {0.40, 0.50, 0.60, 0.70} threshold sweep produces no threshold with recall ≥ 0.60 (best 0.003 at threshold 0.40). Per v2.2 §3 the archetype fails Step 4 with no max-F1 fallback.

The clean cohort exists but is not surface-able with the features available to the protocol at decision time. The IB-trend signal class is not extractable on Pipeline E or D1 as drawn.

## What is permanently eliminated by this closure

Nothing structural to the signal. The cohort is genuinely clean (cluster_0 capturability is strong); the failure is specifically about feature-based extractability at entry / very-early-bar windows.

Signal NOT permanently eliminated. Could potentially be revisited under:
- Richer feature regime (multi-TF context, session/regime conditioning, cross-asset)
- Probability-calibrated classifiers (Platt scaling, isotonic) that allow the existing AUC 0.626+ discrimination to reach the threshold sweep
- A protocol amendment that relaxes the strict {0.40, 0.50, 0.60, 0.70} grid or removes the recall floor for low-prevalence positive classes

The compression-geometry-features hypothesis (specifically: ib_range_ratio, inside_bar_range_atr, break_bar_body_atr, break_close_above_high_atr) IS empirically refuted as an entry-time predictor for this archetype.

## Cross-arc observation (informational, not a §16a candidate)

This is the second arc (after Arc 7, CLEAN-NULL 2026-05-17) where Pipeline D1 clears AUC ≥ 0.60 mechanically but the strict threshold sweep produces no threshold satisfying recall ≥ 0.60. The failure mode is classifier probability mis-calibration relative to the fixed sweep grid, not classifier non-discrimination. Two occurrences is not a calibration trigger; recording here for future cross-arc review. Not eligible for §16a Path A — margins (0.597 here, similar in Arc 7) are far above the 0.03 near-miss threshold.

## Files

| Artefact | Path |
|---|---|
| Live arc doc | results/l_arc_9/ARC_9_LIVE.md |
| Signal module | signals/lchar_inside_bar_break_trend_long.py |
| Config | configs/wfo_l_arc_9.yaml (sha256 in step1 manifest.json) |
| Step 1 trades | results/l_arc_9/step1_verbatim/trades_all.csv |
| Step 1 paths | results/l_arc_9/step1_verbatim/trades_paths.csv |
| Step 1 prefilter events | results/l_arc_9/step1_verbatim/prefilter_events.csv |
| Step 1 manifest | results/l_arc_9/step1_verbatim/manifest.json |
| Step 1 diagnostics | results/l_arc_9/step1_verbatim/{audit_lookahead.txt, audit_right_edge_swing.txt, schema_check.json, cofire_matrix.{md,json}, pool_summary.md, diagnostics.json} |
| Step 2 path features | results/l_arc_9/step2_clustering/path_features.csv |
| Step 2 silhouette + clusters | results/l_arc_9/step2_clustering/{silhouette_summary.csv, silhouette_K{k}.txt, clusters_K{k}.csv, centroids_K{k}.csv} |
| Step 2 archetype assignments | results/l_arc_9/step2_clustering/archetype_assignments.csv |
| Step 2 summary | results/l_arc_9/step2_clustering/STEP2_SUMMARY.md |
| Step 3 SL sweep (per-trade) | results/l_arc_9/step3_capturability/per_trade_sl_sweep.csv |
| Step 3 cluster + archetype aggregates | results/l_arc_9/step3_capturability/{cluster_sl_sweep.csv, archetype_sl_sweep.csv} |
| Step 3 archetype summaries + pass list + routing | results/l_arc_9/step3_capturability/{archetype_summaries.csv, capturability_pass_list.csv, cluster_routing.csv, cluster_0_distribution.csv} |
| Step 3 summary | results/l_arc_9/step3_capturability/STEP3_SUMMARY.md |
| Step 4 entry features | results/l_arc_9/step4_extractability/entry_features.csv |
| Step 4 predictability (E + D1) | results/l_arc_9/step4_extractability/{predictability_angle_E.csv, predictability_angle_D1.csv} |
| Step 4 threshold sweep (D1 only since E never cleared gate) | results/l_arc_9/step4_extractability/threshold_sweep_D1_cluster_0_individual.csv |
| Step 4 extractability pass list (empty / no survivors) | results/l_arc_9/step4_extractability/extractability_pass_list.csv |
| Step 4 summary | results/l_arc_9/step4_extractability/STEP4_SUMMARY.md |

No classifier artefacts (`*_E_classifier.joblib`, `*_D1_classifier.joblib`, `*_E_filter.yaml`, `*_D1_policy.yaml`) emitted — no pipeline cleared its gates.
