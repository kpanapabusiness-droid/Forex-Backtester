# Arc 9 Candidate A — Audit Status Summary

> Formalises the lookahead audit outcome at commit `9dc4f8a` on `claude/bold-brattain-d79817`.
> Resolves spec pre-deployment blocker #1 (lookahead verification audit).

## Resolution statement

**All 8 lookahead audits GREEN at commit `9dc4f8a`. Spec's pre-deployment blocker #1 (audit) is RESOLVED.**

Source of truth: `results/l_arc_9/experiments/lookahead_audit/audit_summary.json` — `overall_verdict = "GREEN"`, `rng_seed = 4242`.

## Per-audit verdicts

| # | Audit | Verdict | Artefact | What was checked |
|---|---|---|---|---|
| 1 | 4H feature timestamps | GREEN | `results/l_arc_9/experiments/lookahead_audit/audit_1_4h_feature_timestamps.csv` | Truncated-series recompute of 4H entry features matches recorded feature values for 3920/3920 (feature, sample) pairs across 280 samples. Zero mismatches → no forward bars contributed. |
| 2 | D1 lag | GREEN | `audit_2_boundary_cases.csv`, `audit_2_d1_lag_samples.csv`, `audit_2_d1_lag_code_review.md` | One-day backward lag verified across 560 samples; min days lag overall = 1; Monday boundary min lag = 3; early-hours boundary min lag = 1. `n_leak = 0`, `hard_stop_triggered = false`. Reference engine (`scripts/phase_kgl_v2_4h_wfo.py:_precompute_d1_exit_arrays`) and Arc-9 D1-lag pattern both pass parity. |
| 3 | Session features | GREEN | `audit_3_session_features.csv` | 80/80 (feature, sample) pairs match across 20 samples for `session_london`, `session_ny_overlap`, `hour_sin`, `hour_cos`. |
| 4 | Label leak | GREEN | `audit_4_correlation_matrix.csv`, `audit_4_feature_list_review.md` | 28 features reviewed; 0 forward-geometry features in entry-time feature set; 0 columns overlap with path-shape features; max absolute pairwise correlation 0.328 (`d1_bars_since_swing_low` × `local_peaks_count` — well below the 0.85 high-correlation flag). |
| 5 | Fold disjointness | GREEN | `audit_5_fold_overlap_table.csv` | TSS(5) walk-forward by construction (`tss_folds_overlap_count = 0`). Step 5 LGBM Pipeline E uses anchored expanding training per KH-24 fold (`train_mask = entry_time < fold.oos_start`); `wfo_folds_train_into_oos_count = 0`. Parity reproduced AUC 0.750766. |
| 6 | Cluster label flow | GREEN | `audit_6_label_flow.md` | `y` is the binary target only (cluster-0 membership); `inference_uses_features_only = true`; no KMeans call in Step 5 inference path. |
| 7 | Execution semantics | GREEN | `audit_7_execution_semantics.csv` | 20/20 samples pass entry-bar-N+1 open execution + SL/exit semantics. Volume veto check N/A (Arc 9 signal has no volume veto). |
| 8 | E2E reproduction | GREEN | `audit_8_e2e_reproduction.csv` | 100/100 samples reproduce within 1e-6 tolerance; `max_abs_diff = 4.82e-11`; 0 mismatches. Deterministic in-memory rebuild matches the trained classifier bit-for-bit. |

## Anchor parity (informational; confirms the Step 5 per-fold AUCs the rebuild must reproduce)

Per `results/l_arc_9/experiments/step5_lgbm_pipeline_e/STEP5_LGBM_E_RESULT.md`:

| Fold | TSS-CV AUC (Pipeline E retry) | Reproduced AUC (Step 5 LGBM) |
|---|---|---|
| F1 | 0.848380 | 0.848380 |
| F2 | 0.735746 | 0.735746 |
| F3 | 0.717263 | 0.717263 |
| F4 | 0.714402 | 0.714402 |
| F5 | 0.738041 | 0.738041 |

Phase 1.1 rebuild must reproduce these byte-identically.

## Commit provenance

- Audit run commit: `9dc4f8a` (arc-9 audit: LOOKAHEAD_AND_LEAK — GREEN, candidates audit-clean)
- Branch: `claude/bold-brattain-d79817`
- RNG seed: `4242`
- Resolution authored on: `feature/arc9-canda-spec-onnx`
