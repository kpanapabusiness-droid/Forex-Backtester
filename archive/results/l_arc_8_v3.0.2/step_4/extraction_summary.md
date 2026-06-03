# l_arc_8_v3.0.2 — Step 4 Extraction Summary

_Generated: 2026-05-25T09:38:36.166676+00:00Z_

- Engine path: `core.steps.step_4_extraction.run_step_4` (canonical, PR #185)
- IS filter: `train_end=2021-01-01T00:00:00Z` — restricts CV + classifier refit to entry_time < cutoff
- Candidate clusters processed: 1
- Classifiers: RF + LGBM + Logistic at Appendix A defaults
- CV: 5-fold TimeSeriesSplit
- HALT canary: AUC > 0.85 on any candidate cluster (per chat auto-run §3)

## Per-cluster, per-model OOS AUC

| Cluster | Model | n trades | Mean OOS AUC | Std AUC | Fold AUCs | AUC-best threshold (mean) | Best? |
|---:|---|---:|---:|---:|---|---:|:---:|
| 2 | rf | 3,337 | 0.4808 | 0.0274 | 0.5034;0.4623;0.5118;0.4377;0.4889 | 0.233 |  |
| 2 | lr | 3,337 | 0.4766 | 0.0323 | 0.5122;0.4828;0.4669;0.5013;0.4200 | 0.271 |  |
| 2 | lgbm | 3,337 | 0.4822 | 0.0425 | 0.5305;0.4746;0.5281;0.4193;0.4584 | inf | ★ |

## Best classifier per cluster (drives Step 5 A2/A6 via Amendment 2)

| Cluster | Best model | Mean AUC | Best threshold | Persisted classifier |
|---:|---|---:|---:|---|
| 2 | lgbm | 0.4822 | inf | `results\l_arc_8_v3.0.2\step_4\classifiers\2.pkl` |

## Amendment 5 Gate 2 decision (drives Step 5 A2/A6 admission)

- **c2 (Bimodal):** mean OOS AUC 0.4822 < 0.65 → Gate 2 does NOT fire → **A2 + A6 skipped**.

## Informational — prior arc references (NOT baselines)

- Arc 8 v3.0 (UTC + multi_tf all-NaN) c2 V-shape RF mean OOS AUC: 0.5300. NOT a baseline — different methodology.
- Prior `arc/l_arc_8_v3.0.2_halted` (UTC + W1 leaking) c2 V-shape mean AUC: 0.6938. NOT a baseline — driven by lookahead per verification doc §1.3.
- v3.0.2 (5ers_eet + canonical W1) per-cluster AUCs above are the canonical observation under this methodology.

## Excluded features (lineage filter, PR #185)

Per L_PROTOCOL §2 Step 4 + Amendment 2: features tagged anything other than `clean` excluded from training.

- `pair`
- `signal_time`
- `dollar_bloc_state`
- `eur_strength_index`
- `signal_density_28`
- `usd_strength_index`

## Top-10 permutation-importance features per cluster

### Cluster 2

| Feature | Mean importance | Std |
|---|---:|---:|
| atr_14 | +0.01007 | 0.02008 |
| swing_low_distance_14 | +0.00568 | 0.00990 |
| d1_close_slope_magnitude | +0.00175 | 0.00267 |
| kijun_26_distance | +0.00156 | 0.00733 |
| prior_session_low_distance | +0.00149 | 0.00741 |
| spread_vs_trailing_100 | +0.00077 | 0.00383 |
| prior_session_high_distance | +0.00070 | 0.00469 |
| spread_percentile_100 | +0.00014 | 0.00167 |
| session_dead | +0.00007 | 0.00082 |
| range_close_ratio | -0.00007 | 0.00445 |


## Amendment 5 architecture admission (drives Step 5)

See `amendment_5_admission.md` for the per-cluster four-gate evaluation and `architectures_skipped_by_amendment_5` projection for the closure tracker.

## Methodology notes

- Engine `core.steps.step_4_extraction.run_step_4` with `train_end=2021-01-01T00:00:00Z` and `persistence_dir=step_4/classifiers/`.
- Target = binary cluster membership (cluster_id == cid).
- TimeSeriesSplit preserves chronological order; PR #185 restricts CV + refit to IS-only.
- Best-AUC classifier refit on full lineage-filtered IS pool; pickled with SHA256 + provenance manifest.
- A2/A6 at Step 5 instantiate via `build_a2_config_from_step4` / `build_a6_config_from_step4` (no Step 5 retrain).
- Amendment 5 four-gate selection applied at dispatch time per observed AUC (no chat override).
