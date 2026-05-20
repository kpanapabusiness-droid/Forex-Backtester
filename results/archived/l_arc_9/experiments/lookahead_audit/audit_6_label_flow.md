# Audit 6 — Cluster label flow

## Question

Does `is_cluster_0` (computed from forward-geometry path features) leak into the
inference pathway as anything other than a target during training?

## Step 1 — `y` is target column, never a feature column

- Feature matrix has `y` column: True (used as target)
- Forbidden label columns appearing in EXPANDED_28: []

## Step 2 — Inference uses features-only X matrix

- `scripts/l_arc_9/experiments/step5_lgbm_pipeline_e.py` contains: `mdl.predict_proba(X[oos_mask])[:, 1]`: True
- The X matrix is `df_clean[EXPANDED_28].to_numpy(...)` — explicitly the feature columns, not the label column.

## Step 3 — No k-means at inference time

- KMeans imported in Step 5 LGBM Pipeline E script: False
- k-means is computed once at Step 2 (`scripts/l_arc_9/step2_clustering.py`) using only path-shape features;
  the resulting `clusters_K3.csv` provides the binary `is_cluster_0` labels used as classification target.

## Conclusion

- Label flows: Step 2 (k-means on path-shape) → `is_cluster_0` target → Step 4/E-retry training → trained model → Step 5 inference on entry-time features only.
- No leakage path: cluster labels never enter inference X. Classifier predicts is_cluster_0 from entry-time features only.
- **Verdict: GREEN**
