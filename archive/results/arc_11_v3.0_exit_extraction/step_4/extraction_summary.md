# Step 4 — Extraction Summary

**Lineage-excluded features** (4): dollar_bloc_state, eur_strength_index, signal_density_28, usd_strength_index

| cluster | n | best clf | mean AUC | threshold | top-3 features |
|---:|---:|---|---:|---:|---|
| 0 | 9103 | lr | 0.5056 | 0.1537 | session_tokyo, prior_session_low_distance, d1_atr_percentile_100 |
| 1 | 9103 | rf | 0.5056 | 0.3546 | prior_session_low_distance, atr_vs_trailing_100, atr_percentile_100 |
