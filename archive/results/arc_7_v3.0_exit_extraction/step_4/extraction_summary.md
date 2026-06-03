# Step 4 — Extraction Summary

**Lineage-excluded features** (4): dollar_bloc_state, eur_strength_index, signal_density_28, usd_strength_index

| cluster | n | best clf | mean AUC | threshold | top-3 features |
|---:|---:|---|---:|---:|---|
| 0 | 2993 | rf | 0.5037 | 0.1446 | prior_session_high_distance, d1_close_slope_magnitude, prior_session_low_distance |
| 1 | 2993 | lgbm | 0.5284 | 0.3315 | atr_14, prior_session_high_distance, kijun_26_distance |
