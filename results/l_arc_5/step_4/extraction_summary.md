# Step 4 — Extraction Summary

**Lineage-excluded features** (4): dollar_bloc_state, eur_strength_index, signal_density_28, usd_strength_index

| cluster | n | best clf | mean AUC | threshold | top-3 features |
|---:|---:|---|---:|---:|---|
| 0 | 121852 | rf | 0.7404 | 0.1495 | w1_close_slope_sign, swing_high_distance_14, day_of_week |
| 1 | 121852 | rf | 0.5403 | 0.3025 | d1_close_slope_magnitude, atr_14, prior_session_high_distance |
