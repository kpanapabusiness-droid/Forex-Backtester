# Step 4 — Extraction Summary

**Lineage-excluded features** (4): dollar_bloc_state, eur_strength_index, signal_density_28, usd_strength_index

| cluster | n | best clf | mean AUC | threshold | top-3 features |
|---:|---:|---|---:|---:|---|
| 0 | 2993 | lr | 0.6192 | 0.1140 | atr_14, atr_vs_trailing_100, w1_close_slope_sign |
| 1 | 2993 | rf | 0.6642 | 0.3534 | w1_close_slope_sign, d1_close_slope_sign, atr_percentile_100 |
