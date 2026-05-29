# Step 5 — WFO Summary

**Total configs evaluated:** 20 (thin)

## Amendment 5 architecture admission per cluster

| Cluster | Archetype | Mean OOS AUC | Admitted | Skipped (vs Amendment 1) |
|---:|---|---:|---|---|
| 0 | bimodal | 0.5059 | A1 | (none) |
| 1 | unclassified | 0.5110 | A1 | (none) |

## Top-K Amendment 3 results

| Rank | Config | Verdict (Amended) | Worst ratio | r_safe | r_hard | Chained DD base | Daily breaches @r_safe |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `A1::A1::cl0::sl2.0::sl_plus_tp_2r::exp2` | fail | -0.95 | 0.0623% | 0.0779% | 99.8602% | 0 |
| 2 | `A1::A1::cl1::sl2.0::sl_plus_tp_2r::exp2` | fail | -0.95 | 0.0623% | 0.0779% | 99.8602% | 0 |
| 3 | `A1::A1::cl0::sl2.0::sl_plus_tp_2r::expinf` | fail | -0.96 | 0.0552% | 0.0690% | 99.9386% | 0 |
