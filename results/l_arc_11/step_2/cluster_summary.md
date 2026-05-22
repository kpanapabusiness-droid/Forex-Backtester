# Arc 11 v3.0 — Step 2 Cluster Summary

Per L_PROTOCOL §2 Step 2.

## Silhouette per K
- K=2: silhouette = 0.3936  ← primary
- K=3: silhouette = 0.3572
- K=4: silhouette = 0.3597
- K=5: silhouette = 0.3340
- K=6: silhouette = 0.3330

Primary K = **2**

## Per-cluster outcomes (K=2)

| cluster | n | mean_R | p50_R | mfe_p50 | mae_p50 | bars_held | dd_depth | recov | peaks | mono | ttp_rel | dominant_tag |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 2266 | 2.004 | -0.176 | 4.662 | -0.978 | 179.7 | 4.204 | 0.101 | 45.96 | 0.517 | 0.637 | choppy |
| 1 | 4752 | -0.995 | -1.000 | 0.472 | -1.159 | 18.2 | 1.446 | -5.236 | 4.29 | 0.361 | 0.286 | mixed |

## Shape-tag distribution (full pool)
- bimodal: 577
- choppy: 2727
- mixed: 1445
- monotonic_down: 838
- stepwise_climber: 36
- v_shape_recovery: 1395

