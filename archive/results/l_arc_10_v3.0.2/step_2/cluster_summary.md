# Arc 10 v3.0 — Step 2 Clustering Summary

- Pool size: 3152  (valid for clustering: 3152)
- Features: ['path_mono', 'path_peaks', 'path_ttp_rel', 'path_drawdown_depth_r', 'path_recovery_ratio']
- K tested: [2, 3, 4, 5, 6]; primary K = **3** (highest silhouette)

## Silhouette per K

| K | Silhouette | Inertia |
|---:|---:|---:|
| 2 | 0.4259 | 9019.20 |
| 3 | 0.4275 | 7768.66 | ◀ primary
| 4 | 0.2831 | 6856.60 |
| 5 | 0.2937 | 5628.81 |
| 6 | 0.2782 | 5010.10 |

## Cluster outcomes (K=3)

| Cluster | Archetype | n | share | mean_r | p25 | p50 | p75 | mfe_p50 | mfe_p75 | mae_p50 | ww_pp | c_path_mono | c_path_peaks | c_path_ttp_rel | c_path_drawdown_depth_r | c_path_recovery_ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1 | monotonic_down | 1658 | 0.526 | -0.843 | -1.000 | -1.000 | -1.000 | 1.964 | 3.512 | -5.127 | 0.530 | -0.514 | 26.608 | 0.222 | 7.717 | 0.293 |
| c0 | v_shape_recovery | 1493 | 0.474 | 0.860 | -1.000 | -1.000 | 1.483 | 5.309 | 7.895 | -1.872 | 0.495 | 0.619 | 26.357 | 0.804 | 4.153 | 0.754 |
| c2 | monotonic_down | 1 | 0.000 | -1.000 | -1.000 | -1.000 | -1.000 | 6.032 | 6.032 | -168.547 | 0.000 | -0.803 | 26.000 | 0.113 | 168.225 | 0.292 |
