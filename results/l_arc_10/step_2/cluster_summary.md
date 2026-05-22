# Arc 10 v3.0 — Step 2 Clustering Summary

- Pool size: 3301  (valid for clustering: 3301)
- Features: ['path_mono', 'path_peaks', 'path_ttp_rel', 'path_drawdown_depth_r', 'path_recovery_ratio']
- K tested: [2, 3, 4, 5, 6]; primary K = **3** (highest silhouette)

## Silhouette per K

| K | Silhouette | Inertia |
|---:|---:|---:|
| 2 | 0.4215 | 9528.91 |
| 3 | 0.4243 | 8050.26 | ◀ primary
| 4 | 0.2711 | 7139.49 |
| 5 | 0.2862 | 6056.18 |
| 6 | 0.2767 | 5376.37 |

## Cluster outcomes (K=3)

| Cluster | Archetype | n | share | mean_r | p25 | p50 | p75 | mfe_p50 | mfe_p75 | mae_p50 | ww_pp | c_path_mono | c_path_peaks | c_path_ttp_rel | c_path_drawdown_depth_r | c_path_recovery_ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c0 | monotonic_down | 1771 | 0.537 | -0.845 | -1.000 | -1.000 | -1.000 | 1.995 | 3.543 | -5.121 | 0.528 | -0.522 | 26.671 | 0.219 | 7.786 | 0.294 |
| c1 | v_shape_recovery | 1528 | 0.463 | 0.928 | -1.000 | -1.000 | 1.602 | 5.267 | 7.781 | -1.893 | 0.506 | 0.617 | 26.469 | 0.806 | 4.224 | 0.754 |
| c2 | monotonic_down | 2 | 0.001 | -1.000 | -1.000 | -1.000 | -1.000 | 3.071 | 4.482 | -143.814 | 0.500 | -0.795 | 28.000 | 0.054 | 140.815 | 0.356 |
