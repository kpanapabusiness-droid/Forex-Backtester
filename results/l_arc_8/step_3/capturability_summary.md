# Arc 8 — Step 3 Capturability Summary

_Generated: 2026-05-22T07:14:29.751958+00:00Z_

SL sweep: [1.5, 2.0, 2.5, 3.0, 3.5, 4.0] x ATR (dispatch §"Step 3", chat F3).
Original Step 1 SL: 2.0 x ATR.

## Per-cluster best-SL metrics

| Cluster | Archetype (modal tag) | n | Best SL | Reach 1R | Reach 2R | Reach 3R | ww_pp | MFE p50 | Mean R | p50 R | SL-fire rate | Composite | Candidate? |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | Monotonic down (Monotonic down) | 1,657 | 4.0 | 0.00% | 0.00% | 0.00% | 2.84% | +0.069 | -0.514 | -0.500 | 2.84% | 0.393 | **no** |
| 1 | Choppy (Mixed) | 3,560 | 1.5 | 59.78% | 29.41% | 12.81% | 40.03% | +1.246 | -0.981 | -1.000 | 99.44% | 0.562 | **no** |
| 2 | V-shape recovery (V-shape recovery) | 1,540 | 1.5 | 100.00% | 99.81% | 97.27% | 0.00% | +7.527 | +2.816 | -1.000 | 55.06% | 1.000 | **YES** |

**Candidate clusters:** 1 (pass: reach_1R >= 0.50 AND ww_pp <= 0.30 AND mfe_p50 >= 1.5R)

## Full SL sweep per cluster

| Cluster | SL | Reach 1R | Reach 2R | ww_pp | MFE p50 | Mean R | SL-fire rate | Composite |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.5 | 2.60% | 0.06% | 97.40% | +0.184 | -1.000 | 100.00% | 0.033 |
| 0 | 2.0 | 1.03% | 0.00% | 98.97% | +0.138 | -1.000 | 100.00% | 0.017 |
| 0 | 2.5 | 0.30% | 0.00% | 37.48% | +0.110 | -0.875 | 37.60% | 0.259 |
| 0 | 3.0 | 0.06% | 0.00% | 15.03% | +0.092 | -0.717 | 15.09% | 0.346 |
| 0 | 3.5 | 0.06% | 0.00% | 6.34% | +0.079 | -0.599 | 6.40% | 0.380 |
| 0 | 4.0 | 0.00% | 0.00% | 2.84% | +0.069 | -0.514 | 2.84% | 0.393 |
| 1 | 1.5 | 59.78% | 29.41% | 40.03% | +1.246 | -0.981 | 99.44% | 0.562 |
| 1 | 2.0 | 47.39% | 17.13% | 52.30% | +0.935 | -0.984 | 99.38% | 0.443 |
| 1 | 2.5 | 37.64% | 9.41% | 22.30% | +0.748 | -0.860 | 36.21% | 0.511 |
| 1 | 3.0 | 29.41% | 4.75% | 10.48% | +0.623 | -0.705 | 14.69% | 0.517 |
| 1 | 3.5 | 22.28% | 2.25% | 4.94% | +0.534 | -0.591 | 6.63% | 0.505 |
| 1 | 4.0 | 17.13% | 0.87% | 2.98% | +0.467 | -0.511 | 3.76% | 0.488 |
| 2 | 1.5 | 100.00% | 99.81% | 0.00% | +7.527 | +2.816 | 55.06% | 1.000 |
| 2 | 2.0 | 100.00% | 98.57% | 0.00% | +5.645 | +2.831 | 39.09% | 1.000 |
| 2 | 2.5 | 100.00% | 95.52% | 0.00% | +4.516 | +2.231 | 16.75% | 1.000 |
| 2 | 3.0 | 99.81% | 90.91% | 0.00% | +3.763 | +1.860 | 8.31% | 0.999 |
| 2 | 3.5 | 99.61% | 85.26% | 0.00% | +3.226 | +1.599 | 4.35% | 0.998 |
| 2 | 4.0 | 98.57% | 76.88% | 0.06% | +2.822 | +1.402 | 2.79% | 0.982 |

## Archetype taxonomy → Step 5 architecture selection

Per dispatch §"Step 5" archetype map:
- **Stepwise climber** → A1, A2, A4
- **V-shape recovery** → A1, A3, A6
- **Bimodal** → A1, A4
- **Monotonic up** → A1, A2, A6
- **Choppy** → no architectures; document and skip
- **A5 (portfolio composition)** → only if >= 2 candidate clusters survive

## Methodology notes

- Per-cluster MFE / MAE / final_R were rescaled from the Step-1 sim (SL=2.0xATR) by the closed-form rule in `_rescale_outcome_at_sl`: at swept SL multiplier s, trade SL-fires iff `mae_r_orig <= -1/(s/2.0)`; otherwise the original exit carries through with R units multiplied by `2.0/s`.
- `ww_pp` is approximated from summary MFE/MAE without per-bar timing — when both MFE and MAE exceed 1R, classified as wrong-way iff `|MAE| >= MFE` (conservative — over-estimates wrong-way on bimodal paths).
- Capturability composite = `0.4 * reach_1R + 0.4 * (1 - ww_pp) + 0.2 * min(mfe_p50/3, 1)`. Ranking aid only, never a gate.
