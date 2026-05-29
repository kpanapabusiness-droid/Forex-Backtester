# l_arc_8_v3.0.2 — Step 3 Capturability Summary

_Generated: 2026-05-25T09:34:37.243751+00:00Z_

SL sweep: [1.5, 2.0, 2.5, 3.0, 3.5, 4.0] x ATR.
Original Step 1 SL: 2.0 x ATR.

## Per-cluster best-SL metrics

| Cluster | Archetype (modal tag) | n | Best SL | Reach 1R | Reach 2R | Reach 3R | ww_pp | MFE p50 | Mean R | p50 R | SL-fire rate | Composite | Candidate? |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | Choppy (Mixed) | 3,418 | 1.5 | 60.42% | 29.05% | 12.55% | 39.50% | +1.291 | -0.997 | -1.000 | 99.85% | 0.570 | **no** |
| 1 | Monotonic down (Monotonic down) | 1,628 | 4.0 | 0.00% | 0.00% | 0.00% | 2.95% | +0.068 | -0.514 | -0.500 | 2.95% | 0.393 | **no** |
| 2 | Bimodal (Bimodal) | 1,566 | 1.5 | 100.00% | 99.74% | 97.06% | 0.00% | +7.550 | +2.621 | -1.000 | 57.15% | 1.000 | **YES** |

**Candidate clusters:** 1 (pass: reach_1R >= 0.50 AND ww_pp <= 0.30 AND mfe_p50 >= 1.5R)

## Full SL sweep per cluster

| Cluster | SL | Reach 1R | Reach 2R | ww_pp | MFE p50 | Mean R | SL-fire rate | Composite |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.5 | 60.42% | 29.05% | 39.50% | +1.291 | -0.997 | 99.85% | 0.570 |
| 0 | 2.0 | 48.71% | 17.26% | 51.11% | +0.968 | -0.991 | 99.62% | 0.455 |
| 0 | 2.5 | 37.86% | 9.25% | 22.38% | +0.775 | -0.866 | 36.69% | 0.514 |
| 0 | 3.0 | 29.05% | 4.59% | 9.77% | +0.646 | -0.708 | 14.10% | 0.520 |
| 0 | 3.5 | 22.73% | 2.19% | 4.92% | +0.553 | -0.593 | 6.35% | 0.508 |
| 0 | 4.0 | 17.26% | 0.64% | 2.84% | +0.484 | -0.513 | 3.51% | 0.490 |
| 1 | 1.5 | 2.83% | 0.12% | 97.11% | +0.182 | -0.999 | 99.94% | 0.035 |
| 1 | 2.0 | 0.98% | 0.00% | 98.89% | +0.136 | -0.999 | 99.88% | 0.017 |
| 1 | 2.5 | 0.37% | 0.00% | 36.86% | +0.109 | -0.873 | 37.04% | 0.261 |
| 1 | 3.0 | 0.12% | 0.00% | 14.43% | +0.091 | -0.714 | 14.43% | 0.349 |
| 1 | 3.5 | 0.00% | 0.00% | 6.08% | +0.078 | -0.597 | 6.08% | 0.381 |
| 1 | 4.0 | 0.00% | 0.00% | 2.95% | +0.068 | -0.514 | 2.95% | 0.393 |
| 2 | 1.5 | 100.00% | 99.74% | 0.00% | +7.550 | +2.621 | 57.15% | 1.000 |
| 2 | 2.0 | 100.00% | 98.15% | 0.00% | +5.663 | +2.689 | 41.00% | 1.000 |
| 2 | 2.5 | 100.00% | 95.53% | 0.00% | +4.530 | +2.112 | 19.48% | 1.000 |
| 2 | 3.0 | 99.74% | 90.55% | 0.06% | +3.775 | +1.761 | 9.39% | 0.999 |
| 2 | 3.5 | 98.91% | 83.59% | 0.19% | +3.236 | +1.513 | 5.49% | 0.995 |
| 2 | 4.0 | 98.15% | 76.76% | 0.26% | +2.831 | +1.325 | 3.77% | 0.980 |

## Archetype taxonomy → Step 5 architecture selection (Amendment 5 four-gate)

Per L_PROTOCOL Amendment 5 (`archive/L_PROTOCOL_v3_0_AMENDMENT_5.md`):
- **Gate 1 (Shape-required):** V-shape→A3; Stepwise→A4; Bimodal→A4; Monotonic up/down/Unclassified→(none)
- **Gate 2 (Classifier-driven):** Step 4 mean OOS AUC ≥ 0.65 → add A2 + A6 regardless of archetype
- **Gate 3 (Universal):** always add A1
- **Gate 4 (Portfolio, Amendment 5.1):** if ≥ 2 candidate clusters survive AND ≥ 1 PASS-tier under Gates 1/2/3 → add A5
- **Cluster skip:** Choppy archetype → skip Step 5 entirely

Final architecture set per cluster = UNION over admitting gates.

## Methodology notes

- Per-cluster MFE / MAE / final_R were rescaled from the Step-1 sim (SL=2.0xATR) by the closed-form rule in `_rescale_outcome_at_sl`.
- `ww_pp` approximated from summary MFE/MAE without per-bar timing.
- Capturability composite = `0.4 * reach_1R + 0.4 * (1 - ww_pp) + 0.2 * min(mfe_p50/3, 1)`. Ranking aid only, never a gate.
