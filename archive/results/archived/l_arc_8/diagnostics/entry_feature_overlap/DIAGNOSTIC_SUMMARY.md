# Arc 8 — Entry-Feature Overlap Diagnostic — Summary

> Post-Step-5 follow-up. Determines whether c1 is separable from c0/c2/c3 at the entry bar using only entry-time features.

## Headline verdict

**c1_NOT_SEPARABLE_AT_ENTRY**

_c1 precision@recall=0.60 = 0.1493 < 0.20_

## Key numbers

- Multiclass RF on full 1,327 pool with 4 cluster labels
- 5-fold TimeSeriesSplit out-of-fold (1105 test predictions)
- OOF accuracy (4-class): **0.2995**
- c1 one-vs-rest AUC: **0.5465** (base rate 0.1334)
- c1 precision: 0.2254, recall: 0.1046, f1: 0.1429 (at default RF prediction threshold)
- **c1 precision@recall=0.60 (one-vs-rest, swept threshold): 0.1493** (decision metric)
  - threshold at this op-point: 0.1257

## Cross-cluster comparison summary

Mean overlap coefficient across all 18 features, by pairwise comparison (higher = more overlap = harder to separate at entry):

| Comparison | Mean overlap (across 18 features) |
|---|---:|
| c1_vs_c3 | 0.8479 |
| c1_vs_c2 | 0.8342 |
| c1_vs_c0 | 0.8305 |

**Hardest pair:** `c1_vs_c3` (mean overlap 0.8479)

## Univariate top-5 separating features (lowest mean overlap coefficient)

| Rank | Feature | Mean overlap | Min overlap | Mean KS stat | Verdict |
|---:|---|---:|---:|---:|:---:|
| 1 | `ret_5bar_atr` | 0.7813 | 0.7665 | 0.0995 | weak |
| 2 | `rsi_14` | 0.7876 | 0.7543 | 0.1568 | weak |
| 3 | `pos_in_20bar_range` | 0.7933 | 0.7874 | 0.1289 | weak |
| 4 | `trigger_close_pos` | 0.8134 | 0.7987 | 0.0714 | weak |
| 5 | `upper_wick_ratio` | 0.8150 | 0.8034 | 0.0714 | weak |

## Univariate bottom-5 (most-overlapping = least separating)

| Rank | Feature | Mean overlap | Verdict |
|---:|---|---:|:---:|
| 14 | `trigger_body_atr` | 0.8488 | weak |
| 15 | `hh_range_atr` | 0.8653 | indistinguishable |
| 16 | `range_to_atr_14` | 0.8669 | indistinguishable |
| 17 | `most_recent_sl_age` | 0.8782 | indistinguishable |
| 18 | `num_higher_lows` | 0.9514 | indistinguishable |

## Verdict distribution (per-feature)

- **separating**: 0 / 18 features
- **weak**: 14 / 18 features
- **indistinguishable**: 4 / 18 features

## Confusion matrix (out-of-fold, 4 classes)

Rows = true cluster, columns = predicted cluster.

| true \ pred | c0 | c1 | c2 | c3 | row sum |
|---|---:|---:|---:|---:|---:|
| c0 | 32 | 18 | 113 | 94 | 257 |
| c1 | 20 | 16 | 67 | 50 | 153 |
| c2 | 62 | 18 | 168 | 101 | 349 |
| c3 | 42 | 19 | 170 | 115 | 346 |

Per-class metrics (out-of-fold):

| Class | Base rate | OOF Precision | OOF Recall | F1 | AUC (1-vs-rest) |
|---:|---:|---:|---:|---:|---:|
| c0 | 0.2381 | 0.2051 | 0.1245 | 0.1550 | 0.4965 |
| c1 | 0.1334 | 0.2254 | 0.1046 | 0.1429 | 0.5465 |
| c2 | 0.3233 | 0.3243 | 0.4814 | 0.3875 | 0.5149 |
| c3 | 0.3052 | 0.3194 | 0.3324 | 0.3258 | 0.4891 |

## Recommended next action

Entry-time prediction is structurally hard for c1 vs c0/c2/c3. Pursue post-entry confirmation: run D1 t=3 + t=5 with multiclass framing (predict cluster ID at bar t, not just within-c1 success). The path-shape features that DEFINE the clusters appear at t≥10 (peak_mfe location), so later t may discriminate where entry features cannot. Alternatively, tighten the PR-HHHL signal trigger to bypass cluster prediction entirely.

## Files

- `entry_feature_univariate_overlap.csv` — long-format per-feature × comparison
- `entry_feature_separation_ranked.csv` — one row per feature, ranked by mean overlap
- `multiclass_diagnostic_results.json` — full multiclass output
- `multiclass_confusion_matrix.csv` — 4×4 confusion matrix
- `entry_feature_distributions_by_cluster.png` — 6×3 KDE grid all 18 features
- `top5_separating_features_detail.png` — deep dive on top-5 most-separating features
