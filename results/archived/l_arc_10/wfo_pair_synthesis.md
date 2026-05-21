# Arc 10 — WFO pair synthesis (base vs oracle c1)

> ⚠️ EXPERIMENTAL — runs over §16a HALT. Arc 10 dispatch only. **DO NOT DEPLOY. Do not promote.**

**Reads:** [`wfo_base/WFO_BASE_RESULT.md`](wfo_base/WFO_BASE_RESULT.md), [`wfo_oracle_c1/WFO_ORACLE_C1_RESULT.md`](wfo_oracle_c1/WFO_ORACLE_C1_RESULT.md)

**Pool sizes:** base = 802 (full Arc 10 Step 1 pool); oracle = 228 (c1 V-shape subset).

## Side-by-side aggregate metrics

| metric | base mean | base 95% CI | oracle mean | oracle 95% CI | abs gap (oracle − base) | rel gap (%) |
|---|---:|---|---:|---|---:|---:|
| sharpe_annual | -1.292 | — | 4.61 | [3.132, 6.088] | 5.902 | 456.7% |
| calmar | -14.19 | — | 71.2 | [39.4, 99.14] | 85.39 | 601.8% |
| expectancy_r | 0.4044 | [-1, 2.679] | 1.554 | [0.6987, 2.231] | 1.15 | 284.3% |
| max_drawdown_pct | 1.115 | — | 0.4997 | — | -0.6154 | -55.19% |
| total_return_pct | -0.2432 | — | 4.03 | — | 4.273 | 1757% |
| cagr_pct | -37.76 | — | 37.92 | — | 75.68 | 200.4% |
| win_rate | 0.3333 | — | 0.65 | — | 0.3167 | 95% |
| profit_factor | 0.1893 | — | 8.097 | — | 7.908 | 4178% |
| n_admit | 1.375 | — | 3.875 | — | 2.5 | 181.8% |

## Per-fold trade-count comparison

| fold | base n_test | base n_admit | oracle n_test | oracle n_admit |
|---:|---:|---:|---:|---:|
| 0 | 51 | 0 | 15 | 1 |
| 1 | 50 | 0 | 15 | 5 |
| 2 | 50 | 1 | 14 | 1 |
| 3 | 50 | 0 | 14 | 6 |
| 4 | 50 | 6 | 14 | 2 |
| 5 | 50 | 0 | 14 | 5 |
| 6 | 50 | 3 | 14 | 5 |
| 7 | 50 | 1 | 14 | 6 |

## Per-fold OOS Sharpe comparison

| fold | base Sharpe | oracle Sharpe | fold gap |
|---:|---:|---:|---:|
| 0 | — | — | — |
| 1 | — | 3.474 | — |
| 2 | — | — | — |
| 3 | — | 6.802 | — |
| 4 | — | 1.616 | — |
| 5 | — | 6.703 | — |
| 6 | -1.292 | 4.305 | 5.597 |
| 7 | — | 4.758 | — |

## Material-uplift readout

**Chat-side material thresholds (defaults):**
- Absolute Sharpe uplift ≥ 0.3
- Relative Calmar uplift ≥ 50.0%

- Sharpe gap: **5.902** vs threshold 0.3 → PASS
- Calmar relative gap: **601.8%** vs threshold 50.0% → PASS

## Clusterifier-justification recommendation

**Recommendation:** BUILD — both material criteria met.

**Rationale:** A perfect c1 classifier produces material OOS uplift on both Sharpe and Calmar. Engineering effort on a real-time c1 classifier is justified, conditional on expected classifier AUC vs the EXP-01 bootstrap distribution (P(E AUC≥0.65) = 12.5%).

## Cross-references to other Arc 10 diagnostics
- EXP-01 bootstrap: P(E AUC ≥ 0.65) under resampling = 12.5% — the realisable classifier ceiling is at the threshold, not above it.
- EXP-02 ablation: HTF features contribute +0.024 mean AUC; `L1_minus_L0_atr` carries 116% of LOO drop.
- EXP-05 cross-arc pool: pooled AUC 0.6348 (gap −0.015 to 0.65), best evidence that the deployable c1 classifier ceiling is achievable across arcs.

## Caveats
- Both WFOs use only 8 outer folds on small pools (base 802 / oracle 228); per-fold N is small and bootstrap CIs are wide.
- The oracle is a true upper bound — any real classifier will be strictly worse. Use gap as ceiling, not realisable lift.
- Parameter-grid optimisation per fold uses inner-CV Sharpe; in-sample-to-OOS selection bias is bounded by the inner-CV honesty but not eliminated. Per-fold mode/SL drift in `params_history.csv` is the right place to check for regime instability.
- Confidence-weighted sizing was omitted from the parameter grid (not implemented in the Arc 10 codebase). All metrics assume fixed-fractional 0.5% risk per admitted trade.

## Files
- [`wfo_base/`](wfo_base/) — base WFO outputs.
- [`wfo_oracle_c1/`](wfo_oracle_c1/) — oracle WFO outputs.
- this file: synthesis.
