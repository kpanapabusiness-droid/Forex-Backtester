# Arc 9 Candidate A Scaled-Risk WFO

> Re-accounts the fixed Candidate A admit set (236 trades, F2-F7) at five
> per-trade risk levels with intraday-resolution worst-day DD measurement.
> No classifier retraining, no admit-set modification, no threshold change —
> only the dollar-size-per-trade parameter varies. Establishes the
> **measured** deployment risk, replacing the projection-based 2% assumption.

## Headline

**Recommended deployment risk: 1.0% per trade.**

| Metric at 1.0% | Value | Margin to constraint |
|---|---|---|
| Worst-fold max DD | 2.52% | 5.48pp under in-system 8% target |
| Worst-day DD | 2.01% | 1.99pp under in-system 4% target / 2.99pp under 5ers 5% hard limit |
| Worst-fold annualised ROI | +20.07% (F3) | 15pp above pass-deployable 5% floor |
| Full-data annualised ROI | +41.45% | — |
| All folds positive (F2-F7) | YES | — |

The projection-based 2% assumption is **NOT supported** by measurement. At 2% per-trade risk the worst-day DD is **3.99%**, which is essentially at the in-system 4% target with effectively zero margin for out-of-sample worse days. The binding constraint is **worst-day DD**, not worst-fold max DD — a metric the original projection did not measure.

## Per-risk-level summary table

| Risk | Worst-fold ann ROI | Mean fold ann ROI | Worst-fold max DD | Worst-day DD | Full-data ann ROI | Full-data max DD | Pass in-system (8% fold / 4% day)? | Pass 5ers hard (10% / 5%)? |
|---|---|---|---|---|---|---|---|---|
| **0.5%** | +9.63% | +22.92% | 1.26% | 1.01% | +19.06% | 1.26% | **PASS** / **PASS** | **PASS** / **PASS** |
| **1.0%** | +20.07% | +51.57% | 2.52% | 2.01% | +41.45% | 2.52% | **PASS** / **PASS** | **PASS** / **PASS** |
| **1.5%** | +31.39% | +87.43% | 3.76% | 3.00% | +67.68% | 3.76% | **PASS** / **PASS** | **PASS** / **PASS** |
| **2.0%** | +43.65% | +132.36% | 5.00% | 3.99% | +98.36% | 5.00% | **PASS** / **PASS** (0.01pp margin) | **PASS** / **PASS** (1.01pp margin) |
| **2.5%** | +56.91% | +188.71% | 6.23% | 4.98% | +134.17% | 6.23% | **PASS** / **FAIL** (4.98% > 4%) | **PASS** / **PASS** (0.02pp margin) |

Recommendation rule applied (in-system + 1pp safety margin):
- Worst-fold max DD ≤ 6% → 0.5%, 1.0%, 1.5%, 2.0% all eligible (2.5% borderline-fail at 6.23%)
- **Worst-day DD ≤ 3%** → 0.5% (1.01%), 1.0% (2.01%) clearly eligible; **1.5% (3.0018%) misses by 0.0018pp**; 2.0% fails (3.99%), 2.5% fails (4.98%)
- All folds positive → all 5 eligible
- ROI scaling sanity → all 5 eligible

**Highest risk level satisfying all four rules: 1.0%.**

## Scaling sanity checks

| Risk vs baseline (0.5%) | ROI ratio observed | ROI ratio vs linear projection | DD ratio observed | DD ratio vs linear |
|---|---|---|---|---|
| 2× (1.0%) | 2.17× | 1.09× (mild compounding lift) | 1.99× | 1.00× (exactly linear) |
| 3× (1.5%) | 3.55× | 1.18× | 2.98× | 0.99× |
| 4× (2.0%) | 5.16× | 1.29× | 3.96× | 0.99× |
| 5× (2.5%) | 7.04× | 1.41× | 4.93× | 0.99× |

**DD scales essentially exactly linearly with risk.** This is the expected result because:
- The trade set is fixed (same admit decisions across risk levels)
- final_r per trade is fixed (same SL distance, same §11 exit)
- Only position-dollar-size varies
- Loss sequencing is unchanged

**ROI scales supra-linearly** due to compounding: each subsequent trade's dollar-size is computed from current (compounded) equity. At 5× risk, observed full-data ROI is 7.04× baseline (linear would be 5×) — the +41% extra capture comes from compounding wins onto wins.

DD ratio vs linear is consistently 0.99× — essentially exact linear scaling. This is the structurally-clean signal: no non-linear compounding distortion in DD, so DD scaling can be projected confidently across risk levels.

## Per-fold detail at each risk level

### 0.5% (baseline)

| F | Trades | Fold ROI % | Ann ROI % | Max DD % | Worst-day DD % | End equity |
|---|---|---|---|---|---|---|
| 1 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | $10,000 (no admits) |
| 2 | 50 | +27.40 | +38.10 | 1.01 | **1.01** | $12,740 |
| 3 | 24 | +7.17 | +9.63 | 0.50 | 0.50 | $10,717 |
| 4 | 32 | +12.25 | +16.72 | 0.80 | 0.51 | $11,225 |
| 5 | 39 | +12.59 | +17.12 | 1.26 | 0.52 | $11,259 |
| 6 | 44 | +19.21 | +26.40 | 1.02 | 1.01 | $11,921 |
| 7 | 47 | +21.54 | +29.57 | 0.71 | 0.52 | $12,154 |

### 1.0% (RECOMMENDED)

| F | Trades | Fold ROI % | Ann ROI % | Max DD % | Worst-day DD % | End equity |
|---|---|---|---|---|---|---|
| 1 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | $10,000 |
| 2 | 50 | +61.68 | +89.74 | 2.01 | 1.03 | $16,168 |
| 3 | 24 | +14.77 | +20.07 | 1.01 | 1.01 | $11,477 |
| 4 | 32 | +25.87 | +36.05 | 1.61 | 1.01 | $12,587 |
| 5 | 39 | +26.58 | +36.91 | 2.52 | 1.05 | $12,658 |
| 6 | 44 | +41.82 | +59.32 | 2.03 | **2.01** | $14,182 |
| 7 | 47 | +47.35 | +67.34 | 1.43 | 1.03 | $14,735 |

### 1.5%, 2.0%, 2.5%

Per-fold detail in `per_risk_0150/per_fold_metrics.csv`, `per_risk_0200/per_fold_metrics.csv`, `per_risk_0250/per_fold_metrics.csv`. All metrics scale linearly from 1.0% as shown in the summary table.

## Worst-day DD analysis (the binding constraint)

| Risk | Worst day | Day DD % | Fold | # trades | # pairs | Pairs | Net day P&L |
|---|---|---|---|---|---|---|---|
| 0.5% | **2024-08-06** | 1.01% | 6 | 2 | 2 | GBP_CHF, NZD_USD | −$101.68 |
| 1.0% | **2024-08-06** | 2.01% | 6 | 2 | 2 | GBP_CHF, NZD_USD | −$205.08 |
| 1.5% | **2024-08-06** | 3.00% | 6 | 2 | 2 | GBP_CHF, NZD_USD | −$310.19 |
| 2.0% | **2024-08-06** | 3.99% | 6 | 2 | 2 | GBP_CHF, NZD_USD | −$417.02 |
| 2.5% | **2024-08-06** | 4.98% | 6 | 2 | 2 | GBP_CHF, NZD_USD | −$525.57 |

**Single day (2024-08-06) is the binding constraint at every risk level.** Two trades on two different pairs (GBP_CHF and NZD_USD) both hit −1R SL on the same calendar day. Net day P&L = −2.02R combined. The cross-pair concentration is the structural risk profile.

Second-worst days (sample at 1.0%):
- 2024-03-06 (F5): 1 trade (cluster 0 admit, SL hit), 1.05% day DD
- 2025-09-02 (F7): 2 trades net 1.03% DD
- 2021-10-27 (F2): 1 trade, 1.03% DD

**Diagnostic reading:** the worst-day risk is dominated by rare same-day multi-pair SL clusters. With only one such event (2024-08-06) in 5+ years of OOS, the small sample limits confidence — a future out-of-sample worse day cluster is plausible. The 1pp safety margin in the recommendation absorbs roughly one additional same-day-2-SL event without breaching the in-system target.

**Pair concentration:** at the recommended 1.0% risk, the worst day involved 2 distinct pairs (no single-pair concentration). This argues against a per-pair scaling rule; the same-day cluster is across pairs, not within one. No pair-cap intervention would have changed the worst-day outcome.

## Comparison to the projection-based 2% assumption

The `ARC_9_CANDIDATE_A_SPEC.md` 2% recommendation was based on linear scaling of the headline `worst-fold DD 1.32%` figure from Step 5 LGBM E: projected 4× = ~5.3% at 2% risk, comfortably under the 8% in-system DD target.

| Axis | Projected (linear from 0.5% baseline at 4× risk) | Measured at 2.0% | Delta |
|---|---|---|---|
| Worst-fold max DD | ~5.3% | 5.00% | −0.30pp (projection slightly conservative) |
| Worst-day DD | **NOT MEASURED in projection** | **3.99%** | **new — binding constraint** |
| Full-data ann ROI | ~76% (linear 19% × 4) | +98.36% | +22pp (compounding lift, projection conservative) |
| Pass in-system 4% daily target | NOT EVALUATED | 0.01pp margin | **Knife-edge — effectively zero out-of-sample headroom** |
| Pass 5ers 5% daily hard limit | NOT EVALUATED | 1.01pp margin | Tight margin |

**Lesson: the worst-fold DD projection was approximately correct; the missing axis was worst-day DD.** Linear DD scaling held empirically (DD ratio observed / linear = 0.99 across all risk multiples). But the 2% projection didn't notice that worst-day DD scales linearly too, and at 2% risk that gets you to ~4% worst-day, leaving no headroom against the in-system target.

If the recommendation strictly followed "in-system target with zero margin" (worst-day DD ≤ 4%, max-fold ≤ 8%), the recommended risk would be **2.0%** — but with no out-of-sample buffer. Adding 1pp safety to both axes (recommendation rules: ≤ 6% fold, ≤ 3% worst-day) brings the recommendation down to **1.0%**.

Risk-tier comparison for analyst decision:
| Tier | Risk | Worst-day DD | Margin to 4% in-system | Margin to 5% 5ers hard | Notes |
|---|---|---|---|---|---|
| Conservative (recommendation) | 1.0% | 2.01% | 1.99pp | 2.99pp | Robust to ~1 additional bad-day cluster in OOS |
| Aggressive in-system | 2.0% | 3.99% | 0.01pp | 1.01pp | Knife-edge against in-system; one bad cluster breaches |
| Maximum 5ers-compliant | 2.5% | 4.98% | breaches in-system | 0.02pp | At 5ers wire; one bad day = account closure |
| Projection (un-measured) | 2.0% | (unknown at projection time) | (unknown) | (unknown) | Based on max-fold DD only |

## Recommendation

**Deploy Candidate A at 1.0% per-trade risk.** Measured worst-fold max DD 2.52%, worst-day DD 2.01%, worst-fold ann ROI +20.07%, full-data ann ROI +41.45%.

This is **half** the projection-based 2% assumption. The reduction is the cost of measuring worst-day DD at intraday compounded resolution — a metric the projection didn't address. The measurement reveals that:

1. **DD scales linearly with risk** (DD ratio observed / linear ≈ 1.00 across all risk multiples), confirming the projection's scaling assumption was correct on the max-fold-DD axis.
2. **Worst-day DD is the binding constraint, not worst-fold max DD.** Worst-fold max DD has 5+ pp headroom under in-system 8% target at all candidate risk levels; worst-day DD hits the 4% target at exactly 2.0% risk with no margin.
3. **The 2024-08-06 single-day cluster (2 same-day SLs across GBP_CHF and NZD_USD) is the single observed event driving the constraint.** With one such event in 5+ years of OOS, out-of-sample risk could produce a worse cluster — the 1pp safety margin is sized for ~one additional cluster of equivalent magnitude.
4. **ROI scales supra-linearly via compounding.** The recommended 1.0% risk produces +41.45% full-data ann ROI (2.18× baseline rather than the linear 2.00× — the +9pp compounding lift partially compensates for the lower-risk choice).

## Cross-arc note

The projection vs measurement gap surfaced here applies to ANY arc deploying through 5ers (or other prop-firm) infrastructure. The standard project convention has been to report worst-fold DD as the headline risk metric; worst-day DD has not been routinely measured. For future arcs reaching deployment candidate status, **measuring worst-day DD at intraday compounded resolution should be a standard Step 5 output**, not a separate dispatch. The dispatch's framing — "scaling assumptions are not measurements" — is the lesson, and it generalises.

Specifically: KH-24's own deployment risk (1.0% per trade per `KH24_SYSTEM_LOCK.md` line 193 "2% breaches 5ers daily cap") was set conservatively without explicit worst-day DD measurement; this scaled-risk methodology applied to KH-24 would either confirm 1.0% or refine it. Worth a follow-on dispatch.

## Determinism

Two independent runs at all 5 risk levels produce byte-identical `summary_table.csv` — **PASS**. The accounting is fully deterministic (no randomness; trade set and final_r are inputs from the prior Step 5 LGBM E run).

## Files

- `summary_table.csv` — 5-risk-level × per-fold-aggregate + full-data + gate columns
- `summary.json` — machine-readable summary + recommendation rationale
- `worst_day_analysis.csv` — single worst day per risk level with contributing trades/pairs
- `per_risk_0050/, per_risk_0100/, per_risk_0150/, per_risk_0200/, per_risk_0250/` — per-risk-level subdirectories with:
  - `per_fold_metrics.csv` — 7-fold table including F1 zero-admits
  - `per_day_dd.csv` — every trading day with closures, ranked by day DD%
  - `full_data_metrics.json` — compounded across all folds
- `determinism_check.json` — 2-run sha256 byte-identical PASS
