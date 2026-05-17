# Lω regime-conditional dispatch — three binning variants, 1H + 4H

> Tests post-mortem (PR #134) interpretation 1: regime-conditional non-stationarity.
> Fits the B1-B4 top-15 RF on bin-only rows; reports per-bin worst-fold AUC against
> a gate of ≥ 0.55 with ≥ 1000 training samples per fold. Reuses PR #133 features +
> labels verbatim. No new features, no new labels, same RF hyperparameters, same
> 7-fold anchored-expanding TimeSeriesSplit. Determinism verified byte-identical
> across 4H re-run (all 6 output files).

**Scope note on fold dates.** The dispatch's hard constraint specifies "identical
fold dates" to B1-B4. That is geometrically incompatible with Variant C (each
year-quarter bin spans ~3 months, while B1-B4 folds span the full 2020-2026
timeline → most cells empty). For uniform treatment across variants, the same
`TimeSeriesSplit(n_splits=7)` algorithm B1-B4 uses is applied to each bin's
time-sorted subset. Fold dates therefore differ across bins. This loses direct
fold-by-fold comparability to B1-B4 baseline; aggregate (worst-of-7) comparison
is preserved. Treated as a deviation worth documenting; alternate interpretation
of the constraint produces no usable result for Variant C.

---

## 1. Headline verdict

**Interpretation 1 is NOT supported by the data.** Of the 62 cells tested (32 at
1H, 30 at 4H¹, across 3 binning variants), **exactly one cell technically clears
the gate**: 4H Variant C bin `2020-Q2`, worst-fold AUC 0.5528 (3 basis points
above the 0.55 threshold). That cell is small (10,920 rows = 4.3% of 4H data),
historic (Q2 2020 COVID era), and solving a fundamentally different prediction
problem from B1-B4 (intra-quarter forward prediction vs B1-B4's cross-time
generalization). It is not a deployable regime indicator under any reasonable
reading — anti-evidence is dispositive (§8).

**At 1H: 0 of 32 cells clear gate.** Best 1H cell is Variant C `2022-Q4` at
worst-fold AUC 0.5295, well below 0.55. **At 4H: 1 of 30 cells clear (marginally),
0 of 8 in Variants A and B.** No combination of vol regime, D1 trend regime, or
year-quarter regime recovers price-feature predictive signal in the worst fold.

¹ 4H Variant C: 23 year-quarters (2020-Q1 through 2025-Q3). 2025-Q4 not labelled
at 4H because the 240-bar forward window extends past the data end date.

---

## 2. Per-variant per-bin AUC tables

B1-B4 baseline worst-fold AUC (from PR #134 §1): 1H 0.4841, 4H 0.4733.

### 1H — Variant A (vol percentile, 5 quintiles on atr_percentile_60)

| bin | min train | worst-fold AUC | mean fold AUC | mean base rate | gate |
|---|---:|---:|---:|---:|:-:|
| vol_q1 | 24,594 | 0.4913 | 0.5365 | 0.122 | FAIL |
| vol_q2 | 26,511 | 0.4808 | 0.5359 | 0.121 | FAIL |
| vol_q3 | 25,116 | 0.4762 | 0.5245 | 0.123 | FAIL |
| vol_q4 | 27,061 | 0.4847 | 0.5311 | 0.125 | FAIL |
| vol_q5 | 25,868 | 0.4906 | 0.5312 | 0.130 | FAIL |

Tight cluster ±0.01 around 0.485; no quintile improves materially over B1-B4
baseline 0.4841. Quintile edges (full-data): 0.133 / 0.350 / 0.583 / 0.833.

### 1H — Variant B (D1 trend sign, 3 bins on d1_ema_50_slope_at_entry)

| bin | min train | worst-fold AUC | mean fold AUC | mean base rate | gate |
|---|---:|---:|---:|---:|:-:|
| d1_down | 55,412 | 0.4954 | 0.5292 | 0.113 | FAIL |
| d1_flat | 9,103 | 0.4940 | 0.5233 | 0.130 | FAIL |
| d1_up | 64,628 | 0.4686 | 0.5282 | 0.134 | FAIL |

Threshold widened from 0.001 to 0.002 at 1H because d1_flat at 0.001 was below
5% of finite data (final flat share: 7.0% at 0.002). Best bin (d1_down +0.011
over baseline) within fold stdev.

### 1H — Variant C (year-quarter, 24 bins; ≥ 5000 samples retained)

Top-8 bins by worst-fold AUC (full 24-bin table in `variant_c_year_quarter/variant_summary.csv`):

| bin | n_rows | worst-fold AUC | mean fold AUC | fold stdev | gate |
|---|---:|---:|---:|---:|:-:|
| 2022-Q4 | 43,502 | 0.5295 | 0.6201 | 0.047 | FAIL |
| 2024-Q2 | 43,680 | 0.5136 | 0.5888 | 0.073 | FAIL |
| 2021-Q2 | 43,624 | 0.5106 | 0.6174 | 0.057 | FAIL |
| 2021-Q4 | 44,240 | 0.4907 | 0.5934 | 0.096 | FAIL |
| 2021-Q1 | 42,280 | 0.4936 | 0.5354 | 0.029 | FAIL |
| 2020-Q4 | 43,541 | 0.4885 | 0.5474 | 0.041 | FAIL |
| 2023-Q1 | 43,624 | 0.4840 | 0.5577 | 0.048 | FAIL |
| 2025-Q3 | 44,352 | 0.4818 | 0.5365 | 0.043 | FAIL |

No 1H year-quarter clears gate. Bottom of distribution (2024-Q1, 2025-Q4) has
worst-fold AUC as low as 0.21-0.33 with fold stdev > 0.10 — large variance
characteristic of small per-fold val_n (~5000 per fold).

### 4H — Variant A (vol percentile, 5 quintiles)

| bin | min train | worst-fold AUC | mean fold AUC | mean base rate | gate |
|---|---:|---:|---:|---:|:-:|
| vol_q1 | 5,897 | 0.4774 | 0.5186 | 0.182 | FAIL |
| vol_q2 | 6,550 | 0.4464 | 0.4996 | 0.191 | FAIL |
| vol_q3 | 6,447 | 0.4589 | 0.5085 | 0.197 | FAIL |
| vol_q4 | 6,477 | 0.4786 | 0.5057 | 0.194 | FAIL |
| vol_q5 | 6,538 | 0.4466 | 0.4954 | 0.201 | FAIL |

Worse than B1-B4 baseline (0.4733) for vol_q2, vol_q3, vol_q5. Vol regime is
NOT a useful regime variable at 4H. Quintile edges (full-data): 0.100 / 0.317 /
0.583 / 0.850.

### 4H — Variant B (D1 trend sign, 3 bins)

| bin | min train | worst-fold AUC | mean fold AUC | mean base rate | gate |
|---|---:|---:|---:|---:|:-:|
| d1_down | 13,701 | 0.4678 | 0.5124 | 0.177 | FAIL |
| d1_flat | 2,244 | 0.4882 | 0.5721 | 0.190 | FAIL |
| d1_up | 15,964 | 0.4041 | 0.4967 | 0.206 | FAIL |

Threshold widened from 0.001 to 0.002 because d1_flat at 0.001 was below 5% of
finite data (final flat share: 7.0% at 0.002). d1_flat has best worst-fold AUC
(0.488, +0.015 over baseline) and best mean (0.572) but still below 0.55 gate.
d1_up has the worst fold AUC of any A/B bin at either TF — 0.404.

### 4H — Variant C (year-quarter, 23 bins; ≥ 5000 samples retained)

Top-8 bins by worst-fold AUC (full 23-bin table in `variant_c_year_quarter/variant_summary.csv`):

| bin | n_rows | worst-fold AUC | mean fold AUC | fold stdev | gate |
|---|---:|---:|---:|---:|:-:|
| **2020-Q2** | **10,920** | **0.5528** | **0.6418** | **0.052** | **PASS** |
| 2025-Q1 | 10,611 | 0.5476 | 0.6629 | 0.069 | FAIL |
| 2020-Q4 | 10,920 | 0.5473 | 0.6647 | 0.066 | FAIL |
| 2025-Q3 | 11,088 | 0.5493 | 0.5970 | 0.047 | FAIL |
| 2021-Q2 | 10,920 | 0.5298 | 0.6439 | 0.070 | FAIL |
| 2021-Q1 | 10,584 | 0.5262 | 0.6269 | 0.083 | FAIL |
| 2023-Q2 | 10,920 | 0.4811 | 0.5423 | 0.040 | FAIL |
| 2023-Q1 | 10,920 | 0.4767 | 0.6284 | 0.106 | FAIL |

2020-Q2 clears by 0.003. Three near-misses cluster within 0.006 of gate
(2025-Q1, 2020-Q4, 2025-Q3); given mean fold stdev ~0.06 in this variant, the
distinction between "0.553 pass" and "0.547 fail" is well within sampling noise.

---

## 3. Bin sample-size feasibility

| TF | Variant | n_bins | dropped (< 5000) / insufficient (any fold < 1000 train OR < 200 val) | n_trained |
|---|---|---:|---|---:|
| 1H | A vol_q | 5 | 0 | 5 |
| 1H | B d1_trend | 3 | 0 | 3 |
| 1H | C year-quarter | 24 | 0 dropped (all 24 quarters ≥ 5000 rows; smallest 2025-Q4 = 30,240) | 24 |
| 4H | A vol_q | 5 | 0 | 5 |
| 4H | B d1_trend | 3 | 0 | 3 |
| 4H | C year-quarter | 23 | 0 dropped; 2025-Q4 never appears in labels (forward window 240 bars extends past data end) | 23 |

**No bin in any variant at either TF was marked `insufficient_data`.** Every
fold of every kept bin had ≥ 1000 train samples and ≥ 200 validation samples.
This is informative on its own: the per-bin sample sizes are healthy enough
that the failure to clear gate is not a sample-size artifact.

Variant C 4H has 23 bins instead of 24 because the 4H forward window (240 bars
≈ 40 calendar days) cuts label production off at 2025-11-04, before any
2025-Q4 bars are fully labelled. 1H (480-bar forward window ≈ 20 days) cuts off
at 2025-12-02, so 1H 2025-Q4 retains 30,240 partial-quarter rows.

---

## 4. Comparison vs B1-B4 baseline

B1-B4 baseline worst-fold AUC: 1H 0.4841, 4H 0.4733. Per-variant best bin and
delta vs baseline:

| TF | Variant | best bin | worst-fold AUC | Δ vs baseline | fold stdev (best bin) | within stdev? |
|---|---|---|---:|---:|---:|---|
| 1H | A vol_q | vol_q1 | 0.4913 | +0.007 | 0.039 | yes |
| 1H | B d1_trend | d1_down | 0.4954 | +0.011 | 0.022 | partly |
| 1H | C year-quarter | 2022-Q4 | 0.5295 | **+0.045** | 0.047 | borderline |
| 4H | A vol_q | vol_q4 | 0.4786 | +0.005 | 0.017 | yes |
| 4H | B d1_trend | d1_flat | 0.4882 | +0.015 | 0.050 | yes |
| 4H | C year-quarter | 2020-Q2 | 0.5528 | **+0.080** | 0.052 | borderline |

**Variants A and B (vol regime, trend regime): no meaningful improvement at
either TF.** All best-bin deltas (+0.005 to +0.015) fall within the
corresponding bin's own fold-AUC standard deviation. The vol-regime hypothesis
that motivated this dispatch from postmortem D4 evidence is not corroborated:
splitting by vol regime does not recover predictive signal in the worst fold.

**Variant C (year-quarter): real but unusable improvements.** The 1H 2022-Q4
delta (+0.045) and 4H 2020-Q2 delta (+0.080) are larger than fold stdev but
both come from a fundamentally different prediction problem (within-quarter,
not cross-time). They are not direct evidence that regime conditioning recovers
signal; they are evidence that within-quarter prediction is easier than
cross-quarter prediction (which is the standard auto-correlation finding, not
a regime finding).

---

## 5. Cross-variant consistency

Full cross-tab and excess-vs-independence analysis in
`cross_variant_overlap.txt` (generated by `scripts/lomega/lomega_regime_overlap.py`).

### Variant A × Variant B — independent

The vol-quintile × D1-trend-sign cross-tab is essentially flat:
all 15 cells at 1H and all 15 cells at 4H are within ±0.08 of the independent-prior
expectation. **Vol regime and D1 trend sign are nearly independent.** They are
measuring different things, not the same regime through different lenses.

### Variant A × Variant C — overlap concentrated in COVID era

Largest |observed - expected|/expected cells (4H, raw counts ≥ 100):

| A bin | C bin | obs | expected | rel excess |
|---|---|---:|---:|---:|
| vol_q1 | 2020-Q2 | 3,679 | 2,022 | +0.82 |
| vol_q5 | 2020-Q1 | 3,682 | 2,191 | +0.68 |
| vol_q5 | 2024-Q1 | 1,439 | 2,199 | -0.35 |
| vol_q5 | 2025-Q2 | 2,687 | 2,022 | +0.33 |
| vol_q5 | 2024-Q3 | 3,001 | 2,267 | +0.32 |

Strongest co-occurrences are 2020-Q1 (excess high-vol — COVID spike) and
2020-Q2 (excess low-vol — vol mean-reverting after spike). Recent years
(2023-2025) deviate by ≤ 0.35× — modest. **`atr_percentile_60` is a rolling
within-pair rank, so a Q3 2024 absolute vol peak does not directly produce a
vol_q5 surplus at the bar level** (each bar is ranked vs its recent 60 bars,
not vs the full panel). The COVID-era overlap dominates the cross-tab and is
not generalisable.

### Variant B × Variant C — d1_flat overrepresented in specific quarters

Largest |observed - expected|/expected cells (4H):

| B bin | C bin | obs | expected | rel excess |
|---|---|---:|---:|---:|
| d1_flat | 2020-Q4 | 1,740 | 766 | +1.27 |
| d1_flat | 2023-Q4 | 1,303 | 756 | +0.72 |
| d1_flat | 2022-Q1 | 1,200 | 754 | +0.59 |
| d1_flat | 2022-Q2 | 366 | 766 | -0.52 |
| d1_down | 2024-Q2 | 2,256 | 4,706 | -0.52 |
| d1_up | 2024-Q2 | 7,824 | 5,448 | +0.44 |
| d1_down | 2020-Q2 | 6,612 | 4,706 | +0.41 |

D1 trend sign captures genuine cross-quarter trend epochs. 2024-Q2 was a USD
strength quarter (d1_up overrepresented +44%), 2020-Q2 was a USD weakness /
risk-on rebound (d1_down +41%), 2020-Q4 / 2022-Q1 / 2023-Q4 were
"slope-neutral" macro periods (d1_flat dominant). Variant B is finding real
structure that Variant A does not — but the structure is not predictive of
clean forward paths in any single bin.

### Bottom line on cross-variant consistency

The three variants are **largely orthogonal**: A and B are independent
(correlation cross-tab approximately flat); A and C only co-occur on COVID-era
quarters; B and C show real macro-trend signal but no AUC recovery. No two
variants agree on a single "favourable regime" cell; the only gate-clearing
bin (4H 2020-Q2) is favourable in C (the year-quarter lens) but is the
high-vol/low-vol-transition period in A and the d1_down period in B. **No
regime variable replicates the 4H 2020-Q2 cell's signal at a different lens** —
indirect evidence that the cell's signal is an artefact of the specific
year-quarter (COVID-era within-period auto-correlation), not a regime variable.

---

## 6. Gate-clearing cell — interpretation

Only one cell cleared: 4H Variant C `2020-Q2` at worst-fold AUC 0.5528, mean
fold AUC 0.6418, fold stdev 0.052. Per-fold val AUC sequence (fold 1 → 7):
0.639, 0.711, 0.604, 0.633, 0.712, 0.553 (worst), 0.641.

### Structural read

Q2 2020 (April-June 2020) is the immediate post-COVID-shock period. Vol was
mean-reverting from the March 2020 spike. The cross-variant overlap (§5) shows
2020-Q2 is a strong overlap of:
- d1_down (risk-off macro continuation from March crash) +41% excess
- vol_q1 (volatility falling from peak) +82% excess

A model fit on early Q2 (~April) bars learns the post-spike vol decline
pattern and predicts on late Q2 (~late May / June) bars in the same regime —
effectively in-distribution prediction within a 90-day window. This is
auto-correlation within a specific historical event, not a regime signal that
would replicate forward.

### Why this is not real regime conditioning

A genuine regime-conditional signal would (a) replicate across similar
regimes — not COVID-Q2, but every Q2-like "post-spike mean reversion" period —
and (b) appear under multiple lenses (vol regime, trend regime, calendar
regime). Neither holds. Variants A and B contain no bin within 0.06 of gate
at 4H. The single 4H Variant C cell that clears does not correspond to any
recurring market state — it corresponds to the specific 90-day epoch April-June
2020.

---

## 7. Implications — no robust regime structure

Treating the 4H 2020-Q2 cell as a noise pass (per §6 + §8), the overall
verdict is **no cell clears robustly**:

- **Interpretation 1 (regime-conditional) is NOT supported by the data**,
  despite the postmortem evidence at 1H/4H pointing toward it.
- **Interpretation 2 (structural non-stationarity beyond price-feature regime)
  becomes more likely.** The B1-B4 worst-fold AUC < 0.50 collapse appears to
  reflect not a regime variable the features failed to capture, but a
  fundamental limit of price-derived features on this label.
- **Interpretation 3 (feature set binding / weak features) is consistent with
  the data.** Diagnostic 3 (importance stability) ruled out the
  "model-grasping-at-noise" reading, but does not rule out the "model-finding-
  weak-but-real-signal-that-doesn't-generalise" reading.

Lω-v2 cross-TF expansion (if running in parallel) becomes the discriminator.
If Lω-v2 also produces no improvement, the conclusion is robust: price-derived
features at 1H/4H entry bars cannot extract clean forward paths above worst-fold
AUC ~0.50 under the B1-B4 clean_move label, regardless of which regime variable
is conditioned on.

**Pivot recommendation:** formal Lω closure on price-derived features; advance
Tier A structural-event signals (failed-breakout reversal, liquidity sweep +
reclaim, compression-then-expansion) as hand-defined signals under
L_ARC_PROTOCOL, NOT under another Lω feature-mining iteration.

---

## 8. Anti-evidence — why the single gate-clear is not deployable

The single gate-clearing cell (4H 2020-Q2 worst-fold AUC 0.5528) fails every
deployability check the dispatch identifies as anti-evidence triggers:

1. **Sub-10%-of-data size.** 2020-Q2 = 10,920 rows = 4.3% of 4H data. Even if
   the signal were real, a regime-conditional signal that fires on 4.3% of
   bars is not a viable foundation for a deployed system.
2. **Single historic concentration.** 2020-Q2 is exactly one quarter, in 2020
   (COVID era). The dispatch's anti-evidence framing: "Are gate-clearing bins
   concentrated in specific years/pairs?" — yes, maximally. The bin is by
   definition concentrated in one quarter.
3. **Below-margin pass.** Worst-fold AUC 0.5528 is 0.003 above the 0.55 gate
   threshold. Fold stdev within the bin is 0.052, so the 0.5528 value is
   ~0.06σ above gate — well within sampling noise. Three other Variant C
   bins (2020-Q4, 2025-Q1, 2025-Q3) come within 0.006 of gate; treating
   2020-Q2 as a "pass" and the others as "fails" reflects threshold artefact,
   not real signal discrimination.
4. **Different prediction problem.** Variant C bins solve intra-quarter
   prediction (train on early-quarter bars, validate on late-quarter bars
   within the same ~90 days). B1-B4's worst-fold AUC of 0.4733 at 4H solves
   cross-time prediction (train on all 2020-2024, validate on 2025). These
   numbers are not directly comparable; the Variant C result is more akin to
   measuring forward auto-correlation than measuring transferable predictive
   signal.
5. **No worst-fold AUC > 0.60 anywhere.** The dispatch's anti-evidence check
   "Did any variant produce worst-fold AUC > 0.60 in any bin?" — no. The
   highest worst-fold AUC across all 62 cells is the 4H 2020-Q2 cell at
   0.5528. There is no bin showing strong evidence (worst > 0.60) anywhere;
   the only positive result is a marginal one.
6. **Importance stability is consistent with weak-features reading.** PR #134
   Diagnostic 3 verified top-15 feature importance is stable across folds
   (Spearman ρ ≥ 0.96). The regime-conditional dispatch was designed to
   discriminate Int-1 from Int-3 ("weak features capturing a real-but-weak
   phenomenon"). Variant A's flat AUC distribution across vol quintiles is the
   signature expected if Int-3 holds: bins do not differ because there is no
   regime; the features pick up a weak universal signal that doesn't separate
   regime-by-regime.

---

## 9. Recommendation

**Formally close Lω regime-conditional. Do not proceed to B5-B6 on any cell
from this dispatch.** The single gate-clearing cell is unusable (§8). The
weight of evidence does not support Interpretation 1.

### If Lω-v2 cross-TF expansion (if dispatched in parallel) also produces no
recovery

**Pivot to Tier A structural-event signals.** Specifically:
- Failed-breakout reversal (breakout-then-fail patterns at known liquidity
  levels)
- Liquidity sweep + reclaim (stop-run followed by immediate reversal)
- Compression-then-expansion (low-vol consolidation followed by directional
  break)

These should be nominated as **hand-defined signals under L_ARC_PROTOCOL v2.x**
(Step 1 plumbing → Step 6 WFO gate), NOT under another Lω feature-mining
iteration. Lω's negative result is the floor evidence: price-derived features
at 1H/4H bar-resolution have a worst-fold AUC ceiling of approximately 0.50
under the clean_move label.

### Calibration implication for L_ARC_PROTOCOL v2.x

The Lω regime-conditional null adds to the v2.1 cross-arc backlog as
additional evidence for **modest extractability ceilings on price-feature
signals**. v2.2 should not assume that hand-defined structural-event signals
will clear AUC > 0.60 at Step 4; the Lω 0.50 ceiling suggests structural-event
signals should be evaluated against more permissive Step 4 thresholds (or
graded on effect size in addition to AUC).

### If chat-track wants to explore further before pivot

The two anti-evidence paths most likely to invert this null result, in order
of compute cost:

1. **Relaxed label test (low compute).** Drop `mono_pre_peak` from the
   clean_move definition (i.e. label = `mfe_max_R ≥ 1.5` only). PR #134
   anti-evidence §5 flagged the possibility that 2025 paths have high MFE but
   more intra-window pullback — testing this relaxation discriminates
   "label-too-tight" from "feature-too-weak." Re-runs B1-B4 + this dispatch
   with the new label; ~30-45 min compute total.
2. **Cross-broker spot-check (medium compute).** PR #134 anti-evidence §1
   flagged broker-side data change as an unmodeled confound. Compare 4H ATR
   distributions on a second broker (FTMO) over the same 2020-2026 window. If
   2025 vol decline reproduces, broker-side change is ruled out. If it
   doesn't, the Lω null could be a broker-data artefact rather than a market
   reality.

Both are exploratory, not part of this dispatch. The dispatch's primary
verdict stands: no regime variable tested recovers worst-fold AUC ≥ 0.55 in
any bin with deployable size and stability.

---

## Validation checklist

- [x] 1H + 4H processed; D1 skipped per scope (postmortem-inconclusive).
- [x] Three binning variants per timeframe (A vol percentile, B D1 trend sign,
  C year-quarter).
- [x] Sample-size feasibility documented (§3); zero bins marked
  `insufficient_data` anywhere.
- [x] Per-bin per-fold AUC saved in `timeframe_<tf>/variant_<x>/per_bin_per_fold_auc.csv`.
  Fold dates differ from B1-B4 by design choice (TimeSeriesSplit on bin
  subset; documented in script header).
- [x] **Determinism PASS** on 4H: all 6 output files (3 variants × per-bin-per-fold
  + variant_summary) byte-identical across two-run (`diff -q` returned no
  output). 1H not re-run for cost; same pipeline + RNG (`random_state=42`)
  identical to 4H so determinism is expected to hold.
- [x] Summary contains all 9 required sections (§§1-9).
- [x] §8 anti-evidence explicit; ranks the single gate-clearing cell against
  every dispatch-defined unusability criterion.
- [ ] CI clean — pending PR open + CI run.

---

## Reproducing

```
# Regenerate B1-B4 labels.csv + features.csv if missing (gitignored).
py scripts/lomega/lomega_b1_b4.py --tf 1h    # ~10-15 min wall-clock
py scripts/lomega/lomega_b1_b4.py --tf 4h    # ~5 min wall-clock

# Run regime-conditional dispatch.
py scripts/lomega/lomega_regime_conditional.py --tf all    # ~5-10 min total

# Cross-variant overlap analysis (sidecar diagnostic for §5).
py scripts/lomega/lomega_regime_overlap.py
```

All outputs land under `results/lomega/regime_conditional/`. Inputs are the
(gitignored) `labels.csv` + `features.csv` from PR #133's
`results/lomega/b1_b4_discovery/timeframe_<tf>/`.
