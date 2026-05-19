# Arc 5 — Result

## Status

| Field | Value |
|---|---|
| Opened | 2026-05-17 |
| Closed | 2026-05-17 |
| Final step | 6 (WFO truth under PR 2 + new spreads) |
| Active protocol | L_ARC_PROTOCOL v2.1.1 |
| Disposition | **SHELVED — Step 6 FAIL** (no ship candidate; signal has admit-set edge but Pipeline D1 selection bias on rejected pool kills full-strategy expectancy) |
| Reopenable | Yes — on Pipeline E re-evaluation with richer feature set, or full Steps 2-4 retrain under new spreads, or alternative pipeline shape that doesn't bear the rejected-pool cost |

---

## Headline result

**No ship candidate.** All three strategy candidates (cluster 1 alone, cluster 3 alone, tiered ensemble) close Step 6 with negative worst-fold ROI at every risk-per-trade level. Risk scaling does not fix negative expectancy.

| Strategy | Best risk | §10 | Worst-fold ROI | Mean fold ROI | Worst-fold DD (compounded) | Total compounded ROI | Total compounded DD |
|---|---|---|---|---|---|---|---|
| Cluster 1 (3×ATR, §11 row 2) | 0.10% | **FAIL** | −5.74% | −5.61% | 6.31% | −26.05% | 27.56% |
| Cluster 3 (2×ATR, §11 row 2) | 0.10% | **FAIL** | −7.95% | −7.61% | 8.44% | −33.80% | 34.41% |
| Ensemble (Tier A+B+C) | 0.10% | **FAIL** | −10.60% | −10.79% | 11.40% | −44.75% | 45.63% |

---

## Executive summary

Arc 5 tested registry Entry 5 (`mtf_alignment.2_down_mixed.kijun`) at time_exit_bars=120 under L_ARC_PROTOCOL v2.1.1 with Pipeline D1 mechanics. The arc proceeded cleanly through Steps 1-5 under PR 1 (close-at-market for rejected trades) + uniform 0.1-pip spread floor, producing apparently strong results: 7/7 positive folds on both clusters, worst-fold annualised ROI +36.5% (c1) / +8.8% (c3), §9 gates clean.

Two corrections changed the picture entirely:

1. **Spread realism (F10):** HistData active-session validation revealed the 0.1-pip floor was materially below real spreads (real p50: 0.3-4.8 pip across 28 pairs). Updated to per-pair active-session floors. Step 5 re-run under new spreads: c1 §9 DD ratio fails (2.34 > 2.0), c3 still passes but at thin margin.

2. **PR 2 + full-pool measurement (F12):** PR 2 (per-archetype §11 row 2 exit policy — MFE-lock at 1R + trail 0.75R) was shipped. Step 6 dispatched under PR 2 mechanics with full WFO equity-curve accounting. **PR 2 fully recovered c1's §9 DD ratio (2.34 → 1.17) — confirming the row 2 mechanism works as designed.** But Step 6's full-pool reckoning revealed that Pipeline D1's 78% rejected pool — which closes at market on bar 2 — has mean R **−0.46** (vs +0.025 unconditional bar-2 R). The classifier's rejection signal is itself a prediction of further adverse drift. Admit-set R remained positive (+0.14 c1, +0.21 c3 per fold) but the full-strategy equity curve was dominated by rejected-pool drag.

**The signal has real path-shape edge on its admits. The pipeline cannot extract it.**

---

## Signal definition

| Field | Value |
|---|---|
| Trial ID | `TRIAL__mtf_alignment__2_down_mixed__kijun__h_024` |
| Registry rank | Entry 5 |
| Family | `mtf_alignment` |
| Base condition | Extremes both down, 4H_mr up — mixed-down state |
| Direction sub-spec | `kijun` — Kijun-sign trend definition |
| Signal TF | 1H |
| Time exit (chat override) | 120 bars (registry h=24 overridden — F1) |
| L4 pooled DSR | 0.999769 (CI [0.932, 1]) |
| L4 raw Sharpe | 0.0321 (CI [0.0220, 0.0416]) |
| L4 per-pair Sharpe | median 0.051, p25 −0.015, p75 0.123 |

Entry trigger shares the byte-identical signal definition with registry Entry 2 (Arc 2 redo). Time exit horizon was the only registry difference; setting h=120 made Arc 5's plumbing pool sha256-identical to Arc 2 redo2's (F2 confirmed at Step 1).

---

## The arc journey

### Step 1 — Plumbing (PASS, sha256-identical to Arc 2 redo2)

12,262 trades across 28 pairs, 2010-02-10 → 2025-12-19. Two-run determinism byte-identical. Lookahead CI clean. Pool sha256-identical to Arc 2 redo2 baseline — F2 confirmed Arc 5 ≡ Arc 2 redo at trade-pool level under v2.1.1 schema.

### Step 2 — Clustering (PASS, K=4 silhouette 0.4834)

K-tie tolerance didn't bind (next-K gap 0.0145 > 0.01). Clusters byte-identical to Arc 2 redo2.

| Cluster | n | Size frac | Centroid (mono/peaks/pullback/ttp) | §11 status |
|---|---|---|---|---|
| 0 | 4,270 | 34.82% | 0.548 / 4.73 / 0.130 / 0.341 | unclassified — ttp gap |
| 1 | 2,285 | 18.63% | 0.538 / 20.07 / 0.340 / 0.774 | Stepwise climber (row 2 clean match) |
| 2 | 4,001 | 32.63% | 0.015 / 0.53 / 0.011 / 0.043 | Early-peak hold (provisional) |
| 3 | 1,706 | 13.91% | 0.512 / 7.46 / 0.700 / 0.591 | unclassified noisy Stepwise (F7 routing to row 2) |

### Step 3 — Capturability (PASS, 2 of 4 clusters)

| Cluster | Disposition | Selected SL | Composite | Failure mechanism |
|---|---|---|---|---|
| 0 | FAIL | none | 0.180 best | reach_1R never crosses 0.70 — "Monotone ascent in shallow water" |
| 1 | **PASS** | 3×ATR | 0.593 | clean rescue via pre-peak shift + SL sweep |
| 2 | FAIL | none | −0.471 best | mono 0.022, reach_1R 0.006 — no_magnitude across all SLs |
| 3 | **PASS** | 2×ATR | 0.370 | thin margins on every floor; routed to row 2 per F7 |

C1 was the v2.1.1 anchor-rescue target: Arc 2 redo (v2.0) killed the structural equivalent at mono 0.541 + frac_wrong_way 0.305 + shape_tag fail. v2.1.1's pre-peak metrics + SL sweep + capturability composite rescued it. First non-self-test pass of v2.1.1's design intent.

### Step 4 — Extractability (PASS via D1; E failed)

Both clusters route through Pipeline D1 at t=1. Pipeline E failed for both (best AUC 0.566 after full Step A/B/C cascade including 30/30 stacking budget exhausted). Feature-set bound (RF−LR gap ≤ 0.014 across all combinations).

| Cluster | D1 RF AUC | D1 exclusion | Anchor parity |
|---|---|---|---|
| 1 | 0.636 at t=1 | 0% | KH-24 anchor 0.638 — dead-even |
| 3 | 0.640 at t=1 | 0% | slightly above anchor |

Path-so-far features (close_r, velocity, MAE, MFE at t=1) dominated importance — top 4 carried ~54% of feature importance for both clusters. The first held bar reveals path quality more than 30 entry-time features combined.

### Step 4b — F9 threshold extension (under-specified protocol)

§8's threshold grid `{0.40, 0.50, 0.60, 0.70}` produced fallback threshold 0.40 with 1-4 TPs admitted (recall 0.0004). Within-arc decision F9: extend grid to `{0.10, ..., 0.40}` per the under-specified D1 threshold rule in §8. Selected thresholds: 0.20 (c1, recall 0.642, precision 0.245), 0.15 (c3, recall 0.607, precision 0.191).

### Step 5 — Cross-fold stability under old spreads (PASS, both clusters)

Strict per-fold classifier retraining (no lookahead in classifier inputs; cluster labels are full-pool — Open-10 leakage unresolved). 7/7 positive folds, both clusters. Headlines: c1 worst-fold +36.46%, c3 worst-fold +8.80%. C3 fold 4 was the regime-stress weakness (mean R +0.05, t-stat +0.40 — barely positive).

### Step 5.5 — Per-fold clustering audit (Open-10 leakage test)

C3: mean Jaccard 0.93, audit-c3 IS clusters reproduce baseline c3 centroids cleanly. **C3 is leak-free.**

C1: Audit invalid. Match formula's `min(1, (30-peaks)/25)` saturates at peaks ≤ 5, causing audit-c1 to consistently pick baseline cluster 0 (peaks 4.73, near-Mono-ascent) instead of baseline cluster 1 (peaks 20.07, true Stepwise). Jaccard 0.0 on every fold — different cluster, not different version of same cluster. Step 5.5b (nearest-centroid-to-baseline rematch) was queued but became moot when c1 died at Step 6.

### Spread validation (cross-arc)

External validation against HistData ASCII tick data (2024-01 → 2025-12) revealed L arc 0.1-pip uniform floor materially understated real active-session spreads (real p50: 0.3-4.8 pip on 1H). Total under-modeled equity cost on the 2024-01 → 2026-01 evaluation window: 52.84% (~26% annualised drag). Floor file updated to per-pair active-session p50 values.

### Step 5 re-run under new spreads (skip-to-5 approximation)

Step 1 plumbing re-run with new spreads (12,348 trades, +86 from concurrent-per-pair guard cascade, 99.04% trade-match to baseline). Baseline classifiers + F9 thresholds applied to new trade pool (admit-set Jaccard 0.91/0.90 — off-distribution effect bounded).

| Cluster | Worst-fold ROI | DD ratio | §9 |
|---|---|---|---|
| 1 | +32.62% (f3) | 2.34 (FAIL) | **FAIL on DD ratio** |
| 3 | +12.49% (f4) | 1.91 (PASS) | PASS |

C1 dies on §9 DD ratio. C3 survives but ROI/DD ratio 0.58 < 5/8 required to satisfy both PASS-DEPLOYABLE gates simultaneously — math says no risk level can clear PASS-DEPLOYABLE.

### Step 6 — WFO truth under PR 2 (FAIL, all three strategies)

PR 2 (per-archetype §11 row 2 exit policy: MFE-lock at 1R + trail 0.75R) shipped. Full WFO equity-curve accounting on the 12,348-trade pool, three strategy candidates (c1, c3, ensemble), risk-per-trade sweep {0.50% → 0.10%}.

**Key finding 1 (positive):** PR 2 fully recovers c1's §9 DD ratio. 2.34 → 1.17. The MFE-lock + trail mechanism rescues the fold-3 regime asymmetry exactly as predicted. C3 DD ratio also tightens (1.91 → 1.18). Admit-set mean R per fold remains positive across all 7 folds for both clusters.

**Key finding 2 (fatal):** Full-strategy equity curve is dominated by rejected-pool cost. Pipeline D1 enters every signal and bears three cohorts:

| Cohort | Count | % of pool | Mean R |
|---|---|---|---|
| Early-exit (SL hit before bar 2) | 976 | 7.9% | −0.74 to −1.11 |
| Admit set (classifier P ≥ threshold) | 1,631-1,810 | 13.2-14.7% | +0.14 to +0.20 |
| Reject + close at bar 2 | ~9,700 | ~78% | **−0.46** |

The rejected pool's −0.46R mean R (vs +0.025 unconditional bar-2 R) is **adverse-selected** — the classifier rejects trades whose path-so-far at bar 1 looks unfavourable, and that adverse signal correlates strongly with continued drift to bar 2. Classifier rejection is itself a prediction of further loss.

Tier B (c3-only-positive admits, mean R +0.015, worst-fold −0.18) is dead weight in the ensemble — adds marginal admits but cannot remove Tier C's −0.46R drag. Ensemble worst-fold ROI −10.60% is worse than c1-alone.

---

## Why it failed — economic mechanism

Pipeline D1 PR 2 + new spreads bears three structural costs that combine to negative expectancy:

1. **Unavoidable bar-0/1 SL hits (7.9% of pool).** Some signals hit SL before the classifier can act at t=1. Includes spread-blowout fills on news days (Jan 2015 SNB, etc.). Tail extends to −6.87R from 440-pip spread events.

2. **Adverse-selected rejection pool (78% of pool, −0.46R mean).** Classifier rejection at bar 1 happens when path-so-far looks adverse. Trades closed at bar 2 carry that adverse momentum. Real cost per rejected trade: spread on entry + spread on exit + 1-bar adverse drift.

3. **Admit-set R compressed under PR 2** (admit mean R +0.14 to +0.21 per fold, down from PR 1's +0.20 to +0.64). PR 2's trail captures wins earlier than PR 1's hold-to-time-exit, which improves DD profile but compresses winning trade R. Still positive — the structural edge survives PR 2 mechanically — but smaller than PR 1's measurement suggested.

The arithmetic: 0.13 × +0.17 (admit contribution) + 0.78 × (−0.46) (rejection drag) + 0.08 × (−1.0) (early SL) = +0.02 + (−0.36) + (−0.08) = **−0.42R per signal**, before risk scaling. No risk level fixes this.

---

## The methodology lesson — full-pool reckoning vs admit-only metrics

**This is the most important cross-arc finding from Arc 5.**

Step 5 §9 sign-consistency and ROI were computed on the **admit set only** — the 13-15% of trades the classifier kept. The 78% rejected pool's R contribution was implicitly assumed to be ≈0 (close at market with small spread cost) and not aggregated into the per-fold ROI metric. Under that framing, both clusters passed §9 cleanly.

Step 6 measured the **full WFO equity curve** including every signal trade. The rejected pool's −0.46R mean R surfaced as the dominant cost. The negative expectancy was always there; the §9 metric just didn't measure it.

**The protocol bug:** §9 sign-consistency and §10 ROI gates need to be computed on full-pool strategy R (admits with their PR-2 outcomes + rejects with their bar-2 outcomes), not admit-only R. Under the corrected framing, Arc 5 would have failed §9 at fold 4 (admit mean barely positive, full-pool mean negative). Step 6 would have been redundant — Step 5 would have caught it.

This isn't unique to Arc 5. Any Pipeline D1 arc with high rejection rate and adverse-selected rejection (which is the natural state of a competent classifier — competent rejection should correlate with bad outcomes by construction) will be vulnerable. Protocol §9 wording needs explicit clarification.

Logged in PROTOCOL_IMPROVEMENT_BACKLOG.md as P-item.

---

## Findings preserved as positive evidence

Despite the Step 6 FAIL, Arc 5 produced meaningful positive findings:

1. **v2.1.1's anchor-rescue mechanism is empirically validated.** C1 (Stepwise climber) cleared §2 conjunctively under combined pre-peak metrics + SL sweep + capturability composite. First non-self-test pass since v2.1.1 amendment. The protocol changes work as designed.

2. **§11 row 2 exit policy (MFE-lock + trail) rescues regime asymmetry.** C1's §9 DD ratio recovered 2.34 → 1.17 under PR 2. Trail closes 24% of c1 admits and 44% of c3 admits — the mechanism is doing real work. PR 2 is validated as a deployment-relevant capability.

3. **D1 path-so-far features at t=1 carry real predictive signal.** AUC 0.636-0.640, anchor-parity to KH-24's 0.638. Top 4 features (close_r, velocity, MAE, MFE at t=1) carry ~54% of importance. The first held bar legitimately reveals path quality.

4. **Open-10 leakage test for c3 confirmed clean.** Mean Jaccard 0.93 across folds — c3's structural identity exists independent of full-pool clustering. Full-pool clustering is empirically safe for this signal at this cohort.

5. **Spread validation methodology is now in repo.** HistData active-session p50 floors validated, applied, documented. Future arcs inherit the calibration.

6. **F2 / F5 confirmed empirically.** Arc 5 ≡ Arc 2 redo at trade-pool level when registry h is overridden. Registry h is descriptive of L-atlas, not a Step 1 plumbing constraint. Important methodological clarification.

---

## Cross-arc backlog items logged

Added to `PROTOCOL_IMPROVEMENT_BACKLOG.md`:

| ID | Title | Priority |
|---|---|---|
| P-§9-FRAMING | §9 sign-consistency and DD-ratio must be measured on full-pool strategy R (admits with their pipeline outcomes + rejects with their pipeline outcomes), not admit-only R. Protocol wording is currently ambiguous and reads as admit-only in practice. | P0 |
| P-D1-VIABILITY | Pipeline D1 viability check: signals with > X% bar-0/1 SL-hit rate at deployed baseline SL should be flagged for D1 unsuitability at Step 4. Arc 5 had 7.9%. Suggested threshold: 5%. | P0 |
| P-D1-REJECT-BIAS | Document rejected-pool selection bias: classifier rejection at bar t is itself a prediction signal correlated with continued adverse drift. Mean R of rejected pool ≠ unconditional bar-t baseline. Add to §3 Pipeline D1 description with calibration data from Arc 5 (−0.46R rejected vs +0.025R unconditional). | P0 |
| P-F9-RESELECT | F9 threshold selection should use the metric that gates ship decision (worst-fold compounded ROI subject to DD ceiling), not an intermediate measurement (admit-set precision/recall). Currently Step 4b selects on admit-only proxies that don't capture full-strategy cost. | P1 |
| P-CLUSTERING-LEAKAGE | Open-10 leakage status: c3 confirmed clean (Arc 5 Step 5.5 audit); c1 unresolved (audit invalid due to match formula saturation). Per-fold clustering should become default rather than full-pool, with full-pool retained as comparison only. | P1 |
| P-SPREAD-FLOOR | Spread floor file docstring drift — claims "applies only when raw spread is zero" but new p50 calibration applies to 58.6% of execution-bar entries. Update docstring + governance. | P2 |
| P-§11-MATCH-FORMULA | §11 row 2 pattern matching formula `min(1, (30-peaks)/25)` saturates at peaks ≤ 5, conflating Monotone-ascent (row 1, peaks ≤ 4) and Stepwise-climber (row 2, peaks 5-30) regions. Add a peaks lower-bound or a "preferred peaks range" specification for cleaner archetype matching. | P2 |
| P-OPEN-18-RECONCILE | STATUS.md / Open-18 priority queue had multiple inaccuracies discovered during Arc 5: `l_arc_4` Step 4/5 scaffolding existed undocumented; Arc 2 redo2 (v2.1.1 schema fork) existed undocumented; Open-18 KH-24 anchor replay scaffolding existed undocumented. Reconcile STATUS with actual repo state. | P2 |

---

## Conditions for reopening

Arc 5 is SHELVED not eliminated. Reopen conditions:

1. **Pipeline E re-evaluation with richer feature set.** Current E failure (AUC 0.566) is feature-set bound (RF−LR gap ≤ 0.014 — model isn't the constraint). Adding cross-pair correlations, regime-switch detectors, longer historical context, or alternative MTF representations could lift E above the 0.65 gate. If E clears, the rejected-pool cost vanishes — E filters at entry, not at bar t=1.

2. **Alternative pipeline shape.** Pipeline D1's structural cost is bearing the rejected pool. A hybrid pipeline (E pre-filter at entry to ~50% of pool, then D1 at t=1 on the survivors) would shrink rejected-pool exposure while preserving D1's predictive value. Not currently in protocol; would be a new pipeline definition.

3. **Full Steps 2-4 retrain under new spreads.** Current numbers use baseline classifiers (trained on old-spread features) applied to new-spread OOS. The off-distribution effect (Jaccard 0.91/0.90) is bounded but real. A full retrain might shift admit sets, F9 thresholds, and per-cluster economics enough to produce a different Step 6 outcome — though the rejected-pool selection bias is structural and unlikely to disappear.

4. **§9 framing fix (P-§9-FRAMING) lands and Arc 5 re-evaluated under corrected metric.** If §9 is computed on full-pool R, Arc 5's Step 5 would have caught the failure earlier. Re-running Step 5 with corrected §9 wouldn't change the Step 6 outcome but would validate the protocol change.

---

## Files and artefacts

| Artefact | Location |
|---|---|
| Step 1 plumbing (old spreads) | `results/l_arc_5/step1/` |
| Step 2 clustering | `results/l_arc_5/step2/` |
| Step 3 capturability | `results/l_arc_5/step3/` |
| Step 4 extractability | `results/l_arc_5/step4/` |
| Step 4b threshold re-sweep | `results/l_arc_5/step4b/` |
| Step 5 stability (old spreads) | `results/l_arc_5/step5/` |
| Step 5.5 clustering audit | `results/l_arc_5/step5_5/` |
| Step 1 re-run (new spreads) | `results/l_arc_5/step1_spread_v2/` |
| Step 5 re-run (new spreads) | `results/l_arc_5/step5_spread_v2/` |
| Step 6 + PR 2 + ensemble | `results/l_arc_5/step6_pr2/` |
| Scripts | `scripts/arc_5/` |
| Configs | `configs/arc_5/`, `configs/wfo_l_arc_5.yaml` |

Live working doc (`ARC_5_LIVE.md`) archived; this result doc supersedes.

---

## Flags carried forward (final state)

| Flag | Status | Resolution |
|---|---|---|
| F1 — registry horizon override 24→120 | RESOLVED | Time exit set to 120 throughout. Plumbing pool sha256-identical to Arc 2 redo2. |
| F2 — entry trigger overlap with Arc 2 | CONFIRMED | Arc 5 ≡ Arc 2 redo at trade-pool level. Logged for cross-arc convention. |
| F3 — overlap with Open-18 Arc 2 redo c2 replay | RESOLVED | Arc 5 ran fresh; Open-18 replay redundant. |
| F4 — kijun direction interpretation | RESOLVED | Applied per Arc 2 redo convention. |
| F5 — schema fork (Arc 2 redo vs Arc 2 redo2) | DOCUMENTED | Arc 5 reproduces Arc 2 redo2 (is_held schema). |
| F6 — Pipeline D1 t-sweep default | RESOLVED | Default {1, 2, 3, 4, 5, 10} preserved. |
| F7 — cluster 3 routing to row 2 | TESTED | c3 routed to row 2 under PR 2; produced positive admit-mean R but full-strategy failed. |
| F8 — undocumented `l_arc_4` + Open-18 scaffolding | LOGGED to P-OPEN-18-RECONCILE | STATUS reconciliation pending. |
| F9 — D1 threshold grid extension | LOGGED to P-F9-RESELECT | Within-arc decision held; cross-arc fix pending. |
| F10 — spread cost-correction overlay → full Step 1 re-run | RESOLVED | Step 1 re-run executed; new-spread numbers superseded cost-overlay. |
| F11 — risk-scaling deployment analysis | RESOLVED | Linear scaling analysis confirmed by Step 6 risk sweep within bounded compounding nonlinearity. |
| F12 — PR 2 + ensemble parallel evaluation | RESOLVED | All three strategies dispositioned in Step 6. |

---

## Acknowledgments and final notes

Arc 5 ran clean methodologically through every step with appropriate flag-and-decision discipline. The Step 6 FAIL is not a methodology failure — the protocol caught the negative expectancy at the appropriate gate (full WFO truth) before any ship decision. The arc closure is the protocol working correctly.

The §9 framing issue (P-§9-FRAMING) is the most important lesson and would have caught Arc 5 earlier with less compute. It's a real protocol bug, not specific to this arc, and applies to every Pipeline D1 arc that follows. Priority P0.

The signal itself has admit-set edge — it's a real path-shape-recognisable pattern. Pipeline D1 isn't the right consumer for this signal class. If a pipeline emerges that doesn't bear the rejected-pool cost, this signal becomes a reopen candidate immediately.

KH-24 is unaffected by Arc 5's findings — KH-24 uses its own price-action exit logic, not Pipeline D1. The spread validation cross-arc finding does apply to KH-24's accounting (modest 1.51% under-modeled cost on the validation window) and should be revisited at KH-24's next anchor-refresh.

**Arc 5 closes SHELVED at Step 6 FAIL. Backlog populated. Next: address P0 items, then signal exploration outside the registry's exhausted h≥120 set.**
