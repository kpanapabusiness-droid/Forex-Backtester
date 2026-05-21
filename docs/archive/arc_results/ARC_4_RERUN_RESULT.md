# ARC_4_RERUN_RESULT — 2026-05-18

> Re-run of Arc 4 under corrected per-pair p50 spread floors (HistData 2024-2025).
> Supersedes the prior `ARC_4_RESULT.md` CLEAN-NULL verdict on transaction-cost framing.
> Verdict: **FAIL at Step 6 under full-pool deployment reckoning.**

---

## Status

**Final verdict: FAIL — pass-deployable and pass-viable both fail under §10.**

The signal class (`bar_range_top_decile`, long, 1H, 28 FX pairs) carries admit-pool edge of +0.125R per trade under cluster 1 (Stepwise climber) with Pipeline D1 at t=1 and §11 row 2 exit policy. This edge does not survive full-pool reckoning. The rejected and early-exit trade pools that any Pipeline D1 system mandatorily incurs in live deployment crush net-of-everything expectancy to −0.076R per signal, producing a 76.98% drawdown over the F2-F7 OOS window at 0.20% risk.

This is the second Pipeline D1 arc (Arc 4, Arc 5) to PASS §9 admit-only stability and FAIL §10 full-pool deployment. The pattern is structural to the protocol's current Pipeline D1 gate sequencing, not specific to either signal class.

---

## TL;DR

| Field | Value |
|---|---|
| Signal class | `bar_range_top_decile__neg__h_001` (top-decile 1H bar-range × bar close < open, long bias) |
| Pool size (under p50 floors) | 10,893 trades, 28 pairs, 2020-10-01 → 2026-01-31 |
| Surviving cluster | Cluster 1 — Stepwise climber, n=1,786 (16.4% of pool), R=3×ATR |
| Pipeline | D1 t=1, §11 row 2 exit policy (MFE-lock 1R, trail 0.75R) |
| Phase 5 §9 verdict | PASS (admit-only stability gates all clear) |
| Phase 6 §10 verdict | **FAIL** (full-pool deployment fails on every gate) |
| Phase 6 full-data ROI (0.20%, 4.5 years) | **−76.98%** |
| Phase 6 max DD (b, 0.20%) | **76.98%** |
| 5ers daily DD breach | 1 day at 5.12% (account-closure event) |
| 5ers all-time DD breach | yes, by ~67pp |

---

## What changed since prior closure

Prior `ARC_4_RESULT.md` (2026-05-17): closed CLEAN-NULL on retroactive HistData spread reconciliation showing F6 admit-only mean R flipped negative under audit-equivalent costs. The closure was triggered by cost-model error in the trade pool generation, not by structural signal-class failure.

The spread floor file was replaced 2026-05-17 with per-pair p50 values from HistData 2024-2025 audit (calibration-curated, supersedes uniform 0.1 pip). This re-run regenerated the full Arc 4 evaluation from Step 1 under the corrected cost model.

Key differences this run:
1. Trade pool regenerated at Step 1 with new spread floors (+129 trades vs prior, +1.2%)
2. All downstream phases re-derived from new pool
3. Step 6 actually executed (prior closure was retroactive at Step 5 without running Step 6)
4. Full-pool economic accounting under §10 (rejected + early-exit pools booked at true cost)
5. S/2 exit spread correction applied to admit pool

---

## Per-phase trail

| Phase | Result | Notes |
|---|---|---|
| 0 — Spread file verification | PASS | New floor body sha matches lock; KH-24 byte-identity via static analysis (KH-24 doesn't load floor file) |
| 1 — Step 1 trade pool re-generation | PASS, with +129 trade delta | Spread changes are path-dependent under exposure cap; protocol-level lesson logged |
| 2 — Step 2 cluster re-verification | PASS | Cluster 1 centroid essentially identical; clusters 0/2 swapped IDs (kmeans relabel) |
| 3 — Step 3 capturability + SL sweep | PASS | Cluster 1 SL=3 selected, composite +0.358; kill-point did not trigger |
| 4 — Step 4 D1 classifier | PASS | t=1 selected, refit AUC 0.6487-0.6779 across F2-F7, all clear §8 |
| 5 — Step 5 §9 stability | PASS | Sign consistency, size variance, DD ratio all clear under admit-only framing |
| 6 — Step 6 §10 deployment | **FAIL** | Full-pool ROI −77%, DD 77%, 5ers account-closure events |

---

## Step 6 headline numbers

### Per-fold at 0.20% risk

| fold | n_total | n_admit | fold ROI% | ann ROI% | DD% (b) | terminal eq |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 1,708 | 937 | −15.69 | −20.91 | 17.86 | 0.8431 |
| 3 | 1,613 | 972 | −15.97 | −21.21 | 18.71 | 0.8403 |
| 4 | 1,559 | 903 | −13.94 | −18.65 | 17.18 | 0.8606 |
| 5 | 1,472 | 870 | +1.92 | +2.56 | 9.18 | 1.0192 |
| **6** | **1,740** | **945** | **−40.43** | **−55.22** | **42.18** | **0.5957** |
| 7 | 1,382 | 795 | −12.34 | −16.39 | 16.81 | 0.8766 |

F5 is the only positive fold. F6 collapses (−55% ann ROI, 42% DD in a single fold).

### Full-data continuous F2→F7 (4.5 years)

| risk | ROI period | ROI ann | Max DD (b) | Max daily DD | days >5% daily |
|---|---:|---:|---:|---:|---:|
| 0.20% | −76.98% | −16.98% | 76.98% | 5.12% | 1 |
| 0.15% | −66.21% | −14.69% | 66.69% | 3.86% | 0 |
| 0.10% | −51.41% | −11.41% | 51.88% | 2.59% | 0 |

At protocol-locked 0.20% risk the strategy is structurally negative-expectancy with catastrophic drawdown. Lower risk levels reduce the magnitude proportionally but do not change the sign — the strategy is mean-negative under full-pool framing at every risk level.

### Trade-flow decomposition (F2-F7, 9,474 total signals)

| segment | n | % of signals | mean R | sum R | contribution at 0.20% |
|---|---:|---:|---:|---:|---:|
| Admit (classifier ≥ threshold) | 5,422 | 57.2% | +0.125 | +678 | +1.36% |
| Reject (bar-2 close-at-market) | 3,047 | 32.2% | −0.232 | −708 | −1.42% |
| Early-exit (pre-t SL hit before t=1) | 1,005 | 10.6% | −0.685 | −689 | −1.38% |
| **All signals** | **9,474** | **100%** | **−0.076** | **−719** | **−1.44%** |

The admit pool is positive-edge (+0.125R) but admits only 57% of signals. The other 43% — reject pool (32%) plus early-exit pool (11%) — are mandatory cost in live deployment and exceed the admit pool's edge.

### 5ers prop firm reality check at 0.20%

| constraint | threshold | measured | survives? |
|---|---|---:|:-:|
| Max DD ever | < 10% | 76.98% | ✗ |
| Daily DD ever | < 5% | 5.12% | ✗ |
| Days exceeding 5% daily DD | 0 | 1 | ✗ |

Account would have been closed on the day daily DD hit 5.12%, AND would have catastrophically breached the 10% max DD constraint multiple times throughout the simulation.

---

## Why it failed

**The classifier works. The architecture forces deployment to take signals the classifier can't help with.**

Three pools, three economic outcomes:

1. **Admit pool (+0.125R mean):** classifier admits at bar 1 close, archetype SL replaces pre-t SL, §11 row 2 exit runs. Real edge, real R.

2. **Reject pool (−0.232R mean):** classifier rejects at bar 1 close, exit at bar 2 open. Cost composes as ~−0.05R spread round-trip + ~−0.18R adverse selection drift (the classifier rejects trades whose bar 1 looked bad; those trades continue to look bad in bar 2). The bail-out is doing its job — exiting losers — but the exit price is already worse than entry by the time the rejection information is actionable.

3. **Early-exit pool (−0.685R mean):** pre-t SL (entry − 2×ATR) fires between bar 1 open and bar 1 close, before the classifier evaluates. Near-full-R loss per trade. The classifier cannot filter these trades; the bail-out doesn't apply. 11% of signal flow at this cost is equal in aggregate damage to the 32% rejection pool.

The arithmetic for Pipeline D1 to deploy is `(admit_rate × admit_mean) > (reject_rate × |reject_mean|) + (early_exit_rate × |early_exit_mean|)`. For Arc 4: `0.572 × 0.125 = 0.072` vs `0.322 × 0.232 + 0.106 × 0.685 = 0.075 + 0.073 = 0.148`. Costs are 2× the edge.

Compounded over 4.5 years at 0.20% risk: −77% terminal equity.

### Why F6 collapsed

F6 has 1,740 total signals — the largest signal count of any fold. With negative expectancy per signal, larger signal flow produces larger absolute loss. F6 admit-pool mean R was the weakest at +0.038, so the admit pool barely produced anything to offset the 1,740-signal stream's reject + early-exit drag. Compounding amplifies: each losing trade reduces equity, the next trade is sized on smaller equity, percentage losses compound. F6 ann ROI −55% / DD 42% / terminal equity 0.5957.

### Why the spread fix didn't save it

The original CLEAN-NULL closure attributed Arc 4 failure to under-modeled spreads (factor 3-48× per pair). The fix corrected that — admit-pool F6 mean R came back from −0.04 to +0.038. Phase 5 §9 PASSED on admit-only framing.

But the cost-model fix only affected admit-pool edge calculation. The reject and early-exit pools were always present, always paying real R, always invisible to §9's admit-only framing. The protocol asked the wrong question for five phases.

---

## Cross-arc structural finding

Arc 4 and Arc 5 both PASS §9 admit-only stability and FAIL §10 full-pool deployment. Both are Pipeline D1 systems. The failure mode shape is identical:

- Admit-pool mean R: positive but modest (Arc 4: +0.125, Arc 5: ~+0.10-0.20 across candidates)
- Reject-pool mean R: materially negative (Arc 4: −0.232, Arc 5: −0.46)
- Early-exit pool: negative, often the dominant drag

For Pipeline D1 with any honest classifier (AUC ~0.65-0.70), adverse selection on the reject pool plus mandatory loss on the early-exit pool swamps the admit pool's edge. This is structural to the architecture: classifiers can only act on information available at bar t, and the pre-t hold period costs real R on the trades the classifier would have rejected if it could.

**The protocol's §9 admit-only gates measure classifier quality, not deployment viability.** §10 is the gate that catches this, but by Step 6 substantial compute has been spent. Two arcs in a row have surfaced this. Three open protocol items spawned (see below).

---

## Open protocol items spawned

### Open-22 — Full-pool gate at §9 or earlier (HIGH)

§9 admit-only stability gates passed both Arc 4 and Arc 5. §10 full-pool deployment failed both. The protocol needs to surface full-pool economics earlier than Step 6. Candidates:
- Add a full-pool variant to §9 (admit-only PASS AND full-pool sign-consistency PASS required to advance)
- Add a Step 4 full-pool preview (admit/reject/early-exit decomposition with expectancy estimate)
- Restructure §9 + §10 sequencing so the deployment-blocking metric is the primary gate

Owner: cross-arc calibration session.
Surface arc: Arc 4 + Arc 5 closures.

### Open-23 — §8 Pipeline D1 cost-language correction (MEDIUM)

§8 currently describes Pipeline D1 reject pool as "near-break-even small loss after spread, given the short hold and pre-t SL." Empirical evidence (Arc 4, Arc 5):

- Reject pool mean R: −0.15 to −0.46R depending on classifier discrimination strength
- Early-exit pool (pre-t SL hits before t): −0.45 to −0.69R, 10-15% of signal flow

Both pools must be reported explicitly in any Pipeline D1 candidate's Step 4-6 economics. §8 wording overstates the bail-out cleanliness.

Owner: cross-arc calibration session.

### Open-24 — Early-exit pool architectural cost (MEDIUM)

Pre-t SL of 2×ATR uniformly fires on 10-15% of signals before the classifier evaluates. This is pure architectural cost — not classifier-induced, not policy-induced, just inherent to "hold to bar t before deciding." Two structural responses possible:

- Widen pre-t SL (reduces early-exit rate at the cost of larger loss per early-exit trade)
- Shorten t (less time for SL to fire, but smaller classifier feature set)
- Hybrid: archetype-specific pre-t SL calibrated to early-exit rate

Currently Pipeline D1 uses uniform pre-t SL = 2×ATR per §8. This is a pre-amendment default — empirical data may justify per-archetype calibration.

Owner: cross-arc calibration session.

---

## Carried items (from Phase 5)

- **Lesson — Spread-floor changes are not population-invariant under exposure caps.** Spread shifts entry/exit fill prices, which shifts when stops fire, which shifts when the concurrent-position cap releases, which shifts admission for subsequent signals. Future arcs that change spread file expect Step 1 trade pool drift of ±1-2%.

- **Lesson — Convention-clarity for prior-run reference.** The phase 5 prompt confused KH-24 closure numbers (1% risk, 4H, separate engine) with Arc 4 step5c output. Future arc re-runs should reference engine-equivalent prior outputs for comparison.

---

## Disposition

**Arc 4 is closed.** The signal class does not survive Step 6 deployment reckoning. The original CLEAN-NULL closure (cost-model) was correct in disposition but for incomplete reason. The corrected re-run reveals the deeper architectural failure mode.

**KH-24 live deployment** is unaffected by this finding. KH-24 uses a different pipeline shape (entry filter from v1.0 protocol, not Pipeline D1) and a different evaluation history (audit-reconciled WFO claim is pass-viable, separately governed).

**Pipeline E feasibility for cluster 1** could be revisited as a long-shot follow-up. Pipeline E does not incur reject-pool or early-exit-pool drag because rejected signals never enter at all. The full-pool economic failure mode disappears under E. The blocker is that Pipeline E AUC at Step 4 was 0.55 (gate 0.65) — entry-time features cannot predict cluster 1 membership for this signal class. Re-trying E would require either richer entry-time features or accepting a lower-AUC operating point with reduced admission, which is a methodology question, not a re-run of existing code.

**Next active work** is the cross-arc calibration session for Open-22/23/24, then Arc 6+ under the corrected floor and (pending calibration outcome) updated §9/§10 gating.

---

## Document control

| Field | Value |
|---|---|
| Re-run date | 2026-05-18 |
| Prior closure | `ARC_4_RESULT.md` (2026-05-17, CLEAN-NULL on transaction-cost truth) |
| Trigger | Spread floor file replacement (`configs/spread_floors_5ers.yaml` uniform 0.1 → per-pair p50) |
| Branch | `calibration/spread-floor-p50-2026-05-17` |
| Protocol version | L_ARC_PROTOCOL v2.1.2 |
| Verdict | FAIL Step 6 under §10 deployment reckoning |
| Supersedes prior closure | Disposition unchanged (Arc 4 closed), reason updated (architecture, not just cost model) |
| Live system impact | None — KH-24 unaffected |

## Artefact index

- `results/l_arc_4_rerun/phase_0_spread_verification.md`
- `results/l_arc_4_rerun/step1/step1_diagnostics.md` + `comparison_to_prior_pool.md`
- `results/l_arc_4_rerun/step2/step2_diagnostics.md` + `phase_2_additional_diagnostics.md`
- `results/l_arc_4_rerun/step3/step3_diagnostics.md` + `phase_3_additional_diagnostics.md`
- `results/l_arc_4_rerun/step4/step4_diagnostics.md`
- `results/l_arc_4_rerun/step5c/phase_4_additional_diagnostics.md`
- `results/l_arc_4_rerun/step5/phase_5_analysis/STEP_5_VERDICT.md`
- `results/l_arc_4_rerun/step6/STEP_6_VERDICT.md`
- `results/l_arc_4_rerun/step6/equity_curve.png`
- `scripts/l_arc_4/step6_wfo.py` (new Step 6 wrapper, adapted from Arc 5's pattern)

## Input sha256s

| File | sha256 |
|---|---|
| `configs/spread_floors_5ers.yaml` (p50, locked) | `8da7644b…` (matches LOCKED_BODY_SHA256) |
| `configs/l_arc_4_rerun.yaml` | `d01361c5…` |
| `results/l_arc_4_rerun/step1/trades_all.csv` | `609f99ed…` |
| `results/l_arc_4_rerun/step1/trades_paths.csv` | `de169567…` |
