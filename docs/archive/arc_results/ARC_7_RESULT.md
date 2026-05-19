# Arc 7 — Liquidity sweep + reclaim (long) — CLOSURE

## Verdict

**CLEAN-NULL at Step 4.** Signal class is capturable (Step 3 PASS: 3 V-shape recovery units survive §2 conjunctively) but not extractable (Step 4 FAIL: zero unit × pipeline pairs clear §8 AUC gate; best 0.536 vs 0.65 required). Arc closes; no deployment candidate produced.

## Status

- **Opened:** 2026-05-17
- **Closed:** 2026-05-17
- **Active protocol:** L_ARC_PROTOCOL v2.1.2
- **Calibration anchor:** KH-24 K=4 archetype 3 (preserved)
- **Branch:** `phase/l_arc_7`

## Signal under test

| Field | Value |
|---|---|
| Signal | Liquidity sweep + reclaim (long) |
| Source spec | `signal_spec_liquidity_sweep_reclaim_long_v0.1.md` (2026-05-17 draft) |
| Family | Level-defined reversal |
| Direction | Long only |
| Signal TF | 4H |
| Horizon | 240 bars |
| Companion arc | Arc 6 (5.4% legitimate co-fire, different reference windows) |

### Trigger

1. `swing_low_N = min(low[t-N..t-1])`, N=20
2. `low[t] < swing_low_N`
3. `close[t] > swing_low_N`
4. `swing_low_N − low[t] >= 0.25 × ATR(14)[t]`
5. `close[t] > open[t]`
6. `(close[t] − swing_low_N) / (swing_low_N − low[t]) >= 0.5`
7. ≥ 20 bars since last signal on pair

## Step results

| Step | Gate | Result | Commit |
|---|---|---|---|
| 1 | Plumbing | PASS | `59de33a` |
| 2 | Path-shape clustering | PASS | `28111e4` |
| 3 | Capturability | PASS | `fa07c4a` |
| 4 | Extractability | **FAIL — CLEAN-NULL** | `4e6f1a6` |
| 5 | Cross-fold stability | not reached | — |
| 6 | WFO + disposition | not reached | — |

## Step 1 — Plumbing (PASS)

Pool 1,288 trades / 28 pairs; smallest per-pair n=32. Bars-held p50 24, p95 240; cap-binding 17% (under §5 auto-extend threshold). Reclaim-strength pre-filter unimodal right-skewed (Hartigan dip p=0.99), mass below 0.5 = 17.4%; 0.5 default removes the left tail cleanly. KH-24 co-fire 0/1,288 (clean independence from live system). Spread p95 2.4 pips, zero weekend tail. Determinism byte-identical; lookahead spot-check 5/5 reproduce.

**Arc 6 overlap finding** — halt rule retired. 69 overlaps are legitimate co-occurrence (Arc 6 uses `min(low[t-N-M..t-M-1])`, Arc 7 uses literal `min(low[t-N..t-1])` — different reference windows). All 69 verified: `t_star_bar_ts < signal_bar_ts` and Arc 6 `swing_low_N ≠ Arc 7 swing_low_used`. 5.4% overlap = portfolio-composition note (Open-05).

## Step 2 — Path-shape clustering (PASS)

K=4 chosen (silhouette 0.4263). All 5 K satisfy §6 gate. Tie tolerance held K=4 over K=6 (gap 0.0021 — the exact Open-12 closure case). No degenerate features (max modal-bin mass 41.8%, under 80%).

| cid | n | size_frac | centroid (mono / peaks / pullback / ttp_rel) | tentative label |
|---|---|---|---|---|
| 0 | 385 | 29.9% | 0.555 / 4.44 / 0.158 / 0.298 | Early-peak hold OR Peak-and-collapse |
| 1 | 185 | 14.4% | 0.536 / 33.51 / 0.567 / 0.730 | V-shape recovery |
| 2 | 353 | 27.4% | 0.014 / 0.59 / 0.009 / 0.082 | Early-peak hold OR Peak-and-collapse |
| 3 | 365 | 28.3% | 0.498 / 9.84 / 0.762 / 0.567 | V-shape recovery |

c1 is a Stepwise-climber near-miss: mono ≥ 0.50 ✓, peaks 5-50 ✓ (under v2.1.2 extended ceiling), ttp_rel ≥ 0.50 ✓, pullback 0.567 vs 0.5 ceiling ✗ (over by 13.4%). Centroid-pattern match fails on one feature.

## Step 3 — Capturability (PASS)

3 units survive §2 conjunctively; all V-shape recovery family.

### Surviving units

| unit | n | size_frac | sel SL | R | composite | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag |
|---|---|---|---|---|---|---|---|---|---|---|
| c1 | 185 | 0.144 | 4×ATR | 4 | 0.617 | 0.567 | 1.000 | 0.000 | 3.45R | unclassified |
| c3 | 365 | 0.283 | 2×ATR | 2 | 0.413 | 0.560 | 0.808 | 0.005 | 2.12R | unclassified |
| agg_c1_c3 | 550 | 0.427 | 4×ATR | 4 | 0.378 | 0.557 | 0.771 | 0.000 | 1.98R | unclassified |

### Killed at Step 3

- **c0:** mono_pp 0.619 ✓ (cleanest in arc) but fwd_mfe_p50 0.97R at every SL — wider SL inflates R denominator faster than absolute MFE. Structurally capacity-limited (early-peakers don't run far). No recoverable pipeline.
- **c2:** mono 0.019 — dead-trade cohort, as predicted.
- **agg_c0_c2:** c2 dilutes c0's clean mono (agg mono 0.332). **Open-14 closure case empirically validated.**

### Other findings

- bimodal_separated test: 0/6 fire. §11 row 7 split-exit not a parallel route.
- All survivors `shape_tag = unclassified` — v2.1.2's relaxed `≠ scattered` floor was load-bearing. **Without v2.1.2, Arc 7 dies at Step 3 instead of Step 4.**
- c1's archetype label = `tentative_V-shape recovery (forward-geometry weak)` — peak_bars≥5 frac 1.000, peak_pos ∈ [0.4, 0.8] frac 0.459 (one of two V-shape conditions met). Effectively a Stepwise climber labelled V-shape because §11 pullback ceiling rejected the cleaner match.

## Step 4 — Extractability (FAIL, arc closure)

| unit | pipeline | n | n_feat | mean AUC | std AUC | gate | margin |
|---|---|---|---|---|---|---|---|
| c1 | E | 185 | 22 | 0.484 | 0.127 | 0.65 | −0.166 |
| c1 | D1 | 185 | 11 | 0.420 | 0.130 | 0.60 | −0.180 |
| c3 | E | 365 | 22 | 0.512 | 0.094 | 0.65 | −0.138 |
| c3 | D1 | 365 | 11 | 0.518 | 0.095 | 0.60 | −0.082 |
| agg_c1_c3 | E | 550 | 22 | 0.536 | 0.029 | 0.65 | −0.114 |
| agg_c1_c3 | D1 | 550 | 11 | 0.496 | 0.065 | 0.60 | −0.104 |

D1 lag audit: 15/15 spot-checks pass. Determinism: 13/13 outputs byte-identical.

### Kill diagnostics

- **c1 / E** (0.484): per-fold range 0.298–0.648. Base success rate 0.778 at SL=4×ATR — only 41 negatives in n=185 — class compression. Wide SL maximised §2 composite but starved §8 of the negative class needed to learn from.
- **c1 / D1** (0.420): actively anti-predictive. D1 regime features carry wrong signal for c1's small regime-dominated cohort.
- **c3 / E** (0.512), **c3 / D1** (0.518): chance. c3's base success rate fluctuates 0.017 → 0.283 across 5 TimeSeriesSplit folds (16× spread). No feature holds across regimes.
- **agg_c1_c3 / E** (0.536): most stable cohort (std 0.029) and highest mean AUC, but still 0.114 short of gate. Aggregating stabilises but does not add signal.
- **agg_c1_c3 / D1** (0.496): chance.

## What we learned

**Capturable ≠ extractable.** Arc 7 is the first arc-of-record where the signal class has confirmed forward-geometry edge (Step 3 PASS with all three V-shape units passing §2 conjunctively at composite > 0.37) but no in-protocol feature set can predict cohort membership. The geometry that makes a sweep run vs reverse is driven by post-entry market dynamics that the 22 entry-time features and 11 D1 regime features don't capture. The §8 gate did its job — it caught an un-deployable cohort that would otherwise have produced misleading WFO results downstream.

**Wide SL at Step 3 can starve Step 4.** c1's selected SL=4×ATR maximised §7 composite (0.617) but drove base success to 78%, leaving only 41 negatives in n=185 for the classifier to learn from. The composite-maximising SL selection is a Step-3-local optimum that can compress the success distribution into extractability-hostile territory. This is a Step 3 / Step 4 tension worth surfacing.

**v2.1.2's `≠ scattered` floor is load-bearing.** All three surviving units at Step 3 carried `shape_tag = unclassified`. Under the prior floor (`∈ {tight_unimodal, heavy_right_tail, bimodal_separated}`), Arc 7 would have died at Step 3 with the wrong diagnosis — the v2.1.2 closure correctly admitted unclassified shapes with genuine §2-clean geometry, and Step 4 then correctly killed the arc for the right reason (no predictability), not the wrong reason (overly strict shape_tag).

**D1 regime is not better than entry-time for this signal class.** The cross-arc intuition that D1 regime features could pre-classify which sweeps will run vs reverse is not supported by Arc 7 evidence. D1 AUCs (0.42 / 0.52 / 0.50) are not systematically above E AUCs (0.48 / 0.51 / 0.54). Trend / volatility regime is not the missing variable.

**The Stepwise pullback ceiling question stays unresolved.** c1 was the test case — geometrically Stepwise except for pullback 0.567 vs 0.5 ceiling — but it didn't reach deployment. We don't know whether the §11 pullback ≤ 0.5R ceiling is over-strict in practice because no Arc 7 unit cleared §8. The question persists for future arcs.

## Cross-arc items (forward to PROTOCOL_IMPROVEMENT_BACKLOG.md)

1. **NEW — Capturable-extractable gap as a recognised closure category.** Arc 7 is the case study. v2.2 should consider an explicit closure pathway and commentary for arcs that PASS §7 but FAIL §8: this is not the same failure mode as a §2-fail at Step 3.
2. **NEW — SL-selection vs class-imbalance tension.** Composite-maximising SLs at §7 can compress the success distribution past extractability viability. Candidate v2.2 amendments: (a) §8 re-sweeps SLs and reports AUC × class-balance jointly; (b) §7 composite includes a class-balance regulariser; (c) leave §7 alone but flag the tension in §17.
3. **NEW — Open-04 external features escalation.** Arc 7 supplies concrete evidence that in-protocol features (Pipeline E + D1) can be insufficient even for capturable cohorts. Macro / session / cross-asset feature pipelines are now backed by empirical case for v2.2 commission.
4. **VALIDATED — v2.1.2 `≠ scattered` floor.** First arc-of-record where the relaxed floor was load-bearing. Floor stays. Closure proves the relaxation admits cohorts that subsequently get killed by other gates for the right reasons.
5. **UNRESOLVED — §11 Stepwise pullback ≤ 0.5R ceiling.** c1 was the test case (mono 0.536, peaks 33.5, ttp_rel 0.73, pullback 0.567). Did not deploy. Question forwarded to future arcs.
6. **CLEANUP — §15a text vs `_flatten_bar_path_for_trade` impl gap** on `mfe_so_far_r` semantics (text says `close_r` running max; impl uses `high_r` intrabar). Arc 7 followed impl. Reconcile protocol text at next calibration review.
7. **CLEANUP — Dispatch halt-criterion phrasing.** "Cross-arc bar-overlap" is a finding, not a halt. Future dispatches use overlap-vs-KH-24 as the live-system check; cross-arc overlap is a portfolio-composition note (Open-05).

## Files

All under `results/l_arc_7/` on `phase/l_arc_7`:
- `step1/` — `59de33a`
- `step2/` — `28111e4`
- `step3/` — `fa07c4a`
- `step4/` — `4e6f1a6`
- `ARC_7_RESULT.md` (this file) — to commit at `docs/arc_results/`

## Notes on protocol version

Arc 7 ran under L_ARC_PROTOCOL v2.1.2. No protocol amendments triggered by this arc directly; v2.2 commissioning candidates listed above.
