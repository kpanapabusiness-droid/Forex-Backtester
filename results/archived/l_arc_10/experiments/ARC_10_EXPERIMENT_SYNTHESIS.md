# Arc 10 — Experimental Step 5 synthesis

> **DO NOT DEPLOY.** Diagnostic evidence only. Open-04 commission, Open-06 threshold relaxation, and any other v2.4 calibration change require separate governance.
> Status: experiments complete (EXP-01 through EXP-06).
> Reads: `EXP_01_auc_bootstrap.md`, `EXP_02_htf_ablation.md`, `EXP_03_threshold_scan.md`, `EXP_04_q2_2022_regime.md`, `EXP_05_v_shape_pool.md`, `EXP_06_open_04_probe.md`.
> Date: 2026-05-18.

---

## Headline findings (one line per experiment)

| Experiment | Headline |
|---|---|
| EXP-01 | Bootstrap 95% CIs straddle both thresholds; P(either pipeline clears under resampling) = **40.5%**. Both AUCs are statistically inside the uncertainty band — Arc 10 is not distinguishable from "would pass" given the sample. |
| EXP-02 | HTF features add **+0.024** total AUC over the generic baseline. The **L_0 bucket** (single feature `L1_minus_L0_atr` = D1 HL slope magnitude) carries 116% of the lift; `age` features contribute only 18%. |
| EXP-03 | Triple-pass threshold (Arc 6 + Arc 7 + Arc 10) requires relaxing E AUC from 0.65 to **≤ 0.536** (binding constraint: Arc 7 agg_c1_c3). At any relaxed threshold the false-positive cost from arc history is **zero documented arcs** — downstream WFO failures (Arc 4 RERUN, Arc 5) are Pipeline D1 deployment-economics failures unrelated to E-AUC. |
| EXP-04 | Fold 2 (date range **2023-07-13 → 2024-06-05** — NOT Q2 2022 as the original closure-doc narration suggested) shows AUC drop, but **no entry-time regime descriptor places fold 2 ≥ 1.5σ from the cohort mean**. The drop is not explained by atr / range / ema / trend / spread descriptors at the entry-time-knowable level. |
| EXP-05 | Pooling Arc 7 c3 (V-shape, n=365) + Arc 10 c1 (V-shape, n=228) at common SL=3.0×ATR with generic features only produces **pooled AUC 0.6348** (gap −0.015 to 0.65) — closer to the gate than Arc 10 c1 alone (+0.029 over its generic baseline). Arc 6 BLOCKED (no step1 artefacts in-branch + wrong archetype). |
| EXP-06 | Both Open-04 candidates (D1 Kijun dist + session dummies) produce **negative** AUC delta on Arc 10 c1 alone. The 228-trade cohort is feature-saturated; more features add noise. Decision-relevant evidence requires the cross-arc reproduction loop. |

---

## Cross-experiment integration

### What the evidence supports (with effect sizes)

**A. Open-06 threshold relaxation (medium support):**

- Arc 10 alone: E AUC 0.6296 (margin −0.020), D1 AUC 0.5897 (margin −0.010). EXP-01 says these are not statistically distinguishable from threshold under bootstrap; joint P(either clears) = 40.5%.
- Cross-arc evidence (EXP-03): triple-pass for Arc 6/7/10 requires a 0.114 relaxation, which is implausibly large. **The binding case is Arc 7 agg_c1_c3 (0.536), not Arc 10.** Arc 6/Arc 10 pair would pass at threshold ≤ 0.600.
- Documented false-positive cost at relaxed threshold: 0 arcs. The deployment failures on record (Arc 4 RERUN, Arc 5) failed for full-pool Pipeline D1 economics — they would not pass E AUC at any plausible relaxed threshold.
- **Effect size summary:** 0.020 to 0.114 of threshold relaxation needed, depending on which arcs the v2.4 cycle wants to include. Arc 7 evidence weakens this case (it's an outlier in margin magnitude).

**B. Cross-arc V-shape pooling (medium-to-strong support):**

- EXP-05: pool n=593 produces AUC 0.6348 with generic features only. **The pooled AUC exceeds both per-arc baselines** (Arc 10 c1 generic 0.6057, Arc 7 c3 generic 0.4954 at re-imposed SL=3.0). +0.029 over Arc 10, +0.140 over Arc 7.
- Pool falls **0.015 short** of the 0.65 gate — narrower than Arc 10 alone (−0.020) despite using only generic features and including a cohort (Arc 7) with much weaker per-cohort AUC.
- This is the **strongest single piece of evidence** that V-shape recovery is a deployable archetype if treated as cross-arc.
- **Effect size summary:** +0.029 pooled-vs-Arc-10-c1 generic baseline; +0.094 pooled-vs-Arc-7-c3-generic. Pool n=593 is small for 5-fold TimeSeriesSplit, but the direction is consistent.

**C. HTF feature class (medium support):**

- EXP-02 attribution: HTF features contribute +0.024 to Arc 10 c1 E AUC over the generic baseline. The load-bearing feature is `L1_minus_L0_atr` (D1 HL slope magnitude in ATR units) — removing it alone drops AUC by 0.028.
- L_1 distance/proximity bucket: +0.019 LOO drop. Age bucket: +0.004 (essentially zero — these features are not load-bearing).
- **Effect size summary:** +0.024 absolute lift from HTF as a class. Single-feature `L1_minus_L0_atr` is the candidate worth cross-arc reproduction. Sibling arcs (8 / 9 / 11) may not have an L_1 / L_0 structure to test this directly.

**D. Regime-aware entry filter (weak / no support from this evidence):**

- EXP-04: no entry-time regime descriptor (atr / range / ema dist / trend slope / spread) places fold 2 ≥ 1.5σ from the cohort mean. The fold-2 AUC drop is not explained by the in-Step-4 feature class.
- Correlations across n=5 folds are not statistically meaningful, but the strongest correlation magnitude is `spread_pips_used` (+0.72 vs E AUC) — and that's a known noise contributor (wider spreads in low-volume windows correlate with worse classifier quality everywhere).
- **Effect size summary:** no candidate regime descriptor passes the |z|≥1.5 + |corr|≥0.5 dual criterion. Negative evidence for the "single-fold-outlier is a regime filter we can engineer" hypothesis.

**E. Open-04 external features (no support from this evidence):**

- EXP-06: both candidates (D1 Kijun dist, session dummies) produce **negative** AUC delta on Arc 10 c1 alone (−0.015 and −0.022 respectively).
- This is a small-n saturation result (n=228 with 25 baseline features → adding more features adds noise faster than signal).
- **Open-04 commission needs cross-arc replication.** The single-arc probe on Arc 10 is uninformative.
- **Effect size summary:** −0.015 to −0.022 on Arc 10 alone; not interpretable until Arc 8/9/11 results land.

---

## Aggregate recommendation packet for v2.4 calibration

Ranked by quantitative support (strongest first):

### Rank 1 — Cross-arc V-shape pooling (Open-05 portfolio adjacent)

**Evidence:** EXP-05 pooled AUC 0.6348 vs Arc 10 c1 alone 0.6057 (+0.029). Pool narrowly misses 0.65 (gap 0.015).

**Action for v2.4:**
1. When Arc 8/9/11 close, identify their V-shape clusters (if any).
2. Re-run EXP-05 with the expanded pool.
3. If pooled AUC clears 0.65 with 3+ arcs, V-shape becomes a deployable cross-arc archetype — discuss adding a §11 row entry or a separate cross-arc "archetype-class" deployment path.

### Rank 2 — `L1_minus_L0_atr`-class feature promotion (Open-06 / signal-spec adjacent)

**Evidence:** EXP-02 single feature carries 116% of HTF LOO drop on Arc 10.

**Action for v2.4:**
1. Note "D1 HL structural magnitude" as a feature-class candidate.
2. If sibling arcs surface analogous structural-magnitude features (Arc 8 PR-HHHL has obvious HL structure; Arc 11 SHB has swing-high structure), reproduce the ablation there.
3. If reproduced, propose feature-class commission to the standard 8-base entry-feature catalog.

### Rank 3 — Open-06 AUC threshold recalibration

**Evidence:** EXP-01 bootstrap CIs straddle threshold; EXP-03 documented FP cost = 0 arcs at threshold relaxations admitting Arc 6 / Arc 10. Arc 7 weakens the case (its 0.536 is outside any reasonable near-miss band).

**Action for v2.4:**
1. Treat as **secondary** to ranks 1 and 2. Threshold relaxation alone admits more cohorts but doesn't change the structural finding that V-shape is partially extractable but below current gate.
2. If V-shape pooling (rank 1) reaches deployable AUC at gate 0.65, threshold relaxation becomes unnecessary.
3. If pooling stays below 0.65 even with Arc 8/9/11, revisit threshold with the larger evidence base.

### Rank 4 — Open-04 external-feature commission

**Evidence:** EXP-06 single-arc probe inconclusive (both candidates negative on n=228). Needs cross-arc replication.

**Action for v2.4:**
1. Defer commission proposal until Arc 8/9/11 results land and EXP-06 can be re-run as a multi-arc probe.
2. If multi-arc evidence is positive for `d1_kijun_dist_atr` or analogous D1 regime features, formal Open-04 commission proposal becomes viable.

### Rank 5 — Regime-aware entry filter

**Evidence:** EXP-04 negative (no candidate descriptor explains fold-2 drop).

**Action for v2.4:** **drop this candidate** unless cross-arc fold-stability analysis surfaces a pattern. v2.3 §1 already removed Step 5 cross-fold stability as a gate; this evidence is consistent with that decision.

---

## Methodological notes

- **EXP-01 method choice:** AUC sampling-distribution bootstrap on out-of-fold (p, y) pairs is the standard approach. A nested per-fold-AUC bootstrap that refits 2000 RFs was rejected as costly and not qualitatively different — the fold-split variance dominates the bootstrap-resample variance for small n.
- **EXP-04 correction:** the original ARC_10_RESULT.md narrated fold 2 as "Q2 2022 USD regime"; this experiment shows the actual fold-2 date range is **2023-07-13 → 2024-06-05**. The closure doc's narrative was wrong; this synthesis corrects it. The fold-2 drop is real but its regime cause is not identified by the descriptors tested here.
- **EXP-05 BLOCKED items:** Arc 6 is doubly blocked — no step1 artefacts in this branch AND it's a Stepwise climber, not V-shape. Including Arc 6 would require both (a) cross-branch artefact access and (b) a different "archetype" definition.
- **EXP-06 framing:** purely informational. The dispatch explicitly excluded Open-04 commission from scope.

## Determinism

All experiments use `random_state=42`, `n_jobs=1` (deterministic), and `lineterminator="\n"` on CSV writes. Re-running any experiment produces byte-identical outputs (verified for EXP-01 + EXP-02 via the bootstrap seed reuse; not explicitly re-run-verified for EXP-03 / EXP-04 / EXP-05 / EXP-06 since they are single-pass).

## Files

- `EXP_01_auc_bootstrap.md` + `raw/exp_01_*`
- `EXP_02_htf_ablation.md` + `raw/exp_02_*`
- `EXP_03_threshold_scan.md` + `raw/exp_03_*`
- `EXP_04_q2_2022_regime.md` + `raw/exp_04_*`
- `EXP_05_v_shape_pool.md` + `raw/exp_05_*`
- `EXP_06_open_04_probe.md` + `raw/exp_06_*`
- `ARC_10_EXPERIMENT_SYNTHESIS.md` (this file)

## Cross-references

- [results/l_arc_10/ARC_10_LIVE.md](results/l_arc_10/ARC_10_LIVE.md)
- [results/l_arc_10/ARC_10_RESULT.md](results/l_arc_10/ARC_10_RESULT.md)
- [docs/arc_results/ARC_6_RESULT.md](docs/arc_results/ARC_6_RESULT.md)
- [docs/arc_results/ARC_7_RESULT.md](docs/arc_results/ARC_7_RESULT.md)
- L_ARC_PROTOCOL v2.3 §1 (Step 5 removal), §16a (HALT/KILL), §8 (extractability gate), Open-04, Open-06.
