# L_ARC History — Arcs 1-11

> Consolidated record of the L_ARC signal-testing programme through Arc 11, plus the KH-24 deployment anchor and a pre-v2.0 summary.
> Generated 2026-05-19 from the canonical closure docs. Source links retained per arc.
> This doc is the consolidated reference. The per-arc closure docs remain the authoritative record for each arc's numerical results.

---

## Overview

The Forex Ignition Rebuild has two parallel tracks. The **KH-24 deployment track** produced the live, gate-passing system that runs on a Contabo VPS at the 5ers prop firm (worst-fold ROI +1.92% / DD 6.37% across 7 OOS folds; long-only, 4H kb_exhaustion signal with D1 regime filter). The **L_ARC research track** tests structurally distinct signal hypotheses through a multi-step pipeline (currently 5 steps under v2.3) whose terminal gate is a worst-fold walk-forward optimisation (WFO) on 7 anchored expanding folds covering Oct-2020 to Jan-2026. The goal is one or more PASS-DEPLOYABLE survivor systems that complement or supersede KH-24.

L_ARC asks of an arc, in sequence: does the signal produce a clean trade pool (Step 1 plumbing)? Do the trade paths cluster into well-separated archetypes (Step 2 clustering)? Do at least one archetype's forward-geometry properties (monotonicity, MFE, wrong-way rate, shape tag) clear pre-committed §2 capturability floors (Step 3)? Can a classifier at entry (Pipeline E) or one bar later (Pipeline D1) separate that archetype's signals from the rest with AUC clearing §8 (E ≥ 0.65 / D1 ≥ 0.60) (Step 4)? And finally, under realistic WFO with full-pool cost accounting, does the strategy clear PASS-DEPLOYABLE (worst-fold ROI ≥ 5%, DD < 8%) or PASS-VIABLE (positive worst-fold ROI, DD < 8%) at 0.5% risk-per-trade (Step 5, was Step 6 pre-v2.3)?

This document exists because eleven arcs of accumulated learning, scattered across closure docs, is hard to navigate. The protocol redesign that follows this dispatch needs one place that summarises what each arc tested, what it found, and what cross-arc patterns now have empirical evidence behind them. This is that place. The forward-looking conclusions — what to do next — belong to the redesign, not to this record.

---

## Cross-arc summary

| Arc | Signal | TF | Protocol | Track | Disposition | Step reached | Key reason | Cross-arc contribution |
|---|---|---|---|---|---|---|---|---|
| 1 (v1.x verbatim) | LCHAR rank 1 (univariate_extreme) | 1H | L6.0 | verbatim WFO | FAIL | 6 (WFO) | Verbatim WFO failed gate | Surfaced CH-001 (concurrent_signals_within_3h ≤ 13) as a passing filter under L6.0 framing |
| 1 (v1.0 redo) | same | 1H | v1.0 | trade-classifier | FAIL | Step 3 calibration | Family-level not feature-level calibration check needed | Triggered v1.1 amendment (family-level calibration check, BH-tier reportorial) |
| 2 (v1.x) | LCHAR rank 2 (`mtf_alignment.2_down_mixed.kijun`) | 1H | L6.0 | verbatim WFO | FAIL | 6 (WFO) | Verbatim WFO failed (DD 39-91%) | Surfaced "real edge, not capturable by fixed-policy exits" pattern |
| 2 redo (v2.0) | same, h=120 | 1H | v2.0 | path-shape clustering | KILL | Step 3 | Cluster 2 Stepwise climber missed mono floor by 0.009 / wrong_way by 0.005 / shape_tag unclassified despite t-stat +52 on n=2,278 | Open-09: hard-floor false-kills of strong-extractability archetypes; shape_tag definition pressure for high-magnitude cohorts |
| 2 redo2 (v2.1.1) | same | 1H | v2.1.1 | reclustered | (folded into Arc 5) | Step 2 | Arc 5 ≡ Arc 2 at trade-pool level when registry h overridden | Documented schema fork; F2/F5 confirmed |
| 3 | `volatility_regime.d1_atr_top_decile.any.h_120` | 1H | v2.0 | path-shape | CLEAN-NULL | Step 3 | Stepwise climber (27.5% of pool) missed shape_tag (bimodal not in §2 allowed set) and wrong_way (38.3%) — §11 row 7 admits bimodal but §2 doesn't | Open-12 (silhouette tie tolerance), Open-13 (§2 vs §11 bimodal incompatibility), Open-14 (same-archetype aggregation), Open-15 (SL/horizon asymmetry) |
| KH-24 v2.0 self-test | `kb_exhaustion_bar` (live KH-24 signal) | 4H | v2.0 | path-shape | HALT | Step 3 | c4 trend-rider (14.5% pool) missed mono 0.55 by 0.020 / shape_tag scattered due to 87.7% 240-bar window-cap censoring | §14 anchor was measured on filtered population not bare signal; §2 floor calibrated against filtered anchor leaves no headroom on bare signal; protocol cannot re-derive what v1.0's filters extracted |
| 4 (initial) | `bar_range_top_decile.neg.h_001` | 1H | v2.1.1 | path-shape (Pipeline D1) | CLEAN-NULL | Step 5 retroactive | First arc to pass Step 3 + Step 4 + Step 5 §9 stability; closed retroactively on HistData spread reconciliation showing modelled spreads understated real spreads 3-48× | Triggered locked-spread-floor replacement with per-pair p50 values; cross-arc spread audit as Phase 0 prerequisite |
| 4 (rerun) | same | 1H | v2.1.2 | path-shape (Pipeline D1) | FAIL | Step 6 §10 | Admit pool +0.125R but reject pool (32% × −0.232R) + early-exit pool (11% × −0.685R) drag swamps edge under full-pool deployment reckoning; full-data ROI −77% at 0.20% risk | Open-22 (full-pool gate at §9), Open-23 (Pipeline D1 cost-language correction), Open-24 (per-archetype pre-t SL) |
| 5 | `mtf_alignment.2_down_mixed.kijun.h_120` | 1H | v2.1.1 | path-shape (Pipeline D1) | SHELVED Step 6 FAIL | Step 6 | Admit-set R +0.14 to +0.21 per fold survives PR 2 exit policy; rejected pool 78% × −0.46R adverse-selected drag overwhelms; bar-0/1 SL hits 7.9% × −0.74R unavoidable | Same Pipeline D1 architectural failure as Arc 4 rerun (3rd consecutive with Arc 8); §9 admit-only framing bug surfaced; signal trade-pool sha-identical to Arc 2 redo2 (registry h descriptive not constraint) |
| 6 | failed-breakout reversal long (out-of-registry) | 4H | v2.1.2 | path-shape | DIES Step 4 | Step 4 | Both surviving clusters (c0 Stepwise-boundary, c2 Stepwise) mechanically clear D1 AUC ≥ 0.60 but threshold sweep collapses to max-F1 fallback at 0.4-0.9% recall; deployability-level FAIL | Open-21 (Step 4 deployability gate — strict max-precision-subject-to-recall ≥ 0.60); §11 stepwise extended ceiling 5→50 was load-bearing for c2 |
| 7 | liquidity sweep + reclaim long (out-of-registry) | 4H | v2.1.2 | path-shape | CLEAN-NULL Step 4 | Step 4 | First capturable-not-extractable closure of record. PASS §7 (3 V-shape units survive §2 conjunctively); FAIL §8 (0/6 unit × pipeline AUCs clear gate; best agg/E 0.536) | Validated v2.1.2 `≠ scattered` floor as load-bearing; SL-selection vs class-imbalance tension; Open-04 external features escalation |
| 8 | pullback resume HH/HL long (PR-HHHL) | 4H | v2.3 | path-shape (E + D1) | HALT_DEPLOYMENT | Step 5 WFO | Steps 1-4 PASS; c1 V-shape FG-weak survives Step 4 (E AUC 0.697 / D1 AUC 0.637 at t=1); Step 5 WFO admit-only PASS but full-pool FAIL (ROI −13% to −15%, DD 15-19%); classifier admits 70-89% of pool because c1 and c2 share entry-bar geometry | Third consecutive Open-22/23/24 admit-only-vs-deployment failure (Arcs 4 RERUN / 5 / 8); v2.4 §1.5 entry-separability gate proposed (Open-25); c1 admit-only economics logged for Open-05 portfolio composition |
| 9 | IB-trend compression-break long | 4H | v2.3 | path-shape (E + D1) | STEP_4_KILL_REAFFIRMED | Step 4 (originally); held-open through Exp 8 | Original KILL: E AUC 0.511, D1 AUC 0.626 threshold-sweep recall 0.003. Held-open Pipeline E retry reached AUC 0.7508 with 12 added features — but two D1 swing features used a ±10-bar centred swing detector (non-causal); causal patch dropped AUC to 0.5190 (LGBM) / 0.5551 (RF). KILL stands. | **Producer-level causal audit dimension** now standard for every classifier audit (distinct from join-level + e2e). v2.x §8 D1 feature-budget expansion WITHDRAWN; v2.x §3 threshold-grid weakened; cohort verified deployable in oracle (+39% ann ROI, 0% DD) but unreachable on causally-clean entry features |
| 10 | D1 swing-low rejection long (DLR, out-of-registry) | 4H | v2.3 | path-shape (E + D1) | STEP_4_HALT | Step 4 | First arc end-to-end under v2.3 (5-step pipeline). c1 V-shape recovery near-miss on disjunctive §8: E AUC 0.6296 (margin −0.0204), D1 AUC 0.5897 (margin −0.0103). §16a Path A compound reading applied. Post-closure WFO oracle Sharpe 4.61 vs base −1.29 (gap +5.90); EXP-05 cross-arc V-shape pool (Arc 7 c3 + Arc 10 c1) AUC 0.6348 closest to gate. EXP-01 P(realisable AUC ≥ 0.65) = 12.5% | Arc 6 reclassified Stepwise (not V-shape) per EXP-05; fold-2 date 2023-07/2024-06 (not Q2 2022) per EXP-04; Open-06 (AUC threshold) weakened; Open-04 (external features) deferred pending Arcs 8/9/11; cross-arc V-shape clusterifier as leading v2.4 candidate |
| 11 | swing-high breakout trend long (SHB, out-of-registry) | 4H | v2.3 | path-shape (E + D1) | CLOSED-HALT (§16a Path A) | Step 4 | 0/3 surviving units clear AUC gate; best 0.5728 (agg_c1_c3 D1 t=5), margin 0.027. Post-closure off-protocol: oracle WFO PASS (worst ROI +101%/yr / 2.48% DD on c1 raw); no-oracle Pipeline E FAIL all gates; best signal-improvement combo DE t=7 + dynamic SL: ROI ann +3.11% but DD/ROI 1.04 not deployable | Second capturable-not-extractable instance (pairs with Arc 6 Stepwise, distinct from Arc 7/10 V-shape pair); Pipeline DE (deferred-entry) as v2.4 candidate; "timing > features as extractability lever" empirical finding; AUC ceiling structural for 4H entry-bar features (three feature regimes capped below 0.55) |

---

## Cumulative findings

### What works (signal)

- **KH-24 itself is the only deployed system.** Worst-fold ROI +1.92%, worst-fold DD 6.37%, 214 trades across 7 OOS folds Oct-2020 → Jan-2026, 7/7 positive folds. Live on Contabo VPS / 5ers. The system is `kb_exhaustion_bar` (C1-C6, C8, C9; C7 disabled) at 4H with D1 regime filter (one-day lag), 2.0×ATR(14) hard SL anchored at entry price, 1.5×ATR trailing stop activating at +2.0×ATR (close-based), 1H CIR ≤ 0.28 filter, exposure cap = 2, per-bar MT5 spreads.

- **V-shape recovery archetype is cross-arc capturable.** Two genuine V-shape near-misses on record (Arc 7 c1/c3/agg at Step 4; Arc 10 c1 at Step 4 disjunctive §8). EXP-05 cross-arc pool (Arc 7 c3 + Arc 10 c1, n=593) reaches AUC 0.6348 vs §8 gate 0.65 with the 17-feature generic subset; +0.029 over Arc 10 alone, +0.140 over Arc 7 alone. WFO oracle on Arc 10 c1: Sharpe 4.61, expectancy 1.55R per trade, max DD 0.50%, gap +5.90 over base. Single load-bearing feature: `L1_minus_L0_atr` (D1 HL slope magnitude) carries 116% of HTF LOO drop on Arc 10. Cohort is real cross-arc; entry-time predictability is the bottleneck.

- **Stepwise climber archetype appears across multiple arcs** with measurably strong forward magnitude when present (Arc 2 redo c2: fwd_mfe_p50 5.83R, t-stat +52 on n=2,278; Arc 4 c1: composite 0.358; Arc 6 c2: composite 0.616; Arc 11 c1: fwd_mfe_p50 4.48R, reach_1R 100%). It is consistently capturable at Step 3; consistently fails at Step 4 (entry-time predictability) or Step 5/6 (Pipeline D1 full-pool reject drag).

- **Pipeline D1 path-so-far features carry real signal.** The first held bar reveals path quality across multiple arcs. Arc 4 c1: D1 RF AUC 0.667 (Pipeline D1 t=1). Arc 5 c1: D1 RF AUC 0.636 at t=1 (KH-24 anchor parity 0.638). Arc 6 c2: D1 AUC 0.630 at t=1 climbing to 0.711 at t=10. Top 4 path-so-far features (close_r, velocity, MAE, MFE at t=1) commonly carry ~54% of feature importance. This is real information; the architectural cost is the reject pool.

- **D1 regime filter (one-day lag) is necessary for KH-24-shaped systems.** Same-day D1 alignment is lookahead by construction. The fix (one-day lag — each 4H bar sees prior calendar day's D1 close) is now permanent invariant.

### What doesn't (anti-signal)

- **Entry-bar features alone (Pipeline E) consistently fall short on V-shape cohorts.** Five arcs (4, 5, 7, 8, 10) show Pipeline E AUC clustering 0.48-0.60 — below the 0.65 deployability bar. The gap is feature-set bound, not classifier-family bound (RF vs Logistic gaps small and frequently negative). Adding richer entry-time features bought ~0.01-0.03 AUC at this regime in arc-level experiments; the structural ceiling for 4H entry-bar features sits below 0.60 across the arcs tested.

- **Pipeline D1 admit-only economics ≠ deployment economics.** Three arcs in a row (Arc 4 rerun, Arc 5, Arc 8) PASSED §9 admit-only stability and FAILED §10 full-pool deployment. The arithmetic is `(admit_rate × admit_mean) > (reject_rate × |reject_mean|) + (early_exit_rate × |early_exit_mean|)`. Reject pool typically −0.15 to −0.46R per trade (adverse-selected — classifier rejection at bar t is itself a prediction of further adverse drift); early-exit pool typically −0.45 to −0.69R, 10-15% of signal flow. Costs ~2× the admit-pool edge across the three arcs.

- **Capturable ≠ extractable.** Arc 7 (CLEAN-NULL Step 4) and Arc 11 (CLOSED-HALT Step 4) are the case studies. A signal class can produce clean §2-passing path-shape archetypes (PASS §7) with confirmed forward-geometry edge, yet have no in-protocol feature set predicting cohort membership (FAIL §8). The geometry that makes a sweep run vs reverse is driven by post-entry market dynamics that entry-time features and D1 regime features don't capture.

- **Mean-reversion class (Arc 4 initial flavour, Arc 14 unrun) struggles on prop-firm cost structures.** Arc 4 demonstrated a real signal class that died on the cost model rather than the methodology — the per-trade edge (0.18 mean_r best refit folds) was too small to support deployment given 0.07R real spreads. Survives or doesn't survives the full-pool reckoning becomes a function of cost realism.

- **Short side has no demonstrated edge on `kb_exhaustion_bar`.** Phase KC short-mirror exploratory run failed; permanently eliminated.

- **C7 volume gate on 5ers data.** Broker-specific, no lift validated; permanently eliminated.

- **TP1 half-off structure.** Inferior to no-TP1; permanently eliminated.

- **Currency exposure cap (KH era).** Superseded by exposure cap = 2.

- **KH-25 re-entry exposure cap, signal_flip exit, kijun_4h exit, D1b slope filter, choppiness gate, FOMC proximity filter, agree_count gate, range/ATR ceiling at 1.25×, 2% risk on 5ers data, same-day D1 alignment, 1H timeframe port of KH-24, L6.0 verbatim-as-gate framing.** All permanently eliminated; see `CLAUDE.md` for the full strike list.

### Methodology lessons

- **Ex-ante population always.** Forward-conditioned dataset construction (population definition that depends on outcome) was the canonical project failure (JL invalidation: 79.8% win rate → −50% ROI when rebuilt ex-ante). Permanently eliminated.

- **Worst-fold WFO at dual-tier disposition is the only judge at step 6 / step 5.** Average fold, best fold — irrelevant.

- **No lookahead / no repainting.** Hard invariant. Tested for lookahead-invariance at every step. D1 lag is the one-day-lag rule.

- **Within-arc thresholds do not move.** Calibration is cross-arc only. JL is the precedent for what happens when you move a threshold to make a result pass.

- **Producer-level causal audit dimension (Arc 9 lesson).** Distinct from join-level causality (`merge_asof` direction, days_lag distribution) and from end-to-end probability reproduction. Standard mathematical definitions of trading concepts (swing detection, pivots, ZigZag, centred smoothers, bilateral change-point detection) are non-causal by default and must be replaced with one-sided or confirmation-lag variants. Now standard for every future classifier audit.

- **Spread realism matters at the order of magnitude level.** Arc 4 demonstrated locked 0.1 pip spread floor under-modeled real first-5-minute execution-bar spreads by 3× (EUR/USD) to 48× (GBP/NZD). Locked file replaced with per-pair p50 values from HistData 2024-2025 audit. Spread audit is now Phase 0 prerequisite for any L arc.

- **Spread-floor changes are not population-invariant under exposure caps.** Changing the spread floor file shifts entry/exit fill prices → shifts when stops fire → shifts when `max_concurrent_per_pair` cap releases → shifts admission for subsequent signals. Trade pool can drift ±1-2% from a pure cost-model change. Path features (mid-based) remain spread-independent; PnL and exposure-derived metrics do not.

- **Convention (b) mark-to-market DD over convention (a) closed-trade ordering.** Convention (a) understates real account DD by 14-63% due to concurrent positions across 28 pairs. 5ers measures equity in real-time; convention (b) is production-truth gate. Should be §10's default gate metric.

- **Per-fold classifier refit, not random-shuffle CV.** Arc 4 surfaced 14% relative worst-fold ROI haircut when refit per fold. Cross-arc structural item.

- **F1 structural leakage.** Pool start coincides with F1 OOS start across many arcs — no honest WFO training data for F1. Affects every L arc with this data window. Options: pool back-extend, drop F1 evaluation across all arcs, or alternate fold structure.

- **Step 4 max-F1 fallback at sub-1% recall.** Arc 6 surfaced this: §8 gate at AUC ≥ 0.60 fires PASS even when threshold sweep falls back to max-F1 at 0.4-0.9% recall. v2.2 §3 closes this — Step 4 threshold sweep failure now kills the archetype.

- **§11 row priors are first-pass.** Cross-arc evidence accumulating that §11 centroid ranges need empirical refinement. v2.1.2 extended Stepwise climber `local_peaks` ceiling 30 → 50; was load-bearing for Arc 6 c2 and Arc 7 c1.

- **Shape_tag definitions for high-magnitude / window-censored cohorts.** Arc 2 redo c2 (heavy_right_tail fails at p95/p50=1.38 because body is already high) and KH-24 v2.0 self-test c4 (scattered because 87.7% of cap-binders dilute final_r) — current shape_tag rules don't compensate for these regimes.

- **Determinism baseline.** `random_state=42`, `n_jobs=1`, `lineterminator="\n"` throughout for any work that must be byte-identical-reproducible. Audited via two-run sha256 comparison. CI-enforced.

### Open structural questions

- Is the entry-feature ceiling for 4H signals (Pipeline E AUC ~0.55-0.60 on V-shape cohorts) FX-4H-specific, or does it generalise to other timeframes / asset classes? Arc 6 D1 AUC growth to 0.711 at t=10 suggests post-entry information has room to run; whether other timeframes admit cleaner entry-time discrimination is untested.

- Whether differentiated exit policy on KH-24 lifts worst-fold ROI without compromising the live system. KH-24's worst-fold +1.92% is tight against the pass-deployable 5% bar under audit-reconciled costs (real-spread KH-24 → ~+1.28% → pass-viable). PR-2-style per-archetype exit policy is feasible in principle but not yet tested on the deployed signal.

- Whether currency-strength, intra-bar microstructure, multi-TF context (W1/M1), or other non-event signals admit edge that the 1H/4H signal-class arcs missed. Arc 9 oracle ceiling demonstrated +39% ann ROI is on the table if the cohort can be reached — what reaches it remains open.

- Whether Pipeline DE (deferred-entry: enter at bar t post-signal or not at all) outperforms Pipeline D1 (enter at signal, classifier at bar t mid-trade). Arc 11 surfaced DE as the only direction that moved AUC; not yet protocol-blessed.

- Whether differentiated SL (per-archetype Step 3 selected SL multiplier vs uniform 2×ATR pre-t SL) materially shifts Pipeline D1 reject + early-exit pool economics. v2.3 §0 added per-archetype `pre_t_sl_atr_multiplier`; not yet validated across multiple arcs.

- Whether v2.4 §1.5 entry-separability gate (one-vs-rest precision@recall=0.60 ≥ 0.30 on entry features alone, before committing Step 1 simulation compute) saves arc-level compute on signals that can't deploy. Proposed by Arc 8 closure; pending design and ratification.

- Whether the broker data (5ers MT5) is portable to other prop firms. KH-24 was developed and locked on 5ers data; the spread regime and execution-bar dynamics are specific to that broker. Cross-broker portability is untested.

---

## Per-arc detail

### Arc 1 — LCHAR rank 1 (univariate_extreme) — FAIL (v1.x + v1.0 redo)

- **Signal:** L characterization arc rank-1 candidate; specific signal from `docs/LCHAR_TOPN_REGISTRY.md` entry 1 family `univariate_extreme`. 1H timeframe.
- **Methodology:** Initially run under L6.0 verbatim-as-gate framing (closed FAIL on verbatim WFO gate); redone under v1.0 protocol's six-step extractability methodology (closed FAIL at Step 3 calibration check on outcome (a) protocol-spec error).
- **Pool:** 45,673 trades (per the v1.3 calibration diagnostic's input metrics).
- **Key results:** L6.0 verbatim WFO failed gate. Under v1.0, the Step 3 calibration check (named-feature) failed for `concurrent_signals_within_3h` — the carrier feature surfaced but at Tier 3, not Tier 1/2 as the v1.0 gate required. Three to four sibling features in the same effect family did clear Tier 2; the carrier was a noisier representative of its family than its siblings. CH-001 (concurrent_signals_within_3h ≤ 13) under L6.0 framing produced a passing WFO with real edge confirmed; this was recorded in `docs/archive/nnfx_era/CANDIDATES.md` and `docs/archive/nnfx_era/PHASE_L6_ARC1_P2_OPEN.md`.
- **Verdict:** FAIL under both v1.x and v1.0. Calibration-check FAIL under v1.0 triggered the v1.1 amendment (family-level calibration check, BH-tier reportorial not eliminative).
- **Cross-arc contribution:** Methodology amendment (family-level check). The original v1.0 spec confused "did the methodology find filterable structure of the right kind" with "did the methodology re-find this exact feature." The first question is the right one.
- **Source closure doc:** `results/l_arc_1/step3_extractability/PHASE_L_ARC_1_STEP3_S15_OUTCOME.md` (referenced by `archive/L_ARC_PROTOCOL_v1_1_AMENDMENT.md`); v1.x WFO closure material in `docs/archive/nnfx_era/PHASE_L6_ARC1_OPEN.md` and `docs/archive/nnfx_era/CANDIDATES.md`. Closure doc status: **partially preserved** — calibration-check FAIL outcome referenced but no standalone `ARC_1_RESULT.md` exists.

### Arc 2 — LCHAR rank 2 (`mtf_alignment.2_down_mixed.kijun`) — FAIL → KILL (v1.x → v2.0 redo)

- **Signal:** `TRIAL__mtf_alignment__2_down_mixed__kijun__h_120` (LCHAR registry entry 2). Base condition: extremes both down, 4H_mr up — mixed-down state. Direction sub-spec: kijun (Kijun-sign trend definition). 1H timeframe.
- **Methodology:** Initial v1.x verbatim WFO; redone under v2.0 path-shape clustering + two-pipeline (E/D1) extractability.
- **Pool (v2.0 redo):** 12,262 trades across 28 pairs, 2010-02-10 → 2025-12-19. K=4 silhouette 0.4778 selected.
- **Key results (v2.0 redo):**
  - Cluster 2 (Stepwise climber, 18.6% pool, n=2,278): fwd_mfe_p50 5.83R, fwd_mfe_p75 8.04R, frac_reach_1R 99.65%, frac_reach_2R 96.44%, final_r_mean +3.18R, t-stat +52.17, mass_gt_5R 0.6102.
  - Failed §2 on three criteria: monotonicity_centroid 0.5414 vs ≥ 0.55 floor (miss 0.0086), frac_wrong_way 0.3051 vs ≤ 0.30 ceiling (miss 0.0051), shape_tag "unclassified" vs `∈ {tight_unimodal, heavy_right_tail}`.
  - PR #129 K=5 archetype 1 concordance: matches on fwd_mfe_p50, frac_reach_1R; divergent on final_r_mean and t-stat (granularity effect K=4 vs K=5).
- **Verdict (v2.0 redo):** KILL ARC at Step 3. v1.x and v2.0 both close FAIL on the same signal — different failure mechanisms (DD-driven on v1.x, capturability-driven on v2.0), same underlying truth: this signal generates large R outcomes on paths that can't be exited cleanly.
- **Cross-arc contribution:** Open-09 evidence — hard floors may false-kill archetypes with weak capturability but strong extractability. Cleanest test case for shape_tag definition pressure on high-magnitude cohorts (heavy_right_tail criterion fails because body is already high). Path-shape vs magnitude gate ordering question raised.
- **Source closure doc:** [results/l_arc_2_redo/ARC_2_REDO_RESULT.md](results/l_arc_2_redo/ARC_2_REDO_RESULT.md).
- **Subsequent re-evaluation:** Arc 2 redo2 (v2.1.1 schema fork with `is_held` schema) reran Step 1 trade-pool generation; pool subsequently confirmed sha256-identical to Arc 5 under v2.1.1 once registry h was overridden to 120 (Arc 5's F2 finding). Arc 2 redo2 closure under v2.1.1 was folded into the Arc 5 closure stream rather than producing a standalone result doc.

### KH-24 v2.0 self-test — HALT (v2.0 protocol calibration arc)

- **Signal:** bare `kb_exhaustion_bar` (C1-C6, C8, C9; C7 disabled) — the live KH-24 signal. Long-only, 4H, 28 FX pairs. 1R hard SL = 2.0×ATR(14). 240-bar forward window.
- **Methodology:** v2.0 protocol applied to bare KH-24 signal as a protocol self-test + Pipeline D1 backtester commissioning arc.
- **Pool:** 842 trades across 28 pairs, 2010-01 → 2025-12. 16.7% cap-binding (240-bar window).
- **Key results:**
  - K=5 clustering, silhouette 0.4327. c4 (trend-rider, 14.5% pool, n=122): centroid mono 0.530, peaks 30.94, ttp_rel 0.760.
  - c4 capturability: fwd_mfe_p50 6.65R, frac_reach_1R 1.000, frac_wrong_way 0.000. Failed §2 on mono 0.530 vs ≥ 0.55 (miss 0.020) and shape_tag "scattered" (87.7% of c4 trades hit 240-bar cap → final_r censored → scattered classification despite underlying heavy-right-tail MFE distribution).
  - c1 (43% of pool): missed clean_shape mono by 0.049; otherwise passed magnitude (1.70R), direction (reach_1R 0.742, wrong_way 0.060), shape_tag heavy_right_tail.
  - §14 anchor measured on filtered deployed 214-trade population (mono 0.576); v2.0 §2 floor 0.55 calibrated against filtered anchor — leaves no headroom on bare 842-trade signal where filters' ~0.05 mono uplift is absent.
- **Verdict:** HALT — STEP3_FAIL_NO_CAPTURABLE_ARCHETYPE. v2.0 as drawn doesn't reach Step 6 on bare KH-24 signal.
- **Cross-arc contribution:** 8 calibration items added — §2 monotonicity floor calibration, shape_tag definitions vs forward-window censoring, 240-bar window for slow 4H signals, §14 anchor population vs §15 pool floor structural mismatch, §17 `frac_wrong_way` Def B ratification, §16 Open-08 closure (pullback_magnitude_median non-degenerate at mode 0.31), §11 archetype-prior empirical refinement, per-pair n distribution stability concern. KH-24 v1.0 deployment unaffected.
- **Source closure doc:** [results/arc_kh24_v2/ARC_KH24_V2_RESULT.md](results/arc_kh24_v2/ARC_KH24_V2_RESULT.md).

### Arc 3 — `volatility_regime.d1_atr_top_decile.any.h_120` — CLEAN-NULL

- **Signal:** `TRIAL__volatility_regime__d1_atr_top_decile__any__h_120` (LCHAR registry entry 3). D1 ATR(14) in top decile of trailing 100 D1 bars per pair. 1H, h=120 (~5 trading days).
- **Methodology:** L_ARC_PROTOCOL v2.0.
- **Pool:** 2,568 trades / 28 pairs. 97.6% skip rate (regime-density signal under max-1-per-pair cap). 95th-percentile bars_held 120.
- **Key results:**
  - K=7 silhouette 0.4177 selected on 0.0021 margin (Open-12 — tie tolerance undefined).
  - Stepwise climber aggregate (clusters 2+4, 27.5% pool, n=707): mono 0.559, local_peaks 16.73, fwd_mfe_p50 3.34R, frac_reach_1R 83.6%, frac_wrong_way 38.3%, shape_tag bimodal.
  - Failed §2 on wrong_way (38.3% vs ≤30%) and shape_tag (bimodal not in allowed {tight_unimodal, heavy_right_tail}).
  - §11 row 7 admits bimodal as valid archetype with own exit policy (half-off at TP1, trail remainder); §2 excludes bimodal — internal protocol inconsistency.
  - Cluster 2 vs cluster 4 disparity: 3× difference in local_peaks (24 vs 8), 3.7× in pct_peak_and_collapse (0.126 vs 0.474) — same §11 row but structurally different. Aggregation masks distinction.
  - Final R distribution Stepwise aggregate: p25 −1.00, p50 +1.85R, p75 +3.80R.
- **Verdict:** CLEAN-NULL at Step 3. Zero archetypes pass §2 conjunctively. Verdict locked per discipline (no within-arc rescue, JL precedent).
- **Cross-arc contribution:** Open-12 (silhouette tie tolerance), Open-13 (§2 vs §11 bimodal incompatibility, highest priority), Open-14 (same-archetype aggregation can destroy capturable sub-clusters), Open-15 (SL-distance / hold-horizon asymmetry inflates frac_wrong_way; 2×ATR_1H on h=120 = 0.18σ of horizon, structurally too tight). Plus Open-07 evidence (Random walk and Peak-and-collapse §11 patterns over-specified).
- **Diagnostic tail (Arc 3D):** SL × aggregation sweep (3 SL × 2 aggregation modes = 6 cells) run post-closure to convert Open-13/14/15 from speculation to evidence; recorded at `results/arc_3d/ARC_3D_SUMMARY.md`.
- **Source closure doc:** [docs/archive/arc_results/ARC_3_RESULT.md](docs/archive/arc_results/ARC_3_RESULT.md) (post-move).

### Arc 4 — `bar_range_top_decile.neg.h_001` — CLEAN-NULL → FAIL (initial → rerun)

- **Signal:** `TRIAL__univariate_extreme__bar_range_top_decile__neg__h_001` (LCHAR registry entry 4). Top-decile 1H bar-range × bar close < open, long bias. 1H, h=1.
- **Methodology:** Initial closure under v2.1.1 (CLEAN-NULL on transaction-cost framing 2026-05-17). Rerun under v2.1.2 with per-pair p50 spread floors and full Step 6 deployment reckoning (FAIL Step 6 2026-05-18 — canonical verdict).
- **Pool (rerun, under p50 floors):** 10,893 trades, 28 pairs, 2020-10-01 → 2026-01-31. +129 trades (+1.2%) from spread-floor change due to exposure-cap path dependence.
- **Key results (rerun):**
  - K=4 silhouette 0.50. Cluster 1 (Stepwise climber, n=1,786, 16.4% pool) PASS Step 3 capturability at SL=3×ATR; composite 0.358; pre-peak mono rescue worked.
  - Pipeline E FAIL both clusters (AUC 0.54-0.56). Pipeline D1 PASS both at t=1 (c1 AUC 0.6487-0.6779 refit across F2-F7; c3 D1 AUC 0.6634).
  - Step 5 §9 admit-only stability PASS.
  - Step 6 §10 full-pool deployment FAIL: full-data ROI −76.98% at 0.20% risk over 4.5-year F2-F7 window. Max DD 76.98%, 1 day breaching 5% daily DD threshold (account-closure event under 5ers).
  - Trade-flow decomposition: admit pool 57.2% × +0.125R = +0.072R contribution; reject pool 32.2% × −0.232R = −0.075R; early-exit pool 10.6% × −0.685R = −0.073R. Costs 2× the admit edge.
  - F6 collapsed: 1,740 signals × negative expectancy → −55% ann ROI, 42% DD, terminal equity 0.5957.
- **Verdict:** FAIL at Step 6 under §10 deployment reckoning. Original CLEAN-NULL closure (transaction-cost framing) was correct in disposition but for incomplete reason; rerun reveals deeper architectural failure mode.
- **Cross-arc contribution:** First arc to expose **Pipeline D1 admit-only ≠ deployment** failure mode. Open-22 (full-pool gate at §9 or earlier, HIGH), Open-23 (§8 Pipeline D1 cost-language correction, MEDIUM), Open-24 (per-archetype pre-t SL multiplier, MEDIUM). Locked spread floor file replaced with per-pair p50 from HistData audit. Spread audit becomes Phase 0 prerequisite. Convention (b) MTM DD as protocol-default. F1 structural leakage cross-arc item.
- **KH-24 impact:** KH-24 doesn't load spread floor file (uses raw MT5 per-bar spread). Real-spread reconciliation on KH-24's audit-window-overlap trades: 1.509% under_pct_equity. Fold 7 published ROI +1.92% → ~+1.28% after correction → pass-viable (was pass-deployable). Live deployment unchanged.
- **Source closure docs:** [docs/archive/arc_results/ARC_4_RESULT.md](docs/archive/arc_results/ARC_4_RESULT.md) (initial), [docs/archive/arc_results/ARC_4_RERUN_RESULT.md](docs/archive/arc_results/ARC_4_RERUN_RESULT.md) (canonical rerun verdict).

### Arc 5 — `mtf_alignment.2_down_mixed.kijun.h_120` — SHELVED Step 6 FAIL

- **Signal:** LCHAR registry entry 5. Byte-identical signal definition to Arc 2; only time exit horizon differs (registry h=24 overridden to h=120 for Arc 5 — F1).
- **Methodology:** L_ARC_PROTOCOL v2.1.1.
- **Pool:** 12,262 trades initially (sha256-identical to Arc 2 redo2 baseline confirming F2); 12,348 under new per-pair p50 spreads.
- **Key results:**
  - K=4 silhouette 0.4834. Cluster 1 (Stepwise climber, 18.63% pool, n=2,285) PASS Step 3 at SL=3×ATR via v2.1.1 pre-peak shift + SL sweep + capturability composite (0.593). First non-self-test pass of v2.1.1's anchor-rescue design intent. Cluster 3 (13.91% pool) PASS at SL=2×ATR, composite 0.370.
  - Pipeline E FAIL both (best AUC 0.566 after full Step A/B/C cascade). Pipeline D1 PASS both at t=1 (c1 AUC 0.636, c3 AUC 0.640).
  - Step 4b F9 threshold grid extended to {0.10, ..., 0.40} for D1 (under-specified protocol detail). Selected thresholds: c1 0.20, c3 0.15.
  - Step 5 under old spreads PASS for both clusters (7/7 positive folds, c1 worst-fold +36.5% ann ROI, c3 +8.80%).
  - Step 5 under new spreads: c1 §9 DD ratio 2.34 FAIL; c3 PASS at DD ratio 1.91 but ROI/DD ratio 0.58 < 5/8 (no risk level clears PASS-DEPLOYABLE).
  - Step 6 PR-2 (per-archetype §11 row 2 exit: MFE-lock at 1R + trail 0.75R) recovered c1 DD ratio 2.34 → 1.17. Admit-set R remained positive: +0.14 to +0.21 per fold.
  - Pipeline D1 cost decomposition: early-exit (SL hit before bar 2) 7.9% × −0.74 to −1.11R; admit 13.2-14.7% × +0.14 to +0.20R; reject + close at bar 2 ~78% × **−0.46R** (vs unconditional bar-2 R +0.025). Rejected pool adverse-selected.
  - Full-strategy ROI per fold negative at every risk level (0.10% to 0.50%). Best (c1 at 0.10%): worst-fold −5.74%, mean −5.61%, DD 6.31%, total compounded ROI −26.05% over 4.5 years.
- **Verdict:** SHELVED — Step 6 FAIL. No ship candidate. The signal has real path-shape edge on admits; Pipeline D1 cannot extract it.
- **Cross-arc contribution:** **P-§9-FRAMING** (full-pool R not admit-only R, P0); **P-D1-VIABILITY** (>5% bar-0/1 SL-hit rate flags D1 unsuitability at Step 4, P0); **P-D1-REJECT-BIAS** (rejected-pool selection bias documented, P0); plus P-F9-RESELECT, P-CLUSTERING-LEAKAGE, P-SPREAD-FLOOR, P-§11-MATCH-FORMULA. Second Pipeline D1 admit-only ≠ deployment failure (with Arc 4 rerun). Spread validation methodology now in repo. v2.1.1's anchor-rescue mechanism (pre-peak metrics + SL sweep + composite) empirically validated.
- **Reopen conditions:** Pipeline E re-evaluation with richer feature set; alternative pipeline shape that doesn't bear rejected-pool cost; full Steps 2-4 retrain under new spreads; §9 framing fix.
- **Source closure doc:** [docs/archive/arc_results/ARC_5_RESULT.md](docs/archive/arc_results/ARC_5_RESULT.md).

### Arc 6 — failed-breakout reversal long (out-of-registry) — DIES Step 4

- **Signal:** Out-of-registry structural spec. Trigger: closed-bar break below 20-bar swing low (magnitude ≥ 0.25×ATR), reclaim within 5 bars on a bullish-close bar. 4H, long-only.
- **Methodology:** L_ARC_PROTOCOL v2.1.2.
- **Pool:** 1,564 trades / 28 pairs. Cap-binding 17.65%. KH-24 co-fire 0 (structural — Arc 6 requires close > open, KH-24 close < open). Spec erratum locked at Step 1 — literal `swing_low_N` definition unsatisfiable; corrected to `min(low[t-N-M..t-M-1])`.
- **Key results:**
  - K=4 silhouette 0.4795. c0 (Stepwise-boundary, 21.4% pool, SL=2×ATR, composite 0.384) and c2 (Stepwise climber, 15.5% pool, SL=3×ATR, composite 0.616) PASS Step 3.
  - c1 (early_peak_family, 30.5% pool): DIES — mfe_p50 max 0.29R at SL=4×ATR, structural magnitude failure.
  - c3 (multi-boundary, 32.7% pool): DIES — reach_1R 0.697 vs 0.70 floor (0.003 margin, within sampling noise on n=511).
  - Pipeline E FAIL both (c0 best 0.600; c2 best 0.5897 forward selection of 2 features). RF/Logistic gap small (positive ~0.01-0.08 for c0, slightly negative for c2). Feature-set bound, not non-linearity bound.
  - Pipeline D1 mechanically PASS AUC gate: c0 at t=4 AUC 0.602, c2 at t=1 AUC 0.630 (growing to 0.711 at t=10).
  - Threshold sweep: neither cluster achieves recall ≥ 0.60 at any threshold ∈ {0.40, 0.50, 0.60, 0.70}. Both fall to max-F1: c0 precision 0.333 recall 0.009 (~3 trades admitted); c2 precision 0.250 recall 0.004 (~1 trade). Sub-1% deployability.
  - Step 5/6 noise-dominated at ≤ 1 trade/fold; not executed.
- **Verdict:** DIES at Step 4 — mechanical PASS at §8 AUC gate, substantive FAIL on deployability via threshold sweep collapse.
- **Cross-arc contribution:** Open-21 (Step 4 deployability gate — strict mode: max-precision subject to recall ≥ 0.60; max-F1 fallback triggers cluster-dies). Open-17 expansion (Tiebreak 1 noise floor). Reach_1R floor noise sensitivity flagged. **Path-so-far signal grows with t** (c2 D1 0.630 → 0.711) — supports deferred-entry or two-stage approaches. v2.1.2 extended Stepwise local_peaks ceiling 5-50 was load-bearing for c2 centroid (32.5).
- **Note (per Arc 10 EXP-05):** Arc 6 c2 archetype is **Stepwise**, not V-shape (previously narrated as V-shape in some downstream docs; corrected 2026-05-18 with EXP-05 evidence).
- **Source closure doc:** [docs/archive/arc_results/ARC_6_RESULT.md](docs/archive/arc_results/ARC_6_RESULT.md).

### Arc 7 — liquidity sweep + reclaim long (out-of-registry) — CLEAN-NULL Step 4

- **Signal:** Out-of-registry. Trigger: `swing_low_N = min(low[t-N..t-1])` with N=20; `low[t] < swing_low_N`; `close[t] > swing_low_N`; `swing_low_N − low[t] ≥ 0.25×ATR`; `close[t] > open[t]`; reclaim-strength ratio ≥ 0.5; ≥ 20 bars since last signal. 4H, long-only, h=240.
- **Methodology:** L_ARC_PROTOCOL v2.1.2.
- **Pool:** 1,288 trades / 28 pairs; smallest per-pair n=32. Bars-held p50 24, p95 240; cap-binding 17%. KH-24 co-fire 0. Arc 6 cross-arc co-fire 5.4% (different reference windows — legitimate co-occurrence).
- **Key results:**
  - K=4 silhouette 0.4263 selected (tie tolerance held K=4 over K=6 at 0.0021 margin — exact Open-12 closure case).
  - 3 units survive §2 conjunctively at Step 3: c1 (V-shape recovery, 14.4% pool, SL=4×ATR, composite 0.617, fwd_mfe_p50 3.45R, wrong_way 0.000); c3 (V-shape recovery, 28.3% pool, SL=2×ATR, composite 0.413); agg_c1_c3 (42.7% pool, SL=4×ATR, composite 0.378). All shape_tag `unclassified` — v2.1.2's `≠ scattered` floor was load-bearing.
  - Pipeline E: 0/3 clear 0.65 gate (c1 mean AUC 0.484, c3 0.512, agg 0.536). Pipeline D1: 0/3 clear 0.60 gate (c1 0.420, c3 0.518, agg 0.496).
  - c1's selected SL=4×ATR drove base success rate to 0.778 — only 41 negatives in n=185; class compression starved Step 4.
  - Steps 5-6 not reached.
- **Verdict:** CLEAN-NULL at Step 4. **First capturable-not-extractable closure of record.** Signal class has confirmed forward-geometry edge (PASS §7 with 3 V-shape units) but no in-protocol feature set predicts cohort membership.
- **Cross-arc contribution:** New closure category (capturable-extractable gap) for v2.2 consideration. SL-selection vs class-imbalance tension at Step 3 / Step 4 boundary. Open-04 external features escalation gets first empirical case. v2.1.2 `≠ scattered` floor validated. §11 Stepwise pullback ≤ 0.5R ceiling unresolved (c1 was test case at 0.567 — geometrically Stepwise except for pullback).
- **Source closure doc:** [docs/archive/arc_results/ARC_7_RESULT.md](docs/archive/arc_results/ARC_7_RESULT.md).

### Arc 8 — PR-HHHL long — HALT_DEPLOYMENT

- **Signal:** Pullback-and-resume in HH/HL uptrend (`signal_pullback_resume_hhhl_long_v0.1`). 4H, long-only.
- **Methodology:** L_ARC_PROTOCOL v2.3 stack (v2.1.2 base + v2.2 + v2.3 amendments).
- **Pool:** 1,327 trades / 28 pairs. KH-24 co-fire 0.0%.
- **Key results:**
  - K=4 silhouette 0.4762. c1 (V-shape, n=177, 13.3% pool, SL=4×ATR), c3 (V-shape, n=405, 30.5% pool, SL=2×ATR), agg (n=582) PASS Step 3.
  - Step 4 PASS: c1 E AUC 0.697 (clears 0.65), c1 D1 AUC 0.637 at t=1 (clears 0.60). c3 and agg die per v2.2 §3 (max-F1 fallback).
  - Step 5 WFO admit-only PASS: Pipeline E worst ROI +18.66%, DD 1.00%, Sharpe 1.44, n=103; Pipeline D1 worst ROI +25.73%, DD 0.54%, Sharpe 1.14, n=133.
  - Step 5 WFO full-pool FAIL: Pipeline E worst ROI −13.21%, DD 15.58%, n=751; Pipeline D1 worst ROI −14.69%, DD 19.02%, n=943.
  - Failure mechanism: both pipelines admit 70-89% of full Step 1 OOS pool. Per-cluster mean_r at SL=4×ATR: c0 −0.46R (23.8% pool); c1 +2.59R (13.3%); c2 −0.47R (32.3%); c3 −0.10R (30.5%). c1 is genuinely +2.59R/trade — only 13.3% of pool. Non-c1 admissions drown signal.
  - Post-Step-5 diagnostics: c1 vs c2 entry-feature mean overlap > 0.78; multiclass RF c1 one-vs-rest AUC 0.547, precision@recall=0.60 = 0.149 vs base rate 0.133 (zero lift). c1 and c2 mechanically indistinguishable at entry. Mid-path classifiers (D1 at t≤12) climb to AUC 0.755 but plateau below 0.40 VIABLE precision threshold.
- **Verdict:** HALT_DEPLOYMENT. No portfolio candidate from §10 ship gates.
- **Cross-arc contribution:** **Third consecutive Open-22/23/24 admit-only-vs-deployment failure** (Arcs 4 RERUN, 5, 8). v2.4 §1.5 **entry-separability gate proposed** as Open-25: pre-Step-1 multiclass RF on smoke pool, target cluster one-vs-rest precision@recall=0.60 ≥ 0.30; below → arc auto-halts before Step 1 simulation compute. c1 V-shape recovery FG-weak archetype logged for Open-05 portfolio composition (Pipeline E Sharpe 1.44 / worst DD 1.00% / worst ROI +18.66% in isolation). `pullback_depth_atr ≥ 1.0` filter improves aggregate +68% but doesn't separate c1 from c2.
- **Source closure doc:** [results/l_arc_8/ARC_8_CLOSURE.md](results/l_arc_8/ARC_8_CLOSURE.md) (co-located).

### Arc 9 — IB-trend (compression geometry → directional break) — STEP_4_KILL_REAFFIRMED

- **Signal:** Inside-bar trend compression-break long (`signal_inside_bar_break_trend_long_v0.1`). 4H, long-only.
- **Methodology:** L_ARC_PROTOCOL v2.1.2 + v2.2 + v2.3 amendments. Held-open lifecycle through 8 diagnostic experiments.
- **Pool:** 2,153 trades. Step 1 sub-gates all green.
- **Key results:**
  - Original Step 4 KILL 2026-05-18: Pipeline E AUC 0.511 (chance), Pipeline D1 AUC 0.626 with threshold-sweep recall 0.003.
  - Held-open experiments 1-3 confirmed cohort reality: Step 5 oracle (cluster 0 only, post-hoc cluster identity) +60% ann ROI / 0% DD across 7/7 folds; Step 5 raw baseline (no filter) −29% worst-fold / 63% DD; calibration recovery OUTCOME_B (rank-bound, not calibration-bound).
  - Experiments 4-7 reported PASS-DEPLOYABLE under "Pipeline E retry" with 8 D1-lagged + 4 session/time features (classifier AUC 0.7508); lookahead audit GREEN on 8 dimensions; scaled-risk recommendation 1.0% per-trade for +41.45% ann ROI at 2.52% DD.
  - **External audit detected producer-level lookahead**: `d1_bars_since_swing_low` and `d1_bars_since_swing_high` used ±10-bar centred swing detector at D1 frame level. The `merge_asof` join was clean; the values inside joined rows depended on up to 10 future D1 bars relative to each signal's entry time. Original audit checked join-level causality and end-to-end probability reproduction; did not check producer-level causal scope within each row.
  - Causal patch (Experiment 8, commit 5b6c547): replaced swing features with confirmed-swing variants requiring 10-day forward confirmation. Patched AUC dropped 0.7508 → **0.5190** (LGBM) / 0.5551 (RF). Forced WFO on patched classifier: Candidate A full-data ROI −0.13% / DD 13.38%; Candidate B +0.08% / DD 22.65%. §8 gate verdict confirmed economically.
- **Verdict:** STEP_4_KILL_REAFFIRMED. Cohort real (Step 3 + Step 5 oracle ceiling unaffected) but unreachable on causally-clean in-protocol features.
- **Cross-arc contribution:** **Producer-level causal audit dimension** now standard for every classifier audit. Distinct from join-level + e2e dimensions. Standard mathematical definitions of trading concepts (swing detection, pivots, ZigZag, centred smoothers) are non-causal by default; dispatch instructions must specify causality at producer level. v2.x §8 D1 feature-budget expansion WITHDRAWN; v2.x §3 threshold-grid replacement WEAKENED; Amendment 1 (producer-level audit) promoted to highest priority. "Features over classifiers" methodology lesson withdrawn (its empirical case was the leaked features). KH-24 pre-PR producer-level audit required before v2.x lands.
- **Source closure doc:** [results/l_arc_9/ARC_9_CLOSURE.md](results/l_arc_9/ARC_9_CLOSURE.md) (co-located). Incident note: [results/l_arc_9/INCIDENT_2026_05_19_ARC_9_PRODUCER_LEAK.md](results/l_arc_9/INCIDENT_2026_05_19_ARC_9_PRODUCER_LEAK.md).

### Arc 10 — DLR (D1 swing-low rejection long) — STEP_4_HALT

- **Signal:** D1 swing-low rejection long (out-of-registry, `signal_spec_d1_swing_low_rejection_long_v0.1.md`). 4H entry on D1 swing-low rejection. Long-only.
- **Methodology:** L_ARC_PROTOCOL v2.3. **First arc end-to-end under v2.3** (5-step pipeline: 1 plumbing, 2 clustering, 3 capturability, 4 extractability, 5 WFO; halt at end of Step 4).
- **Pool:** 802 trades. KH-24 co-fire 0%. 5/5 lookahead spot-checks. 3/3 D1-lag NaN-perturbation. Byte-identical determinism.
- **Key results:**
  - K=3 silhouette 0.4525. c1 V-shape recovery PASS Step 3 at SL=3.0×ATR, composite 0.4934, fwd_mfe_p50 3.08R, wrong_way_pp 0.
  - Step 4 disjunctive §8 near-miss FAIL: c1 E AUC 0.6296 (margin −0.0204), D1 AUC 0.5897 (margin −0.0103). Both Path A near-miss < 0.03.
  - §16a Path A compound reading applied (Step 4 as one §8 gate → HALT). Strict reading available; compound-vs-strict ambiguity flagged for v2.4.
  - **Post-closure research** (over §16a at chat-side direction, NOT a re-disposition):
    - EXP-01 AUC bootstrap: 95% CIs straddle thresholds; P(joint either clears) = 40.5%; P(realisable AUC ≥ 0.65) = 12.5%. Arc 10 alone in noise zone.
    - EXP-02 HTF ablation: `L1_minus_L0_atr` (D1 HL slope magnitude) carries **116%** of HTF LOO drop; age features 18%. Load-bearing single feature.
    - EXP-03 threshold scan: triple-pass threshold E ≥ 0.536 (Arc 7 binding); pair-pass at 0.600. 0 documented FP across arc history at any relaxed threshold.
    - EXP-04 fold-2 regime: **fold-2 date correction 2023-07 → 2024-06** (not Q2 2022). No entry-time regime descriptor places fold-2 ≥ 1.5σ from cohort mean.
    - EXP-05 cross-arc V-shape pool: Arc 7 c3 + Arc 10 c1 (n=593) AUC **0.6348** (gap −0.015 to 0.65). +0.029 over Arc 10 alone, +0.140 over Arc 7. **Arc 6 reclassified Stepwise**, not V-shape — blocked for pool.
    - EXP-06 Open-04 probes: D1 Kijun distance, session dummies both negative on Arc 10 alone. Informational; needs Arc 8/9/11 reproduction.
    - **WFO pair (base + oracle c1):** Base Sharpe −1.29, expectancy 0.40R, Profit factor 0.19, admits/fold 1.4 (4/8 admit zero). Oracle Sharpe **4.61** [CI 3.13-6.09], expectancy 1.55R, win rate 65%, Profit factor 8.10, max DD 0.50%, admits/fold 3.9. Gap +5.90 Sharpe / +85.4 Calmar / +602% relative.
    - Synthesis: BUILD clusterifier (both material thresholds cleared by wide margin) with **explicit DO NOT DEPLOY** (oracle is upper bound; realisable lift = fraction of +5.90 gap; per EXP-01, P(realisable AUC ≥ 0.65) = 12.5%).
- **Verdict:** STEP_4_HALT (unchanged). Disposition reasoning §16a Path A compound interpretation.
- **Cross-arc contribution:** Second V-shape near-miss (with Arc 7). Cross-arc V-shape clusterifier build is leading v2.4 candidate. Two material corrections: Arc 6 reclass Stepwise; fold-2 date 2023-07/2024-06. Open-06 (AUC threshold) WEAKENED. Open-04 (external features) DEFERRED pending Arcs 8/9/11. Vocabulary additions: Reverse FE, Classifier path, Filter path, §16a Path A compound-vs-strict ambiguity. Recommended next dispatches: (1) Reverse-FE diagnostic, (2) cross-arc clusterifier build, (3) filter-path probe, (4) v2.4 calibration packet.
- **Source closure doc:** [docs/archive/arc_results/ARC_10_RESULT.md](docs/archive/arc_results/ARC_10_RESULT.md). Experiment synthesis: [results/l_arc_10/experiments/ARC_10_EXPERIMENT_SYNTHESIS.md](results/l_arc_10/experiments/ARC_10_EXPERIMENT_SYNTHESIS.md).

### Arc 11 — SHB (swing-high breakout trend long) — CLOSED-HALT (§16a Path A)

- **Signal:** Swing-high breakout trend long (`signal_swing_high_breakout_trend_long_v0.1`, SHB). 4H, long-only.
- **Methodology:** L_ARC_PROTOCOL v2.3.
- **Pool:** 2,299 trades. Determinism ✓; right-edge offset 4; cap-bind 15.6%.
- **Key results:**
  - K=4 silhouette 0.469. 3 V-shape recovery units (c1, c3, agg_c1_c3) survive Step 3.
  - Step 4 FAIL: 0/3 clear disjunctive §8 AUC gate. Best 0.5728 (agg_c1_c3 D1 t=5); margin 0.027 (< 0.03 absolute → §16a Path A near-miss).
  - **Post-closure experimental work** (off-protocol, documentation only; no queue/registry/protocol mutation):
    - Exp 1 S5 oracle (post-hoc cluster identity): A) c1 raw SL=3.0 — worst ROI +101.39%/yr / 2.48% DD / 42 trades / PASS oracle; B) agg + D1 t=5 SL=3.0 — worst ROI +26.30%/yr / 4.95% DD / 114 trades / PASS oracle.
    - Exp 2 S5 no-oracle (live classifier): C) E → c1 t=0.50 — worst −20.22% / 33.46% DD / 1 trade / FAIL; D) E → D1 t=5 cascade — worst −22.83% / 33.46% DD / 4 trades / FAIL. Oracle premium: c1 leg −154.48pp, agg leg −56.09pp. **S4 AUC gate vindicated economically.**
    - Exp 3 filter-diagnosis: only **delayed entry t=3** moves AUC (mean 0.6404, 4/6 folds clear 0.65, Δ +0.117). Multi-TF (D1+1H) +0.024 dead; reframed target (reach_1R, mfe≥2R) negative.
    - Exp 4 signal-improvement sweep (7 stages): best candidate `DE t=7 + dynamic SL 4a`: sign-consistency ✓, worst-fold ROI +3.11% ✓, mean-fold +17.23%, DD 18.03% ✗, trades/fold 16 (borderline), DD/ROI 1.04 ✗. **Not deployable.**
  - Strike list (empirically retired for SHB long 4H): Pipeline E on entry-bar features; Pipeline D post-entry on c1; multi-TF feature extension; reframed supervision target; trigger-bar mechanical filters; sizing without filtering; "relax AUC gate when mfe_p50 ≥ 3R"; DD-relaxation amendment.
- **Verdict:** CLOSED-HALT per §16a Path A — numeric near-miss (Step 4 disjunctive E∨D1 fail, margin 0.027).
- **Cross-arc contribution:** **Second capturable-not-extractable instance** (pairs with Arc 6 Stepwise; distinct from Arc 7 + Arc 10 V-shape pair). Confirmed pattern of cohort with real structural edge that entry-time features can't resolve. **Pipeline DE (Deferred-Entry)** proposed as v2.4 candidate: enter at bar t post-signal or not at all (no second-stage classifier); features = path-so-far at bar t; gate AUC ≥ 0.60 + sign-consistency + pre-t SL filter rate < 40%. Distinct from D1 (post-entry decision mid-trade). **`min_observation_bars` registry parameter** also proposed. "Timing > features as extractability lever" — post-signal price action carries discriminating information; multi-TF and feature redesign do not. DD structural for V-shape cohorts (15-20% per-fold DD regardless of filter/classifier/trail). AUC ceiling structural for 4H entry-bar features (three independent feature regimes capped below 0.55).
- **Source closure doc:** [results/l_arc_11/ARC_11_CLOSURE.md](results/l_arc_11/ARC_11_CLOSURE.md) (co-located).

---

## KH-24 — separate track (deployed)

KH-24 is not an L_ARC; it is the deployed system that L_ARC arcs benchmark against. It exists on a separate development lineage (KH series → KH-22 → KH-24) whose endpoint passed the WFO gate and went live.

### Signal spec (locked)

```
Signal:     kb_exhaustion_bar (c1–c6, c8, c9)
            c7 DISABLED — volume gate removed
Direction:  Long only
Timeframe:  4H with D1 regime filter (one-day lag)
Pairs:      28 FX currency pairs
Broker:     5ers
Data:       data/4hr/, data/daily/, data/1hr/
Entry:      Bar N+1 open after signal on bar N close
Stop:       Entry price − 2.0 × ATR(14) [entry price anchor]
Trail:      Activates at close ≥ entry + 2.0 ATR (close-based)
            1.5 ATR behind highest close, bar-close updates only
Exits:      trailing_stop | kijun_d1 | stoploss
Risk:       1.0% of current reset floor balance (KH-24 era; L arc uses 0.5%)
Filters:    exposure cap=2; 1H CIR T=0.28
Spread:     Per-bar MT5 data — never hardcoded
D1 align:   One-day lag — each 4H bar sees prior calendar day's D1 close
```

### WFO gate (locked)

| Criterion | Threshold | Actual | Result |
|---|---|---|---|
| Worst-fold ROI | > 0.0% | +1.92% (F7) | PASS |
| Worst-fold DD | < 8.0% | 6.37% (F1) | PASS |

Per-fold (7 anchored expanding folds, Oct 2020 → Jan 2026):

| Fold | OOS Window | Trades | ROI | DD | Win% | Mean R | Gate |
|---|---|---|---|---|---|---|---|
| 1 | 2020-10-01 → 2021-07-01 | 41 | +13.35% | 6.37% | 43.9% | +0.341 | PASS |
| 2 | 2021-07-01 → 2022-04-01 | 36 | +9.63% | 4.45% | 58.3% | +0.278 | PASS |
| 3 | 2022-04-01 → 2023-01-01 | 25 | +11.90% | 4.43% | 56.0% | +0.479 | PASS |
| 4 | 2023-01-01 → 2023-10-01 | 32 | +3.32% | 3.80% | 46.9% | +0.118 | PASS |
| 5 | 2023-10-01 → 2024-07-01 | 23 | +6.23% | 3.09% | 52.2% | +0.283 | PASS |
| 6 | 2024-07-01 → 2025-04-01 | 30 | +3.24% | 5.03% | 43.3% | +0.140 | PASS |
| 7 | 2025-04-01 → 2026-01-01 | 27 | +1.92% | 4.06% | 51.9% | +0.082 | PASS |

214 trades across all folds. 7 of 7 folds positive. First gate pass in the KH arc.

### Post-Arc-4 audit reconciliation

Arc 4's HistData spread audit triggered a real-spread reconciliation against KH-24. KH-24 doesn't load `spread_floors_5ers.yaml` — it uses raw MT5 per-bar spreads. Audit-window overlap (2024-01 → 2026-01): 69 of 553 trades evaluated, mean under_R 0.02187, total under_pct_equity 1.509%. Fold 7 published +1.92% → ~+1.28% under real-spread reconciliation. Under §10 pass-deployable's worst-fold ann ROI ≥ 5% bar, real-spread KH-24 falls to **pass-viable** (positive but below 5% deploy floor). Live deployment posture unchanged — the live system pays whatever the broker charges, not what the backtester modeled.

### Deployment lineage

KH-24 is the endpoint of the KH series (KH-22 → KH-24). KH-22 was the prior gate-failing iteration; the change to KH-24 was tightening the `h1_last_bar_close_in_range` threshold from 0.624 to 0.28 with no other modifications. Threshold was selected from a sweep of KH-22 OOS trades and validated by the WFO under proper compounded equity and DD calculation. Source: `results/kh24/PHASE_KH24_RESULT.md`. Earlier KH iterations (KH-23, KH-25, KH-27, KH-28, KH-29) are documented in the project memory and `CHANGELOG.md`; KH-25 FAIL, KH-27 preflight KILL, KH-28 entry-side STRUCTURAL, KH-29 exit-side AMBIGUOUS (pivot to B2/B3). Only KH-24 passed the gate.

KH-24 is locked, gate-passing, deployed on Contabo VPS / 5ers. Out of scope for L_ARC modification.

---

## Pre-v2.0 work (summary only)

Two threads predate the L_ARC v2.0 protocol.

**NNFX-era investigations** (pre-2026-04). The original system framing — full NNFX stack as strategy (C1 + C2 + Baseline + Volume + Exit confirmation + ATR) — was pursued through Phases A through D-6F across many indicators and configurations. 57 exit indicators tested, zero passed. The full NNFX stack as a strategy was permanently eliminated. The C1-only sweep (57 indicators), the Phase B / C / D series, the Phase 6 exit indicator research, and the Phase 8 execution-truth scratch all sit under `docs/archive/nnfx_era/` after this dispatch. Their content is historical; the lessons survive as project-permanent invariants in `docs/GOLDEN_STANDARD_LOGIC.md` (slimmed to invariants after KH arc closure).

**L characterization arc / L0 / L6.0** (2026-04 → 2026-05-13). The L atlas was built bottom-up: L0 methodology lock; L1 univariate; L2 multi-timeframe; L3 cross-pair; L4 conditional; L5 synthesis & registry (`docs/LCHAR_TOPN_REGISTRY.md` top 5 candidates by DSR > 0.95). L6.0 was the verbatim-as-gate framing for testing registry candidates (no filter rescue, §9). Arcs 1 and 2 ran under L6.0 and closed FAIL. The L6.0 framing was superseded 2026-05-13 by `L_ARC_PROTOCOL.md` v1.0 (now archived at `archive/L_ARC_PROTOCOL_v1_0.md`), which restructured signal testing into the six-step extractability pipeline that v2.0 (path-shape clustering) then rewrote structurally. Pointers to archived materials: `archive/L_ARC_PROTOCOL_v1_0.md`, `archive/L_ARC_OPERATIONAL_SPEC_v1_0.md`, `archive/L_ARC_PROTOCOL_v1_1_AMENDMENT.md`, `archive/L_ARC_PROTOCOL_v1_2_AMENDMENT.md`, `docs/archive/protocol/L6_0_METHODOLOGY_LOCK.md` (post-move), `docs/archive/nnfx_era/L_ARC_PLAN.md`, `docs/archive/nnfx_era/PHASE_L6_ARC*_OPEN.md` (4 docs), `docs/archive/arc_results/PHASE_L6_ARC2_P3_RESULT.md`, `docs/archive/nnfx_era/CANDIDATES.md`.

The cross-arc evidence base for the v1.x → v2.0 transition included v1.3 calibration (`results/v1_3_calibration/`) and the v2.0 archetype + predictability diagnostics (`results/v2_0_diagnostic/`, `results/v2_0_predictability/`). The Lω discovery track (`results/lomega/`) runs parallel to L_ARC and inverts the question: from the full dataset, find which bar conditions at t=0 predict clean forward paths regardless of any specific signal.
