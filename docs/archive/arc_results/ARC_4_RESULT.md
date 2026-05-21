# Arc 4 — CLEAN-NULL on transaction-cost truth

> Status: CLOSED 2026-05-17
> Verdict: CLEAN-NULL at §9 retroactive fail under real-spread reconciliation
> Signal: `TRIAL__univariate_extreme__bar_range_top_decile__neg__h_001`
> Protocol: L_ARC_PROTOCOL v2.1.1
> Result location: `results/l_arc_4/`

---

## Headline

Arc 4 produced the strongest L arc result to date through Step 5: first arc to pass Step 3 (capturability), first to pass Step 4 D1 extractability (RF AUC 0.667), first to pass Step 5 §9 stability across F2-F7 with refit-corrected leakage. Pass-deployable at 0.20% risk under convention (b) MTM DD with daily DD constraint and modeled exit spread.

A parallel spread audit against HistData tick data (2024-01 → 2025-12, 28 pairs) revealed the locked `configs/spread_floors_5ers.yaml` 0.1 pip uniform floor under-models real first-5-minute execution-bar spreads by factors of 3x (EUR/USD) to 48x (GBP/NZD). 44.59% of Arc 4 entries and 42.74% of exits hit the floor.

Applying real-spread reconciliation: mean additional cost 0.06574 R per trade (6.5x the 0.01R modeled in Step 5B-spread); total under-modeled equity cost across 3,898 evaluated trades = **128.1% under_pct_equity**.

Under real spreads, F6 worst-fold annualised ROI flips from +10.08% to approximately −3%. §9.A sign consistency retroactively fails. Arc 4 closes CLEAN-NULL on transaction-cost truth.

This is a fundamentally different failure mode than prior arcs. The signal is real. The classifier extracts it. The exit policy is valid. The methodology survives. The cost model was wrong, and the cost model is the protocol's responsibility.

---

## What Arc 4 is

| Element | Value |
|---|---|
| Signal | bar_range top decile of trailing 100 1H bars AND signal bar close < open |
| Direction | Long only |
| Timeframe | 1H |
| Pool | 10,764 trades across 28 FX pairs, 2020-10-01 → 2026-01-31 |
| Cluster (survivor) | Cluster 1 — Stepwise climber (near-miss), n=1773, 16.5% of pool |
| R-frame | 1R = 3.0 × ATR(14)_1H |
| Pipeline | D1 (deferred-identification, classifier at bar t=1) |
| Classifier AUC | 0.667 (RF, gate 0.60) |
| Exit policy | §11 row 2 Stepwise climber (MFE-lock at 1R, trail 0.75R from new high) |
| Threshold | 0.1647 (= cluster base rate) |
| Risk | 0.20% (pre-real-spread reconciliation) |

Cluster 3 (Stepwise slow climber, R=4×ATR) died at §9 on F6 sign flip + 31.67% DD blowup. F1 evaluation skipped — pool starts at F1 OOS start, no honest WFO training data (cross-arc v2.2 item).

---

## Step-by-step verdict

| Step | Gate | Result | Notes |
|---|---|---|---|
| 1 — Plumbing | Pool ≥ 500, deterministic, no lookahead | PASS | 10,764 trades, 17.3% cap-binding (under §5 20% threshold), all gates clean |
| 2 — Clustering | Silhouette ≥ 0.30, K-selection | PASS | K=4, silhouette 0.50, four well-separated clusters |
| 3 — Capturability | ≥ 1 cluster passes §2 floors | PASS | Cluster 1 PASS at SL=3×ATR (composite +0.36); Cluster 3 PASS at SL=4×ATR (composite +0.60). v2.1.1 pre-peak mono rescued cluster 1 (+0.055 delta over full-window) |
| 4 — Extractability | RF AUC ≥ 0.65 E OR ≥ 0.60 D1 | PASS | Pipeline E FAIL both clusters (AUC 0.54-0.56). Pipeline D1 PASS both at t=1 (cluster 1 AUC 0.6674, cluster 3 AUC 0.6634) |
| 5 — Stability | §9 sign + size + DD gates | PASS (modeled) → FAIL (real spreads) | Cluster 1 PASS under modeled spreads; F6 sign flip under real-spread reconciliation |
| 6 — WFO truth | §10 pass-deployable / viable | NOT RUN | Closed on Step 5 retroactive fail before engine WFO |

---

## The four corrections to Step 5

In sequence, Step 5 evolved through cumulative corrections to the post-hoc simulator:

1. **Step 5A — D1 threshold re-sweep.** Original {0.40, 0.50, 0.60, 0.70} grid (Pipeline E's grid applied to D1) produced near-zero recall on minority-class targets. Extended to {base_rate, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50} with selection rule unchanged. Cluster 1 selected threshold 0.1647 = base rate. Cross-arc backlog item.
2. **Step 5C — per-fold classifier refit (leakage correction).** Original Step 4 classifier trained on full pool via random-shuffle CV. Refit per fold using anchored-expanding training data. F2-F7 admission overlap 88-96%, per-fold OOS AUC delta ≤ 0.008. Worst-fold ROI haircut 14% relative. F1 untestable — pool starts at F1 OOS start. Cross-arc v2.2 structural item.
3. **Step 5B-refit — convention (b) mark-to-market DD.** Original Step 5 used convention (a) closed-trade PnL ordering, which understates real account DD by 14-63% due to concurrent positions across 28 pairs. Convention (b) hourly MTM is production-truth gate. F3 inflated +62.69%, F6 +33.39%.
4. **Step 5B-spread — exit spread application.** Original Step 5 erroneously skipped exit-bar spread per prompt instruction. Re-applied S/2 exit spread per `SPREAD_SEMANTICS_LOCK.md` round-trip convention. Cost ~0.01R per trade on modeled floor — ROI haircut ~2.5 pp annualised at 0.20% risk.

At end of corrections, cluster 1 stood at: worst-fold ann ROI +10.08% (F6), worst-fold DD 7.60% (F6), daily DD p99 1.63%, deployable at 0.20% risk with headroom on ROI / tight on DD. F6 was the visibly weakest fold across every measurement but cleared every gate.

---

## The fifth correction — real-spread reconciliation (killer)

Spread audit conducted in parallel chat against HistData ASCII tick data, 2024-01 → 2025-12, 28 pairs. Methodology: per-bar first-5-minute median spread as execution-bar proxy. Compared modeled spread (with `spread_floors_5ers.yaml` 0.1 pip floor applied) to real spread from tick aggregation.

### Per-pair gap (1H execution bars, p50 first-5-minute vs 0.1 pip floor)

| Pair | Real p50 (pips) | Floor (pips) | Gap | Multiplier |
|---|---|---|---|---|
| EUR/USD | 0.3 | 0.1 | 0.2 | 3x |
| USD/JPY | 0.6 | 0.1 | 0.5 | 6x |
| AUD/USD | 1.0 | 0.1 | 0.9 | 10x |
| GBP/USD | 0.8 | 0.1 | 0.7 | 8x |
| EUR/NZD | 3.8 | 0.1 | 3.7 | 38x |
| GBP/AUD | 4.0 | 0.1 | 3.9 | 40x |
| GBP/NZD | 4.8 | 0.1 | 4.7 | 48x |

**All 28 pairs have real p50 > floor.** Cross pairs (especially anti-USD pairs and exotics) are dramatically worse than majors.

### Arc 4 trade impact

From `per_arc_summary.csv` reconciliation:
- Trades evaluated (entry/exit within audit window 2024-01 → 2026-01): 3,898 of 10,764
- Floor hit on entry: 44.59%
- Floor hit on exit: 42.74%
- Mean under_R per trade: **0.06574** (6.5x the ~0.01R modeled in Step 5B-spread)
- Total under_pct_equity at 0.20% risk: **128.1%** across the audit window

### F6 reconciliation arithmetic

Per fold, ~900 admitted trades. Mean spread under-modeling 0.0657R × 0.20% risk per trade = ~0.013% equity drag per trade. Per-fold cumulative drag (900 × 0.013%) ≈ 11.8% equity over 9-month fold. Annualised: ~15.7% drag per fold.

Applied to Step 5B-spread fold ROIs at 0.20%:

| Fold | Old ROI ann% | Spread drag (ann) | Real-spread ROI ann% |
|---|---|---|---|
| F2 | ~+34.8 | −15.7 | ~+19.1 |
| F3 | ~+26.0 | −15.7 | ~+10.3 |
| F4 | ~+34.5 | −15.7 | ~+18.8 |
| F5 | ~+43.0 | −15.7 | ~+27.3 |
| F6 | **+10.08** | **−15.7** | **~−5.6** |
| F7 | ~+32.8 | −15.7 | ~+17.1 |

F6 flips negative. §9.A sign consistency retroactively fails. Arc 4 closes CLEAN-NULL.

---

## Why this isn't an arc bug

The signal worked. The classifier worked. The exit policy worked. Every step gate the methodology was designed to test, Arc 4 cleared. The arc only died when the data the methodology depends on (modeled spreads) was shown to be wrong by an order of magnitude on most pairs.

Specifically:
- The signal exists (10,764 trades, four well-separated path-shape clusters)
- The cluster is real (cluster 1, silhouette 0.50, mono-pre-peak 0.564, fwd_mfe_p50 3.81R, frac_reach_1R 0.978)
- The classifier extracts it (D1 RF AUC 0.667 at t=1 with admission overlap 88-96% under proper WFO refit)
- The exit policy converts it to PnL (mean_r +0.136 to +0.243 per fold under modeled spreads)
- Fold stability is genuine (F2-F7 all positive on modeled, t-stats 2-7)

Arc 4's edge under modeled spreads was around 0.18 mean_r per trade in the best refit folds. Real spreads cost 0.07R per trade. That's ~38% of the per-trade edge eaten by the spread we weren't modeling. The signal is too small to support deployment given actual transaction costs — but the methodology found it correctly.

---

## What this means for the project

### KH-24 live deployment posture (re-evaluated)

KH-24's `configs/wfo_kh24.yaml` does NOT apply the 0.1 pip floor — it uses raw MT5 per-bar spread directly. The audit's KH-24 reconciliation:
- Trades evaluated (within 2024-01 → 2026-01 window): 69 of 553 (only Folds 5-7 are inside audit window)
- Mean under_R: 0.02187
- Total under_pct_equity: **1.509%** across evaluated trades
- Fold 7 under_pct_equity: 0.6397% on 27 trades

Fold 7 published ROI: +1.92% (worst fold). After real-spread correction: ~+1.28%. **Annualised over Fold 7's ~5.5-month OOS window: ~+2.8%.**

§10 pass-deployable requires worst-fold ann ROI ≥ 5%. Real-spread KH-24 retroactively falls into pass-viable territory (positive but below 5% deploy floor).

**KH-24 stays live.** The live system pays whatever the broker charges, not what the backtester modeled. The WFO claim of "pass-deployable" was on optimistic spread assumptions; the underlying signal still has positive expectancy. Live deployment posture: continue as currently configured, but the published WFO claim is downgraded to pass-viable.

No live system changes. EA v2.01 on Contabo VPS unchanged. `configs/wfo_kh24.yaml` unchanged (raw spread already).

### Cross-arc invalidation

Arc 4's failure mode applies to all prior arc evaluations that used the floor. Affected arcs:
- Arc 2 redo: closed KILL at Step 3 — verdict unchanged (would have died harder)
- Arc 3: closed CLEAN-NULL at Step 3 — verdict unchanged
- KH-24 v2.0 self-test: closed HALT at Step 3 — verdict unchanged
- Open-18 replays planned for v2.1.1: must use real-spread floor before re-running

Prior closures don't reverse — the magnitude of failure was understated, not the direction. But re-opening any closed arc requires real-spread re-evaluation.

### Cross-arc calibration items added

| Item | Priority | Notes |
|---|---|---|
| Real-spread floor file | HIGHEST | Replace uniform 0.1 pip with per-pair empirical floors from HistData audit, validated against MT5 broker snapshot. Locked-file governance under SPREAD_SEMANTICS_LOCK.md |
| Spread audit becomes Phase Zero prerequisite | HIGH | All future L arcs must validate spread floor file before Step 1 plumbing runs |
| Session-aware spread modeling | MEDIUM | Per-pair × session-time floors (london / NY overlap / off-hours) may be required for accurate cost modeling on cross-pair signals |
| Step 5 simulator default — apply exit spread | MEDIUM | All post-hoc simulators must apply S/2 exit spread by default; Arc 4's bug was prompt-author error, but the simulator template should enforce |
| Convention (b) MTM DD as protocol-default | MEDIUM | 5ers measures equity in real-time; convention (a) closed-trade ordering understates account DD by 14-63%. Should be §10's default gate metric |
| F1 structural leakage | HIGH | Pool start = F1 OOS start, no honest WFO training data for F1. Affects every L arc. Options: pool back-extend, drop F1 evaluation across all arcs, or alternate fold structure |
| D1 threshold grid specification | MEDIUM | §3 locks E grid but not D1 grid; both Arc 4 and Arc 5 hit this. Lock {base_rate, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50} for D1 |
| §11 row 2 deep-pullback tolerance | LOW | Arc 4 cluster 1 carried pullback 0.676R against row 2's ≤0.5R rule; doesn't matter now arc is closed, but cluster centroid pattern boundaries deserve empirical refinement |

---

## Decisions documented

- Cluster 1 archetype assignment to §11 row 2 (Stepwise climber): locked at Step 4→5 boundary, pre-commit, not post-hoc. Pullback 0.676R vs row 2's ≤0.5R was "degree not kind" — exit policy structurally robust to deeper pullbacks.
- Cluster 3 elimination at §9: F6 sign flip (mean_r −0.001) + DD blowup (31.67%) under §9.A + §9.C. Protocol-correct kill.
- F1 skipped throughout Step 5: no honest WFO training data, cross-arc v2.2 item.
- D1 threshold grid extension to {base_rate, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50}: chat decision, within-arc operational interpretation of under-specified §3 detail (§3 locks E grid; D1 grid was never explicitly locked).
- §1.8 (within-arc thresholds don't move) honored throughout — SL selection at Step 3 not re-swept after Step 5 results suggested wider SL might help.

---

## Artefacts

### Step results

| Step | Folder | Key files |
|---|---|---|
| 1 | `results/l_arc_4/step1/` | `trades_all.csv` (10,764 rows), `trades_paths.csv`, `step1_diagnostics.md` |
| 2 | `results/l_arc_4/step2/` | `clusters_K4.csv`, `centroids_K4.csv`, `archetype_assignments.csv`, `step2_diagnostics.md` |
| 3 | `results/l_arc_4/step3/` | `archetype_summaries.csv`, `capturability_pass_list.csv`, `sl_sweep_cluster_{0..3}.csv`, `step3_diagnostics.md` |
| 4 | `results/l_arc_4/step4/` | `cluster_{1,3}_D1_classifier_t1.joblib`, `cluster_{1,3}_D1_policy.yaml`, `extractability_pass_list.csv`, `step4_diagnostics.md` |
| 5 | `results/l_arc_4/step5/` | `fold_stability_cluster_{1,3}.csv`, `per_trade_simulated_{1,3}.csv`, `stability_pass_list.csv`, `step5_diagnostics.md` |
| 5B | `results/l_arc_4/step5b/` | `risk_sweep_cluster_1.csv`, `dd_convention_comparison.csv`, `step5b_diagnostics.md` |
| 5C | `results/l_arc_4/step5c/` | `per_trade_simulated_refit_1.csv`, `per_fold_refit_classifier_metrics.csv`, `leakage_deltas.csv`, `step5c_diagnostics.md` |
| 5B-spread | `results/l_arc_4/step5b_spread/` | `per_trade_simulated_refit_spread_1.csv`, `risk_sweep_refit_spread_b_cluster_1.csv`, `spread_delta_analysis.csv`, `step5b_spread_diagnostics.md` |
| Spread audit | `results/spread_audit/` (separate chat) | `spread_validation_report.md`, `per_pair_distributions.csv`, `execution_bar_spreads.csv`, `per_trade_impact_arc4.csv`, `per_arc_summary.csv` |

### Configs

- `configs/l_arc_4.yaml` (locked, sha256 b3909f17...)
- `configs/spread_floors_5ers.yaml` (FLAGGED FOR UPDATE per cross-arc calibration)

---

## Interesting observations (not core, cherry-on-top)

- Cluster 1's path-so-far features dominated D1 classifier importance (4 of top 5): close_r_at_t1, velocity_first_t, mae_so_far_r_at_t1, mfe_so_far_r_at_t1. One bar of post-entry action is what tells the classifier which cluster a trade belongs to. Entry-time features alone never cleared the 0.65 E gate.
- The v2.1.1 pre-peak monotonicity rescue worked exactly as designed: cluster 1 full-window mono 0.509 (would fail v2.0's 0.55 floor) vs pre-peak mono 0.564 (passes). First arc to demonstrate the amendment's rationale empirically.
- F6 is consistently the regime-risk fold across both surviving clusters: c3 died on F6, c1 weakest on F6, both in original simulator and refit. Worth investigating cross-arc whether F6's date range (likely 2024 macro regime) is structurally hostile to mean-reversion 1H signals.
- Convention (b) MTM DD inflation factors varied 14-63% across folds, with F3 worst (62.69%). This signals concurrent-position concentration is regime-dependent, not uniform. A signal class with low pair-concentration (Arc 4 was broad: top-pair share ~4%) still produces meaningful MTM inflation under volatile periods.

---

## Document control

| Field | Value |
|---|---|
| Arc | 4 |
| Signal | TRIAL__univariate_extreme__bar_range_top_decile__neg__h_001 |
| Opened | 2026-05-17 |
| Closed | 2026-05-17 |
| Verdict | CLEAN-NULL on transaction-cost truth |
| Protocol | L_ARC_PROTOCOL v2.1.1 |
| Calibration items added | 8 (see Cross-arc calibration table above) |
| Live system impact | KH-24 WFO claim downgraded to pass-viable retroactively; live deployment unchanged |
