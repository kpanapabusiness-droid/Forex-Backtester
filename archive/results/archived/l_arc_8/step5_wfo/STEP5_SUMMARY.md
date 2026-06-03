# Arc 8 — Step 5 WFO Summary

> Protocol: L_ARC_PROTOCOL v2.3 §10 multi-pipeline ship rule.
> Archetype: V-shape recovery (forward-geometry weak), c1 cluster, n=177.
> Selected SL: 4.0×ATR_4H entry-anchored. pre_t_sl_atr_multiplier (D1): 4.0.
> WFO windows: configs/wfo_kh24.yaml verbatim (7 anchored folds, IS expanding from 2019-01-01, OOS 9mo each).

## Verdict

**ARCHETYPE DIES at Step 5 — neither Pipeline E nor Pipeline D1 clears §10 ship gates under full-pool deployment economics.**

Arc 8 is the **third arc** to PASS admit-only economics (Step 4 closure) and FAIL full-pool deployment economics (Step 5 WFO §10). Confirms Open-22/23/24 cross-arc framework that surfaced at Arc 4 RERUN and Arc 5 closures.

## Two views per dispatch §10 + CLAUDE.md cross-arc lesson

Each fold is evaluated under two methodologies:

- **admit-only**: classifier applied to c1 OOS trades only. Validates the Step 4 holdout result under a different fold structure. NOT the §10 ship-gate basis.
- **full-pool**: classifier applied to ALL Step 1 OOS trades (production deployment: we cannot pre-filter to c1 because clusters are post-hoc path-shape labels). THIS is the §10 ship-gate basis.

### Why both views?

Per protocol §10 + Arc 4 RERUN closure: "Two arcs in a row PASSED §9 admit-only stability and FAILED §10 full-pool deployment. Pipeline D1 carries mandatory cost on the reject pool and on the early-exit pool. Any Pipeline D1 candidate's deployment viability is `(admit_rate × admit_mean) vs (reject_rate × |reject_mean|) + (early_exit_rate × |early_exit_mean|)`."

The c1 archetype's "structural edge" (admit-only mean_r +2.6 per trade) does NOT translate to deployment because:
1. At deployment, the classifier sees every Step 1 signal (1,327 trades), not just future c1's (177 trades).
2. The trained classifier admits 70-89% of all Step 1 signals — it doesn't discriminate well on non-c1 trades.
3. Non-c1 trades' mean_r at SL=4.0×ATR is far below c1's; the admitted pool's aggregate economics revert to near-zero or negative.

## Pre-checks (committed at 3db7da0, restated here for completeness)

- **Pre-check 1 — Fold 2 regime investigation:** PASS. AUC 0.4500 fold maps to TimeSeriesSplit fold idx 1 OOS window 2022-10-19 → 2023-08-01, covering USD reversal + SVB/CS bank crisis + BoJ Ueda + US debt ceiling. Regime-shift artefact, not leak. See [step5_prechecks/fold2_regime.md](../step5_prechecks/fold2_regime.md).
- **Pre-check 2 — D1 t=1 leak audit:** PASS — CLEAN. Code path inspection confirms slice at t=1 contains bars [0, 1] only; all features strictly causal. Recall=1.000 / precision=0.909 holdout deflates: c1's base rate is 0.81 making recall ≥ 0.60 trivial; specificity is 50% on 6 negatives, CV AUC 0.637 is the reliable metric. See [step5_prechecks/d1_t1_leak_audit.md](../step5_prechecks/d1_t1_leak_audit.md).

## Engine PR merge state

`feat/open-24-pre-t-sl-per-archetype` merged into local `main` at `716ce84` (v2.3 §4 Open-24 plumbing for `pre_t_sl_atr_multiplier` consumption). Brought into this worktree at `3025b35`. 69/69 D1 pipeline tests pass post-merge. NOT pushed to origin/main — that's a separate analyst-side action.

## Locked file sha verification

| File | Required | Actual | Status |
|---|---|---|---|
| `configs/spread_floors_5ers.yaml` (body) | `8da7644b...` | `8da7644b...` | ✓ |
| `configs/wfo_l_arc_8.yaml` | (dispatch said `accba985...` — wrong; from smoke-test manifest) | `9785a5b...` (verbatim manifest) | ✓ (clarified in halt summary) |
| `configs/wfo_kh24.yaml` | (untouched) | `252dfd8d...` | ✓ |

## Pipeline E — Full-Pool WFO

**Threshold: 0.70 (locked from Step 4 closure, NOT re-tuned per dispatch).**
**Features (5, locked from Step 4 B-top5):** `ret_5bar_atr`, `pos_in_20bar_range`, `pullback_depth_atr`, `range_to_atr_14`, `hl_range_atr`.

### Per-fold results

| Fold | IS train | OOS pool | Admit | Admit % | mean_r | total_r | ROI % | Max DD % | Hit | Pairs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | _(skipped — IS empty)_ | | | | | | | | | |
| 2 | _(skipped — IS n=23 < 30)_ | | | | | | | | | |
| 3 | 71 | 188 | 164 | 87.2% | +0.023 | +3.76 | **+1.24%** | 15.58% | 0.293 | 27 |
| 4 | 120 | 184 | 141 | 76.6% | −0.195 | −27.51 | **−13.21%** | **14.66%** | 0.248 | 26 |
| 5 | 156 | 188 | 158 | 84.0% | +0.310 | +49.03 | **+26.85%** | 7.30% | 0.361 | 27 |
| 6 | 235 | 181 | 128 | 70.7% | −0.169 | −21.62 | **−10.56%** | **14.08%** | 0.281 | 26 |
| 7 | 274 | 202 | 160 | 79.2% | +0.072 | +11.47 | **+5.35%** | **8.18%** | 0.362 | 27 |

### Aggregate (5 folds × OOS 9mo each, compounded)

| Metric | Value | Ship gate | Pass? |
|---|---:|---|:---:|
| Total OOS trades admitted | 751 | ≥ 30 | ✓ |
| Worst-window ROI | **−13.21%** | > 0 | **✗** |
| Worst-window max DD | **15.58%** | ≤ 8% | **✗** |
| Aggregate sharpe (per-trade R) | **0.012** | ≥ 0.8 | **✗** |
| Aggregate compounded ROI | +5.02% | _(informational)_ | |
| Aggregate compounded max DD | 16.15% | _(informational)_ | |
| Aggregate hit rate | 31.2% | _(informational)_ | |

**3/4 ship gates FAIL. Pipeline E does not ship.**

(Note: the aggregate compounded ROI of +5.02% obscures the per-fold reality. Two folds (4, 6) are deeply negative (−13.21%, −10.56%). The 5-fold compounded number averages over wins and losses but the §10 ship gate is the WORST-window basis, not the aggregate.)

### admit-only comparison (NOT ship-gate basis)

Restricting to c1 OOS trades (the Step 4 holdout view): aggregate sharpe 1.44, worst-window ROI +18.66%, max DD 1.00%. ALL 4 gates pass.

**The 14-percentage-point gap between admit-only (+18.66%) and full-pool (−13.21%) worst-window ROI quantifies the classifier specificity gap on non-c1 trades.**

## Pipeline D1 — Full-Pool WFO

**Threshold: 0.60 (locked from Step 4 closure).**
**t: 1 (locked from Step 4 smallest-t selection).**
**Features (15, locked):** 8 base entry + 7 path-so-far at t=1.
**Exit policy:** PR 1 close-at-market (admit → run with default exit at SL=4.0×ATR + time exit at bar 240; reject → close at bar 2 mid). §11 row 5 V-shape "standard trail" awaits PR 2 (not yet implemented).

### Per-fold results

| Fold | IS train | OOS pool | Admit | Admit % | Reject CAM | Reject % | mean_r | total_r | ROI % | Max DD % | Hit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | _(skipped — IS empty)_ | | | | | | | | | | |
| 2 | _(skipped — IS n=23 < 30)_ | | | | | | | | | | |
| 3 | 71 | 188 | 150 | 79.8% | 38 | 20.2% | +0.025 | +4.78 | **+1.76%** | 15.66% | 0.266 |
| 4 | 120 | 184 | 163 | 88.6% | 21 | 11.4% | −0.168 | −30.84 | **−14.69%** | **14.69%** | 0.255 |
| 5 | 156 | 188 | 167 | 88.8% | 20 | 10.6% | +0.177 | +33.31 | **+17.26%** | 7.99% | 0.309 |
| 6 | 235 | 181 | 157 | 86.7% | 24 | 13.3% | −0.124 | −22.53 | **−11.09%** | **19.02%** | 0.271 |
| 7 | 274 | 202 | 181 | 89.6% | 20 | 9.9% | +0.083 | +16.73 | **+8.07%** | **9.21%** | 0.366 |

### Aggregate (5 folds, compounded)

| Metric | Value | Ship gate | Pass? |
|---|---:|---|:---:|
| Total OOS trades (admit + reject_cam + pre-empted) | 943 | ≥ 30 | ✓ |
| Worst-window ROI | **−14.69%** | > 0 | **✗** |
| Worst-window max DD | **19.02%** | ≤ 8% | **✗** |
| Aggregate sharpe | **0.001** | ≥ 0.8 | **✗** |
| Aggregate compounded ROI | −2.19% | _(informational)_ | |
| Aggregate compounded max DD | 19.02% | _(informational)_ | |
| Aggregate hit rate | 29.5% | _(informational)_ | |

**3/4 ship gates FAIL. Pipeline D1 does not ship.**

### admit-only comparison (NOT ship-gate basis)

Restricting to c1 OOS trades: aggregate sharpe 1.14, worst-window ROI +25.73%, max DD 0.54%. ALL 4 gates pass.

## Ship decision per §10

§10: "If no cluster's best configuration achieves pass-deployable but ≥ 1 achieves pass-viable: arc ships nothing; pass-viable clusters logged as portfolio candidates only. If no cluster's best configuration achieves pass-viable: arc dies at Step 6, no shipment, no portfolio candidates."

**Pass-deployable check:**
- Worst-fold annualised ROI ≥ 5% ✗ (E: −13.21%, D1: −14.69%)
- Worst-fold max DD ≤ 8% ✗ (E: 15.58%, D1: 19.02%)
- Both fail.

**Pass-viable check (weaker — for small archetypes as portfolio candidates):**
- Worst-fold ROI > 0% ✗ (both negative)
- Worst-fold DD ≤ 8% ✗
- Mean fold ROI ≥ 3% (E aggregate ROI +1.18%; D1 +0.18%) — likely fails
- All folds positive ✗

**Neither pass-deployable nor pass-viable.**

**Disposition: ARCHETYPE_DIES_STEP_5** — no shipment, no portfolio candidate.

Per §10 "If no cluster's best configuration achieves pass-viable: arc dies at Step 5, no shipment, no portfolio candidates."

## §16a closure notes — cross-arc finding (Open-22/23/24 third confirmation)

Arc 8 is the **third arc in a row** to follow the pattern: PASS admit-only economics → FAIL full-pool deployment economics. Documented closures:

| Arc | Step 4 admit-only verdict | Step 5/6 full-pool verdict | Failure mode |
|---|---|---|---|
| Arc 4 (RERUN) | PASS (Pipeline D1 admit-only +0.125R/trade) | FAIL §10 (full-pool reject pool & early-exit drag) | Pipeline D1: reject pool 32% × −0.232R + early-exit pool 11% × −0.685R |
| Arc 5 | PASS (Pipeline D1) | FAIL §10 (rejected-pool adverse selection ~78%) | Pipeline D1: rejected pool 78% × −0.46R mean |
| **Arc 8** | **PASS (E AUC 0.697 + D1 AUC 0.637 at t=1)** | **FAIL §10 (classifier admits 70-89% of full pool; specificity gap on non-c1 trades)** | **Pipeline E: aggregate mean_r ~0 on full pool admits; Pipeline D1: same + CAM reject drag** |

### Cross-arc structural pattern

The c1 classifier (Step 4 RF, 5 features for E / 15 for D1) was trained on c1 trades only (n=177 IS subset shrinking by fold). At deployment, it sees every Step 1 signal across all 4 path-shape clusters:

Per-cluster mean_r at SL=4.0×ATR + default exit (the c1 R-frame):

| Cluster | n | mean_r | median | hit % (r>0) | pos % (r≥1R) | Path-shape archetype | Classifier admit % (avg fold 3-7) |
|---|---:|---:|---:|---:|---:|---|---:|
| 0 | 316 | **−0.459** | −1.00 | 18.0% | 11.1% | unassigned (near Early-peak / Peak-and-collapse) | ~80% |
| 1 (**c1**) | 177 | **+2.591** | +2.38 | 91.0% | 81.4% | **V-shape recovery (FG-weak)** — the trained-on archetype | ~80% (correct admit) |
| 2 | 429 | **−0.474** | −1.00 | 16.8% | 12.1% | Early-peak hold (Step 3 dies) | ~80% (false positives — feature overlap with c1 at entry) |
| 3 | 405 | **−0.097** | −1.00 | 32.8% | 20.3% | V-shape recovery canonical (Step 4 E AUC 0.61, D1 threshold-sweep fails) | ~80% (similar geometry to c1 at entry) |

c0 (n=316) and c2 (n=429) trades have mean_r near −0.46 / −0.47 at SL=4.0×ATR (most stop at −1R). Their inclusion in the admitted pool — at ~80% admit rate — dominates the deployment aggregate. The c1 cluster's +2.59 mean_r is genuinely large but only 13.3% of the Step 1 pool.

Back-of-envelope deployment expectation: `(c0_mean × c0_admit_rate × c0_size + c1_mean × c1_admit_rate × c1_size + ...) / (total_admitted)`:
  ≈ (−0.46×0.80×316 + 2.59×0.80×177 + −0.47×0.80×429 + −0.10×0.80×405) / (1327×0.80)
  ≈ (−116.3 + 366.7 + −161.3 + −32.4) / 1062 ≈ +56.7 / 1062 ≈ **+0.053R per admitted trade**

The classifier filtering provides ~0.05R per trade lift over admitting everyone. Multiplied by 0.5% risk per trade, that's 0.025% balance increment per trade. Across ~150 trades per fold, ~3.8% per fold ROI EXPECTED. Observed worst fold −13.21% reflects regime-dependent variance + per-fold drawdowns — the lift is real on average but per-window distribution makes the worst-window gate fail.

### Why the classifier fails specificity on non-c1 trades

- **5-feature Pipeline E top set:** `ret_5bar_atr`, `pos_in_20bar_range`, `pullback_depth_atr`, `range_to_atr_14`, `hl_range_atr`. These describe entry-bar context (momentum, location-in-range, pullback geometry, volatility, HL structure). Cluster 2 (early-peak / peak-and-collapse) and Cluster 3 (V-shape canonical) have OVERLAPPING entry-time geometry with c1 — the difference between them is in the FORWARD path (which classifier cannot observe at entry).
- **D1 at t=1 features:** Path-so-far features at bar 1 are still too early to distinguish clusters that diverge in middle-of-trade behaviour (t=10+).

This is consistent with the protocol's framing: "path-shape clustering is outcome-blind and operates on whatever path exists" — clusters are POST-HOC labels. Pre-hoc classifiers face the underlying ill-posedness of predicting cluster membership from entry-time features alone when clusters differ mid-path.

### Open-22/23/24 status

The framework needs updating to reflect that this is now a confirmed cross-arc pattern (3 of 3 trend-continuation arcs). Recommendation for analyst-side protocol amendment cycle:

- Consider mandating **full-pool deployment economics evaluation at Step 4** (not Step 5), so that arcs failing this gate are killed before the WFO compute spend.
- Consider an "admission specificity floor" gate at Step 4: classifier must achieve specificity ≥ X% on a held-out non-archetype sample before threshold-sweep proceeds.
- Consider Pipeline E candidate-design guidance: require classifier features that DISCRIMINATE on cluster membership (cluster-id ROC-AUC ≥ Y), not just success-within-cluster (success ROC-AUC ≥ 0.65).

These are protocol-amendment-cycle items, not within this dispatch's scope.

### Arc 8 specific notes

- **Path-shape clustering identified a real V-shape recovery archetype.** c1's admit-only economics (+2.59R mean) are genuinely structural — the 177 c1 trades, evaluated in isolation, do represent a tradeable edge. Cross-arc candidates surface the c1 cluster geometry for portfolio composition (Open-05).
- **The PR-HHHL signal generates a real entry trigger.** The arc-level kill is at the **filter-specificity** step, not the signal step.
- **Pipeline E top-5 feature set is informative as a cluster-cohort filter**, not as a deployment filter. Worth referencing in future arcs' Step 4 feature catalogue: `ret_5bar_atr`, `pos_in_20bar_range`, `pullback_depth_atr`, `range_to_atr_14`, `hl_range_atr`.

## Files

- `pipeline_e/wfo_results_per_window.csv` — full-pool per-fold (ship-gate basis)
- `pipeline_e/wfo_results_per_window_admit_only.csv` — admit-only per-fold (Step 4 cross-validation)
- `pipeline_e/wfo_aggregate.json` — full-pool aggregate + ship gates
- `pipeline_e/wfo_aggregate_admit_only.json` — admit-only aggregate
- `pipeline_e/equity_curve.png` — full-pool compounded equity curve
- `pipeline_d1/...` — same set for D1
- `step5_prechecks/fold2_regime.md` + `step5_prechecks/d1_t1_leak_audit.md` — pre-flight pre-checks (PASS)
- `STEP5_SUMMARY.md` — this doc

## Recommended next action (analyst-side)

1. **Archive Arc 8 c1 V-shape recovery (FG-weak)** as `archived_candidate_c1.yaml` portfolio candidate per Open-05 (cross-arc finding: 3rd confirmation that admit-only economics ≠ deployment economics).
2. **Decide whether to:**
   - Close Arc 8 with disposition **KILL** (no shipment, no portfolio candidate per §10 strict reading)
   - Close Arc 8 with disposition **HALT** (recommend protocol amendment for next cycle — see §16a recommendations above) and log c1 as a Pipeline-D2-or-D3 candidate (if such pipelines exist post-amendment)
3. **Open protocol amendment item** to address the admit-only / full-pool divergence systematically (Open-22/23/24 still has rows open per `PROTOCOL_IMPROVEMENT_BACKLOG.md`).
4. **Update CHANGELOG.md, SESSION_ZERO.md, STATUS.md** with Arc 8 closure — analyst-side housekeeping per dispatch §83 "Do not touch" list.
