# Arc 11 — `signal_swing_high_breakout_trend_long_v0.1` (SHB)

## Status

- **Current step:** Step 4 KILL → HALT — arc closed.
- **Verdict:** **STEP_4_HALT** per §16a Path A (numeric near-miss; agg_c1_c3 best AUC 0.5728 vs gate 0.60, margin 0.027 < 0.03 absolute).
- **Last updated:** 2026-05-18
- **Branch:** `claude/condescending-hoover-72a181` (worktree branch in use for Arc 11; mapping to `phase/l_arc_11` for the dispatch's commit-message convention).

Post-closure experimental documentation: see [ARC_11_CLOSURE.md](ARC_11_CLOSURE.md) for full record.

## Protocol stack (active)

- `L_ARC_PROTOCOL.md` v2.1.2 (base)
- `L_ARC_PROTOCOL_v2_2_AMENDMENT.md` (FIFO queue, §16a, live-execution equivalence, Tier 2 lift cap ≤5, max-F1 fallback removed)
- `L_ARC_PROTOCOL_v2_3_AMENDMENT.md` (Step 5 cross-fold stability **REMOVED**; Step 6 → Step 5 = WFO; halt at end of Step 4; Open-22/23/24 closed)

Effective lifecycle for this arc: Steps 1-4 unattended → halt → chat-dispatched Step 5 WFO.

## Arc-open

| Field | Value |
|---|---|
| Signal under test | `signal_swing_high_breakout_trend_long_v0.1` |
| Signal source | `docs/signal_spec_swing_high_breakout_trend_long_v0.1.md` (analyst spec, Downloads) |
| Signal family | Trend continuation (structural breakout at historical reference) |
| Direction | Long only |
| Signal TF | 4H |
| Anchor TF | n/a (single-TF) |
| Pair set | 28 FX (KH-24 set) |
| Data window | 2020-10-01 → 2026-01-31 |
| Hypothesis | Break magnitude relative to a meaningful historical reference (H_ref), reference freshness, and trigger-bar geometry are entry-time observable → Pipeline E should clear 0.65 AUC. Cluster heterogeneity expected from `H_ref` freshness (4-19 bars). |
| Population builder | `build_ex_ante_bounded_population` (single pass, no folds) |
| Forward window | 240 4H bars (default) |
| Simulation SL (Step 1) | 2.0 × ATR(14)_4H (default per spec) |
| SL sweep candidates (Step 3) | `{0.5, 1.0, 1.5, 2.0, 3.0, 4.0} × ATR_4H` (default per spec) |
| Risk per trade | 0.5% × reset-floor balance |
| Refractory | 20 4H bars between successive signals on same pair |
| Exposure cap | Max 1 open position per pair |
| Spread source | Per-bar MT5 `spread` column; `configs/spread_floors_5ers.yaml` fallback (body sha256 `8da7644b252ae163d963fbd46807572906fa3e5a44fb3e02d771e181b3ecdc05`) |
| Pre-committed step gates | Per v2.3 base (no overrides, no mid-arc sign-off, halt at end of Step 4) |

## Signal trigger (locked at arc-open per §1.8)

Swing definitions (3-bar local extreme):
- Swing-high at bar k iff `high[k] > max(high[k-3..k-1])` AND `high[k] > max(high[k+1..k+3])`
- Swing-low at bar k iff `low[k] < min(low[k-3..k-1])` AND `low[k] < min(low[k+1..k+3])`

1. **Trend filter:** swing-lows in window `t-30..t-1` (only `k ≤ t-4`); require ≥1 exists; require `close[t-1] > min(those swing-low values)`.
2. **Reference swing-high:** identifiable swing-highs in window `t-20..t-1` (`k ≤ t-4`); `H_ref` = most recent.
3. **Break trigger at bar t:**
   - `close[t] > H_ref + 0.10 × ATR(14)_4H[t]`
   - `close[t] > open[t]`
   - `(close[t] − low[t]) / (high[t] − low[t]) ≥ 0.5`
4. **Spacing & entry:** ≥20 bars since last signal on this pair; entry bar t+1 open.

Implemented verbatim in [signals/lchar_swing_high_breakout_trend.py](signals/lchar_swing_high_breakout_trend.py).

## Step results

| Step | Gate | Result | Notes |
|---|---|---|---|
| 1 — Plumbing | Pool ≥ 500; byte-identical determinism; right-edge audit clean | **PASS** | 2,299 trades; det PASS; min `h_ref_bar_offset` = 4. |
| 2 — Clustering | silhouette ≥ 0.30, no cluster > 90%, all clusters ≥ 30 | **PASS** | K=4, silhouette 0.4692, 0/4 degenerate, det PASS. |
| 3 — Capturability | ≥1 archetype passes §2 floors at any swept SL | **PASS** | 3 surviving units (c1, c3, agg_c1_c3 — all V-shape recovery); c0/c2/agg_c0_c2 die. |
| 4 — Extractability | ≥1 capturable archetype clears RF AUC ≥ 0.65 (E) or ≥ 0.60 (D1) with valid threshold (recall ≥ 0.60) | **FAIL — HALT** | 0/3 units clear gate. Best AUC 0.5728 (agg_c1_c3 D1 t=5), margin 0.027 < 0.03 → §16a Path A HALT. |

## Step 1 — Plumbing

### Headline

| Metric | Value |
|---|---|
| Total signals fired (pre-exposure cap) | 5,728 |
| Trades after exposure cap | **2,299** |
| Signals skipped (position open) | 3,429 |
| Prefilter events (all conds except close-upper-half + refractory) | 28,631 |
| `bars_held_p95` | 240 |
| Cap-binding rate (`bars_held ≥ 240`) | **15.57%** (358 / 2,299) — below 20% §5 auto-extend threshold |
| Determinism | **PASS** (byte-identical two-run) |
| Right-edge audit (`min h_ref_bar_offset`) | **4** (PASS) |

Exit reasons: stoploss 1,920 / time_exit 357 / end_of_data 22.

Per-pair trade counts: min 60, median 83, max 104. No pairs with <30 trades; no pairs at zero.

### Step 1 distributions

`break_magnitude_atr` percentiles (5/25/50/75/95): 0.132 / 0.253 / 0.455 / 0.783 / 1.674.

`final_r` percentiles (5/25/50/75/95): −1.026 / −1.017 / −1.011 / −1.006 / +6.098. Mean −0.076R; median −1.01R. ~84% of trades close at the −1R stoploss; the right tail is fat (95th percentile +6.1R). Pre-clustering, this is a classic trend-continuation shape.

`mfe_r` percentiles (50/75/95): 3.18 / 5.53 / 10.09.

### Live-execution compliance

- Entry: bar t+1 open with `open_mid + spread/2` long-fill per `SPREAD_SEMANTICS_LOCK.md` round-trip.
- SL: intrabar mid trigger, fill at `sl_price - spread/2` using execution-bar (SL-hit-bar) spread.
- Time exit: bar t+1+240 open with `open_mid - spread/2` long-close.
- Spread: per-bar MT5 `spread` column / 10 pp-to-pips, floored via `configs/spread_floors_5ers.yaml` (sha-locked).
- D1 features: n/a (signal is single-TF 4H).
- Volume veto: n/a (not in spec).

### Right-edge swing audit (mandatory per spec)

Both swing-high (`H_ref`) and swing-low (trend filter) use `k+1..k+3` lookahead within the detection window only. The signal module enforces `k ≤ t-4` structurally (`RIGHT_EDGE_OFFSET=4`). Pool-level audit confirms `min h_ref_bar_offset = 4` across all 2,299 trades; trend-filter swing-lows enforced structurally. See [results/l_arc_11/step1_verbatim/audit_lookahead.txt](results/l_arc_11/step1_verbatim/audit_lookahead.txt).

### Co-fire matrix

KH-24 + sibling arc co-fire matrix is **deferred** for this arc: Arcs 8/9/10 are being run in parallel by other CC chats on their own branches, so their Step 1 outputs are not present in this worktree. The co-fire computation can run as a cross-arc batch once those Step 1 outputs land — recorded as a known follow-up at the halt summary (the cross-arc co-fire is an analyst input, not a Step 1 gate).

### Artefacts

- `results/l_arc_11/step1_verbatim/trades_all.csv` — sha256 `6fcc2f526bf5d81143385c9413eec01aa2fa63350525d02ab4cde0665910df32`
- `results/l_arc_11/step1_verbatim/trades_paths.csv` — sha256 `1d943ee2e22e23a4ec91b7cf45ce7958b6212b0fcb5c724655d9d488cfa1401b`
- `results/l_arc_11/step1_verbatim/prefilter_events.csv` — sha256 `d90cb38eed4da51fc0e3cda03e34b54ba414cbab0bde56da23301c6061618118`
- `results/l_arc_11/step1_verbatim/audit_lookahead.txt`
- `results/l_arc_11/step1_verbatim/audit_determinism.txt`
- `results/l_arc_11/step1_verbatim/manifest.json`

### Verdict

**PASS** — pool size, determinism, right-edge audit, schema all green. Proceed to Step 2.

## Step 2 — Path-shape clustering

### Headline

| Metric | Value |
|---|---|
| Chosen K | **4** (silhouette 0.4692) |
| Silhouette sweep (K=3/4/5/6/7) | 0.4506 / 0.4692 / 0.4588 / 0.4461 / 0.4559 (all PASS §6 gate) |
| Tie tolerance applied? | No — K=4 wins on raw silhouette. |
| Degenerate features | 0 / 4 (no path-shape feature exceeds 80% single-bin mass) |
| Determinism | **PASS** (byte-identical two-run across all output files) |

### K=4 archetype assignments

| cluster | n | size_frac | centroid (mono / peaks / pullback / ttp_rel) | status | tentative label |
|---:|---:|---:|---|---|---|
| 0 | 744 | 0.324 | 0.011 / 0.45 / 0.008 / 0.062 | tentative | Early-peak hold OR Peak-and-collapse |
| 1 | 324 | 0.141 | 0.536 / 32.48 / 0.543 / 0.768 | tentative | V-shape recovery |
| 2 | 666 | 0.290 | 0.561 / 4.67 / 0.150 / 0.307 | unassigned (near Early-peak; included in that group for Step 3 evaluation) | — |
| 3 | 565 | 0.246 | 0.509 / 9.50 / 0.770 / 0.564 | tentative | V-shape recovery |

Same-archetype clusters: `tentative_V-shape recovery` → {1, 3}. Per-cluster + per-aggregate evaluation at Step 3 per §7 routing.

c2 (unassigned) misses Early-peak's `ttp_rel ≤ 0.30` by 0.007 — included in the Early-peak/Peak-and-collapse Step 3 group so it gets the §2 floor sweep + Step 3 disambiguation.

### Artefacts

- `results/l_arc_11/step2/path_features.csv`, `silhouette_sweep.csv`, `archetype_assignments.csv`, `centroids_K{3..7}.csv`, `clusters_K{3..7}.csv`, `silhouette_K{3..7}.txt`, `feature_histograms.png`, `STEP2_SUMMARY.md`

### Verdict

**PASS** — K=4, silhouette 0.47, structurally clean. Proceed to Step 3.

## Step 3 — Capturability characterisation

### Headline

| Metric | Value |
|---|---|
| Units evaluated | 6 (clusters c0/c1/c2/c3 + aggregates agg_c0_c2, agg_c1_c3) |
| Units surviving §2 floors | **3** (c1, c3, agg_c1_c3 — all V-shape recovery family) |
| Determinism | **PASS** |

### Surviving units (PASS §2 at selected SL)

| unit | type | n | size_frac | archetype | sel SL (×ATR) | composite | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| c1 | cluster | 324 | 0.141 | V-shape recovery (forward-geometry weak) | **3.0** | 0.6119 | 0.562 | 1.000 | 0.000 | 4.481 R | unclassified |
| c3 | cluster | 565 | 0.246 | V-shape recovery | **2.0** | 0.4206 | 0.572 | 0.809 | 0.011 | 2.056 R | unclassified |
| agg_c1_c3 | aggregate | 889 | 0.387 | V-shape recovery | **3.0** | 0.4155 | 0.567 | 0.802 | 0.003 | 2.502 R | unclassified |

Per Open-24 (v2.3 §5): each survivor's `pre_t_sl_atr_multiplier` for Pipeline D1 = its selected SL (c1 → 3.0, c3 → 2.0, agg_c1_c3 → 3.0). Recorded in `cluster_routing.csv`.

### Killed units (FAIL §2 at every swept SL)

| unit | n | size_frac | best SL by composite | best floors passed | failure reasons at best SL |
|---|---:|---:|---:|---:|---|
| c0 | 744 | 0.324 | 4.0 | 4/7 | mono 0.276 < 0.55; reach_1R 0.266 < 0.70; fwd_mfe_p50 0.231 R < 1.5 R |
| c2 | 666 | 0.290 | 1.5 | 5/7 | fwd_mfe_p50 1.176 R < 1.5 R; reach_1R 0.581 < 0.70 |
| agg_c0_c2 | 1410 | 0.613 | 4.0 | 4/7 | mono 0.438 < 0.55; reach_1R 0.325 < 0.70; fwd_mfe_p50 0.527 R < 1.5 R |

The Early-peak/Peak-and-collapse family dies — early-peak archetype (c0, mass 32%) has functionally zero structural edge after the peak (mono 0.011 at SL=2 — almost no in-profit non-decreasing bars). This matches the spec's hypothesised whipsaw failure mode for breaks that don't hold.

### bimodal_separated test

| unit | sl_ref | dip stat | p-value | min mode mass | mode sep (R) | result |
|---|---:|---:|---:|---:|---:|:---:|
| c1 | 3.0 | 0.0317 | 0.214 | 0.0 | 0.0 | no |
| c3 | 2.0 | 0.0241 | 0.486 | 0.0 | 0.0 | no |
| agg_c1_c3 | 3.0 | 0.0140 | 0.901 | 0.0 | 0.0 | no |
| c0 | 2.0 | 0.0173 | 0.999 | 0.0 | 0.0 | no |
| c2 | 1.5 | 0.0235 | 0.500 | 0.0 | 0.0 | no |
| agg_c0_c2 | 2.0 | 0.0090 | 1.000 | 0.0 | 0.0 | no |

No archetype triggers the §11 row 7 split-exit parallel routing at the selected SLs.

### Tentative label disambiguation

| unit | pct_peak_and_collapse @ ref SL | final archetype |
|---|---:|---|
| c0 | 0.019 | Early-peak hold (dies at §2) |
| c2 | 0.464 | Early-peak hold OR Peak-and-collapse (Step 4 disambiguation — dies at §2) |
| agg_c0_c2 | 0.229 | Early-peak hold (dies at §2) |
| c1 | 0.244 | V-shape recovery (forward-geometry weak; peak-bars ≥ 5 conf 0.444 < 0.5 — labelled but with caveat) |
| c3 | 0.719 | V-shape recovery |
| agg_c1_c3 | 0.527 | V-shape recovery |

### Per-cluster / per-aggregate routing

| cluster | individual passes | aggregate passes | disposition |
|---|:---:|:---:|---|
| c0 | no | no | dies |
| c2 | no | no | dies |
| c1 | yes | yes | proceeds both |
| c3 | yes | yes | proceeds both |

### Artefacts

- `results/l_arc_11/step3/archetype_summaries.csv`, `capturability_pass_list.csv`, `cluster_routing.csv`
- Per-unit: `archetype_<unit>_sl_sweep.csv`, `archetype_<unit>_distribution.csv`, histograms for `fwd_mfe` and `final_r`
- `STEP3_SUMMARY.md`

### Verdict

**PASS** — 3 V-shape recovery units survive into Step 4 (1 cluster c1, 1 cluster c3, 1 aggregate agg_c1_c3). Early-peak family fully eliminated at §2.

## Step 4 — Extractability investigation

### Headline

| Metric | Value |
|---|---|
| Units evaluated | 3 (c1, c3, agg_c1_c3) — Step 3 survivors |
| Pipeline E feature count | 23 (8 cross-dataset base + 15 arc-11-specific) |
| Pipeline D1 features (per t) | 15 (8 base entry + 7 path-so-far at bar t) |
| t candidates swept (D1) | {1, 2, 3, 4, 5, 10} per protocol §8 |
| Units passing §8 (AUC + v2.2 §3 threshold) | **0 / 3** |
| Determinism | **PASS** (two-run byte-identical CSVs across summary, fold AUCs, threshold sweeps, feature importance) |

### Per-(unit, pipeline) mean AUCs

| unit | base_succ | E mean AUC (gate 0.65) | D1 best AUC (gate 0.60) | best D1 t |
|---|---:|---:|---:|---:|
| c1 | 0.784 | 0.4194 | 0.4810 | 3 |
| c3 | 0.099 | 0.3911 | 0.5707 | 4 |
| agg_c1_c3 | 0.387 | 0.5140 | **0.5728** | 5 |

### Threshold sweep

Not reached for any unit — all (unit, pipeline) pairs fail the AUC gate before threshold sweep. Empty `threshold_sweep_*.csv` artefacts.

### Class-imbalance observation

- c1 base_success_rate 0.784 (78% reach 1R at SL=3 — minority is the 22% that fail).
- c3 base_success_rate 0.099 (10% reach 1R at SL=2 — severe class imbalance; success is the minority).
- agg_c1_c3 base_success_rate 0.387 (balanced enough that class_weight="none" was used).

c3's per-fold AUCs swing widely (0.222 to 0.512 for Pipeline E; 0.474–0.688 for D1 t=4) — small minority class makes per-fold predictions unstable. agg_c1_c3 is the most stable cohort with std_auc < 0.07 across all configurations.

### Arc-level near-miss analysis

Best AUC across all 21 (unit × pipeline × t) configurations: **0.5728** at agg_c1_c3 Pipeline D1 t=5. Gate is 0.60. Margin = 0.60 − 0.5728 = **0.0272 < 0.03 absolute**.

§16a Path A criteria check:
1. **Single criterion fail**: arc-level extractability is the §8 disjunctive (E∨D1) over the best unit. One criterion. ✓
2. **Cohort viability**: agg_c1_c3 size_fraction = 0.387 ≥ 0.10. ✓
3. **Path A near-miss** (numeric, margin < 0.03 absolute): 0.0272 < 0.03. ✓

All three Path A conditions met → **HALT** with cross-arc calibration candidate, not KILL.

### Artefacts

- `results/l_arc_11/step4/extractability_summary.csv`
- `results/l_arc_11/step4/extractability_pass_list.csv` (includes `pre_t_sl_atr_multiplier` per Open-24)
- `results/l_arc_11/step4/fold_aucs.csv`
- `results/l_arc_11/step4/STEP4_SUMMARY.md`
- (No feature-importance or threshold-sweep CSVs — AUC gate didn't clear for any pipeline.)

### Verdict

**FAIL → STEP_4_HALT** — capturable Step 3 cohorts (V-shape recovery family, mfe_p50 2.06–4.48 R) cannot be distinguished from non-success trades by entry-time features (Pipeline E AUC peak 0.514 on the aggregate) nor by short-horizon path-so-far features (Pipeline D1 best AUC 0.573 on the aggregate at t=5). The cohorts carry structural edge (Step 3 PASS); they lack entry-time / early-path extractability under the §8 feature set.

## Cross-arc candidates

### HALT candidate — SHB extractability under richer feature regime

**Failing criterion:** Pipeline D1 RF AUC (numeric).
**Margin:** 0.60 − 0.5728 = **0.0272 absolute** (Path A near-miss qualifying).
**Magnitude evidence:** agg_c1_c3 fwd_mfe_h240_p50 = 2.502 R; final_r p75 = 1.45 R (Step 3 distribution). c1 alone has fwd_mfe_p50 = 4.48 R with reach_1R = 100% — real structural edge, the extractability problem is identification, not capturability.

**Calibration item type:** entry-time / early-path feature-set extension for trend-continuation breakout signals. Candidates:
- Multi-timeframe context (D1 trend strength + 1H sub-bar context — Arc 11 was single-TF 4H by spec)
- Cross-pair regime / volatility cluster features
- Order-flow proxies (sweep depth pre-break, post-break retest absence)
- Composite ensemble (Pipeline E + Pipeline D1 intersection — not run here under §3 single-classifier-clears-gate rule but mathematically feasible since the unit class balances are not adversarial)

**Reference to open items:** related to **Open-23 / Arc 5 / Arc 4 RERUN cross-arc lesson** that Pipeline D1 admit-only is not the deployment economics — Arc 11 doesn't reach that question because Pipeline D1 doesn't clear AUC. Also related to **Arc 6 closure** (`signal_failed_breakout_long_v0.2`): capturable-not-extractable closure of the same shape (Step 3 PASS path quality clean; Step 4 Pipeline E best AUC 0.60 below 0.65). Arc 11 + Arc 6 are now two arcs in a row where the failure mode is "trend signal with valid forward-geometry edge that the entry-time feature regime can't extract."

### Sibling-arc comparison candidates

Arcs 8, 9, 10 are in-flight parallel CC chats testing other entry-time-feature classes for trend continuation (PR-HHHL, IB-trend, DLR). If any of those reach Step 4 with Pipeline E AUC ≥ 0.65, the cross-arc synthesis (per Arc 11 spec "Arc 8 differential note") can pinpoint which feature class carries the signal — Arc 11's HALT closure is the "structural reference break magnitude" data point.

## Interesting observations

- **Class-imbalance asymmetry between V-shape sub-clusters.** c1 (n=324, SL=3.0): 78% success — the few losers are very hard to identify in advance. c3 (n=565, SL=2.0): 10% success — winners are the minority and hard to predict. The aggregate at SL=3.0 is the most balanced (39% success) yet still doesn't yield ≥ 0.65 AUC.
- **Pipeline E AUC for c3 (0.39) is below 0.50 random baseline** in 4/5 folds. This suggests entry-time features capture a regime-dependent pattern that the time-series CV's expanding window actively trains *against* — the early folds learn one direction; the later folds invert it. Pipeline D1 path-so-far features are more stable across folds (std_auc 0.05–0.10 for c3 D1 vs 0.11 for E).
- **D1 best t is unit-dependent**: c1 best at t=3, c3 best at t=4, agg_c1_c3 best at t=5. The smallest-t rule per §8 (smaller t = larger addressable pool, shorter wait) would favour smaller-t — but no t passes the gate so the rule never fires.
- **Right-edge audit at Step 1 was the only spec-mandated audit gate.** The fact that the signal passed all five preceding gates (Step 1 plumbing, Step 2 clustering, Step 3 capturability per-cluster + per-aggregate) before failing at Step 4 strongly suggests the failure is real (not a plumbing/feature error). The cohort exists and the cohort has structural edge — but the cohort can't be distinguished from non-cohort at decision time using the feature regimes available.
- **Co-fire matrix vs sibling arcs deferred** — Arcs 8/9/10 in flight; cross-arc co-fire is analyst-level synthesis input, not a Step 1 gate. If Arc 11 SHB and Arc 8 PR-HHHL show 20%+ co-fire (per spec expectation), then Arc 8's Pipeline E result (if it clears 0.65) shows what features SHB also has access to. If Arc 8 also lands at HALT, the cross-arc finding is "trend-continuation breakout signals at this protocol's feature regime hit an extractability ceiling around 0.55–0.60 RF AUC."

## Halt Summary — Arc 11

### Status

- **Disposition:** STEP_4_HALT (per §16a Path A numeric near-miss)
- **Closure doc:** _(this live arc doc is the closure record; will be renamed `ARC_11_RESULT.md` at queue closure)_
- **Live arc doc:** `results/l_arc_11/ARC_11_LIVE.md`
- **Branch:** `claude/condescending-hoover-72a181` (worktree branch — Arc 11's commits use the dispatch's `arc-11 step <K>` prefix convention)
- **Queue state:** Per user instruction "Ignore the arc queue, another cc chat is running the previous arcs", no queue mutation performed by this session. Analyst should mark Arc 11 Closed-HALT at queue update time.

### Step pass/fail table

| Step | Gate | Result |
|---|---|---|
| 1 — Plumbing | Pool ≥ 500, det, right-edge audit | **PASS** (2299 trades, det PASS, min h_ref_bar_offset = 4) |
| 2 — Path-shape clustering | sil ≥ 0.30, no cluster > 90%, all clusters ≥ 30, ≤ 1 degenerate feature | **PASS** (K=4, silhouette 0.469, 0/4 degenerate, det PASS) |
| 3 — Capturability | ≥ 1 archetype passes §2 conjunctively at some swept SL | **PASS** (3 V-shape recovery units; Early-peak family fully eliminated) |
| 4 — Extractability | ≥ 1 capturable archetype clears RF AUC + v2.2 §3 threshold | **FAIL → HALT** (best AUC 0.5728, margin 0.027 < 0.03 → Path A near-miss) |

### Surviving archetypes (Step 3 capturable)

None advance to Step 5 (WFO) — Step 4 HALT closes the arc. For analyst reference (would have been the Step 5 dispatch input had Step 4 passed):

| Label | Cluster IDs | Selected SL (= D1 pre_t_sl_atr) | Pipeline | RF AUC (best) | Threshold | Notes |
|---|---|---:|---|---:|---:|---|
| V-shape recovery (path-shape weak) | c1 | 3.0 | (would be both, neither cleared AUC) | E 0.42 / D1 0.48 @ t=3 | n/a | Smallest cohort (n=324), highest base success (78%) — extractability hard at this imbalance |
| V-shape recovery | c3 | 2.0 | (neither cleared) | E 0.39 / D1 0.57 @ t=4 | n/a | n=565, base success 10% — severe class imbalance |
| V-shape recovery | agg_c1_c3 | 3.0 | (neither cleared; closest miss) | E 0.51 / D1 **0.5728** @ t=5 | n/a | n=889, base success 39% — most balanced; near-miss source |

### Cross-arc calibration candidates (HALT-specific)

See **Cross-arc candidates** section above. Single calibration candidate:
- **SHB extractability under richer feature regime.** Failing criterion numeric (AUC), margin 0.027 absolute. Magnitude evidence: agg_c1_c3 fwd_mfe_p50 2.50 R, c1 fwd_mfe_p50 4.48 R with reach_1R 100%. Calibration suggestion: feature-set extension (multi-TF, regime, cross-pair, ensemble) for trend-continuation breakouts. Pairs with Arc 6 (failed-breakout) as a second capturable-not-extractable closure of the same shape.

### Recommended next dispatch

Per v2.3 §9 + §16a:
- Closure recorded as KILL/HALT — no further chat dispatch for this arc.
- HALT closure doc batched with other HALT items for next protocol amendment cycle (per v2.2 §6 / §13 "Chat reviews HALT closure docs in batch for cross-arc calibration cycles").
- Sibling arcs (8, 9, 10) still in flight; cross-arc synthesis happens when all four complete.

---

## Post-closure experimental work

> Off-protocol, documentation only. Four experimental sessions run after HALT to diagnose the failure mode and identify any tradeable signal. None mutated queue / registry / protocol state. Arc 11 status unchanged throughout. Source: [ARC_11_CLOSURE.md](ARC_11_CLOSURE.md).

### Exp 1 — S5 oracle runs

| Run | Config | Sign-cons | Worst ROI%/yr | Mean ROI%/yr | DD% | Trades | Pass |
|-----|--------|-----------|---------------|---------------|-----|--------|------|
| A | c1 raw, SL=3.0 | ✓ | 101.39 | 152.12 | 2.48 | 42 | YES |
| B | agg_c1_c3 + D1 t=5, SL=3.0 | ✓ | 26.30 | 53.33 | 4.95 | 114 | YES |

Both assume cluster-ID-at-entry oracle. Not deployable; established cohort magnitude ceiling.

### Exp 2 — S5 no-oracle runs

| Run | Config | Sign-cons | Worst ROI%/yr | Mean ROI%/yr | DD% | Trades | Pass |
|-----|--------|-----------|---------------|---------------|-----|--------|------|
| C | Live E → c1, t=0.50 | ✗ | −20.22 | −2.36 | 33.46 | 1 | NO |
| C-best | E → c1, t=0.30 | ✗ | −20.22 | +4.78 | 33.46 | 28 | NO |
| D | E → D1 t=5 cascade | ✗ | −22.83 | −2.76 | 33.46 | 4 | NO |

Oracle premium: c1 leg **−154.48 pp**, agg leg **−56.09 pp**. Pipeline E precision at base rate. **S4 AUC gate vindicated.**

### Exp 3 — Filter-diagnosis

| Regime | Mean AUC | Clears 0.65 | Δ vs baseline |
|--------|----------|-------------|---------------|
| A baseline c1 Pipeline E | 0.5238 | 0/6 | — |
| B delayed t=3 | 0.6404 | 4/6 | +0.117 |
| B delayed t=8 | 0.6409 | 1/6 | +0.117 |
| C multi-TF (D1+1H) | 0.5481 | 0/6 | +0.024 |
| D predict reach_1R | 0.4846 | 0/6 | −0.039 |
| D predict mfe ≥ 2R | 0.5102 | 0/6 | −0.014 |

**Only delayed entry moves AUC.** Multi-TF dead. Reframed target dead.

### Exp 4 — Signal improvement sweep (7 stages)

| Stage | Verdict |
|-------|---------|
| 1 Signal-tightening (trigger filters) | DEAD — 0/27 singles, 0/3 pairs pass |
| 2 Pipeline DE t-sweep | Winner t=7 (sign-consist + pre_t<40%) |
| 3 Pipeline D on c1 direct | DEAD — AUC 0.40–0.45 (worse than random) |
| 4 Path-aware dynamic SL | 4a winner (SL=5→2 at t=8); DD 1.66% vs 2.48% |
| 5 Top-10 pair subset | Real but minor; still fails |
| 6 Sizing without filter | DEAD — full pool negative EV at every tier |
| 7 Combinations | No pass-deployable; best S2+S4 below |

**Best candidate:** `DE t=7 + dynamic SL 4a`

| Metric | Value | Gate | Pass |
|--------|-------|------|------|
| Sign-consistency | ✓ | required | ✓ |
| Worst-fold ROI ann | +3.11% | > 0 | ✓ |
| Mean-fold ROI ann | +17.23% | informational | — |
| DD | 18.03% | < 8% | ✗ |
| Trades/fold | 16 | ≥ 15 | borderline |
| DD/ROI ratio | 1.04 | < 0.5 desired | ✗ |

**Verdict: not deployable.** Trade count borderline at 16/fold (gate is 15). DD/ROI ratio 1.04 means risk-scaling does not rescue — doubling sizing doubles DD without doubling expected return relative to capital at risk. Even at half size, DD/ROI is unchanged and trade count remains thin.

---

## Strike list

Empirically retired directions for SHB long 4H:

1. **Pipeline E on entry-bar features** — AUC 0.42–0.52, structurally insufficient
2. **Pipeline D post-entry on c1 cohort** — AUC 0.40–0.45, no discriminating signal inside cluster
3. **Multi-TF feature extension** — +0.024 AUC, dead
4. **Reframed supervision target (reach_1R, mfe≥2R)** — worse than cluster ID
5. **Trigger-bar mechanical filters** — 0/27 single rules, 0/3 pairs separate c1/c2
6. **Sizing without filtering** — full SHB pool negative EV at every tier (−17/−32/−56% worst)
7. **"Relax AUC gate when mfe_p50 ≥ 3R"** — disconfirmed by no-oracle test
8. **DD-relaxation amendment for capturable-not-extractable cohorts** — DD/ROI ratio insufficient

---

## Cross-arc calibration signals

1. **Arc 6 + Arc 11 = capturable-not-extractable pattern (confirmed).** Two arcs, same shape: cohort carries real structural edge (Arc 11 c1: `fwd_mfe_p50` 4.48R, `reach_1R` 100%); entry-time features cannot resolve. Pattern is now empirically documented, not speculative.

2. **Timing > features as extractability lever (new).** Filter-diagnosis shows post-signal price action carries the discriminating information. Multi-TF and feature redesign do not. Any future capturable-not-extractable arc should test deferred entry before declaring HALT.

3. **DD structural for V-shape cohorts (new).** c1's 78% WR + fat-tail wins + clustered −1R losses produces 15–20% per-fold DD regardless of filter, classifier, or trail policy. Cohort character, not policy failure.

4. **Pipeline E ceiling for 4H entry-bar features (new).** Three independent feature regimes (baseline, multi-TF, reframed target) cap below 0.55 AUC on this signal. The ceiling is structural to the feature/timeframe combination, not the classifier choice.

---

## Protocol amendment candidates

For the v2.4 cycle:

**Pipeline DE (Deferred-Entry) — propose for inclusion**
- Architecture: enter at bar `t` post-signal or not at all; no second-stage classifier
- Features: path-so-far at bar `t`
- Default `t`: per-archetype, sweep range `[1, 16]`
- Gate: `AUC ≥ 0.60` + sign-consistency + pre-t SL filter rate `< 40%`
- Distinct from D1 (post-entry decision mid-trade)
- Empirical support: Arc 11 Stage 2 — only direction that moves AUC

**`min_observation_bars` registry parameter — propose for inclusion**
- Every archetype carries a `min_observation_bars` parameter
- Protocol tests Pipeline DE variants before declaring §16a HALT
- Implication: Arc 11 wouldn't have closed HALT under this protocol — would have continued to DE evaluation, producing the same +3.11/+17.23/18% DD result and then HALT'ing on DD/trade-count grounds (i.e. cleaner failure attribution)
