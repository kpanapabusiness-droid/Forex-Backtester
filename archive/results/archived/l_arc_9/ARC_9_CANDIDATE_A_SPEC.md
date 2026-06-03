# Arc 9 Deployment Candidate A — LGBM Pipeline E at threshold 0.40

> Status: **PASS-DEPLOYABLE-PENDING-AMENDMENT** on folds 2-7 (strict 7-fold fails on F1 = 0 admits artifact — data window starts at KH-24 F1 OOS_start, no anchored-expanding training data available for F1). Subject to lookahead verification audit before deployment momentum.
> Source: Arc 9 Step 5 LGBM Pipeline E dispatch, commit 44de1ca on `claude/bold-brattain-d79817`.
> Pre-deployment requirements: (1) lookahead audit pass, (2) v2.x §3 grid amendment OR locked-grid acceptance, (3) Step 5 fold-1 warmup convention OR data-window backfill, (4) formal Step 4 re-pass under amended protocol.

---

## Signal definition

| Field | Value |
|---|---|
| Signal name | IB-trend (compression geometry → directional break, long) |
| Source spec | `signal_spec_ib_trend.md` (Arc 9 origin spec — to be confirmed from worktree) |
| Family | Compression-break (volatility expansion from constricted range) |
| Direction | Long only |
| Signal timeframe | 4H |
| Pair set | 28 FX (KH-24 set) |
| Pool size | 2153 trades |
| Population builder | `build_ex_ante_bounded_population` |

Entry trigger: per Arc 9 signal_spec (compression bar followed by directional break — exact mechanical definition to be pulled from worktree Step 1 script `scripts/l_arc_9/step1_plumbing.py`).

## Cohort identification

| Field | Value |
|---|---|
| Target cluster | `cluster_0_individual` (Step 2 K=3, cid=0) |
| Cluster size in pool | 365 trades (17.0% of pool) |
| Cluster centroid (mono / peaks / pullback / ttp_rel) | (pending readout from `results/l_arc_9/step2_clustering/centroids_K3.csv`) |
| Archetype label | (pending §11 routing readout from `capturability_pass_list.csv`) |
| Selected SL | 2.0 × ATR(14) on 4H |

Step 3 capturability metrics (locked from arc run):

| Metric | Value |
|---|---|
| fwd_mfe_p50 | 6.18R |
| frac_wrong_way_pre_peak | 0.000 |
| final_r_mean | +4.423R |
| t-stat | +21.81 |
| shape_tag | (pending readout) |
| Capturability composite | (pending readout) |

## Admission filter

| Field | Value |
|---|---|
| Pipeline | Pipeline E (entry-time admission, rejected trades never enter book) |
| Classifier | LightGBM (`lgb.LGBMClassifier`) |
| Feature count | 28 features |
| Hyperparameters | n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31, min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1, class_weight='balanced', random_state=42, deterministic=True, force_row_wise=True |
| Training validation | 5-fold TimeSeriesSplit CV |
| Training mean CV AUC | 0.7508 |
| Reproduction parity | byte-identical per-fold AUCs verified (0.848380 / 0.735746 / 0.717263 / 0.714402 / 0.738041) |
| **Admission threshold** | **0.40** |

### Feature composition (28 features)

**Baseline (16 features — Step 4 original Pipeline E feature set):**
- Cross-dataset base 8: `body_to_range_ratio`, `upper_wick_ratio`, `lower_wick_ratio`, `range_to_atr_14`, `ret_5bar_atr`, `ret_20bar_atr`, `pos_in_20bar_range`, `rsi_14`
- Arc 9 specific (8 features, see worktree step4 feature catalogue for exact names)

**Expanded (12 features — Pipeline E retry additions):**

D1-lagged context (8 features, one-day backward lag via `merge_asof`):
- `d1_trend_state` — D1 close > D1 open AND D1 close > D1 close[1] (boolean)
- `d1_atr_ratio_to_4h` — D1 ATR(14) / 4H ATR(14)
- `d1_pos_in_20d_range` — (D1 close − D1 20d low) / (D1 20d high − D1 20d low)
- `d1_ret_5d_atr` — D1 5-day return in D1 ATR units
- `d1_rsi_14` — D1 RSI(14)
- `d1_close_above_kijun` — boolean
- `d1_bars_since_swing_high` — bars on D1 since most recent 20-bar swing high
- `d1_bars_since_swing_low` — bars on D1 since most recent 20-bar swing low **(top feature; dominates importance by 3× gain margin)**

Session / time (4 features):
- `session_london` — boolean (08:00-16:00 UTC)
- `session_ny_overlap` — boolean (12:00-16:00 UTC)
- `hour_sin`, `hour_cos` — cyclic hour-of-day encoding (none in top 15)

### Top 15 feature importance

| Rank | Feature | Origin |
|---|---|---|
| 1 | `d1_bars_since_swing_low` | D1 lag |
| 2-15 | (5 more D1 lag features in top 15, balance from 4H baseline) | mixed |

Structural reading: cluster-0 trades concentrate at the early phase of D1 uptrends, identifiable at entry by recency-to-confirmed-D1-swing-low. The 4H IB-trend bar is the timing trigger; the D1 swing-low context is the cohort discriminator.

## Operating point characteristics

| Metric | Value |
|---|---|
| Admission threshold | 0.40 |
| Per-fold precision range | 0.41 - 0.51 |
| Cluster-0 recall (training CV) | ~0.34 (123 / 365 captured on training set basis) |
| Cluster-0 recall (deployment, Step 5 OOS) | ~30% (109 true positives admitted) |
| Total admits across F2-F7 | 236 trades |
| False-positive rate | ~54% of admit set (127 of 236 admits are non-cluster-0) |

False positives are tolerated because the §11 archetype exit policy gracefully degrades on complement trades — false-positive trades hit SL or trail without catastrophic damage. This is the deployment thesis for tolerating <50% precision.

## Exit policy

| Field | Value |
|---|---|
| Exit policy source | §11 archetype routing for `cluster_0_individual` |
| Stop loss | 2.0 × ATR(14) on 4H, locked from Step 3 SL selection |
| Exit mechanics | (pending readout from `capturability_pass_list.csv` — most likely Monotone ascent or Stepwise climber row of §11; confirm before deployment) |
| Risk per trade | 0.5% of account equity (project default; matches KH-24, all Arc 9 experiments) |
| Forward window | 240 bars on 4H |

## Step 5 WFO economics (folds 2-7)

| Metric | Pass-deployable gate | Candidate A actual | Pass? |
|---|---|---|---|
| Worst-fold annualised ROI | ≥ 5% | **+9.63%** (fold F3) | PASS |
| Mean fold annualised ROI | ≥ 8% | **+22.92%** | PASS |
| Worst-fold max DD | ≤ 8% | **1.32%** | PASS |
| All folds positive | required | TRUE (F2-F7) | PASS |
| Trade count per fold | ≥ 15 | TBD per-fold (236 total / 6 folds ≈ 39 avg) | likely PASS |
| Full-data annualised ROI | ≥ 5% | **+19.06%** | PASS |
| Full-data max DD | ≤ 10% | **1.32%** | PASS |

Strict 7-fold §10 evaluation fails because fold 1 has zero admits — Arc 9 signal data start (2020-10-01) coincides with KH-24 fold 1 OOS_start, leaving no anchored-expanding training data for the classifier on fold 1. This is a data-window artifact, not a signal failure.

## Deployment surface position (folds 2-7 captured fraction of floor → ceiling)

| Axis | Floor (raw) | Ceiling (oracle) | Candidate A | Capture |
|---|---|---|---|---|
| Worst-fold ann ROI | −29.64% | +39.45% | +9.63% | 57% |
| Mean fold ann ROI | +6.10% | +61.40% | +22.92% | 30% |
| Worst-fold DD reduction | from 43.61% | to 0.01% | 1.32% | 97% |
| Full-data ann ROI | +0.80% | +60.50% | +19.06% | 31% |
| Full-data DD reduction | from 62.99% | to 0.01% | 1.32% | 98% |

Candidate A captures most available DD reduction (97-98%) and meaningful but conservative ROI capture (30-57%). Compared to Candidate B, this candidate trades ROI for tighter DD discipline.

## Status notes

**Why "pending amendment":**
- Candidate A operates inside the v2.2 §3 locked threshold grid (threshold 0.40 ∈ {0.40, 0.50, 0.60, 0.70}). The economic case for this candidate does not require any v2.x amendment to the grid.
- However, the **strict 7-fold §10 evaluation fails** due to fold-1 warmup data unavailability. Deployment requires either (a) v2.x Step 5 convention allowing F2-F7 as primary when signal data start ≤ F1 OOS_start, or (b) extending Arc 9 Step 1 backward (engine-touching, expensive).
- Candidate A also requires a **formal Step 4 re-pass** under v2.x with the §8 feature budget expanded to include D1-lagged context (currently §8 budget ≤ 38 features predominantly at signal-bar TF; this candidate uses 28 features but only because Pipeline E retry was experimental).

**What deploying this means in plain terms:**
At every 4H bar across the 28 FX pair set, the IB-trend entry trigger is evaluated. When it fires, the LightGBM classifier reads 28 features (8 cross-dataset baseline + 8 arc-specific 4H + 8 D1-lagged + 4 session/time) and outputs a probability. If probability ≥ 0.40, the trade is taken at the next bar's open with SL = 2.0 × ATR(14). The trade exits per the §11 archetype policy (MFE-lock + trail, exact parameters from §11 routing). Roughly 11% of all signals admitted; of those admitted, ~46% are cluster-0 (true positives), the rest are complement trades that the exit policy handles without catastrophic damage.

**Pre-deployment blockers:**
1. Lookahead verification audit — especially cross-timeframe D1 lag integrity. Project history flag from userMemories: same-day D1 close was the bug that invalidated pre-fix KGL WFO results. This audit is non-negotiable.
2. Formal Step 4 re-pass under v2.x protocol with expanded feature budget
3. Fold-1 warmup resolution (convention or backfill)

## Provenance

- Signal pool: `results/l_arc_9/step1_plumbing/trades_all.csv` (2153 trades, 28 pairs)
- Path features: `results/l_arc_9/step1_plumbing/trades_paths.csv`
- Cluster assignment: `results/l_arc_9/step2_clustering/clusters_K3.csv` (cid=0)
- Capturability: `results/l_arc_9/step3_capturability/capturability_pass_list.csv`
- Classifier training: `results/l_arc_9/experiments/pipeline_e_retry/` (commit 0193334)
- Step 5 evaluation: `results/l_arc_9/experiments/step5_lgbm_pipeline_e/` (commit 44de1ca)
- Worktree branch: `claude/bold-brattain-d79817`

## Document control

| Field | Value |
|---|---|
| Candidate identifier | A (locked-grid 0.40) |
| Status | PASS-DEPLOYABLE-PENDING-AMENDMENT (F2-F7) |
| Drafted | 2026-05-18 |
| Companion candidate | B (recall-floor 0.05) — separate doc |
| Closure | pending audit + Arc 7 cross-test |
| Anchor preservation | KH-24 K=4 archetype 3 — preserved, no interaction |
