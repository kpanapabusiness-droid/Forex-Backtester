# L_ARC_OPERATIONAL_SPEC — Per-Step Deliverables, Angle Catalogues, Scoring

**Status:** LOCKED v1.0
**Locked date:** 2026-05-13
**Companion doc:** `L_ARC_PROTOCOL.md` — methodology, verdict logic, discipline rules.
**Modification rule:** Angles may be added without version bump; angles may not be removed without version bump. Component thresholds and effect-size thresholds are locked until protocol v1.1.

---

## Document History

| Version | Date | Scope |
|---------|------|-------|
| v1.0 | 2026-05-13 | Initial lock. Folder structure, naming, full angle catalogues, component-ranked scoring, effect size definitions, haircut as priority tiers, threshold calibration discipline, arc-open template, candidate slug convention, operational-cost-haircut definition with schedule version recording, signal re-validation procedure, filter intent specs, EA porting acknowledgement. |

---

## 1. Purpose

This document is the operational reference used during arc execution. It complements `L_ARC_PROTOCOL.md` (which is the *what* and *why*) by specifying the *how*: exactly which files are produced at each step, what tests run, what thresholds apply, what folder structure, what naming.

Every chat doing arc execution reads both docs at start.

---

## 2. Folder Structure

```
results/
  l_arc_1/                              # one folder per L arc (1..5)
    PHASE_L_ARC_1_OPEN.md               # arc-level open doc, locked at start (template §12)
    PHASE_L_ARC_1_CLOSURE.md            # arc-level closure, written at step 6 end

    step1_verbatim/
      PHASE_L_ARC_1_STEP1.md            # plumbing report
      trades_verbatim.csv
      signals_log.csv                   # every fire (taken or dropped + reason)
      sanity_checks.txt
      signal_revalidation.txt           # canonical-vs-computed 100-bar comparison
      run_manifest.txt                  # input/output sha256s, determinism

    step2_descriptive/
      PHASE_L_ARC_1_STEP2.md            # narrative, findings, no recommendations
      signals_features.csv              # one row per taken trade, full feature set
      trade_paths.csv                   # per-bar held + forward observations
      distributions/                    # full-distribution CSVs per metric
      conditional_breakdowns/           # by pair, fold, session, regime, hour, DOW
      joint_distributions/              # 2D heatmaps as CSV
      shadow_tradesets/                 # entry-delay, SL-distance, time-exit sweeps
      cost_stress/                      # spread-floor sensitivity sweeps
      random_baseline/                  # matched-N random-entry control
      held_bar_evolution/               # basket, spread, correlation, ATR-regime at t=1,3,5,10,20
      forward_horizon_stability.txt     # h=120 vs h=240 stability check result

    step3_extractability/
      PHASE_L_ARC_1_STEP3.md            # contains the VERDICT
      cluster_assignments.csv           # per-trade cluster ID at K=2,3,4,5,6
      cluster_stability.csv             # per-fold cluster sizes and economics
      cluster_effect_sizes.csv          # forward-geometry effect size per cluster vs pool
      predictor_AUC_by_cluster_by_t.csv # the held-bar predictor scan output
      feature_importance.csv            # top features per cluster per t (across model types)
      multicollinearity_top20.csv       # pairwise correlations among top-20 predictors
      stratifications/                  # pair, regime, momentum, hour-in-bar
      look_elsewhere_haircut.csv        # n_features_scanned, BH-corrected predictor tiers

    step4/
      PHASE_L_ARC_1_STEP4.md            # ONE doc covering ALL candidates
      candidate_component_table.csv     # all candidates × all components for ranking
      <candidate_slug_1>/
        config.yaml
        filter_or_exit_spec.md          # includes target step-3 cluster intent (for filters)
        evaluation_metrics.csv
      <candidate_slug_2>/
        ...

    step5_recharacterisation/
      <candidate_slug>/                 # per filter candidate that advances
        PHASE_L_ARC_1_STEP5_<slug>.md
        signals_features.csv            # post-filter
        trade_paths.csv
        distributions/
        cluster_assignments.csv
        cluster_effect_sizes.csv
        filter_intent_check.txt         # did filter achieve named-target cluster shift
        predictor_AUC_by_cluster_by_t.csv

    step6_wfo/
      PHASE_L_ARC_1_STEP6.md            # ONE doc covering ALL WFO runs, dual-tier disposition
      operational_cost_haircut.csv      # per-config gross vs net ROI; header records schedule version
      <config_slug>/                    # one folder per WFO config tested
        config.yaml
        wfo_fold_results.csv
        trades_all.csv
        equity_curve.csv
        summary.txt

  l_arc_2/                              # same structure
  l_arc_3/
  l_arc_4/
  l_arc_5/

  CROSS_ARC_FEATURE_REGISTRY.md         # cross-arc reusable features
  CANDIDATE_HYPOTHESES.md               # existing, repurposed for cross-arc effects
  NEGATIVE_FINDINGS.md                  # features that consistently don't predict across arcs
```

---

## 3. Naming Conventions

- **Folders:** `l_arc_N` (lowercase, underscored). N is the L registry rank.
- **Phase docs:** `PHASE_L_ARC_N_STEPK[_<candidate>].md` — uppercase, underscored.
- **CSVs:** `snake_case.csv`. Same file purpose uses same name across arcs.
- **Candidate slugs:** `<mechanism>_<feature>_<parameter>`, all lowercase underscored. Examples:
  - `filter_concurrent_t13`, `filter_hour_d1_late_excluded`, `filter_basket_movement_low`
  - `exit_trail_2atr`, `exit_cluster_conditional_k4`, `exit_time_h6`, `exit_partial_2atr_then_trail`

---

## 4. Per-Step Deliverables Checklist

### Step 1 deliverables (plumbing test)

- [ ] `PHASE_L_ARC_N_STEP1.md` written.
- [ ] `trades_verbatim.csv` produced — every taken trade across all 7 folds.
- [ ] `signals_log.csv` produced — every fired signal (taken or dropped, with drop reason).
- [ ] `signal_revalidation.txt` — 100-bar deterministic sample compared against canonical `scripts/lchar/run_layer4.py` output; bit-identical match required.
- [ ] Sanity check block: spread-floor hash matches; SL distance = 2.0 × ATR on all trades; D1 lag-1 verified; same-bar entries = 0; lookahead-invariant test passes.
- [ ] Trade count within ±5% of L4 `n_obs_pooled` expectation (soft WARN if outside).
- [ ] Determinism confirmed: two consecutive runs produce byte-identical outputs.

### Step 2 deliverables (descriptive)

- [ ] `PHASE_L_ARC_N_STEP2.md` written, descriptive only — no recommendations.
- [ ] `signals_features.csv` produced — one row per taken trade with full feature set (per §5 angle catalogue).
- [ ] `trade_paths.csv` produced — per-bar held + forward observations.
- [ ] All angles in §5 catalogue run; corresponding outputs in respective subfolders.
- [ ] Full distributions reported for every metric (mean/std/skew/kurt/min/p1/5/10/20/30/40/50/60/70/80/90/95/99/max).
- [ ] Random-entry baseline produced; visible difference from verbatim or flagged.
- [ ] Shadow trade-sets produced for entry delays {N+2, N+3, N+5, N+10}, SL distances {1.0, 1.5, 2.0, 2.5, 3.0} ATR, and time exits {h=1, 3, 6, 12, 24, 48, 120, 240}.
- [ ] MFE/MAE sequence classification produced (MFE_first, MAE_first, simultaneous_bar) with time-difference distribution.
- [ ] Held-bar evolution outputs produced for basket, spread, correlation, ATR-regime at t ∈ {1, 3, 5, 10, 20}.
- [ ] Forward-horizon stability check at h=120 vs h=240 — if distribution still evolving (medians or p95s differ >10%), extend to h=480 and document.
- [ ] Per-pair and per-stratum samples reported with n; cells with n < 30 flagged or pooled.

### Step 3 deliverables (extractability)

- [ ] `PHASE_L_ARC_N_STEP3.md` written, includes the verdict.
- [ ] Cluster outputs at K ∈ {2, 3, 4, 5, 6} plus hierarchical.
- [ ] Cluster stability across folds reported.
- [ ] **Cluster effect sizes** computed per cluster vs pool (forward-geometry metrics per §8) — `cluster_effect_sizes.csv`.
- [ ] Signal-time predictor scan (3c) AUC per cluster, models: logistic regression, depth-3 tree, random forest, gradient boosting.
- [ ] Held-bar predictor scan (3d) AUC per cluster at t ∈ {1, 3, 5, 10, 20}.
- [ ] Stratifications (3e–3h) per pair, regime, session/DOW/hour/hour-in-4H-bar/hour-in-D1-bar, pre-momentum.
- [ ] Multicollinearity check on top-20 predictors (`multicollinearity_top20.csv`).
- [ ] Look-elsewhere haircut applied; n_features_scanned recorded; predictor priority tiers assigned per §6.7 (reporting only — not a gate).
- [ ] **Verdict committed:** PROCEED / AMBIGUOUS-PROBE / CLOSE based on dual-gate (AUC + effect size + cluster size + stability).
- [ ] If verdict = AMBIGUOUS-PROBE: one probe run with recorded reasoning for probe choice, step 3 outputs re-produced on modified population, final verdict committed.
- [ ] If verdict = CLOSE: arc folder marked CLOSED-NULL; closure registers cross-arc-relevant findings to `results/CANDIDATE_HYPOTHESES.md` and consistently-non-predictive features to `NEGATIVE_FINDINGS.md`.
- [ ] If verdict = PROCEED: routing decision (filter / exit / both) committed.

### Step 4 deliverables (filter / exit derivation)

- [ ] `PHASE_L_ARC_N_STEP4.md` — ONE doc covering all candidates.
- [ ] `candidate_component_table.csv` — all candidates × all components.
- [ ] Per-candidate subfolder with config.yaml, filter_or_exit_spec.md, evaluation_metrics.csv.
- [ ] Candidate auto-generation rule applied (per protocol §8) — every qualifying predictor produces an auto-generated candidate; planner may add additional candidates with recorded reasoning.
- [ ] **Filter candidates only:** spec names the target step-3 cluster path characteristics (median mfe_held, median mae_held, median bars_held, dominant sequence class). Documentation requirement, not a gate.
- [ ] Filter candidate viability check: every component clears floor (per §7.1).
- [ ] Exit candidate held-out fold check: derive on folds 1–5, validate on folds 6–7 (per §7.2).
- [ ] Lookahead-invariant test passes for every candidate.
- [ ] Planner decision: which candidates advance to step 5 (filters) or step 6 (exits / unfiltered+exit), with reasoning that references each candidate's BH priority tier where relevant.
- [ ] If zero candidates clear viability: arc closes as clean null per protocol §8.

### Step 5 deliverables (re-characterisation, filter path only)

- [ ] One subfolder per filter candidate that advances.
- [ ] Full step 2 templates re-run on filtered population.
- [ ] Full step 3 templates re-run; verdict on filtered population committed.
- [ ] **Filter intent check** (`filter_intent_check.txt`): comparison of post-filter cluster mix vs step 4 candidate spec's named target. Reports whether the target cluster's path characteristics are over-represented post-filter. Sanity check — does not block advancement.
- [ ] `PHASE_L_ARC_N_STEP5_<slug>.md` written per candidate.
- [ ] If filtered-population step 3 verdict = CLOSE: candidate dropped, not advanced to step 6.

### Step 6 deliverables (WFO integrity test, dual-tier)

- [ ] `PHASE_L_ARC_N_STEP6.md` — ONE doc covering all WFO runs.
- [ ] Per-config subfolder with config.yaml, wfo_fold_results.csv, trades_all.csv, equity_curve.csv, summary.txt.
- [ ] Operational-cost haircut computed per config (`operational_cost_haircut.csv`) per §7.4, with 5ers schedule version recorded in CSV header.
- [ ] Annualisation and trades-per-fold-floor calculations follow protocol §10.1 conventions (calendar-day annualisation; short-fold exclusion from worst-fold annualisation; scaled trade floor for folds < 180 OOS days).
- [ ] Per-config tier disposition: PASS-DEPLOYABLE / PASS-VIABLE / CLEAN-NULL.
- [ ] Cross-config composite verdict (if multiple pass within or across tiers) per protocol §10.3.
- [ ] If zero configs reach PASS-VIABLE: arc closes as clean null per protocol §10.
- [ ] Arc-level closure: `PHASE_L_ARC_N_CLOSURE.md` written with highest-tier disposition and survivor candidate name.
- [ ] Cross-arc registries updated: `results/CANDIDATE_HYPOTHESES.md`, `CROSS_ARC_FEATURE_REGISTRY.md`, `NEGATIVE_FINDINGS.md` appended as appropriate.

---

## 5. Step 2 Angle Catalogue (FULL)

Every arc runs every angle. Outputs go to the indicated subfolder under `step2_descriptive/`.

### 5.1 Marginal distributions (`distributions/`)

For each of the following metrics, full distribution shape (mean/std/skew/kurt/min/p1/5/10/20/30/40/50/60/70/80/90/95/99/max) plus histogram CSV:

- `net_r` — net R after spread (no commission/swap/slippage modelled at trade level per protocol §5)
- `gross_r` — gross R before spread
- `spread_cost_r` — spread as fraction of R
- `mfe_held_atr` — max favourable excursion while held, ATR-normalised
- `mae_held_atr` — max adverse excursion while held, ATR-normalised
- `bars_held` — count of bars from entry to exit
- `exit_reason` — categorical distribution (sl_hit, time_exit, etc.)
- `peak_to_final_r_ratio` — MFE captured fraction
- `mfe_to_mae_ratio` — path asymmetry within held window

### 5.2 Forward-horizon geometry, unconditional on SL (`distributions/forward/`)

At each h ∈ {1, 3, 6, 12, 24, 48, 72, 120, 240} bars after entry:
- `fwd_logret_h{h}` — full distribution
- `fwd_mfe_h{h}_atr` — peak excursion within first h bars from entry, ATR-norm
- `fwd_mae_h{h}_atr` — trough excursion within first h bars from entry, ATR-norm
- `fwd_mfe_to_mae_ratio_h{h}` — distribution

Plus:
- `bars_to_plus_{0.5, 1, 1.5, 2, 3}_atr_capped_240` — survival-style curve
- `bars_to_minus_{0.5, 1, 1.5, 2, 3}_atr_capped_240`
- Fraction of trades that never reach +0.5/+1/+2 ATR within 240 bars
- **Race condition:** distribution of `bars_to_+1atr − bars_to_-1atr` (negative = price hit +1 ATR first; positive = price hit -1 ATR first). This is the primary path-direction effect-size axis.

### 5.3 Forward-horizon stability check (`forward_horizon_stability.txt`)

Compare distributions at h=120 vs h=240. If any (median, p95) pair differs by more than 10%, mark distribution still evolving and re-run step 2 with horizon extended to h=480. Document outcome.

### 5.4 MFE/MAE sequence classification (`distributions/sequence/`)

For every trade, classify by order of peak excursions while held:
- `MFE_first` — time_to_peak_mfe < time_to_trough_mae
- `MAE_first` — time_to_trough_mae < time_to_peak_mfe
- `simultaneous_bar` — both occur within the same bar

Plus: distribution of `time_to_peak_mfe − time_to_trough_mae` (signed bar count). Per sequence class: count, mean R, fwd_mfe/mae distributions, exit-reason mix.

### 5.5 Path complexity (`distributions/complexity/`)

- `oscillation_count` — sign changes in bar-by-bar return during hold
- `monotonicity_ratio` — bars in trade direction / total bars held
- `max_consecutive_with` — longest run of bars in trade direction
- `max_consecutive_against` — longest run against
- `acf1_returns_during_hold` — first-lag autocorrelation of bar returns while held
- `time_to_peak_mfe` — bars from entry to MFE peak
- `time_to_trough_mae` — bars from entry to MAE trough
- `time_from_peak_to_exit` — bars from MFE peak to actual exit
- `r_given_back_from_peak` — peak MFE minus final R

### 5.6 Survival curves (`distributions/survival/`)

- Fraction of trades still open at bar t ∈ {1, 5, 10, 20, 50, 100, 200}
- Mean R conditional on still-open-at-bar-t
- Win% conditional on still-open-at-bar-t

### 5.7 Early-bar predictivity (`distributions/early_bar/`)

For t ∈ {1, 3, 5, 10}:
- Correlation of cum R at t with final R
- Conditional mean final R given cum R at t in each decile
- Win% by cum-R-at-t decile

### 5.8 Win/loss asymmetry (`distributions/asymmetry/`)

- Median winner R, mean winner R, p95 winner R
- Median loser R, mean loser R, p5 loser R
- Median bars-held for winners vs losers
- Drawdown-during-winners (MAE distribution among eventual winners)
- Run-up-during-losers (MFE distribution among eventual losers)

### 5.9 Conditional breakdowns (`conditional_breakdowns/`)

For every metric in §5.1–§5.8, stratified by:
- Pair (28 per-pair tables)
- Fold (1..7)
- Exit reason
- Session (Asia/London/NY/overlap)
- Day of week (Mon..Fri)
- Hour of day (UTC, 00..23)
- Volatility regime (atr_1h_regime bin)
- Pre-momentum (cum_logret_1h_6 sign+magnitude bin)
- Trigger magnitude decile (for L4 signal trigger value)
- **Hour-in-D1-bar** (0..23)
- **Hour-in-4H-bar** (0..3)
- First-bar direction (N+1 open-to-close: with-trade vs against)
- First-bar range (small / medium / large bins relative to ATR)

**Sample size discipline:** any per-pair or per-stratum cell with n < 30 is flagged in the output. Cells with n < 10 are pooled into an "insufficient-n" aggregate rather than reported as a standalone row.

### 5.10 Joint distributions, 2D heatmaps (`joint_distributions/`)

Full 2D heatmaps (not just correlations):
- (`mfe_held`, `mae_held`)
- (`mfe_held`, `bars_held`)
- (`mae_held`, `bars_held`)
- (`mfe_held`, `time_to_peak_mfe`)
- (`mfe_held`, `exit_reason`)
- (`net_r`, `bars_held`)
- (`net_r`, `oscillation_count`)
- (`first_bar_direction`, `final_r`)
- (`mfe_sequence_class`, `final_r`)

### 5.11 Shadow trade-sets (`shadow_tradesets/`)

- **Entry-delay sensitivity:** trade-sets with entry at N+2, N+3, N+5, N+10 opens. Per shadow: full net R distribution, mean R, win%, exit-reason mix. Output: mean R curve as function of entry delay.
- **SL-distance sensitivity:** trade-sets with SL at 1.0, 1.5, 2.0, 2.5, 3.0 × ATR. Per: full distributions, hit rate, mean R, expectancy.
- **Time-exit horizon sensitivity:** trade-sets with time exits at h=1, 3, 6, 12, 24, 48, 120, 240 bars. Per: full distributions, mean R, capture ratio. Output: mean R curve as function of time-exit horizon.

### 5.12 Cost stress (`cost_stress/`)

Spread floor multipliers {0.5×, 1.0×, 1.5×, 2.0×} applied to baseline. Per multiplier: full net R distribution, fraction floored. At what multiplier does mean R cross zero under verbatim exits.

### 5.13 Random-entry baseline (`random_baseline/`)

Random entries on same pairs, same period, matched count. Same SL, time exit, spread treatment. Per: full net R distribution, marginal stats. Verbatim signal's distribution must differ visibly from random; if it doesn't, flag and investigate before step 3.

### 5.14 Held-bar context evolution (`held_bar_evolution/`)

For each open trade, sample at t ∈ {1, 3, 5, 10, 20}:
- Currency basket move per USD / EUR / JPY / GBP since signal time
- Broker spread (in pips, current quote at sample bar)
- Cross-pair correlation regime (rolling 100h correlation matrix max element vs pool average)
- ATR regime (current bar's atr_1h vs at-signal atr_1h, ratio)

Per t, full distribution of each feature. Conditional breakdowns by cluster (when step 3 has clustered).

### 5.15 Cross-pair / portfolio context (signal-time, in `signals_features.csv`)

- `concurrent_signals_same_bar`, `concurrent_signals_within_3h`
- `currency_basket_3h` per USD / EUR / JPY / GBP
- `trade_overlap_at_execution_time`
- `sequential_same_pair_density`

### 5.16 Standard feature set per trade (`signals_features.csv` columns)

Beyond the metrics above, every row includes the L6.0 §14.3 feature set (carried forward unchanged): identity/context, pre-signal 1H context, signal-bar properties, volume (all TFs, NaN audit), baseline distances (all TFs, ATR-normalised, signed), MTF lag-1 features with timestamps logged, time/session/liquidity, cross-pair, classification labels, trade-level outcome, forward-horizon outcome.

**Additions in this protocol (v1.0):**
- `hour_in_4h_bar` ∈ {0, 1, 2, 3}
- `hour_in_d1_bar` ∈ {0, ..., 23}
- `bars_to_next_4h_close`, `bars_to_next_d1_close`
- `mfe_sequence_class` ∈ {MFE_first, MAE_first, simultaneous_bar}
- `first_bar_direction` ∈ {with_trade, against, flat}
- `first_bar_range_atr` — N+1 range as fraction of signal-time ATR

**Note on ATR(14):** the ATR(14)_1H window is a deliberate convention carried from KH-24 / L6.0, not optimised per-arc. Arcs may report sensitivity but do not change the locked window without a v1.1 patch.

---

## 6. Step 3 Angle Catalogue (FULL)

### 6.1 Cluster discovery (3a)

- Feature subset for clustering: `mfe_held_atr`, `mae_held_atr`, `bars_held`, `time_to_peak_mfe`, `time_to_trough_mae`, `peak_to_final_r_ratio`, `oscillation_count`, `monotonicity_ratio`, `mfe_sequence_class`.
- Algorithm: k-means at K ∈ {2, 3, 4, 5, 6}; agglomerative hierarchical (ward linkage). K=2 provides binary good/bad baseline.
- Per K, per algorithm: silhouette score, cluster size distribution, Adjusted Rand Index against K=2 baseline.
- Per cluster: full distributions of net R, mfe, mae, bars-held, exit-reason mix, fwd_mfe_h{24,120}, fwd_mae_h{24,120}.
- Output: `cluster_assignments.csv`.

### 6.2 Cluster stability (3b)

- Per cluster per fold: trade count, mean R, mfe distribution percentiles, fwd_mfe_h24 percentiles. Coefficient of variation across folds.
- Cluster persistence under feature perturbation: drop one feature at a time, re-cluster, report Adjusted Rand Index vs baseline.
- Stability threshold for verdict: per-fold coefficient of variation of mean R within cluster < 0.5; per-fold cluster-size fraction stays within ±30% of pool average.

### 6.3 Signal-time predictor scan (3c)

- Feature set: all `signals_features.csv` features computable at bar N close.
- Per cluster, four models:
  - Logistic regression (L2-regularised)
  - Decision tree (max depth 3)
  - Random forest (200 trees, max depth 5)
  - Gradient boosting (200 estimators, max depth 3, learning rate 0.05)
- Metrics: ROC AUC pooled and per-fold, top features by importance, calibration, 5-fold CV AUC for fold-stability.
- Multicollinearity check on top-20 features (`multicollinearity_top20.csv`).

### 6.4 Held-bar predictor scan (3d)

- Same as 3c, but adding features observable at t ∈ {1, 3, 5, 10, 20}.
- Features at t=1: bar N+1 open direction, OHLC, cum R, MFE, MAE, which side hit ±0.5R first, first-bar range vs ATR.
- Features at t=3,5,10,20: cum R, MFE, MAE at t, bar-by-bar return signs, range expansion, current basket move, current spread, MFE/MAE sequence so far.
- Report earliest t at which AUC crosses {0.65, 0.70, 0.75} per cluster.

### 6.5 Cluster effect sizes (`cluster_effect_sizes.csv`)

Per cluster, computed from step 2 forward-geometry data, reported vs pool:

| Metric | Pool value | Cluster value | Difference | Std-units | Threshold | Pass? |
|--------|-----------|---------------|------------|-----------|-----------|-------|
| median fwd_mfe_h24 | p50_pool | p50_cluster | absolute diff | (cluster − pool) / pool_std | per §8 | bool |
| median fwd_mfe_h120 | ... | ... | ... | ... | per §8 | ... |
| median fwd_mfe_to_mae_ratio_h24 | ... | ... | ... | ... | per §8 | ... |
| race condition median (bars_to_+1atr − bars_to_-1atr) | ... | ... | ... | ... | per §8 | ... |
| fraction reaching +1ATR within 240 bars | ... | ... | ... | proportion | per §8 | ... |

Pass on at least one metric per §8 is required for cluster to count as "effect size adequate" in the verdict.

### 6.6 Stratifications (3e–3h)

Per-stratum cluster mix and conditional forward-geometry distribution for: pair, volatility regime, session/DOW/hour/hour-in-4H-bar/hour-in-D1-bar, pre-momentum. Sample-size discipline per §5.9.

### 6.7 Look-elsewhere haircut — reporting, not gating

The haircut is a transparency tool, not a verdict. Step 3 features are scanned in bulk; some "good-looking" features will inevitably appear by chance. The haircut tells the planner how much to discount each predictor's apparent strength based on how many features were tested. Step 6 WFO on pre-committed configs is the only true verdict on whether a predictor was real edge.

**Procedure:**
- Count total features scanned across 3c + 3d.
- Apply Benjamini-Hochberg correction to per-feature AUC significance (permutation-derived p-values).
- Categorise each predictor into reporting tiers:

| BH-corrected significance | Priority tier | Meaning |
|---------------------------|---------------|---------|
| Clears α=0.05 | **Tier 1** | Strongest prior on being real edge. Default first-choice for step 4 candidates. |
| Clears α=0.20 but not 0.05 | **Tier 2** | Moderate prior. Viable step 4 candidate, prioritise behind Tier 1 unless effect size is materially larger. |
| Does not clear α=0.20 | **Tier 3** | Weak prior on being real edge. Viable step 4 candidate only when BOTH (a) the predicted cluster makes mechanical sense AND (b) one of: cross-arc precedent in 2+ prior arcs, OR effect size in the top 20% of this arc's predictors, OR step 3 filter dry-run shows ΔP_extractable > 1.5× the median for this arc's candidates. Planner records all checks. |

**Operational use:**
- `look_elsewhere_haircut.csv` lists every predictor with: AUC, raw p-value, BH-corrected p-value, tier assignment.
- `n_features_scanned` reported alongside any AUC value in step 3 narrative.
- Step 4 planner cites each candidate's tier; tier informs prior, does not gate.

**Why tiered rather than gated:** a feature that doesn't clear strict correction isn't necessarily random — it may be a real but underpowered effect. Strict gating throws away weak-but-real edges along with noise. The haircut sets priors; step 6 WFO is the actual evidence. The Tier 3 AND-condition closes the most obvious motivated-reasoning path without forbidding unconventional candidates entirely.

### 6.8 Filter dry-run (3i)

For each top signal-time predictor from 3c (across all tiers), compute what the trade-set would look like under that filter (count, mean R, per-fold sign consistency, ΔP toward extractable cluster). Descriptive only.

---

## 7. Scoring Tables

### 7.1 Filter candidate component table

Each filter candidate evaluated on every component. Candidate **viable** only if ALL components clear their floors.

| Component | Definition | Floor for viability |
|-----------|------------|---------------------|
| ΔP_extractable_archetype | P(extractable_cluster \| filter passes) − P(extractable_cluster \| full population) | > 0 |
| ΔP_non_extractable_archetype | P(non_extractable_cluster \| filter passes) − P(non_extractable_cluster \| full population) | < 0 |
| MFE_geometry_preservation | median(fwd_mfe_h24 \| post-filter, extractable cluster) / median(fwd_mfe_h24 \| pre-filter, extractable cluster) | within [0.85, 1.15] |
| Per_fold_AUC_stability | std(per-fold predictor AUC) / mean(per-fold predictor AUC) | < 0.30 |
| Retained_per_fold_count | min(per-fold count post-filter) | ≥ 100; auto-disqualified if < 50 |
| Mean_R_under_verbatim_exits | mean net R of post-filter trade-set under step 1 verbatim exit | > 0 |
| Target step-3 cluster characteristics | Documentation: spec names target cluster's median mfe_held / mae_held / bars_held / sequence class | Present (no quantitative floor) |

BH priority tier (per §6.7) reported alongside the table as planner context — not a viability floor.

Component table is `candidate_component_table.csv`. Planner reads side-by-side and picks which advance. No single composite score.

### 7.2 Exit candidate evaluation (multi-metric report)

Per exit candidate, report all of:
- Mean net R full distribution (pool and per-cluster)
- Win% and expectancy
- Per-fold mean R and ROI (sign consistency check)
- Capture ratio: median and full distribution of (realised R / fwd_mfe_h_optimal). **Primary exit-quality metric.**
- Drawdown-during-winners reduction vs verbatim
- **5-fold derive, 2-fold validate held-out check** — derive folds 1–5, validate folds 6 and 7.

**Held-out fold floors for advancing to step 6:**
- Mean R on held-out folds (6, 7) > 0
- Capture ratio on held-out folds > capture ratio under verbatim exit on same folds
- Per-fold sign consistency on held-out folds: both folds positive (sign-flip on held-out is auto-disqualification)

### 7.3 Step 6 cross-config verdict

Disposition uses protocol §10.1 calculation conventions (calendar-day annualisation, short-fold exclusion from worst-fold annualisation, scaled trades-per-fold floor) and protocol §10.2 tier thresholds.

Cross-config verdict between multiple passing candidates per protocol §10.3:
1. **Tier disposition first.** PASS-DEPLOYABLE candidates outrank PASS-VIABLE.
2. **Within same tier:** worst-fold ROI (higher better), worst-fold DD (lower better), fold-to-fold ROI consistency (std, lower better), trade count headroom.
3. **Tiebreaker:** more trades held at equal performance preferred.

No single composite formula. Decision recorded in `PHASE_L_ARC_N_CLOSURE.md`.

### 7.4 Operational Cost Haircut (for PASS-DEPLOYABLE evaluation)

**Per-trade cost modelled in steps 1–6 trade accounting:** spread per `configs/spread_floors_5ers.yaml`. This is the only per-trade cost.

**Operational costs NOT modelled per-trade:**
- Commissions (per 5ers schedule; for the funded account these are realised primarily as overnight swap on positions held past 22:00 broker time)
- Overnight swap (per pair, per night held)
- Slippage (assumed neutral over many trades — adverse slippage cancels favourable slippage in expectation)

**Operational-cost haircut procedure at step 6:**
- Compute per-config aggregate haircut as: (total overnight-bars-held across all WFO trades) × (current 5ers swap schedule average per overnight-bar-held per lot, weighted by pair distribution) ÷ (account balance × fold duration). Express as annualised percentage drag.
- Apply haircut to gross-of-haircut ROI numbers to produce net-of-haircut ROI numbers.
- PASS-DEPLOYABLE evaluation uses net-of-haircut numbers.
- PASS-VIABLE evaluation uses gross-of-haircut numbers (since PASS-VIABLE is "is there real edge here" not "is it deployment-ready").
- DD constraint (< 8%) applies to both tiers regardless — prop-firm hard limit, not subject to haircut adjustments.

**Implementation note:** if the precise haircut schedule is unavailable at step 6 time (e.g., swap-free account status under negotiation), compute haircut at a conservative upper bound (e.g., highest 5ers schedule × 1.5) for PASS-DEPLOYABLE evaluation. PASS-DEPLOYABLE under conservative haircut is robust; if the actual cost schedule comes in lower, the candidate looks better — never worse than projected.

**Schedule version recording.** The 5ers swap schedule version used for the haircut calculation is recorded in a header row of `operational_cost_haircut.csv` containing: date of schedule retrieval, source URL or document hash, conservative-upper-bound multiplier (if applied), retriever's identity. If the schedule is updated, prior haircut calculations remain interpretable and re-computable. Output CSV body: per WFO config per fold — gross ROI, applied haircut, net ROI.

---

## 8. Effect Size Definitions and Thresholds

Forward-geometry effect sizes for cluster-vs-pool comparison (used in step 3 verdict logic per protocol §7).

**A cluster passes the effect size condition if AT LEAST ONE of the following holds:**

| Metric | Calculation | Threshold |
|--------|-------------|-----------|
| ΔMedian fwd_mfe_h24 | median(fwd_mfe_h24 \| cluster) − median(fwd_mfe_h24 \| pool) | ≥ 0.10 ATR-norm OR ≥ 0.4 × pool_std(fwd_mfe_h24) |
| ΔMedian fwd_mfe_to_mae_ratio_h24 | median(fwd_mfe / fwd_mae \| cluster) − median \| pool | ≥ 0.25 |
| Δ Race condition median | median(bars_to_+1atr − bars_to_-1atr \| cluster) − median \| pool | ≥ 5 bars |
| Δ Fraction reaching +1ATR within 240 bars | P(reaches +1ATR \| cluster) − P(reaches +1ATR \| pool) | ≥ 0.10 absolute |

**Effect size thresholds are calibrated against L4 magnitudes** that registered as "good signal" in the L registry (DSR > 0.95). Recalibration discipline is in §9.

**Why these metrics, not mean R:** mean R pre-exit is dominated by the wrong exit policy. Differences in mean R between clusters at this stage reflect "which cluster's trades resolved well at h=1," not "which cluster has structural upside available to extract." Forward-geometry metrics are exit-agnostic. Mean R returns as a primary metric in step 4 exit evaluation, where the exit policy is the variable being engineered.

**AUC alone is insufficient.** AUC of 0.99 separating two clusters with identical fwd_mfe distributions is statistically clean but operationally pointless. AUC says the boundary is discriminable; effect size says the discriminated thing matters economically. Both gates conjunctive in the step 3 verdict.

---

## 9. Threshold Calibration Discipline

Pre-committed thresholds make systematic errors in unknown directions. The thresholds in §7.1, §7.4, §8, and protocol §10 are best-estimate priors calibrated against L4 magnitudes that registered DSR > 0.95. They are not assumed correct on first contact with arc data — they are assumed correct enough to commit to, with a disciplined revision path across arcs.

**Within-arc:** thresholds do not move. Loosening a threshold when an arc's results look bad reintroduces the post-hoc rationalisation the protocol exists to prevent. Within-arc verdicts stand on the locked thresholds.

**Cross-arc:** thresholds may be re-evaluated after Arc 1, Arc 2, and Arc 3 close. A v1.1 patch with revised thresholds is justified by **any one** of the following patterns:

1. All 3 arcs produce CLOSE verdicts with clusters falling within 30% of an effect-size threshold without clearing it → thresholds likely too tight.
2. Multiple step 6 WFO configs reach PASS-DEPLOYABLE when their underlying step 3 verdicts were PROCEED with clusters only marginally clearing thresholds → thresholds calibrated correctly or could tighten slightly without losing real edges.
3. Multiple step 6 WFO configs fail PASS-VIABLE when their step 3 verdicts were PROCEED → thresholds likely too loose; false positives reaching step 6.
4. A specific feature consistently predicts PASS-DEPLOYABLE survivors across multiple arcs without clearing Tier 1 BH haircut → haircut tier definition may need revisiting.
5. Cross-arc evidence shows a feature with consistent predicted-cluster type across 2+ arcs but inconsistent threshold clearance → per-arc variance under-accounted; recalibration toward arc-relative effect sizes may be warranted.

**Revision procedure:** re-planning chat, explicit comparison to L4 calibration plus closed-arc evidence, v1.1 lock, document history entry. In-flight arcs continue under their locked protocol version; new arcs run under v1.1.

---

## 10. Lookahead Invariant Requirements

### 10.1 Filter lookahead test

Synthetic perturbation: perturb the OHLC of bars N+1 forward on a random sample of 100 signal bars. Re-evaluate the filter on those signal bars. The filter decision must be byte-identical to the unperturbed result. Hard fail on any difference.

### 10.2 Exit lookahead test

For an exit decision at bar t: perturb OHLC at bars t+1 forward on a random sample of 100 held trades. Re-evaluate the exit decision at bar t. Decision must be byte-identical. Hard fail on any difference.

### 10.3 MTF feature lag test

For every feature that crosses timeframes: assert most-recently-completed convention. 4H feature at 1H bar N uses the 4H bar that closed strictly before the 4H bin containing T_N started. D1 feature uses the D1 bar of the strictly prior calendar date. W1 strictly prior week. Runtime assertions hard-fail on violation.

### 10.4 Feature lag audit deliverable

Every step that produces features writes a `feature_lag_audit.txt` reporting, on a deterministic 100-row sample, the lag relationship between each feature's timestamp and the signal bar timestamp.

---

## 11. Reporting Standards

### 11.1 Full distributions, never medians-only

Every numeric metric reports: mean, std, skew, kurt, min, p1, p5, p10, p20, p30, p40, p50, p60, p70, p80, p90, p95, p99, max. Plus histogram CSV.

### 11.2 2D distributions reported as heatmaps

Full 2D heatmaps (binned counts) as CSV. Not correlation coefficients.

### 11.3 Per-fold breakdowns mandatory

Every aggregated metric also reported per-fold (F1..F7).

### 11.4 Sample size discipline

Per-pair or per-stratum cells with n < 30 are flagged. Cells with n < 10 are pooled into an "insufficient-n" aggregate.

### 11.5 No recommendations in step 2 / step 3 reports

Step 2 and step 3 narrative reports are descriptive only. Action-shaped language lives in step 4 onward or in `results/CANDIDATE_HYPOTHESES.md`.

### 11.6 Determinism receipts

Every phase doc ends with a manifest section reporting input/output sha256s.

### 11.7 Selection-bias context

Wherever an AUC value, predictor ranking, or "best feature" claim appears, n_features_scanned context appears alongside. BH tier reported for every predictor mentioned by name.

---

## 12. Arc-Open Doc Template

Every arc starts with `PHASE_L_ARC_N_OPEN.md` containing:

### §1. Header
- Arc identifier, protocol version locked at, lock date
- Predecessor docs with sha256: `L_ARC_PROTOCOL.md`, `L_ARC_OPERATIONAL_SPEC.md`, `docs/LCHAR_TOPN_REGISTRY.md`, `configs/spread_floors_5ers.yaml`

### §2. Signal definition
Verbatim citation from `docs/LCHAR_TOPN_REGISTRY.md` (or `results/CANDIDATE_HYPOTHESES.md` for hypothesis-derived arcs). Reference to canonical signal source.

### §3. Step 1 configuration
Entry, SL, time exit, spread floor sha256, exposure cap, risk per trade.

### §4. Pre-commit expectations
Expected trade count per fold (range based on L4 measurement); expected most-likely null mode; expected most-likely pass mode.

### §5. Pair set
All 28 pairs, verbatim from L6.0 §5.

### §6. WFO structure
7 anchored expanding folds, OOS coverage 2020-10-01 → 2026-01-01. Folds 1–5 derive; folds 6–7 held-out per §7.2. Per-fold OOS calendar duration recorded for protocol §10.1 annualisation calculations.

### §7. Out of scope
Modifications to registry, spread floor, locked artefacts. Deployment decisions (post-arc per protocol §17).

### §8. Sign-off
Signed off by, date locked, simulator git commit.

---

## 13. Chat Handover Requirements

Every chat ends with a handover note that the next chat reads first:

1. What step is now complete.
2. Path to the just-committed phase doc.
3. Path to the docs the next chat reads first (always: `L_ARC_PROTOCOL.md`, `L_ARC_OPERATIONAL_SPEC.md`, the prior chat's phase doc).
4. Open questions / judgement calls deferred to the next chat (planner's decision, not pre-committed).

Handover note is a section at the end of the phase doc itself, titled `## Handover to Next Chat`.

---

## 14. Cross-Arc Registries

### 14.1 CROSS_ARC_FEATURE_REGISTRY.md

Append-only. One entry per feature that has shown top-10 predictor importance in 2+ arcs (loosened from prior "AUC > 0.65 + effect size" trigger — top-10 importance catches features that consistently surface even when arc-specific thresholds vary).

Entry fields:
- Feature name (exact column name)
- Definition (mechanical, computable with no lookahead)
- Source arcs with effect size and BH tier per arc
- Predicted-cluster description per arc
- Cross-arc consistency assessment
- Notes / open questions

### 14.2 CANDIDATE_HYPOTHESES.md

Existing schema. Every effect observed in any step of any arc that *might* support a fresh-arc hypothesis. Entries are evidence, not a roadmap.

### 14.3 NEGATIVE_FINDINGS.md

Features that consistently DON'T predict (rank outside top-30 across all step 3 models) in 2+ arcs. Saves future arcs time by surfacing known dead ends. Entry fields: feature name, source arcs with rank per arc, notes.

---

## 15. EA Porting and Parity Validation (Acknowledgement)

When a step 6 PASS-DEPLOYABLE survivor is identified, an EA porting workstream opens separately from the arc protocol. This workstream is **out of scope for the per-arc protocol** but flagged here so that arc closure docs include the information needed to start it:
- Complete config.yaml for the surviving system
- Per-trade decision log for the WFO trade-set (used as the parity benchmark)
- Signal-time and held-bar feature requirements with computation conventions
- MTF lag convention and lookahead-invariant proofs

EA implementation, MT5 Strategy Tester parity check, VPS deployment, and 5ers account configuration are addressed in a separate doc opened when the first PASS-DEPLOYABLE survivor exists. No EA work is started before then.

---

## 16. Sign-Off

**Signed off by:** _______________
**Date locked:** _______________

---

*End of L_ARC_OPERATIONAL_SPEC v1.0. Methodology reference at `L_ARC_PROTOCOL.md`.*
