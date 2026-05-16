# L_ARC_PROTOCOL v2.0

> Status: Active protocol for Arcs 3+. Supersedes v1.0 + amendments (v1.1, v1.2, v1.3).
> v1.x is the historical protocol for prior arcs (Arcs 1, 2). v2.0 governs Arc 3 onward.
> Calibration anchor: KH-24 verbatim trade-paths (`results/kh24/trades_paths.csv`).
> Evidence base: v2.0 archetype diagnostic (PR #129), v2.0 predictability investigation (PR #130).

---

## §0. Purpose, changes, and core terminology

### Purpose

L_ARC_PROTOCOL governs the discovery and validation of trading systems through structured arcs. v2.0 is a structural rewrite — the clustering basis and pipeline shape fundamentally change from v1.x.

### Core terminology

- **Cluster** — output of k-means clustering on path-shape features. Algorithmic group of trades. Numbered (cluster 0, cluster 1, ...).
- **Centroid** — mean of a cluster's trades across the four path-shape features. A cluster's centroid is four numbers (e.g., monotonicity=0.58, local_peaks=14, pullback=0.02, time_to_peak_rel=0.85). Describes where the cluster sits in feature space.
- **Centroid pattern** — a rule about centroid values (e.g., "monotonicity ≥ 0.55 AND local_peaks 5-30 AND pullback ≤ 0.5R"). Defined per archetype in §11.
- **Archetype** — descriptive label assigned to a cluster whose centroid matches an archetype's pattern. Examples: "Monotone ascent", "Stepwise climber", "Early-peak hold". Multiple clusters may map to the same archetype.
- **Exit policy** — rules for exiting trades that belong to an archetype. One exit policy per archetype, defined in §11.
- **Filter** — a classifier + probability threshold that admits or rejects trades. Pipeline E filters admit at entry. Pipeline D1 doesn't use filters in this sense — D1 uses a classifier to branch exit policy at bar N.

Flow: `trades → cluster (k-means) → centroid (mean features) → archetype (centroid pattern match in §11) → exit policy (same §11 row)`.

### What v2.0 changes from v1.0

| Dimension | v1.0 | v2.0 |
|---|---|---|
| Clustering basis | Forward-geometry magnitude | Path-shape (outcome-blind) |
| Primary gate | Forward-geometry effect size + cluster predictability | Two gates: capturability + extractability |
| Pipeline shape | Single — entry filter | Two — entry-filtered (E) or deferred-identification (D1) |
| Identification time | Entry only | Entry OR t>0 |
| Filter stacking | Implicitly discouraged | Two-tier rule (§12) |
| Step count | Variable | 6 steps, fixed |
| Documentation | Per-step result docs | One live arc doc, finalised at arc end (§13) |
| Commit workflow | PR for everything | Direct-to-main for analysis; PR for engine (§13) |

### Core thesis

**Extractability and capturability are independent properties. Both required. Neither sufficient.**

- Capturability: archetype path is clean AND magnitude is meaningful (R to extract).
- Extractability: archetype membership predictable from features observable in time to act.

Extractability is the harder constraint. Capturability is about subsequent price action; extractability is about information available at decision time.

---

## §1. Methodology principles (locked)

Short statements. Operational detail lives in §12.

### Inherited from v1.0

1. Structure-first, not indicator-first.
2. No lookahead, no repainting, real execution only. Signal bar N close → entry bar N+1 open.
3. Ex-ante populations always (`build_ex_ante_bounded_population` mandatory).
4. D1 data uses one-day lag (`merge_asof` backward, pre-shifted date).
5. Clean labels = evaluation tool only, never for population selection.
6. Volume = veto only.
7. One change per phase, pre-committed gate.
8. Locked thresholds do not move within an arc. Cross-arc calibration only.
9. Full distributions, never medians-only (p5/p25/p50/p75/p95 minimum).
10. Config-driven via YAML. No hardcoded thresholds.
11. Deterministic outputs, CI enforced. Two-run byte-identical.
12. Python backtester is source of truth. MT5 EA mirrors it; backtester wins.
13. GPT-4 and Aider permanently excluded.

### New in v2.0

14. Clustering features are outcome-blind.
15. Two gates, sequenced: capturability (Step 3) → extractability (Step 4). Each failure kills the archetype.
16. Long-only baseline. Shorts as retrofit only after long-only system passes WFO.
17. Pipeline assignment committed per archetype at Step 4. Dual-assignment runs both at Step 6; ship best.
18. AUC thresholds: Pipeline E ≥ 0.65, Pipeline D1 ≥ 0.60. RF is the gate, logistic is informational.
19. Filter stacking — two-tier rule: Tier 1 clears the gate, Tier 2 lifts above it. Details in §12.
20. KH-24 K=4 archetype 3 is the calibration anchor. Gates that cause it to fail v2.0 extractability overall (both pipelines simultaneously) are wrong by construction.

---

## §2. Capturability and extractability framing

### Capturability (Step 3)

Hard floors, conjunctive — all criteria must pass.

| Property | Threshold | Source |
|---|---|---|
| Clean shape | monotonicity_centroid ≥ 0.55 | Path-shape centroid |
| Limited oscillation | local_peaks_centroid within shape-appropriate ceiling (see §11) | Path-shape centroid |
| Magnitude | fwd_mfe_h240_p50 ≥ 1.5R | Forward geometry |
| Direction | frac_reach_1R ≥ 0.70 AND frac_wrong_way ≤ 0.30 | Forward geometry |
| Internal consistency | shape_tag ∈ {tight_unimodal, heavy_right_tail} | Distribution shape |
| Size viability | size_fraction_of_pool ≥ 0.10 | Cluster size |

KH-24 K=4 archetype 3 passes: monotonicity 0.58, frac_reach_1R 1.00, fwd_mfe_p50 5.4R, frac_wrong_way 0.04, size 0.13.

### Extractability (Step 4)

| Pipeline | Predictability source | AUC threshold | Constraint |
|---|---|---|---|
| E (entry filter) | RF classifier on entry features (≤ 38 features total) | RF AUC ≥ 0.65 | None |
| D1 (deferred policy) | RF classifier on entry features + path-so-far at bar t | RF AUC ≥ 0.60 | Trades-exited-before-t ≤ 30% of archetype pool |

Disjunctive across pipelines: archetype passes extractability if it clears E OR D1. Assignment:
- Clears E only → Pipeline E
- Clears D1 only → Pipeline D1
- Clears both → both run at Step 6 WFO; ship best; may also run in unison

---

## §3. Pipeline shapes

### Pipeline E — Entry-filtered

1. Train entry RF classifier on features at signal bar.
2. Signal fires → classifier evaluates → `P(archetype) ≥ threshold` admits trade.
3. Fixed exit policy per archetype from §11.

**Backtester:** existing infrastructure sufficient.

**Threshold sweep at Step 4:** classifier output threshold ∈ {0.40, 0.50, 0.60, 0.70}. Select threshold maximising precision subject to recall ≥ 0.60.

### Pipeline D1 — Deferred-identification

1. Train t=N RF classifier on features at bar offset N.
2. Signal fires → enter with uniform pre-t SL (entry − 2.0 × ATR(14), matching KH-24 baseline) → hold to bar N. Archetype-specific SL replaces the pre-t SL at bar N once classifier verdict lands (PR 2 — not in PR 1).
3. At bar N: classifier evaluates → apply archetype exit policy from §11.
4. Trades classifier deems untradeable at bar N: close at market on bar N+1 open. Expected outcome is near-break-even small loss after spread, given the short hold and pre-t SL — engine books the realised PnL.

**Backtester:** PR 1 lands plumbing + close-at-market (uniform pre-t SL, classifier evaluation at bar N, untradeable trades close at N+1 open). PR 2 ships per-archetype exit policies (archetype-specific SL replacement at bar N, custom trail distances, TP targets). Pipeline D1 archetypes can reach Step 6 WFO under PR 1 for the close-at-market path; the full archetype-policy WFO awaits PR 2.

**Early-exit attrition:** smaller N preserves addressable pool but weakens signal; larger N improves signal but loses trades. 30% exclusion floor balances this.

### Unison (E + D1)

For dual-assigned archetypes: filter at entry + D1 policy at bar N. Step 6 evaluates E alone, D1 alone, and E+D1 combined; ships best.

---

## §4. Arc lifecycle (6 steps)

```
┌──────────────────────────────┐
│ Step 1: Plumbing             │ → gate: deterministic, no lookahead, pool ≥ 500
├──────────────────────────────┤
│ Step 2: Path-shape           │ → gate: silhouette ≥ 0.30, no cluster > 90%, min cluster ≥ 30
│         clustering           │
├──────────────────────────────┤
│ Step 3: Capturability        │ → gate: ≥ 1 archetype passes §2 floors
│         characterisation     │
├──────────────────────────────┤
│ Step 4: Extractability +     │ → gate: ≥ 1 capturable archetype clears Pipeline E or D1
│         artefact production  │
├──────────────────────────────┤
│ Step 5: Cross-fold stability │ → gate: sign consistency, size variance ≤ 3.0
├──────────────────────────────┤
│ Step 6: WFO truth +          │ → pass-deployable OR pass-viable (small archetypes route here)
│         pass-deployable gate │
└──────────────────────────────┘
              │
              ▼
          DEPLOY
```

Gate failure kills arc (Steps 1-3) or specific archetypes (Steps 4-6). Surviving archetypes proceed independently from Step 5.

---

## §5. Step 1 — Plumbing

**Operation:** generate full trade pool across data period, single pass. WFO fold structure applied later (Steps 5, 6).

**Outputs:**
- `trades_all.csv` — per-trade summary
- `trades_paths.csv` — per-bar trade-paths with forward window to bar offset 240

**Gate:**
- Deterministic (two-run byte-identical)
- No lookahead violations (CI)
- Spread treatment matches `SPREAD_SEMANTICS_LOCK.md`
- Pool size ≥ 500 trades

**Discipline:**
- No filtering, no analysis. Pure population.
- Pairs with n < 30 flagged but not removed.
- Bar 240 = 40 days for 4H. Other timeframes: verify 95th pct bars_held under cap.

---

## §6. Step 2 — Path-shape clustering

### Path-shape features (outcome-blind)

Definitions:
- `monotonicity_ratio_in_profit` — among `close_r > 0` bars, fraction where `close_r ≥` previous in-profit bar
- `local_peaks_count` — count of bars where `mfe_so_far_r >` previous bar
- `pullback_magnitude_median` — for consecutive peak pairs: `earlier peak's mfe_so_far_r − min(close_r between peaks)`; median across pairs
- `time_to_peak_mfe_relative` — `time_to_peak_mfe / max(bars_held, 1)`, capped at 1.0

Edge cases:
- monotonicity: zero in-profit bars → 0
- local_peaks: `bars_held = 0` → 0
- pullback: < 2 peaks → 0
- time_to_peak: trade never in profit → 0

`pullback_magnitude_median`: literal definition evaluates to 0 because mfe_so_far_r is monotone non-decreasing. Operational definition uses `min(close_r)` between peaks (Open-08 tracks possible rethink).

### Clustering

- KMeans, `random_state=42`, `n_init=10`, `max_iter=300`
- StandardScaler pre-applied
- Sweep K ∈ {3, 4, 5, 6, 7}

### Gate
- ≥ 1 K produces silhouette ≥ 0.30
- No cluster > 90% of trades
- All clusters ≥ 30 trades

### K selection

Highest silhouette satisfying gate. If multiple K tie, smaller K preferred (parsimony).

### Cluster → archetype labelling

For each cluster at the chosen K, compute its centroid (mean of the four path-shape features). Match the centroid against archetype patterns in §11. Cluster gets the archetype label of the matching row.

Same-archetype clusters (e.g., two clusters at the chosen K both match "Stepwise climber" pattern) are aggregated for downstream Step 3-6 evaluation — they share an exit policy.

Boundary clusters (centroid between two patterns): assign by empirical test on per-fold internal validation (which exit family produces higher capture ratio for the cluster).

### Degenerate features

If any feature > 80% at single value, flag. Two+ degenerate features → halt arc, signal lacks shape heterogeneity.

**Outputs:** `clusters_K<k>.csv`, `centroids_K<k>.csv`, `path_features.csv`, `silhouette_K<k>.txt`, `archetype_assignments.csv`.

---

## §7. Step 3 — Capturability characterisation

Per archetype (aggregating same-label clusters):

**Identity:** size_count, size_fraction_of_pool, archetype centroid (mean across constituent clusters' trades).

**Forward geometry:** fwd_mfe_h240 percentiles, final_r percentiles, final_r_mean, t-stat, frac_reach_1R, frac_reach_2R, frac_wrong_way, pct_peak_and_collapse.

**Distribution:** shape_tag (no_magnitude / tight_unimodal / heavy_right_tail / bimodal / scattered / unclassified), mass-in-band {0-0.5R, 0.5-1R, 1-2R, 2-5R, >5R}.

Apply §2 capturability criteria. Archetypes passing carry to Step 4.

**Arc-level gate:** zero archetypes pass → arc dies.

**Outputs:** `archetype_summaries.csv`, `archetype_<label>_distribution.csv` (full percentiles), `capturability_pass_list.csv`.

---

## §8. Step 4 — Extractability investigation + artefact production

Single step. Produces predictability profile AND deployable artefact for surviving archetypes.

### Angle E — entry-time predictability

**Features** (≤ 38 total, no lookahead):
- 8 cross-dataset base: `body_to_range_ratio`, `upper_wick_ratio`, `lower_wick_ratio`, `range_to_atr_14`, `ret_5bar_atr`, `ret_20bar_atr`, `pos_in_20bar_range`, `rsi_14`
- Arc-specific expanded from `feature_catalogue.yaml`, ≤ 30 features

**Models:** Logistic Regression + Random Forest (settings in §12). Both reported. RF is the gate (§1.18); logistic is informational.

**Validation:** 5-fold CV ROC-AUC.

### Angle D1 — t>0 predictability

For t ∈ {1, 2, 3, 4, 5, 10}:
- Path-so-far features at t: `close_r_at_t`, `mfe_so_far_r_at_t`, `mae_so_far_r_at_t`, `bars_in_profit_at_t`, `local_peaks_so_far_at_t`, `monotonicity_so_far_at_t`, `velocity_first_t`
- Plus 8 cross-dataset base entry features
- Exclude trades with `bars_held < t`

RF only at each t. Record exclusion count per t.

(t values are default — t ∈ {6, 7, 8, 9} can be added per-arc if the {1, 2, 3, 4, 5, 10} sweep suggests interesting behaviour between 5 and 10.)

**Smallest-t selection rule:** the chosen t for an archetype is the smallest t in the sweep where (a) RF AUC ≥ 0.60 AND (b) trades-exited-before-t ≤ 30% of archetype pool. Smaller t preferred because larger addressable pool and shorter wait before policy switches.

### Filter selection (per archetype)

**Step A — single-classifier baseline.**

Train an RF classifier on the full Angle E feature set (8 base + up to 30 arc-specific). If RF AUC ≥ 0.65, lock as Pipeline E classifier. Done.

**Step B — feature subset selection (if Step A fails).**

If full-feature RF doesn't clear:
- Compute RF feature importances on the full-feature model
- Try smaller subsets: top-5, top-10, top-15 features by importance
- Try forward selection: start with the highest univariate-AUC feature, add features one at a time, keep additions that improve CV AUC
- If any subset clears 0.65, lock as Pipeline E classifier. Done.

**Step C — stacking (only if A and B fail).**

A single classifier hasn't reached the gate. Try stacking:
- Train two classifiers (independent feature subsets, or different model types)
- "Stacked filter passes" = both classifiers admit the trade (intersection)
- Compute combined AUC

Under candidate budget: ≤ 30 total combinations evaluated across all archetypes in this arc (Tier 1 stacking).

If 3+ classifiers needed to clear → archetype is not naturally extractable on Pipeline E.

### Tier 2 lift (after gate cleared)

Once a single classifier (Step A or B) clears the gate, additional classifiers may be stacked freely as lift candidates. Validation: at Step 6, lift must improve WFO worst-fold ROI by ≥ 1% vs single-classifier baseline without DD degradation. Baseline retained as fallback.

### Pipeline assignment & artefact production

Per archetype:
- Clears E threshold (RF AUC ≥ 0.65) → train final E classifier on full training period; sweep threshold {0.40, 0.50, 0.60, 0.70}; select max precision with recall ≥ 0.60. Save `archetype_<label>_E_classifier.joblib` + `archetype_<label>_E_filter.yaml`.
- Clears D1 threshold (RF AUC ≥ 0.60 at chosen t, exclusion ≤ 30%) → train t-bar classifier; sweep threshold; define exit policy from §11. Save `archetype_<label>_D1_classifier.joblib` + `archetype_<label>_D1_policy.yaml`.
- Clears both → produce both artefacts; Step 6 decides.
- Clears neither → archetype dies.

(joblib preferred over pickle for sklearn models. ONNX/PMML for long-term portability is an Open-11 concern.)

### Reporting per archetype

Logistic AUC + RF AUC. RF feature importances (top 10). Confusion matrix at chosen threshold. Per-fold AUC variance.

Large RF-logistic gap (≥ 0.10) = non-linear dynamics in feature relationships. Small gap = feature set is binding (richer features may help; non-linearity won't).

### Arc-level gate

Zero capturable archetypes pass extractability → arc dies.

**Outputs:** `predictability_angle_E.csv`, `predictability_angle_D1.csv`, `extractability_pass_list.csv`, per-archetype classifier files + policy YAMLs.

---

## §9. Step 5 — Cross-fold stability

Per design: apply filter / D1 policy to all 7 WFO folds. Per fold compute archetype size, final_r_mean, t-stat, fold ROI, fold max DD.

**Gate (conjunctive):**
1. Sign consistency: final_r_mean positive across all folds
2. Size variance: max-fold / min-fold ≤ 3.0
3. DD ceiling: worst-fold archetype-attributable DD ≤ 2x median-fold DD

**Discipline notes:**
- Trade-count variance alone (without sign flip or DD blowup) = not a kill. Flag as regime-dependent, advance.
- Single-fold-outlier sign flips → chat-level judgement.
- Trade count per fold informational at Step 5 (reported but not gated). Hard floor applies at Step 6.

Designs failing stability killed before Step 6.

**Outputs:** `fold_stability_<label>.csv`, `stability_pass_list.csv`.

---

## §10. Step 6 — WFO truth + pass-deployable gate

Run full WFO with surviving designs. Plus full-data backtest (all folds combined).

### Pass-deployable

| Metric | Threshold |
|---|---|
| Worst-fold annualised ROI | ≥ 5% |
| Mean fold annualised ROI | ≥ 8% |
| Worst-fold max DD | ≤ 8% |
| All folds positive | (implied by worst-fold > 0) |
| Trade count per fold per design | ≥ 15 |
| Full-data ROI | ≥ 5% |
| Full-data max DD | ≤ 10% |

### Pass-viable (weaker — route for small archetypes)

Small archetypes (size_fraction ~0.10 = ~50 trades pool ~7 per fold) cannot meet pass-deployable's ≥ 15 trades per fold. Pass-viable accepts smaller designs that still demonstrate structural edge.

Pass-viable iff:
- Worst-fold ROI > 0%
- Worst-fold DD ≤ 8%
- Mean fold ROI ≥ 3%
- All folds positive
- Full-data ROI ≥ 3% AND full-data DD ≤ 10%
- Trade count per fold ≥ 5 (relaxed from 15)

Pass-viable = portfolio candidate. May stack with other systems in different regimes. Not deployed solo.

### Tier 2 lift evaluation

For archetypes with lift filters: WFO baseline vs WFO+lift. Ship lift if worst-fold ROI improves ≥ 1% without DD degradation. Else revert.

### Pipeline E vs D1 ship decision

For dual-assigned archetypes: WFO three configs (E alone, D1 alone, E+D1 unison). Ship highest worst-fold ROI subject to DD ≤ 8%.

**Outputs:** `wfo_results.csv`, `wfo_full_data.csv`, `wfo_synthesis.csv`.

---

## §11. Exit-family map

Centroid pattern → archetype label → exit policy. Match the cluster's centroid (its mean across the four features) to a row.

Cells in the "Centroid pattern" column are **rules** the centroid must satisfy — e.g., "monotonicity ≥ 0.55" means the cluster's monotonicity centroid value must be ≥ 0.55.

| Centroid pattern | Archetype label | Initial SL | Exit (Pipeline E) | Exit (Pipeline D1, at bar N) |
|---|---|---|---|---|
| monotonicity ≥ 0.55, local_peaks ≤ 4, time_to_peak_rel ≥ 0.50 | Monotone ascent | 1R | Wide trail (0.75-1.0R from peak) | SL to break-even, trail 1.0R |
| monotonicity ≥ 0.50, local_peaks 5-30, pullback ≤ 0.5R, time_to_peak_rel ≥ 0.50 | Stepwise climber | 1.3R | MFE-lock at 1R, trail 0.75R from new high | MFE-lock at 1R, trail 0.75R |
| time_to_peak_rel ≤ 0.30, fwd_mfe_p50 ≥ 1.5R, pct_peak_and_collapse < 0.30 | Early-peak hold | 0.8R | Wide trail (1.0R from peak) | Trail from peak observed at bar N |
| time_to_peak_rel ≤ 0.30, pct_peak_and_collapse ≥ 0.50 | Peak-and-collapse | 0.5R | Fixed TP at archetype p75 MFE | Tight TP at 0.8x peak observed |
| MAE early (mae before peak ≥ 5 bars), peak in [0.4, 0.8] of trade | V-shape recovery | 1.5R | Standard trail after recovery confirmed | After bar N confirms reversal, standard trail |
| local_peaks ≥ 8, monotonicity ≤ 0.30, pullback ≥ 1R | Random walk | 1R (entry-to-N hold only) | Filter out — never admitted | Close at break-even at bar N |
| Bimodal fwd_mfe distribution (two distinct modes ≥ 1R apart) | Split exit variant of stepwise/monotone | Per the base archetype | Half-off at TP1 (lower mode), trail remainder | Same |

### Early-peak time-to-peak — disambiguation rule

Two patterns share `time_to_peak_rel ≤ 0.30`. Differentiator is `pct_peak_and_collapse`:
- `pct_peak_and_collapse < 0.30` → Early-peak hold
- `pct_peak_and_collapse ≥ 0.50` → Peak-and-collapse
- `0.30 ≤ pct_peak_and_collapse < 0.50` → assign by empirical test on per-fold internal validation (which exit family produces higher capture ratio)

### Map application notes

- Centroid ranges are first-pass priors (Open-07). Empirical refinement during Arc 3: test candidate exit families per archetype.
- Boundary archetypes (centroid between rows): assign by empirical test on per-fold internal validation.
- Same-archetype clusters share an exit policy.
- Random walk archetype with Pipeline D1: the trade is still entered at signal (1R initial SL) — the "close at break-even" exit fires at bar N if classifier predicts random walk. Pipeline E never admits Random walk.

---

## §12. Discipline rules (operational detail)

### Filter stacking — full rule (§1.19 referenced)

**Tier 1 — clearing the gate:**

Step A: single RF classifier on full feature set. Clears 0.65 → done.
Step B: feature subset selection (top-N by importance, forward selection). Clears 0.65 → done.
Step C: stack 2 classifiers (intersection). Budget ≤ 30 total combinations across all archetypes per arc. 3+ classifiers needed → not extractable on E.

**Tier 2 — lifting above the gate:**

Once Tier 1 succeeds with a single classifier, stack additional classifiers freely as lift candidates. Validated at Step 6 by WFO worst-fold ROI improvement ≥ 1% without DD degradation. Baseline always retained as fallback.

All combinations evaluated must be logged in the live arc doc.

### Model and feature

- Logistic Regression: sklearn defaults, `max_iter=2000`, `random_state=42`, StandardScaler on features. Reported for informational purposes (§1.18).
- Random Forest: `n_estimators=200, max_depth=8, min_samples_leaf=20, random_state=42`. The gate.
- Features computed strictly from bars ≤ observation point (entry for E, bar N for D1).
- Cross-dataset 8-feature set mandatory. Arc-specific features from `feature_catalogue.yaml` (per-arc YAML defining available features for that arc).
- Total entry features ≤ 38 (8 base + 30 arc-specific cap).

### Pipeline D1 backtester (next engine PR)

Required capabilities:
- Per-bar trade-paths during simulation (v1.3 capability — already in place)
- Classifier loading at simulation start
- Classifier evaluation at bar offset N
- Branching exit logic on classifier output

Until extension lands: Pipeline D1 archetypes tested via post-hoc simulation in analysis scripts. Step 6 cannot complete for D1. This is the next planned engine PR.

### Cross-arc governance

Calibration changes (thresholds, feature sets, floors) require:
- Calibration result document
- Recalibration against KH-24 anchor
- Chat-level approval before commit

Within-arc thresholds locked at arc start.

---

## §13. Documentation and operational workflow

### Documentation: one live doc per arc

`ARC_<N>_LIVE.md` is the single working document. Starts when arc opens. Travels chat-to-chat by upload. Each chat amends in place. Lives outside the repo while in progress — no commit pressure. No per-step formal docs.

**Structure:**

```
# Arc <N> — <signal name>

## Status
Current step / verdict / last updated.

## Step results
| Step | Gate | Result | Notes |
|---|---|---|---|
| 1 | Plumbing | PASS | 612 trades across 28 pairs, deterministic verified |
| 2 | Clustering | PASS — K=4 chosen, silhouette 0.45 | K=5 also passed; K=4 preferred for parsimony |
| 3 | Capturability | PASS — 2 archetypes survive (Stepwise, Early-peak) | Monotone ascent killed on fwd_mfe_p50 1.2R |
| 4 | Extractability | ... | ... |

## Detailed analysis
(Only for decision-relevant points; 1-3 sections typical)

## Cross-arc candidates
(Filters / exits / features that worked here, candidates for future arcs)

## Interesting observations
(Worth investigating later — cherry on top, not core)
```

**End of arc:** rename to `ARC_<N>_RESULT.md`, commit to `docs/arc_results/`, update project knowledge. Cross-arc candidates fold into anchor docs (`CLAUDE.md`, candidate registry) at this point.

### Workflow: direct-to-main, PR for engine

**Direct-to-main:**
- Arc analysis scripts (`scripts/arc_<N>/...`)
- Arc results data (`results/arc_<N>/...`)
- Ad-hoc diagnostic scripts
- Calibration outputs
- Exploratory code

**PR required (with CI gating before merge):**
- Backtester core (`scripts/phase_kgl_v2_4h_wfo.py` and successors)
- Backtester extensions (including upcoming D1 conditional-exit support)
- Signal definitions (`signals/`)
- Locked configs (`configs/wfo_kh24.yaml`, `configs/spreads_5ers.yaml`, `configs/spread_floors_5ers.yaml`)
- CI workflow files (`.github/workflows/*.yml`) — changing CI logic without review could disable it entirely
- Protocol document (`docs/L_ARC_PROTOCOL.md`)

**CI on main pushes:** triggered by every push to main. Runs full test suite. Failures report as red X on commits but do NOT block subsequent pushes. Engine-path changes still go through PR + CI gating before merge (CI gate is on PR merge, not on direct main pushes). Manual review of red commits on main, fix at convenience.

### Chat ↔ CC ↔ User communication

**Chat → CC:** one `.md` file per dispatch. Prompt only — no scratch context, no commentary outside the file. Detail scaled to prompt complexity. No splits across multiple files.

**CC → Chat:**
- Output **files** (DIAGNOSTIC_REPORT.md, CSVs, etc.): retain analytical structure suitable for the artefact. Reports may be section-organised, distribution-detailed, whatever the analysis needs.
- End-of-prompt chat **summary**: result table first, evidence second, observations third. No process narration. Pass/fail before discussion.

**Chat → User (default response format):**

```
[Result table — optionally with comments column]

What this means:
- Metric A: <plain interpretation>.
- Metric B: <plain interpretation>.
- Overall: <good / bad / useable / inconclusive>.

Next steps:
<single clear next action, or short list if branching>

Notes:
<things to watch for, caveats, what could affect this>
```

For metrics needing explanation (AUC, RF feature importance, t-statistics, statistical significance): spell out per-metric meaning, then aggregate interpretation. Distributions and basic ratios usually don't need explanation.

For decisions: one-sentence recommendation, one-sentence reasoning. Menu of options only when the choice is non-obvious AND consequential.

No reflective preamble, no exhaustive confirmation lists, no "let me reflect that back."

---

## §14. Calibration anchor: KH-24

### Reference archetype: KH-24 K=4 archetype 3 (Stepwise climber)

| Property | Value |
|---|---|
| Size (fraction of pool) | 0.134 |
| Monotonicity centroid | 0.576 |
| Local peaks centroid | 14.19 |
| Pullback magnitude centroid | 0.020 |
| Time-to-peak relative centroid | 0.847 |
| fwd_mfe_h240 p50 | 5.40R |
| frac_reach_1R | 1.000 |
| frac_reach_2R | 0.96 |
| frac_wrong_way | 0.04 |
| final_r_mean | +2.37R |
| final_r_t_stat | +11.0 |
| Pipeline E entry RF AUC (24 features) | 0.642 |
| Pipeline E entry logistic AUC | 0.665 |
| Pipeline D1 t=3 RF AUC (chosen t per smallest-t rule) | 0.638 |
| Pipeline D1 t=3 trade exclusion | 15.4% |
| Pipeline D1 t=5 RF AUC (for reference) | 0.653 |
| Pipeline D1 t=5 trade exclusion | 32.0% (violates ≤ 30% constraint) |
| Shape tag | bimodal (right-mode dominates) |

### How KH-24 passes the v2.0 anchor rule

KH-24 was not built under v2.0. Its v1.0-era deployment used currency-cap and 1H close-in-range filters, designed before path-shape clustering existed. The reference numbers above are measured characteristics of KH-24's verbatim trade-paths re-analysed through v2.0 lenses.

**Pipeline E (entry filter) status:** KH-24 K=4 archetype 3 RF AUC is 0.642 — fails the v2.0 Pipeline E threshold of 0.65 by 0.008. Its logistic AUC is 0.665 (above 0.65) but the protocol gates on RF, so KH-24 fails the Pipeline E gate as drawn.

**Pipeline D1 (deferred policy) status:** Under §8's smallest-t selection rule, the protocol chooses t=3 for KH-24 (smallest t with AUC ≥ 0.60 AND exclusion ≤ 30%). At t=3, RF AUC is 0.638 — passes the v2.0 Pipeline D1 threshold of 0.60 with margin (0.038 over). At t=5 the AUC is higher (0.653) but exclusion is 32%, violating the constraint, so t=5 is rejected.

**Overall v2.0 extractability gate:** §2 is disjunctive — archetype passes if E OR D1 clears. KH-24 K=4 archetype 3 clears via D1 at t=3. Anchor rule satisfied.

This means KH-24 K=4 archetype 3 under v2.0 protocol would be classified as a Pipeline D1 archetype, not Pipeline E. v2.0 protocol applied to KH-24 would not reproduce its current v1.0 deployment — it would produce a different system (D1 instead of entry filter, archetype-specific stepwise-climber exit policy instead of fixed standard exits). Whether to migrate KH-24 from v1.0 deployment to a v2.0-derived deployment is a separate decision from protocol commit.

### Calibration rule

Any v2.0 gate or threshold change is invalid if it causes KH-24 K=4 archetype 3 to fail v2.0 extractability overall (i.e., fail both Pipeline E AND Pipeline D1). The current state (fails E, passes D1 at t=3) is acceptable; any threshold change that also breaks D1 is rejected.

### Promotion rule

If a future arc surfaces a stronger reference archetype, promotion requires a calibration document showing the new anchor is at least as conservative as KH-24 across all gates.

---

## §15. Sample size and operational requirements

| Stage | Requirement | Where checked |
|---|---|---|
| Step 1 plumbing | Pool size ≥ 500 trades | Step 1 gate |
| Step 2 clustering | ≥ 1 K with all clusters ≥ 30 | Step 2 gate |
| Step 3 capturability | Archetype size ≥ 50 (else flagged) | Step 3 reporting |
| Step 4 extractability | Predictability sample n_pos ≥ 50, n_total ≥ 200 | Step 4 reporting |
| Step 5 stability | Trade count per fold informational (reported, not gated) | Step 5 (not enforced) |
| Step 6 WFO | ≥ 15 trades per fold per design for pass-deployable; ≥ 5 trades per fold for pass-viable | Step 6 gate |

Arcs with insufficient pool size cannot reach pass-deployable but may pass-viable as portfolio candidates. The trade-count floor is enforced at Step 6 only.

---

## §16. Open questions and deferred decisions

### Open-01: Pre-peak / post-peak path feature split

Current features average over full held window. Smooth-then-collapse trades get averaged. Proposed: compute features twice (pre-peak, post-peak), cluster on 8 features.

**Status:** v2.1.

### Open-02: Pipeline D2 (delayed entry)

Wait N bars after signal, classify at N, enter only if classifier predicts target archetype.

**Status:** deferred. D1 is v2.0 baseline.

### Open-03: Re-entry pattern detection

Archetypes with "exit well below peak but peak was meaningful" → exit-and-reenter.

**Status:** post-deployment refinement. Cherry on top, not core.

### Open-04: External features for refractory archetypes

If a capturable archetype fails extractability on all in-protocol angles, external information (macro, session, cross-asset, calendar) may be the unlock.

**Status:** v2.1 commission if needed.

### Open-05: Multi-signal portfolio composition

v2.0 governs per-arc. Multi-signal portfolio is v2.2+.

### Open-06: AUC threshold calibration

0.65 / 0.60 are first-pass priors. Refinement after Arcs 3-5.

### Open-07: Exit-family map empirical calibration

§11 centroid ranges are priors. Empirical test during Arc 3 — high priority among Arc 3 side-tasks.

### Open-08: `pullback_magnitude_median` definition

Operational fix locked. Feature was 80%+ degenerate on KH-24. May need fundamental rethink. Lower priority than Open-07 for Arc 3 side-tasks.

### Open-09: Soft trade-off scoring vs hard floors

Hard floors may false-kill archetypes with weak capturability but strong extractability. Needs calibration data not yet available.

**Status:** revisit after Arcs 3-5.

### Open-10: Full-population vs per-fold clustering

Baseline clusters on full population (minor fold-leakage risk). Revisit if leakage shows measurable WFO degradation.

### Open-11: Long-term classifier reproducibility

joblib used for v2.0. ONNX/PMML for cross-version portability is a future concern as sklearn versions drift.

**Status:** not blocking. Address when reproducibility issues surface.

### Open-12: Silhouette tie tolerance for K selection

§6 K selection rule "ties: smaller K preferred" lacks tolerance definition. Arc 3 chose K=7 over K=4 on a 0.0021 silhouette margin (range across five Ks: 0.0165 — effectively noise).

**Status:** v2.1 calibration. Proposed: relative threshold (e.g., 0.01 absolute or 5% relative) below which smaller K is preferred for parsimony.

### Open-13: §2 shape_tag floor / §11 row-7 bimodal incompatibility

§11 row 7 ("Split exit variant — bimodal fwd_mfe distribution, two distinct modes ≥ 1R apart") defines bimodal as a valid archetype shape with its own exit policy (half-off at TP1 lower mode, trail remainder). §2's shape_tag floor `∈ {tight_unimodal, heavy_right_tail}` makes that archetype structurally unreachable. The protocol is internally inconsistent.

Evidence: Arc 3 Stepwise climber (clusters 2+4, 27.5% pool, n=707) — fails §2 only on shape_tag=bimodal and frac_wrong_way 0.383; passes monotonicity 0.559 / mfe_p50 3.34R / reach_1R 83.6% / size cleanly. Final_r distribution textbook split: p25 −1.00, p50 +1.85R, p75 +3.80R.

**Status:** v2.1 amendment. Highest priority among Arc 3 cross-arc candidates. Proposed: §2 shape_tag admits `bimodal` when accompanied by §11 row-7 routing AND modes meet ≥ 1R separation criterion (operationalised via Hartigan dip + mode-separation check).

### Open-14: Same-archetype aggregation can destroy capturable sub-clusters

§6 same-archetype aggregation rule states "same-archetype clusters... are aggregated for downstream Step 3-6 evaluation — they share an exit policy." The rule assumes shared §11 label implies shared forward geometry; the data refutes that.

Evidence: Arc 3 Early-peak hold (clusters 0+3 aggregated). Cluster 0 mono 0.008. Cluster 3 mono 0.579 (passes §2 floor alone). Aggregated mono 0.251 — aggregation killed cluster 3's individual capturability. Within Stepwise climber (clusters 2+4): cluster 2 local_peaks 24.42 vs cluster 4 7.79 (3× difference); cluster 2 pc 0.126 vs cluster 4 0.474 (3.7× difference). Aggregating these as "Stepwise climber" mixes structurally different sub-archetypes.

**Status:** v2.1 amendment. Proposed: same-archetype aggregation conditional on cluster-pair disparity ≤ X across each §2 floor; otherwise treat as separate sub-archetypes carrying the same §11 exit policy.

### Open-15: SL-distance / hold-horizon asymmetry inflates frac_wrong_way

When horizon ≫ SL-distance-in-volatility-units, false stop-outs structurally inflate the §2 `frac_wrong_way ≤ 0.30` floor. 2× ATR_1H on 120-bar horizon = ~0.18 σ of expected horizon price movement — even a directionally neutral random walk should stop out frequently.

Evidence: Arc 3 — all three full-evaluation archetypes failed on frac_wrong_way. Stepwise climber 38.3% (margin 8.3pp over floor). Cluster 1 73.2%. Early-peak hold 98.8%.

**Status:** v2.1 consideration. Proposed options:
- (a) Archetype-specific initial SLs (set at Step 4 with the rest of the artefact) derived from cluster MAE characteristics
- (b) Step 1's initial SL scaled to horizon: SL = 2.0 × ATR × √(h/24) — Brownian-motion-consistent default
- (c) Horizon-aware frac_wrong_way floor scaled to horizon/SL ratio

Path (b) cleanest at arc level. Path (a) is what §11 was meant to do at Step 4 but requires the protocol to reach Step 4.

---

## §17. Glossary

- **Cluster** — output of k-means clustering on path-shape features
- **Centroid** — mean of a cluster's trades across the four path-shape features
- **Centroid pattern** — a rule about centroid values (e.g., "monotonicity ≥ 0.55"), defined per archetype in §11
- **Archetype** — descriptive label assigned to a cluster from centroid pattern matching
- **Exit policy** — rules for exiting trades in an archetype; one per archetype, per §11
- **Filter** — a classifier + probability threshold that admits or rejects trades. Pipeline E filters admit at entry. Pipeline D1 doesn't use filters in this sense — D1 uses a classifier to branch exit policy at bar N.
- **Capturability** — clean path + meaningful magnitude (R to extract)
- **Extractability** — predictable archetype membership from features observable in time to act
- **Pipeline E** — entry-time filter
- **Pipeline D1** — deferred-identification exit policy (enter, classify at bar N, branch exit)
- **Pipeline D2** — delayed entry (Open-02, not in v2.0)
- **Path-shape features** — outcome-blind: monotonicity_ratio_in_profit, local_peaks_count, pullback_magnitude_median, time_to_peak_mfe_relative
- **Forward geometry** — distribution of MFE, MAE, final R within held window
- **MFE** — maximum favourable excursion (best point reached)
- **MAE** — maximum adverse excursion (worst point reached)
- **R-unit** — entry-to-SL distance. SL = 2× ATR, so 1R = 2 ATR.
- **AUC** — Area Under the ROC Curve. Measures how well a classifier ranks positives above negatives. 0.50 random; 0.55-0.60 marginal; 0.60-0.70 usable; 0.70+ strong.
- **Silhouette score** — cluster separability. Range −1 to +1. > 0.50 strong, 0.25-0.50 moderate, < 0.25 weak.
- **WFO** — walk-forward optimisation, 7-fold anchored expanding
- **Pass-deployable** — clears strong thresholds (worst-fold ROI ≥ 5%, mean ≥ 8%, worst DD ≤ 8%)
- **Pass-viable** — clears weak thresholds; portfolio candidate, may stack with other systems

---

## Document control

| Field | Value |
|---|---|
| Version | v2.0 |
| Supersedes | v1.0 + amendments (v1.1, v1.2, v1.3) |
| Evidence base | PR #129 (archetype diagnostic), PR #130 (predictability investigation) |
| Calibration anchor | KH-24 K=4 archetype 3 |
| Active for arcs | Arc 3+ |
