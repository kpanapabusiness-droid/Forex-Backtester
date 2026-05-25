# L_PROTOCOL v3.0 — The Overseer

> **Version:** 3.0
> **Status:** locked. Changes require explicit redesign event.
> **Purpose:** the methodology by which a single arc runs from open to verdict.
> **Scope:** how to run things. Not what to run next.
>
> **Amendment 1 (2026-05-22):** Step 5 search policy clarified — informed by Steps 3-4, not exhaustive. Multi-cluster handling + holdout decision rule specified. See §2 Step 5.
> **Amendment 2 (2026-05-22):** ML architecture mechanics specified for A2, A3, A4, A6. See §2 Step 5 ML mechanics subsection.
> **Amendment 3 (2026-05-22):** Risk-normalised gates. §3 constraints preserved 1:1; evaluation now occurs at scaled risk `r_safe` / `r_hard` rather than at the WFO base risk. Scalability bounds, per-day DD recount, and explicit evaluation order added. Full text archived at `archive/L_PROTOCOL_v3_0_AMENDMENT_3.md`. See §3.
> **Amendment 4 (2026-05-24):** Step 6 causal-audit framework. Six audit categories (lookahead, selection bias, execution realism, statistical integrity, determinism, deployment readiness) as runnable engine code. Auto-dispatches post-gate on Top-1 PASS-tier candidate; manual CLI invokable on any closure. Critical failure downgrades verdict via `step6_causal_audit_fail`. Closure template bumped to v1.3 with `§1 tracker_payload.step_6` block; parser v1.3 + Phase 2 tightening (PR-186-merge cutoff). Full text archived at `archive/L_PROTOCOL_v3_0_AMENDMENT_4.md`. See §2 Step 6.
> **Amendment 5 (2026-05-23):** AUC-gated A2/A6 architecture selection. Amendment 1's uniform archetype gating is split into four gates: Gate 1 preserves A3/A4 archetype gating; Gate 2 admits A2 + A6 whenever Step 4 mean OOS AUC ≥ 0.65 regardless of archetype; Gate 3 always admits A1; Gate 4 admits A5 when ≥2 candidate clusters survive Step 3. Choppy clusters skip all architectures. Enforcement is dispatch-time; engine unchanged (all six architectures wired post-PR-186). Closure template v1.3.1 adds optional `architectures_skipped_by_amendment_5` field; parser v1.3 accepts it and requires it for post-ratification PASS verdicts. Full text archived at `archive/L_PROTOCOL_v3_0_AMENDMENT_5.md`. See §2 Step 5 "Architecture selection (Amendment 5)".
> **Amendment 6 (2026-05-25):** Daily-DD measurement boundary changed from UTC broker-day to **5ers EET broker trading day** (Europe/Athens, EU DST rules). Amendment 3 §"Boundary" was authored under the pre-PR-189 UTC-bar engine assumption; under PR #189's 5ers EET aggregation the bar boundary and the daily-DD reset boundary must match for the gate to be coherent. Engine implementation landed via PR #197: `core.runners._fold_stats_helpers.compute_per_day_max_dd(boundary_convention="5ers_eet")` consuming `core.time_utils.session_boundary.utc_to_eet_trading_day`; `Panel.boundary_convention` carries the choice through orchestrator slicing. `boundary_convention="utc"` opt-in preserved for KH-24 anchor byte-identity. Convention-aware consumers also include `core/features/distance.py` (prior-session HL bucketing) and `core/sim/risk/reset_floor.py` (daily-floor ratchet). See §3 "Boundary" and PROTOCOL_RUNTIME §15.5. No archive file — Amendment 6 is documented inline at §3 only.
> **Amendment 5.1 (2026-05-25):** Gate 4 PASS-tier-constituent qualifier. Sub-amendment to Amendment 5 §Gate 4 only. Gate 4 admits A5 if and only if (a) ≥2 candidate clusters survive Step 3 (preserved from Amendment 5) AND (b) at least one constituent candidate cluster has cleared Step 5 search-WFO at PASS-DEPLOYABLE or PASS-VIABLE tier under an architecture admitted by Gates 1, 2, or 3. If (a) holds but (b) does not, A5 is not admitted; closure records `a5_gate_4_admission_blocked_by_no_pass_tier_constituent` in `architectures_skipped_by_amendment_5`. Rationale: aligns Gate 4 with the standing engine-deferral policy (A5 build deferred until composable PASS-tier candidates exist). Enforcement dispatch-time; engine unchanged; closure template unchanged. Parser v1.3 accepts the new reason string; post-cutoff PASS closures with ≥2 candidate clusters and no PASS-tier constituent MUST cite this string. Cutoff `AMENDMENT_5_1_CUTOFF_ISO` (placeholder `2026-05-25T00:00:00Z`, backfilled to this PR's merge timestamp). No archive file — documented inline at §2 Step 5 Gate 4 only.
>
> This protocol is the umbrella. It accepts any signal, any feature space, any architecture. Sub-protocols may layer on top to add signal-class-specific specificity. The overseer's gates and verdicts apply universally.

---

## §0 Project goal

20% annualised gross ROI on a $100k 5ers funded account, maintained on average over time. First strategy clearing PASS-DEPLOYABLE ships and the primary goal is achieved.

### What an "arc" is under this protocol

An arc is any unit of research that runs through the protocol from open to closure. This includes:

- **Standard arcs** — a signal is defined in the arc-open, the protocol runs Steps 1-5 evaluating it.
- **Discovery arcs** — no signal exists at open; a sub-protocol overrides Step 1 to search for a signal, then Steps 2-5 evaluate what was found.
- **Diagnostic probes** — a specific question is being tested (e.g. "does feature class X help"). Sub-protocol customises which steps run and how. Same closure format, same tracker integration.

All three are arcs. All three use the same closure format and feed the same tracker. Sub-protocols handle the differences.

---

## §1 Non-negotiables

These cannot be violated, ever, regardless of arc, sub-protocol, or chat instruction:

- **No lookahead.** Every feature is computable from data closed strictly before the entry bar's open.
- **Ex-ante population construction.** Population is built before any outcome data is observed. No outcome-aware filtering anywhere in pool construction.
- **D1 one-bar lag rule.** `iClose(D1, 1)` semantics — same-day D1 close is NOT available intraday.
- **Real bid/ask spreads.** HistData M1 bid+ask is the canonical spread source. No fallback mechanism — zero-spread bars are a data quality flag, not silently backfilled.
- **Determinism.** Every result file is reproducible from seed. sha256 manifests on every artefact. `lineterminator='\n'` for cross-platform reproducibility.
- **Config-driven.** All parameters via YAML. No hardcoding.
- **Anchor preservation.** Any cross-arc evaluation framework must reproduce KH-24's documented worst-fold numbers within tolerance (±0.5pp ROI, ±1pp DD). Framework failing reproduction = framework has a methodology bug.
- **Permanently eliminated items** in `CLAUDE.md` never return under any framing.

---

## §2 The five steps + lazy Step 6

The protocol is **gates-as-rankings**. Each step computes its metrics but does not block the arc. The arc runs to completion regardless of intermediate step quality. Step 5 (WFO) is the only deployment gate. Step 6 (causal audit) is lazy — runs only when a Step 5 candidate clears PASS-DEPLOYABLE or PASS-VIABLE.

### Step 1 — Plumbing

**Goal:** build the trade pool and feature matrix.

**Signal source:** vanilla arcs declare the signal at arc-open and Step 1 applies it. Discovery arcs (e.g. signal_discovery_probe sub-protocol) have Step 1 generate the signal via search/discovery procedure; the rest of Step 1 then runs on the discovered signal. Either path produces the same Step 1 output.

**Mechanics:**

1. Apply the arc's signal definition (declared at open OR discovered by sub-protocol) to the in-scope pair set over the arc's window. Trigger times are timestamps of qualifying bar closes. Use ex-ante population construction (`build_ex_ante_bounded_population` or equivalent).
2. For each trigger, simulate the trade at next-bar open. Capture: trade entry timestamp, entry price, SL (per arc's SL rule), TP / trailing / time-exit per arc's exit rule.
3. Compute the feature matrix at signal-bar close. Default feature space includes (subject to per-arc extension):
   - Price geometry: ATR, swing distances, kijun distances, range/close ratios
   - Time: session bucket, hour-of-day, day-of-week
   - Cross-pair: signal density across 28 pairs, dollar-bloc state, currency strength ranks
   - Multi-TF: D1 directional state, D1 ATR percentile, W1 state
   - Volatility regime: current ATR vs trailing
   - Cross-asset (if available): DXY state, US10Y state
   - Distance from round numbers, prior session HL
   - Spread regime (current vs trailing)

   Per-arc signal-specific features are added. All features are tagged with a provisional causal lineage hint.

4. Run integrity checks:
   - Pool size, per-pair n distribution
   - Coverage window, gap report
   - Spread-floor activation rate per pair (informational — high rates suggest data issues)
   - D1-lag NaN-perturbation test (3 random trades)
   - Lookahead spot-check (5 random trades, manually verified by causal lineage trace)
   - KH-24 co-fire rate (informational only — tells whether the signal overlaps the deployed anchor)
   - Determinism: sha256 of pool table reproduces across two runs from same seed

**Output:**
- `step_1/pool.parquet` — trade pool with full feature matrix
- `step_1/integrity_report.md` — all checks with pass/fail/informational status
- `step_1/manifest.json` — sha256 per artefact

**Failure diagnostics (if pool unusable):** surface specifically what failed — pool too small (specify n vs needed), gaps in which pair-months, spread weirdness on which pairs, lookahead detected in which feature. An arc with an unusable pool continues to Step 2 with whatever subset of the pool is usable — never silently drops.

### Step 2 — Clustering

**Goal:** characterise archetypes in forward-path geometry.

**Mechanics:**

1. Compute forward-path geometry per trade: mono (monotonicity), peaks (count), ttp_rel (time-to-peak relative to bars_held), drawdown depth, recovery shape.
2. Cluster on these path features for K in {2, 3, 4, 5, 6}. Use KMeans or HDBSCAN; report silhouette score per K.
3. Pick K with highest silhouette as primary; report all K for diagnostic.
4. Assign each trade a shape-tag label (V-shape recovery, Stepwise climber, Bimodal, Monotonic up, Monotonic down, Choppy, etc. — labels defined by quartile rules on path features).
5. Per cluster, report outcome distribution (mean R, percentile Rs, bars_held, MFE, MAE).

**Output:**
- `step_2/cluster_assignments.parquet` — per-trade cluster ID + shape-tag
- `step_2/cluster_summary.md` — silhouette per K, per-cluster outcome distributions
- `step_2/manifest.json`

**Failure diagnostics:** if silhouette is low (< 0.30) across all K, surface the path-shape variance distribution — is the issue homogeneous trades (all the same), or noisy features (clusters exist but feature scaling broken)? Continue to Step 3 with single-cluster assignment (entire pool as one cluster).

### Step 3 — Capturability

**Goal:** measure per-cluster edge potential.

**Mechanics:**

1. Per cluster, compute:
   - Reach rates: P(MFE ≥ 1R), P(MFE ≥ 2R), P(MFE ≥ 3R)
   - MFE distribution: p25, p50, p75, p90
   - Wrong-way path prevalence (`ww_pp`): fraction of trades that hit −1R MAE before +1R MFE
   - Time-to-peak distribution
   - Per-cluster mean R, p25 R, p50 R
2. Compute the capturability composite per cluster — weighted aggregate of (reach_1R, mfe_p50, 1 − ww_pp). This is a ranking aid only, never a gate.
3. Rank clusters by composite. Flag any cluster with reach_1R ≥ 0.50 AND ww_pp ≤ 0.30 AND mfe_p50 ≥ 1.5R as a "candidate cluster" — these are the cohorts most likely to support extraction.

**Output:**
- `step_3/capturability.csv` — per-cluster metrics
- `step_3/capturability_summary.md` — ranking + candidate cluster flags
- `step_3/manifest.json`

**Failure diagnostics:** if no cluster shows capturability potential, surface what's missing — all clusters have ww_pp > 0.50 (signal fails before it works), or all clusters have mfe_p50 < 1R (signal is anaemic)? Continue to Step 4 with the highest-composite cluster regardless.

### Step 4 — Extraction

**Goal:** test whether entry-time features can predict cluster membership.

**Mechanics:**

1. For each candidate cluster (target = cluster membership, binary):
2. Train three classifiers by default: RandomForest, LightGBM, Logistic Regression. Hyperparameters fixed at sensible defaults (see Appendix A). No per-arc tuning.
3. 5-fold TimeSeriesSplit. Per-fold AUC.
4. Threshold sweep: at AUC-best and F1-best thresholds, report precision, recall, trade count.
5. Permutation importance per feature; rank top contributors.
6. Optionally test rule-based filters (specific feature thresholds). Compare AUC against classifier baseline.
7. Tag each feature with provisional causal lineage (full audit deferred to Step 6).

**Output:**
- `step_4/extraction_metrics.csv` — per-cluster, per-pipeline AUC + threshold sweep
- `step_4/feature_importance.csv` — per-cluster permutation importance
- `step_4/extraction_summary.md` — diagnostic narrative
- `step_4/manifest.json`
- `step_4/classifiers/<cluster_id>.pkl` — joblib pickle of the
  best-AUC classifier per candidate cluster, refit on the full
  lineage-filtered pool after CV selection. Consumed by A2 / A6 at
  Step 5 without retraining (see Step 5 "Architecture-specific
  retraining policy" below).
- `step_4/classifiers/manifest.json` — SHA256 + provenance
  (joblib / sklearn / lightgbm versions, `auc_in_sample`,
  `auc_oos_cv5`, `trained_on_pool_size`, feature order).

**Failure diagnostics:** if all AUC near 0.50, surface top features by importance anyway — tells us where the model thought signal was. Surface whether any individual feature class shows signal even if combined doesn't. Continue to Step 5 with whatever filter candidates the extraction produced (including "no filter" as a valid candidate).

### Step 5 — WFO architecture search

**Goal:** build and test multiple system architectures; rank by ROI/DD ratio.

**Mechanics:**

1. **Architectures tested by default (each is a candidate system):**
   - **A1. System-level filter** — base signal + filter rules from Step 4 + exposure cap + searched SL/exit policy. No classifier.
   - **A2. Classifier filter** — base signal + classifier filter from Step 4 at AUC-best threshold + exposure cap + searched SL/exit policy.
   - **A3. Pipeline DE (deferred entry)** — wait N bars after signal, evaluate path-so-far features, then maybe enter. N searched over {1, 3, 5, 8}.
   - **A4. Pipeline D (differentiated exits)** — keep entry, vary exit policy by early-path shape (per Step 2 cluster on path-so-far).
   - **A5. Portfolio composition** — combine multiple candidate clusters at admit-only economics into a portfolio (when ≥ 2 clusters have capturability potential).
   - **A6. Meta-labeling** — classifier doesn't filter or generate signals; it sizes them. Per-signal output ∈ {0x, 0.5x, 1.0x risk}. Trained on classifier confidence vs realised outcome.

2. For TF-flexible arcs, TF is an axis. Each architecture × TF combination is a candidate.

**Search policy (informed by Steps 3-4, not exhaustive):**

Step 5 search is constrained by what earlier steps surfaced. Exhaustive Cartesian search of the parameter space is explicitly rejected — too much noise exploration, too much selection bias, ignores the methodology's intent.

Inputs from Step 3 (per surviving cluster):
- Selected SL multiplier (capturability composite optimum)
- Archetype label (V-shape, Stepwise, Bimodal, etc.)
- Capturability composite + per-metric percentile distribution

Inputs from Step 4 (per candidate cluster):
- Filter / classifier candidate (rule or model)
- Top features by importance
- AUC and threshold-sweep results

Step 5 search rules:

**Architecture selection (Amendment 5, ratified 2026-05-23):** the set of architectures evaluated per surviving cluster is the UNION over four independent gates. Replaces the Amendment 1 archetype-driven rule. Full text at [archive/L_PROTOCOL_v3_0_AMENDMENT_5.md](archive/L_PROTOCOL_v3_0_AMENDMENT_5.md).

- **Gate 1 — Shape-required (archetype-driven):** adds A3 / A4 per archetype.
  - V-shape recovery → A3
  - Stepwise climber → A4
  - Bimodal → A4
  - Monotonic up / Monotonic down / Unclassified → (none)
- **Gate 2 — Classifier-driven (AUC-driven):** if Step 4 mean OOS AUC ≥ 0.65 for the cluster → add A2 and A6. Fires REGARDLESS of archetype (A2 and A6 are archetype-agnostic by construction; the only relevant input quality is classifier predictive power).
- **Gate 3 — Universal:** always add A1 (system-level filter). No shape or classifier dependence; baseline for every cluster.
- **Gate 4 — Portfolio (revised by Amendment 5.1, 2026-05-25):** A5 (portfolio composition) is admitted if and only if (a) ≥2 candidate clusters survive Step 3 within the arc, AND (b) at least one constituent candidate cluster has cleared Step 5 search-WFO at PASS-DEPLOYABLE or PASS-VIABLE tier under an architecture admitted by Gates 1, 2, or 3. If (a) holds but (b) does not, A5 is not admitted; closure records `a5_gate_4_admission_blocked_by_no_pass_tier_constituent` in `architectures_skipped_by_amendment_5`. If (b) becomes true mid-arc (a Gate 1/2/3 result lands PASS-tier post-Step-5), A5 admission may be added as a closure addendum without re-running the rest of the arc.

**Cluster skip condition:** Choppy clusters skip all architectures. Step 5 evaluation not performed for Choppy.

If Step 4 produced NO usable filter/classifier (AUC at chance, no rule above noise), A1 still runs with "no filter" baseline (raw signal + system-level rules: exposure, SL, exit) — this tests whether the system works without filtering.

The four-gate rule is enforced AT DISPATCH TIME, not engine time. Arc dispatches must list the architectures-tested set per cluster and cite the admitting gate (e.g. `A1: Gate 3; A3: Gate 1 V-shape; A2, A6: Gate 2 AUC=0.72`). Engine receives a fully-resolved set; engine does NOT apply the selection rule itself. Architectures admissible under the prior Amendment 1 rule but skipped under Amendment 5 are recorded in the closure's `architectures_skipped_by_amendment_5` field (optional pre-cutoff, required for post-cutoff PASS closures — see closure template v1.3.1). Gate 4 admission has a sequential dependency under Amendment 5.1: condition (b) is evaluated post-Step-5 search-WFO, so A5 admission decisions are necessarily later than Gates 1–3 admission decisions. Dispatchers schedule A5 conditionally: tentatively admitted at arc-open if condition (a) is projected to hold, finalised after Step 5 search-WFO produces per-cluster verdicts.

**SL multiplier:** centred on Step 3's per-cluster selected SL, ±1 step either side (typically 3 values total). Example: if Step 3 selected SL=2.5×ATR, Step 5 tests SL ∈ {2.0, 2.5, 3.0}.

**Exit policy:** 3-4 variants matched to archetype:
- Stepwise climber → variants of `sl_plus_trailing_atr`, `sl_plus_trailing_swing`
- V-shape recovery → variants of `sl_plus_tp_2r`, `sl_plus_tp_3r`, `sl_partial_close_1r_runner_trail`
- Bimodal → `sl_partial_close_1r_runner_trail` plus `sl_plus_tp_2r`
- Always include `sl_only` baseline for comparison

**Exposure cap:** 2 sensible values per arc (typically: `max_concurrent_per_currency=2` matching KH-24, plus `unlimited` for comparison).

**Pipeline DE bar count (A3 only):** {3, 5} as default test points.

Per-arc config count: roughly 50-100 (3 architectures × 3 exits × 3 SLs × 2 exposure ≈ 54; doubles to ~108 if 2 candidate clusters survive Step 3).

Selection bias accounting:
- Total N (configs evaluated per arc) reported in closure doc
- Bonferroni-equivalent stress: top survivor's worst-fold ratio must materially exceed the noise floor implied by N
- Specifically: arcs reporting fewer than 50 evaluated configs receive a "thin search" flag; arcs reporting 100+ configs receive a "broad search" flag with elevated selection-bias risk

**Multi-cluster handling:**

If Step 3 surfaces 2+ candidate clusters (each passing §3 capturability checks):
- Step 5 runs per-cluster independently (full architecture search per cluster)
- A5 (portfolio composition) additionally runs combining the clusters at admit-only economics
- Top-3 across ALL clusters proceed to holdout

**Holdout decision rule:**

After 11-fold WFO ranking by worst-fold ratio:
- Top-3 candidates evaluated once on 2021-2025 holdout
- Verdict per candidate: PASS-DEPLOYABLE / PASS-VIABLE / FAIL (per §3)
- Arc verdict: best candidate's holdout result, conditional on the same candidate passing the 11-fold WFO gate
- If top-1 fails holdout but top-2 or top-3 passes BOTH 11-fold AND holdout: report all three results; top-survivor becomes the arc's verdict candidate
- If all top-3 fail either WFO or holdout: arc verdict = FAIL with explanatory section per §6 closure format

The holdout is one-shot per candidate. Re-evaluating the same candidate after parameter tweaks is forbidden (selection bias laundering).

**ML architecture mechanics (Amendment 2):**

A1 (system_level_filter) and A5 (portfolio_composition) are rule-based — no ML mechanics needed. The four ML-based architectures require explicit mechanics to ensure cross-arc reproducibility.

**A2 (classifier_filter):**
- Classifier: Step 4's best-AUC classifier (RF, LGBM, or Logistic — whichever scored highest in Step 4 evaluation)
- Threshold: AUC-best threshold from Step 4 threshold sweep
- No retraining at Step 5 (use Step 4 output directly)
- At each new signal: classifier predicts probability of belonging to candidate cluster; take trade if prob ≥ threshold, else skip
- Output recorded: per-signal admit/reject decisions

**A3 (Pipeline DE — deferred entry):**
- Requires NEW classifier training (Step 4's entry-time classifier is not reusable)
- Training data: Step 1 pool
- Target: same as Step 4 (cluster membership)
- Features: "path-so-far" features observable at bar N (entry deferred to bar N):
  - Bar-relative MFE (high - entry) / ATR
  - Bar-relative MAE (entry - low) / ATR
  - Current close vs entry price (R-multiple)
  - Bar count since signal
  - Velocity: (close - signal_close) / N bars / ATR
  - Bid-ask drift since signal
  - Cumulative volume (if available)
- Classifier: RandomForest with default hyperparameters from L_PROTOCOL Appendix A
- N values tested: {3, 5} bars after signal
- Threshold: AUC-best from training threshold sweep per fold
- Decision: at bar N post-signal, classifier predicts; if above threshold, enter at next bar open; else cancel

**A4 (Pipeline D exits — differentiated exit timing):**
- Requires NEW classifier training (different target from A3)
- Training data: Step 1 pool
- Target: "will this trade close profitably?" — binary, 1 if final R > 0, else 0
- Features: same path-so-far feature set as A3, evaluated at each bar post-entry up to time-exit horizon
- Classifier: RandomForest with default hyperparameters
- Decision rule: at each bar post-entry, classifier predicts confidence; if confidence drops below exit threshold, queue exit at next bar open
- Initial SL still applies (classifier exit does NOT override SL)
- Exit threshold: tested at {0.3, 0.4, 0.5} (configurable; higher threshold = exit sooner on uncertainty)

**A6 (meta_labeling):**
- Classifier: Step 4's best-AUC classifier (same as A2)
- Mapping from confidence to position size:
  - prob < lower_threshold → 0x (skip trade)
  - lower_threshold ≤ prob < upper_threshold → 0.5x risk
  - prob ≥ upper_threshold → 1.0x risk
- Default thresholds: lower=0.4, upper=0.6
- Threshold sweep at Step 5: test {(0.3, 0.5), (0.4, 0.6), (0.5, 0.7)} as three variant configs
- Output recorded: per-signal size decisions

**Architecture-specific retraining policy (locked 2026-05-23):**

| Architecture | Retraining at Step 5 | Source of fit |
|---|---|---|
| A1 | N/A — no classifier | rule-based filter only |
| A2 | **No retrain** | Step 4's persisted best-AUC classifier (`step_4/classifiers/<cluster_id>.pkl`), loaded via `core.steps.classifier_persistence.build_a2_config_from_step4` |
| A3 | **Per-fold retrain** | new path-so-far classifier, trained on each WFO fold's IS window |
| A4 | **Per-fold retrain** | new path-so-far classifier with different target ("will trade close profitably?"), per fold |
| A5 | N/A — no classifier | portfolio composition of upstream constituents |
| A6 | **No retrain** | same persisted classifier as A2 (Step 4 best-AUC), loaded via `build_a6_config_from_step4` |

A2 and A6 use the single Step-4-fit classifier across every WFO fold. A3 and A4 fit a fresh classifier on the IS window of each fold. The "global model" caution in the next bullet applies to A3 / A4 only — A2 / A6 are explicitly exempt per Amendment 2.

**A4 same-bar exit precedence (locked 2026-05-23):** when both an A4 classifier-exit predicate AND a trailing-stop trigger fire on the same bar close, **the trailing-stop wins** (`exit_reason = "trailing_stop"`). Intra-bar SL/TP exits remain highest priority (they fire BEFORE either bar-close hook). Matches typical real-world execution where stop-side triggers fire before manual classifier-driven closes on a fast move. Implementation: `core/sim/multipair_backtester.py:_process_bar` step 3 uses direct assignment to `_pending_closes` (not `setdefault`); regression test at `tests/protocol_runtime/test_multipair_backtester_precedence.py`.

**Shared discipline across all ML architectures:**
- All classifiers respect causal lineage tags from Step 1 — no "suspect" or "unverified" features enter training
- Deterministic training: `random_state=42`, `n_jobs=1` per Appendix A
- For A3 / A4: per-fold WFO trains classifier on IS, evaluates on OOS — no global model trained once and used across folds. A2 / A6 are exempt per the Architecture-specific retraining policy table above.
- Selection bias: each unique architecture × parameter set counts toward Step 5's total N

3. **WFO structure (locked):**
   - **Training/search window:** 2010-01-01 → 2020-12-31, 11-fold WFO with 1-year folds
   - **Holdout window:** 2021-01-01 → present (currently ~2026-05) — ONE-SHOT evaluation only
   - Step 5 search runs on the 11-fold WFO. Top-K candidates (default K=3, by worst-fold ratio) are locked, then evaluated ONCE on the 2021-2025 holdout (per item 6 and the Amendment 1 holdout decision rule)
   - Both 11-fold WFO and holdout must pass §3 gates for PASS-DEPLOYABLE

4. For each candidate config:
   - Apply config: filter rules, exposure cap, SL multiplier (Appendix B), exit policy (Appendix B)
   - Run 11-fold WFO 2010-2020 on the FULL POOL (no admit-only metrics)
   - Compute per-fold: trade count, mean R, percentile Rs, ROI annualised, max DD, daily DD breaches, Sharpe, Sortino, ROI/DD ratio, sign-consistency
   - Aggregate: worst-fold ROI, worst-fold DD, worst-fold ratio, mean-fold metrics, std-across-folds

5. **Oracle WFO (standard for every arc):** for each candidate cluster, run WFO using true-label cluster membership (no classifier). Establishes the upper bound on what's possible if entry filtering were free. Reported alongside real WFO for each architecture. Oracle WFO uses the same 11-fold 2010-2020 window.

6. Rank candidates by worst-fold ratio across the 11-fold WFO. Top candidate proceeds to holdout. Other top-K candidates (default K=3) also evaluated on holdout for selection-bias accounting. Apply gate (§3) using BOTH WFO and holdout results.

**Output:**
- `step_5/wfo_results.csv` — per-architecture, per-fold full metrics
- `step_5/wfo_oracle.csv` — oracle WFO per cluster
- `step_5/architectures_ranked.md` — ranked table with verdict per candidate
- `step_5/best_candidate.md` — full detail on the top candidate
- `step_5/manifest.json`

**Verdict assignment:** per §3.

### Step 6 — Causal audit framework (Amendment 4)

> Amendment 4 (2026-05-24) replaces the sparse pre-Amendment Step 6 spec with a six-category runnable framework. Full text at [archive/L_PROTOCOL_v3_0_AMENDMENT_4.md](archive/L_PROTOCOL_v3_0_AMENDMENT_4.md). Summary below; refer to the amendment for severity rules, manifest schema, and Phase 2 parser-tightening details.

**Trigger:** post-gate. Step 6 dispatches AFTER §3 constraints #1-9 clear on at least one top-K candidate (per §3 "Evaluation order"). Auto-dispatch runs on the **Top-1 verdict-carrying candidate only**; feature-set divergence vs Top-2/Top-3 surfaces as a `top_k_feature_set_divergence` warning. Manual CLI (`scripts/run_step_6.py`) invokable on any closure regardless of verdict (read-only).

**Six categories** (each independent, no cross-category dependencies):

1. **§6.1 Lookahead** — per-feature producer trace + D1 lag rule + cluster-feature audit + byte-compare from raw OHLC + threshold-selection lineage.
2. **§6.2 Selection bias** — verifies Step 5's recorded configs_evaluated + Bonferroni-equivalent noise floor + holdout-reuse detector + cluster-selection record.
3. **§6.3 Execution realism** — HistData M1 bid+ask source present + spread regime delta + fill realism + lot rounding at `r_safe` + mid-price refactor active + UTC bar boundary.
4. **§6.4 Statistical integrity** — Lo-corrected Sharpe sample-size + 28-pair survivorship + vol-regime coverage (≥3 years) + cross-pair daily-bucket correlation.
5. **§6.5 Determinism** — `step_4/classifiers/manifest.json` sha256 verify + required artefacts present + seed pinning + LF line endings. (Sha256-verify only; does NOT re-run sims — CI provides the two-run guarantee.)
6. **§6.6 Deployment readiness** — `## §4 deployment_spec` heading + `config_artefact_path` resolvable + all §4.X subsections present + features live-computable + checklist marked.

**Severity rules:**
- `critical` failure → category FAIL → Step 6 FAIL → verdict downgrade
- `warning` failure → category PASS but flagged; counted in `n_warnings`
- `info` failure → recorded; no effect on category outcome

`CategoryAuditResult.passed = AND over critical-severity checks only`.

**Output:** `results/<arc>/step_6/` per auto-dispatch — `manifest.json`, `summary.md`, six `<category>_report.md`, `sha256_manifest.json`. Manual CLI writes to `results/<arc>/step_6_manual_<timestamp>/` (never clobbers auto).

**Verdict effect:** when auto-dispatched Step 6 produces a critical failure, the Top-1's amended gate is re-classified with `causal_audit_clean=False` → `primary_failure_mode = step6_causal_audit_fail` (priority 11 per §3). Manual invocations never modify the verdict.

**Backwards compatibility:** v1.0/v1.1/v1.2/v1.2.1 closures grandfathered. Arc 10's existing hand-written Step 6 stays canonical. Wave 2 arcs close at v1.3; Step 6 auto-dispatches.

---

## §3 Gates

> **Amended 2026-05-22 by Amendment 3 (Risk-Normalised Gates).** Constraint thresholds preserved 1:1 with the prior protocol. Evaluation now happens at a per-arc derived `r_safe` (DEPLOYABLE) or `r_hard` (VIABLE), not at the WFO base risk. Full amendment archived at `archive/L_PROTOCOL_v3_0_AMENDMENT_3.md`.

### Risk normalisation (scaling rule)

WFO runs at base risk `r_base` (typically 0.5%). Engine emits required metrics at `r_base`. Gate evaluation happens at scaled risk.

```
k_safe = 8.0 / worst_fold_dd_base
k_hard = 10.0 / worst_fold_dd_base
r_safe = r_base × k_safe
r_hard = r_base × k_hard
```

Linear scaling applies to: worst-fold ROI, chained max DD, holdout ROI/DD, per-day max-DD (then re-counted for daily breaches).

Linear scaling does **NOT** apply to: daily breach **counts** themselves — step function in the underlying daily P&L distribution. See §"Daily DD measurement".

Constraints not affected by risk scaling (sign-consistency, trade count, fold count) remain unchanged in evaluation procedure.

### Scalability bounds (locked)

- **Floor:** `r_safe ≥ r_min = 0.15%`. Locked value.
- **Ceiling:** `r_safe ≤ r_max = 2.0%`. Locked value.
- **Edge case:** `worst_fold_dd_base = 0%` → `k = ∞` → FAIL `step5_not_scalable`.
- **Sizing convention:** linear DD scaling holds ONLY under reset-floor sizing (L arc convention). Arcs using %-of-current-equity sizing FAIL the scalability check by default unless chat approves a separate scaling treatment.

Same bounds apply to `r_hard` for VIABLE evaluation.

### PASS-DEPLOYABLE (ship)

All of:

1. **Scalable to safe:** `r_safe ∈ [0.15%, 2.0%]` (added by Amendment 3).
2. **Worst-fold ROI/DD ratio at `r_safe`:** ≥ 2.0 (ratio invariant under linear scaling).
3. **Worst-fold ROI at `r_safe`:** > 0 (linearly scaled from `r_base`).
4. **Per-fold positivity:** positive at all 11 IS folds, 0 negative folds (sign does not scale; holdout has its own gate).
5. **Worst-fold DD at `r_safe`:** ≤ 8% (by construction at `k_safe`).
6. **Daily DD breaches at `r_safe`:** exactly 0 across all IS folds + holdout (see §"Daily DD measurement").
7. **Chained max DD at `r_safe`:** ≤ 10% (linearly scaled).
8. **Trades per fold:** ≥ 25 per fold (trade count does not scale).
9. **Holdout at `r_safe`:** re-run holdout simulation at `r_safe`; result clears the prior §3 holdout gate.
10. **Step 6 causal audit clean** (evaluated last per §"Evaluation order").

> The pre-amendment §3 had "positive at all 11 folds" and "0 negative folds" as separate bullets. Merged into constraint #4 here. Same meaning, no threshold change.

### PASS-VIABLE (portfolio candidate, does NOT ship alone)

All of:

1. **Hard-scalable:** `r_hard ∈ [0.15%, 2.0%]` (added by Amendment 3).
2. **Worst-fold ROI/DD ratio at `r_hard`:** ≥ 2.0 (ratio invariant).
3. **Mean-fold ROI/DD ratio at `r_hard`:** ≥ 2.5 (ratio invariant).
4. **Per-fold positivity:** up to 1 negative fold permitted across the 11 IS folds.
5. **Worst-fold DD at `r_hard`:** ≤ 10% (by construction at `k_hard`; matches the 5ers hard limit).
6. **Daily DD breaches at `r_hard`:** exactly 0.
7. **Chained max DD at `r_hard`:** ≤ 10% (linearly scaled).
8. **Trades per fold:** ≥ 25 per fold.
9. **Holdout at `r_hard`:** re-run at `r_hard`; clears the prior §3 holdout gate.
10. **Step 6 causal audit clean.**

A PASS-VIABLE strategy becomes deployable when combined with another PASS-VIABLE / PASS-DEPLOYABLE into an A5 portfolio whose combined account performance passes PASS-DEPLOYABLE gates. Combined-portfolio DD spec when components are at `r_hard` is open — see §"A5 follow-up flag".

### FAIL

Everything else. The arc's closure doc explains why and what would help.

### Daily DD measurement

Linear scaling of breach **counts** is mathematically wrong. Correct procedure:

**Engine emits at `r_base`:** for each trading day in the IS + holdout trajectory, the maximum intraday drawdown percentage of that day's starting equity. Full per-day series persisted to a separate parquet artefact (NOT only summary stats).

**Day-start equity definition:** account equity at 00:00 broker-day (the equity at the start of that calendar trading day in broker timezone), NOT the reset-floor sizing baseline. Under reset-floor sizing the two values are distinct: reset-floor is the per-trade sizing reference; day-start equity is the daily-DD reference. Engine must use day-start equity for daily DD measurement.

**Engine artefact:** `step_5/per_day_max_dd_base.parquet`. Columns: `date`, `pair_set`, `day_max_dd_base_pct`, `n_trades_open_start_of_day`.

**At gate evaluation:** for each day in the series, compute `day_max_dd_scaled = day_max_dd_base × k`. Count days where `day_max_dd_scaled ≥ 5%`. This is per-day re-evaluation, not count scaling.

**Boundary:** EET broker trading day (Europe/Athens, EU DST rules). Locked value, version-amended per **Amendment 6** (supersedes prior `UTC broker-day. Locked value.` framing). Amendment 6 effective from PR #197 merge (2026-05-25); Amendment 6 is documented inline here — no separate archive file. Engine implementation: `core.runners._fold_stats_helpers.compute_per_day_max_dd(boundary_convention="5ers_eet")` consuming `core.time_utils.session_boundary.utc_to_eet_trading_day`. `boundary_convention="utc"` opt-in preserved for KH-24 anchor byte-identity.

**Tolerance:** exactly 0 breaches in both tiers. No safety margin on the daily limit.

### Chained max DD measurement

Computed across the full IS + holdout trajectory: folds concatenated chronologically into one continuous equity curve, peak-to-trough across the whole curve.

Scales linearly with `k`. Threshold preserved from the prior §3 chained-DD spec — only the risk at which evaluated changes.

### Evaluation order

Amendment 3 specifies order explicitly; Amendment 4 makes the Step 6 dispatch concrete:

1. Constraints #1-#9 (all non-Step-6 constraints, both tiers) evaluated in priority order with `causal_audit_clean=True` (default).
2. If ALL clear for at least one top-K candidate → Step 6 framework auto-dispatches on the **Top-1 verdict-carrying candidate** (per Amendment 4 chat Q2).
3. Step 6 critical-clean → PASS-DEPLOYABLE / PASS-VIABLE finalised.
4. Step 6 critical-fails → Top-1's amended gate re-classified with `causal_audit_clean=False` → FAIL with `primary_failure_mode = step6_causal_audit_fail`.

Step 6 is the LAST gate, not concurrent with the others. Manual CLI invocations never modify the verdict regardless of Step 6 outcome.

### Failure-mode priority (tie-break)

When multiple constraints fail simultaneously, `primary_failure_mode` is assigned by first-fail in this order:

1. `pool_too_small` (Step 1 failure — encountered before Step 5 evaluation)
2. `step5_not_scalable`
3. `step5_dd_above_gate` (defensive — should not occur by construction)
4. `step5_chained_dd_above_gate`
5. `step5_daily_dd_breach`
6. `holdout_fail_after_is_pass`
7. `step5_sign_consistency_fail` / `step5_negative_folds`
8. `step5_trade_count_below_gate`
9. `step5_wf_roi_below_gate_after_scaling`
10. `step5_ratio_below_gate_after_scaling`
11. `step6_causal_audit_fail`

Other §3 failure modes not listed (no clusters separable, no capturable cluster, etc.) remain ordered per the wider protocol.

### Failure-mode taxonomy

Added by Amendment 3:

- `step5_not_scalable`
- `step5_daily_dd_breach`
- `step5_chained_dd_above_gate`
- `step5_wf_roi_below_gate_after_scaling`
- `step5_ratio_below_gate_after_scaling`
- `step5_trade_count_below_gate`
- `step5_negative_folds`

Deprecated but retained for historical closures:

- `step5_dd_above_gate` (replaced by scalability check + scaled gates)

### A5 follow-up flag (deferred)

Combined A5 portfolio DD when each component is independently scaled to `r_hard` (worst-fold DD = 10%) can produce combined account-wide DD > 10%. The current §2 Step 5 A5 spec does not address this; Amendment 3 does not either. A separate amendment will specify combined-portfolio DD constraints for VIABLE components. Deferred until the first VIABLE candidate emerges and A5 deployment is concretely on the table.

### Holdout window

Holdout = "2021-01-01 to present at time of arc closure" — matches §2 Step 5 and Appendix B. Moving target by design.

### r_max note

`r_max = 2.0%` is the gate ceiling. Locked value.

---

## §4 Architecture search conventions

The Step 5 search space is bounded by the architectures listed in §2 Step 5. New architectures require a protocol amendment.

**Per architecture, the search dimensions are:**

- Filter set (from Step 4 candidates including "no filter")
- Exposure rule: `max_concurrent_per_currency` ∈ {1, 2, 3, unlimited}, `max_concurrent_total` ∈ {3, 5, 10, unlimited}
- SL multiplier ∈ {1.5, 2.0, 2.5, 3.0, 3.5, 4.0} × ATR(14)
- Exit policy: default is stop-loss only (signal's natural exit). Arcs may declare alternate exit policies at open.
- TF (for TF-flexible arcs): the declared `tf_set`

The arc's closure doc reports the search space size (for selection-bias accounting) and the top-K candidates by worst-fold ratio.

---

## §5 Sub-protocol mechanism

**Default:** an arc runs vanilla overseer.

**Sub-protocol invocation:** the arc's `ARC_OPEN.md` declares `sub_protocol: <name>`. The named file at `docs/sub_protocols/<name>.md` overrides or extends specific overseer steps for that arc.

**Canonical sub-protocols** (defined in `docs/sub_protocols/`):

- `heavy_ml_probe.md` — replaces Step 4 default classifiers with AutoML (autogluon or FLAML) + meta-labeling target + feature selection inside CV. Invoked for arcs with established cohort edge where vanilla Step 4 didn't extract.
- `signal_discovery_probe.md` — replaces Step 1 signal application with rule-search (genetic algorithm or random search over rule combinations under causal-audit constraint). Steps 2-5 then evaluate discovered signals.
- Future sub-protocols added as needed: currency_strength, session_anchored, news_windowed, etc.

**What a sub-protocol can override:**

- Step 1 feature engineering — e.g. currency-strength adds rank-based features and cross-sectional context
- Step 1 signal source — e.g. signal_discovery_probe generates the signal via search
- Step 2 clustering scheme — e.g. regime sub-protocol clusters on regime indicators instead of path shape
- Step 3 capturability metrics — e.g. session sub-protocol adds session-conditional reach rates
- Step 4 classifier choices — e.g. heavy_ml_probe uses AutoML
- Step 5 architecture variants — e.g. cross-sectional adds rank-long-short architectures

**What a sub-protocol CANNOT override:**

- §1 non-negotiables
- §3 gates
- Anchor preservation rule
- Closure doc format

**Sub-protocol changes are arc-isolated.** A sub-protocol's effects appear only in arcs that invoke it. The overseer remains unchanged.

---

## §6 Documentation conventions

### Files

- `L_PROTOCOL.md` (this file)
- `docs/sub_protocols/<name>.md` — sub-protocol definitions
- `ARC_TRACKER.md` — single live tracker, auto-updated
- `results/<arc_name>/ARC_OPEN.md`
- `results/<arc_name>/ARC_CLOSURE.md`
- `results/<arc_name>/step_<N>/` — step artefacts

### ARC_OPEN.md required fields

```
arc_name: <unique identifier>
opened: <ISO timestamp>
signal_class: <e.g. "swing-high breakout">
signal_definition: <reference to spec doc or inline definition>
tf_mode: locked | flexible
tf: <if locked>
tf_set: <if flexible>
sub_protocol: vanilla | <name>
pair_set: <list>
window: <start, end>
risk_per_trade: <default 0.5%>
hypothesis: <one paragraph what this arc tests>
expected_failure_modes: <if any anticipated>
```

### ARC_CLOSURE.md format

All arc closure docs MUST follow `docs/templates/ARC_CLOSURE_TEMPLATE.md` (current v1.3.1).

Required sections:

1. **§1 tracker_payload** — machine-parseable YAML block. Source of truth for ARC_TRACKER updates. Field names and structure are locked.
2. **§2 Why <failed | succeeded>** — required prose, 100-300 words.
3. **§3 Cross-arc observations** — required bullet list.
4. **§4 deployment_spec** — REQUIRED for any verdict in {PASS-DEPLOYABLE, PASS-VIABLE, PASS-*-PROVISIONAL, PASS-*-PENDING-STEP6}. Self-contained porting specification. OPTIONAL for FAIL / HALT / DISCOVERY_COMPLETE verdicts.

Closure docs not conforming are invalid. Tracker updates blocked until repaired.

### ARC_TRACKER.md update mechanism

ARC_TRACKER updates are driven by each closure doc's `§1 tracker_payload` YAML block.

Parser at `scripts/update_tracker_from_closure.py` applies the Section 4 A-K mapping mechanically. Invoke per `scripts/tracker_parser/README.md` — invocation is part of the standard arc-close workflow (run on the arc branch before opening the closure PR; tracker delta lands in the same atomic commit as the closure doc).

Tracker is append-only at the row level. Bad rows flagged with ⚠️ prefix, never deleted. Schema changes require explicit chat-side redesign event documented in the closure doc that introduced them.

---

## §7 Branch strategy

- `main` — canonical
- Per arc: `arc/<arc_name>` worktree, cut from main
- Arcs run in parallel on independent branches. They do not merge into each other; each PRs to main when its closure lands.
- Arc closure: PR opened. Chat reviews and merges to main if clean.
- Auto-cleanup: post-merge git hook removes the worktree and deletes the branch (local + remote). No manual cleanup required.
- ARC_TRACKER.md is the only file multiple arcs touch. Append conflicts at simultaneous-close are resolved chronologically — no content dropped.

---

## §8 Anchor preservation

KH-24 run through the protocol must produce worst-fold numbers consistent with its known deployment lineage (within ±0.5pp ROI, ±1pp DD). If it doesn't, the protocol or backtester has a bug; quarantine results until fixed. KH-24 itself is locked; the protocol does not modify it.

---

## §9 What is NOT in this protocol

The following are deliberately out-of-scope:

- Phase planning (what to run next) — lives in `RE_RUN_PLAN.md`
- Specific arc plans — live in per-arc `ARC_OPEN.md`
- Backtester implementation — lives in `BACKTESTER_ARCHITECTURE.md`
- Data foundation specifics — live in `DATA_FOUNDATION.md`

---

## Appendix A — Default classifier hyperparameters

```
RandomForest:
  n_estimators: 200
  max_depth: 6
  min_samples_leaf: 50
  random_state: 42

LightGBM:
  n_estimators: 200
  num_leaves: 31
  learning_rate: 0.05
  min_child_samples: 50
  random_state: 42

LogisticRegression:
  penalty: l2
  C: 1.0
  max_iter: 1000
  random_state: 42
```

These defaults are locked. Per-arc tuning lives in sub-protocols if needed.

---

## Appendix B — Default architecture search space

```
wfo:
  train_window: 2010-01-01 to 2020-12-31
  fold_size: 1 year  (11 folds)
  holdout_window: 2021-01-01 to present
  holdout_evaluation: one-shot per candidate

exposure:
  max_concurrent_per_currency: [1, 2, 3, unlimited]
  max_concurrent_total: [3, 5, 10, unlimited]

sl_multiplier_atr_14: [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

exit_policy:
  - sl_only
  - sl_plus_tp_2r
  - sl_plus_tp_3r
  - sl_plus_trailing_atr  (trail at 1 ATR distance once 1R reached)
  - sl_plus_trailing_swing  (trail at recent swing low/high)
  - sl_partial_close_1r_runner_trail  (close 50% at 1R, trail rest)
  - time_exit_n_bars  (close at bar N regardless)

pipeline_de_n_bars: [1, 3, 5, 8]

architectures_enabled:
  - system_level_filter
  - classifier_filter
  - pipeline_de
  - pipeline_d_exits
  - portfolio_composition
  - meta_labeling
```

---

## End

Protocol locked. Changes require explicit redesign event documented in chat.
