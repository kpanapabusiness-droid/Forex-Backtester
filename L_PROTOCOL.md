# L_PROTOCOL v3.0 — The Overseer

> **Version:** 3.0
> **Status:** locked. Changes require explicit redesign event.
> **Purpose:** the methodology by which a single arc runs from open to verdict.
> **Scope:** how to run things. Not what to run next.
>
> **Amendment 1 (2026-05-22):** Step 5 search policy clarified — informed by Steps 3-4, not exhaustive. Multi-cluster handling + holdout decision rule specified. See §2 Step 5.
> **Amendment 2 (2026-05-22):** ML architecture mechanics specified for A2, A3, A4, A6. See §2 Step 5 ML mechanics subsection.
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

**Architecture selection:** 2-3 from {A1..A6} that fit the dominant cluster archetype:
- Stepwise climber → A1 (system filter), A2 (classifier filter), A4 (trailing-exit Pipeline D)
- V-shape recovery → A1, A3 (Pipeline DE — deferred entry), A6 (meta-labeling)
- Bimodal → A1, A4 (per-archetype Pipeline D exits)
- Monotonic up → A1, A2, A6
- Choppy → no architectures expected to survive; arc typically dies at Step 3 before reaching Step 5
- A5 (portfolio composition) → only applies if 2+ candidate clusters survive Step 3

If Step 4 produced NO usable filter/classifier (AUC at chance, no rule above noise), A1 still runs with "no filter" baseline (raw signal + system-level rules: exposure, SL, exit) — this tests whether the system works without filtering.

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

**Shared discipline across all ML architectures:**
- All classifiers respect causal lineage tags from Step 1 — no "suspect" or "unverified" features enter training
- Deterministic training: `random_state=42`, `n_jobs=1` per Appendix A
- Per-fold WFO trains classifier on IS, evaluates on OOS — no global model trained once and used across folds
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

### Step 6 — Causal audit (lazy, deployment-only)

**Trigger:** runs only when Step 5 produces ≥ 1 candidate clearing PASS-DEPLOYABLE or PASS-VIABLE.

**Mechanics:**

1. For each feature in the winning candidate's filter / classifier:
   - Producer-level trace: where does this feature's value come from? What columns of the underlying OHLC drive it? What time index? Any aggregations?
   - Verify producer code uses no lookahead, no post-signal data, no leakage from related instruments.
   - Verify the feature value at trade time T is reproducible from data available at time T only.

2. End-to-end check: regenerate a small random sample of feature values from raw OHLC, byte-compare to pool values.

3. If any feature fails — flag, and either:
   - Downgrade candidate (if the failing feature is in a non-critical filter)
   - Kill candidate (if the failing feature is load-bearing)

**Output:**
- `step_6/causal_audit_report.md`
- `step_6/manifest.json`

**Heavy compute, rare invocation.** Not setup until needed.

---

## §3 Gates

### PASS-DEPLOYABLE (ship)

All of:
- worst-fold ROI/DD ratio ≥ 2.0
- worst-fold ROI > 0 at any risk size; positive at all 11 folds (WFO)
- at chosen risk size: DD ≤ 8%
- at chosen risk size: 0 days breaching 5% daily DD
- at chosen risk size: max DD ≤ 10% (5ers hard limit)
- sign-consistency: 0 negative folds
- ≥ 25 trades per fold
- Step 6 causal audit clean

### PASS-VIABLE (portfolio candidate, does NOT ship alone)

All of:
- worst-fold ROI/DD ratio ≥ 2.0
- mean-fold ROI/DD ratio ≥ 2.5
- worst-fold ROI may be negative (single negative fold permitted)
- at chosen risk size: max DD ≤ 10%
- at chosen risk size: 0 days breaching 5% daily DD
- ≥ 25 trades per fold
- Step 6 causal audit clean (if invoked)

A PASS-VIABLE strategy becomes deployable when combined with another PASS-VIABLE / PASS-DEPLOYABLE into a portfolio whose COMBINED account performance passes PASS-DEPLOYABLE gates. Portfolio composition mechanics handled within Architecture A5.

### FAIL

Everything else. The arc's closure doc explains why and what would help.

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

All arc closure docs MUST follow `docs/templates/ARC_CLOSURE_TEMPLATE.md` v1.0.

The template has three required sections:

1. **§1 tracker_payload** — machine-parseable YAML block. Source of truth for ARC_TRACKER updates. Field names and structure are locked.
2. **§2 Why <failed | succeeded>** — required prose, 100-300 words. Preserves cross-arc synthesis quality.
3. **§3 Cross-arc observations** — required bullet list. What does this arc add to cumulative findings?

Closure docs not conforming to the template are invalid. Tracker updates cannot be applied until the closure is repaired (per template Section 4-K).

### ARC_TRACKER.md update mechanism

ARC_TRACKER updates are driven by each closure doc's `§1 tracker_payload` YAML block.

Until parser ships: manual updates per `docs/templates/ARC_CLOSURE_TEMPLATE.md` Section 4 (steps A through K).

Once parser ships (`scripts/update_tracker_from_closure.py`): same Section 4 mapping applied mechanically. Parser specification in template Section 5.

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
