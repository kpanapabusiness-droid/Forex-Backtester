# Arc 7 v3.0.2 — CC intent doc (A2/A6 follow-up retry under architecture-map override)

> **Dispatch source:** chat 2026-05-25 (Arc 7 v3.0.2 A2/A6 follow-up)
> **Branch / worktree:** `arc/l_arc_7_v3.0.2` at `.claude/worktrees/clever-elbakyan-ad50ee`
> **Branch base:** cut from `arc/l_arc_7 @ ba6c5b0` (NOT origin/main — see §1 below)
> **Standing instruction:** continuous run Step 5 + Amendment 3 + Step 6 auto + closure + PR after intent review. End turn ONLY at intent / HALT / PR open.

This retry tests the chat-side architecture-map override: when any candidate
cluster's Step 4 best-classifier mean OOS AUC ≥ 0.65, that cluster's
architecture set is augmented to include A2 and A6 regardless of archetype.
v3.0.1 closure §3 surfaced this gap twice (Arc 7 v3.0 c0 Bimodal AUC 0.6758;
Arc 7 v3.0.1 c1 Unclassified AUC 0.6642). v3.0.2 tests whether closing the
gap flips the verdict.

---

## §0 Branch base correction (dispatch literal vs intent)

Dispatch literal: `git checkout -b arc/l_arc_7_v3.0.2 origin/main`.
Reality: PR #180 (Arc 7 v3.0.1) is **MERGEABLE but not yet merged**;
origin/main HEAD is still `4fccff0`. Cutting from origin/main would NOT
include `results/l_arc_7/` (the v3.0.1 Step 1-4 artefacts the dispatch
explicitly directs CC to reuse).

CC interpretation: dispatch intent = "from a state that includes v3.0.1
artefacts". Cut from `arc/l_arc_7 @ ba6c5b0` (the v3.0.1 commit) — preserves
all Step 1-4 outputs at `results/l_arc_7/` for direct reuse.

Force-push at PR open NOT needed — first push of `arc/l_arc_7_v3.0.2`.

---

## §1 Step 1-4 artefacts reused (no re-execution)

Per dispatch §"Execution mode": Steps 1-4 NOT re-run. All consumed from
`results/l_arc_7/` (Arc 7 v3.0.1 outputs, preserved verbatim on the
branch base):

### Step 1 (pool)
- `results/l_arc_7/step_1/pool.parquet` — 5,175 trades, sha256
  `402048a2e231dcbc9394b38f82d8a1032ceb368fcb1b4ccce85a9fd8720ad460`
- `results/l_arc_7/step_1/paths.parquet`
- `results/l_arc_7/step_1/integrity_report.md`
- `results/l_arc_7/step_1/manifest.json`

### Step 2 (clustering)
- `results/l_arc_7/step_2/cluster_assignments.parquet` (K=4)
- `results/l_arc_7/step_2/cluster_summary.{csv,md}`

### Step 3 (capturability)
- `results/l_arc_7/step_3/capturability.csv` — 2 candidates flagged
- `results/l_arc_7/step_3/capturability_summary.md`

### Step 4 (extraction + persisted classifiers)
- `results/l_arc_7/step_4/extraction_metrics.csv`
- `results/l_arc_7/step_4/feature_importance.csv`
- `results/l_arc_7/step_4/extraction_summary.md`
- `results/l_arc_7/step_4/classifiers/0.pkl` — LR Pipeline, AUC 0.6192, sha-verified ✓
- `results/l_arc_7/step_4/classifiers/1.pkl` — RandomForestClassifier, AUC 0.6642, sha-verified ✓
- `results/l_arc_7/step_4/classifiers/manifest.json`

## §2 Fitted classifier loadability (PR #185 integrity check)

Both persisted classifiers load cleanly via
`core.steps.classifier_persistence.load_classifier`:

```
cluster 0: LR  AUC_oos_cv5=0.6192 threshold=0.1140 → Pipeline loaded ✓
cluster 1: RF  AUC_oos_cv5=0.6642 threshold=0.3534 → RandomForestClassifier loaded ✓
```

SHA256 verified against `manifest.json` per
`ClassifierIntegrityError` contract. No version drift warnings.

## §3 Per-cluster Step 4 AUC vs 0.65 bar (override resolution)

| Cluster | Archetype | Best clf | AUC | ≥ 0.65? | Override applied? |
|---|---|---|---:|---|---|
| c0 | Bimodal | LR | 0.6192 | NO | No augmentation |
| c1 | Unclassified | RF | **0.6642** | **YES** | **+A2, +A6** |
| c2 | Unclassified | — | — (dies Step 3) | n/a | n/a |
| c3 | Monotonic_down | — | — (dies Step 3) | n/a | n/a |

## §4 Per-cluster architecture set under override

| Cluster | Archetype-map set | Override addition | Final set |
|---|---|---|---|
| c0 (Bimodal, AUC 0.6192) | A1, A4 | — | **A1, A4** |
| c1 (Unclassified, AUC 0.6642) | A1 | +A2, +A6 | **A1, A2, A6** |
| A5 portfolio (≥ 2 candidates) | (auto) | — | **A5** runs across c0+c1 best |

**Net new architectures vs v3.0.1:** A2 + A6 for c1 + A5 portfolio (v3.0.1
ran A1 + A4 + A1 only — no A2/A6, no A5).

## §5 Step 5 search dimensions per architecture

Per dispatch + L_PROTOCOL Appendix B:

| Architecture | SL (±1 step around cluster's Step-3 selected) | Trail | Exposure | Arch-specific axis | Configs / cluster |
|---|---|---|---|---|---:|
| A1 | 3 values | {True, False} | {2, unlimited} | — | 12 |
| A2 | 3 values | True (default) | {2, unlimited} | — | 6 |
| A4 | 3 values | True (default) | {2, unlimited} | exit_threshold ∈ {0.3, 0.4, 0.5} | 18 |
| A6 | 3 values | True (default) | {2, unlimited} | threshold pairs ∈ {(0.3,0.5),(0.4,0.6),(0.5,0.7)} | 18 |
| A5 | (composes constituents) | n/a | n/a | (top-1 per cluster × per-fold) | 1 |

**Per-cluster config counts:**
- c0 (A1+A4): 12 + 18 = 30
- c1 (A1+A2+A6): 12 + 6 + 18 = 36
- A5: 1 (combines top-1 per cluster per fold)
- **Total: 67 configs** → `normal` search-scope flag

(v3.0.1 was 28 configs / `thin`. The +A2+A6+A5 expansion accounts for the
delta.)

Skipped exit-policy variants (v3 architecture-side limitation; unchanged
from v3.0.1; explicitly disclosed at `step_5/skipped_configs.md`):
sl_plus_tp_2r, sl_plus_tp_3r, sl_partial_close_1r_runner_trail,
sl_plus_trailing_swing, time_exit_n_bars. v3 A1/A2/A6 Configs don't expose
tp_atr_mult or partial-close mechanics.

## §6 ML mechanics (Amendment 2)

- **A2**: loads `results/l_arc_7/step_4/classifiers/1.pkl` (c1's RF) via
  `core.steps.classifier_persistence.build_a2_config_from_step4(s4_loaded,
  cluster_id=1, threshold=0.3534, ...)`. NO retrain.
- **A6**: same loaded classifier as A2 (per L_PROTOCOL §2 Step 5 Amendment
  2 Architecture-specific retraining policy: A2 + A6 share the persisted
  Step-4 best-AUC classifier). Confidence → 0×/0.5×/1× sizing per the
  threshold-pair sweep.
- **A4**: per-fold retrain via
  `core.steps.path_classifier_per_fold.build_path_classifier_fits_per_fold`
  with `arch="A4"`, target=`final_r > 0`, decide-offset 5. Cluster-
  independent. Trail-first precedence per PR #186 lock.
- **A1**: rule-based, no classifier. `filter_rules=()` (no Step 4 rule
  output; default unfiltered).
- **A5**: composes top-1 per cluster's StrategyResult per fold.

## §7 Files CC will touch / create

### Created at intent stage (this turn)
- `docs/dispatches/arc_7_v3_0_2_intent.md` — this doc

### Created at Step 5+ (driver)
- `scripts/l_arc_7_v3_0_2/__init__.py`
- `scripts/l_arc_7_v3_0_2/run.py` — composition driver. Reuses
  `results/l_arc_7/` Step 1-4 artefacts; runs Step 5 + Amendment 3 + Step 6 +
  Oracle. Composes canonical primitives end-to-end (no hand-rolled step
  implementations per dispatch §"What you do NOT do"; uses canonical
  `core/architectures/a{1,2,4,5,6}*.py`, `core/steps/classifier_persistence`,
  `core/steps/path_classifier_per_fold`, `core/runners/arc_fold_runner`,
  `core/wfo/amended_gates` + `chained_dd` + `holdout_rerun` +
  `compute_per_day_max_dd`, `core/step_6/dispatch`, `core/runners/oracle_fold_runner`).
- `scripts/l_arc_7_v3_0_2/write_closure.py` — adapts
  `scripts/arc_7/write_closure.py` for the v3.0.2 output paths + v3.0.1
  comparison narrative. Closure under template v1.3.

### Output artefacts under `results/l_arc_7_v3.0.2/`
- `ARC_OPEN.md`
- `step_5/wfo_results.csv`
- `step_5/per_fold_metrics.csv`
- `step_5/per_day_max_dd_base__<safe_cid>.parquet` per top-K
- `step_5/holdout_results.csv`
- `step_5/wfo_oracle.csv`
- `step_5/wfo_summary.md`
- `step_5/architectures_ranked.md`
- `step_5/skipped_configs.md`
- `step_5/manifest.json`
- `step_6/` (conditional on PASS-tier candidate clearing §3 #1-9)
- `ARC_CLOSURE.md` — template v1.3 (§1 + §2 + §3 + §4 deployment_spec on PASS)
- `run_summary.json`

### Other artefacts
- `docs/dispatches/arc_7_v3_0_2_log.md` (WORKFLOW §2)
- `ARC_TRACKER.md` updated via parser
- `scripts/tracker_parser/{rolling_state.json, parsed.log}` updated

### NOT touched per dispatch
- `results/l_arc_7/` (v3.0.1 closure preserved; only READ)
- `core/` (no engine code modifications)
- `L_PROTOCOL.md`, `WORKFLOW.md` (no methodology changes)
- KH-24 anywhere

## §8 Window / pair set / base risk / TF / bar boundary

| Field | Value | Source |
|---|---|---|
| Window | 2010-01-01 → 2026-04-30 | dispatch + v3.0.1 reuse |
| Pair set | 28 FX | v3.0.1 pool |
| Base risk | 0.5% (verdict at scaled `r_safe`/`r_hard`) | dispatch + Amendment 3 |
| TF | 4H locked | dispatch |
| Bar boundary | **5ers_eet per PR #189** | v3.0.1 pool (no re-aggregation) |
| Direction | long only | spec |
| SL (Step 1 anchor) | 2.0 × ATR(14) | v3.0.1 pool |
| Sub-protocol | vanilla | dispatch |
| Sizing convention | reset_floor | A1Config default |

No panel re-build needed — Step 5 reads pool/paths/cluster artefacts directly
from `results/l_arc_7/`. ArcFoldRunner does need panels for fold slicing
during architecture execution; will be re-built from the same `5ers_eet`
cache at `data/cache/{H4,D1,W1}_5ers_eet/` (warm from v3.0.1).

## §9 Step 6 framework auto-dispatches on PASS

Per Amendment 4 / PR #188 — same wiring as v3.0.1:

- `core.step_6.dispatch.maybe_dispatch_step_6` invoked after Amendment 3
  evaluation completes
- Auto-dispatch on Top-1 PASS-tier candidate
- Critical failure → `replace_top_1_with_step6_fail` downgrades to FAIL
  with `primary_failure_mode = step6_causal_audit_fail`
- Manual CLI NOT invoked

§6.3 UTC-vs-EET ambiguity per v3.0.1 closure §3: framework gap not arc
failure; master chat owns framework fix. If v3.0.2 produces a PASS and
§6.3 fires on this mismatch, closure §3 flags it explicitly.

## §10 Determinism plan

- `random_state = 42` (per `core.determinism.RANDOM_STATE`)
- `n_jobs = 1` inside per-row work
- `lineterminator = "\n"` on every text artefact
- `seed_everything(42)` at driver entry
- Parquet caches reused (already warm):
  - `data/cache/{H4,D1,W1}_5ers_eet/` (panels)
  - `results/l_arc_7/step_4/classifiers/` (persisted RF + LR via SHA256
    verification each load)
- A4 per-fold retrain seed: derived per
  `core.steps.path_classifier_per_fold.derive_per_fold_seed`

## §11 Expected outcomes (predictions for verdict-comparison narrative)

For closure §2's "compare to v3.0.1" requirement, predictions:

- **v3.0.1 baseline (A1 unfiltered):** worst-fold ratio -0.82, DD base 34.2%,
  r_safe 0.117% (below 0.15% floor) → `step5_not_scalable` FAIL.

- **v3.0.2 A2 for c1:** classifier admits only c1-like trades. c1's
  oracle worst-fold ratio is 6.80 (DD 2-4%). Realised AUC 0.6642 sits
  between chance (0.5) and oracle (1.0). Likely realised DD between A1's
  34% and c1 oracle's 4% — order-of-magnitude estimate, 8-15%. Whether
  this hits `r_safe ≥ 0.15%` depends on actual DD. Predicting borderline.

- **v3.0.2 A6 for c1:** sizes by confidence; smaller positions on
  low-confidence signals. DD scaling depends on sizing distribution.
  Likely similar profile to A2 but smoother.

- **v3.0.2 A5 portfolio:** combines c0 best + c1 best. Diversification
  could improve worst-fold DD vs either alone. Per L_PROTOCOL §3 A5
  follow-up flag, combined-portfolio DD constraints under VIABLE are
  undefined — closure must flag if A5 becomes the verdict-carrying
  candidate.

- **Step 6 dispatch:** activates if any candidate clears §3 #1-9.

These are predictions only. v3.0.2 measures what actually happens.

---

## End of intent

Per dispatch §"Intent doc": **end turn for chat review**.

On continuation: CC executes Step 5 + Amendment 3 + Step 6 auto-dispatch +
closure + tracker update + push + PR continuously, no inter-step pauses,
HALT only on genuine bugs/blockers per dispatch §"HALT triggers".
