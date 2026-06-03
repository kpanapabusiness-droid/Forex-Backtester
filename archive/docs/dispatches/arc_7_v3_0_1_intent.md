# Arc 7 v3.0.1 — CC intent doc (retry under PRs #185/#186/#188/#189)

> **Dispatch source:** chat 2026-05-22 (Arc 7 Retry v3.0.1)
> **Branch / worktree:** `arc/l_arc_7` (hard-reset to `origin/main` @ `4fccff0`)
> **Worktree HEAD:** `4fccff0` [INFRA] TODO.md refresh — post-PR-#189 state
> **Supersedes:** prior Arc 7 v3.0 closure (was FAIL; pre-amendment engine).
> **Standing instruction:** continuous run Steps 1→5 + Step 6 auto-dispatch + closure + PR after this intent doc. End turn ONLY at intent / HALT / PR open.

CC_07 → PR #185 (Step 4 classifier persistence) → PR #186 (Amendment 3 +
A3/A4 wiring) → PR #188 (Step 6 framework + Amendment 4 + template v1.3) →
PR #189 (mid-price features + worst-case fills + EET bar boundaries). All
infrastructure that blocked the prior Arc 7 closure is now live on main.

---

## 1. Resolved signal spec

`core/signals/` carries `pullback_resume_hhhl.py` (Arc 8) only — no
`liquidity_sweep_reclaim_long.py`. Per dispatch §"Read-first" item 7, fall
back to `docs/archive/arc_results/ARC_7_RESULT.md` (v2.1.2 closure) for the
trigger spec.

Spec source: `docs/archive/arc_results/ARC_7_RESULT.md` §"Trigger" — unchanged
from the v3.0 first-attempt intent. Trigger spec re-stated in §2 below; full
producer-level causal trace at §3.

## 2. Signal definition (six elements per dispatch §"Intent doc" item 2)

```
swing_low_N = min(low_bid[t-N..t-1]), N = 20
low_bid[t]  < swing_low_N                                  # sweep
close_bid[t] > swing_low_N                                 # reclaim
swing_low_N − low_bid[t] ≥ 0.25 × ATR(14)[t]               # magnitude
close_bid[t] > open_bid[t]                                  # bullish reclaim bar
(close_bid[t] − swing_low_N) /
    (swing_low_N − low_bid[t]) ≥ 0.5                       # reclaim strength
gap_since_last_signal_on_pair ≥ 20 bars                    # refractory
```

| Spec element | Value |
|---|---|
| swing-low window N | 20 bars |
| magnitude threshold (×ATR) | 0.25 × ATR(14)[t] |
| reclaim condition | `close_bid[t] > swing_low_N` AND `close_bid[t] > open_bid[t]` |
| reclaim-strength ratio | ≥ 0.5 |
| refractory bars | ≥ 20 bars |
| primary TF / direction | 4H, long only |

Wording is unambiguous in the v2.1.2 closure; no HALT under dispatch §"HALT
triggers" item 6.

## 3. Producer-level causal trace of swing-low detection (Arc 9 lesson)

> Arc 9's incident (`results/archived/l_arc_9/INCIDENT_2026_05_19_ARC_9_PRODUCER_LEAK.md`)
> traced a producer-level lookahead in swing detection. This trace verifies
> the LSR swing-low producer has no such leak.

**Producer:** `core/strategies/liquidity_sweep_reclaim_long/signal_module.py`
(to be (re)created — see §4 below). The swing-low computation will be:

```python
# Inside _evaluate_pair:
low_bid = pair_df["low_bid"]
swing_low = (
    low_bid
    .rolling(window=params.swing_window, min_periods=params.swing_window)  # 20
    .min()
    .shift(1)
)
# swing_low.iloc[t] = min(low_bid.iloc[t-20], ..., low_bid.iloc[t-1])
# Strictly prior bars only — low_bid.iloc[t] is NOT in the window.
```

**Why this is causal-clean:**

1. `rolling(window=20).min()` at index `t` returns the min over indices
   `[t-19, t]` (pandas convention: trailing window includes current bar).
2. `.shift(1)` shifts the entire series forward by one index, so the value
   AT index `t` becomes the value previously at index `t-1`.
3. Net effect at index `t`: `swing_low.iloc[t]` = min over indices `[t-20, t-1]`.
4. `low_bid.iloc[t]` itself is NEVER in the window for `swing_low.iloc[t]`.

**Right-edge audit:** the swing-low detection has NO equivalent to Arc 8's
`swing_high(k) AND k+1..k+3` lookahead-within-detection-window pattern.
LSR's swing-low is a pure rolling min over prior bars; no centred-window
or right-edge confirmation lag involved.

**Inputs at bar t (all closed strictly before bar t's open + close):**

- `swing_low[t]` — uses `low_bid[t-20..t-1]` (strictly prior bars) ✓
- `low_bid[t]`, `close_bid[t]`, `open_bid[t]` — current bar; known at bar
  t's close ✓
- `ATR(14)[t]` — Wilder EWM on `high_bid/low_bid/close_bid` shifted via
  `prev_close = close.shift(1)`. The TR at bar t uses `high[t]`, `low[t]`,
  `close[t-1]` — all known by bar t's close. ✓
- `gap_since_last_signal` — stateful walk over prior `True` indices on the
  same pair only ✓

**Causal lineage tag:** `clean`. Step 6's `§6.1 lookahead` audit will
exercise this producer via the byte-compare harness; it WILL fire on any
PASS verdict per Amendment 4.

**Step 1 elevated lookahead check:** dispatch §"Step 1" requires 10 random
trades (vs default 5) with explicit causal-lineage trace per trade. CC will
extend `core.arc.integrity.check_lookahead_spot` invocation to `n_samples=10`
at driver level (the function accepts the kwarg already).

## 4. Files CC will touch / create

### Created this turn (intent only)

- `docs/dispatches/arc_7_v3_0_1_intent.md` — this doc

### Recreated at Step 1 (was discarded by `git reset --hard origin/main`)

- `core/strategies/liquidity_sweep_reclaim_long/__init__.py` (re-create)
- `core/strategies/liquidity_sweep_reclaim_long/signal_module.py` (re-create
  per §2/§3 above; mid-price NOT used for signal evaluation — signal stays
  bid-side per KH-24 / Arc 8 / Arc 11 convention; PR #189's mid-price refactor
  applies to FEATURES, not to signal evaluation per `PROTOCOL_RUNTIME.md` §15.1)
- `tests/test_liquidity_sweep_reclaim_long_signal.py` (re-create; 5 unit
  tests: SignalModule conformance, all-six-conditions fire, C4-violation
  suppression, no-lookahead perturbation, params locked)

### Spec doc

- `docs/archive/signal_specs/signal_spec_liquidity_sweep_reclaim_long_v0.1.md`
  (re-create from the v2.1.2 closure for the conventional location)

### Driver

- `scripts/arc_7/__init__.py`
- `scripts/arc_7/run_arc_7.py` — composition driver that uses canonical
  primitives end-to-end (no hand-rolled step IMPLEMENTATIONS per dispatch).
  Architecture: build panels (H4 + D1 + W1, EET) → `build_arc_pool` (Step 1)
  → `compute_feature_matrix` per pair → `run_step_2` → `run_step_3` →
  archetype-based architecture selection per dispatch §"Step 5" map →
  `run_step_4(persistence_dir=...)` (Step 4 with PR #185 persistence) →
  build `AutoArchSpec` list per archetype → call
  `ArcOrchestrator(cfg, signal_module, panels).run()` which fires
  Step 5 + Amendment 3 evaluation + Step 6 auto-dispatch end-to-end.

  Alternative if orchestrator re-running Steps 1-4 wastes too much wall
  time: drop to direct composition of `run_search` + `run_holdout` +
  `_run_amendment_3_evaluation` components + `maybe_dispatch_step_6`.
  Decision deferred to execution time based on Step 1 wall-time
  measurement.

### Output artefacts (per L_PROTOCOL §6 + Amendment 3 + Amendment 4)

- `results/l_arc_7/ARC_OPEN.md`
- `results/l_arc_7/step_1/pool.parquet`
- `results/l_arc_7/step_1/paths.parquet`
- `results/l_arc_7/step_1/integrity_report.md`
- `results/l_arc_7/step_1/manifest.json`
- `results/l_arc_7/step_2/cluster_assignments.parquet`
- `results/l_arc_7/step_2/cluster_summary.{md,csv}`
- `results/l_arc_7/step_2/manifest.json`
- `results/l_arc_7/step_3/capturability.csv`
- `results/l_arc_7/step_3/capturability_summary.md`
- `results/l_arc_7/step_3/manifest.json`
- `results/l_arc_7/step_4/extraction_metrics.csv`
- `results/l_arc_7/step_4/feature_importance.csv`
- `results/l_arc_7/step_4/extraction_summary.md`
- `results/l_arc_7/step_4/manifest.json`
- `results/l_arc_7/step_4/classifiers/{<cluster_id>.pkl, manifest.json}` (PR #185)
- `results/l_arc_7/step_5/wfo_results.csv`
- `results/l_arc_7/step_5/per_fold_metrics.csv`
- `results/l_arc_7/step_5/per_day_max_dd_base__<safe_cid>.parquet` per top-K (PR #186)
- `results/l_arc_7/step_5/holdout_results.csv`
- `results/l_arc_7/step_5/wfo_summary.md`
- `results/l_arc_7/step_5/wfo_oracle.csv`
- `results/l_arc_7/step_5/skipped_configs.md` (dispatch — explicit list of
  not-tested exit policies; no silent shrinkage)
- `results/l_arc_7/step_5/manifest.json`
- `results/l_arc_7/step_6/` (conditional on PASS-tier candidate clearing
  §3 #1-9) — `manifest.json`, `summary.md`, six `<category>_report.md`,
  `sha256_manifest.json`
- `results/l_arc_7/ARC_CLOSURE.md` — written per template v1.3 (§1
  tracker_payload with Amendment 3 fields + step_6 block; §2 prose; §3
  cross-arc; §4 deployment_spec REQUIRED on PASS)
- `results/l_arc_7/run_summary.json` — driver-level summary the closure
  generator consumes
- `docs/dispatches/arc_7_v3_0_1_log.md` (WORKFLOW §2)
- `ARC_TRACKER.md` updated via `scripts/update_tracker_from_closure.py`
- `scripts/tracker_parser/{rolling_state.json, parsed.log}` updated by parser
- `configs/arc_7/v3_0_1.yaml` — `config_artefact_path` for §4
  deployment_spec REQUIRED if PASS verdict (parser-enforced)

### Conditional artefacts

- HALT diagnostic at `docs/dispatches/arc_7_v3_0_1_diagnostic.md` IF HALT
  triggered (no PR opened in that case)

## 5. Confirmed window / pair set / base risk / TF

| Field | Value | Source |
|---|---|---|
| Window | 2010-01-01 → most-recent-complete-month (2026-04 at run time) | dispatch front-matter |
| Pair set | 28 FX (PAIRS_28 from `scripts.anchor.run_anchor`) | dispatch front-matter |
| Base risk | 0.5% per trade (verdict at scaled `r_safe` / `r_hard` per Amendment 3) | dispatch front-matter |
| TF | 4H locked | dispatch front-matter |
| Direction | long only | spec |
| SL at Step 1 | 2.0 × ATR(14) at entry | dispatch §"Step 1" |
| Entry | next-bar open (`open_ask` per v3 worst-case fill) | dispatch §"Step 1" |
| Hold cap | 240 bars | v2.1.2 spec |
| Sub-protocol | vanilla | dispatch front-matter |
| **Boundary convention** | **`5ers_eet`** per PR #189 (was UTC pre-amendment) | dispatch §"Step 1" |
| Sizing convention | `reset_floor` (Amendment 3 scalability gate requires this) | L_PROTOCOL §3 |

## 6. L_PROTOCOL Step 1 feature space used as-is (canonical mid-price per PR #189)

Per dispatch §"Intent doc" item 6 — confirmed: the 27-feature v3.0 catalogue
emitted by `core.features.pipeline.compute_feature_matrix`. Per
`PROTOCOL_RUNTIME.md` §15.1 + PR #189, all price-derived features compute on
**mid OHLC = `(close_bid + close_ask) / 2`** by default. `spread_regime.*`
features read `spread_close` as a structural regime indicator (exempt by
design per §15.1). Signal evaluation itself stays bid-side per KH-24
convention (mid-feature refactor applies to features, not signals).

Multi-TF features (D1 slope sign/magnitude, D1 ATR percentile, W1 slope sign)
require `panel.aux["d1"]` / `panel.aux["w1"]` per
`core.features.multi_tf._d1_close_slope_sign` and similar. Driver will inject
via `object.__setattr__(panel_h4, "aux", {"d1": ..., "w1": ...})` since Panel
is frozen — same workaround as Arc 7 v3.0 first attempt. The engine
capability audit flags this as `compute_feature_matrix doesn't accept aux
panels` (not a blocker; documented workaround).

Lineage column-name mismatch (`lineage` vs `causal_lineage`) — engine
capability audit §"Lineage filter" calls this out. Driver renames at the
Step 4 call site (existing workaround pattern; same as Arc 7 v3.0 first
attempt). Flagged in closure §3 as remaining infra cleanup.

## 7. All six architectures available for archetype-matched selection at Step 5

Confirmed via canonical module presence:

| Architecture | Module | Status |
|---|---|---|
| A1 system_level_filter | `core/architectures/a1_system_level_filter.py` | WIRED |
| A2 classifier_filter | `core/architectures/a2_classifier_filter.py` + `core.steps.classifier_persistence.build_a2_config_from_step4` | WIRED (PR #185) |
| A3 pipeline_de | `core/architectures/a3_pipeline_de.py` + `core.steps.classifier_persistence.build_a3_config_from_step4` + `core.steps.path_classifier_per_fold` | WIRED (PR #186) |
| A4 pipeline_d_exits | `core/architectures/a4_pipeline_d_exits.py` + `build_a4_config_from_step4` + per-fold retrain + trail-first precedence | WIRED (PR #186) |
| A5 portfolio_composition | `core/architectures/a5_portfolio_composition.py` | WIRED (only if ≥ 2 candidate clusters; Arc 7 v3.0 first attempt had 2 candidates; likely runs) |
| A6 meta_labeling | `core/architectures/a6_meta_labeling.py` + `build_a6_config_from_step4` | WIRED (PR #185) |

Architecture selection per dispatch §"Step 5" archetype map, applied after
Step 3 surfaces archetype labels per cluster:

- V-shape recovery → A1, A3, A6
- Stepwise climber → A1, A2, A4
- Bimodal → A1, A4
- Monotonic up → A1, A2, A6
- Choppy → skip
- A5 if ≥ 2 candidate clusters

**Prior Arc 7 v3.0 found:** c0 Bimodal (n=613, AUC 0.676 ABOVE 0.65),
c1 Unclassified (n=1906, AUC 0.618), c2/c3 die at Step 3. Bimodal → A1+A4
(per map). Under v3.0.1: same archetype detection expected (path-shape
features identical; mid-price refactor + EET boundary may shift the pool
modestly, so cluster IDs / sizes may differ). v3.0.1 specifically tests
whether A2/A6 (now wireable per PR #185) on c0 — should the cluster recur —
unlocks the oracle headroom (6.85 worst-fold ratio under post-hoc cluster
admission).

**Search dimensions per dispatch §"Step 5":**
- SL: Step 3 selected SL ± 1 step from {1.5, 2.0, 2.5, 3.0, 3.5, 4.0}
- Exit policy: 3-4 archetype-matched; always include `sl_only`. Per
  Arc 7 v3.0 first attempt, the v3 A1/A2/A6 architecture Configs don't
  expose `tp_atr_mult` or partial-close mechanics — driver will test
  `trail_enabled ∈ {True (sl_plus_trailing_atr), False (sl_only)}` and
  document skipped exit-policy axis in `step_5/skipped_configs.md` per
  dispatch's "do not silently shrink" rule.
- Exposure cap: `max_concurrent_per_currency ∈ {2, unlimited}`
- A3: `n_defer ∈ {3, 5}` per `DEFAULT_N_DEFER_VALUES`
- A4: `exit_threshold ∈ {0.3, 0.4, 0.5}` per `DEFAULT_EXIT_THRESHOLDS`;
  trail-first precedence per PR #186 (locked in
  `core/sim/multipair_backtester.py:_process_bar` step 3)
- A6: threshold pairs `{(0.3,0.5), (0.4,0.6), (0.5,0.7)}` per `THRESHOLD_PAIRS`

**Oracle WFO:** per dispatch + Arc 10 v3.0 cross-arc lesson — lock oracle
exit policy to the WINNING architecture's exit policy. Driver computes this
post-Step-5 by reading the top-1 amended candidate's exit policy from
its config, then invokes `OracleFoldRunner` with `A1Config` matching that
exit policy + selected SL.

## 8. Step 6 framework will auto-dispatch on PASS

Confirmed via canonical module presence:

- `core/step_6/` framework — orchestrator + 6 categories + dispatch + io
- `core.step_6.dispatch.maybe_dispatch_step_6` — auto-dispatches on
  PASS-tier candidate clearing §3 constraints #1-9
- `core.step_6.dispatch.replace_top_1_with_step6_fail` — downgrades verdict
  to FAIL with `primary_failure_mode = step6_causal_audit_fail` on critical
  failure

Driver flow: after Step 5 + Amendment 3 evaluation produces an
`AmendedWfoSearchResult`, driver calls `maybe_dispatch_step_6(...)` with
the top-K candidates. If any candidate is PASS-tier, Step 6 dispatches
on Top-1 only (per Amendment 4 chat Q2). Results land at
`results/l_arc_7/step_6/`. Manual CLI (`scripts/run_step_6.py`) is NOT
invoked — auto-dispatch only per dispatch §"Step 6".

Step 6 framework expectations for Arc 7 LSR:
- **§6.1 Lookahead** — exercises the swing-low producer (per §3 above) +
  D1 lag rule + cluster-feature audit + byte-compare from raw OHLC +
  threshold-selection lineage. Swing-low producer is the highest-risk
  surface per the Arc 9 lesson (per dispatch).
- **§6.2 Selection bias** — verifies configs_evaluated count + Bonferroni-
  equivalent noise floor + holdout-reuse detector. Arc 7 v3.0 first
  attempt had thin config count (28); v3.0.1 will likely have more
  (4-6 archetype-matched architectures × 3 SL × 2-3 variants × 2 exposure
  per candidate cluster, ~50-100 configs — "normal" scope).
- **§6.3 Execution realism** — mid-price refactor active + UTC bar
  boundary (NOT EET — wait, dispatch says EET per PR #189; this audit
  category will likely flag if execution-realism check expects UTC vs EET
  mismatch — flagged for chat below as §G interpretive call).
- **§6.4 Statistical** — Lo-corrected Sharpe sample-size + 28-pair
  survivorship + vol-regime coverage + cross-pair daily-bucket correlation.
- **§6.5 Determinism** — sha256 verify + required artefacts + seed
  pinning + LF line endings. Driver enforces all four.
- **§6.6 Deployment readiness** — §4 deployment_spec heading +
  `config_artefact_path` resolvable + all §4.X subsections present. Driver
  emits these on PASS via closure-generator helper.

## 9. Determinism plan

Per L_PROTOCOL §1 non-negotiable + dispatch §"Intent doc" item 9:

- `random_state = 42` everywhere (per `core.determinism.RANDOM_STATE`)
- `n_jobs = 1` inside any per-row work (parallel pair compute via
  `core.parallel.parallel_pair_map` only — pair-level parallelism is
  byte-deterministic per PR-D's contract)
- `lineterminator = "\n"` on every text artefact via
  `core.determinism.write_text_deterministic` + explicit pandas kwarg
- `seed_everything(42)` at driver entry — sets numpy + python RNG +
  `PYTHONHASHSEED` for subprocess spawn
- Parquet caches:
  - M1 + H4 + D1 + W1 panels at `data/cache/{m1,H4_5ers_eet,D1_5ers_eet,W1_5ers_eet}/`
    per PR #189 (note `_5ers_eet` suffix is the boundary-convention
    discriminator — coexists with legacy UTC caches; no cross-pollination)
  - Step 4 fitted classifiers under SHA256-verified manifest per PR #185
  - Feature matrix at `data/cache/features/l_arc_7/<feature_set_hash>.parquet`
- Two-run sha256 reproducibility: tracked at `pool.pool_sha256` + Step 4
  classifier manifest sha256. Driver does NOT exercise two-run by default
  (compute cost prohibitive on full WFO); CC will optionally re-run Step 1
  + Step 2 only as a determinism spot-check, time permitting.

---

## Interpretive calls (flagged for chat — small)

These are operational defaults; chat may override at intent review.

### A. Two-pass orchestrator vs composition driver

CC defaults to **composition driver** (Path B in chat-side reasoning above):
runs canonical step functions sequentially, composes Amendment 3 evaluation
+ Step 6 dispatch from canonical primitives, no orchestrator re-entry.

Trade-off: orchestrator path (Path A) is cleaner narrative ("we used the
orchestrator end-to-end") but wastes ~10 min on Steps 1-3 redo + Step 4
classifier retrain. Composition driver saves the waste; identical canonical
implementations under the hood.

Override if chat prefers Path A.

### B. EET vs UTC boundary

Per dispatch §"Step 1": bar aggregation `EET broker timezone per PR #189`.
Driver uses `boundary_convention="5ers_eet"` in
`core.data.aggregator.aggregate(...)` and `build_panel_parallel(...)`.

Note: Arc 7 v3.0 first attempt used UTC (pre-PR-#189). EET vs UTC may
shift the pool — different bar boundaries → different signal_time
indices → different downstream metrics. v3.0.1 numbers are expected to
differ from v3.0 first-attempt numbers regardless of architecture-side
improvements.

Step 6 §6.3 execution-realism check explicitly verifies "UTC bar boundary"
per Amendment 4 §6.3 — this may fail under EET. If so, closure must
reconcile (Amendment 4's literal "UTC" may need amendment for EET arcs;
flag for chat resolution at Step 6 invocation time).

### C. A5 portfolio composition trigger

Per dispatch + L_PROTOCOL §2 Step 5 "A5 only if ≥ 2 candidate clusters
survive Step 3". Arc 7 v3.0 first attempt had 2 candidates (c0 + c1) — A5
likely runs here too if cluster IDs shift consistently. Driver constructs
A5Config from per-cluster best-architecture StrategyResults per
`A5Architecture.run` contract.

### D. `chained_dd_method` = `equity_stitching` (v3.0.1)

Per `PROTOCOL_RUNTIME.md` §8b + template v1.2.1: v3.0.1 default is
`equity_stitching` (multiplicative chaining with continuity adjustment).
v3.0.2 follow-up replaces with `full_window_sim` (per chat directive Q6).
Driver records `chained_dd_method: equity_stitching` in `§1 tracker_payload`
per template requirement.

### E. Step 6 manual CLI

NOT invoked per dispatch §"Step 6" — auto-dispatch only. If Step 6 critical-
fails, verdict downgrades to FAIL with `step6_causal_audit_fail` per
Amendment 4 §3 evaluation order item 4. Closure §1.step_6 block records the
result.

### F. Pre-existing Arc 7 v3.0 artefacts at `results/l_arc_7/`

`git reset --hard origin/main` discards `results/l_arc_7/` since it never
landed on main (the prior closure was on `arc/l_arc_7` only and was never
merged). Fresh start.

### G. Step 6 §6.3 UTC bar-boundary check vs EET deployment

`PROTOCOL_RUNTIME.md` §15.3 documents `boundary_convention` parameter +
EET cache coexistence with UTC. Amendment 4 §6.3 prose says "UTC bar
boundary" — likely written before PR #189 made EET an option. Step 6
execution-realism check may need clarification: does the check require UTC
specifically, or just that bar-boundary convention is consistent?

CC default: proceed with EET; if Step 6 §6.3 critical-fails on
bar-boundary mismatch, downgrade to FAIL per Amendment 4 and document
the EET-vs-UTC ambiguity in closure §3 as cross-arc tag
`step6_bar_boundary_eet_vs_utc_ambiguity` for chat resolution.

---

## End of intent

Per dispatch §"Intent doc": **end turn for chat review**. On continuation:
CC executes Steps 1→5 + Amendment 3 evaluation + Step 6 auto-dispatch
+ closure + tracker update + force-push + PR continuously, no inter-step
pauses, HALT only on genuine bugs/blockers per dispatch §"HALT triggers".
