# Arc 11 v3.0.2 — W1 Producer Contamination HALT

Triggered by cross-arc HALT signal from Arc 10 v3.0.2 audit (commit 5bc149a — `_w1_close_slope_sign` in `core/features/multi_tf.py` uses `merge_asof(direction="backward", allow_exact_matches=False)` against week-START-labelled W1 bars, leaking the same week's Sunday close at every H4 timestamp Mon-Sat).

## Current state at HALT

- **Steps completed:** Step 1 (pool n=17,281; pool_sha256=`4872a0b6e2fb5ad5...`), Step 2 (K=4 selected, silhouettes {2: 0.397, 3: 0.396, 4: 0.494, 5: 0.476, 6: 0.456}), Step 3 (candidate_ids=(0, 1) — c0 Bimodal + c1 Unclassified), Step 4 (c0 best=LR AUC=0.6339; c1 best=RF AUC=0.6357; classifiers persisted to `step_4/classifiers/{0,1}.pkl`)
- **Step currently running at HALT:** Step 5 — ArcOrchestrator.run() started at 16:01:35; in WFO search phase across 54 configs (A1 grid 18 + A4 c0 auto_arch_specs 36) when HALT signal received at ~16:25
- **Process state:** background task `bgkear49h` killed via `TaskStop`; remaining Python processes are unrelated (other terminals/Claude Code).
- **Branch state:** `arc/l_arc_11_v3.0.2` at commit `9e54a478f584084039b370821c11586236a806fe` (= origin/main HEAD after Amendment 5.1 merge).

## Contamination scope for this arc

- **Was `_w1_close_slope_sign` consumed by the feature pipeline executed so far?** **YES.** Arc 11 v3.0.2 ran `core.features.pipeline.compute_feature_matrix` with the canonical 27-feature v3.0 default envelope. `_w1_close_slope_sign` is one of those 27 features. Arc 11 v3.0's closure §4.3 explicitly lists `w1_close_slope_sign` as feature #1 in `features_in_winning_config` for the A2 winner. The producer is in the active feature set both at Step 4 (cluster-membership classifier training) and via the run_context per-trade-feature lookup (for hypothetical A2/A6 admit gates — not exercised under v3.0.2 since both AUCs < 0.65, but still flowed through the pipeline).

- **Which step artefacts are contaminated?**

| Artefact | Status |
|---|---|
| `step_1/pool.parquet` | clean (pool itself is signal-mask-only; no W1 feature consumed in pool construction) |
| `step_1/paths.parquet` | clean (forward-window R-multiples derived from H4 OHLC only) |
| `step_1/integrity_report.md` | clean |
| `step_1/manifest.json` | clean |
| Feature matrix (in-memory, never persisted to disk) | **contaminated** — `w1_close_slope_sign` is one of the 27 columns |
| `step_4/classifiers/0.pkl` (c0 Bimodal, LR, AUC 0.6339) | **contaminated** — trained on lineage-filtered subset of the 27-feature matrix; W1 column was one of the inputs (lineage rule did not exclude it, since `_w1_close_slope_sign` is tagged `clean` per the producer's CausalLineage declaration despite carrying the lookahead bug) |
| `step_4/classifiers/1.pkl` (c1 Unclassified, RF, AUC 0.6357) | **contaminated** — same |
| `step_4/classifiers/manifest.json` | **contaminated** (SHA256 of the contaminated pickles + their feature_order include `w1_close_slope_sign`) |
| `amendment_5_admission_probe.json` | **contaminated** — Step 4 AUCs and best-classifier choices both derived from the contaminated feature matrix |
| `admission_plan.json` | downstream of contamination but methodologically still records the Amendment 5/5.1 admission decision; whether it changes post-fix depends on whether removing W1 shifts c0/c1 AUCs across the 0.65 Gate-2 threshold |
| `step_2/*`, `step_3/*` | not yet emitted to disk in this run (orchestrator hadn't reached its end-of-run `write()` call) |
| `step_5/*`, `step_6/*` | not yet emitted (Step 5 search was still in progress when HALT fired) |

**Causal lineage caveat:** the producer carries a `clean` lineage tag in `core/features/lineage.py`. The contamination is a producer-internal mis-alignment, not a lineage-tag mis-declaration — so the v3.0.2 Step 4 lineage-filter (PR #185 column-name reconciliation) did not exclude it. Fix landing on engine PR `engine/w1_producer_canonical_alignment` is producer-level; the lineage tag stays `clean` once the producer is canonical.

**Cross-arc implication for the v3.0.2 verdict prior:** Arc 11 v3.0 closure §4.3 listed `w1_close_slope_sign` as feature #1 in the winning A2 config's top-10 features under permutation importance — meaning the contaminated producer was load-bearing in v3.0 too (v3.0 ran on UTC bars; the convention-independence of the bug means it leaked under v3.0 as well). The Arc 11 v3.0 FAIL verdict was reached against contaminated AUCs — this does not change v3.0's FAIL determination (worst-fold ratio negative, multi-gate failure), but it does mean v3.0's `step4_auc_above_065_v3_first` cross-arc tag is methodologically suspect and may not survive the post-fix re-evaluation.

## Pending fix

- Tracking via PR `engine/w1_producer_canonical_alignment` (fresh CC chat per cross-arc HALT signal §1).
- Will resume after fix lands on main + this branch rebases (or merges) main + re-executes from Step 1 (full pipeline rebuild required — feature matrix is in-memory only, must be recomputed).

## Action this turn

- **No compute** — orchestrator killed mid-Step-5.
- **No PR** — branch state preserved for chat review and post-fix resume.
- **Branch pushed at commit `9e54a47` + this HALT diagnostic** for chat review.
- **End turn** after push. Await resume signal.

## Notes for resume

- v3.0.2 admission set under contaminated AUCs was: `A1` (Gate 3 universal, 18 configs) + `A4 c0` (Gate 1 Bimodal, 36 configs) = 54 configs total. Post-fix, the admission set may shift: if c0 AUC moves above 0.65 (currently 0.6339 — only 0.0161 below), A2 + A6 admit per Gate 2 for c0; if c1 AUC moves above 0.65 (currently 0.6357), A2 + A6 admit for c1. Either or both could change the architecture-tested set.
- The cross-arc tag `architectures_skipped_by_amendment_5` planned for v3.0.2 closure was `['a5_gate_4_admission_blocked_by_no_pass_tier_constituent', 'A2', 'A6']` under contaminated AUCs; final list pending post-fix re-derivation.
- Driver script at `scripts/l_arc_11_v3_0_2/run.py` is functional and deterministic; intent doc at `docs/dispatches/arc_11_v3_0_2_intent.md`; closure writer skeleton at `scripts/l_arc_11_v3_0_2/write_closure.py`. All three preserved on branch.
- Pre-fix run.log + amendment_5_admission_probe.json + admission_plan.json + step_1 artefacts + step_4 classifier pickles preserved under `results/l_arc_11_v3.0.2/` for post-fix delta comparison (per chat discretion at resume time).
