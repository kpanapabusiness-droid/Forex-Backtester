# Arc 7 v3.0.2 — A2/A6 follow-up (architecture-map override) — execution log

**Branch:** `arc/l_arc_7_v3.0.2`
**Base:** `arc/l_arc_7 @ ba6c5b0` (the v3.0.1 commit — PR #180 not yet merged; see dispatch §0)
**Worktree:** `.claude/worktrees/clever-elbakyan-ad50ee/`
**Dispatch:** `docs/dispatches/arc_7_v3_0_2_intent.md`
**Closure:** `results/l_arc_7_v3.0.2/ARC_CLOSURE.md`

---

## Verdict

**FAIL** — primary failure mode `step5_not_scalable` (different cause than v3.0.1).

Top-1: `A6::A6::cl1::sl2.0::thr0.5-0.7::exp2` — worst-fold ratio **+4.363**, mean-fold ratio +8.703, 0 negative folds, chained max DD base **2.09%**, r_safe **3.16%** (above 2.0% prop cap).

---

## What was reused vs re-run

Reused from `results/l_arc_7/` (v3.0.1) verbatim:
- Step 1 pool (5,175 trades, 5ers EET, sha256 verified via v3.0.1 manifest)
- Step 2 cluster_assignments (K=4)
- Step 3 capturability (2 candidate clusters: c0 Bimodal, c1 Unclassified)
- Step 4 persisted classifiers (c0 LR AUC 0.6192, c1 RF AUC 0.6642) loaded via SHA256-verified `core.steps.classifier_persistence.load_classifier`

Re-run only Step 5+ with **architecture-map override**:
- c0 (AUC 0.6192 < 0.65): {A1, A4} — unchanged from v3.0.1
- c1 (AUC 0.6642 ≥ 0.65): {A1, **A2**, **A6**} — augmented per dispatch override
- A5: portfolio composition across both candidate clusters

**Step 5 grid: 45 configs** (v3.0.1 had 28 / +17 from A2(c1)+A6(c1)+A5).

---

## Execution timeline

| Phase | Wall time |
|---|---:|
| Step 1–4 artefact load (v3.0.1 reuse) | ~30s |
| Per-trade A2/A6 feature reconstruction (27 features × 28 pairs) | ~3min |
| Step 5 WFO (440 candidate × fold pairs, 17.1s/run avg) | ~125min |
| A5 portfolio composition | ~5s |
| Holdout (top-3) | ~4min |
| Amendment 3 evaluation | ~2min |
| Step 6 dispatch (no-op — verdict FAIL) | <1s |
| Oracle WFO per cluster | ~3min |
| Writes (CSVs, parquets, summary, manifest) | <5s |
| **Total wall** | **~139min (8366s)** |

Budget: `--wall-time-budget-h 5.5` (5h30m). Used: 2h19m. Margin: 3h11m.

---

## Incidents

**Incident 1 — driver crashed at line 1094 (run.py) on first attempt.** Python format-spec syntax error: `{ce.best_classifier_mean_auc:.4f if ce else 0:.4f}` — conditional inside format spec is not valid Python. Crash occurred *after* the entire expensive flow had completed (WFO + holdout + Amendment 3 + Step 6 dispatch + Oracle WFO + Step 5 CSV writes), specifically while building `wfo_summary.md`. Only `wfo_summary.md`, `skipped_configs.md`, `manifest.json`, and `run_summary.json` were missing on disk. Patch: restructure to `{(ce.best_classifier_mean_auc if ce else 0.0):.4f}` (evaluate conditional first, then format). Crashed artefacts preserved as `results/l_arc_7_v3.0.2/run.log.crashed` and `results/l_arc_7_v3.0.2/step_5.crashed/` for byte-identical-reproducibility audit on demand. Driver re-run produced clean outputs.

No other incidents.

---

## Cross-arc findings (full text in ARC_CLOSURE.md §3)

1. **Ratify the architecture-map override.** The rule "any cluster with E AUC ≥ 0.65 gains A2+A6 regardless of archetype" changed Arc 7's Step 5 surface materially even though the verdict letter was unchanged (v3.0.1 best ratio -0.82 → v3.0.2 best +4.36). Without the override, c1 (Unclassified, AUC 0.6642) would have produced thin-grid FAIL with no information. Recommend embedding as a permanent L_PROTOCOL Amendment 1 update.
2. **Amendment 3 "too clean to scale" gap.** v3.0.1's best A1 hit `step5_not_scalable` because DD was *too high* (r_safe 0.117% < 0.15% floor). v3.0.2's best A6 hit `step5_not_scalable` because DD was *too low* (r_safe 3.16% > 2.0% ceiling). First L-arc closure where a candidate fails the gate by being too risk-efficient. Open question for v3.0.x calibration: should fixed-risk-cap deployment (r=2% with k_realised below 8 budget headroom) be acceptable as PASS-VIABLE when scalability binds at the ceiling only?
3. **Deployable candidate one grid step away.** `A6::cl1::sl2.0::thr0.4-0.6::exp2` (rank 3) is `scalable_to_safe=true` (r_safe=1.43%, inside [0.15%, 2.0%]). Fails only on `step5_daily_dd_breach` (2 days breach 5% daily-DD). Finer threshold sweep or a `max_concurrent_total` cap could plausibly find a PASS-DEPLOYABLE config without changing architecture.
4. **Oracle-realised gap recurs (4th instance).** c0 oracle worst-fold ratio 11.90, c1 6.80; realised top-1 4.36. A6 closed roughly half the gap vs v3.0.1's A1 (-0.82 vs c1 oracle 11.90). Cross-arc clusterifier (Arc 10 v2.4 backlog) remains the standing follow-up.

---

## Definition-of-Done checklist (from dispatch §10)

| # | Item | Status |
|---|---|---|
| 1 | Step 5 ran under architecture-map override | ✅ c0 {A1,A4}; c1 {A1,A2,A6}; A5 |
| 2 | Step 5 → 6 → tracker per Amendment 4 framework | ✅ Step 6 not dispatched (no PASS) |
| 3 | Closure §1 tracker_payload populated | ✅ v1.3 YAML |
| 4 | Closure §2 explains why failed | ✅ direct v3.0.1↔v3.0.2 comparison table |
| 5 | Closure §2 compares to v3.0.1 | ✅ same letter, opposite cause; A2/A6 mechanism documented |
| 6 | Closure §3 documents override + amendment recommendation | ✅ ratify recommendation conditional on cross-arc, calibration gap flagged |
| 7 | Tracker updated via parser | ✅ `scripts/update_tracker_from_closure.py` |
| 8 | Step 1-4 artefacts reused, not re-run | ✅ load via `_load_v301`; SHA256 verification clean |
| 9 | Dispatch log written | ✅ this file |
| 10 | PR opened with verdict in title | ⏳ next step |
