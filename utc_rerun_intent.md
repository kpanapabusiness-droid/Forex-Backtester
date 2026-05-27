# Intent — Arc 10 v3.0.2 UTC Convention Rerun

> **Dispatch:** `phase/arc_10_v3_0_2_utc_rerun` (commit history will land this file first, then compute).
> **Anchor:** tag `arc-10-v3.0.2-DEPLOYABLE` = commit `244fb76`. Working tree clean; `HEAD == tag`.
> **Branch cut:** `phase/arc_10_v3_0_2_utc_rerun` created from `244fb76` on 2026-05-27.
> **Status of this doc:** PRE-COMPUTE. Listed file paths and parameter values are the contract for the rerun; nothing outside this list will be touched.

---

## §1 Goal

Re-run Arc 10 v3.0.2's signal + winning config **under `boundary_convention="utc"` instead of `"5ers_eet"`**, leaving every other parameter byte-identical to the locked v3.0.2 configs. Compare the resulting WFO numbers to the v3.0.2 EET closure baseline. **Not a reproducibility test** — the convention switch is expected to produce different numbers; the question is whether the verdict survives.

The §"Why" premise (dispatch): live 5ers MT5 publishes UTC-anchored H4 bars natively; the v3.0.2 verdict was earned on EET-aggregated bars. The rerun characterises convention sensitivity for the validated edge.

---

## §2 Anchor verification

`HEAD` and `arc-10-v3.0.2-DEPLOYABLE` both resolve to `244fb763a8ecffd45d9da4eafabf50caff7bc468`. `git status --short` is empty. Working-tree sha256 of the four anchor files:

| Path | sha256 |
|---|---|
| `signals/lchar_dlr_long.py` | `a68406c8dc3c860b14a57d1360836d5c7cd8a0e304fe45a55c5b8fa4cb82896d` |
| `core/sim/exit_policies/sl_partial_close_1r_runner_trail.py` | `9abec19c10753237deba4379c38e4f22f38cafbac1a52456c79177b268abff6d` |
| `configs/l_arc_10_v3.0.2/winning_config.yaml` | `09ed7ce1606b925389d5b51fb60bc56e8cf001c97d26eb130e8ff71d1a71c4ec` |
| `configs/l_arc_10_v3.0.2/arc_open.yaml` | `d50525fd0450d087b117eb155e0d4c5549c4ef91db6526ffe9f7eff00466bf34` |

(All four are at their HEAD-tracked git blob OIDs since the tree is clean at tag-commit; the sha256 values above will be re-verified at the head of `COMPARISON_REPORT.md`.)

---

## §3 Pipeline reality check — full Steps 1→5 are required

The dispatch §1.3 single command (`py -m scripts.l_arc_10_v3.step_5 -c <override> --out <dir>`) is insufficient. Two findings from reading the scripts:

1. **`step_5.py` reads upstream artefacts from Steps 1-4.** Specifically (paths derived from `cfg["output"]["results_dir"]`):
   - `<results_dir>/pool.parquet` + `<results_dir>/trade_paths.parquet` (Step 1)
   - `<arc_root>/step_2/cluster_assignments.parquet`
   - `<arc_root>/step_3/capturability.csv`
   - `<arc_root>/step_4/manifest.json`

   Under UTC convention, every one of those upstream artefacts is different from EET — pool composition shifts (D1-alignment differs), features differ, clusters differ, classifiers differ. Running only Step 5 against EET upstream artefacts would either crash (file-not-found at the UTC results_dir) or — if I redirected results_dir at Step 5 but not upstream — produce nonsense (Step 5 evaluating UTC search_pool against EET cluster-assignment labels).

2. **The dispatch's `--out` flag does not exist.** `step_5.py main()` accepts only `-c/--config`; the output directory is derived from the config's `output.results_dir` field. The override config will therefore own the output routing.

**Action:** run the full Steps 1→5 pipeline via the standard sequence (Step 1 standalone, then `run_all.py` chains Steps 2-5). All five steps consume the same override config and produce siblings under `results/l_arc_10_v3_0_2_utc_rerun/`. This matches the dispatch's intent (the wall-clock estimate of 6-12 hours is consistent with a full Steps 1→5 run; Step 5 alone is fast).

This is a substantive divergence from the dispatch's literal command, surfaced before running per dispatch §1.3 ("If the orchestrator does not accept the override config path cleanly, surface the integration question before running — do not patch silently"). Proceeding under the assumption that "v3.0.2's signal + winning config under UTC" requires regenerating the upstream pool/clusters/features under UTC, since the winning config is identified by Step 5's search over Step 1-4 outputs.

---

## §4 Files CC will touch

### §4.1 New files (created by this branch)

| Path | Purpose |
|---|---|
| `utc_rerun_intent.md` | This doc. |
| `configs/l_arc_10_v3.0.2_utc_rerun/arc_open.yaml` | Sole-field override of `configs/l_arc_10_v3.0.2/arc_open.yaml`. |
| `configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml` | Sole-field override of `configs/l_arc_10_v3.0.2/winning_config.yaml` (provenance artefact; not consumed by the pipeline run itself — kept for symmetry with the v3.0.2 deployment-artefact convention). |
| `results/l_arc_10_v3_0_2_utc_rerun/step_1/{pool.parquet, trade_paths.parquet, integrity_report.md, manifest.json}` | Step 1 outputs. |
| `results/l_arc_10_v3_0_2_utc_rerun/step_2/{cluster_assignments.parquet, ...}` | Step 2 outputs. |
| `results/l_arc_10_v3_0_2_utc_rerun/step_3/{capturability.csv, ...}` | Step 3 outputs. |
| `results/l_arc_10_v3_0_2_utc_rerun/step_4/{manifest.json, ...}` | Step 4 outputs. |
| `results/l_arc_10_v3_0_2_utc_rerun/step_5/{wfo_results.csv, wfo_oracle.csv, architectures_ranked.md, best_candidate.md, manifest.json}` | Step 5 outputs. |
| `results/l_arc_10_v3_0_2_utc_rerun/trade_ledger_utc.parquet` | Per-trade UTC ledger (admitted Top-1 config trades; see §7). |
| `results/l_arc_10_v3_0_2_utc_rerun/trade_matching.csv` | EET-vs-UTC trade-level diff (see §7). |
| `results/l_arc_10_v3_0_2_utc_rerun/COMPARISON_REPORT.md` | The deliverable comparison artefact. |

### §4.2 Files CC will NOT touch (under any circumstance)

- Any file under `configs/l_arc_10_v3.0.2/` (locked — STOP condition per dispatch §5).
- Any file under `core/`, `signals/`, `scripts/l_arc_10_v3/`, `scripts/l_arc_10_v3_0_2/`.
- Any file under `results/l_arc_10/`, `results/l_arc_10_v3.0.2/`, or any other arc-results directory.
- `L_PROTOCOL.md`, `TODO.md`, `ARC_TRACKER.md`, `WORKFLOW.md`, `CLAUDE.md`, `docs/`.
- `data/histdata/`, the M1 source-of-truth.

### §4.3 Worktree-local infrastructure (gitignored)

- `data/cache` Windows directory junction → `C:/Users/panap/Documents/Forex-Backtester/data/cache` (main-repo cache). Required so the aggregator can read pre-built parquet caches (UTC: `<cache_root>/H4/<PAIR>.parquet`, `D1/`, `W1/`) instead of re-aggregating from M1. The main-repo's UTC cache directories (`H4`, `D1`, `W1`) are confirmed present (listed `AUDCAD.parquet ...`). Junction creation is the worktree-local convention used by the v3.0.2 addendum (closure §11). NOT committed.

If the junction step fails or the UTC cache directories are unexpectedly empty for a pair, surface and STOP — do not silently re-aggregate from M1 (a full M1→H4 rebuild across 28 pairs × ~16 years would add many hours of wall-clock not in the dispatch budget).

---

## §5 Parameter overrides

### §5.1 `configs/l_arc_10_v3.0.2_utc_rerun/arc_open.yaml`

Byte-identical copy of `configs/l_arc_10_v3.0.2/arc_open.yaml` with exactly **two** changes:

| Field | v3.0.2 (locked) | UTC rerun (override) | Reason |
|---|---|---|---|
| `boundary_convention` | `"5ers_eet"` | `"utc"` | The single substantive override the dispatch authorises. |
| `output.results_dir` | `results/l_arc_10_v3.0.2/step_1` | `results/l_arc_10_v3_0_2_utc_rerun/step_1` | Required to redirect Step 1-5 outputs away from the locked v3.0.2 results directory (otherwise Step 1 would clobber `results/l_arc_10_v3.0.2/step_1/pool.parquet` — a STOP condition per dispatch's "do not modify locked v3.0.2 artefacts"). The output redirect is mechanically necessary; it is not a methodology change. |

All other fields preserved byte-for-byte: `arc_name`, `protocol_version`, `sub_protocol`, `data`, `window`, `tf`, `pairs` (28 pairs), `risk_per_trade` (0.005), full `signal` block (module + every locked constant), full `step_1` block (SL/exit/exposure/features/integrity gates).

Note on `arc_name`: I will leave `arc_name: l_arc_10_v3.0.2` unchanged in the override config — the manifest's `arc_name` field is for provenance display; it does not drive any path resolution (the script reads from `output.results_dir`, and the manifest is hardcoded to `arc_name="l_arc_10"` in `step_5.py:839`). Changing `arc_name` would be a no-op gesture; leaving it preserves the truth that we are rerunning the v3.0.2 spec under a different convention.

### §5.2 `configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml`

Byte-identical copy of `configs/l_arc_10_v3.0.2/winning_config.yaml` with exactly **one** change:

| Field | v3.0.2 (locked) | UTC rerun (override) | Reason |
|---|---|---|---|
| `boundary_convention` | `"5ers_eet"` | `"utc"` | The single substantive override the dispatch authorises. |

This file is not consumed by `scripts/l_arc_10_v3/*.py` — it's a deployment-spec artefact, the symmetric companion to `arc_open.yaml`. It will be kept in sync per dispatch §4 deliverables. I do **not** override its `arc_name` or `provenance` fields; the `verdict: PASS-DEPLOYABLE` field is the *v3.0.2 EET verdict* — preserving it is correct (this rerun does not re-issue the verdict; §3 of COMPARISON_REPORT will compute a separate UTC verdict per dispatch §3 framework).

### §5.3 Diff verification

After creating the overrides, I will `git diff --no-index configs/l_arc_10_v3.0.2/arc_open.yaml configs/l_arc_10_v3.0.2_utc_rerun/arc_open.yaml` and confirm exactly the two lines (`boundary_convention`, `output.results_dir`) differ in the arc_open override, and exactly one line (`boundary_convention`) differs in the winning_config override. Diff output goes in COMPARISON_REPORT §appendix.

---

## §6 Run plan

### §6.1 Steps 1→5 invocation

```powershell
# Step 1 (separate — cache hits + signal scan + path sim across 28 pairs)
py -m scripts.l_arc_10_v3.step_1 -c configs/l_arc_10_v3.0.2_utc_rerun/arc_open.yaml

# Steps 2-5 chained via run_all
py -m scripts.l_arc_10_v3.run_all -c configs/l_arc_10_v3.0.2_utc_rerun/arc_open.yaml
```

Determinism per protocol: `random_state=42`, `n_jobs=1`, `lineterminator="\n"` — all enforced in-code by `core.determinism.seed_everything(RANDOM_STATE)` called at the head of each step.

### §6.2 Expected wall-clock

- Step 1: ~30-60 min (cache hits on H4/D1/W1; ~28k DLR triggers; trade-path simulation across 28 pairs × 16 years).
- Steps 2-4: ~10-30 min combined.
- Step 5: ~30-90 min (single candidate cluster after Amendment 5 strips A6 — though the UTC cluster-AUC may differ and admit different architectures; this is one of the outcomes we want to observe).
- **Total: ~1.5-3 hours wall-clock** (substantially below the dispatch's 6-12h estimate; the estimate appears to assume cold M1→H4 cache rebuild, which the junction avoids).

If actual runtime exceeds 12h: STOP per dispatch §5 wall-clock condition.

### §6.3 Run-in-background pattern

Step 1 will be backgrounded with the Bash tool's `run_in_background` parameter; status checked via Monitor. Same for `run_all` once Step 1 completes. Manifest sha256s confirmed after each step (Step 1 writes `manifest.json` with input/output hashes; Steps 2-5 likewise).

### §6.4 What is NOT in the run plan

- No grid sweep beyond what `step_5.py` natively does (sl_range ±0.5 around per-cluster best_sl, all archetype-mapped exit policies, both exposure caps, all architectures eligible after Amendment 5 AUC gate).
- No re-running on any other engine version or any other commit.
- No skip of any fold.
- No early stopping.
- No interactive parameter adjustment.

---

## §7 Comparison plan (post-WFO)

### §7.1 Baselines

**Primary (v3.0.2 EET, the actual comparison target):**

Verified from `results/l_arc_10_v3.0.2/step_5/wfo_results.csv` Top-1 row + `results/l_arc_10_v3.0.2/step_5/amendment_3/amended_gate_classification.json` per-fold table + `architectures_ranked.md` holdout block:

| Metric | EET value |
|---|---|
| Top-1 config | A1 / SL=3.5 / sl_partial_close_1r_runner_trail / unlimited / cluster c0 v_shape_recovery |
| Search worst-fold ROI | 0.22456 (22.456%) — F9 |
| Search worst-fold DD | 0.07354 (7.354%) — F4 |
| Search worst-fold ratio | 6.4273 — F6 |
| Search mean ROI | 0.49882 (49.882%) |
| Search mean DD | 0.03493 (3.493%) |
| Search mean ratio | 16.237 |
| Sign consistency | 11 / 11 |
| Search total trades | 2,059 |
| Holdout ROI (r_base, single one-shot fold) | 0.52831 (52.831%) |
| Holdout DD (r_base) | 0.05499 (5.499%) |
| Holdout ratio | 9.607 |
| Holdout trades | 1,093 |
| Daily-DD breaches (r_base, all 11 IS + holdout) | 0 |
| Amendment 3 k_safe | 1.0879 → r_safe 0.5439% |
| Amendment 3 k_hard | 1.3598 → r_hard 0.6799% |

Per-fold IS table (from amended_gate_classification.json, r_base):

| Fold | n_trades | ROI | DD | ratio | breaches |
|---|---|---|---|---|---|
| F1 | 201 | 0.40172 | 0.03235 | 12.418 | 0 |
| F2 | 182 | 0.44975 | 0.02963 | 15.180 | 0 |
| F3 | 179 | 0.52130 | 0.02028 | 25.706 | 0 |
| F4 | 195 | 0.72323 | 0.07354 | 9.835 | 0 |
| F5 | 192 | 0.51969 | 0.02405 | 21.608 | 0 |
| F6 | 190 | 0.32331 | 0.05030 | 6.427 | 0 |
| F7 | 170 | 0.46983 | 0.03248 | 14.466 | 0 |
| F8 | 191 | 0.73656 | 0.02460 | 29.945 | 0 |
| F9 | 175 | 0.22456 | 0.03051 | 7.361 | 0 |
| F10 | 196 | 0.53284 | 0.04034 | 13.207 | 0 |
| F11 | 188 | 0.58424 | 0.02577 | 22.672 | 0 |

**Secondary (v3.0 UTC sanity reference):**

From `results/l_arc_10/step_5/wfo_results.csv` Top-1 + `best_candidate.md`:

| Metric | v3.0 UTC value |
|---|---|
| Top-1 config | A1 / SL=3.5 / sl_partial_close_1r_runner_trail / unlimited / cluster c1 v_shape_recovery |
| Search worst-fold ROI | 0.26491 (26.491%) |
| Search worst-fold DD | 0.09224 (9.224%) |
| Search worst-fold ratio | 5.4185 |
| Search mean ROI | 0.49865 (49.865%) |
| Sign consistency | 11 / 11 |
| Search total trades | 2,162 |
| Holdout ROI (per closure §10) | 59.07% |
| Holdout DD | 5.03% |
| Holdout ratio | 11.73 |

(Cluster id label is c0 in v3.0.2 vs c1 in v3.0 — clustering is order-dependent and the labels are nominal; both are the v_shape_recovery archetype. Sanity check: v3.0.2 UTC rerun expected to land near v3.0 UTC numbers since Amendment 5 + plumbing PRs do not affect A1's signal+SL+exit path materially.)

### §7.2 Comparison artefacts

| Artefact | Content |
|---|---|
| `trade_ledger_utc.parquet` | UTC rerun's admitted trades for the Top-1 config, columns: `trade_id, pair, signal_bar_time, entry_time, exit_time, entry_price, sl_price, exit_price, final_r, bars_held, exit_reason`. |
| `trade_matching.csv` | Row per EET trade. Columns: `eet_trade_id, eet_pair, eet_signal_bar_time, eet_final_r, match_kind ∈ {exact, near_1bar, none}, utc_trade_id, utc_signal_bar_time, utc_final_r, bar_offset, r_delta`. "Near" = same pair, signal_bar_time within ±1 H4 bar (±4h). Plus EET-side-unmatched UTC trades appended at the bottom (`eet_trade_id = NaN`). |
| `COMPARISON_REPORT.md` | The deliverable. Sections: §1 anchor verification (re-state sha256s); §2 per-fold table (EET vs UTC, ΔROI / ΔDD / Δtrades); §3 aggregate metrics table; §4 Amendment 3 evaluation under UTC (compute k_safe / k_hard / r_safe / r_hard / scalable verdicts using `core.wfo.amended_gates.classify_amended_fold_stats`); §5 trade-level matching summary (exact / near / none rates overall + per-pair); §6 sanity vs v3.0 UTC reference (delta on each headline metric, with an explicit pass/fail note on the §5 dispatch stop condition "v3.0.2 UTC rerun differs >5pp on worst-fold ROI from v3.0 UTC reference"); §7 verdict per dispatch §3 framework (UTC worst-fold ratio band + hard gates). |

### §7.3 Amendment 3 evaluation under UTC

Compute r_safe / r_hard from the UTC rerun's chained DD by re-using the existing Amendment 3 primitive `core.wfo.amended_gates.classify_amended_fold_stats` (the same primitive the v3.0.2 addendum used). Inputs:
- Per-fold OOS equity curves from the Step 5 Top-1 config (need to re-instrument `_wfo_run` for equity-curve export, OR reuse the closure §11 pattern of consuming `wfo_results.csv` + replaying via `core.sim.exit_policies.simulate_path` against the UTC pool / trade_paths).

Pattern follows `scripts/l_arc_10_v3_0_2/amendment_3_addendum.py` (the closure addendum that did exactly this for EET — I'll mirror that script for UTC against the UTC artefacts). New script: `scripts/l_arc_10_v3_0_2_utc_rerun/amendment_3_addendum.py` (sibling directory; addendum scripts are not in the dispatch's NO-TOUCH list).

Wait — adding a new script under `scripts/` is technically a file CC will touch beyond §4.1. I will write it as a small inline script in `results/l_arc_10_v3_0_2_utc_rerun/` (path `results/l_arc_10_v3_0_2_utc_rerun/amendment_3_addendum.py`) to keep it scoped to the rerun's results directory and avoid polluting `scripts/`. If the user prefers it under `scripts/`, that's a small move; flagging here.

### §7.4 Verdict framework (dispatch §3)

| UTC worst-fold ratio | Verdict |
|---|---|
| ≥ 5.0 | PASS-DEPLOYABLE under UTC |
| 3.0 – 5.0 | PASS-DEPLOYABLE under UTC if hard gates hold |
| 2.0 – 3.0 | MARGINAL — escalate |
| < 2.0 | FAIL under UTC — sidecar must aggregate M1→EET internally |

Hard gates (all must hold for PASS-DEPLOYABLE-UTC):
- Worst-fold DD ≤ 9.5%
- Sign consistency ≥ 10/11 IS folds positive
- Holdout positive ROI
- Amendment 3 r_safe (intrinsic) > 0.3%

**Hypothesis** (for orientation, not a gate): UTC rerun lands near v3.0 UTC numbers (~26% ROI / ~9.2% DD / ~5.4 ratio), since v3.0.2 vs v3.0 differs by Amendment 5 architecture-skip + plumbing PRs that don't affect A1's signal+SL+exit path. Under that hypothesis, the rerun clears PASS-DEPLOYABLE in the 5.0+ band with margin on hard gates (9.2% DD is just under the 9.5% threshold but within tolerance; sign 11/11; holdout 59% > 0; Amendment 3 should give k_safe ≈ 0.87 → r_safe ≈ 0.43% > 0.3%).

If the rerun lands materially *worse* than v3.0 UTC (>5pp ROI delta worst-fold), that triggers the dispatch §5 STOP condition "v3.0.2 UTC rerun differs materially from v3.0 UTC reference — surface for investigation".

---

## §8 Stop conditions (echoed from dispatch §5)

- Anchor checksum mismatch — STOP.
- Override config requires modifying any locked file under `configs/l_arc_10_v3.0.2/` — STOP. (Mitigated by sidecar dir per §4.1.)
- Engine error during WFO that suggests code drift since anchor — STOP.
- UTC rerun matches EET numbers exactly — STOP (engine bug: convention switch didn't propagate).
- v3.0.2 UTC rerun differs >5pp on worst-fold ROI from v3.0 UTC reference — STOP, surface for investigation.
- Hard gates in §3 fail — STOP.
- Wall-clock exceeds 24h — STOP.
- Cache junction setup fails or main-repo UTC cache missing pairs — STOP (do not silently re-aggregate M1).

Escalation: write `results/l_arc_10_v3_0_2_utc_rerun/ESCALATION.md`, surface to user, do not proceed.

---

## §9 Deliverables (echoed from dispatch §4)

- [x] `utc_rerun_intent.md` (this doc; commit immediately, before any compute)
- [ ] `configs/l_arc_10_v3.0.2_utc_rerun/arc_open.yaml` (sole-field override + output redirect)
- [ ] `configs/l_arc_10_v3.0.2_utc_rerun/winning_config.yaml` (sole-field override)
- [ ] `results/l_arc_10_v3_0_2_utc_rerun/step_{1..5}/` — full pipeline outputs
- [ ] `results/l_arc_10_v3_0_2_utc_rerun/COMPARISON_REPORT.md`
- [ ] `results/l_arc_10_v3_0_2_utc_rerun/trade_ledger_utc.parquet`
- [ ] `results/l_arc_10_v3_0_2_utc_rerun/trade_matching.csv`
- [ ] (small addendum script for Amendment-3-under-UTC; see §7.3)
- [ ] PR opened against `main` with verdict in description

---

## §10 Out of scope (echoed from dispatch §6)

- Modifying signal code or exit policy code.
- Modifying any file under `configs/l_arc_10_v3.0.2/`.
- Re-running on 5ers data (separate phase branch).
- Grid searches over any parameter.
- Audit re-execution.

---

End of intent.
