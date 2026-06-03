# Arc 8 v3.0.2 — Intent Doc (CORRECTED retry under canonical 5ers_eet)

> **Dispatch:** `DISPATCH_arc_8_v3_0_2_CORRECTED.md` — Arc 8 v3.0.2 final retry under canonical `boundary_convention="5ers_eet"` engine.
> **Worktree:** `C:/Users/panap/Documents/Forex-Backtester/.claude/worktrees/focused-knuth-93d932`
> **Branch:** `arc/l_arc_8_v3.0.2` — FRESH, cut from `origin/main` at `dd391cd`, pushed (`git push -u origin arc/l_arc_8_v3.0.2` ✓).
> **Prior preserved:** `arc/l_arc_8_v3.0.2_halted` at `aa641dc` (local + remote) — UTC run + verification HALT + Step 4 re-resolution work, kept for historical reference; not consumed by this retry.
> **Protocol:** `L_PROTOCOL.md` v3.0 + Amendments 1–6 + 5.1. Sub-protocol `vanilla`.
> **Status:** plumbing-only intent. No Step 1 compute yet. End-turn after this doc per corrected dispatch §4.

---

## §0 Why this retry exists (succinct)

Two correctable defects in the prior `arc/l_arc_8_v3.0.2` (now `_halted`) make its outputs invalid for any 5ers-deployment interpretation:

1. **Convention mismatch.** The prior run used `boundary_convention="utc"` end-to-end, justified at the time as "v3 default for KH-24 anchor parity." That justification is incorrect under Amendment 6: `5ers_eet` is canonical for any arc whose target deployment is 5ers MT5. The prior 0.6938 c2 V-shape Step 4 mean OOS AUC is **not** a baseline; it is a UTC-convention result with no direct comparability to a 5ers_eet run.
2. **W1 producer lookahead (separately fixed pre-retry).** The prior run's pre-Step-5 verification audit (`docs/dispatches/arc_8_v3_0_2_verification.md` on `_halted`) confirmed `_w1_close_slope_sign` was using raw `pd.merge_asof(direction="backward", allow_exact_matches=False)` instead of the canonical PR-#193 `get_htf_value_at(..., require_fully_closed=True)`. W1 bars are labelled at start-of-week → the producer matched the current week's bar → its `close_bid`/`close_ask` was the FUTURE Sunday close. Sign-conditional V-shape rate 5.6% vs 30.2% (5.4× lift); permutation importance #1 at 28× the next multi-TF feature. Fixed on `origin/main` via PR #208 (commit `8ce3b3d`, merged 2026-05-25 — see §7 pre-flight receipt).

Defect (1) and defect (2) are independent: the W1 lookahead would have inflated AUC under either convention. The corrected retry runs against a clean engine (W1 producer canonical) under the canonical convention (5ers_eet end-to-end). **Expectation: the v3.0.2 5ers_eet AUC may be higher, lower, or near Arc 8 original's 0.530 — we do not assume any particular drift.** The 0.6938 prior is treated as informational only; it is not a reference value.

---

## §1 File paths to be touched

Working directory: this worktree's repo root. All paths repo-relative.

### Created (Step 1 → closure)

- `scripts/l_arc_8_v3_0_2/` — per-arc driver scripts (mirrors Arc 8 original `scripts/l_arc_8/`):
  - `shared.py` — pair set, window, panel construction with `boundary_convention="5ers_eet"`, signal-module instantiation
  - `step_1_pool.py` — pool builder (PR-HHHL long, 28 pairs, 2010-01-01 → 2025-12-31)
  - `step_2_clustering.py` — KMeans K-sweep + shape-tag assignment
  - `step_3_capturability.py` — per-cluster reach/MFE/ww_pp + composite
  - `step_4_extraction.py` — RF + LGBM + LR with `train_end=2021-01-01T00:00:00Z` (PR #185 IS-only CV)
  - `step_5_wfo.py` — 11-fold WFO + holdout via `ArcOrchestrator`; Amendment 3 emission automatic
- `configs/l_arc_8_v3_0_2/winning_config.yaml` — populated at closure from `step_5/best_candidate.md`
- `results/l_arc_8_v3.0.2/` — all step artefacts (clean directory; no files copied from prior run):
  - `ARC_OPEN.md` — opened at start of Step 1
  - `step_1/{pool.parquet, paths.parquet, features.parquet, feature_lineage.csv, integrity_report.{md,json}, manifest.json, step_1_run.log}`
  - `step_2/{cluster_assignments.parquet, cluster_metrics.csv, cluster_summary.md, path_features.parquet, silhouette_sweep.csv, manifest.json}`
  - `step_3/{capturability.csv, capturability_summary.md, sl_sweep.csv, manifest.json}`
  - `step_4/{extraction_metrics.csv, feature_importance.csv, extraction_summary.md, classifiers/<cid>.pkl, classifiers/manifest.json, manifest.json}`
  - `step_5/{wfo_results.csv, wfo_oracle.csv, architectures_ranked.md, best_candidate.md, per_fold_metrics.csv, per_day_max_dd_base__<safe_cid>.parquet (per top-K), manifest.json}`
  - `step_6/` — auto-dispatched only if a Top-1 candidate clears §3 #1–9 (verdict-prior §6 below: not expected)
  - `ARC_CLOSURE.md` — template v1.3.1, mandatory §10 retroactive re-evaluation
- `docs/dispatches/arc_8_v3_0_2_log.md` — execution log per `WORKFLOW.md` §2

### Modified

- `ARC_TRACKER.md` — appended by `scripts/update_tracker_from_closure.py` at closure (atomic commit with the closure doc per `WORKFLOW.md` §2 arc-close artefact set)
- `scripts/tracker_parser/rolling_state.json` and `scripts/tracker_parser/parsed.log` — parser side-effects committed alongside

### Cache namespace (new)

- `data/cache/4H_5ers_eet/<PAIR>.parquet`, `D1_5ers_eet/<PAIR>.parquet`, `W1_5ers_eet/<PAIR>.parquet` — per `core.data.aggregator` 5ers_eet aggregation outputs. Cache is cold for these (the prior run populated `4H` / `D1` / `W1` UTC caches, not `5ers_eet`-namespaced caches). First Step-1 run materialises them; subsequent runs reuse.

### NOT touched

- `arc/l_arc_8_v3.0.2_halted` — preserved on remote, no edits, no overwrites.
- `results/l_arc_8/` — Arc 8 v3.0 original (UTC, multi-TF-features all-NaN per closure §2 caveat 2). Preserved as historical record; no edits.
- `core/*` — no engine changes planned. All required capability is on `main`: PR #208 (W1 producer canonical), PR #197 (EET session semantics + Amendment 6), PR #195 (canonical exit policy registry incl. `sl_partial_close_1r_runner_trail`), PR #194 (Amendment 5 + parser v1.3 field), PR #193 (signal-EET timezone alignment + `get_htf_value_at`), PR #189 (mid-anchored features + 5ers EET bar boundaries + worst-case fills), PR #186 (Amendment 3 risk-normalised gates), PR #185 (Step 4 classifier persistence + IS-only CV via `train_end`), PR #201 (Amendment 5.1 Gate 4 PASS-tier-constituent qualifier).

---

## §2 Steps 1–5 plan with `boundary_convention="5ers_eet"` at every subsystem

### Step 1 — Plumbing

- **Signal:** `pullback_resume_hhhl_long_v0.1` at H4 bar close. Module `core/signals/pullback_resume_hhhl.py` — classified `State A` (single-TF H4, no HTF lookup) under both UTC and 5ers_eet per `docs/audits/signal_module_eet_audit_2026_05.md`. No signal-module changes required for the convention switch.
- **Pair set:** 28 KH-24 pairs (Arc 8 original closure §4.1 list).
- **Window:** 2010-01-01 → 2025-12-31 (matches L_PROTOCOL §2 Step 5: 2010-2020 IS / 2021-present holdout, one-shot).
- **Panels:** H4 primary + D1 + W1 aux, all with `boundary_convention="5ers_eet"` — `core.data.aggregator` produces tz-aware UTC output regardless of convention (per `core/signals/htf_alignment.py` module docstring); the EET convention shifts the bar boundaries (D1 EET-day-N starts at UTC 22:00 of day N−1 in winter, UTC 21:00 in summer; H4 + W1 boundaries shift correspondingly). Cache namespace `4H_5ers_eet` / `D1_5ers_eet` / `W1_5ers_eet`.
- **Feature matrix:** all 7 v3.0 feature classes (27 features), with `panel.aux={"d1": d1_panel, "w1": w1_panel}` passed to `compute_feature_matrix`. This is the **critical engine fix from the prior intent doc preserved** — Arc 8 v3.0 original had this gap (multi-TF features all-NaN per closure §2 caveat 2); the prior `arc/l_arc_8_v3.0.2` already fixed it; we preserve the fix.
- **W1 producer canonical** — PR #208 verified in place; producer uses `get_htf_value_at(..., require_fully_closed=True)`. No `merge_asof` in W1 body (only in module docstring describing D1 lag mechanics). Source verified in §7 below.
- **Integrity report:** standard 6-check set per `core.arc.integrity`. PARTIAL items (D1-lag NaN-perturbation, spread-floor activation rate, KH-24 co-fire) hand-rolled at the arc-script level matching Arc 11's pattern.
- **KH-24 co-fire:** expected at 0.000% (Arc 8 signal is bearish-exhaustion long-resume vs KH-24's bullish exhaustion bar). Re-confirmed at integrity report.
- **Population:** `build_ex_ante_bounded_population` (canonical, uncapped) — no outcome-aware filtering.
- **Output:** per L_PROTOCOL §6.

### Step 2 — Clustering

- KMeans over K ∈ {2, 3, 4, 5, 6}, silhouette selection. `random_state=42`, `n_init=10`, `max_iter=300` per `core/determinism.py`.
- Expected (under 5ers_eet): 3 clusters reproducing Arc 8 original archetype assignments approximately (c0 Monotonic_down ~25%, c1 Choppy ~52%, c2 V-shape ~23%). 5ers_eet bar boundaries shift intraday session-position features but path-shape clustering is dominated by close-to-close geometry; centroids should be close to Arc 8 original. Flag in `cluster_summary.md` if archetype assignments shift materially (V-shape splits, Choppy disappears, etc.).

### Step 3 — Capturability

- Per-cluster reach_1R/2R/3R, MFE p25/p50/p75/p90, ww_pp, composite via `core.steps.step_3_capturability`.
- Candidate flag: `reach_1R ≥ 0.50 AND ww_pp ≤ 0.30 AND mfe_p50 ≥ 1.5R`.
- SL sweep `{1.5, 2.0, 2.5, 3.0, 3.5, 4.0} × ATR(14)` per corrected dispatch §"Step 3" (inherited from the original Arc 8 v3.0 dispatch via the prior `_halted` intent doc §1 Flag F3 default — chat ratified at the prior dispatch).
- Expected: c2 V-shape surfaces as the lone candidate cluster (Arc 8 original c2: composite 1.000, mfe_p50 7.527R, ww_pp 0.0, reach_1R 1.0). Highly unlikely Gate 4 fires (would need ≥2 candidate clusters).

### Step 4 — Extraction

- RF + LGBM + Logistic per L_PROTOCOL Appendix A defaults; 5-fold TimeSeriesSplit.
- `train_end=2021-01-01T00:00:00Z` passed to `run_step_4` per PR #185 — CV restricted to IS-only (2010-2020), holdout (2021-present) never touches Step 4 training.
- Best-AUC classifier per candidate cluster persisted to `step_4/classifiers/<cid>.pkl` with SHA256 manifest (PR #185 protocol).
- See §4 below for AUC summary plan.

### Step 5 — WFO architecture search

- Architectures admitted per **Amendment 5 four-gate** + **Amendment 5.1 Gate 4 PASS-tier-constituent qualifier**, evaluated at dispatch time (end-of-Step-4) per the 5ers_eet observed AUC. See §3 below for the admission table.
- **Exit slate (V-shape archetype canonical, per L_PROTOCOL §2 Step 5 + verification doc §2):** 4 exits = `{sl_only, sl_plus_tp_2r, sl_plus_tp_3r, sl_partial_close_1r_runner_trail}`. This matches Arc 10 v3.0's post-CC_18 canonical slate, not Arc 8 original's 3-exit pre-CC_18 slate.
- SL multiplier ∈ {Step 3 c2 optimum − 1 step, optimum, optimum + 1 step}.
- Exposure cap ∈ {`max_concurrent_per_currency=2`, unlimited}.
- A3 N-bar values ∈ {3, 5} (Pipeline DE per-fold retrain configurations).
- WFO 11-fold 2010-2020 at `r_base=0.5%` via `core.wfo.folds.build_v3_folds`.
- **Amendment 6 EET daily-DD boundary (NATIVE, not opt-in).** `compute_per_day_max_dd(boundary_convention="5ers_eet")` is invoked via `panel.boundary_convention` forwarding through `ArcOrchestrator`. Under 5ers_eet panels (Step 1 above), this is now the panel-native convention rather than a bucketing-only override. The bar-storage convention and the daily-DD-reset convention agree end-to-end.
- Amendment 3 emissions automatic via `ArcOrchestrator._run_amendment_3_evaluation`: `per_day_max_dd_base__<safe_cid>.parquet` per top-K, `chained_max_dd_base_pct` via `core.wfo.chained_dd` equity-stitching (`chained_dd_method: equity_stitching` — full-window-sim follow-up remains a separate deferred engine PR, out of scope here).
- Top-3 by worst-fold ratio → 2021-2025 holdout one-shot per L_PROTOCOL §2 Step 5 holdout-decision-rule. Holdout re-runs at `r_safe` / `r_hard` per `core.wfo.holdout_rerun.rescale_arch_config_risk`.

### Step 6 — Causal audit (auto-dispatch only)

- `core/step_6/dispatch.py:maybe_dispatch_step_6` runs on Top-1 if §3 #1–9 clear. Verdict-prior says FAIL → Step 6 unlikely to fire. Framework patch (vacuous pass for `features_in_winning_config: []` when A1 wins) is merged on main per the resume signal §5 — handled automatically if triggered.

### Closure

- `results/l_arc_8_v3.0.2/ARC_CLOSURE.md` per template v1.3.1.
- `tracker_payload.boundary_convention: 5ers_eet` (mandatory).
- `tracker_payload.architectures_skipped_by_amendment_5` populated per §3 admission outcome (post-cutoff PASS closures require this field; pre-emptively populated for all post-cutoff arcs per the standing convention).
- `step_6` block with `ran: false` / `trigger: not_applicable` if FAIL (verdict-prior).
- §3 cross-arc — explicit note: this is a fresh test under canonical 5ers_eet engine + canonical W1 producer; NOT a delta vs the `_halted` UTC prior; NOT a delta vs Arc 8 v3.0 original (UTC + W1 producer was clean but multi-TF was all-NaN).
- §10 retroactive re-evaluation — quantitative comparison vs Arc 8 v3.0 (`results/l_arc_8/`). Mandatory caveat: UTC vs 5ers_eet is a methodologically different test; AUC/ratio drifts are not attributable to single causes.

---

## §3 Architecture-admission table — Amendment 5 four-gate + Amendment 5.1 Gate 4

**Expected outcome (single candidate cluster — c2 V-shape).** Amendment 5.1 Gate 4 condition (a) requires ≥2 candidate clusters → does not fire regardless of Step 5 outcomes:

| Cluster | Expected archetype | Step 3 candidate? | Expected Step 4 mean OOS AUC | Gate 1 (archetype) | Gate 2 (AUC ≥ 0.65) | Gate 3 (universal A1) | Gate 4 (Amendment 5.1) | **Architecture set** |
|---|---|---|---|---|---|---|---|---|
| c0 | Monotonic_down | no (`dies_step3`) | not processed | n/a | n/a | n/a | n/a | (none — does not reach Step 5) |
| c1 | Choppy | no (`dies_step3`, Choppy cluster-skip) | not processed | n/a (Choppy skips Gate 1) | n/a | n/a (Choppy supersedes Gate 3) | n/a | (none — Choppy cluster-skip) |
| c2 | V-shape | **yes (candidate)** | unknown under 5ers_eet — no UTC baseline applies | **A3** (V-shape) | TBD | **A1** | n/a (cond. (a) fails: only 1 candidate cluster) | **{A1, A3}** if AUC < 0.65; **{A1, A2, A3, A6}** if AUC ≥ 0.65 |

**`architectures_skipped_by_amendment_5`** (closure §1 field) — populated post-Step-4:
- Expected case (AUC < 0.65): `[A6]` for c2 V-shape (A6 admissible under Amendment 1's archetype rule but skipped by Amendment 5 Gate 2). A2 not in the skipped list (not admitted under Amendment 1's archetype rule either).
- Contingency (AUC ≥ 0.65): `[]` for c2 (A6 admitted via Gate 2).
- Amendment 5.1 condition (a) fails → A5 not admitted, but the reason string `a5_gate_4_admission_blocked_by_no_pass_tier_constituent` is NOT applicable (that string is for condition (a) holding AND (b) failing). Closure leaves A5 out of `architectures_skipped_by_amendment_5` since Amendment 5.1 explicitly governs the "≥2 candidate cluster" case only.

**Contingency — if a second candidate cluster emerges at Step 3** (low prior given Arc 8 original surfaced only c2, but 5ers_eet bar boundaries could shift cluster boundaries): Amendment 5.1 condition (a) holds. A5 admission becomes condition-(b)-dependent — tentatively admitted at dispatch time and finalised post-Step-5 (if any constituent cluster lands PASS-DEPLOYABLE or PASS-VIABLE under Gates 1/2/3, A5 stays in the search; otherwise A5 is removed and the closure records `a5_gate_4_admission_blocked_by_no_pass_tier_constituent`).

### Config count (per cluster, with 4-exit canonical V-shape slate)

For c2 V-shape under the expected `{A1, A3}` set:
- A1: 1 × 3 SLs × 4 exits × 2 exposures = **24 configs**
- A3: 1 × 2 N-bars × 3 SLs × 4 exits × 2 exposures = **48 configs**
- **Subtotal c2: 72 configs.** `search_scope_flag: normal` (50–99 per closure template §1).

For c2 under the contingency `{A1, A2, A3, A6}` set:
- + A2: 1 × 3 SLs × 4 exits × 2 exposures = 24
- + A6: 3 threshold pairs × 3 SLs × 4 exits × 2 exposures = 72
- **Total: 168 configs.** `search_scope_flag: broad` (≥ 100).

(Note: the prior `_halted` intent doc reported 108 / 252 using a 6-exit slate. The verification doc §2 corrected this to 4 exits per L_PROTOCOL canonical V-shape; 72 / 168 is the corrected count.)

---

## §4 Step 4 AUC summary plan

Per corrected dispatch §4 item 4 and Amendment 5 §9 dispatch-discipline #3. Step 4 outputs and the closure §1 `clusters.<cid>.step4_e_auc` / §2 prose will report:

- **Per candidate cluster** (c2 V-shape expected; any others if Step 3 surfaces them):
  - Mean OOS AUC across 5 TimeSeriesSplit folds per classifier (RF, LGBM, Logistic)
  - Std AUC across folds + fold-by-fold AUC list
  - Best classifier identity, its mean OOS AUC, and AUC-best threshold
  - Amendment 5 Gate 2 outcome — "AUC X.XXX < 0.65 → A2 + A6 SKIPPED" or "AUC X.XXX ≥ 0.65 → A2 + A6 ADMITTED"
- **Informational comparison (NOT a baseline test):**
  - Arc 8 v3.0 original (UTC + multi-TF all-NaN): c2 RF mean OOS AUC = 0.5300 — reported as historical context only.
  - `arc/l_arc_8_v3.0.2_halted` (UTC + multi-TF restored + W1 producer leaking): c2 mean AUC = 0.6938 — reported as informational only. **This is NOT a baseline.** It was driven primarily by the W1 lookahead (verification doc §1.3: sign-conditional V-shape rate 5.4× lifted; permutation importance 28× the next multi-TF feature). With the W1 fix and the convention switch, drift is unconstrained.
  - v3.0.2 5ers_eet (this run): observed mean AUC — reported as the canonical value for Amendment 5 Gate 2 enforcement.
- **No HALT-on-AUC-divergence trigger.** The prior intent doc §0 carried a HALT trigger calibrated against the (now-known-to-be-leaky) 0.6938 prior. Removed here. Under 5ers_eet + canonical W1, the c2 AUC has no reference value to diverge from. Whatever surfaces is the canonical 5ers_eet result.
- **`step4_d1_auc` field**: populated `null` matching Arc 8 original's convention (Step 4 emits a single OOS-mean AUC per (cluster, model); the `_d1_auc` field is a legacy split from older pipeline conventions).

---

## §5 Compute estimate

Wall-clock estimates, single-process (`n_jobs=1` per determinism contract). Includes cache-cold 5ers_eet panel materialisation cost per corrected dispatch §4 item 5.

| Phase | Estimate | Notes |
|---|---:|---|
| 5ers_eet panel materialisation (cache-cold, 28 pairs × H4 + D1 + W1) | **+15–30 min** | Per corrected dispatch. First Step-1 run pays this cost; subsequent runs reuse `data/cache/{4H,D1,W1}_5ers_eet/<PAIR>.parquet`. |
| Step 1 (pool + feature matrix on 5ers_eet panels) | ~25–35 min | Feature matrix dominates; ~10% slower than Arc 8 original due to multi-TF feature restoration. |
| Step 2 (clustering, K-sweep) | ~2–3 min | KMeans on path-shape features, deterministic. |
| Step 3 (capturability) | ~1 min | Per-cluster aggregations. |
| Step 4 (RF + LGBM + LR × 1 candidate cluster × 5 folds + refit for persistence) | ~10–15 min | LGBM dominates. |
| Step 5 expected case ({A1, A3}, 72 configs) | ~2–3 hr | A3 per-fold retrain × 11 folds × 48 configs is the bulk. |
| Step 5 contingency case ({A1, A2, A3, A6}, 168 configs) | ~4–5 hr | If observed AUC ≥ 0.65. |
| Step 6 (if auto-dispatched — verdict-prior says no) | ~5 min | Six-category framework, mostly metadata. |
| Closure + parser + commit + PR | ~30 min | Includes tracker delta + atomic commit. |
| **Total (expected case)** | **~3.5–4.5 hr** | Cache-cold 5ers_eet adds 15-30 min vs the prior `_halted` UTC run wall-clock baseline. |
| **Total (contingency case)** | **~5.5–7 hr** | If Amendment 5 Gate 2 admits A2 + A6. |

User-side workstation, single-process. No remote/cloud compute.

---

## §6 Verdict prior

**Unknown.** All available priors are non-applicable:

- Arc 8 v3.0 (`results/l_arc_8/ARC_CLOSURE.md`): UTC + multi-TF all-NaN. FAIL with worst-fold ratio 1.749; primary failure mode re-classified to `step5_ratio_below_gate_after_scaling` post-Amendment-3. Different engine state → not directly comparable.
- `arc/l_arc_8_v3.0.2_halted`: UTC + multi-TF restored + W1 leaking. Halted pre-Step-5; Step 4 c2 AUC 0.6938 was an artefact of the W1 lookahead, not a true engine attribution.

The corrected 5ers_eet + canonical-W1 retry is its own test. No prior anchor.

Best guess (low confidence): clean engine + canonical convention removes the AUC inflation → c2 V-shape Step 4 AUC lands sub-0.65 → Amendment 5 admits only `{A1, A3}` for c2 → 72-config search → worst-fold ratio likely > 2.0 reproducing Arc 8 original's search-WFO-FAIL pattern. Holdout behaviour is harder to call (Arc 8 original's search-fail + holdout-pass anomaly was anchored in regime asymmetry; that prior holds independent of convention).

---

## §7 Pre-flight gate receipt (corrected dispatch §1 — inlined per chat directive)

Per chat directive: "treat Arc 5 corrected cross-references as placeholders; inline best-effort §1/§3/§7/§8 inferences from CLAUDE.md, L_PROTOCOL.md, prior intent on `arc/l_arc_8_v3.0.2_halted`, and the canonical 5ers_eet conventions."

Pre-flight checks (canonical engine state on `origin/main` @ `dd391cd`):

- ✅ **PR #208 (W1 producer canonical alignment)** — merged at `ab03be9` / commit `8ce3b3d`. Verified: `core/features/multi_tf.py::_w1_close_slope_sign` uses `get_htf_value_at(..., require_fully_closed=True)`; no `merge_asof` in the W1 producer body (line 13 occurrence is module docstring describing D1 lag mechanics, not the W1 producer).
- ✅ **PR #197 (Amendment 6 EET session semantics)** — merged. `core.runners._fold_stats_helpers.compute_per_day_max_dd(boundary_convention="5ers_eet")`, `core.time_utils.session_boundary.utc_to_eet_trading_day`, `SUPPORTED_CONVENTIONS == ('utc', '5ers_eet')`.
- ✅ **PR #195 (canonical exit policy registry)** — merged. `core.sim.exit_policies.available_policies()` includes `sl_partial_close_1r_runner_trail`.
- ✅ **PR #194 (Amendment 5 + parser v1.3)** — merged. Closure template `architectures_skipped_by_amendment_5` field accepted.
- ✅ **PR #201 (Amendment 5.1 Gate 4 PASS-tier-constituent qualifier)** — merged. Parser accepts `a5_gate_4_admission_blocked_by_no_pass_tier_constituent` reason string.
- ✅ **PR #193 (signal-EET timezone alignment + `core/signals/htf_alignment.py`)** — merged. `get_htf_value_at`, `get_htf_row_at`, `get_htf_index_at` all importable.
- ✅ **PR #189 (mid-anchored features + 5ers EET bar boundaries + worst-case fills)** — merged. `boundary_convention="5ers_eet"` opt-in available end-to-end.
- ✅ **PR #186 (Amendment 3 risk-normalised gates)** — merged. `ArcOrchestrator._run_amendment_3_evaluation` and `core.wfo.holdout_rerun.rescale_arch_config_risk` wired.
- ✅ **PR #185 (Step 4 classifier persistence + IS-only CV)** — merged. `run_step_4(train_end=...)` supported.
- ✅ **Branch hygiene (corrected dispatch §2).** Prior `arc/l_arc_8_v3.0.2` renamed to `arc/l_arc_8_v3.0.2_halted` (local + remote), preservation commit `aa641dc` pushed. New `arc/l_arc_8_v3.0.2` cut FRESH from `origin/main` at `dd391cd`, pushed (`origin/arc/l_arc_8_v3.0.2` tracks).

HALT triggers (inlined — Arc 5 corrected dispatch §7 cross-reference treated as placeholder):

- Step 1 integrity report fails any of the 6 standard checks → HALT, diagnostic doc, end turn.
- Step 1 multi-TF feature population reproduces the Arc 8 v3.0 all-NaN failure (i.e. `panel.aux={"d1":..., "w1":...}` not threaded correctly) → HALT.
- Step 2 archetype assignment shifts materially from Arc 8 original (V-shape splits across clusters, Choppy disappears, etc.) → flag in `cluster_summary.md`, not HALT (5ers_eet shift is expected to be modest).
- Step 4 feature importance ranks any multi-TF feature suspiciously high relative to D1 producers' rank (canary against re-introduction of an HTF lookahead) → diagnostic comparison; HALT if a 5× gap recurs (mirroring the W1 lookahead signature).
- Step 5 worst-fold ratio fails Amendment 3 gates → not a HALT; produce closure as FAIL per verdict-prior §6.
- Any cross-arc contamination flag from the parser → HALT.

Definition of done (corrected dispatch §8 — Arc 5 corrected §9 placeholder treated as standard arc-close DOD):

- All five step artefact sets present and SHA256-manifested.
- `ARC_CLOSURE.md` per template v1.3.1 with §10 retroactive re-evaluation.
- Tracker delta atomic commit (closure + parser side-effects in one commit).
- PR opened titled `[ARC 8 v3.0.2] pullback_resume_hhhl_long — <verdict>`.
- `boundary_convention: 5ers_eet` recorded in closure §1.

---

## End of intent. END TURN per corrected dispatch §4 closing line.

No blocking decisions outstanding. The two prior-intent flags (UTC convention; AUC-discrepancy interpretation) are both resolved by the corrected dispatch: convention is 5ers_eet end-to-end; AUC is whatever Step 4 produces under the clean engine, with Amendment 5 enforcement at dispatch time per the observed value (no chat-side override of the gate).

Step 1 begins on chat go-ahead.
