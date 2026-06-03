# Amendment 3 Re-evaluation — Intent Doc (chat review before any closure-doc edits)

> Drafted: 2026-05-22
> Branch: `claude/confident-mestorf-0b36a7` (worktree off `main`; `main` has commit `8a9723e` — Amendment 3 + template v1.1)
> Dispatch: Re-evaluate Arcs 8, 10, 11 under Amendment 3 (risk-normalised gates)
> Status: read-first complete. No edits to closure docs / tracker / summary until chat reviews this intent.

---

## Read-first artefacts consumed

1. `L_PROTOCOL.md` §3 amended gate definitions (DEPLOYABLE + VIABLE, scalability bounds, daily DD measurement spec, failure-mode priority order, taxonomy extensions).
2. `docs/templates/ARC_CLOSURE_TEMPLATE.md` v1.1 — new `best_architecture` fields (`worst_fold_dd_base_pct`, `worst_fold_roi_base_pct`, `chained_max_dd_base_pct`, `per_day_max_dd_*`, `k_safe`/`k_hard`, `r_safe_pct`/`r_hard_pct`, scaled ROI/DD/holdout, `sizing_convention`), and the schema-versioning rule (closures landed before v1.1 retain v1.0 field names; parser detects via `template_version`).
3. Closure docs `results/l_arc_8/ARC_CLOSURE.md`, `results/l_arc_10/ARC_CLOSURE.md`, `results/l_arc_11/ARC_CLOSURE.md` in full.
4. Supporting per-fold artefacts inspected for what additional data is recoverable without engine re-runs:
   - `results/l_arc_8/step_5/per_fold_metrics.csv` — per-config × per-fold n_trades, ROI, max_dd, `daily_5pct_breaches`, ratio.
   - `results/l_arc_10/step_5/wfo_results.csv` — per-config aggregated metrics (worst/mean ROI, worst/mean DD, sign-consistency, n_total). No per-fold breakdown saved separately for Arc 10.
   - `results/l_arc_11/step_5/per_fold_metrics.csv` — per-config × per-fold n_trades, ROI, max_dd, `days_breaching_daily_5pct`, ratio.

All three closure docs are v1.0 template (no `template_version` field). Pre-amendment field names (`worst_fold_roi_pct`, `worst_fold_dd_pct`) are used. Per template §"Schema versioning", these are treated as the v1.0 fields and the renamed-field equivalence applies.

---

## Per-arc metric availability

### Arc 8 — `pullback_resume_hhhl_long_v0.1` (4H, vanilla, original verdict: FAIL)

| Metric | Status | Value | Source |
|---|---|---|---|
| `worst_fold_dd_base_pct` (= old `worst_fold_dd_pct`) | PRESENT | **1.9826%** | closure §1 |
| `worst_fold_roi_base_pct` (= old `worst_fold_roi_pct`) | PRESENT | **+1.7486%** | closure §1 |
| Per-fold ROI / sign-consistency | PRESENT | 11/11 positive (closure §1 `sign_pos_folds: "11/11"`); per-fold values reconstructable from `step_5/per_fold_metrics.csv` if needed | closure §1 + per-fold CSV |
| `chained_max_dd_base_pct` (IS + holdout concatenated equity curve) | **MISSING** | — | not in closure §1; not in `step_5/` artefacts |
| Per-day max-DD series (`per_day_max_dd_base.parquet`) | **MISSING** | — | Amendment 3 artefact, not built under v3.0 pre-amendment |
| Per-fold `daily_5pct_breaches` count at `r_base` | PRESENT | 0 across all 11 folds of the winning A6 config (per-fold CSV) | `step_5/per_fold_metrics.csv` |
| Trades per fold | PRESENT | min ≈ 247 (per-fold CSV, A6 winner config) — well above the 25 floor | `step_5/per_fold_metrics.csv` |
| Holdout ROI @ `r_base` | PRESENT | +42.78% | closure §1 |
| Holdout DD @ `r_base` | PRESENT | 1.7369% | closure §1 |
| `sizing_convention` | reset-floor (L arc default per CLAUDE.md / L_PROTOCOL §3 scalability sizing-convention rule) | implicit |

Note. The closure's `worst_fold_ratio: 1.749` does NOT equal `worst_fold_roi_pct / worst_fold_dd_pct = 0.882`. The closure's gate-comparison narrative ("worst-fold ratio 1.749 against a 2.0 threshold") uses the 1.749 number directly. Under the amendment, the ratio is explicitly defined as `worst_fold_roi / worst_fold_dd` and is invariant under linear scaling. The re-evaluation will compute the ratio from `worst_fold_roi_base / worst_fold_dd_base = 0.882` and flag the discrepancy with the closure's reported value. **This affects the verdict logic.** Chat may want to instruct whether to (a) trust the closure's `worst_fold_ratio: 1.749` as the gate quantity, or (b) recompute as 0.882 per the §3 definition.

### Arc 10 — `d1_swing_low_rejection_long` (H4, vanilla, original verdict: PASS-VIABLE)

| Metric | Status | Value | Source |
|---|---|---|---|
| `worst_fold_dd_base_pct` | PRESENT | **9.22%** | closure §1 |
| `worst_fold_roi_base_pct` | PRESENT | **+26.49%** | closure §1 |
| Per-fold ROI / sign-consistency | PRESENT | 11/11 positive (closure §1) | closure §1 |
| `chained_max_dd_base_pct` | **MISSING** | — | not in closure §1; not in `step_5/` artefacts |
| Per-day max-DD series | **MISSING** | — | Amendment 3 artefact, not built |
| Per-fold daily-breach counts at `r_base` | **MISSING** | — | per-fold CSV not present for Arc 10; only aggregated `wfo_results.csv` |
| Trades per fold | PARTIAL | n_total=2162 / 11 folds = ~197 avg; per-fold breakdown not saved in tracked artefacts | closure §1 (total only) |
| Holdout ROI @ `r_base` | PRESENT | +59.07% | closure §1 |
| Holdout DD @ `r_base` | PRESENT | 5.03% | closure §1 |
| `sizing_convention` | reset-floor | implicit |

Note. Arc 10 closure §2 caveats already flag (a) per-fold equity reset (cumulative DD across folds not tracked — relevant to amendment's `chained_max_dd`), (b) oracle exit-policy mismatch. The `chained_max_dd` caveat in particular maps directly to the amendment's missing-data flag and is the binding gap for upgrading Arc 10 from PASS-VIABLE to PASS-DEPLOYABLE definitively.

### Arc 11 — `swing_high_breakout_trend_long` (H4, vanilla, original verdict: FAIL)

| Metric | Status | Value | Source |
|---|---|---|---|
| `worst_fold_dd_base_pct` | PRESENT | **38.36%** | closure §1 |
| `worst_fold_roi_base_pct` | PRESENT | **−24.03%** | closure §1 |
| Per-fold ROI / sign-consistency | PRESENT | 6/10 positive (closure §1 — NB. 10 folds, not 11; raises a separate methodology issue documented in closure §3 `canonical_orchestrator_step5_run_context_gap`) | closure §1 + per-fold CSV |
| `chained_max_dd_base_pct` | **MISSING** | — | not in closure §1 |
| Per-day max-DD series | **MISSING** | — | Amendment 3 artefact, not built |
| Per-fold `days_breaching_daily_5pct` at `r_base` | PRESENT | non-zero across multiple folds (e.g. A2 winner: folds 6/9/10 = 1 each, fold 11 = 2) | `step_5/per_fold_metrics.csv` |
| Trades per fold | PRESENT | min n_trades = 15 (A2 winner fold 5) — **below 25-trade floor** at r_base | `step_5/per_fold_metrics.csv` |
| Holdout ROI @ `r_base` | PRESENT | +1.30% | closure §1 |
| Holdout DD @ `r_base` | PRESENT | 57.50% | closure §1 |
| `sizing_convention` | reset-floor | implicit |

---

## What "MISSING" means for verdict definitiveness

Per dispatch:
- `chained_max_dd_base_pct` missing → constraint #7 cannot be definitively evaluated. Provisional pass on assumption.
- Per-day max-DD series missing → constraint #6 cannot be definitively evaluated. Provisional pass on assumption.
- Both missing across all three arcs ⇒ all three re-evaluations will carry the PROVISIONAL flag if scalability + ratio + ROI + sign-consistency + trade-count constraints clear.

If both are missing AND any of the other constraints fails, the verdict can still be **definitive FAIL** (the failing constraint is independent of the missing data).

---

## Preliminary verdict shape (chat review before this is committed)

> All numbers below use the §3 mathematical definition: `worst_fold_ratio_base = worst_fold_roi_base / worst_fold_dd_base`. This differs from the closure's `worst_fold_ratio` field for Arc 8 (1.749 closure vs 0.882 §3 math) — flag noted above.

### Arc 8
- `worst_fold_dd_base = 1.9826%` → `k_safe = 8 / 1.9826 = 4.0351`; `r_safe = 0.5 × 4.0351 = 2.018%`.
- **`r_safe = 2.018% > r_max = 2.0%` → scalability FAIL** (DEPLOYABLE).
- `k_hard = 10 / 1.9826 = 5.044`; `r_hard = 2.522% > 2.0%` → scalability FAIL (VIABLE).
- Also: `ratio_base = 1.7486 / 1.9826 = 0.882 < 2.0` → ratio gate FAILS under amendment regardless of scaling (invariant).
- **Re-evaluated verdict: FAIL.** Primary failure mode (priority order): `step5_not_scalable`.
- Cross-arc observation: novel failure mode — "DD too small relative to gate, can't scale risk high enough at r_max". The closure's narrative attributed the FAIL to entry-feature AUC ceiling capping the ratio; under Amendment 3 the *same data* now fails the scalability bound first. Worth a cross-arc tag: `scalability_cap_due_to_low_dd`.

### Arc 10
- `worst_fold_dd_base = 9.22%` → `k_safe = 8 / 9.22 = 0.8677`; `r_safe = 0.5 × 0.8677 = 0.434%`. Within [0.15%, 2.0%] ✓.
- `k_hard = 10 / 9.22 = 1.0846`; `r_hard = 0.5 × 1.0846 = 0.542%`. Within bounds ✓.
- `worst_fold_roi_at_r_safe = 26.49 × 0.8677 = +22.98%`; ratio invariant 5.42 (or 2.87 under strict §3 math). Both ≥ 2.0 ✓.
- 11/11 positive folds ✓.
- Trade count per fold ≥ 25 ✓ (avg 197; need data for definitive min but very likely OK).
- Holdout linear-proxy at `r_safe`: ROI ≈ +51.27%, DD ≈ 4.36% (< 8% gate) — PROXY clears.
- Chained max DD and per-day breaches: MISSING → PROVISIONAL.
- **Re-evaluated verdict: PASS-DEPLOYABLE-PROVISIONAL** (upgrade from PASS-VIABLE).
  - Hinges on missing chained DD and per-day breach data passing under linear scaling.
  - Note: `k_safe < 1` (we're scaling *down* from r_base because base DD already > 8%). Under downward scaling, daily breach count can only decrease or stay flat — but the dispatch's procedure flags this as PROVISIONAL regardless. If chat wants to upgrade to DEFINITIVE on the daily-breach constraint specifically, the mathematical argument is available; the chained-DD constraint still requires a re-run.
- Step 6 audit: original PASS-VIABLE verdict implies Step 6 was run (per §3 lazy rule). Re-evaluation assumes Step 6 carries over.

### Arc 11
- `worst_fold_dd_base = 38.36%` → `k_safe = 8 / 38.36 = 0.2085`; `r_safe = 0.5 × 0.2085 = 0.104%`. **Below `r_min = 0.15%` floor** → scalability FAIL (DEPLOYABLE).
- `k_hard = 10 / 38.36 = 0.2607`; `r_hard = 0.5 × 0.2607 = 0.130%`. **Below floor** → scalability FAIL (VIABLE).
- Also: ratio_base = −0.626 < 2.0; ROI base negative; sign-consistency 6/10 (4 negative folds — VIABLE allows ≤1); min trades per fold 15 < 25. Multiple independent FAILs.
- **Re-evaluated verdict: FAIL** (definitive — multiple independent gate failures including scalability). Primary failure mode (priority): `step5_not_scalable`. Notable secondaries: `step5_negative_folds`, `step5_trade_count_below_gate`, `step5_ratio_below_gate_after_scaling`.

---

## Open question for chat (before closure-doc edits)

1. **Arc 8 `worst_fold_ratio` discrepancy.** Closure §1 reports `worst_fold_ratio: 1.749`. Computed `worst_fold_roi_pct / worst_fold_dd_pct = 1.7486 / 1.9826 = 0.882`. The §3 amendment explicitly defines the gate as ROI/DD. Should the re-evaluation:
   - (a) Use 0.882 (the §3 definition) — current proposed approach. This makes the §10 block self-consistent with the amendment but disagrees numerically with §1 of the closure.
   - (b) Use 1.749 (the closure's reported ratio) — keeps internal consistency with the closure but contradicts the amendment's explicit definition.
   - The verdict outcome (FAIL) is the same either way (scalability fails first), but the §10 block's "ratio at r_safe" row will print very different numbers.
   - Same question latently applies to Arc 10 (5.42 closure vs 2.87 math) and Arc 11 (−0.769 closure vs −0.626 math). Arc 10's verdict outcome is also the same either way (both ratios > 2.0). Arc 11 is the same either way (both fail).

2. **Provisional vs definitive for Arc 10 daily-breach constraint.** Under `k_safe = 0.87` (downward scaling), daily breach count at r_safe ≤ count at r_base. The per-fold breach data was NOT recorded for Arc 10 (no `per_fold_metrics.csv`), so we don't even have the r_base count — but we DO have `worst_fold_dd_pct = 9.22%`, which suggests daily DD profile was probably benign. Without the per-fold or per-day data, the constraint is PROVISIONAL under either reading. The intent doc proposes leaving it PROVISIONAL.

3. **Folder/file location confirm.** Intent doc landed at `results/re_evaluation_2026_05/re_evaluation_intent.md`. Summary doc will land at `results/re_evaluation_2026_05/SUMMARY.md` per dispatch. OK to proceed?

4. **Branch name.** Dispatch specifies `infra/re-evaluate-pre-amendment-arcs` cut from main after Amendment 3 lands. I'm on `claude/confident-mestorf-0b36a7` (worktree branch). Should I rename / re-branch, or proceed on the current worktree branch and let PR title carry the `[INFRA]` label?

5. **`primary_failure_mode` for Arc 8.** The original closure uses `entry_feature_auc_ceiling`. Under the amendment, the scaling-cap-due-to-low-DD failure isn't in the §3 priority order or taxonomy explicitly (the taxonomy's `step5_not_scalable` covers it). Re-evaluation will assign `step5_not_scalable` as the new `primary_failure_mode`. Confirm?

---

## End turn

Awaiting chat confirmation before producing the §10 re-evaluation blocks, tracker update, and SUMMARY.md.
