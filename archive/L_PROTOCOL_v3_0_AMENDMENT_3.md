# L_PROTOCOL Amendment 3 — Risk-Normalised Gates (v5, locked)

> **Date:** 2026-05-22
> **Status:** locked and landed into `L_PROTOCOL.md` §3 on 2026-05-22.
> **Triggered by:** Protocol failed configs with high raw DD but linearly-scalable risk.
> **Replaces:** v1, v2, v3, v4 drafts (all superseded by this v5; no historical drafts preserved).
> **Scope:** PURE risk-normalisation. Adds scaling, scalability bounds, daily/chained DD evaluation at scaled risk. Does NOT redesign §3 gate constraints — every constraint matched 1:1 to current §3 wording; only the risk at which constraints are evaluated changes.
> **Supersedes within §3:** the procedure by which DD constraints are evaluated. Constraint thresholds preserved.

---

## Core principle

A config passes if it can be **scaled to a per-trade risk** at which all current §3 constraints hold. Evaluation happens at the scaled risk, not the base risk used in the WFO run.

Two tiers preserved: PASS-DEPLOYABLE (single-strategy ship) and PASS-VIABLE (A5 portfolio composition eligible).

---

## Scaling rule

WFO runs at base risk `r_base` (typically 0.5%). Engine emits required metrics at `r_base`.

Compute:

```
k_safe = 8.0 / worst_fold_dd_base
k_hard = 10.0 / worst_fold_dd_base
r_safe = r_base × k_safe
r_hard = r_base × k_hard
```

Linear scaling applies to: worst-fold ROI, chained max DD, holdout ROI/DD, per-day max-DD (then re-counted for daily breaches).

Linear scaling does **NOT** apply to: daily breach **counts** themselves (step function — see §"Daily DD measurement").

Constraints that are not affected by risk scaling (sign-consistency, trade count, fold count): unchanged from current §3.

---

## Scalability bounds (locked)

- **Floor:** `r_safe ≥ r_min = 0.15%`. Locked value.
- **Ceiling:** `r_safe ≤ r_max = 2.0%`. Locked value.
- **Edge case:** `worst_fold_dd_base = 0%` → `k = ∞` → FAIL `step5_not_scalable` (no DD information to anchor scaling).
- **Sizing convention:** linear DD scaling holds ONLY under reset-floor sizing (the L arc convention). Arcs using %-of-current-equity sizing FAIL the scalability check by default unless chat approves a separate scaling treatment for that arc.

Same bounds apply to `r_hard` for VIABLE evaluation.

---

## PASS-DEPLOYABLE gate (matched 1:1 to current §3 PASS-DEPLOYABLE)

All of:

1. **Scalable to safe:** `r_safe ∈ [0.15%, 2.0%]` (NEW constraint added by this amendment — required for scaled evaluation).
2. **Worst-fold ROI/DD ratio at `r_safe`:** ≥ 2.0 (ratio invariant under linear scaling; same value as at `r_base`; constraint unchanged in threshold).
3. **Worst-fold ROI at `r_safe`:** > 0 (linearly scaled from `r_base`).
4. **Per-fold positivity:** positive at all 11 IS folds, 0 negative folds (matches current §3; sign does not scale; holdout has separate gate).
5. **Worst-fold DD at `r_safe`:** ≤ 8% (by construction at `k_safe`; matches current §3 "at chosen risk: DD ≤ 8%").
6. **Daily DD breaches at `r_safe`:** exactly 0 across all IS folds + holdout (matches current §3 "0 days breaching 5% daily"; see §"Daily DD measurement" for evaluation).
7. **Chained max DD at `r_safe`:** ≤ 10% (matches current §3 "at chosen risk: max DD ≤ 10%"; linearly scaled).
8. **Trades per fold:** ≥ 25 per fold (trade count does not scale; unchanged from current §3).
9. **Holdout at `r_safe`:** re-run holdout sim at `r_safe`; result clears existing §3 holdout gate (unchanged thresholds).
10. **Step 6 causal audit clean** (evaluated last per §"Evaluation order").

> Note: current §3 DEPLOYABLE has a known internal duplicate ("positive at all 11 folds" + "0 negative folds" as separate bullets). Preserved here for 1:1 mapping fidelity; flagged as §3 housekeeping for a future protocol clean-up, not this amendment.

Every constraint references current §3 wording. The only change is "evaluated at `r_safe`" replacing "at chosen risk" or "at `r_base`" depending on the constraint.

## PASS-VIABLE gate (matched 1:1 to current §3 PASS-VIABLE; A5 portfolio composition eligible)

All of:

1. **Hard-scalable:** `r_hard ∈ [0.15%, 2.0%]` (NEW).
2. **Worst-fold ROI/DD ratio at `r_hard`:** ≥ 2.0 (ratio invariant under linear scaling; unchanged threshold).
3. **Mean-fold ROI/DD ratio at `r_hard`:** ≥ 2.5 (ratio invariant under linear scaling; unchanged threshold).
4. **Per-fold positivity:** up to 1 negative fold permitted across the 11 IS folds (matches current §3 VIABLE per-fold rule; sign-related, does not scale).
5. **Worst-fold DD at `r_hard`:** ≤ 10% (by construction at `k_hard`; matches 5ers hard limit).
6. **Daily DD breaches at `r_hard`:** exactly 0.
7. **Chained max DD at `r_hard`:** clears existing §3 VIABLE chained-DD gate (unchanged threshold; linearly scaled).
8. **Trades per fold:** clears existing §3 VIABLE trade-count gate (unchanged).
9. **Holdout at `r_hard`:** re-run at `r_hard`; clears existing holdout gate.
10. **Step 6 causal audit clean.**

A config can be VIABLE without being DEPLOYABLE if it scales into 5ers-hard compliance but not in-system safety compliance. VIABLE strategies enter A5 portfolio composition per existing §3 A5 spec (subject to follow-up — see §"A5 follow-up flag").

---

## Daily DD measurement (handles step-function correctly)

Linear scaling of breach **counts** is mathematically wrong. Correct procedure:

**Engine emits at `r_base`:** for each trading day in the IS + holdout trajectory, the maximum intraday drawdown percentage of that day's starting equity. Full per-day series persisted to a separate parquet artefact (NOT only summary stats in the payload).

**Day-start equity definition:** the account equity at 00:00 broker-day (i.e., the equity at the start of that calendar trading day in broker timezone), NOT the reset-floor sizing baseline. Under reset-floor sizing the two values are distinct: reset-floor is the per-trade sizing reference; day-start equity is the daily-DD reference. Engine must use day-start equity for daily DD measurement.

**Engine artefact:** `step_5/per_day_max_dd_base.parquet`. Columns: `date`, `pair_set`, `day_max_dd_base_pct`, `n_trades_open_start_of_day`.

**At gate evaluation:** for each day in the series, compute `day_max_dd_scaled = day_max_dd_base × k`. Count days where `day_max_dd_scaled ≥ 5%`.

This is per-day re-evaluation, not count scaling — correct under linear-scaling assumption.

**Boundary:** UTC broker-day. Locked value.

**Tolerance:** exactly 0 breaches in both tiers. No safety margin on the daily limit.

---

## Chained max DD measurement

Computed across the full IS + holdout trajectory (folds concatenated chronologically into one continuous equity curve, peak-to-trough across the whole curve).

Scales linearly with `k`. Threshold preserved from current §3 chained-DD spec — only the risk at which evaluated changes.

---

## Evaluation order (Step 6 ordering)

Step 6 is lazy per current protocol §2. This amendment specifies order explicitly:

1. Constraints #1-#9 (all non-Step-6 constraints, both tiers) evaluated in priority order.
2. If ALL clear → Step 6 causal audit dispatched.
3. Step 6 clean → PASS-DEPLOYABLE / PASS-VIABLE finalised.
4. Step 6 fails → FAIL with `primary_failure_mode = step6_causal_audit_fail`.

Step 6 is the LAST gate.

---

## Failure-mode priority (tie-break)

When multiple constraints fail simultaneously, `primary_failure_mode` is assigned by first-fail in this order:

1. `pool_too_small` (Step 1 failure — encountered before Step 5 evaluation)
2. `step5_not_scalable` (scalability bounds)
3. `step5_dd_above_gate` (defensive — should not occur by construction)
4. `step5_chained_dd_above_gate`
5. `step5_daily_dd_breach`
6. `holdout_fail_after_is_pass`
7. `step5_sign_consistency_fail` / `step5_negative_folds`
8. `step5_trade_count_below_gate`
9. `step5_wf_roi_below_gate_after_scaling`
10. `step5_ratio_below_gate_after_scaling`
11. `step6_causal_audit_fail`

Other §3 failure modes not listed (no clusters separable, no capturable cluster, etc.) remain ordered per current protocol.

---

## Failure mode taxonomy update

Add to `primary_failure_mode` enum:

- `step5_not_scalable`
- `step5_daily_dd_breach`
- `step5_chained_dd_above_gate`
- `step5_wf_roi_below_gate_after_scaling`
- `step5_ratio_below_gate_after_scaling`
- `step5_trade_count_below_gate`
- `step5_negative_folds`

Deprecate but retain for historical closures:

- `step5_dd_above_gate` (replaced by scalability check + scaled gates)

---

## A5 follow-up flag (NOT in scope of this amendment)

Combined A5 portfolio DD when each component is independently scaled to `r_hard` (worst-fold DD = 10%) can produce combined account-wide DD > 10%. Current §3 A5 spec does not address this; current amendment does not either.

**Follow-up required:** separate A5 amendment specifying combined-portfolio DD constraint when components are VIABLE-tier. Candidate approaches: cap combined `r_hard` allocation; require pairwise correlation analysis; require combined-portfolio chained-DD simulation at proposed allocation. Defer until first VIABLE candidate emerges and A5 deployment is concretely on the table.

---

## Engine-side changes

Scoped:

1. **Per-day max-DD emission.** Full per-day series to `step_5/per_day_max_dd_base.parquet` at `r_base`. Required for daily DD evaluation at scaled risk.
2. **Chained max DD emission.** Across IS + holdout trajectory at `r_base`. (Likely partially implemented for sign-consistency already.)
3. **Verdict logic.** Step 5 closure-writer reads per-day artefact, computes `k_safe`, `k_hard`, `r_safe`, `r_hard`, scaled metrics, applies gate per priority order.
4. **Holdout re-run.** For each candidate reaching gate stage, re-run holdout sim at `r_safe` (for DEPLOYABLE) and `r_hard` (for VIABLE if separately evaluated). 1-2 additional sims per candidate.
5. **Determinism.** Each holdout re-run produces its own sha256 manifest entry. Two holdout re-runs (one at `r_safe`, one at `r_hard`) produce two separate manifest entries. Same seed, same data → same output bytes.

---

## Why holdout re-runs but IS doesn't

IS is the scaling input — linear-scaling assumption applied analytically to per-fold metrics is the foundation. Holdout is the one-shot deployment proxy and must match deployment risk exactly for the verdict to be honest. Re-running holdout at scaled risk costs little (1-2 sims) and removes the strongest assumption from the most consequential gate.

---

## Tracker payload additions

`§1 tracker_payload.best_architecture` in `ARC_CLOSURE_TEMPLATE.md` v1.0 → v1.1.

### Field rename map

| Old field (v1.0) | New field (v1.1) |
|---|---|
| `worst_fold_dd_pct` | `worst_fold_dd_base_pct` |
| `worst_fold_roi_pct` | `worst_fold_roi_base_pct` |

Closures landed before this amendment retain v1.0 field names. Parser handles both schemas via version detection on the closure doc's template version reference.

### New fields

```yaml
  best_architecture:
    # ── renamed fields above ──
    chained_max_dd_base_pct: <float>            # NEW — IS+holdout chained at base risk
    per_day_max_dd_artefact_path: <path>        # NEW — relative path to per-day series parquet
    per_day_max_dd_base_summary:                # NEW — summary for human inspection only
      n_days: <int>
      p50_pct: <float>
      p95_pct: <float>
      p99_pct: <float>
      max_pct: <float>
    k_safe: <float>                             # NEW
    k_hard: <float>                             # NEW
    r_safe_pct: <float>                         # NEW
    r_hard_pct: <float>                         # NEW
    scalable_to_safe: <bool>                    # NEW
    scalable_to_hard: <bool>                    # NEW
    worst_fold_roi_at_r_safe_pct: <float>       # NEW
    worst_fold_roi_at_r_hard_pct: <float>       # NEW
    chained_max_dd_at_r_safe_pct: <float>       # NEW
    chained_max_dd_at_r_hard_pct: <float>       # NEW
    daily_dd_breaches_at_r_safe: <int>          # NEW (recounted per-day, not count-scaled)
    daily_dd_breaches_at_r_hard: <int>          # NEW
    holdout_roi_at_r_safe_pct: <float or null>  # NEW (from re-run; null if not re-run)
    holdout_dd_at_r_safe_pct: <float or null>   # NEW
    holdout_roi_at_r_hard_pct: <float or null>  # NEW
    holdout_dd_at_r_hard_pct: <float or null>   # NEW
    sizing_convention: reset_floor | equity_pct # NEW — gate FAILs equity_pct unless chat approves
```

---

## Holdout window

Holdout = "2021-01-01 to present at time of arc closure" — matches current §2 Step 5 and Appendix B definition. Moving target by design.

---

## r_max note

`r_max = 2.0%` is the gate ceiling. Locked value.

---

## Application to in-flight work

Forward arcs use amended gate from first run. Closures produced before this amendment lands are re-evaluated at chat discretion — none currently in scope.

---

## Action items

1. Land amendment into `L_PROTOCOL.md` §3. **[DONE 2026-05-22]**
2. Update `ARC_CLOSURE_TEMPLATE.md` v1.0 → v1.1 with rename map + new fields. **[DONE 2026-05-22]**
3. Update parser for v1.0/v1.1 schema detection. (Parser not yet built — spec updated in template §5.)
4. Update Step 5 closure-writer + verdict logic with priority-ordered gate evaluation.
5. Add per-day max-DD parquet emission to engine.
6. Add chained max DD emission to engine if not already present.
7. Add holdout re-run at scaled risk(s) to candidate evaluation pipeline, with separate manifest entries.
8. Add sizing-convention check to gate (FAIL if `equity_pct` without chat-approved alternative scaling).
9. **A5 follow-up amendment** — separate amendment defining combined-portfolio DD constraints for VIABLE components. Deferred until first VIABLE candidate emerges.

---

## Risks

1. **Per-day max-DD granularity.** Engine must compute per-day at M1 simulation tick (or finer); verify no coarser aggregation.
2. **Sizing-convention silent invalidation.** Future arcs switching to equity-pct must FAIL by default at the gate.
3. **A5 portfolio combined DD.** Open follow-up item, not blocker for this amendment.

---

End of v5.
