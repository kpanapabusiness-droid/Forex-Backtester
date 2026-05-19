# Arc 9 Tradeability Audit — Intent

> Branch: `audit/arc-9-tradeability` (cut from `origin/main` @ `fb2e7ab`).
> Dispatch: independent tradeability verification of Arc 9 Candidate A at **1.0% per-trade risk**.
> Scope: Candidate A only. Candidate B explicitly out of scope (no spec, audit-decided 2026-05-19).
> Disposition rule: hard short-circuit on first audit-fatal phase fail.
> Verdict format: binary TRADEABLE / NOT-TRADEABLE at 1% risk. No third option.

---

## Confirmation of understanding

I have read in order:

1. `L_ARC_PROTOCOL.md` (v2.1.2 in repo)
2. `L_ARC_PROTOCOL_v2_3_AMENDMENT.md` (DRAFT, governs Arc 8+; the document Arc 9 was opened under)
3. `results/l_arc_9/ARC_9_CANDIDATE_A_SPEC.md` (the audit subject)
4. `docs/SPREAD_FLOOR_AUDIT_FINDING.md`, `docs/calibration_decisions/SPREAD_FLOOR_CALIBRATION_DECISION_2026-05-17.md`, `docs/SPREAD_SEMANTICS_LOCK.md`
5. `docs/arc_results/ARC_4_RESULT.md` and `docs/arc_results/ARC_4_RERUN_RESULT.md` (cost-truth precedents)
6. `docs/KH24_SYSTEM_LOCK.md` (anchor)
7. `scripts/l_arc_9/step1_plumbing.py` and `scripts/l_arc_9/experiments/pipeline_e_retry.py`
8. The Pipeline E retry source on `claude/bold-brattain-d79817` (commits `0193334`, `44de1ca`)
9. `results/l_arc_9/ARC_9_CLOSURE.md`, `results/l_arc_9/experiments/step5_lgbm_pipeline_e/STEP5_LGBM_E_RESULT.md`, `results/l_arc_9/experiments/scaled_risk/SCALED_RISK_RESULT.md`, and the existing producer-branch lookahead audit at `results/l_arc_9/experiments/lookahead_audit/AUDIT_REPORT.md` (reference only — Phase 1 re-audits independently).

## Target system under audit

| Field | Locked value |
|---|---|
| Signal | IB-trend (`signals.lchar_inside_bar_break_trend_long`, 4H, long-only, 28 FX) |
| Target cluster | `cluster_0_individual` (Step 2 K=3, cid=0) |
| Pipeline | Pipeline E (entry-time admission, LightGBM) |
| Classifier hyperparameters | `n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31, min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1, class_weight='balanced', random_state=42, deterministic=True, force_row_wise=True` |
| Feature set | 28 features (16 baseline + 8 D1-lagged + 4 session) |
| **Admission threshold** | **0.40** (locked grid {0.40, 0.50, 0.60, 0.70}; v2.2 §3 max-precision with recall ≥ 0.60 selection) |
| §11 exit | Stepwise climber Pipeline E — MFE-lock at 1R, trail 0.75R from peak_close |
| Selected SL | 2.0 × ATR(14)_4H at signal bar (= cluster_0_individual Step 3 selected SL) |
| **Per-trade risk** | **1.0%** (audit override; not the L-arc-default 0.5%, not Arc-4's 0.20%) |
| WFO structure | 7 KH-24-anchored expanding folds, OOS Oct 2020 – Jan 2026 |
| Spread treatment | `configs/spread_floors_5ers.yaml` p50 per-pair (sha256 `8da7644b…`); raw MT5 per-bar when non-zero, p50 floor when raw=0 |
| Pool generator | `build_ex_ante_bounded_population` semantics via `scripts/l_arc_9/step1_plumbing.py` |

## Reproduction parity figures (locked targets for Phase 0)

Per-fold TimeSeriesSplit(5) CV AUCs from Pipeline E retry (`lgbm_expanded_28`):

- F1: 0.848380
- F2: 0.735746
- F3: 0.717263
- F4: 0.714402
- F5: 0.738041
- Mean: 0.7508 (reproduced 0.750766 in Step 5 parity check, within 1e-4)

These are TSS(5) CV folds — chronological training-time CV. They are NOT the KH-24 7-fold WFO folds. Step 5 LGBM E uses anchored-expanding refit per KH-24 fold; Pipeline E retry uses TSS(5) for training-time AUC.

Step 5 Candidate A per-fold deployment (at 0.5% risk in producer output):

| Fold | OOS | trades | mean R | ann ROI % | max DD % |
|---|---|---|---|---|---|
| F1 | 2020-10-01 → 2021-07-01 | 0 (data-window artefact) | — | 0.00 | 0.00 |
| F2 | 2021-07-01 → 2022-04-01 | 50 | +0.977 | +38.10 | 1.01 |
| F3 | 2022-04-01 → 2023-01-01 | 24 | +0.580 | +9.63 | 0.50 |
| F4 | 2023-01-01 → 2023-10-01 | 32 | +0.725 | +16.72 | 0.51 |
| F5 | 2023-10-01 → 2024-07-01 | 39 | +0.612 | +17.12 | 1.32 |
| F6 | 2024-07-01 → 2025-04-01 | 44 | +0.804 | +26.40 | 0.79 |
| F7 | 2025-04-01 → 2026-01-01 | 47 | +0.835 | +29.57 | 0.52 |

Total admits 236 / 2,153 OOS signals (10.96%).

## Risk-parameter framing

```
risk_per_trade = 0.01   # 1.0% — KH-24-parity, NOT the L-arc-default 0.5%
```

All WFO re-runs, DD computations, daily-DD checks, 5ers rule simulations, and Phase 7 cost reconciliation operate at 1% risk. The `SCALED_RISK_RESULT.md` already measured Candidate A at 1.0% risk on the existing admit set (236 trades, F2-F7); the audit re-verifies and stress-tests these measurements, it does not blindly accept them. Linearity of DD scaling is **rejected as a default assumption** and re-verified per Phase 8.

## Phase status entering Phase 0

| Phase | Status | Disposition rule |
|---|---|---|
| 0 — Reproducibility | not run | If sha256 of trained-classifier outputs and per-fold AUCs mismatch the spec figures, audit halts. |
| 1 — Lookahead | not run | A/B/C/D each individually audit-fatal. E individually pass-criterion. F informational. |
| 2 — Population | not run | Outcome-aware filter, pre-2026-05-17 spread floor, or >25% per-pair concentration each kills. |
| 3 — Labels | not run | 3A informational. 3B and 3C each individually fail. |
| 4 — Classifier | not run | 4B p10 < 5%, 4C p10 < 5% or p90−p10 > 4pp, 4D cold-block any-criterion-fail each fail. 4A explanation requirement. |
| 5 — Threshold | not run | Threshold 0.40 must derive strictly from protocol §3 sweep. (Candidate B not audited.) |
| 6 — WFO | not run | Worst-fold ROI ≥ 5% / DD ≤ 8% / no daily DD breach / sign-consistent at 1% risk on the primary fold set (per the F1 convention chosen in 6B). |
| 7 — Cost truth | not run | Combined 7A+7B+7C worst-fold ROI < 5% kills. Most likely failure point given Arc 4 precedent. |
| 8 — Risk scaling | not run | 1% risk must fit inside safe envelope with margin. |
| 9 — 5ers compliance | not run | Zero daily DD breaches; max all-time DD ≤ 8%; max daily DD ≤ 4%. |
| 10 — KH-24 concurrency | not run | Joint behaviour must remain compliant with 5ers rules under a documented joint exposure rule. |
| 11 — Anchor | not run | KH-24 K=4 archetype 3 numbers must reproduce within ±0.5pp under the audit's reconstructed pipeline; otherwise the audit framework has a bug. |
| 12 — Forward-walk | data-limited | Pool ends 2026-01-31. Local 4H data ends 2025-12-31. **No post-pool data exists locally for forward-walk.** Phase 12 will report data-unavailability and the audit's rollup will treat this as a verdict caveat. |

## Critical context from prior closures

- **Arc 4 RERUN structural lesson** (`docs/arc_results/ARC_4_RERUN_RESULT.md`): Pipeline D1 admit-only economics ≠ deployment economics due to mandatory reject-pool and early-exit-pool costs. **Arc 9 uses Pipeline E, not D1.** Pipeline E does not have a reject pool — rejected signals never enter the book. The Arc 4 RERUN failure mode does not directly transfer to Arc 9. **The cost-truth concern remains (Phase 7).**
- **Arc 9 official disposition** (`results/l_arc_9/ARC_9_CLOSURE.md`): **STEP_4_KILL.** Original Step 4 RF AUC = 0.511 (Pipeline E) and 0.626 at t=1 (Pipeline D1, but threshold-sweep recall maxed at 0.003). **Candidate A is from a post-closure held-open experiment** (Pipeline E retry: LightGBM + 12 additional features). The Candidate A spec's status "PASS-DEPLOYABLE-PENDING-AMENDMENT" depends on (1) lookahead audit pass, (2) v2.x §3 grid amendment or locked-grid acceptance, (3) Step 5 fold-1 warmup convention or data backfill, and (4) formal Step 4 re-pass under amended protocol. This audit is what (1) tests.
- **Same-day-D1 historical bug:** Pre-fix KGL WFO results were invalidated by a Python bug using same-day D1 close. The Arc 9 `pipeline_e_retry.py` D1 lag uses `merge_asof(direction='backward')` on `signal_date − 1 day`, the KH-24 engine's pattern. Phase 1B re-verifies this independently.

## Verdict ground rules

- TRADEABLE at 1% risk requires Phase 0, 1A-D, 1E, 2, 3B, 3C, 4A, 4B, 4C, 4D, 5, 6A, 6B, 6C, 7D, 8, 9, 10, 11 all to pass. Phase 1F and 3A are informational. Phase 12 is data-limited and recorded as a caveat.
- NOT-TRADEABLE at 1% risk is recorded if any individually-fatal phase fails. The verdict doc names the failing phase, its evidence, and explicitly lists which downstream phases were not run.
- "TRADEABLE pending X" and "TRADEABLE at lower risk" are NOT permitted as audit outcomes. The audit may separately recommend a lower-risk follow-up analysis in its Recommendations section, but the verdict at 1% risk stands as binary.

## Discipline acknowledgements

- The audit's job is to find holes, not confirm pre-existing optimism. Every phase is a hostile witness.
- An ambiguous result is a fail; ambiguity does not get the benefit of the doubt.
- Per-fold TSS-CV AUC ordering (0.848 → 0.738 across folds 1→4, monotone-descending where smaller-training-set folds usually have LOWER OOS AUC) is either explained or it is a Phase 4A fail.
- Single-feature 3× gain dominance on `d1_bars_since_swing_low` is either ablation-robust (Phase 1E) or it is a fail.
- If a methodology hole in the protocol itself is found, it is logged as a cross-arc calibration item but does not soften the Arc 9 verdict.

## Out-of-scope or downscoped

- **Candidate B (threshold 0.05).** No `ARC_9_CANDIDATE_B_SPEC.md` exists on any branch. Per audit-time chat decision, Candidate B is not audited. The Phase 5 Bonferroni cross-candidate adjustment is therefore not needed; threshold-sweep-internal multiple-testing for Candidate A still runs.
- **Phase 12 forward-walk.** Local 4H data ends 2025-12-31, pool ends 2026-01-31, so no post-pool data exists locally. Phase 12 will execute on whatever sliver may be available; otherwise it records "not auditable, no data" as a verdict caveat. The audit does not pull external data.

---

Audit run started 2026-05-19 from `audit/arc-9-tradeability` branch HEAD `924cfe9`.
