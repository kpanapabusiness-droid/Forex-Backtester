# AUDIT_ARC_9_TRADEABILITY — 2026-05-19

> Verdict: **NOT-TRADEABLE** at 1% risk
> Subject: Arc 9 Candidate A (LightGBM Pipeline E, threshold 0.40, 28 features)
> Risk parameter: 1.0% per trade
> Audit branch: `audit/arc-9-tradeability`
> Anchor preserved: KH-24 K=4 archetype 3 — yes (does not use the contaminated feature; not at risk; Phase 11 not run per short-circuit because the audit decided on Phase 1)

## Headline

Arc 9 Candidate A is NOT-TRADEABLE at 1% risk because the top-importance feature `d1_bars_since_swing_low` and the second-importance feature `d1_bars_since_swing_high` (combined 27.0% of total LightGBM gain) are computed using D1 bars that occur AFTER the 4H signal timestamp. The lookahead is structural and confirmed empirically: 18/50 random Arc 9 trades (36%) show a materially different feature value when the D1 history is truncated to signal-date − 1 day vs the producer's full-history computation. The classifier's AUC 0.7508 and the Step 5 economics (worst-fold ann ROI +9.63% / DD 1.32% at 0.5% risk, extrapolated to +20.07% / 2.52% at 1.0% risk) are therefore not valid measurements of live-equivalent classifier behaviour.

## Phase results

| Phase | Sub-phase | Verdict | Key number | Evidence file |
|---|---|---|---|---|
| 0 | reproducibility | PASS | 9/9 Step 5 LGBM E + 6/6 Pipeline E retry artefacts byte-identical (LF-normalized); `summary.json` raw-byte-identical | `verdict_evidence/phase_0/phase_0_reproducibility.md`, `sha256_manifest.txt` |
| 1A | per-feature dependency cards | PARTIAL FAIL | 26/28 features clean; `d1_bars_since_swing_{low,high}` reference D1 bars after signal date | `verdict_evidence/phase_1/phase_1_lookahead.md` §1A |
| 1B | merge_asof verification | FAIL on joined-row content | join lag ≥ 1 day verified (matches producer's own audit on 560 samples); features inside the joined row reference future D1 bars via the swing-confirmation mechanism | `verdict_evidence/phase_1/phase_1_lookahead.md` §1B |
| **1C** | **swing-low windowing** | **FAIL** | **18/50 trades (36%) leak; median magnitude +26 bars; max +88 bars; combined contaminated-feature share 27.0% of total LGBM gain** | `verdict_evidence/phase_1/phase_1_lookahead.md` §1C, `d1_bars_since_swing_low_leak_test.csv` |
| 1D | permutation null | not run | short-circuit | — |
| 1E | top-feature ablation | not run | short-circuit | — |
| 1F | top-5 ablation | not run | short-circuit | — |
| 2 | population integrity | not run | short-circuit | — |
| 3 | label construction | not run | short-circuit | — |
| 4 | classifier robustness | not run | short-circuit | — |
| 5 | threshold integrity | not run | short-circuit | — |
| 6 | WFO integrity | not run | short-circuit | — |
| 7 | cost truth | not run | short-circuit | — |
| 8 | 1% risk scaling | not run | short-circuit | — |
| 9 | 5ers rule compliance | not run | short-circuit | — |
| 10 | KH-24 concurrency | not run | short-circuit | — |
| 11 | anchor preservation | not run | short-circuit | — |
| 12 | forward-walk | not run | short-circuit (and would have been data-limited: 4H data ends 2025-12-31, pool ends 2026-01-31; no post-pool data exists locally) | — |

The dispatch's hard short-circuit rule states: "If Phase 1A finds lookahead, stop." Phase 1B and 1C are individually audit-fatal per the dispatch's pass criterion ("Sub-phases A, B, C, D fail the audit if they fail."). The audit therefore stops at Phase 1 and records the verdict against the failed sub-phases.

## Failures

### Phase 1C — Swing-low windowing (primary fail)

**What the protocol requires.** A feature observed at the 4H signal bar's close must depend only on bars whose timestamp is strictly less than the 4H signal bar's open. For D1-lagged features (the protocol's allowed exception under the one-day-lag convention), the joined D1 row must itself contain only information available at or before the prior calendar day's D1 close.

**What the producer's code does.** `scripts/l_arc_9/experiments/pipeline_e_retry.py:_d1_swing_high_low` computes a centred ±10-bar swing detector: bar k is a swing iff its extreme exceeds bars k±10. `_build_d1_feature_frame` then applies a "confirmation" mask of `confirmed[n - 10:] = False` where `n` is the entire D1 history length. The mask zeroes only the LAST 10 bars of the full frame, not "the last 10 bars relative to each signal's join date".

**The empirical consequence.** For any signal whose joined D1 bar lies more than 10 bars before the end of the full D1 frame (essentially every Arc 9 trade), the swing flag at the join point was computed using D1 bars from after the signal date. `_bars_since` then reports a small number when there's been a recent — but not yet live-confirmable — swing.

Concrete numbers from the 50-sample test seeded at `4242`:

| Statistic | Value |
|---|---|
| Trades leaked (different live vs prod value) | 18/50 (36%) |
| Among leaked — min delta (live − prod) | +11 bars |
| Among leaked — median delta | +26 bars |
| Among leaked — max delta | +88 bars |
| Direction of bias | live > prod always |

The bias is structural — production sees future swing-lows that haven't yet been confirmed at live signal time, so production reports smaller `bars_since` values; live reports the more conservative larger value.

**Feature importance impact.**

| Rank | Feature | importance_gain | share of total | Leaks? |
|---|---|---|---|---|
| 1 | `d1_bars_since_swing_low` | 5796.1 | **20.2%** | YES |
| 2 | `d1_bars_since_swing_high` | 1960.9 | **6.8%** | YES (n=10 sample: 4/10 leak; same mechanism) |
| 3 | `swing_low_dist_atr` | 1589.0 | 5.5% | no |
| ... | ... | ... | ... | ... |
| Total (28 features) | — | 28701.7 | 100% | — |

**Combined leaked share: 27.0% of total LightGBM gain.**

The top feature alone is 3.65× the next-largest gain — matching the spec's "top feature dominates by 3× gain margin" assertion. That dominance is built on contaminated information.

### Why the producer's existing audit GREEN-ed this

`results/l_arc_9/experiments/lookahead_audit/AUDIT_REPORT.md` (commit `9dc4f8a`) Audit 2 verified the `merge_asof` join itself: pattern matches KH-24 engine, days_lag ≥ 1 across 560 samples, Monday boundary cases correctly find Friday's D1 (days_lag = 3). All of those checks are valid and still pass. But Audit 2 did not verify that the joined D1 row's *feature values* were themselves causally constructible from data ≤ the joined date. The bug lives one layer deeper than what was audited. The producer's GREEN verdict is correct at its scope; this audit's FAIL adds a layer the producer did not check.

### What the author wrote vs what the code does

`_build_d1_feature_frame` docstring (`pipeline_e_retry.py` lines 215-233) says:

> "For 4H bars looking at lag-1 D1, the most recent confirmable swing as of that lag is `min(lag1_idx - half, ...)`. We compute `bars_since` ON THE D1 FRAME using the confirmed swing flags (only flags where k <= n-half-1)..."

The first sentence describes the correct rule: confirmation must be relative to each signal's join point (`lag1_idx - half`). The second sentence describes what the code actually does: confirmation relative to the frame end (`n - half - 1`). The two diverge for every signal not within 10 bars of the frame end.

The author's continuation — "For most arc 9 trades (median entry several years into history), the confirm window is non-binding because there are always >half future bars in the D1 history. Only the last `half` D1 bars (most recent ~10 days) would have an unconfirmed-swing edge artifact" — has the logic inverted. "There are always > half future bars in the D1 history" is the *source* of the lookahead, not protection against it. At live signal time, those future bars do not yet exist, so the production-trained feature value is not reproducible live.

## How this propagates to the deployment claim

The Candidate A spec reports:

- TimeSeriesSplit(5) mean CV AUC: 0.7508 (per-fold 0.848380 / 0.735746 / 0.717263 / 0.714402 / 0.738041)
- Step 5 anchored-expanding refit per KH-24 fold, threshold 0.40, at 0.5% risk:
  - worst-fold ann ROI +9.63% (F3), mean +22.92%, worst DD 1.32%, all F2-F7 positive
- SCALED_RISK extrapolation to 1.0% risk:
  - worst-fold ann ROI +20.07%, mean +51.57%, worst DD 2.52%, worst-day DD 2.01%

All three sets of measurements share the same feature pipeline. Both training and OOS-evaluation feature matrices contain the leaked features. The TSS-CV AUCs, the Step 5 OOS admit decisions, the per-fold mean R values, the DD time series, and the SCALED_RISK projections all reflect the model's performance on lookahead-augmented inputs. In live deployment, those inputs would not exist; the classifier's probability outputs would shift, the admit/reject decisions at threshold 0.40 would differ, and the empirically-measured deployment metrics would change in directions this audit has no way to predict from the contaminated artefacts.

The producer audit's end-to-end probability reproduction check (Audit 8, 100/100 trades within 4.82e-11) confirms that **retraining is deterministic**, not that the trained classifier would behave identically on live-equivalent inputs. Reproducibility under identical (leaked) inputs does not exonerate the leaked-input problem.

## Recommendations

Arc 9 Candidate A is not deployable at 1% risk as currently constructed. Two structural paths exist:

**(a) Fix the feature pipeline and re-audit.** The minimal fix is one of:
- Replace `_d1_swing_high_low`'s centred window with a strictly one-sided lookback detector. Loses the "confirmed by subsequent recovery" semantic; gains live correctness by construction.
- Move swing confirmation into `_attach_d1_features` per-trade, using each trade's lookup_date as the right-edge cutoff. Preserves centred-swing semantic; constructs features causally per join.

After either fix, regenerate `pipeline_e_retry/per_fold_aucs.csv`, the threshold sweep, and `step5_lgbm_pipeline_e/candidate_A_thr0.40/`. The AUC will be lower (by some unknown amount; the two leaked features carry 27% of total gain) — possibly below the 0.65 Pipeline E gate. The Step 5 deployment metrics will be different and possibly insufficient for pass-deployable. Re-run this audit from Phase 0 on the corrected pipeline.

**(b) Accept Arc 9 Candidate A as not deployable.** The IB-trend signal class itself is not eliminated — Step 1 pool, Step 2 path-shape clustering, and Step 3 capturability (cluster_0_individual, fwd_mfe_p50 6.18R, frac_wrong_way 0.0, mean R +4.42R t=21.81) are all computed from forward path geometry and remain valid characterisations of a real cohort. What's killed is the *Pipeline E classifier extractability claim* that this cohort can be identified at entry time. The official Arc 9 closure was already STEP_4_KILL on the original Step 4 RF AUC 0.51 (no D1-lagged features). The held-open Pipeline E retry experiment introduced the leaky features and produced the AUC 0.7508 result — that result is now invalidated.

The audit does NOT recommend deploying at a lower risk level. Lower risk doesn't fix the lookahead; it just scales down the leverage on the same broken signal extraction.

## What is safe and what is not

| Item | Status |
|---|---|
| KH-24 live deployment | UNAFFECTED. KH-24 uses signal `kb_exhaustion_bar`, no D1 swing-bars-since features. Phase 11 not run, but by static analysis the anchor doesn't touch the contaminated code path. |
| Step 1 IB-trend trade pool (2,153 trades) | Computed without the leaked features; remains valid as a population. The Phase 2 population integrity check was not run but no a-priori reason to suspect a problem. |
| Step 2 K=3 cluster_0 assignment (365 trades) | Computed from forward path-shape features (monotonicity, local_peaks, pullback, ttp_rel), no lookahead. Cluster identity is real. |
| Step 3 cluster_0_individual capturability (fwd_mfe_p50 6.18R, etc.) | Computed from forward observation bars (`is_held=0`) per §15a — these are post-hoc characterisations, not predictors. No lookahead in the population/characterisation sense; they describe the cohort's actual forward behaviour. |
| Original Step 4 RF Pipeline E AUC 0.511 (closure verdict KILL) | UNAFFECTED. Original Step 4 used 16 features (8 base + 8 arc-specific 4H) with no D1-lagged features. No leak path. |
| **Pipeline E retry AUC 0.7508 and Step 5 Candidate A economics** | **INVALIDATED** by this audit. |
| SCALED_RISK extrapolation to 1.0% risk | INVALIDATED (depends on the same contaminated Step 5 admit set). |
| Producer-branch lookahead audit GREEN verdict | Correct at its scope; does not exonerate this audit's finding. |

## Cross-arc carry-forward

Two protocol-level items this audit surfaces:

**Open-Phase-0-prereq** — *Feature-causality audit must precede or accompany every classifier-bearing arc.* The producer's existing lookahead audit (audit_2_d1_lag_code_review.md) was sufficient as a *join-pattern* check but insufficient as a *feature-causality* check. A protocol amendment could mandate per-feature live-equivalent-vs-production comparison on a sample of trades for every D1-lagged or path-derived feature, similar to this audit's leak-test methodology.

**Open-swing-detector-pattern** — *Centred swing detectors are easy to misuse.* This is the second arc to flirt with this pattern (the project history flag about same-day D1 close was different mechanism, same family of bugs). A protocol convention could state: "Swing/peak/trough detectors used as classifier features must be one-sided lookback-only, or implemented per-join with explicit cutoff." Centred detectors are fine for offline characterisation (Step 2 path-shape features computed over the full held window) but not for entry-time predictors.

Both items are recorded here for cross-arc calibration consideration. Per dispatch discipline, neither softens the Arc 9 verdict.

## Appendix — Evidence manifest

```
results/l_arc_9/audit/
├── audit_intent.md
├── AUDIT_ARC_9_TRADEABILITY.md  ← this document
└── verdict_evidence/
    ├── phase_0/
    │   ├── phase_0_reproducibility.md
    │   ├── sha256_manifest.txt
    │   ├── pe_retry_rerun/      (15 files, all reproducible from script)
    │   └── step5_rerun/         (4 files, all reproducible from script)
    └── phase_1/
        ├── phase_1_lookahead.md
        ├── d1_bars_since_swing_low_leak_test.csv  (50 trades, prod vs live)
        └── leak_test_summary.txt
```

The pe_retry_rerun/ and step5_rerun/ subfolders contain the re-run scratch outputs from Phase 0 — they are byte-identical (after LF normalization) to the committed `results/l_arc_9/experiments/` artefacts and are not separately committed (reproducible from the scripts on the audit branch).

## Document control

| Field | Value |
|---|---|
| Audit run date | 2026-05-19 |
| Audit branch | `audit/arc-9-tradeability` |
| Branch cut from | `origin/main` @ `fb2e7ab` |
| Arc 9 artefact source | `claude/bold-brattain-d79817` (commits `0193334`, `44de1ca`) |
| Candidate A spec source | uploaded by chat to `results/l_arc_9/ARC_9_CANDIDATE_A_SPEC.md` |
| Risk parameter | 1.0% per trade (audit override) |
| Subject | Candidate A only (Candidate B not audited per chat decision) |
| Verdict | **NOT-TRADEABLE** at 1% risk |
| Killing phase | Phase 1 (1B + 1C) |
| Phases not run | 1D, 1E, 1F, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 — short-circuit per dispatch |
| Anchor preservation status | Not directly tested (Phase 11 not run); static analysis says KH-24 doesn't use the contaminated feature path; anchor not at risk from this finding |
| Auditor | Claude Opus 4.7 (1M context) via Claude Code |
