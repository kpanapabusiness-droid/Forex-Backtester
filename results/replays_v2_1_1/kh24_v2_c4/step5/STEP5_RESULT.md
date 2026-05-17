# KH-24 v2.0 self-test — Step 5 Cross-Fold Stability — RESULT

## Status

Step 5 cross-fold stability per L_ARC_PROTOCOL v2.1.2 §9 on Step 4 survivors
c1 (Stepwise climber) and c4 (Stepwise climber under v2.1.2 ceiling extension,
patched in commit 4e18d67).

Both classifier targets are cluster membership (policy-independent). Per-fold
final_r and fold_ROI computed under engine-default exit (hard SL 2.0×ATR +
kijun_d1 + 240-bar cap) as proxy for §11 Stepwise climber exit policy
(MFE-lock at 1R + 0.75R trail). True §11-policy fold metrics require D1 PR 2;
deferred per §12.

**Verdicts:**
- **c1: FAIL §9** — all three gates fail (sign 3/7, size 4.20, DD 4.70)
- **c4: BLOCKED** — Step 4 §8 threshold sweep produced null admit_threshold;
  §9 admission cannot fire

Both clusters die at Step 5 in deployment-track terms. §10 multi-cluster
ship rule yields no shipment for this arc. Cross-arc portfolio archival
(Open-05) is the residual option.

## Cohort set

| Cluster | Archetype (v2.1.2) | n | R-frame | Step 4 D1 ROC-AUC | admit_threshold | chosen t |
|---|---|---|---|---|---|---|
| c1 | Stepwise climber | 365 | 2.0×ATR | 0.618 | 0.50 | 1 |
| c4 | Stepwise climber (5-50 ceiling) | 122 | 2.0×ATR | 0.615 | null | 2 |

WFO folds per `configs/wfo_kh24.yaml` (locked): 7 anchored expanding folds,
IS always starts 2019-01-01, OOS rolls 9 months from 2020-10-01 to 2026-01-01.

## Per-fold metrics — c1 (Pipeline D1 at t=1, admit_threshold=0.50)

| Fold | OOS window | OOS n | OOS pos | Admit n | Admit pos | final_r mean | t-stat | ROI (annualised %) | max DD (R) | sign |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2020-10 → 2021-07 | 69 | 25 | 21 | 8 | **+1.84** | +2.07 | **+25.9%** | 3.03 | **+** |
| 2 | 2021-07 → 2022-04 | 48 | 20 | 5 | 4 | -0.15 | -0.17 | -0.5% | 3.01 | − |
| 3 | 2022-04 → 2023-01 | 35 | 16 | 9 | 6 | +0.80 | +0.65 | +4.8% | 3.00 | **+** |
| 4 | 2023-01 → 2023-10 | 44 | 17 | 17 | 8 | **-0.66** | -2.45 | **-7.5%** | **14.14** | − |
| 5 | 2023-10 → 2024-07 | 35 | 15 | 6 | 5 | +0.80 | +0.97 | +3.2% | 2.01 | **+** |
| 6 | 2024-07 → 2025-04 | 42 | 22 | 11 | 6 | -0.09 | -0.19 | -0.7% | 3.01 | − |
| 7 | 2025-04 → 2026-01 | 41 | 18 | 18 | 11 | -0.25 | -0.61 | -3.0% | 6.32 | − |

3/7 folds positive (F1, F3, F5). 4/7 negative. F4 is the disaster fold: -0.66R mean, -7.5% ROI, 14.14R drawdown — driven by 17 admitted trades, 8 true positives but the other 9 false positives evidently dominated. The admitted-n/positive-n ratio in F4 (17 admitted vs 17 OOS positives total, but only 8 of admitted were true positives) shows classifier overfitting to IS — at this fold the model admits roughly half false positives. F7 is also bad (18 admitted, 11 positives, but final_r mean -0.25 means the positives weren't profitable under engine-default exit either).

## Per-fold metrics — c4 (Pipeline D1 at t=2, admit_threshold=null)

**N/A — cluster BLOCKED.**

`fold_stability_c4.csv` written as schema-conformant empty CSV. The c4
classifier (RF ROC-AUC 0.615) cleared the §8 AUC gate at Step 4, but the
threshold sweep produced no candidate satisfying recall ≥ 0.60. Without an
admit_threshold, §9 admission cannot fire — there's no decision rule to apply
to OOS trades.

## §9 gate evaluation

| Gate | Threshold | c1 | c4 |
|---|---|---|---|
| Sign consistency (all folds positive final_r_mean) | all 7 | **FAIL (3/7)** | blocked |
| Size variance (max-fold / min-fold) | ≤ 3.0 | **FAIL (4.20)** | blocked |
| DD ceiling (worst-fold / median-fold) | ≤ 2.0 | **FAIL (4.70)** | blocked |
| **Overall §9** | conjunctive | **FAIL** | **BLOCKED** |

c1 fails all three gates decisively. Not a single gate is close to passing —
sign consistency is 3/7 (need 7/7), size variance is 40% over the threshold,
DD ceiling is more than 2× over the threshold.

## Per-pair stability (informational)

c1: 27 distinct pairs admit at least one trade. Top pair CAD_CHF at 10.3% share;
top 5 pairs (CAD_CHF, EUR_GBP, EUR_CHF, USD_CHF, USD_CAD) hold 44.8% combined —
below the §9 "> 50% in < 5 pairs" concentration flag. **Concentration flag does
NOT fire.** The cluster is geographically diverse; instability is temporal
(regime-dependent), not pair-concentrated.

c4: blocked — no admitted trades to analyse.

## Discipline notes triggered

Per §9: "Single-fold-outlier sign flips → chat-level judgement." c1 has 4
negative folds and 3 positive folds — not a single-fold outlier, this is
sustained instability. Not a chat-level escalation; a real failure.

Per §9: "Trade-count variance alone (without sign flip or DD blowup) = not a
kill." Both sign flips AND DD blowup present for c1, so size variance is
not the determining factor — c1 would fail even without the size-variance gate.

## Proxy framing reminder

Engine-default exit (hard SL 2.0×ATR + kijun_d1 + 240-bar cap) was used as
proxy for §11 Stepwise climber exit (MFE-lock at 1R + 0.75R trail from new
high). Step 6 D1 WFO under true §11 policy is blocked on D1 PR 2 backtester
extension per §12.

This proxy is conservative-to-realistic for failure judgments: a Stepwise
climber exit would lock 1R when MFE reaches 1R, then trail. Trades that
reached 1R MFE and then collapsed (the path-shape pattern these clusters
are defined by) would lock the 1R gain — likely turning some engine-default
losses into +1R wins. So c1's true §11 Stepwise-exit performance is likely
**better** than the proxy here suggests, but not enough to flip 4 losing
folds into wins, given the magnitude of F4's drawdown and the -7.5% ROI.

It would take an unusually optimistic §11-exit assumption to rescue c1's
Step 5 verdict. The right diagnostic interpretation: c1 is real-edge under
ideal conditions (F1: +25.9% ROI) but unstable across the recent regime
(F4-F7 all negative under engine-default).

## Verdict + next steps

**c1: §9 FAIL.** Per §9, "Designs failing stability killed before Step 6."
c1 does not proceed to Step 6 WFO under v2.1.2 protocol.

**c4: §9 BLOCKED.** The Step 4 null-threshold caveat compounds with §9 — c4
cannot be evaluated for stability without an admission rule. Three resolution
paths exist:
1. Loosen recall floor in Step 4 threshold sweep (calibration question)
2. Expand threshold candidate set in Step 4 (calibration question)
3. Accept c4 as non-deployable on Pipeline D1

All three are chat-side calibration decisions; not auto-resolvable.

**Arc-level outcome.** No cluster ships from KH-24 v2.0 under the v2.1.2
protocol on this analysis pass. Per §10 multi-cluster ship rule: arc dies
at Step 5; no Step 6 dispatch. Residual option per Open-05: c1 and c4 archive
as portfolio candidates (`archived_candidate_c1.yaml`, `archived_candidate_c4.yaml`)
for future cross-arc composition.

**Likely cross-arc takeaway** (for chat-side synthesis, not auto-actioned here):
the rescue of c4 under v2.1.2 amendment was directionally correct at Step 3
(capturability passes), but Step 4 D1 classification was marginal (PR-AUC 0.24
just under "meaningful" threshold) and Step 5 stability fails. The v2.1.2
amendment is anchor-preserving but doesn't structurally rescue the KH-24 v2.0
self-test arc. KH-24 v1's filtered deployed population may still be the only
stable form of this signal — the bare-signal v2.0 pool does not pass §9 under
the current §11 Stepwise climber exit policy.

## Files

All under `results/replays_v2_1_1/kh24_v2_c4/step5/`:

- `fold_stability_c1.csv` — per-fold metrics for c1 (7 rows)
- `fold_stability_c4.csv` — empty (cluster blocked)
- `stability_pass_list.csv` — per-cluster §9 gate results + overall verdict
- `pair_stability_c1.csv` — 27-pair share table for c1
- `pair_stability_c4.csv` — empty (cluster blocked)
- `pair_stability_summary.csv` — concentration flag info per cluster
- `STEP5_RESULT.md` (this file)

Step 4 patch commit: 4e18d67 (v2.1.2 alignment)
Step 5 commit: filled in post-commit
