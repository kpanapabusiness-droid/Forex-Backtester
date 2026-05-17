# Arc 6 — Failed-breakout reversal (long) — RESULT

## Status

- **Opened:** 2026-05-17
- **Closed:** 2026-05-17 (same-day; CC ran Steps 1–4 consecutively)
- **Active protocol:** `L_ARC_PROTOCOL.md` v2.1.2
- **Calibration anchor:** KH-24 K=4 archetype 3 (passes via Pipeline D1 at t=3)
- **Verdict:** **DIES at Step 4 — deployability-level failure on Pipeline D1 recall collapse**
- **Branch:** `discovery/lomega_regime_conditional`

## Final disposition

Arc 6 produces a mechanical PASS at the §8 gate (Pipeline D1 RF AUC ≥ 0.60 for both surviving clusters) but a substantive FAIL on deployability.

Both surviving clusters fail Pipeline E (best AUC 0.600, 0.590 vs 0.65 floor). Both clear the Pipeline D1 AUC ≥ 0.60 floor mechanically but their threshold sweeps fall back to max-F1 because no threshold achieves the §8 design-intent recall ≥ 0.60. Resulting admission rates: 0.9% (c0) and 0.4% (c2), or ~3–4 trades across the entire 5-year, 28-pair, 1,564-trade pool.

WFO at Step 6 would see ≤ 1 trade/fold — three orders of magnitude below PASS-VIABLE (5/fold). Step 5 stability with that admission rate is noise-dominated and provides no decision-relevant information.

Steps 5 and 6 not executed. Arc dies at Step 4 deployability.

## Signal under test

| Field | Value |
|---|---|
| Origin | Structural signal spec (not LCHAR registry) |
| Spec | `signal_spec_failed_breakout_long_v0.1.md` |
| Family | Failed-breakout reversal at swing low |
| Direction | Long only |
| Signal TF | 4H |
| Trigger | Closed-bar break below 20-bar swing low (magnitude ≥ 0.25 × ATR), reclaim within 5 bars on a bullish-close bar |
| Multi-TF | 4H only |

## Arc-6 backtester configuration

| Field | Value |
|---|---|
| Direction | Long only |
| Pair set | All 28 FX (KH-24 list) |
| Entry | Bar t+1 open per `SPREAD_SEMANTICS_LOCK` |
| Initial SL (simulation) | `entry_price − 2.0 × ATR(14)_4H[t]` |
| Forward window | 240 bars |
| Exposure cap | Max 1 open position per pair |
| Risk per trade | 0.5% × reset floor balance |
| Spread | `configs/spread_floors_5ers.yaml` |
| Data window | 2020-10-01 → 2026-01-31 |
| Arc config | `configs/wfo_l_arc_6.yaml` |

### Signal parameters (defaults; no sweeps performed)

| Parameter | Value |
|---|---|
| N (swing lookback) | 20 bars |
| M (reclaim window) | 5 bars |
| Breakout magnitude floor | 0.25 × ATR(14) |
| ATR period | 14 |

## Step results

| Step | Gate | Result | Notes |
|---|---|---|---|
| 1 | Plumbing | **PASS** | Pool 1,564 / 28 pairs; det e57528; KH-24 cofire 0 (structural); no cost flags; cap-bind 17.65% |
| 2 | Path-shape clustering | **PASS** | K=4, silhouette 0.4795; 1 Stepwise (c2) + 1 early_peak (c1) + 2 unassigned (c0, c3) (54% of pool) |
| 3 | Capturability | **PASS** (2/4) | c2 (Stepwise, SL=3.0×ATR, composite 0.616); c0 (Stepwise-boundary, SL=2.0×ATR, composite 0.384); c1, c3 die |
| 4 | Extractability | **PASS mechanically / FAIL substantively** | Both clusters fail E (best 0.600, 0.590); both clear D1 AUC ≥ 0.60 floor but recall sweep collapses to max-F1 fallback at 0.4–0.9% recall |
| 5 | Stability | NOT EXECUTED | Step 4 deployability failure makes Step 5 noise-dominated |
| 6 | WFO + disposition | NOT EXECUTED | Same reason; pool would be ≤ 1 trade/fold |

## Detailed analysis

### Step 1 — Plumbing (PASS)

Pool 1,564 trades across 28 FX pairs. Determinism hash `e57528...`. Below the spec's prior estimate range (1,800–5,500) but well above the §5 floor (500). All six validation checks passed.

Bars-held distribution bimodal: p25=7, p50=21, p75=107, p95=240. Cap-binding 17.65% (below the 20% §5 auto-extend threshold). Exit reasons: 1,288 sl_hit / 276 bar_240_system_exit.

KH-24 co-fire rate = 0.0000 by structural exclusion: Arc 6 requires `close[t] > open[t]` (bullish reclaim) while KH-24 C1 long requires `close[t] < open[t]` (bearish exhaustion). Mechanical independence is guaranteed by construction, not measured empirically. Cost-clearance: no pairs flagged.

**Spec erratum locked:** `swing_low_N = min(low[t-N-M..t-M-1])` rather than the literal spec `min(low[t-N..t-1])`. The literal definition is mathematically unsatisfiable: with `t* ∈ [t-M..t-1] ⊂ [t-N..t-1]`, the breakout-bar `close[t*] < swing_low_N` cannot hold because `low[t*] ≥ swing_low_N` by definition of min, and `close[t*] ≥ low[t*]` by OHLC. CC's reading anchors the swing low to bars strictly preceding the breakout window; this preserves the N=20-bar swing evidence and is consistent with the structural story. Spec v0.2 erratum required.

### Step 2 — Path-shape clustering (PASS)

K=4 selected, silhouette 0.4795. All five K ∈ {3, 4, 5, 6, 7} admissible (silhouettes 0.45–0.48). K=4 wins on highest silhouette (margin 0.0092 over K=6, just inside the 0.01 parsimony tolerance).

| Cluster | n | Frac | mono | peaks | pullback | ttp_rel | Label |
|---|---|---|---|---|---|---|---|
| 0 | 334 | 0.214 | 0.507 | 8.64 | 0.615 | 0.555 | unassigned (Stepwise-boundary) |
| 1 | 477 | 0.305 | 0.014 | 0.58 | 0.012 | 0.050 | early_peak_family |
| 2 | 242 | 0.155 | 0.535 | 32.52 | 0.353 | 0.754 | Stepwise climber |
| 3 | 511 | 0.327 | 0.554 | 4.64 | 0.105 | 0.328 | unassigned (multi-boundary) |

54% of pool labelled unassigned — c0 misses Stepwise only on pullback (0.615 vs ≤0.5 floor); c3 sits in the ttp_rel dead zone (0.328 — past the early-peak ceiling 0.30, well short of Stepwise/Monotone floor 0.50). No feature degeneracy.

v2.1.2 amendment matters: Stepwise climber centroid `local_peaks = 32.5` sits comfortably inside the extended [5, 50] range; under v2.1.1 priors (ceiling 30) it would have missed.

### Step 3 — Capturability (PASS, 2/4 clusters pass §2)

Shape_tag classifier inherited from `scripts/replays_v2_1_1/kh24_v2_c4/step3.py` (scattered_std=1.0, KDE bandwidth=scott, dip α=0.05). `pct_peak_and_collapse` threshold from `scripts/arc_3/step3_capturability.py` (final_r ≤ 0.4 × peak_mfe with peak_mfe ≥ 1.0R).

| Cluster | Disposition | Sel SL | Composite | mfe_p50_R | mono_pp | reach_1R | ww_pp | Shape | Key |
|---|---|---|---|---|---|---|---|---|---|
| 0 | PASS | 2.0×ATR | 0.384 | 1.69 | 0.604 | 0.731 | 0.000 | heavy_right_tail | All floors clear; no clean §11 routing |
| 1 | DIES | — | — | — | — | — | — | — | mfe_p50 max 0.29R at X=4.0 — structural, not narrow |
| 2 | PASS | 3.0×ATR | 0.616 | 4.47 | 0.566 | 1.000 | 0.000 | unclassified | Clean §11 row 2 routing |
| 3 | DIES | — | — | — | — | — | — | — | reach_1R 0.697 vs 0.70 by 0.003 |

**c0 routing decision (chat-level):** centroid mono=0.507, peaks=8.64, pullback=0.615, ttp_rel=0.555. Three of four features pass §11 row 2 (Stepwise climber); pullback misses by 0.115 — the narrowest of the boundary failures. ttp_rel=0.555 places it past the early-peak ceiling. Routed to row 2 with empirical trail-distance tuning carried forward.

**c2 SL selection (Open-17 expansion):** composites tied at X=2.0 and X=3.0 (both 0.6162). Tiebreak 1 (larger peak_mfe in ATR units) gave X=3.0 by 0.02 ATR (13.40 vs 13.38) — a 0.15% relative margin, well below measurement noise on ATR estimation, sampling, or float precision. Protocol-literal X=3.0 applied. Economic case for X=2.0 (~50% better capital efficiency at identical composite — same dollar risk per trade, ~1.5× the dollar MFE) is non-trivial but interpretive. Logged as Open-17 expansion item.

**c3 noise-margin failure:** reach_1R = 0.697 vs 0.70 floor — a 0.003 absolute margin. Well within sampling noise for a 511-trade cluster. Within-arc thresholds don't move per §1.8. Cross-arc question: does the reach_1R floor need a noise tolerance (e.g., `reach_1R ≥ 0.70 − 1.96 × se`)?

**c1 structural death:** early_peak_family path is dominated by trades that peak at offset 0–1 with mfe_p50 near zero. Even at the widest SL=4.0×ATR (R = 4×ATR, biggest possible mfe denominator), mfe_p50 reaches only 0.29R — magnitude not in the data. No SL-frame rescues it. Pipeline E or D1 won't help either.

### Step 4 — Extractability (PASS mechanically, FAIL substantively)

Feature catalogue: 8 cross-dataset base + 19 arc-specific (signal geometry, breakout-bar shape, pre-breakout structure, trend/regime, velocity, cyclic time-of-day) = 27 entry features. Path-so-far feature set for D1 = 8 base + 7 path = 15 features.

**Pipeline E fails for both clusters across all four investigation steps.**

c0 (Stepwise climber, n=334, R=2.0×ATR):

| Step | n_features | RF AUC | Logistic AUC | Gap | Clears 0.65 |
|---|---|---|---|---|---|
| A all | 27 | 0.5495 | 0.5397 | +0.010 | NO |
| B top-5 | 5 | 0.5971 | 0.5215 | +0.076 | NO |
| B top-10 | 10 | 0.5692 | 0.5308 | +0.038 | NO |
| B top-15 | 15 | 0.5596 | 0.5258 | +0.034 | NO |
| B forward | 6 | 0.5996 | 0.5396 | +0.060 | NO |
| C stack 1 | 26 | 0.5572 | — | — | NO |
| C stack 2 | 27 | 0.5586 | — | — | NO |

c2 (Stepwise climber, n=242, R=3.0×ATR):

| Step | n_features | RF AUC | Logistic AUC | Gap | Clears 0.65 |
|---|---|---|---|---|---|
| A all | 27 | 0.5546 | 0.5789 | −0.024 | NO |
| B top-5 | 5 | 0.5544 | 0.5580 | −0.004 | NO |
| B top-10 | 10 | 0.5600 | 0.5654 | −0.005 | NO |
| B top-15 | 15 | 0.5506 | 0.5755 | −0.025 | NO |
| B forward | 2 | 0.5897 | 0.5173 | +0.073 | NO |
| C stack 1 | — | 0.5484 | — | — | NO |
| C stack 2 | — | 0.5463 | — | — | NO |

**RF/Logistic gap interpretation:**
- c0: small positive gaps (+0.01 to +0.08) — feature set is binding more than non-linearity. Richer arc-specific catalogue might help, but the structural failure is unlikely to be feature-engineering-driven (best AUC 0.600 vs 0.65 target is a 0.05 gap; doubling the catalogue typically buys 0.01–0.03 on AUC at this regime).
- c2: small negative gaps (−0.005 to −0.025) on full / top-k subsets — logistic marginally outperforms RF. Relationship is roughly linear and shallow; no non-linear structure for RF's depth-8 trees to exploit.

**Pipeline D1 mechanically passes AUC gate.**

c0:

| t | RF AUC | Logistic AUC | Gap | Exclusion |
|---|---|---|---|---|
| 1 | 0.561 | — | — | 0.000 |
| 2 | 0.566 | — | — | 0.000 |
| 3 | 0.591 | — | — | 0.000 |
| **4** | **0.602** | 0.562 | +0.040 | 0.000 |
| 5 | 0.602 | — | — | 0.000 |
| 10 | 0.618 | — | — | 0.003 |

Smallest-t selection: t=4, AUC 0.602. Locked.

c2:

| t | RF AUC | Logistic AUC | Gap | Exclusion |
|---|---|---|---|---|
| **1** | **0.630** | 0.648 | −0.018 | 0.000 |
| 2 | 0.631 | — | — | 0.000 |
| 3 | 0.659 | — | — | 0.000 |
| 4 | 0.683 | — | — | 0.000 |
| 5 | 0.673 | — | — | 0.000 |
| 10 | 0.711 | — | — | 0.000 |

Smallest-t selection: t=1, AUC 0.630. Locked.

D1 exclusion essentially zero across all t — Stepwise paths hold long enough that no trades exit before t. Favourable: no addressable-pool tax from D1 attrition.

**The deployability collapse:**

Threshold sweep rule (§8): "max precision subject to recall ≥ 0.60; else max-F1." Neither cluster achieves recall ≥ 0.60 at any threshold ∈ {0.40, 0.50, 0.60, 0.70}. Both fall back to max-F1:

- c0 at threshold 0.40: precision = 0.333, recall = 0.009 → ~3 trades admitted across 5-year pool
- c2 at threshold 0.40: precision = 0.250, recall = 0.004 → ~1 trade admitted across 5-year pool

The §8 design intent (recall ≥ 0.60) is the deployability line; max-F1 was the graceful fallback for marginal cases. Both clusters land far below the line — not marginal, decisive.

The arithmetic: AUC 0.60 at 15–21% positive class allows a maximum achievable precision at recall 0.60 of roughly 0.20–0.25 — barely better than base rate. The rule's design point (recall ≥ 0.60 AND meaningful precision) effectively requires AUC ≈ 0.75+ at this class balance. Neither cluster gets close.

c2's D1 AUC growth with t (0.630 at t=1 → 0.711 at t=10) is the strongest single finding in Arc 6 — path-so-far information has real discriminative power. The smallest-t rule selects t=1 for addressable-pool reasons; even at t=10, however, the threshold sweep would still collapse to max-F1 (AUC 0.71 at 15% positive class still doesn't permit recall ≥ 0.60 with meaningful precision).

## Cross-arc candidates

### Open-XX (new): Step 4 deployability gate

**Problem:** Step 4 mechanically passes when D1 AUC ≥ 0.60 even if the threshold sweep falls back to max-F1 at sub-1% recall. The classifier admits effectively zero trades but the §8 arc-level gate fires PASS.

**Proposal (one of):**
- **(a) Strict mode:** threshold sweep must select on max-precision subject to recall ≥ 0.60. Max-F1 fallback triggers cluster-dies, not graceful pass. The §8 rule's stated intent already encodes recall ≥ 0.60 as the deployability line; make it the gate.
- **(b) Recall floor at gate:** admit threshold must achieve recall ≥ 0.30 (or 0.40) regardless of selection rule. Below that, cluster dies.
- **(c) Higher AUC floor:** raise the §8 D1 AUC threshold from 0.60 to 0.70. At 0.70, recall-0.60 thresholds become achievable for class balances in 15–25% range.

Recommend (a) — minimal protocol change, preserves the rule's design intent, kills the deployability hole cleanly. Cross-arc calibration discussion before next arc opens.

### Open-17 expansion: Tiebreak 1 noise floor

Step 3 c2 SL selection demonstrated Tiebreak 1 firing on a 0.02 ATR margin (~0.15% relative) — well below measurement noise. Rule fires mechanically but its stated purpose ("reward larger physical capture") isn't satisfied at noise-level differences.

**Proposal:** require `peak_mfe_atr_margin ≥ 0.10 ATR OR ≥ 1% relative` before Tiebreak 1 applies; otherwise fall through to Tiebreak 2 (parsimony / smaller SL). Preserves the rule's intent while eliminating the noise-driven flip.

### Cross-arc note: reach_1R floor noise sensitivity

Step 3 c3 dies at 0.697 vs 0.70 by 0.003 — within sampling noise for a 511-trade cluster. Cross-arc question: does the reach_1R floor need a binomial-noise tolerance (e.g., `reach_1R ≥ 0.70 − 1.96 × se` where `se = sqrt(p(1−p)/n)`)?

At n=511, p=0.70, `1.96 × se ≈ 0.040`. Tolerance would meaningfully widen the gate but also weaken its discriminative power.

Arguments both sides; cross-arc discussion.

### Spec erratum: failed-breakout swing_low_N definition

Spec v0.1 literal definition is mathematically unsatisfiable. v0.2 erratum:

```
swing_low_N = min(low[t-N-M..t-M-1])
```

The N=20-bar swing evidence is anchored to bars strictly preceding the M=5-bar breakout window. Verified at Step 1; locked as-built.

## Interesting observations

- **The signal is real but weak.** Pipeline E AUC ~0.60 across all configurations means there IS discriminative entry-time signal for the Stepwise outcome — it just doesn't rise above the 0.65 deployability bar. The pattern exists; it's not strong enough to filter on at this feature set.
- **Pipeline D1 signal grows with t.** c2 D1 AUC 0.630→0.711 from t=1 to t=10. Path-so-far information adds real value. The smallest-t rule + threshold-sweep design point combined to bury what could have been the deployable hook in a different regime.
- **Failed-breakout long is partially arbitraged on majors at 4H.** The structural story is real (trapped breakout shorts at objective swing lows), and c2's path quality is textbook clean (mono_pp 0.566, reach_1R 1.000, ww_pp 0.000). But the entry-time predictability is thin — consistent with a well-known pattern traded by many systematic actors, with residual edge below the filterable bar.
- **Cap-binding 17.65% at Step 1 didn't haunt Step 3.** c2 selected SL=3.0×ATR with zero pre-peak SL touches across 242 trades. The cap-binding subset distributed across clusters rather than concentrating; auto-extend not needed.
- **Anchor parallelism holds.** KH-24 K=4 archetype 3 also failed Pipeline E (0.642 — just barely below 0.65) and routed D1 successfully with deployable recall. Arc 6 produces the same archetype shape but with uniformly weaker entry-side signal.

## Lessons

1. **Mechanical AUC pass ≠ deployable.** The §8 gate at AUC ≥ 0.60 allows arcs to limp through Step 4 with classifiers that admit ~1% of positives. Cross-arc calibration priority (Open-XX).
2. **Path-so-far information has real discriminative value.** D1 AUC growth with t (c2: 0.630 → 0.711) is the strongest signal-quality finding in Arc 6. Cross-arc design idea: deeper-t evaluation, ensemble approaches, or two-stage classifiers (admit at t=1, re-evaluate at t=5).
3. **Failed-breakout long on majors at 4H is partially arbitraged.** Path quality exists post-entry; pre-entry predictability is too thin to filter on. Future arcs of mechanically-objective patterns on majors at 4H should expect this regime and plan accordingly (richer features, multi-TF entry context, ensembles).
4. **Spec drafts need a mathematical sanity check.** The literal `swing_low_N` definition was unsatisfiable. Trivial to spot at review — lesson: spec checklist should include "verify event-definition predicates are satisfiable on synthetic inputs" before dispatch.
5. **Reach_1R = 1.000 with mono_pp = 0.566 is a meaningful within-arc pattern.** c2 has every trade reaching 1R (= 3 ATR physical distance) but only 56.6% of in-profit bars are monotone — implies the path is "always reaches the target but with significant chop." Suggests trail-from-MFE may not be the right exit policy for this profile; a fixed-target take-profit at 1R (and let runners trail thereafter) may capture more cleanly. Cross-arc design exploration.

## Artefacts

- Pool: `results/arc_6/step1/trades_all.csv` (1,564 rows), `trades_paths.csv`
- Clusters: `results/arc_6/step2/clusters_K4.csv`, `archetype_assignments.csv`
- Capturability: `results/arc_6/step3/cluster_routing.csv`, `capturability_pass_list.csv`
- Extractability: `results/arc_6/step4/extractability_pass_list.csv`, `predictability_angle_E.csv`, `predictability_angle_D1.csv`
- D1 classifiers (not shipped — arc dies at Step 4): `archetype_cluster_0_D1_classifier.joblib`, `archetype_cluster_2_D1_classifier.joblib` (artefacts retained for post-mortem only; do not deploy)
- Scripts: `scripts/arc_6/step1_build_pool.py`, `step2_cluster.py`, `step3_capturability.py`, `step4_extractability.py`
- Config: `configs/wfo_l_arc_6.yaml`, `configs/arc_6_feature_catalogue.yaml`
- Commits: `b178ea4` (Step 1), `fd660ab` (Step 2), `f56b945` (Step 3), `8648c1b` (Step 4)

## v2.1.1 → v2.1.2 alignment

Arc opened referencing v2.1.1 (spec draft date); v2.1.2 amendment landed same day as arc open. All three v2.1.2 changes integrated and verified:
- §2 categorical shape_tag floor `≠ scattered` — applied at Step 3
- §11 Stepwise local_peaks ceiling extended to 5–50 — c2 centroid (32.5) used the extended range; would have missed under v2.1.1's 30 ceiling
- §15a `trades_paths.csv` schema with `is_held` — applied at Step 1, validated at gates

---

*Closed 2026-05-17. ARC_6_LIVE.md → ARC_6_RESULT.md. Branch: discovery/lomega_regime_conditional. Calibration items Open-XX and Open-17 expansion queued for cross-arc session before next arc opens.*
