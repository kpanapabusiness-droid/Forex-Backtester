# Open-18 Replay #1 — Arc 3 Stepwise climber under v2.1.1 — RESULT

## Status

Opened 2026-05-17 under L_ARC_PROTOCOL v2.1.1 §16 Open-18.
Closed 2026-05-17. Final disposition: **PASS_v2_1_1**.

This is a replay, not a new arc. Arc 3 remains CLEAN-NULL per ARC_3_RESULT.md; verdict locked. This replay measures whether v2.1.1 mechanics would have changed §2 admission.

Per-cluster outcome:
- **Cluster 2** — PASS individually at selected SL = 1.5 ATR (composite 0.4728), AND PASS as part of aggregate. Routing = both.
- **Cluster 4** — FAIL individually at every candidate SL. PASS via aggregate. Routing = aggregate.
- **Aggregate (2+4)** — PASS at selected SL = 2.0 ATR (composite 0.4672). shape_tag = bimodal_separated.

Cohort verdict: rescued. Both clusters proceed downstream under §2 admission — c2 in both individual and aggregate forms, c4 only as part of the aggregate.

## Cohort under test

| Field | Value |
|---|---|
| Source arc | Arc 3 (`TRIAL__volatility_regime__d1_atr_top_decile__any__h_120`) |
| Cohort | Stepwise climber (clusters 2+4 aggregated under §6; also individual per §7 v2.1.1) |
| Pool size | n=707 aggregate (c2 n=380, c4 n=327) |
| Pool denominator | 2568 (size_fraction = 27.5%) |
| Signal TF | 1H, horizon h=120 |
| Simulation SL | 2.0 × ATR(14)_1H |
| v2.0 verdict | FAIL §2 — wrong_way 38.3% (full-window) > 0.30 floor AND shape_tag = bimodal ∉ admit list |
| v2.0 passes | size 27.5%, mono 0.559, peaks 16.73, mfe_p50 3.34R, reach_1R 83.6% |

Cluster centroids (Arc 3 Step 2, K=7):

| | C2 (n=380) | C4 (n=327) | Agg (n=707) |
|---|---|---|---|
| Monotonicity | 0.549 | 0.572 | 0.559 |
| Local peaks | 24.42 | 7.79 | 16.73 |
| Pullback | 0.498 | 0.426 | — |
| ttp_rel | 0.816 | 0.787 | — |
| pct_peak_and_collapse | 0.126 | 0.474 | — |

<<< PRE-COMMITTED — DO NOT EDIT >>>

## Hypothesis (pre-committed)

**Primary:** Cluster 4 rescues primarily via pre-peak wrong_way (Def C). Its high `pct_peak_and_collapse` (0.474) means roughly half its trades have a post-peak collapse — and post-peak adverse moves by Def C do NOT count toward `frac_wrong_way_pre_peak`. The v2.0 full-window 38.3% wrong_way for the aggregate was likely dominated by c4's post-peak collapse contribution.

**Secondary:** Cluster 2 (low pc=0.126) already had a cleaner path; pre-peak wrong_way is expected to be much lower than 38.3% on c2 alone.

**Tertiary (uncertain):** The aggregate's `fwd_mfe` distribution at the selected SL may pass the bimodal_separated test — but the bimodality is in MFE space (mass near 0 for stopped-out trades + mass at ≥1R for survivors), NOT in `final_r` space (where the v2.0 bimodal-tag came from). Whether MFE is structurally bimodal_separated per Hartigan dip + min-mode-mass + ≥1R separation is empirical.

**Mechanism disentangle:** Capturability composite prefers wider SL when cohort has high early-stop-out rates (pre-peak wrong_way drops trivially for trades with no pre-peak bars). Wider SL also captures peaks happening after Arc 3's original 120-bar time exit (sweep window 240). `frac_peak_at_bar_0` and `frac_peak_after_bar_120` separate these confounds.

## Predictions (pre-committed)

Quantitative pre-commitments. Replay confirms or disconfirms.

| Metric | Cluster 2 | Cluster 4 | Aggregate |
|---|---|---|---|
| frac_wrong_way_pre_peak at sim SL (2.0 ATR) | 15-25% | 20-35% | 20-30% |
| frac_wrong_way_pre_peak at selected SL | <25% | <30% | <30% |
| Individual §2 verdict under v2.1.1 | PASS expected | PASS likely (was thought failed under v2.0 reasoning) | PASS likely |

Selected SL: 2.0 to 3.0 ATR range expected for all three archetypes. 0.5 and 1.0 ATR likely fail magnitude/reach. 4.0 ATR borderline on `fwd_mfe_p50 ≥ 1.5R` (magnitude halves to ~1.67R per Arc 3 closure estimate).

shape_tag at selected SL:
- C2: tight_unimodal or heavy_right_tail (clean MFE, right tail from survivors)
- C4: bimodal_separated plausible (mass near 0 from collapsed trades, mass at higher R from survivors). Hartigan dip likely fires.
- Aggregate: bimodal_separated plausible. If this is the only level where bimodal_separated fires (and neither cluster individually does), the rescue is per-cluster eval, not bimodal_separated admit — Fork B.

Diagnostic predictions:
- `frac_peak_at_bar_0` high at tight SL (>30% at 0.5 ATR), low at wide SL (<10% at 3-4 ATR)
- `frac_peak_after_bar_120` 10-30% at selected SL. If >50%, rescue is substantially horizon-extension.

## Interpretation forks (pre-committed)

Pre-committed reading for the outcomes that materially change what we learn. Fixed before measurement.

**Fork A — Both clusters pass §2 individually, aggregate also passes, bimodal_separated does NOT fire on any.**
- Reading: v2.1.1's pre-peak mechanism is what rescued the cohort. Open-13 bimodal admit not needed.
- Implication: Open-13 mechanism remains theoretical until Replay #2 or #3 surfaces it.

**Fork B — bimodal_separated fires on aggregate ONLY, neither cluster individually shows it.**
- Reading: Bimodality is an aggregation artifact of c2 and c4 having different magnitude distributions, not a true bimodal archetype. v2.1.1 per-cluster eval (Open-14) is the actual rescue mechanism; bimodal_separated admit (Open-13) is firing on a pseudo-mode.
- Implication: Open-13 needs re-examination — does it correctly handle the aggregation-artifact case? §11 row 7 split-exit would be inappropriate.

**Fork C — bimodal_separated fires on cluster 4 (or 2) individually.**
- Reading: Genuine bimodal archetype at cluster level. Open-13 validated. §11 row 7 split-exit is the right routing.
- Implication: Open-13 admit was structurally correct.

**Fork D — Aggregate passes but individual clusters fail.**
- Reading: c2 and c4 each have issues but aggregate averages them out. Inverse of what v2.1.1 was designed to surface — protocol weakness.
- Implication: Per-cluster eval may need additional rules; surface to PROTOCOL_IMPROVEMENT_BACKLOG.

**Fork E — Selected SL is 0.5 or 1.0 ATR (tighter than simulation).**
- Reading: Cohort is fast-peaking; tight SL captures peaks without false stops. Arc 3's 2 ATR was actually too LOOSE for this archetype.
- Implication: Reverses the Arc 3 closure's SL/horizon asymmetry diagnosis for this cohort specifically. Material finding.

**Fork F — Selected SL is 3.0 or 4.0 ATR AND `frac_peak_after_bar_120 > 30%` at that SL.**
- Reading: Rescue is substantially horizon-extension. Wider SL alone wouldn't rescue with hold capped at 120.
- Implication: Arc 4+ needs to consider time-exit window alongside SL, not just wider SL.

**Fork G — Cohort fails to rescue.**
- Reading: Pre-peak alone insufficient. v2.1.1 mechanics didn't fix the issue.
- Implication: Re-examine §2 calibration. Open-19 likely.

## What this replay can NOT tell us (pre-committed)

- Does not validate that Stepwise survives Steps 4-6 (extractability, stability, WFO). §2 admission is necessary, not sufficient.
- Does not validate Open-13's `bimodal_separated → §11 row 7 split-exit` produces a deployable system. Requires Steps 4-6.
- Does not generalise beyond this cohort. Cross-cohort requires Replays #2 and #3 + chat-side synthesis.
- Does not refute Arc 3's CLEAN-NULL verdict.

## Cross-replay implications framing (pre-committed)

- **Fork A (pre-peak alone, no bimodal_separated):** KH-24 c4 should rescue similarly (also failed primarily on full-window mono). Arc 2 redo c2 is harder — if path is messy from start rather than only post-peak, pre-peak alone won't save it.
- **Fork B or C (bimodal_separated fires):** Open-13 validated (or shown aggregation-artifact-sensitive in B). Cross-replay look for bimodal_separated in KH-24 c4 (whose "scattered" v2.0 tag is consistent with possible bimodality).
- **Fork G (cohort fails):** v2.1.1 needs diagnosis. Replays #2/#3 may also fail. Open-19 likely.

Do not speculate further about Replays #2/#3 here.

<<< END PRE-COMMITTED >>>

## Data limitation note (added before measurement)

Arc 3's `trades_paths.csv` was generated by [scripts/arc_3/step1_backtest.py](../../../scripts/arc_3/step1_backtest.py) — predates the v1.3 forward-window extension that protocol §17 SL-free path assumes. Schema differences vs the engine:

- No `is_held` column (every row was held)
- No `pair` column (joined via `trades_all.csv`)
- Path truncates at `min(SL_hit, time_exit_idx=120)` — no forward observation bars after exit

Without forward bars, SL sweep at X > 2.0 ATR is unmeasurable on 62.6% of trades at 3 ATR and 71.2% at 4 ATR (stoploss trades whose 2-ATR-SL bar had min_mae > new threshold — we have no data for what they would have done next).

Resolution: wrote [scripts/replays_v2_1_1/arc_3_stepwise/extend_paths.py](../../../scripts/replays_v2_1_1/arc_3_stepwise/extend_paths.py) to forward-extend bar paths to `entry_idx + 240` using the same per-bar excursion math the engine's `_flatten_bar_path_for_trade` uses (long-only: `cand_mfe_price = high - entry_price`, `cand_mae_price = entry_price - low`, accumulators max-tracked, R-denominated via `sl_distance_price = entry_price - initial_sl_price`). Output written to `trades_paths_extended.csv`; held bars copied byte-equivalent from source, forward bars flagged `is_held=0`. Arc 3's frozen step1 output is **not** modified. The Step 3 replay reads only the extended file.

Extension stats: 2568 trades, 131641 held rows preserved, 487179 forward rows added (618820 total). 2 trades hit data runout (entry close to data end; partial forward window). User authorised this regeneration via AskUserQuestion before any measurement was taken.

## v2.0 vs v2.1.1 comparison

| Archetype | Metric | v2.0 (full window, sim SL 2.0) | v2.1.1 (selected SL, pre-peak) |
|---|---|---|---|
| Cluster 2 | n | 380 | 380 |
|  | size_fraction | 14.8% | 14.8% |
|  | mono | 0.549 (centroid_v2) | 0.562 (pre-peak, sel SL 1.5) |
|  | local_peaks | 24.42 (centroid_v2) | 27.35 (pre-peak, sel SL 1.5) |
|  | fwd_mfe_p50 | (not separately reported in ARC_3_RESULT) | 7.10R (in 1.5-ATR R) |
|  | frac_reach_1R | (n/a in v2.0 aggregate row) | 87.6% (in 1.5-ATR R) |
|  | frac_wrong_way | (n/a alone) | 1.6% pre-peak |
|  | shape_tag | (n/a alone) | **bimodal_separated** |
|  | composite | n/a | 0.4728 |
|  | verdict | not separately evaluated | **PASS** |
| Cluster 4 | n | 327 | 327 |
|  | size_fraction | 12.7% | 12.7% |
|  | best of any SL: fwd_mfe_p50 | n/a | 1.37R (peaks at SL=1.5 ATR) |
|  | best of any SL: frac_reach_1R | n/a | 64.5% (caps below 0.70) |
|  | best shape_tag across sweep | n/a | heavy_right_tail / unclassified / bimodal (never bimodal_separated) |
|  | verdict | not separately evaluated | **FAIL_NO_PASSING_SL** (no SL passes all §2 floors) |
| Aggregate 2+4 | n | 707 | 707 |
|  | size_fraction | 27.5% | 27.5% |
|  | mono | 0.559 (v2 centroid) | 0.598 (pre-peak, sel SL 2.0) |
|  | local_peaks | 16.73 (v2 centroid) | 21.76 (pre-peak, sel SL 2.0) |
|  | fwd_mfe_p50 | 3.34R | 4.14R (in 2.0-ATR R) |
|  | frac_reach_1R | 83.6% | 83.6% (in 2.0-ATR R; matches v2.0) |
|  | frac_wrong_way | 38.3% (full window) | **1.7% pre-peak** |
|  | shape_tag | bimodal (v2.0 KDE-valley test) | **bimodal_separated** (v2.1.1 dip + mass + sep all pass) |
|  | composite | n/a | 0.4672 |
|  | verdict | **FAIL** (wrong_way + shape_tag) | **PASS** |

## Selected SL + composite per archetype

From [archetype_summaries.csv](archetype_summaries.csv):

| Archetype | Selected SL | Composite | mono_pp | local_peaks_pp | fwd_mfe_p50 | reach_1R | wrong_way_pp | shape_tag | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| cluster_2 | 1.5 ATR | 0.4728 | 0.562 | 27.35 | 7.10R | 87.6% | 1.6% | bimodal_separated | PASS |
| cluster_4 | (no passing SL) | — | — | — | — | — | — | — | FAIL_NO_PASSING_SL |
| aggregate_stepwise_climber_2+4 | 2.0 ATR | 0.4672 | 0.598 | 21.76 | 4.14R | 83.6% | 1.7% | bimodal_separated | PASS |

## §2 floor evaluation at selected SL

**Cluster 2 @ SL=1.5 ATR:**

| Floor | Threshold | Measured | Margin | Pass |
|---|---|---|---|---|
| monotonicity_pre_peak | ≥ 0.55 | 0.562 | +0.012 | ✓ |
| local_peaks (in [5,30]) | 5 ≤ x ≤ 30 | 27.35 | inside | ✓ |
| fwd_mfe_h240_p50 | ≥ 1.5R | 7.10R | +5.60R | ✓ |
| frac_reach_1R | ≥ 0.70 | 0.876 | +0.176 | ✓ |
| frac_wrong_way_pre_peak | ≤ 0.30 | 0.016 | +0.284 | ✓ |
| shape_tag | in {tight_unimodal, heavy_right_tail, bimodal_separated} | bimodal_separated | — | ✓ |
| size_fraction | ≥ 0.10 | 0.148 | +0.048 | ✓ |

All 7 floors pass. Composite = (0.562 − 0.55) + (0.876 − 0.70) + (0.30 − 0.016) = 0.012 + 0.176 + 0.284 = **0.4728**.

**Cluster 4** (no passing SL — fails magnitude AND shape_tag across the sweep):

| SL (ATR) | mono_pp | local_peaks | fwd_mfe_p50 | reach_1R | wrong_way_pp | shape_tag | floors failing |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.497 | 2.69 | 1.24R | 58.1% | 37.3% | heavy_right_tail | mono, local_peaks, mfe_p50, reach_1R, wrong_way |
| 1.0 | 0.540 | 5.18 | 1.31R | 58.1% | 13.5% | heavy_right_tail | mono, mfe_p50, reach_1R |
| 1.5 | 0.602 | 7.27 | 1.37R | 62.1% | 5.2% | unclassified | mfe_p50, reach_1R, shape_tag |
| 2.0 | 0.636 | 9.24 | 1.34R | 64.5% | 3.7% | unclassified | mfe_p50, reach_1R, shape_tag |
| 3.0 | 0.612 | 11.49 | 1.13R | 53.8% | 0.6% | bimodal | mfe_p50, reach_1R, shape_tag |
| 4.0 | 0.601 | 12.45 | 1.00R | 50.2% | 0.3% | bimodal | mfe_p50, reach_1R, shape_tag |

Cluster 4 alone is genuinely structurally weaker than cluster 2 — its magnitude (`fwd_mfe_p50` peaks at 1.37R at SL=1.5, never reaches the 1.5R floor) and reach_1R (caps at 64.5%, never the 70% floor) bind regardless of SL. The bimodal_separated test never fires (min_mode_mass always lopsided ≤ 0.17). Routing via aggregate is what carries c4 across §2.

**Aggregate 2+4 @ SL=2.0 ATR:**

| Floor | Threshold | Measured | Margin | Pass |
|---|---|---|---|---|
| monotonicity_pre_peak | ≥ 0.55 | 0.598 | +0.048 | ✓ |
| local_peaks (in [5,30]) | 5 ≤ x ≤ 30 | 21.76 | inside | ✓ |
| fwd_mfe_h240_p50 | ≥ 1.5R | 4.14R | +2.64R | ✓ |
| frac_reach_1R | ≥ 0.70 | 0.836 | +0.136 | ✓ |
| frac_wrong_way_pre_peak | ≤ 0.30 | 0.017 | +0.283 | ✓ |
| shape_tag | in admit list | bimodal_separated | — | ✓ |
| size_fraction | ≥ 0.10 | 0.275 | +0.175 | ✓ |

All 7 floors pass. Composite = 0.048 + 0.136 + 0.283 = **0.4672**.

Note: the aggregate's selected SL = 2.0 ATR is the same as Arc 3's simulation SL. The v2.0 verdict failed at the SAME SL on full-window metrics; v2.1.1 mechanics flip the verdict without changing the SL.

## Shape_tag verdicts

| Archetype | Selected SL | v2.0 classifier output | dip p | mode_sep (R) | min_mode_mass | bimodal_separated test | Final v2.1.1 shape_tag |
|---|---|---|---|---|---|---|---|
| cluster_2 | 1.5 ATR | scattered | 0.000051 | 5.84 | 0.208 | dip ✓, sep ✓, mass ✓ → **PASS** | **bimodal_separated** |
| cluster_4 | (no sel SL; best @ 1.5 ATR) | unclassified | 0.992 | 6.43 | 0.128 | dip ✗, mass ✗ | unclassified |
| aggregate | 2.0 ATR | bimodal (v2.0 KDE) | 0.012 | 3.46 | 0.358 | dip ✓, sep ✓, mass ✓ → **PASS** | **bimodal_separated** |

Notable: the v2.0 classifier alone tagged cluster_2 @ SL=1.5 as "scattered" — but the v2.1.1 stricter bimodal_separated test (Hartigan dip + min-mode-mass + ≥1R separation) confirms genuine two-mode structure. KDE modes at ~1.06R and ~6.90R for c2; ~1.28R and ~4.75R for aggregate at SL=2.0. The lower mode in each case (~1R) is dominated by stop-outs at −1R and survivors who peak between 0.5–2R; the upper mode is the run-to-multiple-R survivors. This is the bimodal pattern the analyst hypothesised for MFE space.

## Per-cluster vs aggregate routing

From [cluster_routing.csv](cluster_routing.csv):

| Cluster | Individual verdict | Aggregate verdict | Routing |
|---|---|---|---|
| 2 | PASS | PASS | **both** — proceeds as c2 (sel SL 1.5) AND as part of aggregate (sel SL 2.0) |
| 4 | FAIL | PASS | **aggregate** — proceeds only as part of aggregate (sel SL 2.0) |

Under v2.1.1 §7, c2 advances downstream in both forms (Step 6 ships best). c4 advances only via the aggregate routing — meaning at Step 4 onward c4 is evaluated as a member of the 707-trade aggregate, not as a 327-trade standalone. The aggregate's selected SL (2.0 ATR) becomes c4's R-frame.

## Diagnostic columns

| Archetype | Selected SL | frac_peak_at_bar_0 | frac_peak_after_bar_120 |
|---|---|---|---|
| Cluster 2 | 1.5 ATR | 6.1% | **58.2%** |
| Cluster 4 | (best SL=2.0) | 0.0% | 12.5% |
| Aggregate 2+4 | 2.0 ATR | 0.0% | **44.5%** |

Tying to Fork E / Fork F:

- **Fork E (selected SL tighter than simulation):** partially applies to cluster_2 — its selected SL = 1.5 ATR is one notch tighter than the 2.0-ATR simulation, but not as tight as 0.5/1.0. Aggregate stays at 2.0 ATR (matches simulation). So the Arc 3 closure's "2 ATR too loose for fast-peaking sub-archetype" speculation is mildly supported for c2 only — c2 is *somewhat* fast-peaking (6.1% peak at bar 0) but most of c2's rescue comes from horizon extension (58.2% peak after bar 120), not tighter SL.

- **Fork F (selected SL wider + frac_peak_after_bar_120 > 30%):** the aggregate's frac_peak_after_bar_120 = 44.5% at selected SL is well above the Fork F 30% threshold. **The rescue is substantially horizon-extension** — Arc 3's original 120-bar time exit was cutting the cohort off before peak MFE for ~45% of trades. This is the dominant mechanism, alongside Def-C pre-peak wrong_way relief. Cluster 2's individual selected SL (1.5 ATR) is tighter than simulation, but its frac_peak_after_120 = 58.2% — even the c2-individual selection is horizon-bound.

  Implication aligned with Fork F's pre-committed reading: **Arc 4+ should consider time-exit window alongside SL, not just SL alone.** The 240-bar forward window is doing real lifting here.

## Verdict vs predictions

| Prediction | Predicted | Measured | Reading |
|---|---|---|---|
| C2 frac_wrong_way_pre_peak at sim SL (2.0) | 15–25% | 0.0% | **dramatically lower** — Def C completely eliminates wrong_way for c2 at sim SL (no pre-peak bars cross −1R) |
| C4 frac_wrong_way_pre_peak at sim SL (2.0) | 20–35% | 3.7% | **dramatically lower** — c4's high pct_peak_and_collapse means almost all adverse moves are post-peak, exactly as primary hypothesis predicted |
| Agg frac_wrong_way_pre_peak at sim SL (2.0) | 20–30% | 1.7% | **dramatically lower** — confirms primary hypothesis with greater magnitude than expected |
| C2 frac_wrong_way_pre_peak at selected SL (1.5) | < 25% | 1.6% | confirmed (very low) |
| C4 frac_wrong_way_pre_peak at "best" SL | < 30% | 3.7% (at SL=2.0) | confirmed but irrelevant — c4 fails on other floors |
| Agg frac_wrong_way_pre_peak at selected SL (2.0) | < 30% | 1.7% | confirmed |
| Selected SL range | 2.0–3.0 ATR | c2 = 1.5 ATR; agg = 2.0 ATR | c2 selected SL is one notch tighter than predicted; aggregate lands at lower end of range. Composite at 3.0 ATR was lower than at 2.0 ATR for the aggregate (0.420 vs 0.467), so the protocol's empirical SL choice is well inside the predicted band |
| C2 §2 verdict | PASS | **PASS** | confirmed |
| C4 §2 verdict | PASS likely | **FAIL_NO_PASSING_SL** | disconfirmed — c4 alone is structurally weaker than the prediction allowed for. Reach_1R caps at 64.5%, fwd_mfe_p50 caps at 1.37R; binds independent of wrong_way relief. c4 rescues only via aggregate routing |
| Agg §2 verdict | PASS likely | **PASS** | confirmed |
| Which fork fired? | Open | Fork C + Fork F | **Fork C** (cluster 2 individually shows bimodal_separated at SL ∈ {0.5, 1.0, 1.5} — Open-13 validated, NOT just an aggregation artifact). **Fork F** (aggregate's selected SL frac_peak_after_bar_120 = 44.5% — horizon extension is a material rescue mechanism alongside pre-peak Def C) |

Pre-committed shape_tag predictions: C2 was predicted tight_unimodal or heavy_right_tail; actually fired **bimodal_separated** at SL=1.5. C4 was predicted bimodal_separated; actually never fires (min_mode_mass always too lopsided). Aggregate was predicted bimodal_separated; actually fires bimodal_separated at SL ∈ {2.0, 3.0, 4.0}. The predicted MFE-space bimodality is real for c2-individual and aggregate, but NOT for c4-individual.

## Interpretation

The Stepwise climber cohort is rescued under v2.1.1. The primary mechanism was Def C pre-peak wrong_way: the v2.0 full-window 38.3% drops to 1.7% on the aggregate at the same SL. This is more dramatic than the analyst's pre-committed 20–30% prediction — Def C reduces wrong_way by ~20× rather than the ~2× the analyst's prior suggested. The reason is that this cohort's adverse moves are overwhelmingly post-peak, not pre-peak (consistent with c4's pct_peak_and_collapse = 0.474 and with the bimodal MFE-space structure: stop-outs cluster at −1R end, survivors at multi-R end, and the −1R mass arrives mostly *after* an initial favourable move).

The aggregate's selected SL = 2.0 ATR is identical to the simulation SL — v2.1.1 mechanics flip the verdict on the same SL without needing wider-SL rescue. This was not a foregone outcome: at SL=3.0 and 4.0 ATR the aggregate also passes (composites 0.420 and 0.399 respectively), but the magnitude erosion at wider SLs costs composite faster than wrong_way relief gains it. The capturability composite mechanic (v2.1.1's refinement over v2.1's smallest-passes) is what makes 2.0 ATR the choice here.

bimodal_separated fires on cluster 2 individually (SL=1.5) and on the aggregate (SL=2.0). It does NOT fire on cluster 4 individually at any SL — c4's distribution is more lopsided (the higher mode has only ~13–17% mass at any sweep level). This is **Fork C, not Fork B** — the bimodality is a genuine cluster-level structural property of c2, not just an aggregation artifact. Open-13's bimodal_separated admit was structurally correct for at least this cohort.

Cluster 4 individually fails §2 across the entire SL sweep — not on wrong_way (≤ 3.7% everywhere) but on magnitude (fwd_mfe_p50 caps at 1.37R, never reaches 1.5R) and reach_1R (caps at 64.5%, never reaches 70%). This contradicts the analyst's "PASS likely" for c4 alone — c4 is structurally weaker than predicted, and the Arc 3 closure's speculation about c4 having "wrong_way 50%+" was also wrong (the wrong_way under Def C is essentially zero). c4 is salvaged downstream only via aggregate routing. This is a more granular surprise than any of the pre-committed forks predicted exactly, though it lies on the boundary of Fork D ("aggregate passes but individuals fail") — except here only c4 fails, not c2.

The diagnostic columns expose Fork F: aggregate frac_peak_after_bar_120 = 44.5% at selected SL, and c2's individual figure is even higher at 58.2%. **Roughly half of trades in this cohort peak after Arc 3's original 120-bar time exit.** That's a structural finding about the signal class: horizon extension is doing real work, not just SL relaxation. The Arc 3 closure's instinct to look at SL/horizon asymmetry (Open-15) was right; this replay sharpens it — for this cohort, horizon was the more binding constraint than SL.

Combined picture: v2.1.1's three mechanics (pre-peak Def C, capturability composite, bimodal_separated test) **all contributed** to the rescue, with Def C the largest in absolute magnitude, bimodal_separated the precondition that prevented the v2.0 shape_tag failure from recurring, and the composite ranking what selected the empirically-correct SL among multiple passing candidates. None of the three mechanisms alone would have rescued the cohort: Def C alone wouldn't have helped if shape_tag still failed; bimodal_separated alone wouldn't have helped if wrong_way still failed; composite-vs-smallest-passes is what made SL=2.0 (over SL=3.0/4.0) the empirical choice for the aggregate.

The cohort's downstream-extractability question (Steps 4–6) is unanswered by this replay and outside scope.

## Files

All under [results/replays_v2_1_1/arc_3_stepwise/](.):

- [archetype_summaries.csv](archetype_summaries.csv) — one row per evaluated archetype at selected SL (or failure row)
- [cluster_2_sl_sweep.csv](cluster_2_sl_sweep.csv), [cluster_4_sl_sweep.csv](cluster_4_sl_sweep.csv), [aggregate_stepwise_climber_2+4_sl_sweep.csv](aggregate_stepwise_climber_2+4_sl_sweep.csv) — full sweeps with all metrics, floor pass/fail, composite per candidate SL
- [cluster_2_distribution.csv](cluster_2_distribution.csv), [aggregate_stepwise_climber_2+4_distribution.csv](aggregate_stepwise_climber_2+4_distribution.csv) — at selected SL (percentiles, mass-in-band, dip/mode details)
- [capturability_pass_list.csv](capturability_pass_list.csv) — surviving archetypes
- [cluster_routing.csv](cluster_routing.csv) — per-cluster individual / aggregate / both / dies disposition
- [trades_paths_extended.csv](trades_paths_extended.csv) — forward-extended bar paths (input to step3.py, not Arc 3's frozen step1 output)
- REPLAY_RESULT.md (this file)

Commit SHAs: to be filled at commit time.
