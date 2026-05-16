# Arc 2 redo — Result (CLOSED: KILL at Step 3)

## Status

Opened 2026-05-16 under L_ARC_PROTOCOL v2.0. Closed 2026-05-16 at Step 3 — arc-level capturability gate (v2.0 §7) returned zero passing archetypes. Final disposition: **KILL ARC**.

Signal: `TRIAL__mtf_alignment__2_down_mixed__kijun__h_120` (LCHAR_TOPN_REGISTRY.md Entry 2).

Ran in parallel to Arc 3 under v2.0. Original Arc 2 ran under v1.x verbatim-as-gate framing and closed FAIL (DD 39-91%); the v1.x outcome is not material to this disposition.

## Step results

| Step | Gate | Result |
|------|------|--------|
| 1 | Plumbing | PASS — pool size 12,262 across 28 pairs, deterministic, no lookahead, spread treatment matches lock |
| 2 | Path-shape clustering | PASS — K=4 chosen via silhouette 0.4778; gate clean on all K ∈ {3,4,5,6,7} |
| 3 | Capturability | FAIL — 0/4 archetypes passed §2 hard floors; arc dies per §7 |
| 4 | Extractability | N/A — blocked by Step 3 |
| 5 | Cross-fold stability | N/A |
| 6 | WFO + disposition | N/A |

## Headline finding

Cluster 2 (Stepwise climber, 18.6% of pool, n=2,278) carries overwhelming forward magnitude:

- fwd_mfe_p50 **5.83R**, fwd_mfe_p75 **8.04R**
- frac_reach_1R **99.65%**, frac_reach_2R **96.44%**
- final_r_mean **+3.18R**, t-stat **+52.17**
- mass_gt_5R **0.6102**

Yet fails §2 capturability on three criteria:

- monotonicity_centroid **0.5414** vs ≥ 0.55 floor (miss by 0.0086)
- frac_wrong_way **0.3051** vs ≤ 0.30 ceiling (miss by 0.0051)
- shape_tag **"unclassified"** vs ∈ {tight_unimodal, heavy_right_tail}

The Arc 2 `mtf_alignment.2_down_mixed.kijun` signal generates strong R outcomes on oscillatory, drawdown-prone paths — the pattern v2.0 §2 path-shape floors are designed to filter out. **Edge is real, edge is not capturable by v2.0 §11 exit families.**

### Per-archetype kill criteria

| Cluster | Step 2 label | Step 3 final label | Size | Kill criteria |
|---|---|---|---|---|
| 0 | unclassified | unclassified | 13.6% | monotonicity (0.5120), frac_wrong_way (0.7640) |
| 1 | unclassified | unclassified | 35.1% | monotonicity (0.5485), frac_wrong_way (0.8988) |
| 2 | Stepwise climber | Stepwise climber | 18.6% | monotonicity (0.5414), frac_wrong_way (0.3051), shape_tag (unclassified) |
| 3 | Early-peak hold OR Peak-and-collapse | Early-peak hold | 32.8% | monotonicity (0.0166), frac_reach_1R (0.6628), frac_wrong_way (0.9507) |

Cluster 3 §11 disambiguation: pct_peak_and_collapse = 0.2973 → Early-peak hold (rule: < 0.30).

## Detailed analysis

### Path-shape vs magnitude tension

The protocol gave the right answer for the right reason. Three things are independently true and need separating:

**The signal has real predictive content.** Cluster 2's t-stat of +52 on n=2,278 is not noise. The signal mechanically identifies trades that, on average, produce +3.18R over the 120-bar horizon with 99.65% reaching 1R. This is not a marginal effect.

**The signal's edge is not extractable by fixed-policy exits.** monotonicity 0.5414 means barely over half of in-profit bars made new in-profit highs — that's noisy ascent. frac_wrong_way 0.3051 means 30% of trades touched -1R at some point along the way. A trailing stop policy trying to capture that 5R median MFE would be repeatedly stopped at the -1R touchpoint long before maturation. This is exactly the path-shape pattern §2 floors filter.

**The shape_tag "unclassified" is more diagnostic than the monotonicity / wrong-way margin misses.** Cluster 2's fwd_mfe distribution has p50 5.83R, p75 8.04R, mass_gt_5R 0.61 — fat-right but `p95/p50 > 3` heavy_right_tail criterion fails because the body is already high. No clean structure exists for a fixed TP or trail to harvest. Even if monotonicity and frac_wrong_way thresholds were loosened, the distribution-shape problem remains.

### Methodology cross-validation

Both Arc 2 v1.x (verbatim WFO DD 39-91%) and Arc 2 v2.0 redo (Step 3 KILL) close FAIL on this signal. The failure mechanisms differ — v1.x measured failure as drawdown on the whole pool, v2.0 measured failure as no capturable archetype — but the underlying truth is the same: this signal generates large R outcomes on paths that can't be exited cleanly. Two protocols, two methodologies, same verdict. The signal is genuinely intractable rather than a methodology artefact.

### PR #129 concordance

Redo cluster 2 (K=4) vs PR #129 K=5 archetype 1:
- fwd_mfe_p50: 5.83R (redo) vs 6.86R (PR #129) — OK
- frac_reach_1R: 0.9965 vs 0.998 — OK
- final_r_mean: +3.18R vs +4.45R — divergent
- final_r_t_stat: +52.2 vs +38.7 — divergent

K=4 vs K=5 granularity accounts for the divergence. The redo's K=4 clustering merged what K=5 was splitting, producing a larger but slightly less concentrated archetype. Both runs identify the same underlying capturable cohort.

## Cross-arc candidates

These get carried into the post-Arc-5 cross-arc calibration review per v2.0 §12:

**Open-09 evidence.** Arc 2 redo is the cleanest test case so far for "hard floors may false-kill archetypes with weak capturability but strong extractability". Specific numbers worth quoting at calibration review: cluster 2 monotonicity 0.5414 (miss by 0.0086), fwd_mfe_p50 5.83R, t-stat +52.17 on n=2,278. The miss margin is essentially noise; the magnitude is overwhelming.

**Shape_tag definition pressure.** Cluster 2's fwd_mfe distribution is fat-right but `p95/p50 > 3` heavy_right_tail criterion fails at 1.38 because the body is already high. Definition may need revision for high-magnitude cohorts where the right tail is *not* unusually fat *relative to the body*. Current §7 definition implicitly assumes the body is small relative to the tail; with p50 already 5.83R, no realistic right tail can exceed 3× without being absurd. Worth a calibration look at the shape-tag rule set.

**Path-shape vs magnitude gate ordering.** v2.0 sequences capturability (Step 3) → extractability (Step 4). Arc 2 redo confirms this ordering catches cohorts with strong magnitude but unextractable paths early. Whether the gate ordering should be relaxed for "exceptional magnitude" cohorts (e.g., t-stat > 30 admits to Step 4 even if path-shape marginal) is an Open-09 question.

## Interesting observations

- Path-shape clustering cleanly separated the high-magnitude cohort (cluster 2, 18.6%) from the dominant "SL'd quickly" cohort (cluster 3, 32.8%, mono 0.017, ttp 0.05) — v2.0 §6 working as designed.
- Cluster 3 §11 disambiguation resolved to "Early-peak hold" at pct_peak_and_collapse 0.2973, narrowly under the 0.30 threshold. Doesn't affect capturability outcome but the disambiguation rule fired cleanly.
- The K=4 silhouette of 0.4778 is healthy clustering structure — no degenerate features, no oversized clusters. The signal has internal structure; that structure just doesn't include a capturable archetype.
- pullback_magnitude_median 44% at 0.0 but modal-bin mass 46.27% (below the 80% degeneracy threshold). Open-08 concern about pullback degeneracy doesn't materialise on this pool.

## Flags carried from prior steps

1. Step 1 — 240-bar forward-window filter excludes late-data signals; would have affected fold 7 trade count at Step 6 had the arc reached it.
2. Step 1 — weekend-spread tail: 32 trades with spread_pips_exit > 50, worst -7.9R AUD_JPY 2015-01-05. Faithful to SPREAD_SEMANTICS_LOCK.
3. Step 2 — pullback_magnitude_median 44% at 0.0 but not degenerate (modal mass 46.27% < 80%). Open-08 concern doesn't materialise.
4. Step 2 — PR #129 K=5 concordance documented as match but v2.0 chose K=4 by silhouette.
5. Step 2 — clusters 0 and 1 unclassified under §11; Step 3 evaluated them under §2 floors with "no §11 ceiling" treatment on local_peaks (passed on absence of constraint).
6. Step 3 — forward-geometry method spot-checked, raw market closes correctly reconstructed for post-exit bars.
7. Step 3 — Open-09 calibration tension explicitly surfaced; cluster 2 contributes evidence.

## Permanently eliminated by this arc

Nothing permanently eliminated. The signal is shelved as "real edge, not extractable under v2.0 as drawn". A future v2.x calibration patch that loosens monotonicity floor or revises shape_tag definitions could reopen the question, contingent on cross-arc calibration review producing the relevant amendment.

## Verdict and disposition

**KILL ARC.** Arc 2 redo closes at Step 3 with documented finding. v2.0 protocol gave the methodologically correct answer; the signal is real-but-intractable under v2.0 as drawn.

## Files

All under `results/l_arc_2_redo/`:

- `step1/` — trades_all.csv, trades_paths.csv, STEP1_SUMMARY.md, configs and helper scripts
- `step2/` — path_features.csv, clusters_K{3..7}.csv, centroids_K{3..7}.csv, silhouette_K{3..7}.txt, silhouette_sweep.csv, archetype_assignments.csv, STEP2_SUMMARY.md, feature_histograms.png
- `step3/` — forward_geometry.csv, forward_geometry_spot_check.txt, capturability_pass_list.csv, archetype_summaries.csv, archetype_{0..3}_distribution.csv, archetype_{0..3}_{fwd_mfe,final_r}_histogram.png, STEP3_SUMMARY.md
- `ARC_2_REDO_RESULT.md` (this file)

Step 1 commit: `1557d03` (Step 2 commit referenced in Step 2 summary)
Step 2 commit: `1557d03`
Step 3 commit: `b973600`

## Handover note

Arc 2 redo is closed. No live doc to inherit. Arc 3 continues under v2.0 unaffected by this closure.

Cross-arc carryover for future v2.x calibration review:
- Open-09 evidence: cluster 2 metrics (monotonicity 0.5414, fwd_mfe_p50 5.83R, t-stat +52.17, n=2,278)
- Shape_tag definition revisit for high-magnitude cohorts (current heavy_right_tail criterion fails on high-body distributions)
- Path-shape vs magnitude gate ordering question (Open-09)

Surface these naturally at the post-Arc-5 cross-arc calibration review per v2.0 §12.
