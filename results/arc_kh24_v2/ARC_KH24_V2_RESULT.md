# Arc — KH-24 v2.0 self-test — RESULT (CLOSED: HALT at Step 3)

## Status

Opened 2026-05-16 under `L_ARC_PROTOCOL.md` v2.0 as a protocol self-test + Pipeline D1 backtester commissioning arc. Closed 2026-05-16 at Step 3 — arc-level capturability gate (§7) returned zero passing archetypes. Final disposition: **HALT — STEP3_FAIL_NO_CAPTURABLE_ARCHETYPE**.

Signal under test: bare `kb_exhaustion_bar` (C1-C6, C8, C9, with C7 disabled). Long-only, 4H, 28 FX, 1R hard SL = 2.0 × ATR(14), 240-bar forward window.

Pipeline D1 backtester extension being built in a separate chat. Its value extends beyond this arc to all future D1-routed archetypes; that work continues independently.

## The actual question this arc answered

**Does v2.0 protocol applied to bare KH-24 signal produce a better system than v1.0 KH-24?**

**No.** Zero clusters cleared §2 capturability floors. v1.0 KH-24 (worst-fold ROI +1.92%, DD 6.37%, deployed since 2026-04-20) remains the only system that has reached deployment on this signal. v2.0 as drawn does not reach Step 6 here.

The §14 "calibration anchor reproduction" question that initially shaped this arc's framing was dropped at Step 2 once the anchor was understood to be measured on a different population than v2.0 can produce. The real question — "better than v1.0?" — has its mechanical answer above. The structurally interesting answer follows.

## c4 in plain English (the best contender)

c4 was the arc's only structurally credible candidate. It was the **trend-rider archetype** — provisionally labelled `unresolved_11fee4a0`. Closest §11 match would have been Stepwise climber, had it cleared §2.

**Proportions:** 122 trades out of 842 (14.5% of pool), roughly 7.6 trades per year over 16 years.

**This cluster behaved like:** a slow, chunky trend ride. Trades entered, took their time to mature (median time-to-peak at 76% of the trade's duration — late peakers), and ran for a long time (87.7% of c4 trades held until the 240-bar forward-window cap). Lots of intra-trade peaks (centroid 30.94, the noisiest cluster of all) — the trades climbed in many small steps with frequent small pullbacks rather than one clean shot.

**It had potential due to:**
- p50 forward MFE of **6.65R** (every other surviving cluster was 1.7R or worse)
- **100%** of trades reached 1R MFE — perfect direction
- **0%** wrong-way trades — the signal never fired against the direction it eventually went
- Sizable cluster (14.5% — well above the 10% size floor)

These are the economics of a real edge cohort. Trades that always reach 1R and have a median maximum favourable excursion of 6.65R are the kind of trade you build a system around.

**It failed because:**

1. **Monotonicity 0.530 vs the 0.55 floor — a 0.020 miss.** The trades climb choppily (lots of small peaks and pullbacks) rather than cleanly. The §2 floor was calibrated against KH-24's filtered deployed population (mono 0.576), where v1.0's 1H CIR + currency cap filters had already pre-selected for cleaner-shaped paths. On the bare signal, the same edge cohort exists but with a slightly choppier path shape — choppy enough to miss 0.55.

2. **shape_tag classified as "scattered."** Because 87.7% of c4 trades hit the 240-bar forward window cap, their `final_r` values are censored — recorded at whatever point on a still-running trajectory the cap happened to fall. Different trades got clipped at different points, producing a wide spread in `final_r` even though the underlying MFE distribution was clean and right-tailed. The protocol's shape_tag rules judge the censored `final_r` distribution and call it scattered. The protocol couldn't see the heavy-right-tail shape the underlying paths actually had.

Two failures, both technical, both from §2 calibrations that don't fit this signal's actual character. Real edge, killed on the gates.

## Pattern across arcs (24-hour window)

Two arcs closed FAIL at Step 3 §2 floors on 2026-05-16:

- **Arc 2 redo:** cluster 2 missed mono 0.55 by **0.009** with t-stat +52.17, n=2,278.
- **KH-24 v2.0 self-test (this arc):** cluster c4 missed mono 0.55 by **0.020** with frac_reach_1R 1.000, frac_wrong_way 0.000, fwd_mfe_p50 6.65R.

Different signals, different mechanisms, **same gate failure.** Both arcs surfaced strong-edge cohorts. Both got filtered out by mono < 0.55 and/or shape_tag.

If Arc 3 closes the same way it becomes three-of-three. The §2 calibration is becoming the dominant signal from v2.0's first three arcs. The post-Arc-5 calibration review now has concrete inputs and is no longer a "wait and see" item.

## Step results summary

| Step | Gate | Result |
|------|------|--------|
| 1 | Plumbing | **PASS** — pool 842 trades, 28 pairs, deterministic, no lookahead, spread per lock (commit `f655b3d`) |
| 2 | Path-shape clustering | **PASS** at K=5 — silhouette 0.4327; all 5 K passed gate; K=5 selected per highest-silhouette rule (commit `2f6e239`) |
| 3 | Capturability | **FAIL** — 0/5 clusters passed §2 conjunctively; arc dies per §7 |
| 4 | Extractability | N/A — blocked by Step 3 |
| 5 | Cross-fold stability | N/A |
| 6 | WFO truth | N/A |

## Per-step detail

### Step 1 — Plumbing (PASS)

- 842 trades across 28 pairs over 2010-01 to 2025-12.
- 701 hard_sl exits, **141 bar_240_system_exits (16.7% cap-binding at pool level).**
- Per-pair n range 17 (GBP_JPY) – 50 (CAD_CHF), median 28; **15 of 28 pairs flagged < 30** per §5 but retained.
- Calendar distribution roughly uniform across years (32–77 trades/yr, 2010–2025).
- Spread sourcing: **per-bar historical from `data/4hr/`**, not fixed values from `spreads_5ers.yaml`. Correct per project's "per-bar MT5 spread data, never hardcoded" rule. Conservative for older data (2010–2012 historical spreads were 5–8 pips on AUD_CAD; current 5ers spread on AUD_CAD is 1.4 pips). My original Step 1 prompt's reference to `spreads_5ers.yaml` was sloppy — CC followed engine convention correctly.
- All §5 gates passed; 21 CI tests under `tests/arc_kh24_v2/`.

### Step 2 — Path-shape clustering (PASS at K=5)

| K | silhouette | max frac | min size | gate |
|---|---|---|---|---|
| 3 | 0.3895 | 0.591 | 139 | PASS |
| 4 | 0.4164 | 0.494 | 85 | PASS |
| **5** | **0.4327** | **0.434** | **50** | **PASS — selected** |
| 6 | 0.3996 | 0.265 | 54 | PASS |
| 7 | 0.4035 | 0.246 | 41 | PASS |

K=5 clusters (selected K):

| cid | size | frac | mono | peaks | pullback | ttp_rel | label |
|---|---|---|---|---|---|---|---|
| c0 | 193 | 0.229 | 0.028 | 0.32 | 0.005 | 0.060 | peak_and_collapse |
| c1 | 365 | 0.434 | 0.501 | 8.40 | 0.222 | 0.459 | unresolved (near stepwise) |
| c2 | 112 | 0.133 | 0.878 | 1.52 | 0.043 | 0.268 | peak_and_collapse |
| c3 | 50 | 0.059 | 0.532 | 3.64 | 0.698 | 0.612 | unresolved (near monotone) |
| c4 | 122 | 0.145 | 0.530 | 30.94 | 0.233 | 0.760 | unresolved (near stepwise) |

Step 2 findings:

- **`pullback_magnitude_median` non-degenerate.** Mode fraction 0.31, well under 0.80. **Refutes §16 Open-08 prior** of 80%+ degeneracy on KH-24 paths. The operational definition (`min(close_r)` between peaks, not literal `min(mfe)`) works.
- **No `random_walk` cluster.** Bare signal does not produce paths failing all three (monotonicity / peaks / pullback) simultaneously. The signal's selectivity prevents pure-noise paths — modest evidence that the bare signal has real structural integrity even without v1.0's filters.
- **No clean §11 centroid matches.** Three clusters labelled `unresolved_*`. §14 calibration anchor doesn't reproduce on bare signal (see below).
- **§14 anchor non-reproduction.** Closest K=4 candidate is c0 (mono 0.527, peaks 28.62, pullback 0.236, ttp 0.740) — off on mono, off by 2× on peaks, off by an order of magnitude on pullback vs anchor (mono 0.576, peaks 14.19, pullback 0.020, ttp 0.847). Explanation: §14 was measured on KH-24's *deployed* 214-trade population (with 1H CIR + currency cap), not the bare 842-trade signal. The protocol's §15 pool floor excludes the deployed population, so v2.0 cannot self-validate on its own calibration anchor.

### Step 3 — Capturability characterisation (FAIL)

Per-cluster §2 evaluation:

| cid | label_final | size | mono | fwd_mfe_p50 | reach_1R | wrong_way_b | shape_tag | §2 |
|---|---|---|---|---|---|---|---|---|
| c0 | peak_and_collapse | 193 | 0.028 | 0.150 | 0.016 | 0.891 | no_magnitude | FAIL |
| c1 | unresolved_f22d041e | 365 | 0.501 | 1.703 | 0.742 | 0.060 | heavy_right_tail | FAIL (clean_shape only) |
| c2 | peak_and_collapse | 112 | 0.878 | 0.440 | 0.071 | 0.607 | no_magnitude | FAIL |
| c3 | unresolved_dd77bf0f | 50 | 0.532 | 0.910 | 0.460 | 0.160 | no_magnitude | FAIL |
| c4 | unresolved_11fee4a0 | 122 | 0.530 | 6.650 | 1.000 | 0.000 | scattered | FAIL (clean_shape + shape) |

Two near-misses:
- **c1** (43% of pool): misses clean_shape by 0.049 (mono=0.501). Otherwise passes everything: magnitude (1.70R), direction (frac_reach_1R 0.742, frac_wrong_way 0.060), shape_tag heavy_right_tail, size 43%.
- **c4** (14.5% of pool): misses clean_shape by 0.020 (mono=0.530) AND shape_tag scattered. Magnitude huge (fwd_mfe_p50 6.65R), direction perfect (frac_reach_1R 1.000, frac_wrong_way 0.000). 87.7% cap-binders.

Early-peak family split (c0, c2): both resolved to `peak_and_collapse` (pct_peak_and_collapse 0.933 and 0.616 respectively). Both die on §2 magnitude floor (no_magnitude shape_tag).

`frac_wrong_way` definition ratified as **Def B (wrong-from-outset)**: MAE ≤ -1R hit before MFE > 0.5R, or MFE > 0.5R never reached. Def A (final_r ≤ -0.5) gives nonsense numbers on a hard-SL design. Def B reconciles with §14 anchor's 0.04. Cross-arc carryover: §17 glossary needs explicit definition.

77/77 CI tests passing (21 Step 1 + 18 Step 2 + 38 Step 3). Determinism confirmed across all three steps.

## Detailed analysis

### Why v2.0 didn't beat v1.0 on this signal

Two structural reasons surfaced by the data:

**1. The §14 calibration anchor was measured on a filtered population, not bare signal.** §14 reference numbers (mono 0.576, peaks 14.19, ttp 0.847) were derived from KH-24's deployed 214-trade population — already filtered through 1H CIR T=0.28 + currency cap=2. The filters do real work: they push the surviving population's path-shape monotonicity up by approximately 0.05. v2.0's §2 floor of 0.55 was calibrated against this filtered anchor, leaving only 0.026 headroom against the anchor itself. On the bare-signal population (842 trades), the trend-rider cohort lands at mono 0.530 — close to where it should be on a bare population, but below the floor calibrated against a filtered one. The protocol cannot mechanically rediscover, on bare signal, what v1.0's hand-crafted filters extracted.

**2. The 240-bar forward window censors the highest-quality cohort.** c4's trades — by every economic measure — are the system's winners. 100% reach 1R, 0% wrong-way, median MFE 6.65R. 87.7% of them hit the 240-bar cap. This means their paths are still running when the forward window closes; their final_r values are dispersed across whatever point on the trajectory they happened to be at bar 240. This dispersion drives shape_tag toward `scattered` rather than the `heavy_right_tail` the underlying MFE distribution would suggest. The protocol cannot see the actual distribution shape on a cohort it has censored.

These two factors compound: c4 misses §2 on both clean_shape (calibration) and shape_tag (censoring). Either alone would have failed it.

### What v1.0 was actually doing

The 1H CIR T=0.28 filter accepts only signals where the prior 1H bar's close was in the bottom 28% of its range — a momentum-confirmation gate. The currency exposure cap prevents portfolio concentration. Together they selected for trades that, post-entry, exhibited more monotone ascent. The §14 anchor mono 0.576 reflects that selection.

v2.0's premise was that path-shape clustering would mechanically rediscover this selection. The evidence here says: it isolates a structurally similar cohort (c4), but slightly differently shaped — enough to miss the §2 floors calibrated against the filtered version.

This is not a methodology failure. It is the protocol working as designed (rejecting candidates that miss thresholds), against a calibration set narrowly.

### If the §2 calibration were relaxed

c4's economics — 7.6 trades/year × p50 6.65R MFE × frac_reach_1R 1.0 × frac_wrong_way 0.0 — suggest that if its MFE could be captured at ~40% via an exit policy, it could clear PASS-DEPLOYABLE on its own. The capture mechanism is the question, not the existence of edge. **This is the testable hypothesis that the cross-arc calibration review will eventually need to resolve.**

## Definitions used in this arc (for clarity in future reading)

- **Monotonicity (`monotonicity_ratio_in_profit`):** Of the bars where the trade is in profit (close above entry), what fraction are at-or-above the previous in-profit bar's level. A clean staircase up = 1.0. A choppy back-and-forth in profit = lower. Never in profit = 0.
- **Local peaks (`local_peaks_count`):** Count of bars where `mfe_so_far_r` strictly increased — i.e., new MFE highs along the path.
- **Pullback magnitude median (`pullback_magnitude_median`):** Across all consecutive pairs of new-MFE-peak bars, the median of (earlier peak's MFE — min `close_r` between the two peaks). Operationally defined on `close_r` not literal `mfe` dip — refutes Open-08's degeneracy concern on KH-24.
- **Time to peak MFE relative (`time_to_peak_mfe_relative`):** `(bar_offset of max mfe) / max(bars_held, 1)`, capped at 1.0. Early peakers ≤ 0.30, late peakers ≥ 0.50.
- **R-unit:** 1R = entry-to-SL distance = 2.0 × ATR(14) for this arc's signal.
- **`frac_wrong_way` (Def B, ratified):** Fraction of trades where MAE ≤ -1R was hit before MFE > 0.5R was reached, OR MFE > 0.5R was never reached. The "wrong from outset" definition — Def A (`final_r ≤ -0.5`) gives nonsense on a hard-SL design.
- **shape_tag (`scattered`):** Classification where `final_r` distribution has std > 2.0 with no clear modal structure. Note: can be a censoring artifact when most trades hit the forward-window cap.
- **Cap-binding:** Trades that exit at `bar_240_system_exit` rather than hard_sl. Their paths are still running; final_r is censored.
- **Pipeline E / D1 (per §3 of protocol):** E = entry filter (admit by classifier at signal time, exit per archetype policy). D1 = deferred classification (admit all trades at entry, classifier evaluates at bar N, branching exits). Distinction matters at Step 4.

## Cross-arc candidates (carried forward to post-Arc-5 calibration review)

1. **§2 monotonicity floor (0.55) calibration.** Direct evidence accumulating across arcs:
   - This arc: c4 missed by 0.020 with otherwise structurally perfect profile; c1 missed by 0.049 with heavy_right_tail and t-stat-worthy direction.
   - Arc 2 redo: cluster 2 missed by 0.009 with t-stat +52.17 on n=2,278.
   - Calibration review should consider whether 0.55 is too high, whether the conjunctive AND should soften to "must clear k of 6 §2 criteria," or whether monotonicity should be measured at median-across-cluster-trades rather than centroid.

2. **`shape_tag` definitions vs forward-window censoring.** When N% of a cohort hits the window cap, their final_r is censored. Current shape_tag rules don't compensate. Candidates: censor-aware shape_tag, dual-measure (final_r and forward-bar-N close_r), shape_tag derived from MFE distribution instead of final_r, or skip shape_tag entirely for cohorts with > 50% cap-binders.

3. **240-bar forward window for 4H signals.** 16.7% pool-level cap-binding on bare KH-24 signal; 87.7% on c4 specifically. The protocol's 240-bar cap is too tight for slow trend-following signals. Candidates: per-arc-configurable window, window scaled to expected hold time of the signal class, or extending the window when cap-binding exceeds a threshold.

4. **§14 anchor population definition vs §15 pool floor.** §14 numbers were measured on the filtered 214-trade deployed population. v2.0's §15 pool floor (≥ 500) excludes this population. The anchor and the protocol describe non-overlapping populations. Calibration review should either: (a) re-derive §14 on a v2.0-compatible population, or (b) explicitly document that §14 is a filtered reference not reproducible under v2.0 mechanics.

5. **§17 `frac_wrong_way` definition.** Ratify Def B (wrong-from-outset) as the protocol's intent. Add explicit definition to §17 glossary. Def A produces nonsense on hard-SL designs.

6. **§16 Open-08 closure.** `pullback_magnitude_median` operational definition is non-degenerate on the bare KH-24 signal (mode fraction 0.31). Open-08 can be closed as resolved.

7. **§11 stepwise-climber and other archetype priors.** None of the §11 patterns matched cleanly. Most clusters got `unresolved_*` provisional labels. Cross-arc evidence is accumulating that §11 centroid ranges are first-pass priors needing empirical refinement (Open-07).

8. **Per-pair n distribution in pools.** 15 of 28 pairs flagged < 30 trades in this arc's pool. §5 keeps them in the pool but the structural concern remains: if a downstream archetype concentrates in low-n pairs, cross-pair stability is suspect. Candidates: per-archetype per-pair stability check at Step 5, or pool-level rule that low-n pairs are excluded from clusters they don't reach a minimum count in.

## Process notes (workflow observations from this arc)

- **CC branch-naming discipline:** every step of this arc landed initially on a non-main local branch (Step 1 on `arc/l_arc_3_step1`, Step 2 unspecified, Step 3 on `feat/d1-pipeline`), then was fast-forwarded or cherry-picked to main. The branch state was inherited from parallel chats (Arc 2 redo, Arc 3, D1 backtester extension). All commits landed correctly on main; no contamination occurred. Discipline note: explicit branch-naming instruction in CC prompts continues to be necessary; CC's session-start branch is not reliable.
- **Spread sourcing convention:** the Step 1 prompt named `spreads_5ers.yaml` as the spread source, but the engine convention (per-bar historical from `data/4hr/`) is what was actually used, correctly. Future arc prompts should reference "per `SPREAD_SEMANTICS_LOCK.md` + engine convention," not a specific yaml file.
- **Tests directory convention:** flat `tests/arc_kh24_v2/` rather than nested `tests/arc_kh24_v2/step{N}/`. Established at Step 1, kept at Steps 2 and 3. Fine; no rework needed.
- **No yaml configs for Steps 2 and 3:** CC correctly noted that within-arc thresholds are protocol-locked (§1.8), so no tunable hyperparameters exist for Steps 2/3. The "config-driven via YAML, no hardcoding" rule is upheld at the protocol level for these steps; no per-step yaml needed.

## Interesting observations

- v1.0 KH-24's filters did more work than expected. The ~0.05 monotonicity uplift from currency cap + 1H CIR is the difference between "v2.0 reproduces v1.0" and "v2.0 produces nothing." The filters aren't aesthetic — they're structurally extracting a different population.
- No `random_walk` cluster despite K up to 7. The bare signal has structural integrity — even without v1.0's filters, the worst clusters are early-peak failures (c0, c2), not pure-noise paths. This is modest evidence that the signal itself is sound; the v2.0 capturability failure is about thresholding the surviving high-quality cohort, not about the signal being noise.
- Both KH-24 v2.0 self-test (this arc) and Arc 2 redo (2026-05-16) close FAIL at Step 3 on §2 floors. Different signals, different mechanisms, same gate failure. Two data points now; if Arc 3 makes three, §2 calibration jumps from "watch item" to "blocking the protocol from producing systems."

## Permanently eliminated by this arc

Nothing. The signal is not eliminated — KH-24 v1.0 deployment continues unchanged. The arc's verdict is on v2.0's ability to *re-derive* a system on bare signal, not on the signal itself.

## Verdict and disposition

**HALT — STEP3_FAIL_NO_CAPTURABLE_ARCHETYPE.** Arc closed.

KH-24 v1.0 deployment unaffected. v2.0 cross-arc calibration backlog grows by 8 items above.

The protocol gave the methodologically correct answer: no archetype clears §2 conjunctively, arc dies. The protocol's strictness in this case filtered out a structurally-edge-carrying cohort (c4). Whether that strictness is right is a cross-arc question, surfaced for post-Arc-5 calibration review.

## Files

All under `results/arc_kh24_v2/`:

- `step1/` — trades_all.csv (842 trades), trades_paths.csv (60,878 rows), plumbing_report.md, configs and helper scripts (commit `f655b3d`)
- `step2/` — path_features.csv, clusters_K{3..7}.csv, centroids_K{3..7}.csv, silhouette_K{3..7}.txt, archetype_assignments.csv, step2_report.md (commit `2f6e239`)
- `step3/` — archetype_summaries.csv, archetype_<label>_distribution.csv per cluster, capturability_pass_list.csv (empty), step3_report.md (commit cherry-picked from `feat/d1-pipeline` to main)
- `ARC_KH24_V2_RESULT.md` — this document

CI tests under `tests/arc_kh24_v2/`: 77 tests across all three steps. Determinism confirmed.

## Handover

Arc is closed. No live doc to inherit.

**For chat continuity:**
- Pipeline D1 backtester extension work in a separate chat: continues. Independent value for future D1-routed archetypes.
- Arc 2 redo: closed (HALT at Step 3, 2026-05-16) for related §2-floor reasons. See `ARC_2_REDO_RESULT.md`.
- Arc 3 (running in parallel chat): unaffected by this closure; continues under v2.0.

**Cross-arc carryover priority surfaces (for post-Arc-5 calibration review):**
- §2 monotonicity floor (0.55) — two arcs provide near-miss evidence; Arc 3 outcome will determine whether this is signal-specific or systemic
- shape_tag definitions vs forward-window censoring
- 240-bar forward window for 4H signals
- §14 anchor population definition vs §15 pool floor structural mismatch
- §17 frac_wrong_way disambiguation
- §11 archetype priors empirical refinement
- Per-pair n distribution stability concern
- §16 Open-08 closure (resolved by this arc's empirical refutation)

Local branch cleanup: `arc/l_arc_3_step1` (Step 1) and `feat/d1-pipeline` (Step 3) still exist locally, identical to main after cherry-picks. Delete at convenience.
