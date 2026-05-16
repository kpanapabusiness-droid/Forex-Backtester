# Arc 3 — TRIAL__volatility_regime__d1_atr_top_decile__any__h_120

## Status
- **Opened:** 2026-05-16
- **Closed:** 2026-05-16
- **Active protocol:** L_ARC_PROTOCOL v2.0
- **Calibration anchor:** KH-24 K=4 archetype 3 (passes via Pipeline D1 at t=3)
- **Verdict:** **CLEAN-NULL at Step 3** — zero archetypes pass §2 capturability floors as drawn

> **Reviewer note:** This arc has real cross-arc signal worth investigating. See **"What we had — the Stepwise climber opportunity"** and **"Diagnostic tail (Arc 3D) — recommended"** sections below. The CLEAN-NULL verdict is locked per protocol discipline; the diagnostic recommendations are about generating evidence for v2.1, not rescuing Arc 3.

## Signal under test
| Field | Value |
|---|---|
| Trial ID | `TRIAL__volatility_regime__d1_atr_top_decile__any__h_120` |
| Registry rank | 3 of 5 |
| Family | volatility_regime |
| Base condition | D1 ATR(14) in top decile of trailing 100 D1 bars (per pair) |
| Direction sub-spec at L4 | `any` (no signal-bar directional filter) |
| Signal TF | 1H |
| Horizon | 120 bars (~5 trading days) |
| L4 pooled DSR | 0.999964 (CI [0.978, 1]) |
| L4 raw Sharpe | 0.0212 (CI [0.0152, 0.0270]) |
| L4 pooled mean (ATR-norm) | 0.1290 |
| L4 n_obs_pooled | 106,560 |
| L4 per-pair Sharpe | median 0.061, p25 −0.037, p75 0.248, range [−0.370, 0.490] |

## Arc-3 backtester configuration
| Field | Value |
|---|---|
| Direction | Long only |
| Pair set | All 28 FX (same list as KH-24) |
| Entry | Bar N+1 open after signal on bar N close |
| Initial SL | entry_price − **2.0 × ATR(14)_1H** *(see SL flag below)* |
| Time exit | Bar N+1+120 open |
| Exposure cap | Max 1 open position per pair |
| Risk per trade | 0.5% × reset floor balance |
| Spread | `configs/spread_floors_5ers.yaml` |
| Data window | 2020-10-01 → 2026-01-31 |
| Arc config | `configs/wfo_l_arc_3.yaml` |

## Pre-committed gates (locked at arc open)

- **Step 1 (plumbing):** pool ≥ 500; deterministic; lookahead-invariant; 95th-pct bars_held ≤ 240.
- **Step 2 (clustering):** ≥ 1 K with silhouette ≥ 0.30, no cluster > 90%, min ≥ 30.
- **Step 3 (capturability):** ≥ 1 archetype passes §2 (mono ≥ 0.55, fwd_mfe_p50 ≥ 1.5R, frac_reach_1R ≥ 0.70, frac_wrong_way ≤ 0.30, shape_tag ∈ {tight_unimodal, heavy_right_tail}, size_fraction ≥ 0.10).
- **Step 4–6:** as per v2.0 §8–§10.

Locked. Did not move within the arc.

## Step results

| Step | Gate | Result | Notes |
|---|---|---|---|
| 1 | Plumbing | **PASS** | 2568 trades / 28 pairs; 95th-pct bars_held 120; determinism byte-identical |
| 2 | Path-shape clustering | **PASS** | K=7 chosen; silhouette 0.4177; 4 clusters → 2 archetypes, 3 unassigned |
| 3 | Capturability | **FAIL** | Zero archetypes pass §2 conjunctive floors |
| 4 | Extractability | NOT REACHED | |
| 5 | Stability | NOT REACHED | |
| 6 | WFO | NOT REACHED | |

## Step 1 — detailed result

**Commit:** `debc7ae`. Pool: 2,568 trades; 97.6% skip rate (structural to regime-density signal under max-1-per-pair cap). 95th-pct bars_held = 120 (most trades run to time exit). Determinism PASS (`47c5290d70...`).

Mid-run engine fix in `scripts/arc_3/step1_backtest.py` (analysis-path): SL evaluation on time-exit bar now respects open-fill semantics. Read-only import of `core.spread_floor` primitives — library use, not engine modification (per v2.0 §13).

## Step 2 — detailed result

**Determinism:** PASS. **K=7 chosen** (silhouette 0.4177).

K sweep silhouettes 0.4027–0.4177 across K ∈ {3..7} (range 0.0165). K=7 wins under strict §6 reading; **silhouette tie tolerance is effectively undefined** (see Open-12 below).

Cluster → archetype assignments:

| Cluster | n | Size frac | Centroid (mono / peaks / pullback / ttp_rel / pc) | Archetype |
|---|---|---|---|---|
| 0 | 591 | 23.0% | 0.008 / 0.25 / 0.000 / 0.041 / 0.007 | Early-peak hold |
| 1 | 493 | 19.2% | 0.499 / 8.70 / 0.473 / 0.385 / 0.663 | unassigned |
| 2 | 380 | 14.8% | 0.549 / 24.42 / 0.498 / 0.816 / 0.126 | Stepwise climber |
| 3 | 437 | 17.0% | 0.579 / 2.53 / 0.044 / 0.253 / 0.265 | Early-peak hold |
| 4 | 327 | 12.7% | 0.572 / 7.79 / 0.426 / 0.787 / 0.474 | Stepwise climber |
| 5 | 110 | 4.3% | 0.018 / 1.55 / 0.097 / 0.645 / 0.018 | unassigned |
| 6 | 230 | 9.0% | 0.504 / 8.37 / 1.083 / 0.641 / 0.557 | unassigned |

Aggregated under §6: Early-peak hold (0+3) = 40.0%; Stepwise climber (2+4) = 27.5%; three unassigned standalone groups.

Degeneracy audit clean (no feature > 80% modal — contrast KH-24's pullback degeneracy).

## Step 3 — detailed result

§2 capturability evaluation per group:

| Archetype | Size | Mono | Local peaks | MFE p50 | Reach 1R | Wrong way | Shape tag | §2 verdict |
|---|---|---|---|---|---|---|---|---|
| Early-peak hold (0+3) | 1028 (40.0%) | 0.251 | 1.22 | 0.35R | 12.0% | 98.8% | no_magnitude | FAIL |
| Stepwise climber (2+4) | 707 (27.5%) | **0.559** | **16.73** | **3.34R** | **83.6%** | 38.3% | bimodal | **FAIL** |
| Cluster 1 | 493 (19.2%) | 0.499 | 8.70 | 1.95R | 80.9% | 73.2% | no_magnitude | FAIL |
| Cluster 6 (diagnostic) | 230 (9.0%) | 0.504 | 8.37 | 2.18R | 83.5% | 61.7% | heavy_right_tail | FAIL-size |
| Cluster 5 (diagnostic) | 110 (4.3%) | 0.018 | 1.55 | 0.27R | 1.8% | 100.0% | no_magnitude | FAIL-size |

Failing floors:
- **Early-peak hold (0+3):** 5/6 fail — mono, mfe_p50, reach_1R, wrong_way, shape_tag. Aggregation effect dominates (see flag below).
- **Stepwise climber (2+4):** 2/6 fail — wrong_way (38.3% vs 30% floor) and shape_tag (bimodal not in allowed set). **Passes magnitude / reach / size / mono cleanly.**
- **Cluster 1 (unassigned):** 3/6 fail — mono (0.499), wrong_way (73.2%), shape_tag. Peak-and-collapse signature (pc=0.663) explains the wrong_way concentration.

**Arc 3 verdict: CLEAN-NULL at Step 3.** Per v2.0 §7 and project methodology (§1.7-1.8, KH-29 precedent): locked gates not loosened post-hoc. No within-arc rescue.

---

## 🔴 What we had — the Stepwise climber opportunity

This is the most important finding from Arc 3 and the reason the CLEAN-NULL verdict deserves close reading.

Stepwise climber (clusters 2+4 aggregated, 27.5% of pool, 707 trades) presents the profile of a real, structural, archetype-level edge:

| Property | Value | §2 floor | Margin |
|---|---|---|---|
| Size fraction | 27.5% | ≥ 0.10 | +17.5pp |
| Monotonicity | 0.559 | ≥ 0.55 | +0.009 |
| Local peaks | 16.73 | 5–30 | within range |
| **MFE p50** | **3.34R** | **≥ 1.5R** | **+1.84R (2.2× floor)** |
| **Reach 1R** | **83.6%** | **≥ 0.70** | **+13.6pp** |
| Wrong way | 38.3% | ≤ 0.30 | **−8.3pp (fail)** |
| Shape tag | bimodal | {tight_unimodal, heavy_right_tail} | **fail** |

**Final R distribution:** p25 = −1.00, p50 = **+1.85R**, p75 = +3.80R, p95 not yet measured.

Median trade in this archetype closes at +1.85R. 83.6% reach 1R MFE. p75 is +3.80R.

**This isn't a borderline case. It's a real archetype-level edge.**

### Why §11 already knows how to extract it

§11 row 7 reads verbatim:

> "Bimodal fwd_mfe distribution (two distinct modes ≥ 1R apart) | Split exit variant of stepwise/monotone | Per the base archetype | Half-off at TP1 (lower mode), trail remainder | Same"

The protocol has the exit policy ready: half-off at TP1 (lower mode) captures the wrong_way mass that currently dies as stop-outs; trail remainder captures the upper mode. The bimodality *is* the signal — some trades stop out cleanly, others run.

### Why §2 cuts it

§2's `shape_tag ∈ {tight_unimodal, heavy_right_tail}` floor excludes bimodal at the capturability gate, despite §11 having a dedicated row for it. **§2 and §11 are internally inconsistent on this point.** The protocol's own exit-family map admits bimodal as a valid archetype; the protocol's own capturability floor denies it admission.

### Why we don't rescue within-arc

Two precedents matter:

- **JL invalidation** (project_brief.md): 79.8% win rate → −50% ROI when rebuilt ex-ante. Retrofit-after-result is the canonical project failure.
- **KH-29 closure** (CHANGELOG.md): closest miss in project history (0.016R). Accepted the null. Methodology held.

If we change §2 within Arc 3 to admit bimodal because Stepwise is right there, we're moving a threshold to make a result pass. That's JL.

**The Stepwise climber finding is worth more as cross-arc evidence than as a within-arc rescue.** As evidence → Open-13 → v2.1 amendment → applies cleanly to Arcs 4, 5, and the KH-24 calibration anchor. As rescue → ad-hoc protocol modification → result can't be trusted, can't compare to other v2.0 arcs.

The discipline is what makes the finding meaningful.

---

## 🟡 Flag: Aggregation may be hiding more

Cluster 2 and cluster 4 are mapped to the same §11 row (Stepwise climber) by centroid pattern, but their forward-geometry properties differ structurally:

| | Cluster 2 (n=380) | Cluster 4 (n=327) |
|---|---|---|
| Monotonicity | 0.549 | 0.572 |
| Local peaks | **24.42** | **7.79** |
| Pullback | 0.498 | 0.426 |
| ttp_rel | 0.816 | 0.787 |
| **pct_peak_and_collapse** | **0.126** | **0.474** |

Local peaks differ by 3× (24 "many small steps" vs 8 "few bigger moves"). pct_peak_and_collapse differs by **3.7×** (cluster 2 rarely collapses; cluster 4 collapses ~half the time). These are not the same archetype.

**Likely un-aggregated picture (not measured yet):**
- Cluster 2 alone: lower wrong_way (likely 20–30%), tighter monotonicity, possibly passes §2 except for shape_tag (Open-13 issue regardless).
- Cluster 4 alone: higher wrong_way (likely 50%+) from its higher collapse rate, fails §2.

The aggregate's 38.3% wrong_way may be masking cluster 2 ~25–30% and cluster 4 ~50%+. Same applies to Early-peak hold (0+3), where cluster 0 (mono 0.008) dominates the aggregate down from cluster 3's 0.579.

This is Open-14 evidence. **Within Arc 3 we cannot test it without changing the verdict. As cross-arc evidence it is among the most important findings.**

**Recommended for diagnostic tail (see below):** measure un-aggregated capturability per cluster, including the ones currently masked by aggregation.

---

## 🟡 Flag: Stop-loss distance / horizon asymmetry

Current SL = 2.0 × ATR(14)_1H on a 120-bar (5-day) horizon. The stop is sized in volatility units; the hold is in time units. Over 120 1H bars, total expected price-movement std dev ≈ √120 × per-bar-vol ≈ 11 × per-bar-vol.

So **the 2 ATR stop is ~0.18 std deviations of expected horizon movement.** Even a directionally neutral random walk should stop out frequently at this distance. This is the structural reason all three full-evaluation archetypes failed on `frac_wrong_way`.

| SL choice | σ of horizon | Comment |
|---|---|---|
| 2.0 × ATR_1H (current) | 0.18 σ | Way too tight for horizon |
| 3.0 × ATR_1H | 0.27 σ | Tight but workable |
| 4.0 × ATR_1H | 0.36 σ | Reasonable for horizon |
| 2.0 × ATR × √(h/24) ≈ 4.5 × ATR_1H | 0.40 σ | Brownian-motion-consistent |
| 2.0 × ATR_4H | ~5–6 × ATR_1H | Mismatched units, often used in practice |

### What wider stops would do to Stepwise climber (estimated)

Doubling SL distance:
- **Halves wrong_way mass.** Estimate Stepwise → ~20%, passes the 0.30 floor comfortably.
- **R-rescales the magnitude floors.** "Reach 1R" becomes "reach to 2× original distance" — reach rate likely drops from 83.6% to ~60%. Could narrowly fail or narrowly pass the 0.70 floor depending on actual MAE distribution.
- **MFE p50 in R units halves** to ~1.7R. Still passes 1.5R, narrowly.

**Honest answer: wider stops might rescue Stepwise climber on wrong_way without sinking it on reach_1R, but not guaranteed.** Measurement is what tells us.

### Where to fix it

Three structural paths, in increasing scope:

1. **Within-archetype SL** (Step 4 artefact). §11 already provides per-archetype initial SLs (Monotone ascent 1R, Stepwise 1.3R, Early-peak hold 0.8R, Peak-and-collapse 0.5R, V-shape 1.5R, Random walk 1R). These are R-multipliers on top of whatever arc-level base SL exists. v2.0 §11 expects archetype-specific SLs to be applied at Step 4. **The arc-level 2× ATR base is what's miscalibrated for the horizon — the §11 archetype multipliers don't compensate for arc-level SL/horizon asymmetry.**

2. **Arc-level SL scaled to horizon.** Set SL = 2.0 × ATR × √(h/24) at arc open. For h=120 this is ~4.5× ATR_1H. Simple, principled (Brownian-consistent), applies cleanly to any horizon. Would need adoption as v2.1 default.

3. **Horizon-aware frac_wrong_way floor.** Keep SL fixed, scale the §2 wrong_way ceiling by horizon/SL ratio. Less clean — embeds the asymmetry in the gate rather than fixing it at source.

Path 2 is the cleanest fix. Path 1 is what §11 was meant to do but requires Step 4 to actually run, which Arc 3 didn't reach.

### Compute cost of a diagnostic sweep

Per SL variant:
- Step 1 re-run with new SL: ~5–10 minutes (same signal generation, different exit simulation)
- Step 2 re-run on new trade pool: seconds (clustering is fast)
- Step 3 re-run with new metrics: seconds

Per-variant cost: <15 minutes. **Three SL variants × two aggregation modes = 6-cell sweep ≈ 30–60 minutes one CC dispatch.**

### Pros and cons

**Pros:**
- Cheap and deterministic
- Converts Open-13/14/15 from speculation to evidence
- Direct input to v2.1 calibration decision
- Same pattern as KH-27/28/29 (post-closure diagnostic tail)
- Doesn't change Arc 3 verdict

**Cons:**
- Scope creep before Arc 4 opens
- Motivated-reasoning risk — needs explicit framing as cross-arc evidence, not arc-rescue
- One more thing to coordinate before v2.1 work

---

## 🟢 Diagnostic tail (Arc 3D) — recommended

**Status: flagged, awaiting reviewer decision.**

Run a diagnostic-only phase Arc 3D after Arc 3 closure commits, before Arc 4 opens. Locked frame: Arc 3 verdict is CLEAN-NULL regardless of what 3D shows; 3D produces evidence for v2.1, not a re-evaluation of Arc 3.

Scope (2D sweep, one CC dispatch):

| | SL=2.0×ATR_1H | SL=3.0×ATR_1H | SL=4.0×ATR_1H |
|---|---|---|---|
| Aggregated (§6 rule as-is) | baseline (matches Arc 3) | diagnostic | diagnostic |
| Un-aggregated (per cluster) | diagnostic | diagnostic | diagnostic |

Outputs per cell: §2 floor evaluation per cluster/archetype, fwd_mfe and final_r distributions, frac_wrong_way breakdown.

**Questions Arc 3D answers:**
1. Does cluster 2 alone pass §2 (except shape_tag) → confirms Open-14 has teeth
2. Does Stepwise climber pass wrong_way at wider SL → quantifies Open-15
3. Does the SL/aggregation interaction surface any cells that pass all 5 non-shape_tag floors → strongest evidence for Open-13 amendment

**Decision required from reviewer:**
- (a) Authorise Arc 3D before Arc 4 — adds ~1 chat session, generates evidence while Arc 3 is fresh
- (b) Defer Arc 3D until after Arc 5 — fold into v2.1 calibration pass alongside Arcs 4 and 5 evidence
- (c) Skip Arc 3D — accept Open-13/14/15 on Arc 3 evidence alone

My recommendation: **(a)**. The data is fresh, scripts are warm, the diagnostic is cheap. Worth knowing Stepwise climber's true profile before deciding whether Arc 4 should adjust SL convention or aggregation rule.

---

## Cross-arc candidates (full list)

**Open-12 — Silhouette tie tolerance** (Step 2)
K selection rule needs a tolerance definition for "ties: smaller K preferred." Arc 3 chose K=7 on a 0.0021 margin. Propose 0.01 absolute or 5% relative threshold below which smaller K wins. v2.1 calibration.

**Open-07 evidence — §11 Random walk over-specified** (Step 2)
Cluster 6 matches Random walk's positive criteria (local_peaks 8.37, pullback 1.083R) but fails the monotonicity ≤ 0.30 ceiling at 0.504. Propose splitting Random walk into stricter and looser monotonicity variants.

**Open-07 evidence — §11 Peak-and-collapse time_to_peak ceiling tight** (Step 2)
Cluster 1 matches collapse signature strongly (pc=0.663) but with peak timing 0.385 vs §11's ≤ 0.30 ceiling. Consider widening to 0.40 or splitting.

**Open-13 — §2 shape_tag / §11 row-7 incompatibility** (Step 3, **highest priority**)
§11 row 7 defines bimodal as a valid archetype with its own exit policy; §2 shape_tag floor excludes bimodal entirely. Stepwise climber (27.5% pool) is the evidence — passes 4/6 floors with mfe_p50 = 3.34R and reach_1R = 83.6%, killed only by bimodal exclusion. v2.1 amendment proposed: §2 admits `bimodal` when accompanied by §11 row-7 routing AND modes meet ≥ 1R separation criterion (operationalised via Hartigan dip + mode-separation check).

**Open-14 — Same-archetype aggregation can destroy capturable sub-clusters** (Step 3)
§6 same-archetype rule aggregated cluster 0 (mono 0.008) with cluster 3 (mono 0.579) → aggregate mono 0.251, kills cluster 3's capturability. Cluster 2 vs cluster 4 within Stepwise climber show 3× difference in local_peaks and 3.7× in pct_peak_and_collapse — clearly different archetypes despite sharing §11 row. v2.1 amendment proposed: same-archetype aggregation conditional on cluster-pair disparity ≤ X across each §2 floor; otherwise treat as separate sub-archetypes carrying the same §11 exit policy. **Arc 3D (recommended) measures this directly.**

**Open-15 — SL-distance / hold-horizon asymmetry inflates frac_wrong_way** (Step 3)
2.0 × ATR_1H on h=120 = 0.18 σ of horizon — structurally too tight. All three full-evaluation archetypes failed on wrong_way. v2.1 consideration: SL scaled to horizon (e.g., 2.0 × ATR × √(h/24)) as arc-level default, OR archetype-specific Step-4 SL adjustments per §11. **Arc 3D (recommended) quantifies the magnitude.**

---

## Interesting observations

- **97.6% skip rate** structural to regime-density signal converted by exposure cap. Cross-arc relevance: Arc 5 (mtf_alignment h_120) likely similar; Arc 4 (h_001) very different.
- **D1 ATR top-decile is a regime, not an event.** Surviving sub-population would have been "trades that entered early in a vol regime survive cleanly" — but bridge from pooled edge to capturable archetype failed under §2 as drawn.
- **Signal works on average but not for any captured sub-population.** L4 pooled DSR preserved; the bridge fails. This is exactly the failure mode v2.0 was designed to surface. Protocol working as intended — except where §2 and §11 disagree on bimodal.
- **K=7 surfaced 7 distinct clusters with only 2 §11 archetypes covering ~67%.** §11 likely under-specified for the data variety Arcs 3+ will cumulatively generate. v2.1 §11 refresh warranted after Arcs 4 and 5.

## Disposition

Arc 3 closes CLEAN-NULL at Step 3. Verdict locked. Five cross-arc items logged for v2.1.

**Diagnostic tail (Arc 3D) recommended before Arc 4 opens** to convert Open-13/14/15 from speculation to evidence. Reviewer decision required.

**Next regardless of 3D decision:** Arc 4 — registry Entry 4 (`TRIAL__univariate_extreme__bar_range_top_decile__neg__h_001`). Structurally different signal class; will not reproduce Arc 3's failure modes.

## Handover

This document → rename to `ARC_3_RESULT.md`, commit to `docs/arc_results/`. Cross-arc candidates fold into project knowledge as v2.1 backlog. Arc 3D decision pending reviewer.
