# Arc 10 Signal-Parity Re-run Procedure & Expected Delta (2026-05)

> **Source PR:** PR #189 (signal parity engine) — original procedure
> **Related PRs:** PR #193 (signal-level EET timezone audit + DLR `_date_to_d1_index` State-B fix) extends the rerun trigger; PR #197 (Amendment 6 EET daily-DD boundary) does NOT affect Arc 10 PASS-VIABLE numbers under UTC convention
> **Author:** CC (jovial-mcnulty-1b0855); updated CC_22 docs refresh 2026-05-25
> **Date:** 2026-05-24 (original); 2026-05-25 (post-PR-#193 / #197 update)
> **Status:** procedure documented; actual re-run is a follow-up workstation operation, now gated on TWO prerequisites (per PR #193 audit OPEN-ARC-10-RESUMPTION-PREREQS): canonical `sl_partial_close_1r_runner_trail` (landed in PR #195) AND signal-EET fix (landed in PR #193) — both prerequisites now met as of PR-#195 merge.
> **No closure modification** — Arc 10 `docs/archive/arc_results/ARC_10_RESULT.md` is unchanged.
>
> **PR-numbering note:** the source dispatch (`CC_15_SIGNAL_PARITY_ENGINE.md`) used the notional label "PR #187"; the actual GitHub PR is #189. See TODO.md §"PR numbering convention".

---

## §1 Context

PR #189 (signal parity engine) made three engine changes:
1. Two mid-leak fixes in [core/features/distance.py](../../core/features/distance.py)
   (prior_session_high uses mid_high; prior_session_low uses mid_low)
2. Trail manager activation + ratchet switched to mid close (hit detection
   stays bid; see [core/sim/trailing_stop.py](../../core/sim/trailing_stop.py))
3. 5ers EET bar-boundary convention added to
   [core/data/aggregator.py](../../core/data/aggregator.py) (UTC default
   unchanged, so existing caches are byte-identical)

Arc 10 (DLR — D1 swing-low rejection long) used the A1 architecture
with `sl_partial_close_1r_runner_trail` exit policy. Per
[ARC_10_RESULT.md](../archive/arc_results/ARC_10_RESULT.md), the
load-bearing feature is `L1_minus_L0_atr` (D1 high-low slope magnitude,
EXP-02 contributed 116% of HTF LOO drop).

**Pre-run delta hypothesis:** near-zero. Justification:

- `L1_minus_L0_atr` lives in [core/features/multi_tf.py](../../core/features/multi_tf.py)
  `_d1_close_slope_magnitude` and **already computed on mid** pre-PR-187
  (the multi_tf module was correctly mid-anchored from PR #185 onwards).
- Arc 10's other features come from price_geometry, vol_regime,
  cross_pair, all of which were already mid-priced in the V3 path.
- The only features that change in Arc 10's feature envelope are the
  two distance leaks: `prior_session_high_distance` and
  `prior_session_low_distance` (now use mid_high/mid_low instead of
  high_bid/low_ask). These are 2 of ~17 features in the load-bearing
  subset.
- A1 architecture (system_level_filter) uses the trail manager. Arc 10's
  exit policy `sl_partial_close_1r_runner_trail` uses a runner trail
  whose activation/update was switched from bid-close to mid-close in
  PR #189. This may shift trail-exit fills by half a spread on average
  (~0.1 pip on majors → ~0.005R per trade on 2.5×ATR SL).
- 5ers EET aggregation is opt-in (UTC default); Arc 10 ran on UTC and
  remains on UTC unless explicitly re-aggregated.

**Acceptable range:** worst-fold ROI delta within ±2pp (vs the original
26.49% worst-fold), worst-fold DD delta within ±2pp (vs the original
9.22%). Anything outside → investigate the trail-manager delta first
(half-spread × trail exits is the leading expected driver).

### §1.1 PR #193 + Amendment 6 impact on the rerun

The original §1 hypothesis was written before PR #193's signal-EET audit and
PR #197's Amendment 6 daily-DD boundary change. Status under each:

- **PR #193 (signal-EET):** Arc 10's `signals/lchar_dlr_long.py` `_date_to_d1_index`
  was classified **State B (silent same-EET-day D1 lookahead) under EET storage**
  but State A under UTC. The original Arc 10 PASS-VIABLE numbers were computed
  under UTC convention, so the original verdict was NOT affected by the bug.
  The signal module has been fixed (canonical `get_htf_index_at(..., require_fully_closed=False)`);
  re-running under UTC convention should produce numerically identical results
  to the original (the legacy idiom was byte-identical to the canonical utility
  under UTC).
- **Amendment 6 (PR #197, EET daily-DD boundary):** Arc 10's daily-DD breach
  count was computed under UTC convention. Amendment 6 only affects per-day
  bucketing when `boundary_convention="5ers_eet"`; under `"utc"` (Arc 10's
  setting) `compute_per_day_max_dd` takes the legacy `.normalize()` branch
  byte-identically. No impact on the existing PASS-VIABLE numbers.

**Net:** rerun procedure unchanged from §1. The PR #193 and Amendment 6 work
is forward-hygiene for any future EET-convention rerun of Arc 10 (a separate
exercise from this UTC-parity rerun).

---

## §2 Original Arc 10 closure numbers (search WFO, 11 folds)

From [results/l_arc_10/step_5/wfo_results.csv](../../results/l_arc_10/step_5/wfo_results.csv)
row 1 (winning A1 config):

| Metric | Value |
|---|---|
| Architecture | A1 |
| SL multiplier | 3.5 × ATR |
| Exit policy | sl_partial_close_1r_runner_trail |
| Exposure cap | unlimited |
| Search worst-fold ROI | 0.2649 (26.49%) |
| Search worst-fold DD | 0.0922 (9.22%) |
| Search worst-fold ratio | 5.4185 |
| Search mean ROI | 0.4987 (49.87%) |
| Search mean DD | 0.0385 (3.85%) |
| Search mean ratio | 16.18 |
| Sign consistency | 11 / 11 folds positive |
| Negative folds | 0 |
| Total trades | 2162 |
| Verdict (closure §1) | PASS-VIABLE |
| Verdict (closure §10 amendment 3) | PASS-DEPLOYABLE-PROVISIONAL |

---

## §3 Re-run procedure

Workstation invocation (paths assume parent project, not worktree):

```powershell
# 1. Sync worktree's PR #189 code into project root (if running from project root)
git checkout claude/jovial-mcnulty-1b0855  # or whatever the merged main branch is

# 2. (Optional) Rebuild 5ers_eet cache if the re-run will use EET bars.
#    For parity with original Arc 10 (which used UTC), KEEP UTC convention.
#    No cache rebuild needed.

# 3. Re-run Arc 10 Step 5 WFO
py -m scripts.l_arc_10_v3.step_5 \
    --config configs/l_arc_10_v3/arc_open.yaml \
    --out results/l_arc_10_pr187_rerun/step_5/
```

Expected runtime: 6-12 hours (full 11-fold WFO across 28 pairs ×
2010-2020 search window).

---

## §4 Comparison

After the re-run completes:

```powershell
py -m scripts.l_arc_10_v3.compare_runs \
    --original results/l_arc_10/step_5/wfo_results.csv \
    --rerun    results/l_arc_10_pr187_rerun/step_5/wfo_results.csv \
    --out      docs/calibration/arc_10_signal_parity_delta.md
```

(The `compare_runs` script is a candidate follow-up; if not present, do
the diff manually.)

Per-fold comparison checklist:
- ROI delta per fold (should be near-zero with possible -0.005R/trade
  trail-spread bias)
- DD delta per fold (should be near-zero)
- Signal count (should be unchanged — signal logic is V3 path unchanged)
- Trade count (should be unchanged — entries fire at same bars; only
  trail exits may shift by ≤1 bar)

---

## §5 Outcomes

### §5.1 Expected outcome (near-zero delta)

If the re-run confirms ±2pp ROI/DD per fold:
- Arc 10's edge is genuinely signal-derived, not bid-price/spread
  artefact
- Arc 10 verdict UNCHANGED: PASS-VIABLE → PASS-DEPLOYABLE-PROVISIONAL
- Document the run with a delta summary in this doc's §5.1 section
- No closure modification

### §5.2 Surprise outcome (delta > ±2pp)

If the re-run shifts worst-fold by more than ±2pp:
- Investigate trail-manager change first (most likely driver)
- Run with trail manager pinned to bid-only via a debug switch (if added)
  to isolate
- If isolated to trail: document as "trail-mid causes Xpp ROI shift on
  Arc 10" in this doc's §5.2 section
- If NOT isolated to trail: investigate distance.py mid changes
  (less likely to be material since they're 2/17 features and Arc 10's
  load-bearing feature is unaffected)
- Closure §1 still NOT modified per dispatch rule; instead add a §12
  "Signal-parity re-run" appendix to the closure noting the delta

### §5.3 Documentation update

After the run completes, fill in §6 below with the actual per-fold delta
table and an outcome verdict (5.1 vs 5.2).

---

## §6 Re-run results

*To be filled after workstation re-run.*

---

End of doc.
