# CANDIDATES.md — Arc 1 Extended fresh-arc candidate hypotheses

**Source arc:** PHASE_L6_ARC1 (L6+ verbatim WFO of L registry rank 1) — FAILed gate on
conditions 1 and 2; closed with `PHASE_L6_ARC1_RESULT.md` (sha to be logged at commit).

**Source characterisation:** This chat's Phase 3 (univariate + bivariate + combination filter
sweeps), Phase 1b-lite (close-to-close horizon, no intrabar SL), and Phase 2c-lite (time-to-payoff,
race outcomes). See `filter_univariate.md`, `filter_bivariate_and_combinations.md`,
`horizon_sweep_lite.md`, `excursion_analysis_lite.md`.

**Status convention:** All entries start `OPEN`. Becomes `PRE-COMMITTED` once a fresh-arc YAML
is written and its expectations are recorded in the arc's `_OPEN.md` *before* the WFO runs.
Becomes `CLOSED-PASS` or `CLOSED-FAIL` once the arc result doc is produced.

**Discipline reminder (L6.0 v1.1 §14.5):** Each candidate below frames its observation as a
factual record of the in-sample sweep, and its fresh-arc gate as a *mechanical* trigger
specification — not an advocacy. The decision about which candidate (if any) becomes the
next arc is made by the user **after seeing this whole document**, not during the sweep.

**Selection-bias reminder:** Phase 3 evaluated ~148 distinct cells against the same 46,597-trade
in-sample set. A binomial null would produce ~7–8 cells at 6/7 or 7/7 fold sign-consistency by
chance alone. Each candidate's selection-bias exposure is recorded explicitly.

---

## CH-001 — Concurrent-density filter (threshold ≤ 13)

| Field | Value |
|---|---|
| ID | CH-001 |
| Status | OPEN |
| Date logged | 2026-05-11 |
| Source arc | PHASE_L6_ARC1 (closed, FAIL) |
| Source phase | Arc 1 Extended Phase 3a univariate + Phase 3c combination |
| Recommended priority | **Highest** — only non-low-survival 7/7-fold-positive cell |

### Observation summary

Trades on which the cross-pair signal-density metric `concurrent_signals_within_3h` evaluated
at signal-bar close (≤ 13 — the 4th decile boundary across all 46,597 in-sample signals) exhibit
mean net R **+0.0346** on n=19,000 (40.8% of pooled), versus pooled mean −0.0051. The lift is
**positive in mean R on every one of the 7 anchored WFO folds** with worst-fold mean
+0.0002R and best-fold +0.0937R; SL hit rate falls from 3.53% pooled to 1.33%; win rate from
50.54% to 54.18%.

The cell is one slice of a fully monotone decile pattern (d1–d10 of the same feature) in mean R,
SL rate, win%, and per-fold sign-consistency.

| Fold | n_in_cell | mean net R | sign |
|---:|---:|---:|:---:|
| 1 | 2,118 | +0.0002 | + |
| 2 | 2,794 | +0.0069 | + |
| 3 | 2,593 | +0.0466 | + |
| 4 | 2,821 | +0.0444 | + |
| 5 | 2,732 | +0.0524 | + |
| 6 | 2,847 | +0.0230 | + |
| 7 | 3,095 | +0.0604 | + |

### Mechanical fresh-arc filter definition (verbatim spec)

- **Computation:** at every 1H bar `N` close, for every of the 28 pairs, evaluate the L4
  univariate-extreme signal (top decile `abs_return_top_decile` neg branch, threshold computed
  from pooled IS sample per L4 spec).
- **Concurrent metric:** for each fired signal at bar `N`, count the total number of fired
  signals across all 28 pairs in the 3-hour window `[N-2, N]` inclusive (i.e. ending at bar
  `N` and including bars `N-2`, `N-1`, `N`). The candidate's own signal is included in the
  count.
- **Gate:** the trade is taken if and only if `concurrent_signals_within_3h <= 13`. Otherwise
  the signal is rejected at signal-bar close and no trade is opened.
- **All other Arc 1 parameters unchanged:** LONG only, entry at bar N+1 open, SL = 2.0 ATR
  (1H ATR at signal bar), time-exit at bar N+2 open, spread floor = 1 native point per pair,
  no exposure cap (the gate replaces the rejected-trade slots; remaining trades are still
  taken without per-pair or per-currency caps in Arc 2).

### Expected per-fold trade volume (from in-sample characterisation)

Min/fold = 2,118 (F1), max/fold = 3,095 (F7). Well above L6 standard `≥ 15` floor;
likely above any reasonable per-fold sample-floor up to ~500.

### Proposed Arc 2 pre-commit gate

Per L6.0 v1.1 §4 (standard conjunctive gate):
1. Worst-fold ROI > 0% across all 7 folds.
2. Worst-fold max DD < 8% across all 7 folds.
3. Trades-per-fold ≥ 15 (or a tighter floor of ≥ 500 if planner re-anchors).

In-sample worst-fold mean R is +0.0002R (essentially zero); the ROI translation depends on the
position-sizing rule. With Arc 1's position size of 0.50% account risk per trade and ~2,118
trades in F1, expected F1 ROI under perfect in-sample = +0.05% × n × mean_R ≈ +0.05% × 2118 ×
0.0002 = ~+0.02% (i.e. essentially zero in the worst fold). **In-sample is a knife-edge on
condition 1.** Any out-of-sample regression on F1 would push it negative. Pre-commit must
acknowledge this thin margin.

### Expectations to log in `_OPEN.md` before run

- **Most-likely PASS mode:** all 7 folds positive ROI but F1 small (< 0.5%), other folds
  +5% to +20% ROI; worst-fold DD 3–6%.
- **Most-likely FAIL mode:** F1 OOS regresses to negative on the order of −0.001 to −0.005
  mean R, breaking condition 1; worst-fold DD still well under 8% so condition 2 likely passes.
- **Lookahead-bias concern:** `concurrent_signals_within_3h` requires real-time knowledge of
  all 28 pairs' signal state at bar N close. Verify the simulator computes this without
  using future bars from any pair. Test: a signal at pair X bar N close cannot count
  pair Y's signal at bar N+1 in its concurrent count.
- **Survival-vs-effect-size tradeoff documented:** alternative cuts at d3 (n=15,272, +0.038R)
  or d5 (n=24,159, +0.020R) are NOT pre-committed for this arc; if Arc 2 fails on F1,
  candidate CH-001a/CH-001b can be opened with alternative thresholds.

### Selection-bias notes

- Cell selected from 11 univariate features × ~10 cells each ≈ 110 univariate observations.
- Distinguishing evidence beyond single-cell observation: the same feature exhibits
  monotone effect across all 10 deciles in mean R, SL rate, win%, and per-fold sign-consistency.
  This level of monotone structure is not consistent with multiple-comparisons noise. See
  `filter_univariate.md` §1 for the full decile table.
- The 7/7 fold sign-consistency in this cell occurs at a worst-fold mean of +0.0002R, which
  is statistically indistinguishable from zero given F1's ~2,100 trades. The cell is at the
  edge of "robust" by sign-count alone; the case for the candidate rests heavily on the
  monotone-decile structure.

### References

- `filter_univariate.md` §1, §8
- `filter_bivariate_and_combinations.md` §1, §2
- `horizon_sweep_lite.md` §2 (close-to-close 7/7 at H=1 and H=6 under same filter)
- `excursion_analysis_lite.md` §1, §3 (race-first delta flips from −1.3pp to +7.0pp)

---

## CH-002 — Concurrent-density filter + structural-pattern restriction

| Field | Value |
|---|---|
| ID | CH-002 |
| Status | OPEN |
| Date logged | 2026-05-11 |
| Source arc | PHASE_L6_ARC1 |
| Source phase | Arc 1 Extended Phase 3b bivariate |
| Recommended priority | **Medium — LOW SURVIVAL** (25% pooled, below 30% threshold) |

### Observation summary

Two-filter cell `concurrent_signals_within_3h ≤ 13 AND structural_pattern ∈ {mixed, pullback_in_uptrend}`:
n=11,675 (25.1% pooled), mean net R **+0.0374**, win 54.6%, SL 1.41%, **7/7 folds positive**,
worst-fold +0.0002R, best-fold +0.094R.

| Fold | n_in_cell | mean net R | sign |
|---:|---:|---:|:---:|
| 1 | 1,285 | +0.0028 | + |
| 2 | 1,738 | +0.0002 | + |
| 3 | 1,534 | +0.0436 | + |
| 4 | 1,724 | +0.0502 | + |
| 5 | 1,731 | +0.0616 | + |
| 6 | 1,789 | +0.0280 | + |
| 7 | 1,874 | +0.0651 | + |

### Mechanical fresh-arc filter definition

CH-001's gate AND `structural_pattern ∈ {mixed, pullback_in_uptrend}` evaluated at bar `N` close.

`structural_pattern` is a categorical computed in the characterisation feature set; its derivation
must be replicated exactly (no IS/OOS leakage — the categorical computed only from data available
at bar N close).

### Expected per-fold trade volume

Min/fold = 1,285. Above 1,000-trade floor by 28%. Likely above any sample floor.

### Proposed Arc pre-commit gate

Same as CH-001 (L6.0 §4 conjunctive). Note: F2's mean R is +0.0002 (lowest among folds),
the knife-edge concern is on F2 here rather than F1.

### Survival caveat

This candidate trims population to 25.1% pooled. The L6.0 informal survival rule treats < 30%
as "low survival" — a flag, not a blocker. The decision to open this as an arc despite the
flag depends on whether the +0.003R effect-size gain over CH-001 alone is judged worth
the ~7,000-trade reduction in OOS signal frequency.

### Selection-bias notes

- Composite of two features each independently selected from the univariate sweep.
- Lift over the parent univariate (CH-001 alone, +0.0346R) is +0.0028R — small.
- The 7/7 sign-consistency improvement over CH-001 (which is already 7/7) is zero. Population
  shrinks ~38% for negligible robustness gain.

### References

- `filter_bivariate_and_combinations.md` §1
- `filter_combinations.csv` row 1 (after the trivial concurrent_le_d8 dedupe row)

---

## CH-003 — Concurrent-density filter + session restriction

| Field | Value |
|---|---|
| ID | CH-003 |
| Status | OPEN |
| Date logged | 2026-05-11 |
| Source arc | PHASE_L6_ARC1 |
| Source phase | Arc 1 Extended Phase 3b bivariate |
| Recommended priority | **Low — LOW SURVIVAL** (11.8% pooled, well below threshold) |

### Observation summary

Two-filter cell `concurrent_signals_within_3h ≤ 13 AND session ∈ {ny, sydney}`:
n=5,498 (11.8% pooled), mean net R **+0.0446**, win 55.5%, SL 1.22%, **7/7 folds positive**,
worst-fold +0.0033R, best-fold +0.085R.

| Fold | n_in_cell | mean net R |
|---:|---:|---:|
| 1 | 608 | +0.0289 |
| 2 | 868 | +0.0033 |
| 3 | 716 | +0.0510 |
| 4 | 764 | +0.0383 |
| 5 | 707 | +0.0524 |
| 6 | 845 | +0.0461 |
| 7 | 990 | +0.0846 |

### Mechanical fresh-arc filter definition

CH-001's gate AND `session ∈ {'ny', 'sydney'}` evaluated at bar `N` close. The session
attribute is derived from the bar's UTC hour per the characterisation feature spec.

### Survival caveat

This candidate trims to 11.8% pooled — well below the 30% survival rule. Min/fold = 608 trades.

The strictly-positive worst-fold mean (+0.0033 vs +0.0002 for CH-001) is the largest among
the 7/7-fold-positive cells, but it comes at a 71% population reduction vs CH-001 alone.

### Selection-bias notes

- Composite of two features each independently selected. NY-or-Sydney specifically picks
  the two non-overlap-active sessions, which is plausibly mechanistically related to the
  concurrent-density effect (overlap sessions have higher concurrency by construction).
  The interaction strength is small (−0.007R), consistent with independent rather than
  amplifying composition.

### References

- `filter_bivariate_and_combinations.md` §1
- `filter_univariate.md` §2 (sydney session standalone +0.059R, 5/2 folds)

---

## CH-004 — Horizon extension to H=6 under concurrent-density filter

| Field | Value |
|---|---|
| ID | CH-004 |
| Status | OPEN — requires OHLC follow-up to fully specify |
| Date logged | 2026-05-11 |
| Source arc | PHASE_L6_ARC1 |
| Source phase | Arc 1 Extended Phase 1b-lite + Phase 2c-lite |
| Recommended priority | **Medium — pending OHLC-based Phase 1b proper verification** |

### Observation summary

Under filter `concurrent_signals_within_3h ≤ 13`, the **close-to-close R** at H=6 bars (i.e.,
extended time exit at bar `N+6` open, with intrabar SL still active at 2.0 ATR) is
**+0.0907 (close-to-close, no SL active)**, win 55.1%, **7/7 folds positive** with
worst-fold mean +0.0335R — a 4-bar lift from H=1's +0.022 worst-fold to +0.034 worst-fold.

The time-to-payoff data show the favorable-vs-adverse spread peaks at H=3 (+11.4pp) and
decays past H=24 — the short-horizon window (H ≤ 6) is where the directional bias is strongest.

### Critical caveat — CSV-lite proxy only

The +0.0907 mean is **close-to-close**, computed without intrabar SL active in the held window.
Trades that touched −2 ATR (SL) between bars `N+1` and `N+6` are still counted at their close-to-close
return rather than the realized −1R loss. The bias direction is **upward**: the proper Phase 1b
with SL active intrabar will yield a *different number*, likely smaller magnitude (and possibly
flipping fold signs).

The deferred Phase 1b proper run must compute:

`R_realized(H_exit) = -1 if intrabar low ≤ entry - 2*ATR_at_N within bars [N+1, N+H_exit]; else (close[N+H_exit] - entry_price) / (2*ATR_at_N) - spread_cost_r`

### Mechanical fresh-arc filter + exit definition (when proper Phase 1b confirms)

CH-001's gate, plus exit policy change:
- **Time exit moves from H=1 to H=6**: hold the trade until bar `N+6` open if SL not hit.
- **SL unchanged**: 2.0 ATR at signal-bar 1H ATR, active intrabar throughout the held window.
- **No trail** (separate candidate CH-007 would add trail; see below).

### Expectations to log in `_OPEN.md` before run

These can only be made precise after Phase 1b proper produces the SL-aware H=6 numbers
on the same trade-set. Best-case scenario (handover anchor): H=6 with SL realized mean R
stays positive on 5+/7 folds, with mean R ≈ 0.5× of the close-to-close proxy (≈ +0.045R).
Pre-flight Claude Code run must produce these numbers before pre-committing.

### Selection-bias notes

- Horizon was swept over 6 values {1, 6, 24, 72, 120, 240}. H=6 was selected as the largest
  H value that preserves 7/7 fold sign-consistency under the close-to-close measurement.
  H=24 dropped one fold to small-negative (−0.003R).
- Horizon selection within the same filter is a 6-cell sweep, expected ~0.4 cells at 7/7
  consistency under noise. The signal is at the edge of multiple-comparisons strength;
  joint evidence with the time-to-payoff peak at H=3 (independently observed in Phase 2c)
  is the supporting structure.

### References

- `horizon_sweep_lite.md` §2
- `excursion_analysis_lite.md` §3
- `CC_PROMPT_for_OHLC_work.md` — Phase 1b proper run spec

---

## CH-005 — Exclusion of top-2 concurrent deciles only

| Field | Value |
|---|---|
| ID | CH-005 |
| Status | OPEN |
| Date logged | 2026-05-11 |
| Source arc | PHASE_L6_ARC1 |
| Source phase | Arc 1 Extended Phase 3a univariate |
| Recommended priority | **Low** — large population preservation but small effect; F1 still negative |

### Observation summary

Lighter version of CH-001: exclude only the worst 2 deciles of the concurrent-density feature
(d9 + d10, both 0/7 folds positive). Threshold: `concurrent_signals_within_3h < 27`.

n=37,426 (80.3% pooled), mean net R **+0.0110**, win 51.86%, SL 1.92%, **6/7 folds positive**,
worst-fold (F1) **−0.0218R** (sign-failure), best-fold +0.0274R.

| Fold | n_in_cell | mean net R | sign |
|---:|---:|---:|:---:|
| 1 | 3,817 | −0.0218 | **−** |
| 2 | 5,652 | +0.0036 | + |
| 3 | 5,456 | +0.0153 | + |
| 4 | 5,676 | +0.0241 | + |
| 5 | 5,474 | +0.0190 | + |
| 6 | 5,634 | +0.0088 | + |
| 7 | 5,717 | +0.0173 | + |

### Mechanical fresh-arc filter definition

CH-001 mechanism but with threshold relaxed to `< 27` (or equivalently `≤ 26`, the d8 boundary).

### Tradeoff

Preserves 80.3% of pooled population (twice what CH-001 keeps) but **F1 is negative**
(−0.0218R). Condition 1 of L6.0 §4 standard gate is unlikely to pass in-sample, let alone OOS.

### Selection-bias notes

- Same parent feature as CH-001; threshold is the only difference. The decile-split selection
  is a 9-cell choice (threshold could be at d1..d9 boundary); this entry records the d8
  boundary specifically.

### References

- `filter_univariate.md` §1 (decile sweep)

---

## CH-006 — Exclusion of structural-pattern `continuation_in_downtrend`

| Field | Value |
|---|---|
| ID | CH-006 |
| Status | OPEN |
| Date logged | 2026-05-11 |
| Source arc | PHASE_L6_ARC1 |
| Source phase | Arc 1 Extended Phase 3a univariate |
| Recommended priority | **Low** — wide population preserved; effect too small to flip pooled mean |

### Observation summary

Exclude the 5,570-trade `structural_pattern = continuation_in_downtrend` slice (the single
most-negative categorical cell, 1/6 folds positive, mean −0.0161R).

Remaining n=41,027 (88.0% pooled), mean net R **−0.0036**, win 50.65%, **4/7 folds positive**,
worst-fold (F1) **−0.0266R**.

| Fold | n_in_cell | mean net R | sign |
|---:|---:|---:|:---:|
| 1 | 4,128 | −0.0266 | − |
| 2 | 6,211 | −0.0209 | − |
| 3 | 6,090 | +0.0010 | + |
| 4 | 6,270 | +0.0134 | + |
| 5 | 6,201 | +0.0072 | + |
| 6 | 6,051 | −0.0113 | − |
| 7 | 6,076 | +0.0044 | + |

### Tradeoff

Removes the worst-pattern slice but does not flip the pooled mean to positive. Conditions 1
of L6.0 §4 standard gate fails on F1, F2, F6 in-sample.

### Selection-bias notes

- Single-cell categorical exclusion. Pattern feature has 5 categorical values; this entry picks
  the worst.

### References

- `filter_univariate.md` §3

---

## Composite note — about adding CH-006 to CH-001

Both Phase 3b bivariate sweep and Phase 3c combination sweep tested
`concurrent_le_d4 AND pattern_not_cont_in_down` (n=17,187, 36.9% pooled). The cell has:

- Mean net R +0.0348 (essentially unchanged from CH-001 alone's +0.0346)
- 6/1 folds positive (loses one fold vs CH-001's 7/0) — F1 mean R becomes −0.0017
- Population −1,813 vs CH-001 alone

Adding the pattern exclusion on top of the concurrent filter **degrades** the fold consistency
without improving mean R. Not recommended as a composite. Documented here so no future
synthesis attempts to "stack" exclusions assuming they always compose favorably.

---

## What is NOT a candidate (and why)

| Cell observed | Reason omitted |
|---|---|
| `hour_utc = 0` (+0.28R, 6/0 folds, n=804) | 1.7% pooled, far below survival. Effect size implausibly large for a 24-cell sweep (one cell hits 6/0 by chance with reasonable probability). Re-test on independent data first. |
| `session = sydney` standalone (+0.059R, 5/2) | 5.3% pooled, low survival. Effect substantially weaker than concurrent_le_d4. |
| `pre_momentum_label = down` (+0.006R, 5/2) | Effect size smaller than +0.01R. Not worth a fresh arc on its own. |
| `mtf_up_or_mixed` exclusion of `aligned_down` | Reduces population only 21%; mean R lift +0.001; fold consistency 4/3 → 4/3. No effect. |

---

## Dependency on OHLC follow-up

Candidates **CH-001, CH-002, CH-003, CH-005, CH-006** are gateable purely on the existing
backtester's filter mechanism (no exit-policy change). Each is implementable as a one-line
YAML config addition to the L4 univariate-extreme spec.

Candidate **CH-004** requires Phase 1b proper (SL active intrabar) to verify that the lite
proxy's +0.091 H=6 mean R remains positive on 7/7 folds with SL realized. The CC prompt
specifies this run; without it, CH-004's "horizon = 6" component should not be pre-committed.

---

## Recommended ordering for fresh-arc selection

If exactly one fresh arc is to be opened next:

1. **CH-001** as Arc 2 — single-feature, parsimonious, only non-low-survival 7/7 candidate. Worst-fold knife-edge documented in expectations.

If a second fresh arc follows (or runs in parallel):

2. **CH-004** as Arc 3, *conditional on* Phase 1b proper confirming H=6 SL-realized mean R stays 7/7 fold-positive. The CC prompt produces this go/no-go input.

If the user judges CH-001's worst-fold knife-edge unacceptable:

3. **CH-002** as Arc 2 (low-survival flag explicit) — tightens to 25% pooled but
keeps worst-fold positive on F2 at +0.0002. F1's mean improves slightly from CH-001 (+0.000 → +0.003).

---

*End of CANDIDATES.md. This file is the master input for the next fresh-arc planner.
On opening Arc 2, copy the chosen candidate's `Mechanical fresh-arc filter definition`
block verbatim into the arc's `_OPEN.md` §3 (planner spec), and record the chosen
`Status` here as `PRE-COMMITTED → Arc N`.*
