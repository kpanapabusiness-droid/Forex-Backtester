> # ⚠️ SUPERSEDED 2026-05-13
>
> This Phase 2 doc was written under L6.0 framing as a fresh arc with pre-committed CH-001 filter, now SUPERSEDED by `L_ARC_PROTOCOL.md` v1.0.
>
> **What this phase established (preserved in cross-arc registry):** `concurrent_signals_within_3h ≤ 13` filter produced a passing WFO under L6.0 framing — real edge confirmed. CH-001 is documented in `CANDIDATE_HYPOTHESES.md`.
>
> **Under the new protocol:** the filter mechanism is one candidate among several in step 4 of the Arc 1 redo. The threshold of 13 is not pre-anchored — step 3's predictor scan may surface a sharper or different boundary. CH-001 must surface as ≥ Tier 2 predictor in step 3 per the protocol calibration check (`L_ARC_PROTOCOL.md` §15); if it doesn't, the protocol is miscalibrated.
>
> **Do not use this doc as authority for new arc work.** Read `L_ARC_PROTOCOL.md` first.
>
> ---

# PHASE_L6_ARC1_P2_OPEN — Arc 1 Phase 2: concurrent-density filter (CH-001)

**Phase:** `l6_arc1_p2_concurrent_filter`
**Status:** OPEN — locked once Keanu signs off; no modifications post-lock per L6.0 §10.
**Parent arc:** Arc 1 (`TRIAL__univariate_extreme__abs_return_top_decile__neg__h_001`),
verbatim phase closed FAIL per `PHASE_L6_ARC1_RESULT.md` (gate conditions 1+2 failed all 7 folds).
**Candidate source:** `CANDIDATE_HYPOTHESES.md` CH-001 (to be appended on lock).
**One-change-per-phase:** addition of a single boolean filter on `concurrent_signals_within_3h`
applied at signal-bar close. No other parameter modified vs Arc 1 verbatim.

---

## §1. Scope and disposition

Arc 1 verbatim FAILed on conditions 1 (worst-fold ROI > 0%) and 2 (worst-fold DD < 8%)
across all 7 folds. Characterisation per L6.0 §14 identified a monotone effect across all
10 deciles of `concurrent_signals_within_3h` in mean net R, SL rate, win%, and per-fold
sign-consistency. The bottom 4 deciles (`concurrent_signals_within_3h ≤ 13`) form the only
non-low-survival cell in the characterisation sweep with 7/7 folds positive in mean R.

This phase tests whether adding that filter — and nothing else — produces a WFO that passes
the L6.0 §4 gate on the same 7 anchored folds with the same OOS window. The hypothesis is
that the concurrent-density filter removes correlated-cluster signal-stacking days, which
the Arc 1 verbatim result doc identified as the dominant DD mechanism (basket-blow during
high-concurrency periods, with no exposure cap to brake it).

Per L6.0 §1, this is a permitted subsequent filter phase. Per L6.0 §3, this phase does NOT
preempt Arc 2 (already opened on a different registry signal) — Arc 1 P2 and Arc 2 run on
independent schedules.

**Disposition rules (inherited from L6.0):**

- Gate PASS → Arc 1 candidate-survives. Phase 3 (likely exit-policy variation per CH-004)
  may follow under one-change-per-phase, opened in a separate doc.
- Gate FAIL → Phase 2 closes as clean null. Findings recorded; no further filter phases on
  Arc 1 unless a new candidate graduates via §14.7 from this phase's characterisation.

---

## §2. Signal definition (unchanged from Arc 1)

Verbatim from `scripts/lchar/run_layer4.py` for `univariate_extreme` family,
`abs_return_top_decile` base condition, `neg` sub-spec, 1H signal TF, h=1 horizon.

No modification to the signal definition, threshold computation, IS/OOS split convention,
or pair universe (28 pairs) vs Arc 1.

---

## §3. The one change — concurrent-density filter

### §3.1 Definition

Add a single boolean filter at signal-bar close, applied AFTER the L4 signal fires and
BEFORE the trade is opened at bar N+1 open:

```
take_trade = (signal_fires) AND (concurrent_signals_within_3h <= 13)
```

If the filter rejects, the signal is dropped; no trade opens; no position is taken.
The bar N+1 open is not reserved or otherwise consumed.

### §3.2 `concurrent_signals_within_3h` computation

At every 1H bar N for every of the 28 pairs:

1. Evaluate the L4 signal for that pair at that bar (binary fires/does-not-fire).
2. Count: across all 28 pairs, how many fired the L4 signal in the rolling 3-hour window
   `[N-2, N]` inclusive (i.e., bars N-2, N-1, N). The candidate's own signal is included in the count.
3. The value of `concurrent_signals_within_3h` for the signal at pair X bar N is that count.

### §3.3 Lookahead invariant (BLOCKING; CI-enforced)

The count at pair X bar N close must use signal evaluations at bars `[N-2, N]` across all 28
pairs — and **only** those bars. It must NOT use any data from bar N+1 or later on any pair.

A test harness must verify, on a small synthetic sample: if pair Y fires at bar N+1, the
count at pair X bar N does NOT change. Hard error if violated. This is non-negotiable per
the project's general lookahead discipline.

### §3.4 Everything else unchanged

| Parameter | Value | Source |
|---|---|---|
| Direction | LONG only | Arc 1 §3 |
| Entry | bar N+1 open | Arc 1 §3 |
| SL | 2.0 × ATR_1H at signal bar | Arc 1 §3 |
| Time exit | bar N+2 open (h=1) | Arc 1 §3 |
| Trail | none | Arc 1 §3 |
| Pair universe | 28 pairs (locked) | L6.0 §5 |
| OOS folds | 7 anchored expanding | L6.0 §3 |
| Position size | 0.50% account risk per trade | Arc 1 §3 |
| Spread floors | `configs/spread_floors_5ers.yaml` | Arc 1 §3 |
| Exposure cap | none | Arc 1 §3 |
| Currency cap | none | Arc 1 §3 |

The filter REPLACES the rejected-trade slots with nothing — surviving trades are still
taken without per-pair or per-currency caps. Adding any cap would be a second change
and is not permitted in this phase.

---

## §4. Gate (verbatim from L6.0 §4)

Conjunctive across all 7 OOS folds:

1. Worst-fold ROI > 0% across all 7 folds.
2. Worst-fold max DD < 8% across all 7 folds.
3. Trades-per-fold ≥ 15 across all 7 folds.

All three must pass on every fold. No soft-pass. No median-fold pass.

---

## §5. Output artefacts (per L6.0 §6 + §14)

- `results/l6/arc1_p2/wfo_fold_results.csv` — per-fold ROI, DD, trade count, mean R, gate disposition
- `results/l6/arc1_p2/fold_{01..07}/{in_sample,out_of_sample}/{trades.csv,equity_curve.csv,summary.txt}`
- `results/l6/arc1_p2/fold_{01..07}/fold_dates.json`
- `results/l6/arc1_p2/signals_features.csv` — characterisation output per §14.4 (one row per taken signal)
- `results/l6/arc1_p2/PHASE_L6_ARC1_P2_RESULT.md` — closure record

Determinism: each run twice; all outputs byte-identical sha256.

---

## §6. Files modified

- `configs/wfo_l6_arc1_p2.yaml` — new file, derived from `configs/wfo_l6_arc1.yaml` with
  the filter block added (see §7 below for the diff spec).
- `CANDIDATE_HYPOTHESES.md` — append CH-001 entry on lock (status: PRE-COMMITTED → Arc 1 P2).
- `scripts/walk_forward.py` — only if needed to wire the filter through. If the simulator
  already supports a `filter:` block in YAML, no code change. If not, the minimal
  implementation that reads the filter expression and applies it at signal time.

---

## §7. YAML diff spec (for Cursor)

Starting from `configs/wfo_l6_arc1.yaml`, produce `configs/wfo_l6_arc1_p2.yaml` identical
in all respects EXCEPT:

```yaml
# (existing arc1 contents above unchanged)

# NEW: filter block applied at signal-bar close, before trade open
filter:
  feature: concurrent_signals_within_3h
  op: '<='
  threshold: 13
  # The count is computed at signal-bar N close as the sum across all 28 pairs of
  # L4 signal fires in the rolling 3-hour window [N-2, N] inclusive.
  # Lookahead invariant: counts use data only from bars [N-2, N] across pairs.

# Output redirect
output:
  results_dir: results/l6/arc1_p2
  trades_csv: trades.csv
  fold_results_csv: wfo_fold_results.csv
  summary_txt: summary.txt
```

If the simulator does not currently support `filter:` as a top-level block, the
walk_forward.py changes are: (a) read the filter, (b) at signal time compute
`concurrent_signals_within_3h` from the L4 signal mask for all 28 pairs at the current
bar and the previous 2 bars, (c) gate the trade.

---

## §8. Determinism + CI hooks

Per L6.0 §6:

- No RNG (verified in v1.0 of Arc 1; this phase adds no RNG-dependent code).
- Re-run twice. All output sha256s byte-identical.
- Pre-flight: confirm `wfo_l6_arc1.yaml` re-runs to byte-identical Arc 1 artefacts on the
  simulator commit used for this phase. If Arc 1's verbatim is not reproducible on the
  current simulator commit, **stop and investigate before running P2**.

---

## §9. L4-bar-identity & spread-floor application sanity (per L6.0 §7)

Same audit ranges as Arc 1. The filter shrinks the trade-set to ~40.8% of pre-filter pool,
which will narrow the spread-floor distribution but not eliminate it; the audit range
needs interpretation in that context.

Expectation: the trade-set under the filter has fewer high-volatility, post-tail bars (those
typically appear in concurrent_signals_within_3h > 13 cells). Spread-floor application rate
should fall vs Arc 1's 38.75% — but remain elevated vs the unconditional audit range.
Soft WARN if the floor rate stays > 50%; hard fail if any of the simulator's other audit
checks trip.

---

## §10. Post-lock modification policy

Per L6.0 §10: post-lock, no modifications. Specifically:

- The threshold value `13` is locked. If P2 fails, a P3 with a different threshold
  (e.g., d3 boundary at 11, or d5 boundary at 16) requires a fresh OPEN doc.
- The lookback `[N-2, N]` is locked. A different window (e.g., `[N-5, N]`) requires a
  fresh OPEN doc.
- The filter feature `concurrent_signals_within_3h` is locked. Using a different concurrent-
  density metric (e.g., USD-basket only, or weighted by per-pair Sharpe) is a different
  hypothesis and requires a separate phase.

If gate condition 1 fails by a narrow margin on F1 specifically (the known knife-edge),
the result is a clean FAIL; the temptation to "just relax the F1 expectation" is the
exact post-hoc rationalization the methodology lock exists to prevent.

---

## §11. Pre-commit expectations (locked at sign-off, vs actual at close)

### §11.1 Trade frequency per fold

| Fold | Pre-commit expected | In-sample anchor | Comment |
|---:|---:|---:|---|
| 1 | 2,000 – 2,800 | 2,118 (characterisation) / ~2,650 (extrapolated to full F1 OOS) | F1 OOS in WFO covers 3 more months than characterisation CSV (2020-10-01 → 2021-07-01); add ~25% to characterisation count. |
| 2 | 2,500 – 3,300 | 2,794 | |
| 3 | 2,300 – 3,100 | 2,593 | |
| 4 | 2,500 – 3,300 | 2,821 | |
| 5 | 2,400 – 3,200 | 2,732 | |
| 6 | 2,500 – 3,300 | 2,847 | |
| 7 | 2,800 – 3,600 | 3,095 | |

Far above L6.0 §4 condition 3 floor of 15. No expectation that condition 3 binds.

### §11.2 Per-trade economics

In-sample mean net R per fold ranges +0.0002 (F1) to +0.0604 (F7); aggregate +0.0346.

Pre-commit expected:
- F1 OOS mean R: **−0.005 to +0.010** (knife-edge; in-sample is +0.0002).
- F2–F7 OOS mean R: **+0.005 to +0.045** (regression toward mean from in-sample range).

### §11.3 Worst-fold DD

Arc 1 verbatim had DD 39.7% to 91.3% across folds, driven by correlated-pair basket-stack
during high-concurrency periods. This phase's filter directly removes the highest-density
bars (top 6 deciles, where 0–6 folds were positive). Mechanism for basket-stack is
structurally reduced but not eliminated (residual concurrency in d1–d4 still exists,
just at lower magnitude).

Pre-commit expected: **worst-fold DD 3–8%**. Condition 2 (DD < 8%) is borderline.
If DD blows past 8%, the filter is necessary-but-not-sufficient and an exposure cap
becomes the natural next phase.

### §11.4 Highest-prior-risk DD folds

Arc 1 verbatim DD ranking: F2 (91.3%) > F1 (88.8%) > F6 (74.0%) > F5 (65.1%).

Pre-commit expectation: F1, F2, F6 remain the highest-DD-risk folds in this phase,
but at much lower absolute magnitudes (5–8% range vs 60–90% range).

### §11.5 Most-likely PASS mode

All 7 folds ROI > 0%; F1 small positive (0% – 1% ROI); F2–F7 in the 3% – 20% ROI range.
Worst-fold DD 4% – 7%. Gate PASS on all three conditions. This becomes Arc 1 candidate-
survives; phase 3 (likely exit-policy / horizon extension per CH-004) opens.

### §11.6 Most-likely FAIL mode

F1 OOS mean R regresses to small-negative (~ −0.003 to −0.008 R), breaking condition 1.
F2–F7 still positive ROI. Worst-fold DD 4–7%, condition 2 passes. Trade count well above
condition 3 floor.

Specifically: condition 1 FAIL on F1 alone is the highest-probability null mode given
the in-sample +0.0002 worst-fold knife edge.

### §11.7 Other plausible failure modes

- **Lookahead-bias slip:** filter implementation accidentally uses bar N+1 data on a
  cross-pair signal evaluation. Result would be unrealistically good (filter is essentially
  trained on the future). The §3.3 invariant test prevents this.
- **Spread-floor application rate spike:** filter inadvertently selects bars where the
  broker reports zero spread on all 28 pairs (correlated-quiet moment), causing the floor
  to apply almost universally and inflate spread costs beyond the per-trade R estimate. 
  Monitored via §9 audit.
- **Trade count below 1,500/fold:** in-sample F1 minimum is 2,118; if OOS drops by >30%
  on any fold it suggests a population effect not captured by the characterisation.
  Condition 3 still passes at 15-trade floor but the gate-rich-side of the trade pool
  is thinned.

---

## §12. What ships at close

Per L6.0 §6:

- All §5 output artefacts produced and committed.
- `PHASE_L6_ARC1_P2_RESULT.md` closure record, with:
  - §11 pre-commit vs §11 actual side-by-side.
  - Gate disposition (PASS / FAIL with reasons).
  - Determinism confirmation.
  - L4-bar-identity report.
  - Spread-floor application stats per pair.
  - Characterisation deliverable per §14 (separate from gate disposition).
  - "What ships next" — next phase planning input.

- `CANDIDATE_HYPOTHESES.md` updated: CH-001 status moves from PRE-COMMITTED to either
  CLOSED-PASS or CLOSED-FAIL.

- Any new candidate hypotheses surfaced during this phase's characterisation graduate
  per §14.7.

---

## §13. Sign-off

This phase is locked when Keanu signs off in writing on this document. From sign-off forward:
no modification to §3 (the one change), §4 (the gate), §5 (output spec), §7 (YAML diff),
§10 (post-lock policy), §11 (pre-commit expectations). Modifications would require a fresh
OPEN doc and a new phase identifier.

Signed off (Keanu): ____________________
Date locked: ____________________
Simulator commit (`git rev-parse HEAD`): ____________________
Methodology lock version: L6.0 v1.1
Methodology lock sha256: 4fd870b1d17380e4fc4fbfda5a43f7775d313c7a5f50dbfd1f06a3e49c519c26 (per Arc 2 OPEN — verify at lock)

End of PHASE_L6_ARC1_P2_OPEN.
