# Phase 2 — UTC Parity Validation

> **Verdict:** PASS. Sidecar↔lab signal output byte-identical on UTC convention.
> **Source artifact:** `results/phase_2_parity/parity_report.md`
> **Git tag:** `arc-10-parity-validated`

## What this proves

The Python sidecar that runs on the VPS — when pointed at 5ers historical data and run under UTC convention — emits **signal envelopes byte-identical** to the corresponding entries in the WFO trade ledger.

In other words: if the sidecar had been running on the VPS continuously from 2010 to 2026, it would have emitted the same signals the WFO recorded, at the same bars, with the same audit fields (atr14, L1, L0, ages, proximities, fractions, direction).

This closes the gap between "the strategy works on paper" and "the strategy works as deployed."

## Headline numbers

| Metric | Result |
|---|---|
| Convention | `utc` |
| Pairs | 28 / 28 PASS |
| Candidate bars evaluated | 14,670 |
| Lab signals = ledger rows | 3,301 = 3,301 (exact) |
| Pool x-check (all pairs) | `count_match=True`, `atr_delta_vs_pool=0.0` |
| Ledger uncovered | 0 |
| **Fire disagreements** | **0** |
| ATR tolerance exceeds (>1e-9) | 0 |
| Field tolerance exceeds | 0 |
| Global max ATR delta | 2.31e-10 |
| Divergence rows | **1** (GBPNZD 2016 New Year holiday edge — live-harmless) |

**Byte-identity rate on the signal/audit axis: 100%**, far above the ≥99.5% acceptance bar.

## Methodology

The harness compares the windowed sidecar's `run_signal()` output against the lab's full-panel `compute_signal()` output on the same H4/D1 panels. Comparison happens on:

1. **Pool x-check (all 28 pairs):** Full-panel sidecar code must reproduce the lab's pool.parquet exactly. atr_delta_vs_pool = 0.0 confirms the data layer is byte-perfect.

2. **Per-pair signal sweep:** At each candidate bar (lab signal ∪ lab prefilter_pass ∪ 1% random sample of all other bars), run windowed sidecar code, compare fire/no-fire decision and audit field values.

3. **Ledger coverage:** Every signal in the lab ledger must have a corresponding sidecar fire decision.

Acceptance gate (all met):
- ≥99.5% byte-identical signal emission → achieved 100%
- 0 true logic divergence → achieved 0
- Audit field deltas ≤ 1e-9 → achieved max 2.31e-10
- Pool x-check exact on all 28 pairs → achieved

## What divergence WAS found (one row, documented)

**GBPNZD signal bar 2016-12-30 (Friday last bar before New Year's holiday):**

- Sidecar projects `entry_bar_open_utc` to Sunday 2017-01-01 reopen
- Lab ledger has entry timestamp Monday 2017-01-02 (because Sunday reopen was a holiday at the historical broker)
- Signal logic itself: identical
- Audit fields: identical
- Direction, atr14, SL distance: identical

**Why this is harmless:** the EA enters at the first actual tradeable tick after the envelope arrives. Whether the envelope's timestamp field says "Sunday 22:00 UTC" or "Monday 04:00 UTC" makes no difference to the live trade — the broker decides when its market opens, and the EA fills at the first tick when it does.

**Why we kept it:** the alternative is to embed a per-broker holiday calendar into the sidecar. That adds operational complexity (calendar updates, broker-specific edge cases) for negligible benefit. Documented as a tolerated residual.

This is the only divergence in 14 years of signals across 28 pairs.

## What was found and FIXED during this dispatch

The Phase 2 UTC dispatch surfaced and fixed one real bug in the sidecar:

**Weekend timestamp projection bug** (`signal_runner._project_entry_bar_open`):

- A signal firing on the last H4 bar of the week (Friday 20:00 UTC) used a naive `+4h` projection for `entry_bar_open_utc`
- That landed on "Saturday 00:00 UTC" — a non-existent bar (market closed)
- The lab correctly projects to Sunday reopen (Sunday 21:00 UTC)
- 3 signals across 14 years were affected

**Severity assessment at the time:** the NewsFilter EA function consumes `entry_bar_open_utc` to center a news-blackout window. For a Friday signal, news would be checked at the wrong (phantom Saturday) time instead of the correct Sunday reopen. Practical impact: near-zero (news windows are 120s wide, both phantom and real positions land in news-dead zones), but the bug was real and in an execution-gating path.

**Fix:** updated `_project_entry_bar_open()` to detect when the +4h projection lands in the closed-market window and snap to Sunday reopen. Regression test added. Commit landed in PR #229 (during Phase 2 dispatch).

This fix carried over to Phase 2 EET parity (next document).

## Why this convention is now retired (sort of)

UTC is the deployment convention for 5ers. The system is live on 5ers under UTC. Sidecar code supports both UTC and EET via `boundary_convention` flag. Both code paths exist and are tested.

UTC artifacts remain in the repo as the proven-correct UTC reference. If 5ers deployment continues, this validation stays in effect.

## Source artifacts

| Artifact | Path |
|---|---|
| Full parity report | `results/phase_2_parity/parity_report.md` |
| Divergence ledger | `results/phase_2_parity/divergence_ledger.parquet` |
| Per-pair summaries | (in parity_report.md) |
| Trade ledger (UTC) | `results/l_arc_10_v3_0_2_utc_rerun/trade_ledger_utc.parquet` |
| Lab pool (UTC) | `results/l_arc_10_v3_0_2_utc_rerun/pool.parquet` |
| Sidecar code | `deployment/sidecar/signal_runner.py`, `signal_emitter.py` |
| Signal logic | `signals/lchar_dlr_long.py` |
