# Arc 10 Gate-Fidelity Defect — the lesson that reset this repo

> **Required reading before opening any new arc.** This is the single most
> valuable artefact carried out of the pre-2026-06-02 research. It is kept
> LIVE (not archived) on purpose. Everything else from that era was archived
> and its numbers are not to be trusted.
>
> **One line:** Arc 10 (DLR) was declared PASS-DEPLOYABLE and deployed on the
> strength of a backtest gate that was scored by a fast *replay* with a
> stop-skipping bug. An SL-honest re-run flips the worst-fold from **+13.77%
> to −8.32%** and folds-positive from **11/11 to 2/11**. Arc 10 does **not**
> clear PASS-DEPLOYABLE. It is KILLED.

---

## The defect

The Step-5 gate ranked candidates using a fast path *replay* (`simulate_path`
/ `realized_r_3p5`, now retired) instead of walking bars through the SL-honest
`MultiPairBacktester`. For the `sl_partial_close_1r_runner_trail` exit, that
replay only applied the −3.5×ATR stop to the **runner, after** the +1R partial
fired (a `sl_breach > tp1` guard). It **ignored stop breaches that happened
before the partial** — and it suppressed a same-bar stop. So a trade whose low
pierced the stop on the way up was booked as a partial+runner **win** instead
of a −1R **loss**.

A live SL-honest engine (and the deployed MQL5 EA, whose broker stop is always
binding) stops those trades at −1R first. The replay flattered them.

## The footprint (canonical Arc 10 v3.0.2 pool, 3152 trades)

- **699 trades (22.2%)** had a pre-partial stop breach.
- **695 of them** were booked by the replay as **wins** that are actually
  **losses**: a gross swing of **+608.8 R → −695 R**.

## The flip — replay gate vs SL-honest engine (search folds F1–F11, r_base 0.40%)

| metric | replay (published) | SL-honest | verdict |
|---|---|---|---|
| worst-fold ROI % | **13.77** | **−8.32** | fail |
| mean-fold ROI % | 28.85 | −4.50 | fail |
| per-trade win % | 71.35 | 49.30 | — |
| worst-fold trailing DD % | 7.73 | 14.13 | fail (>8%) |
| folds positive (of 11) | 11 | 2 | fail |
| kills (8% close-all) | 0 | 3 | fail |

On the SL-honest engine **Arc 10 NO LONGER CLEARS PASS-DEPLOYABLE on any
axis.** The live EA already executes SL-honest, so live results track the
SL-honest column — but the backtest gate that *justified deployment* rested on
the inflated replay numbers.

## The deeper lesson — internal consistency ≠ correctness

The most dangerous part: the replay **self-validated**. Re-running the
canonical reconstruction reproduced the published gate numbers *exactly*
(worst-fold ROI 13.77% vs 13.77%, trailing DD 7.73% vs 7.73%). Every internal
reconciliation passed. The numbers agreed with each other — and were still
wrong, because the bug lived in the **shared scorer** that produced all of
them. Agreement between two paths that share a component proves nothing about
that component.

A result is only as honest as the single engine that scored the P&L. If P&L
can be reconstructed from a precomputed shortcut, the shortcut — not the
bar-walking engine — is what you actually deployed.

## What changed because of this (the standing rules)

1. **The fast replay is RETIRED (2026-06-02).** `MultiPairBacktester`
   (`core/sim/`) is the **sole** engine that may score a trade for a gate.
   There is no `realized_r_3p5` / `sl_breach > tp1` path in the live tree.
2. **Take-the-loss is a locked invariant.** Any stop breach at or before the
   +1R partial bar resolves to −1R (full position); a same-bar stop is SL-first;
   ambiguity never resolves to a win. Pinned by
   [`tests/sim/test_take_the_loss_invariant.py`](../tests/sim/test_take_the_loss_invariant.py)
   (CI-gated).
3. **No trusted carry-forward numbers.** Every pre-2026-06-02 gate number was
   produced by the retired replay and is recorded only as "what was attempted",
   never as evidence. Deployable-system count = **0**.

## Provenance

The full audit (fold-by-fold deltas, the 699 disagreement trades, SHAs) is
preserved read-only under
`archive/results/diagnostics/arc_10_sl_honest_gate/ARC_10_SL_HONEST_GATE.md`
and `archive/results/diagnostics/arc_10_path_peakr/PATH_PEAKR_DIAGNOSTIC.md`.
The audit is conservative throughout (its residual assumptions all *under*-count
the inflation), so the real defect is at least as large as reported.
