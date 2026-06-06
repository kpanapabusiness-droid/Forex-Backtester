# arc 2037 — independent §11 OUTCOME-layer audit of gap (1006) + me_long (1011)

**Chat:** 2000s · **Date:** 2026-06-06 · **Verdict:** DIAGNOSTIC → **KILL** (no new component; gap 1006 + me_long 1011 UNCHANGED, PORTFOLIO — now INDEPENDENTLY outcome-verified)
**Disposition:** KILL · **passed:** N · **Component touched:** gap 1006, me_long 1011 (verification only; dispositions unchanged)

> Extends arc 2036 (which independently outcome-verified fbr) to the two other **LONG, simple-exit** book
> legs: the weekend gap-fill (1006, H4 JPY crosses) and month-end reversion LONG (me_long, 1011, D1 USD
> majors) — both exit only via `{stop_loss, time_exit}` (no trail, no partial close). Exactly as arc 2035
> extended 2034's SIGNAL audit to the whole book, this extends 2036's OUTCOME audit. **Result: BOTH legs'
> gross OUTCOME layer CONFIRMED HONEST from raw price — all 6 checks 100%** (gap 263/263 over 263 IS trades;
> me_long 98/98). The honest take-the-loss signature holds on both: **`stop_loss` exits cluster at ≈ −1R**
> (gap −0.999, me_long −0.977). With fbr (2036), **3 of the 4 book components are now OUTCOME-verified**;
> only **me_short (1019, SHORT + partial-close multi-leg)** remains — the one named remaining OUTCOME slice,
> alongside per-trade COST re-derivation for all legs.

---

## Log reading (step a)

Pulled main (chat 1000s landed arc 1037 — a book signal-layer audit — auto-merged via the union log
driver; verified my 2036 row + theirs both present). No `discovery/STOP`. Highest 2000s id = 2036 → resumed
at **2037**. Corpus state unchanged from my arc-2036 reading: edge-hunt exhausted, 4-way book mean-positive
but not all-folds-positive (2015/2018 wall), lever = operator path-A, recent arcs are DIAGNOSTIC book
characterizations + the §11 independent-verification thread (signal: 2034/2035/1037; outcome: 2036). The
owed §11 step is the OUTCOME layer for the whole book; 2036 did fbr, so the contiguous next move is the
remaining long simple-exit legs.

Fresh-eyes edge check: nothing new — every frontier item (shorts, rel-value, 2018 leg) is mapped dead;
commodity lead-lag DATA-gated (1033/2033). The §11 outcome completion is the genuine highest-EV move.

## What this arc does

Same §11 discipline as 2036 (BUILT `discovery/tools/independent_outcome_audit_book.py`, a parameterized
generalization): read ONLY the trusted `Panel.from_pairs` loader + the engine's `ClosedTrade` ledger (the
CLAIM), re-derive every check from raw OHLC + the documented conventions, never the engine exit code. Both
legs are LONG and exit only via stop_loss or the n-bar time-exit predicate, so the convention set is fbr's
non-trail subset. One verified convention point: the **time-exit predicate fires at bar close and queues a
NEXT-BAR-OPEN fill** (`open_bid` long) — its returned `close_bid` is discarded by the engine
(`_check_exits` stores only the reason → `_fill_pending_closes` fills at the next open), so a `time_exit`
exit_price must equal `open_bid[exit bar]`, identical to fbr's trail fills.

Components (committed configs, per `scripts/cosim_validation/validate_4way_book.py`):
- **gap (1006):** `WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36)`, H4 JPY crosses, time-exit
  n_bars=24, `A1Config(sl_atr_mult=2.0, trail_enabled=False, exit_policy=None)`.
- **me_long (1011):** `MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2)`, D1 USD majors,
  time-exit n_bars=2, `A1Config(sl_atr_mult=2.0, trail_enabled=False, exit_policy="sl_only")`.

The 6 per-trade checks vs raw price: (1) entry==`open_ask[entry]`; (2) SL==`close_ask[sig]−2·indepATR[sig]`
(arc-2034 independent Wilder ATR → R-denominator from raw price); (3) take-the-loss / no missed earlier stop
(Arc-10 defect test); (4) stop-bar consistency; (5) exit px stop→`sl` / time→`open_bid`; (6) gross pnl==
`(exit−entry)·size`.

## Result — all checks 100%

```
gap (1006, H4 JPY)  263 trades   exit mix: time_exit 119 / stop_loss 144
  entry 263/263 · SL 263/263 · take-the-loss 263/263 · stop-bar 263/263 · exit-px 263/263 · pnl 263/263
  R[stop_loss]: -0.999 (-1.046..-0.844, n=144)   R[time_exit]: +1.449 (-0.895..+6.342, n=119)

me_long (1011, D1 USD)  98 trades   exit mix: time_exit 94 / stop_loss 4
  entry 98/98 · SL 98/98 · take-the-loss 98/98 · stop-bar 98/98 · exit-px 98/98 · pnl 98/98
  R[stop_loss]: -0.977 (-1.126..-0.781, n=4)     R[time_exit]: +0.123 (-0.955..+1.423, n=94)
```

The `stop_loss` clusters at ≈ −1R on both (the honest take-the-loss signature; the small spread off −1R is
real per-trade entry slippage — `open_ask` fill vs an SL anchored on the signal bar's `close_ask`, plus the
exit half-spread; favorable entry gaps push a few stops to milder-than-−1R). gap's `time_exit` mean +1.449R
matches its known overshoot-edge character; me_long's `time_exit` mean +0.123R is consistent with its thin
published +0.232% IS edge (gross, pre-cost, pre-risk-scaling). No violations on either leg.

## Verdict & scope

**DIAGNOSTIC → KILL** (no new component). **gap 1006 + me_long 1011 UNCHANGED — PORTFOLIO**, gross OUTCOME
layer now independently verified honest (joining their signal verification in 2035). No canonical change, no
FLAG (engine correct), no council, no null, no OOS (IS-only, folds 2011–2020).

**Remaining §11 slices (for a future chat):**
- **me_short (1019):** the one un-outcome-verified component — SHORT (`open_bid` entry / non-stop exits at
  `open_ask` / `high_ask ≥ sl` SL trigger; SL = `close_bid[sig] + 2·ATR`) AND `sl_partial_close_1r_runner_trail`
  (PARTIAL close at +1R → multi-leg `ClosedTrade` records to group by `position_id`). Meatier; named target.
- **per-trade COST re-derivation** for all four legs (FundedNext, netted at the canonical chokepoint
  `build_fold_stats_from_run`; honest-sweep Part C / PR #264 RESOLVED) — the gross layer is now verified for
  3/4 legs; cost is the transparent deduction on top.

**Tooling:** BUILT `independent_outcome_audit_book.py` (registered) — generalizes the 2036 fbr audit to any
long simple-exit component; the me_short extension needs the short-side fill conventions above.
