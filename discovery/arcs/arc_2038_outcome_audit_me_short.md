# arc 2038 — independent §11 OUTCOME-layer audit of me_short (1019) — the LAST book leg

**Chat:** 2000s · **Date:** 2026-06-06 · **Verdict:** DIAGNOSTIC → KILL (no new component;
me_short UNCHANGED, PORTFOLIO) · **Disposition:** KILL

## Step (a) — log read (fresh eyes, honest-era)
Resumed 2000s at 2038 (highest in range = 2037, +1). No `discovery/STOP`. The edge-hunt is
**exhausted**: the deployable candidate is the 4-component reversion book (gap 1006 JPY-H4 + me_long
1011 USD-D1 + fbr 1013 USD-H4 + me_short 1019 USD-D1), mean-positive but NOT all-folds-positive; the
2018-positive "clean leg" is unfound across ~12 routes (structure shorts 1014/2009/2011/3011, trend
short 3010, up-gap 1016, rel-value 2010, deep-continuation 2012, carry-unwind 1017, vol-state 3012,
weekly 3014, shock-continuation 3019…). MENU closed; the lever sits with the operator (path-A). My
chat's recent thread is the **§11 independent re-verification** — the Arc-10 institutional defense
("never deploy on a single engine's word"): SIGNAL layer DONE for all 4 (2034 fbr, 2035 the other
three); OUTCOME layer done for 3 of 4 (2036 fbr, 2037 gap + me_long). Both 2036 and 2037 explicitly
flagged the SAME owed next step for a fresh chat: **me_short — the last un-outcome-verified leg**
(SHORT fill conventions + `sl_partial_close_1r_runner_trail` partial→multi-leg grouping). Fresh-eyes
edge check: no untouched mechanism (commodity lead-lag DATA-gated per 1033/2033; all else dead ground)
→ completing the book's gross-outcome §11 verification is the genuine highest-EV autonomous move.

## Why this arc
me_short is the hardest / most Arc-10-prone outcome config in the book: it is the only SHORT leg AND
the only one running `sl_partial_close_1r_runner_trail` — a +1R PARTIAL close (50%) then a runner that
trails 1R above its favorable extreme, producing MULTI-LEG `ClosedTrade`s grouped by `position_id`.
The reset-causing Arc-10 defect *was* an outcome-layer bug (a gate that scored profits while skipping
pre-partial stop breaches) — and the partial/runner machinery is exactly where a same-bar
stop-vs-partial precedence error would hide. So this is the single highest-stakes remaining §11 slice.

## Method (the §11 discipline — independent, not reproduction)
Built GEOMETRY-ONLY `discovery/tools/independent_outcome_audit_me_short.py` (generalizes 2036/2037 to
SHORT + partials). Reads ONLY (a) the trusted `Panel.from_pairs` loader and (b) the engine's
`ClosedTrade` ledger (the CLAIM); re-derives every check from raw OHLC + the documented conventions;
NEVER calls the engine's exit/partial code. The engine source (`fill.py`, `multipair_backtester.py`,
`sl_partial_close_1r_runner_trail.py`) was READ to know WHAT to assert, not copied into the
re-derivation — the audited invariants are convention-robust, not a re-coded partial/trail walk
(that = transcription).

Committed config audited (validate_4way_book.py): `MonthEndReversionShortSignal(threshold_atr=1.0,
into_bars=2)`, D1 USD majors, `A1Config(config_id="arc_1019", sl_atr_mult=2.0, trail_enabled=False,
exit_policy="sl_partial_close_1r_runner_trail")` (native A1 TrailManager is short-deferred, so only the
exit-policy partial/trail binds).

SHORT conventions re-derived from raw price:
- ENTRY  == `open_bid[entry bar]` (short sells the bid).
- SL     == `close_bid[sig] + 2·indepATR[sig]` (sig = entry−1; ATR = arc-2034 INDEPENDENT Wilder(14)
  mid, shift1 → the R-denominator is re-derived from raw price). R unit = `sl − close_bid[sig]` =
  `2·ATR[sig]`; the exit policy's `r_atr` = `sl_atr_mult · atr_at_entry` = the same value.
- TAKE-THE-LOSS / ARC-10: no bar strictly between entry and a leg's exit has `high_ask ≥ sl`.
- PARTIAL leg (`partial_close_1r`): intra-bar fill at `tp1 = entry − r_atr`. The engine checks the
  intra-bar stop in `_check_exits` BEFORE the partial hook (`_process_bar` step 2a→2b), so a partial
  leg EXISTING implies `high_ask[partial bar] < sl` — the **take-the-loss precedence for the partial**
  is asserted directly.
- RUNNER `stop_loss` leg: intra-bar `high_ask ≥ sl`, fill at `sl`.
- RUNNER `runner_trail_stop` leg: queued at prior bar close, fills NEXT-bar `open_ask`
  (`_fill_pending_closes` runs BEFORE the intra-bar SL check → the take-the-loss reference is the FILL
  price: `open_ask[xi] < sl`, the SHORT mirror of fbr's 2036 long resolution).
- PNL == `direction.sign·(exit−entry)·size` (short sign = −1).

Plus a MULTI-LEG STRUCTURAL check (within each fold, where `position_id` is unique): every
partial-bearing position reconstructs as exactly 1 partial + 1 runner leg, linked by
`parent_position_id`, with equal halves summing to the original size (no size leakage).

## Result — ALL CHECKS PASS
153 unique legs across IS folds 2011–2020 (exit mix: 58 `partial_close_1r`, 57 `runner_trail_stop`,
38 `stop_loss`). **6/6 checks 153/153 (100%)** — entry, SL geometry, take-the-loss/no-missed-stop
(the Arc-10 test), per-reason stop-bar consistency, exit-px, gross pnl all clean.

Honest take-the-loss R signature (gross, pre-cost):
- `partial_close_1r`: **+1.000 / +1.000 / +1.000** (n=58) — the partial fills exactly at +1R, as
  designed; a deceptive engine could not pin all 58 to exactly +1R.
- `stop_loss`: **−1.003 mean** (−1.293 … −0.841, n=38) — honest ≈ −1R; the off-−1R spread is real
  per-trade entry slippage (`open_bid` fill vs an SL anchored on the signal bar's `close_bid`) + the
  exit half-spread; favorable entry gaps push a few stops milder than −1R.
- `runner_trail_stop`: **+0.865 mean** (−0.373 … **+7.938**, n=57) — the runner's convex right tail
  (the reversion-overshoot character), bounded below near −1R+ as the trail gives back ≤1R from peak.

Multi-leg structural: **58/58 partial-bearing positions accounted** — 57 closed partial+runner pairs
(all linked, equal halves summing to full size) + 1 partial-only position. That 1 (fold 2011, pos 12)
is a partial that fired with its runner **still OPEN at the fold's last bar** — confirmed independently:
the engine's `run()` does NOT force-close at run end (it only reports `n_open_at_end`), and
`n_open_at_end == 1` for exactly that fold. So a Dec-2011 month-end short that partialed and whose
runner trailed into 2012 (outside the one-year OOS fold) legitimately contributes only its
already-realized partial leg — boundary truncation, NOT size leakage (the partial leg itself passed
6/6). Structural: **CLEAN.**

## Verdict
**DIAGNOSTIC → KILL** (no new component). me_short 1019 UNCHANGED — PORTFOLIO, its gross OUTCOME layer
now independently confirmed honest from raw price (joins its 2035 signal verification). Engine correct
→ **no canonical change, no FLAG.** No council, no null, no OOS (IS-only).

**§11 milestone: the WHOLE deployable book is now gross-verified end-to-end.** Signal layer (2034 fbr +
2035 gap/me_long/me_short) AND gross outcome layer (2036 fbr + 2037 gap/me_long + 2038 me_short) are
all independently re-derived from raw price — no geometry bug, no price-lookahead, no missed-stop
(Arc-10) defect, honest take-the-loss, honest partial/runner multi-leg accounting. The book rests on
no un-audited bespoke code.

## Threads (remaining §11 slice, for a fresh chat)
ONE layer left before deployment-eligibility: **per-trade COST re-derivation** for all four legs
(FundedNext netted at the canonical chokepoint `build_fold_stats_from_run`, the explicit subject of
honest-sweep Part C / PR #264 RESOLVED, applied as a transparent deduction on the now-verified gross
numbers). The gross entry/exit/R layer is done for 4/4; the cost deduction is the last independent
re-derivation owed. (The 4-way book's all-folds-positive FAILURE is already cost-robust per arc 3022 —
2/10 neg even at κ=0 — so cost re-derivation is a deployment-honesty check, not an edge question.)

## Tooling
BUILT `discovery/tools/independent_outcome_audit_me_short.py` (registered, step (i)). Reusable for any
SHORT partial-runner-trail component; the multi-leg reconstruction + `n_open_at_end`-backed
open-runner accounting + SHORT take-the-loss-at-fill-price pattern generalize. No canonical change.
