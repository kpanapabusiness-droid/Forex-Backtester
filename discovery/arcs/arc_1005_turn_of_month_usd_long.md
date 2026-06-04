# Arc 1005 — Turn-of-Month USD-Long (Calendar-Flow Mechanism)

> **Arc id:** 1005 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-04
> **Final verdict:** **FAIL (cheap-kill at triage).** A REAL turn-of-month USD-strength drift exists
> (USDCHF +0.177 ATR / 5d, 73% years positive) but is SUB-COST: USDCHF triage worst −2.78%, mean −0.79%;
> USDXXX basket worst −3.55%, mean −1.61%. The drift is too small to clear FundedNext cost + SL-first.
> **Lever tested:** MECHANISM CHANGE — a non-price-direction CALENDAR-FLOW edge (month-end rebalancing).

Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first). First non-directional mechanism of
the run. Built a reusable time-exit predicate (BUILT tool); canonical engine realizes all P&L.

## (a) Log read — FRESH EYES

Pulled main. Corpus = 5 directional-long FAILs (arc 0/1000/1001 H4 majors, 1002 D1, 1003 crosses) + arc 1004
(exit engineering on the cross trend, FAIL). Binding constraint = **EDGE < COST**, and it's an entry/cost
problem not an exit problem. Steer: **change the MECHANISM** (non-price-direction). No STOP.

## (b) Idea + observation

**Mechanism = CALENDAR FLOW (turn-of-month).** Month-end institutional rebalancing/hedging creates documented
FX flow. Tested with the CORRECT metric — mean forward DRIFT in ATR units (the +1R-before-SL capture is blind
to small drifts) — PER PAIR (month-end USD flow pushes XXXUSD vs USDXXX oppositely, so pooling washes out).

D1 turn-of-month window (last 2 + first 3 trading days), mean forward 5-day drift in ATR, IS 2010–2020:

| pair | TOM mean fwd5 (ATR) | years positive |
|---|---|---|
| EURUSD | −0.180 | 36% |
| GBPUSD | **−0.263** | 18% |
| USDCHF | **+0.177** | **73%** |
| EURGBP | +0.051 | 55% |
| NZDUSD | +0.031 | 55% |
| USDJPY | +0.028 | 64% |
| AUDUSD | −0.012 | 45% |
| USDCAD | −0.023 | 36% |

**A real turn-of-month USD-strength signal exists** (XXXUSD drift DOWN — EURUSD/GBPUSD; USD/CHF drifts UP),
consistent with month-end USD demand and concentrated in EUR/GBP/CHF (European flows). **Long-only-exploitable
via USDCHF** (+0.177 ATR/5d, 73% years positive) — the most promising long signal of the run. (Day-of-month
buckets pooled are noise; month-of-year per-pair swings are 11-sample noise.)

## (c)+(d) Characterize + cheap kill

**Signal:** long at the close of the 3rd-to-last trading day of each month (captures the run into month-end +
first days of next month via a 6-bar TIME exit), SL=2·ATR (disaster stop, take-the-loss). Time exit via the
BUILT signal-class `make_time_exit_predicate` (A1Config.time_exit_bars is NOT wired into the Order — FLAG).
A1Config: `exit_policy=None, trail_enabled=False` → only SL + the time-exit predicate. Caveat: USDCHF is
1-of-8 (multiple-comparison / cherry-pick risk); the OOS judge is the real test.

| universe | IS trades | +1R capture | mean final_r | 2013 | 2016 | 2019 | worst | mean |
|---|---|---|---|---|---|---|---|---|
| USDXXX basket (CHF/JPY/CAD) | 384 | 0.268 | −0.0105 | −0.01% | −3.55% | −1.26% | −3.55% | −1.61% |
| **USDCHF only** | 128 | 0.281 | −0.0100 | +0.68% | −2.78% | −0.27% | −2.78% | −0.79% |

Exit mix ~70% time_exit / ~30% hard_sl (the predicate fired correctly). **Both sub-cost → KILL.** USDCHF is
the least-negative result of the whole run (mean −0.79%, tiny DDs 1–3%, mean final_r ≈ break-even gross) — but
still net-negative every-fold-but-one. The real +0.177 ATR gross drift is too small to overcome FundedNext
cost + the ~31% 2ATR stop-out rate.

## Council — NOT convened

No worthwhile-ceiling/diagnosis fork: the gross drift is real but measurably too small to clear cost; the
remaining moves (sweep window/hold/entry-day on a cherry-picked single pair) are rescue-sweeps the council
would call "ritual." Cheap-killed on the apparatus.

## Final verdict — FAIL (cheap-kill)

The **turn-of-month USD-long is not deployable**. The mechanism is REAL (a genuine, mechanistically-grounded,
cross-year-consistent month-end USD-strength drift — the first real gross signal of the run) but the drift
magnitude (~0.18 ATR/5d) is **sub-cost** once FundedNext spread/commission/slippage + the SL-first take-the-loss
apply. EDGE < COST.

## Lessons (candidate for LESSONS.md)

1. **A REAL turn-of-month USD-strength drift exists on D1** (USDCHF +0.177 ATR/5d, 73% years positive; XXXUSD
   drift down — EURUSD −0.18, GBPUSD −0.26), concentrated in EUR/GBP/CHF. But it is **sub-cost** — too small
   to clear FundedNext cost + SL-first. The first NON-directional mechanism tested, and EDGE<COST holds for it
   too: the cost hurdle is **mechanism-general**, not specific to price-direction signals.
2. **The binding constraint is now extremely well-supported across SIX families** (5 directional + 1 calendar)
   and three levers (entry, timeframe, universe, exit, mechanism): on liquid FX with FundedNext costs + SL-first,
   the realizable gross edge of simple long-only signals (~0.1–0.2 ATR or coin-flip capture) is below the cost
   hurdle. A deployable long needs a MUCH larger gross edge per trade, or a structurally different cost profile.
3. **Calendar/seasonality month-of-year per-pair swings are 11-sample noise** — do not chase them.

## Threads

- **Closed:** turn-of-month USD-long (real but sub-cost); calendar month-of-year seasonality (noise).
- **Open / next:** the search needs a signal with a LARGE gross edge per trade (to clear cost), or a different
  cost regime. Candidates not yet tried: (a) rare high-conviction multi-condition CONFLUENCE events (few, large
  moves) — though prior AUC≈0.5 cautions; (b) event/news-anchored drift; (c) a frank assessment that liquid-FX
  long-only under FundedNext costs may have no easily-discoverable edge (6 families across all levers say the
  realizable edge < cost) — worth flagging to the operator if the next 2–3 mechanism-changes also fail.

## Flags (code NOT merged — human-gated, per protocol §9)

1. **`A1Config.time_exit_bars` is defined but NOT wired** into the `Order` by `A1Architecture._build_a1_strategy`
   — so a config-level time exit silently does nothing. Worked around with the signal-class `make_time_exit_predicate`
   (BUILT tool, committed). If a config-level time exit is wanted, A1 should pass `cfg.time_exit_bars` to the
   `Order` (the engine already supports `Order.time_exit_bars`) via the normal human-gated PR path.

## Reproduction

Signal + driver scratch (`_disco_work/arc1005_observe.py`, `arc1005_kill.py`); time-exit predicate committed at
`discovery/tools/time_exit_predicate.py`. Data: `histdata_root=C:\Users\panap\histdata_backup`, **tf D1** 5ers_eet.
Universes: {USDCHF,USDJPY,USDCAD} and {USDCHF}. Signal: long at 3rd-to-last trading day of month, 6-bar time
exit, SL=2·ATR. Engine: `A1Architecture`+`ArcFoldRunner` (`exit_policy=None, trail_enabled=False`), FundedNext
costs at `build_fold_stats_from_run`, triage folds `build_v3_folds` ids 4/7/10.
