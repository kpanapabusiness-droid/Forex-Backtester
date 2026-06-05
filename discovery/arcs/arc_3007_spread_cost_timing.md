# Arc 3007 — Intraday-spread cost-timing lever on EDGE<COST

> **Arc id:** 3007 · **Chat:** 3000–3999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **FAIL → KILL.** The one untried cost lever on the corpus-wide EDGE<COST
> constraint — restricting a positive-gross-drift signal to the tightest-spread intraday window to cut
> realized cost — **backfires.** Entry spread and gross edge are entangled (both liquidity-driven): the
> tight-spread / liquid hours, where cost is lowest, have the **weakest** (negative) gross edge; the lone
> positive-gross bucket is mid-spread, and it nets **≤ 0** on the SL-honest engine (mid-spread-band
> cross-trend triage mean −0.02%, not all-folds-positive, on momentum-friendly years). You cannot cut
> cost without cutting edge.
> **Idea source:** arcs 1003/1004 sharpened the binding constraint to **EDGE<COST** and named the fix as
> "raise gross edge OR cut per-trade cost"; every cost lever tried was frequency-via-exits (1004).
> Intraday **spread timing** is the one untried cost lever. Closing it completes the EDGE<COST coverage.

Observation via canonical `observe_long_capture` (honest take-the-loss capture + gross drift) +
the BUILT `DonchianBreakoutLongSignal`; confirmation via the SL-honest engine (`ArcFoldRunner` →
`MultiPairBacktester`, FundedNext costs). Real bid/ask, H4 5ers_eet, IS 2010–2020, 12 trending crosses.

## (a) Log read + synthesis

Pulled `origin/main`. 20 arcs read (0, 1000–1009, 2000–2004, 3000–3006). STOP absent; highest 3000s id 3006
→ resume 3007. State: shallow directional prediction closed under both metrics, all instruments/TF/regimes/
exits/stop-geometry; volume, calendar, convexity, triangulation, and (arc 3006) multi-TF structure all closed.
The binding constraint, sharpened by arcs 1003/1004, is **EDGE<COST** — on crosses the gross drift is
*positive* (+0.10R, arc 1003) but the wider spreads eat it; the fix must "raise gross edge or **cut per-trade
cost**." Arc 1004 cut cost via exit/frequency (didn't flip it). The one untried cost lever is **intraday
spread timing**: FX spreads vary 3–5× across the day, and the honest engine charges 1.5× the *actual bar
spread* — so trading a +gross signal only in the tightest-spread window directly cuts realized cost. Tested on
arc 1003's cross-trend signal (the textbook +gross-but-net-negative case). The arc-1006 gap-fill (the one
PORTFOLIO edge) remains the only live long-only component; no 2nd has been found.

## (b) Idea + because (observe before believing)

**Hypothesis.** Restricting the cross-trend long to the tightest-spread intraday window cuts the realized
spread cost enough to flip its +0.10R-gross-but-net-negative profile net-positive — attacking EDGE<COST from
the cost side, the only lever arcs 1003/1004 named but did not try. **Because:** intraday FX spread is large
and predictable (tight in the London/NY overlap, wide in the Asian/rollover window); the engine's 1.5×-spread
charge scales directly with it.

## (c) Observation → the lever backfires

**Intraday spread profile (median spread / 2·ATR, pooled crosses, IS):** ranges 0.0256 (tight, ~London hours)
to 0.0764 (wide, the thin-liquidity bar) — a **2.99× range.** The lever has real cost range to exploit.

**Cross-trend pool (Donchian-20, SMA200-uptrend, 12 crosses): n=4539, honest capture 0.4807, gross drift
+0.0218R** (matches arc 1003/3001 — crosses trend gross-positive but tiny). Split by entry-bar spread tercile:

| tercile | n | spread_R | honest capture | gross drift | net-proxy* |
|---|---|---|---|---|---|
| **tight** | 1513 | 0.0138 | 0.4878 | **−0.0199R** | −0.0531 |
| **mid** | 1513 | 0.0328 | 0.4977 | **+0.1207R** | +0.0590 |
| **wide** | 1513 | 0.0858 | 0.4567 | −0.0354R | −0.1768 |

*net-proxy = gross − 1.5·spread_R − slippage_R; commission omitted ⇒ **optimistic.**

Tightest-6-UTC-hour window {2,6,9,10,13,14}: n=2769, gross **−0.0158R** (negative), net-proxy −0.0908R.

**The lever backfires.** The cost-saving direction — tight spread, the liquid London/NY hours — has the
**weakest, NEGATIVE** gross edge (−0.020R), because tight spread = high market efficiency = breakouts most
arbitraged. The positive gross drift lives in the **mid**-spread bucket (+0.121R), where cost is *not*
minimized. Spread and edge are entangled (both liquidity-driven), so you cannot separately optimize them:
timing to low spread = timing to low edge.

## (d) Confirm — the lone positive bucket dies SL-honest (no hand-waving the +0.12R)

The mid-spread +0.121R gross is the most positive cell the 3000s range has produced — so, per the Arc-10
discipline (never trust/dismiss a number without the engine), it was scored on the SL-honest engine. Built a
mid-spread-band filter (entry spread_R ∈ [0.020, 0.050]) over the cross-trend signal (2,955 fires), 3-fold
honest triage (2013/2016/2019, A1, `sl_partial_close_1r_runner_trail`, SL=2·ATR, FundedNext costs):

| fold (OOS yr) | ROI | DD | n |
|---|---|---|---|
| 2013 | +0.08% | 0.06% | 165 |
| 2016 | +0.01% | 0.03% | 95 |
| 2019 | −0.13% | 0.14% | 217 |

**worst −0.13%, mean −0.02%, 2/3 positive — net-negative, not all-folds-positive,** on the momentum-*friendly*
triage years (arc 3002: those years inflate, so full-IS is ≤ this). The +0.121R gross collapses to ~break-even-
to-negative under take-the-loss + cost — the arc-3003 signature (positive gross drift → ≤0 SL-honest
expectancy). The mid-spread bucket is a non-monotone post-hoc artifact, not a real edge.

## Final verdict — FAIL → KILL

The intraday-spread cost-timing lever does not flip the cross-trend net-positive. (1) The cost-saving
direction (tight spread) has *negative* gross edge — the lever backfires; (2) the only positive-gross bucket
(mid-spread) nets ≤0 SL-honestly. **KILL** disposition (net-negative — not a PORTFOLIO component; §11). This
closes the last untried lever (cost-side) on the EDGE<COST constraint.

## Lessons (candidate for LESSONS.md compression)

1. **The cost-timing lever on EDGE<COST is closed — spread and edge are entangled.** Intraday spread varies ~3×
   on crosses, but the tight-spread/liquid hours (cheapest to trade) carry the *weakest* gross edge (−0.02R) —
   high efficiency arbitrages the breakout. You cannot cut per-trade cost by spread-timing without cutting the
   gross edge by the same liquidity mechanism. The one positive-gross bucket (mid-spread) is non-monotone and
   dies SL-honest (triage mean −0.02%). Arcs 1003/1004's "cut per-trade cost" steer is now tested and dead.
2. **Gross drift is non-monotone in spread, which is the tell it isn't a lever.** A monotone cost/edge relation
   would be exploitable; a single bulging middle bucket is regime/selection noise — confirmed by the engine
   collapse. Re-applies the arc-3002/3003 discipline (a positive cell must clear the SL-honest engine).
3. **EDGE<COST is closed from BOTH sides now** — edge-side (raise gross: every entry/TF/universe/regime, arcs
   0–3006) and cost-side (cut cost: exits/frequency arc 1004, spread-timing arc 3007). Cements the arc-3004
   escalation: the apparatus cannot express a deployable edge; a structural unlock (shorts / second leg /
   genuinely tighter-cost broker regime) is required = the operator's decision.

## Threads / what didn't help

- **Closed:** intraday-spread cost-timing as an EDGE<COST lever (backfires; mid-spread bucket dies SL-honest).
- **Did NOT re-run the exit sweep:** the cross-trend entry's exit-as-hyperparameter sweep is already done
  (arc 1004, no exit flips it net-positive); arc 3007 tests a NEW lever (cost-timing) on the same entry, and
  kills the *lever*, not the entry. The §5f sweep is not re-triggered.
- **Operative state unchanged:** no 2nd net-positive long-only component (still only the arc-1006 gap-fill); the
  arc-3004 escalation stands, now reinforced from the cost side. Next high-leverage move = the operator's
  structural unlock (`ESCALATION_apparatus_capability.md`, 2001/2003 FLAG-1).

## Flags (code NOT merged)

None. No canonical-core change; no new BUILT tool (the mid-spread filter is a one-off conditioning wrapper kept
in scratch). Reused BUILT `observe_long_capture` + `DonchianBreakoutLongSignal`.

## Reproduction

`Panel.from_pairs([AUDJPY EURJPY GBPJPY NZDJPY CADJPY CHFJPY EURAUD EURNZD EURCAD GBPAUD GBPNZD GBPCAD], "H4",
histdata_root=C:\Users\panap\histdata_backup, cache_root=data/cache, boundary_convention="5ers_eet")`. Spread_R
= (close_ask−close_bid)/(2·ATR). Cross-trend = `DonchianBreakoutLongSignal(lookback=20, sma_filter=200,
spacing_bars=6)`; observe via `observe_long_capture(..., restrict=signal_mask)`; mid-spread band [0.020,0.050]
triage via `ArcFoldRunner` over the 2013/2016/2019 `build_v3_folds` folds, `A1Config(exit_policy=
"sl_partial_close_1r_runner_trail", sl_atr_mult=2.0, risk_pct=0.005)`. Drivers:
`_disco3_work/arc3007_observe.py`, `_disco3_work/arc3007_triage.py`.
