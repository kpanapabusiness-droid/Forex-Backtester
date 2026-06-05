# arc 3016 — Intraday session structure: Asian-range / London-open break (continue OR fade)

**Chat:** 3000s · **Date:** 2026-06-05 · **Verdict:** FAIL (cheap-kill at observation) → **KILL**
**Disposition:** KILL · **passed:** N

> The single genuinely-UNMAPPED data modality: every prior arc is H4/D1/W1 daily-scale or
> calendar-flow; intraday session/liquidity structure has never been tested as a construction (arc
> 1000 only did H4 hour-of-day CONDITIONING of capture, dry). The classic London-open break of the
> Asian range. **Falsified at observation: the break neither continues nor cleanly fades — honest
> capture ~0.38 BOTH ways (whipsaw into the 2·ATR SL), net drift ≈0 after spread, sub-cost.**

---

## Log reading (step a — FRESH EYES, honest-era only; pulled main, no STOP)

Resumed 3000s at arc 3016 (just closed my own 3015). Pulled main — three NEW arcs landed from the
other chats since my last read, and they sharpen the picture decisively:

- **arc 1018 (1000s):** weekend gap-fill on USD-NEUTRAL crosses (EURGBP/AUDNZD/…) → KILL. **Lesson:
  the gap-fill edge is JPY-cross/carry-SPECIFIC, not a universal weekend-gap property; USD-neutral
  gaps are efficient/random-walk; edge & tail are the SAME exposure → net-zero, can't diversify.**
  (10th dead route to the 2018 leg.) This **pre-kills cross-universe flow extensions** (`me`/gap-fill
  on other crosses).
- **arc 1017 (1000s):** carry-unwind cascade short → KILL (sub-cost); **new lesson: a
  correlated-cascade signal is un-scalable into a portfolio leg** (currency-exposure cap guts
  clustered same-direction fires).
- **arc 2013 (2000s):** up-gap weekend short JPY crosses → KILL (regime-luck; converges with 1016).

State after 42 arcs: directional space fully closed (long/short, all TF incl. W1, majors+crosses);
**three PORTFOLIO components** (gap-fill 1006, month-end `me` 1011, failed-breakdown-reclaim 1013); the
3-way book is **provably blocked by 2015 AND 2018**; the **2018-leg hunt is exhausted across ~10
directional/flow/structure/calendar routes**, all sharing one USD/carry/risk-reversion-fade exposure.
My own arc 3015 just closed the weekly-calendar-flow route (mechanical reversion is month-end-SPECIFIC).
The 1000s/2000s chats are **saturating the flow/short/portfolio space** — so for collision-freedom AND
information value I targeted the one region nobody has touched: **intraday session structure**.

## Idea + why (the unmapped modality, with a *because*)

The low-liquidity Asian session sets a tight range; the **London open** (the highest-liquidity FX
event) frequently breaks it. Two documented, opposite mechanisms — (a) **continuation**: the break is
a liquidity-ignition that trends through London/NY; (b) **fade**: the initial break is a stop-run /
false breakout that reverses. This is a **genuinely different exposure** (intraday liquidity dynamics)
— decorrelated from every daily-fade edge in the corpus, so a positive result would be portfolio-
relevant regardless of the 2018 wall. Untested as a construction (arc 1000's dry hour-of-day result was
capture CONDITIONING, not a session-RANGE break).

**Cost reality stated up front (arc-3008 prior):** on H1, 1R = 2·ATR ≈ 24 pips and the FundedNext
round-turn (1.5× spread + 0.5pip/fill ×2 + $5/lot) ≈ 3 pips ≈ **~0.125 R** — roughly **2× the H4
cost-in-R**. So an intraday edge must show a convincing (~2×) gross signal, not a marginal one.

**Acceptance test (pre-registered):** a continuation OR fade with honest capture > ~0.56 (clears 1:1-RR
cost) AND robust positive drift across pairs, with the 2015/2018 sign checked. Observation-first.

## Method (CALLED canonical; observation only)

`Panel.from_pairs` (H1 — generated from M1, 1.6 s/pair cached) on EURUSD/GBPUSD/USDJPY/AUDUSD.
**DST-clean** session definition via Europe/London local time (neutralizes the arc-1000 DST artifact):
Asian range = high/low over London-hours 0–7; break window = first London-hour ∈ {8,9,10,11} whose
close breaks the range; entry next H1 bar, 2·ATR SL, hold 8 bars (London+NY), drift over 6 bars. Honest
+1R-before-SL capture via `reached_1r_before_sl`; measured BOTH the continuation and the fade as their
OWN direction (the fade's honest capture ≠ 1−continuation_cap — the SLs are asymmetric). Drivers:
`_disco3_work/arc3016_observe_session_break.py` + `arc3016_observe_fade.py` (reproducible).

## What happened — FALSIFIED at observation

**The break does NOT continue.** Long-continuation (buy the Asian-high break): honest cap **0.375**,
drift_mean **−0.116**. Short-continuation (sell the Asian-low break): cap **0.396**, drift −0.039. Low
capture + negative drift = no liquidity-ignition momentum; the break tends to reverse.

**But the fade does NOT cleanly work either — the "fade edge" was an asymmetric-SL artifact.** The
naive `1−continuation_cap ≈ 0.62` is NOT the fade's honest capture. Measuring the fade as its own
direction (short the up-break / long the down-break, own 2·ATR SL):

| fade | n | honest cap | drift_mean | drift_med |
|---|---|---|---|---|
| short the up-break | 3972 | **0.3862** | +0.021 | −0.007 |
| long the down-break | 4010 | **0.3828** | −0.057 | −0.100 |

Honest fade capture is **0.38**, not 0.62. Continuation cap 0.375 + fade cap 0.386 = **0.76 < 1** — the
break entry sits at a **high-volatility London-open whipsaw point where a 2·ATR SL is hit BOTH ways**
before +1R. Net fade drift after the entry spread (ask/bid ≈ 0.08 ATR on H1) is **~+0.02 R ≈ 0**, an
order of magnitude below the ~0.125 R cost.

**No robustness, and the 2015/2018 drift-positivity is inside a losing cell.** short_up_break fade is
positive in only **1/4 pairs (EURUSD +0.19; AUDUSD/USDJPY negative)** — the single-pair noise tell
(arcs 1010/2011/3012). long_down_break fade is uniformly negative (4/4 pairs). The fade IS drift-positive
in 2015 (+0.174) and 2018 (+0.251) — but at capture **0.385 / 0.433 (both < 0.50)**: the gross drift
does NOT survive the take-the-loss 2·ATR SL (the exact "+drift inside sub-0.50 capture → collapses
SL-honest" pattern, arcs 3004/3012). §5f does not bite (capture far below 0.50, net drift ≈0).

## Diagnosis — the break entry is a whipsaw point, not a directional signal

Entering immediately after a London-open break sits on a **volatility spike**: the next bars extend
(stopping the fade) or snap back (stopping the continuation) with near-equal frequency, so ANY
directional bet with a 2·ATR SL there loses to the stop (cap ~0.38 both sides). The small residual fade
drift (~0.02 R) is the ceiling on any exit — and it is ~10× below the H1 cost. No exit/SL sweep can
manufacture an edge from ≈0 net drift, so the §5f engine sweep is not warranted (no non-coin-flip
entry). This extends the corpus's **H1 cost wall** (gotobi 1008, month-end-fix 3008, round-number-H1
1010 — all sub-cost) from discrete-flow events to **session-RANGE structure**.

## Verdict: KILL (cheap-kill at observation)

The Asian-range / London-open break neither continues (cap 0.375, drift −0.116) nor cleanly fades
(honest cap 0.386, net drift ≈0 after spread); the lone positive cell is a single-pair (EURUSD)
artifact; the 2015/2018 drift sits inside sub-0.50 capture. Sub-cost on the H1 cost wall. No
pool/engine/null/council spent (arc-3010/3012 efficiency discipline).

## Threads / lessons

1. **Intraday session structure is sub-cost — the last unmapped data modality is now mapped dead.**
   The London-open break of the Asian range is a **whipsaw entry** (honest cap ~0.38 in BOTH
   directions = a 2·ATR SL hit both ways), with net drift ≈0 after spread. The corpus's H1 cost wall
   (1008/3008/1010) extends from discrete-flow events to session-RANGE constructions: an intraday
   structural edge on liquid majors must clear ~2× the H4 cost and none does.
2. **`1 − continuation_capture` is NOT the fade's capture** (asymmetric SL) — a reusable measurement
   caution: always measure the contrarian leg in its own direction. The naive flip here suggested a
   0.62 "fade edge" that the honest fade capture (0.38) refuted.
3. **A gross-drift cell inside sub-0.50 honest capture is dead, even when 2015/2018-positive** — re-
   confirms arcs 3004/3012: the take-the-loss 2·ATR SL is the arbiter, and a positive forward drift
   that the SL eats is not a tradeable edge. (So the fade's +2015/+2018 drift is NOT a 2018-leg lead.)
4. **Surviving frontier (further narrowed):** with intraday session structure now mapped dead, the
   in-apparatus (H1/H4/D1/W1 FX-major OHLCV) search is comprehensively exhausted — direction, flow,
   calendar (monthly taken / weekly+intraday dead), structure, vol, regime, relative-value, and now
   session structure. The portfolio route stands one regime-orthogonal 2018-leg component from
   deployable, and the corpus's repeated conclusion holds: that component must come from a lever the
   in-apparatus search cannot supply (operator-gated tighter-cost execution regime — arc-3004
   escalation #3 — or a non-OHLCV data source). The honest read is that the H4/D1-FX-major idea well
   is nearly dry; remaining arcs are low-EV map-completion until the operator opens a new lever.

## Tooling

No new BUILT tool — focused scratch observers (`arc3016_observe_session_break.py`,
`arc3016_observe_fade.py`) using canonical `Panel.from_pairs` (H1) + `reached_1r_before_sl`, same
pattern as arcs 3012/3014/3015. Not promoted (one-off session conditioner).

## FLAGS (code not merged)

None. No canonical-core change. Carries standing FLAG-1 (2018-leg leads exhausted in-apparatus) + the
`A1Config.time_exit_bars`-unwired flag. Drivers reproducible from this doc.
