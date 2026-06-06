# arc_2061 — US-bank-holiday thin-session reversion (scheduled-thinness gap-fill)

**Chat:** 2000s | **Date:** 2026-06-06 | **Verdict:** KILL (obs cheap-kill) | **Disposition:** KILL

## (a) Log read + synthesis (FRESH EYES, honest-era only)
Resumed at the highest 2000-range arc (2060) + 1. No `discovery/STOP`. Read protocol §0-§11,
TOOL_REGISTRY (CALL vs BUILT), LESSONS + DISCOVERY_DIRECTION, full Tier-1 ledger (0→2060) + recent
Tier-2.

State of the corpus: 4 PORTFOLIO components (`gap` 1006 weekend down-gap-fill JPY-cross H4, `me_long`
1011 month-end reversion long USD-major D1, `fbr` 1013 failed-breakdown-reclaim long USD-major H4,
`me_short` 1019 month-end reversion short USD-major D1). The 4-way book is mean-positive but never
all-folds-positive (2015/2018 strong-USD wall, combination-invariant; path-B densification
quantitatively closed, 3021). The §5f exit-honesty audit (2040-2046/1042-1046) collapsed the honest
deploy object 4→2-way {me_long+fbr}→**me_long SOLO** (gap+me_short flip mean-neg under honest nested
exits; fbr's IS diversification dies OOS); 2059/2060 found {me_long,fbr} is the OOS *vehicle* optimum
(Calmar hump peaks at 2 legs) yet vehicle-infeasible at every leg-count. §11 verification complete
end-to-end (signal 2034/35, outcome 2036-39/2046, cost 2039). The explore-now MENU is exhausted
(M1 1027/2023, O1 1029/1030/1055, L1 1031/1054/2028, Q1 1028, G1 2052, S1 modifier). Lever = operator
path-A gate-governance call.

**Unified survivor theory (the *because* filter):** a capturable FX edge needs an identifiable
institutional forced flow + a LARGE/surprise displacement + information-free + not coincident with a
directional-flow window. Calendar-flow failure modes so far catalogued: (i) **sub-cost** (displacement
too small — fix flows 2025/3008), (ii) **priced-in** (large but anticipated → front-run, fiscal-YE
2026), (iii) **instrument-neutral** (real in a derivative but cash-neutral in spot — IMM roll 2057),
(iv) **fat-tail mirage** (gross drift clears cost but R≈0 under take-the-loss, OOS-neg — quarter-end
2058).

## (b) Idea (log-dry; the one inelasticity proxy O1 never tested)
The surviving FLOW edge — the weekend gap-fill (`gap`, 1006) — works because a **closed-market thinness**
lets a displacement overshoot, then revert when liquidity returns. The O1 *inelasticity-state* thesis
(thin book → bigger move → bigger reversion) was killed via the **spread-z** proxy (1030/1055), where it
**INVERTED** — but the documented reason was that a wide spread marks **info-STRESS** (a genuine
breakdown → continues). A **scheduled US bank holiday** is a *different* inelasticity proxy: thinness
that is **calendar-known and NOT info-stress-driven** → it should escape the 1055 inversion and behave
like the weekend gap (overshoot → revert). Genuinely untested (no holiday arc in the corpus; distinct
from 1005 turn-of-month / 3015 end-of-week / 2026 fiscal-YE / 2057 IMM / 1048 NFP). Has a working
precedent (`gap` 1006), is calendar-derivable from the data, and was checked specifically for a 2015 &
2018 sign. Applied the survivors' **magnitude gate** (condition on a ≥1-ATR holiday move) for a fair
above-cost shot.

## (c)/(d) Characterize + obs cheap-kill
Canonical D1 cache (7 USD majors, 2010-2026), Wilder(14) MID ATR shift1 (ex-ante), session date =
ts+1day (5ers_eet 22:00-UTC open convention, arc-1048). Holidays = `pandas` `USFederalHolidayCalendar`
(US bank holidays) ∪ `GoodFriday` (FX-thin), matched on session date → **1179 holiday-session bars
(3.6%)**. Lenses: holiday-session intraday move `(close-open)/ATR`, fade bet `−sign(move)·fwd_k` over
k∈{1,2,3} D1 bars, `corr(move, fwd)`, vs a non-holiday control; per-year & per-pair on the magnitude-
gated subset. Cost reference: D1 RT ~0.085 ATR.

**FALSIFIED — and the thesis INVERTS on its core premise:**
- **[1] Holidays do NOT amplify displacement — they MUTE it:** holiday |move| mean **0.323 ATR** vs
  non-holiday **0.500** (median 0.236 vs 0.380). 24h-global FX stays liquid on a US holiday (London +
  Tokyo carry on); only US participation thins → the move is SMALLER, not bigger. The premise (thin book
  → larger displacement) is **false for FX**.
- **[2] No reversion — faint continuation:** `corr(holiday move, fwd_k)` = **+0.042 / +0.002 / +0.013**
  (k=1/2/3), all ≈0-to-positive; the non-holiday control is faintly NEGATIVE (−0.008/−0.007/−0.006, the
  generic tiny mean-reversion). The holiday *removes* the faint reversion rather than adding it.
- **[3] Magnitude gate kills it on thinness + sign:** |move|≥1.0 leaves only **n=51 holiday bars** over
  16yr × 7 pairs (~3/yr = untradeable), and the fade drift is **NEGATIVE** (fade2 −0.138 / fade3 −0.227,
  frac+ 0.471) = the rare large holiday move **CONTINUES**, worse than the control (≈flat −0.016).
- **[4] Not the 2015/2018 leg:** per-year noise on n=1-7; **2015 fade3 −1.40, 2018 −0.12 (both neg)**.
- **[5] Pair-mix (USD-quote-beta tell):** 5/7 pairs neg (USDCAD +0.74 / GBPUSD +0.39 lone positives) =
  the arc-2009/3012/2011 confound, not a coherent holiday effect.

Coin-flip-to-continuation + thin + pair-mix + premise-false → §5f does not bite (no above-cost,
above-0.50 cell); §5d cheap-kill. No engine/null/council spent. OOS untouched.

## (e) Diagnosis — the *because* it fails (NEW lesson)
The weekend gap-fill (`gap` 1006) works because the market is genuinely **CLOSED** for ~48h: no price
discovery occurs, so a true repricing gap accumulates over the closure and reverts when trading resumes.
A US bank holiday leaves FX **fully open globally** — only one center's participation is removed → the
"thinness" is mild, displacement is MUTED, and what little move happens is ordinary global price
discovery that **continues** (corr +0.01). **Scheduled single-center thinness ≠ closed-market thinness.**

This is a **FIFTH failure mode** of documented calendar/structural flows (after sub-cost / priced-in /
instrument-neutral / fat-tail-mirage): **(v) thinness-too-mild** — the inelasticity the gap-fill needs
requires an actual market CLOSURE window (weekend), not merely reduced participation in one session. It
ALSO cleanly distinguishes from the O1/1055 spread-z inversion: the holiday lane does not fail because
holiday thinness is info-stress (the 1055 reason) — it fails because holiday thinness in 24h FX **barely
exists** (|move| is smaller, not larger), so there is no overshoot to fade in the first place.

**Sharpens the survivor DNA:** the only tradeable thinness in 24h FX is a genuine market-CLOSURE window
(the weekend, `gap`). Reduced-participation thinness (single-center holiday) neither amplifies nor reverts
— it is not an inelasticity-state lever, closing the scheduled-thinness sub-route of O1 with a cleaner
mechanism than the spread-z proxy did.

## Outcome
- Closes the US-holiday / scheduled-thinness gap-fill lane (a new entry in the "documented
  calendar/structural condition → not capturable in 24h spot FX" series: gotobi 1008/2025, fiscal-YE
  2026, daily-fix 1051/3008, NFP 1048, IMM 2057).
- Components UNCHANGED (all 4 PORTFOLIO). Deploy object UNCHANGED (me_long-solo mean / {me_long,fbr}
  vehicle, 1046/2059/2060). Lever = operator path-A. No new BUILT tool (canonical loader + inline obs +
  a stock `pandas` holiday calendar; scratch removed). No canonical change, no FLAG, no council, OOS
  untouched.

**Repro:** D1 5ers_eet, 7 USD majors, `USFederalHolidayCalendar ∪ GoodFriday` on session=ts+1day;
holiday-session `(close-open)/ATR` vs non-holiday control + `−sign(move)·fwd_k` fade, magnitude gate
|move|≥{0.5,1.0,1.5}. (Inline obs, deleted post-run; numbers above.)
