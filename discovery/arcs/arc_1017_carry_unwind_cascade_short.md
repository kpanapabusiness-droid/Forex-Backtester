# arc 1017 — Carry-unwind cascade SHORT on JPY crosses (vol-ignition conditioned)

**Chat:** 1000s · **Date:** 2026-06-05 · **Disposition:** KILL · **passed:** N

## Idea + why (the 2018-leg hunt, a genuinely novel flow mechanism)
The programme's deployable gate is an all-folds-positive COMBINED book. The strongest book (3-way:
gap-fill 1006 + month-end 1011 + failed-breakdown-reclaim 1013) is provably blocked by **2015 & 2018**
(strong-USD / risk-off years) — every existing component is a calm-year *fade* that bleeds when trends
run. The missing 4th leg must be **net-positive AND positive in 2015 & 2018**. Every prior attempt failed:
structural shorts (1014/2009/2011·3011), trend-short (3010), up-gap flow short (1016), relative-value
(2010), deep-continuation long (2012).

The one documented mechanism intrinsically aligned with BOTH 2015 and 2018 (both risk-off) that is NOT a
symmetric directional coin-flip and was UNMAPPED: the **carry-unwind cascade** on JPY crosses. *Because:*
leveraged carry (long high-yield risk-cross / short JPY) builds up in calm risk-on; a risk-off shock
force-unwinds it, JPY is bought back, and because it is forced deleveraging the move is **persistent and
one-directional** ("up the stairs, down the elevator"). The IGNITION is a vol-expansion big-RED bar fired
from a built-up carry position. Shorting that ignition bets on forward continuation DOWN = entering at the
*start* of the unwind (**forward-confirming**, the arc-1013 property), distinct from arc 3010 (established
downtrend → reverts). Uses the newly-enabled shorts (PR #273) non-trivially.

## Construction (BUILT `CarryUnwindCascadeShortSignal`, registered)
Universe = JPY carry crosses AUDJPY, NZDJPY, EURJPY, GBPJPY, CADJPY; H4. Fire at bar i (SHORT, enter i+1
sell open_bid) when: `close_mid > SMA100 & SMA100 rising/20` (carry built up) AND `TR/ATR ≥ vol_ignition`
(vol-expansion ignition) AND `(open_mid−close_mid)/ATR ≥ down_atr` (big-red body). ATR = Wilder(14) MID
shift1; all reads ex-ante (bar-i close known, entry next bar). Default vi=1.5, down=1.0 (best obs cell).

## What happened
**Observation (direction-aware `observe_long_capture(direction="short")`, H4 carry crosses, IS):**
- Base short: cap 0.4833, drift −0.038 (the carry up-drift headwind, as expected).
- Ignition cell (uptrend & vol≥1.5 & down≥1.0): cap **0.5000 (coin-flip)**, drift **+0.134 ATR**
  (median +0.068), 3/5 pairs positive (AUDJPY +0.44 / CADJPY +0.34 / EURJPY +0.11; GBP/NZD negative).
- **Structure control PASSES (the strongest point for the mechanism):** in-carry-uptrend median **+0.068**
  (continues DOWN) vs same big-red+vol bar NOT-in-uptrend median **−0.178** (reverts UP). The carry context
  is genuinely load-bearing — it flips a generically-reverting big-red bar into a (weakly) continuing one.
- **But two artifact tells:** capture is coin-flip 0.50 (positive drift under coin-flip capture = the
  arc-1010/2011 warning); and the **deeper cell INVERTS** (vol2.0/down1.5 → drift −0.017; down1.5 cells
  2015/2018 negative) — the opposite of arc 1013's deepens-the-edge signature, the 2011/3011 tell.
  Per-year drift is pure noise (−1.26 [2012] .. +1.21 [2011]).

**Honest engine (§5f — entry shows gross structure-controlled drift → exit menu swept before FAIL):**
Pool n=1204 (capture matches obs); gross mean final_r per year positive most years (2014 +0.86R, 2016
+0.76R, **2018 +1.04R**) but negative 2012/2013. 5-pair IS WFO, best exit **sl_only mean +0.013%, 4/10
folds positive, worst −0.148% — NOT all-folds-positive** (tp_2r/trailing/partial all mean ≤ 0).
- **The "2015 & 2018 both +0.19%" first read was a low-trade-count artifact** of the **JPY currency-exposure
  cap**: all 5 pairs share JPY quote and the carry-unwind ignitions fire *simultaneously* (one correlated
  risk-off cascade), so the 2-per-currency cap guts the clustered fires (2015/2016/2019 OOS folds → n=0–3
  on the engine, vs 92–109 in the uncapped pool). The cap structurally kills exactly the clustered trades
  the mechanism is about.
- **Single-pair (exposure-cap removed), best-version §5f + null:** AUDJPY best (sl_plus_trailing_atr) mean
  **+0.011%**, 6/10 pos, beats null by **+0.022pp**; EURJPY +0.007% / +0.010pp; NZDJPY ≈0 / +0.0001pp
  (dead). **None all-folds-positive.** 2015/2018 OOS folds positive-but-tiny and inconsistent across exits
  (EURJPY 2015 flips negative under tp_2r/trailing).
- Fair same-exit null (5-pair, 5 seeds): real mean +0.013% vs null −0.031% → beats null by **+0.044pp** —
  real but **noise-floor** edge (arc 1006 beat its null by ~+0.36pp; this is ~10× smaller).

## Verdict: KILL
Real-but-sub-cost, the arc-0/1000 signature now confirmed for the carry-unwind short: beats random by a
noise-floor margin (+0.02–0.04pp), NOT all-folds-positive under any exit/pair, capture coin-flip, drift on
the cost line and inverting in deeper cells. Mean (+0.011% best) is 60× below the PORTFOLIO archetype
(1006 +0.69%) and within noise → not a confident mean-positive edge; §11 + conservative bias (§8) forbid
building a portfolio on +0.02pp-over-null noise (cannot diversify a noise-floor component positive, arcs
3000/3001). OOS 2021+ holdout NOT touched (IS not all-folds-positive, §4).

## Threads / lessons
1. **The carry-unwind cascade short is a REAL structural effect but sub-cost.** The structure control
   passes (carry-uptrend context flips a reverting big-red bar to weakly-continuing, +0.134 vs +0.017 ATR),
   yet the magnitude sits on the JPY-cross cost line and capture is coin-flip → net noise-floor, like every
   other directional construction. Forward-confirming is necessary-not-sufficient (re-confirms 1013/2011/2012).
2. **NEW: a correlated-cascade signal is structurally un-scalable into a portfolio leg.** All-JPY-quote
   crosses fire together in risk-off, so the currency-exposure cap guts precisely the clustered cascade
   trades that carry the hoped-for edge (uncapped pool 92–109/yr → capped engine 0–3/yr in 2015/16/19). Even
   a genuinely risk-off-positive flow signal on a single-currency cross-basket cannot become the
   magnitude-meaningful 2018 leg — its book is internally redundant (one bet) and capped too thin.
3. **2018 is genuinely engine-positive here (AUDJPY +0.07–0.10%, pool +1.04R) — the only short construction
   that IS positive in 2018 on the honest engine** — but tiny, with 2015 marginal/inconsistent, and
   un-scalable (lesson 2). "Directionally right, magnitude-dead." The 3-way book's 2015/2018 deficits
   (−0.77% risk-parity 2018) dwarf a +0.07% contribution.
4. **The 2018-positive 4th leg is now unfound across EVERY route:** structure (1014/2009/2011), climax
   (2009), reject (2011·3011), trend (3010), up-gap flow (1016), relative-value (2010), deep-continuation
   long (2012), and now **carry-unwind flow short (1017)** — the last named risk-off flow idea. Reinforces
   the arc-3004 escalation: the apparatus's only net-positive edges are calm-year fades; risk-off-positive
   constructions are at the noise floor / un-scalable.

## Tooling
**BUILT + registered:** `CarryUnwindCascadeShortSignal` (`discovery/tools/carry_unwind_signals.py`) —
reusable JPY-cross carry-unwind ignition short. Reused BUILT `observe_long_capture` (direction-aware short
lens) and `build_null_signal_evaluation`; CALLED canonical `build_arc_pool`, `ArcFoldRunner`,
`run_config_over_folds`, `judge_all_folds_positive`, `build_v3_folds`. Scoring stayed canonical throughout.

## FLAGS (code not merged)
None requiring the canonical core. Observed (not a defect, a realistic constraint): the FundedNext
currency-exposure cap makes a single-quote-currency cross-basket un-scalable for a correlated-event signal
(lesson 2) — a property of the gate, correctly applied. Carries the standing `A1Config.time_exit_bars`-
unwired flag (1005/3004) + FLAG-1 (the 2018 leg the route needs is unexpressible as any tested directional
bet). Drivers scratch `_disco_work/arc1017_*.py` (reproducible from this doc).

## Reproduction
`PYTHONPATH=. py _disco_work/arc1017_engine.py` (5-pair WFO), `arc1017_singlepair.py` (per-pair §5f+null),
`arc1017_observe_carry_unwind_short.py` (observation), `arc1017_null.py` (5-pair null). Signal:
`CarryUnwindCascadeShortSignal(sma_period=100, vol_ignition=1.5, down_atr=1.0)`, H4, 5ers_eet,
`histdata_root=C:\Users\panap\histdata_backup`. SL=2·ATR, risk 0.5%, FundedNext costs ON, SL-first.
