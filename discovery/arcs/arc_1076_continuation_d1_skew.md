# arc_1076 — positive-skew trend-CONTINUATION on the HONEST ENGINE at the D1 timeframe (the untested axis)

**Chat:** 1000s · **Range:** 1000–1999 · **Date:** 2026-06-07 · **TF/universe:** **D1** (the new axis),
two universes — 7 USD majors (controlled vs arc 1074/1002) AND the full 28-pair set (trendy-cross
steelman) · **Window:** IS 2010–2020 (OOS 2021+ FROZEN, one-shot, untouched unless IS guard holds).

**Disposition: KILL** (all 8 cells KILL at IS; the one mean-positive cell dies under the pre-registered
tail-removed guard; OOS never touched). The timeframe axis is now closed too.

---

## Why this arc — the one axis the skew-lens arcs never varied

The operator's 2026-06-06 LESSONS compression named ONE open in-charter thread: **positive-skew
continuation** (the corpus called trend-following dead using ONLY +1R-capture and mean-forward-drift —
win-rate lenses structurally blind to a low-capture / fat-right-tail payoff). Arcs **1074** (Donchian
breakout-in-trend), **1075** (pullback-resume), **2081** (continuation-book + shock under trailing) and
**2082** (vol-expansion breakout) closed it under the mandated mean + median-per-fold + tail-removed lens
on the honest engine — and declared it *"airtight-closed across every entry GEOMETRY."*

**But every one of those arcs ran at H4.** Entry *geometry* was varied exhaustively; **timeframe was held
fixed at H4** — the single worst timeframe for trend-persistence-relative-to-noise. Timeframe is an axis
*orthogonal* to entry geometry, and the positive-skew premise is intrinsically timeframe-sensitive: a
trailing-runner exit on **D1** can ride a multi-week / multi-month trend into a genuinely fat right tail,
whereas an H4 "trend burst" is a few ATR before mean-reversion noise dominates. **Daily is the CTA
timeframe** — the resolution where the documented-fundable positive-skew trend edge (LESSONS run-2
practitioner lens; `NEEDS_ENABLEMENT` #1 cross-asset trend) actually lives. So the "airtight" closure
covers geometry-at-H4, not the timeframe axis.

**The closed-ground prior says directional coin-flip is "timeframe-invariant" — but that was measured
under the CAPTURE/DRIFT lens** (arc 3004), and the operator's entire redirection is that capture is
*structurally blind to positive skew*. TF-invariance-under-capture does NOT imply TF-invariance-under-
mean+tail-removed. The skew lens has never been applied across timeframes.

### Two priors point at the exact untested SYNTHESIS cell

- **arc 1002 (D1 trend-following long).** Triaged the D1 Donchian-20 long on the honest engine with a
  runner exit (`sl_partial_close_1r_runner_trail`): **mean −4.13% (3-fold triage, all negative).** But it
  was **long-only, 3-fold triage only, capture-as-primary-lens**, and **never** ran the mandated
  mean+median+**tail-removed** lens, both directions, the §5f nested exit-menu selection, or full 8-fold
  WFO. A fat right tail can hide under a negative triage-ROI; arc 1002 never looked for one.
- **arc 1003 (crosses, H4).** **Crosses TREND** — gross mean final_r **+0.1021/trade** ("crosses DO
  trend; the runner harvests it") — but the **wider cross spreads sink the net** (triage −5.49%). The
  death was *cost*, not absence-of-trend.

**The synthesis neither arc tested:** **D1 × trendy crosses.** D1 amortizes the wide cross spread over a
multi-week trend hold (one entry/exit spread per large move, not a spread paid frequently as on H4) while
*keeping* the stronger cross trend that arc 1003 documented. This is simultaneously (a) the timeframe
where positive-skew trend structurally lives and (b) the cell where arc-1003's cost-drag death is
structurally minimized. It is a genuinely-different, in-charter, buildable-now construction on an axis the
skew-lens arcs never touched — not a re-derivation of closed H4 geometry.

### Falsifiable prediction

If positive-skew trend lives anywhere in-charter, it is on **D1 crosses**: mean per-trade R > 0 **and** a
fat right tail that SURVIVES tail-removal (G1 + G2 both pass), with a runner exit selected at a *looser*
SL (letting the multi-week move breathe). **Falsifier:** if D1 crosses are STILL mean-negative with a thin
tail (the same shape as H4), positive-skew continuation is closed on the **timeframe axis too**, not just
entry geometry — extending the airtight closure to TF and confirming edge-absence (the conservation law on
the continuation axis) rather than TF-/geometry-specificity.

---

## ⚠️ MANDATORY PRE-REGISTERED KILL-RULE (written BEFORE the run) — tail-luck ≠ skew

A positive-skew result is real ONLY if it survives removal of its biggest winners. **If the edge is
mean-positive ONLY because of the top-K winners — i.e. it goes flat or negative under the tail-removed /
winsorized expectancy — it is KILL.** No relabeling a handful of outlier trades (a lucky 2020-2021 trend
run) as "positive skew." Concretely, a cell is KILL unless ALL hold together:
- **G1** mean per-fold ROI > 0 **AND** mean per-trade R > 0 (the skew edge must exist at all);
- **G2 (decisive honesty check)** under **+2R cap**, **drop-top-5%/fold**, AND **drop-top-K{1,3} global**,
  the mean stays **> 0** (the edge is broad-based, not 2–3 monster trades carrying everything — cf. the
  thin-tail traps arcs 2011/2063 caught);
- **G3** per-fold median ROI > 0 in a majority of folds (breadth, not tail-only);
- plus **beats a fair same-side random-entry null** (§11: beats-null-but-net-negative = KILL).

The OOS holdout is touched ONLY for a cell that passes **G1+G2+G3** on IS. Exit menu is selected §5f-nested
(select on IS fold → score that fold's OOS → freeze); never full-sample best-pick.

---

## Method

Reuse `discovery/tools/trend_continuation_signal.py::TrendContinuationBreakoutSignal` (fresh Donchian-N
break inside a dual-SMA trend, long AND short, mid-OHLC shift1, crossing-bar-only) with `primary_tf="D1"`.
Scored SOLELY by the honest `ArcFoldRunner → A1Architecture → MultiPairBacktester` (FundedNext costs ON,
risk 0.005), under the positive-skew **RUNNER exits ONLY** — `sl_plus_trailing_atr`,
`sl_plus_trailing_swing`, `sl_partial_close_1r_runner_trail` × SL{1.5, 2.0, 2.5}, §5f nested walk-forward
selection (NO `tp_2r/3r`, which cap the tail and defeat the premise). Cells: long & short × Donchian{20,55}
× SMA50/200 (Turtle-canonical daily params), per universe. Fair same-side random-entry null (warmup 210 D1
bars ≈ SMA200). IS folds = `build_v3_folds`, oos_year ≥ 2011 (8 evaluable). OOS NEVER touched on IS-KILL.

The experiment side computes the entry MASK + ATR geometry + tail-removal/null ARITHMETIC only; it NEVER
realizes P&L for a gate. Anti-Arc-10: canonical core scores every trade, take-the-loss / SL-first.

Driver `discovery/_disco1_work/arc1076_continuation_d1_skew.py`.

---

## Result — KILL in all 8 cells; OOS preserved frozen

8 cells = {long, short} × Donchian{20, 55} × {USD7, ALL28}, each with §5f-nested runner-exit selection
over the 3×3 menu, 8 evaluable folds (2013–2020; 2011/2012 are §5f warmup). Pooled trades 100–412/cell.

**7 of 8 cells fail G1 (mean-negative).** The lone G1-pass dies on the tail-removed guard:

| Universe · cell | mean ROI/yr | mean R | median R | folds+ | +2R-cap | drop-top5%/fold | vs null | guard |
|---|---|---|---|---|---|---|---|---|
| USD7 · long Donch20 | −2.81% | −0.44 | −0.96 | 1/8 | −2.81% | −3.04% | beats | KILL |
| USD7 · short Donch20 | −3.13% | −0.50 | −0.98 | 0/8 | −3.30% | −3.52% | LOSES | KILL |
| **USD7 · long Donch55 (Turtle)** | **+0.16%** | **+0.025** | −0.06 | 4/8 | **−0.11%** | **−0.95%** | beats | **KILL (G2)** |
| USD7 · short Donch55 | −0.07% | −0.009 | −0.98 | 4/8 | −1.01% | −1.64% | beats | KILL |
| ALL28 · long Donch20 | −3.66% | −0.17 | −0.93 | 2/8 | −4.69% | −6.90% | LOSES | KILL |
| ALL28 · short Donch20 | −4.29% | −0.17 | −0.92 | 1/8 | −6.31% | −8.27% | LOSES | KILL |
| ALL28 · long Donch55 (Turtle) | −0.88% | −0.043 | −0.84 | 2/8 | −3.56% | −5.03% | beats | KILL |
| ALL28 · short Donch55 | −2.21% | −0.124 | −0.94 | 3/8 | −2.52% | −4.64% | LOSES | KILL |

**The decisive cell is USD7 Turtle-long — the textbook tail-luck KILL the guard was written to catch.**
It is the only cell that is mean-positive (+0.16%/yr, mean R +0.025, 4/8 folds+, beats null +2.81pp) — and
it **collapses the instant you remove the tail**: +2R-cap mean → **−0.11%**, drop-top-5%/fold → **−0.95%**,
drop-top-K{1} global → −0.12%. Its largest winner is **+4.39R**; cap it at +2R and the cell is negative.
This is *exactly* the pre-registered "mean-positive ONLY because of the top-K winners → KILL" scenario — a
handful of outlier trades carrying a flat-to-negative book, not a broad right tail. The guard fired verbatim.

**No broad skew anywhere; the median trade still takes the −1R stop at D1.** Median per-trade R is ≈ **−0.84
to −0.98** in every cell — the same adverse-first / near-martingale signature as the H4 arcs (1074/1075). The
largest single winners (USD7 +5.64R short, ALL28 +7.89R Turtle-long) are bigger than H4's, and on ALL28 a
single +7.89R trade is what produced the lone +10.09% 2015 fold — but tail-removal erases it (Turtle-long
ALL28 → −3.56% capped). **2018 is negative in all 8 cells** (−1.0% to −9.3%); 2015 is mixed (a long-vol
year that helps the longs, hurts the shorts) — the classic long-vol/short-vol regime split, but not enough
to lift any cell past the guard.

## The hypothesis is REFUTED — and that closes the timeframe axis

The "D1 amortizes the cross spread over a multi-week trend hold while keeping the stronger cross trend" thesis
**did not survive contact with the engine.** Even at D1 — the CTA timeframe, on the trendiest 28-pair
universe, with §5f-selected runner exits — the continuation entry's median trade still takes the −1R stop
(median R ≈ −0.9) and the only mean-positive cell is tail-carried, not broad. **The falsifier was hit:** D1
crosses are still mean-negative-or-tail-carried with the same median-R structure as H4. So the failure is NOT
H4-timeframe-specificity and NOT cross-spread-cost (arc 1003's death) — it is the same root cause arc 1075
named: **liquid FX price has no exploitable post-trend continuation**, and that is **timeframe-invariant under
the positive-skew lens** just as it was under the capture lens (arc 1002's D1 capture ≈ 0.49 → now confirmed
on the *mean+tail-removed* lens too). A trend-structural entry samples the near-martingale at *any* sampling
interval; take-the-loss + a runner cannot manufacture broad skew from a martingale, and the discrete tail that
does form is too thin to survive the honesty check (conservation law, LESSONS run-2).

**Frontier status.** Positive-skew continuation was closed across every entry GEOMETRY at H4 (arcs
1074/1075/2081/2082); this arc closes the orthogonal **TIMEFRAME axis** (D1) **and** the **universe axis**
(trendy crosses) under the same mandated mean + median-per-fold + tail-removed lens on the honest engine. The
closure now spans geometry × timeframe × universe — the operator's ONE open in-charter thread (LESSONS
2026-06-06) is comprehensively closed on all three axes by direct engine measurement. In-charter frontier
returns to the convergent terminus: 0 PASS, 4 PORTFOLIO, deployable = 0; the documented-fundable version of
this exact positive-skew shape lives only on a **less-efficient universe** (`NEEDS_ENABLEMENT.md` #1
cross-asset trend — operator-gated, out-of-band). Components UNCHANGED; no canonical change; no FLAG; no
council (decisive negative, not a survivor/fork).

**Reusable.** `TrendContinuationBreakoutSignal` now exercised at D1 (`primary_tf` param confirmed
TF-agnostic); driver `arc1076_continuation_d1_skew.py` (parameterized over TF × universe); and the lesson:
the continuation closure is **timeframe-invariant under the skew lens**, so future "try a slower/faster TF"
ideas for trend-continuation on liquid FX resolve to this closed ground.

