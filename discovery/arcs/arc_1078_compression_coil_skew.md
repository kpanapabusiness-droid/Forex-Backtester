# arc 1078 — positive-skew COMPRESSION-COIL breakout on the HONEST ENGINE (H4) — the last untested entry geometry

**Chat:** 1000s · **Range:** 1000–1999 · **Disposition: KILL** (all 8 cells KILL at IS; no positive
skew, no right tail, the small-stop premise refuted; OOS never touched).

Driver `discovery/_disco1_work/arc1078_compression_coil_skew.py`; new BUILT tool
`discovery/tools/compression_coil_breakout_signals.py` (`CompressionCoilBreakoutSignal`). Scored solely
by `ArcFoldRunner → A1Architecture → MultiPairBacktester`, FundedNext costs ON, take-the-loss; experiment
side = entry signal + tail-removal/null arithmetic only.

## Log reading (step a — what's been tried, what's open)

Pulled main (up to date). Read `DISCOVERY_PROTOCOL.md` v1.1, the `LESSONS.md` 2026-06-06 operator
compression (Closed-ground prior + run-2 conservation law + the ONE open in-charter thread),
`DISCOVERY_DIRECTION.md` (runs 1 & 2), `TOOL_REGISTRY.md` (canonical + BUILT), recent Tier-2 tail (arcs
1072–1077, 2084/2085). No `discovery/STOP`. Firsthand: `passed/` EMPTY (deployable=0),
`portfolio-candidates/` = 4 (gap 1006 / me_long 1011 / fbr 1013 / me_short 1019). Cross-range termini:
1000s → 1077, 2000s → 2085, 3000s → 3022.

The operator's 2026-06-06 redirection opened ONE in-charter thread: **positive-skew CONTINUATION** — the
corpus called trend-following dead using only +1R-capture and mean-forward-drift (win-rate lenses blind to
a low-capture / fat-right-tail payoff). It must be judged on **MEAN + median-per-fold + TAIL-REMOVED** with
**take-the-loss + a trailing/runner exit**, with the mandatory pre-registered guard **tail-luck ≠ skew**
(mean-positive ONLY via top-K winners ⇒ KILL).

**State of that thread at step (a) — nearly airtight, one geometry left:**
- **Entry GEOMETRY (all H4) under the mandated lens:** Donchian breakout-in-trend (1074) KILL;
  pullback-resume / favorable-first (1075, 2083) KILL; vol-EXPANSION breakout (2082) KILL; shock +
  continuation-book (2081) KILL.
- **TIMEFRAME:** D1 (1076, both universes/dirs/§5f) KILL; W1 (1077) KILL → the −1R median stop-wall is
  timeframe-invariant (H4/D1/W1).
- **The unifying finding:** the median continuation trade is ≈ −0.9R at every TF; the mean improves as the
  clock slows but asymptotes from below, never crossing zero. Liquid-FX price has no exploitable post-trend
  continuation (conservation law).

**The specific gap I found.** One entry geometry was NEVER run on the honest engine under the mandated lens:
the **compression-COIL breakout** (break out of a *tight* coil). It was killed only on the BLIND lens —
**arc 1001** (long-only, capture 0.4880 ≈ unconditional + a single-exit 3-fold triage, *pre-redirect*) and
**arc 1062** (both directions, gross capture 0.4535 + gross drift −0.069, explicitly *"no engine / no
null"*, framed as a 5th-leg regime-orthogonality test, not a positive-skew mean/tail test). This is exactly
the **arc-1064 → 1074 situation**: arc 1064 gross-drift-killed *continuation*, the operator said "that lens
is blind to skew, re-run it on the engine under trailing," and arc 1074 did. By identical logic the
compression-coil's gross/capture kill (1001/1062) earns an honest-engine re-run under the runner exits +
the mandated lens.

## Idea + because (why the compression-coil is the strongest residual skew candidate)

A tight coil → small recent range → **small ATR** → the engine's −1R stop is a small ABSOLUTE move sitting
just back inside the coil. If the post-break expansion runs, the trailing runner harvests a **large
R-multiple** (move / small-stop-distance) = the bounded-loss / unbounded-win geometry the positive-skew
premise needs. This is the textbook "volatility-contraction → expansion" (squeeze) setup. It is the
mechanistic **opposite** of arc-2082's vol-EXPANSION entry (which fired on an already-LARGE bar → 2·ATR
stop is WIDE → small R-multiples → died): the compression entry is the untested mirror where the skew
geometry is *a priori strongest*.

**Falsifiable prediction.** If the small-stop premise is real, the compression-coil break should show
**mean per-trade R toward 0/positive and a genuine right tail of large R winners** (median can stay ≈ −1R
— that is allowed for a skew edge), surviving the tail-removed guard. **Falsifier:** if the breaks are
adverse-first false-breaks (like 1001's diagnosis), the median takes the −1R stop, no right tail develops,
and tail-removal worsens the (negative) mean — the small-stop is irrelevant because nothing runs.

## Method

New BUILT `CompressionCoilBreakoutSignal` (mask + ATR geometry ONLY; ex-ante: ATR(14) shift1, Donchian
hi/lo over prior N shift1, coil width = (hi−lo)/ATR ≤ `coil_thresh` measured strictly before the
entry-decision bar; crossing-bar + spacing refractory; entry fills next-bar open in the engine). Long fires
on up-breaks of a tight coil, short on down-breaks. Same engine machinery as arcs 1074/1076: honest
`ArcFoldRunner → A1Architecture → MultiPairBacktester` (FundedNext, risk 0.005, linear regime), positive-skew
RUNNER exits only (`sl_plus_trailing_atr` / `sl_plus_trailing_swing` / `sl_partial_close_1r_runner_trail` ×
SL{1.5, 2.0, 2.5}, §5f nested walk-forward exit/SL selection — no tp_2r/3r, which cap the tail and defeat
the premise). Judged by the pre-registered G1/G2/G3 guard + a fair same-side random-entry null.

**Cells:** {long, short} × {coil12 ≤ 3.5·ATR (the EXACT arc-1062 config, now on the engine), coil20 ≤
5.0·ATR (wider window, proportionally tight)} × {USD7 majors, full 28-pair}. TF = **H4** (the structural TF
where the small-stop premise is most natural and where 1001/1062/2082 ran). IS 2010–2020, 8 evaluable folds
(2013–2020); OOS NEVER touched.

**Pre-registered KILL-rule (written before the run, applied verbatim):** a cell is a real positive-skew
edge ONLY if **G1** (mean per-fold ROI > 0 AND mean per-trade R > 0) AND **G2** (tail-removed still positive:
+2R-cap mean > 0 AND drop-top-5%/fold mean > 0 AND drop-top-K{1,3} global > 0) AND **G3** (per-fold median
ROI > 0 in a majority of folds) AND it beats the same-side null. Mean-positive-ONLY-via-top-K ⇒ KILL;
beats-null-but-net-negative ⇒ KILL (§11).

## Result — KILL in all 8 cells, decisively (no positive skew, no right tail)

| universe | cell | fires | n_trades | mean ROI/yr | mean R | **median R** | folds+ | +2R-cap | drop-5%/fold | largest +R | null |
|---|---|---|---|---|---|---|---|---|---|---|---|
| USD7 | TIGHT long coil12≤3.5 | 4893 | 303 | −8.26% | −0.44 | **−0.88** | 0/8 | −8.37% | −9.06% | +2.67 | LOSES −1.29pp |
| USD7 | TIGHT short coil12≤3.5 | 4711 | 486 | −3.87% | −0.13 | **−0.66** | 1/8 | −4.86% | −7.41% | +3.78 | LOSES −0.13pp |
| USD7 | WIDER long coil20≤5.0 | 3969 | 150 | −5.46% | −0.58 | **−0.63** | 0/8 | −5.46% | −5.44% | **+0.34** | beats +2.68pp |
| USD7 | WIDER short coil20≤5.0 | 3898 | 315 | −9.70% | −0.49 | **−0.93** | 0/8 | −9.92% | −10.73% | +3.16 | LOSES −5.50pp |
| ALL28 | TIGHT long coil12≤3.5 | 20015 | 633 | −21.17% | −0.54 | **−0.68** | 0/8 | −21.17% | −21.11% | +0.33 | LOSES −2.77pp |
| ALL28 | TIGHT short coil12≤3.5 | 19165 | 1837 | −18.06% | −0.16 | **−0.79** | 0/8 | −22.62% | −31.65% | +7.49 | LOSES −9.22pp |
| ALL28 | WIDER long coil20≤5.0 | 16112 | 642 | −21.32% | −0.53 | **−0.64** | 0/8 | −21.32% | −21.23% | +0.33 | LOSES −3.00pp |
| ALL28 | WIDER short coil20≤5.0 | 15718 | 1298 | −17.61% | −0.22 | **−0.84** | 0/8 | −25.17% | −30.46% | +25.16 | LOSES −6.08pp |

Every cell **G1 FAIL / G2 FAIL / G3 FAIL**; 0/8 AFP; the single beats-null cell (USD7 WIDER-long) is
**net −5.46%/yr** ⇒ beats-null-but-net-negative = KILL (§11).

**Three decisive findings:**

1. **The small-stop premise is REFUTED — there is no right tail.** The whole thesis was "tight coil → small
   stop → large R-multiple winners." It fails outright: on the long cells the **largest single positive R is
   +0.33–0.34R** (coil20), i.e. the trailing runner essentially never gets a winner to run; even the best
   long cell tops out at +2.67R. The breaks are **adverse-first false-breaks** (arc 1001's diagnosis,
   confirmed on the engine under the mandated lens): the median trade is −0.63 to −0.93R (takes the −1R
   stop), the same near-martingale signature as every other continuation geometry. Tail-removal **worsens**
   most cells (+2R-cap and drop-5% push the already-negative mean further down) — the opposite of a skew edge.

2. **The one big winner is single-outlier tail-luck, not broad skew.** The only fat right-tail value anywhere
   is ALL28 WIDER-short's **+25.16R** (a 2015 SNB-class CHF-cross short, cf. 2082's +132R, 1077's +18R) — and
   the guard erases it exactly as designed: +2R-cap −25.17%, drop-top-5%/fold −30.46% (REMOVING the tail makes
   the short cells *much* more negative because the tail was the only thing offsetting a wall of small −1R
   losers). Textbook thin-tail trap.

3. **The compression gate fires constantly and the cost wall dominates on crosses.** The tight-coil condition
   is NOT selective — 4.7–20k fires; on the full 28-pair universe (15–20k fires, wide cross spreads × many
   small trades) the book is −17 to −21%/yr vs USD7's −4 to −10%, the same cost-amplification arc 1003 found
   for crosses. A "small stop" entry pays the spread on a small absolute move = the worst possible
   cost-to-edge ratio.

## Diagnosis + meaning

The compression-coil's appealing geometry (small stop) does not survive contact with the market: a tight coil
on liquid FX breaks into a **coin-flip-to-adverse** next move, so the small stop is hit on the median trade
and no winner runs far enough to manufacture the R-multiple the premise needed. The small stop *amplifies*
the cost-to-move ratio rather than the win-to-loss ratio. This is the conservation law (LESSONS run-2) on the
compression axis: a price-structural entry samples the near-martingale; take-the-loss + a runner cannot
manufacture skew from a martingale, regardless of how tight the stop is.

**Relation to the gross-lens prior (a clean methodological note).** Unlike arc 1064→1074 (where I had to
*correct* the "gross is strictly worse than the engine" bound because take-the-loss caps the gross left
tail), here the gross kill (1062: capture 0.4535, drift −0.069) and the honest-engine kill **agree, and the
engine is even more decisive.** Why the difference from 1074: 1074's continuation entry had a modest gross
left tail the stop could cap; the compression breakout is *adverse-first with no compensating right tail at
all* (largest long winner +0.34R), so capping the left tail (take-the-loss) just books the −1R on the median
and there is nothing on the right to offset it. The corrected 1074 bound ("gross-vs-engine bias direction
depends on path shape") predicted exactly this: a no-right-tail adverse-first entry is killed *harder* by the
honest engine, not rescued.

## Closure

With the entry-geometry closures (Donchian breakout 1074/2081, pullback-resume 1075/2083, vol-expansion 2082,
shock + continuation-book 2081) and the timeframe closures (D1 1076, W1 1077), **the compression-coil breakout
was the last untested entry geometry** in the positive-skew continuation shape. It is now KILLED on the honest
engine under the mandated mean + median-per-fold + tail-removed lens. Positive-skew continuation is therefore
closed across **entry geometry (breakout / vol-expansion / pullback-resume / shock / continuation-book /
compression-coil) × timeframe (H4/D1/W1) × direction × universe × exit menu** — every axis the apparatus can
vary. The operator's ONE open in-charter thread (LESSONS 2026-06-06) is comprehensively closed by direct
engine measurement under the mandated metric.

The genuine long-vol / positive-skew diversifier the vehicle wall needs lives only in a **less-efficient
instrument universe** (cross-asset trend, `NEEDS_ENABLEMENT.md` #1 — the same positive-skew shape on a
less-arbitraged universe where the tail may actually pay; operator-gated, blocker = historical data). The
in-charter FX version is what these arcs closed. The only live levers are out-of-band: operator path-A
gate-governance call on the 4-component mean-positive book, and/or a charter unlock.

Components UNCHANGED (4 PORTFOLIO; me_long-solo the honest deploy object; deployable = 0). No canonical
change; no FLAG (carries the standing arc-3017 risk-convention FLAG; linear regime); no council (decisive
clean negative under an objective pre-registered guard, not a survivor/fork); OOS NEVER touched.

## Threads / handoff

- **New BUILT tool:** `discovery/tools/compression_coil_breakout_signals.py` (`CompressionCoilBreakoutSignal`)
  — reusable for any future compression/squeeze-breakout test; registered at step (i).
- **Reusable lesson:** the small-stop→large-R-multiple ("squeeze") premise is **empirically false on liquid
  FX** — the compression-coil break is adverse-first with no right tail; the tight stop amplifies cost, not
  skew. Future "tight-stop / squeeze / volatility-contraction breakout" ideas on liquid FX resolve here.
- **Frontier:** in-charter positive-skew continuation is closed on ALL axes; convergent terminus (0 PASS, 4
  PORTFOLIO, deployable=0). Next event is out-of-band (operator STOP / range-drop / one charter unlock — the
  cross-asset-trend unlock #1 is the SAME positive-skew shape on a less-efficient universe).
