# Arc 3000 — Mean-Reversion Long on Less-Efficient / Coupled Crosses

> **Arc id:** 3000 · **Chat:** 3000–3999 (continuous, first arc) · **Date:** 2026-06-04
> **Final verdict:** **FAIL (cheap-kill at triage)** — best reversion version (RSI<25 oversold
> long on the 9 most-coupled crosses) scores 3/3 negative folds (worst −20.22%, mean −12.76%);
> pool honest +1R-before-SL capture 0.4936 < 0.50.
> **Idea-family:** the two most-emphasized arc-1002 steers at once — the **instrument-universe**
> lever (less-efficient CROSSES, not majors) and a **mean-reversion** construction (the first
> non-continuation mechanism tested across the whole programme).

First continuous arc of chat 3000–3999. Scored solely by `MultiPairBacktester` (FundedNext costs ON,
SL-first take-the-loss). Engine/measurement **called, never re-rolled**; the signal is an experiment tool.

---

## (a) Log read + synthesis — FRESH EYES

Pulled `origin/main`. Honest-era corpus = Arc 0 (trial) + arcs 1000/1001/1002 (chat 1000s). LESSONS.md
still empty (no operator compression yet). No pre-reset eliminated-list consulted (fresh eyes; only
carry-forward is the Arc-10 gate-fidelity *methodology* lesson).

**What's been tried (all FAIL, all the same signature):**
- **arc 0** — pullback-in-uptrend long (H4 majors). Selection lever closed (AUC≈0.50); real-but-sub-cost.
- **arc 1000** — cross-sectional momentum long (H4 majors). Dispersion *inversion* (falsified); sub-cost.
- **arc 1001** — volatility-contraction breakout long (H4 majors). Coil predicts vol not direction; sub-cost.
- **arc 1002** — D1 trend-following long (majors). Coin-flip base is **timeframe-invariant** (D1≈H4); sub-cost.

**The accumulated steer (arc 1002, explicit):** *"stop testing directional-long entries on majors; change a
more fundamental lever — instrument universe (less-efficient CROSSES; the backup has ~28 pairs),
portfolio/selection of decorrelated sub-cost signals, or a non-directional construction."* Four independent
directional-long attempts share one signature: a ~0.49 directional base that beats random entry but cannot
clear FundedNext costs + SL-first. The binding constraint is the **cost/SL-first hurdle against a coin-flip
base**, not the entry construction.

**Framing chosen (fresh eyes, distinct from the 1000s chat's continuation/breakout/momentum line).** Attack
the steer on TWO axes at once, both untouched by prior arcs:
1. **Instrument universe.** Majors (esp. EURUSD) are the most efficient, most-arbitraged instruments on
   earth. *Crosses* — derived, less liquid, and (for the tightly-coupled ones: EUR/GBP, EUR/CHF, AUD/NZD,
   NZD/CAD, …) behaving like spreads between linked economies — should retain more exploitable structure.
   The backup has 28 pairs; arcs 0–1002 only ever used the 8 liquid majors. **Mechanism:** efficiency
   gradient.
2. **Mean-reversion.** Every prior arc bet on *continuation* (pullback-resume, XS-momentum, contraction
   breakout, trend-following). **None** tested *reversion*. The coupled crosses are exactly where reversion
   should be strongest (linked economies → the cross oscillates around a slow equilibrium rather than
   trending). **Mechanism:** economic coupling → spread reverts to equilibrium.

Long-only is forced by the apparatus (`build_arc_pool` / `MultiPairBacktester` simulate the long side only),
so "reversion" here = **buy the oversold / stretched-down side** and ride the bounce.

## (b) Observation → idea (observe before hypothesizing)

Honest +1R-before-SL **LONG** capture (the in-tree take-the-loss label; SL=2·ATR, hold 120, entry next-bar
open at ask; SL-first same-bar), **all 28 pairs**, IS 2010–2020, vectorized to match `core.sim.honest_label`
semantics. Baseline from arcs 0/1000: majors ~0.4877; **>0.50 = a real directional-long edge exists.**

**Observation #1 — unconditional capture is INSTRUMENT-INVARIANT (and crosses are *worse*):**

| group | n | uncond capture |
|---|---|---|
| MAJOR (7 USD pairs) | 121,465 | **0.4859** |
| COUPLED (9 intra-bloc crosses) | 156,179 | **0.4742** |
| TREND_X (12 cross-bloc / JPY crosses) | 208,139 | **0.4719** |

Nothing reaches 0.50. The least-efficient instruments have the **lowest** long capture (more negative drift
+ wider spread/ATR for a long-only ±2ATR race). A handful of single cells touch ~0.50–0.51 (EURGBP 0.5000,
NZDCAD 0.5036, EURCAD 0.5008) but scattered, within noise (n≈17k ⇒ CI ±0.0075), not group-consistent. The
**instrument-universe steer does not break the coin-flip base** — the base is instrument-invariant, just as
arc 1002 found it timeframe-invariant.

**Reversion state (z = (close − SMA50)/ATR, causal):** bucketing capture by z-decile per group shows **no
monotone reversion lift** as z goes negative on any group. The highest-capture cells sit at *positive* z
(faint continuation) — the already-dry axis. COUPLED oversold (z<−1) = 0.4752 vs 0.4742 uncond ≈ zero lift.

**Observation #2 — fail the BEST reversion version** (multiple oversold proxies × two barrier scales, on the
9 most range-bound coupled crosses, trade-weighted):

| reversion proxy | cap @ SL2 (base .4742) | cap @ SL1 (base .4522) | pairs w/ lift @SL1 |
|---|---|---|---|
| RSI<30 | 0.4724 (−.002) | 0.4690 (+.017) | 5/9 |
| **RSI<25** | **0.4853 (+.011)** | **0.4907 (+.039)** | **6/9** |
| z<−2 | 0.4751 (+.001) | 0.4548 (+.003) | 4/9 |
| z<−3 | 0.4719 (−.002) | 0.4590 (+.007) | 3/9 |
| consec≥3 down | 0.4684 (−.006) | 0.4555 (+.003) | 4/9 |
| big drop (<−1.5ATR) | 0.4621 (−.012) | 0.4531 (+.001) | 4/9 |
| Bollinger lower-2σ | 0.4609 (−.013) | 0.4483 (−.004) | 4/9 |

**RSI<25 is the only proxy with a non-trivial lift — and it peaks at 0.4907, still BELOW 0.50**, thin
(~30 fires/pair/yr), and not cross-pair-robust (3 of 9 pairs show no/negative lift; the headline is partly
carried by AUDNZD's small-n 0.628 cell). A 1:1-RR barrier needs capture > 0.50 just to break even *before*
costs. **The reversion tilt is real but sub-coin-flip — the same sub-cost signature as every prior arc.**

**Idea (best reasoned version to formalize):** long when **RSI(14) < 25** (deepest robust oversold) on the
9 coupled crosses, refractory 6, SL=2·ATR. *Because:* if economic coupling makes these crosses revert, a
deep-oversold long should bank the bounce; this is the strongest reversion signal on the instruments where
reversion should be strongest.

## (c) Characterize — ex-ante population

`build_arc_pool`, 9 coupled crosses (EURGBP EURCHF AUDNZD NZDCAD CADCHF AUDCHF AUDCAD NZDCHF GBPCHF), H4
5ers_eet, IS 2010–2020, SL=2·ATR, hold 120, 1% risk. `pool_sha256 bc606a89945ea83a…`.
- **1,009 IS trades** (92–141/pair). Exit mix 812 `hard_sl` (80%) / 197 `time_exit`. Mean final_r
  **−0.1409**, median −1.0. Honest +1R-before-SL **0.4936** (matches observation — sanity confirmed).

## (d) Cheap kills

- **Pool floor:** 1,009 ≫ 50 → **PASS.**
- **Oracle-best-cluster ceiling: SKIPPED — deliberately.** Observation already established capture < 0.50
  for the best entry, so there is no *reachable* upside to diagnose; a high oracle ceiling on a sub-0.50
  entry is the Arc-0 hindsight trap (clusters defined from realised path; AUC≈0.50 ⇒ unbridgeable). Running
  it would be ritual, not rigor (the arc-1000 council's exact warning). When the triage is decisively
  negative, the ceiling adds nothing — same call arcs 1001/1002 made.
- **3-fold honest triage** (A1, SL=2·ATR, 1% reset-floor, exposure 1/pair 2/ccy,
  `sl_partial_close_1r_runner_trail` — the excursion-banking exit, i.e. reversion's BEST shot; FundedNext
  costs ON, SL-first; OOS years 2013/2016/2019, matching arc 1000):

  | fold OOS | ROI | DD | n |
  |---|---|---|---|
  | 2013 | **−20.22%** | 22.54% | 107 |
  | 2016 | −5.17% | 12.26% | 126 |
  | 2019 | −12.89% | 20.67% | 104 |

  worst **−20.22%**, mean **−12.76%**, **3/3 negative**. → **KILL at triage** (protocol §5d), before full WFO.

## (e)–(h) Diagnose / address / validate / council — NOT REACHED

Cheap-killed at triage; no worthwhile/reachable ceiling, so no diagnosis fork and (correctly) **no council**
(council fires at a diagnosis fork with reachable upside, or on a survivor — neither exists here). No survivor,
so no `passed/` record. The partial/runner banking exit (reversion's most favorable exit) was already in the
triage config and still lost −12.76% mean — the asymmetric-payoff escape hatch is covered.

## Final verdict — FAIL (cheap-kill)

Mean-reversion long on coupled crosses is **not deployable**. It dies for the now-familiar reason, reached
twice over: (1) the directional-long base is instrument-invariant (crosses ≤ majors, all ~0.47–0.49 < 0.50),
and (2) the reversion tilt, even in its best version on the most-coupled instruments, never crosses 0.50 —
a real but sub-cost edge. The triage confirms it (−12.76% mean, 3/3 negative) once FundedNext costs + SL-first
+ partial-close-pays-spread-twice apply.

## Lessons (candidate for LESSONS.md compression)

1. **The FX directional-long coin-flip base is INSTRUMENT-INVARIANT.** Across all 28 pairs, unconditional
   long capture is 0.47–0.49 (majors 0.486, coupled crosses 0.474, trending crosses 0.472) — the
   less-efficient crosses are *worse*, not better. The "less-efficient crosses" steer is **closed**: changing
   the instrument does not break the base, the same way changing the timeframe (arc 1002) did not.
   **Independently corroborated by arc 1003** (landed on main mid-arc): the 1000s chat hit crosses from the
   *trend-momentum* angle and found the identical cross base **0.4712** < majors 0.4877. Two chats, two
   different cross signals, one conclusion — strong convergence. Arc 1003 adds a sharp mechanistic nuance:
   crosses *do* trend (gross drift **+0.10R/trade**) but **wider cross spreads eat it** → the constraint is
   **EDGE < COST**, not "no edge." My reversion result is the complementary case: buying oversold dips does
   NOT even catch that positive cross drift (pool mean final_r **−0.14R**, capture < 0.50) — reversion is
   gross-coin-flip *and* cost-bled, whereas trend is gross-positive *but* cost-bled. Either way: net sub-cost.
2. **Mean-reversion (buy oversold) confers NO deployable long edge** — the first non-continuation mechanism
   tested across the programme, and it lands sub-cost too. Best version (RSI<25, tight target) lifts capture
   only to ~0.491 (< 0.50), thin and not cross-pair-robust; all other oversold proxies (z-score, Bollinger,
   consec-down, big-drop) are flat-to-negative. The reversion tilt is real but sub-coin-flip.
3. **SIX independent directional-long arcs now FAIL identically, across both directional mechanism families.**
   Continuation/trend family: pullback (arc 0), XS-momentum (1000), vol-contraction-breakout (1001),
   D1 trend-following (1002), cross trend-momentum (1003). Reversion family: oversold-revert (this arc, 3000)
   — the lone non-continuation mechanism, and it fails too. Coverage now spans **two timeframes** (H4, D1),
   the **full 28-pair instrument universe** (majors + coupled + trending crosses), and **two independent
   chats** (1000s + 3000s) converging. This is no longer "entry construction X is dry" — it is overwhelming
   evidence that **a long-only, fixed-±R-barrier, single-instrument directional bet on FX is net sub-cost,
   period** — independent of mechanism (continuation OR reversion), timeframe, and instrument. The binding
   constraint is structural (gross edge too small × FundedNext cost hurdle × SL-first), not the signal.
4. **An excursion-banking exit (`sl_partial_close_1r_runner_trail`) does not rescue a sub-0.50-capture entry**
   — re-confirmed here on reversion (−12.76% mean). Consistent with arc 0. **Independently re-confirmed by arc 1004** (landed mid-arc): on the
   positive-drift cross-trend signal, no exit variant (wide-trail let-it-run, 3R target, standard trail)
   flipped net-positive either — EDGE<COST is an entry/cost problem, not an exit problem.

## Methodological note (honest caveat)

The +1R-before-SL capture metric used in observation is **blind to small drifts** (it only registers a
≥1R favourable excursion before the stop). Arc 1004 flagged this for calendar/flow ideas, where the
expected edge is a small mean forward drift. It does NOT weaken this arc's verdict: a *reversion* long bets
on a ≥1R bounce (exactly what the metric measures), and the binding evidence is the **3-fold honest
triage's real P&L** (−12.76% mean, 3/3 negative) — not the capture scan. The capture scan is the cheap
pre-filter; the triage is the verdict.

## Threads / what didn't help

- **Closed:** mean-reversion (oversold) long on crosses; the instrument-universe lever as a route to a
  *directional* long edge (base is instrument-invariant).
- **Open (NOT a rescue of this arc) — two surviving steers, both away from trade-level price-direction:**
  (1) **Non-price-direction / calendar-flow** — arc 1003's steer: turn-of-month rebalancing flow, NOT yet
  tested (arc 1000 covered hour/day-of-week but not day-of-month). A flow effect's gross edge is not bounded
  by the 0.49 directional wall. (2) **Portfolio/selection of decorrelated sub-cost edges** — six mechanisms
  each beat random but fail costs *standalone*; combining decorrelated weak edges into a portfolio whose
  *aggregate* fold-ROI is all-positive is the one lever that doesn't require beating 0.50 trade-by-trade (a
  different ex-ante population + a portfolio-level framing of the all-folds-positive judge). This subsumes
  arc 0/1000's portfolio thread. Both are the highest-value untested directions; either is a fresh arc, not
  a rescue of this one. **Coordination note:** arc 1003 already named the calendar/flow steer, so the 3000s
  range should prefer the portfolio/selection steer to avoid two chats colliding on turn-of-month.
- **Low-priority / likely-dry:** reversion on D1 crosses (arc 1002's timeframe-invariance makes a different
  outcome unlikely); a true market-neutral *spread* (pairs trade) is not expressible in the long-only
  single-instrument apparatus (a single cross like EURGBP is already a relative bet, and it failed here).

## Flags (code NOT merged — human-gated, per protocol §9)

None requiring the canonical core. The signal module + observation/triage drivers are scratch
(`_disco3_work/`, not committed; reproducible below). No reusable experiment tool was needed (the null
baseline was not required — an all-negative triage is decisive; the null distinguishes a sub-cost edge that
*survives to a full WFO*, which this did not).

## Reproduction

- **Data:** `histdata_root = C:\Users\panap\histdata_backup`, `cache_root = data/cache`,
  `boundary_convention = "5ers_eet"`, TF H4. **Pairs:** EURGBP EURCHF AUDNZD NZDCAD CADCHF AUDCHF AUDCAD
  NZDCHF GBPCHF (the 9 most-coupled crosses). Observation #1 scanned all 28 pairs.
- **Signal:** `_disco3_work/arc3000_signal.py` — `RSIOversoldRevertLong(rsi_period=14, rsi_thresh=25,
  refractory=6)`, SL=2·ATR, hold 120. `pool_sha256 bc606a89945ea83a…`.
- **Drivers (scratch, not committed):** `observe.py` (b, 28-pair capture + z-buckets),
  `observe2.py` (b, reversion proxies × 2 barrier scales), `arc3000_triage.py` (c/d, pool + 3-fold triage).
  Run with `PYTHONPATH=. py _disco3_work/<script>.py`.
- **Engine:** `MultiPairBacktester` via `A1Architecture` + `ArcFoldRunner`; FundedNext costs netted at
  `build_fold_stats_from_run`; IS folds = `build_v3_folds` (is_days≥365), triage = OOS years 2013/2016/2019;
  judge = `judge_all_folds_positive`.
