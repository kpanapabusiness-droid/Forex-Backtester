# arc_1065 — lens-A: the CORRELATION-vs-SIGN crux of me_long's strong-USD losses

- **Chat:** 1000s
- **Date:** 2026-06-06
- **Type:** observation cheap-kill (§5d), council-surfaced forward thread (arc 1063 thread (a),
  logged-not-executed — the LAST open thread in the 1000s range); BUILT/reused tools only; no engine /
  null / council; no canonical change; no FLAG; OOS untouched.
- **Disposition:** **KILL**
- **Components UNCHANGED** (all 4 PORTFOLIO; me_long-solo the honest deploy object).

## Fresh-eyes (step a)

Pulled main (highest in-range arc 1064 → resume 1065; `discovery/STOP` ABSENT). Read
`DISCOVERY_PROTOCOL` (v1.1, followed), `DISCOVERY_LOG` (both tiers, all three ranges incl. disposition),
`LESSONS`, `TOOL_REGISTRY` (canonical LOCKED + all BUILT), front-door state. Honest-era synthesis:

- **4 PORTFOLIO components**, all forced-flow REVERSION: gap (1006), me_long (1011), fbr (1013),
  me_short (1019). The book is mean-positive but NEVER all-folds-positive — blocked
  combination-invariantly by the strong-USD **2015 & 2018** wall (1015/2008/3009/1020; co-sim item E
  FUNDAMENTAL). Honest §5f exits weakened 3/4 legs → deploy object collapses to me_long-solo, itself
  marginal/tail-fragile (1042-1046/1056-1058/2063).
- **The route's lever is the operator's path-A gate-governance call** (1023/2019/3021/1032). Edge-hunt
  closed: path-B densification proven impossible at empirical correlation (3021), explore-now MENU
  exhausted (M1/O1/L1/Q1/G1/S1), 5th-leg unfound across ~22 routes, shorts + relative-value all KILL,
  OHLC charter mined out down to M1 intrabar (1031/1052/1054).
- **The one fresh thread.** arc 1063 (POWER AUDIT of the all-folds judge) reframed "frontier exhausted"
  and logged **two** un-executed council forward threads. Thread (b) — continuation/positive-skew shapes
  — was **closed by arc 1064 (KILL)**. Thread (a) — the **USD-factor correlation-vs-sign crux** — was
  still logged-not-executed. This arc executes thread (a), the last open thread in my range.

## The thread (the *because* — and why it is genuinely distinct, not a re-run)

The 1063 council CHALLENGED its own lens-A premise: *"2015/2018 is a SIGN conflict, not co-located
noise → a symmetric veto can't make a year prefer the leg it rejects."* lens-A SETTLES that empirically:
**is me_long's strong-USD loss a hedgeable COMMON-USD-FACTOR component** (then a beta-sized USD-basket
short — shorts now enabled, PR #273 — could strip it, possibly rescuing the worst fold) **or an
irreducible IDIOSYNCRATIC-to-the-regime / sign-conflict effect** (then no symmetric factor hedge helps)?

Distinct from the two nearest prior arcs:
- **arc 1059 (USD-breadth FILTER)** conditioned ENTRY — dropping systematic fires loses their count AND
  removes me_long's directional-WMR +2018 help ("edge & tail are the same exposure"). lens-A keeps ALL
  fires and subtracts the beta·factor component — a **HEDGE, not a filter** → isolates pure idiosyncratic
  alpha. A hedge that neutralizes only the common factor (keeping each pair's full idiosyncratic
  reversion) is a strictly different operation from a filter that removes whole trades.
- **arc 2018 (cross-sectional / fully USD-neutral)** neutralizes the WHOLE common move (long-laggard/
  short-leader). lens-A removes only the **beta-sized** common-factor exposure — the least-aggressive
  hedge that could still strip the regime risk while preserving the most edge.

## Method

Obs-level, gross, no engine (the engine + the hedge's doubled FundedNext cost only matter IF the gross
decomposition shows a rescue — it does not). Reused arc 1059's `build_common_usd_move` (canonical
`_month_end_into_move`) + BUILT `observe_long_capture` (honest take-the-loss capture / 2-bar fwd-drift).
me_long committed config (`MonthEndReversionLongSignal` threshold_atr=1.0, into_bars=2, D1, 7 USD
majors), IS 2010-2020, n=126 fires. Driver
`discovery/_disco1_work/arc1065_melong_factor_vs_sign.py`.

For each fire: its 2-bar reversion outcome `fwd_drift_atr`; the month's **common USD factor**
`f` = mean USD-signed 2-day move into ME across all 7 majors (sign = USD direction, |·| = breadth); the
position's USD exposure = `+USD_FACTOR[p]` (long USDXXX gains when USD rises, long XXXUSD when USD
falls); `factor_signal = USD_FACTOR[p]·f`. Three readouts:
1. **Pooled regression** `drift ~ alpha + beta·factor_signal` → factor loading + R².
   *(sign-invariant: flipping the factor_signal convention flips beta but leaves R², the fitted
   component, and the hedged residual identical — verified by re-running both conventions.)*
2. **Within-month sign concordance** of simultaneously-held fires, per year (factor-wide ⇒ high).
3. **The crux:** per-year RAW drift vs factor component vs **de-factored (hedged) alpha** `y − beta·factor_signal`.

## Results (decisive — beta ≈ 0, the hedge is INERT)

```
unconditional me_long: drift_mean +0.1451  drift_med +0.2199  capture 0.5159

(1) POOLED  drift ~ alpha + beta·factor_signal
    alpha +0.1626   beta +0.0263   R2 0.0003   corr +0.0185      <-- ZERO factor loading

(2) within-month sign concordance (1.0 = all held fires same outcome sign):
    2010 .833 / 2011 .875 / 2012 .667 / 2013 .833 / 2014 .667 / 2015 .667 /
    2016 .500 / 2017 .500 / 2018 .667 / 2019 .819 / 2020 1.000   (avg ~0.72)

(3) per-year RAW vs HEDGED alpha:
    RAW          neg-years 3/11  mean +0.1474  worst-yr -0.2369 (2016)
    HEDGED alpha neg-years 3/11  mean +0.1646  worst-yr -0.2272 (2016)   <-- ~identical to RAW
    2014-16 block:  RAW -0.0155   HEDGED +0.0003   (still ~0/negative)
    2018:           RAW -0.0596   HEDGED -0.0360   (drift lens; engine fold +0.90, see nuance)
```

- **me_long's per-fire reversion outcome has essentially ZERO common-USD-factor loading** (beta +0.026,
  **R² 0.0003**, corr +0.019). There is no directional-USD factor to hedge.
- **A beta-sized USD-basket hedge is INERT** — the de-factored alpha is ~identical to raw (mean +0.165 vs
  +0.147; worst-year −0.227 vs −0.237; same 3/11 negative years; the 2014-16 block stays ≈0/negative,
  2016 robustly −0.23 hedged). The hedge removes nothing because there is nothing factor-shaped to remove.
- **Within-month sign concordance is HIGH (~0.72)** — when me_long holds several majors at one month-end
  they DO succeed/fail together — **but this co-movement is ORTHOGONAL to the USD factor** (R²≈0). The
  positions co-move because the month-end reversion *as a regime* either works or fails for all of them
  that month, NOT because of a shared USD-directional drift a symmetric short could neutralize.

## Read / verdict

**lens-A CLOSES — KILL.** me_long's strong-USD-block loss is **NOT a hedgeable common-USD factor**; it is
**idiosyncratic to the month-end-reversion regime** (the reversion genuinely fails when the down-move
into ME *is* the USD trend, exactly as the signal's *because* and arc 1059 predicted — and that failure
survives a beta-sized factor hedge because it is non-directional). The 1063 council's pre-emption — *the
2015/2018 wall is a SIGN conflict, not co-located noise a symmetric veto could rescue* — is **empirically
confirmed.** §5f does not bite (the hedge is inert → there is no above-baseline best-version to take to
the engine). No new component; me_long UNCHANGED (PORTFOLIO).

**Nuance (honest).** The 2-bar drift lens reads 2018 slightly negative (−0.06); the engine fold ROI for
me_long-2018 is +0.90% (arc 2017, CI>0). The drift lens is a gross pre-pool screen, not the gate
(TOOL_REGISTRY `observe_long_capture` note). lens-A's conclusion rests on (i) the factor R²≈0 (a
sign- and aggregation-invariant fact) and (ii) the **2014-16 block**, which is robustly negative in BOTH
the drift lens AND the engine folds — so the drift-vs-engine 2018 distinction does not touch the verdict.

## NEW lesson

A thin reversion edge's per-year fold COHERENCE (here: ~0.72 within-month outcome-sign concordance — the
held positions fail together) can be **REGIME co-movement, not FACTOR co-movement** — and is therefore
**un-hedgeable by a symmetric directional (USD-basket) overlay** (R²≈0). You cannot strip the strong-USD
loss with a factor short because the loss is not a directional factor exposure; it is the
month-end-reversion-fails-in-a-trend regime itself. This generalizes arc 1059's "edge & tail are the same
exposure" from the **filter axis** to the **hedge axis**: neither filtering systematic entries (1059) nor
hedging the common factor (this arc) lifts the worst fold, because the exposure that carries the edge IS
the (non-directional) reversion regime. Before reaching for a factor hedge to rescue a thin book's worst
fold, regress its per-trade outcomes on the candidate factor — an R²≈0 says the co-movement is regime,
not factor, and the hedge will be inert.

## Programme position

This **closes lens-A** — the last logged-not-executed council forward thread in the 1000s range —
completing the forward-thread closure begun by arc 1064 (lens-B). The arc-1063 "frontier exhausted"
reframe now carries **no remaining caveat** in this range: edge-hunt closed, MENU exhausted, both 1063
council threads resolved, the 4-way book combination-invariantly blocked, the deploy object marginal and
tail-fragile. Lever remains the **operator path-A gate-governance call**. Components UNCHANGED (all 4
PORTFOLIO); deployable-system count = 0. No engine / null / council / canonical change / FLAG; OOS
untouched. No STOP sentinel set (only the operator sets it, §9).
