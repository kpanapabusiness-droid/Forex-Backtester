# arc_1062 — DIRECTION-AGNOSTIC COMPRESSION-EXPANSION convexity leg (long-vol 5th-leg candidate)

- **Chat:** 1000s
- **Date:** 2026-06-06
- **Type:** observation cheap-kill (§5d), fresh-eyes novel construction (§5a); BUILT tools only;
  no engine / null / council; no canonical change; no FLAG; OOS untouched.
- **Disposition:** **KILL**
- **Components UNCHANGED** (all 4 PORTFOLIO; me_long-solo the honest deploy object).

## Fresh-eyes (step a)

Pulled main (highest in-range arc 1061 → resume 1062); read protocol + DISCOVERY_LOG (both tiers, all
ranges) + LESSONS + TOOL_REGISTRY (canonical + all BUILT); no `discovery/STOP`. The corpus is at a
**rigorously-proven** within-charter terminus — confirmed here by independently re-checking, not
inheriting, the load-bearing closures:
- 4 PORTFOLIO components (gap 1006, me_long 1011, fbr 1013, me_short 1019); the 4-way book is never
  all-folds-positive — its **two binding negative folds are 2015 and 2018** (the strong-USD wall);
  co-sim (item E) confirmed the failure is **fundamental, not a combiner artifact**.
- The **5th-leg route is structurally closed** (arc **2022** weighting dilemma — even the strongest real
  +2015&+2018 leg, the *directional* shock-continuation 3019, is throttled to an overfit-tiny honest
  weight) and the **densification/path-B route is closed by portfolio-math PROOF** (arc **3021** — a
  shared dollar factor floors book variance, P(AFP) plateaus ~0.30, unreachable at any N).
- Every dispatch frontier item is independently closed by data: short-side asymmetries (up-gap short
  1016; fbr-mirror short 1014/2011/3011/1035 — all KILL), relative-value / market-neutral (cointegration
  2010, laggard 2003 — KILL, doubled-cost wall on majors), the USD-neutral-cross dollar-factor escape
  (1018/2049/2064 — all KILL), and the documented calendar-flow family (1005/1008/1010/2057/2058/3008/
  1061 — all KILL).

Per §2 / §5a + the arc-3004 *"apparatus is incapable is seductive"* warning, I did **not** declare the
terminus by argument. Arc 1061 asserted the long-vol/convexity leg is "already closed by 2022" — but 2022
only tested the *directional* construction (shock-continuation: enter AFTER a confirmed big bar in that
bar's direction). The **direction-agnostic** convexity construction — now buildable since shorts are
verified-enabled (PR #273) — has never been tested. So I tested it empirically rather than assume it.

## Idea + because

The two binding folds 2015 and 2018 SHARE one property: **elevated realized volatility** (a CHF-de-peg
shock and a persistent strong-USD trend are both big-range regimes). A **long-volatility / convexity**
leg that does NOT pick direction in advance should be positive precisely when realized vol spikes — i.e.
in 2015 AND 2018 regardless of direction — making it the regime-orthogonal complement of the reversion
book. The construction: a multi-bar **COMPRESSION** (low-ATR coil) that **breaks**, taken in **whichever
direction it breaks** (long the up-break, short the down-break — a single-leg straddle analog).

Why this is **not** closed ground / **not** arc 1001 / **not** arc 2022:
- **≠ arc 1001** (vol-contraction breakout LONG, killed at triage): that could only catch up-breaks; this
  is the **direction-agnostic** version shorts now unlock — the symmetric straddle 1001 structurally
  could not be.
- **≠ arc 2022** shock-continuation: that is post-confirmation *directional* (continuation of a completed
  move); this enters AT the break of a tight coil, BEFORE the expansion forms — capturing the **convexity
  of the expansion itself**.
- The KILL/PROCEED criterion is the **regime-orthogonal test**, not overall edge: is it RELIABLY POSITIVE
  in BOTH binding folds 2015 AND 2018? A coin-flip-overall leg is still the 5th leg IF its positive years
  are the book's negative years.

Falsifiable hypotheses: **(A)** positive honest capture/drift in BOTH 2015 and 2018; **(B)** a TIGHTER
coil separates positively from a loose break (a genuine compression-convexity gradient, not closed
breakout); **(C)** positive years concentrate in the high-vol/strong-USD folds (regime concentration).

## Method

BUILT tools only. Causal compression-break per pair at bar t: coil width = (12-bar high-low range)/ATR
≤ 3.5 (tight); up-break = close_t > prior-12-bar high (shift(1), excludes t) → long next bar; down-break
= close_t < prior-12-bar low → short next bar. All quantities known at t's close. `observe_long_capture`
(direction-aware honest take-the-loss capture + N-bar fwd drift) run long on up-break bars, short on
down-break bars, via `restrict`; concatenated; IS 2010-2020, **H4, 7 USD majors** (the structural TF
where fbr lives). Levers swept (coil/break lookback, tightness threshold, hold) for regime-orthogonality,
NOT optimization. Driver: `discovery/_disco1_work/arc1062_compression_expansion_convexity.py`.

## Result — gross-negative, sub-null, no compression gradient, binding folds not robustly covered

**(A) FALSIFIED.** Tight-coil default (coil12 / ≤3.5 ATR / break12 / hold24): n=5870, **capture 0.4535**
— *below* the ~0.49 unconditional H4-majors null (the entry is WORSE than random), **drift_mean −0.069**,
**mean-of-year −0.083** (gross-negative on the typical trade, before costs). The binding folds are NOT
robustly covered: 2015 **+0.029** (marginal), 2018 **+0.066** in the default but **−0.181** in the
longer-hold cell (sign-flips across cells = not real). The OTHER strong-USD years are strongly negative:
2014 **−0.397**, 2016 **−0.191**.

**(B) FALSIFIED — no compression gradient.** Splitting the default fires by actual coil tightness:
tight-tercile drift −0.050 vs loose-tercile −0.129 — **both negative**; tightness buys no positive edge.
(On the wider inert-threshold set the gradient ran the *other* way, tight −0.069 vs loose +0.088 — i.e.
across both scalings the genuine tight coil is never the positive side.) There is no convexity signal in
compression; the tight-coil break is the classic **false-break trap**.

**(C) FALSIFIED — no regime concentration.** Strong-USD folds 2014 −0.397 / 2015 +0.029 / 2016 −0.191 /
2018 +0.066 — dominated by negatives, only 2015 weakly positive. The convexity leg does NOT
systematically pay in high-vol/strong-USD years.

## Diagnosis + meaning

A direction-agnostic compression-expansion convexity leg — the construction that *should* be positive in
both high-vol binding folds — is gross-negative, sub-null on capture, has no compression gradient, and
does not robustly cover 2015 **or** 2018. The reason is the arc-1060 lesson, sharpened: **high realized
volatility is not the same as expansion-convexity payoff under take-the-loss.** 2018's vol is a persistent
**one-directional TREND** whose pullbacks stop-run the straddle's wrong-side entries to −1R (and 2014/2016
strong-USD are the same trend-grind); only a genuinely **bidirectional shock** (2015 de-peg) marginally
rewards the straddle, and even that is thin and sub-cost. §5f exit-sweep is moot: the entry base is a
coin-flip (capture 0.45 < null), so per §5f it does not even trigger the mandatory exit-menu sweep, and
the failure is at the capturability / fold-sign level, not the exit.

## NEW lesson

**"Both binding folds are high-vol ⇒ a direction-agnostic convexity leg covers both" is FALSE.** Even a
straddle-analog (long-OR-short = whichever way a coil breaks) does not systematically pay in the strong-USD
high-vol years (2014 −0.40, 2016 −0.19 strongly negative; 2018 fragile/sign-flipping; 2015 only marginal),
because **2018-class vol is directional-trend vol** (one-way move + stop-running pullbacks that whipsaw the
wrong side to −1R), not the bidirectional expansion vol a straddle needs. This closes the **long-vol /
convexity 5th-leg route from the CONSTRUCTION side**, complementing arc 2022's closure from the *weighting*
side (which used the directional shock-continuation): both the directional (2022) and the direction-agnostic
(this arc) long-vol constructions fail to be the +2015&+2018 regime-orthogonal leg. It also reconfirms the
dispatch's own prediction that **shorts do not revive closed breakout/momentum** — the direction-agnostic
(symmetric) breakout is sub-null capture (0.45), exactly the long base's coin-flip mirrored. Adds
direction-agnostic compression-expansion to the mapped-dead 5th-leg attempts; the 2015/2018 wall is a
**two-different-regime** barrier (whipsaw-shock vs directional-trend), not a single-volatility-exposure gap.

## Threads / handoff

Components UNCHANGED (all 4 PORTFOLIO; me_long-solo the honest deploy object, vehicle-infeasible). Lever =
**operator path-A** (gate-governance on a sound mean-positive ~3-bet book — 1023/2021/3021) **OR a charter
unlock** (operator-gated macro/options/COT data, `NEEDS_ENABLEMENT.md`). This arc empirically closes the
last argued-but-untested EDGE candidate (the long-vol/convexity leg) from the construction side, so the
within-charter terminus is now proven on **both** the weighting axis (2022) and the construction axis (this
arc). **Datum:** direction-agnostic tight-coil break (H4 USD majors, coil12/≤3.5ATR/break12/hold24) capture
0.4535 (sub-null), drift_mean −0.069, mean-of-yr −0.083; binding folds 2015 +0.029 (marginal) / 2018 +0.066
default but −0.181 longer-hold (sign-flips); strong-USD 2014 −0.40 / 2016 −0.19; no compression gradient
(tight −0.05 vs loose −0.13). Driver `discovery/_disco1_work/arc1062_compression_expansion_convexity.py`
(BUILT tools only, single-use; no new BUILT tool registered — pure `observe_long_capture` + a causal
compression-break mask).
