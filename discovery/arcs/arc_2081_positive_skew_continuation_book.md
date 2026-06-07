# arc 2081 — Positive-skew CONTINUATION book (the long-vol sleeve), judged on MEAN + median-per-fold + TAIL-REMOVED

> **Arc id:** 2081 · **Chat:** 2000–2999 (continuous) · **Date:** 2026-06-07
> **Type:** EDGE-HUNT on the operator-redirected frontier (positive-skew / continuation; LESSONS
> compression 2026-06-06). **Disposition:** _(filled at end)_
> **Scored solely by** `MultiPairBacktester` (FundedNext costs ON, SL-first / take-the-loss). Canonical
> apparatus CALLED (`Panel.from_pairs`, `build_arc_pool`, `ArcFoldRunner`→`A1Architecture`,
> `build_v3_folds`, `discovery_measure`); the trend entry is the registered EXPERIMENT tool
> `DonchianBreakoutLongSignal` (arc 2000). The tail-removed metric reuses arc 2063's geometry-only
> upside-winsorization arithmetic (`net_capped = min(net_pnl, K·$500)`, losses UNTOUCHED).

## Why this arc (the redirected frontier — NOT a re-derivation of closed ground)

The operator's 2026-06-06 LESSONS compression closed the reversion/directional terminus and opened ONE
new in-charter thread: **positive-skew CONTINUATION is UNTESTED.** The corpus's "trend-following = dead"
verdict was measured ONLY by +1R-before-SL **capture** and **mean-forward-drift** — win-rate-style lenses
**structurally blind to a positive-skew payoff** (low capture, near-zero *median* drift, but strongly
positive *MEAN* from a fat right tail). A genuine continuation edge (take-the-loss at −1R, ride winners
via trailing) judged on **MEAN + median-per-fold + TAIL-REMOVED expectancy** has never been run.

What the corpus actually tested (and why each leaves this cell open):
- **arc 2000** — Donchian breakout + full-size trailing, but H4 **majors only** (8), **3-fold triage**,
  long-only, MFE/capture lens. Mean-negative; tail found GENERIC. Its own #1 open thread: *"trend-following
  may only work diversified across MANY decorrelated instruments"* → the full 28-pair universe, untested.
- **arc 2012 / 2014** — deep-conjunction continuation longs, killed by the **capture/drift** lens (the
  mean-blind metric). Structure-control inverted, but the mean lens under a trailing exit was never run.
- **arc 3019** — extreme-shock continuation, strongest IS continuation (trailing_atr 7/10 IS, +2015/+2016)
  but the §5f nested-WFO selected **tp_3r** (a tail-CAPPING exit) for the frozen OOS and it died
  epoch-dependently. The trailing-exit IS edge was never carried to OOS; H4 majors only.
- **arc 2063** — winsorization applied to the **reversion** book (fbr runner), never a continuation book.

So the genuinely-untested cell = **a diversified continuation/breakout book across the FULL 28-pair
universe, exited take-the-loss + tail-preserving trailing, judged on per-trade MEAN R + median-per-fold +
TAIL-REMOVED.** Framed as the **long-vol sleeve**: structurally it should be positive in the reversion
book's death years (2015 / 2018) and could lift the combined book's Calmar (the binding *vehicle wall*).

## Hypothesis

A Donchian-breakout continuation entry across the full 28-pair universe, exited with a tail-preserving
trailing stop (take-the-loss at −1R, winners run), has a **mean-positive, broad-based** positive-skew
edge net of FundedNext costs — i.e. one that does NOT collapse when its biggest winners are removed —
and is positive in 2015/2018. *Because:* a trend/breakout book is intrinsically long-vol; rare large
right-tail trends across many decorrelated instruments could net positive where a single-pair / majors-
only cut (arc 2000) was generic-and-sub-cost, and could complement the short-vol reversion book.

## ⚠️ PRE-REGISTERED KILL-RULE (written BEFORE results — applied verbatim afterwards)

Tail-luck ≠ skew. The book is a real positive-skew edge **only if it survives removal of its biggest
winners.** Concretely, with per-trade realized R = `net_pnl / ($500 = risk_pct·SB)`:

1. **Mean expectancy** — per-trade mean R and per-fold mean ROI must be **> 0 net of costs**. If the raw
   net mean is ≤ 0 (or deeply negative, arc-2000 majors style) → **KILL** (you cannot diversify a
   negative-mean component positive — §11, arcs 3000/3001).
2. **TAIL-REMOVED guard (decisive).** Re-compute the mean with the right tail capped/dropped:
   (a) **+2R upside cap** (`min(net, 2·$500)`, losses untouched), and (b) **top-5% (and top-K) winners
   removed.** **If the mean goes ≤ 0 (flat or negative) under EITHER tail-removal — i.e. the edge is
   mean-positive ONLY because of the top-K winners — it is KILL.** No relabeling 2–3 monster trades / a
   lucky 2024-25 run as "positive skew" (cf. the thin-tail traps arcs 2011, 2063 caught).
3. A real edge shows **broad-based** right-tail behaviour: the median-per-fold is supportive (not a single
   monster fold) and the tail-removed mean stays clearly positive across folds.

Disposition (§11): **PASS** = all-folds-positive IS+OOS + mandatory council + survives the tail guard.
**PORTFOLIO** = mean-positive net of costs AND survives the tail guard (and ideally +2015/+2018 /
Calmar-lifting), but not all-folds-positive. **KILL** = everything else (incl. mean-positive-only-via-tail,
beats-null-but-net-negative, coin-flip).

## Method (CALLS canonical; experiment-side = entry mask + winsorization arithmetic only)

1. Load all **28 pairs**, H4, `5ers_eet`, real bid/ask (cache at the main-repo `data/cache`).
2. Entry = `DonchianBreakoutLongSignal(lookback=40, spacing_bars=6)` (registered, arc 2000), SL=2·ATR,
   `risk_pct=0.005` (LINEAR regime per the arc-3017 risk-convention FLAG; cap ~never binds).
3. Exit = the **trailing family** (`sl_plus_trailing_atr`, `sl_plus_trailing_swing`,
   `sl_partial_close_1r_runner_trail`) + `sl_only` reference. **NOT** tp_2r/tp_3r (tail-capping defeats
   the premise). §5f nested-WFO discipline if it reaches OOS.
4. Score over the full IS folds (`build_v3_folds`, 2011–2020), capture per-trade `net_pnl` via
   `apply_cost_model(runner.last_result.run_result, CostModel.fundednext()).breakdown` filtered to each
   fold's OOS slice — the arc-2063 unit. Compute per-trade mean R, median R, **+2R-capped mean**,
   **top-5%-removed mean**; per-fold mean+median ROI; majors-vs-crosses split; 2015/2018 signs.
5. Null = `PeriodicLongSignal` (being-long-anytime) under the same exit — does the breakout beat generic
   long-vol exposure?
6. Apply the pre-registered kill-rule verbatim. If it survives the cheap kill + tail guard → full OOS +
   mandatory council before any PORTFOLIO/PASS recording.

## Results

Drivers: `_disco2000_work/arc2081_continuation_skew.py` (Donchian, full universe, all trailing exits +
periodic null) and `_disco2000_work/arc2081_shock_addendum.py` (the corpus's strongest continuation,
arc 3019, re-measured under trailing + tail-removed on the full universe). IS folds = `build_v3_folds`
(2011–2020, 10 folds); per-trade net R from `apply_cost_model(...).breakdown.net_pnl / $500`.

### (1) Donchian-40 breakout long, full 28-pair universe — mean-NEGATIVE, tail INSUFFICIENT

| exit | fold mean ROI | folds neg | per-trade mean R | +2R-cap mean | top5%-rm mean |
|---|---|---|---|---|---|
| sl_only | −21.35% | 9/10 | −0.1294 | −0.2086 | −0.3216 |
| sl_plus_trailing_atr | −22.60% | 10/10 | −0.1301 | −0.1948 | −0.3091 |
| sl_plus_trailing_swing | −23.11% | 10/10 | −0.1358 | −0.2071 | −0.3216 |
| sl_partial_close_1r_runner_trail | −22.10% | 10/10 | −0.1274 | −0.1390 | −0.2432 |

- **Mean is NEGATIVE everywhere** (per-trade −0.13R; per-fold ≈ −22%, 9–10/10 folds negative). Fails
  pre-registered guard #1 outright — you cannot diversify a negative-mean book positive (§11, arcs 3000/01).
- **Tail-removal makes it MORE negative** (+2R cap and top-5%-removed both worsen the mean). This is the
  *inverse* of positive skew: the right tail is real (max single trade +12.6R) but **insufficient** to
  offset the frequent −1R losses (median trade −0.81R) + FundedNext costs.
- **Majors ≈ crosses** (both ≈ −0.13R) — the arc-2000 majors-only open thread is closed: the full 28-pair
  universe does NOT rescue it; crosses behave identically to majors here (unlike gap-fill, where cross≠major).
- **Loses to the null:** Donchian −0.127→−0.136R vs periodic being-long-anytime −0.106R — the breakout
  ENTRY adds *negative* value (re-confirms arc 2000's "trend entry ≈ random", here marginally worse).
- **NOT the long-vol complement:** 2015 (−33 to −37%) and 2018 (−12 to −13%) are among the *worst* folds —
  a breakout book gets chopped up in those years, it does not catch the vol-expansion. The hoped-for
  reversion-book complementarity is absent.

### (2) Shock-continuation (arc 3019, the corpus's strongest) re-measured w/ trailing + tail-removed, full universe

Most cells mean-negative. The only mean-positive cells were `shock3.0 short` (n≈369), which the
**pre-registered guard KILLS as tail-luck** — the exact case the guard exists for:

| cell | fold mean | folds neg | per-trade mean R | +2R-cap | top5%-rm | verdict by guard |
|---|---|---|---|---|---|---|
| shock3.0 short + trailing_atr | +0.98% | 5/10 | +0.0575 | **−0.0685** | −0.1610 | KILL (cap2<0) |
| shock3.0 short + partial_runner | +0.78% | 5/10 | +0.0464 | +0.0146 (31%) | **−0.0856** | KILL (top5%-rm<0) |

Both look like a +2015 PORTFOLIO candidate on the headline (mean-positive, +2015 +4–9%) — but the
positivity is carried entirely by the top ~5% of winners; cap or drop them and the mean goes ≤ 0. The
edge is also **majors-only** (crosses ≈ 0 / negative), independently re-confirming arc 3019's majors
finding and consistent with its OOS epoch-kill. Every Donchian/shock long cell, and every cross cell,
is mean-negative or tail-luck.

## Diagnosis — the right tail is GENERIC and INSUFFICIENT; positive-skew continuation is structurally dead on liquid FX OHLC

The operator's redirect was the correct question to ask, and the tail-AWARE metric answers it cleanly —
not by re-using a tail-blind lens, but by showing directly that:
1. A continuation/breakout book's per-trade **mean is negative** after take-the-loss + costs (Donchian),
   OR positive **only via the top ~5% of winners** (shock-short) — i.e. tail-luck, the precise failure
   the guard pre-registers against.
2. The right tail is a **generic property of being long FX vol** (arc 2000), not entry-selectable — the
   breakout loses to the being-long-anytime null. Conditioning harder (extreme 3-ATR shocks) only thins
   the sample until the apparent mean is 2–3 monster trades.
3. The long-vol-complement hope fails empirically: these books **bleed in 2015/2018**, the reversion
   book's death years — a chopped-up breakout book is not the long-vol sleeve the vehicle wall needs.

This is the conservation law made concrete (`frequency × edge ≈ const`): the rare large right-tail trends
do not pay for the frequent −1R losses + costs. The one payoff shape the shallow envelope had not directly
covered is now covered, under the operator's own mandated lens, across two canonical continuation
constructions (Donchian breakout + forward-confirmed shock), both directions, the full 28-pair universe,
the tail-preserving trailing exits, and a being-long null.

## Council — NOT convened (clean cheap-kill / objective pre-registered guard is the rigor)

§5d cheap-kill: Donchian mean deeply negative (no worthwhile ceiling); the only mean-positive cells
(shock3-short) are adjudicated KILL by the **objective pre-registered tail-removed rule** I committed to
applying verbatim — which is the operator's stand-in judgment for exactly this thin-tail case (and arc
3019 already spent shock-continuation's OOS → epoch-kill, so a full-WFO escalation would re-grind a
known-dead OOS on a config the IS guard already fails). Convening would re-derive the arc-2000/2012/3019
continuation verdict ("ritual not rigor", arc-1002 precedent). OOS NEVER touched.

## Verdict: KILL

No new component. The positive-skew / continuation thread — the operator's ONE open in-charter thread —
is **closed under its own mandated metric (mean + median-per-fold + tail-removed):** a continuation book
is mean-negative (Donchian, full universe, all trailing exits, loses to null) or mean-positive only via
the top-5% tail (shock-short, majors-only — the thin-tail trap the guard catches), and bleeds in
2015/2018 rather than complementing the reversion book. Deployable count = 0; components UNCHANGED
(4 PORTFOLIO: gap 1006 / me_long 1011 / fbr 1013 / me_short 1019).

## Lessons (candidate for LESSONS.md)

1. **Positive-skew continuation is dead on liquid-FX OHLC under the tail-AWARE lens.** A Donchian-breakout
   book across the full 28-pair universe with take-the-loss + tail-preserving trailing is per-trade
   **mean-negative** (−0.13R, 9–10/10 folds neg, majors ≈ crosses), **loses to a being-long-anytime null**,
   and the right tail (max +12.6R) is INSUFFICIENT — removing it makes the mean MORE negative. The fat
   tail is generic to being-long FX vol (arc 2000), not entry-selectable.
2. **The pre-registered tail-removed guard EARNED ITS KEEP first time out.** Forward-confirmed extreme-shock
   continuation (arc 3019's edge), re-measured short on the full universe under trailing, shows two
   mean-positive +2015 cells (+0.78–0.98%/yr) that would read as PORTFOLIO candidates on the headline —
   but the mean is carried entirely by the top ~5% of winners (top-5%-removed < 0), i.e. tail-luck, KILL.
   This is the thin-tail trap (arcs 2011/2063) the operator mandated the guard for; it is now a one-call
   BUILT tool (`tail_removed_expectancy`).
3. **The long-vol-sleeve hope is empirically false on OHLC continuation:** breakout/continuation books
   bleed in 2015/2018 (chopped up), they do not catch the vol-expansion — so they do not lift the combined
   book's Calmar / break the vehicle wall. A genuine long-vol diversifier likely needs a cross-asset
   trend universe (the NEEDS_ENABLEMENT #1 lever) or a continuous non-price state, not an in-charter FX
   price functional.

## Threads / handoff

- **Closed (under the mandated metric):** positive-skew continuation via Donchian breakout (full 28-pair
  universe, all trailing exits) and forward-confirmed shock continuation (both directions) — mean-negative
  or tail-luck, bleeds 2015/2018, loses to null. This was the ONE open in-charter thread the 2026-06-06
  compression named.
- **Not directly run (residual, low expected value):** a vol-regime-EXPANSION-conditioned breakout
  tail-select — but arc 2000's generic-tail finding + the null-loss here strongly predict it dies too
  (the tail is not entry-selectable). A fresh chat may run it as a final closer if desired.
- **Beyond charter (operator-gated):** the genuine long-vol / positive-skew diversifier is the
  cross-asset trend universe (`NEEDS_ENABLEMENT.md` #1; same shape, less-efficient universe, blocker =
  historical data) — the in-charter FX-OHLC version is now closed.

## Flags / Tooling

No canonical-core change (no FLAG). Carries the standing arc-3017 `risk_pct` percent-vs-fraction FLAG
(ran the LINEAR regime, risk 0.005, cap ~never binds — consistent with arcs 2063/3019). NEW BUILT tool:
`discovery/tools/tail_removed_expectancy.py` (`tail_removed_expectancy` + `TailRemovedExpectancy`) —
the reusable positive-skew honesty guard (registered). Reused EXPERIMENT tools: `DonchianBreakoutLongSignal`
/ `PeriodicLongSignal` (arc 2000), `ShockContinuationSignal` (arc 3019), arc-2063 winsorization arithmetic.
Drivers in `_disco2000_work/` (reproducible: `PYTHONPATH=. py _disco2000_work/arc2081_*.py`).
Data: 28 pairs, H4, `5ers_eet`, `histdata_root=C:\Users\panap\histdata_backup`,
`cache_root=C:/Users/panap/Documents/Forex-Backtester/data/cache`.
