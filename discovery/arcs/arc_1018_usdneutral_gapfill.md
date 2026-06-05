# arc 1018 — Weekend gap-FILL long on USD-NEUTRAL crosses

**Chat:** 1000s · **Date:** 2026-06-05 · **Disposition:** KILL · **passed:** N

## Idea + why (target the (+2015,+2018) leg via DECORRELATION, not USD-prediction)
The 3-way book (1015) is blocked by 2015 & 2018; the missing 4th leg must be net-positive AND positive in
BOTH. Nine routes that tried to *predict* the strong-USD/risk-off regime failed (structure/trend/vol/flow
shorts, deep-continuation long, relative-value, carry-unwind 1017). A genuinely different principle: get the
(+,+) fold profile by **decorrelation from the USD regime** — apply the PROVEN net-positive gap-fill flow
mechanism (arc 1006, JPY crosses) to **USD-NEUTRAL crosses** (EURGBP, AUDNZD, GBPCHF, AUDCAD, NZDCAD), whose
weekend gaps are driven by intra-bloc news, not USD direction. If the fill edge survives there (it's a flow
fade, not a directional bet) and the fold profile is orthogonal to the USD regime, it could be +2015 AND
+2018 where the USD/JPY fades bleed. Not closed ground (gap-fill on USD-neutral crosses is untested; the
arc-3000 closure was a directional reversion *capture* cut, not the weekend-gap flow event).

## What happened — FALSIFIED at observation
Weekend down-gap (≥0.5·ATR) fill long, i+1 entry, 24-bar fill drift, H4 IS, 5 USD-neutral crosses (BUILT
`observe_long_capture` + the 1006 gap mask). n=463 (thin, ~8/pair/yr):
- **No robust fill edge:** cap **0.4881 (coin-flip ≈ the 0.49 base)**, fill drift mean +0.078 ATR but
  **median 0.000** → the positive mean is a thin-tail artifact (the arc-2011/1010 tell). EURGBP — the
  cleanest USD-neutral, where the decorrelation thesis is strongest — drift **−0.014 ≈ 0** (cap 0.512).
  Per-pair mixed (AUDCAD −0.276, NZDCAD +0.152, AUDNZD +0.154, GBPCHF +0.456); median-zero pooled.
- **Acceptance test FAILS on the binding folds:** 2015 drift +0.390 (cap 0.525, barely >coin-flip) but
  **2018 drift −0.085 (cap 0.500, NEGATIVE)**. USD-neutrality did NOT deliver +2018; it delivered noise.
  Per-year is pure noise (2012 +0.92 .. 2019 −1.34).

## Verdict: KILL (observation cheap-kill, §5d)
Cap ≈ base (coin-flip) + median fill-drift 0.000 + 2018-negative → no robust edge, acceptance test fails.
§5f doesn't bite (sub-0.50/median-zero, no gross drift to bank); no pool/engine/null/council spent (arc
1010/1016/2011 discipline — a coin-flip-capture/median-zero/2018-negative cell collapses SL-honest). OOS
untouched.

## Threads / lessons
1. **The weekend gap-fill edge is JPY-cross-SPECIFIC (carry/USD-related), NOT a universal weekend-gap
   property.** On USD-neutral crosses the fill drift is coin-flip / median-zero — those gaps are
   efficient/random-walk with no systematic reversion. This sharpens arc 1006/1009: 1006's edge (and its
   null's +0.33% JPY-basket drift) both ride JPY-cross carry/USD structure; remove that exposure and the
   fill edge vanishes with it.
2. **Decorrelation-from-USD is NOT a free path to the (+2015,+2018) leg.** Removing the USD/carry exposure
   that makes 2015/2018 negative *also* removes the edge — the edge and the tail are the SAME exposure. A
   USD-neutral instrument is decorrelated from the blockers but has no edge to contribute (net-zero ⇒ can't
   diversify a book positive, §11 / arcs 3000/3001). This closes the "find a USD-neutral net-positive fade"
   sub-route: net-positive FX fades are USD/JPY/carry-driven (hence 2015/2018-exposed), so a USD-neutral one
   is net-zero. 10th dead route to the 2018 leg.
3. **The 3-way book (1015) remains the strongest non-deployable result.** The portfolio's 4th-leg spec
   (net-positive AND +2015 AND +2018) is now empirically shown unreachable by USD-prediction (9 routes) AND
   USD-decorrelation (this arc) — converging hard on the arc-3004 escalation (the apparatus's net-positive
   edges are USD/carry-exposed calm-year fades, structurally 2015/2018-tailed).

## Tooling
No new BUILT tool — reused BUILT `observe_long_capture` (long lens, `restrict` mask) + the arc-1006 weekly-
gap mask logic (one-off scratch). CALLED canonical `Panel.from_pairs`. Scoring N/A (observation cheap-kill).

## FLAGS (code not merged): none. Driver `_disco_work/arc1018_observe_usdneutral_gapfill.py` (reproducible).
