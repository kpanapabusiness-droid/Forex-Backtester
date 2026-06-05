# arc 2012 — DEEP multi-factor trend-CONTINUATION long (1013's forward-confirm property applied to continuation) — KILL

- **Chat / range:** 2000s (2000–2999)
- **Arc id:** 2012
- **Date:** 2026-06-05
- **Disposition:** **KILL** (cheap-kill at observation — structure-control INVERSION + sub-0.50 capture)
- **Council:** none (a structure-control inversion is a falsification, not a tuning fork — arc-3003/1014
  discipline; capture<0.50 ⇒ no reachable ceiling, the oracle-best-cluster there is the Arc-0 hindsight trap)
- **Tooling:** reused BUILT `observe_long_capture(direction="long")`; conditioning is a one-off scratch
  observer. No canonical-core change, no new BUILT tool. Driver:
  `_disco2000_work/arc2012_observe_trend_continuation.py`.

---

## Idea + why (the one untested face of the 1013 template)

The portfolio route (arcs 2006/2008/3009) is **blocked on 2018** (strong-USD trend year): all three
PORTFOLIO components (gap-fill 1006, month-end 1011, failed-breakdown-reclaim 1013) are fade/reversion
edges → tail-correlated, all bleed trend years. 2008 proved **0/5151** convex weightings clear
all-folds-positive. The missing 4th leg must be **trend / 2018-POSITIVE** — the one flavor that profits
when fades bleed. The short/relative-value routes to that leg are exhausted: structural shorts are dead
(1014/2009/2011 — the swing-level sweep is a reversal structure with no tradeable short mirror),
trend-following short is dead (3010 — downtrend regime INVERTS), relative-value is cost-bound on majors
(2010). The remaining open face (dispatch item d) is a **deep multi-factor directional LONG** in the
arc-1013 template.

Every prior trend cut was **shallow / single-condition** (Donchian/SMA: arcs 0, 1000–1004, 2000,
3000–3003) → coin-flip, and 3003 sharpened it: strong-trend regimes REVERT. BUT arc 1013 proved a
**DEEP conjunction (structure × sequence × magnitude) extracts a clean edge where the shallow version is
dead** — and its load-bearing property was being **forward-confirming**: the reclaim enters AFTER the
adverse low, so the i+1 entry is not buying into adverse continuation. Arc 0's pullback-long failed
partly because it bought INTO the dip (not forward-confirmed). **Nobody has applied 1013's forward-confirm
property to trend CONTINUATION** — which is intrinsically trend-positive, a LONG (no short-mirror death),
and in-scope/in-range.

**Hypothesis:** a deep continuation conjunction — established uptrend × a shallow controlled pullback
that holds the higher-low × a strong prior impulse × a **forward-confirming resume** (close reclaims the
prior bar's high = resumption restarted) — extracts the trend edge the shallow versions cannot, and is
net-positive in 2018 where the three fades bleed.

## Construction (ex-ante, shift1; entry i+1; H4 USD majors, IS 2010–2020)

- **structure:** `close>SMA200 & SMA50>SMA200` (established uptrend)
- **sequence:** a 5-bar local pullback that HOLDS above the prior 40-bar swing low (higher-low intact),
  now `dip_depth = (close−pullback_low)/ATR ≥ 0.5` (a real dip, resuming)
- **magnitude:** `impulse = (SMA50−SMA50[−10])/ATR ≥ thr` (strong prior up-leg)
- **resume (forward-confirm):** `close > high[i−1]` (bullish reclaim of prior-bar high)
- Honest +1R-before-SL LONG capture + i+1 forward drift via direction-aware `observe_long_capture`.

## What happened — FALSIFIED at observation (decisive, no engine needed)

Base (all 121,430 IS bars, long lens): cap 0.4860, drift −0.0474 ATR.

- **Q1 — the deep continuation cell does NOT clear 0.50, and impulse is anti-predictive.** cap
  0.4787–0.4865 (≤ base, all < 0.50); a STRONGER impulse gate makes it WORSE (impulse≥0.5 → cap 0.4787,
  drift −0.105; impulse≥1.0 → 0.4780). The more "established" the trend, the worse — a trade-level echo of
  arc-3003's regime inversion.
- **Q2 — the forward-confirm RESUME does NOT rescue continuation.** RESUME-confirmed drift −0.105 vs
  into-the-dip −0.156 — **both negative, both sub-0.50 capture.** The 1013 property fails to generalize.
  **Mechanism (the new lesson):** forward-confirming a *reversal* (1013) enters BEFORE the move starts (the
  up-move begins at the reclaim) → works; forward-confirming a *continuation* enters AFTER the bounce has
  already moved (late) → mean-reverts. The asymmetry is WHERE the i+1 entry lands relative to the move.
- **Q3 — STRUCTURE CONTROL is ANTI-load-bearing (decisive, the inverse of 1013).** FULL deep conjunction
  cap 0.4787 / drift −0.105 is **WORSE than the generic** resume-in-uptrend cap 0.4956 / drift +0.0070.
  Adding the deep pullback/higher-low/impulse structure pulls a near-coin-flip generic bounce NEGATIVE.
  1013's discriminator (AT-structure ≫ elsewhere) **inverts** here: the trend structure HURTS. A structure
  whose presence helps the OPPOSITE outcome is dead (the arc-2009 rule).
- **Q4 — 1/7 pairs positive** (only USDJPY +0.094, the carry/Abenomics pair); 6/7 negative — below the
  3–4/7 noise floor, i.e. genuinely negative, not noise.
- **Q5 — 2018 drift = −0.233 (NEGATIVE).** Even this continuation construction bleeds in 2018; per-year is
  a coin-flip (4/11 positive, range −1.33 [2019] to +0.81 [2020]). It does **not** provide the
  trend-positive leg — the portfolio's 2018 wall stands.

**Verdict: KILL (cheap-kill at observation).** The deep continuation conjunction is sub-0.50, the
forward-confirm property is reversal-specific (does not rescue continuation), the trend structure is
anti-load-bearing (structure-control inversion), and 2018 is negative.

## Threads / lessons

1. **arc 1013's forward-confirm property is REVERSAL-SPECIFIC, not a general edge.** It works because the
   reclaim enters *before* the up-move begins; applied to continuation the same trigger enters *after* the
   bounce (late) and reverts. Forward-confirming is necessary-not-sufficient — it only helps when the
   confirmed event is the START of the bet's move, not its tail. This is why 1013 has no continuation sibling.
2. **Deep multi-factor structure does NOT rescue trend continuation** — the trade-level completion of
   arc-3003 (strong-trend regimes revert): the cleaner/stronger the trend structure (impulse gate ↑,
   full-conjunction vs generic), the MORE negative. Structure helps reversals (1013), HURTS continuation.
3. **The portfolio's 2018-positive leg is NOT a structural-continuation long either** (2018 drift −0.233)
   — adding to the already-closed structural-short (1014/2009/2011) and trend-short (3010) routes. The
   trend/2018-positive flavor is not expressible as a price-structure directional bet in EITHER direction
   → the arc-3004 escalation is reinforced from the long-continuation angle (every directional face now
   mapped: shallow long/short dead, deep reversal long = the 3 fades, deep continuation long dead here,
   structural/trend short dead).
4. **Surviving frontier (unchanged):** the FLOW-event up-gap weekend SHORT (1000s' arc-1015 lane, acc
   0.64 — the one place direction beats 0.50, but a fade → likely inherits the 2018 tail). The
   trend-continuation 4th leg the portfolio needs is, with this arc, structurally unfindable within the
   FX price-direction apparatus — the route is escalation-bound short of a genuinely non-price-direction
   construction.

## FLAGS (code not merged)

None. No canonical-core change; reused the BUILT direction-aware `observe_long_capture`. Conditioning is
a one-off scratch observer (no reusable tool warranted). Driver:
`_disco2000_work/arc2012_observe_trend_continuation.py` (reproducible from this doc).
