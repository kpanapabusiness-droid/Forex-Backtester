# Arc 1028 — Q1: central-bank PEG / boundary-defense persistence (the long-shot)

> **Arc id:** 1028 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **KILL (cheap-kill at observation)** — the one clean defended-boundary in the corpus
> (SNB EURCHF 1.20 floor, 2011-09 .. 2015-01) does NOT yield a fadeable edge: in-regime near-floor-touch
> capture **0.526 ≈ the out-of-regime 0.525 coin-flip**, and the in-regime forward drift is **median −0.237
> ATR** (the +1.19 *mean* is a thin-tail artifact). A hard floor **PINS** price (no upward bounce to fade),
> it does not mean-revert it; the apparent vol-collapse "lift" (0.598) is **generic low-vol reversion, not
> the peg** (the detector fires in every year, not 2012–2014). And the regime is a single pair-epoch ending
> in a **−18.8% / ~182-ATR gap-through-stop** break (Jan 15 2015) — un-gateable by absence + un-hedgeable tail.
> **Idea source:** `DISCOVERY_DIRECTION.md` MENU item **Q1** (the divergent council seat's "new forced
> actor" — a persistent, non-reverting flow you JOIN, vs the corpus's reversions-of-overextension).

Scored by inspection of the canonical EURCHF H4 panel (real bid/ask, 5ers_eet) via the canonical honest
capture harness. No pool/engine — the observation is decisive (§5d). OOS never touched.

## (a) Log read + synthesis

(Carried from arc 1027, same chat/session — corpus is mature; path-B closed, 3021; lever = operator
gate-governance.) The strategist `DISCOVERY_DIRECTION.md` MENU's **Q1** is the one item that targets the
book's actual deficit — a **THICK** component (many trades/year) from a genuinely NEW mechanism: a
price-insensitive actor *defending a band* is a forced, infinite-balance-sheet, repeated counterparty
pinning price to one side. Every surviving corpus edge is a reversion-of-an-overextension; Q1 asks for the
inverse — a persistent flow you join / fade-the-boundary-of. The OHLC-detectable instance in the corpus is
the SNB EURCHF 1.20 floor (announced 2011-09-06, broken 2015-01-15) — in-IS, H4 cached, the textbook peg.

## (b) Idea + because

A defended floor (EURCHF 1.20) means a repeated buyer absorbs all selling at the level → fading touches of
the boundary (long near the floor) *should* be a high-win-rate, low-variance, THICK edge that appears
**with** the regime and vanishes outside it. **Falsifiable prediction (Q1):** in-regime near-floor fade
shows high capture / positive R; the SAME rule outside the regime is a coin-flip; the edge tracks the regime,
not price. **Falsifier:** if in-regime capture/drift ≈ the out-of-regime coin-flip, a defended boundary does
not create a fadeable edge (it pins, it doesn't bounce) → dead.

## (c)/(d) Observation → verdict (cheap-kill)

EURCHF H4. Established floor = causal `rolling-min(mid_low, 250).shift(1)`; near-floor touch = a bar whose
`mid_low` dipped within **0.5 ATR** of the floor. Honest +1R-before-SL capture + 24-bar fwd drift via the
canonical `observe_long_capture` restricted to near-floor bars; split by hindsight regime label and by a
causal bottom-decile-realized-vol detector.

**Mechanism test (hindsight regime label):**

| | n | capture | fwd_drift mean | fwd_drift **median** |
|---|---|---|---|---|
| in-regime (2011-09 .. 2015-01) | 346 | **0.526** | +1.194 | **−0.237** |
| out-of-regime | 1629 | 0.525 | +0.100 | −0.007 |

- **No asymmetry in the honest win-rate:** in-regime capture 0.526 ≈ out-of-regime 0.525 ≈ coin-flip. The
  defended floor confers NO +1R-before-SL edge.
- **The in-regime median drift is NEGATIVE (−0.237 ATR).** The +1.194 *mean* is a thin-tail artifact (a few
  large moves). Mechanistically: the SNB held 1.20 by **pinning** price *at* the floor (2012–2014 EURCHF sat
  in a ~1.5%/yr band glued to 1.2000), absorbing selling but **not pushing price up** — so the
  reversion-to-a-higher-level the fade needs never happened. A truncated distribution at a floor ≠ a bounce.

**Causal vol-collapse detector (bottom-decile realized vol):** near-floor × low-vol capture 0.598 vs 0.509 —
but the detector fires in **every year** (2010:273, 2011:423, 2013:313, … 2022:395, 2025:226), NOT
concentrated in the 2012–2014 floor regime. So the "lift" is **generic low-vol mean-reversion**, which is
**closed ground** (vol-level / regime conditioning, all dead — LESSONS), not the peg mechanism. (Consistent
with thin-tail: near-floor low-vol dips in quiet markets bounce slightly, every year, sub-cost.)

**Un-gateable + un-hedgeable (decisive even if the fade had worked):**
- The defended-boundary regime exists on **one pair for ~3 years** (EURCHF 2012–2014). It cannot populate the
  per-year folds 2010/2011/2015–2020 → fails all-folds-positive **by absence of regime**, not by being wrong
  (exactly the shock-continuation epoch-dependence death, arc 3019, but worse — one pair).
- The break is an **un-hedgeable fat left tail:** pre-break 1.2010 → intraday low 0.9751 = **−18.8% gap =
  ~182 ATR through any stop** (Jan 15 2015). The −1R invariant caps the *backtest* per-trade, but live a
  long-the-floor position gaps ~150 R; across a book it blows the daily-DD cap. Cost in R was NOT the killer
  here (in-regime ATR ~12 pips, spread ~1 bp → ~0.048 R) — the killer is no-edge + the tail.

## Final verdict — KILL (cheap-kill at observation)

A defended boundary does not create a fadeable edge — it **pins** price (coin-flip capture, negative median
drift in-regime), and its inevitable **break** is an un-hedgeable gap-through-stop tail. The "new forced
actor / persistent non-reverting boundary" idea is dead on the corpus's one clean instance. §5f does not bite
(falsified at observation, no non-coin-flip entry). No engine / null / council spent; OOS never touched.
Components UNCHANGED (all 4 PORTFOLIO); the corpus's lever remains the operator gate-governance call.

## Lessons (candidate for LESSONS.md compression)

1. **A defended boundary PINS, it does not BOUNCE — there is no fade edge.** On the SNB EURCHF 1.20 floor
   (the textbook peg, in-IS), near-floor-touch honest capture is 0.526 ≈ the out-of-regime 0.525 coin-flip
   and the in-regime median forward drift is **negative** (−0.237 ATR). A hard floor absorbs selling at the
   level without pushing price up → a truncated distribution, not a mean-reversion. The "join a persistent
   forced actor" inverse-of-reversion idea (the council's divergent seat) is dead at observation.
2. **Peg-defense is un-gateable AND un-hedgeable** even where it exists: a single pair-epoch (one regime,
   ~3 years) can't populate the per-year folds (absence-of-regime FAIL, cf. arc 3019), and the break is a
   gap-through-stop fat tail (EURCHF −18.8% / ~182 ATR in one session) the −1R invariant flatters in
   backtest but cannot hedge live. A regime-flagged probe at most, never a normally-sized component.
3. **"Mean-vs-median" + "detector-fires-every-year" are the two cheap tells that re-killed it.** The +1.19
   in-regime *mean* drift looked promising; the −0.237 *median* exposed it as thin-tail (the recurring 2011/
   3012 tell). The 0.598 vol-collapse capture looked like a regime lift; its every-year firing exposed it as
   generic low-vol reversion (closed ground). Always check median vs mean and the detector's year-spread.

## Threads / what didn't help

- **Closed:** Q1 (central-bank peg / boundary-defense fade). The corpus's surviving mechanisms remain
  reversions-of-overextension; the join-a-persistent-flow inverse has no tradeable instance in the corpus's
  pairs (EURCHF being the only clean defended band, and it pins + breaks).
- **Operative state unchanged:** lever = operator gate-governance (path-A; arcs 2019/3021). Strategist MENU
  remaining: L1 (triangulation second moment — cost-walled, low-EV), O1 (inelasticity-state — fold-resolution
  attack, but path-B closed). The `explore-now` frontier is now nearly exhausted (M1 1027, Q1 1028 closed).

## Flags (code NOT merged)

None (engine/canonical untouched; reused the canonical `observe_long_capture`). The floor / vol-collapse
detectors are arc-specific and remain in the driver (not registered — not proven reusable on a KILL).
Driver: `discovery/_disco1_work/arc1028_q1_peg_defense.py`.

## Reproduction

`Panel.from_pairs(["EURCHF"], "H4", histdata_root=C:\Users\panap\histdata_backup, cache_root=data/cache,
boundary_convention="5ers_eet")`; floor = `mid_low.rolling(250).min().shift(1)`; near-floor = `mid_low`
within 0.5 ATR of floor; honest capture via `observe_long_capture(restrict=near_floor)`; regime label
2011-09-07 .. 2015-01-14; vol-collapse = bottom-decile rolling realized vol. Driver:
`discovery/_disco1_work/arc1028_q1_peg_defense.py`.
