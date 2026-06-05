# Arc 3006 — Multi-TF Breakout-Retest (resistance→support flip), long-only

> **Arc id:** 3006 · **Chat:** 3000–3999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **FAIL (cheap-kill at observation)** — the retest-and-hold of a freshly broken
> prior-swing-high confers **no** capturable long edge on H4 majors. Conditional honest +1R-before-SL
> capture is ≤ the unconditional base in **every** parametrization, and forward 24-bar drift is **negative
> in every cell** (−0.35 to +0.00 ATR). The setup catches **fading breakouts**, not defended supports —
> the textbook "broken resistance flips to support" mechanism is falsified on liquid FX.
> **Idea source:** the one long-only lane arc 1008 flagged as still untested — *multi-timeframe STRUCTURE as
> a setup (not a regime filter)* — and the arc-3004-escalation frontier item "structural mechanisms available
> NOW (no shorts needed)." Closing it completes the structural-long coverage and reinforces the escalation.

Scored by the canonical observation harness (`discovery/tools/observe_long_capture.py`, honest take-the-loss
capture + gross forward drift, real bid/ask, H4 5ers_eet). No pool / engine / council needed — the entry is
coin-flip-or-worse at observation (below the 0.4877 base, negative drift), so the §5f exit-sweep does not bite
(it bites only for entries that beat the null or show gross **positive** drift). Like arcs 1001, 1002, 3005, 1008.

## (a) Log read + synthesis

Pulled `origin/main` (at `8f4047a`, arc 1008). Read DISCOVERY_PROTOCOL, the full two-tier DISCOVERY_LOG (19
arcs: 0, 1000–1008, 2000–2003, 3000–3005), LESSONS.md, TOOL_REGISTRY.md. STOP absent. Highest id in range
3000–3999 = 3005 → resume at 3006. Fresh eyes, honest-era only.

State: shallow directional prediction is comprehensively closed (momentum / breakout / mean-reversion / trend,
H1/H4/D1, 28 pairs, capture AND drift lenses, all exits + SL multiples, stop removed — arc 3004). Regime
conditioning dead across 3 measures (strong trends invert). Volume = magnitude not direction. TOM drift real
but sub-cost; gotobi absent; triangulation ≈0. The one live edge is the weekend gap-fill long on JPY crosses
(arc 1006, PORTFOLIO). Arc 3004 escalation: the apparatus can only bet direction → coin-flip; deployable edges
need a structural unlock. **My pre-shorts lane:** a 2nd net-positive long-only component, or a novel structural
mechanism with a *because*. **Arc 1008 named the one untested long-only lane: multi-TF structure as a setup.**
That is this arc.

## (b) Idea + because (observe before believing)

**Hypothesis.** When price clears a well-established prior swing-high (resistance) and then **retests and holds**
that level, the broken level flips to **support**: trapped breakout-sellers cover and fresh buyers defend = an
order-flow **memory** anchored to a real structural level. A long entered on the holding retest should show
forward upside **above** the 0.4877 coin-flip base — unlike arc-0 SMA-pullback (no level memory) or the generic
breakout (which fades, arcs 1001/2000). This is a **two-stage** structural setup (break THEN retest-hold),
mechanistically distinct from every prior single-condition directional cut, and it lives in the explicitly-open
multi-TF-structure lane.

**Operationalization (H4, 7 USD majors, IS 2010–2020).** Prior structural high = `mid_high.rolling(L).max().shift(1)`.
Fresh breakout at bar *b*: `close[b] > level` and `close[b-1] ≤ level[b-1]`. Watch ≤ *K*=20 bars for a
**retest-hold**: a bar *t* with `low[t] ≤ level + tol·ATR` (dipped to the old resistance) **and** `close[t] ≥ level`
(closed back above → held); if `close < level − tol·ATR` first, the level is decisively lost → failed setup,
abandon. Entry = next-bar open (honest i+1). Conditioning mask joined onto the canonical per-bar observation.

## (c) Observation → verdict

Primary (L=120, tol=0.25·ATR, "hold" = close ≥ level):

| group | n | honest capture | 24-bar drift (ATR) | lift vs base |
|---|---|---|---|---|
| **unconditional base** | 121,430 | 0.4860 | −0.0474 | — |
| **retest-hold** | 1,200 | **0.4733** | **−0.2353** | **−0.0127** |

Per-pair retest capture / drift: AUDUSD 0.497 / −0.379 · EURUSD 0.417 / −0.573 · GBPUSD 0.424 / −0.363 ·
NZDUSD 0.489 / −0.193 · USDCAD 0.467 / −0.048 · USDCHF 0.465 / +0.021 · USDJPY 0.535 / −0.161. Only USDJPY's
capture nudges above base (0.535) but its drift is **negative** (−0.161) — hindsight-noise on a 1-of-7 cherry,
not real upside. Frequency ~15.6 fires/pair/year (ample, not a thinness artifact).

**Robustness sweep (3 L × 3 tol × 2 hold-strictness = 18 cells), conditional capture / drift vs base:**

- Capture lift ∈ [−0.0246, **+0.0012**] — every cell ≤ base except one (+0.0012, the loosest L=250 tol=0.40
  cell, noise).
- Forward drift ∈ [−0.3514, **+0.0028**] — **negative in 17/18 cells** (the lone +0.0028 is the thinnest
  L=250-strict cell, noise).
- **Monotone in L:** longer (more "significant") levels give *less negative* drift (L=60 ≈ −0.30 → L=250 ≈ −0.05)
  — i.e. genuinely significant structural levels only **converge the retest TO the coin-flip base, never above it.**

**The mechanism is falsified, parameter-robustly.** The retest-hold as defined catches **fading breakouts** (the
level is retested precisely because the breakout stalled), and stalled H4-major breakouts then drift down —
re-deriving arc 0/1000's "breakouts fade" from the structural-level angle. "Broken resistance → support" memory
does not survive as a capturable long on liquid majors.

## Final verdict — FAIL (cheap-kill at observation)

Multi-TF breakout-retest confers no long edge on H4 majors; dry-to-negative across L / band / hold-strictness.
This closes the multi-TF-structure long lane that arc 1008 flagged as the last untested long-only direction.
**KILL** disposition (entry below the coin-flip base, gross drift negative — not even net-positive, so not a
PORTFOLIO component; §11).

## Lessons (candidate for LESSONS.md compression)

1. **Structural-level memory is arbitraged on liquid H4 majors.** A retest-and-hold of a freshly broken
   prior-swing-high gives no long edge — capture ≤ base and forward drift negative in every parametrization; the
   setup selects *fading* breakouts. The "former resistance flips to support" textbook idea is dead here.
2. **The monotone-in-L signature is the tell:** more-significant levels don't *help*, they merely stop hurting —
   the retest converges TO coin-flip as L grows, never above it. A directional structural bet asymptotes to
   ~0.49 no matter how "important" the level. Re-confirms the directional coin-flip prior from a 2-stage
   structural angle (the 20th arc to land there).
3. **The one untested long-only lane (multi-TF structure, arc-1008 flag) is now closed** — leaving, pre-shorts,
   only portfolio construction (a 2nd net-positive component to combine with the gap-fill, still not found) and
   the standing arc-3004 escalation (shorts / second-leg / tighter-cost unlock = operator decision).

## Threads / what didn't help

- **Closed:** breakout-retest / resistance→support-flip long (dry-to-negative, parameter-robust). A stricter
  "hold" (bullish + upper-half close) did not help; longer lookbacks only neutralize, never lift.
- **Did NOT pursue exits/pool/engine:** the entry is coin-flip-or-worse at observation, so the §5f
  exit-as-hyperparameter sweep does not apply (reserved for entries that beat the null or show +gross drift).
- **Operative state unchanged:** no 2nd net-positive long-only component yet; the arc-3004 escalation stands. The
  next high-leverage move remains the operator's structural unlock (shorts / second leg), per
  `ESCALATION_apparatus_capability.md` and the 2001/2003 FLAG-1.

## Flags (code NOT merged)

None. No canonical-core change; no new BUILT tool (the retest mask is a one-off conditioning helper kept in
scratch, like arc 3005's triangulation observer). Reused BUILT `observe_long_capture`.

## Reproduction

`Panel.from_pairs([AUDUSD EURUSD GBPUSD NZDUSD USDCAD USDCHF USDJPY], "H4",
histdata_root=C:\Users\panap\histdata_backup, cache_root=data/cache, boundary_convention="5ers_eet")`;
`observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=24, direction="long")`, IS 2010–2020; join the
retest mask (L=120, K=20, tol=0.25·ATR; fresh `close>rolling(L).max().shift(1)` break, retest-hold =
`low≤level+tol·ATR AND close≥level` within K bars). Drivers: `_disco3_work/arc3006_observe.py` (primary),
`_disco3_work/arc3006_sweep.py` (18-cell robustness).
