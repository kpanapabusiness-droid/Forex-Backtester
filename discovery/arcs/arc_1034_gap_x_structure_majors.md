# arc 1034 — Weekend down-gap × structural-support confluence on USD majors

**Chat:** 1000s · **Date:** 2026-06-06 · **Verdict:** FAIL (cheap-kill at observation) → **KILL**
**Disposition:** KILL · **passed:** N

> The plain weekend gap-fill LONG fails on USD majors (arc 2001: mean-neg, adverse continuation MAE
> −1.1R) but works on JPY crosses (1006); separately, fbr (1013) proved USD-major swing-lows are
> *defended structural pivots*. **Hypothesis:** a weekend down-gap that opens INTO/THROUGH a prior
> swing-low (support) combines forced weekend-repositioning flow with a defended level → the support
> arrests the adverse continuation that kills the plain-majors gap → a structurally-distinct candidate
> component (the arc-1013 conjunction template, structure × event, on a genuinely new combination).
> **FALSIFIED at obs: the conjunction does NOT rescue the majors gap.** The coherent cell — gap sweeps
> the swing low AND reclaims (the long-reversion setup) — is capture **0.4667 < 0.50** (coin-flip),
> per-pair MIXED (0.0–0.71), thin (n=30 ~3/yr), median drift ≈0. The only cell that *looks* positive
> (full through-gap 0.5513) is a label artifact of the 62% NON-reclaim falling-knife continuations
> (buying a confirmed breakdown = arc-2007 "worse than null under take-the-loss"). **A weekend gap
> cannot substitute for fbr's live intrabar rejection wick** — the gap sweeps the level over the closed
> market, removing exactly the real-time absorption information fbr keys on.

---

## Log reading (step a — FRESH EYES, honest-era only; pulled main, no STOP)

Fresh chat, resumed 1000s at the range floor: highest 1000s id in the ledger was **1033** → I open
**1034**. Read the full Tier-1 ledger (~100 arcs across the three chats) + recent Tier-2 + LESSONS +
TOOL_REGISTRY. State synthesis that matters for the frontier:

- **Four net-positive PORTFOLIO components:** gap-fill 1006 (JPY-cross H4), month-end-long `me` 1011
  (USD-major D1), failed-breakdown-reclaim `fbr` 1013 (USD-major H4, the STRONGEST: IS +1.85% 9/10),
  month-end-SHORT `me_short` 1019 (USD-major D1, first robust +2018).
- **The 4-way book (arc 1020) is mean-positive (+0.59%/yr RP, t=2.66, P(mean<0)=0.004), cost-robust
  (break-even κ=3.32), temporally stable, ~3 independent bets, shallow DD (1.59%) — but NOT
  all-folds-positive**, blocked by within-noise 2015/2016 dips. Risk geometry (arc 1033): low Calmar
  (~0.24–0.36), ~98% of decade underwater — the deployment weakness is duration + risk-adjusted quality,
  not depth/crash.
- **The edge-hunt is structurally closed:** path-B densification proven impossible (3021: a shared USD
  factor ρ≈+0.12 floors P(AFP) at any N); leg-hunt closed (2022); explore-now MENU exhausted
  (M1/O1/L1/Q1/G1/S1 — triangulation 3005/1027/2023/1031/2028, inelasticity 1029/1030/2027, session
  3016/1026, fix-flows 2025/3008/2026); shorts revive nothing symmetric (1014/2009/2011/3010/1016/2013);
  the 2018 trend-leg unfound across ~15 routes. The repeatedly-stated lever = **operator path-A**
  (gate-governance call), quantified in 1032 (coarsening the calendar gate doesn't work; path-A = adopt
  a mean/pooled/CI gate).
- **Closed ground:** single-condition shallow directional (long OR short) on liquid FX is dead; carry is
  OFF (FundedNext swap-free). Edges are universe-SPECIFIC: gap-fill is JPY-cross-specific (1018,
  2031 confirmed structural edges don't transfer off USD majors either).

**Why I still opened a fresh-mechanism arc rather than another book diagnostic.** The last three arcs
(1032/1033 1000s, 2031 2000s) all concluded "no new mechanism → diagnostic." The arc-3004 council
warned that *"the apparatus is incapable" is a seductive, search-ending conclusion*. So fresh eyes:
(1) I first verified arc-1033's one named-untouched angle (non-FX commodity→commodity-currency lead-lag)
is genuinely **data-gated** — the 65 GB histdata backup is FX-pairs-only (28 pairs, no XAU/oil/index),
confirmed by directory listing. That frontier stays closed. (2) Within OHLC-FX, the one genuinely
untested *conjunction* I could find with a real *because* is gap × structure — never combined.

## Idea + why (a documented *because*)

Two settled corpus facts point at an untested seam:
1. The plain weekend gap-fill is mean-NEGATIVE on USD majors (arc 2001 — honest i+1 lands in adverse
   continuation, MAE −1.1R) yet mean-POSITIVE on JPY crosses (1006). The gap *alone* is not enough on
   majors.
2. fbr (1013) proved USD-major swing-lows are *defended structural pivots* with dense resting stop
   liquidity — the corpus's only clean directional structural edge (structure-control load-bearing).

**Hypothesis:** a weekend down-gap that opens INTO or THROUGH a prior K=40 swing-low is a CONJUNCTION of
forced weekend-repositioning flow AND a defended structural support. The level should arrest the adverse
continuation that kills the plain-majors gap, converting the negative single-factor base into a reversion
edge — a structurally-distinct, plausibly-decorrelated candidate component on the universe where the
single-factor gap fails. This is the arc-1013 multi-factor template (structure × event), genuinely
untested (no arc conjoins gap + swing-low). Falsifier: gap-at-support honest +1R capture on majors stays
≤ the plain base / ≤ a far-from-support structure control, OR is coin-flip / pair-mixed / thin.

## Observation (cheap-kill — H4, IS 2010-2020, canonical `observe_long_capture`, no engine)

Setup: weekend down-gaps (gap_atr ≤ −0.5, the 1006 threshold), bucketed by
`dist_supp = (open_mid − priorK_swing_low)/ATR` — `through` (≤0, gapped through/swept), `into`
(0–0.5, opens just above support), `near` (0.5–1.5), `far` (>1.5, structure control).

**USD majors (n=503 gaps):**
- Plain gap base: cap **0.4732** (sub-0.50, reproduces the arc-2001 adverse majors gap), drift +0.150 /
  median +0.320 (the +drift-inside-sub-0.50-capture signature that 3004/3012 warn collapses SL-honest).
- By bucket: `through` cap **0.5513** (drift +0.70) · `into` 0.4615 · `near` 0.4691 · `far` 0.4555.
  Only the through-bucket clears 0.50.

The through-bucket looked promising (gap-driven liquidity sweep, the gap-trigger analog of fbr;
fbr-overlap = **0.00** → genuinely distinct from fbr, not re-tagging). **But interrogating it kills it:**
- Only **38%** of through-gaps RECLAIM (close back above the swept swing low). The other 62% are
  CONFIRMED breakdowns that keep falling.
- `through + RECLAIM` (the coherent long-reversion setup) = cap **0.4667 < 0.50**, n=30 (~3/yr), drift
  mean +0.55 but **median +0.107 ≈ 0** (thin-tail), per-pair MIXED (USDCHF 0.0, EURUSD 0.333, USDCAD
  0.333, AUDUSD 0.400, GBPUSD 0.500, USDJPY 0.714, NZDUSD 0.667 on n=2–7), per-year noise (2011 +4.6 /
  2020 −3.4 on n=1–5 folds = regime-luck).
- The full-bucket 0.5513 is therefore a **label artifact** of the non-reclaim falling-knife
  continuations: the oversold dead-cat bounce reaches +1R *then* continues down (capture is blind past
  +1R), and a LONG into a confirmed breakdown is buying structural weakness = arc-2007's "worse than
  null under take-the-loss." Not a tradeable reversion.

**JPY crosses (n=253, reference):** plain base cap 0.5455 (reproduces the working 1006 gap-fill); the
structure split is FLAT (`through` 0.5769 ≈ `far` 0.5368; confluence 0.5467 vs far 0.5368) — structure
adds nothing where the plain flow edge already works.

## Diagnosis (the *why*)

A weekend gap **cannot substitute for fbr's live intrabar rejection wick.** fbr's edge is a swing low
that is pierced *and reclaimed within one live bar* with a deep lower shadow — the wick is the visible,
real-time evidence that resting bids ABSORBED the stop-run (genuine support holding). A weekend gap
sweeps the level over the **closed** market (48 h, no live trading) → there is no rejection wick, no
absorption signal. By the weekly-open bar's close the setup is one of two already-decided states, both
backward-confirming (the established arc-1014/2009/1016 mode):
- still below the swept level (no reclaim) → a CONFIRMED breakdown continuing down → long = falling knife;
- already back above (reclaim) → the bounce already happened over the weekend → i+1 entry lands at the
  local low → reverts/gives back (coin-flip).

This sharpens the corpus's existing separation: the gap-fill (1006) is a pure **FLOW** edge (reversion of
forced weekend repositioning; works on JPY crosses via the carry/USD-basket channel), NOT a structural
one; fbr (1013) is a **STRUCTURAL** edge requiring a live rejection. The two mechanisms do **not
compose** — bolting structure onto the gap doesn't import fbr's edge, because the gap destroys exactly
the intrabar absorption information fbr depends on.

## Verdict / disposition

**KILL (obs cheap-kill).** The coherent conjunction cell is coin-flip + pair-mixed + thin + noise →
§5d cheap-kill, no engine, no null, no council (the same handling as 1014/2009/1016/2029 for the same
backward-confirming structure; arc 2001 already engine-killed the plain majors gap). §5f does not bite
(the only long-coherent cell is sub-0.50 with median drift ≈0 — not a non-coin-flip / gross-drift entry).
**OOS never touched.** Components UNCHANGED (all 4 PORTFOLIO). Operative lever unchanged = operator
path-A.

## Threads

- The "gap-driven sweep" idea is closed on the LONG/reclaim side. By symmetry the up-gap-through-swing-high
  short would inherit the same backward-confirming death (the up-gap short is already KILL, 1016/2013) —
  not worth a fresh arc.
- Reinforces that the only forward-confirming structural reversal in the corpus is fbr's live wick;
  any trigger that resolves the rejection over a closed market (gap) or after the fact loses the edge.

## NEW lesson

A weekend gap cannot substitute for fbr's live intrabar rejection wick: the gap sweeps the level over the
closed market, so by the weekly-open bar's close the setup is either still-falling (confirmed breakdown,
no reclaim) or already-reclaimed (backward-confirming, reverts at i+1) — removing exactly the real-time
absorption signal fbr keys on. The gap-fill is a FLOW edge (1006, JPY-cross carry channel) and fbr is a
STRUCTURAL edge (1013, live wick); the two mechanisms do NOT compose. A high aggregate +1R capture on a
"buy-the-sweep" bucket can be a take-the-loss LABEL artifact of oversold dead-cat bounces in confirmed
breakdowns (the +1R is reached then given back) — always split by reclaim and check the median drift and
per-pair before reading capture as an edge.
