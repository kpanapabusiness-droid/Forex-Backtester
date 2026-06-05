# arc 1025 — Does the corpus's only fold-RESOLVING edge (fbr) THICKEN? (arc-2017 option B)

**Chat:** 1000s · **Date:** 2026-06-05 · **Verdict:** FAIL (§5f honest engine) → **KILL**
**Disposition:** KILL · **passed:** N · **Component touched:** arc-1013 fbr (UNCHANGED, still PORTFOLIO)

> Arc 2017 named the ONE surviving productive spec — **option (B): a component THICK enough that its
> per-year folds RESOLVE and clear zero** (the thin components make the all-folds-positive gate vacuous;
> only `fbr` resolves folds, and its sole real hole is the mechanism-intrinsic −2018). Arc 2018 attacked
> (B) by thickening `me` cross-sectionally → killed by multi-leg cost. This arc attacks (B) by thickening
> the corpus's single fold-resolving edge **itself**: relax `fbr`'s deep/long trigger (shadow & swing
> lookback `K`) to fire more often. **It does not thicken.** The reclaim DEPTH *is* the edge — shallower/
> shorter triggers dilute monotonically to a coin-flip AND add whipsaw variance, so the denser cell has
> **lower mean AND higher across-fold SD** → a strictly **worse** fold-resolver. And **2018 stays a
> coin-flip-or-wipeout at every density** (HYP-B). Option (B) is closed for the best edge via the depth
> lever; the per-year gate stays structurally unsatisfiable for this thin-edge corpus.

---

## Log reading (step a — FRESH EYES, honest-era only)

Resumed 1000s at arc 1025 (highest in-range = 1024; pulled main, no `discovery/STOP`). State: four
net-positive PORTFOLIO components — gap-fill (1006, JPY-cross H4), month-end-long `me_long` (1011,
USD-major D1), failed-breakdown-reclaim `fbr` (1013, USD-major H4, the strongest, 9/10 IS), month-end-SHORT
`me_short` (1019, USD-major D1, first robustly +2018). The 4-way book (arc 1020) is the strongest in the
corpus (worst fold ≈ −0.115%, 2018 wall breached) but **0/all convex weightings all-folds-positive**,
blocked by **marginal 2015/2016**.

The recent diagnostic chain is decisive and convergent (2016 → 2017 → 1023 → 1024, three chats):
- **arc 2016/1023:** the 4-way book's residual negative folds are **below the noise floor**; across-fold
  SD (~0.74%) ≈ within-fold sampling SD ⇒ year-folds are sampling-dominated. The book **mean is
  significantly positive** (t=2.66, CI [+0.22%,+1.09%]); it fails ONLY the too-fine per-year gate. The
  ~18-arc 5th-leg hunt was fold-painting inside the floor.
- **arc 2017:** components split — gap/me_long/me_short are **too thin to resolve any fold** (gate
  vacuous); only **`fbr` resolves folds**, and it carries the corpus's **only statistically-real negative
  fold, 2018** (−4.20%, CI<0; arc 2014: 18/19 −1R, mechanism-intrinsic, entry-time-unconditionable).
  Named the only productive spec: **option (B) a component thick enough that folds resolve AND clear
  zero** (hard — closed ground makes dense/shallow directional = coin-flip).
- **arc 2018:** attacked (B) via cross-sectional USD-neutral `me` → KILL (stripping USD beta strips the
  edge; multi-leg cost vs coin-flip).
- **arc 1024:** `risk_pct` is a FRACTION; the whole picture is at 0.5% deployable risk, fold signs stable.

Closed ground (LESSONS): shallow single-condition directional (long OR short) is dead at every TF/universe/
metric; carry is OFF. **Open frontier:** a genuinely-novel mechanism, or the operator-gated path-A
gate-resolution / escalation. Leg-hunting a thin 5th fold-painter is explicitly deprecated.

## Idea + because (attack option B with the BEST edge, via its OWN density lever)

Arc 2018 thickened `me` (cross-sectional) and cost killed it. The **un-tried** (B) route is to thicken
the corpus's single fold-resolving edge directly. `fbr` (1013) fires only ~17–21/yr **because** it needs
a DEEP (shadow ≥ 1.25 ATR) reclaim of a LONG (`K`=40) swing low — rare, deep forced-flow stop-runs. The
**same stop-run-reversal mechanism** fires far more often at shorter lookbacks (`K`=10/20, swept more
often) and shallower reclaims (shadow ≥ 0.5/0.75). Two genuinely-untested questions, one cheap obs sweep:

- **HYP-A (thicken / option B):** does a denser `(K, shadow)` cell keep enough edge — capture > 0.50 +
  load-bearing structure control + null-beatable drift — to be a **fold-resolving** thicker component?
  (arc 1013 found "deeper grab = stronger", which *predicts* shallow dilutes — but the full count/edge
  tradeoff surface was never published, and **whether the extra trades shrink per-fold SE faster than the
  weaker mean falls** — the actual fold-resolution test — was never measured.)
- **HYP-B (the book's one real obstacle):** is `fbr`'s 2018 wipeout `(K, shadow)`-INVARIANT (every
  fbr-class trade loses in strong-USD 2018 → mechanism-intrinsic, closes the obstacle harder) or do
  shallower/shorter triggers catch tradeable 2018 trades (→ a real lead on the one blocking fold)?

USD majors only (arc 1022: `fbr` is USD-major-specific; crosses carry no edge). All IS (2018 is an IS
year) → OOS never touched.

## What happened

### Step 1 — observation surface (gross capture + drift, K×shadow, IS) — `arc1025_fbr_density_sweep.py`

`observe_long_capture` (honest +1R-before-SL, long) restricted to the BUILT `fbr` fire mask at each cell,
plus the arc-1013 **load-bearing structure control** (deep wick AT swept-low vs the SAME deep wick
ELSEWHERE). 16 cells, K∈{10,20,40,60} × shadow∈{0.5,0.75,1.0,1.25}:

| K | shadow | n (/yr) | capture | drift | struct_cap | struct_drift | pp>.50 | 2018 cap | 2018 drift |
|---|---|---|---|---|---|---|---|---|---|
| **40** | **1.25 (ref)** | 237 (21.5) | **+0.5865** | **+0.2365** | **+0.1077** | **+0.3212** | **7/7** | 0.4737 | −0.729 |
| 40 | 1.00 | 507 (46.1) | +0.5227 | +0.0977 | +0.0436 | +0.1877 | 5/7 | 0.5000 | −0.187 |
| 40 | 0.75 | 1092 (99) | +0.5110 | +0.0538 | +0.0278 | +0.1272 | 3/7 | 0.4845 | +0.155 |
| 40 | 0.50 | 2400 (218) | +0.5004 | +0.0169 | +0.0113 | +0.0590 | 3/7 | 0.5070 | +0.131 |
| 20 | 1.25 | 318 (29) | +0.5503 | +0.0222 | +0.0701 | +0.0364 | 5/7 | 0.5600 | −0.831 |
| 10 | 1.25 | 430 (39) | +0.5395 | −0.0004 | +0.0722 | +0.0332 | 5/7 | 0.6333 | −0.010 |
| 60 | 1.00 | 411 (37) | +0.5134 | +0.1580 | +0.0289 | +0.2427 | 5/7 | 0.5128 | +0.016 |

(full 16-cell table in the driver output.) **The edge degrades MONOTONICALLY with density** along BOTH
levers: capture 0.5865 → ~0.50, structure-excess +0.108 → ~0, drift +0.237 → ~0 as the trigger gets
shallower/shorter. The committed **K40/s1.25 is the cleanest cell** (the deepest grab). The one genuine
"denser but still non-coin-flip" candidate is **K40/s1.00** (n=507, ~2× ref; cap 0.5227, struct +0.044,
drift +0.098, 5/7 pairs). **HYP-B obs:** 2018 capture is **coin-flip-or-worse at every cell** (min 0.458,
max 0.633 — the 0.633 is the thin n=30 deep K10 cell) and drift ≤ ~0 wherever the cell has any real edge
→ **2018-positivity and the fbr-edge are mutually exclusive across the whole surface.**

### Step 2 — §5f honest engine on the candidate vs ref — `arc1025_fbr_density_engine.py`

K40/s1.00 is non-coin-flip (cap 0.5227) → §5f mandates the exit-menu sweep before a FAIL. Ran the
candidate AND the committed ref through the **same** 7-exit menu on the same 10 IS folds + the **fair
same-exit null** (`build_null_signal_evaluation`), risk_pct 0.005 (=0.5% deployable, arc 1024). Fold 9 =
OOS 2018.

**ref K40/s1.25** — committed exit `sl_plus_trailing_atr`+trail reproduces the headline **EXACTLY**
(+1.854%, **9/10**, fold-9/2018 the lone neg) → apparatus validated. mean/SD at the committed exit
**0.604**.

**DENSER K40/s1.00** — best honest exit `sl_plus_trailing_atr`:
- mean **+0.576%** (vs ref +1.854%) — collapses to ~⅓; only **6/10** positive
- across-fold **SD 4.53%** (vs ref's 3.07% at its committed exit) — **rises**, not falls
- **mean/SD = 0.127** (vs ref **0.604**) — the fold-resolution proxy is **~5× WORSE**
- 2018 (fold 9) = **−4.118%** (still NEG), and it ADDS 2014 (−7.42%) & 2019 (−3.81%) negatives
- beats the fair null by **+0.870pp** → a REAL but diluted edge (not noise), strictly dominated by the
  committed deep `fbr`.

## Diagnosis

**Reclaim DEPTH is the forced-flow signal, and depth is exactly what makes the event rare.** A deep
lower-wick that pierces a structural swing low and reclaims it = a genuine stop-run-and-liquidation
reversal (the dense USD-major stop book, arc 1013/3013/1022). A *shallow* sweep is a routine probe with no
liquidation behind it → it carries no edge (capture → 0.50) AND it whipsaws (the bar re-breaks), so it
ADDS across-fold variance. That is why thickening fails on BOTH terms at once: the denser cell has a
**lower mean** (less edge per trade) AND a **higher SD** (more whipsaw) → mean/SD craters. **Fold-
resolution and trade-count are NOT independent levers for a forced-flow edge** — depth couples them.

**HYP-B confirmed (obs + engine):** 2018 is a coin-flip (the shallow cells) or a wipeout (the deep
edge-bearing cells) at every `(K, shadow)`; thickening adds MORE negative folds (2014/2019), not fewer.
`fbr`-2018 is density-invariant and mechanism-intrinsic — independently re-confirms arc 2014/2017 from a
new (density) angle.

## Verdict — KILL (no new component; `fbr` UNCHANGED, PORTFOLIO at the committed deep config)

HYP-A FALSE: `fbr` does **not** thicken into a fold-resolving component — **arc-2017 option (B) is closed
for the corpus's best edge via the depth lever** (as arc 2018 closed it for `me` via the universe/
cross-sectional lever). No pool/council needed beyond the §5f engine (the obs is a §5d coin-flip-toward-
density read; the one non-coin-flip candidate got its mandated §5f sweep + null). OOS never touched.

## Threads / lessons

1. **NEW (the escape hatch closes):** for a forced-flow edge, **fold-resolution (thickness) and edge-
   strength are COUPLED through the trigger depth — you cannot trade depth for trade-count without trading
   edge for whipsaw variance.** The denser `fbr` has BOTH lower mean (+0.58 vs +1.85%) AND higher
   across-fold SD (4.5 vs 3.1%) → its fold-resolution (mean/SD) is ~5× worse. This closes the "just
   thicken the best edge to satisfy the per-year gate" route (arc-2017 option B) for the strongest edge,
   the complement to arc 2018's cross-sectional-`me` closure.
2. **Strengthens the corpus's unified theory:** every real edge here is a forced-flow reversion (gap
   liquidity-vacuum, WMR rebalancing, stop-run liquidation); the forced flow is what beats cost, and
   forced-flow events are **intrinsically rare** → all real edges are thin → the per-year all-folds gate
   is **structurally unsatisfiable** for this corpus (the operator's path A vs B governance call stands;
   B has now been attacked from both its obvious levers — universe/cross-section (2018) and depth (1025)
   — and is closed for both proven edges).
3. **`fbr`-2018 is density-invariant** (obs cap < 0.52 every cell; engine 2018 negative every thickened
   exit) — a 4th independent confirmation (1013/2014/3013 + now density) that the one real fold-obstacle
   is mechanism-intrinsic, not a missing leg.
4. **Reusable:** before claiming a denser variant "thickens" a thin component, measure mean/SD (the
   fold-resolution proxy), not just trade count — extra trades can RAISE across-fold SD when they add
   whipsaw rather than edge.

## Tooling

No new BUILT tool. **CALLED** `Panel.from_pairs`, the BUILT `FailedBreakdownReclaimLongSignal` at swept
`(swing_lookback, min_shadow_atr)`, `observe_long_capture(direction="long")`, the BUILT
`build_null_signal_evaluation`, and the canonical `A1Architecture`/`ArcFoldRunner`/`run_config_over_folds`
(scoring stays canonical). Drivers: `_disco_work/arc1025_fbr_density_sweep.py` (obs surface),
`_disco_work/arc1025_fbr_density_engine.py` (§5f engine + null; reproduces the committed +1.854% 9/10 fbr
headline EXACTLY → apparatus validated).

**FLAGS (code not merged):** none new. Carries the standing `A1Config.time_exit_bars`-unwired flag
(arc 1005). OOS never touched.
