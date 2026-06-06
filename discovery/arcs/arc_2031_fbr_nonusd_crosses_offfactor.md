# arc_2031 — fbr on non-USD crosses: does the clean structural edge transfer OFF the dollar factor?

**Chat:** 2000s · **Date:** 2026-06-06 · **Verdict:** KILL (obs cheap-kill, §5d) · **Disposition:** KILL

## Log synthesis (step a — fresh eyes, honest-era only)
Pulled main; read DISCOVERY_PROTOCOL, the Tier-1 table + recent Tier-2 (2027–2030, 1031, 3020–3022), LESSONS, TOOL_REGISTRY. State of the corpus:
- **Deeply converged.** ~70 honest-era arcs. Closed ground = single-condition shallow directional (long OR short, every TF incl. **W1** — arc 3014 — and pair/lens). The 4-component PORTFOLIO book (gap-fill 1006 · me-long 1011 · fbr 1013 · me-short 1019/3017) is mean-positive (+0.589%), cost-robust (κ break-even 3.32), but NOT all-folds-positive — binding wall = the strong-USD risk-off folds **2015 & 2018**.
- **Two structural closures frame the only remaining edge-question:** (i) arc 2022 — even a perfectly-targeted, maximally-decorrelated 5th leg fails the honest gate (variance-weighting dilemma); the precise unmet spec is *a moderate-variance leg robustly + in BOTH 2015 AND 2018*. (ii) arc 3021 — densification is **provably** off the table: at the corpus's residual correlation ρ≈+0.12 a **shared dollar/risk factor floors book variance**, so P(AFP) plateaus ~0.30 at any N. ⇒ the only escape from the floor is a component **off the shared dollar factor**.
- 3 of the 4 components are USD-majors-based; the one clean directional edge (`fbr`) has only ever run on USD majors. **The off-dollar-factor leg arc 3021 needs has never been attempted by porting the clean edge to a non-USD universe** — that is this arc.
- **Guard against the search-ending trap (arc-3004 council):** "apparatus exhausted" is seductive and wrong to assume. This arc is a *fresh, because-grounded* test of the one untried route to the structurally-needed leg, not a menu item.

## Idea (step b — observe, with a mechanism)
`fbr` (arc 1013) longs a sweep-and-reclaim of a survived rolling-40-bar swing low: resting sell-stops cluster below a visible swing low; a bar pierces it (sweeps stops), closes back above with a deep rejection wick (failed breakdown) → the down-move was liquidity-driven, the reclaim confirms the adverse excursion is over *at entry*. PORTFOLIO (IS 9/10, OOS +0.94%, structure control-proven), the corpus crown jewel.

**The distinct *because* (why this is not arc 1018 redux).** Arc 1018 ported the **gap-fill** to USD-neutral crosses and found it dead, with the lesson *"the gap-fill edge is USD/carry-specific; the edge and the strong-USD tail are the SAME exposure."* But gap-fill is a **carry/USD-driven flow reversion** — factor-specific by construction. `fbr` is a **liquidity-grab / stop-run** mechanism, and stop-clustering below swing lows is a **microstructure-universal** phenomenon (every liquid instrument has it), economically distinct from carry. So fbr is the *best candidate* to break arc 1018's lesson — if its structural lift is genuinely universal, it should survive transfer to non-USD crosses, and being off the dollar factor those crosses' **2018 fold is not the USD grind** → a plausible 2015/2018-orthogonal leg (arc 2022/3021's exact spec).

**Falsifiable prediction:** fbr's structural capture lift (majors ~0.55–0.59 vs 0.4877 base) transfers to non-USD crosses (capture clears the cross directional base ~0.4712, arc 1003) AND is + in 2015 & 2018. **Falsifier:** if cross fbr collapses to the coin-flip base, or is net-coin-flip / not +both-folds, the lift is USD-majors-specific (arc 1018 generalizes to fbr) and the route to the off-factor leg via the clean edge is closed.

## What I did (steps c–d — cheap-kill observation)
Honest +1R-before-SL capture + 24-bar drift via canonical `observe_long_capture` (gross, characterization only — NOT a gate), on the fbr fire-mask (BUILT `FailedBreakdownReclaimLongSignal`, K=40, shadow≥1.25), IS 2010–2020, H4.
- **Universe:** 8 **non-USD, non-CHF** crosses — EURGBP, EURAUD, EURNZD, EURCAD, GBPAUD, AUDNZD, AUDCAD, NZDCAD. (CHF crosses excluded on purpose: the Jan-2015 SNB de-peg is a ~182-ATR gap-through-stop, arc 1028 — it would let one catastrophic day define the 2015 fold I most need to read cleanly.)
- **Arc-10 anchor (mandatory before trusting cross numbers):** reproduced fbr on the 4 cached USD majors.
- **Rigor pass:** unconditional cross base (does fbr *lift* it?); 2018 per-pair (one-pair-carried?); shadow sweep {1.0, 1.25, 1.5}.

## What happened — FALSIFIED (structural lift does NOT transfer)
Baseline references: majors unconditional 0.4877; cross unconditional 0.4788.

| population (IS, K40/sh1.25) | n | capture | drift_mean (ATR) | 2015 | 2018 |
|---|---|---|---|---|---|
| **MAJORS fbr** (anchor) | 132 | **0.5833** | +0.068 | cap 0.833 / +2.23 | cap 0.455 / **−2.17** |
| **NON-USD CROSS fbr** | 233 | **0.5021** | −0.052 | cap 0.500 / **−1.05** | cap 0.708 / **+1.59** |
| cross unconditional base | 138 757 | 0.4788 | −0.033 | — | 0.490 |

1. **The structural lift is USD-majors-specific.** On crosses fbr-fire capture is **0.5021** vs the unconditional cross base **0.4788** — a **+0.02 non-lift**, versus majors' **+0.10** lift (0.4877→0.583). The shadow sweep confirms there is no edge to find: cross fbr cap **0.486** (sh1.0) / **0.502** (sh1.25) / **0.470** (sh1.5) — all ≈ the coin-flip base, deeper shadow doesn't build an edge (the opposite of majors, where it does). **fbr ≈ random entry on non-USD crosses.**
2. **A clean regime FLIP, mirror of majors.** Majors fbr = +2015 / −2018; cross fbr = **−2015 / +2018**. The crosses' 2018 IS positive (cap 0.708, drift +1.59; monotone in shadow 0.644→0.708→0.813; broad across 7 pairs, not one-pair-carried) — but 2015 is negative and every other year is a coin-flip, which is exactly why the full-IS aggregate is 0.50 / drift −0.05.

## Why it fails (the mechanism) — generalizes arc 1018 to the structural family
Stripping the USD-majors exposure (porting fbr to non-USD crosses) **removes the 2018 hole** (cross 2018 +0.71) — **but also removes the edge everywhere else** (capture collapses to the 0.50 coin-flip base). The edge and the strong-USD tail are **the same factor exposure** — the *exact* arc-1018 gap-fill lesson, now confirmed for the **structural-reclaim** family. So fbr's liquidity-grab lift is **not** microstructure-universal as hypothesized; on liquid FX it is realized through the USD-majors factor, and a swing-low reclaim on EURGBP/AUDNZD/… carries no reliable stop-grab edge. The +2018 cross result is the absence-of-a-dominant-USD-trend showing up as a single-fold positive inside a coin-flip population — **regime-luck within a coin-flip** (the arc-1016/3010 signature), not a harvestable edge: net-coin-flip over IS, **negative in 2015**, n=24 in the one good fold.

**This closes the one untried route to arc 3021's needed off-factor leg.** You cannot manufacture a leg off the shared dollar factor by porting a USD-majors edge to non-USD crosses, because the edge *is* the dollar-factor exposure — shedding the factor sheds the edge. arc 3021's floor stands by mechanism, not just by induction.

## Verdict + what this closes
**KILL (obs cheap-kill, §5d).** §5f does not bite: the full-IS cross base is a coin-flip (0.502 ≈ 0.479 unconditional, drift −0.05); the only positive cell is a single fold (2018, n=24) inside that coin-flip, and a signal cannot be selected to "fire only in 2018" without lookahead. Net-coin-flip / not mean-positive ⇒ **KILL, not PORTFOLIO** (§11: beating null is necessary; cross fbr does not even beat the cross base). No engine / null / council spent (matches 2029/2030/2016 §5d discipline). Components UNCHANGED (all 4 PORTFOLIO). Deployable-system count = 0.

**Threads / lessons.**
1. **NEW lesson:** `fbr`'s clean structural edge is **USD-majors-factor-specific, NOT microstructure-universal** — porting it to non-USD crosses collapses it to the coin-flip base (+0.02 non-lift vs +0.10 on majors). Generalizes arc 1018 (gap-fill is USD/carry-specific) from the **flow-reversion** family to the **structural-reclaim** family: *in liquid FX, a directional edge and its strong-USD regime tail are the same factor exposure.*
2. **arc 3021's USD-factor floor is now mechanistically closed on the clean-edge route:** an off-dollar-factor leg cannot be built by transferring a USD-majors edge to crosses (edge ≡ factor). The off-factor leg, if it exists, must come from a mechanism that is *natively* non-USD — and arcs 1018 (USD-neutral gap), 1003/3000 (cross directional base 0.4712 < majors, wider spread), and now this one all say non-USD crosses carry no transferable structural edge.
3. The operative deployability lever remains the operator's **path-A gate-governance** call on the existing mean-positive, cost-robust, ~3-bet book — not a 5th leg (arc 2019/2022) and not densification (arc 3021).

**Tooling:** no new tool — canonical `Panel.from_pairs`, BUILT `FailedBreakdownReclaimLongSignal` (arc 1013) + `observe_long_capture` only; no TOOL_REGISTRY append. **FLAGS:** none. Driver: `_arc2031_work/` (scratch, not committed).
