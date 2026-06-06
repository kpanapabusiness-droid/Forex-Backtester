# arc_2029 — Prior-day / prior-week-low sweep-reclaim (is "human-watched liquidity" cleaner than fbr's rolling swing?)

**Chat:** 2000s · **Date:** 2026-06-06 · **Verdict:** KILL (obs cheap-kill, §5d) · **Disposition:** KILL

## Log synthesis (step a — fresh eyes, honest-era only)
Pulled main; read DISCOVERY_PROTOCOL, the Tier-1 table + recent Tier-2 (esp. 2027/2028), LESSONS, TOOL_REGISTRY, DISCOVERY_DIRECTION. State of the corpus:
- **Deeply converged.** ~70 honest-era arcs. Closed ground = single-condition shallow directional (long OR short, every TF/pair/lens). The 4-component PORTFOLIO book (gap-fill 1006 · me-long 1011 · fbr 1013 · me-short 1019/3017) is mean-positive (+0.589%) but NOT all-folds-positive (~3 bets; binding wall = 2018 strong-USD); the operative deployability lever is the operator's **path-A gate-governance** call, not a 5th reversion leg (arc 2019).
- **The autonomous `explore-now` menu is exhausted:** M1 (1027/2023), O1 (1029 density + 2027 spread-z), L1 (2028 OU amplitude), Q1 (1028), G1 (2018) all closed; S1 is a tautological entry-geometry modifier.
- **But "apparatus exhausted" is a search-ending trap (arc-3004 council warning),** and the dispatch + frontier explicitly invite **extending the arc-1013 deep-structural template** (`fbr` = the corpus's ONLY clean directional structural edge) with a documented *because*. That is the open thread this arc attacks — not a menu item, a fresh extension of the one win.

## Idea (step b — observe, with a mechanism)
**fbr (arc 1013) longs a sweep-and-reclaim of a rolling-40-bar swing low** — resting sell-stops cluster below a visible swing low; a bar pierces it (sweeps stops), closes back above (failed breakdown), deep rejection wick → liquidity-driven down-move, reclaim confirms the adverse excursion is over *at entry*. It is PORTFOLIO (IS 9/10, OOS +0.94%, structure control-proven).

The reference level fbr uses — a rolling-K-bar swing low — is an **algorithmic** construct. The levels human desks and retail platforms actually *mark and cluster stops at* are the **prior-day low (PDL)** and **prior-week low (PWL)** — the single most-watched intraday/swing references in FX. *because:* if the fbr edge is about concentrated resting liquidity getting grabbed, the most-watched calendar-anchored levels should hold **more** concentrated stops → a **cleaner, higher-conviction** grab-reversal.

**Falsifiable prediction:** honest +1R capture at a PDL/PWL sweep-reclaim should **EXCEED** fbr's rolling-swing capture (0.52–0.61), robustly, above the fair baseline (~0.49). **Falsifier:** if PDL/PWL capture ≤ fbr, or the genuinely-distinct (non-fbr) part is a coin-flip, the "human-watched = cleaner liquidity" premise is dead.

## What I did (steps c–d — cheap-kill observation)
Population = H4, the 4 cached majors (AUDUSD/EURUSD/GBPUSD/USDJPY), IS 2010–2020. Per bar, computed ex-ante (no-lookahead, all from completed prior groups):
- **PDL** = min `low_bid` of the prior completed EET trading day (`utc_to_eet_trading_day`, factorized integer day-ids, prior-group `.shift(1)`).
- **PWL** = min `low_bid` of the prior completed ISO week.
- **fbr swing** = rolling-40 `low_bid.shift(1).min()` (arc-1013 reference, reproduced as the anchor).
Fire (each level) = `low_bid < level & close_mid > level & shadow ≥ 1.0·ATR` (the fbr sweep-reclaim test, identical except the level). Scored honest +1R-before-SL capture + 24-bar drift via canonical `observe_long_capture` (gross, characterization only — NOT a gate). Plus the decisive **distinct-from-fbr** subsets (`PDL ∧ ¬fbr`, `PWL ∧ ¬fbr`).

## What happened — FALSIFIED three ways
Population-level (IS, shadow ≥ 1.0; baseline unconditional capture ≈ 0.4877):

| population | n | capture | 24-bar drift (ATR) |
|---|---|---|---|
| **fbr-swing** (anchor) | 287 | **0.547** | **+0.086** |
| PDL sweep-reclaim | 424 | 0.524 | −0.007 |
| PWL sweep-reclaim | 175 | 0.560 | +0.003 |

1. **PDL is WEAKER than fbr, not stronger** (cap 0.524 < 0.547; drift ≈ 0 vs fbr +0.086) — the prediction's direction is refuted at the population level.
2. **PWL marginally exceeds fbr on capture but is 70% the SAME trades** (123/175 PDL∖... PWL fires also fire fbr) and its drift is ≈ 0 — it is mostly fbr re-tagged, not a new edge; PDL overlaps fbr 38% (163/424).
3. **The genuinely-distinct part is a pure coin-flip** (the §5f-decisive check):

| distinct-from-fbr | n | capture | drift |
|---|---|---|---|
| **PDL ∧ ¬fbr** | 261 | **0.502** | −0.052 |
| **PWL ∧ ¬fbr** | 52 | **0.500** | −0.221 |

The new trades PDL/PWL add beyond fbr capture at exactly 0.50 with zero/negative drift. Both populations are also **pair-mix confounds** (the arc-2011/3011 tell): PDL∖fbr per-pair GBPUSD 0.571 vs AUDUSD 0.486/USDJPY 0.449/EURUSD 0.479 — 3 of 4 sub-0.50; per-year drift sign flips yearly (negative in 6 of 11 IS years).

## Why it fails (the mechanism)
**Significance-by-survival beats calendar-anchoring for liquidity-grab edges.** A rolling-K-bar swing low that is *still intact* is a level that has SURVIVED K bars of testing — that survival is precisely what makes it a genuinely dense, much-watched pool of resting stops. The prior-day / prior-week low is **mechanically refreshed every day/week regardless of structural significance**, so the bulk of PDL/PWL sweeps occur at trivially-recent, low-significance levels with no concentrated stops → no reliable grab-reversal (capture collapses to 0.50 once fbr's survived-swing subset is removed). So fbr's rolling-swing reference is **not** an arbitrary algorithmic choice to be "improved" by a human-watched level — it is the *better* liquidity proxy because the survival filter is load-bearing. The calendar anchor removes that filter and dilutes quality (PDL adds n=424 vs fbr 287, but the marginal trades are noise).

## Verdict + what this closes
**KILL (obs cheap-kill, §5d).** §5f does not bite — the distinct base is a coin-flip (0.50, drift ≈ 0), so there is no non-coin-flip entry to put on the engine; no engine / null / council spent (matches 2016/2027/2028). Closes the single most natural extension of the corpus's one directional win: **prior-day / prior-week-low sweep-reclaim is NOT a cleaner fbr.** Strengthens arc 1013 rather than extending the book — it isolates *why* fbr works (the swing-survival significance filter, not the calendar visibility of the level). Components UNCHANGED (all 4 PORTFOLIO). Deployable-system count = 0.

**Threads.** None on the reference-level axis (swing-survival is the load-bearing feature; calendar-anchored levels dilute it). The operative deployability lever remains operator **path-A (gate-governance)** on the existing mean-positive, cost-robust, ~3-bet book.

**Tooling:** no new tool — canonical `Panel.from_pairs`, `observe_long_capture`, `utc_to_eet_trading_day`, BUILT `_atr_shift1_mid` only; no TOOL_REGISTRY append. **FLAGS:** none. Driver: `_arc2029_work/` (scratch, not committed).
