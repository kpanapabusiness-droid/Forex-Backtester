# arc_2032 — Up-gap weekend CONTINUATION long: is the documented gap-asymmetry's productive leg an edge?

**Chat:** 2000s · **Date:** 2026-06-06 · **Verdict:** KILL (obs cheap-kill, §5d) · **Disposition:** KILL

## Log synthesis (step a — fresh eyes, honest-era only)
Pulled main; read DISCOVERY_PROTOCOL, the Tier-1 table + recent Tier-2 (2026–2031, 1029–1033, 3020–3022), LESSONS, TOOL_REGISTRY. State of the corpus:
- **Deeply converged (~75 honest-era arcs).** Closed ground = single-condition shallow directional (long OR short, every TF incl. W1, every pair/lens). The 4-component PORTFOLIO book (gap-fill 1006 · me_long 1011 · fbr 1013 · me_short 1019/3017) is mean-positive (+0.589% RP, t=2.66), cost-robust (κ=3.32), temporally stable, ~3 independent bets (ENB 3.32) — but NOT all-folds-positive; binding wall = the strong-USD risk-off folds **2015 & 2018**.
- **The lever is the operator's path-A gate-call.** Leg-hunt structurally closed (2022 weighting dilemma; 3021 densification provably impossible at ρ≈+0.12; 2031 closed the off-factor escape on the clean edge). Explore-now MENU exhausted (M1/O1/L1/Q1/G1/S1). arc 1032 quantified path-A; arc 1033 the book's risk geometry.
- **Guard against the search-ending trap (arc-3004 council):** "apparatus exhausted" is seductive and wrong to *assume*. §2/§5a demand a genuine fresh-eyes look for a novel, because-grounded mechanism before accepting closure.

**The gap modality is the corpus's most productive family yet has ONE untested face.** Coverage to date: down-gap fill LONG = arc 1006 PORTFOLIO (the first real edge); up-gap fill SHORT = arc 1016/2013 KILL; USD-neutral cross gaps = arc 1018 efficient. The corpus repeatedly *documented an asymmetry* — down-gaps revert/fill, but **up-gaps CONTINUE up at i+1** (arc 1016's own diagnosis: the up-gap short fails *because* the gap keeps going up). The **productive leg of that asymmetry — BUYING the up-gap continuation — was never measured directly.** 1016/2013 only ever evaluated the *failed short*; the long-continuation capture was never read.

## Idea (step b — observe, with a mechanism)
**Because:** a weekend up-gap is a one-shot weekend-news repricing (risk-on / positive surprise). The corpus found down-gaps overshoot and fill (1006), while up-gaps were observed to *continue* up at the tradeable i+1 bar (1016). IF that continuation is a genuine central-tendency drift, a LONG entered at i+1 rides it — and on **USD majors a USD-strength up-gap continuing = a strong-USD/risk-off-aligned move**, making it a candidate **+2018 (and +2015)** leg (the exact arc-2022/3021 spec the book needs).

**Falsifiable prediction:** up-gap LONG honest i+1 capture clears the 0.4877 majors base AND shows positive central-tendency (median) forward drift, robust across pairs and monotone-ish in gap depth, with + 2015 & 2018. **Falsifier:** coin-flip capture / zero-or-negative median drift / pair-mix / non-monotone in depth / 2015–2018 failing ⇒ the documented "up-gaps continue" is the same untradeable hindsight-gap-open artifact (arc 2001's −0.57 ATR was the gap-bar OPEN) seen from the long side — no edge either way by i+1.

## What I did (steps c–d — cheap-kill observation)
Honest +1R-before-SL capture + 24-bar forward drift via canonical `observe_long_capture(direction="long")` (gross, characterization only — NOT a gate), restricted to weekly-open UP-gap bars (gap detection replicated from BUILT `WeekendUpGapShortSignal`: index-gap >20 h flags the weekly open; `gap_atr = (open_mid − prior_close_mid)/ATR ≥ +thr`, ATR Wilder(14) MID shift1 — all ex-ante). IS 2010–2020, H4.
- **Universe:** the 4 **cached** USD majors — AUDUSD, EURUSD, GBPUSD, USDJPY (the 2015/2018-relevant USD universe; §5d less-data-early — extend only if the first read is alive).
- **Arc-10 anchor:** unconditional base capture reproduced **0.4889** (corpus anchor 0.4877 ✓ — apparatus faithful).
- **Rigor pass:** depth sweep thr∈{0.5, 1.0, 1.5}; per-pair (pair-mix?); per-year (2015/2018 acceptance); mean **and median** drift (thin-tail tell).

## What happened — FALSIFIED (no continuation edge by i+1)

| up-gap LONG (IS, H4, 4 majors) | n | capture | drift_mean | drift_med | 2015 (cap/drift) | 2018 (cap/drift) |
|---|---|---|---|---|---|---|
| unconditional base | 69 405 | 0.4889 | — | — | — | — |
| **thr ≥ 0.5 ATR** | 228 | 0.5263 | +0.077 | **0.000** | 0.444 / −1.538 | 0.706 / +0.751 |
| **thr ≥ 1.0 ATR** | 72 | **0.4861** | −0.119 | −0.248 | 0.000 (n1) | 0.571 / −0.533 |
| **thr ≥ 1.5 ATR** | 34 | 0.5588 | −0.023 | **−0.300** | — | 0.000 (n1) / −5.155 |

1. **No robust capture lift, NON-MONOTONE in depth.** thr0.5 0.526 (marginal), **thr1.0 0.486 — BELOW the coin-flip base** at the *meaningful* gap depth, thr1.5 0.559 (thin). The deeper/more-significant the up-gap, the *worse* the capture turns first — the arc-3007 "not-a-lever" + arc-2011 thin-tail signature, the opposite of a real edge that strengthens with depth (cf. fbr 1013).
2. **Drift MEDIAN ≤ 0 at every threshold** (0.000 / −0.248 / −0.300). There is no central-tendency continuation; the only positive *mean* (thr0.5 +0.077) is a thin upper tail. The honest i+1 up-gap has no systematic drift — the short couldn't fade it (1016), the long can't ride it (here).
3. **Pure pair-mix (USD-quote-beta confound).** USDJPY drift +1.0..+1.4 (its secular USD/carry up-drift), GBPUSD ~flat-to-+, vs **EURUSD strongly NEGATIVE −0.9..−1.1** and AUDUSD negative → 2/4 pairs >0.50 capture. What looks like "up-gap continuation" is USDJPY's per-pair structural drift (the arc-2009/3012 USD-quote-beta tell), **not a gap effect**.
4. **2015/2018 acceptance FAILS.** 2015 negative at every threshold (−1.54 / n1-zero / absent). 2018 positive *only* at the shallow coin-flip thr0.5 (cap 0.706) and **inverts NEGATIVE at thr1.0 (−0.533) and thr1.5 (−5.155)** — regime-luck-within-a-coin-flip (arc-1016/3010 signature), not a harvestable +2018 leg.

## Why it fails (the mechanism)
A weekend up-gap is a discrete one-shot repricing. By the tradeable i+1 open the repricing is **already complete** — there is no systematic continuation to ride (median drift 0) any more than there was reversion to fade (1016). The "up-gaps continue" the corpus noted was the same **untradeable hindsight gap-bar-OPEN** observation (arc 2001's −0.57 ATR was open→close, in hindsight) read from the long side; the honest entry sees **nothing in either direction**. The only apparent "continuation" is **USDJPY's structural up-drift** (USD/carry-beta) — a per-pair regime exposure that survives FundedNext's carry-off only as price drift and is the dead trend/momentum family, not a gap signal. This is consistent with carry being OFF on the target broker (LESSONS): a JPY-cross extension would merely surface more carry-drift, not a gap edge, so it was not spent (§5d compute discipline).

## Verdict + what this closes
**KILL (obs cheap-kill, §5d).** §5f does not bite: there is no non-coin-flip entry — every above-0.50-capture cell has ≤0 median drift (thin-tail), and the one meaningful-depth cell (thr1.0) is *below* coin-flip; the only positive *mean* is a coin-flip-capture thin tail, USDJPY-carried, with 2015 negative and 2018 inverting with depth. Net-coin-flip / not mean-positive ⇒ **KILL, not PORTFOLIO** (§11). No engine / null / council spent (matches 2029/2030/2031/2016 §5d discipline). Components UNCHANGED (all 4 PORTFOLIO). Deployable-system count = 0.

**This closes the LAST untested face of the weekend-gap modality.** The full map now: down-gap fill LONG = PORTFOLIO (1006, JPY-cross-carry-specific) · up-gap fill SHORT = KILL (1016/2013) · USD-neutral cross gaps = efficient (1018) · **up-gap continuation LONG = KILL (2032)**. The documented gap *asymmetry* is real only as an *untradeable hindsight gap-open* fact — by the honest i+1 entry the gap is efficient in **both** directions; the only surviving gap edge remains the down-gap **fill** on carry-driven JPY crosses (1006).

**Threads / lessons.**
1. **NEW lesson:** the weekend-gap asymmetry ("down-gaps fill, up-gaps continue") does NOT yield a tradeable continuation long — the i+1 up-gap has zero median drift and sub-0.50 capture at meaningful depth; the apparent continuation is USDJPY USD/carry-beta (pair-mix), not a gap effect. *A documented hindsight-gap-open displacement is untradeable from i+1 in BOTH directions* (extends 1016's short-side finding to the long, and arc 2001's "the −0.57 ATR was the untradeable open" to the continuation reading).
2. The candidate +2018 leg is once again **not** in this construction — 2018 inverts negative with gap depth (regime-luck within a coin-flip). The route's 2018/2015 wall stands; the off-factor / +both-folds leg remains unfound across every directional, structural, flow, and now gap-continuation route.
3. Operative deployability lever unchanged: the operator's **path-A gate-governance** call on the existing mean-positive, cost-robust, ~3-bet book (arc 2019/3021/1032), not a 5th leg.

**Tooling:** no new tool — canonical `Panel.from_pairs` + BUILT `observe_long_capture` (direction="long") only; up-gap mask replicated inline from BUILT `WeekendUpGapShortSignal` (no new committed signal — would only build one if it reached the engine). No TOOL_REGISTRY append. **FLAGS:** none. Driver: scratch (not committed).
