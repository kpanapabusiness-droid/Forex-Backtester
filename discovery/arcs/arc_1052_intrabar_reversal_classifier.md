# arc 1052 — Intrabar (M1-path) reversal-velocity as a forced-vs-informed displacement classifier

**Chat:** 1000s · **Date:** 2026-06-06 · **Disposition:** KILL (obs cheap-kill, §5d) · no engine / null / council · OOS untouched · components UNCHANGED.

Full driver: [`../_disco1_work/arc1052_intrabar_reversal_obs.py`](../_disco1_work/arc1052_intrabar_reversal_obs.py) (pure-pandas characterization; reuses the canonical `core.data.histdata_loader.load_m1` data loader; replicates the canonical `5ers_eet` H4 grouping for intrabar alignment; GROSS fade-drift in ATR units, never realizes P&L, no canonical reimplementation; single-use scan → no BUILT registry entry).

## Log read (step a)
Pulled main. Read DISCOVERY_PROTOCOL, LESSONS (Closed ground + open-frontier ranking), TOOL_REGISTRY (canonical + 30+ BUILT), and the Tier-1 ledger + recent Tier-2 (arcs 1042–1051, plus 2x-range cross-refs). State on entry: **frontier independently re-confirmed exhausted** by the prior 1000s handoff and by 1032/1033. The 4 PORTFOLIO components (gap 1006 / me_long 1011 / fbr 1013 / me_short 1019) are all forced-flow reversions; the book is mean-positive (+0.55–0.59%, t≈2.1–2.7) but NOT all-folds-positive (2018 strong-USD year binds); §11 verification COMPLETE end-to-end (signal 2034/35 + outcome 2036/37/38 + cost 2039); honest deploy object = {me_long+fbr} 2-way (1045/1046/2045), vehicle-infeasible (Calmar ~0.245, ~5yr underwater). Every documented lane is closed: explore-now MENU (M1/O1/L1/Q1/G1/S1 all dead), triangulation (3005/1027/1031/2028), relative-value/market-neutral (2003/2010/2018 = doubled-cost-vs-coin-flip), boundary-defense (1028), all calendar flows (gotobi 1008 / NFP 1048 / fiscal-YE 2026 / quarter-end 1012/1029 / Friday 3015), gap×structure conjunction (1034), fbr off-factor (2031 = edge≡factor), option-B thickening (5 constructions), vol-target overlay (1036), path-A quantified (1032). The lever is the operator's gate-governance call.

**Fresh-eyes obligation (§2 / arc-3004 "apparatus-incapable is a seductive search-ending conclusion").** Rather than declare done, I looked for a mechanism OUTSIDE the entire mapped set. The single dimension the corpus has NEVER touched: the **M1 intrabar path** — every arc runs on H4/H1/D1 *resampled* OHLC, which destroys the within-bar trajectory (the timing and sequence of the high/low). The M1 bid+ask is in the corpus (it is the loader's source) but its intrabar information has never been used as a signal.

## Idea + because (step b)
The corpus's hard-won unified theory: a tradeable reversion needs a SURPRISE displacement that is **MECHANICAL/forced** (overshoots into a thin book → reverts), NOT **informed** (continues). arc 1050 showed bar-SHAPE proxies (range/vacuum) FAIL to separate these → collapse to generic momentum. **But bar shape is an H4-OHLC property; it cannot see intrabar timing.** Hypothesis: a mechanical overshoot prints its extreme EARLY in the bar and RETRACES by the bar close (liquidity returns within the 4h), whereas an informed move trends and closes AT its extreme. The M1 retrace-fraction and extreme-timing are genuinely new features → could isolate the reverting (forced) subset that magnitude (1050) and bar-shape cannot. Natively testable on the non-USD cross (EURGBP) too → a candidate off-the-dollar-factor angle (the unmet 2015/2018 spec).

**Falsifiable prediction:** among large H4 displacement bars (|close−open|/ATR ≥ thr), the HIGH-reversal subset (high in-bar retrace + early extreme) shows POSITIVE next-K-bar FADE drift (reverts); the LOW-reversal subset continues (negative, as 1050's base). If intrabar reversal does NOT separate forward reversion from continuation, the "intrabar order-flow proxy" lane is CLOSED.

## Method (steps c/d — obs cheap-kill)
11 M1-cached pairs (7 USD majors + AUDJPY/EURJPY/GBPJPY + EURGBP), IS 2010-2020, H4 (5ers_eet). Per H4 bar from M1: mid OHLC, `retrace` = fraction of the displacement-direction extension given back by close (high = faded-in-bar), `ext_t` = normalized timing of the extreme (early = mechanical). ATR Wilder(14) on H4 mid, shift1. `disp=(close−open)/ATR`; `fade_k = −sign(disp)·(close[i+k]−close[i])/ATR` (positive = displacement reverted). No lookahead (features known at bar i close; entry implied i+1; ATR shift1).

## Result — FALSIFIED (the gradient is INVERTED and sub-cost)
**Base reproduces arc 1050:** fade drift negative everywhere, frac+ ≈ coin-flip (thr1.0 k1/k2/k3 = −0.014/−0.022/−0.032; thr1.5 = −0.031/−0.048/−0.067) → displacement CONTINUES.

**By intrabar RETRACE quartile (thr1.0, k3) — monotone but the WRONG SIGN:**
| retrace | mean_fade | frac+ |
|---|---|---|
| Q1_lo (closed AT extreme = "informed") | **+0.0165** | 0.523 |
| Q2 | −0.047 | 0.495 |
| Q3 | −0.041 | 0.489 |
| Q4_hi (faded in-bar = "mechanical") | **−0.0575** | 0.482 |

The gradient is real and monotone but **exactly inverted vs the hypothesis**: bars that closed AT their extreme (low retrace) REVERT; bars that already faded in-bar CONTINUE. And the best cell (+0.017 ATR) is **far sub-cost** (H4 round-trip ≈0.05–0.15 ATR; cf. 1051's H1 RT ≈0.308) at frac+ 0.523 ≈ coin-flip.

**By EXTREME-TIMING quartile (thr1.0, k3):** early −0.028 / Q2 −0.054 / Q3 −0.031 / late −0.015 → **non-monotone, all negative, no separation** → extreme-timing is NON-LOAD-BEARING (the timing half of the hypothesis is decisively dead).

**Coherent cell (retrace≥Q3 & extreme early≤Q1):** sign-FLIPS with threshold (thr0.75 −0.025 / thr1.0 −0.020 / thr1.5 +0.008) — the thin-tail-inversion signature (1050/3010). At thr1.5 the marginal +0.008 is per-year NOISE: 2014 −0.166, **2018 −0.171 (the exact strong-USD year a leg needs = the WORST)**, 2019 −0.100 vs 2016 +0.173/2017 +0.196 — alternating, no regime coherence.

## Verdict + diagnosis
**KILL** (§5d cheap-kill: base coin-flip-continuation; the one real gradient is inverted AND ≤+0.017 ATR sub-cost; timing non-load-bearing; coherent cell threshold-unstable + per-year noise with 2018 negative → §5f does not bite — nothing clears cost to score).

**Diagnosis.** Even the M1 intrabar path — finer than arc-1050's bar shape — does NOT separate forced from informed displacement. Worse for the hypothesis, the within-bar retrace gradient runs the *opposite* way: a bar that already snapped back intrabar has **released** its over-extension and then drifts on; a bar that closed at its extreme still has it "loaded" and reverts marginally — a within-bar mean-reversion that *consumes* the next-bar reversion, not a forced-flow tell. The signal is tiny and sub-cost either way.

## NEW lesson
**The "price-trajectory proxy for forced flow" is dead at EVERY resolution — intrabar included.** arc 1050 closed it at bar-SHAPE (H4 range/vacuum); arc 1052 extends the closure to the M1 intrabar PATH (retrace + extreme-timing). A price-path classifier cannot manufacture the forced-vs-informed split that the survivors (gap/me/fbr) get *for free* from an **identifiable institutional flow** (weekend settlement gap / WMR-fix rebalancing / stop-run at a structural pivot). Sharpens the unified theory: tradeable reversion requires a NAMED inelastic counterparty, not a finer price-shape/timing feature — going to higher data resolution does not reveal a hidden order-flow edge (mirrors 1031's "go finer → variance collapses faster than cost", now on the directional/classifier axis). The intrabar lane is the last untouched data resolution; closing it confirms the OHLC-only charter is genuinely mined out (only the operator-gated macro/options unlocks could re-light the EDGE term — `NEEDS_ENABLEMENT.md`).

Operative frontier unchanged: lever = operator path-A gate-governance call (1032); honest deploy object = {me_long+fbr} 2-way (1045/1046/2045); deployable-system count = 0.
