# arc 1051 — DAILY 16:00-London WM/Reuters fix dislocation-reversion (does the fix edge thicken to a standalone?)

**Chat:** 1000s | **Range:** 1000-1999 | **Date:** 2026-06-06
**Disposition: KILL** (obs cheap-kill — corr≈0, faint horizon-growing fade ~+0.04 ATR but ~8× SUB-COST on H1; not a clean 2015/2018 leg)

---

## (a) Log read / synthesis

(Carried from arc 1050, same chat — same corpus state.) 4 PORTFOLIO components (gap 1006 / me_long 1011 / fbr 1013 / me_short 1019); portfolio route structurally closed (path-B proof 3021; AFP failure = noise-floor artifact 2016/17/19/1023); honest deploy object collapses to me_long-solo OOS (1042–1046); 2018 leg unfound across 15+ routes; the lever is the operator path-A gate call. Survivor DNA (sharpened 1050 + 2026): a tradeable forced-flow edge needs a **SURPRISE displacement AND a documented institutional flow** — known dates are priced-in (2026), self-constructed price-shape proxies are just momentum (1050). **One open route remains named:** arc-2017 **option B** — a component THICK enough that per-year folds RESOLVE; closed so far on 4 constructions (fbr-depth 1025, me-cross-sectional 2018, thick-by-design 1026, full-month-window 2024).

## (b) Idea (observe, don't guess)

`me` (1011/1019) works via the **WMR 16:00-London month-end fix** rebalancing flow — but that fix runs **EVERY day**, not just month-end. arc 3008 tested the **month-end** fix on H1 and found it sub-cost; the corpus has NEVER tested the **daily** (all-days) fix as a standalone. *Because:* passive fix-tracking funds rebalance to the daily 16:00 fix → an abnormal pre-fix push (a SURPRISE displacement driven by fix-hedging) should over-shoot and revert post-fix. This is the one **THICK-by-design** (≈250 fires/pair/yr) expression of the proven fix mechanism — the arc-2017 option-B candidate for the fix family. Documented institutional flow + surprise push → passes the 1050/2026 necessary-condition bar.

## (c)/(d) Characterization + cheap kill (OBSERVATION ONLY — no engine/P&L)

Driver `discovery/_disco1_work/arc1051_daily_fix_reversion_obs.py`. H1 mid-OHLC, Wilder(14) ATR shift1, IS 2010–2020, 8 H1-cached pairs (EURUSD/GBPUSD/AUDUSD/USDJPY + EURJPY/GBPJPY/AUDJPY/EURGBP). Fix bar = the **15:00–16:00 London** H1 bar (DST-robust via `Europe/London`), mirroring arc 3008's construction but for **all days**. push = `(close−open)/ATR` of the fix bar; post-fix reversion = fade-direction forward drift over k∈{1,2,3} next H1 bars; cost gauge = `1.5·spread_close/ATR`.

**Result — real but faint, and DECISIVELY sub-cost (n=22,799 fix bars, ~259/pair/yr):**

| horizon | corr(push, fwd) | push≥1.0 fade drift | frac+ |
|---|---|---|---|
| k=1 | +0.014 | −0.021 | 0.504 |
| k=2 | −0.006 | +0.014 | 0.518 |
| k=3 | −0.012 | **+0.040** | 0.525 |

- **corr(push, post-fix drift) ≈ 0** at every horizon (+0.014 / −0.006 / −0.012) — no real reversion signal, like NFP (1048) / triangulation (3005).
- A FAINT horizon-growing fade exists (k=3, push≥1.0: **+0.040 ATR**, frac+ 0.525 barely above coin-flip) — but the **H1 round-trip spread cost alone is ~0.308 ATR** (mean 1-leg ~0.154 ATR; commission + slippage on top) → the signal is **~8× SUB-COST**.
- **Per-year (push≥0.5, k=2):** 2016 +0.083 is the lone standout (inconsistent); **2018 +0.003 ≈ 0, 2015 +0.012 weak**; 2012/2013/2020 negative → not a clean 2015/2018 leg, and noise-level even gross.

## (e)–(h) Diagnose / validate / council

Base is corr≈0 / frac+ ≈ coin-flip / signal ~8× sub-cost → §5d cheap-kill; §5f exit step does not bite (no non-coin-flip, above-cost entry to optimize). No engine/null/council. OOS untouched.

**Diagnosis.** The daily fix dislocation IS real (a faint reversion appears at k=3 as the push grows) but is **far too small** — the daily fix's rebalancing notional is a tiny fraction of the month-end fix, and the H1 cost wall (documented 1008/1010/3008/3016) buries it ~8×. This pins down WHY `me` is tradeable: not the fix mechanism per se, but the **MONTH-END rebalancing MAGNITUDE** (large enough to clear cost at the D1 scale, ~0.05–0.10R hurdle). The daily fix has the mechanism without the magnitude.

## (i) Verdict + lesson

**KILL** (obs cheap-kill). Closes **option-B for the fix family** — the daily fix does not thicken `me` into a fold-resolving standalone (the **5th** option-B construction closed: fbr-depth 1025, me-cross-sectional 2018, thick-by-design 1026, full-month-window 2024, now **daily-fix 1051**). Components UNCHANGED.

**NEW lesson — the surviving fix edge (`me`) is MAGNITUDE-gated, not mechanism-gated.** The same WMR-fix reversion mechanism runs daily but is ~8× sub-cost on H1; only the MONTH-END instance carries enough forced-rebalancing notional to clear cost (at D1). Thickening a forced-flow edge by firing its mechanism more often (daily vs month-end) trades magnitude for frequency and drops below the cost wall — the corpus-wide "real FX edges are intrinsically RARE forced-flow dislocations" theory now confirmed on the fix axis: you cannot keep the edge and add the trades. Extends 3008 (month-end fix sub-cost on H1) to the all-days fix, and complements 1025's "fbr depth↔strength coupled" — both the structural (fbr) and the flow (me) survivors resist thickening, for the same magnitude-vs-frequency reason.

Operative frontier unchanged (operator path-A; deploy object me_long-solo, 1046). No engine/null/council; no canonical change; no FLAG; OOS untouched.
