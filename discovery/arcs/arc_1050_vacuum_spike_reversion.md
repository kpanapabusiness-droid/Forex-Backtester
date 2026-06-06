# arc 1050 — liquidity-vacuum → spike REVERSION (fade a displacement that follows a narrow-range bar)

**Chat:** 1000s | **Range:** 1000-1999 | **Date:** 2026-06-06
**Disposition: KILL** (obs cheap-kill — the spike CONTINUES not reverts; coin-flip; the vacuum conditioner is non-load-bearing; not 2015/2018-positive)

---

## (a) Log read / synthesis

Read DISCOVERY_PROTOCOL, the full DISCOVERY_LOG Tier-1 ledger (arcs 0→3022, ~90 rows across 1000s/2000s/3000s), LESSONS, TOOL_REGISTRY. State of the corpus:

- **4 PORTFOLIO components:** gap-fill (1006, JPY-cross H4), me_long (1011, USD-major D1), fbr (1013, USD-major H4, strongest), me_short (1019, USD-major D1, first robustly-2018+).
- **The portfolio route is structurally closed.** arc 3021 proved path-B (densification) cannot satisfy the per-year AFP gate at realistic residual correlation (ρ≥0.1 → P(AFP) plateaus, never 0.9 at any N); arcs 2016/2017/2019/1023 proved the book's AFP failure is a **measurement-floor / gate-resolution artifact** (a sound ~3-bet mean-positive PORTFOLIO whose every-calendar-year gate trips on within-noise single-leg dips). The lever is the operator's gate-governance call (path-A).
- **Honest-exit audit (1042–1046)** collapsed the deploy object: under §5f honest nested exits + frozen OOS, the 4-way book's mean halves (+0.59%→+0.27%), significance is lost (t 2.16→~1.3), ENB halves (3.32→~1.8), and OOS it collapses to **me_long-solo** (fbr's IS-diversification dies OOS). §11 programme COMPLETE (signal/outcome/cost all independently verified honest).
- **The 2018 leg is unfound across 15+ routes** (structure 1014/2009/2011/3011, trend 3010, flow 1016/2013, vol 3012, continuation 2012, carry 1017, rel-value 2010, fiscal 2026, cross-universe 1022/3018, all fbr-refinements 2014/2020/1025/3013/1040/3020). fbr-2018 is mechanism-intrinsic/entry-unconditionable across 5 axes.
- **Closed ground:** shallow single-condition directional (momentum/breakout/MR/trend, long+short, H1/H4/D1/W1, majors+crosses), session structure (3016/1026/1047), calendar flows beyond month-end WMR (weekly 3015, NFP 1048, gotobi 1008, round-number 1010, fiscal-end 2026), triangulation (3005/1027/2023).
- **Binding design constraint (arc 2026):** the survivors (gap/me/fbr) key off a **SURPRISE displacement, NOT a known calendar date** — known-date flows fail via (i) sub-cost or (ii) priced-in/front-run. A new mechanism must combine a surprise displacement with a documented forced/mechanical flow.

## (b) Idea (observe, don't guess — log-dry, examined data directly)

Calendar-flow lane is closed (2026). Survivors need a **surprise displacement + forced flow + reversion**. The one surprise-mechanical-reversion the corpus has NOT tested: a **liquidity-vacuum spike**. *Because:* a large displacement bar that fires immediately after an unusually QUIET (narrow-range, low-participation) bar is more likely a **thin-book / forced mechanical move** (a stop-cascade into a vacuum) than an informational move → it should over-shoot and revert when normal liquidity returns. This is the equity overnight-inventory-reversal logic (arc 1026) re-aimed at the intraday liquidity *gradient* rather than a session close. Distinct from arc 1001 (vol-contraction *breakout* — which traded the CONTINUATION and died at triage); here I FADE the spike and use the prior-bar vacuum as the mechanical tell. Surprise (spike direction unpredictable → not priced-in, escapes 2026's trap), forced-flow (thin book), reversion. Candidate decorrelated/regime-orthogonal leg if the vacuum isolates non-informational moves.

## (c)/(d) Characterization + cheap kill (OBSERVATION ONLY — no engine/P&L)

Driver `discovery/_disco1_work/arc1050_vacuum_spike_obs.py`. Mid-OHLC, Wilder(14) ATR shift1, IS 2010–2020, 10 pairs (7 USD majors + AUDJPY/EURJPY/GBPJPY). Definitions: **vacuum** = prior bar TrueRange/ATR in the bottom tercile (≤0.713); **spike** = current-bar body `(close−open)/ATR ≥ threshold`; **fade** = short an up-spike / long a down-spike; measure the fade-direction forward drift over K=6 H4 bars, with a NON-vacuum spike control.

**Result — the fade is DEAD; the spike CONTINUES; vacuum is non-load-bearing:**

| spike≥ | n | fade mean drift (ATR) | median | frac+ | NON-vacuum control mean | control frac+ |
|---|---|---|---|---|---|---|
| 0.75 | 9179 | **−0.028** | 0.000 | 0.499 | −0.026 | 0.500 |
| 1.0 | 4807 | **−0.040** | +0.008 | 0.502 | −0.040 | 0.499 |
| 1.5 | 1531 | **−0.083** | −0.042 | 0.488 | −0.056 | 0.496 |

- **Fade drift NEGATIVE at every threshold** = the displacement CONTINUES in its own direction; fading it loses. frac+ ≈ **0.499–0.502 (coin-flip)**.
- **Vacuum conditioner decisively NON-load-bearing:** vacuum and non-vacuum controls are essentially identical (spike≥1.0: −0.040 ≡ −0.040). The prior-bar narrow range tells you nothing about the spike's reversion.
- **Bigger spikes continue MORE** (−0.083 at ≥1.5) — the closed-ground "bigger moves continue" texture (arc 1048).

**Per-pair / per-year (spike≥1.0, vacuum), fade drift:**
- Per-pair only **2/10 positive** (USDCAD +0.13, AUDUSD +0.01 = noise); 8/10 negative.
- Per-year **2015 −0.099 / 2018 −0.192 both NEG** — the spike continues *hardest* in exactly the strong-USD trend years a 2018 leg needs (momentum, the worst possible sign). 2016 +0.111 is the lone positive (1/11 yrs), inconsistent with 2015/2018 → isolated noise.

## (e)–(h) Diagnose / validate / council

Base is a **coin-flip with a non-load-bearing conditioner** → §5d cheap-kill; §5f exit-menu step does NOT bite (it only bites for a non-coin-flip entry that beats the null or shows gross drift — here there is neither). No engine, no null baseline, no council spent. OOS never touched.

**Diagnosis.** A *constructed* "mechanical" displacement (narrow-range → spike) does NOT isolate the forced-flow reversion the survivors capture — at H4 it behaves as **generic momentum** (continues, coin-flip, worst in trend years). The "liquidity vacuum → thin-book → forced → reverts" intuition is FALSE at the H4 scale: a vacuum-then-spike is informational/momentum, not a thin-book artifact — consistent with arc 1026 (FX never closes → no warehoused-inventory price concession to mean-revert; the intraday liquidity gradient is efficient). The vacuum proxy is the wrong instrument for "forced vs informational."

## (i) Verdict + lesson

**KILL** (obs cheap-kill). Closes the **liquidity-vacuum spike-reversion** lane (the last clean surprise-mechanical-reversion construction the calendar-flow closure left open). Components UNCHANGED.

**NEW lesson — a surprise displacement is necessary but NOT sufficient (the OTHER half of arc 2026's principle).** arc 2026 showed a *known-date* forced flow fails (priced-in); this shows a *surprise* displacement that is NOT tied to a documented forced/mechanical flow ALSO fails — it is just momentum (continues, coin-flip, trend-year-negative). The survivors (gap/me/fbr) need BOTH: a surprise displacement AND a documented forced mechanism (weekend settlement gap / WMR-fix rebalancing / stop-run at a structural pivot). A self-constructed "liquidity/vacuum" proxy supplies the surprise but not a real forced flow, so it collapses to closed-ground momentum. Don't proxy "forced flow" with a price-shape conditioner; it must be an identifiable institutional mechanism.

Operative frontier unchanged (operator path-A gate-governance call is the lever, arcs 2019/3021; the honest deploy object is me_long-solo, arc 1046). No canonical change, no FLAG.
