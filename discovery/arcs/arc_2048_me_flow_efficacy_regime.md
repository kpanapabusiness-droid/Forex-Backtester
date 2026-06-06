# arc_2048 — Month-end flow-EFFICACY regime persistence (can the WMR reversion's own recent efficacy separate me_long's dead 2014/15/16 block?)

**Chat:** 2000s · **Date:** 2026-06-06 · **Verdict:** KILL (obs cheap-kill, §5d) · **Disposition:** KILL

## Log synthesis (step a — fresh eyes, honest-era only)
Pulled main; read DISCOVERY_PROTOCOL (followed it), the full Tier-1 ledger (arcs 0→2047) + recent Tier-2 (the honest-exit deploy thread 1042→1046 / 2040→2046, the §11 audits 2034→2039, the exploratory cheap-kills 2026→2032/2047), LESSONS, TOOL_REGISTRY. State of the corpus:
- **The honest-exit deploy thread (1042→1046, 2040→2046) just reframed the whole programme.** Under honest §5f nested-WFO exit selection, the 4-component reversion book (gap 1006 · me_long 1011 · fbr 1013 · me_short 1019) loses ~half its advertised deploy mean, all its significance (book t 2.66→≤0.96), and ~half its effective bets (ENB 3.32→~1.8). **gap + me_short FLIP mean-NEGATIVE; fbr −40% (stays +); only `me_long` is exit-ROBUST.** On the FROZEN holdout the whole "portfolio" collapses to **me_long-SOLO** (~+0.4%/yr, 5/6 OOS+, Sharpe ~0.5, borderline-significant, NOT all-folds-positive, vehicle-infeasible per 2033/2045). fbr's IS-diversification dies OOS. Signal+outcome+cost layers §11-verified honest end-to-end.
- **me_long is now the de-facto sole survivor** — yet, unlike fbr (attacked toward a solo PASS across ~10 arcs: 2014/2017/2020/2029/2030/2031/3013/3020/1025/1040), me_long's solo-PASS potential was attacked far less and never post-reframe: 1012 (exit menu, 0 AFP), 1029 (calendar density, worsens), 2024 (full-month window, 2018 contaminates), 2018 (cross-sectional, fails 2018). Its binding negative folds = the contiguous **2014/2015/2016 strong-USD block** (1012); every PRICE-trend regime gate to separate it FAILED (2014 SMA-slope, 1012 trend-filter).
- **Lever to deployment = operator path-A gate-governance call** (1032/2019/3021); the autonomous leg-hunt is structurally closed for a 5th component, but improving the *sole survivor* toward a standalone PASS would bypass the book-gate stalemate entirely.

## Idea (step b — log-seeded, with a mechanism)
The ONE regime detector never tried on me_long: **the flow's OWN RECENT EFFICACY.** *Because:* the WMR month-end rebalancing flow's tradeable strength tracks (i) the standing stock of cross-border equity hedges (slow-moving) and (ii) whether the prevailing trend is currently overrunning it (also slow-moving) — both SLOW states → the month-end reversion's realized strength should be **positively autocorrelated month-to-month**, and a month where last month's month-end reversion FAILED signals the flow is being overrun (a strong-trend regime) → **skip the current month.** This is a STATE conditioner (the edge's own recent outcome as the regime tell), categorically distinct from the price-trend regime gates that died (2014), and aimed squarely at separating the 2014/15/16 dead block → a me_long solo all-folds-positive PASS.

**Falsifiers (cheap-kill, §5d):** (1) month-to-month autocorr of the cross-sectional reversion coefficient ≈ 0 (within ~2SE) → efficacy not persistent → no regime to detect; (2) prior-month efficacy does NOT separate the 2014/15/16 block. Either → KILL; only BOTH passing → §5f engine.

## What I did (steps c–d — cheap-kill observation)
Population = D1, 7 USD majors (me_long's universe), 2010–2026. Driver: `discovery/_disco2_work/arc_2048_me_flow_efficacy_obs.py` (scratch). Ex-ante throughout (canonical `_month_end_into_move` + `observe_long_capture`):
- **Dense monthly flow-efficacy** (the regime proxy, every month-end, unconditional, all 7 pairs): `into = (close[me]−close[me−2])/ATR`, `after = (close[me+2]−close[me])/ATR`. Two efficacy measures per month: `rev_coef = −corr(into, after)` across the ≥4 pairs that month (positive = reversion active; ≤0 = continuation/overrun), and a magnitude-aware `fade_ret = mean(−sign(into)·after)` ATR (the realised return of a month-end fader). 195 months.
- **Autocorrelation** of each (falsifier 1).
- **Tradeable me_long fires** (`into ≤ −1.0 ATR`, n=182): honest +1R capture + 2-bar drift, joined to the PRIOR month's efficacy (shift the monthly series by 1), split prior-ON vs prior-OFF, then the decisive cut WITHIN the 2014/15/16 block (falsifier 2).

## What happened — FALSIFIED both ways
**Falsifier 1 — efficacy is NOT persistent (sub-2SE autocorrelation):**

| measure | lag-1 autocorr | ~2SE band (n=195) | mean |
|---|---|---|---|
| `rev_coef` | **+0.060** | ±0.143 | +0.119 |
| `fade_ret` | **+0.124** | ±0.143 | +0.123 ATR |

Both autocorrelations sit INSIDE the ±0.143 white-noise band → the month-to-month variation of the WMR reversion's strength is statistically indistinguishable from noise. The +0.124 on `fade_ret` is suggestive of faint persistence but cannot be leaned on (< 2SE).

**Mechanism partly real but noisy.** Dead 2014-16 mean `fade_ret` −0.058 vs good other-IS +0.082 — the block IS lower-efficacy on average, validating the *direction* of the mechanism. But per-year it is noisy and does NOT map cleanly to the binding folds: **2015 is positive (+0.079, rev_coef +0.110)**; only 2014 (−0.047) and **2016 (−0.206)** are low. (Corroboration aside: OOS 2022-2026 efficacy is strongly positive +0.18→+0.45, consistent with me_long's 5/6 OOS+ in 1046.)

**Falsifier 2 — prior-month efficacy gives a pooled lift but FAILS to separate the dead block:**

| split | n | capture | 2-bar drift (ATR) |
|---|---|---|---|
| all tradeable fires (base) | 182 | 0.5110 | +0.188 |
| prior `fade_ret` > 0 (flow ON) | 100 | 0.5300 | +0.279 |
| prior `fade_ret` ≤ 0 (flow OFF) | 82 | 0.4878 | +0.076 |
| **dead 2014-16 & prior-ON** | **14** | 0.5714 | **−0.074** |
| **dead 2014-16 & prior-OFF** | **14** | 0.5000 | +0.002 |

The pooled prior-ON lift (cap 0.511→0.530, drift +0.188→+0.279) is exactly the modest re-weighting a +0.12 autocorr predicts — it concentrates fires into already-good months. But **within the binding 2014-16 block the conditioner does nothing**: prior-ON gives a higher *capture* (0.571) yet a **NEGATIVE drift** (−0.074) — capture and drift disagree (the take-the-loss-label-vs-forward-move split seen across the corpus), and n=14/cell is squarely regime-luck (the 1017/3010/2031 disqualifier). The dead block stays drift-≤0 regardless of prior efficacy.

## Why it fails (the mechanism)
The WMR flow's *tradeable strength is not a persistent, slow-moving, detectable regime* — its month-to-month variation is ≈ noise (autocorr ~0), so the edge's own recent outcome carries no usable forward information. The 2014/15/16 block is NOT a "flow-dormant" state that persists and can be read off last month-end; it is per-month idiosyncratic strong-USD overruns, un-foreshadowed by the prior month's reversion. This **extends arc 2019's lesson** ("binding folds are a REGIME property, not entry-noise — can't lift them by sharpening entry quality") to a new conditioner class: even a STATE gate built from the edge's *own recent efficacy* can't separate them, because the efficacy itself isn't autocorrelated. It also re-confirms arcs 2016/2017 (me_long folds are within-noise → so is their month-to-month efficacy) and 1029/2024 (any me_long conditioner that re-weights toward good months thins the pool without touching the binding folds).

## Verdict + what this closes
**KILL (obs cheap-kill, §5d).** Both falsifiers trip: efficacy autocorrelation is sub-2SE, and prior-month efficacy fails to separate the binding 2014-16 block (the only place it would matter). §5f does NOT bite — the conditioner does not produce a new all-folds-positive candidate (it leaves the dead-block fires drift-negative and thins the pool 182→100, worse fold resolution per 1029/2024/2030); me_long is already engine-validated PORTFOLIO and is UNCHANGED. No engine / null / council spent (matches 2016/2027/2028/2029/2047). Closes the **flow-self-efficacy / state-persistence** conditioner on the sole survivor — the last categorically-new regime-detection axis for me_long (after price-trend 2014, calendar-density 1029, full-month window 2024, cross-sectional 2018). The me_long-solo-PASS route is now closed on the regime/state axis as well as the exit (1012) and entry-window (2024) axes; its 2014/15/16 block is regime-intrinsic and un-conditionable from the flow's own history.

**Threads.** None on the flow-efficacy axis. me_long stays the lone exit-robust PORTFOLIO survivor (1046), not a solo PASS; the 2014/15/16 block is intrinsic. Operative deployability lever remains operator **path-A (gate-governance)** on the existing mean-positive book.

**Tooling:** no new BUILT tool — canonical `Panel.from_pairs`, BUILT `_month_end_into_move`, `observe_long_capture` only; no TOOL_REGISTRY append (observation was inline scratch, matching 2026/2027/2047). **FLAGS:** none (no canonical change). Components UNCHANGED (all 4 PORTFOLIO; deploy core = me_long-solo per 1046). Deployable-system count = 0.
