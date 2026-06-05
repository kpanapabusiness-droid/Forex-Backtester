# Arc 1027 — M1: driver-shock-CONDITIONAL cross-rate triangulation residual

> **Arc id:** 1027 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **KILL (cheap-kill at observation)** — conditioning the triangular residual on a large
> D1 driver-leg shock does NOT lift its next-H4-bar directional predictability above efficiency: pooled
> `corr(driver_shock, next-bar residual) = 0.0135` (within-triangle-standardized 0.0144) ≈ arc 3005's
> **unconditional** ~0.01, well below the M1 falsifier (~0.05), and the per-triangle correlations are
> **sign-INCONSISTENT** (+,−,+,+) — the tell of noise, not a shared lag mechanism. The residual stays
> sub-spread even at shock bars (median 0.168 bp; >0.8 bp on only 16.5%).
> **Idea source:** `DISCOVERY_DIRECTION.md` MENU item **M1** (the strategist council's #1-EV `explore-now`
> candidate) — the one angle arc 3005 left open: it killed the residual **unconditionally and at H4 only**.

Scored by inspection of the canonical panel (real bid/ask, H4/D1 5ers_eet). No pool/engine — the
observation is decisive (§5d). OOS never touched.

## (a) Log read + synthesis (FRESH EYES, honest-era only)

Pulled `origin/main`. Read DISCOVERY_PROTOCOL, the full DISCOVERY_LOG (both tiers, arcs 0–3021),
LESSONS.md, TOOL_REGISTRY.md, DISCOVERY_DIRECTION.md. State of the corpus:

- **Mature / comprehensively closed.** ~70 honest-era arcs. Single-leg shallow directional (long AND
  short, every TF/pair), calendar, microstructure/H1-flow, regime conditioning, naive relative-value,
  convexity/trend, carry — all dead (LESSONS "Closed ground"). Shorts merged + verified (PR #273) and
  exhaustively explored (1014/2009/2011/3011 structure, 3010 trend, 1016/2013 up-gap flow, 3012 vol,
  2010 rel-value, 1017 carry-unwind) — no short revived a directional/structural base.
- **The book.** 4 PORTFOLIO components exist (gap-fill 1006, me_long 1011, fbr 1013, me_short 1019). The
  4-way book is a sound ~3-bet (ENB 3.32), mean-positive (t=2.66, arc 1023), temporally-robust (arc 2021)
  PORTFOLIO that fails ONLY the per-year all-folds gate. Arcs 2019 (ENB/tail) and **3021 (portfolio-math
  proof)** establish a 5th reversion leg / densification CANNOT make it all-folds-positive — **path-B is
  closed; the lever is the operator's gate-governance call (path-A).**
- **Where the open frontier is.** `DISCOVERY_DIRECTION.md` (the strategist generative council, 2026-06-05)
  ranks a THIN, conditional `explore-now` MENU. Its #1-EV item is **M1**: arc 3005 killed the cross-rate
  triangulation residual but **only unconditionally and only at H4** — a *driver-shock-conditional* lag was
  never tested. M1 is one-leg-expressible, cheap-obs-first, and attacks a NAMED unconditional corpus result.
  Picked it as arc 1027.

## (b) Idea + because

`EURJPY ≡ EURUSD × USDJPY` is a triangular identity; the residual `r = log(EURJPY) − log(EURUSD) −
log(USDJPY)` is pinned ≈ 0 by arbitrage (arc 3005: median ≈ 0, |r|>spread 1.3–6.1%, fwd-conv corr ≈ 0.01).
**Because:** a large directional shock in a *driver leg* (EURUSD on the D1 close) *forces* the dependent
quoted cross to move; IF the JPY market re-prices the quoted cross with a lag, the **next H4 bar** would
carry a transient residual signed by the shock — even though the *same-bar* H4 residual (3005) averages 0.
You would ride a transient identity dislocation from asynchronous re-pricing, not forecast price.

**Falsifiable prediction (M1):** `corr(driver_shock, next_bar_cross_residual) > 0.05`, materially above
3005's ~0.01. **Falsifier:** ≤ ~0.05 → the cross re-prices within the bar (efficient) → dead. A *real* lag
must also be **same-signed across all four XXXJPY triangles** (shared USDJPY leg, shared lag structure).

## (c)/(d) Observation → verdict (cheap-kill)

Four product-form triangles (driver = XXXUSD, cross = XXXJPY = XXXUSD·USDJPY), IS 2010-2020, H4 residual
(mid closes, bp), D1 driver shock `z = ΔcloseMid / Wilder-ATR(14).shift(1)`, re-keyed (no-lookahead) to the
**first H4 bar after the D1 close** (D1 ⊂ H4 on the shared 22:00-UTC EET boundary). Shock = `|z| > 1.5`.

| triangle | n_shock | resid_med (bp) | shock_resid_std (bp) | corr_uncond | **corr_shock** | conv_corr_shock |
|---|---|---|---|---|---|---|
| EURJPY=EURUSD×USDJPY | 103 | −0.012 | 1.086 | 0.007 | **+0.112** | +0.034 |
| GBPJPY=GBPUSD×USDJPY | 120 | −0.019 | 1.077 | −0.049 | **−0.196** | +0.262 |
| AUDJPY=AUDUSD×USDJPY | 100 | −0.041 | 1.306 | 0.008 | **+0.029** | +0.104 |
| NZDJPY=NZDUSD×USDJPY | 100 | −0.029 | 1.245 | 0.030 | **+0.152** | −0.076 |

**Pooled (n=423 shock bars):** `corr(z, next-bar resid) = 0.0135`; within-triangle-standardized `0.0144`;
|resid| median 0.168 bp / mean 0.530 bp; **>0.8 bp (single-cross spread) on only 16.5%** of shock bars.

**Three facts kill it at observation:**
1. **Directional predictability does NOT clear the bar.** Pooled corr 0.0135 (standardized 0.0144) ≈ 3005's
   unconditional ~0.01, far below the 0.05 falsifier. The shock conditioning buys nothing.
2. **The per-triangle corrs are sign-INCONSISTENT (+,−,+,+)** and each within ~2 SE of 0 (n≈100 → SE≈0.10).
   A genuine asynchronous-repricing lag would be the SAME sign across all four XXXJPY crosses (shared
   USDJPY leg). Mixed signs = sampling noise, not a mechanism (and pooling them — raw or standardized —
   washes to ≈0, confirming no shared effect).
3. **Still sub-spread.** Even at large-shock bars the residual exceeds a single-cross spread (~0.8 bp) only
   16.5% of the time; the modest amplitude bump (std 1.08–1.31 bp vs unconditional 0.85–1.11) is a
   *variance* effect, not a *directional* one — i.e. it belongs to L1 (second-moment), not M1.

**Apparatus validated against arc 3005:** unconditional convergence corr reproduces 3005 (0.00–0.04 here vs
its 0.009–0.016), residual median ≈ 0 and std ~0.85–1.1 bp match — confidence the measurement is faithful.

**Mechanistically expected FAIL:** triangular arb is the most-policed FX relationship and the next H4 bar is
~4 h after the D1 close — ample time for the quoted cross to catch up. M1's hoped-for lag is sub-H4 (tick),
below the apparatus and the cost floor — exactly 3005's reasoning, now confirmed to survive driver-shock
conditioning.

## Final verdict — KILL (cheap-kill at observation)

The triangulation thread is now closed **BOTH unconditionally (arc 3005) AND driver-shock-conditionally
(arc 1027).** No tradeable directional dislocation at H4, conditioned or not. §5f does not bite (the angle
is falsified at observation — no non-coin-flip entry to run an exit menu on). No engine / null / council
spent; OOS never touched. The only untouched triangulation sub-thread is **L1** (the residual's *second
moment* / OU amplitude at finer resolution) — but it walks straight into the H1 cost wall that killed the
whole microstructure cluster (gotobi / round-number / WMR-fix / session-break), so it is a low-EV door.

## Lessons (candidate for LESSONS.md compression)

1. **No driver-shock-conditional triangulation lag at H4 either.** Conditioning the EUR/GBP/AUD/NZD-vs-JPY
   residual on a large D1 driver-leg shock leaves next-bar directional predictability at corr ≈ 0.014 (≈ the
   unconditional 0.01) with sign-inconsistent per-triangle correlations → no shared asynchronous-repricing
   mechanism. Extends arc 3005's unconditional closure to the conditional axis; the triangulation thread is
   directionally closed at H4.
2. **Sign-consistency across structurally-identical triangles is the decisive noise test.** Four XXXJPY
   crosses share the USDJPY leg; a real lag must be same-signed across all four. Mixed signs (+,−,+,+) within
   sampling error = the per-triangle nonzero corrs are noise, not a mechanism — a cheaper, sharper kill than
   any single-triangle number. (Re-usable: when a mechanism predicts a common sign across parallel
   instruments, test the sign agreement, not just the pooled magnitude.)
3. **A shock bumps the residual's VARIANCE, not its directional MEAN.** Shock-bar residual std rises modestly
   (~1.1–1.3 vs ~0.85–1.1 bp) while directional corr stays ≈ 0 — a clean separation of M1 (first-moment lag,
   dead) from L1 (second-moment amplitude, untested but cost-walled). Don't conflate amplitude with edge.

## Threads / what didn't help

- **Closed:** M1 (driver-shock-conditional triangulation residual, directional). Adds to 3005 (unconditional).
- **L1 (open, low-EV):** the residual second moment / OU amplitude at H1→M1 — genuinely untested but the
  ~1.3 bp amplitude vs ~1 bp cost is a knife-edge inside the H1 cost wall. A cheap multi-resolution amplitude
  observation could close the triangulation door fully (level + variance); expect cost to eat it.
- **Operative state unchanged:** the corpus's lever remains the operator's gate-governance call (path-A; arcs
  2019/3021). Components UNCHANGED (all 4 PORTFOLIO).

## Flags (code NOT merged)

None (engine/canonical untouched). New EXPERIMENT tool committed + registered:
`discovery/tools/triangulation_residual.py` (`triangle_log_residual_bp`, `d1_driver_shock`,
`shock_available_at_next_day`). Observation driver: `discovery/_disco1_work/arc1027_m1_triangulation.py`.

## Reproduction

`Panel.from_pairs([AUDJPY AUDUSD EURJPY EURUSD GBPJPY GBPUSD NZDJPY NZDUSD USDJPY], "H4", ...)` +
`Panel.from_pairs([AUDUSD EURUSD GBPUSD NZDUSD], "D1", ...)`, `histdata_root=C:\Users\panap\histdata_backup`,
`cache_root=data/cache`, `boundary_convention="5ers_eet"`; IS 2010-2020; residual via
`triangle_log_residual_bp(op="mul")`, shock via `d1_driver_shock` + `shock_available_at_next_day`, `|z|>1.5`.
Driver: `discovery/_disco1_work/arc1027_m1_triangulation.py`.
