# arc 2023 — M1: driver-shock-CONDITIONAL cross-timeframe triangulation residual (DISCOVERY_DIRECTION menu)

**chat:** 2000s | **date:** 2026-06-05 | **disposition:** KILL (obs cheap-kill; no engine/null/council) | **passed:** N

## Step (a) — log read
Pulled main; STOP absent. Highest 2000s arc = 2022 → resume **2023**. Read protocol/log/LESSONS/registry + the freshly-landed **`discovery/DISCOVERY_DIRECTION.md`** strategist MENU (which independently corroborates my arc 2022: the leg-hunt route for the 4-way book is closed; the lever is the operator gate-call). The menu lists `explore-now` candidates that attack a *named unconditional* corpus result; **M1** is its freshest prior.

## Step (b) — idea + because
**M1 (menu, ranked #1 explore-now).** Arc 3005 killed the cross-rate triangulation residual but **only unconditionally and only at H4** (convergence corr ≈ 0.01, residual std 1.23–1.58 bp). `EURJPY ≡ EURUSD × USDJPY` is an identity; a large directional shock in a **driver leg** (EURUSD) *forces* the dependent cross to move. *Because:* if the quoted cross re-prices the identity with a **lag**, the next bar's identity residual should continue in the shock direction even though the same-bar residual is ≈0 on average — a transient identity dislocation, not a price forecast (one-leg expressible, cost ~cross spread). Falsifiable: `corr(driver_shock, next-bar cross residual) > 0.05` (materially above 3005's ≈0.01) → real lag; ≈0 → efficient/dead.

## Step (c) — method (CHARACTERIZATION only; cheap obs, no engine)
Script `_disco2000_work/arc2023_driver_shock_residual.py`. All H4, IS 2010–2020, 3 clean XXXUSD-driver triangles: EUR(EURUSD,USDJPY→EURJPY), GBP(GBPUSD,USDJPY→GBPJPY), AUD(AUDUSD,USDJPY→AUDJPY). EX-ANTE: the "D1 driver shock" is proxied by a 6-H4-bar (~1 day) driver log return / rolling-std(250), measured at **t-1** (predicts bar t; no lookahead, no D1/H4 alignment ambiguity), flagged |z|>1.5. Two next-bar residual measures: **(b) the pure identity residual** `idr = r_cross − r_driver − r_other` (≈0 by identity — arc 3005's residual in return space, the clean dislocation test) and **(a) the menu-literal** `r_cross − β·r_USDJPY` (flagged as conflating driver MOMENTUM, which is closed ground). Decisive = `corr(driver_shock[t-1], idr[t])` within the shock subset.

## Result — FALSIFIED at obs (efficient; extends arc 3005 to the conditional case)

| triangle | n_shocks | idr_std | cross spread | **corr(shock, next-bar idr)** | not-expl-by-USDJPY corr | directional (bp) |
|---|---|---|---|---|---|---|
| EURJPY | 2235 | 0.77 bp | 1.23 bp | uncond −0.000 / **shock −0.009** | +0.007 | −0.013 |
| GBPJPY | 2272 | 0.92 bp | 1.95 bp | uncond +0.000 / **shock −0.025** | +0.005 | −0.025 |
| AUDJPY | 2342 | 0.95 bp | 1.63 bp | uncond +0.001 / **shock +0.001** | −0.033 | +0.004 |
| **POOLED** | **6849** | — | — | **−0.012** | — | **−0.011 bp** |

- **The decisive conditional corr is ≈0** (pooled −0.012; all three within ±0.025) — statistically indistinguishable from arc 3005's unconditional ≈0.01. Conditioning on a large driver shock does **not** lift it above the ~0.05 falsifier.
- **No driver-momentum leak either:** the menu-literal "not-explained-by-USDJPY" residual is also ≈0 (+0.007/+0.005/−0.033) — consistent with closed-ground directional momentum being dead.
- **Directional continuation ~−0.01 bp**, ~100–200× below the cross spread (1.2–2.0 bp) — no harvestable amplitude.

## Verdict + meaning
**KILL (obs cheap-kill).** After a large driver-leg shock the quoted cross re-prices the triangular identity **within the same H4 bar** — there is no next-bar lag dislocation to ride. This **extends arc 3005's "residual ≈ 0" closure from unconditional to the driver-shock-CONDITIONAL case** (the exact angle 3005 never tested), and is the high-value closure the menu intended ("a death at the cheap-obs stage is a high-value closure"). The triangulation **first-moment** thread is now closed unconditionally AND conditionally. No engine/null/council spent (corr ≈ 0 + sub-spread directional → §5d cheap-kill, like 3005/3016).

## Threads
1. **Triangulation first moment is fully closed** (level — 3005; driver-shock-conditional — this arc). The only residual triangulation door left is the menu's **L1 — the residual SECOND MOMENT (OU amplitude) at finer resolution** (M1→H1 data), explicitly flagged by 3005 as out-of-apparatus-scope; but it walks into the H1 cost wall (knife-edge ~1.3 bp amplitude vs ~1 bp cost). A cheap multi-resolution OU observation would close it cleanly; not taken here (left for a chat with budget).
2. **Heuristic reinforced:** triangular arbitrage is the most-policed FX relationship; even a 1-day driver shock is fully absorbed by the next H4 bar. Identity-dislocation lead-lag is not an H4 edge.
3. The operative frontier conclusion is unchanged (arcs 2019/2022 + the menu §0.2): edge-hunting for the 4-way book is closed; the deployability lever is the operator's gate-governance call.

## Tooling / FLAGS
No new BUILT tool (pure observation on the canonical `Panel`). No canonical code touched; no FLAGs. Scratch driver `_disco2000_work/arc2023_driver_shock_residual.py` (reproducible).
