# Arc 1031 — L1: triangulation residual SECOND moment (OU amplitude) at finer resolution

> **Arc id:** 1031 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-06
> **Final verdict:** **KILL (cheap-kill at observation) — the triangulation thread is now FULLY closed
> (level AND variance, all conditionings, all resolutions H4→M5).** The residual's OU amplitude does NOT
> become harvestable at finer resolution — it SHRINKS (σ H4 1.41 → H1 0.65 → M15 0.70 → M5 0.58 bp) while
> the one-leg round-turn cost is TF-INVARIANT (~3.0 bp), so a |z|>2 full-convergence harvest nets
> NEGATIVE at every resolution and every triangle (mean net_z2: H4 −0.24 → H1 −1.80 → M15 −1.65 → M5
> −1.88 bp; the sole non-negative cell, EURJPY H4 +0.22 bp, is 1/16 and within-noise). The fraction of
> bars where |residual| even exceeds the round-turn cost (the ceiling on a theoretically-profitable
> convergence) stays FAR below the menu's 5% bar and DECREASES finer: **H4 0.021 → H1 0.008 → M15 0.0066
> → M5 0.0063.** Half-life is sub-2-bars everywhere (H4 0.27 → M5 1.30) → the residual reverts within ~1
> bar = the arb is near-instantaneous (confirms 3005's "re-prices within the bar" for the second moment).
> **Falsifier met cleanly at every TF.**
> **Idea source:** `DISCOVERY_DIRECTION.md` MENU item **L1** — the LAST named-live explore-now thread.
> Arc 3005 killed the residual LEVEL unconditionally at H4; arcs 1027/2023 killed it driver-shock-
> conditionally at H4; all three closed only the FIRST moment and 3005 explicitly flagged the residual's
> std (1.23–1.58 bp) and sub-H4 as "out of apparatus scope." This arc closes the variance dimension.

Scored by a pure microstructure OBSERVATION (no engine): the BUILT `triangle_log_residual_bp`
(arc 1027) across H4/H1/M15/M5 panels (real bid/ask, 5ers_eet) for 3 JPY triangles + EURGBP, with the
OU half-life (AR(1)) and the residual amplitude charged against the honest FundedNext one-leg round-turn
cost. §5f does not bite (no entry clears cost → nothing to carry to the engine). OOS never touched.

## (a) Log read + synthesis

Pulled main (picked up the concurrent 2000s arc 2027 + cosim item-E validation). Corpus state unchanged
from arc 1030's read: the 4-component PORTFOLIO (gap/me_long/fbr/me_short) is mean-positive, cost-robust,
temporally stable, ~3 independent bets, but fails the per-year all-folds-positive gate below the
components' noise floor; **path-B (more legs) is quantitatively closed (3021), edge-hunting is closed, the
lever is the operator's path-A gate call.** My own arc 1030 closed O1 (inelasticity-state) on all three
collinear proxies and flagged **L1 as the one named-live MENU thread remaining.** The explore-now MENU is
otherwise tested/dead: M1 driver-shock residual (1027/2023 KILL), O1 (1029/1025/1030 KILL), Q1 peg-defense
(1028 KILL), G1 dollar-factor (covered by 2018), S1 take-the-loss-geometry (a modifier, tautological on
fbr). L1 is the last door; closing it completes the MENU.

FRESH EYES: pre-reset conclusions ignored; only honest-era log carries weight.

## (b) Idea + because

**L1 hypothesis.** Triangular arbitrage pins the residual `r = 1e4·(log(cross) − log(legA·legB))` MEAN to
≈0 (3005), but the residual is the difference of three ASYNCHRONOUSLY-updating quotes; its AMPLITUDE
spikes when one leg's quote is temporarily stale (session handovers, one-currency news, fix windows). It
should behave as a mean-reverting OU process around 0 — you harvest the AMPLITUDE (a convergence trade),
not the direction. **Because** 3005 reported a non-trivial residual std (1.23–1.58 bp) and |resid|>spread
on 1.3–6.1% of H4 bars but only tested the LEVEL and only at H4, the variance dimension at finer
resolution is genuinely virgin. **Falsifiable prediction:** as resolution rises H4→H1→M15→M5, the OU
half-life shortens; find a resolution where half-life < ~5 bars AND the amplitude exceeds the single-cross
ROUND-TURN cost on > ~5% of bars. **Falsifier:** if at EVERY resolution `frac(|resid| > rt_cost)` stays
below ~5% AND a |z|>2 harvest nets ≤ 0, the triangulation thread is FULLY closed (level AND variance).

Honest cost: FundedNext one-leg round-turn = 1.5×spread + 0.5pip/fill×2 slippage + $5/lot RT (0.5 bp).
No-lookahead: all metrics are descriptive on the contemporaneous residual series; a convergence from
residual R nets at most |R| − rt_cost, so |resid| > rt_cost is the (optimistic) ceiling on even a
theoretically-profitable trade.

## (c)/(d) Observation → verdict (cheap-kill)

Per (triangle × TF): residual std σ (bp), AR(1) φ → half-life (bars), median cross spread + round-turn
cost (bp), the 2σ full-convergence harvest of a |z|>2 entry, its net of cost, and the fraction of bars
where |resid| exceeds the round-turn cost (and, looser, the raw spread — to reproduce 3005).

**Per-TF means (across the 4 triangles):**

| TF | σ_resid (bp) | half-life (bars) | rt_cost (bp) | net |z|>2 harvest (bp) | frac(|r| > cost) |
|---|---|---|---|---|---|
| H4 | 1.41 | 0.27 | 3.07 | **−0.24** | **0.0210** |
| H1 | 0.65 | 0.80 | 3.09 | −1.80 | 0.0080 |
| M15 | 0.70 | 0.80 | 3.05 | −1.65 | 0.0066 |
| M5 | 0.58 | 1.30 | 3.03 | −1.88 | 0.0063 |

- **frac(|r| > cost) < 0.05 at every TF, and DECREASING finer** (0.021 → 0.006). The optimistic ceiling on
  tradeable opportunity never approaches the menu's 5% bar.
- **net |z|>2 harvest NEGATIVE at every TF / every triangle** (mean −0.24 to −1.88 bp). The one exception
  is EURJPY @ H4 (+0.22 bp) — 1 of 16 cells, within-noise, and a one-leg cross trade additionally carries
  directional exposure between entry and convergence that this gross lens ignores → not real.
- **The amplitude SHRINKS at finer resolution** (σ 1.41 → 0.58 bp) — the decisive physics (see diagnosis).
- **Half-life sub-2-bars everywhere** (0.27 → 1.30) → the residual reverts within ~1 bar (the arb is
  near-instantaneous; AR(1) φ ≈ 0.06 at H4). The menu HOPED a short half-life would mean tradeability — it
  instead confirms the residual re-prices within the bar (3005's first-moment finding, now for the second).
- **3005 reproduction:** `frac(|r| > raw spread)` at H4 = EURJPY 0.100 / GBPJPY 0.052 / AUDJPY 0.088 /
  EURGBP 0.051 — reproduces 3005's "1.3–6.1% of H4 bars" order. But vs the HONEST round-turn cost (~2.5×
  the raw spread) that collapses to 1.5–2.6% (H4) and < 1.3% finer. Even 3005's looser non-triviality
  evaporates once the real cost is charged.

## (e) Diagnosis (the why — §2)

**Why does the amplitude SHRINK at finer resolution (the opposite of "more staleness visible finer")?**
The residual's standing dispersion is dominated by ASYNCHRONOUS SAMPLING of the three legs' closes. Over a
4-hour bar, the three close prints can be separated by a long, variable interval during which one leg moved
and another did not yet → large residual. Over a 5-minute bar the three closes are far MORE synchronized
(less time for one leg to drift away from the identity within the bar) → SMALLER residual. So σ is largely
a sampling artifact of the bar width, not a sub-H4 OU process you can ride. Meanwhile the round-turn cost
is TF-INVARIANT (same instrument, same spread+commission+slippage). Net: the harvest (∝ σ) collapses
faster than the cost as you go finer, so the convergence trade gets STRICTLY WORSE at higher resolution —
the exact inverse of the menu's hope. The short half-life confirms the arb closes within the bar at every
TF; there is no resolution at which a tradeable amplitude opens up. (No council: the obs is decisive,
falsifier met at every TF — §5d, like 3005/1027/2023.)

## (f)/(g) Best version / validation

§5f does not bite: no entry clears cost at any resolution → there is no above-cost best-version to score on
the honest engine. No engine, no null, no council spent — a §5d cheap-kill. OOS never touched.

## (h) Council

Not invoked (cheap-kill at observation).

**Independent convergence (Arc-10 norm).** The concurrent 2000s arc **2028** tested the SAME L1 thread
with the same BUILT `triangle_log_residual_bp` and reached the SAME FALSIFIED verdict — at H1 (the
cost-wall TF). My arc spans H4→H1→M15→M5 and charges the cost-vs-amplitude net at each, which adds the
decisive **amplitude-shrinks-finer** physics (σ 1.41 → 0.58 bp while cost is TF-invariant) that closes the
door at EVERY resolution, not just H1. Two independent constructions, one verdict: the triangulation
residual's second moment is not harvestable.

## (i) Disposition + NEW lesson

**KILL (cheap-kill at observation).** Components UNCHANGED (all 4 still PORTFOLIO). **The triangulation
thread is now FULLY CLOSED** — level (unconditional 3005, driver-shock 1027/2023) AND variance/second-
moment (this arc), across all resolutions H4→M5. With L1 closed, **the explore-now MENU is exhausted**:
every named candidate is tested/dead. There is no autonomous edge-hunt left above the noise floor.

**NEW lesson.** The triangulation residual's AMPLITUDE is an artifact of asynchronous bar-sampling, not a
harvestable OU process: it SHRINKS at finer resolution (legs more synchronized within a shorter bar) while
the one-leg round-turn cost is TF-invariant → the convergence trade gets strictly WORSE finer, and the
sub-1-bar half-life means the arb closes within the bar at every resolution. "Go finer to find tradeable
variance" fails because amplitude collapses faster than cost. Generalizes the H1 cost wall (the whole
microstructure cluster — gotobi/round-number/WMR-fix/session-break — died on it) to the residual-variance
case. The deployability lever remains the operator's path-A gate-governance call (arcs 2019/3021/2016/2017/
1023/2021/3022).

**Frontier note.** With the MENU exhausted (M1/O1/L1/Q1/G1/S1 all dead) and path-B closed (3021), the
autonomous discovery programme has no positive-EV edge-hunt remaining. The next arcs are corpus-completing
(close residual doors, independent reproductions) rather than route-opening; the live decision is the
operator's gate-governance call, which is explicitly NOT an autonomous-chat action (§5g firewall — the
book's OOS stays unspent).

## Reproduction

`discovery/results/arc_1031_l1_triangulation_second_moment/observe_ou_amplitude.py` (H4/H1/M15) +
`observe_ou_m1.py` (M5; M1 is not a supported aggregation output, M5 is the finest — the monotone trend is
decisive). Triangles EURJPY/GBPJPY/AUDJPY (mul) + EURGBP (div) from `C:\Users\panap\histdata_backup`.
Outputs saved alongside (`output_h4_h1_m15.txt`, `output_m5.txt`). Base conda env (pyarrow). Reuses the
BUILT `triangle_log_residual_bp` (arc 1027); no new tool, no TOOL_REGISTRY append; no FLAGS.
