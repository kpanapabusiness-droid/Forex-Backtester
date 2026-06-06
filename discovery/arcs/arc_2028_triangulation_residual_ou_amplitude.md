# arc 2028 — L1: triangulation residual SECOND MOMENT (OU amplitude)

**Chat:** 2000s | **Date:** 2026-06-06 | **Range:** 2000–2999 | resumes after arc 2027.

## (a) Log read + synthesis
Continued from arc 2027 (same session): protocol, full Tier-1 ledger, LESSONS, TOOL_REGISTRY,
DISCOVERY_DIRECTION read. No `discovery/STOP`. Pulled main (brought in concurrent co-sim Item E build —
a measurement, not a new component). Corpus state unchanged: 4-component PORTFOLIO book, mean-positive /
cost-robust / ~3-bet but not all-folds-positive; path-B math-closed (3021), 5th-leg structurally closed
(2022); lever = operator path-A.

## (b) Idea (the *because*)
The **last genuinely-untested `DISCOVERY_DIRECTION.md` menu door (L1).** Arc 3005 killed the cross-rate
triangulation residual *level* (mean ≈ 0, fwd-convergence corr ≈ 0.01); arcs 1027/2023 killed its
*driver-shock-conditional first moment*. Both attacked the residual's DIRECTION. Its **second moment —
the OU amplitude — was never traded.** Triangular arb pins the residual *mean* to ≈ 0, but the residual is
the difference of three asynchronously-updating quotes; its **amplitude** spikes when one leg is
temporarily stale (session handovers, one-currency news, fix windows) and behaves as a mean-reverting OU
process around 0. The trade is a **convergence** harvest, not a forecast: when `|residual_z| > 2`, take
the cheap (one-leg quoted-cross) convergence side, exit on convergence — you are paid for the amplitude,
not for being right about direction. `EURJPY ≡ EURUSD × USDJPY`.

**Falsifiable prediction (the menu's).** Fit `dr = −θ·r·dt + σ·dW` at H4 → H1 → M15 → M1; find a
resolution where **half-life < ~5 bars AND OU amplitude σ > single-cross cost on > ~5% of bars.**
**Falsifier:** if amplitude > cost stays below ~5% at every resolution down to M1, triangulation is fully
closed (level AND variance). Prior is FAIL — the council's own read: "expect cost to eat it; going to
H1/M1 walks into the H1 cost wall (~2× H4)."

## (c)/(d) Observation — FALSIFIED (§5d cheap-kill)
Reused the BUILT `triangle_log_residual_bp` (canonical residual, MID closes) on 3 JPY triangles at **H1**
(the cost-wall resolution where the whole microstructure cluster died), IS 2010–2020, n≈68k bars each.
OU half-life via AR(1) `phi` (half-life = −ln2/ln φ). Cost hurdle = cross `spread_close` in bp.

| triangle | resid std (bp) | \|r\| med | OU half-life | cross spread (bp) | frac \|r\|>1×spr | frac \|r\|>2×spr | harvest \|r\| @\|z\|>2 vs RT cost |
|---|---|---|---|---|---|---|---|
| EURJPY | 0.606 | 0.135 | **1.09 bars** | 1.224 | 4.3% | 1.9% | 1.56 bp vs ~3.67 bp |
| GBPJPY | 0.573 | 0.119 | **0.60 bars** | 1.895 | 1.2% | 0.4% | 1.52 bp vs ~5.69 bp |
| AUDJPY | 0.800 | 0.138 | **0.99 bars** | 1.626 | 2.5% | 0.8% | 1.90 bp vs ~4.88 bp |

**Three decisive findings:**
1. **OU half-life ≈ 0.6–1.1 bars — reverts within ONE H1 bar.** The convergence is gone before you can
   act (entry is next-bar); there is no slow, capturable convergence to harvest.
2. **Amplitude is arb-bounded sub-cost.** Residual std 0.57–0.80 bp and |r| median 0.12–0.14 bp vs cross
   spread ~1.2–1.9 bp (~10× larger). |r| > 1× spread on only **1.2–4.3%** of bars (below the ~5%
   falsifier); σ itself < the spread.
3. **The convergence trade loses on cost even at the extreme trigger.** At |z| > 2 (n≈3k/triangle) the
   mean harvest |r| = 1.5–1.9 bp vs an RT cost ~3.7–5.7 bp (2 × FundedNext 1.5×-spread) — **~2–3× below
   cost.** §5f does not bite (no above-cost entry to place on the engine).

**Why M1 was not loaded (reasoned, not skipped).** The residual amplitude is **pinned near 0 by
arbitrage** — it cannot grow at finer resolution — while the spread cost **rises** going finer (the H1
wall, ~2× H4, that killed gotobi/round-number/WMR-fix/session-break). The trend is monotone against
tradeability: H4 (arc 3005) |r| > spread on 1.3–6.1%; H1 (here) 1.2–4.3% with harvest ~half the RT cost
and half-life already sub-bar. There is no resolution at which an arb-bounded amplitude overtakes a rising
cost. Loading M1 (≈5M bars × 7 pairs) for a door already shut is unwarranted; the falsifier is met at H1
and the cost/resolution argument forecloses finer scales.

## Verdict — KILL (obs cheap-kill)
The triangulation residual's **second moment is sub-cost and reverts within a bar** — the convergence
harvest the menu hoped for does not clear cost at H1 and cannot at M1 (arb-bounded amplitude vs rising
cost). **This fully closes the cross-rate triangulation thread: level (arc 3005), driver-shock first
moment (1027/2023), AND second-moment/OU amplitude (this arc).** It is the last open
`DISCOVERY_DIRECTION.md` menu door — the only remaining item, S1, is a tautological entry-geometry
modifier (fbr already embodies it), not a standalone arc.

**NEW lesson.** Triangular arbitrage pins BOTH moments out of reach on liquid FX: the mean to ≈ 0
(direction dead, 3005/1027/2023) and the amplitude below the cross spread with sub-bar half-life (variance
dead, this arc). An arb-enforced residual cannot be a tradeable edge in either moment, because the same
arb that creates the transient dislocation also collapses it faster than the spread can be paid. Generalizes
the H1 cost wall from discrete-flow events to the continuous cross-rate identity.

**Frontier.** The autonomous `explore-now` menu is now exhausted (M1/O1/L1/Q1/G1 all closed; S1 is a
non-standalone modifier). The operative lever is unchanged: operator **path-A (gate-governance)** on the
existing mean-positive 4-component book (arc 2019/3021/2021/3022). Components UNCHANGED. OOS untouched.

## (i)/(k) Tooling + re-orient
No new tool — reused canonical `Panel.from_pairs`, `mid_close`, and BUILT `triangle_log_residual_bp`. No
TOOL_REGISTRY append. FLAGS: none. Arc documented; log appended; committed + pushed. Re-orient: chat
2000s, range 2000–2999, next id 2029.
