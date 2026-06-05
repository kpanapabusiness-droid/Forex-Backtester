# Arc 2005 — Attacking EDGE<COST from the COST side (cost-regime conditioning)

> **Arc id:** 2005 · **Chat:** 2000–2999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **FAIL (KILL) for the signal — but a high-value methodological finding.** Every prior arc
> attacked the EDGE side of the programme's EDGE<COST wall; arc 3004's peer review flagged "the cost is a wall
> INDEPENDENT of the stop." This arc lifts the COST side: restricting the cross trend-momentum entry (arc 1003:
> +0.10R GROSS, net sub-cost because crosses' WIDE spreads eat it) to its CHEAPEST bars (low spread/ATR) lifts
> IS net **monotonically** from mean −7.96% (all bars) to **+1.72%** (cheapest 15%) — the best long-only IS mean
> in the programme — and it **beats a matched random-on-cheap-bars null** (≈ −0.3%), so the cross-trend
> *selection* does real work; the edge was cost-masked, not absent. **BUT it does NOT survive OOS** (mean
> −2.26%, 4/6 neg) and the q=0.15 was full-sample-swept → the IS edge is q-overfit + regime, not durable.
> Conservative disposition (§8/§11): **KILL** (OOS-negative ⇒ net-negative forward ⇒ not a PORTFOLIO
> component). **Convergent with arc 3007 (3000s), which attacked the cost side via absolute spread-timing and
> found it backfires (cost↓ entangled with edge↓); together the two chats close the COST side of EDGE<COST.
> DIRECTION remains the binding wall (reconfirms 3004); cost and stop are secondary ~5–10pp drags, not the
> wall.**
> **Idea source:** log-seeded (arc 3004 peer-review "cost is an independent wall" + arc 1003's +gross-drift,
> wide-spread-eaten cross-trend cell).

Scored solely by `MultiPairBacktester` (FundedNext costs ON; the engine applies the REAL per-bar cost on the
surviving cheap bars). Engine/measurement **called, never re-rolled**. Tools (geometry/observation only):
`DonchianBreakoutLongSignal` (BUILT arc 2000, reused), `make_low_cost_mask` (BUILT this arc), random-cheap null
(driver-local, mask randomization only; scoring canonical).

---

## (a) Log read + synthesis — FRESH EYES

Pulled `origin/main`. Honest-era corpus = 20 arcs across 3 chats (0; 1000–1008; 2000–2004; 3000–3005), all
FAIL. The directional long is comprehensively dead under BOTH the +1R-capture and forward-drift lenses, all
instruments/timeframes/regimes/exits, with the **stop removed** (3004 momentum, 2004 reversion — the SL-first
tax is a real drag but not the wall for either family), and one microstructure flow (gotobi, arc 1008) is
sub-cost. One PORTFOLIO edge exists (arc 1006 JPY-cross weekend gap-fill, 1000s thread). The arc-3004
ESCALATION stands: the apparatus expresses only direction (coin-flip on liquid FX); durable FX edges are
relative/market-neutral (needs shorts/2nd-leg, operator-gated). Pre-shorts lane = a 2nd net-positive long-only
component for the PORTFOLIO route.

**The untouched lever.** The programme's binding constraint is **EDGE < COST**. Every arc attacked the EDGE
side (raise per-trade gross edge: entry, exit, regime, instrument, timeframe). **No arc attacked the COST
side.** Arc 3004's peer review made it explicit: *"the cost is a wall independent of the stop — a time-exit
still pays the full per-round-turn spread."* FundedNext cost = 1.5×spread + slippage + $5/lot RT; the spread
term dominates and **varies a lot by bar/session**. Arc 1003 found cross trend-momentum has a real **+0.10R
gross** drift but nets sub-cost because **crosses' wide spreads eat it**. So the natural, never-run experiment:
restrict that +gross-drift entry to its CHEAPEST bars and see whether cutting the cost flips the net sign. This
also tests *which half* of EDGE<COST binds — if cost-cutting rescues it, COST was binding; if not, EDGE is.

## (b) Idea — log-seeded, no idea-council

Hypothesis (because): a trade's cost *in R units* scales with spread/ATR (a move of sl_mult·ATR = 1R, so a
spread S costs ~S/(sl_mult·ATR) in R). Restricting entries to bars whose spread/ATR is in its own trailing LOW
quantile cuts per-trade cost without touching the edge. Built `make_low_cost_mask` (BUILT, registered) — a
causal/ex-ante low-cost-regime mask (spread_close known at the signal bar; ATR shift1; the per-bar threshold is
a trailing rolling quantile shift1, so a bar's own value never enters its own threshold). Entry =
`DonchianBreakoutLongSignal(lookback=20, sma_filter=200)` on the 12 trending crosses (the arc-1003 +gross cell).

## (c)+(d) The cost-side A/B — full IS WFO (no lucky-3-fold triage)

`partial/runner` exit, sl 2.0, full 10-fold IS WFO, FundedNext costs ON. AND the low-cost mask into the entry
at successively tighter quantiles:

| version | per-fold ROI% (2011…2020) | worst | mean | maxDD | neg | AFP | n |
|---|---|---|---|---|---|---|---|
| all bars (baseline = arc-1003 cell) | −13.31/+0.59/−7.90/−23.95/−7.25/−11.36/−2.03/−6.82/−12.44/+4.84 | −23.95% | −7.96% | 27.55% | 8/10 | N | 2632 |
| cheapest 50% | −7.51/−4.35/+2.91/−5.91/−1.71/−11.78/−3.63/+2.29/−6.47/+2.32 | −11.78% | −3.38% | 13.79% | 7/10 | N | 1639 |
| cheapest 30% | −8.45/−2.00/+0.66/−0.93/+5.40/−7.52/−0.57/−0.94/−1.72/+12.28 | −8.45% | −0.38% | 10.76% | 7/10 | N | 1129 |
| **cheapest 15%** | +1.42/+0.97/+3.92/+1.13/+4.16/−2.41/+0.81/−1.69/+2.61/+6.28 | −2.41% | **+1.72%** | 6.01% | **2/10** | N | 682 |

**A clean, MONOTONE cost effect:** as bars get cheaper, mean rises −7.96→−3.38→−0.38→+1.72%, neg folds fall
8→7→7→2, maxDD falls 27.55→6.01%, worst rises −23.95→−2.41%. The monotonicity + the sound mechanism (less cost
= better net on a +gross-drift entry) make this a real effect, not a q-fluke. The cost side of EDGE<COST is a
**~10pp net lever** here. (n is summed over expanding/overlapping IS folds, so it is not a unique-trade count.)

## (e) The decisive control — is it the cross-trend EDGE, or just "cheap bars are cheaper"?

Restricting to low-spread bars also selects high-liquidity sessions — so the gain could be (a) real cost
saving (deployable) or (b) the trivial "cheap bars cost less" that would help ANY entry. Control = a **random
entry restricted to the SAME cheap bars**, matched fire count, 3 seeds (mask randomization only; scoring
canonical):

| version (cheapest 15%) | mean | worst | neg | 
|---|---|---|---|
| **REAL cross-trend** | **+1.72%** | −2.41% | **2/10** |
| RANDOM, seed 42 | +0.47% | −6.80% | 4/10 |
| RANDOM, seed 7 | −0.27% | −4.35% | 5/10 |
| RANDOM, seed 123 | −1.08% | −6.99% | 5/10 |

The real cross-trend **clearly beats random-on-cheap-bars** (+1.72% vs avg ≈ −0.3%; 2/10 neg vs ~5/10).
Random-cheap is ≈ break-even (as a ~zero-gross-edge long should be even at low cost) — so the cheap-bar
restriction does NOT manufacture a spurious positive; the cross-trend *selection* is doing real work. The
+0.10R gross edge (arc 1003) was being **cost-masked by wide cross spreads** and emerges net-positive once cost
is cut. **The edge is real.**

## (g) OOS (2021+) — the edge does NOT survive

q=0.15 was selected on full IS (not OOS) → measuring OOS once is characterization, not contamination. Per-year
2021–2026, same cheap-15% config:

| OOS fold | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | worst | mean | neg |
|---|---|---|---|---|---|---|---|---|---|
| ROI% | +0.22 | −8.33 | −3.64 | +0.79 | −2.49 | −0.13 | −8.33% | **−2.26%** | 4/6 |

**OOS is mean-NEGATIVE (4/6 neg).** The holdout contradicts the IS edge. Combined with the fact that IS
positivity is concentrated ONLY at the most aggressive cut (q=0.15 +1.72% vs q=0.30 −0.38%, q=0.50 −3.38%) and
q was full-sample-swept, the honest read is: the IS +1.72% is **q-overfit + favorable-regime**, not a durable
forward edge.

## Final verdict — FAIL; disposition = KILL (conservative, §8/§11)

This is a genuine PORTFOLIO-vs-KILL judgment fork (IS mean-positive + beats random null = the arc-1006 PORTFOLIO
profile), but **two things push it to KILL**: (1) **OOS is mean-negative** (−2.26%, 4/6 neg) — §11 is explicit
that a net-negative-forward signal is KILL, not PORTFOLIO (you cannot diversify net-negative components
positive); (2) the IS +1.72% was a **full-sample q-sweep best-pick** (§5f exit-fishing risk), so it is not the
clean IS-positive that arc 1006's un-swept +0.69% was. Conservative bias (§8) → **KILL the signal.** A proper
**nested cost-quantile WFO** (select q on each IS fold, score that fold's OOS, freeze onto holdout) is the
honest test that could *in principle* re-qualify a less-aggressive version as PORTFOLIO — flagged as the
recommended next step (and a HEAVY council on it), but the current evidence (OOS-negative) does not support
PORTFOLIO.

## Convergence with arc 3007 (chat 3000s, landed on main mid-arc) — reconciling a contradiction

The 3000s chat **independently** attacked the cost side on the SAME signal (Donchian-20 + SMA200 cross-trend,
12 crosses) via **intraday spread-timing** — and reached the *opposite-looking* conclusion: spread-timing
**backfires**, because the tight-spread tercile (= liquid London/NY hours) has the **weakest/NEGATIVE gross
edge** (tight gross −0.0199R, mid +0.12R, wide −0.035R — non-monotone); "you cannot cut cost without cutting
edge — spread×edge are entangled (both liquidity-driven)." → KILL.

**Reconciliation (Arc-10: two honest measurements disagree — find why).** The two arcs condition on DIFFERENT
notions of "cheap": 3007 uses **absolute** spread terciles / tightest-UTC-hours → selects liquid hours, which
have low gross edge. Arc 2005 uses **trailing-relative** cheapness (spread/ATR vs each pair's own recent
250-bar norm) → selects *different* bars (cheap relative to that pair's recent regime, not necessarily liquid
hours), and the Donchian selection on those bars beats random-cheap **in-sample**. But **my OOS is −2.26%
(negative)** — so the IS +1.72% is relative-cheapness q-overfit + favorable-regime that does NOT generalize.
**Both chats reach the same KILL**, and 3007 supplies the mechanism that explains why my IS edge can't persist:
cutting cost is entangled with cutting edge. **The cost side of EDGE<COST is now closed from two independent
angles (3007 absolute spread-timing; 2005 trailing-relative cheapness), both KILL.**

## What this arc ADDS (the value — a methodological finding, reconciled with arc 3007)

1. **The COST side of EDGE<COST is closed (both chats), and the closure is now well-understood.** Restricting a
   +gross-drift entry to its cheapest bars lifts IS mean −7.96%→+1.72% (monotone) and beats a matched
   random-cheap null **in-sample** — but it does NOT survive OOS (−2.26%), and arc 3007 shows the mechanism:
   the cost-saving direction (tight spread = liquid hours) is **entangled with the weakest gross edge**, so you
   cannot durably cut cost without cutting edge. The `make_low_cost_mask` tool (BUILT) is reusable, but the
   honest lesson is that **cost-conditioning is NOT a free lever on a directional FX entry** — it needs a gross
   edge that is *decoupled* from liquidity (none found). Apply it only to a signal whose gross edge is
   demonstrably independent of the spread regime.
2. **Cutting cost does NOT rescue a DIRECTIONAL edge — completing the "stop + cost" closure.** Arc 3004 showed
   removing the STOP lifts mean but not to all-folds-positive; arcs 2005 + 3007 show cutting the COST lifts IS
   mean (2005) or backfires (3007) but never yields a durable edge. **Both halves of the "maybe it's the stop /
   maybe it's the cost" objection are now lifted across both chats, and DIRECTION is still the binding wall.**
   The escalation's cost item (#3, "tighter-cost regime") is thus informed: even a raw-spread ECN would help
   margins but, on this evidence (cost↓ entangled with edge↓), would NOT manufacture a durable directional edge
   — reinforcing that the real unlock is the 2nd leg / shorts (relative-value), not cheaper execution.

## Threads / what didn't help

- **Closed:** the cheap-bar directional cross-trend as a standalone/portfolio signal (OOS-negative, q-overfit).
- **HIGH-VALUE open thread:** apply `make_low_cost_mask` to a signal with a *durable* gross edge. The cost
  lever is real; it needs a real, OOS-surviving gross edge to act on. Candidates: the gap-fill (though it fires
  at high-spread weekly opens, so likely incompatible) or a future portfolio combination. **Recommended:** a
  nested-q WFO + HEAVY council to settle whether any cost-conditioned version is PORTFOLIO-worthy.
- **Stands:** the arc-3004 escalation (now reinforced from the cost side). Direction is the wall; cost and stop
  are secondary ~5–10pp drags, neither the binding constraint.

## Flags (code NOT merged — human-gated, §9)

- None new. Carries FLAG-1 (long-only blocks the stronger UP-gap short side; operator-gated) and the
  `A1Config.time_exit_bars`-unwired flag (arcs 1005/3004).
- Reminder (driver convention, not a code flag): `FoldStats.roi_pct`/`max_dd_pct` are FRACTIONS (×100 for %).

## Reproduction

- **Data:** `histdata_root = C:\Users\panap\histdata_backup`, `cache_root = data/cache`,
  `boundary_convention = "5ers_eet"`, TF H4. **Pairs (12 trending crosses):** EURJPY GBPJPY AUDJPY NZDJPY
  CADJPY CHFJPY EURAUD EURNZD EURCAD GBPAUD GBPNZD GBPCAD.
- **Entry (BUILT, reused):** `discovery/tools/trend_entry_signals.DonchianBreakoutLongSignal(lookback=20, spacing_bars=6, sma_filter=200)`.
- **Cost mask (BUILT this arc):** `discovery/tools/cost_regime_mask.make_low_cost_mask(panel, pairs, quantile=Q, window=250)`, ANDed into each pair's `PerPairSignalState.signal_mask` via `dataclasses.replace`. Q∈{0.50,0.30,0.15}.
- **Exit:** `A1Config(exit_policy="sl_partial_close_1r_runner_trail", sl_atr_mult=2.0)`. WFO: `build_v3_folds()` IS (is_days≥365); OOS `build_oos_year_folds(start_year=2021)`; judge `judge_all_folds_positive`.
- **Null control:** random entry placed only on cheap bars (matched per-pair count, warmup 220, seeds 42/7/123).
- **Drivers (scratch):** `_disco2000_work/arc2005_cost_side.py`, `arc2005_null_control.py`, `arc2005_oos.py`.
