# arc 2051 — fbr (failed-breakdown reclaim) on H1: can the crown-jewel edge be THICKENED on a finer TF? (arc-2017 option-B)

**Chat 2000s · VERDICT: KILL (obs cheap-kill, §5d).** No engine/null/council; OOS untouched; components
UNCHANGED (all 4 PORTFOLIO; honest deploy core = me_long-solo, arc 1046).

## Because
The only NON-operator-gated path to an all-folds-positive book is arc-2017's **option-B: a component
THICK enough that its year-folds RESOLVE.** The 4-way book's AFP failure is a NOISE FLOOR — thin
components have year-fold sampling-sd ≥ mean (arcs 1023/2016), so a 5th thin leg is fold-painting. `fbr`
(1013) is the corpus's ONE robust structural edge (H4 USD majors, capture 0.582 vs baseline 0.4877,
control coin-flip, IS 9/10) but was **only ever run on H4** (~210 IS trades, thin). The swing-low
stop-run mechanism is *prima facie* scale-free (sell-stops cluster below visible H1 swing lows too) → fbr
on H1 should fire ~4× as often → thicker. If the capture edge SURVIVES the finer TF (beats the H1
baseline ≈0.431 [arc 2050] AND a matched structure control), the extra thickness is the explicit option-B
lever → engine + §5f. Cost-skeptic prior: H1 displacements are smaller → the edge may erode (the corpus's
repeated intraday finding).

## Method (observation only; no engine/P&L)
H1, 4 cached USD majors (EURUSD/GBPUSD/AUDUSD/USDJPY — fbr's universe; USDCAD/USDCHF/NZDUSD not H1-cached,
noted as a universe limit). Replicated the BUILT `FailedBreakdownReclaimLongSignal` fire logic inline on
H1: `low_bid < prior-K-bar-min-low` (swept stops) AND `close_mid > that level` (failed breakdown) AND
lower-rejection shadow `(min(open,close)−low)/ATR ≥ 1.25` (canonical). K∈{40 (fast/thick ≈1.7d), 160
(horizon-matched to H4 K40 ≈6.7d)}. Matched STRUCTURE control = the SAME deep rejection wick NOT at a
swept swing low (no pierce) — fbr's 1013 control discipline. Honest +1R capture (take-the-loss) + 24-bar
drift via canonical `observe_long_capture`; IS 2010-2020 only (OOS preserved). Driver
`discovery/_disco2_work/arc_2051_fbr_h1_obs.py` (scratch).

## FALSIFIED — fbr does NOT survive on H1
H1 baseline (4 USD majors) = capture **0.4753**, drift −0.0718.

| signal | n | capture | drift ATR |
|---|---|---|---|
| H1 baseline | 273142 | 0.4753 | −0.072 |
| **fbr H1 K40** | 720 | **0.4431** | **−0.413** |
| structure control K40 | 1344 | 0.4702 | −0.355 |
| **fbr H1 K160** | 395 | **0.4481** | **−0.077** |
| structure control K160 | 1825 | 0.4597 | −0.476 |

- **fbr capture is BELOW baseline AND below the structure control at both K** (K40 0.4431 < 0.4702 <
  0.4753; K160 0.4481 < 0.4597 < 0.4753) — the swing-low pierce makes capture WORSE, not better. The
  structure is **anti-load-bearing on H1** (vs H4, where AT-swing-low 0.582 ≫ control coin-flip).
- **Drift strongly negative** (K40 −0.413; the K160 −0.077 is the wick mean, still negative). The H1
  "failed-breakdown reclaim" does not revert — the move continues down.
- **Strong-USD years negative** (K40 2014 −1.31 / 2015 −0.93 / 2018 −0.24; K160 similar); positive only in
  choppy 2012/2017 (thin n, ≤67). Per-pair: USDJPY/EURUSD drift strongly negative; no robust pair.

## Why it fails (mechanism)
fbr works because an **H4/D1 swing low is a SIGNIFICANT, watched level** with genuinely clustered resting
stops — sweeping it is a real, information-free stop-cascade that reverts. An **H1 swing low is
microstructure noise**: minor pivots every few hours where stops are NOT clustered, so an H1
"failed-breakdown reclaim" is just a deep down-wick — which is exactly fbr's OWN H4 *control* ("a big
down-wick NOT at a swing low"), a coin-flip-to-negative non-event. On H1 the fbr fire-set IS dominated by
those non-events → it inverts below baseline and below its own structure control. The ~4× thickness is
illusory: the extra fires are not stop-runs. So fbr cannot be thickened to fold-resolution by trading the
same edge on a finer TF.

## What this closes
arc-2017's **option-B via the most natural route** — thicken a proven edge by going to a finer timeframe —
is closed: the corpus's one structural edge is TF-specific (lives at H4/D1, dies at H1). Combined with arc
2018 (thicken via cross-sectional → cost-killed), both tractable thickening routes for option-B are now
dead. The 2018/2014 regime-orthogonal leg remains unfound; the fold-resolution path stays operator-gated
(path-A). me_long-solo deploy object UNCHANGED (1046). Deployable-system count = 0.

## NEW lesson
The structural stop-run reversal edge (fbr) is **TIMEFRAME-SPECIFIC**: it requires the swing low to be a
SIGNIFICANT, watched level with genuinely clustered resting stops (H4/D1), which does NOT exist at H1
(swing lows are microstructure noise, no stop cluster). So **option-B "thicken a proven edge to resolve
folds" cannot be reached by trading the SAME edge on a finer TF** — finer-TF fires are mostly non-events
that DILUTE, not thicken, the signal (fbr H1 fires invert BELOW baseline and BELOW its own structure
control — worse than random, the structure flips sign, not merely shrinks). Generalizes the corpus's
"intraday erodes toward cost" to "intraday erodes the STRUCTURE itself." Corollary: a structural edge's
thickness is bounded by how many genuinely-significant levels exist on its native TF — you cannot
manufacture more by subdividing the timeframe.

## Tooling
No new BUILT tool — single-use exploratory scan (matching 2047-2050), canonical `Panel.from_pairs` +
`observe_long_capture` + BUILT `_atr_shift1_mid` + the inline fbr fire logic (replicated from the BUILT
`FailedBreakdownReclaimLongSignal` source). No TOOL_REGISTRY append. No canonical change, no FLAG, no
council, OOS untouched. Driver `discovery/_disco2_work/arc_2051_fbr_h1_obs.py`.

## Threads / handoff
Both tractable option-B thickening routes (finer-TF here, cross-sectional 2018) are dead — fold-resolution
for an AFP book stays operator-gated (path-A). The surviving edges are information-free MECHANICAL forced
flows keyed to a LARGE surprise displacement at a NON-directional-flow time, living ON the USD factor,
**at H4/D1 scale** (the structure dies intraday). 2018/2014 regime-orthogonal leg unfound (~22 routes);
deployability lever = operator path-A (arcs 2019/3021/1046) + the arc-2033/2045 vehicle wall. Datum: H1
long capture baseline ≈0.475 on 4 USD majors (≈0.431 incl. JPY crosses, arc 2050 — the crosses drag it).
