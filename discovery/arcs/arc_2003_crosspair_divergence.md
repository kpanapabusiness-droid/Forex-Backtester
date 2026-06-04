# Arc 2003 — Cross-Pair Relative-Value Divergence (Laggard Catch-Up Long)

> **Arc id:** 2003 · **Chat:** 2000–2999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **FAIL (cheap-kill at observation).** The under-performing leg does NOT catch up —
> it CONTINUES to lag (z<−2 forward drift −0.05 to −0.11 ATR; per-group z<−1 fwd10 −0.20 EURvGBP / −0.15
> AUDvNZD). There is no relative-value reversion in correlated majors; relative performance is
> momentum/coin-flip (re-confirms arcs 1000, 3001). The LIGHT generative council's strongest call —
> **CONCEDE**: the binding blocker is the long-only constraint, not the search space — stands.
> **Lever tested:** a LIGHT generative council (§5b) for a non-obvious direction, then its best testable
> idea (cross-pair divergence catch-up — relative-value, distinct from arc 3000's absolute oversold).

Observation-grade drift scan (the lens arcs 1004/3001 endorsed); no pool built (cheap-killed before
characterization). No canonical-core change; reused the `_atr_shift1_mid` helper from `trend_entry_signals`.

## (a) Log read — FRESH EYES

Pulled main (`011517f`). 15 arcs, all FAIL. The price-only long-only directional space is metric-robustly
closed; the one real edge (weekend gap-fill reversion) is closed on the long side from 4 convergent angles
(universe 1006, exit-target 1007, entry-resolution 2002, intraday-analog 3001) and its stronger side (UP-gap
short) is blocked by the long-only apparatus (FLAG-1). The obvious price-structure ideas are exhausted — a
genuine **dry-log idea-fork**. No STOP.

## (b) LIGHT generative council (§5b) — surface a non-obvious direction

Convened 3 generative lenses (isolated) on the mapped state + hard constraints (price-only data, long-only,
FundedNext costs incl. no swaps, all-folds-positive judge):
- **Mechanism** → *cross-pair relative-value divergence*: when two correlated XXX/USD majors diverge (the
  synthetic cross at an extreme), the under-performing LEG should catch up; long-only = long the laggard.
  Low-frequency (cost-tolerant), relative-value not single-pair direction.
- **Alternative-framing** → *bid-ask spread-spike mean-reversion* (M1): after a spread spike (liquidity
  withdrawal), mid reverts toward fair value over 1–5 bars. Microstructure dislocation, not direction.
- **Soundness/Devil** → **CONCEDE is the stronger call**: "15 fails with one confirmed real edge that
  points SHORT is not a 'keep searching' pattern — it is the programme telling you what it found." The
  binding blocker is the long-only constraint; the divergence idea is "worth one arc," but the honest
  meta-move is to flag the short-side constraint to the operator first.

**CC synthesis (generative = CC decides):** test the convergent best idea (divergence catch-up — 2 lenses
named it, and it is genuinely distinct from arc 3000's *absolute* RSI-oversold-on-the-cross: this is the
*relative* under-performance of one leg vs a correlated leg). The M1 spread-spike idea is deferred (a
1–5-bar M1 reversion is sub-pip vs ~3-pip round-turn cost — almost certainly EDGE<COST; low-odds seed).

## (c) Observation — divergence catch-up (D1, ex-ante)

For correlated XXX/USD groups (EURUSD↔GBPUSD, AUDUSD↔NZDUSD, both directions), spread = log(A_mid) −
log(B_mid), z-score over a 60-bar rolling window (shift1, ex-ante). LOW z = A under-performed B = catch-up
long candidate. Forward drift of A (mean, in ATR), IS 2010–2020:

| z bin | n | fwd5 (mean ATR / frac+) | fwd10 | fwd20 |
|---|---|---|---|---|
| **z<−2 (A weak)** | 808 | −0.109 / 0.49 | −0.046 / 0.51 | −0.007 / 0.53 |
| −2..−1 | 2651 | −0.055 / 0.49 | −0.122 / 0.49 | −0.169 / 0.51 |
| −1..0 | 2826 | +0.069 / 0.53 | +0.041 / 0.53 | −0.097 / 0.51 |
| 1..2 | 2651 | −0.058 / 0.50 | −0.169 / 0.49 | −0.312 / 0.47 |
| z>2 (A strong) | 808 | −0.121 / 0.49 | −0.095 / 0.50 | −0.120 / 0.52 |

Per-group at z<−1 (long A), fwd10: EURUSDvGBPUSD **−0.202** / AUDUSDvNZDUSD **−0.150** / GBPUSDvEURUSD −0.057 /
NZDUSDvAUDUSD +0.034 (frac+ 0.51, coin-flip). **The catch-up hypothesis is FALSIFIED:** the laggard does NOT
revert up — it drifts flat-to-NEGATIVE (continues lagging). The relative performance of correlated majors is
weak momentum / coin-flip, not mean-reverting. → cheap-kill at observation (no worthwhile drift to
characterize; the drift is the wrong sign for a long-only catch-up).

## Final verdict — FAIL (cheap-kill)

Cross-pair relative-value divergence is not a long-only edge: the under-performing leg continues to
under-perform. This re-confirms, from the pairwise-divergence angle, what arc 1000 (cross-sectional momentum)
and arc 3001 (drift-lens scan) found — there is no separable relative-value reversion in FX majors. The
generative council's best testable idea is spent and dead; its **CONCEDE** call stands.

## Lessons (candidate for LESSONS.md)

1. **No relative-value reversion in correlated FX majors** (pairwise synthetic-cross divergence): the
   under-performing leg drifts flat-to-negative at z<−2 (−0.05 to −0.11 ATR/12 bars), it does not catch up.
   Relative performance is momentum/coin-flip — a third independent confirmation (with arcs 1000, 3001) that
   the cross-sectional/relative axis carries no long-only edge.
2. **The LIGHT generative council's strongest, self-judged call was CONCEDE** — and it INDEPENDENTLY reached
   the same conclusion as arc 2001's FLAG-1: across 16 arcs / 3 chats, the one real edge (weekend gap-fill)
   is confirmed in mechanism, direction, and magnitude, and its STRONGER leg is the UP-gap SHORT — which the
   long-only apparatus structurally cannot trade. **The binding blocker is the long-only constraint, not the
   search space.** This is the highest-value, most-actionable standing result for the operator.
3. A generative council on a heavily-mapped space mostly re-surfaces the known answer-space (relative-value,
   short side, microstructure) — its value here was the independent CONCEDE corroboration + ruling the
   divergence idea in/out cheaply, not a new survivor.

## Threads / FLAGS

- **Closed:** cross-pair relative-value divergence catch-up (laggard continues lagging — no reversion).
- **FLAG-1 (REINFORCED — the programme's most actionable item):** the long-only apparatus blocks the
  STRONGER side of the only real edge (the UP-gap weekend SHORT: −0.57 ATR drift, 0.64 accuracy vs the
  down-side's 0.59). Independently corroborated now by the generative council's CONCEDE lens. Adding
  short support is a canonical-core change (operator/human-gated, NOT self-merged) and would unlock the
  programme's dominant leg — the single highest-leverage operator decision.
- **Low-odds seed (not tested):** M1 bid-ask spread-spike mean-reversion (liquidity-withdrawal dislocation);
  almost certainly EDGE<COST at M1 frequency (sub-pip reversion vs ~3-pip round-turn), but the only
  genuinely-untested microstructure axis if the operator wants it scanned.
- **Standing result:** the long-only price-only H1/H4/D1 FX space is comprehensively mapped across 16 arcs /
  3 chats with no capturable EDGE>COST edge; the live value is the gap-portfolio (1000s) and the short-side
  unlock (operator decision).

## Reproduction

Driver scratch `_disco2000_work/arc2003_divergence_observe.py`; `PYTHONPATH=. py
_disco2000_work/arc2003_divergence_observe.py`. Data: `histdata_root=C:\Users\panap\histdata_backup`,
`cache_root=...\data\cache`, **tf D1** 5ers_eet, correlated XXX/USD groups (EURUSD↔GBPUSD, AUDUSD↔NZDUSD).
Spread z-score 60-bar rolling, shift1. IS 2010–2020. No pool/engine run (cheap-killed at observation).
