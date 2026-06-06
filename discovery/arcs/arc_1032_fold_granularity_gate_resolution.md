# Arc 1032 — fold-granularity → all-blocks-positive map (operator path-A gate-resolution, quantified)

> **Arc id:** 1032 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-06
> **Final verdict:** **KILL (diagnostic; no new component).** Quantifies the corpus's sole remaining
> deployability lever — the operator's path-A "reconsider fold/gate resolution" call (arc 2016 flag) —
> which no prior arc had measured. **The 4-component book is robustly all-blocks-positive ONLY at the
> trivial 1-block (= whole-IS mean) level**; at EVERY meaningful granularity (≥3 blocks) it is NOT
> robustly all-blocks-positive under all block alignments. The worst block shrinks MONOTONICALLY as the
> gate coarsens (risk-parity: calendar-year −0.459% → 2-year −0.153% → 3-year −0.064% → 5-year-halves
> −0.012% → whole-IS **+0.266%**) — the noise-absorption signature (the 2015 & 2018 holes are within-noise
> dips that average out only in the limit), but it does NOT cleanly flip positive at any ≥3-block
> resolution (3-year: only 33% [RP] / 67% [equal] of alignments pass; 5-year: 80%, and a 2-block gate is
> not a meaningful gate). **⇒ "coarsen the calendar-year gate" is NOT a viable path-A resolution** — the
> holes persist through 2- and 3-year blocks and vanish only at the 1-block mean. Path-A is therefore
> effectively binary: adopt a **mean / pooled / CI-based** gate (the book passes — mean +0.59% RP, t=2.66
> arc 1023, P(mean<0)=0.004 arc 2019) OR keep all-blocks-positive at any meaningful resolution (the book
> fails). There is no moderate-coarsening middle ground.
> **Idea source:** the operator FLAG raised by arc 2016 and reinforced by 2017/2019/3021/2021/3022/1023 +
> cosim item E — "the AFP calendar-year gate is applied below its noise floor; reconsider fold/gate
> resolution (operator governance call)." Stated qualitatively everywhere, quantified nowhere. This arc
> measures it.

A pure re-analysis of the CANONICAL COMMITTED per-year component ROIs (cosim item-E validation doc,
reproduced byte-exact through `MultiPairBacktester`; arc-2019 frozen risk-parity weights). NO new engine
run, NO null, NO council. **OOS NOT touched** — the book fails IS AFP, so the frozen 2021+ holdout stays
unspent (§4/§5g; spending it under a path-A gate is the operator's firewall decision, not an autonomous
chat's, per arc 2022).

## (a) Log read + synthesis

Pulled main (picked up concurrent 2000s arcs 2028 [L1, converged with my 1031], 2029, 2030 [fbr touch-count
refinement — edge-hunt continuing] and the cosim item-E validation). Corpus terminus, as of this arc:

- **The 4-component PORTFOLIO** (gap 1006 / me_long 1011 / fbr 1013 / me_short 1019) is mean-positive
  (t=2.66, arc 1023; P(mean<0)=0.004, arc 2019), cost-robust (break-even κ=3.32, arc 3022), temporally
  stable (arc 2021), ~3 independent bets (ENB 3.32/4, arc 2019) — but **fails the per-calendar-year
  all-folds-positive gate**, blocked by 2015 & 2018 folds that sit BELOW the components' noise floor
  (arcs 2016/2017). The cosim item-E gate CONFIRMS this is FUNDAMENTAL, not a linear-combiner artifact
  (2015 & 2018 negative under every weighting and every combiner; cap-on adds 2017).
- **Every autonomous edge-hunt route is closed:** the explore-now MENU is EXHAUSTED (M1 1027/2023, O1
  1029/1025/**1030**, L1 **1031**/2028, Q1 1028, G1 2018, S1 modifier); path-B (more decorrelated legs) is
  quantitatively closed (3021: shared dollar factor floors book variance below the gate at any N); option-B
  (a thick fold-resolving standalone) closed on 4 constructions (1025/2018/1026/2024); the leg-hunt is
  structurally closed (2022). **The sole deployability lever is the operator's path-A gate-governance call.**
- That lever has been stated QUALITATIVELY by ~6 arcs but never quantified. The natural, OOS-preserving,
  decision-relevant capstone is: **at what fold granularity does the IS book become (robustly) all-blocks-
  positive?** — i.e., does "coarsen the gate" actually work, and at what block count?

FRESH EYES; honest-era only.

## (b) Idea + because

**Because** arc 2016 flagged the all-folds-positive *calendar-year* gate as mismatched to a thin book whose
negative folds are within-noise, and named "reconsider fold/gate resolution" as the operator's lever, the
concrete question is whether COARSENING the fold blocks (2-year, 3-year, 5-year) absorbs the sub-noise
2015/2018 dips into net-positive blocks — and if so, at what block count, and robustly to alignment.
**Falsifiable framing:** if the book becomes all-blocks-positive at some meaningful granularity (≥3 blocks)
under EVERY block alignment, "coarsen the gate" is a viable path-A resolution; if the holes persist until
the trivial 1-block (= mean) limit, then path-A is specifically "switch to a mean/pooled/CI gate," not
"coarsen the calendar gate."

**Method (transparent, no engine).** Book per-year ROI = Σ wₖ·ROIₖ from the canonical committed component
per-year vectors (cosim doc, reproduced byte-exact), under arc-2019 frozen risk-parity weights AND equal
weights. At each block length L ∈ {1,2,3,5,10}, partition the 10 IS years into contiguous L-year blocks
under EVERY alignment offset ∈ [0,L) (block ROI = equal-time mean of its years; leading/trailing partials
counted as their own shorter blocks — every year in exactly one block, the stricter accounting). Report
robust-AFP (all blocks > 0 under all alignments), the fraction of alignments that pass, and the worst block
over all alignments. (Block ROI mean vs compounded is sign-invariant at these ~0.5% magnitudes.)

## (c)/(d) Result

**Risk-parity book** (mean +0.593%, negative years 2015 −0.46 / 2018 −0.28):

| granularity | robust AFP | frac alignments pass | worst block (any align) | n_blocks |
|---|---|---|---|---|
| calendar-year | False | 0.00 | −0.459% | 10 |
| 2-year | False | 0.00 | −0.153% | 5 |
| 3-year | False | 0.33 | −0.064% | 4 |
| 5-year (halves) | False | 0.80 | −0.012% | 2 |
| whole-IS | **True** | 1.00 | **+0.266%** | 1 |

**Equal-weight book** (mean +0.920%, negative years 2013/2015/2018): same shape — year False (−2.355%) →
2-year False/0.50 → 3-year False/0.67 (−0.217%) → 5-year False/0.80 (−0.013%) → whole-IS **True (+0.231%)**.

- **Robust all-blocks-positive ONLY at the 1-block (whole-IS = mean) level** for both weightings.
- **The worst block shrinks MONOTONICALLY as the gate coarsens** (RP: −0.459 → −0.153 → −0.064 → −0.012
  → +0.266) — the textbook noise-absorption curve: the negative folds are shallow within-noise dips
  (arcs 2016/2017) that average toward the +0.59% mean as blocks widen, but they do NOT flip the worst
  block cleanly positive until the single whole-IS block.
- **No meaningful (≥3-block) granularity passes robustly:** 3-year (4 blocks) passes only 33% (RP) / 67%
  (equal) of alignments with a still-negative worst block; 5-year passes 80% but is only 2 blocks (not a
  meaningful gate). The 2015/2018 holes are deep enough that some alignment always straddles a net-negative
  block until the holes are diluted into the whole-decade mean.

## (e) Diagnosis / read for the operator

"Coarsen the fold gate" is NOT a viable path-A resolution: there is no fold granularity that is BOTH a
meaningful gate (≥3 blocks) AND robustly all-blocks-positive. The all-blocks-positive framing only succeeds
at the trivial 1-block limit, which is just the already-established mean-positive result. **So path-A is
effectively binary:**
1. **Adopt a mean / pooled / CI-based gate** → the book PASSES on IS (mean +0.59% RP, t=2.66 [1023],
   P(mean<0)=0.004 [2019], cost-robust to κ=3.32 [3022], temporally stable [2021]). This is the only
   gate-resolution that both (a) is a real gate and (b) the book clears.
2. **Keep all-blocks-positive at any meaningful resolution** → the book FAILS (2015 & 2018 within-noise
   dips trip it, combiner-invariant per cosim item E).

The monotone worst-block→mean curve IS the quantitative statement that the per-year gate trips on sub-noise
dips. This does not change the verdict (still FAIL under the strict gate) and does NOT loosen the gate — it
characterizes the lever precisely so the operator's governance call is informed. Spending the frozen OOS
under whichever gate the operator adopts remains the operator's §5g firewall decision (arc 2022), not an
autonomous chat's.

## (f)/(g)/(h) Best version / validation / council

N/A — a re-analysis diagnostic, not an edge. No engine, null, or council (a measurement informing a
measurement, like arcs 1023/2017/3021). OOS never touched.

## (i) Disposition + NEW lesson

**KILL (diagnostic → no new component).** Components UNCHANGED (all 4 PORTFOLIO). The book stays strict-gate
FAIL; this arc does not alter the gate — it quantifies the operator's path-A lever.

**NEW lesson.** For a thin mean-positive book whose negative folds are within-noise, coarsening the fold
gate shrinks the worst block MONOTONICALLY toward the mean but does NOT yield a robust all-blocks-positive
pass at any meaningful (≥3) block count — the holes vanish only at the 1-block (mean) limit. ⇒ the
"all-folds-positive" framing and a "use bigger fold blocks" fix are INCOMPATIBLE for this corpus; the
operator's path-A is specifically "adopt a mean/pooled/CI gate," not "coarsen the calendar-year gate."
This sharpens the standing operator flag (arcs 2016/2017/2019/3021) from "reconsider fold resolution"
(ambiguous) to a precise binary. The autonomous programme has nothing further to add to deployability:
edge-hunt closed (MENU exhausted, path-B/option-B closed), book-gate quantified — the decision is the
operator's.

## Reproduction

`discovery/results/arc_1032_fold_granularity_gate_resolution/granularity_map.py` (+ `output.txt`).
Inputs are the canonical committed per-year component ROIs from `discovery/COSIM_ITEM_E_VALIDATION.md`
(reproduced byte-exact through `MultiPairBacktester`) + arc-2019 frozen risk-parity weights. No new tool,
no TOOL_REGISTRY append, no FLAGS, no OOS. Base conda env.
