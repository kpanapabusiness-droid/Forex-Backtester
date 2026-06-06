# arc 1042 — Honest-exit 4-way BOOK mean (completing the §5f exit audit on gap + me_long; recomputing the deploy number)

> chat: 1000s | range 1000-1999 | timestamp: 2026-06-06
> disposition: **DIAGNOSTIC → KILL** (no new component; book deploy-mean recomputed under honest exits)
> components UNCHANGED (all 4 PORTFOLIO). OOS frozen-scored once, never re-selected.

## (a) Log read / synthesis
Read DISCOVERY_LOG (both tiers, arcs 0–2041), LESSONS, TOOL_REGISTRY, checked STOP (absent). State:
single-condition shallow directional prediction is closed ground (H1/H4/D1/W1, long+short); **4
PORTFOLIO components** exist — gap (1006), me_long (1011), fbr (1013, 9/10), me_short (1019, only
robust +2018). The **4-way book is mean-positive (+0.59% RP, t=2.66, arc 1023) but NOT all-folds-
positive** (2015/2018 within-noise dips). Edge-hunt structurally closed (3021 path-B proof; MENU
M1/O1/L1/Q1/G1/S1 exhausted; ~18 routes to a +2015/+2018 leg dead). §11 independent verification
COMPLETE (signal 2034/2035/1037 + outcome 2036/2037/2038/1038/1039 + cost 2039, all honest). Sole
deployability lever = operator path-A gate call (1032 quantified: coarsening the gate doesn't yield a
meaningful-resolution AFP pass; path-A = adopt a mean/CI gate). Live fresh thread = arcs **2040
(fbr) + 2041 (me_short)**: committed exits are §5f-forbidden full-sample picks; honest nested-WFO
selection cut fbr ~40% (stayed +) and FLIPPED me_short NEGATIVE — but **gap + me_long never checked.**

## (b) Idea (the because)
The corpus is at: edge-hunt structurally closed (3021 path-B proof, MENU exhausted, ~18 routes to a
+2015/+2018 leg dead); §11 independent verification COMPLETE (signal+outcome+cost, all honest); the
sole deployability lever is the operator's **path-A gate-governance call** (adopt a mean/pooled/CI
gate vs the all-folds-positive calendar-year gate, which sits below the thin book's noise floor).

The live fresh thread: arcs **2040** (fbr) + **2041** (me_short) showed the committed component
headlines rest on **§5f-forbidden full-sample-best exit picks**. Under honest nested-WFO exit
selection fbr lost ~40% (stayed +) and **me_short FLIPPED NEGATIVE**. They framed it "BOTH
engine-positive legs are exit-optimistic" but only tested 2 of 4 — **gap (1006) + me_long (1011)
were never checked.**

Because: path-A would deploy the book's **MEAN** (committed +0.59% RP, t=2.66). That mean is a
weighted combination of the 4 component headlines — and 2 of the 4 are now known exit-inflated
(me_short to the point of sign-flip). **The deploy-relevant honest book mean — every component scored
under its HONEST §5f nested-WFO exit instead of its committed full-sample exit — has NEVER been
computed.** This arc computes it: (1) complete gap + me_long honest exit selection (the untested
legs), (2) recombine all 4 honest per-fold series into the honest book mean / worst / AFP, (3)
quantify the book-level exit-optimism gap vs the committed +0.59%. This is the synthesis 2040/2041
set up but neither finished, and it directly informs whether path-A's deploy number is real.

## (c)/(d)/(f) Method
- Canonical scoring throughout (`ArcFoldRunner` → `A1Architecture` → `MultiPairBacktester`,
  FundedNext), selection via BUILT `nested_exit_selection` (2040), book combination via BUILT
  `combine_fold_roi` (2006). No gate reimplemented.
- **gap** exit menu = the chosen hyperparameter (the OVERSHOOT-harvest HORIZON; arc 1007 proved the
  edge is overshoot and capping at target is WORSE) → pure-time (`exit_policy=None`) × horizon
  {12,18,24,36,48} × SL {1.5,2,2.5} = 15 configs; committed = horizon 24 / SL 2.0. Registry SL-exits
  scored as a side-check at horizon 24 (different family; confirm they don't beat pure-time).
- **me_long** = 6 registry exits × 3 SL = 18 configs, the 2-bar reversion horizon held FIXED as
  mechanism (arc 2024: lengthening contaminates with trend); committed = sl_only / SL2.0.
- **fbr / me_short** = 6 registry exits × 3 SL (re-derived here for the book series; reproduces
  2040/2041).
- Nested selection: 3 metrics (mean_roi / afp_then_mean / worst_then_mean), per fold choose the
  config best over STRICTLY-EARLIER folds; first 2 folds = warmup (excluded from verdict). Frozen =
  best over all IS; scored ONCE on 2021+.
- Book: combine the 4 honest per-fold ROI series (equal + risk-parity, weights fit on IS), vs the
  committed series at committed RP weights. Also honest-series @ committed-weights (isolates the exit
  effect from the weight effect).

## Results

**Reproduction anchor (committed configs vs arc-1020).** gap +0.685% (max|diff| 0.005pp), me_long
+0.232% (0.004pp), me_short +0.683% (0.005pp) byte-close. fbr +2.084% (1.493pp off arc-1020's
+1.854%) — this is the **trail-OFF** book config (the deployable book scores `trail_enabled=False`;
arc-1020's +1.854% used the `trail_enabled=True` double-trail, arc-2040 F2). My +2.084% reproduces
arc-2040's trail-off anchor EXACTLY → consistent, no defect.

**Component §5f picture (the 2 NEW legs + reproduction of 2040/2041's 2):**

| leg | RP wt | committed exit | committed meanIS | full-sample-best = committed? | honest nested meanIS (eval) | honest verdict |
|---|---|---|---|---|---|---|
| **gap** | 0.078 | pure-time h24/SL2.0 | **+0.685%** | **YES — h24 IS the best-mean pick (textbook §5f violation)** | **−0.85% to −1.14%** (5-neg) | **mean-NEGATIVE** (NEW) |
| **me_long** | 0.531 | sl_only/SL2.0 | +0.232% | no (best=partial-runner +0.24%, ≈same) | **+0.147% to +0.203%** | **ROBUST, ≈committed** (NEW) |
| fbr | 0.107 | trailing_atr/SL2.0 | +2.084% | no (best=trailing_swing/SL1.5 +4.26%) | +1.00% to +1.20% (afp/worst) | ~40% haircut, stays + (≡2040) |
| me_short | 0.284 | partial-runner/SL2.0 | +0.683% | no (best=tp_3r/SL2.5 +0.78%) | −0.03% to −0.41% | NEGATIVE (≡2041) |

- **gap** is the cleanest §5f violation in the corpus: its committed horizon-24 IS the full-sample-
  best-mean pick (the headline and the fished number are the SAME config). Under honest no-lookahead
  horizon selection gap is mean-NEGATIVE (eval) / ≈0% (all-10) — its committed +0.685% rests almost
  entirely on the single 2012 +8.23% outlier fold, which honest selection cannot rely on (gap is
  thin/threshold-fragile, arc 1009). gap JOINS me_short on the exit-fragile list.
- **me_long is ROBUST** (honest +0.203% ≈ committed +0.232%; honest selection lands on partial-runner
  ≈ baseline). The heaviest RP leg (0.531) is the one NOT exit-fished → vindicates arc 1012.
- gap registry side-check (h24): pure-time/sl_only (+0.685%) is mid-pack; trailing_atr (+0.898%) is
  the full-sample best but that is itself fishing — confirms the meaningful gap hyperparameter is the
  horizon, not the SL-exit family (arc 1007).

**BOOK MEAN — committed vs honest (the deploy number).** Committed RP weights gap=0.078 /
me_long=0.531 / fbr=0.107 / me_short=0.284 (reproduces arc 1033).

| book | mean (all-10) | eval-mean (8) | worst | AFP |
|---|---|---|---|---|
| **COMMITTED RP** | **+0.594%** | +0.275% | −0.46% | False |
| HONEST RP, commit-wts, mean_roi | +0.407% | +0.337% | −1.08% | False |
| HONEST RP, commit-wts, afp_then_mean | +0.291% | +0.034% | −0.79% | False |
| HONEST RP, commit-wts, worst_then_mean | +0.267% | +0.109% | −0.68% | False |
| HONEST RP, **honest-refit wts**, mean_roi | +0.156% | +0.088% | −0.67% | False |

⇒ the committed **+0.59% overstates the honest §5f deploy mean by ~1.5–2×**: honest ≈ **+0.27% to
+0.41%** RP (commit weights), or ~+0.16% if RP weights are re-fit on the honest (lower-vol) series.
The book **survives mean-positive under every honest metric & weighting** — carried by the robust
me_long (wt 0.531) + haircut-but-positive fbr, diluting the now-negative gap + me_short. Still never
AFP (unchanged).

**FROZEN-EXIT OOS book (2021+, each leg's nested-frozen exit scored ONCE, §4/§5f).** Per-component
freeze-and-score is §5f-mandated (as 2040/2041 did per leg); the combined-book line is a
CHARACTERIZATION data-point, NOT a book-OOS gate (the combined-book holdout AFP gate stays the
operator's §5g firewall, arc 2022/1032).
- mean_roi frozen exits → OOS RP **−0.033%** (the §5f trap: mean_roi picks fbr trailing_swing/SL1.5,
  the high-variance exit that dies forward — 2040/2041 signature).
- afp_then_mean → OOS RP **+0.261%**; worst_then_mean → OOS RP **+0.194%** (both conservative metrics
  pick partial-runner exits; IS≈OOS consistent at ~+0.2%).
- ⇒ the trustworthy honest deploy estimate (conservative metrics, IS≈OOS) is **~+0.2–0.3%/yr**, all
  NOT-AFP.

**Significance implication (labelled estimate, exact recompute owed).** arc-1023's book-mean t=2.66
(p≈0.026) was computed on the COMMITTED-exit series. The honest deploy mean is ~half; at a similar
per-fold sd (~0.7%, arc 1023) that implies a honest-exit book t≈1.2 → **likely no longer significant
at 0.05.** This is an estimate from the mean-haircut, not a computed sd — an exact honest-series
t-stat recompute is the owed follow-up. It MATTERS for path-A: the "statistically-significant mean-
positive" pillar (arc 1023/2019) weakens under honest exits.

## Verdict
**DIAGNOSTIC → KILL** (no new component). Components UNCHANGED (all 4 PORTFOLIO). The book is NOT
deployable-improved; this CHARACTERIZES the honest deploy mean for the operator's path-A call.

**FLAGS (docs only, no code change):**
- **F1 — the book's deploy-relevant mean is exit-optimistic by ~1.5–2×.** Committed +0.59% RP →
  honest §5f ≈ +0.27–0.41% (commit wts) / ~+0.2–0.3% IS≈OOS (conservative metrics). path-A should
  use the honest number, not the committed headline.
- **F2 — gap JOINS me_short as a component mean-NEGATIVE under honest §5f exit selection** (gap's
  committed exit IS the full-sample-best pick). By §11 (can't diversify net-negatives positive) the
  PORTFOLIO status of BOTH gap and me_short is questionable on the conservative reading — which would
  leave the book leaning on me_long (robust) + fbr (haircut). FAIR caveat (§8, as 2041): the
  committed exits are defensible FIXED mechanism-motivated choices (gap's overshoot-harvest horizon,
  me_short's partial-runner) and as fixed choices stay +; the nested negativity is partly small-n (8
  eval folds) selection variance → this is a FLAG (which standard governs = operator call), not a
  unilateral downgrade.
- **F3 — arc-1023/2019's t=2.66 significance pillar likely does not survive honest exits** (estimated
  t≈1.2); exact recompute owed.

**What HOLDS:** me_long (heaviest RP leg) is exit-ROBUST (confirms 1012); the honest book stays
mean-positive; the AFP failure is unchanged (always was, not an exit artifact). The lever remains
operator path-A — now with a corrected (lower) deploy mean as input.

## (i) New lesson
The full-sample-best-exit optimism is **component-dependent and spans the full range** — ~0 haircut
(me_long, robust) → ~40% (fbr, stays +) → SIGN-FLIP to negative (gap, me_short). At BOOK level it
**partially washes out** because risk-parity puts the heaviest weight (0.531) on the one robust leg
(me_long) and down-weights the high-variance fished legs (gap 0.078) — so "2 of 4 legs are exit-
fished" does NOT mean "the book is fake," but it roughly HALVES the deploy mean (+0.59%→~+0.27%) and
concentrates the residual edge on me_long. Re-derive a thin book's deploy mean leg-by-leg under §5f
nested-WFO selection before trusting the committed headline; the honest book mean (not the committed
one) is the number path-A deploys, and it likely loses the t=2.66 significance the deploy case leaned
on.

## (k) Re-orient
Detail persisted (this doc + log). Tools: REUSED `nested_exit_selection` (2040) + `combine_fold_roi`
(2006) — no new BUILT tool (gap horizon-sweep was inline config construction). Driver:
`_disco_work/arc1042_honest_exit_book.py`. No canonical change, no council. OOS touched only via the
§5f-mandated per-component frozen-exit score (as 2040/2041); combined-book AFP gate left to the
operator (§5g). Next: resume at arc 1043.
