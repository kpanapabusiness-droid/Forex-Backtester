# arc 1045 — Honest-exit 2-way {me_long + fbr}: is the clean pair a better deploy object than the 4-way?

> chat: 1000s | range 1000-1999 | timestamp: 2026-06-06
> disposition: **DIAGNOSTIC → KILL** (no new component; 2-way still NOT all-folds-positive)
> components UNCHANGED (all 4 PORTFOLIO). OOS NOT touched (IS characterization; fails IS AFP → §5g firewall).

## (a) Log read / synthesis
Pulled main; read DISCOVERY_LOG (both tiers, arcs 0→2043), LESSONS, TOOL_REGISTRY; STOP absent.
State: single-condition shallow directional prediction is closed ground (H1/H4/D1/W1, long+short); the
data corpus is **pure FX** (28 majors/crosses — verified `histdata_backup`; no metals/indices, so the
"instrument-universe-beyond-FX" lever is closed by data availability). **4 PORTFOLIO components** exist —
gap (1006), me_long (1011), fbr (1013, the 9/10 leg), me_short (1019, the only robustly-+2018 leg). The
4-way book is mean-positive but NOT all-folds-positive (blocked by 2015 & 2018); ~18 routes to a
+2015/+2018 leg are dead; §11 independent verification COMPLETE (signal+outcome+cost honest).

The live recent thread (my arcs 1042/1043/1044 + 2000s 2040/2041/2042/2043): the §5f exit-honesty audit.
Under honest nested-WFO exit selection **gap + me_short flip mean-NEGATIVE** (exit-selection artifacts —
their committed exits were full-sample-best picks), **fbr takes a ~40% haircut but stays +**, and
**me_long is exit-robust**. Consequences for the deploy case: book mean ~halved (+0.59%→+0.27%, 1042),
significance lost (t 2.66→~1.3, CI spans zero, 1043), ENB halved (3.32→~1.8, 1044). arc 1044 found the
4-way still beats me_long-SOLO under conservative metrics because **fbr is THE diversifier** (honest corr
−0.44…−0.65 with me_long) — but it never tested the obvious next object.

## (b) Idea (the because)
1044 compared the 4-way vs the 1-leg solo and stopped. The unanswered, deploy-relevant question sits
exactly between them: **if 2 of the 4 legs (gap, me_short) are mean-negative exit-artifacts and the real
diversification lives entirely in the me_long↔fbr anti-correlation, is the cleanest honest deploy object
the 2-way {me_long + fbr} — and does DROPPING gap + me_short IMPROVE the book** (higher Sharpe, fewer neg
folds, recovered significance, closer to AFP)? me_long carries 2018 (+0.90), fbr carries 2015 (+3.17) —
the two binding folds — and they are the most strongly anti-correlated pair in the book. No arc has tested
this 2-way on the honest-exit basis (arc 2006 = gap+me_long; 2008/3009/1015 = 3-way incl. gap; the gap=0
edge of 2008's committed convex search is the closest, but never on the honest §5f series, never as the
headline deploy candidate, never with the Sharpe/significance comparison). A negative-mean leg only earns
its place if its decorrelation benefit exceeds its mean drag — under honest exits that is now in doubt for
gap + me_short, and this arc tests it directly.

## (c)/(f) Method
100% canonical scoring (`ArcFoldRunner` → `A1Architecture` → `MultiPairBacktester`, FundedNext); honest
§5f per-leg series via BUILT `nested_exit_selection` (3 metrics: mean_roi / afp_then_mean /
worst_then_mean); book combination via BUILT `combine_fold_roi` (equal / risk-parity / a full convex
w_me∈[0,1] scan). **Honest series = the EVALUABLE (non-warmup) folds only** — 8 folds 2013-2020; the 2
warmup folds (2011/2012) are dropped because their nested value is the global frozen pick (a lookahead
taint), per arc 2043. Both the 2-way and the 4-way are scored on the SAME 8 evaluable folds, so the
comparison is apples-to-apples honest (this also corrects arc 1044, which mixed in the 2 warmup folds).
Fold-bootstrap mean/sd/t/Sharpe/CI/P(mean<0) (seed 42, N=10000). Driver
`_disco_work/arc1045_honest_2way_melong_fbr.py`. No new tool, no canonical change, OOS untouched.

## Results

**Honest corr(me_long, fbr) = −0.54 to −0.68** (strongly anti-correlated — the real diversification pair).

**The 2-way is NOT all-folds-positive — the convex scan finds NO weighting that passes** (best worst-fold
−0.641% / −0.209% / −0.217% across the 3 metrics). Binding folds: **2014** (both legs mildly negative:
me_long −0.23, fbr −0.14 — fundamentally unrescuable by a 2-leg book) **+ the 2015↔2018 mutual-exclusion**
(2015 positive ONLY via fbr-heavy +3.5; 2018 positive ONLY via me_long-heavy +0.90, fbr −3.4 — opposite
weight demands, neither solo is AFP). Same wall as the 4-way (arcs 2008/3009/1015) → **KILL on the gate.**

**But the 2-way is a materially BETTER deploy object than the 4-way** (honest §5f, 8 evaluable folds):

| object | metric | mean | Sharpe(fold) | t | P(mean<0) | 95% CI | neg folds |
|---|---|---|---|---|---|---|---|
| me_long SOLO | afp | +0.203% | +0.31 | +0.72 | 0.219 | [−0.34,+0.69] | 3/8 |
| 4-way BOOK (RP) | afp | +0.096% | +0.14 | +0.39 | 0.333 | [−0.37,+0.51] | 3/8 |
| **2-way me_long+fbr (RP, w_me=0.74)** | **afp** | **+0.411%** | **+0.75** | **+2.12** | **0.008** | **[+0.07,+0.77]** | 2/8 |
| 4-way BOOK (RP) | worst | +0.120% | +0.20 | +0.57 | 0.278 | [−0.26,+0.51] | 4/8 |
| **2-way me_long+fbr (RP, w_me=0.80)** | **worst** | **+0.357%** | **+0.69** | **+1.95** | **0.015** | **[+0.02,+0.69]** | 3/8 |
| **2-way me_long+fbr (RP, w_me=0.94)** | **mean_roi** | **+0.398%** | **+0.66** | **+1.87** | **0.032** | **[−0.03,+0.73]** | 1/8 |

- **~2× the Sharpe of the 4-way** under every metric (0.66–0.75 vs 0.13–0.20), and **it RECOVERS the
  statistical significance the honest 4-way LOST** (arc 1043: honest 4-way t≈1.3, CI spans zero). Under
  the afp metric the 2-way RP is **t=+2.12, P(mean<0)=0.008, CI=[+0.07,+0.77] EXCLUDING ZERO**; under
  worst it is t=+1.95, CI=[+0.02,+0.69] excluding zero.
- **gap + me_short are NET DRAGS under honest exits, not diversifiers** — adding them to the clean 2-way
  HALVES the Sharpe and kills the significance. Their negative honest mean outweighs their decorrelation
  benefit; ALL the real diversification value is in the single me_long↔fbr anti-correlation (−0.57). This
  sharpens arc 1044 ("not reducible to just me_long") to its precise form: **the honest book IS reducible
  to me_long + fbr; the other two legs subtract.**
- RP heavily down-weights fbr (w_me 0.74–0.94) because fbr is so volatile (sd 2.3–12.2 vs me_long 0.7) —
  so the 2-way is "me_long with a small decorrelating fbr sleeve," and that sleeve is exactly what lifts
  the non-significant me_long-solo (t≈0.7) to significant.

**Honesty caveats (§8, Arc-10 defense — not over-claimed):**
- Significance is **weighting- and metric-dependent**: the equal-weight 2-way is positive but NOT
  significant (afp t=+1.76, CI=[−0.04,+1.18] spans zero); the significance appears under RP weighting.
  RP is a principled no-fold-sign-lookahead weighting (inverse-vol fit on IS variance only), so the claim
  is legitimate — but it is **borderline at n=8** and rests on the RP choice. Honest statement: the 2-way
  is **borderline-to-significant under RP** (t 1.87–2.12, CI excludes zero in 2 of 3 metrics), clearly
  better than the 4-way, but not a robust >2σ result.
- This is **IS only.** OOS (2021+) was deliberately NOT touched: the 2-way fails IS AFP, so the §5g
  combined-book holdout firewall (arcs 1042/2043) blocks an OOS book gate — consistent with every recent
  book arc. A real deploy claim for the 2-way needs it frozen (weights + per-leg §5f-frozen exits) and
  scored ONCE on OOS — an explicit OWED operator/§5g step, not done here (conservative bias).

## Verdict
**DIAGNOSTIC → KILL** (no new component; the 2-way is NOT all-folds-positive — 2014 + the 2015↔2018
mutual-exclusion block every convex weighting, same wall as the 4-way). Components UNCHANGED (all 4
PORTFOLIO; the 4-way book's AFP failure is intrinsic, not fixed by trimming to 2 legs).

**FLAG (operator-facing, docs only, no code change) — the honest deploy object is the 2-way, not the
4-way.** Under honest §5f exits the path-A deploy book should be **{me_long + fbr} at risk-parity**, NOT
the 4-way: it has ~2× the Sharpe (0.66–0.75 vs 0.13–0.20), recovers borderline significance the 4-way
lost (afp t=+2.12, P(mean<0)=0.008, CI excludes zero vs 4-way t=0.39, CI spans zero), and fewer negative
folds — because gap + me_short, whose committed exits were full-sample-best picks, are net drags once
exits are honest (negative mean > decorrelation benefit). This **partially reverses** the implicit
"deploy all 4 legs" framing of the committed-exit characterization (2016/2019/1023): two of the four legs
should be DROPPED for the honest-exit deploy. Caveat: the 2-way's significance is RP-weighting-dependent
and borderline at n=8; it remains NOT all-folds-positive; and its OOS book confirmation (frozen weights +
exits, scored once) is owed before any deploy. The discovery gate (AFP) verdict is unchanged — KILL.

## (i) New lesson
Once the §5f exit-honesty correction is applied, a thin multi-leg book is NOT improved by keeping every
decorrelated leg — a leg whose committed headline was a full-sample-best exit pick (gap, me_short) can go
mean-negative under honest exits, and then its decorrelation no longer pays for its mean drag, so it
SUBTRACTS from the book. Dropping the two exit-artifact legs and keeping only the two robust,
strongly-anti-correlated legs (me_long + fbr, corr −0.57) **doubled the honest book Sharpe and recovered
the statistical significance the full 4-way lost** — yet the trimmed book is STILL not all-folds-positive
(2014 + the 2015↔2018 mutual-exclusion are leg-count-invariant). So: (1) re-select a book's leg SET, not
just its weights, under honest exits — fewer-but-cleaner can beat more-but-exit-fished; (2) a Sharpe/
significance improvement from trimming is real and deploy-relevant even when the AFP gate verdict is
unchanged — characterize the honest book on its best leg subset, not the historically-accumulated full
set.

## (k) Re-orient
Detail persisted (this doc + log). Tools REUSED (`nested_exit_selection` 2040, `combine_fold_roi` 2006);
no new BUILT tool. Driver `_disco_work/arc1045_honest_2way_melong_fbr.py`. No canonical change, no
council (not a survivor; portfolio-construction characterization), OOS untouched. Next: resume at arc 1046
(or graceful handoff if context low).
