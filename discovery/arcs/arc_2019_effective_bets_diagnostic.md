# arc 2019 — DIAGNOSTIC: how many INDEPENDENT bets is the 4-component book? (council-driven)

**Chat:** 2000s | **Date:** 2026-06-05 | **Disposition:** KILL (diagnostic; no new component; components UNCHANGED)

## Idea + because

Resuming behind arcs 2016/2017/2018 (the route is noise-floor-capped, operator-flagged, and my arc 2018
closed the cross-sectional thicken-`me` lane). At this genuine strategic fork I convened
`/llm-council-discovery` (generative juncture, light weight). The five lenses split — refine `fbr`-2018
via an M1 reclaim-confirmation filter (Refinement), test `fbr` alone on OOS (Soundness, warning of a
"laundering" of a noise-mined 100%-reversion book), hunt a novel event-anchored liquidity mechanism
(Mechanism/Steelman) — but the **peer-review round converged, 3 of 5 reviewers independently, on the one
thing all five lenses MISSED:** nobody had measured the **effective number of independent bets in the
EXISTING 4-component book**. That single unmeasured quantity gates every branch:

- If the book is effectively **rank-~1** (one reversion factor sliced four ways), then "0/all convex
  weightings all-folds-positive" is **mechanically inevitable** → no 5th thin leg or gate reframe helps →
  the lever is the operator gate-governance call.
- If genuinely **multi-bet in the body but rank-low in the TAIL** (all bleed together in strong-USD
  risk-off 2018), then no 5th **reversion** leg can fix it (it shares the tail) → only a NON-reversion
  factor (trend/continuation = coin-flip-dead) or the gate call remains.

This is cheap (already-scored components, no OOS spend), decisive, and extends the arcs-2016/2017
per-component noise-floor diagnostic to the **cross-component** dimension. CC committed to measuring it
(the council recommends; CC commits).

## Method

Reproduced the 4 committed components EXACTLY (reusing arc 2015's configs: gap-fill JPY-crosses H4 sl_only
24-bar; me_long USD-majors D1 sl_only 2-bar; fbr USD-majors H4 trailing_atr; me_short USD-majors D1
partial-runner), all via the canonical apparatus (`ArcFoldRunner` → `MultiPairBacktester`,
`run_config_over_folds`, `build_v3_folds`). Computed on the 10 IS folds (2011–2020):
(1) per-fold ROI correlation matrix; (2) eigenvalue spectrum + **effective number of bets ENB = (Σλ)²/Σλ²**
(participation ratio); (3) bootstrap CI of ENB (resample folds, 5000×, seed 42); (4) tail co-movement
(co-negativity per fold; pairwise corr in the book's worst vs best folds); (5) book pooled per-fold MEAN
CI under frozen equal + risk-parity weights. Driver `_disco2000_work/arc2019_effective_bets.py`.
Headlines reproduce committed values exactly (gap +0.685 / me_long +0.232 / fbr +1.854 / me_short +0.683).

## What happened — the rank-1 / shared-tail fears are REFUTED; the gate-thinness diagnosis is CONFIRMED

**(1) Correlation matrix (n=10) — low, with a useful anti-correlation:**
gap·me_long +0.117, gap·fbr +0.189, gap·me_short +0.188, me_long·fbr **−0.366**, me_long·me_short +0.157,
fbr·me_short +0.406. Not rank-1 (a rank-1 book has all pairwise ≈ +1).

**(2) Effective number of bets = 3.32 / 4.** Eigenvalues [1.556, 1.257, 0.829, 0.358]; the top eigenvalue
explains only **38.9%** of cross-fold variance. The book is genuinely **~3 independent bets** — style-
homogeneous (all four are reversion/fade) but **statistically decorrelated**, because the events are
disjoint (weekends vs month-ends vs stop-runs). The Soundness lens's "one bet sliced four ways" is
empirically false.

**(3) Bootstrap ENB 95% CI = [2.02, 3.33]** (mean 2.75), **P(ENB<2)=0.021**. The diversification is
**robust to n=10 sampling noise** — ≥2 effective bets with 97.9% confidence. (This is the cross-component
complement to arcs 2016/2017: per-component folds don't resolve, but the cross-component INDEPENDENCE
does resolve as real.)

**(4) Tail co-movement is NEGATIVE — there is NO common crash (the decisive finding):**
mean pairwise corr in the book's **5 WORST folds = −0.201**, vs +0.123 in the 5 best (and +0.115 overall).
Co-negativity never exceeds **2/4** in any year (2013/2015/2016/2018), and is 0/4 in 2012/2019. The famous
**2018** binding fold is **2-down (gap −6.79, fbr −4.20) / 2-up (me_long +0.90, me_short +0.86) near-
cancellation**, NOT a book-wide drawdown. arc 2008's "the components share a 2018 tail" is refuted — when
one bleeds, the others tend to hold or profit.

**(5) Book MEAN is robustly positive:** risk-parity mean **+0.589%, 95% CI [+0.120%, +1.088%],
P(mean<0)=0.004**; equal-weight mean +0.863%, CI [−0.150%, +1.863%], P(mean<0)=0.046. The risk-parity
book is a genuine mean-positive diversified PORTFOLIO (the Soundness "−0.115% rounding-error" was the
worst-FOLD under optimized weights, not the mean).

## Read + verdict — KILL (diagnostic), and the operator FLAG is now decisively grounded

The book is **NOT** rank-1, **NOT** tail-correlated, and its mean is **NOT** a fake — all three of the
council's "laundering" fears are refuted with numbers. It is a genuinely ~3-independent-bet, negative-
tail-co-movement, robustly-mean-positive reversion portfolio. **Therefore its failure to pass all-folds-
positive is purely the arcs-2016/2017 thinness problem expressed at the book level:** the every-calendar-
year gate is applied below the thin components' per-fold noise floor, so in any given year ONE thin leg
dips within-noise-negative and drags that fold marginally under zero — even though nothing co-crashed
(every worst fold is a single-thin-leg noise dip, never a co-drawdown).

This **resolves the generative-vs-governance fork**: a 5th *decorrelated reversion* mechanism **cannot make
the book all-folds-positive** — the book already has no diversification deficit and no common tail to
hedge; an added thin leg merely adds one more series that can itself dip within-noise in some year (one
more chance to trip the every-year gate). The only thing that could pass the *all-folds-positive* route is
a component **thick enough that its own folds resolve positive** (arc 2017 option B) — but such a component
would be all-folds-positive *solo* (a deployable system on its own, dissolving the "book" framing), and
arc 2017 established no such thick non-coin-flip edge exists on this data (closed ground). **The binding
lever is definitively the operator's gate-resolution governance call** — now proven from the cross-
component/diversification angle, not only the per-component angle.

**Verdict: KILL** (diagnostic; no new component; components UNCHANGED, still PORTFOLIO). No OOS spent
(book still fails IS all-folds-positive). No council-survivor stress-test (no survivor). The generative
council's best NOVEL candidate (option-expiry/gamma-pin, Mechanism lens) was unanimously flagged by
reviewers as **not constructible from FX OHLC** (needs options/strike data the programme lacks → collapses
into round-number behavior, already dead, arc 1010) — recorded so it is not re-proposed.

## Threads / lessons

1. **NEW (decisive): the 4-component book is genuinely diversified — ENB 3.32, robust CI [2.02, 3.33] —
   with NEGATIVE tail co-movement (worst-fold pairwise corr −0.201).** Style-homogeneity (all reversion)
   does NOT imply factor-homogeneity: disjoint event-timing makes the fold-ROIs ~3 independent bets. The
   "one reversion bet sliced four ways" intuition is empirically wrong.
2. **NEW: every worst fold is a single-thin-leg within-noise dip, not a co-drawdown** (max 2/4 co-negative;
   2018 is 2-up/2-down near-cancellation). So the all-folds-positive failure is a *gate-resolution* artifact
   (per-year granularity below the noise floor), NOT a portfolio-construction failure. This upgrades arcs
   2016/2017's operator FLAG from "per-component vacuous" to "the book is a sound PORTFOLIO that the
   calendar-year gate cannot certify."
3. **A 5th decorrelated reversion component is LOW-EV for the book** — it cannot fix a gate that trips on
   single-leg noise dips, and the book needs no more diversification. Edge-hunting *for the book via the
   all-folds-positive route* is closed; only a thick fold-resolving standalone (arc 2017 option B, none
   known) or the operator gate-resolution call remains.
4. **Council process worked exactly as intended** (Arc-10 insurance): the generative lenses each proposed
   a different action; the anonymous peer-review surfaced the decisive unmeasured quantity ALL of them
   missed; measuring it refuted the strongest dissent (Soundness's laundering charge) with data rather
   than rhetoric.
5. The option-expiry/gamma-pin mechanism (Mechanism lens) is **not OHLC-constructible** (no strike/options
   data) — do not re-propose; it reduces to round-number behavior (dead, arc 1010).

## Tooling

No new BUILT tool — reused the canonical apparatus + the BUILT `combine_fold_roi` (arc 2006) and the four
committed component signals; the diagnostic is eigenvalue/bootstrap arithmetic (`numpy.linalg.eigvalsh` +
resampling) on canonical per-fold ROI. No canonical core touched. Driver
`_disco2000_work/arc2019_effective_bets.py` (reproduces the 4 committed headlines exactly).

**FLAGS (code not merged):** none new. **OPERATOR FLAG reinforced (3rd time, now from the cross-component
angle):** the 4-component book is a genuine ~3-independent-bet, negative-tail, mean-positive PORTFOLIO; the
all-folds-positive *calendar-year* gate cannot certify it because the gate resolution sits below the thin
legs' per-fold noise floor. Honest operator options unchanged from arc 2017: (A) a noise-aware /
pooled-trade / coarser-window gate; (B) a thick fold-resolving standalone component (none known — closed
ground). Discovery has mapped the accessible reversion frontier; the lever is the gate-resolution call.
