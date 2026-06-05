# arc 2021 — ADVERSARIAL temporal-stability stress of the 4-component book's mean-positive edge

**chat:** 2000s | **date:** 2026-06-05 | **disposition:** KILL (DIAGNOSTIC; no new component; components UNCHANGED, still PORTFOLIO; book stays strict-gate FAIL)

## Step (a) — log read
Pulled main (concurrent **arc 1025** from 1000s landed: `fbr` does NOT thicken — fold-resolution & edge-strength
are coupled through trigger depth → arc-2017 option-B closed for the best edge via the *depth* lever, complementing
my arc 2020 closing it via the *entry-resolution* lever, and arc 2018 closing it for `me`). STOP absent. Highest
arc-id in range (2000–2999) = 2020 → resume at **2021**.

## Step (b) — idea (the unasked adversarial question)
Arc 2019 established the book is a sound ~3-independent-bet, mean-positive PORTFOLIO (risk-parity **+0.589%**,
P(mean<0)=0.004) whose all-folds-positive failure is a *gate-resolution artifact* → the deployability lever is the
operator's gate-governance call. Arcs 2016/2017/1023/2019 all characterized the book's **noise** (within-fold CI,
across-fold SD, ENB, tail-corr, mean-CI) — **none characterized its behavior in TIME.** Before anyone leans on the
book's mean for a deploy decision, the first thing any quant asks is: *is the edge stable across the decade, or
front-loaded in an early-era (2011–2015) regime that has since decayed?* A decayed edge makes path-A hopeless (and
should downgrade the book); a stable edge genuinely strengthens the deploy case. **Conservative bias: try to BREAK
the mean — show it is a front-loaded artifact.** No gate loosened, no OOS spent.

## Step (c) — method (canonical only)
Reproduced the 4 committed components EXACTLY via arc 2019's frozen configs (script:
`_disco2000_work/arc2021_temporal_stability.py`); headlines verify to the decimal (gap +0.685% / me_long +0.232% /
fbr +1.854% / me_short +0.683%). Combined to the per-year (2011–2020) risk-parity & equal-weight book ROI (frozen
weights), split EARLY (2011–2015) vs LATE (2016–2020), bootstrapped each half's book-mean CI and the early−late
decay CI (5000×, seed 42), per component and per book.

## Result — the adversarial hypothesis is NOT supported (edge is temporally robust)
**Per-year book ROI (risk-parity):** 2011 +2.06 / 2012 +1.62 / 2013 +0.20 / 2014 +0.40 / **2015 −0.42** ‖ 2016 +0.03
/ 2017 +0.18 / **2018 −0.31** / 2019 +1.54 / 2020 +0.58 (%).

1. **The LATE half is still mean-positive** — risk-parity **+0.404%** (bootstrap P(<0)=0.060, ~94% one-sided), 1/5
   neg (2018 only). The edge does NOT die after 2015.
2. **Decay is NOT statistically significant** — EARLY +0.773% vs LATE +0.404%, difference +0.365% with 95% CI
   **[−0.636%, +1.326%] (spans zero, P(decay>0)=0.76)**. Cannot reject "late ≥ early."
3. **Components decay HETEROGENEOUSLY** — fbr is front-loaded (early +2.975% → late +0.733%, 0/5→1/5 neg) but
   me_long *strengthens* late (+0.055% → +0.409%, 2/5→1/5 neg); gap +0.97→+0.40, me_short +1.11→+0.26. The book's
   time-stability comes from the SAME multi-bet decorrelation arc 2019 found in the fold-correlation — now on the
   **time axis** (me_long backfills as fbr fades). The book is not a single front-loaded factor.

## Honest caveats (conservative — what this does NOT prove)
- **Low power:** n=5 per half → wide CIs; "no significant decay" is weak evidence of stability, not strong proof
  (the test also can't confidently assert decay).
- **Weighting-dependent:** the late-half robustness leans on risk-parity; under equal weight the late mean is
  +0.449% but P(<0)=0.249 (noisier, driven by 2018's −0.31 and 2016's +0.03 thinness).
- **fbr front-loading is real:** the corpus's strongest leg is materially weaker recently (early +2.98 → late
  +0.73); the book leans less on fbr and more on me_long in the late half. Not a decay of the BOOK, but worth flagging.
- **IS only:** OOS (2021+) is untouched (the book has not earned it). Late-IS persistence is the best available
  proxy for 2021+, but it is a proxy, not the holdout.

## Verdict + meaning
**The book's mean-positive edge is NOT a front-loaded 2011–2015 artifact — it persists into 2016–2020 with no
statistically significant decay.** This is the adversarial test failing to break the mean: it **removes one
objection** to the operator's path-A (the edge is not a decayed early-era regime) and modestly strengthens the
deploy case — while changing nothing about deployability (the book still fails strict all-folds-positive; no gate
was altered; this is characterization, not a verdict). DIAGNOSTIC → KILL (no new component). Components UNCHANGED
(all 4 PORTFOLIO). No OOS, no council (a measurement informing a measurement; not an edge fork or survivor).

## Threads
- The book is now characterized on noise (2016/2017/1023/2019) AND time (2021): a sound ~3-bet, mean-positive,
  temporally-robust PORTFOLIO that fails only the calendar-year AFP gate sitting below its noise floor. The lever
  remains the operator's gate-governance call; this arc supplies the temporal-robustness input that call needs.
- Edge-frontier for the book is closed (option-B via fbr closed at depth [1025] + entry-resolution [2020]; via me
  [2018]; 5th reversion leg can't help [2019]). Open only: operator gate call, or a thick fold-resolving standalone
  (none known; data FX-only).
