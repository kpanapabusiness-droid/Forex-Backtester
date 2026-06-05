# arc 3021 — PATH-B tractability: how many decorrelated legs would the per-year AFP gate need?

**Chat:** 3000s · **Date:** 2026-06-05 · **Verdict:** DIAGNOSTIC → **KILL** (no new component; components UNCHANGED, all 4 PORTFOLIO; book stays strict-gate FAIL)
**Disposition:** KILL · **passed:** N · **Component touched:** none (portfolio-math diagnostic)

> The corpus's sole remaining deployability lever is the operator's path-A (change the gate) vs
> path-B (build a denser book) call — left QUALITATIVE by every prior arc. This quantifies path-B.
> **Result: path-B is intractable, and at the corpus's own realistic residual correlation (ρ≈+0.115)
> it is IMPOSSIBLE — P(all-folds-positive) plateaus at ~0.30 regardless of how many legs you add.**
> Adding decorrelated positive-mean legs cannot satisfy the per-year gate. Path-A is the only route.

---

## Log reading (step a — FRESH EYES, honest-era only)

Pulled main; no `discovery/STOP`. Resumed 3000s after my arc 3020 (highest in-range now 3020 → 3021).
Concurrent landings since my last read: **arc 1025** (1000s — `fbr` does NOT thicken; option-B closed via
the depth lever), **arc 2021** (2000s — the book's mean-positive edge is temporally robust, NOT a
front-loaded 2011–2015 artifact, no significant decay), **arc 1026** (1000s — session/overnight reversal).

Converged corpus state (~56 arcs, honest-era): closed-ground directional space; 4 net-positive PORTFOLIO
components (gap 1006 / me_long 1011 / fbr 1013 / me_short 1019), none all-folds-positive; the 4-way book
is a sound ~3-independent-bet (ENB=3.32, arc 2019), mean-positive (t=2.66, P(mean<0)=0.004, arc 1023),
**temporally robust** (arc 2021) PORTFOLIO whose all-folds-positive (AFP) FAILURE is purely the per-year
gate sitting **below the legs' noise floor** (arcs 2016/2017/1023/2019/2021). The edge frontier is closed
on every documented lever: option-B (a thick fold-resolving standalone) closed for `fbr` via depth (1025)
+ entry-resolution (2020) + **breadth (my 3020)** + level (3013) + regime (2014), and for `me` (2018);
arc 2019 concluded "a 5th *reversion* leg can't help." The repeatedly-flagged remaining lever is the
operator's **path-A (gate-resolution) vs path-B (denser components)** call — and **no arc has quantified
path-B.** That gap is this arc.

## Idea + why (sharpen arc 2019 from qualitative to a portfolio-math number)

arc 2019's "a 5th leg can't help" is a *fold-painting* argument (an added thin leg adds another
every-year trip-chance). But basic portfolio math points the other way and was never checked: averaging
N decorrelated positive-mean legs shrinks the book's per-fold variance (~1/N) → book per-fold
Sharpe = per-leg-Sharpe·√N_eff → **P(AFP over 10 folds) = Φ(book-Sharpe)¹⁰ rises with N** → AFP is
reachable at SOME N even with thin legs. So the honest question is not "can a 5th leg fix a fold" but
**"at what N does densification clear the gate, and is that N tractable?"** — the exact input the
operator's path-A/B call needs. PRIME DIRECTIVE: this also TESTS arc 2019's conclusion rather than
inheriting it.

## Method (CALLED canonical reproduction + a transparent portfolio simulation)

Reproduced the 4 committed components EXACTLY via the canonical apparatus (same configs/exits as arcs
2019/2021: `ArcFoldRunner` → `MultiPairBacktester`, FundedNext costs, 0.5% risk FRACTION per arc 1024),
giving each leg's 10-year (2011–2020) per-fold ROI vector. Calibrated each leg's **per-fold Sharpe**
(mean/SD across folds) and the mean pairwise leg correlation. Then simulated an equal-weight book of N
i.i.d. legs, each leg's per-fold ROI ~ N(Sharpe, 1) (so mean/SD = the calibrated Sharpe), with optional
equicorrelation ρ across legs within a fold; computed **P(AFP) = P(all 10 book-fold ROIs > 0)** vs N, at
calibrated Sharpe levels and ρ ∈ {0, 0.1, 0.2}. DIAGNOSTIC only — no new component, no OOS, no gate
change, no council (a measurement informing a measurement). Driver:
`_disco3_work/arc3021_pathb_tractability.py`.

## What happened — path-B quantified and closed

**Reproduction verified (Arc-10 discipline):** gap +0.685% / me_long +0.232% / fbr +1.854% / me_short
+0.683% — exact. Per-leg per-fold Sharpe: gap 0.141, me_long 0.327, me_short 0.515, **fbr 0.604**
(median 0.42, mean 0.40). Mean pairwise leg corr **+0.115** (consistent with arc 2019's −0.366..+0.406 / ENB 3.32).

**Analytic gate requirement:** P(AFP over 10 folds) = 0.9 needs **book** per-fold Sharpe ≥ **2.31** (0.5
needs ≥ 1.50). The real 4-leg book is far below this (its worst fold ≈ 0 within noise → it fails AFP).

**(1) Decorrelated (ρ=0) — N required is large but finite:**

| per-leg quality (Sharpe) | N for P(AFP)≥0.9 |
|---|---|
| median corpus leg (0.42) | **≈ 30** |
| mean corpus leg (0.40) | ≈ 34 |
| best leg = fbr-class (0.60) | ≈ 15 |
| optimistic (0.50) | ≈ 22 |

i.e. even in the idealized zero-correlation case the book needs **~15–34** decorrelated positive-mean
legs of corpus-typical quality. The corpus has produced **4 in ~55 arcs (~1 per 14 arcs)** and has
declared the edge frontier closed → reaching N≈30 would take on the order of **hundreds more arcs** with
no known remaining mechanisms. Intractable.

**(2) Realistic residual correlation — IMPOSSIBLE at any N (the decisive result):**

| ρ | N=8 | N=15 | N=30 | N=50 | N=80 |
|---|---|---|---|---|---|
| 0.0 | 0.29 | 0.59 | 0.90 | 0.99 | 1.00 |
| **0.1** | 0.14 | 0.20 | 0.27 | 0.31 | **0.33 (plateau)** |
| 0.2 | 0.08 | 0.11 | 0.13 | 0.14 | 0.14 |

At ρ = 0.1 — **below** the corpus's empirical +0.115 — P(AFP) **plateaus around 0.33 and never reaches
0.9 no matter how many legs are added.** A shared common factor floors the book's per-fold variance at
ρ·σ² (it does NOT →0 with N), so the book Sharpe ceilings at (μ/σ)/√ρ and Φ(·)¹⁰ stays well under the
gate. FX reversion legs demonstrably share such a factor — the dollar / risk regime (arc 2008/2015/3009
found the shared 2015 & 2018 tails) — so the realistic case is the ρ>0 row, not ρ=0.

## Diagnosis — densification cannot satisfy the per-year gate; arc 2019 is right for a deeper reason

The per-year AFP gate demands a book per-fold Sharpe ~2.3. The corpus's legs are per-fold Sharpe ~0.4
(thin, by arc 1025's unified theory: real FX edges = intrinsically-rare forced-flow reversions). Closing
a 2.3/0.4 ≈ 6× Sharpe gap by averaging requires N≈30 IF the legs were perfectly decorrelated — already
intractable at the corpus's discovery rate — and is OUTRIGHT UNREACHABLE because the legs share a dollar/
risk-regime factor (ρ≈+0.12 > 0), which caps the achievable book Sharpe below the gate regardless of N.
This converts arc 2019's qualitative "a 5th reversion leg can't help" into a portfolio-math proof: it is
not that one more leg fails to fix one fold — it is that **NO number of legs at this Sharpe and this
residual correlation can make the per-year gate pass.** Path-B is closed.

## Verdict: DIAGNOSTIC → KILL (no new component)

No edge tested; reuses canonical reproduction; quantifies the operator's path-B option and finds it
non-viable. Components UNCHANGED (all 4 PORTFOLIO). No OOS spent (book hasn't earned it; this is
characterization). No council (a measurement informing a measurement, not an edge fork or survivor).
**The strict per-year all-folds-positive gate is NOT loosened** — this arc quantifies *why* path-B
cannot clear it, leaving path-A (the operator's gate-governance call) as the sole deployability route.

## Honest caveats (conservative)

- **Equicorrelation is a simplification:** real legs have heterogeneous (even negative, me_long·fbr
  −0.366) pairwise corrs; a negative-corr pair HELPS. But the MEAN is +0.115 (positive) and, decisively,
  the binding folds (2015/2018) are exactly where the legs CO-MOVE in the dollar tail (arc 2008) — so the
  ρ>0 model is fair, arguably optimistic about the tail that actually trips the gate.
- **i.i.d.-normal per-fold model** idealizes fat-tailed, thin-n fold ROIs; the qualitative conclusion
  (a positive shared factor caps P(AFP) below the gate) is robust to the distributional form — it is a
  variance-flooring argument, not a normality argument.
- This bounds path-B's *tractability*; it is not a claim that the book is or isn't deployable — that is
  the operator's path-A call, which this arc informs (path-B is not a substitute for it).

## Threads / lessons

1. **NEW, decision-critical: path-B (denser book) is quantitatively non-viable.** Decorrelated, it needs
   ~15–34 corpus-quality legs (hundreds of arcs at the discovery rate, frontier closed); at the realistic
   shared-factor correlation (ρ≈+0.12) P(AFP) plateaus ~0.30 and is unreachable at ANY N. ⇒ path-A (the
   operator's gate-governance call) is the ONLY route to deployability.
2. **Why the per-year gate is structurally unsatisfiable for this corpus (unified):** it demands book
   per-fold Sharpe ~2.3; FX forced-flow legs are per-fold Sharpe ~0.4 and share a dollar/risk factor →
   neither single-leg thickening (1025) nor multi-leg densification (this arc) can bridge it. This is the
   portfolio-level statement of arc 1025's "edges are intrinsically thin" + arc 2008's "shared tail."
3. **Independent-method confirmation of arc 2019** (ENB → here portfolio-variance simulation): same
   conclusion (sound book, AFP-uncertifiable), now with the quantitative mechanism (Sharpe × ρ).
4. **For the operator:** the realistic decision is path-A (mean/pooled/regime-block gate, with arc 2021's
   temporal-robustness + arc 1023's t=2.66 mean-positivity as support) — path-B is off the table. No
   autonomous gate change made.

## Tooling

No new BUILT tool — reused the canonical reproduction (arc 2019/2021 configs) + BUILT
`combine_fold_roi`. The path-B Monte-Carlo is a one-off diagnostic in the driver (not a reusable
experiment filter). Driver `_disco3_work/arc3021_pathb_tractability.py` (reproducible from this doc;
uses `scipy.stats.norm` for the analytic gate threshold).

## FLAGS (code not merged)

None. No canonical-core change. Reinforces the standing OPERATOR FLAG (arc 2016/2017/2019/1023/2021): the
per-year all-folds-positive gate sits below this corpus's noise floor and — now shown — cannot be cleared
by densification; the path-A vs path-B governance call is the operator's, and path-B is quantitatively
closed.
