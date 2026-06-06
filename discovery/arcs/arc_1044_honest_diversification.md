# arc 1044 — Honest-exit DIVERSIFICATION: does the book beat me_long SOLO? (the 3rd deploy pillar under honest exits)

> chat: 1000s | range 1000-1999 | timestamp: 2026-06-06
> disposition: **DIAGNOSTIC → KILL** (no new component; arc-1043 F2 diversification-pillar slice)
> components UNCHANGED (all 4 PORTFOLIO). OOS NOT touched (IS characterization only).

## (a) Log read / synthesis
Continuing the honest-exit deploy-case correction (my arcs 1042/1043, this chat). Established: under
honest §5f exits the book deploy mean is ~half (+0.27%, 1042) and NOT statistically significant
(fold-bootstrap t≈1.3, CI spans zero, 1043); 2 of 4 legs (gap, me_short) go mean-negative, fbr takes a
~40% haircut, me_long (RP wt 0.531) is the lone exit-robust leg. arc 1043 F2 flagged that the OTHER
deploy-case pillars — temporal stability (2021), cost cushion (3022), and the **diversification /
~3-independent-bets** count (2019, ENB 3.32/4) — were ALL computed on committed-exit series and are
suspect by the same mechanism. (Independent arc 2042 converged on the component verdicts.)

## (b) Idea (the because)
The diversification pillar is the structural justification for the whole multi-component programme
(arcs 1006→2019 built 4 legs precisely because arc-2019 measured ~3.32 independent bets — a real
portfolio, not one signal). If honest exits collapse that — if 2 legs go negative and the book
honestly reduces to "me_long with noise" — then the portfolio thesis itself, not just its magnitude,
was an exit artifact. The sharpest single test that subsumes ENB: **does the 4-way book beat its one
robust leg (me_long) SOLO on a risk-adjusted basis, under honest exits?** Plus the honest ENB as
supporting structure.

## (c)/(f) Method
- Canonical scoring; honest §5f series via BUILT `nested_exit_selection` (3 metrics); honest-refit RP
  weights via `combine_fold_roi`. Fold-bootstrap (seed 42, N=10000) me_long-SOLO vs 4-way BOOK
  (mean/sd/t/fold-Sharpe). Honest ENB = (Σλ)²/Σλ² of the 4-leg fold-ROI covariance (arc-2019 method);
  honest pairwise fold-ROI correlations. Driver `_disco_work/arc1044_honest_diversification.py`. OOS
  untouched.

## Results

**Honest ENB collapses ~3.32 → ~1.8/4** (1.41 mean_roi / 1.93 afp / 1.77 worst) — arc-2019's
committed-exit ENB 3.32 is roughly **HALVED**: the honest book is effectively ~1.5–2 independent bets,
not ~3.3. The diversification pillar is also exit-optimistic.

**Book vs me_long-SOLO (fold-bootstrap, honest-refit RP weights ~0.55 me_long):**
| honest metric | me_long SOLO (mean / Sharpe / t) | 4-way BOOK (mean / Sharpe / t) | risk-adjusted | ENB |
|---|---|---|---|---|
| mean_roi | +0.217% / +0.307 / +0.97 | +0.156% / +0.262 / +0.83 | **SOLO ≥ book** | 1.41 |
| afp_then_mean | +0.217% / +0.307 / +0.97 | +0.347% / +0.437 / +1.38 | **BOOK > solo** | 1.93 |
| worst_then_mean | +0.170% / +0.286 / +0.90 | +0.285% / +0.481 / +1.52 | **BOOK > solo** | 1.77 |

- Under the **conservative metrics** (afp/worst — the ones that held up OOS in 1042) the honest book
  **beats me_long-solo on risk-adjusted Sharpe** (+0.44–0.48 vs +0.29–0.31). The load-bearing
  diversifier is **fbr**, whose honest fold-ROI correlation with me_long is **−0.44 to −0.65** — even
  though gap + me_short are individually negative-mean, the decorrelation reduces book variance enough
  that the book Sharpe exceeds the solo leg. So the book is **NOT reducible to "just me_long."**
- Under the **mean_roi metric** (the §5f-trap that picks high-variance fbr trailing_swing) the book is
  **WORSE than solo** (+0.262 vs +0.307) — so even the "book beats solo" conclusion is itself
  exit-metric-dependent.
- me_long-SOLO is itself NOT significant (t≈0.9–0.97) — consistent with 1043 (everything thin at n=10).

## Verdict
**DIAGNOSTIC → KILL** (no new component; components UNCHANGED, all 4 PORTFOLIO). The honest portfolio
thesis **survives in WEAKENED, exit-metric-dependent form:** the effective-bet count is HALVED
(ENB 3.32→~1.8), but the book is NOT merely its one robust leg — under the trustworthy conservative
§5f metrics the decorrelated fbr (corr −0.44…−0.65 with me_long) still lifts risk-adjusted return over
me_long-solo (Sharpe ~+0.48 vs ~+0.29). The lift reverses under the optimistic mean_roi metric and the
book remains non-significant (1043). **Completes the honest-exit correction of all three magnitude/
significance/diversification deploy pillars** (mean 1042, significance 1043, ENB 1044): each is
materially weaker honest than committed, but none fully collapses — the honest deploy object is a thin,
~1.8-bet, non-significant, +0.27%/yr book carried by me_long + decorrelated fbr.

**FLAG (rolls up 1042-F1 / 1043-F1,F2 into one operator-facing statement):** the path-A deploy case as
characterized in the committed-exit arcs (2016/2017/2019/1023/2021/3022) is **systematically
exit-optimistic** — deploy mean ~2× too high (+0.59%→+0.27%), significance lost (t 2.16→~1.3, CI spans
zero), effective bets ~2× too high (3.32→~1.8). The remaining un-rechecked committed-exit pillars are
**temporal stability (2021)** and **cost cushion (3022, κ=3.32)** — both owed an honest-exit re-run,
though the cost cushion almost-certainly halves with the mean (the honest book is still net-positive at
κ=1, so no qualitative flip expected there). The honest deploy decision is the operator's path-A call
on a WEAKER object than the committed characterization advertised.

## (i) New lesson
Honest §5f exits roughly HALVE the book's effective-bet count (ENB 3.32→~1.8) — the diversification
pillar (arc 2019) is exit-optimistic too — BUT the book is NOT reducible to its one robust leg: under
the conservative §5f metrics a decorrelated-but-individually-negative-mean leg (fbr, corr −0.44…−0.65
with me_long) can still lift risk-adjusted return over the solo leg, because the variance reduction
outweighs the mean drag. So a thin multi-leg book whose magnitude/significance collapse under honest
exits can still retain REAL (if modest, exit-metric-dependent) diversification value — "2 of 4 legs go
negative" weakens but does not necessarily collapse the portfolio. Characterize the honest book on
risk-adjusted (Sharpe) terms, not just mean, before declaring the multi-leg thesis dead.

## (k) Re-orient
Detail persisted (this doc + log). Tools REUSED (`nested_exit_selection`, `combine_fold_roi`); no new
BUILT tool. Driver `_disco_work/arc1044_honest_diversification.py`. No canonical change, no council,
OOS untouched. Next: resume at arc 1045 (or graceful handoff if context low).
