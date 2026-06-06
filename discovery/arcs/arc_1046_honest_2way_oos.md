# arc 1046 — Honest frozen-exit OOS: does the 2-way {me_long+fbr} beat the 4-way out-of-sample?

> chat: 1000s | range 1000-1999 | timestamp: 2026-06-06
> disposition: **DIAGNOSTIC → KILL** (no new component; nothing reaches AFP; me_long UNCHANGED PORTFOLIO)
> OOS measured as CHARACTERIZATION (frozen on IS, scored ONCE, never tuned — §4 / arc-1042 precedent); the
> combined-book AFP holdout GATE remains the operator's §5g firewall.

## (a) Log read / synthesis
Direct continuation of my arc 1045 (this chat, same session). 1045 (IS) found the clean 2-way
{me_long + fbr} is a materially better honest deploy object than the 4-way — ~2× Sharpe, and it RECOVERS
the statistical significance the honest 4-way LOST (afp metric t=+2.12, CI excludes zero) — because gap +
me_short flip mean-negative under honest §5f exits (their committed exits were full-sample-best picks) and
ALL the real diversification is the me_long↔fbr anti-correlation (−0.54…−0.68). But 1045 was IS-only and
explicitly flagged the OOS durability as the owed question. The corpus state otherwise unchanged: 4
PORTFOLIO components, 4-way book blocked by 2015/2018, ~18 routes to a +2015/+2018 leg dead, §11
verification complete, deploy = operator path-A call.

## (b) Idea (the because)
1045's headline FLAG ("the honest deploy object is the 2-way, drop gap+me_short") rests entirely on an IS
characterization. The single thing that would confirm or break it is the holdout: under the §5f FROZEN
discipline — choose each leg's exit on ALL IS (the nested `frozen_label`), fit RP weights on IS, FREEZE
both, score the 2021+ holdout ONCE (never re-selected, §4) — does the 2-way's IS superiority persist OOS?
Two sub-questions: (1) does 2-way > 4-way hold OOS (is "drop gap+me_short" durable)? (2) does the 2-way's
fbr-driven advantage over me_long-solo hold OOS, or was fbr's IS decorrelation an in-sample artifact?

## (c)/(f) Method
100% canonical scoring (`ArcFoldRunner`→`A1Architecture`→`MultiPairBacktester`, FundedNext) on BOTH the
IS folds (2011-2020) AND `build_oos_year_folds(2021)` (2021-2026, 6 holdout years). Per leg, per metric:
`nested_exit_selection.freeze_best_over_folds` picks the exit best over ALL IS (the §5f holdout freeze);
that frozen exit's OOS series is read off the OOS grid. RP weights fit on the IS honest evaluable series,
FROZEN. Combine the OOS frozen-exit series at frozen weights → 2-way and 4-way OOS books, vs me_long-solo
OOS. 3 metrics (mean_roi / afp / worst). Nothing is selected on OOS. Driver
`_disco_work/arc1046_honest_2way_oos.py`. No new tool, no canonical change, OOS measured-once only.

## Results

**OOS 2021-2026 (frozen exits + frozen RP weights, scored once):**

| object | metric | mean | Sharpe(fold) | t | worst | neg | AFP |
|---|---|---|---|---|---|---|---|
| **me_long SOLO** | afp | **+0.449%** | **+0.534** | +1.31 | −0.628% | **1/6** | False |
| 4-way BOOK (RP) | afp | +0.210% | +0.195 | +0.48 | −1.249% | 3/6 | False |
| 2-way me_long+fbr (RP) | afp | +0.402% | +0.457 | +1.12 | −0.605% | 3/6 | False |
| **me_long SOLO** | worst | +0.334% | +0.514 | +1.26 | −0.503% | 1/6 | False |
| 4-way BOOK (RP) | worst | +0.175% | +0.217 | +0.53 | −0.808% | 2/6 | False |
| 2-way me_long+fbr (RP) | worst | +0.321% | +0.457 | +1.12 | −0.510% | 3/6 | False |
| 4-way BOOK (RP) | mean_roi | **−0.039%** | **−0.037** | −0.09 | −1.635% | 3/6 | False |
| 2-way me_long+fbr (RP) | mean_roi | +0.262% | +0.265 | +0.65 | −1.177% | 2/6 | False |
| me_long SOLO | mean_roi | +0.449% | +0.534 | +1.31 | −0.628% | 1/6 | False |

**Finding 1 — ROBUST: the 2-way beats the 4-way OOS too** (every metric, mean & Sharpe): 2-way Sharpe
+0.27/+0.46/+0.46 vs 4-way −0.04/+0.20/+0.22; under mean_roi the **4-way goes mean-NEGATIVE OOS**
(−0.039%) while the 2-way stays +0.262%. So arc-1045's "drop gap + me_short" is **durable** — those two
exit-artifact legs are net drags IS AND OOS.

**Finding 2 — SURPRISE, the IS ranking INVERTS: me_long SOLO beats the 2-way OOS** (every metric, higher
mean, higher Sharpe, fewer neg folds: solo 1/6 vs 2-way 3/6). **fbr — the IS diversifier — is mean-NEGATIVE
OOS** (per-year, afp-metric partial-runner exit: 2021 −0.54, 2022 −0.00, 2023 −1.18, 2024 +5.33, 2025
−1.57, 2026 −0.44; under the mean_roi trailing_swing exit it is far worse, 2021 −9.5). Its IS −0.57
decorrelation with me_long lifted the 2-way IS Sharpe, but OOS that benefit reverses: adding fbr to
me_long INTRODUCES negative folds (2023, 2026) me_long alone did not have. fbr's OOS-negativity is robust
to the exit choice (negative under both partial-runner and trailing_swing frozen picks), not a
trailing_swing artifact — consistent with arc 1013's thin OOS (+0.94%, 3/6) and arc 2040's "fbr fished
exit dies OOS."

**Net:** under honest frozen exits scored on the holdout, the entire 4-component portfolio's deploy value
**collapses to its single robust leg, me_long** — the only component positive both under its IS-robust
exit (arcs 1042/2042) AND OOS (here): +0.33–0.45%/yr, Sharpe ~0.51–0.53, **5/6 OOS years positive** (only
2021 negative). me_long is NOT all-folds-positive (2021 OOS neg + 3/8 IS neg, arc 1045) → it stays a
PORTFOLIO component, NOT a survivor/PASS.

**Caveats (§8, Arc-10 defense):** n=6 OOS years is small; me_long-solo OOS t≈1.3, P(mean<0)≈0.06–0.08 —
borderline, NOT significant at 0.05. This is one honest frozen read (no OOS tuning); it does not "use up"
the holdout (§4 — measuring ≠ optimizing). me_long-solo is not AFP, so no §5g gate pass is claimed.

## Verdict
**DIAGNOSTIC → KILL** (no new component; nothing reaches AFP). me_long UNCHANGED (PORTFOLIO); the other 3
components UNCHANGED. The 1045 IS finding is RESOLVED on the holdout with a sharper, partly-inverted
answer.

**FLAG (operator path-A, docs only — supersedes/sharpens arc-1045's FLAG):**
- **Durable:** the 2-way {me_long+fbr} beats the 4-way OOS on every metric → **drop gap + me_short** for
  the honest-exit deploy (net drags IS AND OOS; the 4-way even goes mean-negative OOS under mean_roi).
- **But the honest, OOS-validated deploy CORE is me_long SOLO, not the 2-way** — fbr's IS-diversification
  benefit does NOT survive the holdout (fbr is mean-negative OOS under its own frozen exit), so adding fbr
  to me_long HURTS OOS (introduces negative folds; lowers mean and Sharpe). me_long-solo is the cleanest
  object in the book (OOS +0.33–0.45%/yr, Sharpe ~0.5, 5/6 years +), though borderline-significant
  (t≈1.3) and not AFP.
- **Implication:** the multi-component PORTFOLIO thesis (the structural justification for building 4 legs,
  arcs 1006→2019) does NOT survive honest frozen exits + OOS — under those conditions the "portfolio" is
  effectively me_long with three non-additive legs (gap + me_short exit-artifacts, fbr OOS-negative). The
  path-A deploy object is a single thin PORTFOLIO leg, not a 4-bet (1044) or 2-bet (1045) book.

## (i) New lesson
An IS diversification benefit can REVERSE out-of-sample when the diversifying leg is mean-negative on the
holdout: fbr's IS −0.57 anti-correlation with me_long lifted the 2-way IS Sharpe above solo (1045), but
OOS fbr is mean-negative under its frozen exit, so the 2-way Sharpe falls BELOW me_long-solo (the
decorrelation no longer pays once the mean is negative on fresh data). Judge a diversifier on whether its
OOS MEAN survives, not just its IS correlation — a −corr leg only helps if it is also non-negative-mean
out-of-sample. The robust half (drop the exit-artifact legs gap/me_short) holds both ways; the fragile
half (fbr adds value) was IS-only. Re-derive a thin multi-leg book's deploy ranking on the FROZEN holdout
before trusting an IS Sharpe ranking — here it collapses a 4-leg "portfolio" to one robust leg (me_long).

## (k) Re-orient
Detail persisted (this doc + log). Tools REUSED (`nested_exit_selection` incl. `freeze_best_over_folds`,
`combine_fold_roi`); no new BUILT tool. Driver `_disco_work/arc1046_honest_2way_oos.py`. No canonical
change, no council (not a survivor; portfolio-characterization), OOS measured-once as characterization
(not tuned, not a gate). Next: graceful handoff (context budget) — a fresh 1000s chat resumes at arc 1047.
