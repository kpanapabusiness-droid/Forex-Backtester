# arc 2040 — §5f NESTED walk-forward exit/SL selection on `fbr` (the load-bearing leg)

**Chat:** 2000s · **Range:** 2000–2999 · **Disposition:** **KILL** (methodology/verification arc; no
new component — `fbr` UNCHANGED as PORTFOLIO, with an honest-headline FLAG for the operator) · **Council:** none

---

## 1. READ + SYNTHESIZE THE LOG (step a) — FRESH EYES, honest-era only

The corpus is mature. State at entry (highest 2000s arc = 2039):

- **The deployable artifact = a 4-component PORTFOLIO book** — gap (1006, JPY crosses H4), me_long
  (1011, USD majors D1), **fbr (1013, USD majors H4 — the load-bearing leg, arc 2033)**, me_short
  (1019, USD majors D1). All key off a *surprise large displacement → fade/reclaim*.
- **Book status:** mean-positive (risk-parity +0.589%, P(mean<0)=0.004; ENB 3.32, arc 2019), temporally
  robust (2021), cost-robust (break-even κ=3.32, 3022). **FAILS strict all-folds-positive.** Arc 2017
  isolated the book's ONLY statistically-real negative fold: **fbr-2018 (−4.20%, n18, CI<0)**; every
  other worst-fold is within the noise floor (2016/2017).
- **§11 verification COMPLETE** (2034–2039): signal + gross-outcome + cost layers independently honest.
- **Routes closed:** leg-hunt structurally closed (2019/2022); relative-value dead (2010/2018); all
  short structural mirrors dead (1014/2009/2011/3011); explore-now menu exhausted (2023/2027/2028);
  calendar forced-flows sub-cost-or-priced-in (2024/2025/2026); **fbr ENTRY-quality axis closed across
  three levers** (2014 regime-gate, 2017 per-fold CI, 2020 M1-microstructure, 2029 PDL/PWL, 2030
  touch-count, 2031 crosses). The repeated frontier conclusion: the lever is the **operator path-A
  gate-governance call**, not autonomous edge-hunting.

**The gap I found (step a → b).** Every prior fbr arc attacked the **ENTRY** side. The **EXIT** side was
set ONCE, in arc 1013, by picking `sl_plus_trailing_atr`/SL2.0 as the **single full-IS-best-MEAN exit**
from a 5-exit menu (arc 1013 §4 table). That is exactly the full-sample best-pick **§5f explicitly
forbids**: *"NEVER pick the single best exit across the full sample and report its number — that is
exit-fishing, an Arc-10-class gate inflation."* §5f mandates the exit/SL be a **nested WFO
hyperparameter**: selected on the IS portion of each walk-forward fold, scored on that fold's OOS, the
all-IS choice frozen onto the holdout. fbr — the corpus's load-bearing component (arc 2033) and the one
whose 2018 hole is the book's only real obstacle — never received this. Arc 1012 ran a "§5f nested
sweep" on me_long but as a *fixed-config all-folds-positive scan* (is any ONE config AFP), not the
walk-forward per-fold selection. So the proper nested-WFO exit selection on fbr is genuinely untested,
§5f-mandated, and the highest-stakes Arc-10-defense target left on the exit axis.

## 2. THE IDEA + WHY (step b)

**because:** the number the whole book leans on (fbr +1.854% IS, arc 2033 load-bearing) was produced
under an exit chosen with full-sample hindsight. Two honest questions follow: **(Q1)** is that headline
inflated by the full-sample exit pick — what is fbr's HONEST, no-lookahead §5f nested-WFO exit number?
**(Q2)** does any honestly-selected exit make fbr's 2018 hole shallow enough to matter — i.e. is the
2018 obstacle EXIT-conditionable (the one fbr axis never tested), or is it exit-invariant like it is
entry-invariant (2014/2017/2020)?

This is a verification/methodology arc (in the spirit of 2034–2039), not an edge hunt — but it directly
tests the load-bearing number and could, in principle, upgrade fbr toward solo-PASS (if a nested-honest
exit were all-folds-positive) or correct the book's headline downward.

## 3. METHOD (canonical scoring; §5f selection)

- **Signal:** `FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25)`, 7 USD majors,
  H4 (the committed fbr), via the BUILT tool. Trusted loader `Panel.from_pairs` (histdata_backup + cache).
- **Grid:** the 6 registry exits {`sl_only`, `sl_plus_tp_2r`, `sl_plus_tp_3r`, `sl_plus_trailing_atr`,
  `sl_plus_trailing_swing`, `sl_partial_close_1r_runner_trail`} × SL∈{1.5, 2.0, 2.5} = **18 configs**,
  all `trail_enabled=False` (the exit_policy is the SOLE exit mechanism — clean, no double-trail
  confound). Scored over the IS folds (2011–2020) by the canonical `ArcFoldRunner` →
  `MultiPairBacktester` → FundedNext netting (`run_config_over_folds`). 100% canonical scoring.
- **Selection (BUILT `nested_exit_selection.py`, pure arithmetic, scores nothing):** for each fold-year
  Y, choose the config best over STRICTLY EARLIER folds (pure in-sample to Y), report Y scored with that
  config → the honest walk-forward OOS series. Three selection metrics for robustness: `mean_roi`,
  `afp_then_mean` (rewards all-folds-positive-ness first — aligned with the discovery gate),
  `worst_then_mean`. Folds with <2 priors = warmup (excluded from the verdict). The all-IS frozen pick
  is then scored on the 2021+ holdout ONCE (NOT re-selected per year, §4).
- Driver: `_disco_work/arc2040_fbr_nested_exit.py`.

**Fidelity anchor.** `sl_plus_trailing_atr`/SL2.0/trail-off reproduces arc-1020's recorded fbr per-year
column closely (max|diff| 1.49pp on 2016; per-fold signs all match incl. **2018 −4.39 vs recorded
−4.20**; my mean +2.084% vs recorded +1.854%). Harness CONFIRMED reproducing the committed/book fbr.

## 4. RESULTS

**(A) The full-sample best-pick view (§5f-forbidden) — and why §5f forbids it.** Over the full 18-config
grid, the full-sample best-**MEAN** config is **`sl_plus_trailing_swing`/SL1.5 at +4.26%** — but it is
only **5/10 folds-positive, worst −7.63%**. The highest full-sample mean is a high-variance
fold-disaster. (The committed `sl_plus_trailing_atr`/SL2.0 is +2.08%, 8/10, −4.39 — robust but not the
mean-max once the menu is widened.) A naive "pick the best mean" on the wider menu would have chosen a
materially WORSE deployable exit. This is the §5f trap made concrete.

**(B) The HONEST §5f nested walk-forward series (evaluable folds 2013–2020):**

| selection metric | nested IS mean | worst fold | n neg | all-folds-positive | frozen-all-IS pick |
|---|---|---|---|---|---|
| `mean_roi` | +3.357% | −7.63% | 4 | **No** | `sl_plus_trailing_swing\|sl1.5` |
| `afp_then_mean` | +1.001% | −3.37% | 3 | **No** | `sl_partial_close_1r_runner_trail\|sl1.5` |
| `worst_then_mean` | +1.204% | −4.39% | 2 | **No** | `sl_partial_close_1r_runner_trail\|sl1.5` |

**Under EVERY honest selection metric, fbr is NOT all-folds-positive** → robustly confirms PORTFOLIO,
never a solo PASS. The `mean_roi` metric chases the high-variance trailing_swing and pays for it
(4 neg); the fold-aware metrics (the correct objective for an AFP gate) settle on the conservative
**partial-close-1R-runner-trail** and give fbr a HONEST headline of **~+1.0–1.2% IS** — versus the
committed full-sample **+1.85%**. **≈40% of fbr's committed headline is full-sample exit-selection
optimism.**

**(C) 2018 is EXIT-INVARIANT negative.** 2018 is negative in all three nested series (−6.23 / −3.37 /
−4.39) and the best-fold-count config in the grid (partial-runner/SL1.5, 9/10) carries its lone neg fold
at 2018 (−2.53). The shallowest honest 2018 any exit buys is ≈−2.5 to −3.4% (the partial-runner family,
vs trailing's −4.2/−4.4). So the exit axis JOINS the entry axis as closed for fbr-2018: the 2018 hole is
**mechanism-intrinsic and unconditionable on BOTH entry (2014/2017/2020) and exit (here)** — a faster
exit only trades book-mean for a shallower-but-still-negative 2018, never erases it.

**(D) Frozen holdout (2021+), scored once:**

| frozen pick | OOS mean | worst | pos/6 |
|---|---|---|---|
| `mean_roi` → trailing_swing/SL1.5 | **−2.587%** | −9.51% | 1/6 |
| `afp`/`worst` → partial-runner/SL1.5 | +0.269% | −1.57% | 1/6 |

The mean-fished exit (trailing_swing) **collapses OOS** (−2.59%, 1/6) — direct evidence of full-sample
optimism failing forward. The fold-aware frozen exit (partial-runner) is OOS mean-positive but thin
(+0.269%, 1/6). For reference, arc 1013's committed `sl_plus_trailing_atr` reported OOS +0.936% (3/6).
So **no honest nested choice beats the committed config's reported OOS** — the committed number is what
you get WITH exit hindsight; every honest walk-forward choice lands lower. Classic full-sample optimism
signature.

## 5. VERDICT — KILL (methodology/verification; `fbr` UNCHANGED as PORTFOLIO, headline FLAGGED)

No new component; this is an Arc-10-defense methodology correction on the load-bearing leg. Findings:

1. **fbr is not all-folds-positive under ANY honest §5f nested-WFO exit selection** → PORTFOLIO confirmed
   on the exit axis (the last untested axis), never PASS.
2. **The committed +1.854% headline is mildly full-sample-optimistic.** Honest fold-aware nested-WFO
   exit selection gives fbr **~+1.0–1.2% IS / +0.27% OOS** — ≈40% lower. Since the whole book's
   mean-positive case (risk-parity +0.589%, arc 2019) leans on fbr as the strongest leg (arc 2033), the
   book's deploy-relevant mean is correspondingly **modestly optimistic on the exit axis** — material to
   the operator's path-A gate-governance call (which would lean on that mean).
3. **fbr-2018 is exit-invariant negative** — the exit axis joins entry-regime (2014) / per-fold-CI
   (2017) / M1 (2020) as a closed axis; the 2018 hole cannot be conditioned away from either side.
4. **NEW lesson:** widening the exit menu does NOT rescue a non-AFP entry; it surfaces a higher-mean
   high-variance exit (trailing_swing) that a full-sample pick would grab and that collapses OOS. The
   honest §5f nested-WFO selection is precisely the discipline that avoids this — and it confirms the
   conservative partial-runner (not the committed best-mean trailing) is the fold-honest exit, at a
   lower-but-real number. Reporting a directional entry's headline under its *full-sample-best* exit
   over-states it by the exit-selection optimism gap (~40% here); the deployable number is the nested one.

**FLAGS (documentation only — no canonical change, code human-gated §9):**
- **F1 — fbr committed-headline optimism:** the recorded fbr +1.854% rests on a §5f-forbidden
  full-sample exit pick. Honest §5f nested-WFO = ~+1.0–1.2% IS / +0.27% OOS. Recommend the operator note
  this when weighing the book's mean for a path-A deploy decision. (Components left UNCHANGED — I do not
  unilaterally rewrite a committed component's number; this is the operator's call.)
- **F2 — fbr config discrepancy in the corpus:** the deployable BOOK validation
  (`scripts/cosim_validation/validate_4way_book.py`, arc 1020) scores fbr with **`trail_enabled=False`**
  (single `sl_plus_trailing_atr`), and my anchor reproduces the book's recorded fbr under trail-off. But
  the §11 OUTCOME audit (arc 2036, `independent_outcome_audit_fbr.py`) asserts and verifies
  **`trail_enabled=True`** (the A1-default *double-trail*, flagged 1015/3009). **The §11 outcome
  verification audited a NON-book fbr variant** — the deployable book's trail-off fbr has NOT had its
  per-trade outcome independently re-derived. Recommend a follow-up §11 outcome audit on the trail-off
  config before deployment (small gap; the book's fbr is trail-off per arc 1020).

**Components UNCHANGED** (all 4 PORTFOLIO). Lever unchanged = operator path-A gate-governance call. No
OOS tuned (frozen picks scored once; selection used IS only). No council (verification arc).

BUILT + registered: `discovery/tools/nested_exit_selection.py` (the §5f nested-WFO exit/SL selector,
reusable for every future non-coin-flip entry). Driver scratch: `_disco_work/arc2040_fbr_nested_exit.py`.
