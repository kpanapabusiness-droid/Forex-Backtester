# arc 2041 — §5f NESTED exit/SL selection on `me_short`: does arc-2040's optimism generalize?

**Chat:** 2000s · **Range:** 2000–2999 · **Disposition:** **KILL** (methodology/verification; no new
component — `me_short` UNCHANGED, but its PORTFOLIO status FLAGGED for the operator) · **Council:** none

---

## 1. SEED (step a/b) — arc 2040 generalization

Arc 2040 found `fbr`'s committed +1.854% is ≈40% full-sample exit-selection optimism (honest §5f
nested-WFO = ~+1.0–1.2% IS / +0.27% OOS, still positive). **Does the same exit-optimism afflict the other
book legs?** `me_short` (1019) is the cleanest next test: its committed exit IS a registry exit
(`sl_partial_close_1r_runner_trail`, NO time-predicate in the book's `validate_4way_book` config), so the
6-exit registry grid maps cleanly — same apples-to-apples as fbr. `me_short` is also the highest-stakes
short leg: arc 2015 named it the FIRST robustly +2018 component and the one that "moves the convex-search
wall," and the BUILT-tool note already FLAGGED it **EXIT-SENSITIVE** (arc 3017: "a long 120-bar hold
washes it toward noise; the time exit / partial-runner is load-bearing for this reversion edge"). This arc
QUANTIFIES that flag via honest §5f nested-WFO selection.

## 2. METHOD

Identical to arc 2040 (reusing the BUILT `nested_exit_selection.py`): `MonthEndReversionShortSignal(1.0,
2)`, 7 USD majors, D1; 6 registry exits × SL{1.5,2.0,2.5} = 18 cfgs (trail_off, exit_policy = sole
mechanism), scored over IS folds 2011–2020 by canonical `ArcFoldRunner`→`MultiPairBacktester`→FundedNext;
per-fold walk-forward selection on STRICTLY-EARLIER folds (3 metrics); frozen-all-IS pick scored on 2021+
once. SHORT signal (engine short-symmetric, PR#273). Driver `_disco_work/arc2041_meshort_nested_exit.py`.

**Fidelity anchor — PERFECT.** Committed partial-runner/SL2.0 reproduces arc-1020 recorded me_short
per-year **exactly** (max|diff| **0.005pp**; +0.683% mean, 7/10). Harness confirmed.

## 3. RESULTS — the optimism is WORSE than fbr (positivity does not survive honest selection)

**(A) me_short is EXIT-SENSITIVE (the grid).** Full-IS mean ranges from +0.78% (`sl_plus_tp_3r`/SL2.5) to
**−1.63%** (`sl_only`/SL1.5); the committed partial-runner/SL2.0 (+0.68%, 7/10) is near the top, but
**sl_only and trailing_swing at low SL are mean-NEGATIVE**. Positivity depends on landing in the
partial-runner / tp_3r family — confirming arc 3017's exit-sensitivity flag concretely.

**(B) Honest §5f nested-WFO selection → MEAN-NEGATIVE (the headline):**

| selection metric | nested IS mean | worst | n neg | AFP | frozen pick | frozen OOS mean | OOS pos |
|---|---|---|---|---|---|---|---|
| `mean_roi` | **−0.192%** | −1.32% | 5 | No | tp_3r/SL2.5 | **−0.557%** | 1/6 |
| `afp_then_mean` | **−0.405%** | −2.61% | 5 | No | partial-runner/SL2.0 | **−0.489%** | 2/6 |
| `worst_then_mean` | **−0.034%** | −0.90% | 5 | No | partial-runner/SL2.5 | **−0.247%** | 2/6 |

Under **EVERY** honest selection metric, me_short is **mean-negative-to-≈zero IS (−0.03% to −0.41%) AND
mean-negative OOS (−0.25% to −0.56%)** — versus the committed +0.683% IS. The exit choice is NOT stable
across folds: selecting on prior folds does not reliably recover the partial-runner, and the honest
walk-forward series is negative. Where fbr's honest number stayed positive (lower), **me_short's honest
number goes negative** — the optimism is qualitatively worse.

**(C) The §5f trap, again.** Full-grid best-MEAN = `sl_plus_tp_3r`/SL2.5 (+0.78%) collapses OOS to
−0.557% (1/6) when frozen — the full-sample best-pick failing forward, as in arc 2040.

## 4. VERDICT — KILL (methodology; `me_short` UNCHANGED, PORTFOLIO status FLAGGED)

**Findings.**
1. **me_short's committed +0.683% does NOT survive honest §5f nested-WFO exit selection** — mean-negative
   IS (−0.03% to −0.41%) and OOS (−0.25% to −0.56%) under all three metrics. This QUANTIFIES arc 3017's
   qualitative exit-sensitivity flag into a §5f-honest verdict: the mean-positivity is exit-selection-
   dependent and not recoverable without lookahead.
2. **Generalizes + SHARPENS arc 2040.** Both engine-positive book legs whose headline rests on a chosen
   exit are exit-optimistic; **me_short critically so** (honest = negative, vs fbr honest = lower-but-
   positive). The book's mean-positive case (risk-parity +0.589%, arc 2019) leans on BOTH — under honest
   §5f exit accounting it is materially weaker than the committed component headlines imply.
3. **§11 implication (FLAG, operator's call):** §11 requires a PORTFOLIO component be "mean-positive net
   of costs"; "you cannot diversify net-negative components positive." If me_short's mean-positivity is an
   exit-selection artifact (honest §5f → mean-negative IS+OOS), it sits on the **KILL** side of the
   PORTFOLIO/KILL line, not PORTFOLIO. This would also remove the book's only robustly-+2018 leg (arc
   2015) — weakening the whole 4-way book's 2018 story.

**FAIR caveat (conservative bias, §8 — why this is a FLAG, not a unilateral downgrade).** The committed
partial-runner IS a defensible FIXED, mechanism-motivated exit (a fix-flow reversion needs the +1R partial
to bank the bounce and the runner for the follow-through — arc 1019/3017), and as a fixed choice it is
+0.683%/7-of-10 and reproduces exactly. The nested-WFO negativity is partly small-n (8 evaluable folds)
selection variance — adaptive per-fold exit switching is noisy at n=8. So the honest reading is: me_short
is **exit-fragile to the point that no-lookahead exit selection yields negative**, which is a strong
caution about its mean-positivity, but granting the fixed mechanism-motivated partial-runner it remains
+0.683%. **Which standard governs (fixed mechanism-exit vs §5f-honest selection) is the operator's
gate-governance call** — the same path-A decision the corpus keeps converging on, now with a sharper input.

**FLAG F1 (docs only):** me_short's recorded +0.683% (the 4th PORTFOLIO component, arc 2015's robust-+2018
leg) is the MOST exit-fragile headline in the book — mean-negative under honest §5f nested-WFO exit
selection (IS and OOS). Recommend the operator weigh whether me_short qualifies as PORTFOLIO under §5f-
honest exit accounting before leaning on it for the book's 2018 story / path-A deploy case.

**Components UNCHANGED** (all 4 PORTFOLIO; I do not unilaterally rewrite a committed component's
disposition). Lever unchanged = operator path-A gate-governance call. No new BUILT tool (reused arc-2040's
`nested_exit_selection`). No council. No OOS tuned (frozen scored once). Driver scratch:
`_disco_work/arc2041_meshort_nested_exit.py`.

**NEW lesson (extends arc 2040):** the full-sample-best-exit optimism gap is component-dependent and can
flip the SIGN — fbr (strong, deep-edge) loses ~40% of its headline but stays positive; me_short (thin,
exit-sensitive reversion) goes NEGATIVE. A thin engine-positive component's mean-positivity should be
re-checked under honest §5f nested-WFO exit selection before it is trusted as a PORTFOLIO input — a fixed
best-version exit can manufacture the entire edge.
