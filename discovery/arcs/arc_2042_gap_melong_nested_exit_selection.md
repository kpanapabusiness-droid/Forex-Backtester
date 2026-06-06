# arc 2042 — §5f NESTED exit/SL selection on `gap` (1006) + `me_long` (1011): completing the book's exit-honesty audit

**Chat:** 2000s · **Range:** 2000–2999 · **Disposition:** **KILL** (methodology/verification; no
new component — both legs UNCHANGED, `gap`'s PORTFOLIO status FLAGGED alongside `me_short`) ·
**Council:** none

---

## 1. SEED (step a/b) — finish the 2040/2041 thread

Arc 2040 found `fbr`'s committed +1.854% is ≈40% full-sample exit-selection optimism (honest §5f
nested-WFO ≈ +1.0–1.2% IS / +0.27% OOS, still positive). Arc 2041 found `me_short`'s +0.683% is WORSE
— honest §5f **flips the sign to mean-negative** (IS −0.03…−0.41%, OOS −0.25…−0.56%). Both arcs only
covered the legs whose committed exit is a PURE registry exit (clean 6-exit grid). **The two remaining
book legs — `gap` (1006) and `me_long` (1011) — were skipped because their committed exits carry a
TIME-EXIT predicate** (a fill/reversion plays out over a fixed horizon, not at +1R; `make_time_exit_
predicate`, arc 1005). Arc 2041's own lesson is explicit: *re-check any thin PORTFOLIO input under
honest §5f nested-WFO exit selection before trusting it.* This arc does that for the last two legs,
completing the whole 4-component book's exit-honesty audit — the direct input to the book's
mean-positive deploy case (arc 1023 t=2.66 / arc 2019 risk-parity +0.589%), which was computed on the
COMMITTED (potentially exit-optimistic) component numbers.

**The methodological wrinkle (why these legs are different).** For a time-exit reversion leg the
load-bearing exit hyperparameter that was full-sample-chosen is the **time horizon** (and SL), not the
registry exit_policy. So the honest §5f exit menu is `{exit_policy} × {SL} × {time-horizon}`. Two
analyses per leg: **(A)** 6 registry exit_policy × SL{1.5,2.0,2.5} at the committed horizon (apples-to-
apples with 2040/2041); **(B)** committed exit_policy × SL × a time-horizon grid (the binding knob).

## 2. METHOD

Reused the BUILT `nested_exit_selection.py` (arc 2040); scoring 100% canonical (`ArcFoldRunner` →
`A1Architecture` → `MultiPairBacktester` → FundedNext) over IS folds 2011–2020; per-fold
walk-forward selection on STRICTLY-EARLIER folds (3 metrics: `mean_roi` / `afp_then_mean` /
`worst_then_mean`); frozen-all-IS pick scored on 2021+ ONCE (§4, no per-year re-selection). Time-exit
predicate lives on the signal eval, so one runner per horizon. Committed configs (from
`scripts/cosim_validation/validate_4way_book.py`):
- `gap`: `WeekendGapFillLongSignal(0.5, 36)` H4 JPY crosses + time-exit 24 + `exit_policy=None`, SL2.0
- `me_long`: `MonthEndReversionLongSignal(1.0, 2)` D1 USD majors + time-exit 2 + `sl_only`, SL2.0

Driver `_disco_work/arc2042_gap_melong_nested_exit.py`.

**Fidelity anchors — BOTH PERFECT.** Committed configs reproduce arc-1020 recorded per-year IS ROI
**exactly**: `gap` max|diff| **0.005pp** (+0.685% mean, 5/10); `me_long` max|diff| **0.004pp**
(+0.232% mean, 7/10).

## 3. RESULTS — the two legs split: `gap` FLIPS NEGATIVE, `me_long` is ROBUST

### `gap` (1006) — committed +0.685% does NOT survive honest §5f (flips negative)

**(A) exit_policy × SL @ h24.** Full-sample best-MEAN = `sl_plus_trailing_atr`/SL2.0 **+0.90%** (5/10)
— so the committed exit was not even the full-sample best. Honest §5f nested: **mean-NEGATIVE every
metric (IS −0.92% / −1.02% / −1.41%, all 5-neg, none AFP); frozen OOS −0.45% / −0.45% / −0.33%
(2/6).**

**(B) horizon × SL (exit=None).** Full-sample best = **`h24`/SL2.0 (+0.69%)** — i.e. **the COMMITTED
horizon 24 IS the full-sample best-pick** (h12 −0.74%, h18 +0.10%, h36 −0.11%) = textbook §5f exit-
fishing. Honest §5f nested: **IS −0.97% / −0.97% / −0.59% (5-neg, none AFP); frozen OOS +0.00% / +0.00%
/ +0.81% (2–3/6).**

⇒ **`gap`'s +0.685% is full-sample horizon/exit-selection optimism** — like `me_short`, it goes
mean-NEGATIVE under honest no-lookahead exit selection (IS), worst of all four legs. Consistent with
arc 1009 (gap is threshold-fragile; its edge over a fair null is ~half the headline) and with gap's
extreme fold volatility (−6.79%…+8.23%): the +0.69% full-sample mean rests on a couple of huge folds
(2012 +8.23, 2019 +7.45) that the exit choice cannot stabilize forward.

### `me_long` (1011) — committed +0.232% IS ROBUST to honest §5f (the only exit-robust leg)

**(A) exit_policy × SL @ h2.** All 6 registry exits ~IDENTICAL (+0.23–0.24%, 7/10) → at a 2-bar D1
hold the registry exit_policy is **irrelevant** (nothing triggers but SL/time); the horizon is the
real knob. Honest §5f nested **STAYS POSITIVE: IS +0.203% / +0.203% / +0.147%; frozen OOS +0.449% /
+0.449% / +0.334% (5/6 pos)**.

**(B) horizon × SL (sl_only).** Full-sample best = `h3`/SL2.0 **+0.50%** > committed `h2` (+0.23%) →
**the committed horizon is CONSERVATIVE, not fished.** Honest §5f nested **STAYS POSITIVE: IS +0.238% /
+0.238% / +0.283%; frozen OOS +0.42% / +0.65% / +0.58% (3–5/6 pos)**.

⇒ **`me_long`'s honest §5f number (~+0.20–0.28% IS / +0.33–0.65% OOS) ≈ or BETTER than its committed
+0.232%.** It is the LEAST exit-optimistic component — its thin (~12/yr) edge lives in the month-end
fix-flow ENTRY timing, not an exit artifact (vindicates arcs 1011/1012, which found the baseline won).

### The book-level payoff — all 4 legs now have honest §5f exit numbers

| leg | committed IS | honest §5f IS | honest §5f OOS | exit-honesty verdict |
|---|---|---|---|---|
| `fbr` (2040) | +1.854% | ~+1.0–1.2% (−40%) | +0.27% | stays + (lower) |
| `me_short` (2041) | +0.683% | −0.03…−0.41% | −0.25…−0.56% | **FLIPS −** |
| **`gap` (2042)** | +0.685% | **−0.59…−1.41%** | −0.45…+0.81% | **FLIPS −** |
| **`me_long` (2042)** | +0.232% | +0.15…+0.28% | +0.33…+0.65% | **ROBUST +** |

**THREE of four legs are exit-optimistic; TWO (`gap`, `me_short`) flip mean-negative under honest §5f;
only `me_long` (the thinnest) survives robustly.** The book's mean-positive deploy case (risk-parity
+0.589%, t=2.66 — arc 1023/2019) was computed on the COMMITTED component numbers; under honest §5f
exit accounting the book's mean is **materially weaker than +0.589%** — substantially an artifact of
full-sample exit/horizon selection in 3 of 4 legs.

## 4. VERDICT — KILL (methodology; both legs UNCHANGED, `gap` PORTFOLIO status FLAGGED)

**Findings.**
1. **Completes the book's exit-honesty audit.** `gap` joins `me_short` on the exit-fragile side (honest
   §5f → mean-negative IS); `me_long` is the lone exit-robust leg; `fbr` is in between (lower but +).
   The full-sample-best-exit optimism gap is component-dependent and can flip the sign — now confirmed
   on 3 of 4 legs.
2. **`gap`'s committed horizon=24 IS the full-sample-best horizon** (analysis B) — a concrete instance
   of the §5f-forbidden exit-fishing baked into a committed PORTFOLIO headline.
3. **Sharpens the deploy case substantially.** The book's +0.589% mean — the entire path-A case —
   rests largely on exit-selection optimism: 2 legs flip negative, `fbr` loses 40%, only `me_long`
   (+0.23%) is honest-robust. An honest-§5f book mean is plausibly near-zero (a quantitative
   recomputation at each leg's nested-selected exit is the clean next arc, 2043).

**FAIR caveat (conservative bias, §8 — why FLAG not unilateral downgrade), identical in spirit to
arc 2041.** The committed exits are defensible FIXED, mechanism-motivated choices — a weekend gap-fill
needs a time exit to let the reversion play out (arc 1006/1007), month-end reversion is a 2-bar
fix-flow over-extension (arc 1011/2024). As FIXED choices they reproduce the committed numbers exactly
(gap +0.685%, me_long +0.232%). The nested-WFO negativity is partly small-n (8 evaluable folds)
adaptive-selection variance, and `gap`'s folds are extremely volatile so the nested estimator is noisy.
So the honest reading: `gap` is **exit/horizon-fragile to the point that no-lookahead selection yields
negative**, a strong caution on its mean-positivity; granting the fixed mechanism-motivated exit it
remains +0.685%. **Which standard governs (fixed mechanism-exit vs §5f-honest selection) is the
operator's path-A gate-governance call** — the same lever the corpus keeps converging on, now with a
much sharper input.

**FLAG F1 (docs only):** `gap`'s recorded +0.685% (PORTFOLIO component, arc 1006/1009) flips
mean-negative under honest §5f nested-WFO exit selection (IS), with its committed horizon=24 being the
full-sample-best horizon pick. Combined with arc 2041's `me_short` flag and arc 2040's `fbr` −40%, the
operator should weigh that the book's +0.589% mean-positive deploy case is substantially exit-selection-
optimistic — only `me_long` is honest-§5f-robust. (Recommended next: recompute the book mean / t-stat
at each leg's honest nested-selected exit — arc 2043 spec.)

**Components UNCHANGED** (all 4 stay PORTFOLIO; a chat does not unilaterally rewrite a committed
component's disposition). Lever unchanged = operator path-A gate-governance call. No new BUILT tool
(reused arc-2040's `nested_exit_selection`). No council. No OOS tuned (frozen scored once). Driver
scratch `_disco_work/arc2042_gap_melong_nested_exit.py`.

**NEW lesson (extends 2040/2041).** The honest §5f exit menu for a TIME-EXIT reversion leg is
`{exit_policy} × {SL} × {time-horizon}` — and the time-horizon is the binding knob (the registry
exit_policy is inert at a short hold, e.g. me_long's 2-bar D1 hold makes all 6 registry exits
identical). Auditing it completed the book's exit-honesty map: 3 of 4 thin reversion/fill legs carry
full-sample exit/horizon-selection optimism (gap & me_short flip negative, fbr −40%); only the leg
whose committed knob is CONSERVATIVE relative to the fishable optimum (me_long, h2 < fishable h3) is
honest-robust. A committed PORTFOLIO headline whose chosen horizon equals the full-sample-best horizon
(gap's h24) is a flag for exit-fishing.
