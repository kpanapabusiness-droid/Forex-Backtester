# arc_1063 — POWER AUDIT of the all-folds-positive discovery judge

**Chat:** 1000s | **Date:** 2026-06-06 | **Disposition:** KILL (diagnostic — no new component; quantifies the judge, does NOT loosen it) | **passed:** N

> Council-recommended meta-arc. The discovery council (light, generative) was convened at a
> genuine idea-fork (every fresh-eyes idea kept landing on already-tested ground). The convergent
> finding of the **peer-review layer** — surfaced by 4 of 5 reviewers, raised by none of the 5
> lenses — was: *before trusting the ~90-arc "frontier exhausted" verdict, audit the JUDGE.* The
> `all-folds-positive-per-calendar-year` gate is a high-variance, near-binary, low-power estimator;
> with ~10–16 annual folds a genuinely positive-EV edge can be rejected by per-fold sign-test
> sampling noise (Type-II). This arc measures that power. It does **not** loosen, replace, or
> re-implement the gate (cf. arc 1023, which "quantifies why, does NOT loosen it").

---

## §5a log read (FRESH EYES, honest-era only) — what I synthesized before acting

- **Survivors (4, all PORTFOLIO, all forced-flow REVERSION):** gap-fill (1006, JPY-cross down-gap),
  me_long (1011), me_short (1019), fbr (1013). Mean-positive, none all-folds-positive.
- **Book:** fails all-folds, blocked combination-invariantly by 2015 (only fbr +) & 2018 (only
  me_long +) — opposite legs, no convex weighting passes both (arcs 1015/2008/3009). Item E co-sim
  confirmed the failure is FUNDAMENTAL (not a combiner artifact). Book mean +0.624%, across-fold sd
  0.741%, 8/10 (arc 1023, t=2.66 on committed numbers) — but 3/4 legs are exit-selection-optimistic
  under honest §5f (arc 2042: gap & me_short flip negative; only me_long survives), and me_long-solo's
  mean is itself non-sig + 2025-tail-carried (1056/1057/1058).
- **This session's frontier is dead:** both dispatch short leads (up-gap short 1016/2013; fbr
  short-mirror 1014/2009/2011/3011); explore-now MENU exhausted (M1 1027, O1 1029/1025/1055, L1 1054,
  Q1 peg-defense 1028, G1 2052, S1); me/fbr ported to crosses (2049/1021/3018); intraday session
  (3016/2050/1026); calendar-flow seasonals (gotobi/ToM/IMM/quarter-end/JPY-FYE 1061); the 5th
  regime-orthogonal leg unfound across ~22 routes (1059–1062 the latest); peg-defense; vol-breakout +
  direction-agnostic straddle (1062). My own fresh-eyes brainstorm (me-on-crosses, first-of-month
  inflow, turn-of-year, cross-pair cascade) each resolved to already-tested ground.
- **Ledger fact (cheap, decisive — council crux #3):** disposition tallies = **0 PASS, 7 PORTFOLIO,
  146 KILL**. The all-folds-positive gate has **NEVER been satisfied** in the honest era — not solo,
  not combined. That 0/90 is the datum this arc explains.

This is a genuine stuck-state → §5b LIGHT generative council (the protocol's idea-generation tool).

## Council (light, generative — `/llm-council-discovery`)
- **Lenses proposed** (CC weights lightly, commits its own): C=cross-pair liquidation-cascade fade;
  D=mechanism-anchored (phase-locked) fold boundaries; A=export fbr's USD-factor-idiosyncratic
  separator as a shared cross-leg veto; E=spot short-gamma (self-refuting — loads onto 2015/2018);
  B=continuation/positive-skew shapes (the corpus only ever tested reversion).
- **Reviewer convergence (the load-bearing output):** 4/5 reviewers, unprompted, flagged that *no
  lens questioned the judge's statistical power*; the all-folds-per-year gate may reject true
  positive-EV edges by sampling noise, so "exhaustion" may be a measurement artifact. Chairman
  recommendation: **investigate-the-judge-first** (cheaper than any arc, dual honest outcome). E was
  unanimously parked. Lens A's premise was challenged (the 2015/2018 block is a *sign* conflict, not
  merely co-located noise → a symmetric veto can't make a year prefer the leg it rejects).
- **CC commit:** adopt the redirection. Arc 1063 = the judge power audit (crux #2) + the ledger fact
  (crux #3). Logged forward (not executed here): lens-A's correlation-vs-sign crux (cheap: are
  me_long's 2018 losing bars USD-factor-wide?); lens-B continuation/positive-skew (the one untested
  *shape* family — guard: pre-register median-per-fold + tail-removed expectancy before believing it).

## Method (no price / no engine / no OOS touched — pure statistics of the judge)
- **Tool (BUILT):** `discovery/tools/judge_power_audit.py` — builds synthetic per-fold `FoldStats` of
  KNOWN expected value and feeds them to the **canonical** `judge_all_folds_positive` (rule = every
  `roi_pct > 0`). Fold COUNTS come from the canonical builders (`build_v3_folds` IS, `build_oos_year_folds`
  OOS). Determinism `default_rng(42)`. Driver `discovery/_disco1_work/arc1063_judge_power_audit.py`.
- **Backbone:** a year's ROI ~ (mean μ, sd σ); P(year>0)=Φ(μ/σ)=Φ(S) where S = per-year Sharpe (the
  realized mean-over-sd of the annual ROI series, NOT annualized). Folds ~independent (distinct
  scored years) ⇒ P(all-K-folds-positive)=Φ(S)^K. Validated three ways: (i) the canonical judge MC
  reproduces the vectorized rule and the analytic Φ(S)^K; (ii) the **independent cross-check** — at
  the book's published (μ=0.624%, σ=0.741%, K=10) the backbone gives P=0.1076 vs arc-1023's
  independent per-trade/fold BOOTSTRAP 0.1130 (≈ exact; also confirms fold-dependence is negligible,
  since 1023's bootstrap carried the real dependence and my independent-Gaussian matches it);
  (iii) a convex SL-honest take-the-loss per-trade Monte-Carlo for the fat-tail penalty.

## Results

**Canonical judged fold-set sizes:** K_IS = 10, K_OOS = 6, K_JOINT = 16 annual folds.

**The judge's power — per-year Sharpe needed to PASS all-folds-positive:**

| target P(pass) | K_IS=10 | K_OOS=6 | joint=16 |
|---|---|---|---|
| 0.50 | **1.50** | 1.23 | **1.72** |
| 0.80 | **2.01** | 1.79 | **2.20** |
| 0.95 | 2.57 | 2.39 | 2.73 |

**P(all-folds-positive) at representative per-year Sharpe:**

| per-year Sharpe | P(IS 10) | P(OOS 6) | P(joint 16) |
|---|---|---|---|
| 0.50 | 0.025 | 0.109 | 0.003 |
| **0.84** (book) | **0.107** | 0.261 | **0.028** |
| 1.00 | 0.178 | 0.355 | 0.063 |
| 1.50 | 0.501 | 0.660 | 0.331 |
| 2.00 | 0.794 | 0.871 | 0.692 |

**Real survivors placed on the curve (published per-year ROI series):**
- **me_long:** IS Sharpe **+0.254** (5/8 pos) → P(all-IS)=0.017; OOS Sharpe +0.501 (5/6 pos) → P(all-OOS)=0.110. Judge fails both (as observed).
- **4-way book:** Sharpe **0.842** → P(all-10)=0.108 ≡ arc-1023's independent 0.113.

**Fat-tail penalty (convex take-the-loss vs matched-mean Gaussian):** real but **secondary** —
×1.1–1.2 at survivor thinness (12–25 trades/yr). The CLT mostly normalizes the year-sum; the binding
problem is the **low per-year Sharpe**, not the skew. (At the survivors' true per-trade edge
~+0.05–0.10R the per-year Sharpe is only ~0.2–0.45 → P(all-IS) ≈ 0.4–1.8%.)

## Interpretation (decision-grade — the honest both-halves)

1. **The all-folds-positive gate is a Sharpe≈2/year CONSISTENCY screen.** To pass it ≥80% of the time
   over the joint 16 folds an edge needs a per-year Sharpe ≈ 2.2 — world-class systematic consistency
   (deployed CTAs typically run annual Sharpe 0.5–1.0). Real, genuinely-positive-mean edges of the
   survivors' magnitude (per-year Sharpe ~0.25–0.84) pass it only ~2–11% of the time.
2. **⇒ The 0/90 PASS rate is consistent with the gate's LOW POWER, not only with edge-absence.** A
   programme running ~90 thin candidate edges through a Sharpe≈2 screen is *expected* to produce zero
   passers even if several real, deployable-magnitude edges existed. This **reframes "frontier
   exhausted"**: the corpus has shown there is no edge so strong it survives the all-folds gate — a
   much WEAKER statement than "no deployable edge exists."
3. **BUT low judge power does NOT manufacture deployability (Soundness / §8 conservative bias).** A
   real-but-thin edge must still be REAL (positive mean net of costs, beats the null) AND certifiable
   by a PROPER estimator (mean + CI / tail-robustness) — and the corpus already established
   (1023/1056/1057/1058/2042) that the book and me_long means are only borderline-to-non-significant
   and tail-fragile under honest §5f exit accounting. So the audit's real service is to **separate two
   things the programme had conflated:** *"fails all-folds"* (a low-power consistency screen the thin
   survivors were always going to fail) vs *"mean not certifiable"* (the actual binding deployment
   obstacle, on the proper estimator). They converge on the SAME operator decision but for a cleaner,
   correctly-attributed reason.
4. **The right deployment question is the mean significance / tail-robustness one, NOT the all-folds
   screen** — which is exactly the operator's standing **path-A** governance call (a mean / pooled /
   regime-block gate vs the strict all-folds gate; arcs 1023/2019). This arc quantifies *why* the
   strict gate never fires and *why* it is the wrong instrument for a thin-but-real edge — decision
   support for that call. **The strict all-folds gate STAYS the gate (not loosened);** the honest
   path to a deployable verdict runs through certifiable-mean, which the corpus has separately judged
   marginal.

## Verdict
**KILL** (diagnostic; no new component; the strict gate is unchanged). The "frontier exhausted"
verdict is REFRAMED, not overturned: the in-apparatus *edge* search is genuinely thin, but the 0/90
all-folds result is largely a property of an extremely strict (≈Sharpe-2/year) consistency screen, not
proof of edge-absence — and the binding deployment obstacle is the (separately-established, marginal)
certifiable-MEAN question, an operator path-A governance call.

## Threads / lessons
1. **NEW lesson — the all-folds-positive-per-year discovery judge is a ≈Sharpe-2/year consistency
   screen** (≈2.0 per-year Sharpe for an 80% pass over 16 folds; 1.7 for 50%). Real positive-mean
   edges at the survivors' thinness (Sharpe ~0.25–0.84) pass it only ~2–11% of the time → a 0/N
   all-folds result on thin candidates is weak evidence of edge-absence; it is mostly the screen's low
   power. SEPARATE "fails all-folds" (consistency screen) from "mean not certifiable" (the real
   deployment test) — do not read the former as the latter.
2. **Method banked:** `judge_power_audit.py` — power/false-rejection characterization of ANY
   all-folds-positive verdict via the canonical judge; reusable to put a confidence interval on
   *future* KILLs (is this a real KILL or a power KILL?). Cross-validated against arc 1023's
   independent bootstrap (0.108 vs 0.113).
3. **Forward threads (council-surfaced, not executed):** (a) lens-A correlation-vs-sign crux — are
   me_long's 2018 losing bars predominantly USD-factor-wide (cheap panel query; settles whether a
   shared USD-factor veto could help the book or whether 2015/2018 is an irreducible sign conflict);
   (b) lens-B — the corpus tested ONLY reversion shapes; *continuation/positive-skew* shapes survive
   take-the-loss differently (need skew, not >0.50 capture) — the one untested shape family, with a
   pre-registered guard (median-per-fold + tail-removed expectancy BEFORE believing it) against
   relabeling 2025/top-5 tail-luck as "positive skew."

**FLAGS (code not merged):** none — `judge_power_audit.py` is an EXPERIMENT tool (statistics over the
canonical judge), no canonical-core change. **No engine / no price / no OOS touched.** Components
UNCHANGED (all 4 PORTFOLIO). Lever = operator path-A (now with the gate's power quantified).
