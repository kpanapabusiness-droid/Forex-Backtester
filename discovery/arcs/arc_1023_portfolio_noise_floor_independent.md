# arc 1023 — INDEPENDENT verification of the 4-way book's NOISE-FLOOR (arc 2016, by a different method)

**Chat:** 1000s · **Date:** 2026-06-05 · **Verdict:** KILL (no new component; strict all-folds gate stays FAIL)
**Disposition:** KILL · **passed:** N · **Components touched:** none (all 4 UNCHANGED, still PORTFOLIO)

> Arc 2016 (2000s) made a programme-redirecting claim: the 4-way book's residual negative folds (2015
> −0.047%, 2018 −0.124%) are statistically indistinguishable from zero, so the ~18-arc 5th-leg hunt was
> moving inside the noise floor. The Arc-10 norm is **independent reproduction by a DIFFERENT method.**
> This arc re-runs all 4 components through the canonical apparatus (reproducing every headline EXACTLY),
> reconstructs the book at arc 2016's frozen weights, and confirms the noise floor by an **across-fold**
> lens (vs 2016's within-fold per-trade bootstrap). **Confirmed — and extended: the book MEAN is
> significantly positive (t=2.66, p≈0.026), the per-year all-folds gate is below its noise floor.**

---

## Log reading (step a — FRESH EYES, honest-era only; pulled main, no STOP)

Resumed 1000s at arc 1023 (prior in-range 1022; pulled main → two concurrent arcs landed: my arc 1022
and 3000s' arc 3018 are **independent reproductions of the same KILL** — fbr does NOT transfer off USD
majors; and 2000s' **arc 2016** is the programme-redirecting noise-floor diagnostic). State: four
net-positive PORTFOLIO components (gap 1006 JPY-cross H4, me_long 1011 USD-major D1, fbr 1013 USD-major
H4, me_short 1019 USD-major D1). The 4-way book (arc 1020) breached the 2018 wall (worst −0.115%) but is
not all-folds-positive; arc 2016 then showed the residual block is **statistically zero**.

## Idea + why (independent verification, not another leg-hunt)

My own arc 1022 (and 3018) were exactly the thin-leg hunt arc 2016 deprecates — both correctly KILLed.
Continuing to hunt thin legs to flip a 0.07–0.18σ fold is **fold-painting** (selecting against the test
statistic). The highest-value, non-redundant move is to **independently verify the noise-floor claim**,
since if it holds it closes the book route and stops the treadmill, and if it fails it catches an error in
a programme-redirecting result. arc 2016 used a within-fold per-trade bootstrap of the COMBINED book; I
use an **across-fold** lens (the 10 IS fold ROIs as 10 annual samples) + a fold-resample bootstrap with a
different seed — genuinely independent method AND chat (the corpus norm: 2008/3009, 1019/3017, 1022/3018).
**No gate-loosening** (arc 2016's council rejected that lens): the strict all-folds-positive gate stays a
FAIL; I only quantify *why* it fails and whether the failure is real.

## Method (CALLED canonical; IS-only; combined-book OOS NOT spent — §5g, book fails IS AFP)

Driver `_disco_work/arc1023_noise_floor.py`. Re-ran each component through the canonical apparatus
(`Panel.from_pairs` → registered signal → `A1Architecture`/`ArcFoldRunner` →
`run_config_over_folds` over `build_v3_folds` IS folds; FundedNext costs netted in `FoldStats`), at the
EXACT committed configs from each `portfolio-candidates/*/config.yaml`:
- **gap** `WeekendGapFillLongSignal(0.5, 36)`, 5 JPY crosses H4, 24-bar `make_time_exit_predicate`, no TP/trail.
- **me_long** `MonthEndReversionLongSignal(1.0, 2)`, 7 USD majors D1, `sl_only` + 2-bar time exit.
- **fbr** `FailedBreakdownReclaimLongSignal(40, 1.25)`, 7 USD majors H4, `sl_plus_trailing_atr`, `trail_enabled=True` (the committed double-trail).
- **me_short** `MonthEndReversionShortSignal(1.0, 2)`, 7 USD majors D1, `sl_partial_close_1r_runner_trail`.

**Risk-convention calibration (arc-3017 FLAG-1, load-bearing).** At `A1Config.risk_pct=0.5` (the
intended PERCENT) the daily-5%-DD cap BLOWS THROUGH and ROIs explode to the hundreds of % with flipped
fold signs (gap mean −334%, me_short +100%) — arc 3017's nonlinear-DD warning, reproduced. The committed
headlines live in the **low-risk LINEAR regime at `risk_pct=0.005`**, where I reproduce all four
**EXACTLY** (see below). The noise-floor σ-distance is scale-invariant *within* the linear regime, so the
verdict is robust to the convention; but the standing FLAG-1 (the whole portfolio characterization is at
0.005, possibly 100× below deployable risk, where the DD-cap nonlinearity changes fold signs) is restated.

## What happened — every headline reproduced EXACTLY (Arc-10), book matches arc 2016

| component | my mean | headline | pos folds | total trades |
|---|---|---|---|---|
| gap | **+0.685%** | +0.685% | 5/10 | 260 |
| me_long | **+0.232%** | +0.232% | 7/10 | 98 |
| fbr | **+1.854%** | +1.854% | 9/10 | 208 |
| me_short | **+0.683%** | +0.683% | 7/10 | 145 |

**Book at arc 2016's frozen weights** w={gap 0, me_long .65, fbr .2, me_short .15}: mean **+0.624%**,
worst **−0.124%** (fold 9 = 2018), 2015 (fold 6) **−0.047%**, **8/10** positive, across-fold sd **0.741%**
— reproduces arc 2016 (mean +0.624%, worst −0.124%, 2015 −0.047%, sd 0.703%) to the decimal. Per-fold ROI
%: {2011:+2.28, 2012:+1.05, 2013:+0.67, 2014:+0.04, 2015:−0.047, 2016:+0.04, 2017:+0.37, 2018:−0.124,
2019:+0.96, 2020:+1.01}.

**Noise floor — INDEPENDENT confirmation (across-fold lens + fold bootstrap):**
- **Worst fold −0.124% = −0.167 across-fold-sd from zero** ≈ arc 2016's within-fold **0.176σ**. Two
  independent noise models agree the worst fold is ~0.17σ from zero = **statistically zero.**
- **across-fold sd (0.741%) ≈ arc 2016's within-fold sampling sd (0.703%)** — the **decisive
  method-independent statement.** If the per-year folds carried real regime signal beyond sampling noise,
  the across-fold dispersion would EXCEED the within-fold sampling sd. They are nearly equal ⇒ the
  fold-to-fold ROI variation is **essentially all sampling noise**; the year-folds carry almost no real
  signal beyond the pooled mean. This is the cleanest possible confirmation of arc 2016.
- Fold-resample bootstrap (seed 123, B=20k): P(any negative fold in a 10-fold draw) = **0.887** — having
  ≥1 negative fold is EXPECTED, not anomalous; the all-folds-positive gate asks for a 10/10 streak from a
  process whose per-fold P(neg) ≈ 0.4.

**NEW decision-critical finding — the book MEAN is significantly positive:**
- one-sample t on the 10 annual fold ROIs: **t = +2.66 (df=9, p ≈ 0.026 two-sided)** → reject mean = 0.
- fold-resample bootstrap mean 95% CI = **[+0.22%, +1.09%]**, **P(mean ≤ 0) = 0.000**.
- ⇒ the 4-way book is a **genuine positive-expectancy strategy**; it fails ONLY the per-year
  all-folds-positive gate, and that gate sits below its own noise floor.

## Diagnosis + meaning

The all-folds-positive **calendar-year** gate, applied to a book of thin (10–28 trade/fold) decorrelated
components, demands a 10/10 positive-year streak from a process whose per-year ROI sampling sd (~0.7%) is
LARGER than its per-year mean (~0.6%). Such a streak is statistically near-impossible regardless of edge
quality — the gate is **below its noise floor.** The book has a real, significant positive mean; the two
"failing" folds are 0.17σ negative draws. The ~18-arc 5th-leg hunt's worst-fold "improvements"
(−0.77 → −0.124) were largely **moving inside the noise floor**, exactly as arc 2016 concluded — now
confirmed by a second method and a second chat.

This sharply scopes the two forward paths the operator already flagged (arc 2016):
- **Path A (gate resolution):** the per-year all-folds gate is the wrong resolution for thin-component
  books. A pooled-trade or regime-block gate with explicit SE — or simply "mean significantly > 0 with a
  bounded worst regime-block" — would treat this book as the real positive edge it is. **(Decision-support
  only; I do NOT adopt a looser gate or claim a pass — operator's call, §9 code/governance is human-gated.)**
- **Path B (denser components):** to make the per-year folds individually significant under the EXISTING
  gate, components need more trades/fold and/or a larger per-trade edge (so per-fold SE shrinks below the
  per-fold mean). The corpus's mapping (closed ground: dense/shallow = coin-flip; the real edges are
  calendar/structural and inherently sparse) suggests this is hard in-apparatus — but it is the only path
  that keeps the strict gate.

## Verdict: KILL (no new component; strict gate stays FAIL)

Independent reproduction CONFIRMS arc 2016: the 4-way book's residual negative folds are statistically
zero; the per-year all-folds-positive gate on this thin-component book is below its noise floor. The book
mean is significantly positive (t=2.66) but the strict all-folds gate is a FAIL (8/10) and stays one —
this arc quantifies the failure, it does not loosen the gate. Components UNCHANGED (all 4 PORTFOLIO); no
new portfolio-candidate. Combined-book OOS NOT spent (book fails IS AFP, §5g). No council spent (this is a
measurement that resolves a measurement question, not an idea-fork; arc 2016 already ran the council that
redirected the leg-hunt).

## Threads / lessons

1. **CONFIRMED (independent method + chat): the portfolio-book per-year all-folds gate has a NOISE FLOOR
   set by component trade-counts.** across-fold sd (0.741%) ≈ within-fold sampling sd (0.703%) ⇒ the
   year-folds are sampling-noise-dominated. A 5th leg to flip a 0.17σ fold is fold-painting. **Reusable
   rule (now twice-confirmed): quantify a marginal fold's CI BEFORE hunting a component to flip it; and
   for a thin-component book, judge the MEAN (with SE), not a 10/10 per-year streak.**
2. **NEW: the 4-way book's mean is statistically positive (t=2.66, CI [+0.22%,+1.09%]).** The book is a
   real edge that fails only a too-fine gate — this is decision-support for the operator's path A, not a
   pass (the strict gate is sovereign and stays FAIL).
3. **Risk-convention FLAG-1 (arc 3017) re-confirmed and load-bearing here:** all component headlines + the
   noise-floor analysis live at `A1Config.risk_pct=0.005` (linear regime); at the intended `0.5` the
   daily-DD cap blows through and flips fold signs. The portfolio's deployability picture is
   risk-convention-dependent until FLAG-1 is resolved (code human-gated). Any deployable-risk evaluation
   must re-derive the noise floor at that risk (the linear-regime σ-distances will NOT carry over through
   the DD-cap nonlinearity).
4. **Programme status:** the in-apparatus 5th-leg route is noise-floor-blocked (path B hard given closed
   ground); the live levers are operator-gated (path A gate-resolution decision, or escalation #3
   tighter-cost regime / a non-OHLCV data source). Autonomous discovery has mapped the book route to its
   honest floor.

## Tooling

No new BUILT tool — all CALLED (`Panel.from_pairs`, the 4 registered signals, `make_time_exit_predicate`,
`A1Architecture`/`ArcFoldRunner`, `run_config_over_folds`; scoring canonical via `MultiPairBacktester`).
Noise-floor statistics are one-off arithmetic on canonical `FoldStats.roi_pct` outputs. Driver
`_disco_work/arc1023_noise_floor.py` (reproducible; reproduces the 4 headlines EXACTLY at `risk_pct=0.005`
before any analysis — Arc-10 discipline).

## FLAGS (code not merged)

- **FLAG-1 (arc 3017, restated, load-bearing):** `A1Config.risk_pct` PERCENT-vs-FRACTION unit split + the
  daily-DD-cap nonlinearity in `risk_pct`. The portfolio characterization is risk-convention-dependent.
  Code human-gated — NOT patched.
- **Governance (operator's call, not mine):** the per-year all-folds-positive gate is, for thin-component
  books, evaluated below its noise floor (arc 2016 OPERATOR FLAG, independently confirmed here). Whether
  to adopt a mean/pooled/regime-block gate (path A) is an operator/protocol decision — FLAGGED, not taken.
- Carries the standing `A1Config.time_exit_bars`-unwired flag (arc 1005). OOS never touched.
