# arc 2016 — DIAGNOSTIC: is the 4-way book's residual block REAL signal or measurement-floor NOISE?

**Chat:** 2000s · **Range:** 2000–2999 · **Disposition:** **KILL** (no new component; the leg-hunt-to-paint-folds route is shown to hit its measurement floor) · **Council:** **HEAVY (generative idea-fork, §5b) — and it changed the arc.** The council redirected a planned 19th +2015-leg hunt into this diagnostic.

> **The headline finding (programme-level): the 4-way book's two negative folds — 2015 (−0.047%) and 2018 (−0.124%) — are STATISTICALLY INDISTINGUISHABLE FROM ZERO** (0.067σ and 0.176σ of the book's own per-fold ROI std). Bootstrapping the honest per-trade P&L, the book's 2015 ROI is −0.047% with sd **0.71%** and a 95% CI of **[−1.22%, +1.55%]** (P(neg)=0.42). **Hunting a 5th component to flip a 0.07σ fold positive is chasing measurement noise** — and any leg that *appears* to fix it would be fold-painting. The "structural 2015↔2018 anti-correlation" that motivated ~18 prior leg-hunts is an **n=10 artifact**: every pairwise component fold-ROI correlation's 95% CI spans zero. This is a **FLAG to the operator** on gate resolution (a governance/methodology call that is theirs, not mine — I did **not** touch the gate).

---

## 1. READ + SYNTHESIZE THE LOG (step a)

Pulled main (clean; arc 2015 = my last arc, the 4-way combination; 1019 = the co-discovered me_short component; 3015/3016/1018 = recent 2018/2015-leg KILLs). No STOP. Read protocol, full Tier-1 ledger (0–3016), LESSONS, TOOL_REGISTRY, the 4 portfolio-candidate configs, and the arc-2015 Tier-2 reasoning.

**State.** The corpus's live route is the PORTFOLIO book. **Four** net-positive decorrelated PORTFOLIO components exist: gap-fill long (1006, JPY-cross H4), month-end long (1011, USD-major D1), failed-breakdown-reclaim long (fbr, 1013, USD-major H4), month-end SHORT (1019/2015, USD-major D1). Arc 2015's 4-way convex combination: 0/1771 weightings all-folds-positive, **but me_short was the first new leg to MOVE the wall** (best worst-fold −0.222% [3-way] → −0.124% [4-way]), robustly fixing 2018. Arc 2015's stated thread-1: the residual block is **2015**, and the sharpest open spec is "a robust +2015 leg" (acute SNB/China/ECB-QE 2015 → an event/vol-shock mechanism). ~18 prior arcs hunted the 2015/2018 leg as a separate short/trend/vol/flow/rel-value component; all dead.

## 2. FORM AN IDEA → the council redirected it (step b, HEAVY council)

**First move (observe, §5b).** The arc-2015 spec pointed at vol-shock reversion. I observed it directly (D1 USD majors, direction-aware honest-capture lens, `arc2016_observe_volshock.py`): **fading an extreme 1-bar move (≥1.5/2.0/2.5 ATR; down→long, up→short) is DEAD** — capture **0.20–0.35** (far below 0.50), forward drift NEGATIVE (the extreme move CONTINUES). 2015 not positive (only +cells are single-trade SNB-unpeg USDCHF outliers). Vol-shock reversion is not the +2015 leg. This sharpened a structural read: the 2015 carrier (fbr, +3.17) is 2018-negative (−4.20); the 2018 carriers (me_short/long) are 2015-negative/luck → an apparent USD-directional **2015↔2018 anti-correlation**, so a clean +2015 leg would need to be non-USD-directional / 2018-neutral.

**Council (`/llm-council-discovery`, generative idea-fork).** Supplied the full structured input (the spec, the observations, the ~18 dead routes, the metrics). The verdict **redirected the arc**:
- Peer review was unusually unanimous — **all 5 reviewers ranked the Soundness lens strongest** and **all 5 named the Alternative-framing (loosen the gate) lens the biggest blind spot** (changing the ruler after seeing which fold fails = the failure mode that reset this repo; **rejected**).
- **Recommendation: investigate-first, do NOT open a 19th leg-hunt.** Compute three numbers never computed: (1) is worst-fold −0.124% statistically distinguishable from zero at ~10–15 trades/fold? (2) the 2015↔2018 correlation CI at n≈10 — does it span zero? (3) per-component 2015 decomposition — one component's hole (reweightable) or four-way co-drawdown (needs a leg)? *"If −0.124% is inside noise OR single-component-driven, skip the 5th leg entirely."*
- **Strongest dissent:** there is **no out-of-sample 2015** — a leg engineered to clear a fixed historical year is in-sample by construction; the real test is forward survival, which the book (failing IS all-folds-positive) has not earned.

**I committed to the council** (heavy/evaluative weight): converted arc 2016 from a leg-hunt into the diagnostic it recommended. Stayed **IS-only** — did not spend the combined-book OOS one-shot (§5g: the book fails IS all-folds-positive, so it has not earned the OOS look).

## 3. THE DIAGNOSTIC (steps c–g) — `arc2016_diagnostic.py`

Reproduced all 4 components at their EXACT committed configs via the canonical apparatus (Panel / `build_arc_pool`-backed signals / `ArcFoldRunner` / `build_v3_folds`), capturing each fold's honest **per-trade P&L** (`StrategyResult.closed_trades[].pnl`) and initial equity. Headlines verify EXACTLY before trusting anything: gap **+0.685%**, me_long **+0.232%**, fbr **+1.854%**, me_short **+0.683%**. IS-best convex weights (step-0.05 grid, maximize worst-fold) = **{gap 0.0, me_long 0.65, fbr 0.2, me_short 0.15}**, worst-fold −0.124%, mean +0.624% — reproduces arc 2015.

Book per-year ROI: 2011 +2.28 · 2012 +1.05 · 2013 +0.67 · 2014 +0.04 · **2015 −0.05** · 2016 +0.04 · 2017 +0.37 · **2018 −0.12** · 2019 +0.96 · 2020 +1.01. (Book per-fold std **0.703%**.)

**(1) 2015 decomposition** — book 2015 (−0.047%) is **single-component-driven, NOT a co-drawdown**: me_long contributes **−0.742%** (w0.65 × −1.141%, n=10), nearly offset by fbr **+0.635%** (w0.2 × +3.174%) and me_short +0.060%; gap is at weight 0. It is a me_long-vs-fbr near-cancellation, netting −0.047%.

**(2) NOISE FLOOR (decisive)** — bootstrap (10k, seed 42) of each component's 2015 ROI from honest per-trade P&L contributions, combined at the frozen IS weights:
- Per-component 2015 sampling sd is HUGE at these trade counts: gap ±2.64%, fbr ±2.62%, me_short ±1.27%, me_long ±0.68%.
- **BOOK 2015 ROI: point −0.047%, bootstrap sd 0.71%, 95% CI [−1.22%, +1.55%], P(book2015<0) = 0.418.**
- **|worst-fold| / book-2015-sd = 0.067.** The "block" is **6.7% of one standard deviation.** 2018's −0.124% = 0.176σ. Cross-check (model-free): both negative folds sit deep inside one book-σ; **6 of 10 folds are within ±1σ of zero** — only 2011/12/19/20 clear the noise floor robustly.

**(3) Correlation CI** — Fisher-z 95% CIs on the 6 pairwise component fold-ROI correlations (n=10) **ALL span zero**: gap·me_long +0.117 [−0.55,+0.70]; gap·fbr +0.189 [−0.50,+0.73]; gap·me_short +0.188 [−0.50,+0.73]; **me_long·fbr −0.366 [−0.81,+0.34]** (the "anti-correlation" — spans 0 widely); me_long·me_short +0.157 [−0.52,+0.72]; fbr·me_short +0.406 [−0.30,+0.83]. The structural 2015↔2018 anti-correlation narrative is **statistically unfounded at n=10**.

## 4. VERDICT + WHAT IT MEANS

**KILL** — not a tradeable outcome but a **programme-redirecting diagnostic**. The chairman's skip-the-5th-leg condition is satisfied on **both** clauses (inside noise AND single-component-driven). Concretely:

- **The 4-way book's residual negative folds are measurement noise, not a fixable hole.** No 5th component can *reliably* flip a 0.07σ fold; one that *appears* to is fold-painting (selecting a leg against the test statistic — the council's unanimous-strongest concern). The "find a +2015 leg" route (and the "+2018 leg" route before it) has been **optimizing convex weights and hunting thin components at a resolution finer than ~10–30-trade yearly folds support** (per-fold noise floor ±0.7–2.6%).
- **The 4-way book remains a strict-gate FAIL** (not all-folds-positive) → components UNCHANGED (still PORTFOLIO); the book is KILL like arcs 2006/2008/1015/2015. **I did NOT loosen the gate** (the rejected Alternative-framing lens) — I report that the residual is noise.
- **OPERATOR FLAG (governance, theirs to decide).** The all-folds-positive-on-calendar-year gate, evaluated on books of thin decorrelated components, is being applied **below its own noise floor** — a worst-fold of −0.1% on ±0.7% per-fold noise is a coin-flip sign. Two honest paths follow, both the operator's call: **(A)** reconsider the fold/gate resolution for thin-component books (e.g. pooled-trade or regime-block gating with explicit SE), or **(B)** redirect discovery toward components whose *per-fold* ROI clears the noise floor (more trades/fold and/or larger per-trade edge), rather than ever-thinner decorrelated legs whose yearly ROIs are sampling-error-dominated. The genuine forward test (the strongest dissent) — the 4-way's 2021+ OOS survival — remains **deferred** until a book earns the OOS look.

## 5. THREADS / LESSONS

1. **NEW reusable lesson (high value): the portfolio-book gate has a NOISE FLOOR set by component trade-counts.** A negative fold smaller than ~1 book-σ (~0.7% here) is statistically zero; "fix it with a 5th leg" is chasing noise and invites fold-painting. **Quantify a marginal fold's bootstrap CI BEFORE hunting a component to flip it.** This retroactively reframes the ~18-arc 2018/2015-leg hunt: me_short's "robust +2018" was real *as a component*, but the convex-search worst-fold improvements (−0.222→−0.124) were largely **moving inside the noise floor**, not closing a real gap.
2. **The 2015↔2018 anti-correlation is an n=10 story, not a structural constraint** — all six component correlations' CIs span zero. Future combination arcs should treat per-fold correlations at n≈10 as uninformative.
3. **me_long is the 2015 drag** (−1.141% its single worst fold) at the IS-best weights — but reweighting away from it would surrender its robust +2018 (its reason for inclusion). The tug is real but sub-noise.
4. **Council process worked exactly as designed** — it caught a sunk-cost treadmill (about to spend the 19th arc painting a noise fold) and redirected to the measurement that dissolves the question. Generative council at a genuine fork earned its cost.

**FLAGS (code not merged):** none touching the canonical core. No new BUILT tool (the diagnostic is one-off arithmetic on canonical outputs; the `closed_trades` per-trade P&L capture is a read of `StrategyResult`, not a new measurement path). Drivers in scratch `_disco2000_work/arc2016_*.py` (`observe_volshock`, `diagnostic`), reproducible; the diagnostic reproduces the 4 committed component headlines EXACTLY before any bootstrap. Council transcript summarized in §2 (not separately saved — generative fork, not a survivor stress-test).
