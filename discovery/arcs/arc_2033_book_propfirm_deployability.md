# arc 2033 — prop-firm-challenge deployability of the 4-way book (Sharpe + profit-target-vs-maxDD)

**Chat:** 2000s · **Date:** 2026-06-06 · **Verdict:** DIAGNOSTIC → **KILL** (no new component; all 4 components UNCHANGED, PORTFOLIO; book stays strict-gate FAIL and is shown INFEASIBLE on the operator's actual deployment vehicle)
**Disposition:** KILL · **passed:** N · **Component touched:** none (deployment-vehicle book characterization)

> Every prior book-characterization arc measured an *academic* property — noise floor (2016/2017/1023),
> ENB (2019), temporal stability (2021), cost robustness (3022), gate-coarsening map (1032), raw risk
> geometry (1033). **None connected the book's risk-geometry to the operator's ACTUAL deployment vehicle:**
> a FundedNext / 5ers PROP-FIRM account, whose pass/fail rule is a **profit-target vs max-drawdown** hurdle,
> not all-folds-positive. And the corpus never reported the book's **Sharpe** (it has Calmar [1033] + the mean
> t-stat [1023], but not the single most-standard deployment metric). This arc fills both gaps.
> **Result — two separate, decisive findings:** (1) the book's risk is **DRAWDOWN-SHAPED, not vol-shaped**:
> daily-annualized Sharpe ~0.4–0.5 / annual-fold Sharpe ~0.67 (modest-but-real, reconciles t=2.66) yet
> Calmar only 0.24–0.36 → **Calmar ≪ Sharpe**, the signature of persistent, serially-correlated multi-year
> drawdowns (the ~4.7yr underwater chain of arc 1033). (2) **The book CANNOT pass a prop-firm CHALLENGE in
> any realistic timeframe** — for every realistic config it needs **1.4–8.4 yr operating AT the max-DD limit**
> (safe operation ~2× that) to reach the profit target by leverage; not a near-miss, off by years. The
> binding wall is Calmar (arc 1024: risk-invariant → leverage can't rescue it); the daily-DD cap always has
> headroom. **Deployability is VEHICLE-METRIC-dependent:** a DD-gated vehicle (prop-firm challenge) → infeasible;
> already-funded capital at low risk (the operator owns dormant funded accounts) → a marginal slow-grind
> diversifier (~0.4–0.9%/yr, Sharpe ~0.5, ~98% time-underwater), the operator's risk-appetite call.

---

## Log reading (step a — FRESH EYES, honest-era only)

Pulled main; no `discovery/STOP`. Resumed 2000s at the highest in-range id (2032) + 1 → 2033.

Converged corpus state (~70 honest-era arcs across 3 chats):
- **Closed ground:** single-condition shallow directional prediction (momentum / breakout / mean-reversion /
  trend) is dead on liquid FX across H1/H4/D1/**W1** (3014), majors + crosses, **long AND short**, capture
  AND drift lenses, every exit/SL, stop-removed (3004). Forward drift ≈ cost everywhere.
- **4 net-positive PORTFOLIO components:** gap-fill 1006 (JPY-cross H4 weekend down-gap fill), month-end-long
  `me_long` 1011 (USD-major D1), failed-breakdown-reclaim `fbr` 1013 (USD-major H4 — the crown jewel,
  +1.854%/9-of-10, the only fold-resolving edge), month-end-short `me_short` 1019 (USD-major D1, the first
  robustly-+2018 leg).
- **The 4-way book** is a sound ~3-independent-bet (ENB 3.32, arc 2019), mean-positive (t=2.66,
  P(mean<0)=0.004, arc 1023; RP +0.589%, arc 2019), temporally robust (2021), cost-robust (break-even κ=3.32,
  3022) PORTFOLIO whose all-folds-positive (AFP) failure is purely the per-year gate sitting **below the legs'
  noise floor** (2016/2017). Its raw risk geometry (1033): contiguous maxDD ~1.6%, but ~4.7yr deepest drawdown
  (Oct-2014→Jun-2019), ~98% time-underwater, **Calmar ~0.24–0.36**.
- **Both edge-levers are closed:** path-B densification is PROVABLY closed (3021: shared USD factor ρ≈0.12
  floors P(AFP) below 0.9 at any N); the explore-now MENU (M1/O1/L1/Q1/G1/S1) is exhausted; every forced-flow /
  directional / structural / short / lead-lag / relative-value / session / calendar angle is mapped dead. The
  2018 leg is unfound across ~18 routes. The sole remaining deployability lever is the operator's **path-A**
  (adopt a mean/pooled/CI gate; gate-coarsening doesn't work, arc 1032).

Open threads at resume: by the log's own account the **autonomous edge-hunt is at genuine exhaustion** and the
remaining value is **operator-decision support** on the one artifact the programme produced (the 4-way book).

## Idea (step b — observe, don't guess)

**Fresh-eyes check FIRST (don't accept exhaustion on faith, §2 / arc-3004 council warning that "the apparatus
is incapable" is a seductive search-ending conclusion):**

- Arc 1033 asserted the one untouched cross-instrument angle (commodity → commodity-currency lead-lag, e.g.
  oil→CAD / gold→AUD — which would be a mechanism genuinely OFF the USD factor that arc 3021 proved the book
  needs) is **"DATA-GATED (histdata corpus is FX-pairs-only)"** — but stated, not verified. **I verified it by
  inspection:** `C:\Users\panap\histdata_backup` holds exactly **28 FX pairs, no XAU/XAG/WTI/SPX** (full listing
  checked). The commodity-lead-lag angle is genuinely data-gated → **closed by verification, not assumption.**
- Every other OHLC-constructible mechanism I could brainstorm is **reason-killable by an existing closure**, not
  needing a fresh engine run: cross-pair / hub→cross **lead-lag** is dead because the triangulation identity
  re-prices WITHIN the H4 bar (3005/1027/2023/2028) and finer TF is cost-walled; a **common-factor dollar-shock**
  directional bet is MAXIMALLY cross-pair-clustered → worst-case for the 5%-daily-DD cap → structurally
  un-scalable (the arc-1017/3019 currency-/daily-cap death, by construction); **vol-direction** is dead
  (vol = magnitude not direction, closed ground); the **forced-flow well is enumerated dry** (weekend gap ✓,
  month-end WMR ✓, stop-runs ✓, Tokyo/London fix sub-cost, quarter/fiscal-year priced-in, peg-defense
  un-gateable, option-expiry data-gated). No novel mechanism survives a-priori screening — manufacturing one
  would grind proven-dead ground (explicitly warned against, §5a).

**Conclusion: the highest-EV arc is NOT another edge cheap-kill — it is the one missing operator-decision input.**
Idea (log-dry → the artifact itself): every book-characterization arc measured an *academic* property; **none
asked whether the book is deployable on the operator's ACTUAL vehicle.** The operator owns dormant FundedNext /
5ers accounts (CLAUDE.md). A prop-firm account's gate is a **profit-target vs max-drawdown** challenge — a
fundamentally different test from all-folds-positive. And the corpus never reported the book's **Sharpe** (the
standard deployment metric). Observe the book through the deployment vehicle's lens.

## Characterize / method (steps c–f)

GEOMETRY ONLY on the already-computed canonical contiguous curve — no engine re-run for the analytics, no new
edge, **no OOS spent**. Built `discovery/tools/propfirm_feasibility.py` (BUILT), which reuses arc 1033's
validated `_build_4way_contiguous` (canonical A1 + `MultiPairBacktester`, the same configs as
`scripts/cosim_validation/validate_4way_book.py`) and `compute_risk_profile`, and adds:

1. **Sharpe / Sortino** — annualized from business-day-resampled returns of the contiguous net-equity curve
   (deployment-relevant: a continuous curve is what a prop-firm judges), plus the **annual-fold Sharpe** straight
   from the per-year book ROIs (reconciles arc 1023: t = Sharpe·√n).
2. **Prop-firm-challenge feasibility theorem.** Arc 1024 proved ROI and max-DD both scale **linearly** with
   per-trade risk (Calmar invariant). So leverage `f` maps (ann_ret, maxDD) → (f·ann_ret, f·maxDD). Reaching a
   profit target `P` over `T` years needs `f = P/(ann_ret·T)`; the resulting max-DD is `f·maxDD = P/(Calmar·T)`.
   The challenge max-DD limit `D` is cleared only if **`T ≥ (P/D)/Calmar`** — the feasibility horizon, *at* the
   DD limit (zero margin). Parameter-robust: the binding quantity is the ratio `P/D` against the book's Calmar,
   so the exact published numbers barely matter. (Conservative: uses the *historical* maxDD as the forward maxDD —
   a lower bound; forward paths can only be worse → reinforces the verdict.)

Reproduction cross-check passed (max |mine − arc-1020 record| = 1.493 pp = the documented fbr native-trail
imprecision; verdict-invariant). Reproduce: `PYTHONPATH=. py discovery/tools/propfirm_feasibility.py`.

## Results

**Risk-adjusted metrics (contiguous 2011–2020 IS curve):**

| weighting | cap | ret %/yr | maxDD % | Calmar | Sharpe (daily,ann) | Sortino | vol %/yr |
|---|---|---|---|---|---|---|---|
| risk-parity | OFF (bound) | +0.577 | 1.592 | **0.363** | **+0.480** | +0.494 | 0.835 |
| risk-parity | ON (faithful) | +0.362 | 1.533 | **0.236** | **+0.381** | +0.370 | 0.660 |
| equal | OFF (bound) | +0.899 | 4.521 | 0.199 | +0.446 | +0.498 | 1.408 |
| equal | ON (faithful) | +0.578 | 3.934 | 0.147 | +0.325 | +0.337 | 1.248 |

Annual-fold Sharpe (RP cap-OFF): per-year ROI mean +0.603% / sd 0.899% → **Sharpe 0.67**, implied t = 2.12
(reconciles arc 1023's t=2.66 — the small gap is the fbr-trail reproduction imprecision + calendar-year slicing;
same ballpark, verdict-invariant).

**Finding 1 — the book's risk is DRAWDOWN-SHAPED, not volatility-shaped.**
Sharpe ~0.4–0.7 (vol-adjusted: modest but real) coexists with Calmar 0.24–0.36 (DD-adjusted: poor). **Calmar ≪
Sharpe** is the diagnostic signature of *persistent, serially-correlated* drawdowns — the bad folds (2015, 2016,
2018) chain into one ~4.7yr underwater stretch (arc 1033), so daily volatility is low but the drawdown is deep-in-
*time*. A vol-gated allocator (e.g. a multi-strat fund sleeve, Sharpe-judged) sees an acceptable small sleeve; a
DD-gated vehicle (prop-firm, max-DD-judged) sees an unpassable hurdle. **The deployment verdict depends on which
risk metric the vehicle uses.**

**Finding 2 — the book CANNOT pass a prop-firm CHALLENGE in any realistic timeframe.** `T_min = (P/D)/Calmar`
years operating *at* the max-DD limit; safe operation needs ~2× (so a within-noise bad fold doesn't breach):

| challenge (representative published terms) | P/D | T_min (RP cap-OFF / cap-ON) | safe ~2× |
|---|---|---|---|
| FundedNext Stellar 2-step P1 (8% / maxDD 10%) | 0.80 | 2.2 / 3.4 yr | 4.4–6.8 yr |
| FundedNext Stellar 2-step P2 (5% / maxDD 10%) | 0.50 | 1.4 / 2.1 yr | 2.8–4.2 yr |
| FundedNext Stellar 1-step (10% / maxDD 6%) | 1.67 | 4.6 / 7.1 yr | 9.2–14.1 yr |
| 5ers Hyper-growth rep. (8% / maxDD 5%) | 1.60 | 4.4 / 6.8 yr | 8.8–13.6 yr |

Even the **most lenient** real config (5% target / 10% maxDD) needs **1.4–2.5 yr at the DD limit**; the standard
8%/10% needs **2.2–4.0 yr**; the stricter 1-step / 5ers **4.4–8.4 yr**. Prop-firm challenges expect passing in
**weeks to a few months**. This is **off by years — not a near-miss.** And at `T_min` the account sits exactly at
the max-DD limit (zero margin): since the book trips a within-noise negative fold *every year* (precisely why it
fails AFP), it would near-certainly breach the limit during the multi-year wait — so even unlimited-time
challenges are infeasible. The **daily-DD cap is never the binding constraint** (worst single-day DD 0.17–0.54%;
× the needed leverage stays well under the 3–5% daily limit) — **max-DD is the sole wall**, and Calmar's
risk-invariance (arc 1024) means leverage cannot move it.

## Verdict & disposition

**DIAGNOSTIC → KILL** (no new component; all 4 components UNCHANGED, PORTFOLIO). The fresh-eyes edge check
confirmed exhaustion (commodity angle data-gated by *verification*; every novel mechanism reason-killable), so
no edge was run. The deployment characterization is decisive:

- **Prop-firm CHALLENGE (the operator's nominal route to new/scaled capital): INFEASIBLE** — Calmar 0.24–0.36 is
  structurally far below every real challenge's `P/D` hurdle (0.5–1.67); passing needs years at the DD limit.
- **Already-funded capital, run at low risk as a slow diversifier (the operator's dormant funded accounts):
  marginally viable** — ~0.4–0.9%/yr at maxDD 1.5–4.5%, Sharpe ~0.4–0.7, but ~98% time-underwater and a ~4.7yr
  deepest drawdown. An endurance / opportunity-cost / risk-appetite question — **the operator's call** (discovery
  characterizes, never deploys, §11).

This **sharpens the whole portfolio-route conclusion**: there are TWO separate deployment walls, not one. Arc
1032 quantified the **gate wall** (path-A: the AFP gate must be relaxed to a mean/CI gate or the book never
"passes"). This arc quantifies the **vehicle wall** (even if path-A is adopted and the book "passes" the academic
gate, it is undeployable via a prop-firm challenge — its risk is drawdown-shaped, Calmar too low). The vehicle
wall is the more fundamental: it does not depend on the gate philosophy at all.

## NEW lesson

**A thin mean-positive book can clear a vol-adjusted bar (Sharpe ~0.5, t=2.66) yet be undeployable on a
DD-gated vehicle (prop-firm challenge), because its risk is drawdown-SHAPED (Calmar ≪ Sharpe; persistent
multi-year underwater), and Calmar's risk-invariance (arc 1024) means leverage cannot rescue it.** Deployment
feasibility is **vehicle-metric-dependent** — always test the actual deployment vehicle's risk metric (DD-gate
vs vol-gate), not just the academic gate. For this corpus: the binding deployment constraint is **Calmar (the
~4.7yr drawdown chain), not the all-folds-positive gate**; a future component that would make the book deployable
on a prop-firm challenge must lift the book's *Calmar* (shorten/shallow the drawdown chain), not merely add
positive mean or decorrelation — i.e. it must be positive *during* the 2015–2018 underwater stretch (the same
+2015/+2018 spec the leg-hunt already proved unreachable, now re-derived from the deployment side).

**Operator-facing bottom line:** the autonomous programme's deployment dossier on the 4-way book is now complete
on every axis — noise (2016/2017/1023), ENB (2019), time (2021), cost (3022), gate-resolution (1032), raw risk
geometry (1033), and now **risk-adjusted quality + deployment-vehicle feasibility (this arc)**. The book is a
genuine ~3-bet, mean-positive, cost-/time-robust PORTFOLIO that (a) fails the academic AFP gate only below its
noise floor and (b) is infeasible on a prop-firm challenge by a Calmar shortfall. Both remaining levers are the
operator's: relax the gate AND accept a low-Calmar slow-grind on already-funded capital, or shelve the book.
There is nothing further the autonomous edge-hunt can add.

## Tooling

- **BUILT:** `discovery/tools/propfirm_feasibility.py` — Sharpe/Sortino reader (`compute_sharpe`) + annual-fold
  Sharpe (`annual_sharpe_from_fold_rois`) + the challenge feasibility theorem (`feasibility_horizon_years`) over
  the canonical contiguous curve. Reuses arc 1033's `_build_4way_contiguous` + `compute_risk_profile`. Registered
  in `TOOL_REGISTRY.md`.
- **Canonical:** reproduction via `_build_4way_contiguous` (A1 + `MultiPairBacktester`, configs identical to the
  committed item-E validator). No canonical change; no FLAG.
- No council (a measurement resolving a deployment characterization; §7 council junctures don't apply). No OOS.
