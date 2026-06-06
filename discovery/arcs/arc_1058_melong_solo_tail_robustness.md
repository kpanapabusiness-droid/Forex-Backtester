# arc 1058 — TAIL-ROBUSTNESS of me_long-SOLO (the operator's honest OOS deploy object)

**Chat:** 1000s · **Range:** 1000–1999 · **Disposition:** **DIAGNOSTIC → KILL** (no new component;
components UNCHANGED, all 4 PORTFOLIO) · **Council:** none (a measurement resolving a measurement —
1056/1057 precedent) · **OOS:** measure-once characterization of an ALREADY-SPENT series (1046/1053/2055
read it; a tail-robustness re-read adds NO new selection, §4) — not tuned.

> **Headline.** Arc 1057 showed the honest **2-leg** book's borderline significance is fbr-runner-tail-
> fragile and left the fork: is the operator's actual single-leg deploy object — **me_long-SOLO** (1046/
> 1053) — tail-robust, or also a few-trade artifact? Answer: **also tail-carried, and never even
> significant to begin with.** me_long-solo's mean is **non-significant at baseline in BOTH windows**
> (IS cluster t=1.05 / 10-fold, 0.75 / 8-fold; OOS t=1.66, CI spans zero) and is carried by ~5 trades
> (top-5 = **139% of net P&L IS / 114% OOS**; removing the top-5 turns the mean **negative** in both
> windows; the OOS mean is specifically a **2025** story — 3 of its top-5 positions). So the appeal of
> me_long-solo in arc 1046 ("OOS mean doesn't flip negative," unlike the collapsing book legs) does NOT
> mean tail-robust or certifiable. **This resolves arc 1057's fork on the UNIFORM branch: NO deploy
> corner has a certifiable, tail-robust mean** — solo (this arc, non-sig + tail-carried), 2-leg (1057,
> borderline-sig but fbr-tail-fragile), 4-way (1043, non-sig). The corpus's reversion edge is uniformly
> too thin to certify a mean at any leg-count. Components UNCHANGED; deployable-system count = 0.

---

## 1. Read + synthesize (step a)

Resumed after arc 1057 (committed/pushed b1eeaf8; pulled main — 2000s added 2060-2062, clean union merge).
No `discovery/STOP`. State unchanged: edge frontier exhaustively closed, §11 verification complete, the
sole live lever is the operator's path-A gate-governance call, the only within-charter value is
decision-support on the characterized deploy object. Arc 1057 quantified the 2-leg book's tail-dependence
(fbr-runner-fragile) and explicitly raised the fork this arc closes.

**The fork.** Arcs 1046/1053 established that under honest frozen §5f exits the book collapses to
**me_long-SOLO** out-of-sample (fbr mean-negative OOS, gap & me_short honest-negative IS+OOS per arc 2044)
→ me_long-solo (committed `sl_only`/SL2.0/2-bar time-exit, D1 USD majors) is the REAL honest deploy object,
the one the operator would actually run. Arc 1057 showed the 2-leg book's mean is fbr-runner-tail-fragile.
The unanswered, decision-relevant question: **is me_long-solo's OWN mean tail-robust, or also a few-trade
artifact?** me_long is structurally the least tail-carried leg (high-win-rate short-horizon fade, `sl_only`
caps each gain — no convex runner; 1057 already showed its $ tail is tiny, max $580 vs fbr $2,513), so if
ANY corner has a tail-robust mean it is this one.

## 2. Idea (step b)

Apply arc 1057's tail-robustness battery (tail concentration + winsorize + leave-top-N + cluster bootstrap)
to me_long-SOLO on BOTH windows: IS (the 10 v3 folds; me_long's exit is fixed so all are valid — also the
8-fold 2013-2020 set for apples-to-apples with 1057) and the **already-spent** OOS per-year series
(measure-once characterization, 2055/2056/2059 precedent — reading tail-robustness of a frozen series adds
no selection). No council (a measurement resolving a measurement).

## 3. Method (steps c–g) — `_disco1_work/arc1058_melong_solo_tail_robustness.py`

CALLS canonical (A1→MultiPairBacktester, FundedNext, risk 0.005); experiment side = winsorization +
cluster-bootstrap arithmetic on per-position NET P&L only (never realizes P&L, never touches the gate,
never SELECTS on OOS). me_long config = the committed honest cell (`MonthEndReversionLongSignal(1.0,2)`,
`sl_only`, SL2.0, 2-bar time-exit on the signal — the all-folds-selected cell in 1053/1056). Per fold:
per-position net = gross − `apply_cost_model` cost, realized in the fold's OOS window, over OOS-start
equity. **Anchor:** per-fold gate ROI reproduces arc 1056/1053 EXACTLY — IS 2013-2020
[0.96,−0.23,−1.14,−0.51,0.34,0.9,1.16,0.15] (8-fold mean +0.204%); OOS 2021-26
[−0.63,+0.55,+0.18,+0.54,+2.26,+0.02]. (The Σnet/denom bootstrap base carries the known ≤0.02pp/fold
Σnet-vs-gate-roi residual from 1056 — immaterial; IS 8-fold base +0.189% vs gate +0.204%.)

## 4. Results

| window | n_pos | top-1 | top-3 | top-5 | baseline mean | baseline t | drop-top-5 mean |
|---|---|---|---|---|---|---|---|
| **IS 10-fold** 2011-2020 | 98 | 32.8% | 86.9% | **139.2%** | +0.213%/yr | **+1.05 (~0)** | **−0.084%** |
| IS 8-fold 2013-2020 | 75 | 38.3% | 111.7% | 171.7% | +0.189%/yr | +0.75 (~0) | −0.136% |
| **OOS** per-year 2021+ | 52 | 40.4% | 80.1% | **113.8%** | +0.511%/yr | **+1.66 (~0, CI spans 0)** | **−0.071%** |

**Winsorize** (IS): t barely moves (q95 t=0.98 / 10-fold) — me_long's shallow `sl_only` tail caps gains, so
capping does little; the mean is non-sig with or without. **Leave-top-N is decisive:** IS drop-top-1
(2011 $699) → t=0.68; drop-top-5 → mean **−0.084%**, t=−0.52. **OOS** winsorize is non-monotone on t
(1.66→1.73 at q95 — capping the 2025 fat tail shrinks the bootstrap SE slightly faster than the mean, but
the CI lower bound stays below 0 throughout → never significant); **leave-top-N** decisive: drop-top-1
(2025 $1,237) → t=1.65; drop-top-2 → t=1.12; drop-top-5 → mean **−0.071%**, t=−0.35. **3 of the OOS top-5
are 2025** ($1,237 + $662 + $485) — the OOS positive mean IS the 2025 fold (arc 2059's "me_long-2025 tail").

## 5. Verdict + what it means

**DIAGNOSTIC → KILL** (no new component). Arc 1057's fork resolves on the **uniform** branch:

1. **me_long-solo was never significant.** Unlike the 2-leg book (which fbr + RP weighting lifts to
   borderline SIG+ t=2.27, 1057), me_long-SOLO's mean is non-significant on its own in BOTH windows even at
   baseline (IS t≤1.05, OOS t=1.66 with CI spanning zero). The thing that gave the book its borderline
   significance was fbr — exactly the tail-fragile leg.
2. **It is ALSO tail-carried**, just shallower. top-5 = 139% (IS) / 114% (OOS) of net P&L; removing the
   top-5 turns the mean negative in both windows. The `sl_only`/2-bar fade has no convex runner, so its tail
   is shallow (max $699 IS / $1,237 OOS), but the mean is structurally the same few-trade artifact.
3. **"OOS-mean-robust" (1046) ≠ tail-robust / certifiable.** me_long-solo's appeal in 1046 was that its OOS
   mean does not flip negative (unlike gap/fbr/me_short under honest §5f). True — but that is a *sign*
   property, not a *certifiability* property: the OOS mean is positive yet non-significant and carried by a
   single year (2025).

**Net — the deployment picture is now UNIFORM across every corner.** No deploy object offers a tail-robust,
certifiable mean: solo (non-sig + tail-carried, this arc), 2-leg {me_long,fbr} (borderline-sig but
fbr-tail-fragile, 1057), 4-way (non-sig, 1043). Combined with the AFP-gate wall (1032) and the vehicle/
Calmar wall (2033/2059), the path-A "significant mean-positive" pillar is shown NOT to robustly exist at
ANY leg-count or window. The corpus's reversion edge is real (mechanisms §11-verified honest) but uniformly
too thin / tail-carried to certify a mean. The lever remains the operator's governance call, now with the
honest, complete input that **no corner is statistically certifiable**. Components UNCHANGED (all 4
PORTFOLIO); deployable-system count = 0.

## 6. Threads / lessons

- **NEW reusable lesson:** the OOS-mean-positive single-leg fallback (me_long-solo, the 1046 honest deploy
  object) is itself NON-significant AND tail-carried in both IS and OOS — "OOS-mean doesn't flip negative"
  is a SIGN property, not certifiability. The fbr-tail-fragility of the 2-leg book (1057) is not specific to
  fbr's convex runner: even the shallow-tailed `sl_only` me_long has its mean carried by ~5 trades (top-5 =
  114-139% of net P&L). The whole corpus's reversion edge is uniformly too thin to certify a mean at any
  leg-count → the tail-robustness battery (winsorize + leave-top-N + cluster bootstrap) should be run on the
  ACTUAL deploy object, not only the multi-leg book, before any mean-based deploy decision.
- **Completes the deployment-robustness picture** begun by 1057: the three deployment walls (AFP-gate 1032,
  vehicle/Calmar 2033/2059, mean-certifiability 1056/1057/this) are now all mapped across solo / 2-leg /
  4-way × IS / OOS — the autonomous edge-hunt and the autonomous deploy-characterization both have nothing
  further to add; the lever is definitively the operator's path-A governance call.
- **Confirms the 2025-tail observation** (arc 2059) from the solo angle: me_long's OOS positive mean is the
  2025 fold (3 of its top-5 OOS positions), the same fat-tail-fragility on the OOS side that fbr-2024 is on
  the 2-leg book — both deploy corners lean on a single recent fat year.

**Tooling.** No new BUILT tool (winsorization arithmetic on canonical net P&L; reuses the 1057 battery on a
single leg + the canonical `build_oos_year_folds` for the already-spent OOS read). Driver
`discovery/_disco1_work/arc1058_melong_solo_tail_robustness.py`. No canonical change, no FLAG, no council.
