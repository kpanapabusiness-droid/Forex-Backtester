# arc 2058 — Turn-of-quarter USD funding-squeeze (Basel/G-SIB snapshot) on spot USD majors

**Chat:** 2000s · **Date:** 2026-06-06 · **Verdict:** KILL (honest-engine §5f; real IS timing effect that
beats null but R≈0 under take-the-loss + frozen-exit OOS collapses + correlated with me_short) · **Council:** none
(not a survivor; §5f resolved it on the engine).

## Step (a) — log read / synthesis (what's live, what's dead)
Read DISCOVERY_PROTOCOL, LESSONS, TOOL_REGISTRY, DISCOVERY_LOG (both tiers) at resume. State of the corpus:
- **4 PORTFOLIO components** exist (gap-fill 1006, me_long 1011, fbr 1013, me_short 1019); **no PASS**. The
  combined book is provably NOT all-folds-positive — blocked by **2015 & 2018** (strong-USD years where every
  reversion edge bleeds simultaneously; 0/5151 convex weightings pass, arcs 1015/2008/3009).
- The honest OOS deploy object collapses to **me_long-solo** (1046), vehicle-infeasible (prop-firm T_min years).
- The strategist **explore-now MENU (M1/O1/L1/Q1/G1) is EXHAUSTIVELY closed** (1027/2023, 1029/1025/1055, 1054,
  1028, 2052); the deploy-object vehicle matrix is **complete** (2053–2056); the 1000s chat gracefully handed off
  (1055) declaring the OHLC-only EDGE frontier mined out.
- **Calendar-flow failure-mode series** sharpened: (i) sub-cost (gotobi 2025, WMR-fix 3008), (ii) priced-in /
  front-run (JP fiscal-YE 2026), (iii) instrument-neutral (IMM futures roll 2057). Survivor DNA (arc 2057):
  *a forced flow must displace the **actual traded spot instrument** directly*.
- Frontier per dispatch: highest-EV is a regime-orthogonal **2015/2018-positive** component; ~22 routes dead.

## Step (b) — idea + *because*
**Turn-of-quarter USD funding squeeze.** Basel III leverage-ratio / G-SIB reporting is a balance-sheet
**snapshot** on the quarter-end date → dealers demand USD on balance sheet **at** the reporting date → a spot
USD bid into the turn that snaps back on the first business day of the new quarter (the documented
cross-currency-basis turn-of-quarter / turn-of-year spike, strongest post-2014 = our late IS window).
Chosen because it is the one untested documented calendar flow that is **(1) directional-USD** (the 2015/2018
direction the route lacks), **(2) a hard-deadline snapshot that cannot be front-run** (distinct from the
priced-in fiscal-YE repatriation, arc 2026), and **(3) a spot balance-sheet position** (distinct from the
instrument-neutral IMM futures roll, arc 2057). Distinct from `me` (subset = quarter-ends; direction =
USD-factor, not generic month-end magnitude reversion).

## Step (c)/(d) — observation (cheap characterization, D1, 7 USD majors, IS 2010-2020)
`_disco2000_work/arc2058_turn_of_quarter_usd.py`. usd_long = long-USD per-pair return; into = 5d cum into the
quarter-end day, fwd = 3d cum after; /Wilder(14)-ATR-frac shift1. **The hypothesis INVERTED but a real
quarter-end-specific effect appeared:**
- **into** usd_long **−0.255** ATR (USD *weakens* into the turn, not strengthens) vs control other-month-ends
  −0.038, all-days +0.025. Year-end especially: into −0.792.
- **fwd** usd_long **+0.296** ATR (median **+0.251**, frac+ **0.601**) — USD **strengthens after** the turn —
  vs control other-month-ends **−0.016** / all-days +0.016. So a **quarter-end-specific post-turn USD-strength**
  (excess ≈ +0.31 ATR over control), median≈mean (not a thin tail), **clears the ~0.085 ATR D1 cost gross**.
- Carried by the **European/commodity XXXUSD** majors (EUR +0.44 / GBP +0.56 / AUD +0.35 / NZD +0.18 fwd usd_long
  → those pairs *fall* after the turn); USDJPY weak (+0.09), **USDCAD negative (−0.08)** = the recurring
  USD-quote-beta pair-mix tell.
- **Per-year: 2015 fwd mean −0.038 / median −0.282 = NEGATIVE** (the strong-USD year the route needs FAILS —
  in 2015 USD strengthened into the turn, into +0.572, and did not revert); 2018 +0.18 OK but is
  USD-strength-continuation (into +0.648), not the reversion. 2011/2019 also negative.

Non-coin-flip (frac+ 0.601, gross drift >> cost, quarter-end-specific vs control) → **§5d does NOT cheap-kill;
§5f REQUIRES the honest engine under the exit menu** before any verdict (the gross drift could be gutted by the
currency-cap/cost — the arc-1017 lesson).

## Step (f)/(g) — honest engine §5f (the clean short leg: 4 XXXUSD majors, uniform SHORT after quarter-end)
Built `discovery/tools/quarter_end_signals.py::QuarterEndUsdReversionShortSignal` (BUILT, registered); engine
driver `_disco2000_work/arc2058_engine_wfo.py` (canonical Panel / build_arc_pool / ArcFoldRunner /
build_v3_folds / build_oos_year_folds / discovery_measure — nothing reimplemented).
- **Pool n=168, mean `final_r` +0.012R ≈ 0, win 0.202.** The +0.296 ATR gross drift does **NOT** survive
  take-the-loss — the post-turn reversion is too slow/shallow relative to the 2·ATR stop (low win-rate; a
  **mean-driven displacement, not a capture edge**).
- **IS §5f exit menu — EXIT-FRAGILE, none all-folds-positive:** only `sl_only` (+1.06%) and
  `sl_plus_trailing_swing` (+1.27%) positive-mean; the other 4 exits **negative** (tp2r −0.35 / tp3r −0.41 /
  trail_atr −0.52 / partial-runner −0.49); best 5/10 folds negative. **The IS positivity rests on ONE fat-tail
  fold** (fold 5 +13.88%, n=4) plus a 0-trade artifact fold — **ex-fold-5 IS mean is negative (~−0.13%)**.
- **NULL:** REAL +1.27% vs fair same-side random-short −0.36% → **excess +1.62pp** (a *real* IS timing effect —
  quarter-end short beats random short).
- **OOS (frozen best IS exit `trailing_swing`, measure-once):** **−1.054%, 5/6 negative, worst −3.40%** (only
  2024 +5.58%, n=2 = another thin fat tail). **The IS edge collapses out-of-sample** — the §5f exit-fishing trap
  made concrete (cf. fbr's fished trailing_swing dying OOS, arc 2040).
- **Correlation vs me_short (1019): +0.188** per-fold IS — the **same month-end-reversion family**, not
  decorrelated; me_short is the more robust version (7/10 IS folds+, far smaller swings).

## Verdict — KILL (§11: beats-null-but-net-negative)
A *real* quarter-end timing effect (beats the fair null +1.62pp IS) but **not portfolio-worthy and not a PASS**:
the gross drift is a **fat-tail mirage under take-the-loss** (pool R≈0, win 0.20), the IS mean is carried by a
**single fold**, the **frozen-exit OOS is net-negative** (−1.05%, 5/6 neg), and it is **correlated (+0.19) with
the more-robust me_short** so it adds nothing decorrelated. Per §11, real-but-net-negative = **KILL**, not
PORTFOLIO. Components UNCHANGED (all 4 PORTFOLIO); deploy object UNCHANGED (me_long-solo, 1046); lever = operator
path-A. OOS measured-once as characterization (1046/2055 precedent), not tuned.

## NEW lesson
**A gross forward-drift ATR that clears cost is NOT a tradeable edge when win-rate is low — it is a
*mean-driven* displacement, not a *capture* edge, and take-the-loss collapses it to R≈0.** Quarter-end carries a
real post-turn USD-strength drift (+0.30 ATR, beats null IS) but with win 0.20 the 2·ATR stop eats it; the IS
"mean-positive" is single-fat-tail-fold + exit-fishing that dies OOS. Sharpens the corpus's capture-vs-drift
distinction (the survivors gap/me/fbr all clear cost **on capture/win-rate**, not merely on mean drift) and adds
a **FOURTH** calendar-flow failure mode to the 2057 series: **(iv) real-but-fat-tail-mirage** — a documented
spot-displacing calendar flow whose drift is real and beats null yet evaporates under take-the-loss + frozen-exit
OOS (joins sub-cost / priced-in / instrument-neutral). Also: a quarter-end restriction of the me-reversion family
inherits me's 2015-fragility (corr +0.19) and is *thinner* → no decorrelated gain. Closes the turn-of-quarter /
Basel-snapshot USD-funding named gap.

## Tooling / FLAGS
- **BUILT (registered):** `discovery/tools/quarter_end_signals.py::QuarterEndUsdReversionShortSignal` — reusable
  quarter-end (or year-end-only) directional signal, mask + ATR + `Direction.SHORT`, canonical-scored.
- Drivers: `_disco2000_work/arc2058_turn_of_quarter_usd.py` (obs), `arc2058_engine_wfo.py` (engine §5f).
- **Soft note (not a FLAG):** the IS exit menu showed one fold with n=0 trades but +5.77% ROI
  (`trailing_swing`, fold 6) — an open-runner / per-year-reset book-keeping artifact in the canonical
  per-fold reporting; it inflates the IS mean but does not change the KILL (OOS-negative regardless). Worth a
  glance by a future arc that depends on per-fold n vs ROI consistency; not patched (canonical, human-gated).
- No canonical change, no council, OOS holdout untouched (measure-once).
