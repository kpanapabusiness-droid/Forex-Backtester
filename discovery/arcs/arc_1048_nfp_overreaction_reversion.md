# arc 1048 — NFP-day (first-Friday) overreaction REVERSION

**Chat:** 1000s | **Range:** 1000-1999 | **Date:** 2026-06-06
**Disposition: KILL** (obs cheap-kill — falsified: NFP move is priced efficiently, corr≈0, no fade, not 2018)

---

## (a) Log read / synthesis

(Carried from arc 1047, same chat — log re-read at 1047 step a; main pulled, no STOP.) The honest-exit
deploy thread (1042→1046, 2040→2046) is closed; the deploy object collapses to me_long-solo, not AFP,
vehicle-infeasible. The unfound **2018/2014 regime-orthogonal component** is the only path to an AFP book
(~18-19 dead routes incl. arc 1047 session-flow). Closed ground: all shallow directional (long+short,
H1/H4/D1/W1), pure-FX instrument universe, cost-side levers, intraday flow (sub-cost, re-confirmed 1047).
Survivor DNA = **LARGE forced dislocation → reversion clearing cost** (swing scale; intraday is sub-cost).

## (b) Idea (observe, don't guess)

Apply the survivor template to the **highest-impact scheduled US event**, conditioned PURELY on the
calendar (no external data): **NFP-day = first Friday of each month → overreaction REVERSION.** *Because*:
the algo knee-jerk to the payroll surprise overshoots; real-money fades it (the classic "fade the NFP
spike"). Candidate **regime-orthogonal (2018-positive)** leg: NFP surprises are noise not trend, so the
fade could work in the strong-USD/risk-off years the reversion book bleeds. Distinct from the KH-era
FOMC-proximity filter (untrustworthy, different event) and from unscheduled vol-spike reversion (3012).

## (c)/(d) Characterization + cheap kill (OBSERVATION ONLY — no engine/P&L)

Driver `_disco_work/arc1048_nfp_reversion.py`. D1, 7 USD majors (cached), 2010-2026. **Measurement-integrity
fix:** under `5ers_eet` D1 bars are timestamped at the START of the EET trading day (~22:00/21:00 UTC of the
prior calendar day), so the Friday-SESSION bar carries a THURSDAY timestamp — a naive `weekday==4` mask
selects the empty Saturday session. Corrected to **session date = timestamp + 1 day** (DST-robust), verified
NFP n=4053 (~12/yr × 16yr × 7 pairs). Method: NFP-day net move (close−open)/ATR; FADE capture =
−sign(move)·forward k-day drift in ATR (reversion = positive); controls = non-first Fridays (FRI) + all days
(ALL) for specificity (arc-1011 method); per-year + 2018 sign; vs the ~0.05-0.10 ATR/R cost hurdle.

**Finding 1 — NO reversion and NO continuation (corr ≈ 0).** corr(NFP move, forward k-day drift):
k=1d −0.008, k=2d −0.010, k=3d −0.007 (n1351) — essentially zero. FRI control similar (−0.02 to −0.04),
ALL −0.005. The NFP-day move has **no predictive relationship** with subsequent drift: the surprise is
priced efficiently at the daily scale, so there is nothing to fade. (A coin-flip base.)

**Finding 2 — the tiny small-threshold fade is below cost AND not NFP-specific.** FADE capture at thr≥0.5:
NFP +0.047 ATR (k=2d) vs FRI +0.039 vs ALL −0.000 — NFP ≈ FRI (no NFP specificity), and +0.047 < the
~0.05-0.10 ATR cost hurdle. Bigger NFP moves **CONTINUE** (thr≥1.0 fade −0.025 to −0.065; thr≥1.5 noisy) —
consistent with NFP often *starting* a Fed-repricing move, not overshooting.

**Finding 3 — NOT the 2018 leg, unstable.** Per-year NFP-fade (thr≥1.0, k=2d): positive **5/17** years;
**2018 −0.255 and 2015 −0.196 both negative**; 2014 +0.923 is a thin-tail outlier (n13). Fails the
regime-orthogonality test for the same reason as every other route.

## (e)–(h) Diagnose / validate / council

Not reached — §5d cheap-kill. The base is a **coin-flip** (corr≈0, no gross drift either way, no
NFP-specificity), so the §5f nested-exit step does not bite (it bites only for a non-coin-flip entry that
beats the null OR shows gross drift — this shows neither). No engine, null, or council spent (3003 lesson
#2). OOS never touched.

## (i) Verdict

**KILL (obs cheap-kill).** The NFP-day overreaction-reversion hypothesis is FALSIFIED: at the daily/swing
scale the NFP move has corr≈0 with forward drift (priced efficiently — information, not a faddable
overshoot), the tiny small-threshold fade is sub-cost and not NFP-specific, bigger moves continue, and it
is not 2018-positive (2018 & 2015 negative, 5/17). No new component; no canonical change.

**NEW lesson.** The highest-impact scheduled US macro release (NFP) shows **NO daily-scale
overreaction-reversion** on liquid USD majors — the payroll surprise is priced efficiently (corr≈0 with
forward drift, no specificity vs generic Fridays), so the "fade the spike" edge does not survive to the
swing horizon (and the intraday knee-jerk fade, if any, is sub-cost per 1047). Closes the
**scheduled-macro-event overreaction-reversion** lane: scheduled high-impact releases are information, not
a faddable dislocation, at the swing scale — unlike the survivors' MECHANICAL forced flows (weekend
rebalancing, WMR-fix, stop-runs), a *news* surprise carries genuine information so it does not
systematically revert.

## Tooling

No new BUILT tool (pure observation; reused `Panel.from_pairs` + `wilder_atr`). Driver
`_disco_work/arc1048_nfp_reversion.py`. No canonical change, no council, OOS untouched.

## Threads / handoff

The 2018/2014 regime-orthogonal component remains unfound (~19-20 dead routes). The distinction sharpening:
the survivors exploit **mechanical** forced flow (rebalancing, fix, stop-runs) that reverts because it
carries no information; **information** events (macro releases) are priced efficiently and do not revert.
The remaining edge lane is a genuinely novel **mechanical/structural** dislocation not yet tested — the
news/event-overreaction angle is now closed at the swing scale.
