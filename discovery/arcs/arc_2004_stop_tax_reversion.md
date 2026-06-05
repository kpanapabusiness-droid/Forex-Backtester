# Arc 2004 — The Stop-Tax on the FILL/REVERSION family (completing the arc-3004 escalation)

> **Arc id:** 2004 · **Chat:** 2000–2999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **FAIL (KILL).** Arc 3004 lifted the SL-first stop confound for a MOMENTUM/continuation
> entry (real ~4–5pp drag, not the wall). This arc lifts it for the OTHER mechanism family — a REVERSION/fill
> entry (the weekend gap-fill, where the honest i+1 entry buys INTO a −1.1 to −1.3R adverse excursion, so the
> 2·ATR SL-first take-the-loss tax should bite HARDEST). Result: removing the stop **roughly halves the
> drawdown and loss tail** (a *larger* relative effect than the momentum case) and lifts mean fold ROI toward
> zero — **but still no config, any horizon, is all-folds-positive.** The residual edge ≈ cost, regime-
> dependent. **The stop-tax is a real drag for BOTH mechanism families but is NOT the binding wall for either.**
> **FLAG-2 answered** (H4+FundedNext IS hostile to fill/reversion — via the same EDGE≈COST wall, not the stop).
> **Idea source:** log-seeded (arc 2001's explicit FLAG-2 seed + arc 3004's stop-confound method, applied to
> the reversion family it never covered).

Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first). Engine/measurement **called, never
re-rolled**. Both experiment tools **reused** (geometry/timing only): `WeekendGapFillLongSignal` (BUILT arc
2001), `make_time_exit_predicate` (BUILT arc 1005). **OOS (2021+) deliberately NOT touched** — see §(g).

---

## (a) Log read + synthesis — FRESH EYES

Pulled `origin/main` (`8114d70`). Honest-era corpus = 18 arcs across 3 chats (0; 1000–1007; 2000–2003;
3000–3005), all FAIL. State:

- **Directional long is comprehensively dead** — momentum/breakout/reversion/trend, H1/H4/D1, 28 pairs, both
  the +1R-capture and the forward-drift lens, every exit/SL-multiple, **and with the stop removed** (arc 3004,
  for a momentum entry), regime detection (3 measures), volume, calendar, convexity, triangulation. Forward
  drift ≈ cost everywhere (best conditioned cell ~+0.023R gross vs a ~0.05–0.10R hurdle).
- **One PORTFOLIO edge:** arc 1006 weekend gap-down-fill long on **JPY crosses** (mean +0.69% IS, beats
  random, but fold-fragile → not all-folds-positive). The 1000s chat owns the gap-portfolio thread.
- **Arc 3004 ESCALATION:** the apparatus can only bet direction (coin-flip on liquid FX) → needs a structural
  unlock (2nd leg / shorts / tighter-cost regime). Shorts code merged (PR #273) but **the immediate lane stays
  long-only PORTFOLIO until the operator flips to shorts** (CLAUDE.md) → I stay pre-shorts.
- **2000s lane coordination (arc 2002):** avoid the majors-gap-fill-as-candidate (done, arc 2001) and the
  1000s gap-portfolio thread; **open: FLAG-2 diagnostic** — *is H4+FundedNext generically hostile to
  fill/reversion entries?*

**The gap I picked.** Arc 3004's stop-confound test was run on ONE entry: the post-up-spike **momentum**
continuation (the strongest +gross-drift cell). It found the 2·ATR SL-first stop is a real ~4–5pp drag but not
the wall. **It never tested a REVERSION/fill entry** — and that is exactly where the take-the-loss tax should
bite hardest: a fill entry buys into weakness, so it sits in large adverse excursion (arc 2001 measured the
majors gap-fill MAE at −1.1 to −1.3R, P(MAE≤−0.5R)=0.77–0.89) → a 2·ATR SL converts a large fraction of
eventual-reverters into −1R losers. So arc 3004's "stop is a drag but not the wall" conclusion was only
established for the momentum/continuation family; the reversion/fill family was an open confound. Closing it
(a) completes the escalation's coverage and (b) directly answers arc 2001's FLAG-2.

## (b) Idea — log-seeded, no council

This is a focused, confirmatory **A/B diagnostic** on an already-characterized signal (arc 2001's majors
gap-fill) — not a stuck-point idea-fork and not a survivor, so no council (cf. arcs 2002 / 3005, confirmatory
cheap-kills). Hypothesis with a *because*: **the SL-first take-the-loss tax is a (large) drag on a reversion
entry — larger than on a momentum entry — because reversion buys into adverse excursion; test whether removing
it rescues the gap-fill into all-folds-positive.** Universe = **majors** (the 2000s-owned arc-2001 result;
keeps this off the 1000s JPY-cross portfolio component — this is a family diagnostic, not a portfolio build).

## (c) Characterize

Reused arc 2001's characterization (not re-derived): big down-gaps (gap ≤ −1·ATR) fill, but the honest i+1
entry lands inside a large adverse continuation (median MAE −1.1 to −1.3R). Pool sanity this arc: **n=176
big-gap trades** over IS 2010–2020 (matches arc 2001's ~177 ✓), `WeekendGapFillLongSignal(threshold_atr=1.0)`,
8 majors, H4 5ers_eet. Fires/pair 16–40 (USDCAD 16 … AUDUSD 40).

## (d)+(g) The clean stop-tax A/B — full IS WFO (no lucky-3-fold triage; arc 2001/3002 lesson)

Three exit families, full 10-fold IS WFO, FundedNext costs ON, scored only by `MultiPairBacktester`. The
**STOP-PRESENT vs STOP-REMOVED pair shares the identical N-bar time exit**, so their contrast isolates the
SL-first tax (exactly arc 3004's method): "stop removed" = `sl_only` + `sl_atr_mult=6.0` (wide, sizing-only) +
`make_time_exit_predicate(n_bars=N)`; "stop present" = the same time exit with `sl_atr_mult=2.0`. Baseline =
the all-arc partial/runner geometry.

| config | per-fold ROI% (2011…2020) | worst | mean | maxDD | neg | AFP |
|---|---|---|---|---|---|---|
| BASELINE partial/runner sl2.0 | −2.82/+2.80/+0.56/−3.12/+2.88/+2.78/−2.08/−0.14/+0.45/−3.04 | −3.12% | −0.17% | 3.89% | 5/10 | **N** |
| STOP-PRESENT sl2.0 + time N=6 | −2.84/+2.83/+0.86/−3.11/+2.19/+0.61/−2.87/−2.33/+1.69/−2.04 | −3.11% | −0.50% | 3.76% | 5/10 | N |
| **STOP-REMOVED sl6.0 + time N=6** | −1.12/+0.88/+0.28/−0.30/+0.86/+0.22/−0.46/−0.66/+0.59/−0.95 | **−1.12%** | −0.07% | **1.56%** | 5/10 | N |
| STOP-PRESENT sl2.0 + time N=12 | −1.77/+0.72/−1.41/−4.90/+2.89/+0.13/−2.32/−2.58/+0.63/−2.18 | −4.90% | −1.08% | 5.13% | 6/10 | N |
| STOP-REMOVED sl6.0 + time N=12 | −1.26/+0.33/−0.03/−0.73/+0.45/+0.57/−0.32/+0.17/+0.63/−1.35 | −1.35% | −0.16% | 1.94% | 5/10 | N |
| STOP-PRESENT sl2.0 + time N=24 | −2.90/+1.07/+2.06/−3.37/+3.38/+2.59/−1.18/−0.87/+0.23/−1.39 | −3.37% | −0.04% | 3.94% | 5/10 | N |
| STOP-REMOVED sl6.0 + time N=24 | −1.92/+0.97/+0.27/+0.16/+0.20/+0.99/−0.08/+0.46/−0.07/−1.42 | −1.92% | −0.04% | 2.22% | **4/10** | N |
| STOP-PRESENT sl2.0 + time N=48 | −2.90/+2.17/+1.50/−2.62/+3.85/+4.31/−1.72/−0.29/+0.23/−1.68 | −2.90% | +0.29% | 4.11% | 5/10 | N |
| STOP-REMOVED sl6.0 + time N=48 | −2.51/+1.76/−0.07/+0.67/−0.44/+0.96/−0.20/+1.35/+0.12/−0.37 | −2.51% | +0.13% | 2.70% | 5/10 | N |

**Reading the two numbers (arc-3004 pre-committed framing):**

1. **STOP isolated (present vs removed, same time exit):** removing the 2·ATR SL-first stop on the reversion
   entry **roughly HALVES drawdown and the loss tail** — maxDD 3.76→1.56% (N=6), 5.13→1.94% (N=12),
   3.94→2.22% (N=24); worst fold −3.11→−1.12% (N=6), −4.90→−1.35% (N=12), −3.37→−1.92% (N=24) — and lifts mean
   toward zero (−0.50→−0.07% N=6; −1.08→−0.16% N=12). This is a **larger relative DD/tail effect than arc
   3004 saw on the momentum entry**, exactly as the MAE −1.1R mechanism predicts (the stop was manufacturing
   −1R losers out of would-be reverters). The take-the-loss tax is real, and it is *worse* for fill entries.
2. **COST/REGIME isolated:** but it is **NOT the binding wall.** **No config — present or removed, any horizon
   — is all-folds-positive.** The best stop-removed config (N=24) is still worst −1.92%, 4/10 neg, mean
   −0.04% (≈ break-even). The only mean-positive readings (stop-present N=48 +0.29%, removed N=48 +0.13%) are
   the cherry-picked best of a sweep — claiming them would be exit-fishing (§5f); the *robust* read across the
   whole exit/SL menu is break-even and regime-dependent (the same +2.8% trending folds vs −2 to −3% chop
   folds, stop or no stop). The residual gross edge after FundedNext cost ≈ 0.

This is arc 3004's "stop-removed but still sub-cost" branch, now established for the reversion family with a
clean, unconfounded proof: the gap-fill on majors is closed not because the stop was killing it, but because
the residual edge ≈ cost and is regime-dependent.

## (g) OOS — PRESERVED PRISTINE

IS is not all-folds-positive for any config → there is no finished system to validate on OOS. Per §4 + the
arc-2001 HEAVY council, **OOS (2021+) was deliberately NOT measured** (measuring an IS-failed system adds
nothing; and the majors big-gap cohort is ~6–8 trades/OOS-year — too thin to be a meaningful all-folds-
positive judge regardless). Arc 2001 preserved this holdout; arc 2004 keeps it preserved.

## Final verdict — FAIL (KILL); disposition = KILL

The weekend-gap-fill long on majors is **not deployable**, and removing the SL-first stop does not change that.
Disposition **KILL** (not PORTFOLIO): the robust, non-exit-fished read is break-even/regime-dependent (≈ the
null line), and §11 is explicit that beating-the-null is necessary-but-not-sufficient — a break-even,
all-folds-failing signal is KILL, not a portfolio component. Arc 2001 already KILLED majors gap-fill; this arc
confirms no exit/SL (including the stop-removed extreme) rescues it, and extracts the deeper diagnostic finding.

## What this arc ADDS (the value — confirmatory + diagnostic)

1. **Completes the arc-3004 escalation coverage.** Arc 3004 proved the SL-first stop is a real ~4–5pp drag but
   not the wall for the **momentum/continuation** family. Arc 2004 proves the SAME for the **reversion/fill**
   family — and that the stop-tax is *larger* there (it ~halves DD, vs ~shrinks-DD for momentum), consistent
   with the −1.1R adverse-excursion mechanism. **For BOTH mechanism families: the stop fattens the loss tail
   (removal fixes that), but it does not manufacture the deficit — the EDGE≈COST wall is unchanged.** The
   escalation now stands on both families with the stop confound lifted on both.
2. **FLAG-2 answered (arc 2001's open seed):** *is H4+FundedNext generically hostile to fill/reversion
   entries?* **Yes — but via the same EDGE≈COST wall as everything else, not via a fill-specific stop tax.**
   The honest i+1 entry's residual edge after costs is ≈ 0; the stop only worsened the variance/DD, it did not
   create the sub-cost result. A cheap diagnostic that retires the "maybe the stop is uniquely punishing
   reversion" hypothesis.

## Threads / what didn't help

- **Closed (clean):** weekend-gap-fill long on majors under the FULL exit/SL menu including stop-removed —
  no config is all-folds-positive; break-even/regime-dependent. The stop-geometry confound is now lifted for
  BOTH mechanism families (3004 momentum, 2004 reversion).
- **Stands:** the arc-3004 escalation (structural unlock needed). The one PORTFOLIO edge remains arc 1006's
  JPY-cross gap-fill (1000s thread); majors are the weaker gap universe (arc 2001) and stop-removal does not
  promote them.
- **Carried FLAGs (unchanged):** FLAG-1 long-only blocks the stronger UP-gap short side (operator/human-gated
  code); `A1Config.time_exit_bars` defined but UNWIRED (arcs 1005/3004 — used the `ExitPredicate` workaround).

## Flags (code NOT merged — human-gated, §9)

- None new. Carries FLAG-1 (long-only blocks short side) and the `A1Config.time_exit_bars`-unwired flag
  (worked around with the BUILT `make_time_exit_predicate`, as arcs 1005/3004 did).
- **Methodological note (for any reader of a 2000s driver):** `FoldStats.roi_pct` / `max_dd_pct` are
  **FRACTIONS** (0.0056 = 0.56%); multiply by 100 for percent. (A first-pass driver printed them raw and looked
  100× too small — corrected before any verdict was read. Not a code flag; a driver-convention note.)

## Reproduction

- **Data:** `histdata_root = C:\Users\panap\histdata_backup`, `cache_root = data/cache`,
  `boundary_convention = "5ers_eet"`, TF H4. **Pairs (8 majors):** EURUSD GBPUSD USDJPY USDCHF AUDUSD USDCAD
  NZDUSD EURGBP.
- **Signal (BUILT, reused):** `discovery/tools/gap_signals.WeekendGapFillLongSignal(threshold_atr=1.0, gap_hours=20.0)`.
- **Exit (test, BUILT, reused):** `discovery/tools/time_exit_predicate.make_time_exit_predicate(..., n_bars=N)`
  for N∈{6,12,24,48}, injected via `dataclasses.replace` onto each pair's `PerPairSignalState.exit_predicate`.
  STOP-REMOVED: `A1Config(exit_policy="sl_only", sl_atr_mult=6.0)`; STOP-PRESENT: same exit, `sl_atr_mult=2.0`.
  Baseline: `exit_policy="sl_partial_close_1r_runner_trail", sl_atr_mult=2.0`.
- **WFO:** IS folds `build_v3_folds()` (is_days≥365), judged by `judge_all_folds_positive`; OOS NOT run.
- **Driver (scratch):** `_disco2000_work/arc2004_gap_stoptax.py` (`PYTHONPATH=. py _disco2000_work/arc2004_gap_stoptax.py`).
