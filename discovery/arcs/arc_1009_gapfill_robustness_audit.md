# Arc 1009 — Weekend Gap-Fill Robustness/Firmness Audit (the one PORTFOLIO component)

> **Arc id:** 1009 · **Chat:** 1000–1999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **PORTFOLIO — re-affirmed but THINNER (and threshold-fragile).** This is an AUDIT
> of arc 1006's component (the SAME edge — not a new signal). It (1) REPRODUCES the +0.69% IS headline
> EXACTLY via the registered tools (+0.685%; first committed-tool reproduction — the transcribed record
> was never re-run), (2) shows it is NOT a single-pair artifact (leave-one-out positive everywhere), but
> (3) is threshold-fragile (edge concentrated at 0.5 ATR, ~gone by 1.25), and (4) de-inflates the
> apparent edge: vs a FAIR random null (same pairs/exit/SL) the gap-SPECIFIC excess is ~+0.36pp, about
> HALF the +0.69% headline — the rest is JPY-cross basket drift + 24-bar/2·ATR geometry that random
> entries share. The component still beats the fair null and is mean-positive → PORTFOLIO holds.
> **Lever:** firm/audit the one asset for the PORTFOLIO route (frontier item 3), NOT a new mechanism.

Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first), via the canonical entry point.
IS ONLY — the disposition is already PORTFOLIO/FAIL; OOS not re-run (arc 1006 has it: 4/6 neg, worst −4.13%).
Reuses three BUILT tools (`WeekendGapFillLongSignal`, `make_time_exit_predicate`, `build_null_signal_evaluation`).

## (a) Log read — FRESH EYES (honest-era only)

Pulled main. 17 honest-era arcs: shallow directional prediction comprehensively closed (both metrics, all
instruments/TFs/regimes/exits, stop-removed); calendar/flow now mapped (TOM sub-cost 1005, gotobi absent
1008); triangulation ≈0 (3005); relative-value catch-up falsified (2003). The arc-3004 escalation: the
apparatus can only bet direction = coin-flip; durable edges need a structural unlock (shorts/second-leg,
operator-gated, not yet flipped). **The ONE net-positive long-only edge is the weekend gap-down-fill on JPY
crosses (arc 1006) → PORTFOLIO.** STOP absent.

My pre-shorts lane = hunt a 2nd net-positive long-only component OR a novel mechanism. After arc 1008
(gotobi absent) the genuinely-novel-long-only well is near-dry and a generative council on a heavily-mapped
space re-surfaces FLAG-1 (arcs 2003 found this). The highest-value CHEAP arc that doesn't grind dead
directional ground: **firm the one asset.** The gap-fill record was TRANSCRIBED, not re-run (no committed
turnkey repro), and arc 1006 picked ONE config (thr 0.5) — so two artifact risks were never checked:
single-pair dependence and threshold-fragility, and the null was not a same-exit apples-to-apples. If the
programme's only asset is an artifact, the PORTFOLIO route is empty — a decision-relevant thing to know.

## (b)–(c) Idea + method

Audit, not a new signal. Reproduce arc 1006 via the registered `WeekendGapFillLongSignal(threshold_atr=θ,
gap_hours=36)` + `make_time_exit_predicate(n_bars=24)` over the 5 JPY crosses (EURJPY, GBPJPY, AUDJPY,
CADJPY, CHFJPY), SL=2·ATR, `A1Config(exit_policy=None, trail_enabled=False, risk_pct=0.005)`, IS folds
`build_v3_folds` (is_days≥365), judged by `judge_all_folds_positive`. Four cuts:
1. **Threshold sweep** θ ∈ {0.5, 0.75, 1.0, 1.25} — is +0.69% a 0.5-specific number?
2. **Leave-one-out** at θ=0.5 — is the +mean driven by ONE pair?
3. **Fair random null** (`build_null_signal_evaluation`, matched fire-rate, SAME 24-bar exit + 2·ATR SL).

## (d) Results

**Threshold robustness (all 5, 24-bar exit, SL 2·ATR):**

| threshold | IS mean fold ROI | worst fold | neg folds | total IS trades |
|---|---|---|---|---|
| **0.5 ATR** | **+0.685%** | −6.79% | 5/10 | 260 |
| 0.75 ATR | +0.191% | −4.49% | 6/10 | 143 |
| 1.0 ATR | +0.244% | −4.85% | 4/10 | 83 |
| 1.25 ATR | +0.021% | −4.35% | 4/10 | 57 |

Reproduces arc 1006 (+0.685% ≈ +0.69%). The edge **decays sharply with threshold** and is ~zero by 1.25
ATR — it lives in the many small-to-moderate (0.5 ATR) gaps, NOT the rare huge ones. None all-folds-positive
(consistent with PORTFOLIO/FAIL). [thinner-but-more-events at 0.5 is the only viable cut.]

**Leave-one-out (θ=0.5):**

| variant | IS mean | note |
|---|---|---|
| all 5 (baseline) | +0.685% | — |
| drop EURJPY | +0.268% | EURJPY = strongest contributor (dropping it ~halves the mean) |
| drop GBPJPY | +0.964% | GBPJPY = a drag |
| drop AUDJPY | +0.584% | ~neutral |
| drop CADJPY | +1.336% | CADJPY = biggest drag (dropping it ~doubles the mean) |
| drop CHFJPY | +0.606% | ~neutral |

**Mean stays positive under every leave-one-out (+0.27% to +1.34%)** → NOT a single-pair artifact (the main
artifact risk is cleared). The edge is spread across pairs; EURJPY leads, CADJPY/GBPJPY drag (a future
PORTFOLIO-combination arc could weight toward EUR/CHF/AUD-JPY).

**Fair random-entry null (θ=0.5, SAME pairs/exit/SL, only entry TIMING randomized):**

| | IS mean | worst | neg | trades |
|---|---|---|---|---|
| REAL gap-fill | +0.685% | −6.79% | 5/10 | 260 |
| NULL random entry | **+0.327%** | −5.05% | 6/10 | 309 |

The real edge BEATS the fair null, but the **gap-specific excess is only ~+0.36pp — about HALF the +0.69%
headline.** A FAIR null (random long JPY-cross entries, identical 24-bar exit + 2·ATR SL) is itself
IS-mean-**positive** (+0.327%): being long a JPY-cross basket in 2010–2020 (Abenomics JPY weakness 2012–15)
with this exit/SL geometry already earns ~+0.33pp regardless of the gap. Arc 1006's reported null (−0.60%)
was NOT a same-exit apples-to-apples, which inflated the apparent edge ~3.6×. The honest gap-specific edge
over a fair null is thin but positive and direction-consistent.

## Council — NOT convened

No survivor (not all-folds-positive → not a `passed/` promotion, so the mandatory §5h council doesn't fire)
and no diagnosis fork. This is a measurement/audit arc on an existing PORTFOLIO component.

## Final verdict — PORTFOLIO (re-affirmed, thinner) → disposition PORTFOLIO

The weekend gap-fill **holds as a PORTFOLIO component** — it reproduces exactly, beats a fair null, is
mean-positive net of costs, and is NOT a single-pair artifact. But it is materially THINNER than the record
implied (gap-specific excess ~+0.36pp vs the +0.69% headline) and **threshold-fragile** (lives at 0.5 ATR).
Disposition unchanged (PORTFOLIO, not all-folds-positive); the component is now reproduced-via-committed-tools
and honestly re-characterized. NOT a 2nd component — the SAME edge, audited.

## Lessons (candidate for LESSONS.md)

1. **The gap-fill reproduces exactly via the registered tools (+0.685% ≈ arc 1006's +0.69%) and is NOT a
   single-pair artifact** (leave-one-out positive everywhere; EURJPY leads, CADJPY/GBPJPY drag) — but its
   edge over a FAIR null (same pairs/exit/SL, only timing randomized) is **~+0.36pp, ~half the headline**;
   the rest is JPY-cross basket drift + 24-bar/2·ATR geometry that random entries share. **Always null with
   IDENTICAL exit/SL/universe** — a null that differs in exit (arc 1006's −0.60%) inflates the apparent edge
   several-fold. (Arc-10 lesson in null-construction space.)
2. **A random long basket can be IS-mean-positive purely from in-sample regime drift** (random JPY-cross
   longs +0.327% over 2010–2020) — IS-mean-positive ALONE is weak evidence; the fair null + OOS are what
   separate edge from drift.
3. **The gap-fill edge is threshold-fragile** (concentrated at 0.5 ATR, ~zero by 1.25) — it is in the many
   small-to-moderate gaps, not the rare huge ones; "bigger gap = cleaner reversion" is false here (consistent
   with arc 1007's overshoot reframe — the snapback is a frequent small effect, not a rare large one).

## Threads

- **The gap-fill remains the one PORTFOLIO component, now firmer (reproduced, not single-pair) but thinner
  and threshold-fragile.** Best cut = θ=0.5; a future combination arc should weight EUR/CHF/AUD-JPY over
  CAD/GBP-JPY.
- **A 2nd decorrelated net-positive long-only component is still NOT found** (17 arcs). The thin gap-specific
  excess (~+0.36pp) + threshold-fragility reinforce that even the best long-only edge is marginal under
  FundedNext costs → the **shorts/second-leg unlock (arc-3004 escalation, FLAG-1, reinforced by 2001/2003)
  remains the highest-leverage operator move.**
- The PORTFOLIO-candidate deep record is updated additively (`robustness_audit_arc1009.md`) with the
  committed reproduction + the fair-null de-inflation + leave-one-out profile.

## Flags (code NOT merged)

None. No canonical-core change; no new BUILT tool (reused three existing BUILT tools). Driver scratch.

## Reproduction

Driver scratch `_disco_work/arc1009_gapfill_robustness.py` (`PYTHONPATH=. py ...`). Data:
`histdata_root=C:\Users\panap\histdata_backup`, tf H4, `5ers_eet`, pairs EURJPY/GBPJPY/AUDJPY/CADJPY/CHFJPY.
Signal `WeekendGapFillLongSignal(threshold_atr=θ, gap_hours=36)`; exit `make_time_exit_predicate(n_bars=24)`;
`A1Config(sl_atr_mult=2.0, exit_policy=None, trail_enabled=False, risk_pct=0.005)`; IS folds
`build_v3_folds` (is_days≥365); `judge_all_folds_positive`; null `build_null_signal_evaluation(seed=42)` with
the same exit re-injected. Costs ON at `build_fold_stats_from_run`. OOS not re-run (preserved; arc 1006 holds it).
