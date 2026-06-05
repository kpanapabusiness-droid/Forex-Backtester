# arc 2013 — Weekend UP-gap weekend SHORT, JPY crosses (the gap-fill's stronger leg, finally tested) — PORTFOLIO

- **Chat / range:** 2000s (2000–2999)
- **Arc id:** 2013
- **Date:** 2026-06-05
- **Disposition:** **PORTFOLIO** — mean-positive net of costs (best overshoot exit +0.745% IS), beats its
  fair same-side null by +1.74%, but NOT all-folds-positive (5/10 neg). The corpus's **first SHORT
  component** and a candidate **2018-leg** for a future 4-way combination arc.
- **Council:** none (PORTFOLIO ≠ PASS; the mandatory survivor council §5h applies only to IS+OOS
  all-folds-positive survivors, as for the 3 existing components).
- **Tooling:** BUILT `WeekendUpGapShortSignal` (discovery/tools/gap_signals.py); made the BUILT
  `build_null_signal_evaluation` direction-aware (carries `direction` through — additive, longs
  byte-identical). Reused BUILT direction-aware `observe_long_capture(direction="short")`. **No
  canonical-core change** (the short Step-1/engine path was already merged, PR #273). Drivers:
  `_disco2000_work/arc2013_observe_upgap_short.py`, `arc2013_robustness.py`, `arc2013_engine_wfo.py`,
  `arc2013_null.py`.

---

## Idea + why (the still-unrun highest-acc directional edge in the corpus)

arc 2001 (2000s) found the weekly-open gap-fill is clean, monotone, **symmetric** on H4: DOWN >1ATR gaps
drift +0.45 (frac+ 0.59), UP >1ATR gaps drift −0.57 (**frac DOWN 0.64**). The UP-gap leg has the bigger
drift / higher hindsight accuracy — flagged repeatedly (FLAG-1, arcs 2001/2003/1014) as the gap-fill's
STRONGER leg, but untradeable under long-only. arc 1006 (1000s) found the DOWN-gap FILL LONG is
mean-POSITIVE on **JPY CROSSES** (+0.685% IS, the corpus's one PORTFOLIO long) but mean-NEGATIVE on majors.
Shorts merged (PR #273); the 1000s chat referenced the up-gap short as the frontier but used arc 1015 for
the 3-way portfolio combination — so it was **still unrun**. It descends from arc 2001 (2000s lineage) → my
range. Natural test: the **UP-gap weekend SHORT on JPY crosses** — the direction-mirror of 1006's
mean-positive long, on its stronger leg, and (1015's spec) a candidate SHORT 2018-leg.

## What happened

### Observation (honest i+1 SHORT capture/drift, direction-aware `observe_long_capture`)
- **JPY crosses, up-gap≥1.0:** honest capture **0.518 (>0.50)**, drift **+0.08** (median **+0.146**); ≥0.5:
  cap 0.515 but drift ~0 / median −0.115; **≥1.5 inverts** to −0.22 (non-monotone — the thin-tail tell).
- **Majors:** up-gap short drift **−0.355** (price CONTINUES up) → dead. The 1006 cross>major asymmetry
  holds for the short.
- **Symmetry surprise:** honestly entered, the DOWN-gap LONG is the STRONGER leg (short-lens drift −0.46 =
  +0.46 long) and the UP-gap SHORT the weaker (+0.08) — the **opposite** of arc 2001's hindsight "up-gap
  stronger." Cause: the **JPY-basket upward drift** (arc 1009) helps the long, fights the short.

### Robustness gauntlet (the arc-2011 discipline, before any engine spend)
- median (≥1.0) **+0.146 > mean +0.081** (central tendency positive — NOT a thin-tail artifact at this gate,
  unlike arc 2011); leave-one-pair-out 5/6 positive but **drop AUDJPY → −0.107** (single-pair sensitivity).
- **Fair-null lift (decisive):** a random weekly-open SHORT loses (drift −0.188 — basket headwind); up-gap≥1.0
  **+0.27 ATR over the null**. The timing carries information → a NON-coin-flip entry → §5f REQUIRES the
  honest engine before any FAIL.

### Honest engine IS WFO (§5f exit menu) — the FIRST end-to-end short engine run
The short pool builds sign-correctly (mean final_r +0.137 at thr 1.0); WFO runs clean, no canonical change.
- **thr ≥0.5:** all exits net-negative → dead.
- **thr ≥1.0:** mean-POSITIVE under the whole OVERSHOOT-exit family — `sl_plus_trailing_atr` **+0.745%**,
  `tp_3r` +0.510%, `partial_runner` +0.324%, `tp_2r` +0.015% (mechanism-aligned, 1006/1007 overshoot);
  non-overshoot `sl_only` −1.97% / `trailing_swing` −1.43% negative. **NOT all-folds-positive** (5/10 neg,
  worst −3.18%); thin (1–14 trades/fold).
- **Per-fold (trailing_atr):** +2011 +2012 −2013 −2014 −2015(n2) −2016 +2017 **+2018(+5.08)** **+2019(+5.92)**
  −2020(n1). **Strongly positive on the binding 2018 fold** (where gap-fill −6.79 & fbr −4.20 both bleed);
  bleeds a complementary set {2013/14/16}.

### Fair same-side null on the engine (decisive PORTFOLIO-vs-KILL test)
Random JPY-cross weekly-open SHORT (5 seeds): trailing_atr **−0.99%**, tp_3r **−1.17%** → REAL beats null by
**+1.74% / +1.68%** (seed-tight). Much cleaner than 1006's +0.36pp. The up-gap TIMING is the edge, not the
JPY-cross-short basket.

## Verdict: PORTFOLIO

Mean-positive net of costs, beats its fair same-side null by a large margin, decorrelated by construction
(first SHORT component), and **strongly 2018-positive** (+5.08%) — the most portfolio-relevant new component
since 1013. NOT all-folds-positive (thin, 5/10 neg) → not PASS. Recorded under
[`../portfolio-candidates/arc_2013_weekend_upgap_short_jpy/`](../portfolio-candidates/arc_2013_weekend_upgap_short_jpy/).
**OOS preserved** (not AFP; exit IS-selected — the combination arc is the proper AFP gate that touches OOS).

## Threads / lessons

1. **The gap-fill's UP-gap SHORT is REAL and net-positive on JPY crosses** — the first short edge to survive
   the honest engine. But honestly entered (i+1, costs) it is the **WEAKER** leg vs the down-gap long
   (+0.745% best-exit vs 1006's +0.69% at a far higher null margin), NOT the "stronger" leg the hindsight
   gap-bar-open measure suggested — because the **JPY-basket upward drift** flatters the long and taxes the
   short (the same drift that inflated 1006's null, arc 1009). The "up-gap is stronger" flag (2001/2003) is
   a hindsight artifact; honest direction matters.
2. **First end-to-end short engine run — the canonical short path works** (pool + architecture + engine
   build sign-correctly; FundedNext costs net symmetrically; no FLAG). The PR #273 short path is now
   exercised, not just observation-verified (closes the arc-2011 "never exercised end-to-end" note).
3. **A candidate 2018-leg exists at last** — strongly positive on the binding 2018 fold (+5.08%) with a
   complementary bleed set. It does NOT cover 2015 (the spec's other half), so it is not a guaranteed
   portfolio-closer, but it materially loosens the 2018 constraint. **Named next step: a 4-way combination
   WFO** (gap-fill 1006 + month-end 1011 + fbr 1013 + up-gap-short 2013) — 2018 may now be satisfiable by
   up-gap-short + month-end, potentially freeing fbr weight for 2015. That arc is the proper all-folds-positive
   gate (and the OOS touch).
4. **The overshoot mechanism mirrors for the short** (arc 1007): the gap-fill edge is the overshoot PAST the
   prior close, captured by a let-it-run trailing exit; tp/partial that cap it are weaker, `sl_only`/swing
   negative. Same as the down-gap long.

## FLAGS (code not merged)
None requiring the canonical core. Two EXPERIMENT-tool changes (discovery/tools/, flow freely per §9):
BUILT `WeekendUpGapShortSignal`; `build_null_signal_evaluation` made direction-aware (additive, longs
byte-identical). Both registered in `TOOL_REGISTRY.md`. The standing `A1Config.time_exit_bars`-unwired flag
(arc 1005) is unchanged (I used the registered trailing exit, not a time exit).
