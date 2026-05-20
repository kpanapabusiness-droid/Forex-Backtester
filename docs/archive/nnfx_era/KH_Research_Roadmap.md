# KH Research Roadmap

**Forex Ignition Rebuild**

KH arc closed: 2026-05-09 | KH-24 validated, locked, live | All Path A items closed

---

## Current State

KH-24 is the validated base system as of 2026-04-20. It is the first configuration in the KH arc to pass the WFO gate across all 7 out-of-sample folds and remains live and locked on the production VPS.

As of 2026-05-09, three post-lock diagnostic phases (KH-27, KH-28, KH-29) have collectively established that fold 7's +1.92% ROI is the structural worst-fold ceiling for KH-24. The Path A direction (improving KH-24) is closed. Future research moves to Path B and the new L characterization arc. This document records what was tested, what was closed, and what remains as future work.

| Metric | Value |
| --- | --- |
| Gate | PASS — all 7 OOS folds positive |
| Worst-fold ROI | +1.92% (Fold 7) │ Gate: > 0% |
| Worst-fold DD | 6.37% (Fold 1) │ Gate: < 8% |
| OOS Trades | 214 across Oct 2020 – Jan 2026 (328 baseline) |
| Components | Baseline signal + Exposure cap=2 + 1H CIR T=0.28 |
| EA | KH24_EA.mq5 v2.0 — built, audited, deployed |
| Live status | Running on Contabo VPS connected to 5ers account |

### KH-24 WFO fold results

| Fold | OOS Period | Trades | ROI | DD | Gate |
| --- | --- | --- | --- | --- | --- |
| 1 | Oct 2020 – Jul 2021 | 41 | +13.35% | 6.37% | PASS |
| 2 | Jul 2021 – Apr 2022 | 36 | +9.63% | 4.45% | PASS |
| 3 | Apr 2022 – Jan 2023 | 25 | +11.90% | 4.43% | PASS |
| 4 | Jan 2023 – Oct 2023 | 32 | +3.32% | 3.80% | PASS |
| 5 | Oct 2023 – Jul 2024 | 23 | +6.23% | 3.09% | PASS |
| 6 | Jul 2024 – Apr 2025 | 30 | +3.24% | 5.03% | PASS |
| 7 | Apr 2025 – Jan 2026 | 27 | +1.92% | 4.06% | PASS |

Signal flow: 328 baseline signals → 107 filtered by 1H CIR → 7 blocked by cap → 214 trades taken.

---

## Path A — KH-24 Improvement (CLOSED)

The Path A direction is closed as of 2026-05-09. Three diagnostic phases established that further KH-arc tuning will not move worst-fold ROI beyond approximately +1.92%. Each individual A item is recorded below with its closure status.

### A1 — Re-entries with exposure cap applied

| Field | Detail |
| --- | --- |
| What | KH-25 was characterized as failing because re-entry losses concentrated in fold 7. Hypothesis was that extending the cap to re-entries would prevent those losses. |
| Status | **CLOSED by KH-27 (KILL).** Pre-flight analysis showed re-entries fire AFTER the original has already exited via kh14_bar6 (10/10 sampled trades). The original is never in the open-positions set at re-entry time, so the cap could never have applied. The three fold 7 losses are months apart (2025-04-30, 2025-10-01, 2025-12-26) with zero overlapping exposure. The "correlated losses" framing in the original KH-25 phase doc was not supported by the data. |

### A2 — 1H filter threshold refinement on live data

| Field | Detail |
| --- | --- |
| What | Re-run the T=0.28 threshold sweep as live data accumulates. |
| Status | **DEFERRED.** Insufficient live data accumulated to make refinement meaningful. Revisit only after 6+ months of live trading produces enough new data to warrant a sweep. Do not optimize on small live samples. |

### A3 — 1H timeframe deeper research (was KH-26)

| Field | Detail |
| --- | --- |
| What | Test additional 1H features beyond h1_cir as entry conditions. |
| Status | **DEFERRED.** Entry-side regime filtering shown ineffective for fold 7 by KH-28 STRUCTURAL verdict. Adding 1H features without addressing fold 7 regime is unlikely to improve worst-fold ROI. Reopening would require new evidence that 1H features specifically protect fold 7. |

### A4 — D1 close-in-range on KH-24 population

| Field | Detail |
| --- | --- |
| What | Re-run D1 CIR discrimination analysis on the filtered KH-24 population. |
| Status | **DEFERRED.** Same reasoning as A3 — entry-side filtering is not the binding constraint. KH-28 STRUCTURAL verdict stands. |

### A5 — Fold 7 regime investigation

| Field | Detail |
| --- | --- |
| What | Identify a signal-time regime indicator that gates signals during unfavourable conditions. |
| Status | **CLOSED by KH-28 (STRUCTURAL).** No candidate signal-time regime variable (R1 cross_pair_atr, R2 cross_pair_trend_strength, R3 cross_pair_dispersion, R4 pair_atr_ratio control) discriminated fold 7 losers from non-fold-7 winners with both p<0.05 and protective direction. R2 was closest miss (right direction, p=0.077, n=13 — underpowered). Fold 7 weakness is not addressable via signal-time regime selection from this candidate set. Caveat: R1 was JPY-pair dominated by literal interpretation of the spec; an equal-weight per-pair-normalized rebuild was not performed and would be the right starting point if cross-pair regime work is ever revisited. |

### A6 — Position sizing refinement

| Field | Detail |
| --- | --- |
| What | Re-run ATR-conditional sizing analysis on KH-24 population. |
| Status | **DEFERRED.** Sizing refinement requires live validation against 5ers reset floor simulation; the priority is too low given the structural fold 7 ceiling. Revisit only if a different research arc materially improves DD headroom. |

### A7 — Kijun_d1 exit refinement on KH-24 population

| Field | Detail |
| --- | --- |
| What | Test 2-bar kijun_d1 confirmation on the cleaner KH-24 population. |
| Status | **DEFERRED.** KH-29 exit-side excursion analysis returned AMBIGUOUS — neither EXIT_DEFECT_KIJUN nor TREND_EXTENSION cleanly indicted. Fold 7's kijun_d1-exited winners actually show *higher* capture (0.536) than its trail-exited winners (0.454), so the kijun pathway is not robbing capture. There is no exit-defect signal to justify A7. Reopening would require new evidence of an exit-side defect specifically. |

---

## Fold 7 Ceiling Finding

The +1.92% worst-fold ROI in KH-24 fold 7 is structural to the exhaustion bar signal in low-extension regimes. KH-28 ruled out signal-time regime selection from the candidate set tested; KH-29 ruled out clean exit-side indictment. Fold 7 winners have normal win rates (51.9%, matching good folds) but moderate MFE shrinkage (median 1.695R vs 1.961R in folds 1–3) and a +16.8pp shift toward kijun_d1 exits. Neither effect is large enough to act on via the KH-arc levers tested.

The project's realistic worst-fold ROI ceiling for the KH-24 signal is approximately +1.92%. This is the structural finding that closes the Path A direction.

---

## Path B — Independent Research Directions

Path B items are independent of KH-24. KH-24 continues to run live while Path B research proceeds on its own merits. The one-change-per-phase rule applies within each new arc.

### B1 — Apply KH-24 system to 1H timeframe

| Field | Detail |
| --- | --- |
| What | Port the KH-24 signal conditions to 1H data. |
| Status | **CLOSED by KI arc.** Tested in early 2026. Mean R collapsed to 0.004 (t=0.095) on 1H data. Edge does not survive 1H compression. The exhaustion bar pattern is structurally native to 4H. |

### B2 — New signal discovery on 4H

| Field | Detail |
| --- | --- |
| What | Identify a different 4H entry pattern, complementary or independent. |
| Status | **SUPERSEDED by L arc.** The L characterization arc subsumes B2 and approaches it differently — bottom-up data characterization first, then mechanically pre-registered signal candidates from the atlas. See `L_ARC_PLAN.md`. |

### B3 — Daily timeframe system

| Field | Detail |
| --- | --- |
| What | Build a new D1 system from scratch with higher firing frequency than the exhaustion bar. |
| Status | **DEFERRED.** Bottlenecked by D1 frequency (~10 trades/pair/year is below WFO worst-fold power). Only revisit if a D1 signal discovery shows materially higher firing rate than the exhaustion bar. The L arc's atlas may surface D1-specific findings worth investigating, in which case B3 reopens via the L pipeline. |

### B4 — Short side research

| Field | Detail |
| --- | --- |
| What | Independent short system, mirror or otherwise. |
| Status | **GATED.** Two prior attempts failed to find short-side edge with t-stat comparable to long side. Do not reopen without fresh population analysis showing comparable t-stat as a prerequisite. |

---

## Recommended Path Forward

The active research direction as of 2026-05-09 is the **L characterization arc**. Path A is closed by evidence; B1 is closed by evidence; B2 is superseded by L; B3 and B4 are deferred or gated.

The L arc is a from-scratch bottom-up exploratory data analysis effort. It builds a quantitative atlas of patterns and regularities in the FX data before any new signal building. Output is descriptive (statistics with confidence intervals), not predictive. Top-N candidates from the atlas proceed mechanically to signal-testing arcs via a pre-registered ranking rule locked in L0.

The L arc is independent of KH-24. No assumption that any pattern found will or should complement KH-24. No commitment to KH-24's choices (4H, long-only, exhaustion bar, 28-pair universe) carrying forward. The L arc evaluates everything fresh.

Source of truth for the L arc: `L_ARC_PLAN.md`.

---

## Permanently Closed

These were tested and rejected. Evidence is in the arc record. Do not reopen without extraordinary new evidence.

- **Phase JL: jd_rf_evt_02 bounded event system** — Forward bias in population selection. Invalidated.
- **Clean label conditioning in population selection** — Lookahead violation by definition. Permanently excluded.
- **Full NNFX stack as strategy** — Adopted as structural starting point only. Stack not viable as-is.
- **C7 volume gate** — 5ers data incompatibility confirmed. Tick volume is broker-specific.
- **kijun_4h exit** — Destroys fold 5 by cutting valid trends on normal 4H pullbacks.
- **signal_flip exit** — Fires after single counter-move bars during valid trend continuations.
- **Indicator-driven C1 sweeps as primary research method** — All prior GPT-4 indicator conversions found broken or hallucinated.
- **Exit indicator sweeps as standalone research** — Tested exhaustively in KG phase. No WFO-passing result found.
- **Bar-1 exit for State 2** — Cost of scratch (~0.3R/trade) equals or exceeds benefit from cutting State 2 drag.
- **KH-25 re-entry exposure hypothesis** — KH-27 KILL. Re-entries fire post-original-exit; cap was structurally never applicable.
- **KH-24 fold 7 entry-side regime selection** — KH-28 STRUCTURAL. No signal-time regime variable (R1–R4) discriminated fold 7 with both significance and protective direction.
- **KH-24 fold 7 exit-side indictment** — KH-29 AMBIGUOUS. Neither EXIT_DEFECT nor TREND_EXTENSION cleanly indicted.
- **KI arc: 1H timeframe port of KH-24 signal** — Mean R 0.004, t=0.095. Edge does not survive 1H compression.
- **Aider** — permanently excluded from implementation.
- **GPT-4** — permanently excluded from implementation.

---

## Non-Negotiable Rules

These apply to all future research without exception. The methodology that produced KH-24 and the three clean diagnostic verdicts (KH-27, KH-28, KH-29) is not edited.

- `build_ex_ante_bounded_population` — mandatory on every test
- Signal bar N close → entry bar N+1 open — absolute
- D1 one-day lag — never revert
- No lookahead anywhere in signal, indicator, alignment, or stop logic
- WFO worst-fold is the only judge of success
- One change per phase, pre-committed gate, accept the result
- Per-bar MT5 spread data — never hardcoded defaults
- Risk = 1% of reset floor balance (5ers validated ceiling)
- Clean labels = evaluation tool only, never used for population selection
- GPT-4 and Aider permanently excluded from implementation
- Every phase gets a result document regardless of pass or fail

---

*KH-24 is locked, live, and unchanged. The KH arc is closed as a system-improvement project. The L arc is the active research direction. See `L_ARC_PLAN.md` for L arc scope and methodology.*
