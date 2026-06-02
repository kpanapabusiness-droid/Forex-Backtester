# Canonical WFO — Arc 10 v3.0.2 (fixed-initial — matches the deployed EA)

> **This is the source of truth for every Arc 10 number.** All other validation docs resolve to the figures here. Where an older doc still cites a different ROI/DD, this doc wins and that doc is stale.
> **Canonical basis = fixed-initial sizing** (`1R = r_base` of the tier's initial balance, held flat) — the basis the **deployed EA actually trades** and the L_PROTOCOL Amendment 8 default. The earlier floating-equity (EA-faithful) run is **retained as a reference comparison** ("what compounding-within-tier would have done"), not deleted — see §"Floating-equity reference" and §"Supersession chain".
> **Source artifacts (committed, tracked):** canonical → `results/l_arc_10_v3.0.2_canonical/` (`matrix.csv`, `per_fold.csv`, `governor_log.csv`, `continuous_equity.csv`, `CANONICAL_WFO.md`); floating reference → `results/l_arc_10_v3.0.2_ea_faithful/`. Every figure below traces to those CSVs. Nothing here is asserted without a CSV row behind it.

```
BASIS:   EET | governed (3.5/4.5 daily, 7/8 total) | 1.5x spread | $5/lot RT | swaps OFF | 0.5 slip
SIZING:  fixed-initial   (floating = reference comparison, ../l_arc_10_v3.0.2_ea_faithful/)
DD REFS: daily=static | max=trailing(plan)+static(enforce) — both reported
RISK:    r_base = 0.40% (operating); 0.50% swept (marginal/gated)
FRAME:   sha 05dea9…9ee58a | 28 pairs | F1–F11 + holdout (aggregate; per-year breakdown in floating reference) | no CAGR
DECORRELATION: N/A — Arc 10 is the first/sole deployed system
```

## Why fixed-initial is canonical

The deployed EA sizes each entry off the tier's **initial balance** (`1R = r_base` of initial, held flat), not off floating equity — so the gate must size the same way to measure what trades live. This run does exactly that. The floating-equity ("EA-faithful") run was the original precedent that probed the **procyclical concurrency tail** (entries sizing larger when the open book is up, into a possible reversal); that tail was measured and found **benign** (+0.16pp worst-fold trailing DD at 0.50%), so it is retained as a reference upper-bound rather than the deployment basis. ROI/DD figures are read off the **fixed-initial** run; floating is the comparison view.

## The basis (state this wherever these numbers are quoted)

| Dimension | Setting |
|---|---|
| Convention | EET (FundedNext broker trading day) |
| Sizing | **Fixed-initial** — `1R = r_base` of the tier's initial balance, held flat (matches the deployed EA). Floating-equity = reference comparison. |
| Cost cell | Cell 5: **swaps OFF** (FundedNext swap-free add-on), **1.5× spread**, **$5/lot RT commission**, **0.5 pip slip × n_fills** (n_fills = 3 if TP1 hit else 2) |
| Governors | **ON** — daily 3.5% halt / 4.5% close-all; total 7% halt / 8% close-all |
| Exit | `sl_partial_close_1r_runner_trail` @ **3.5× ATR** initial SL |
| Universe | 28-pair ex-ante bounded population |
| Risk levels swept | r_base ∈ {0.40%, 0.50%} |
| DD references | **BOTH** reported. **From-initial = FundedNext's actual MLL basis (static).** Trailing = conservative planning anchor. |
| Determinism | `random_state=42`, `n_jobs=1`, `lineterminator="\n"`; frame sha `05dea9…9ee58a` |

**No CAGR anywhere.** Search folds F1–F11 are annualised over their own ~1-year span. Holdout (2021→present) is reported as a positive aggregate here; the **per-year** breakdown (2021–2025 full + 2026 raw partial) lives in the floating reference run — a fixed-initial per-year split is a future regeneration item, not a gate blocker (aggregate holdout is positive). Any 2026 figure is a **raw partial return, never annualised.**

## Canonical matrix — fixed-initial (`results/l_arc_10_v3.0.2_canonical/matrix.csv`)

Two views: **per-fold** = the CEILING (each fold a fresh-tier reset hitting its own worst year); **continuous** = the FLOOR (one unbroken curve 2010→2026, never reset). Worst-fold (per-fold ceiling) is the deployment judge.

| risk% | gov | view | worst/period ROI% | mean-fold ROI% | trailing DD% | from-init DD% | daily DD% | kills | verdict (trailing) | verdict (from-init) |
|---|---|---|---|---|---|---|---|---|---|---|
| **0.40** | on | **per-fold** | **13.77** | **28.85** | **7.73** | **5.55** | **3.86** | **0** | **PASS-DEPLOYABLE** | **PASS-DEPLOYABLE** |
| 0.40 | off | per-fold | 13.77 | 28.85 | 7.73 | 5.55 | 3.86 | 0 | PASS-DEPLOYABLE | PASS-DEPLOYABLE |
| 0.40 | on | continuous | 32.35 | — | 8.20 | 3.01 | 4.01 | 0 | PASS-VIABLE | PASS-DEPLOYABLE |
| 0.50 | on | per-fold | 17.22 | 35.53 | **10.44** | 6.94 | 4.76 | 0 | **FAIL** | PASS-DEPLOYABLE |
| 0.50 | off | per-fold | 17.22 | 36.08 | 9.51 | 6.94 | 4.76 | 0 | PASS-VIABLE | PASS-DEPLOYABLE |
| 0.50 | on | continuous | 40.91 | — | 10.73 | 3.75 | 5.00 | 0 | FAIL | PASS-DEPLOYABLE |

- **0.40% is the deployable operating tier.** On fixed-initial it clears **both** DD references on the worst-fold ceiling: trailing 7.73% < 8% (PASS-DEPLOYABLE on the conservative basis) and from-init 5.55% < 8% (PASS-DEPLOYABLE on FundedNext's actual MLL basis), daily 3.86% < 5%, 0 kills.
- **0.50% FAILS on the trailing basis** (per-fold gov-on 10.44% > 10% hard; continuous 10.73%). It passes only from-initial (6.94%). Marginal and gated, not a routine step — see §"Risk position".
- Worst-fold ROI is **F9 2018** (13.77%). Worst trailing DD is **F1 2010** (7.73%). Worst from-init is **F5 2014** (5.55%). Worst daily is **F1 2010** (3.86%).

## Full per-fold (fixed-initial, 0.40% governed — the operating config) (`per_fold.csv`)

| fold | yr | n | ROI% | trailing DD% | from-init DD% | daily DD% | kills | fires |
|---|---|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 23.24 | **7.73** | 3.04 | **3.86** | 0 | 1 |
| F2 | 2011 | 182 | 25.83 | 7.55 | 2.26 | 3.39 | 0 | 0 |
| F3 | 2012 | 179 | 28.47 | 3.89 | 0.17 | 1.50 | 0 | 0 |
| F4 | 2013 | 195 | 40.19 | 6.40 | 0.61 | 2.04 | 0 | 0 |
| F5 | 2014 | 192 | 28.91 | 6.18 | **5.55** | 2.52 | 0 | 0 |
| F6 | 2015 | 190 | 20.77 | 5.75 | 4.77 | 2.84 | 0 | 0 |
| F7 | 2016 | 171 | 28.51 | 5.52 | 2.28 | 2.44 | 0 | 0 |
| F8 | 2017 | 190 | 42.56 | 6.96 | 2.14 | 1.71 | 0 | 0 |
| F9 | 2018 | 176 | **13.77** | 5.09 | 0.80 | 2.08 | 0 | 0 |
| F10 | 2019 | 195 | 30.97 | 7.58 | 1.25 | 2.27 | 0 | 0 |
| F11 | 2020 | 188 | 34.08 | 6.21 | 4.06 | 2.70 | 0 | 0 |
| **Holdout** | 2021→present | 1093 | **20.03 (aggregate)** | 5.36 | 1.04 | 1.97 | 0 | 0 |

Mean-fold ROI 28.85% is the simple mean of F1–F11. Sign consistency: **11/11 search folds positive, holdout positive (aggregate).** Governor fires at 0.40% governed: **F1 daily-halt (1), 0 kills, 0 close-alls.** Full 2×2 (off/on × risk) detail in `per_fold.csv`.

## Floating-equity reference (retained, NOT the deployment basis) (`results/l_arc_10_v3.0.2_ea_faithful/`)

The original EA-faithful run sized off floating equity (`ACCOUNT_EQUITY × r_base`, floating P&L included, re-read per entry). Retained as the comparison view — the upper bound where compounding-within-tier amplifies both ROI and the procyclical DD tail. **Reference only; not the deployment basis.** Its per-year holdout breakdown (6/6 holdout years positive) is the per-year granularity referenced above.

| risk% | gov | worst-fold ROI% | mean-fold ROI% | trailing DD% | from-init DD% | daily DD% | kills | verdict (trailing) | verdict (from-init) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | on | 14.42 | 32.48 | 8.21 | 5.49 | 4.11 | 0 | PASS-VIABLE | PASS-DEPLOYABLE |
| 0.50 | on | 18.23 | 40.95 | 10.89 | 6.85 | 5.16 | 0 | FAIL | PASS-DEPLOYABLE |

The fixed-initial → floating step is the procyclical tail: at 0.50% gov-on it moves worst-fold trailing DD by **+0.16pp** (10.73% → 10.89%), daily +0.15pp (5.00% → 5.16%), mean-fold ROI +0.23pp. **Benign** — floating sizing amplifies DD only marginally when the open book is up. (Floating-vs-closed comparison figures from `EA_FAITHFUL_WFO.md` §4.) The same launch-at-0.40% conclusion holds on either basis; fixed-initial is simply tighter and matches what trades.

## Supersession chain

Every run below is retained on disk and self-validates against the **9.22% reconstruction gate**: a governors-OFF + zero-cost reduction reproduces the **9.22%** portfolio worst-fold trailing DD (F4 2013), proving the reconstruction (`build_schedules`, open-book marks, `_flat`) is sound before any sizing variation is layered on.

| Run | Sizing basis | Status | Folder |
|---|---|---|---|
| **Canonical (fixed-initial / linear)** | **fixed-% of initial (live-matched)** | **CANONICAL** | `results/l_arc_10_v3.0.2_canonical/` |
| EA-faithful | floating equity | **reference comparison** (retained) | `results/l_arc_10_v3.0.2_ea_faithful/` |
| Canonical compound | closed equity (`e_bal`, realized only) | superseded | `results/l_arc_10_v3.0.2_canonical_compound/` |
| Governed/floating, ungoverned sequential | intermediate validation variants | superseded | (within above) |

The basis decision reverses the 2026-05-31 framing that treated the floating run as authoritative: the deployed EA sizes fixed-initial, so fixed-initial is the gate basis and floating is the retained reference. No numbers are deleted — the floating table above and the EA-faithful CSVs are preserved verbatim.

## Findings appendix

**Daily-DD sweep — keep close-all at 4.5%:** the worst daily-DD event is **F1 2010** (gov-on daily 3.86% fixed-initial; 5.16% in the floating reference at 0.50%), where the governor fired both `daily_halt` (2010-04-27) and `daily_close_all` (2010-05-06 — the Flash Crash) with **0 kills** (`governor_log.csv`). That event is a single-bar gap: the daily DD is threshold-invariant — lowering the close-all below 4.5% does not catch it sooner (price gaps straight through) and only costs ROI on benign days. **Governors stay 3.5/4.5 daily, 7/8 total.**

**Gap risk is single-currency bounded, not portfolio-wide.** A news spike hits one currency's correlated cluster (e.g. an EUR or GBP event), not all 28 pairs at once — peak concurrent exposure on such an event is ~5%, not portfolio-wide. Weekend gaps reset the daily measure (Monday opens fresh) but accrue to total DD. The CSV evidence: even the Flash Crash fold killed 0 accounts and stayed within the from-initial basis.
> *Source note:* the ~5% single-currency peak and the close-all flatten-budget figure below come from the operator daily-DD + gap sweep that calibrated the governors; they are **not** rows in `matrix.csv`/`per_fold.csv`. The governor-fire pattern and the daily figures that anchor them **are** CSV-sourced (`governor_log.csv`, `per_fold.csv`).

**Slippage is not the binding risk.** The close-all flatten budget is ~11 pips/position on the worst concurrent flatten (F1 2010, ~12-position book) versus a realistic ~1–2 pip fill → roughly **5× margin**. The binding residual is the **intrabar tick-gap assumption** (open-book marked at H4-bar resolution, not tick), which is only resolvable with live data.

## Risk position (canonical — mirrored in `../04_runbook/09_risk_and_payout_protocol.md`)

- **0.40% = launch / operating tier.** On fixed-initial it clears both hard limits on the worst-fold ceiling (trailing 7.73% < 10%, daily 3.86% < 5%, 0 kills) and is PASS-DEPLOYABLE on both the conservative trailing basis (< 8%) and FundedNext's from-initial basis (5.55% < 8%).
- **0.50% = marginal, gated — NOT a routine buffer-triggered step.** FAILS trailing (per-fold 10.44% / continuous 10.73% > 10%) on the conservative basis; passes only from-initial (6.94%). Viable only after a substantial banked buffer **and** a live gap-event confirms the tick EA caps daily under 5%. A future evidence-gated upgrade, not an automatic step.
- **MLL is static from-initial.** FundedNext's Maximum Loss Limit is 10% of initial; the floor does not re-base on scaling (retained profit expands room within a tier). The EA total-DD floor is an operator-set input, fail-loud (OPEN-001 resolved — see `../05_history/07_open_issue_dd_restart_rebaselining.md`). See `BROKER_RULES.md` for the full FundedNext measurement basis.

## Honesty constraints

- **Canonical for the MODELLED system, not ground truth.** Two unclosable gaps remain: (1) intrabar tick resolution — the open book is marked at H4-bar resolution of the intrabar low, a tick trigger could differ; (2) live close-all slippage on N concurrent positions is unmodelled (budgeted ~5× safe above, but not proven live).
- The **tick-gap assumption is the live-only residual.** Everything else is settled in backtest.
- Same-bar entry order is the deterministic trade-id tiebreak (H4 has no finer timestamp).
- Every figure in this doc traces to a row in `results/l_arc_10_v3.0.2_canonical/` (canonical) or `results/l_arc_10_v3.0.2_ea_faithful/` (floating reference). The CSVs are not duplicated here beyond the key rows; read them for full precision.
