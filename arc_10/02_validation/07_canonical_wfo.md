# Canonical WFO — Arc 10 v3.0.2 (EA-faithful, floating-equity)

> **This is the source of truth for every Arc 10 number.** All other validation docs resolve to the figures here. Where an older doc still cites a different ROI/DD, this doc wins and that doc is stale.
> **Source artifacts (committed, tracked):** `results/l_arc_10_v3.0.2_ea_faithful/` — `matrix.csv`, `per_fold.csv`, `governor_log.csv`, `EA_FAITHFUL_WFO.md`. Every figure below traces to those CSVs. Nothing here is asserted without a CSV row behind it.

## Why this run is canonical

This is the first WFO whose **position sizing matches what trades live.** The live EA sizes every entry as `risk_amount = ACCOUNT_EQUITY × r_base` with equity **including floating open P&L**, re-read per entry (`PositionManager.mqh:143`). This run reproduces that exactly: `mult = r_base × (realized + Σ open floating MtM)` at each entry, equity stepping continuously on floating marks and on closes, per-fold reset. Earlier runs sized linearly or off closed balance — close, but not what the broker sees. The basis correction is the whole point: it measures the **procyclical concurrency tail** (entries size larger when the open book is up, into a possible reversal) that linear/closed sizing cannot see.

## The basis (state this wherever these numbers are quoted)

| Dimension | Setting |
|---|---|
| Convention | EET (FundedNext broker trading day) |
| Sizing | **Floating-equity** — `ACCOUNT_EQUITY × r_base`, floating P&L included, re-read per entry (matches live EA) |
| Cost cell | Cell 5: **swaps OFF** (FundedNext swap-free add-on), **1.5× spread**, **$5/lot RT commission**, **0.5 pip slip × n_fills** (n_fills = 3 if TP1 hit else 2) |
| Governors | **ON** — daily 3.5% halt / 4.5% close-all; total 7% halt / 8% close-all |
| Exit | `sl_partial_close_1r_runner_trail` @ **3.5× ATR** initial SL |
| Universe | 28-pair ex-ante bounded population |
| Risk levels swept | r_base ∈ {0.40%, 0.50%} |
| DD references | **BOTH** reported. **From-initial = FundedNext's actual MLL basis.** Trailing = conservative planning anchor. |
| Determinism | `random_state=42`, `n_jobs=1`, `lineterminator="\n"`; frame sha `05dea9…9ee58a` |

**No CAGR anywhere.** Search folds F1–F11 are annualised over their own ~1-year span. Holdout is reported **per year** (2021–2025 full + 2026 raw partial). The 2026 figure is a **raw ~4-month partial return, never annualised.**

## Canonical matrix (`matrix.csv`)

| risk% | gov | worst-fold ROI% | mean-fold ROI% | trailing DD% | from-init DD% | daily DD% | kills | verdict (trailing) | verdict (from-init) |
|---|---|---|---|---|---|---|---|---|---|
| **0.40** | on | **14.42** | **32.48** | **8.21** | **5.49** | **4.11** | **0** | PASS-VIABLE | **PASS-DEPLOYABLE** |
| 0.40 | off | 14.42 | 32.48 | 8.21 | 5.49 | 4.11 | 0 | PASS-VIABLE | PASS-DEPLOYABLE |
| 0.50 | off | 18.23 | 41.86 | 10.16 | 6.85 | 5.16 | 0 | **FAIL** | PASS-DEPLOYABLE |
| 0.50 | on | 18.23 | 40.95 | 10.89 | 6.85 | 5.16 | 0 | **FAIL** | PASS-DEPLOYABLE |

- **0.40% is the deployable operating tier.** Only level clearing both hard limits on the conservative trailing basis (8.21% < 10% trailing, 4.11% < 5% daily) AND PASS-DEPLOYABLE on FundedNext's actual from-initial basis (5.49% < 8% target). 0 kills.
- **0.50% FAILS on the trailing basis** (10.89% gov-on > 10% hard limit; daily 5.16% > 5%). It passes only on the from-initial basis (6.85%). It is marginal and gated, not a routine step — see §"Risk position" and `../04_runbook/09_risk_and_payout_protocol.md`.
- Worst-fold ROI is **F9 2018** (14.42% / 18.23%). Worst trailing DD is **F10 2019** (8.21%) at 0.40%, **F1 2010** (10.89%) at 0.50% gov-on. Worst from-init is **F5 2014** (5.49% / 6.85%). Worst daily is **F1 2010** (4.11% / 5.16%).

## Full per-fold + per-holdout-year (`per_fold.csv`, 0.40% governed — the operating config)

| fold | yr | n | ROI% | trailing DD% | from-init DD% | daily DD% | kills | fires |
|---|---|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 25.07 | 8.13 | 3.00 | 4.11 | 0 | 1 |
| F2 | 2011 | 182 | 28.09 | 7.88 | 2.53 | 3.55 | 0 | 1 |
| F3 | 2012 | 179 | 31.81 | 4.06 | 0.17 | 1.69 | 0 | 0 |
| F4 | 2013 | 195 | 46.88 | 7.45 | 0.61 | 2.14 | 0 | 0 |
| F5 | 2014 | 192 | 32.30 | 6.12 | **5.49** | 2.50 | 0 | 0 |
| F6 | 2015 | 190 | 22.26 | 5.71 | 4.73 | 3.08 | 0 | 0 |
| F7 | 2016 | 171 | 31.96 | 5.60 | 2.35 | 2.47 | 0 | 0 |
| F8 | 2017 | 190 | 50.83 | 7.19 | 2.40 | 2.17 | 0 | 0 |
| F9 | 2018 | 176 | **14.42** | 5.32 | 0.84 | 2.20 | 0 | 0 |
| F10 | 2019 | 195 | 34.76 | **8.21** | 1.26 | 2.41 | 0 | 0 |
| F11 | 2020 | 188 | 38.92 | 6.21 | 4.05 | 3.28 | 0 | 0 |
| **2021** | 2021 | 212 | 38.79 | 5.95 | 1.04 | 2.50 | 0 | 0 |
| **2022** | 2022 | 179 | 24.11 | 7.05 | 1.66 | 2.25 | 0 | 0 |
| **2023** | 2023 | 189 | 27.26 | 6.71 | 0.90 | 2.02 | 0 | 0 |
| **2024** | 2024 | 191 | 42.86 | 5.57 | 2.62 | 2.46 | 0 | 0 |
| **2025** | 2025 | 251 | 52.55 | 5.83 | 3.54 | 3.22 | 0 | 0 |
| **2026p** | 2026 | 71 | **2.72 (raw ~4-mo partial)** | 4.74 | 0.61 | 1.99 | 0 | 0 |

Mean-fold ROI 32.48% is the simple mean of F1–F11. Sign consistency: **11/11 search folds positive, 6/6 holdout years positive.** Governor fires at 0.40% governed: F1 + F2 daily-halt (1 each), **0 kills, 0 close-alls.** Full per-row figures (incl. 0.50% cells) in `per_fold.csv`.

## Supersession chain

Every run below is retained on disk and self-validates against the **9.22% reconstruction gate**: a linear (`mult ≡ r_base`) + zero-cost reduction reproduces the **9.22%** portfolio worst-fold trailing DD (F4 2013), proving the reconstruction (`build_schedules`, open-book marks, `_flat`) is sound before any sizing variation is layered on.

| Run | Sizing basis | Status | Folder |
|---|---|---|---|
| **EA-faithful** | **floating equity (live-matched)** | **CANONICAL** | `results/l_arc_10_v3.0.2_ea_faithful/` |
| Canonical compound | closed equity (`e_bal`, realized only) | superseded | `results/l_arc_10_v3.0.2_canonical_compound/` |
| Canonical linear | linear `mult ≡ r_base` | superseded | `results/l_arc_10_v3.0.2_canonical/` |
| Governed/floating, ungoverned sequential | intermediate validation variants | superseded | (within above) |

The only difference between EA-faithful and the compound run is per-entry sizing (floating equity vs closed `e_bal`); reconstruction is reused verbatim. The linear→compound step was a benign +0.30pp; the compound→floating step is the procyclical tail (§ below).

## Findings appendix

**Procyclical amplification (`EA_FAITHFUL_WFO.md` §4, 0.50% gov-on):** sizing off floating equity vs closed equity moves worst-fold trailing DD from 10.73% → 10.89% = **+0.16pp**; from-init 6.89% → 6.85% (−0.04pp); daily 5.00% → 5.16% (+0.15pp); mean-fold ROI 40.72% → 40.95% (+0.23pp). **Benign** — floating sizing amplifies DD only marginally when the open book is up. (These four figures are from the cross-run comparison in `EA_FAITHFUL_WFO.md`, derived from the compound and floating CSVs.)

**Daily-DD sweep — keep close-all at 4.5%:** the worst daily-DD event is **F1 2010 at 5.16%** (`per_fold.csv`, 0.50%), where the governor fired both `daily_halt` (2010-04-27) and `daily_close_all` (2010-05-06 — the Flash Crash) with **0 kills** (`governor_log.csv`). That event is a single-bar gap: the daily DD is threshold-invariant — lowering the close-all below 4.5% does not catch it sooner (price gaps straight through) and only costs ROI on benign days. **Governors stay 3.5/4.5 daily, 7/8 total.**

**Gap risk is single-currency bounded, not portfolio-wide.** A news spike hits one currency's correlated cluster (e.g. an EUR or GBP event), not all 28 pairs at once — peak concurrent exposure on such an event is ~5%, not portfolio-wide. Weekend gaps reset the daily measure (Monday opens fresh) but accrue to total DD. The CSV evidence: even the Flash Crash fold killed 0 accounts and stayed within the from-initial basis.
> *Source note:* the ~5% single-currency peak and the close-all flatten-budget figure below come from the operator daily-DD + gap sweep that calibrated the governors; they are **not** rows in the EA-faithful `matrix.csv`/`per_fold.csv`. The governor-fire pattern and the 5.16% daily figure that anchor them **are** CSV-sourced (`governor_log.csv`, `per_fold.csv`).

**Slippage is not the binding risk.** The close-all flatten budget is ~11 pips/position on the worst concurrent flatten (F1 2010, ~12-position book) versus a realistic ~1–2 pip fill → roughly **5× margin**. The binding residual is the **intrabar tick-gap assumption** (open-book marked at H4-bar resolution, not tick), which is only resolvable with live data.

## Risk position (canonical — mirrored in `../04_runbook/09_risk_and_payout_protocol.md`)

- **0.40% = launch / operating tier.** The only level clearing both hard limits (trailing 8.21% < 10%, daily 4.11% < 5%, 0 kills) and PASS-DEPLOYABLE on FundedNext's from-initial basis (5.49%).
- **0.50% = marginal, gated — NOT a routine buffer-triggered step.** FAILS trailing (10.89% > 10%) and daily (5.16% > 5%) on the conservative basis; passes only from-initial (6.85%). Viable only after a substantial banked buffer **and** a live gap-event confirms the tick EA caps daily under 5%. It is a future evidence-gated upgrade, not an automatic step.
- **MLL is static from-initial.** FundedNext's Maximum Loss Limit is 10% of initial; the floor does not re-base on scaling (retained profit expands room within a tier). The EA total-DD floor is now an operator-set input, fail-loud (OPEN-001 resolved — see `../05_history/07_open_issue_dd_restart_rebaselining.md`).

## Honesty constraints

- **Canonical for the MODELLED system, not ground truth.** Two unclosable gaps remain: (1) intrabar tick resolution — the open book is marked at H4-bar resolution of the intrabar low, a tick trigger could differ; (2) live close-all slippage on N concurrent positions is unmodelled (budgeted ~5× safe above, but not proven live).
- The **tick-gap assumption is the live-only residual.** Everything else is settled in backtest.
- Same-bar entry order is the deterministic trade-id tiebreak (H4 has no finer timestamp); same-bar opens contribute ~0 floating (offset-0 mark).
- Every figure in this doc traces to a row in `results/l_arc_10_v3.0.2_ea_faithful/`. The CSVs are not duplicated here beyond the key rows; read them for full precision.
