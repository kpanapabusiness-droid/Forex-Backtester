# Canonical WFO — Arc 10 v3.0.2 (deploy-faithful: fixed-initial + daily_ref=initial)

> **This is the source of truth for every Arc 10 number.** All other validation docs resolve to the figures here. Where an older doc still cites a different ROI/DD, this doc wins and that doc is stale.
> **⟳ CANONICAL REPOINTED (2026-06-02) to [`_final_canonical`](../../results/l_arc_10_v3.0.2_final_canonical/FINAL_CANONICAL_WFO.md).** The matrix, basis, and per-fold numbers below are the **fixed-initial + `daily_ref=initial`-resetting** deploy-faithful run that matches the live EA post-FIX-2b (PR #252/#253). This is a **numbers-only** repoint; the full prose reconciliation (the legacy 0.50% sweep narrative, the floating-equity procyclical appendix in §"Findings appendix", and the §"Risk position" 0.50% framing) is a **separate pending task** — those sections still reference the superseded floating run and are flagged inline.
> **Source artifacts (committed, tracked):** `results/l_arc_10_v3.0.2_final_canonical/` — `matrix.csv`, `per_fold.csv`, `governor_log.csv`, `FINAL_CANONICAL_WFO.md`. Every matrix/per-fold figure below traces to those CSVs. Nothing here is asserted without a CSV row behind it. (Superseded floating run retained at `results/l_arc_10_v3.0.2_ea_faithful/`.)

## Why this run is canonical

This is the first WFO whose **position sizing matches what trades live.** The live EA sizes every entry as `risk_amount = ACCOUNT_EQUITY × r_base` with equity **including floating open P&L**, re-read per entry (`PositionManager.mqh:143`). This run reproduces that exactly: `mult = r_base × (realized + Σ open floating MtM)` at each entry, equity stepping continuously on floating marks and on closes, per-fold reset. Earlier runs sized linearly or off closed balance — close, but not what the broker sees. The basis correction is the whole point: it measures the **procyclical concurrency tail** (entries size larger when the open book is up, into a possible reversal) that linear/closed sizing cannot see.

## The basis (state this wherever these numbers are quoted)

| Dimension | Setting |
|---|---|
| Convention | EET (FundedNext broker trading day) |
| Sizing | **Fixed-initial** — 1R ≡ r_base of the **INITIAL** balance, constant per trade (matches the deployed EA `ArcComputeLots` post-FIX-1). NOT floating-equity. |
| Cost cell | Cell 5: **swaps OFF** (FundedNext swap-free add-on), **1.5× spread**, **$5/lot RT commission**, **0.5 pip slip × n_fills** (n_fills = 3 if TP1 hit else 2) |
| Governors | **ON** — daily 3.5% halt / 4.5% close-all; total 7% halt / 8% close-all |
| Daily-DD basis | **`daily_ref="initial"`** — fixed-$/day off initial, **window RESETS each EET day** (deployed EA `Daily_DD_Basis=INITIAL`, EquityGuards.mqh FIX 2b). NOT the quarantined `static_noreset` freeze-mode. |
| Exit | `sl_partial_close_1r_runner_trail` @ **3.5× ATR** initial SL |
| Universe | 28-pair ex-ante bounded population |
| Risk levels swept | r_base ∈ {0.40%, 0.42%, 0.45%} — **0.40% = canonical operating tier**; 0.42%/0.45% = margin-check only (NOT adoption candidates). 0.50% is no longer swept (0.40% is the ceiling on the deploy-faithful basis). |
| DD references | **BOTH** reported. **From-initial = FundedNext's actual MLL basis.** Trailing = conservative planning anchor. |
| Determinism | `random_state=42`, `n_jobs=1`, `lineterminator="\n"`; frame sha `05dea9…9ee58a` |

**No CAGR anywhere.** Search folds F1–F11 are annualised over their own ~1-year span. Holdout is reported **per year** (2021–2025 full + 2026 raw partial). The 2026 figure is a **raw ~4-month partial return, never annualised.**

## Canonical matrix (`matrix.csv`)

| risk% | gov | worst-fold ROI% | mean-fold ROI% | trailing DD% | from-init DD% | daily DD% | kills | verdict (trailing) | verdict (from-init) |
|---|---|---|---|---|---|---|---|---|---|
| **0.40** | on | **13.77** | **28.85** | **7.73** | **5.55** | **4.06** | **0** | **PASS-DEPLOYABLE** | **PASS-DEPLOYABLE** |
| 0.40 | off | 13.77 | 28.85 | 7.73 | 5.55 | 4.06 | 0 | PASS-DEPLOYABLE | PASS-DEPLOYABLE |
| 0.42 | on | 14.46 | 30.29 | 8.09 | 5.83 | 4.27 | 0 | PASS-VIABLE | PASS-DEPLOYABLE |
| 0.45 | on | 15.49 | 31.97 | 9.47 | 6.24 | 4.57 | 0 | PASS-VIABLE | PASS-DEPLOYABLE |

- **0.40% is the deployable operating tier — now PASS-DEPLOYABLE on BOTH bases.** On the deploy-faithful fixed-initial basis it clears both hard limits on the conservative trailing reference (7.73% < 8% deploy AND < 10% hard; daily 4.06% < 5%) AND is PASS-DEPLOYABLE on FundedNext's actual from-initial basis (5.55% < 8% target). 0 kills. (Fixed-initial improves trailing DD vs the superseded floating run's 8.21% → now under the 8% deploy threshold, so the verdict lifts from PASS-VIABLE to PASS-DEPLOYABLE.)
- **0.42% / 0.45% are margin-check only — NOT adoption candidates.** They quantify proximity to the limits (trailing 8.09% / 9.47%, both under the 10% hard limit; daily 4.27% / 4.57%, both under 5%; 0 kills) and confirm 0.40% is the ceiling. Daily DD is a fixed $/day off initial, so it only tightens in %-terms as the account grows — there is no case to raise risk. 0.50% is no longer swept.
- Worst-fold ROI is **F9 2018** (13.77%). Worst trailing DD is **F1 2010** (7.73% at 0.40%). Worst from-init is **F5 2014** (5.55%). Worst daily is **F1 2010** (4.06%). All at 0.40% gov-on.

## Full per-fold + per-holdout-year (`per_fold.csv`, 0.40% governed — the operating config)

| fold | yr | n | ROI% | trailing DD% | from-init DD% | daily DD% | kills | fires |
|---|---|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 23.24 | **7.73** | 3.04 | 4.06 | 0 | 1 |
| F2 | 2011 | 182 | 25.83 | 7.55 | 2.26 | 3.54 | 0 | 1 |
| F3 | 2012 | 179 | 28.47 | 3.89 | 0.17 | 1.69 | 0 | 0 |
| F4 | 2013 | 195 | 40.19 | 6.40 | 0.61 | 2.46 | 0 | 0 |
| F5 | 2014 | 192 | 28.91 | 6.18 | **5.55** | 2.49 | 0 | 0 |
| F6 | 2015 | 190 | 20.77 | 5.75 | 4.77 | 3.08 | 0 | 0 |
| F7 | 2016 | 171 | 28.51 | 5.52 | 2.28 | 2.44 | 0 | 0 |
| F8 | 2017 | 190 | 42.56 | 6.96 | 2.14 | 2.28 | 0 | 0 |
| F9 | 2018 | 176 | **13.77** | 5.09 | 0.80 | 2.23 | 0 | 0 |
| F10 | 2019 | 195 | 30.97 | 7.58 | 1.25 | 2.43 | 0 | 0 |
| F11 | 2020 | 188 | 34.08 | 6.21 | 4.06 | 3.32 | 0 | 0 |
| **2021** | 2021 | 212 | 33.84 | 5.36 | 1.04 | 2.51 | 0 | 0 |
| **2022** | 2022 | 179 | 22.40 | 6.87 | 1.65 | 2.24 | 0 | 0 |
| **2023** | 2023 | 189 | 24.92 | 5.34 | 0.90 | 2.04 | 0 | 0 |
| **2024** | 2024 | 191 | 36.74 | 4.88 | 2.64 | 2.46 | 0 | 0 |
| **2025** | 2025 | 251 | 43.54 | 5.86 | 3.57 | 3.31 | 0 | 0 |
| **2026p** | 2026 | 71 | **2.78 (raw ~4-mo partial)** | 4.59 | 0.61 | 2.00 | 0 | 0 |

Mean-fold ROI 28.85% is the simple mean of F1–F11. Sign consistency: **11/11 search folds positive, 6/6 holdout years positive.** Governor fires at 0.40% governed: F1 + F2 daily-halt (1 each), **0 kills, 0 close-alls.** **F5 2014 / F6 2015 recover to +28.91% / +20.77% with 0 daily-governor fires** (under the old non-resetting `static_noreset` they froze at −5.99% / −5.45% — the daily-reset alignment is live; see `FINAL_CANONICAL_WFO.md` §"F5/F6 recovery proof"). Full per-row figures (incl. 0.42%/0.45% cells) in `per_fold.csv`.

## Supersession chain

Every run below is retained on disk and self-validates against the **9.22% reconstruction gate**: a linear (`mult ≡ r_base`) + zero-cost reduction reproduces the **9.22%** portfolio worst-fold trailing DD (F4 2013), proving the reconstruction (`build_schedules`, open-book marks, `_flat`) is sound before any sizing variation is layered on.

| Run | Sizing basis | Status | Folder |
|---|---|---|---|
| **Final canonical** | **fixed-initial + `daily_ref=initial`-resetting (deploy-faithful, matches live EA post-FIX-2b)** | **CANONICAL** | `results/l_arc_10_v3.0.2_final_canonical/` |
| EA-faithful | floating equity (procyclical-tail measurement) | superseded | `results/l_arc_10_v3.0.2_ea_faithful/` |
| Canonical compound | closed equity (`e_bal`, realized only; non-resetting daily) | superseded | `results/l_arc_10_v3.0.2_canonical_compound/` |
| Canonical linear | linear `mult ≡ r_base` (pre-daily-reset-alignment) | superseded | `results/l_arc_10_v3.0.2_canonical/` |
| Governed/floating, ungoverned sequential | intermediate validation variants | superseded | (within above) |

The final canonical run differs from the (now superseded) EA-faithful floating run in two aligned, validated params only: **sizing** (fixed-initial vs floating equity — the live EA sizes fixed-initial via `ArcComputeLots`) and **daily-DD basis** (`daily_ref="initial"` resetting each EET day — the live EA `Daily_DD_Basis=INITIAL`, EquityGuards.mqh FIX 2b). v3.0.2 signal/exit/cost are LOCKED and identical. Fixed-initial moves 0.40% worst-fold trailing DD from 8.21% → **7.73%** (−0.47pp; it drops the procyclical tail floating sizing measured), lifting the trailing verdict from PASS-VIABLE to **PASS-DEPLOYABLE**.

## Findings appendix

> **⚠ LEGACY — pending full reconciliation.** This appendix reports the floating-vs-closed *procyclical amplification* at 0.50%, which was the point of the now-superseded floating run. Under the canonical fixed-initial basis the relevant fixed-vs-floating delta at **0.40%** is in `FINAL_CANONICAL_WFO.md` §4 (fixed-initial trailing 7.73% vs floating 8.21% = −0.47pp). The 0.50% figures below are retained for historical context only and are NOT canonical.

**Procyclical amplification (`EA_FAITHFUL_WFO.md` §4, 0.50% gov-on):** sizing off floating equity vs closed equity moves worst-fold trailing DD from 10.73% → 10.89% = **+0.16pp**; from-init 6.89% → 6.85% (−0.04pp); daily 5.00% → 5.16% (+0.15pp); mean-fold ROI 40.72% → 40.95% (+0.23pp). **Benign** — floating sizing amplifies DD only marginally when the open book is up. (These four figures are from the cross-run comparison in `EA_FAITHFUL_WFO.md`, derived from the compound and floating CSVs.)

**Daily-DD sweep — keep close-all at 4.5%:** the worst daily-DD event is **F1 2010 at 5.16%** (`per_fold.csv`, 0.50%), where the governor fired both `daily_halt` (2010-04-27) and `daily_close_all` (2010-05-06 — the Flash Crash) with **0 kills** (`governor_log.csv`). That event is a single-bar gap: the daily DD is threshold-invariant — lowering the close-all below 4.5% does not catch it sooner (price gaps straight through) and only costs ROI on benign days. **Governors stay 3.5/4.5 daily, 7/8 total.**

**Gap risk is single-currency bounded, not portfolio-wide.** A news spike hits one currency's correlated cluster (e.g. an EUR or GBP event), not all 28 pairs at once — peak concurrent exposure on such an event is ~5%, not portfolio-wide. Weekend gaps reset the daily measure (Monday opens fresh) but accrue to total DD. The CSV evidence: even the Flash Crash fold killed 0 accounts and stayed within the from-initial basis.
> *Source note:* the ~5% single-currency peak and the close-all flatten-budget figure below come from the operator daily-DD + gap sweep that calibrated the governors; they are **not** rows in the EA-faithful `matrix.csv`/`per_fold.csv`. The governor-fire pattern and the 5.16% daily figure that anchor them **are** CSV-sourced (`governor_log.csv`, `per_fold.csv`).

**Slippage is not the binding risk.** The close-all flatten budget is ~11 pips/position on the worst concurrent flatten (F1 2010, ~12-position book) versus a realistic ~1–2 pip fill → roughly **5× margin**. The binding residual is the **intrabar tick-gap assumption** (open-book marked at H4-bar resolution, not tick), which is only resolvable with live data.

## Risk position (canonical — mirrored in `../04_runbook/09_risk_and_payout_protocol.md`)

- **0.40% = launch / operating tier.** On the deploy-faithful fixed-initial basis it clears both hard limits with margin (trailing **7.73% < 8% deploy** and < 10% hard, daily **4.06% < 5%**, 0 kills) and is **PASS-DEPLOYABLE on both bases** (from-initial 5.55%). It is the only swept tier PASS-DEPLOYABLE on the conservative trailing reference.
- **0.42% / 0.45% = margin-check only — NOT adoption candidates.** Trailing 8.09% / 9.47% (both < 10% hard but over the 8% deploy threshold → PASS-VIABLE), daily 4.27% / 4.57% (< 5%), 0 kills. They quantify proximity to the limits and confirm 0.40% is the ceiling. Because daily DD is a fixed $/day off initial (it only tightens in %-terms as the account grows), there is no case to raise risk.
- **0.50% is no longer swept.** Legacy note (superseded floating run): 0.50% FAILED trailing (10.89% > 10%) and daily (5.16% > 5%). The deploy-faithful sweep stops at 0.45% to bound the margin; 0.40% remains the operating tier.
- **MLL is static from-initial.** FundedNext's Maximum Loss Limit is 10% of initial; the floor does not re-base on scaling (retained profit expands room within a tier). The EA total-DD floor is now an operator-set input, fail-loud (OPEN-001 resolved — see `../05_history/07_open_issue_dd_restart_rebaselining.md`).

## Honesty constraints

- **Canonical for the MODELLED system, not ground truth.** Two unclosable gaps remain: (1) intrabar tick resolution — the open book is marked at H4-bar resolution of the intrabar low, a tick trigger could differ; (2) live close-all slippage on N concurrent positions is unmodelled (budgeted ~5× safe above, but not proven live).
- The **tick-gap assumption is the live-only residual.** Everything else is settled in backtest.
- Same-bar entry order is the deterministic trade-id tiebreak (H4 has no finer timestamp); same-bar opens contribute ~0 floating (offset-0 mark).
- Every figure in this doc traces to a row in `results/l_arc_10_v3.0.2_ea_faithful/`. The CSVs are not duplicated here beyond the key rows; read them for full precision.
