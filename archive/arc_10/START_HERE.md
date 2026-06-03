# Arc 10 — START HERE

> **Status:** Arc 10 v3.0.2 — analysis complete, documented, live-ready. **Operating risk 0.40%.** Nothing left to decide; execute the go-live checklist and run it.

---

## The system in one paragraph

Arc 10 is a **D1 swing-low rejection long** signal traded across **28 FX pairs** on the **EET** (FundedNext broker trading day) convention. Initial stop is **3.5× ATR**; exits run the canonical `sl_partial_close_1r_runner_trail` three-stage policy (partial close at 1R, runner on a trail). Position sizing is **EA-faithful floating-equity** — `ACCOUNT_EQUITY × r_base` with floating open P&L included, re-read per entry, exactly matching the live EA — at **0.40% risk**. Portfolio-level **governors** are on: daily **3.5%** halt / **4.5%** close-all, total **7%** halt / **8%** close-all. Deployed on **FundedNext** under a static-MLL (Maximum Loss Limit from initial) basis.

---

## Expected performance (canonical — see `02_validation/07_canonical_wfo.md`)

EA-faithful, floating-equity, 0.40% governed — the operating config:

| Metric | Value |
|---|---|
| Worst-fold ROI | **14.42%** (F9 2018) |
| Mean-fold ROI | **32.48%** (F1–F11) |
| Per-year holdout | **~24–53%** (2021–2025 full; 2026 = 2.72% raw ~4-month partial, never annualised) |
| Worst-fold trailing DD | **8.21%** (F10 2019) |
| Worst-fold from-init DD | **5.49%** (F5 2014) — FundedNext's actual MLL basis |
| Worst-fold daily DD | **4.11%** (F1 2010) |
| Kills | **0** |
| Sign consistency | 11/11 search folds positive, 6/6 holdout years positive |

> These are **backtest** numbers for the modelled system. Live-validate against the first N trades; the one unclosable gap is the intrabar tick-gap assumption (open book marked at H4-bar resolution, not tick).

**0.40% is the operating tier** — the only level clearing both hard limits on the conservative trailing basis (8.21% < 10% trailing, 4.11% < 5% daily) and PASS-DEPLOYABLE on FundedNext's from-initial basis (5.49% < 8%). **0.50% is a future, evidence-gated upgrade — not a routine step** (it FAILS trailing 10.89% / daily 5.16%).

---

## Where everything is

| You want… | Read |
|---|---|
| **Canonical numbers (source of truth)** | `02_validation/07_canonical_wfo.md` |
| The risk + payout rule | `04_runbook/09_risk_and_payout_protocol.md` |
| Kill criteria | `04_runbook/07_kill_criteria.md` |
| Weekly review | `04_runbook/08_sunday_weekly_check.md` |
| Daily health check | `04_runbook/01_daily_health_check.md` |
| What the system is, end to end | `00_executive_summary.md` + `README.md` |
| Live-only residuals + operator go-live checklist | `../TODO.md` (Arc 10 — Live operations) |

---

## Do NOT relitigate

The risk analysis — **0.40% vs 0.50%, the sizing basis (EA-faithful floating-equity), the governors, gap and slippage** — is **CANONICAL and CLOSED**. Do not re-open it in a new chat. If a number is ever questioned, it traces to a row in the EA-faithful CSVs at `results/l_arc_10_v3.0.2_ea_faithful/` (`matrix.csv`, `per_fold.csv`, `governor_log.csv`; frame sha `05dea9…9ee58a`). The earlier UTC / signal-parity-rerun and "PASS-VIABLE → Amendment-3" framings are **legacy** — superseded by the EA-faithful run.

---

## Next action

Execute the **Operator go-live checklist** in `../TODO.md`:

1. Set EA input `Initial_Equity_Floor = 100000` on FundedNext (fail-loud if unset).
2. Pull `main` → local + VPS.
3. Buy FundedNext $100k; attach EA at **0.40%**; confirm journal `equity init: floor=100000.00 source=input`.
