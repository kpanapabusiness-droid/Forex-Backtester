# Arc 10 v3.0.2 — CANONICAL WFO (FundedNext, EET, 3.5R)

> **THE single authoritative gate run.** Every risk/DD/ROI number at the exact settings the system trades, consolidated here so all docs cite one source. **Supersedes every prior gate run** (ungoverned per-trade-sequential 49.87 / 22.46 / 7.80; `governed_wfo`; `fundednext_floating`; `whole_period_dd`) — those are superseded, not deleted. v3.0.2 logic LOCKED; governors are EA-faithful re-modelling, **not tuned**. Frame sha `05dea9…9ee58a` (manifest-matched). Costs ON in every cell (FundedNext cell 5: swap-OFF, 1.5× spread, $5/lot RT commission, 0.5 pip slip × n_fills; `cost_r` imported from `scripts/audit/arc_10/fundednext_cost_sweep.py`). r_base parameterised {0.40%, 0.50%}; deterministic (two-run sha identity); EET only; PR-gated.

> **Validation gate PASSED:** governors-OFF + zero-cost reproduces the **9.22%** portfolio worst-fold trailing DD (abort-if-not check in driver). Costed output trusted.

## Governor firing reference (stated)

> The canonical governed cells fire the total governors (7% halt / 8% close-all) on the **from-initial** reference — the rule the live EA / FundedNext actually enforces (DD measured from the tier's initial balance). The **trailing-peak** DD of that governed curve is the PRIMARY *planning* number reported alongside (the honest fresh-tier worst case). The stricter **trailing-FIRED** governor variant (which can freeze / kill the account) is a Cut-3 sensitivity + 8% kill flag, NOT a matrix cell.

## 1. THE canonical matrix (2 risk × 2 governance × 2 view; both DD refs)

| risk% | governance | view | worst/period ROI% | mean ROI% | trailing DD% | from-init DD% | daily DD% | kills | verdict_trailing | verdict_from_initial |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | off | per-fold (ceiling) | 13.77 | 28.85 | 7.73 | 5.55 | 3.86 | 0 | PASS-DEPLOYABLE | PASS-DEPLOYABLE |
| 0.40 | on | per-fold (ceiling) | 13.77 | 28.85 | 7.73 | 5.55 | 3.86 | 0 | PASS-DEPLOYABLE | PASS-DEPLOYABLE |
| 0.40 | off | continuous (floor) | 32.35 | nan | 8.20 | 3.01 | 4.01 | 0 | PASS-VIABLE | PASS-DEPLOYABLE |
| 0.40 | on | continuous (floor) | 32.35 | nan | 8.20 | 3.01 | 4.01 | 0 | PASS-VIABLE | PASS-DEPLOYABLE |
| 0.50 | off | per-fold (ceiling) | 17.22 | 36.08 | 9.51 | 6.94 | 4.76 | 0 | PASS-VIABLE | PASS-DEPLOYABLE |
| 0.50 | on | per-fold (ceiling) | 17.22 | 35.53 | 10.44 | 6.94 | 4.76 | 0 | FAIL | PASS-DEPLOYABLE |
| 0.50 | off | continuous (floor) | 41.51 | nan | 10.15 | 3.75 | 5.00 | 0 | FAIL | PASS-DEPLOYABLE |
| 0.50 | on | continuous (floor) | 40.91 | nan | 10.73 | 3.75 | 5.00 | 0 | FAIL | PASS-DEPLOYABLE |

> Per-fold = the CEILING (each fold a fresh-tier reset hitting its own worst year; linear). Continuous = the FLOOR (one unbroken compounding curve 2010→2026, never reset). True risk lives between. `mean ROI%` is per-fold only (continuous is a single curve → NaN). Continuous period ROI is compound-annualised (illustrative; no withdrawals / firm caps modelled).

## 2. Risk-level decision (0.40% vs 0.50% vs the limits)

| risk_pct | worst_fold_trailing_dd_pct | under_10pct_hard | under_8pct_deploy | worst_daily_dd_pct | under_5pct_daily | worst_fold_roi_pct | mean_fold_roi_pct |
|---|---|---|---|---|---|---|---|
| 0.40 | 7.73 | YES | YES | 3.86 | YES | 13.77 | 28.85 |
| 0.50 | 10.44 | NO | NO | 4.76 | YES | 17.22 | 35.53 |

> **Decision:** launch-at-0.40%, step-to-0.50% only after a buffer is banked (0.40% keeps the fresh-tier worst-fold trailing DD under 10% where 0.50% does not). Per-fold (fresh-tier ceiling) worst-fold trailing DD is **7.73%** at 0.40% vs **10.44%** at 0.50%; per-fold worst daily DD 4.76% vs the 5% limit (governed by the 4.5% daily close-all). Reinforcing the call: at 0.50% the continuous-view worst daily DD reaches **5.00%** — right at the 5% limit (modelled bar-overshoot past the 4.5% close-all; the unmodelled N-position close-all slippage would push it over) — whereas at 0.40% it is 4.01%.

## 3. Governor value (ON vs OFF, per risk level; both DD refs)

| risk_pct | view | trailing_dd_saved_pp | from_initial_dd_saved_pp | roi_cost_pp | kills_from_initial |
|---|---|---|---|---|---|
| 0.40 | per_fold | 0.00 | 0.00 | 0.00 | 0 |
| 0.40 | continuous | 0.00 | 0.00 | 0.00 | 0 |
| 0.50 | per_fold | -0.93 | 0.00 | 0.55 | 0 |
| 0.50 | continuous | -0.58 | 0.00 | 0.60 | 0 |

**0.40% — from-initial firing:** daily_halt: 1. **8% total-kill flags (trailing-fired sensitivity):** continuous 2010-05-17.

**0.50% — from-initial firing:** daily_close_all: 1, daily_halt: 2. **8% total-kill flags (trailing-fired sensitivity):** per-fold F2 2011-03-17; per-fold F8 2017-04-12; per-fold F10 2019-07-19.

> The from-initial (live-EA) governors add only a mild tax and zero 8% kills. Under the stricter trailing-fired reference the 8% close-all CAN kill / freeze the account (flags above) — the sharpest reference-dependence in the study, a FINDING, not a trigger to retune. The negative 'DD saved' under governed-static per-fold (governors deepen trailing DD by locking a recoverable dip) is real, not an artifact.

## 4. PASS verdict per cell (vs gates)

> Gates: worst-fold ROI > 5% AND mean-fold ROI > 8% (PASS-DEPLOYABLE); DD ≤ 8% in-system / ≤ 10% hard. Verdict shown on BOTH DD references (trailing-peak primary / from-initial = FundedNext basis).

| risk_pct | governance | view | trailing_dd_pct | verdict_trailing | from_initial_dd_pct | verdict_from_initial |
|---|---|---|---|---|---|---|
| 0.40 | off | per-fold | 7.73 | PASS-DEPLOYABLE | 5.55 | PASS-DEPLOYABLE |
| 0.40 | on | per-fold | 7.73 | PASS-DEPLOYABLE | 5.55 | PASS-DEPLOYABLE |
| 0.40 | off | continuous | 8.20 | PASS-VIABLE | 3.01 | PASS-DEPLOYABLE |
| 0.40 | on | continuous | 8.20 | PASS-VIABLE | 3.01 | PASS-DEPLOYABLE |
| 0.50 | off | per-fold | 9.51 | PASS-VIABLE | 6.94 | PASS-DEPLOYABLE |
| 0.50 | on | per-fold | 10.44 | FAIL | 6.94 | PASS-DEPLOYABLE |
| 0.50 | off | continuous | 10.15 | FAIL | 3.75 | PASS-DEPLOYABLE |
| 0.50 | on | continuous | 10.73 | FAIL | 3.75 | PASS-DEPLOYABLE |

## 5. Continuous-vs-per-fold gap (cross-fold accumulation)

| risk_pct | worst_fold_trailing_dd_pct | continuous_trailing_dd_pct | gap_pp |
|---|---|---|---|
| 0.40 | 7.73 | 8.20 | 0.47 |
| 0.50 | 10.44 | 10.73 | 0.30 |

> Longest continuous time-underwater (0.50% gov-on compound): **130 days** (0.36y). A positive gap means the continuous curve's worst drawdown exceeds the worst single fold (cross-fold accumulation); a non-positive gap means the deepest drawdowns are contained single episodes, not troughs straddling a year boundary.

> **Floor/ceiling nuance:** the dispatch frames per-fold as the *ceiling* and continuous as the *floor*. That ordering holds on the **from-initial** reference (continuous banks a buffer → its from-init DD is far lower: e.g. 3.75% continuous vs 6.94% per-fold at 0.50%). On the **trailing-peak** reference the small positive gaps above mean the continuous high-water-mark drawdown marginally EXCEEDS the worst single fold — trailing-peak accumulates across the unbroken curve where the per-fold reset clears it. Both are honest; the per-fold trailing remains the planning anchor (fresh-tier ceiling) and the continuous trailing is the realised-path max-DD.

## Reviewer notes / honesty constraints

- **Canonical for the MODELLED system, not ground truth.** Two unclosable modelled-vs-live gaps: (1) **intrabar tick resolution** on governor firing — the sim marks the open book at H4-bar resolution of the intrabar low (via cumulative MAE), so a bar-resolution trigger UNDERSTATES a true tick trigger; (2) **close-all slippage on N concurrent positions** is unmodelled — it bites hardest exactly where the daily-DD margin sits near 5%. Frame these as canonical modelled results, to be live-validated against the first N trades.
- **FundedNext's actual total-DD rule is semi-static** (limit scales with the account; floor stays at initial within a tier, re-bases at scale) — it sits BETWEEN the static (from-initial) and trailing bounds reported, nearer static once a buffer exists. Scaling timeline unconfirmed.
- **Every figure traces to the committed CSVs** (`matrix.csv`, `per_fold.csv`, `continuous_equity.csv`, `governor_log.csv`). No number is asserted that is not reproducible from this run.

## Done-state

> Matrix (2×2×2, both DD refs) + risk-level decision + governor value + PASS verdict + continuous-vs-per-fold gap, all reproducible from the committed CSVs; prior gate runs marked superseded; `.gitignore` re-include confirmed.
