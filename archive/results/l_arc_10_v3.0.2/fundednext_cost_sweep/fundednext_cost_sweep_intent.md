# Arc 10 v3.0.2 — FundedNext Cost Sweep: Intent (Pre-Compute)

> Dispatch: `CC_DISPATCH_FUNDEDNEXT_COST_SWEEP.md`
> Analysis-only post-hoc cost overlay on the **EET** WFO pool for the FundedNext
> deployment target (EET broker day, swap-free). No engine changes; `simulate_path`
> is invoked only to reconstruct the winning-config per-trade R, exactly as the
> bespoke step_5 / amendment_3 addendum did. The cost overlay itself is pure
> post-hoc R-arithmetic.
>
> **Status at write time:** pre-compute. Per dispatch §1 ("no compute until
> committed"), this intent is committed before the sweep is run.

---

## §1 — Anchor verification (amended for this dispatch)

The dispatch §11 stop condition as literally written ("verify worktree HEAD ==
anchor 244fb76") is **structurally unsatisfiable** for this dispatch: §2 (winning-
config replay tooling) and §8 (EET-vs-UTC delta) depend on commits that *post-date*
244fb76 — the UTC re-run cost-sweep grid and the `core/sim/costs/` primitives both
landed after the anchor. HEAD is `5612e5d`, 36 commits ahead of `244fb76`.

Per explicit user amendment, the anchor condition is reinterpreted for this dispatch:

> **Anchor verification = EET pool _input_ byte-identity to 244fb76, NOT worktree HEAD.**

- Anchor commit: `244fb76` (Merge PR #216 — `arc/l_arc_10_v3.0.2_addendum`).
- EET pool input: `results/l_arc_10_v3.0.2/step_1/pool.parquet`
  sha256 = `d624212bf10b24e986a4cd1a625becfd08b7ce4e8dc7922df4c543e6aacab486`.
- Verified byte-identical to its committed state at 244fb76 (3152 rows × 62 cols);
  the regenerated companion `trade_paths.parquet` was produced deterministically
  from the same step_1 config and the regenerated `pool.parquet` reproduced the
  committed sha exactly before `trade_paths` was retained.

**Precedent.** The prior UTC cost sweep resolved the identical contradiction the
same way: it merged `origin/main` to acquire the UTC re-run pool (which post-dates
244fb76) and treated EET/UTC pool input integrity — not worktree HEAD — as the
binding anchor condition. This dispatch follows that precedent.

**All other §11 stop conditions remain in force:**
- Baseline reproduction (1× / 0 slip / commission-OFF must reproduce the unmodified
  EET pool worst-fold numbers within 0.5pp) — enforced in-driver, return code 2 on
  mismatch.
- Zero-spread > 1% of pool — enforced in-driver, return code 4.
- EET pool / trade_paths not found — return code 3.

---

## §2 — Winning config (replay target)

EET step_5 Top-1 by `search_worst_ratio` (`results/l_arc_10_v3.0.2/step_5/wfo_results.csv`):

| field | value |
|-------|-------|
| architecture | A1 |
| cluster_id | c0 |
| sl_multiplier | 3.5 |
| exit_policy | `sl_partial_close_1r_runner_trail` |
| exposure | unlimited |

Replay mirrors `scripts/l_arc_10_v3_0_2/amendment_3_addendum.py`: A1 admits every
trade in each OOS window; `unlimited` applies no exposure cap; `simulate_path`
recomputes per-trade final R at SL=3.5 over `trade_paths.parquet`; R clamped to
[-1.5, 20] exactly as `step_5._fold_metrics`.

**Pool-SL normalization.** Pool `final_r` / `sl_distance_price` are normalized to
SL=2.0; the winning config is SL=3.5 (scale factor 3.5/2.0 = 1.75). TP1 (=+1R at
SL=3.5) is detected when pool `mfe_r ≥ 1.75` (pool-SL units).

---

## §3 — Cost model (FundedNext, swap-free)

| component | rule |
|-----------|------|
| **Commission** | **$5/lot round-turn** (NOT the $4 5ers default), ON in all 15 grid cells. `compute_commission_usd(lots, rate_per_lot_rt=5.0)`. |
| **Spread** | widen recorded EET per-bar spread (entry+exit close) by `(mult−1)`, **no floor**. `compute_extra_spread_price(se, sx, mult)`. |
| **Slippage** | adverse `slip_per_fill_pips × n_fills`; `n_fills = 3` if TP1 hit else `2`. `compute_slippage_pips(slip, tp1_hit)`. |
| **Swap** | **OFF** — FundedNext swap-free add-on. Primitive **not invoked**. |

All components are converted to R-multiple decrements against the per-trade risk
amount ($500 = 0.5% × $100k) using per-pair lot sizing:
`lots = risk_usd / (sl_distance_pips × pip_value_usd_per_lot)`, then
`final_r_adj = final_r_replayed − (comm_r + slip_r + extra_spread_r)`.

### §3.4 — FX constants (reused verbatim from prior UTC intent §4.1)

Per dispatch §3.4 (reuse representative per-pair FX constants from the prior sweep),
the per-pair pip-value reference rates are reused unchanged:

| pair | ref rate |
|------|----------|
| USDJPY | 109.6032 |
| USDCAD | 1.2412 |
| USDCHF | 0.9324 |
| NZDUSD | 0.7047 |
| GBPUSD | 1.4067 |
| AUDUSD | 0.7760 |

USD-quote pairs: $10/lot/pip. JPY/CAD/CHF/NZD/GBP/AUD quotes converted via the
constants above.

---

## §4 — Correctness gates

1. **G4 commission test** — `compute_commission_usd(lots, rate_per_lot_rt=5.0)` returns
   `5×lots` (parametrized: 1.0→5.0, 0.5→2.5, 2.0→10.0, 0.0→0.0). Added to
   `tests/sim/costs/test_commission.py`; the existing $4-default test is preserved
   (still correct for the 5ers default).
2. **Baseline reproduction** — cell (1× / 0 slip / commission-OFF) must reproduce
   the unmodified EET pool worst-fold numbers within 0.5pp:
   - worst-fold ROI = 0.224563
   - worst-fold DD = 0.073539
   - worst-fold ratio = 6.4273
   - search n_total = 2059
   Enforced in-driver before any grid compute; return code 2 on mismatch.

---

## §5–§6 — Grid + per-fold tables

- **Grid:** 5 spread mults {1×, 1.5×, 2×, 3×, 4×} × 3 slippage levels {0, 0.5, 1.0}
  = **15 cells**, commission ON in every cell, swap axis removed.
- Each cell recorded at **r_base = 0.5%** AND at **r_recommended** (§7): worst/mean-
  fold ROI/DD/ratio, holdout ROI/DD/ratio, sign-consistency (n positive of 11),
  neg-fold count, trade counts, zero-spread count, verdict.
- **Per-fold tables** (F1–F11 + Holdout) emitted for the **9 realistic cells**
  (spread ∈ {1×, 1.5×, 2×} × slip ∈ {0, 0.5, 1.0}).

---

## §7 — Recommended risk band

`r_recommended = 0.005 × (0.080 / worst_fold_DD_at_r_base)` (Amendment 3 linear DD
scaling), evaluated on three reference cells, output to 4 decimals:

- central realistic: 1.5× spread / 0.5 slip
- optimistic: 1.0× spread / 0.5 slip
- adverse: 2.0× spread / 1.0 slip

The grid's `r_recommended` column uses the **central** value.

---

## §8 — EET vs UTC delta

Cross-comparison table (9 realistic cells, swap-off, r_base) pulling UTC worst-fold
ratio/DD and holdout ROI from `results/l_arc_10_v3.0.2/cost_sweep/grid_results.csv`.

**Caveat (to be restated in the report):** the prior UTC grid carries **$4**
commission; this EET sweep uses **$5**. The delta table therefore conflates the
EET-vs-UTC boundary effect with the $4→$5 commission step. The report will flag
this as a directional comparison, not a clean boundary isolation.

---

## §9 — Verdict rule (per cell)

- **PASS-DEPLOYABLE:** worst-fold ratio ≥ 2.0 AND worst-fold ROI ≥ 5% AND worst-fold DD < 8%
- **DD_WATCH:** worst-fold DD in [8%, 10%) (and not otherwise FAIL)
- **FAIL:** worst-fold ratio < 2.0 OR worst-fold ROI < 5%
- **FAIL_DD_HARD:** worst-fold DD ≥ 10%

---

## §10 — Deliverables

Under `results/l_arc_10_v3.0.2/fundednext_cost_sweep/`:

- `fundednext_cost_sweep_intent.md` (this file, pre-compute)
- `fundednext_cost_sweep_report.md` (post-compute)
- `grid_results_eet.csv` — 15 cells × {r_base, r_recommended}
- `per_fold_realistic.csv` — per-fold detail, 9 realistic cells
- `eet_vs_utc_delta.csv` — §8 cross-comparison
- `diagnostics.json` — baseline check, zero-spread, FX constants, r-band
- `scripts/audit/arc_10/fundednext_cost_sweep.py` — reproducible driver

---

## Reproduce

```
py scripts/audit/arc_10/fundednext_cost_sweep.py --baseline-only   # gate only
py scripts/audit/arc_10/fundednext_cost_sweep.py                   # full grid
```
