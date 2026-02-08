# Phase B — Indicator Quality & Signal Research

## What Phase B Is and Is Not

**Phase B is** a deterministic, config-driven, non-competitive diagnostic pipeline that evaluates C1 and Volume indicators as *signals* (behavior and controllability). It produces response curves, overlap/timing diagnostics, scratch/MAE analysis, and a controlled-overfit diagnostic. Outputs feed a **Quality Gate** table (PASS / REDESIGN / DISCARD) and an **approved indicator pool** for Phase C eligibility.

**Phase B is NOT**:
- WFO v2 selection or worst-fold leaderboard ranking
- Profitability optimization or winner selection
- A trading system build
- A change to engine semantics or spread logic (Phase A is immutable)

## Data & Execution

- **Date window**: 2019-01-01 to 2026-01-01 (config-driven; must be explicit).
- **Pairs**: 28 (canonical list; reuse from phase5_base / configs).
- **Spreads**: ON (bar-by-bar; affects PnL only, not trade counts).
- **Execution realism**: Entry next open, system exit next open, intrabar SL/BE/TS. No changes to engine.

## Metrics, Thresholds, and Formulas

### Flip density
- **Definition**: Fraction of bars where the signal *sign* changes from the previous nonzero sign.
- **Handling of 0**: Transitions into or out of 0 do not count as a “flip”; only +1↔-1 (or -1↔+1) count. Consecutive 0s are ignored for flip counting.
- **Formula**: `flip_density = (number of bars with sign change from previous nonzero) / (number of bars with current nonzero signal)`; if denominator is 0, use 0.

### Persistence (run length)
- **Definition**: For nonzero signal values, the distribution of consecutive run lengths (number of bars the signal stays in +1 or -1 before changing or going to 0).
- **Output**: Summary stats (mean, median, percentiles) of run lengths per indicator/param.

### Scratch (single definition)
- **Definition**: A trade is a **scratch** if either:
  - **Hold ≤ 3 bars**: For daily data, `(exit_date - entry_date)` in calendar days ≤ 3, or
  - **Realized R in a small band around 0**: `|realized_pnl / (risk_at_entry)| ≤ 0.1` (0.1R band).
- We use **one** definition per run; document which in the run (default: hold ≤ 3 bars for Phase B diagnostics).

### Scratch clustering
- **Definition**: 3+ scratches in a rolling window of 30 trades.
- **Index**: Count of such windows (or fraction of trade sequence covered by at least one such cluster). Document exact formula in `quality_gate` (e.g. rolling 30-trade window, count windows with ≥3 scratches).

### MAE (Max Adverse Excursion) in R
- **Definition**: Maximum adverse excursion before exit, normalized by initial risk (R). Use existing trade audit fields if present; otherwise compute from price path consistent with engine (intrabar high/low vs SL/entry).
- **Tail**: Distribution of MAE for scratch/loss trades; “MAE tail” = high percentiles (e.g. 90th, 95th).

### Elasticity
- **Definition**: Step deltas across *adjacent* parameter values for: trade count, scratch rate, hold time (e.g. mean hold bars).
- **Categories**:
  - **Dead**: No material change in metric across step (e.g. Δ trade count ≈ 0).
  - **Cliff**: Very large change in one step (e.g. Δ trade count > 50% of max).
  - **Meaningful step**: Moderate, smooth change.
- **Metrics**: Percent of steps that are dead, cliff, or meaningful (document exact thresholds in config or code in one place).

### Directional symmetry sanity
- For C1: Long vs short signal counts or trade counts should not be wildly asymmetric (e.g. ratio within a bounded range). Thresholds in quality_gate (e.g. ratio between 0.25 and 4.0).

### Volume veto
- **Invariant**: Volume-on entries must be a **subset** of volume-off entries (same C1 baseline). No new trades created by volume; only veto.
- **Smoothness / selectivity**: Participation rate (fraction of volume-off entries that pass volume gate); avoid binary cliffs across param steps.

## Controlled Overfit Protocol

- **Input**: Predefined “diagnostic fold pairs” from config: each pair has **Fit window** and **Check window** (e.g. Fold N fit, Fold N+1 check). At least 3 transitions.
- **On Fit window**: Run a **bounded dense param search**. Select “best-ish” by **stability proxy** (not ROI alone).  
  **Stability proxy formula** (document in code and here):
  - `stability = -w_dd * max_drawdown_pct - w_scratch * scratch_cluster_penalty + w_trades * trade_count_sanity`
  - Where: `scratch_cluster_penalty` = count of rolling-30-trade windows with ≥3 scratches; `trade_count_sanity` = 1 if total_trades in [min_trades, max_trades] else 0 (configurable bounds). Weights `w_dd`, `w_scratch`, `w_trades` in config or single place in code.
- **Freeze** selected params and evaluate on **Check window** (no further tuning).
- **Repeat** for each fold pair; write convergence metrics (param region stability) and behavior preservation metrics.
- **Explicit “not shippable” clause**: Outputs of the controlled-overfit diagnostic are for structural/robustness assessment only. They are **not** used to select live systems or to rank indicators for deployment.

## Output Structure

```
results/phaseB/
  c1_diagnostics/
    <c1_name>/
      response_curves.csv
      signal_stats.json
      scratch_mae.csv
  volume_diagnostics/
    <volume_name>/
      veto_response_curves.csv
      on_off_comparison.csv
      mae_tail.csv
  overlap/
    c1_overlap_matrix.csv
    c1_leadlag_summary.csv
  controlled_overfit/
    <indicator_name>/
      overfit_pairs.csv
      overfit_summary.csv
  quality_gate.csv
  approved_pool.json
  approved_pool.md
```

## How to Run

1. **C1 diagnostics** (response curves, overlap, scratch/MAE):
   ```bash
   python scripts/phaseB_run_diagnostics.py --config configs/phaseB/phaseB_c1_diagnostics.yaml
   ```

2. **Volume diagnostics** (veto response, ON vs OFF, MAE tail):
   ```bash
   python scripts/phaseB_run_diagnostics.py --config configs/phaseB/phaseB_volume_diagnostics.yaml
   ```

3. **Controlled overfit** (fold-pair fit/check, stability proxy):
   ```bash
   python scripts/phaseB_run_controlled_overfit.py --config configs/phaseB/phaseB_controlled_overfit.yaml
   ```

4. **Quality gate** (aggregate all Phase B outputs, produce gate table and approved pool):
   ```bash
   python -m analytics.phaseB_quality_gate --input results/phaseB --output results/phaseB
   ```

## Interpreting PASS / REDESIGN / DISCARD

- **PASS**: Indicator meets all quality thresholds (trade-count sanity, flip density, persistence, elasticity not dead/cliff-dominated, directional symmetry, scratch clustering, volume veto subset/smoothness where applicable). Eligible for Phase C.
- **REDESIGN**: Structural or controllability issues (e.g. too many dead/cliff steps, poor persistence, or overfit diagnostic shows param instability) but not fundamentally broken. Improve parametrization or logic before Phase C.
- **DISCARD**: Fundamental issues (e.g. leakage, binary cliff, violation of volume veto subset, or consistently failing overfit checks). Not eligible for Phase C without substantial change.

Reasons are pipe-separated in the `reasons` column of `quality_gate.csv`; `key_metrics_json` holds compact numeric metrics used for the decision.
