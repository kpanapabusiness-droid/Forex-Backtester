# Phase C — C1 Identity WFO v2

## Purpose

Phase C runs **identity-only** walk-forward evaluation for the 6 approved C1 indicators from Phase B.1. There is **no ranking obsession**: each identity is judged PASS or REJECT by worst-fold gates only. No parameter sweeps, no optimization, no volume/baseline/C2/exit research.

## System (locked)

- **Entry**: C1 only; entry on flip into +1 or -1.
- **Exit**: Full directional flip only (+1 → -1 or -1 → +1), next-open semantics. Neutral 0 never forces exit.
- **Stops**: Fixed SL (existing fixed SL mechanism). No new stop types.
- **Spreads**: ON. Phase A realism is immutable.

## Folds (canonical 4)

| Fold | Train start | Train end   | Test start  | Test end    |
|------|-------------|------------|-------------|-------------|
| 1    | 2019-01-01  | 2020-12-31 | 2021-01-01  | 2021-12-31  |
| 2    | 2019-01-01  | 2021-12-31 | 2022-01-01  | 2022-12-31  |
| 3    | 2019-01-01  | 2022-12-31 | 2023-01-01  | 2023-12-31  |
| 4    | 2019-01-01  | 2023-12-31 | 2024-01-01  | 2025-12-31  |

## Input universe (6 C1s)

- `c1_persist_momo__binary`
- `c1_persist_momo__neutral_gate`
- `c1_regime_sm__binary`
- `c1_regime_sm__neutral_gate`
- `c1_vol_dir__binary`
- `c1_vol_dir__neutral_gate`

The list is read from `results/phaseB1/approved_pool.json` (key `C1`) if present; otherwise from `configs/phaseC/phaseC_c1_identities.yaml` (`c1_identities`). The runner enforces exactly these six; no extras, no unknowns.

## Output files and locations

| Output | Location |
|--------|----------|
| Leaderboard (PASS/REJECT) | `results/phaseC/aggregate/leaderboard_c1_identity.csv` |
| Survivors justification | `results/phaseC/aggregate/phaseC_survivors.md` |
| Overlap matrix (optional) | `results/phaseC/aggregate/overlap_matrix.csv` |
| Per-identity WFO runs | `results/phaseC/wfo_runs/<c1_name>/<run_id>/fold_XX/` |

## PASS/REJECT rules

**Hard rejects**

- **zero-trade collapse**: any fold OOS trades == 0 → REJECT  
- **trade starvation**: worst-fold OOS trades < 300 → REJECT  
- **regime collapse**: worst-fold trades < 20% of median-fold trades → REJECT  
- **catastrophic DD**: worst-fold max_dd ≥ 25% (0.25) → REJECT  

**Soft gate (all required for PASS)**

- Worst-fold ROI > -5% (-0.05) else REJECT  
- Worst-fold scratch_rate ≤ 0.65 else REJECT  
- Worst-fold max_dd ≤ 1.5 × median_fold_max_dd else REJECT (median=0 handled: gate not applied)  

If fewer than 5 identities PASS, `phaseC_survivors.md` must state:  
**"Phase C failed: <N> survivors < 5. Return to Phase B.1 Round 2."**

## No ranking obsession

Phase C does not rank identities. It only assigns PASS or REJECT per identity using the gates above. Leaderboard CSV columns are for audit and reporting; the decision is binary.

## Config and scripts

- **Configs**: `configs/phaseC/` — `phaseC_base.yaml`, `phaseC_wfo_shell.yaml`, `phaseC_c1_identities.yaml`
- **Runner**: `scripts/phaseC/run_phaseC_c1_identity_wfo.py`
- **Aggregation**: `analytics/phaseC/phaseC_leaderboard.py` (runnable with `python -m analytics.phaseC.phaseC_leaderboard`)

## How to run Phase C

From repo root:

```bash
# 1) Run WFO v2 once per C1 identity (writes to results/phaseC/wfo_runs/<c1_name>/)
python scripts/phaseC/run_phaseC_c1_identity_wfo.py --wfo-shell configs/phaseC/phaseC_wfo_shell.yaml --results-root results/phaseC

# 2) Build leaderboard and survivors (writes to results/phaseC/aggregate/)
python -m analytics.phaseC.phaseC_leaderboard --results-root results/phaseC
```

Optional: use `--approved-pool results/phaseB1/approved_pool.json` if not using the default path; use `--no-skip` to re-run all identities even when `wfo_done.json` exists.

