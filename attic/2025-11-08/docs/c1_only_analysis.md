### C1-only Exits Analysis Toolkit

This toolkit summarizes, sanity-checks, and visualizes the completed C1-only exits sweep at `results/c1_only_exits/<PAIR>/<C1>/{trades.csv, summary.txt}`.

### Commands

Run the aggregator (builds all-runs table and leaderboards):

```bash
python tools/aggregate_c1_only.py --root results/c1_only_exits --outdir results
```

Run the stability scan (per-year profitability and monthly variability):

```bash
python analytics/stability_scan.py --root results/c1_only_exits --out results/stability/c1_only_exits_stability.csv --min-trades 100
```

Select candidates by joining leaderboard and stability:

```bash
python tools/select_candidates.py --lb results/leaderboard_c1_only_exits_overall.csv --stab results/stability/c1_only_exits_stability.csv --out results/candidates/c1_only_exits_candidates.csv --min-trades 300 --min-profitable-years 60
```

Plot top-N equity curves and drawdowns (dedup by C1 if needed):

```bash
python analytics/plot_equity_curves.py --from-leaderboard results/leaderboard_c1_only_exits_by_pair.csv --root results/c1_only_exits --top 12 --outdir results/plots
```

### Outputs

- `results/c1_only_exits_all_runs.csv`: pair×C1 rows, columns include `pair,c1,trades,roi_pct,win_rate_pct,avg_r,max_dd_pct,sharpe,sortino`.
- `results/leaderboard_c1_only_exits_by_pair.csv`: sorted within each pair by ROI desc, WinRate desc, MaxDD asc.
- `results/leaderboard_c1_only_exits_overall.csv`: averaged across pairs per C1, trades summed, sorted by ROI desc then MaxDD asc.
- `results/stability/c1_only_exits_stability.csv`: `pair,c1,trades,years,profitable_years_pct,monthly_cv,low_sample_flag`.
- `results/candidates/c1_only_exits_candidates.csv`: filters by `trades≥300`, `profitable_years_pct≥60`, non-NaN `monthly_cv`.
- `results/plots/c1_only_exits_topN_equity.png` and `..._drawdowns.png`.

### Interpreting Columns

- **roi_pct**: total ROI% from `summary.txt` (if available; else 0.0). Used for ranking.
- **win_rate_pct**: win rate % of non-scratch trades (if provided; else 0.0).
- **avg_r**: average R multiple (if provided; else 0.0).
- **max_dd_pct**: max drawdown % (if provided; else 0.0).
- **trades**: count; falls back to `trades.csv` row count if not present in summary.
- **profitable_years_pct**: % of years with positive total (R or PnL proxy).
- **monthly_cv**: coefficient of variation of monthly totals (std/mean); NaN if insufficient data or zero mean.

### Next

- Feed `results/candidates/c1_only_exits_candidates.csv` into Walk-Forward and Monte Carlo drivers.
- Prefer per-trade Monte Carlo (order reshuffle) using each candidate’s `trades.csv`.

### Assumptions about summary.txt

- Keys parsed if present: `ROI%`, `Win Rate %`, `Avg R`, `Max DD %`, `Sharpe`, `Sortino`, `Trades`.
- Symbols like `%` are stripped; missing values default to `0.0` (tolerant read).


