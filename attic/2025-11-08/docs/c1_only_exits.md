### C1-only Exits Sweep

This sweep evaluates all current Confirmation-1 (C1) indicators with:

- Pure C1-driven exits (no exit indicator)
- Two-leg model: TP1 leg closes at TP1; runner leg has SL only
- Exit only by SL or C1 flip (no trailing stop, no baseline cross)
- No spread, no one-candle rule, no volume/DBCVIX/session filters, no baseline gating

Scope:
- Timeframe: D1
- Date window: 2016-01-01 → 2025-10-31
- Pairs: EUR_USD, GBP_USD, USD_JPY, AUD_USD, GBP_JPY, AUD_NZD, EUR_JPY

Outputs are written per combination under:
`results/c1_only_exits/<pair>/<c1>/` with `trades.csv`, `summary.txt`, and `equity_curve.csv`.

### Run

```bash
conda activate forex_backtester
python scripts/run_c1_only_exits.py --config sweeps/c1_only_exits.yaml
```

After completion, you can scan results with your preferred tooling. A simple consolidated CSV is also produced by the standard sweeper at `results/c1_batch_results.csv` when using `scripts/batch_sweeper.py`, but this helper writes directly to `results/c1_only_exits/` for easier per-combination inspection.

### Notes

- Classification follows the Golden Standard: TP1 hit → WIN; SL before TP1 → LOSS; system exits (C1 flip) before TP1 → SCRATCH.
- Leg A: TP1 + SL only (no breakeven; no trailing stop effect intended).
- Leg B (runner): SL or C1 flip; breakeven after TP1 is allowed; trailing stop is enabled and activates per project defaults (activation uses engine defaults so it does not interfere pre-TP1).
- Baseline/volume/session/exit indicators are fully disabled; only C1 determines entries and flips determine exits; spread=0; one-candle rule off.


