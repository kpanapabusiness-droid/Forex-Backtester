# Forex Backtester

See [docs/README.md](docs/README.md) for complete documentation.

## Repo Layout (after cleanup 2025-11-08)

- `core/` — Backtesting engine (`backtester.py`, `signal_logic.py`, `utils.py`)
- `indicators/` — Indicator modules (C1/C2, baseline, volume, exit functions)
- `analytics/` — Performance analysis (`metrics.py`, `monte_carlo.py`)
- `scripts/` — Entry point scripts:
  - `run_single_debug.py` — Single backtest debug runner
  - `run_from_yaml.py` — Run backtest from YAML config
  - `batch_sweeper.py` — Parallel parameter sweeps
  - `walk_forward.py` — Walk-forward optimization
  - `smoke_test_selfcontained_v198.py` — Comprehensive smoke test
- `configs/` — YAML configuration files (`config.yaml`, `sweeps.yaml`)
- `results/` — Output directory (only `c1_only_exits/**` is tracked; others ignored)
- `attic/` — Quarantined files from 2025-11-08 cleanup (temporary, reversible)
- `tests/` — Test suite (pytest)