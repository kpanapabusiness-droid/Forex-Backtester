# Forex Backtester - AI Coding Agent Guide

## Build & Test Commands
- **Lint**: `ruff check .` (gates CI, line-length=100, Python 3.12)
- **Test (all)**: `pytest -q` or `python -m pytest -q -c /dev/null`
- **Test (single)**: `pytest tests/test_<module>.py::test_function -v`
- **Smoke test**: `python smoke_test_full_v198.py` (non-gating)
- **CI gating**: ruff + pytest (smoke is non-gating)

## Architecture & Structure
- **Core Engine**: `backtester.py` (main backtesting execution)
- **Signal Logic**: `signal_logic.py` (trading rules & entry/exit logic)
- **Indicators**: `indicators/` (baseline, confirmation, exit, volume functions)
- **Config-driven**: Uses `config.yaml`, `sweeps.yaml`, `batch_config.yaml`
- **Results**: Output to `results/` (trades, summary, equity_curve CSV/TXT files)
- **Data**: Market data in `data/`, cached computations in `cache/`

## Code Style & Conventions
- **Python 3.12**, line length 100, ruff linting (E, F, I rules)
- **Config-driven only**: No hardcoded params, everything in YAML configs
- **Indicator contracts**: Return df, write {-1,0,+1} to signal_col, baseline writes df["baseline"]
- **TDD approach**: Write failing test first, then minimal implementation
- **Immutable audit fields**: tp1_at_entry_price, sl_at_entry_price never change
- **Exit logic**: Use current_sl for dynamic stops, set sl_at_exit_price on exit
- Every PR must add/update tests and pass CI (ruff + pytest)
