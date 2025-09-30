# Forex Backtester v2.0.0

**Current Stable: v2.0.0 (2025-09-30)** | [Changelog](../CHANGELOG.md)

A comprehensive NNFX-style forex backtesting system with advanced features for testing trading strategies.

## 🎯 Purpose

This project helps you test and evaluate different trading indicators and systems using a fully customizable backtester.

**You can:**
- Test C1 indicators individually or in combination
- Add filters (C2, volume, baseline)
- Include full risk, exit, and continuation rules
- Run walk-forward optimization
- Perform Monte Carlo analysis
- Visualize performance and save results

📖 **For complete trading logic specification, see [Golden Standard Logic](GOLDEN_STANDARD_LOGIC.md)**

## 🎯 Hard-Stop Realism (Current Standard)

The backtester implements **Hard-Stop Realism** for exit arbitration:

- **Intrabar TP/SL/BE/TS touch → immediate exit** (highest priority)
- **Trailing Stop**: activation & updates on **closes only** (monotone, never loosens)
- **Breakeven (BE)**: becomes **effective the same bar** TP1 is first hit
- **System exits** (C1 reversal, baseline cross): only if no intrabar touch occurred
- **Pre-TP1 SCRATCH**: C1 reversal before TP1 → PnL ≈ 0 within tolerance

This aligns backtests with broker-style hard stops and avoids ambiguity between intrabar touches and close-level system signals.

## 📁 Project Structure

```
Forex_Backtester/
├── core/                           # Core engine modules
│   ├── backtester.py              # Main backtesting engine
│   ├── signal_logic.py            # Trading signal logic
│   ├── backtester_helpers.py      # Helper functions for backtester
│   └── utils.py                   # Utility functions (ATR, equity tracking, etc.)
├── configs/                       # Configuration files
│   ├── config.yaml               # Main strategy configuration
│   ├── batch_config.yaml         # Batch processing configuration
│   ├── sweeps.yaml               # Parameter sweep configuration
│   └── config_smoke_autofix.yaml # Smoke test configuration
├── scripts/                       # Executable scripts
│   ├── batch_sweeper.py          # Parallel parameter sweeping
│   ├── walk_forward.py           # Walk-forward optimization
│   ├── run_meta.py               # Meta-analysis runner
│   └── smoke_test_selfcontained_v198.py  # Comprehensive smoke test
├── indicators/                    # Trading indicators
│   ├── confirmation_funcs.py     # C1/C2 confirmation indicators
│   ├── baseline_funcs.py         # Baseline indicators
│   ├── volume_funcs.py           # Volume indicators
│   └── exit_funcs.py             # Exit indicators
├── data/                         # Market data
│   ├── daily/                    # Daily OHLCV data (1 file per pair)
│   └── external/                 # External data (e.g., DBCVIX)
├── results/                      # Output folder for current run
├── analytics/                    # Analytics modules
│   ├── metrics.py               # Performance metrics
│   └── monte_carlo.py           # Monte Carlo analysis
├── notebooks/                    # Jupyter notebooks
│   ├── test_runner.ipynb        # Testing and visualization
│   ├── batch_runner.ipynb       # Batch processing
│   └── visualizations.ipynb     # Results visualization
├── tests/                        # Unit tests
└── tools/                        # Utility tools
    └── compare_runs.py           # Compare backtest results
```

## 🚀 Quick Start

### 1. Run Smoke Test
```bash
python scripts/smoke_test_selfcontained_v198.py --mode fast
# Or with quiet output:
python scripts/smoke_test_selfcontained_v198.py -q --mode fast
```

### 2. Basic Backtest
```bash
python core/backtester.py configs/config.yaml
```

### 3. Batch Parameter Sweep
```bash
python scripts/batch_sweeper.py configs/sweeps.yaml
```

### 4. Walk-Forward Optimization
```bash
python scripts/walk_forward.py configs/config.yaml
```

## ⚙️ Configuration

### Main Config (configs/config.yaml)

```yaml
timeframe: daily
pairs: [EUR_USD, GBP_JPY, AUD_NZD, USD_CAD]

indicators:
  c1: rsi_2        # Primary confirmation
  c2: macd         # Secondary confirmation
  volume: ad       # Volume filter
  baseline: ema_50 # Baseline/trend filter
  exit: atr_stop   # Exit indicator

entry:
  one_candle_rule: true
  pullback_rule: true
  bridge_too_far_days: 7

exit:
  exit_on_c1_flip: true
  exit_on_baseline_cross: true
  trailing_stop: true

risk:
  risk_percent: 2.0
  overlap_filter: true
  dbcvix_filter: false

continuation:
  allow_continuation: true

tracking:
  equity_curve: true
  summary_stats: true
```

## 📊 Indicator Contracts

All indicators must follow specific contracts:

### Confirmation Indicators (C1/C2)
```python
def c1_rsi(df, *, signal_col="c1_signal", **kwargs):
    """
    Writes df[signal_col] with values {-1, 0, +1}
    Returns modified df
    """
    # Implementation here
    return df
```

### Baseline Indicators
```python
def baseline_ema(df, *, signal_col="baseline_signal", **kwargs):
    """
    Writes BOTH df["baseline"] and df[signal_col] {-1, 0, +1}
    Returns modified df
    """
    # Implementation here
    return df
```

### Volume Indicators
```python
def volume_ad(df, *, signal_col="volume_signal", **kwargs):
    """
    Writes df[signal_col] with values {-1, 0, +1}
    Returns modified df
    """
    # Implementation here
    return df
```

### Exit Indicators
```python
def exit_atr_stop(df, *, signal_col="exit_signal", **kwargs):
    """
    Writes df[signal_col] with values {0, 1} ONLY
    Returns modified df
    """
    # Implementation here
    return df
```

## 🧪 Testing & Validation

### Run All Tests
```bash
pytest -q
```

### Run Specific Test
```bash
pytest tests/test_smoke_end_to_end.py::test_smoke_backtest_outputs -v
```

### Linting
```bash
ruff check .
```

## 📈 Advanced Features

### Walk-Forward Optimization
- Automatically splits data into training/testing periods
- Optimizes parameters on training data
- Validates on out-of-sample data
- Generates comprehensive performance reports

### Monte Carlo Analysis
- Randomizes trade order to assess strategy robustness
- Provides confidence intervals for performance metrics
- Available in both per-trade and daily return modes

### Batch Processing
- Parallel execution of multiple parameter combinations
- Automatic result aggregation and ranking
- Configurable parameter sweeps via YAML

### DBCVIX Risk Filter
- Optional volatility-based risk management
- Reduce position size or block trades during high volatility
- Configurable thresholds and modes

## 🔧 Development

### Adding New Indicators
1. Add function to appropriate `indicators/*_funcs.py` file
2. Follow the indicator contract for that role
3. Test with smoke test: `python scripts/smoke_test_selfcontained_v198.py --mode fast` (add `-q` for quiet output)

### TDD Workflow
1. Write failing test first
2. Apply minimal implementation (≤3 files for Cursor; else use Aider)
3. Run `pytest` locally until green
4. Open PR and ensure CI passes

## 🏗️ Architecture

- **Config-Driven**: All parameters come from YAML, no hardcoded values
- **Indicator Contracts**: Strict interfaces ensure consistency
- **Immutable Audit Fields**: Trade entry prices never change
- **Cache-Aware**: Intelligent caching for expensive indicator calculations
- **Equity Tracking**: Real-time equity curve generation with drawdown analysis

## 📋 Requirements

- Python 3.12+
- pandas, numpy, pyyaml
- pytest (for testing)
- ruff (for linting)

See `requirements-dev.txt` for full dependency list.

## 🚨 Safety & Validation

- No lookahead bias in signals or trades
- Safeguards for mathematical operations (log/div)
- Comprehensive test suite with smoke tests
- CI/CD pipeline with GitHub Actions
- Discord webhook notifications for failures

---

For detailed implementation notes and advanced usage, see the additional documentation files in this directory.
