# Project Folder Structure

Updated structure after reorganization (v1.9.8):

```
Forex_Backtester/
├── analytics/                         # Analytics and performance modules
│   ├── metrics.py                     # Performance metrics calculations
│   └── monte_carlo.py                 # Monte Carlo analysis
├── archive/                           # Archived files and old configs
│   ├── backtester.bak                 # Backup of old backtester
│   ├── signal_logic.ipynb             # Old signal logic notebook
│   └── config_*.yaml                  # Various archived configurations
├── cache/                             # Indicator computation cache
├── configs/                           # All configuration files
│   ├── config.yaml                    # Main strategy configuration
│   ├── batch_config.yaml              # Batch processing settings
│   ├── sweeps.yaml                    # Parameter sweep definitions
│   └── config_smoke_autofix.yaml      # Smoke test configuration
├── core/                              # Core engine modules
│   ├── __init__.py                    # Package initialization
│   ├── backtester.py                  # Main backtesting engine
│   ├── signal_logic.py                # Trading signal logic
│   ├── backtester_helpers.py          # Helper functions for backtester
│   └── utils.py                       # Utility functions (ATR, equity, etc.)
├── data/                              # Market data storage
│   ├── daily/                         # Daily OHLCV data files
│   │   ├── EUR_USD.csv                # Individual currency pair data
│   │   ├── GBP_JPY.csv
│   │   └── ... (other pairs)
│   ├── external/                      # External data sources
│   │   └── dbcvix_synth.csv          # DBCVIX volatility data
│   └── test/                          # Test data for unit tests
├── docs/                              # Documentation (NEW)
│   ├── README.md                      # Main project documentation
│   ├── INDICATORS.md                  # Available indicators reference
│   └── FOLDER_STRUCTURE.md            # This file
├── indicators/                        # Trading indicator implementations
│   ├── __init__.py                    # Package initialization
│   ├── confirmation_funcs.py          # C1/C2 confirmation indicators
│   ├── baseline_funcs.py              # Baseline/trend indicators
│   ├── volume_funcs.py                # Volume filter indicators
│   └── exit_funcs.py                  # Exit signal indicators
├── notebooks/                         # Jupyter notebooks
│   ├── Auditor.ipynb                  # Trade audit and analysis
│   ├── C1_Func_Finder.ipynb           # C1 indicator testing
│   ├── batch_runner.ipynb             # Batch processing interface
│   ├── data_extraction.ipynb          # Data preparation tools
│   ├── test_runner.ipynb              # Testing and visualization
│   ├── visualizations.ipynb           # Results visualization
│   ├── data_prep/                     # Data preparation notebooks
│   └── indi_builder/                  # Indicator development notebooks
│       ├── baseline_funcs.ipynb
│       ├── confirmation_indicators.ipynb
│       ├── exit_indicators.ipynb
│       └── volume_indicators.ipynb
├── optimizations/                     # Optimization modules (legacy)
│   └── grid_search.py                 # Grid search optimization
├── results/                           # Backtest output directory
│   └── results_history/               # Historical results archive
├── scripts/                           # Executable scripts (NEW)
│   ├── __init__.py                    # Package initialization
│   ├── batch_sweeper.py               # Parallel parameter sweeping
│   ├── walk_forward.py                # Walk-forward optimization
│   ├── run_meta.py                    # Meta-analysis runner
│   └── smoke_test_selfcontained_v198.py # Comprehensive smoke test
├── tests/                             # Unit test suite
│   ├── conftest.py                    # Test configuration
│   ├── test_baseline_contract.py      # Baseline indicator tests
│   ├── test_resolver_and_pipeline_smoke.py # Pipeline tests
│   ├── test_signal_exits.py           # Signal logic tests
│   ├── test_smoke_end_to_end.py       # End-to-end tests
│   ├── test_smoke.py                  # Basic smoke tests
│   └── test_writer.py                 # File writing tests
├── tools/                             # Utility tools
│   └── compare_runs.py                # Compare backtest results
├── AGENTS.md                          # AI agent guidelines
├── PROJECT_RULES.md                   # Development rules and guidelines
├── folder_tree.txt                    # Text version of folder structure
├── indicators_cache.py                # Indicator caching system
├── pytest.ini                        # Pytest configuration
├── requirements-dev.txt               # Development dependencies
├── ruff.toml                          # Ruff linter configuration
├── validators_config.py               # Configuration validation
└── validators_util.py                 # Utility validation functions
```

## Key Changes in v1.9.8 Structure

### New Directories
- `core/` - Contains main engine modules (backtester, signal_logic, utils, helpers)
- `configs/` - All YAML configuration files consolidated here
- `scripts/` - Executable scripts for various operations
- `docs/` - Consolidated documentation in Markdown format

### Moved Files
- `backtester.py` → `core/backtester.py`
- `signal_logic.py` → `core/signal_logic.py`
- `utils.py` → `core/utils.py`
- `backtester_helpers.py` → `core/backtester_helpers.py`
- `config.yaml` → `configs/config.yaml`
- `batch_config.yaml` → `configs/batch_config.yaml`
- `sweeps.yaml` → `configs/sweeps.yaml`
- `batch_sweeper.py` → `scripts/batch_sweeper.py`
- `walk_forward.py` → `scripts/walk_forward.py`
- `smoke_test_selfcontained_v198.py` → `scripts/smoke_test_selfcontained_v198.py`

### Benefits of New Structure
- **Logical Grouping**: Related files are grouped together
- **Clear Separation**: Core engine, configs, and scripts are distinct
- **Better Imports**: Cleaner import paths with package structure
- **Documentation**: Centralized docs in Markdown format
- **Maintainability**: Easier to navigate and understand project layout

## Usage with New Structure

### Running Scripts
```bash
# Smoke test
python scripts/smoke_test_selfcontained_v198.py --mode fast

# Main backtest
python core/backtester.py configs/config.yaml

# Batch sweeping
python scripts/batch_sweeper.py configs/sweeps.yaml
```

### Imports in Code
```python
# Core modules
from core import backtester
from core.signal_logic import apply_signal_logic
from core.utils import calculate_atr

# Indicators
from indicators.confirmation_funcs import c1_rsi
from indicators.baseline_funcs import baseline_ema
```

This structure maintains backward compatibility while providing better organization and clearer module boundaries.
