Released v2.0.0 on 2025-09-30 (Australia/Sydney)

# Forex Backtester v2.0.0 — Current Status

*Last updated: 2025-09-30*

## 1. Version & Scope

**Version**: v2.0.0 (stable baseline release)

This is a mature, production-ready NNFX-style forex backtesting system. The current version represents a stable baseline with no functional logic changes from v1.9.x series. Key characteristics:

- **Config-driven architecture**: All parameters sourced from YAML configurations
- **Strict indicator contracts**: Enforced function signatures and return value domains
- **Immutable audit fields**: Entry-time prices and levels never mutated post-entry
- **Cache-aware indicators**: Intelligent caching for expensive calculations
- **Comprehensive validation**: Multi-layer testing from unit to integration levels

## 2. Architecture & Data Flow

**High-level flow**: `configs/*.yaml` → `core/backtester.py` → indicators → `core/signal_logic.py` → trade simulation → output writers

### Core Components

1. **Configuration Layer** (`configs/`)
   - `config.yaml`: Main strategy configuration
   - `sweeps.yaml`: Parameter sweep definitions
   - `batch_config.yaml`: Batch processing settings

2. **Data Loaders** (`core/backtester.py`)
   - CSV file discovery and loading
   - OHLCV data normalization
   - ATR calculation pipeline

3. **Indicator Pipeline** (`indicators/` + `core/backtester_helpers.py`)
   - Role-based indicator discovery and execution
   - Cache-aware computation with configurable backends
   - Contract validation and signal coercion

4. **Signal Logic** (`core/signal_logic.py`)
   - Entry/exit signal generation from indicator outputs
   - One-Candle Rule and Pullback Rule enforcement
   - Bridge-Too-Far logic for baseline-triggered entries

5. **Trade Simulation** (`core/backtester.py`)
   - Intrabar execution with configurable priority (tp_first/sl_first/best/worst)
   - Partial fills (TP1) with breakeven and trailing stop logic
   - DBCVIX risk filtering and position sizing

6. **Output Writers**
   - `results/trades.csv`: Individual trade records with audit fields
   - `results/equity_curve.csv`: Time-series equity progression
   - `results/summary.txt`: Aggregated performance metrics

## 3. Indicator Contracts

All indicators follow strict contracts enforced by `core/backtester_helpers.py` and `validators_util.py`:

### Confirmation Indicators (C1/C2)
```python
def c1_<name>(df, *, signal_col="c1_signal", **kwargs) -> pd.DataFrame
def c2_<name>(df, *, signal_col="c2_signal", **kwargs) -> pd.DataFrame
```
- **Input**: DataFrame with OHLCV columns
- **Output**: Modified DataFrame with `df[signal_col] ∈ {-1, 0, +1}`
- **Signal Domain**: `-1` (short), `0` (neutral), `+1` (long)

### Baseline Indicators
```python
def baseline_<name>(df, *, signal_col="baseline_signal", **kwargs) -> pd.DataFrame
```
- **Input**: DataFrame with OHLCV columns
- **Output**: Modified DataFrame with:
  - `df["baseline"]`: Numeric price series (for pullback calculations)
  - `df[signal_col] ∈ {-1, 0, +1}`: Direction signal based on close vs baseline

### Volume Indicators
```python
def volume_<name>(df, *, signal_col="volume_signal", **kwargs) -> pd.DataFrame
```
- **Input**: DataFrame with OHLCV columns
- **Output**: Modified DataFrame with `df[signal_col] ∈ {-1, 0, +1}` or `{0, 1}` (pass/fail)

### Exit Indicators
```python
def exit_<name>(df, *, signal_col="exit_signal", **kwargs) -> pd.DataFrame
```
- **Input**: DataFrame with OHLCV columns
- **Output**: Modified DataFrame with `df[signal_col] ∈ {0, 1}` ONLY
- **Signal Domain**: `0` (hold), `1` (exit now)

## 4. Entry/Exit Logic Implementation

### Entry Logic
- **Primary Trigger**: C1 signal flip (`+1` for long, `-1` for short)
- **Confirmation Filters**: C2, volume, baseline directional agreement
- **Entry Rules**:
  - **One-Candle Rule**: Entry only on signal bar (mutually exclusive with Pullback)
  - **Pullback Rule**: Entry when price returns within 1 ATR of baseline
  - **Bridge-Too-Far**: Cancel baseline-triggered entries if C1 last signaled ≥ N days ago
- **Continuation Trades**: Configurable re-entries while position open

### Exit Logic
- **C1 Reversal**: Exit when C1 flips opposite to position direction
- **Baseline Cross**: Exit when price crosses baseline against position
- **Exit Indicator**: Exit when exit indicator signals (if enabled)
- **Take Profit**: TP1 at 1x ATR, partial close (half position)
- **Stop Loss**: Initial SL at 1.5x ATR from entry
- **Breakeven**: Move SL to entry price after TP1 hit
- **Trailing Stop**: Activate after +2x ATR move, trail at 1.5x ATR behind price

## 5. Risk & PnL Management

### Position Sizing
- **Base Risk**: Configurable percentage per trade (default: 2%)
- **ATR-Based Sizing**: Position size calculated from SL distance in ATR units
- **Overlap Filter**: Prevents simultaneous trades in correlated pairs

### Risk Filters
- **DBCVIX Integration**: Volatility regime filtering
  - **Reduce Mode**: Lower position size during high volatility periods
  - **Block Mode**: Prevent new entries during extreme volatility
  - **CSV Source**: External volatility data integration
  - **Status**: Present but disabled by default in configurations

### Spread Modeling
- **PnL-Only Impact**: Spreads affect profit/loss calculations only
- **Trade Count Invariant**: Spread settings never change entry/exit timing
- **Configurable Sources**: Per-pair fixed spreads or ATR-multiple dynamic spreads

## 6. Output Schema

### trades.csv Fields
**Core Lifecycle**:
- `pair`, `entry_date`, `entry_price`, `direction`, `direction_int`
- `exit_date`, `exit_price`, `exit_reason`

**Risk & Sizing**:
- `atr_at_entry_price`, `atr_at_entry_pips`, `lots_total`, `lots_half`, `lots_runner`
- `risk_pct_used`, `dbcvix_val`, `dbcvix_flag`

**Immutable Audit Fields** (set at entry, never mutated):
- `tp1_at_entry_price`, `sl_at_entry_price`

**Dynamic State**:
- `tp1_hit`, `breakeven_after_tp1`, `ts_active`, `ts_level`
- `sl_at_exit_price` (final stop level at exit)

**Results & Spread**:
- `pnl`, `win`, `loss`, `scratch`, `spread_pips_used`

**Empty File Behavior**:
When no trades are generated, `trades.csv` is still created with all standard column headers but zero data rows. This maintains backward compatibility with scripts and tools that expect the file to exist. The writer logs `[WRITE TRADES SKIP] reason=empty` followed by `[WRITE TRADES OK] wrote=0 path=... (empty file with headers)`.

### equity_curve.csv Fields
- `date`, `equity`, `peak`, `drawdown`
- Real-time equity progression with running drawdown calculation

### summary.txt Metrics
- Trade counts (total, wins, losses, scratches)
- Win/loss rates (non-scratch basis)
- ROI (dollars and percentage)
- Expectancy per trade
- Performance metrics (if equity curve available): Sharpe, Sortino, CAGR, Max DD, MAR

## 7. Scripts & Commands

### Core Operations
```bash
# Basic backtest
python core/backtester.py -c configs/config.yaml

# Walk-forward optimization
python scripts/walk_forward.py --config configs/config.yaml

# Batch parameter sweeps
python scripts/batch_sweeper.py --config configs/sweeps.yaml

# Self-contained smoke test (fast mode)
python scripts/smoke_test_selfcontained_v198.py --mode fast

# Self-contained smoke test (quiet)
python scripts/smoke_test_selfcontained_v198.py -q --mode fast

# Full smoke test
python scripts/smoke_test_selfcontained_v198.py --mode full
```

### Development & Validation
```bash
# Linting (gates CI)
ruff check .

# Test suite (gates CI)
pytest -q

# Test with custom config
python -m pytest tests/ -c /dev/null
```

## 8. Configuration Surface

### Main Config Groups (`config.yaml`)
- **pairs**: Currency pair list
- **timeframe**: Data frequency ("D" for daily)
- **data**: Directory paths and date ranges
- **indicators**: C1/C2/baseline/volume/exit selections with use flags
- **indicator_params**: Per-function parameter overrides
- **rules**: One-candle, pullback, bridge-too-far, continuation settings
- **entry**: ATR multiples for TP/SL levels
- **exit**: Exit condition toggles and trailing stop configuration
- **risk**: Position sizing, account currency, FX quotes
- **spreads**: Spread modeling configuration
- **tracking**: Output options (equity curve, summary stats)
- **cache**: Indicator caching settings
- **walk_forward**: WFO window definitions
- **monte_carlo**: MC analysis configuration

### Batch Processing (`sweeps.yaml`)
- **role_filters**: Which indicator roles to sweep
- **discover**: Auto-discovery flags per role
- **allowlist/blocklist**: Indicator filtering
- **default_params**: Parameter grids per role
- **static_overrides**: Config overrides for all runs
- **parallel**: Worker count and execution limits
- **scoring**: Composite score weighting

## 9. Tests & CI

### Test Coverage (`tests/`)
- **test_smoke.py**: Basic functionality verification
- **test_smoke_end_to_end.py**: Full backtest pipeline
- **test_baseline_contract.py**: Indicator contract validation
- **test_signal_exits.py**: Exit logic verification
- **test_resolver_and_pipeline_smoke.py**: Indicator discovery and execution
- **test_writer.py**: Output file generation
- **conftest.py**: Test fixtures and utilities

### CI Pipeline (`.github/workflows/ci.yml`)
**Gating Checks** (must pass for merge):
- `ruff check .` (linting)
- `pytest -q` (test suite)

**Non-Gating Checks**:
- Smoke test execution (informational)

**Environment**:
- Ubuntu latest with Python 3.12
- Pip cache enabled
- Clean pytest environment (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`)
- Discord webhook notifications on failure

## 10. Limitations & Assumptions

### Current Limitations
- **Repaint Detection**: Not yet implemented for indicator validation
- **Cache Behavior**: Cache invalidation relies on data/parameter hashing
- **Dataset Expectations**: Assumes clean OHLCV data with consistent timestamps
- **Currency Conversion**: FX quotes required for cross-currency PnL calculation
- **Intrabar Modeling**: Simplified intrabar execution (no tick-level simulation)

### Key Assumptions
- **Daily Timeframe**: Primary focus on daily bar data
- **NNFX Methodology**: Follows No Nonsense Forex indicator hierarchy
- **ATR-Based Sizing**: All position sizing and levels based on Average True Range
- **Pip-Based Spreads**: Spread costs modeled in pip units
- **Single-Leg Entries**: No complex order types or staged entries

### Performance Considerations
- **Memory Usage**: Full dataset loaded into memory for each pair
- **Computation Time**: Scales with indicator complexity and data volume
- **Cache Dependencies**: Indicator changes require cache invalidation
- **Parallel Execution**: Batch sweeps benefit from multi-core systems

## 11. Key File Structure

```
Forex_Backtester/
├── core/                    # Engine components
│   ├── backtester.py       # Main backtesting engine (v1.9.8)
│   ├── signal_logic.py     # Entry/exit signal generation
│   ├── backtester_helpers.py # Indicator pipeline & validation
│   └── utils.py            # ATR, equity, FX utilities
├── indicators/             # Trading indicators
│   ├── confirmation_funcs.py  # 60+ C1/C2 indicators
│   ├── baseline_funcs.py      # 20+ baseline indicators
│   ├── volume_funcs.py        # Volume/volatility filters
│   └── exit_funcs.py          # Exit signal indicators
├── analytics/              # Performance analysis
│   ├── metrics.py          # Sharpe, Sortino, CAGR, drawdown
│   └── monte_carlo.py      # Trade shuffling & bootstrapping
├── scripts/                # Execution scripts
│   ├── batch_sweeper.py    # Parallel parameter optimization
│   ├── walk_forward.py     # Out-of-sample validation
│   ├── run_meta.py         # Meta-analysis utilities
│   └── smoke_test_selfcontained_v198.py # Comprehensive testing
├── configs/                # Configuration files
│   ├── config.yaml         # Main strategy config
│   ├── sweeps.yaml         # Parameter sweep definitions
│   └── batch_config.yaml   # Batch processing settings
├── tests/                  # Test suite (7 test files)
├── results/                # Output directory
├── data/daily/             # Market data (OHLCV CSVs)
└── cache/                  # Indicator computation cache
```

---

**Status**: Production-ready with comprehensive testing and validation. All core functionality implemented and verified through automated CI pipeline.
