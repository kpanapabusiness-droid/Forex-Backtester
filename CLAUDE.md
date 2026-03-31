# CLAUDE.md — Forex Backtester Project Brief

> Briefing for a new senior developer (or AI assistant) joining this project.
> The owner is a **complete beginner with coding and repos** — all instructions must be step-by-step, exact, and leave no guesswork. See `PROJECT_RULES.md` for the full delivery standard.

---

## Project Overview

This is a **rule-based Forex backtesting engine** built on the NNFX (No-Nonsense Forex) framework. It simulates multi-indicator trading strategies on historical daily/H1/H4 OHLCV data and produces performance metrics to evaluate strategy quality before any live deployment.

The goal is to **find robust, non-overfitted strategies** that hold up out-of-sample across multiple currency pairs and timeframes. The backtester is the core research tool — everything feeds into it or reads from it.

**What it is NOT:** There is no live trading, broker integration, or real-time data feed. It is a pure research and simulation system.

---

## Folder Structure

```
Forex-Backtester/
├── core/                        # Simulation engine (heart of the system)
│   ├── backtester.py            # Main entry point: loads data, runs simulation loop
│   ├── signal_logic.py          # Entry/exit signal generation (633 LOC)
│   ├── backtester_helpers.py    # Trade finalization helpers
│   └── utils.py                 # ATR calculation, pip conversion, summary helpers
│
├── indicators/                  # All trading indicators, organized by role
│   ├── confirmation_funcs.py    # C1 & C2 confirmation indicators → {-1, 0, +1}
│   ├── baseline_funcs.py        # Baseline/trend filters → df["baseline"] + {-1,0,+1}
│   ├── volume_funcs.py          # Volume/AD veto signals → {-1,0,+1} or {0,1}
│   ├── exit_funcs.py            # Exit rule indicators → {0,1} ONLY
│   └── legacy_rejected/         # Quarantined indicators (repaint-suspected)
│
├── analytics/                   # Performance metrics
│   ├── metrics.py               # Sharpe, Sortino, CAGR, max DD, MAR, skew, kurtosis
│   └── monte_carlo.py           # Trade-sequence shuffling for robustness testing
│
├── scripts/                     # Runners / entry points
│   ├── run_single_debug.py      # One backtest with CLI args (pair, date range, indicators)
│   ├── run_from_yaml.py         # YAML-driven single run
│   ├── smoke_test_selfcontained_v198.py  # Health check — run this first
│   ├── batch_sweeper.py         # Parallel parameter sweeps
│   ├── walk_forward.py          # Walk-forward optimization (WFO)
│   └── phase{B,C,D}*/           # Research scripts (parameter sensitivity, diagnostics)
│
├── configs/                     # YAML strategy configs
│   ├── config.yaml              # Main config (the one you edit)
│   ├── sweeps.yaml              # Batch sweep parameter ranges
│   ├── batch_config.yaml        # Aggregation settings for batch runs
│   ├── BACKTESTER_TEMPLATE.yaml # Canonical template with all options documented
│   └── phase*/                  # WFO and phase-specific configs
│
├── data/                        # Static market data (never auto-downloaded)
│   ├── daily/                   # Daily OHLCV CSVs — one file per pair (EUR_USD.csv, etc.)
│   ├── test/                    # Smaller test datasets (2009–2026)
│   └── external/                # External indicators (e.g., dbcvix_synth.csv)
│
├── results/                     # All output goes here (mostly gitignored)
│   ├── trades.csv               # Per-trade detail (30 columns)
│   ├── summary.txt              # Aggregate performance metrics
│   └── equity_curve.csv         # Time-series equity & drawdown
│
├── tests/                       # pytest suite (50+ test files)
│
├── validators_config.py         # Pydantic-based YAML config validation
├── validators_util.py           # Indicator contract enforcement
├── indicators_cache.py          # Parquet/feather caching layer
│
├── BACKTESTER_SCHEMA.json       # Full config schema
├── BACKTESTER_TEMPLATE.yaml     # Template with all defaults
├── BACKTESTER_AUDIT.md          # P&L pipeline audit / decision log
├── PROJECT_RULES.md             # Team playbook — read this early
└── docs/README.md               # User documentation
```

---

## Key Dependencies

```
Python 3.12+
pandas          # DataFrames & time series
numpy           # Numerical computing
pyyaml          # YAML config parsing
pydantic        # Config validation & typed models
pytest          # Test framework
ruff            # Linting & formatting
```

No broker APIs, no live data feeds, no external databases. All data is local CSV files.

---

## How to Run the Backtester

### 1. Smoke test (fastest validation — run this first after any change)

```bash
python scripts/smoke_test_selfcontained_v198.py -q --mode fast
```

Generates its own synthetic data. If this passes, the engine is healthy.

### 2. Single backtest from YAML config

```bash
python core/backtester.py configs/config.yaml
```

Outputs to `results/trades.csv`, `results/summary.txt`, `results/equity_curve.csv`.

### 3. Single backtest with CLI args (debug mode)

```bash
python scripts/run_single_debug.py \
  --pair EUR_USD \
  --from 2019-01-01 \
  --to 2022-12-31 \
  --timeframe D \
  --c1 rsi_2 \
  --c2 macd \
  --baseline ema_50
```

### 4. Batch parameter sweep (parallel)

```bash
python scripts/batch_sweeper.py configs/sweeps.yaml
```

Results aggregated to `results/batch_summary.csv`.

### 5. Walk-forward optimization (WFO)

```bash
python scripts/walk_forward.py -c configs/wfo_c1_smoothed_momentum.yaml
```

Outputs: `results/wfo_c1_<indicator>/wfo_folds.csv`, `oos_summary.txt`, `equity_curve.csv`.

### Disable indicator cache

```bash
FB_NO_CACHE=1 python core/backtester.py configs/config.yaml
```

---

## Data Format

**CSV files in `data/daily/`:**
```
date,open,high,low,close,volume
2010-01-02,1.4312,1.43425,1.42926,1.43036,1027
```

- Required: `open, high, low, close`
- Optional: `volume` (needed for volume indicators)
- Common column name variants are auto-normalized

**External data in `data/external/`:**
```
date,value
2010-01-02,15.2
```
Used for the DBCVIX volatility filter.

---

## Output Metrics & How to Interpret Them

### Trade Statistics

| Metric | What it means | Target |
|--------|---------------|--------|
| Win Rate | % of trades that hit TP1 (excl. scratches) | ≥ 45% |
| Scratch Rate | % of trades that exit at breakeven (C1 reversed pre-TP1) | < 25% |
| Avg Win / Avg Loss | Mean PnL on wins vs. losses | Ratio ≥ 1.2 |
| Expectancy | Expected PnL per trade (AvgWin×WinRate − AvgLoss×LossRate) | Positive |

### Return Metrics

| Metric | What it means | Target |
|--------|---------------|--------|
| ROI (%) | Total return over the period | ≥ 10% annualized |
| CAGR (%) | Annualized compound return | ≥ 10% |

### Risk Metrics

| Metric | What it means | Target |
|--------|---------------|--------|
| Max Drawdown | Worst peak-to-trough decline (negative %) | ≥ −20% |
| Sharpe Ratio | Return per unit of total volatility | ≥ 1.0 |
| Sortino Ratio | Return per unit of downside volatility only | ≥ 1.5 |
| MAR Ratio | CAGR / |Max Drawdown| — recovery rate | ≥ 1.5 |
| Volatility (ann.) | Annualized daily return std dev | < 10% |

### Distribution Metrics

| Metric | What it means | Target |
|--------|---------------|--------|
| Skewness | > 0 = right-tail wins (rare big wins); < 0 = left-tail losses | ≥ 0.3 |
| Kurtosis | > 3 = fat tails (surprise moves) | 3–5 |

### Quality Thresholds

**Minimum viable (continue researching):**
- Win rate ≥ 45%, Sharpe ≥ 0.5, Max DD ≥ −25%, Total trades ≥ 20

**Good (worth pursuing):**
- Win rate ≥ 52%, Sharpe ≥ 1.0, Max DD ≥ −15%, MAR ≥ 1.5, OOS Sharpe ≥ 0.6

**Excellent (live-ready candidate):**
- Win rate ≥ 55%, Sharpe ≥ 1.5, Max DD ≥ −10%, MAR ≥ 2.0, consistent across ≥ 3 pairs

**Red flags — reject immediately:**
- Win rate < 40%, Max DD > −40%, Sharpe < 0.3, any negative expectancy
- OOS Sharpe < (IS Sharpe × 0.7) — this means overfitting

---

## Architecture & Conventions

### NNFX Signal Stack

The engine implements NNFX (No-Nonsense Forex) rules with four indicator roles:

1. **C1 (primary confirmation)** — triggers the entry; must be ±1
2. **C2 (secondary confirmation)** — optional direction filter; must agree with C1
3. **Baseline** — price trend filter; close must be on correct side
4. **Volume** — veto signal; must be +1 (pass) to allow entry
5. **Exit** — optional early exit indicator

All four are optional except C1. Each has a strict function contract (see below).

### Indicator Contracts (never break these)

Every indicator is a pure function with a fixed signature:

```python
# C1 / C2
def c1_<name>(df: pd.DataFrame, *, signal_col: str = "c1_signal", **kwargs) -> pd.DataFrame:
    # Must write df[signal_col] ∈ {-1, 0, +1}
    return df

# Baseline
def baseline_<name>(df, *, signal_col: str = "baseline_signal", **kwargs) -> pd.DataFrame:
    # Must write df["baseline"] (numeric series) AND df[signal_col] ∈ {-1, 0, +1}
    return df

# Volume
def volume_<name>(df, *, signal_col: str = "volume_signal", **kwargs) -> pd.DataFrame:
    # Must write df[signal_col] ∈ {-1, 0, +1} or {0, 1}
    return df

# Exit
def exit_<name>(df, *, signal_col: str = "exit_signal", **kwargs) -> pd.DataFrame:
    # Must write df[signal_col] ∈ {0, 1} ONLY — no -1
    return df
```

`validators_util.py` enforces these contracts on every run.

### Immutable Audit Fields

Once a trade is opened, these fields are **permanently fixed** and must never change:
- `entry_price`, `entry_date`, `entry_idx`
- `tp1_at_entry_price`, `sl_at_entry_price` (ATR multiples frozen at entry)
- `atr_at_entry_price`, `atr_at_entry_pips`

Dynamic stops track in `current_sl`; the final value is recorded as `sl_at_exit_price`.

### Config-Driven Only

No hardcoded strategy parameters. Every tunable value lives in YAML configs. Pydantic (`validators_config.py`) validates the config before any data is loaded.

### No-Lookahead Guarantee

- Indicators are computed on bar `i-1` close, used to signal bar `i`
- Entries execute at bar `i` close
- Intrabar TP/SL arbitration is configurable (`tp_first` default)

### Three-Tier Exit Model

1. **Pre-TP1:** Check TP1 → SL → C1 reversal (scratch exit)
2. **Post-TP1:** Trailing stop → C1 reversal → breakeven
3. **Intrabar arbitration:** Configurable priority (default: TP first)

### TDD Loop (always follow this)

```
Write failing test → minimal patch → pytest -q green → open PR → CI green → merge
```

### CI Requirements (every PR must pass)

```bash
ruff check .
pytest -q
python scripts/smoke_test_selfcontained_v198.py -q --mode fast
```

---

## Tier-1 C1 Indicators (approved, non-repaint)

These have been vetted and are safe for WFO:

- `smoothed_momentum`
- `kalman_filter`
- `trend_akkam`
- `lwpi`
- `disparity_index`
- `twiggs_money_flow`

**Quarantined (suspected repaint — do not use):**
- `fisher_transform`, `laguerre`, `doda_stochastic`, `supertrend`, `ehlers_reverse_ema`, `turtle_trading_channel`

These live in `indicators/legacy_rejected/` and are excluded from WFO configs.

---

## Known Limitations

1. **No live trading** — backtester only; no MT4/MT5, Oanda, or other broker integration
2. **Fixed spreads only** — no dynamic spread modeling; slippage is not modeled beyond entry/exit
3. **No liquidity constraints** — assumes any position size fills at any price
4. **No regime detection** — indicators are static; no built-in market regime switching
5. **Daily data is primary** — H1/H4 supported but less battle-tested
6. **Single-account simulation** — no portfolio correlation or multi-account modeling
7. **Repaint risk** — cache system assumes indicators are non-repainting; suspects quarantined but new indicators need manual audit

---

## Active Research Areas (TODOs)

- Dynamic spread model (currently only fixed pips)
- Regime filter (market structure detection in `signal_logic.py`)
- Monte Carlo with correlation matrix (currently independent shuffles)
- ML-style opportunity labeling (`docs/PHASE_D6F_CLEAN_LABELS.md`)
- Feature engineering pipeline (`docs/PHASE_D2_LIFT_HARNESS.md`)
- Execution bar off-by-one validation (`docs/phase8_execution_truth.md`)
- Kurtosis overflow guard for extreme trade sequences (`analytics/metrics.py`)

---

## Tool Assignments (from PROJECT_RULES.md)

| Task | Tool |
|------|------|
| Bug fix / small patch (≤ 3 files, ≤ 100 LOC) | Cursor |
| New module / multi-file feature / refactor | Aider |
| Find where X is used across the repo | Cody |
| Architecture, acceptance criteria, analysis | ChatGPT |
| YAML, CI, docs | Cursor |

Rule: **one branch = one tool** — don't mix Cursor and Aider on the same branch.

---

## Key Documents to Read First

| Document | Why |
|----------|-----|
| `PROJECT_RULES.md` | Delivery standard, guardrails, tool assignments |
| `BACKTESTER_TEMPLATE.yaml` | All config options with defaults |
| `BACKTESTER_AUDIT.md` | How the P&L pipeline works internally |
| `docs/README.md` | Full user guide |
| `BACKTESTER_SCHEMA.json` | Config schema reference |
