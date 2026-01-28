# C1-Only Batch Sweep Guide (2023-2025)

## 1. C1 Indicators & Pairs Discovery

### 1.1 C1 Indicators (57 total)

The following C1 indicators are available for the sweep:

```
aroon, aso, band_pass_filter, bears_bulls_impulse, chandelier_exit, coral,
cyber_cycle, decycler_oscillator, disparity_index, doda_stochastic,
dpo_histogram, ease_of_movement, ehlers_cg, ehlers_deli, ehlers_eot,
ehlers_reverse_ema, ehlers_roofing_filter, ergodic_tvi, fisher_transform,
forecast, glitch_index, grucha_percentage_index, hacolt_lines, hlc_trend,
is_calculation, kalman_filter, kuskus_starlight, laguerre,
linear_regression_slope, lwpi, metro_advanced, perfect_trend_line,
polarized_fractal_efficiency, price_momentum_oscillator, rsi,
schaff_trend_cycle, sherif_hilo, sma_cross, smooth_step, smoothed_momentum,
supertrend, third_gen_ma, top_bottom_nr, top_trend, tp_trend_pivot,
trend_akkam, trend_continuation, trend_lord, trendilo, ttf,
turtle_trading_channel, twiggs_money_flow, vortex_indicator, vulkan_profit,
wpr_ma, zerolag_macd
```

### 1.2 Currency Pairs (28 total)

All pairs from `data/daily/`:

```
AUD_CAD, AUD_CHF, AUD_JPY, AUD_NZD, AUD_USD, CAD_JPY, CHF_JPY,
EUR_AUD, EUR_CAD, EUR_CHF, EUR_GBP, EUR_JPY, EUR_NZD, EUR_USD,
GBP_AUD, GBP_CAD, GBP_CHF, GBP_JPY, GBP_NZD, GBP_USD,
NZD_CAD, NZD_CHF, NZD_JPY, NZD_USD,
USD_CAD, USD_CHF, USD_JPY, USD_SGD
```

**Total combinations:** 28 pairs × 57 C1 indicators = **1,596 backtest runs**

---

## 2. Sweep Configuration

**File:** `configs/c1_only_sweep_2023_2025.yaml`

This YAML config defines:
- **Timeframe:** D (daily)
- **Date range:** 2023-01-01 to 2025-02-01
- **Fixed settings:** No baseline, no C2, no volume, no exit indicators
- **Risk:** 2% per trade, $10,000 starting balance
- **All C1 indicators:** Auto-discovered from `indicators/confirmation_funcs.py`
- **All pairs:** Discovered from `data/daily/` directory

---

## 3. Batch Run Command

Run the C1-only sweep over all pairs:

```bash
python scripts/run_c1_sweep_all_pairs.py --config configs/c1_only_sweep_2023_2025.yaml --results-root results/c1_sweep_2023_2025
```

**Options:**
- `--config`: Path to sweep config (default: `configs/c1_only_sweep_2023_2025.yaml`)
- `--results-root`: Root directory for all run results (default: `results/c1_sweep_2023_2025`)
- `--workers`: Number of parallel workers (default: auto = CPU count - 1)
- `--base-config`: Base config file (default: `configs/config.yaml`)

**Example with custom workers:**
```bash
python scripts/run_c1_sweep_all_pairs.py --config configs/c1_only_sweep_2023_2025.yaml --results-root results/c1_sweep_2023_2025 --workers 8
```

**What it does:**
1. Discovers all C1 indicators and pairs
2. Creates all (pair, C1) combinations
3. Runs backtests in parallel
4. Saves results to `results/c1_sweep_2023_2025/PAIR__C1_NAME__TIMESTAMP/`
5. Creates `results/c1_sweep_2023_2025/aggregated_results.csv` (intermediate)

---

## 4. Aggregator Script

**File:** `scripts/aggregate_c1_results.py`

After the batch completes, aggregate all results into a single comparison CSV:

```bash
python scripts/aggregate_c1_results.py --results-root results/c1_sweep_2023_2025 --output results/c1_comparison_2023_2025.csv
```

**What it does:**
1. Recursively scans `--results-root` for all run directories
2. Extracts metrics from `summary.txt` (or `trades.csv` as fallback)
3. Parses pair and C1 name from directory structure or config files
4. Creates a single CSV with one row per (pair, C1) combination
5. Sorts by pair, then by ROI descending

---

## 5. End-to-End Usage Guide

### Step 1: Activate Environment (if using virtual environment)

```bash
# If using conda:
conda activate forex_backtester

# If using venv:
# source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows
```

### Step 2: Run the C1-Only Sweep

```bash
python scripts/run_c1_sweep_all_pairs.py --config configs/c1_only_sweep_2023_2025.yaml --results-root results/c1_sweep_2023_2025
```

**Expected output:**
- Progress updates every 10 runs
- Final summary with total runs and duration
- Results saved to `results/c1_sweep_2023_2025/`

**Note:** With 1,596 runs, this may take several hours depending on your CPU. Monitor progress in the terminal.

### Step 3: Aggregate Results

Once the batch finishes, run the aggregator:

```bash
python scripts/aggregate_c1_results.py --results-root results/c1_sweep_2023_2025 --output results/c1_comparison_2023_2025.csv
```

**Expected output:**
- Number of runs found
- Top 5 by ROI preview
- Final CSV: `results/c1_comparison_2023_2025.csv`

### Step 4: View Results

Open the comparison CSV:

```bash
# View in terminal (first 20 rows)
head -20 results/c1_comparison_2023_2025.csv

# Or open in Excel/Google Sheets/LibreOffice
```

---

## 6. Final CSV Columns

The aggregated CSV (`c1_comparison_2023_2025.csv`) contains:

| Column | Description |
|--------|-------------|
| `pair` | Currency pair (e.g., EUR_USD) |
| `c1_name` | C1 indicator name (e.g., fisher_transform) |
| `timeframe` | Timeframe used (D) |
| `from_date` | Start date (2023-01-01) |
| `to_date` | End date (2025-02-01) |
| `total_trades` | Total number of trades executed |
| `wins` | Number of winning trades |
| `losses` | Number of losing trades |
| `scratches` | Number of scratch trades |
| `win_rate_pct` | Win rate % (non-scratch) |
| `roi_pct` | Return on Investment % |
| `max_dd_pct` | Maximum Drawdown % |
| `sharpe` | Sharpe ratio |
| `sortino` | Sortino ratio |
| `mar` | MAR ratio (CAGR / Max DD) |
| `expectancy` | Expectancy per trade ($) |
| `run_id` | Run identifier (directory name) |
| `output_dir` | Full path to run results directory |

**How to interpret:**
- **Sort by ROI:** Find best-performing combinations
- **Filter by pair:** Compare C1 indicators for a specific pair
- **Filter by C1:** See which pairs work best with a specific indicator
- **Check win_rate_pct:** High ROI with low win rate may indicate high risk
- **Check max_dd_pct:** Lower is better (less drawdown risk)
- **Check sharpe/sortino:** Higher is better (risk-adjusted returns)

**Example analysis:**
```python
import pandas as pd

df = pd.read_csv("results/c1_comparison_2023_2025.csv")

# Top 10 by ROI
print(df.nlargest(10, "roi_pct")[["pair", "c1_name", "roi_pct", "win_rate_pct", "max_dd_pct"]])

# Best C1 for EUR_USD
eur_usd = df[df["pair"] == "EUR_USD"].nlargest(5, "roi_pct")
print(eur_usd[["c1_name", "roi_pct", "win_rate_pct", "max_dd_pct"]])

# Best pairs for supertrend
supertrend = df[df["c1_name"] == "supertrend"].nlargest(5, "roi_pct")
print(supertrend[["pair", "roi_pct", "win_rate_pct", "max_dd_pct"]])
```

---

## Notes

- **Runtime:** With 1,596 runs, expect several hours of computation time
- **Storage:** Each run creates a results directory; ensure sufficient disk space
- **Parallelization:** Uses all CPU cores by default (minus 1); adjust with `--workers` if needed
- **Resume:** If interrupted, re-run the aggregator to collect completed runs; the sweep script will skip existing runs if you modify it (or manually delete incomplete runs)
- **Cache:** Indicator caching is enabled, so subsequent runs with same data will be faster

