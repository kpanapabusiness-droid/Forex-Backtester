# Results Schema Audit Report
## Standardize Results Schema for Max Comparability

**Date**: 2025-01-25  
**Objective**: Identify why baseline vs volume outputs fail to join and propose canonical schema

---

## 1. Result Producers Map

### 1.1 Core Backtester (`core/backtester.py`)

**Output Files**:
- `results/<run_dir>/trades.csv`
- `results/<run_dir>/summary.txt`
- `results/<run_dir>/equity_curve.csv`

**trades.csv Schema** (from `TRADES_COLS`):
- Identity: `pair`, `entry_date`, `entry_idx`
- Entry: `entry_price`, `direction`, `direction_int`, `atr_at_entry_price`, `atr_at_entry_pips`
- Risk: `lots_total`, `lots_half`, `lots_runner`, `risk_pct_used`
- Levels: `tp1_price`, `sl_price`, `tp1_at_entry_price` (immutable), `sl_at_entry_price` (immutable)
- Exit: `exit_date`, `exit_price`, `exit_reason`, `sl_at_exit_price`
- Results: `pnl`, `win`, `loss`, `scratch`, `spread_pips_used`
- **Missing**: `timeframe`, `c1`, `exit`, `volume`, `from_date`, `to_date`, `run_id`, `config_hash`

**summary.txt Schema**:
- Text format with key-value pairs
- Metrics: `total_trades`, `wins`, `losses`, `scratches`, `Win% (NS)`, `ROI ($)`, `ROI (%)`
- **Missing**: Identity keys (pair, timeframe, strategy components)

**equity_curve.csv Schema**:
- `date`, `equity`, `peak`, `drawdown`
- **Missing**: Identity keys

---

### 1.2 Batch Sweeper (`scripts/batch_sweeper.py`)

**Output Files**:
- `results/c1_batch_results.csv` (consolidated)
- `results/results_history/<run_slug>/trades.csv` (per-run)
- `results/results_history/<run_slug>/summary.txt` (per-run)
- `results/results_history/<run_slug>/equity_curve.csv` (per-run)

**c1_batch_results.csv Schema**:
- Identity: `run_slug` (e.g., "c1-coral__20260125_145522"), `timestamp`
- Strategy: `roles` (JSON: `{"c1": "coral", "c2": null, "baseline": null, "volume": null, "exit": null}`)
- Params: `params` (JSON)
- Metrics: `total_trades`, `wins`, `losses`, `scratches`, `win_rate_ns`, `loss_rate_ns`, `scratch_rate_tot`, `win_rate`, `loss_rate`, `scratch_rate`, `roi_dollars`, `roi_pct`, `max_dd_pct`, `expectancy`, `score`
- **Missing**: `pair`, `timeframe`, `from_date`, `to_date` (lost at consolidation stage)

**Root Cause**: `batch_sweeper.py` aggregates across ALL pairs in `static_overrides.pairs` but does NOT include pair in consolidated CSV. Pair info exists only in archived configs.

---

### 1.3 Volume Sweep (`scripts/run_c1_volume_sweep.py`)

**Output Files**:
- `results/c1_w_exit_plus_vol_results/<run_slug>/trades.csv`
- `results/c1_w_exit_plus_vol_results/<run_slug>/summary.txt`
- `results/c1_w_exit_plus_vol_results/<run_slug>/equity_curve.csv`

**Run Slug Format**: `c1_<c1_name>__vol_<volume_name>__exit_<exit_name>__pair_<pair>`

**Per-Run Outputs**: Same as core backtester (trades.csv, summary.txt, equity_curve.csv)

**Identity Preserved**: Pair is in run_slug, but not in CSV columns until aggregation.

---

### 1.4 C1 Sweep All Pairs (`scripts/run_c1_sweep_all_pairs.py`)

**Output Files**:
- `results/<results_root>/<pair>__<c1_name>__<timestamp>/trades.csv`
- `results/<results_root>/aggregated_results.csv`

**aggregated_results.csv Schema**:
- Identity: `pair`, `c1_name`, `run_slug`, `run_dir`
- Metrics: `total_trades`, `wins`, `losses`, `scratches`, `win_rate_ns`, `roi_pct`, `max_dd_pct`, `sharpe`, `sortino`
- **Missing**: `timeframe`, `from_date`, `to_date`, `exit`, `volume`

---

### 1.5 Walk-Forward (`scripts/walk_forward.py`)

**Output Files**:
- `results/<wfo_dir>/wfo_folds.csv` (per fold)
- `results/<wfo_dir>/wfo_summary.csv` (aggregated)

**wfo_folds.csv Schema**:
- Identity: `fold_idx`, `train_start`, `train_end`, `test_start`, `test_end`
- Metrics: `trades`, `wins`, `losses`, `scratches`, `win_pct_ns`, `loss_pct_ns`, `expectancy`, `roi_pct`, `max_dd_pct`
- **Missing**: `pair`, `timeframe`, `c1`, `exit`, `volume` (strategy components not in fold-level CSV)

---

## 2. Aggregators Map

### 2.1 Aggregate C1 Results (`scripts/aggregate_c1_results.py`)

**Input**: Per-run directories under `results/<results_root>/`

**Output**: `results/<output_path>.csv`

**Schema**:
- Identity: `pair`, `c1_name`, `timeframe`, `from_date`, `to_date`, `run_id`, `output_dir`
- Metrics: `total_trades`, `wins`, `losses`, `scratches`, `win_rate_pct`, `roi_pct`, `max_dd_pct`, `sharpe`, `sortino`, `mar`, `expectancy`
- **Extraction Method**: Parses directory names and config.yaml files
- **Missing**: `exit`, `volume` (not extracted from directory structure)

---

### 2.2 Aggregate C1 Volume Results (`scripts/aggregate_c1_volume_results.py`)

**Input**: `results/c1_w_exit_plus_vol_results/<run_slug>/`

**Output**: `results/c1_volume_aggregated.csv` or `results/c1_volume_aggregated_full.csv`

**Schema**:
- Identity: `pair`, `c1`, `volume`, `exit`, `run_dir`
- Metrics: `total_trades`, `wins`, `losses`, `scratches`, `win_rate_ns`, `roi_pct`, `max_dd_pct`, `mar`, `expectancy`, `sharpe`, `sortino`
- **Extraction Method**: Parses run_slug pattern: `c1_<c1>__vol_<vol>__exit_<exit>__pair_<pair>`
- **Complete**: Has all required identity keys

---

### 2.3 Aggregate C1 Exit Baseline (`scripts/aggregate_c1_exit_baseline_from_batch.py`)

**Input**: `results/c1_batch_results.csv` + `results/results_history/<run_slug>/config.yaml`

**Output**: `results/c1_exit_baseline_aggregated.csv`

**Schema**:
- Identity: `pair`, `c1`, `exit`
- Metrics: `total_trades`, `wins`, `losses`, `scratches`, `roi_pct`, `max_dd_pct`, `expectancy`, `win_rate_ns`
- **Extraction Method**: 
  1. Extracts `c1` and `exit` from `roles` JSON in batch CSV
  2. Extracts `pair` from archived config.yaml files (fallback to sweeps.yaml)
- **Issue**: Pair extraction is fragile (depends on archived configs existing)

---

### 2.4 Aggregate V1 Results (`scripts/aggregate_v1_results.py`)

**Input**: `results/<wfo_v1_*>/wfo_folds.csv`

**Output**: `results/v1_folds_summary.csv`, `results/v1_leaderboard.csv`

**Schema**:
- Identity: `c1_name`, `volume_name`, `fold_idx`, `train_start`, `train_end`, `test_start`, `test_end`
- Metrics: `trades`, `wins`, `losses`, `scratches`, `win_pct_ns`, `loss_pct_ns`, `expectancy`, `roi_pct`, `max_dd_pct`
- **Missing**: `pair`, `timeframe` (WFO aggregates across pairs)

---

### 2.5 Aggregate WFO C1 Results (`scripts/aggregate_wfo_c1_results.py`)

**Input**: `results/<wfo_c1_*>/wfo_folds.csv`

**Output**: `results/wfo_c1_summary.csv`, `results/wfo_c1_leaderboard.csv`

**Schema**:
- Identity: `indicator` (C1 name), `fold_idx`, `train_start`, `train_end`, `test_start`, `test_end`
- Metrics: `trades`, `wins`, `losses`, `scratches`, `win_pct_ns`, `loss_pct_ns`, `expectancy`, `roi_pct`, `max_dd_pct`, `scratch_pct`
- **Missing**: `pair`, `timeframe`, `exit`, `volume`

---

## 3. Root Cause Analysis

### 3.1 Phase 1 Join Failure

**Baseline CSV** (`results/c1_exit_baseline_aggregated.csv`):
- Columns: `pair`, `c1`, `exit`, `total_trades`, `wins`, `losses`, `scratches`, `roi_pct`, `max_dd_pct`, `expectancy`, `win_rate_ns`
- Unique pairs: **12** (from sweeps.yaml static_overrides)
- Unique C1s: **54**

**Volume CSV** (`results/c1_volume_aggregated.csv`):
- Columns: `pair`, `c1`, `volume`, `exit`, `run_dir`, `total_trades`, `wins`, `losses`, `scratches`, `win_rate_ns`, `roi_pct`, `max_dd_pct`, `mar`, `expectancy`, `sharpe`, `sortino`
- Unique pairs: **28** (from data/daily/ discovery)
- Unique C1s: **6** (TOP_C1_NAMES hardcoded)

**Join Keys**: `['pair', 'c1', 'exit']`

**Failure Points**:
1. **Pair Mismatch**: Baseline has 12 pairs (sweeps.yaml), volume has 28 pairs (data discovery) → **12 vs 28 = no overlap**
2. **C1 Mismatch**: Baseline has 54 C1s (full universe), volume has 6 C1s (top performers) → **6 overlap, but pair mismatch prevents join**
3. **Missing Timeframe**: Neither CSV includes `timeframe` or `from_date`/`to_date`, so cannot verify date range compatibility

### 3.2 Where Identity Keys Are Lost

| Stage | Script | Lost Keys | Reason |
|-------|--------|-----------|--------|
| **Consolidation** | `batch_sweeper.py` | `pair`, `timeframe`, `from_date`, `to_date` | Consolidated CSV uses `run_slug` only; pair info not extracted |
| **Aggregation** | `aggregate_c1_exit_baseline_from_batch.py` | `timeframe`, `from_date`, `to_date` | Only extracts pair from configs; date range not preserved |
| **Volume Aggregation** | `aggregate_c1_volume_results.py` | `timeframe`, `from_date`, `to_date` | Run slug doesn't include dates; not extracted from configs |

### 3.3 Schema Inconsistencies

| Field | Batch Results | Baseline Agg | Volume Agg | Issue |
|-------|---------------|--------------|------------|-------|
| `pair` | ❌ Missing | ✅ Present | ✅ Present | Batch loses pair at consolidation |
| `timeframe` | ❌ Missing | ❌ Missing | ❌ Missing | Never captured |
| `from_date` | ❌ Missing | ❌ Missing | ❌ Missing | Never captured |
| `to_date` | ❌ Missing | ❌ Missing | ❌ Missing | Never captured |
| `c1` | ✅ In `roles` JSON | ✅ Present | ✅ Present | Consistent |
| `exit` | ✅ In `roles` JSON | ✅ Present | ✅ Present | Consistent |
| `volume` | ✅ In `roles` JSON | ❌ N/A (baseline) | ✅ Present | Consistent |
| `run_id` | ✅ `run_slug` | ❌ Missing | ❌ Missing | Lost in aggregation |
| `mar` | ❌ Missing | ❌ Missing | ✅ Present | Inconsistent metrics |

---

## 4. Canonical Results Schema Proposal

### 4.1 Required Identity Keys (Minimum for Joinability)

```yaml
identity_keys:
  # Market context
  pair: str                    # e.g., "EUR_USD"
  timeframe: str               # e.g., "D", "H4"
  
  # Date range
  from_date: str               # ISO format: "YYYY-MM-DD"
  to_date: str                 # ISO format: "YYYY-MM-DD"
  
  # Strategy components
  c1: str                      # e.g., "twiggs_money_flow" (short name, no prefix)
  c2: Optional[str]            # e.g., null or "disparity_index"
  baseline: Optional[str]      # e.g., null or "ema"
  volume: Optional[str]        # e.g., null or "adx"
  exit: Optional[str]          # e.g., "exit_twiggs_money_flow" or null
  
  # Run identification
  run_id: str                  # Unique run identifier (e.g., run_slug or UUID)
  config_hash: Optional[str]   # Hash of config for reproducibility (optional)
```

### 4.2 Required Metrics Columns

```yaml
metrics:
  # Counts
  total_trades: int
  wins: int
  losses: int
  scratches: int
  
  # Rates (percentages)
  win_rate_ns: float           # Win% on non-scratch trades
  loss_rate_ns: float           # Loss% on non-scratch trades
  scratch_rate_tot: float       # Scratch% of total trades
  
  # Performance
  roi_dollars: float            # Absolute ROI in account currency
  roi_pct: float                # ROI as percentage
  max_dd_pct: float             # Maximum drawdown percentage
  mar: float                    # MAR (ROI% / |MaxDD%|)
  expectancy: float             # Expectancy per trade ($)
  
  # Risk-adjusted (optional but recommended)
  sharpe: Optional[float]
  sortino: Optional[float]
```

### 4.3 Complete Canonical Schema

**File**: `results/schema/canonical_results_schema.yaml`

```yaml
version: "1.0"
description: "Canonical results schema for all backtest outputs"

required_identity_keys:
  - pair
  - timeframe
  - from_date
  - to_date
  - c1
  - exit
  - run_id

optional_identity_keys:
  - c2
  - baseline
  - volume
  - config_hash

required_metrics:
  - total_trades
  - wins
  - losses
  - scratches
  - win_rate_ns
  - roi_pct
  - max_dd_pct
  - expectancy

optional_metrics:
  - loss_rate_ns
  - scratch_rate_tot
  - roi_dollars
  - mar
  - sharpe
  - sortino
  - score

column_types:
  pair: str
  timeframe: str
  from_date: str  # ISO date format
  to_date: str    # ISO date format
  c1: str
  c2: str | null
  baseline: str | null
  volume: str | null
  exit: str | null
  run_id: str
  config_hash: str | null
  total_trades: int
  wins: int
  losses: int
  scratches: int
  win_rate_ns: float
  loss_rate_ns: float
  scratch_rate_tot: float
  roi_dollars: float
  roi_pct: float
  max_dd_pct: float
  mar: float
  expectancy: float
  sharpe: float | null
  sortino: float | null
  score: float | null

join_keys:
  primary: ["pair", "timeframe", "from_date", "to_date", "c1", "exit"]
  secondary: ["pair", "c1", "exit"]  # For date-agnostic comparisons
  volume_comparison: ["pair", "c1", "exit"]  # Phase 1 use case
```

**Location**: `results/schema/canonical_results_schema.yaml`

---

## 5. Migration Plan (Minimal Changes First)

### Step 1: Add Manifest + Enforce Required Keys in Consolidated CSV

**Target**: `scripts/batch_sweeper.py`

**Changes**:
1. Extract `pair`, `timeframe`, `from_date`, `to_date` from `merged_cfg` in `worker_job()`
2. Add these fields to `FIELDNAMES` in consolidated CSV
3. Include in row dict at line 490-510

**Impact**: Minimal (adds 4 columns to existing CSV)

**Files Modified**: `scripts/batch_sweeper.py` (1 file)

---

### Step 2: Unify Aggregators to Read from Canonical Schema

**Targets**: All aggregator scripts

**Changes**:
1. Create `scripts/schema_validator.py` with:
   - `validate_canonical_schema(df: pd.DataFrame) -> bool`
   - `normalize_to_canonical(df: pd.DataFrame) -> pd.DataFrame`
2. Update each aggregator to:
   - Call `normalize_to_canonical()` before writing output
   - Ensure required identity keys are present
   - Map existing columns to canonical names

**Impact**: Medium (touches 5 aggregator scripts)

**Files Modified**:
- `scripts/schema_validator.py` (new)
- `scripts/aggregate_c1_exit_baseline_from_batch.py`
- `scripts/aggregate_c1_volume_results.py`
- `scripts/aggregate_c1_results.py`
- `scripts/aggregate_v1_results.py`
- `scripts/aggregate_wfo_c1_results.py`

---

### Step 3: Deprecate Old Outputs Gradually

**Strategy**: 
1. Keep old outputs for 2-3 releases
2. Add deprecation warnings to old aggregators
3. Update documentation to reference canonical schema
4. Phase out old formats after migration complete

**Impact**: Low (documentation + warnings only)

---

## 6. Immediate Fix for Phase 1

**Quick Win**: Update `scripts/batch_sweeper.py` to include pair/timeframe/dates in consolidated CSV:

```python
# In worker_job(), extract identity from merged_cfg:
pair = (merged_cfg.get("pairs") or [""])[0] if merged_cfg.get("pairs") else ""
timeframe = merged_cfg.get("timeframe") or ""
date_range = merged_cfg.get("date_range") or {}
from_date = date_range.get("start") or ""
to_date = date_range.get("end") or ""

# In main(), add to row dict:
row = {
    "run_slug": run_slug,
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "pair": pair,                    # ADD
    "timeframe": timeframe,           # ADD
    "from_date": from_date,           # ADD
    "to_date": to_date,               # ADD
    "roles": json.dumps(role_names),
    # ... rest of metrics
}
```

**Then**: Update `aggregate_c1_exit_baseline_from_batch.py` to use pair from CSV instead of config extraction.

---

## 7. Summary Tables

### 7.1 Result Producers Summary

| Producer | Output Files | Identity Keys Present | Metrics Present | Missing Keys |
|----------|-------------|----------------------|-----------------|--------------|
| `core/backtester.py` | trades.csv, summary.txt, equity_curve.csv | `pair` (in trades.csv) | All core metrics | `timeframe`, `from_date`, `to_date`, `c1`, `exit`, `volume`, `run_id` |
| `scripts/batch_sweeper.py` | c1_batch_results.csv | `run_slug`, `roles` (JSON) | All metrics | `pair`, `timeframe`, `from_date`, `to_date` |
| `scripts/run_c1_volume_sweep.py` | Per-run CSVs | `pair` (in run_slug) | All metrics | `timeframe`, `from_date`, `to_date` (not in CSV) |
| `scripts/run_c1_sweep_all_pairs.py` | aggregated_results.csv | `pair`, `c1_name` | Core metrics | `timeframe`, `from_date`, `to_date`, `exit`, `volume` |
| `scripts/walk_forward.py` | wfo_folds.csv | `fold_idx`, date ranges | All metrics | `pair`, `timeframe`, strategy components |

### 7.2 Aggregators Summary

| Aggregator | Input | Output | Join Keys | Missing Keys |
|------------|-------|--------|-----------|--------------|
| `aggregate_c1_results.py` | Per-run dirs | comparison CSV | `pair`, `c1_name` | `exit`, `volume`, `timeframe` (optional) |
| `aggregate_c1_volume_results.py` | Volume sweep dirs | volume_aggregated.csv | `pair`, `c1`, `volume`, `exit` | `timeframe`, `from_date`, `to_date` |
| `aggregate_c1_exit_baseline_from_batch.py` | batch_results.csv + configs | baseline_aggregated.csv | `pair`, `c1`, `exit` | `timeframe`, `from_date`, `to_date` |
| `aggregate_v1_results.py` | WFO fold CSVs | v1_folds_summary.csv | `c1_name`, `volume_name`, `fold_idx` | `pair`, `timeframe` |
| `aggregate_wfo_c1_results.py` | WFO fold CSVs | wfo_c1_summary.csv | `indicator` (C1), `fold_idx` | `pair`, `timeframe`, `exit`, `volume` |

---

## 8. Recommendations

### 8.1 Immediate Actions (Phase 1 Fix)

1. **Update `batch_sweeper.py`** to include `pair`, `timeframe`, `from_date`, `to_date` in consolidated CSV
2. **Update `aggregate_c1_exit_baseline_from_batch.py`** to read pair from CSV instead of config extraction
3. **Re-run baseline aggregation** to get proper pair values

### 8.2 Short-Term (Schema Standardization)

1. **Create canonical schema YAML** at `results/schema/canonical_results_schema.yaml`
2. **Create schema validator** (`scripts/schema_validator.py`)
3. **Update all aggregators** to normalize to canonical schema before writing

### 8.3 Long-Term (Full Migration)

1. **Deprecate old output formats** with warnings
2. **Update documentation** to reference canonical schema
3. **Add schema validation tests** to CI
4. **Phase out legacy formats** after 2-3 releases

---

## 9. Files to Create/Modify

### New Files
- `results/schema/canonical_results_schema.yaml`
- `scripts/schema_validator.py`

### Files to Modify (Priority Order)
1. `scripts/batch_sweeper.py` (add identity keys to consolidated CSV)
2. `scripts/aggregate_c1_exit_baseline_from_batch.py` (use CSV pair instead of config extraction)
3. `scripts/aggregate_c1_volume_results.py` (add timeframe/dates)
4. `scripts/aggregate_c1_results.py` (normalize to canonical)
5. `scripts/aggregate_v1_results.py` (normalize to canonical)
6. `scripts/aggregate_wfo_c1_results.py` (normalize to canonical)

---

**End of Audit Report**




