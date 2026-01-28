# Repository Cleanup Plan — 2025-11-08

**Date**: 2025-11-08  
**Branch**: `cleanup/repo-slim-2025-11-08`  
**Strategy**: Quarantine unused files to `attic/` (reversible), purge non-c1_only_exits results

---

## 1. Kept — Referenced by Imports/Entrypoints/CI

### Core Engine Modules
| Path | Why Kept | Reference |
|------|----------|-----------|
| `core/backtester.py` | Main backtesting engine | Imported by all scripts, tests |
| `core/signal_logic.py` | Entry/exit signal generation | Imported by `backtester.py` |
| `core/backtester_helpers.py` | Indicator pipeline & validation | Imported by `backtester.py` |
| `core/utils.py` | ATR, equity, FX utilities | Imported by `backtester.py`, `run_single_debug.py |
| `core/__init__.py` | Package marker | Python import system |

### Indicator Modules
| Path | Why Kept | Reference |
|------|----------|-----------|
| `indicators/confirmation_funcs.py` | C1/C2 indicators | Imported by `backtester.py`, `batch_sweeper.py`, `indicators_cache.py` |
| `indicators/baseline_funcs.py` | Baseline indicators | Imported by `backtester.py`, `indicators_cache.py` |
| `indicators/volume_funcs.py` | Volume indicators | Imported by `backtester.py`, `indicators_cache.py` |
| `indicators/exit_funcs.py` | Exit indicators | Imported by `backtester.py`, `indicators_cache.py` |
| `indicators/__init__.py` | Package marker | Python import system |
| `indicators_cache.py` | Indicator caching layer | Imported by `backtester.py`, `run_single_debug.py` |

### Analytics Modules
| Path | Why Kept | Reference |
|------|----------|-----------|
| `analytics/metrics.py` | Performance metrics (Sharpe, Sortino, CAGR) | Imported by `batch_sweeper.py`, `core/utils.py`, smoke test |
| `analytics/monte_carlo.py` | Monte Carlo analysis | Imported by `walk_forward.py` (optional) |
| `analytics/plot_equity_curves.py` | Equity curve plotting | Standalone utility (may be used) |
| `analytics/stability_scan.py` | Stability analysis | Standalone utility (may be used) |

### Authoritative Entry Point Scripts
| Path | Why Kept | Reference |
|------|----------|-----------|
| `scripts/run_single_debug.py` | Single backtest debug runner | User-facing CLI, referenced in docs |
| `scripts/run_from_yaml.py` | Run backtest from YAML config | User-facing CLI |
| `scripts/batch_sweeper.py` | Parallel parameter sweeps | User-facing CLI, imports core engine |
| `scripts/walk_forward.py` | Walk-forward optimization | User-facing CLI, imports core engine |
| `scripts/smoke_test_selfcontained_v198.py` | Comprehensive smoke test | CI smoke test (non-gating), referenced in docs |
| `scripts/__init__.py` | Package marker | Python import system |

### Config & Validation
| Path | Why Kept | Reference |
|------|----------|-----------|
| `validators_config.py` | Config validation | Imported by `backtester.py` |
| `validators_util.py` | Contract validation | Imported by `backtester.py`, smoke test |
| `BACKTESTER_TEMPLATE.yaml` | Config template | Schema reference |
| `BACKTESTER_SCHEMA.json` | JSON schema | Schema validation |
| `BACKTESTER_AUDIT.md` | Audit documentation | Documentation |
| `configs/config.yaml` | Main strategy config | Used by all scripts |
| `configs/sweeps.yaml` | Parameter sweep definitions | Used by `batch_sweeper.py` |
| `configs/batch_config.yaml` | Batch processing settings | May be used by batch scripts |

### CI & Test Infrastructure
| Path | Why Kept | Reference |
|------|----------|-----------|
| `pytest.ini` | Pytest configuration | CI runs `pytest -q` |
| `ruff.toml` | Ruff linting config | CI runs `ruff check .` |
| `.gitignore` | Git ignore rules | Version control |
| `requirements-dev.txt` | Development dependencies | CI environment setup |

### Documentation (Essential)
| Path | Why Kept | Reference |
|------|----------|-----------|
| `README.md` | Project overview | Repository entry point |
| `docs/README.md` | Complete documentation | Referenced by README.md |
| `docs/STATUS.md` | Current status | Project status reference |
| `CHANGELOG.md` | Version history | Project history |
| `PROJECT_RULES.md` | Project rules | Development guidelines |
| `AGENTS.md` | Agent guidelines | AI agent context |

---

## 2. Quarantine Candidates — Not Referenced

### Legacy/Experimental Scripts
| Path | Reason | Evidence |
|------|--------|---------|
| `scripts/run_meta.py` | Minimal utility (only `make_run_id`), not imported | No imports found, minimal functionality |
| `scripts/run_c1_only_exits.py` | Specialized script, not core entrypoint | Only used for specific C1-only analysis |
| `scripts/export_mt5_daily.py` | Data export utility, not core engine | Standalone data export tool |
| `scripts/mt5_compare.py` | MT5 comparison utility | Only used for MT5 parity validation |
| `scripts/mt5_parity_run.py` | MT5 parity runner | Only used for MT5 parity validation |
| `scripts/dump_indicator_series.py` | Indicator debugging utility | Standalone debugging tool |
| `scripts/data_coverage_report.py` | Data coverage analysis | Standalone reporting tool |

### Tools (Not Core Engine)
| Path | Reason | Evidence |
|------|--------|---------|
| `tools/aggregate_c1_only.py` | C1-only aggregation tool | Standalone analysis tool |
| `tools/compare_runs.py` | Run comparison utility | Standalone comparison tool |
| `tools/make_indicator_table.py` | Indicator table generator | Standalone reporting tool |
| `tools/mt5_html_to_csv.py` | MT5 HTML converter | Data conversion utility |
| `tools/select_candidates.py` | Candidate selection tool | Standalone selection utility |

### Optimization (Not Core Engine)
| Path | Reason | Evidence |
|------|--------|---------|
| `optimizations/grid_search.py` | Grid search optimization | Not imported by core engine |

### Notebooks (Development/Exploration)
| Path | Reason | Evidence |
|------|--------|---------|
| `notebooks/*.ipynb` | All Jupyter notebooks | Development/exploration artifacts |
| `Untitled.ipynb` | Untitled notebook | Temporary file |

### Archive (Old Configs)
| Path | Reason | Evidence |
|------|--------|---------|
| `archive/*.yaml` | Old configuration files | Historical configs, not used |
| `archive/signal_logic.ipynb` | Old notebook | Historical artifact |

### Legacy Documentation
| Path | Reason | Evidence |
|------|--------|---------|
| `docs/c1_only_analysis.md` | C1-only analysis notes | Analysis notes, not core docs |
| `docs/c1_only_exits.md` | C1-only exits notes | Analysis notes, not core docs |
| `docs/mt5_fisher_parity.md` | MT5 parity notes | Parity validation notes |
| `MT5_PARITY_RERUN.md` | MT5 parity rerun guide | Parity validation guide |
| `docs/FOLDER_STRUCTURE.md` | Folder structure docs | May be outdated |
| `docs/INDICATORS.md` | Indicator documentation | May be outdated (check if referenced) |
| `docs/GOLDEN_STANDARD_LOGIC.md` | Logic documentation | May be outdated (check if referenced) |

### Legacy Configs
| Path | Reason | Evidence |
|------|--------|---------|
| `configs/config_fisher_c1_d1_2024.yaml` | Specific test config | Test config, not core |
| `configs/fisher_transform_test.yaml` | Test config | Test config, not core |
| `configs/parity/*.yaml` | Parity test configs | Validation configs, not core |
| `configs/validation/mt5_parity_d1.yaml` | MT5 parity config | Validation config, not core |

### Other
| Path | Reason | Evidence |
|------|--------|---------|
| `spotcheck/` | Spot check runs | Temporary test results |
| `sweeps/` | Sweep configs | May be experimental |

---

## 3. Results to Purge

All subfolders under `results/` EXCEPT `results/c1_only_exits/**`:

- `results/results_history/` (if exists)
- `results/single_debug/` (if exists)
- `results/smoke_selfcontained_*/` (if exists)
- `results/validation/` (if exists)
- `results/wfo_*/` (if exists)
- Any other subfolders not matching `c1_only_exits/**`

**Note**: If `results/c1_only_exits/` doesn't exist, create it with `.gitkeep` to maintain folder structure.

---

## 4. Tests to Keep

All tests in `tests/` that import core engine or cover indicator contracts:

- `tests/test_smoke.py` — Basic smoke test
- `tests/test_smoke_end_to_end.py` — Full pipeline test
- `tests/test_baseline_contract.py` — Indicator contract validation
- `tests/test_signal_exits.py` — Exit logic verification
- `tests/test_resolver_and_pipeline_smoke.py` — Indicator discovery
- `tests/test_writer.py` — Output file generation
- `tests/test_apply_signal_logic.py` — Signal logic tests
- `tests/test_audit_immutability.py` — Audit field immutability
- `tests/test_entry_rules_enforcement.py` — Entry rules
- `tests/test_tp1_classification.py` — TP1 classification
- `tests/test_continuations_wls_invariant.py` — Continuation invariants
- `tests/test_no_lookahead_decisions.py` — Lookahead prevention
- `tests/test_no_same_bar_duplicate_entries.py` — Duplicate entry prevention
- `tests/test_duplicate_open_guard.py` — Duplicate open guard
- `tests/test_date_slice.py` — Date slicing
- `tests/test_slice_enforce.py` — Slice enforcement
- `tests/test_metrics_rates.py` — Metrics rate calculations
- `tests/test_trades_writer.py` — Trades writer
- `tests/test_writer.py` — Writer tests
- `tests/test_wfo_truthiness.py` — WFO truthiness
- `tests/test_sweeps_schema_guard.py` — Sweeps schema guard
- `tests/conftest.py` — Test fixtures
- `tests/utils_synth.py` — Test utilities

### Tests to Quarantine (Specialized/Validation)

- `tests/test_c1_only_exits_config.py` — C1-only specific test (may keep if used)
- `tests/test_aggregate_and_stability.py` — Aggregation test (may keep if used)
- `tests/test_csv_schema_normalization.py` — Schema normalization test (may keep if used)
- `tests/test_c1_supertrend.py` — Specific indicator test (may keep if used)
- `tests/test_indicator_dump.py` — Indicator dump test (may keep if used)
- `tests/test_mt5_parity.py` — MT5 parity test (validation, not core)
- `tests/test_mt5_parity_smoke.py` — MT5 parity smoke test (validation, not core)
- `tests/test_single_debug.py` — Single debug test (may keep if used)
- `tests/test_golden_standard_logic.py` — Golden standard test (may keep if used)

**Decision**: Keep all tests for now (they're small and may be useful). Only quarantine if they break after moving other files.

---

## 5. Risk Notes & Undo Plan

### Risks
1. **Import Paths**: Moving Python modules may break imports if they're dynamically loaded.
2. **Test Dependencies**: Some tests may reference files in quarantined locations.
3. **Config References**: Some configs may reference paths that move.

### Undo Plan
All quarantined files are moved (not deleted) to `attic/2025-11-08/` with original folder structure preserved. To restore:

```bash
# Restore from attic
git checkout attic/2025-11-08/
# Or manually copy files back
cp -r attic/2025-11-08/* .
```

### Verification Steps
1. After quarantine, run `ruff check .` — should pass
2. Run `pytest -q` — should pass (may need to fix import paths)
3. Run `python scripts/smoke_test_selfcontained_v198.py -q` — should pass
4. Check that core scripts still work:
   - `python scripts/run_single_debug.py --help`
   - `python scripts/batch_sweeper.py --help`
   - `python scripts/walk_forward.py --help`

---

## 6. WFO Readiness Assessment

### Current State
- ✅ **Walk-forward driver exists**: `scripts/walk_forward.py` with `run_wfo(cfg: dict)` function
- ✅ **Config support**: WFO config block supported in YAML (`walk_forward` section)
- ✅ **Engine integration**: Imports `core.backtester.run_backtest` cleanly
- ✅ **Output artifacts**: Generates `wfo_folds.csv`, `equity_curve.csv`, `oos_summary.txt`
- ✅ **Smoke test coverage**: `smoke_test_selfcontained_v198.py` includes WFO test

### Example Command
```bash
python scripts/walk_forward.py --config configs/config.yaml
```

Or with explicit WFO config:
```yaml
# In config.yaml
walk_forward:
  enabled: true
  train_years: 2
  test_months: 6
  step_months: 6
  run_name: "wfo_default"
data:
  start: "2018-01-01"
  end: "2024-12-31"
```

### Gaps (if any)
- ⚠️ **Leaderboard/top-K**: No explicit leaderboard generation in WFO output (but folds CSV can be sorted)
- ⚠️ **Documentation**: WFO usage not prominently documented in README (but exists in STATUS.md)

### Verdict
**READY** ✅ — WFO is functional and can be run with a single command. Minor documentation gap, but core functionality is complete.

---

## Summary

- **Files to Keep**: ~50 core files (engine, indicators, entrypoints, tests, essential docs)
- **Files to Quarantine**: ~30 files (legacy scripts, tools, notebooks, archive, experimental configs)
- **Results to Purge**: All except `c1_only_exits/**`
- **WFO Status**: READY ✅

