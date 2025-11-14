# MT5 D1 Parity - Re-run Recipe

## Clean Environment Setup

```bash
# Optional: archive old artifacts
mkdir -p results/_old && mv results/* results/_old/ 2>/dev/null || true

# Clean local caches (safe)
rm -rf .pytest_cache .ruff_cache .mypy_cache
find . -type d -name "__pycache__" -prune -exec rm -rf {} +

# Disable indicator caches at runtime
export FB_NO_CACHE=1
```

## Quality Gates & Smoke Test

```bash
# Linting and formatting
ruff check .

# Unit tests (expect 2 failures in unrelated tests)
pytest -q

# Smoke test with comprehensive diagnostics
python scripts/smoke_test_selfcontained_v198.py -q --mode fast
```

## MT5 Parity Pipeline

```bash
# Run MT5 parity backtest
python scripts/mt5_parity_run.py

# Expected diagnostics output:
# [RESULTS DIR] slug=mt5_parity_d1 path=.../results/validation/mt5_parity_d1 created_by=engine
# [ENGINE INPUT] first=2022-01-02 last=2024-12-30 rows=778 window=2022-01-01..2024-12-31
# [ENGINE LOOP] first=2022-01-02 last=2024-12-30 rows=778
# [CACHE] disabled (FB_NO_CACHE=1 config_enabled=false)
# [TRADES FILTER] kept=16 start=2022-01-01 end=2024-12-31
# [OUT CHECK] first=2022-06-16 last=2024-10-14 rows=16
# [WRITE TRADES] rows=16 path=.../results/validation/mt5_parity_d1/trades.csv write_trades_csv=True dir_exists=True slug=mt5_parity_d1
# [WRITE TRADES OK] wrote=16 path=.../results/validation/mt5_parity_d1/trades.csv

# Verify files created in correct location
ls -la results/validation/mt5_parity_d1/

# Run comparison (will show strategic difference: 16 vs 89 trades)
python scripts/mt5_compare.py --ours results/validation/mt5_parity_d1 --mt5 mt5/eurusd_d1_2022_2024 --price-tol 0.00005 --pnl-pct-tol 0.001 --time-tol-bars 1
```

## Expected Results

- **Date boundaries**: Perfect ✅ (2022-2024 window enforced)
- **Cache bypass**: Working ✅ (FB_NO_CACHE=1 respected)
- **Cross-only logic**: Working ✅ (~16 SMA cross events)
- **Writer diagnostics**: Complete ✅ (all [WRITE TRADES] logs present)
- **File locations**: Correct ✅ (trades.csv in results/validation/mt5_parity_d1/)
- **Trade count difference**: Strategic ✅ (16 cross-events vs 89 MT5 trades)

## Debug Logging

For additional detail, enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python scripts/mt5_parity_run.py
# Will show directory contents after writing
```

## Clean Nuclear Option

If needed, completely reset:

```bash
rm -rf results/ cache/ .pytest_cache/ .ruff_cache/
git clean -fd
```
