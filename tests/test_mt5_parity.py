"""
Tests for MT5 parity functionality.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from scripts.mt5_compare import compare_totals, match_trades  # noqa: E402


class TestMT5ParityConfig:
    """Unit tests for MT5 parity configuration validation."""

    def test_mt5_parity_config_validation(self):
        """Test that MT5 parity config has exact required settings."""
        config_path = project_root / "configs/validation/mt5_parity_d1.yaml"

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Critical assertions - these must never change
        assert cfg["roles"]["c1"] == "sma_cross", "MT5 parity must use sma_cross indicator"
        assert cfg["c1"]["fast_period"] == 20, "MT5 parity fast SMA must be 20"
        assert cfg["c1"]["slow_period"] == 50, "MT5 parity slow SMA must be 50"
        assert cfg["symbol"] == "EURUSD", "MT5 parity must use EURUSD"
        assert cfg["timeframe"] == "D1", "MT5 parity must use D1 timeframe"
        assert str(cfg["date_from"]) == "2022-01-01", "MT5 parity start date must be 2022-01-01"
        assert str(cfg["date_to"]) == "2024-12-31", "MT5 parity end date must be 2024-12-31"
        assert cfg["spread"]["enabled"] is False, "MT5 parity must have spreads disabled"
        assert cfg["commission"]["enabled"] is False, "MT5 parity must have commission disabled"
        assert cfg["slippage_pips"] in [0, 0.0], "MT5 parity must have zero slippage"


class TestMT5CrossOnlyEngine:
    """Unit tests for cross-only engine behavior."""

    def test_sma_cross_events_only(self):
        """Test that SMA cross generates signals only on cross events, not continuous state."""
        # Create toy price series with known SMA crosses

        # Price series designed to create specific SMA crosses
        prices = [
            1.1000,
            1.1010,
            1.1020,
            1.1030,
            1.1040,  # Rising trend
            1.1050,
            1.1060,
            1.1070,
            1.1080,
            1.1090,  # Continue rising
            1.1100,
            1.1110,
            1.1120,
            1.1130,
            1.1140,  # More rising
            1.1150,
            1.1160,
            1.1170,
            1.1180,
            1.1190,  # Peak
            1.1200,
            1.1190,
            1.1180,
            1.1170,
            1.1160,  # Start falling
            1.1150,
            1.1140,
            1.1130,
            1.1120,
            1.1110,  # Continue falling
            1.1100,
            1.1090,
            1.1080,
            1.1070,
            1.1060,  # More falling
            1.1050,
            1.1040,
            1.1030,
            1.1020,
            1.1010,  # Bottom
            1.1000,
            1.0990,
            1.0980,
            1.0970,
            1.0960,  # Continue down
            1.0950,
            1.0960,
            1.0970,
            1.0980,
            1.0990,  # Start rising again
            1.1000,
            1.1010,
            1.1020,
            1.1030,
            1.1040,
        ]  # Rising again

        df = pd.DataFrame(
            {
                "close": prices,
                "high": [p + 0.0005 for p in prices],
                "low": [p - 0.0005 for p in prices],
                "open": prices,
            }
        )

        # Add project root to path for imports
        import sys
        from pathlib import Path

        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))

        from indicators.confirmation_funcs import c1_sma_cross

        # Apply SMA cross indicator
        result = c1_sma_cross(df, fast_period=5, slow_period=10, signal_col="c1_signal")

        # Check that signals are only -1, 0, or +1
        unique_signals = result["c1_signal"].unique()
        assert all(s in [-1, 0, 1] for s in unique_signals), f"Invalid signals: {unique_signals}"

        # Check that we have some cross events (not all zeros)
        non_zero_signals = result[result["c1_signal"] != 0]
        assert len(non_zero_signals) > 0, "No cross events detected"

        # Check that signals are sparse (not continuous state)
        signal_density = len(non_zero_signals) / len(result)
        assert signal_density < 0.5, (
            f"Too many signals ({signal_density:.2f}), should be sparse cross events"
        )

        # Verify no NaN values
        assert not result["c1_signal"].isna().any(), "Signal column contains NaN values"

    def test_cross_only_engine_behavior(self):
        """Test cross-only engine with toy data to verify single position and reversals."""
        # Create simple test data with known crosses
        df = pd.DataFrame(
            {
                "close": [1.1000, 1.1010, 1.1020, 1.1030, 1.1040, 1.1030, 1.1020, 1.1010, 1.1000],
                "high": [1.1005, 1.1015, 1.1025, 1.1035, 1.1045, 1.1035, 1.1025, 1.1015, 1.1005],
                "low": [1.0995, 1.1005, 1.1015, 1.1025, 1.1035, 1.1025, 1.1015, 1.1005, 1.0995],
                "open": [1.1000, 1.1010, 1.1020, 1.1030, 1.1040, 1.1030, 1.1020, 1.1010, 1.1000],
                "atr": [0.001] * 9,
            }
        )

        # Manually set cross signals: cross up at index 2, cross down at index 6
        df["c1_signal"] = [0, 0, 1, 0, 0, 0, -1, 0, 0]

        # Test config for cross-only engine
        config = {
            "engine": {"cross_only": True, "reverse_on_signal": True, "allow_pyramiding": False},
            "indicators": {"use_c2": False, "use_baseline": False, "use_volume": False},
            "rules": {"one_candle_rule": False, "pullback_rule": False},
            "exit": {"exit_on_c1_reversal": True},
        }

        # Import signal logic
        from core.signal_logic import apply_signal_logic

        result = apply_signal_logic(df, config)

        # Should have exactly 2 entry signals (long at index 2, short at index 6)
        entry_signals = result[result["entry_signal"] != 0]
        assert len(entry_signals) == 2, f"Expected 2 entries, got {len(entry_signals)}"

        # First entry should be long (+1) at index 2
        assert result.loc[2, "entry_signal"] == 1, "Expected long entry at index 2"
        assert result.loc[2, "entry_allowed"], "Entry should be allowed at index 2"

        # Second entry should be short (-1) at index 6 with simultaneous exit
        assert result.loc[6, "entry_signal"] == -1, "Expected short entry at index 6"
        assert result.loc[6, "exit_signal_final"] == 1, "Expected exit signal at index 6 (reversal)"
        assert result.loc[6, "exit_reason"] == "reverse_signal", (
            "Exit reason should be reverse_signal"
        )


class TestMT5Comparator:
    """Unit tests for MT5 comparison logic using synthetic data."""

    def test_trade_matching_exact(self):
        """Test trade matching with exact matches."""
        # Create synthetic trade data
        our_trades = pd.DataFrame(
            {
                "open_time": pd.to_datetime(["2022-01-01", "2022-01-02"]),
                "close_time": pd.to_datetime(["2022-01-01", "2022-01-02"]),
                "symbol": ["EURUSD", "EURUSD"],
                "side": [1, -1],
                "open_price": [1.1000, 1.1050],
                "close_price": [1.1010, 1.1040],
                "sl": [1.0950, 1.1100],
                "tp": [1.1050, 1.1000],
                "pnl_pips": [10.0, 10.0],
                "pnl_currency": [100.0, 100.0],
                "tag": ["tp1", "tp1"],
            }
        )

        mt5_trades = pd.DataFrame(
            {
                "open_time": pd.to_datetime(["2022-01-01", "2022-01-02"]),
                "close_time": pd.to_datetime(["2022-01-01", "2022-01-02"]),
                "symbol": ["EURUSD", "EURUSD"],
                "side": [1, -1],
                "open_price": [1.1000, 1.1050],
                "close_price": [1.1010, 1.1040],
                "sl": [1.0950, 1.1100],
                "tp": [1.1050, 1.1000],
                "pnl_pips": [10.0, 10.0],
                "pnl_currency": [100.0, 100.0],
                "tag": ["mt5", "mt5"],
            }
        )

        matches, unmatched_ours, unmatched_mt5 = match_trades(
            our_trades, mt5_trades, price_tol=0.00005, time_tol_bars=1
        )

        assert len(matches) == 2
        assert len(unmatched_ours) == 0
        assert len(unmatched_mt5) == 0
        assert (0, 0) in matches
        assert (1, 1) in matches

    def test_trade_matching_with_tolerance(self):
        """Test trade matching with price and time tolerances."""
        our_trades = pd.DataFrame(
            {
                "open_time": pd.to_datetime(["2022-01-01 00:00:00"]),
                "close_time": pd.to_datetime(["2022-01-01 00:00:00"]),
                "symbol": ["EURUSD"],
                "side": [1],
                "open_price": [1.1000],
                "close_price": [1.1010],
                "sl": [1.0950],
                "tp": [1.1050],
                "pnl_pips": [10.0],
                "pnl_currency": [100.0],
                "tag": ["tp1"],
            }
        )

        # MT5 trade with slight time and price difference
        mt5_trades = pd.DataFrame(
            {
                "open_time": pd.to_datetime(["2022-01-01 12:00:00"]),  # 12 hours later
                "close_time": pd.to_datetime(["2022-01-01 12:00:00"]),
                "symbol": ["EURUSD"],
                "side": [1],
                "open_price": [1.1000 + 0.00003],  # Within tolerance
                "close_price": [1.1010],
                "sl": [1.0950],
                "tp": [1.1050],
                "pnl_pips": [10.0],
                "pnl_currency": [100.0],
                "tag": ["mt5"],
            }
        )

        matches, unmatched_ours, unmatched_mt5 = match_trades(
            our_trades, mt5_trades, price_tol=0.00005, time_tol_bars=1
        )

        assert len(matches) == 1
        assert len(unmatched_ours) == 0
        assert len(unmatched_mt5) == 0

    def test_trade_matching_no_match(self):
        """Test trade matching when trades don't match."""
        our_trades = pd.DataFrame(
            {
                "open_time": pd.to_datetime(["2022-01-01"]),
                "close_time": pd.to_datetime(["2022-01-01"]),
                "symbol": ["EURUSD"],
                "side": [1],  # Buy
                "open_price": [1.1000],
                "close_price": [1.1010],
                "sl": [1.0950],
                "tp": [1.1050],
                "pnl_pips": [10.0],
                "pnl_currency": [100.0],
                "tag": ["tp1"],
            }
        )

        mt5_trades = pd.DataFrame(
            {
                "open_time": pd.to_datetime(["2022-01-01"]),
                "close_time": pd.to_datetime(["2022-01-01"]),
                "symbol": ["EURUSD"],
                "side": [-1],  # Sell - different side
                "open_price": [1.1000],
                "close_price": [1.1010],
                "sl": [1.0950],
                "tp": [1.1050],
                "pnl_pips": [10.0],
                "pnl_currency": [100.0],
                "tag": ["mt5"],
            }
        )

        matches, unmatched_ours, unmatched_mt5 = match_trades(
            our_trades, mt5_trades, price_tol=0.00005, time_tol_bars=1
        )

        assert len(matches) == 0
        assert len(unmatched_ours) == 1
        assert len(unmatched_mt5) == 1

    def test_compare_totals_pass(self):
        """Test total comparison with matching statistics."""
        our_trades = pd.DataFrame({"pnl_currency": [100.0, -50.0, 0.0, 75.0]})

        mt5_trades = pd.DataFrame({"pnl_currency": [100.0, -50.0, 0.0, 75.0]})

        result = compare_totals(our_trades, mt5_trades, pnl_pct_tol=0.001)

        assert result["passed"] is True
        assert len(result["errors"]) == 0
        assert result["our_stats"]["total_trades"] == 4
        assert result["our_stats"]["wins"] == 2
        assert result["our_stats"]["losses"] == 1
        assert result["our_stats"]["scratches"] == 1

    def test_compare_totals_fail_count(self):
        """Test total comparison with mismatched trade counts."""
        our_trades = pd.DataFrame({"pnl_currency": [100.0, -50.0]})

        mt5_trades = pd.DataFrame({"pnl_currency": [100.0, -50.0, 75.0]})

        result = compare_totals(our_trades, mt5_trades, pnl_pct_tol=0.001)

        assert result["passed"] is False
        assert len(result["errors"]) >= 1
        assert any("Trade count mismatch" in error for error in result["errors"])

    def test_compare_totals_fail_pnl(self):
        """Test total comparison with PnL outside tolerance."""
        our_trades = pd.DataFrame({"pnl_currency": [100.0]})

        mt5_trades = pd.DataFrame(
            {
                "pnl_currency": [110.0]  # 10% difference
            }
        )

        result = compare_totals(our_trades, mt5_trades, pnl_pct_tol=0.05)  # 5% tolerance

        assert result["passed"] is False
        assert len(result["errors"]) >= 1
        assert any("difference too large" in error for error in result["errors"])


class TestMT5ParityE2E:
    """End-to-end tests for MT5 parity (skipped if MT5 data not available)."""

    @pytest.mark.slow
    def test_full_parity_pipeline(self):
        """
        Full E2E test: run parity backtest and compare with MT5 data.
        Skipped if MT5 data is not available.
        """
        mt5_trades_file = Path("mt5/eurusd_d1_2022_2024/eurusd_d1_2022_2024_trades.csv")

        if not mt5_trades_file.exists():
            pytest.skip(
                f"MT5 trades file not found: {mt5_trades_file}. "
                "Please provide MT5 backtest data to run E2E parity test."
            )

        # Run our parity backtest
        print("\n🚀 Running MT5 parity backtest...")
        result = subprocess.run(
            [sys.executable, "scripts/mt5_parity_run.py"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        if result.returncode != 0:
            pytest.fail(
                f"Parity backtest failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

        # Run comparison
        print("\n🔍 Running MT5 comparison...")
        result = subprocess.run(
            [sys.executable, "scripts/mt5_compare.py"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        if result.returncode != 0:
            pytest.fail(f"MT5 comparison failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

        print("✅ Full MT5 parity pipeline completed successfully!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
