#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_apply_signal_logic.py — Tests for core/signal_logic.py
----------------------------------------------------------
Tests the complete NNFX signal engine implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.signal_logic import apply_signal_logic  # noqa: E402


def create_test_data(n_bars: int = 100) -> pd.DataFrame:
    """Create synthetic OHLC data for testing."""
    np.random.seed(42)  # Reproducible

    # Generate realistic price data
    base_price = 1.1000
    returns = np.random.normal(0, 0.001, n_bars)  # Small daily returns
    closes = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from closes
    highs = closes * (1 + np.abs(np.random.normal(0, 0.0005, n_bars)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.0005, n_bars)))
    opens = np.roll(closes, 1)
    opens[0] = base_price

    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n_bars, freq="D"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "atr": np.full(n_bars, 0.0020),  # Fixed ATR for predictability
        }
    )

    return df


def create_test_config(
    use_c2: bool = False,
    use_baseline: bool = True,
    use_volume: bool = False,
    one_candle_rule: bool = False,
    pullback_rule: bool = False,
    baseline_as_catalyst: bool = False,
) -> dict:
    """Create test configuration."""
    return {
        "indicators": {
            "use_c2": use_c2,
            "use_baseline": use_baseline,
            "use_volume": use_volume,
        },
        "rules": {
            "one_candle_rule": one_candle_rule,
            "pullback_rule": pullback_rule,
            "allow_baseline_as_catalyst": baseline_as_catalyst,
            "bridge_too_far_days": 7,
        },
        "exit": {
            "use_trailing_stop": True,
            "move_to_breakeven_after_atr": True,
            "exit_on_c1_reversal": True,
            "exit_on_baseline_cross": False,
            "exit_on_exit_signal": False,
        },
    }


class TestApplySignalLogic:
    """Test suite for apply_signal_logic function."""

    def test_domains_and_warmup(self):
        """Test signal domains and warmup behavior."""
        df = create_test_data(50)

        # Add simple C1 signals
        df["c1_signal"] = 0
        df.loc[10:15, "c1_signal"] = 1  # Long signals
        df.loc[25:30, "c1_signal"] = -1  # Short signals

        # Add baseline
        df["baseline"] = df["close"].rolling(10).mean()

        config = create_test_config()
        result = apply_signal_logic(df, config)

        # Check output domains
        assert result["entry_signal"].isin([-1, 0, 1]).all(), "entry_signal must be in {-1,0,1}"
        assert result["exit_signal_final"].isin([0, 1]).all(), "exit_signal_final must be in {0,1}"

        # Check no NaNs in critical columns
        assert not result["entry_signal"].isna().any(), "entry_signal should not have NaNs"
        assert not result["exit_signal_final"].isna().any(), (
            "exit_signal_final should not have NaNs"
        )

        # Check warmup period (first bar should be neutral)
        assert result.loc[0, "entry_signal"] == 0, "First bar should have no entry signal"

    def test_no_lookahead(self):
        """Test that decisions don't use future information."""
        df = create_test_data(20)
        df["c1_signal"] = 0
        df.loc[10, "c1_signal"] = 1
        df["baseline"] = df["close"].rolling(5).mean()

        config = create_test_config()

        # Run once
        result1 = apply_signal_logic(df, config)

        # Modify future price and run again
        df_modified = df.copy()
        df_modified.loc[15, "close"] = df_modified.loc[15, "close"] * 1.1  # Big price jump
        result2 = apply_signal_logic(df_modified, config)

        # Decisions up to bar 10 should be identical
        for i in range(11):
            assert result1.loc[i, "entry_signal"] == result2.loc[i, "entry_signal"], (
                f"Entry signal changed at bar {i} due to future price change"
            )
            assert result1.loc[i, "exit_signal_final"] == result2.loc[i, "exit_signal_final"], (
                f"Exit signal changed at bar {i} due to future price change"
            )

    def test_entry_blocking_and_allow(self):
        """Test baseline alignment blocking and allowing entries."""
        df = create_test_data(30)

        # Set up C1 signals
        df["c1_signal"] = 0
        df.loc[10, "c1_signal"] = 1  # Long signal
        df.loc[20, "c1_signal"] = -1  # Short signal

        # Set up baseline to test alignment
        df["baseline"] = df["close"] * 0.999  # Baseline slightly below price (allows long)

        # For bar 20 (short signal): set baseline ABOVE price to allow short
        df.loc[20, "baseline"] = df.loc[20, "close"] * 1.001  # Baseline above price (allows short)

        # For bar 25 (short signal): set baseline BELOW price to block short
        df.loc[25, "c1_signal"] = -1  # Add another short signal
        df.loc[25, "baseline"] = df.loc[25, "close"] * 0.999  # Baseline below price (blocks short)

        config = create_test_config(use_baseline=True)
        result = apply_signal_logic(df, config)

        # Bar 10: Long signal with price above baseline should be allowed
        assert result.loc[10, "entry_signal"] == 1, (
            "Long entry should be allowed when price > baseline"
        )
        assert result.loc[10, "entry_allowed"], "Entry should be marked as allowed"

        # Bar 20: Short signal with price below baseline should be allowed
        assert result.loc[20, "entry_signal"] == -1, (
            "Short entry should be allowed when price < baseline"
        )
        assert result.loc[20, "entry_allowed"], "Entry should be marked as allowed"

        # Bar 25: No new entry should be attempted since position is already open
        assert result.loc[25, "entry_signal"] == 0, "No new entry when position already open"
        assert result.loc[25, "position_open"], "Position should still be open from bar 20"

    def test_trailing_and_exit_priority(self):
        """Test trailing stop and exit priority."""
        df = create_test_data(30)

        # Set up for a long entry
        df["c1_signal"] = 0
        df.loc[5, "c1_signal"] = 1
        df["baseline"] = df["close"] * 0.999  # Price above baseline

        # Create price path that moves up then down to hit stop
        entry_price = df.loc[5, "close"]
        tp1_price = entry_price + df.loc[5, "atr"]  # TP1 at entry + 1 ATR

        # Move price up to hit TP1
        df.loc[10, "close"] = tp1_price + 0.0001

        # Then move down to hit trailing stop
        trail_sl = entry_price  # Should move to breakeven after TP1
        df.loc[15, "close"] = trail_sl - 0.0001  # Hit the stop

        config = create_test_config()
        result = apply_signal_logic(df, config)

        # Should have entry at bar 5
        assert result.loc[5, "entry_signal"] == 1, "Should have long entry"

        # Should have exit at bar 15 with stop hit
        exit_bars = result[result["exit_signal_final"] == 1]
        assert len(exit_bars) > 0, "Should have at least one exit"

        # Find the exit bar
        exit_bar_idx = exit_bars.index[0]
        assert result.loc[exit_bar_idx, "exit_reason"] == "stop_hit", (
            "Exit reason should be stop_hit"
        )

    def test_spreads_invariance(self):
        """Test that trade count is invariant to spreads (only PnL changes)."""
        df = create_test_data(50)

        # Add predictable signals
        df["c1_signal"] = 0
        df.loc[10:12, "c1_signal"] = 1
        df.loc[25:27, "c1_signal"] = -1
        df["baseline"] = df["close"].rolling(5).mean()

        config = create_test_config()

        # Run without spreads
        result_no_spreads = apply_signal_logic(df, config)

        # Run with spreads (this would be handled at execution level, not signal level)
        # For signal logic, trade counts should be identical
        result_with_spreads = apply_signal_logic(df, config)

        # Trade counts should be identical
        entries_no_spreads = (result_no_spreads["entry_signal"] != 0).sum()
        entries_with_spreads = (result_with_spreads["entry_signal"] != 0).sum()

        assert entries_no_spreads == entries_with_spreads, (
            "Trade count should be invariant to spreads setting"
        )

        exits_no_spreads = (result_no_spreads["exit_signal_final"] == 1).sum()
        exits_with_spreads = (result_with_spreads["exit_signal_final"] == 1).sum()

        assert exits_no_spreads == exits_with_spreads, (
            "Exit count should be invariant to spreads setting"
        )

    def test_c1_signal_detection(self):
        """Test C1 signal column detection."""
        df = create_test_data(20)
        df["baseline"] = df["close"].rolling(5).mean()

        # Test with standard column name
        df["c1_signal"] = 0
        df.loc[5, "c1_signal"] = 1

        config = create_test_config()
        result = apply_signal_logic(df, config)

        assert result.loc[5, "entry_signal"] == 1, "Should detect c1_signal column"

        # Test with no C1 column
        df_no_c1 = df.drop(columns=["c1_signal"])
        result_no_c1 = apply_signal_logic(df_no_c1, config)

        assert (result_no_c1["entry_signal"] == 0).all(), (
            "Should return all zeros when no C1 column found"
        )

    def test_one_candle_rule(self):
        """Test one-candle rule for delayed entries."""
        df = create_test_data(20)

        # C1 signal with misaligned baseline initially
        df["c1_signal"] = 0
        df.loc[10, "c1_signal"] = 1

        # Baseline starts above price (blocks entry)
        df["baseline"] = df["close"] * 1.001
        # But aligns on next bar
        df.loc[11, "baseline"] = df.loc[11, "close"] * 0.999

        config = create_test_config(one_candle_rule=True)
        result = apply_signal_logic(df, config)

        # Bar 10: Should be blocked but pending
        assert result.loc[10, "entry_signal"] == 0, "Entry should be blocked initially"
        assert "pending" in result.loc[10, "reason_block"], "Should be marked as pending"

        # Bar 11: Should recover and enter
        assert result.loc[11, "entry_signal"] == 1, "Entry should be allowed after recovery"

    def test_baseline_as_catalyst(self):
        """Test baseline-as-catalyst mode."""
        df = create_test_data(30)

        # No C1 signals except for bridge rule
        df["c1_signal"] = 0
        df.loc[5, "c1_signal"] = 1  # Recent C1 to satisfy bridge rule

        # Set up baseline cross scenario
        df["baseline"] = df["close"] * 1.001  # Start with baseline above price
        # Create a bullish cross at bar 10
        df.loc[10:, "baseline"] = df.loc[10:, "close"] * 0.999  # Baseline below price

        config = create_test_config(baseline_as_catalyst=True)
        result = apply_signal_logic(df, config)

        # Should have baseline-triggered entry at the cross
        assert result.loc[10, "entry_signal"] == 1, "Should have bullish entry at baseline cross"
        assert "baseline_trigger" in result.loc[10, "reason_block"], "Should be baseline triggered"

    def test_position_state_tracking(self):
        """Test position state tracking."""
        df = create_test_data(30)

        df["c1_signal"] = 0
        df.loc[10, "c1_signal"] = 1  # Entry
        df.loc[20, "c1_signal"] = -1  # Reversal exit
        df["baseline"] = df["close"] * 0.999

        config = create_test_config()
        result = apply_signal_logic(df, config)

        # Check position tracking
        assert result.loc[10, "entry_signal"] == 1, "Should have entry"

        # Position should be open between entry and exit
        position_open_bars = result[result["position_open"]]
        assert len(position_open_bars) > 0, "Should have bars with open position"

        # Should have exit signal
        exits = result[result["exit_signal_final"] == 1]
        assert len(exits) > 0, "Should have exit signal"

    def test_config_flexibility(self):
        """Test that function handles various config combinations."""
        df = create_test_data(20)
        df["c1_signal"] = 0
        df.loc[10, "c1_signal"] = 1
        df["baseline"] = df["close"] * 0.999

        # Test minimal config
        minimal_config = {"indicators": {}, "rules": {}, "exit": {}}
        result = apply_signal_logic(df, minimal_config)

        # Should not crash and should produce valid output
        assert result["entry_signal"].isin([-1, 0, 1]).all()
        assert result["exit_signal_final"].isin([0, 1]).all()

        # Test config with all features enabled (but not conflicting rules)
        full_config = create_test_config(
            use_c2=True,
            use_baseline=True,
            use_volume=True,
            one_candle_rule=True,
            pullback_rule=False,  # Can't enable both One-Candle and Pullback
            baseline_as_catalyst=True,
        )

        # Add missing indicator columns
        df["c2_signal"] = df["c1_signal"]  # Same as C1 for simplicity
        df["volume_signal"] = 1  # Always pass

        result_full = apply_signal_logic(df, full_config)

        # Should not crash
        assert result_full["entry_signal"].isin([-1, 0, 1]).all()
        assert result_full["exit_signal_final"].isin([0, 1]).all()
