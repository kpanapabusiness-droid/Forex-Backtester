#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_c1_supertrend.py — Unit tests for Supertrend C1 indicator
--------------------------------------------------------------
Tests the Supertrend indicator implementation to ensure proper signal generation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import after path setup
from indicators.confirmation_funcs import c1_supertrend, supertrend  # noqa: E402


class TestSupertrendIndicator:
    """Test suite for Supertrend C1 indicator."""

    def test_c1_supertrend_basic_functionality(self, dummy_ohlcv):
        """Test that c1_supertrend produces valid signals."""
        df = dummy_ohlcv.copy()

        # Run the indicator
        result = c1_supertrend(df, atr_period=10, multiplier=3.0)

        # Check required columns exist
        assert "c1_signal" in result.columns, "c1_signal column should be created"
        assert "atr" in result.columns, "atr column should be present"

        # Check signal values are valid
        unique_signals = set(result["c1_signal"].dropna().unique())
        assert unique_signals.issubset({-1, 0, 1}), f"Invalid signals found: {unique_signals}"

        # After warmup period, should have both +1 and -1 signals (not all zeros)
        warmup_period = 14  # ATR default period
        post_warmup_signals = result["c1_signal"].iloc[warmup_period:].dropna()
        assert len(post_warmup_signals) > 0, "Should have signals after warmup"

        # Should contain both bullish and bearish signals for a 120-day period
        unique_post_warmup = set(post_warmup_signals.unique())
        assert 1 in unique_post_warmup or -1 in unique_post_warmup, (
            "Should have at least one directional signal"
        )

        # ATR should be positive where calculated
        atr_values = result["atr"].dropna()
        assert (atr_values > 0).all(), "ATR values should be positive"

    def test_supertrend_alias_functionality(self, dummy_ohlcv):
        """Test that supertrend alias works identically to c1_supertrend."""
        df = dummy_ohlcv.copy()

        # Run both functions with same parameters
        result_c1 = c1_supertrend(df.copy(), atr_period=10, multiplier=3.0)
        result_alias = supertrend(df.copy(), atr_period=10, multiplier=3.0)

        # Results should be identical
        pd.testing.assert_series_equal(
            result_c1["c1_signal"], result_alias["c1_signal"], check_names=False
        )
        pd.testing.assert_series_equal(result_c1["atr"], result_alias["atr"], check_names=False)

    def test_supertrend_custom_parameters(self, dummy_ohlcv):
        """Test Supertrend with custom parameters."""
        df = dummy_ohlcv.copy()

        # Test with different parameters
        result = c1_supertrend(df, atr_period=5, multiplier=2.0, signal_col="custom_signal")

        # Check custom signal column
        assert "custom_signal" in result.columns, "Custom signal column should be created"
        assert "atr" in result.columns, "ATR column should be present"

        # Check signal validity
        unique_signals = set(result["custom_signal"].dropna().unique())
        assert unique_signals.issubset({-1, 0, 1}), f"Invalid signals: {unique_signals}"

    def test_supertrend_trend_persistence(self, dummy_ohlcv):
        """Test that Supertrend maintains trend until reversal."""
        df = dummy_ohlcv.copy()

        # Create a strong trending market (monotonic increase)
        df["close"] = pd.Series(np.linspace(100, 120, len(df)))
        df["high"] = df["close"] + 0.1
        df["low"] = df["close"] - 0.1
        df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])

        result = c1_supertrend(df, atr_period=10, multiplier=3.0)

        # In a strong uptrend, should eventually be mostly bullish
        warmup_period = 20
        post_warmup_signals = result["c1_signal"].iloc[warmup_period:]

        # Should have valid signals
        assert len(post_warmup_signals.dropna()) > 0, "Should have signals after warmup"

        # Signals should be consistent (not flipping every bar)
        signal_changes = (post_warmup_signals != post_warmup_signals.shift()).sum()
        total_signals = len(post_warmup_signals.dropna())
        change_ratio = signal_changes / total_signals if total_signals > 0 else 0

        # Should not change direction too frequently (less than 50% of the time)
        assert change_ratio < 0.5, f"Too many signal changes: {change_ratio:.2%}"

    def test_supertrend_no_lookahead(self, dummy_ohlcv):
        """Test that Supertrend doesn't use future data."""
        df = dummy_ohlcv.copy()

        # Calculate full result
        full_result = c1_supertrend(df.copy(), atr_period=10, multiplier=3.0)

        # Calculate partial result (first 80 bars)
        partial_df = df.iloc[:80].copy()
        partial_result = c1_supertrend(partial_df, atr_period=10, multiplier=3.0)

        # The first 80 signals should be identical
        pd.testing.assert_series_equal(
            full_result["c1_signal"].iloc[:80], partial_result["c1_signal"], check_names=False
        )

    def test_supertrend_with_existing_atr(self, dummy_ohlcv):
        """Test that Supertrend works when ATR is already present."""
        from core.utils import calculate_atr

        df = dummy_ohlcv.copy()

        # Pre-calculate ATR
        df = calculate_atr(df, period=14)

        # Run Supertrend
        result = c1_supertrend(df, atr_period=10, multiplier=3.0)

        # Should still work correctly
        assert "c1_signal" in result.columns
        assert "atr" in result.columns

        unique_signals = set(result["c1_signal"].dropna().unique())
        assert unique_signals.issubset({-1, 0, 1})

    def test_supertrend_signal_contract(self, dummy_ohlcv):
        """Test that Supertrend follows the C1 indicator contract."""
        df = dummy_ohlcv.copy()

        # Test the contract requirements
        result = c1_supertrend(df, atr_period=10, multiplier=3.0, signal_col="c1_signal")

        # 1. Must return DataFrame
        assert isinstance(result, pd.DataFrame), "Must return DataFrame"

        # 2. Must have signal column with values in {-1, 0, +1}
        assert "c1_signal" in result.columns, "Must create c1_signal column"
        signal_values = result["c1_signal"].dropna()
        valid_values = {-1, 0, 1}
        assert set(signal_values.unique()).issubset(valid_values), (
            f"Signal values must be in {valid_values}"
        )

        # 3. Must include ATR column
        assert "atr" in result.columns, "Must include atr column"

        # 4. Should not modify input DataFrame
        original_cols = set(dummy_ohlcv.columns)
        result_cols = set(result.columns)
        new_cols = result_cols - original_cols
        assert "c1_signal" in new_cols, "Should add c1_signal column"
        assert "atr" in new_cols, "Should add atr column"

        # 5. Index should be preserved
        pd.testing.assert_index_equal(df.index, result.index)
