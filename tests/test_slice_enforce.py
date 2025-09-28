"""
Tests for slice enforcement in backtesting pipeline.
"""

# Add project root to path for imports
import sys
from pathlib import Path

import pandas as pd
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils import slice_df_by_dates  # noqa: E402


class TestSliceEnforcement:
    """Unit tests for slice enforcement in backtesting."""

    def test_slice_enforce_basic(self):
        """Test that slice enforcement works with simulated backtest data."""
        # Build a daily df from 2010-01-01 to 2024-12-31
        date_range = pd.date_range(start="2010-01-01", end="2024-12-31", freq="D")
        df = pd.DataFrame(
            {
                "date": date_range,
                "close": [1.1000 + 0.0001 * i for i in range(len(date_range))],
                "high": [1.1005 + 0.0001 * i for i in range(len(date_range))],
                "low": [1.0995 + 0.0001 * i for i in range(len(date_range))],
                "open": [1.1000 + 0.0001 * i for i in range(len(date_range))],
                "pair": ["EUR_USD"] * len(date_range),
            }
        )

        # Simulate what would happen with date filtering
        config = {"date_from": "2022-01-01", "date_to": "2024-12-31"}

        date_start = config.get("date_from")
        date_end = config.get("date_to")

        # Apply slice (simulating backtester behavior)
        sliced_df, (first_ts, last_ts, rows_before, rows_after) = slice_df_by_dates(
            df, date_start, date_end
        )

        # Verify slice worked correctly
        assert rows_before == len(df)  # Original size
        assert rows_after == len(sliced_df)  # Sliced size
        assert rows_after > 0, "Should have data in target range"

        # Verify date boundaries
        assert pd.to_datetime(first_ts).date() >= pd.to_datetime("2022-01-01").date()
        assert pd.to_datetime(last_ts).date() <= pd.to_datetime("2024-12-31").date()

        # Verify all dates are within range
        sliced_dates = pd.to_datetime(sliced_df["date"])
        assert (sliced_dates >= pd.to_datetime("2022-01-01")).all()
        assert (sliced_dates <= pd.to_datetime("2024-12-31")).all()

        # Should be approximately 3 years of data (2022, 2023, 2024)
        expected_days = pd.date_range(start="2022-01-01", end="2024-12-31", freq="D")
        assert rows_after == len(expected_days), (
            f"Expected {len(expected_days)} days, got {rows_after}"
        )

    def test_slice_enforce_with_indicators(self):
        """Test that slice enforcement works even when indicators might expand data."""
        # Create base data
        base_dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="D")
        df = pd.DataFrame(
            {
                "date": base_dates,
                "close": range(len(base_dates)),
                "high": [x + 0.1 for x in range(len(base_dates))],
                "low": [x - 0.1 for x in range(len(base_dates))],
                "open": range(len(base_dates)),
                "pair": ["EUR_USD"] * len(base_dates),
            }
        )

        # Add some "indicator" columns that might come from cache
        df["c1_signal"] = 0
        df["atr"] = 0.001

        # Simulate slice enforcement
        target_start = "2022-01-01"
        target_end = "2024-12-31"

        sliced_df, metadata = slice_df_by_dates(df, target_start, target_end)

        # After slicing, all data should be within range
        assert len(sliced_df) > 0
        dates = pd.to_datetime(sliced_df["date"])
        assert (dates >= pd.to_datetime(target_start)).all()
        assert (dates <= pd.to_datetime(target_end)).all()

        # Indicator columns should still be present
        assert "c1_signal" in sliced_df.columns
        assert "atr" in sliced_df.columns

    def test_trades_filter_simulation(self):
        """Test simulated trades filtering to date range."""
        # Create simulated trades data spanning wider range
        trade_dates = [
            "2019-06-15",
            "2021-03-10",
            "2022-01-15",
            "2022-06-20",
            "2023-03-10",
            "2023-11-05",
            "2024-02-14",
            "2024-11-30",
            "2025-01-15",
        ]

        trades_df = pd.DataFrame(
            {
                "pair": ["EUR_USD"] * len(trade_dates),
                "entry_date": trade_dates,
                "entry_price": [1.1000 + 0.01 * i for i in range(len(trade_dates))],
                "direction": ["long"] * len(trade_dates),
                "pnl": [100, -50, 75, -25, 150, -75, 200, -100, 50],
            }
        )

        # Simulate the trades filter logic
        date_start = "2022-01-01"
        date_end = "2024-12-31"

        if len(trades_df) > 0:
            start_ts = pd.to_datetime(date_start)
            end_ts = pd.to_datetime(date_end)

            # Filter trades by entry_date
            entry_dates = pd.to_datetime(trades_df["entry_date"])
            mask = (entry_dates >= start_ts) & (entry_dates <= end_ts)
            filtered_trades = trades_df[mask].copy()

        # Should keep only trades within the target range
        expected_kept = [
            "2022-01-15",
            "2022-06-20",
            "2023-03-10",
            "2023-11-05",
            "2024-02-14",
            "2024-11-30",
        ]

        assert len(filtered_trades) == len(expected_kept)

        # Verify all kept trades are within range
        kept_dates = pd.to_datetime(filtered_trades["entry_date"])
        assert (kept_dates >= pd.to_datetime(date_start)).all()
        assert (kept_dates <= pd.to_datetime(date_end)).all()

    def test_empty_slice_handling(self):
        """Test handling of empty slice results."""
        # Create data outside target range
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2010-01-01", "2010-01-02", "2010-01-03"]),
                "close": [1.1, 1.2, 1.3],
                "pair": ["EUR_USD"] * 3,
            }
        )

        # Try to slice to a range with no data
        sliced_df, (first_ts, last_ts, rows_before, rows_after) = slice_df_by_dates(
            df, "2025-01-01", "2025-12-31"
        )

        assert rows_before == 3
        assert rows_after == 0
        assert len(sliced_df) == 0
        assert first_ts is None
        assert last_ts is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
