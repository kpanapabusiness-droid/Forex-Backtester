"""
Tests for date slicing functionality.
"""

# Add project root to path for imports
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils import slice_df_by_dates  # noqa: E402


class TestDateSlicing:
    """Unit tests for date slicing utility."""

    def test_slice_df_by_dates_basic(self):
        """Test basic date slicing functionality."""
        # Create test data from 2019-01-01 to 2024-12-31 daily
        date_range = pd.date_range(start="2019-01-01", end="2024-12-31", freq="D")
        df = pd.DataFrame(
            {
                "date": date_range,
                "close": range(len(date_range)),
                "high": [x + 0.1 for x in range(len(date_range))],
                "low": [x - 0.1 for x in range(len(date_range))],
                "open": range(len(date_range)),
            }
        )

        # Slice to 2022-01-01 to 2024-12-31
        sliced_df, (first_ts, last_ts, rows_before, rows_after) = slice_df_by_dates(
            df, "2022-01-01", "2024-12-31"
        )

        # Verify metadata
        assert rows_before == len(df)
        assert rows_after == len(sliced_df)
        assert rows_after > 0, "Should have data in target range"

        # Verify date boundaries
        assert pd.to_datetime(first_ts).date() >= pd.to_datetime("2022-01-01").date()
        assert pd.to_datetime(last_ts).date() <= pd.to_datetime("2024-12-31").date()

        # Verify expected row count (approximately 3 years of daily data)
        expected_days = pd.date_range(start="2022-01-01", end="2024-12-31", freq="D")
        assert rows_after == len(expected_days), (
            f"Expected {len(expected_days)} days, got {rows_after}"
        )

        # Verify all dates are within range
        sliced_dates = pd.to_datetime(sliced_df["date"])
        assert (sliced_dates >= pd.to_datetime("2022-01-01")).all()
        assert (sliced_dates <= pd.to_datetime("2024-12-31")).all()

    def test_slice_df_by_dates_inclusive_options(self):
        """Test different inclusive options."""
        # Create small test dataset
        dates = ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05"]
        df = pd.DataFrame({"date": pd.to_datetime(dates), "value": range(len(dates))})

        # Test "both" (default)
        sliced, _ = slice_df_by_dates(df, "2022-01-02", "2022-01-04", inclusive="both")
        assert len(sliced) == 3  # includes both boundaries

        # Test "left"
        sliced, _ = slice_df_by_dates(df, "2022-01-02", "2022-01-04", inclusive="left")
        assert len(sliced) == 2  # includes left boundary only

        # Test "right"
        sliced, _ = slice_df_by_dates(df, "2022-01-02", "2022-01-04", inclusive="right")
        assert len(sliced) == 2  # includes right boundary only

        # Test "neither"
        sliced, _ = slice_df_by_dates(df, "2022-01-02", "2022-01-04", inclusive="neither")
        assert len(sliced) == 1  # excludes both boundaries

    def test_slice_df_by_dates_empty_result(self):
        """Test slicing that produces empty result."""
        df = pd.DataFrame(
            {"date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]), "value": [1, 2, 3]}
        )

        # Slice outside available range
        sliced, (first_ts, last_ts, rows_before, rows_after) = slice_df_by_dates(
            df, "2025-01-01", "2025-12-31"
        )

        assert rows_before == 3
        assert rows_after == 0
        assert len(sliced) == 0
        assert first_ts is None
        assert last_ts is None

    def test_slice_df_by_dates_different_date_formats(self):
        """Test slicing with different date input formats."""
        df = pd.DataFrame(
            {"date": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03"]), "value": [1, 2, 3]}
        )

        # Test with string dates
        sliced1, _ = slice_df_by_dates(df, "2022-01-01", "2022-01-02")

        # Test with datetime objects
        sliced2, _ = slice_df_by_dates(df, datetime(2022, 1, 1), datetime(2022, 1, 2))

        # Test with date objects
        sliced3, _ = slice_df_by_dates(df, date(2022, 1, 1), date(2022, 1, 2))

        # All should produce same result
        assert len(sliced1) == len(sliced2) == len(sliced3) == 2

    def test_slice_df_by_dates_missing_date_column(self):
        """Test error handling when date column is missing."""
        df = pd.DataFrame(
            {"timestamp": pd.to_datetime(["2022-01-01", "2022-01-02"]), "value": [1, 2]}
        )

        with pytest.raises(ValueError, match="DataFrame must have 'date' column"):
            slice_df_by_dates(df, "2022-01-01", "2022-01-02")

    def test_slice_df_by_dates_invalid_inclusive(self):
        """Test error handling for invalid inclusive parameter."""
        df = pd.DataFrame({"date": pd.to_datetime(["2022-01-01", "2022-01-02"]), "value": [1, 2]})

        with pytest.raises(ValueError, match="inclusive must be"):
            slice_df_by_dates(df, "2022-01-01", "2022-01-02", inclusive="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
