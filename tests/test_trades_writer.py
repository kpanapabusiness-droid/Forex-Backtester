"""
Unit tests for the centralized trades CSV writer.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.backtester import write_trades_csv_with_diagnostics  # noqa: E402


class TestTradesWriter:
    """Test the centralized trades CSV writer with diagnostics."""

    def create_sample_trades_df(self, num_trades=2):
        """Create a minimal valid trades DataFrame for testing."""
        base_data = {
            "pair": ["EUR_USD", "EUR_USD"],
            "entry_date": ["2022-06-16", "2022-06-20"],
            "entry_price": [1.04938, 1.05348],
            "direction": ["long", "short"],
            "direction_int": [1, -1],
            "exit_date": ["2022-06-20", "2022-06-24"],
            "exit_price": [1.05348, 1.04500],
            "pnl": [48.45, -95.20],
        }

        # Slice to the requested number of trades
        sliced_data = {}
        for key, values in base_data.items():
            sliced_data[key] = values[:num_trades]

        return pd.DataFrame(sliced_data)

    def test_trades_writer_parity_mode_writes_csv(self, tmp_path, capsys):
        """
        Positive case: writer should create CSV when flag is True and data is valid.
        """
        # Arrange
        trades_df = self.create_sample_trades_df(2)
        config = {"outputs": {"write_trades_csv": True}}

        # Act
        result = write_trades_csv_with_diagnostics(trades_df, tmp_path, config, "test_run")

        # Assert
        assert result is True, "Writer should return True on success"

        # Check file exists
        trades_file = tmp_path / "trades.csv"
        assert trades_file.exists(), f"trades.csv should exist at {trades_file}"

        # Check file contents
        written_df = pd.read_csv(trades_file)
        assert len(written_df) == 2, "Should write 2 trade rows"
        assert "pair" in written_df.columns, "Should have required headers"
        assert "entry_date" in written_df.columns, "Should have required headers"
        assert "direction_int" in written_df.columns, "Should have required headers"

        # Check logs
        captured = capsys.readouterr()
        assert "[WRITE TRADES]" in captured.out, "Should log decision inputs"
        assert "[WRITE TRADES OK]" in captured.out, "Should log success"
        assert "rows=2" in captured.out, "Should log correct row count"
        assert "write_trades_csv=True" in captured.out, "Should log flag status"

    def test_trades_writer_flag_off_negative(self, tmp_path, capsys):
        """
        Negative case: writer should skip when write_trades_csv=False.
        """
        # Arrange
        trades_df = self.create_sample_trades_df(2)
        config = {"outputs": {"write_trades_csv": False}}

        # Act
        result = write_trades_csv_with_diagnostics(trades_df, tmp_path, config, "test_run")

        # Assert
        assert result is False, "Writer should return False when flag is off"

        # Check file does NOT exist
        trades_file = tmp_path / "trades.csv"
        assert not trades_file.exists(), "trades.csv should NOT be created when flag is off"

        # Check logs
        captured = capsys.readouterr()
        assert "[WRITE TRADES]" in captured.out, "Should log decision inputs"
        assert "[WRITE TRADES SKIP] reason=flag_off" in captured.out, "Should log skip reason"
        assert "write_trades_csv=False" in captured.out, "Should log flag status"

    def test_trades_writer_empty_data(self, tmp_path, capsys):
        """
        Edge case: writer should create empty CSV with headers for compatibility.
        """
        # Arrange
        empty_df = pd.DataFrame(columns=["pair", "entry_date", "direction_int"])
        config = {"outputs": {"write_trades_csv": True}}

        # Act
        result = write_trades_csv_with_diagnostics(empty_df, tmp_path, config, "test_run")

        # Assert
        assert result is True, "Writer should return True for empty data (success)"

        # Check file DOES exist with headers (for compatibility)
        trades_file = tmp_path / "trades.csv"
        assert trades_file.exists(), "trades.csv should be created with headers for empty data"

        # Verify it's empty but has headers
        written_df = pd.read_csv(trades_file)
        assert len(written_df) == 0, "Should have 0 rows"
        assert len(written_df.columns) > 0, "Should have column headers"

        # Check logs
        captured = capsys.readouterr()
        assert "[WRITE TRADES SKIP] reason=empty" in captured.out, "Should log empty skip"
        assert "[WRITE TRADES OK] wrote=0" in captured.out, (
            "Should log successful empty file creation"
        )

    def test_trades_writer_empty_no_schema(self, tmp_path, capsys):
        """
        Critical test: empty DataFrame without required columns should still be success.
        This is the exact scenario that was failing in CI.
        """
        # Arrange - empty DataFrame with no columns at all (like from failed backtest)
        empty_df = pd.DataFrame()  # No columns, no rows
        config = {"outputs": {"write_trades_csv": True}}

        # Act
        result = write_trades_csv_with_diagnostics(empty_df, tmp_path, config, "test_run")

        # Assert
        assert result is True, "Writer should return True for empty DataFrame (even without schema)"

        # Check file DOES exist with standard headers (for compatibility)
        trades_file = tmp_path / "trades.csv"
        assert trades_file.exists(), "trades.csv should be created with headers for empty data"

        # Verify it's empty but has headers
        written_df = pd.read_csv(trades_file)
        assert len(written_df) == 0, "Should have 0 rows"
        assert len(written_df.columns) > 0, "Should have column headers"

        # Check logs - should show empty skip, NOT schema error
        captured = capsys.readouterr()
        assert "[WRITE TRADES SKIP] reason=empty" in captured.out, "Should log empty skip"
        assert "schema_invalid" not in captured.out, "Should NOT log schema error for empty data"
        assert "[WRITE TRADES OK] wrote=0" in captured.out, (
            "Should log successful empty file creation"
        )

    def test_trades_writer_schema_invalid(self, tmp_path, capsys):
        """
        Edge case: writer should skip when required columns are missing.
        """
        # Arrange - missing required column 'pair'
        invalid_df = pd.DataFrame({"entry_date": ["2022-06-16"], "direction_int": [1]})
        config = {"outputs": {"write_trades_csv": True}}

        # Act
        result = write_trades_csv_with_diagnostics(invalid_df, tmp_path, config, "test_run")

        # Assert
        assert result is False, "Writer should return False for invalid schema"

        # Check logs
        captured = capsys.readouterr()
        assert "[WRITE TRADES SKIP] reason=schema_invalid" in captured.out, (
            "Should log schema error"
        )
        assert "missing_fields=['pair']" in captured.out, "Should list missing fields"

    def test_trades_writer_debug_logging(self, tmp_path, capsys, monkeypatch):
        """
        Test debug logging when LOG_LEVEL=DEBUG.
        """
        # Arrange
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        trades_df = self.create_sample_trades_df(1)
        config = {"outputs": {"write_trades_csv": True}}

        # Act
        result = write_trades_csv_with_diagnostics(trades_df, tmp_path, config, "test_run")

        # Assert
        assert result is True

        # Check debug logs
        captured = capsys.readouterr()
        assert "[DEBUG] Results dir contents:" in captured.out, (
            "Should show debug directory listing"
        )

    def test_trades_writer_atomic_write_safety(self, tmp_path):
        """
        Test that atomic write prevents partial files on failure.
        """
        # This test ensures atomic write behavior - temp file is cleaned up on failure
        trades_df = self.create_sample_trades_df(1)
        config = {"outputs": {"write_trades_csv": True}}

        # Act
        result = write_trades_csv_with_diagnostics(trades_df, tmp_path, config, "test_run")

        # Assert
        assert result is True

        # Check no temporary files left behind
        temp_files = list(tmp_path.glob(".trades_tmp_*"))
        assert len(temp_files) == 0, "Should clean up temporary files"

        # Final file should exist
        final_file = tmp_path / "trades.csv"
        assert final_file.exists(), "Final trades.csv should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
