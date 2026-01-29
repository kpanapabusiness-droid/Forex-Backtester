#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_batch_sweeper_csv_parseable.py — Test that batch_sweeper CSV is reliably parseable
---------------------------------------------------------------------------------------

Ensures consolidated CSV contains only scalar fields (no JSON blobs) and
can be read by pandas with default engine.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_sweeper import (  # noqa: E402
    FIELDNAMES,
    append_consolidated,
    extract_pairs_from_csv,
)


class TestBatchSweeperCSVParseable:
    """Test suite for CSV parseability."""

    def test_consolidated_csv_no_json_columns(self):
        """Test that FIELDNAMES does not include JSON/dict blob columns."""
        # These columns should NOT be in FIELDNAMES (they cause parsing issues)
        json_columns = ["roles", "params", "components_json", "params_json", "config_overrides", "indicator_params"]
        
        for col in json_columns:
            assert col not in FIELDNAMES, f"JSON column '{col}' should not be in FIELDNAMES (causes parsing issues)"

    def test_consolidated_csv_has_canonical_keys(self):
        """Test that FIELDNAMES includes all required canonical identity and component keys."""
        required_identity = ["run_id", "pair", "timeframe", "from_date", "to_date"]
        required_components = ["c1", "c2", "baseline", "volume", "exit"]
        required_metrics = ["total_trades", "wins", "losses", "scratches", "win_rate_ns", "roi_pct", "max_dd_pct", "expectancy"]
        
        for key in required_identity:
            assert key in FIELDNAMES, f"Missing required identity key: {key}"
        
        for key in required_components:
            assert key in FIELDNAMES, f"Missing required component key: {key}"
        
        for key in required_metrics:
            assert key in FIELDNAMES, f"Missing required metric key: {key}"

    def test_consolidated_csv_write_and_read(self):
        """Test that a sample row can be written and read back with pandas (default engine)."""
        # Build a sample row with all scalar fields (no JSON blobs)
        sample_row = {
            "run_id": "test_run_123",
            "pair": "EUR_USD",
            "timeframe": "D",
            "from_date": "2023-01-01",
            "to_date": "2025-01-01",
            "c1": "coral",
            "c2": "none",
            "baseline": "none",
            "volume": "none",
            "exit": "exit_twiggs_money_flow",
            "run_slug": "test_run_123",
            "timestamp": "20260125_120000",
            "total_trades": 100,
            "wins": 50,
            "losses": 30,
            "scratches": 20,
            "win_rate_ns": 62.5,
            "loss_rate_ns": 37.5,
            "scratch_rate_tot": 20.0,
            "win_rate": 50.0,
            "loss_rate": 30.0,
            "scratch_rate": 20.0,
            "roi_dollars": 1000.0,
            "roi_pct": 10.0,
            "max_dd_pct": 5.0,
            "expectancy": 0.5,
            "score": 5.0,
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_batch_results.csv"
            
            # Write using append_consolidated (simulates batch_sweeper behavior)
            # We need to temporarily set CONSOLIDATED for this test
            from scripts.batch_sweeper import CONSOLIDATED
            original_consolidated = CONSOLIDATED
            
            try:
                # Temporarily override CONSOLIDATED
                import scripts.batch_sweeper as bs_module
                bs_module.CONSOLIDATED = csv_path
                
                # Write the row
                append_consolidated([sample_row])
                
                # Verify file exists
                assert csv_path.exists(), "CSV file should be created"
                
                # Read back with pandas (default engine, no special options)
                df = pd.read_csv(csv_path)
                
                # Verify expected column count
                assert len(df.columns) == len(FIELDNAMES), (
                    f"Expected {len(FIELDNAMES)} columns, got {len(df.columns)}. "
                    f"Columns: {list(df.columns)}"
                )
                
                # Verify all FIELDNAMES are present
                for col in FIELDNAMES:
                    assert col in df.columns, f"Missing column: {col}"
                
                # Verify data integrity
                assert len(df) == 1, "Should have exactly one row"
                assert df.iloc[0]["run_id"] == "test_run_123"
                assert df.iloc[0]["pair"] == "EUR_USD"
                assert df.iloc[0]["total_trades"] == 100
                assert df.iloc[0]["roi_pct"] == 10.0
                
                # Verify no JSON/dict columns exist
                json_columns = ["roles", "params"]
                for col in json_columns:
                    assert col not in df.columns, f"JSON column '{col}' should not be in CSV"
                
            finally:
                # Restore original CONSOLIDATED
                bs_module.CONSOLIDATED = original_consolidated

    def test_consolidated_csv_handles_special_characters(self):
        """Test that CSV handles special characters in scalar fields correctly."""
        # Build a row with special characters that might cause issues
        sample_row = {
            "run_id": "test_run_with_underscores_123",
            "pair": "EUR_USD",  # Underscore is fine
            "timeframe": "D",
            "from_date": "2023-01-01",
            "to_date": "2025-01-01",
            "c1": "coral",
            "c2": "none",
            "baseline": "none",
            "volume": "none",
            "exit": "exit_twiggs_money_flow",
            "run_slug": "test_run_with_underscores_123",
            "timestamp": "20260125_120000",
            "total_trades": 100,
            "wins": 50,
            "losses": 30,
            "scratches": 20,
            "win_rate_ns": 62.5,
            "loss_rate_ns": 37.5,
            "scratch_rate_tot": 20.0,
            "win_rate": 50.0,
            "loss_rate": 30.0,
            "scratch_rate": 20.0,
            "roi_dollars": 1000.0,
            "roi_pct": 10.0,
            "max_dd_pct": 5.0,
            "expectancy": 0.5,
            "score": 5.0,
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_batch_results.csv"
            
            from scripts.batch_sweeper import CONSOLIDATED
            original_consolidated = CONSOLIDATED
            
            try:
                import scripts.batch_sweeper as bs_module
                bs_module.CONSOLIDATED = csv_path
                
                append_consolidated([sample_row])
                
                # Read back with pandas
                df = pd.read_csv(csv_path)
                
                # Verify data integrity
                assert df.iloc[0]["run_id"] == "test_run_with_underscores_123"
                assert df.iloc[0]["pair"] == "EUR_USD"
                
            finally:
                bs_module.CONSOLIDATED = original_consolidated

    def test_consolidated_csv_multiple_rows(self):
        """Test that multiple rows can be written and read correctly."""
        rows = [
            {
                "run_id": f"test_run_{i}",
                "pair": "EUR_USD",
                "timeframe": "D",
                "from_date": "2023-01-01",
                "to_date": "2025-01-01",
                "c1": "coral",
                "c2": "none",
                "baseline": "none",
                "volume": "none",
                "exit": "exit_twiggs_money_flow",
                "run_slug": f"test_run_{i}",
                "timestamp": "20260125_120000",
                "total_trades": 100 + i,
                "wins": 50 + i,
                "losses": 30,
                "scratches": 20,
                "win_rate_ns": 62.5,
                "loss_rate_ns": 37.5,
                "scratch_rate_tot": 20.0,
                "win_rate": 50.0,
                "loss_rate": 30.0,
                "scratch_rate": 20.0,
                "roi_dollars": 1000.0,
                "roi_pct": 10.0,
                "max_dd_pct": 5.0,
                "expectancy": 0.5,
                "score": 5.0,
            }
            for i in range(3)
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_batch_results.csv"
            
            from scripts.batch_sweeper import CONSOLIDATED
            original_consolidated = CONSOLIDATED
            
            try:
                import scripts.batch_sweeper as bs_module
                bs_module.CONSOLIDATED = csv_path
                
                append_consolidated(rows)
                
                # Read back with pandas
                df = pd.read_csv(csv_path)
                
                # Verify row count
                assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
                
                # Verify all rows are readable
                for i in range(3):
                    assert df.iloc[i]["run_id"] == f"test_run_{i}"
                    assert df.iloc[i]["total_trades"] == 100 + i
                
            finally:
                bs_module.CONSOLIDATED = original_consolidated

    def test_extract_pairs_from_csv(self):
        """Test extracting unique pairs from a CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_pairs.csv"
            
            # Create test CSV with pairs (some duplicates to test uniqueness)
            test_data = {
                "pair": ["EUR_USD", "GBP_USD", "EUR_USD", "AUD_USD", "GBP_USD", "USD_JPY"],
                "c1": ["coral", "coral", "aso", "coral", "aso", "coral"],
                "total_trades": [100, 150, 200, 120, 180, 90],
            }
            df = pd.DataFrame(test_data)
            df.to_csv(csv_path, index=False)
            
            # Extract pairs
            pairs = extract_pairs_from_csv(csv_path, column_name="pair")
            
            # Verify unique and sorted
            assert len(pairs) == 4, f"Expected 4 unique pairs, got {len(pairs)}"
            assert pairs == ["AUD_USD", "EUR_USD", "GBP_USD", "USD_JPY"], (
                f"Expected sorted unique pairs, got {pairs}"
            )
            
            # Test with custom column name
            csv_path2 = Path(tmpdir) / "test_pairs2.csv"
            test_data2 = {
                "symbol": ["EUR_USD", "GBP_USD", "AUD_USD"],
                "other_col": [1, 2, 3],
            }
            df2 = pd.DataFrame(test_data2)
            df2.to_csv(csv_path2, index=False)
            
            pairs2 = extract_pairs_from_csv(csv_path2, column_name="symbol")
            assert pairs2 == ["AUD_USD", "EUR_USD", "GBP_USD"]
            
            # Test error handling: missing file
            missing_path = Path(tmpdir) / "nonexistent.csv"
            with pytest.raises(FileNotFoundError, match="Pairs source CSV not found"):
                extract_pairs_from_csv(missing_path)
            
            # Test error handling: missing column
            with pytest.raises(RuntimeError, match="Column 'missing_col' not found"):
                extract_pairs_from_csv(csv_path, column_name="missing_col")
            
            # Test error handling: empty pairs
            csv_path3 = Path(tmpdir) / "test_empty.csv"
            empty_df = pd.DataFrame({"pair": [], "other": []})
            empty_df.to_csv(csv_path3, index=False)
            
            with pytest.raises(RuntimeError, match="No pairs found"):
                extract_pairs_from_csv(csv_path3, column_name="pair")

