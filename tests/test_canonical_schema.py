#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_canonical_schema.py — Test canonical results schema enforcement
--------------------------------------------------------------------

Unit tests to ensure batch_sweeper.py outputs canonical schema and
baseline aggregation uses it correctly.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aggregate_c1_exit_baseline_from_batch import (  # noqa: E402
    aggregate_baseline,
)
from scripts.batch_sweeper import (  # noqa: E402
    FIELDNAMES,
    extract_canonical_identity,
)


class TestCanonicalSchema:
    """Test suite for canonical schema enforcement."""

    def test_extract_canonical_identity_required_keys(self):
        """Test that extract_canonical_identity extracts all required keys."""
        merged_cfg = {
            "pairs": ["EUR_USD"],
            "timeframe": "D",
            "date_range": {
                "start": "2023-01-01",
                "end": "2025-01-01",
            },
        }
        role_names = {
            "c1": "coral",
            "c2": None,
            "baseline": None,
            "volume": None,
            "exit": "exit_twiggs_money_flow",
        }
        
        identity = extract_canonical_identity(merged_cfg, role_names)
        
        # Required identity keys
        assert identity["pair"] == "EUR_USD"
        assert identity["timeframe"] == "D"
        assert identity["from_date"] == "2023-01-01"
        assert identity["to_date"] == "2025-01-01"
        
        # Components (always present)
        assert identity["c1"] == "coral"
        assert identity["c2"] == "none"
        assert identity["baseline"] == "none"
        assert identity["volume"] == "none"
        assert identity["exit"] == "exit_twiggs_money_flow"

    def test_extract_canonical_identity_fails_missing_pair(self):
        """Test that extract_canonical_identity fails fast if pair is missing."""
        merged_cfg = {
            "timeframe": "D",
            "date_range": {"start": "2023-01-01", "end": "2025-01-01"},
        }
        role_names = {"c1": "coral"}
        
        with pytest.raises(ValueError, match="Required identity key 'pair' missing"):
            extract_canonical_identity(merged_cfg, role_names)

    def test_extract_canonical_identity_fails_missing_timeframe(self):
        """Test that extract_canonical_identity fails fast if timeframe is missing."""
        merged_cfg = {
            "pairs": ["EUR_USD"],
            "date_range": {"start": "2023-01-01", "end": "2025-01-01"},
        }
        role_names = {"c1": "coral"}
        
        with pytest.raises(ValueError, match="Required identity key 'timeframe' missing"):
            extract_canonical_identity(merged_cfg, role_names)

    def test_extract_canonical_identity_fails_missing_dates(self):
        """Test that extract_canonical_identity fails fast if dates are missing."""
        merged_cfg = {
            "pairs": ["EUR_USD"],
            "timeframe": "D",
        }
        role_names = {"c1": "coral"}
        
        with pytest.raises(ValueError, match="Required identity keys 'from_date'/'to_date' missing"):
            extract_canonical_identity(merged_cfg, role_names)

    def test_consolidated_csv_includes_canonical_keys(self):
        """Test that FIELDNAMES includes all canonical identity and component keys."""
        required_identity = ["run_id", "pair", "timeframe", "from_date", "to_date"]
        required_components = ["c1", "c2", "baseline", "volume", "exit"]
        required_metrics = ["total_trades", "wins", "losses", "scratches", "win_rate_ns", "roi_pct", "max_dd_pct", "expectancy"]
        
        for key in required_identity:
            assert key in FIELDNAMES, f"Missing required identity key: {key}"
        
        for key in required_components:
            assert key in FIELDNAMES, f"Missing required component key: {key}"
        
        for key in required_metrics:
            assert key in FIELDNAMES, f"Missing required metric key: {key}"

    def test_baseline_aggregation_uses_canonical_pair_column(self):
        """Test that baseline aggregation uses canonical 'pair' column from batch CSV."""
        # Create synthetic batch CSV with canonical schema
        batch_data = {
            "run_id": ["run1", "run2"],
            "pair": ["EUR_USD", "GBP_JPY"],  # Canonical pair column
            "timeframe": ["D", "D"],
            "from_date": ["2023-01-01", "2023-01-01"],
            "to_date": ["2025-01-01", "2025-01-01"],
            "c1": ["coral", "aso"],
            "c2": ["none", "none"],
            "baseline": ["none", "none"],
            "volume": ["none", "none"],  # Baseline: no volume
            "exit": ["exit_twiggs_money_flow", "exit_twiggs_money_flow"],
            "run_slug": ["run1", "run2"],
            "total_trades": [100, 150],
            "wins": [50, 75],
            "losses": [30, 50],
            "scratches": [20, 25],
            "roi_pct": [10.5, 15.2],
            "max_dd_pct": [5.0, 7.0],
            "expectancy": [0.5, 0.6],
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_csv = Path(tmpdir) / "batch.csv"
            df_batch = pd.DataFrame(batch_data)
            df_batch.to_csv(batch_csv, index=False)
            
            # Aggregate (should use canonical pair column directly)
            result_df = aggregate_baseline(batch_csv)
            
            # Verify pair column contains real FX pairs
            assert "pair" in result_df.columns
            pairs = result_df["pair"].unique().tolist()
            assert "EUR_USD" in pairs
            assert "GBP_JPY" in pairs
            assert len(pairs) == 2
            
            # Verify grouping columns
            assert "c1" in result_df.columns
            assert "exit" in result_df.columns
            
            # Verify metrics
            assert "total_trades" in result_df.columns
            assert result_df["total_trades"].sum() == 250  # 100 + 150

    def test_baseline_aggregation_filters_volume(self):
        """Test that baseline aggregation filters out rows with volume indicators."""
        batch_data = {
            "run_id": ["run1", "run2", "run3"],
            "pair": ["EUR_USD", "EUR_USD", "EUR_USD"],
            "timeframe": ["D", "D", "D"],
            "from_date": ["2023-01-01", "2023-01-01", "2023-01-01"],
            "to_date": ["2025-01-01", "2025-01-01", "2025-01-01"],
            "c1": ["coral", "coral", "coral"],
            "c2": ["none", "none", "none"],
            "baseline": ["none", "none", "none"],
            "volume": ["none", "adx_volume", "none"],  # run2 has volume
            "exit": ["exit_twiggs_money_flow", "exit_twiggs_money_flow", "exit_twiggs_money_flow"],
            "run_slug": ["run1", "run2", "run3"],
            "total_trades": [100, 150, 200],
            "wins": [50, 75, 100],
            "losses": [30, 50, 70],
            "scratches": [20, 25, 30],
            "roi_pct": [10.5, 15.2, 12.0],
            "max_dd_pct": [5.0, 7.0, 6.0],
            "expectancy": [0.5, 0.6, 0.55],
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_csv = Path(tmpdir) / "batch.csv"
            df_batch = pd.DataFrame(batch_data)
            df_batch.to_csv(batch_csv, index=False)
            
            # Aggregate (should filter out run2 with volume)
            result_df = aggregate_baseline(batch_csv)
            
            # Should only have baseline rows (run1 and run3)
            assert result_df["total_trades"].sum() == 300  # 100 + 200 (run2 excluded)




