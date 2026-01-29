#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_aggregate_c1_exit_baseline.py — Test baseline aggregation from batch CSV
---------------------------------------------------------------------------

Unit tests for aggregate_c1_exit_baseline_from_batch.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aggregate_c1_exit_baseline_from_batch import (  # noqa: E402
    aggregate_baseline,
    extract_pairs_from_config,
)


class TestAggregateC1ExitBaseline:
    """Test suite for baseline aggregation."""

    def test_extract_pairs_from_config(self):
        """Test extracting pairs from config file."""
        import yaml
        
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            run_slug = "c1-coral__20260125_145522"
            run_dir = history_dir / run_slug
            run_dir.mkdir(parents=True)
            
            config_path = run_dir / "config.yaml"
            config_data = {
                "pairs": ["EUR_USD"],
                "indicators": {"c1": "coral"},
            }
            with config_path.open("w") as f:
                yaml.dump(config_data, f)
            
            pairs = extract_pairs_from_config(run_slug, history_dir)
            assert pairs == ["EUR_USD"]

    def test_aggregate_baseline_uses_real_pair_column(self):
        """Test that aggregation uses real FX pairs, not run_id/timestamp."""
        # Create synthetic batch CSV
        batch_data = {
            "run_slug": ["c1-coral__20260125_145522", "c1-aso__20260125_145523"],
            "timestamp": ["20260125_145552", "20260125_145553"],
            "roles": [
                json.dumps({"c1": "coral", "c2": None, "baseline": None, "volume": None, "exit": None}),
                json.dumps({"c1": "aso", "c2": None, "baseline": None, "volume": None, "exit": None}),
            ],
            "params": [
                json.dumps({"c1": {}, "c2": {}, "baseline": {}, "volume": {}, "exit": {}}),
                json.dumps({"c1": {}, "c2": {}, "baseline": {}, "volume": {}, "exit": {}}),
            ],
            "total_trades": [100, 150],
            "wins": [50, 75],
            "losses": [30, 50],
            "scratches": [20, 25],
            "roi_pct": [10.5, 15.2],
            "max_dd_pct": [5.0, 7.0],
            "expectancy": [0.5, 0.6],
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create batch CSV
            batch_csv = Path(tmpdir) / "batch.csv"
            df_batch = pd.DataFrame(batch_data)
            df_batch.to_csv(batch_csv, index=False)
            
            # Create history dir with configs
            history_dir = Path(tmpdir) / "history"
            history_dir.mkdir()
            
            # Create config for first run
            run1_dir = history_dir / "c1-coral__20260125_145522"
            run1_dir.mkdir()
            config1_path = run1_dir / "config.yaml"
            import yaml
            with config1_path.open("w") as f:
                yaml.dump({"pairs": ["EUR_USD"]}, f)
            
            # Create config for second run
            run2_dir = history_dir / "c1-aso__20260125_145523"
            run2_dir.mkdir()
            config2_path = run2_dir / "config.yaml"
            with config2_path.open("w") as f:
                yaml.dump({"pairs": ["AUD_USD"]}, f)
            
            # Aggregate
            result_df = aggregate_baseline(batch_csv, history_dir=history_dir)
            
            # Verify pair column contains FX pairs, not timestamps
            assert "pair" in result_df.columns
            pairs = result_df["pair"].unique().tolist()
            assert "EUR_USD" in pairs or "AUD_USD" in pairs
            assert "20260125_145522" not in pairs  # Should NOT be timestamp
            assert "20260125_145523" not in pairs  # Should NOT be timestamp
            
            # Verify other columns
            assert "c1" in result_df.columns
            assert "exit" in result_df.columns
            assert "total_trades" in result_df.columns

    def test_csv_quality_gate_passes_with_one_bad_line(self):
        """Test that one malformed line among ~200 lines passes quality gate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_csv = Path(tmpdir) / "batch.csv"
            
            # Create CSV with ~200 valid rows + 1 malformed row
            valid_rows = []
            for i in range(200):
                valid_rows.append({
                    "pair": "EUR_USD",
                    "c1": "coral",
                    "exit": "exit_twiggs_money_flow",
                    "volume": "none",
                    "total_trades": 100 + i,
                    "wins": 50,
                    "losses": 30,
                    "scratches": 20,
                    "roi_pct": 10.5,
                    "max_dd_pct": 5.0,
                    "expectancy": 0.5,
                })
            
            # Write valid rows
            df_valid = pd.DataFrame(valid_rows)
            df_valid.to_csv(batch_csv, index=False)
            
            # Append malformed line (extra commas, unquoted JSON)
            with batch_csv.open("a", encoding="utf-8") as f:
                f.write('EUR_USD,coral,exit_twiggs_money_flow,none,100,50,30,20,10.5,5.0,0.5,extra,commas,here\n')
            
            # Should pass (1 bad line out of 201 total = 0.5%, threshold allows max(1, 201*0.005) = 1)
            result_df = aggregate_baseline(batch_csv)
            assert len(result_df) > 0  # Should have aggregated results

    def test_csv_quality_gate_fails_with_many_bad_lines(self):
        """Test that many malformed lines fail quality gate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_csv = Path(tmpdir) / "batch.csv"
            
            # Create CSV with ~200 valid rows
            valid_rows = []
            for i in range(200):
                valid_rows.append({
                    "pair": "EUR_USD",
                    "c1": "coral",
                    "exit": "exit_twiggs_money_flow",
                    "volume": "none",
                    "total_trades": 100 + i,
                    "wins": 50,
                    "losses": 30,
                    "scratches": 20,
                    "roi_pct": 10.5,
                    "max_dd_pct": 5.0,
                    "expectancy": 0.5,
                })
            
            # Write valid rows
            df_valid = pd.DataFrame(valid_rows)
            df_valid.to_csv(batch_csv, index=False)
            
            # Append many malformed lines (exceeds 0.5% threshold)
            # Threshold: max(1, 200*0.005) = 1, so 2+ bad lines should fail
            with batch_csv.open("a", encoding="utf-8") as f:
                f.write('EUR_USD,coral,exit_twiggs_money_flow,none,100,50,30,20,10.5,5.0,0.5,extra,commas,here\n')
                f.write('EUR_USD,coral,exit_twiggs_money_flow,none,100,50,30,20,10.5,5.0,0.5,more,commas,here\n')
                f.write('EUR_USD,coral,exit_twiggs_money_flow,none,100,50,30,20,10.5,5.0,0.5,even,more,commas\n')
            
            # Should fail (3 bad lines out of 203 total > threshold of 1)
            with pytest.raises(RuntimeError, match="CSV quality check failed"):
                aggregate_baseline(batch_csv)

    def test_pair_level_aggregation_preserves_pairs(self):
        """Test that aggregation preserves pair-level granularity (pair × c1 × exit)."""
        # Create batch CSV with multiple pairs and C1s
        pairs = ["EUR_USD", "GBP_USD", "AUD_USD"]
        c1s = ["coral", "aso", "cyber_cycle"]
        exit_name = "exit_twiggs_money_flow"
        
        rows = []
        for pair in pairs:
            for c1 in c1s:
                rows.append({
                    "pair": pair,
                    "c1": c1,
                    "exit": exit_name,
                    "volume": "none",
                    "total_trades": 100,
                    "wins": 50,
                    "losses": 30,
                    "scratches": 20,
                    "roi_pct": 10.5,
                    "max_dd_pct": 5.0,
                    "expectancy": 0.5,
                })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_csv = Path(tmpdir) / "batch.csv"
            df_batch = pd.DataFrame(rows)
            df_batch.to_csv(batch_csv, index=False)
            
            # Aggregate
            result_df = aggregate_baseline(batch_csv)
            
            # Verify pair-level granularity
            assert "pair" in result_df.columns, "Output must include 'pair' column"
            assert "c1" in result_df.columns
            assert "exit" in result_df.columns
            
            # Should have pair_count × c1_count rows (for single exit)
            expected_rows = len(pairs) * len(c1s)
            assert len(result_df) == expected_rows, (
                f"Expected {expected_rows} rows (3 pairs × 3 C1s), got {len(result_df)}"
            )
            
            # Verify all pairs are present
            result_pairs = set(result_df["pair"].unique())
            assert result_pairs == set(pairs), f"Missing pairs. Expected {set(pairs)}, got {result_pairs}"
            
            # Verify all C1s are present
            result_c1s = set(result_df["c1"].unique())
            assert result_c1s == set(c1s), f"Missing C1s. Expected {set(c1s)}, got {result_c1s}"
            
            # Verify pairs are real FX pairs (not timestamps or run_ids)
            for pair_val in result_df["pair"].unique():
                assert "_" in pair_val, f"Pair '{pair_val}' should be FX pair format (e.g., EUR_USD)"
                assert pair_val.isupper() or pair_val.replace("_", "").isalnum(), (
                    f"Pair '{pair_val}' should be valid FX pair"
                )
            
            # Verify no pooling across pairs (each pair×c1 combination should be separate)
            pair_c1_combos = result_df.groupby(["pair", "c1"]).size()
            assert all(pair_c1_combos == 1), "Each pair×c1 combination should appear exactly once"

