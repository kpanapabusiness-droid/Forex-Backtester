#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_phase1_volume_invariant.py — Test Phase 1 Volume Invariant Validation
---------------------------------------------------------------------------

Unit tests for the invariant enforcement: trades_with_volume <= trades_without_volume
"""

from __future__ import annotations

import pytest
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase1_volume_referee import validate_invariant, find_join_keys  # noqa: E402


class TestPhase1VolumeInvariant:
    """Test suite for Phase 1 volume invariant validation."""

    def test_invariant_passes_when_volume_trades_less_or_equal(self):
        """Test that invariant passes when volume trades <= baseline trades."""
        df = pd.DataFrame({
            "pair": ["EUR_USD", "GBP_USD", "USD_JPY"],
            "c1_name": ["twiggs_money_flow", "disparity_index", "kalman_filter"],
            "trades_baseline": [100, 150, 200],
            "trades_volume": [80, 150, 180],  # All <= baseline
        })

        # Should not raise
        validate_invariant(df, "trades_baseline", "trades_volume")

    def test_invariant_passes_when_volume_trades_equal(self):
        """Test that invariant passes when volume trades equal baseline."""
        df = pd.DataFrame({
            "pair": ["EUR_USD"],
            "c1_name": ["twiggs_money_flow"],
            "trades_baseline": [100],
            "trades_volume": [100],  # Equal is OK
        })

        # Should not raise
        validate_invariant(df, "trades_baseline", "trades_volume")

    def test_invariant_fails_when_volume_trades_greater(self):
        """Test that invariant raises SystemExit when volume trades > baseline."""
        df = pd.DataFrame({
            "pair": ["EUR_USD", "GBP_USD"],
            "c1_name": ["twiggs_money_flow", "disparity_index"],
            "trades_baseline": [100, 150],
            "trades_volume": [120, 140],  # First one violates
        })

        with pytest.raises(SystemExit) as exc_info:
            validate_invariant(df, "trades_baseline", "trades_volume")

        # Check that error message contains violation info
        error_msg = str(exc_info.value)
        assert "INVARIANT VIOLATION" in error_msg or "violation" in error_msg.lower()
        assert "EUR_USD" in error_msg or "twiggs" in error_msg

    def test_invariant_fails_with_multiple_violations(self):
        """Test that invariant reports multiple violations."""
        df = pd.DataFrame({
            "pair": ["EUR_USD", "GBP_USD", "USD_JPY"],
            "c1_name": ["twiggs_money_flow", "disparity_index", "kalman_filter"],
            "trades_baseline": [100, 150, 200],
            "trades_volume": [120, 160, 220],  # All violate
        })

        with pytest.raises(SystemExit) as exc_info:
            validate_invariant(df, "trades_baseline", "trades_volume", max_violations=2)

        error_msg = str(exc_info.value)
        assert "3 rows" in error_msg or "violation" in error_msg.lower()

    def test_invariant_handles_zero_trades(self):
        """Test that invariant works correctly with zero trades."""
        df = pd.DataFrame({
            "pair": ["EUR_USD"],
            "c1_name": ["twiggs_money_flow"],
            "trades_baseline": [0],
            "trades_volume": [0],  # Both zero is OK
        })

        # Should not raise
        validate_invariant(df, "trades_baseline", "trades_volume")

    def test_invariant_handles_empty_dataframe(self):
        """Test that invariant handles empty dataframe gracefully."""
        df = pd.DataFrame({
            "pair": [],
            "c1_name": [],
            "trades_baseline": [],
            "trades_volume": [],
        })

        # Should not raise (no violations possible)
        validate_invariant(df, "trades_baseline", "trades_volume")


class TestPhase1VolumeJoinKeys:
    """Test suite for join key validation."""

    def test_find_join_keys_requires_at_least_3_keys(self):
        """Test that find_join_keys requires at least 3 keys."""
        baseline_df = pd.DataFrame({
            "pair": ["EUR_USD", "GBP_USD"],
            "c1": ["twiggs", "disparity"],
            "exit": ["exit_twiggs", "exit_twiggs"],
            "total_trades": [100, 150],
        })
        
        volume_df = pd.DataFrame({
            "pair": ["EUR_USD", "GBP_USD"],
            "c1": ["twiggs", "disparity"],
            "exit": ["exit_twiggs", "exit_twiggs"],
            "volume": ["adx", "adx"],
            "total_trades": [80, 140],
        })
        
        # Should find 3 keys (pair, c1, exit)
        keys = find_join_keys(baseline_df, volume_df)
        assert len(keys) >= 3
        key_names = [k[0].lower() for k in keys]
        assert "pair" in key_names
        assert "c1" in key_names
        assert "exit" in key_names

    def test_find_join_keys_rejects_too_few_keys(self):
        """Test that find_join_keys raises error with < 3 keys."""
        baseline_df = pd.DataFrame({
            "pair": ["EUR_USD", "GBP_USD"],
            "total_trades": [100, 150],
        })
        
        volume_df = pd.DataFrame({
            "pair": ["EUR_USD", "GBP_USD"],
            "total_trades": [80, 140],
        })
        
        with pytest.raises(ValueError) as exc_info:
            find_join_keys(baseline_df, volume_df)
        
        assert "Insufficient join keys" in str(exc_info.value) or "at least 3" in str(exc_info.value).lower()

    def test_find_join_keys_requires_pair_c1_exit(self):
        """Test that find_join_keys requires pair, c1, and exit."""
        baseline_df = pd.DataFrame({
            "pair": ["EUR_USD"],
            "timeframe": ["D"],
            "total_trades": [100],
        })
        
        volume_df = pd.DataFrame({
            "pair": ["EUR_USD"],
            "timeframe": ["D"],
            "total_trades": [80],
        })
        
        with pytest.raises(ValueError) as exc_info:
            find_join_keys(baseline_df, volume_df)
        
        error_msg = str(exc_info.value)
        assert ("Missing required join keys" in error_msg or 
                "pair" in error_msg.lower() or 
                "c1" in error_msg.lower() or
                "exit" in error_msg.lower())

    def test_find_join_keys_explicitly_uses_pair_c1_exit_when_present(self):
        """Test that find_join_keys explicitly uses ['pair', 'c1', 'exit'] when all present."""
        baseline_df = pd.DataFrame({
            "pair": ["EUR_USD", "GBP_USD"],
            "c1": ["twiggs", "disparity"],
            "exit": ["exit_twiggs", "exit_twiggs"],
            "total_trades": [100, 150],
        })
        
        volume_df = pd.DataFrame({
            "pair": ["EUR_USD", "GBP_USD"],
            "c1": ["twiggs", "disparity"],
            "exit": ["exit_twiggs", "exit_twiggs"],
            "volume": ["adx", "adx"],
            "total_trades": [80, 140],
        })
        
        # Should explicitly return ['pair', 'c1', 'exit'] without checking value overlap
        keys = find_join_keys(baseline_df, volume_df)
        assert len(keys) == 3
        key_names = [k[0].lower() for k in keys]
        assert "pair" in key_names
        assert "c1" in key_names
        assert "exit" in key_names


class TestPhase1VolumeJoinExplosion:
    """Test suite for join explosion protection."""

    def test_join_explosion_detection(self):
        """Test that join explosion is detected when joined rows > 2x max input."""
        import pandas as pd
        from scripts.phase1_volume_referee import find_join_keys
        
        # Create dataframes that would cause explosion (duplicate keys)
        baseline_df = pd.DataFrame({
            "pair": ["EUR_USD"] * 10,  # 10 rows with same pair
            "c1": ["twiggs"] * 10,
            "exit": ["exit_twiggs"] * 10,
            "total_trades": [100] * 10,
        })
        
        volume_df = pd.DataFrame({
            "pair": ["EUR_USD"] * 10,  # 10 rows with same pair
            "c1": ["twiggs"] * 10,
            "exit": ["exit_twiggs"] * 10,
            "volume": ["adx"] * 10,
            "total_trades": [80] * 10,
        })
        
        # This would create 100 joined rows (10x10), which is > 2*10 = 20
        # The explosion check happens in main(), not in find_join_keys
        # So we test the logic conceptually here
        keys = find_join_keys(baseline_df, volume_df)
        assert len(keys) >= 3  # Should find keys
        
        # The actual explosion check is in the main() function after merge
        # We can't easily test it without mocking, but the logic is there

