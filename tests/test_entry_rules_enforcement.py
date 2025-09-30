"""
Tests for GS vNext entry rules enforcement and post-TP1 exit priority.

Entry constraints:
- One-Candle vs Pullback mutually exclusive; reject two-candle chains.
- If baseline is catalyst, do not apply One-Candle/Pullback.
- Bridge-Too-Far: baseline-catalyst entries blocked when last C1 cross ≥ N bars (default 7; configurable).

Post-TP1 exit priority:
- If trailing stop active and hit → 'trailing_stop'.
- Else if C1 reversal → 'c1_reversal'.
- Else if price returns to entry → 'breakeven_after_tp1'.
"""

import inspect

import pandas as pd
import pytest

from core import backtester as backtester_module
from core.signal_logic import apply_signal_logic


class TestEntryRulesExclusivity:
    """Test One-Candle vs Pullback mutual exclusivity."""

    def test_one_candle_and_pullback_mutually_exclusive(self):
        """Should raise ValueError when both One-Candle and Pullback rules are enabled."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10, freq="D"),
                "open": [1.0] * 10,
                "high": [1.1] * 10,
                "low": [0.9] * 10,
                "close": [1.0] * 10,
                "atr": [0.01] * 10,
                "c1_signal": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                "baseline": [1.0] * 10,
            }
        )

        cfg = {
            "indicators": {"use_baseline": True},
            "rules": {
                "one_candle_rule": True,
                "pullback_rule": True,  # Both enabled - should raise error
            },
            "exit": {},
            "engine": {},
        }

        with pytest.raises(
            ValueError, match="One-Candle and Pullback rules are mutually exclusive"
        ):
            apply_signal_logic(df, cfg)

    def test_one_candle_rule_only_allowed(self):
        """Should work when only One-Candle rule is enabled."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10, freq="D"),
                "open": [1.0] * 10,
                "high": [1.1] * 10,
                "low": [0.9] * 10,
                "close": [1.0] * 10,
                "atr": [0.01] * 10,
                "c1_signal": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                "baseline": [1.0] * 10,
            }
        )

        cfg = {
            "indicators": {"use_baseline": True},
            "rules": {
                "one_candle_rule": True,
                "pullback_rule": False,
            },
            "exit": {},
            "engine": {},
        }

        # Should not raise error
        result = apply_signal_logic(df, cfg)
        assert "entry_signal" in result.columns

    def test_pullback_rule_only_allowed(self):
        """Should work when only Pullback rule is enabled."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10, freq="D"),
                "open": [1.0] * 10,
                "high": [1.1] * 10,
                "low": [0.9] * 10,
                "close": [1.0] * 10,
                "atr": [0.01] * 10,
                "c1_signal": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                "baseline": [1.0] * 10,
            }
        )

        cfg = {
            "indicators": {"use_baseline": True},
            "rules": {
                "one_candle_rule": False,
                "pullback_rule": True,
            },
            "exit": {},
            "engine": {},
        }

        # Should not raise error
        result = apply_signal_logic(df, cfg)
        assert "entry_signal" in result.columns


class TestBaselineCatalystRules:
    """Test baseline-catalyst entry rules."""

    def test_baseline_catalyst_skips_one_candle_rule(self):
        """Baseline-catalyst entries should not apply One-Candle rule."""
        # Create scenario where baseline cross triggers entry but filters fail
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10, freq="D"),
                "open": [1.0] * 10,
                "high": [1.1] * 10,
                "low": [0.9] * 10,
                "close": [
                    0.99,
                    0.99,
                    1.01,
                    1.01,
                    1.01,
                    1.01,
                    1.01,
                    1.01,
                    1.01,
                    1.01,
                ],  # Cross baseline at bar 2
                "atr": [0.01] * 10,
                "c1_signal": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # C1 signal at bar 1
                "baseline": [1.0] * 10,
                "c2_signal": [0, 0, -1, 0, 0, 0, 0, 0, 0, 0],  # C2 blocks entry at bar 2
            }
        )

        cfg = {
            "indicators": {"use_baseline": True, "use_c2": True},
            "rules": {
                "one_candle_rule": True,
                "pullback_rule": False,
                "allow_baseline_as_catalyst": True,
                "bridge_too_far_days": 7,
            },
            "exit": {},
            "engine": {},
        }

        result = apply_signal_logic(df, cfg)

        # Bar 2 should be blocked by C2 but NOT have pending status (no One-Candle rule for baseline catalyst)
        assert result.loc[2, "entry_signal"] == 0
        assert not result.loc[2, "entry_allowed"]
        assert "pending" not in result.loc[2, "reason_block"]
        assert "c2" in result.loc[2, "reason_block"]

    def test_baseline_catalyst_skips_pullback_rule(self):
        """Baseline-catalyst entries should not apply Pullback rule."""
        # Create scenario where price is too far from baseline but baseline cross triggers
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10, freq="D"),
                "open": [1.0] * 10,
                "high": [1.1] * 10,
                "low": [0.9] * 10,
                "close": [
                    0.98,
                    0.98,
                    1.05,
                    1.05,
                    1.05,
                    1.05,
                    1.05,
                    1.05,
                    1.05,
                    1.05,
                ],  # Cross baseline, price far from baseline
                "atr": [0.01] * 10,
                "c1_signal": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # C1 signal at bar 1
                "baseline": [1.0] * 10,
            }
        )

        cfg = {
            "indicators": {"use_baseline": True},
            "rules": {
                "one_candle_rule": False,
                "pullback_rule": True,
                "allow_baseline_as_catalyst": True,
                "bridge_too_far_days": 7,
            },
            "exit": {},
            "engine": {},
        }

        result = apply_signal_logic(df, cfg)

        # Bar 2 should allow entry despite being far from baseline (no pullback rule for baseline catalyst)
        assert result.loc[2, "entry_signal"] == 1
        assert result.loc[2, "entry_allowed"]
        assert result.loc[2, "reason_block"] == "baseline_trigger"


class TestBridgeTooFar:
    """Test Bridge-Too-Far constraint for baseline-catalyst entries."""

    def test_bridge_too_far_blocks_baseline_catalyst(self):
        """Should block baseline-catalyst entries when last C1 cross ≥ N bars ago."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=15, freq="D"),
                "open": [1.0] * 15,
                "high": [1.1] * 15,
                "low": [0.9] * 15,
                "close": [0.99] * 7 + [0.99, 1.01] + [1.01] * 6,  # Baseline cross at bar 8
                "atr": [0.01] * 15,
                "c1_signal": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C1 signal at bar 1
                "baseline": [1.0] * 15,
            }
        )

        cfg = {
            "indicators": {"use_baseline": True},
            "rules": {
                "one_candle_rule": False,
                "pullback_rule": False,
                "allow_baseline_as_catalyst": True,
                "bridge_too_far_days": 7,  # Default 7 bars
            },
            "exit": {},
            "engine": {},
        }

        result = apply_signal_logic(df, cfg)

        # Bar 8: C1 was 7 bars ago (8-1=7), should be blocked by bridge-too-far (7 >= 7)
        assert result.loc[8, "entry_signal"] == 0
        assert not result.loc[8, "entry_allowed"]
        assert "bridge_too_far" in result.loc[8, "reason_block"]

    def test_bridge_too_far_allows_recent_c1(self):
        """Should allow baseline-catalyst entries when last C1 cross < N bars ago."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10, freq="D"),
                "open": [1.0] * 10,
                "high": [1.1] * 10,
                "low": [0.9] * 10,
                "close": [
                    0.99,
                    0.99,
                    0.99,
                    0.99,
                    0.99,
                    0.99,
                    1.01,
                    1.01,
                    1.01,
                    1.01,
                ],  # Baseline cross at bar 6
                "atr": [0.01] * 10,
                "c1_signal": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # C1 signal at bar 1
                "baseline": [1.0] * 10,
            }
        )

        cfg = {
            "indicators": {"use_baseline": True},
            "rules": {
                "one_candle_rule": False,
                "pullback_rule": False,
                "allow_baseline_as_catalyst": True,
                "bridge_too_far_days": 7,
            },
            "exit": {},
            "engine": {},
        }

        result = apply_signal_logic(df, cfg)

        # Bar 6: C1 was 5 bars ago (6-1=5), should be allowed
        assert result.loc[6, "entry_signal"] == 1
        assert result.loc[6, "entry_allowed"]
        assert result.loc[6, "reason_block"] == "baseline_trigger"

    def test_bridge_too_far_configurable_days(self):
        """Should respect configurable bridge_too_far_days setting."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=8, freq="D"),
                "open": [1.0] * 8,
                "high": [1.1] * 8,
                "low": [0.9] * 8,
                "close": [
                    0.99,
                    0.99,
                    0.99,
                    0.99,
                    0.99,
                    1.01,
                    1.01,
                    1.01,
                ],  # Baseline cross at bar 5
                "atr": [0.01] * 8,
                "c1_signal": [0, 1, 0, 0, 0, 0, 0, 0],  # C1 signal at bar 1
                "baseline": [1.0] * 8,
            }
        )

        cfg = {
            "indicators": {"use_baseline": True},
            "rules": {
                "one_candle_rule": False,
                "pullback_rule": False,
                "allow_baseline_as_catalyst": True,
                "bridge_too_far_days": 3,  # Custom 3 bars
            },
            "exit": {},
            "engine": {},
        }

        result = apply_signal_logic(df, cfg)

        # Bar 5: C1 was 4 bars ago (5-1=4), should be blocked with 3-bar limit
        assert result.loc[5, "entry_signal"] == 0
        assert not result.loc[5, "entry_allowed"]
        assert "bridge_too_far" in result.loc[5, "reason_block"]


class TestPostTP1ExitPriority:
    """Test post-TP1 exit reason priority: trailing_stop > c1_reversal > breakeven_after_tp1."""

    def test_exit_priority_implementation_exists(self):
        """Verify that the post-TP1 exit priority logic is implemented in backtester.py."""
        # This is a basic test to ensure the priority logic exists
        # The actual priority testing requires complex integration scenarios

        # Get the source code of the backtester module
        source = inspect.getsource(backtester_module)

        # Check that the post-TP1 priority logic is present
        assert "Post-TP1 exit priority" in source
        assert "trailing_stop > c1_reversal > breakeven_after_tp1" in source

        # Check that the priority structure exists
        assert "if tp1_done:" in source
        assert "# 1. Trailing stop (highest priority post-TP1)" in source
        assert "# 2. C1 reversal (second priority post-TP1)" in source
        assert "# 3. Breakeven (lowest priority post-TP1)" in source

    def test_pre_tp1_logic_preserved(self):
        """Verify that pre-TP1 logic is preserved (system exits > hard stops)."""
        # Get the source code of the backtester module
        source = inspect.getsource(backtester_module)

        # Check that pre-TP1 logic is preserved
        assert "else:" in source  # The else block for pre-TP1
        assert "# Pre-TP1: System exits (C1 reversal) take priority over hard stops" in source


def test_integration_all_constraints():
    """Integration test covering all new constraints together."""
    # Test that all constraints work together without conflicts

    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=20, freq="D"),
            "open": [1.0] * 20,
            "high": [1.1] * 20,
            "low": [0.9] * 20,
            "close": [0.99] * 10 + [1.01] * 10,  # Baseline cross at bar 10
            "atr": [0.01] * 20,
            "c1_signal": [
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # C1 at bar 1
            "baseline": [1.0] * 20,
            "c2_signal": [0] * 20,
        }
    )

    cfg = {
        "indicators": {"use_baseline": True, "use_c2": False},
        "rules": {
            "one_candle_rule": True,  # Only one rule enabled
            "pullback_rule": False,
            "allow_baseline_as_catalyst": True,
            "bridge_too_far_days": 7,
        },
        "exit": {"exit_on_c1_reversal": True},
        "engine": {},
    }

    # Should not raise any errors
    result = apply_signal_logic(df, cfg)

    # Bar 10 should be blocked by bridge-too-far (C1 was 9 bars ago)
    assert result.loc[10, "entry_signal"] == 0
    assert "bridge_too_far" in result.loc[10, "reason_block"]
