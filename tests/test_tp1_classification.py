"""
Test TP1-based classification logic.

Tests that WL/S classification is based solely on TP1 leg:
- If TP1 touched: WIN (regardless of runner outcome)
- If SL before TP1: LOSS
- If system exit before TP1: SCRATCH
- Runner affects ROI/DD only, not WL/S classification
"""

from __future__ import annotations

from tests.utils_synth import create_synthetic_ohlc, run_synthetic_backtest


class TestTP1Classification:
    """Test suite for TP1-based conceptual classification."""

    def test_tp1_hit_then_be_exit_is_scratch(self):
        """
        Hard-Stop Realism: TP1 hit → BE exit → SCRATCH classification.

        This tests the hard-stop realism rule: breakeven exits are always SCRATCH,
        even if TP1 was hit previously.

        Scenario:
        - Entry at 1.0000, TP1 at 1.0020, SL at 0.9970
        - TP1 hit (half closed)
        - Runner moves to breakeven (1.0000)
        - Price drops to hit runner SL at breakeven
        - Expected: SCRATCH classification (BE exit overrides TP1 for classification)
        """
        bars = [
            # Bar 0: Setup
            {
                "date": "2023-01-01",
                "open": 1.0000,
                "high": 1.0000,
                "low": 1.0000,
                "close": 1.0000,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 1: Entry signal
            {
                "date": "2023-01-02",
                "open": 1.0000,
                "high": 1.0005,
                "low": 0.9995,
                "close": 1.0000,
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 2: TP1 hit
            {
                "date": "2023-01-03",
                "open": 1.0000,
                "high": 1.0025,  # Hits TP1 at 1.0020
                "low": 1.0000,
                "close": 1.0010,
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 3: Price drops to hit breakeven SL
            {
                "date": "2023-01-04",
                "open": 1.0010,
                "high": 1.0010,
                "low": 0.9995,  # Hits breakeven SL at 1.0000
                "close": 0.9995,
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
        ]

        df = create_synthetic_ohlc(bars, atr_value=0.002)
        result = run_synthetic_backtest(df)

        trades = result["trades"]
        assert len(trades) == 1, f"Expected 1 trade, got {len(trades)}"

        trade = trades[0]

        # Verify TP1 was hit
        assert trade["tp1_hit"], "TP1 should have been hit"
        assert trade["breakeven_after_tp1"], "Should move to breakeven after TP1"

        # Verify exit reason is stop-related
        assert trade["exit_reason"] in ["stoploss", "breakeven_after_tp1"], (
            f"Expected SL-related exit, got '{trade['exit_reason']}'"
        )

        # CORE TEST: Hard-Stop Realism: BE exit = SCRATCH (even if TP1 hit)
        assert not trade["win"], "Trade should not be WIN (breakeven exit = SCRATCH)"
        assert not trade["loss"], "Trade should not be LOSS"
        assert trade["scratch"], "Trade should be SCRATCH (breakeven exit, regardless of TP1)"

        # Verify PnL includes both halves (half profit from TP1, runner loss from SL)
        # Half should be profitable (TP1), runner should be breakeven or small loss
        assert trade["pnl"] >= 0, "Should have non-negative PnL (TP1 half profit >= runner loss)"

    def test_sl_before_tp1_is_loss(self):
        """
        SL hit before TP1 → LOSS classification.

        This verifies the existing behavior is preserved.
        """
        bars = [
            # Bar 0: Setup
            {
                "date": "2023-01-01",
                "open": 1.0000,
                "high": 1.0000,
                "low": 1.0000,
                "close": 1.0000,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 1: Entry signal
            {
                "date": "2023-01-02",
                "open": 1.0000,
                "high": 1.0005,
                "low": 0.9995,
                "close": 1.0000,
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 2: SL hit before TP1
            {
                "date": "2023-01-03",
                "open": 1.0000,
                "high": 1.0005,
                "low": 0.9965,  # Hits SL at 0.9970 before TP1
                "close": 0.9975,
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
        ]

        df = create_synthetic_ohlc(bars, atr_value=0.002)
        result = run_synthetic_backtest(df)

        trades = result["trades"]
        assert len(trades) == 1, f"Expected 1 trade, got {len(trades)}"

        trade = trades[0]

        # Verify TP1 was NOT hit
        assert not trade["tp1_hit"], "TP1 should NOT have been hit"

        # Verify LOSS classification
        assert not trade["win"], "Trade should not be WIN"
        assert trade["loss"], "Trade should be LOSS (SL before TP1)"
        assert not trade["scratch"], "Trade should not be SCRATCH"

        # Verify negative PnL
        assert trade["pnl"] < 0, "Should have negative PnL (full SL loss)"

    def test_system_exit_before_tp1_is_scratch(self):
        """
        System exit before TP1 → SCRATCH classification.

        This verifies the existing behavior is preserved.
        """
        bars = [
            # Bar 0: Setup
            {
                "date": "2023-01-01",
                "open": 1.0000,
                "high": 1.0000,
                "low": 1.0000,
                "close": 1.0000,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 1: Entry signal
            {
                "date": "2023-01-02",
                "open": 1.0000,
                "high": 1.0005,
                "low": 0.9995,
                "close": 1.0000,
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 2: C1 reversal before TP1 or SL
            {
                "date": "2023-01-03",
                "open": 1.0000,
                "high": 1.0015,  # Not high enough for TP1
                "low": 0.9985,  # Not low enough for SL
                "close": 1.0005,
                "c1_signal": -1,  # C1 reversal triggers exit
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
        ]

        df = create_synthetic_ohlc(bars, atr_value=0.002)
        result = run_synthetic_backtest(df)

        trades = result["trades"]
        assert len(trades) == 1, f"Expected 1 trade, got {len(trades)}"

        trade = trades[0]

        # Verify TP1 was NOT hit
        assert not trade["tp1_hit"], "TP1 should NOT have been hit"

        # Verify SCRATCH classification
        assert not trade["win"], "Trade should not be WIN"
        assert not trade["loss"], "Trade should not be LOSS"
        assert trade["scratch"], "Trade should be SCRATCH (system exit before TP1)"

        # PnL can be positive or negative, but should be small
        assert abs(trade["pnl"]) < 50, "SCRATCH PnL should be reasonable"
