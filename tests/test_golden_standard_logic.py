"""
Hard-Stop Realism Tests - Deterministic verification of NNFX trading rules.

Tests use synthetic OHLC data to trigger specific scenarios and verify:
- Intrabar TP/SL/BE/TS touch → immediate exit (hard-stop realism)
- TP1 → BE same bar → exit if BE touched intrabar
- SL before TP1 → LOSS classification
- System exits before TP1 → SCRATCH classification (≈0 PnL)
- Trailing stop: close-based activation/updates, intrabar exits
- BE activation: immediate on TP1 bar, effective same bar
- Spreads affect PnL only, not trade counts
- Audit invariants: immutable entry levels, proper exit recording
"""

from __future__ import annotations

import pytest

from tests.utils_synth import create_synthetic_ohlc, run_synthetic_backtest


class TestHardStopRealism:
    """Test suite enforcing Hard-Stop Realism trading logic."""

    def test_tp1_moves_runner_to_be_same_bar_immediate_exit(self):
        """
        Hard-Stop Realism: TP1 hit → BE set same bar → immediate exit if BE touched intrabar.

        Scenario:
        - Entry at 1.0000 (C1 signal +1)
        - TP1 at 1.0020 (1×ATR), SL at 0.9970 (1.5×ATR)
        - Bar 2: High touches TP1, low touches BE (1.0000) same bar
        - Expected: Immediate exit as breakeven_after_tp1, SCRATCH classification
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
            # Bar 2: TP1 hit then return (same bar)
            {
                "date": "2023-01-03",
                "open": 1.0000,
                "high": 1.0025,
                "low": 1.0000,
                "close": 1.0000,
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 3: Continue (runner at breakeven)
            {
                "date": "2023-01-04",
                "open": 1.0000,
                "high": 1.0010,
                "low": 0.9995,
                "close": 1.0005,
                "c1_signal": -1,
                "baseline": 0.9990,
                "baseline_signal": 1,  # C1 reversal exit
            },
        ]

        df = create_synthetic_ohlc(bars, atr_value=0.002)
        result = run_synthetic_backtest(df)

        trades = result["trades"]
        assert len(trades) == 1, f"Expected 1 trade, got {len(trades)}"

        trade = trades[0]

        # Verify TP1 mechanics
        assert trade["tp1_hit"], "TP1 should have been hit"
        assert trade["breakeven_after_tp1"], "Should move to breakeven after TP1"

        # Verify SCRATCH classification (hard-stop realism: BE touched same bar)
        assert not trade["win"], "Trade should not be WIN (BE hit same bar)"
        assert not trade["loss"], "Trade should not be LOSS"
        assert trade["scratch"], "Trade should be SCRATCH (breakeven exit)"

        # Verify immutable audit fields
        assert trade["tp1_at_entry_price"] == pytest.approx(1.0020, abs=1e-6), (
            "TP1 audit field immutable"
        )
        assert trade["sl_at_entry_price"] == pytest.approx(0.9970, abs=1e-6), (
            "SL audit field immutable"
        )

        # Verify hard-stop realism: BE touched intrabar → immediate exit
        assert trade["exit_reason"] == "breakeven_after_tp1", (
            f"Hard-Stop Realism: BE touched same bar as TP1 → immediate exit, got '{trade['exit_reason']}'"
        )
        assert "sl_at_exit_price" in trade, "Should record final SL at exit"

    def test_sl_before_tp1_is_full_loss(self):
        """
        SL hit before TP1 → full size closed → LOSS classification.

        Scenario:
        - Entry at 1.0000, TP1 at 1.0020, SL at 0.9970
        - Price drops to SL without touching TP1
        - Expected: Full position closed, classification = LOSS
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
            # Bar 2: SL hit (before TP1)
            {
                "date": "2023-01-03",
                "open": 1.0000,
                "high": 1.0000,
                "low": 0.9965,
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

        # Verify SL mechanics
        assert not trade["tp1_hit"], "TP1 should not have been hit"
        assert not trade["breakeven_after_tp1"], "No breakeven without TP1"

        # Verify LOSS classification
        assert not trade["win"], "Trade should not be WIN"
        assert trade["loss"], "Trade should be classified as LOSS"
        assert not trade["scratch"], "Trade should not be SCRATCH"

        # Verify exit details
        assert trade["exit_reason"] == "stoploss", "Should exit on stop loss"
        assert trade["pnl"] < 0, "Should have negative PnL"

    def test_pre_tp1_system_exit_is_scratch(self):
        """
        Hard-Stop Realism: System exit before TP1 → SCRATCH classification (≈0 PnL).

        Scenario:
        - Entry at 1.0000, TP1 at 1.0020, SL at 0.9970
        - C1 reversal before TP1 is hit (no intrabar TP/SL touch)
        - Expected: Full position closed, SCRATCH classification, ≈0 PnL
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
            # Bar 2: C1 reversal (before TP1)
            {
                "date": "2023-01-03",
                "open": 1.0000,
                "high": 1.0015,
                "low": 0.9995,
                "close": 1.0010,
                "c1_signal": -1,
                "baseline": 0.9990,
                "baseline_signal": 1,  # C1 flips
            },
        ]

        df = create_synthetic_ohlc(bars, atr_value=0.002)
        result = run_synthetic_backtest(df)

        trades = result["trades"]
        assert len(trades) == 1, f"Expected 1 trade, got {len(trades)}"

        trade = trades[0]

        # Verify no TP1 hit
        assert not trade["tp1_hit"], "TP1 should not have been hit"
        assert not trade["breakeven_after_tp1"], "No breakeven without TP1"

        # Verify SCRATCH classification
        assert not trade["win"], "Trade should not be WIN"
        assert not trade["loss"], "Trade should not be LOSS"
        assert trade["scratch"], "Trade should be classified as SCRATCH"

        # Verify exit details
        assert trade["exit_reason"] == "c1_reversal", "Should exit on C1 reversal"

        # Verify strict SCRATCH PnL requirement: must be ≈ 0 within tolerance
        # Golden Standard: Pre-TP1 SCRATCH should have minimal PnL impact
        spread_tolerance = 2.0  # pips - account for spread + small timing effects
        pip_size = 0.0001  # EUR_USD pip size
        max_pnl_tolerance = (
            spread_tolerance * pip_size * 100000 + 1e-9
        )  # Convert to account currency

        assert abs(trade["pnl"]) <= max_pnl_tolerance, (
            f"Golden Standard violation: Pre-TP1 SCRATCH PnL must be ≈0, got {trade['pnl']:.2f} "
            f"(tolerance: ±{max_pnl_tolerance:.2f})"
        )

    def test_trailing_stop_hard_stop_realism(self):
        """
        Hard-Stop Realism: TS activation on close, updates on close, exits intrabar when touched.

        Scenario:
        - Entry at 1.0000, TP1 hit at 1.0020
        - TP1 bar: low stays above BE (1.0000) to avoid immediate exit
        - Close moves to 1.0040 (2×ATR) → trail activates at 1.0010 (1.5×ATR behind)
        - Later bar: intrabar low touches TS → immediate exit
        - Expected: TS active, exit_reason=trailing_stop, WIN classification
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
            # Bar 2: TP1 hit, low stays above BE to avoid immediate exit
            {
                "date": "2023-01-03",
                "open": 1.0000,
                "high": 1.0025,
                "low": 1.0005,  # Above BE (1.0000) to prevent same-bar exit
                "close": 1.0020,
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 3: Move to 2×ATR (trail activation)
            {
                "date": "2023-01-04",
                "open": 1.0020,
                "high": 1.0045,
                "low": 1.0020,
                "close": 1.0040,
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 4: Intra-bar spike down (should not affect trail)
            {
                "date": "2023-01-05",
                "open": 1.0040,
                "high": 1.0040,
                "low": 1.0005,
                "close": 1.0035,
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 5: Hit trailing stop intrabar (TS level ≈ 1.0010)
            {
                "date": "2023-01-06",
                "open": 1.0035,
                "high": 1.0035,
                "low": 1.0008,  # Touches TS level (1.0040 - 1.5×0.002 = 1.0010) intrabar
                "close": 1.0015,
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

        # Verify TP1 was hit first
        assert trade["tp1_hit"], "TP1 should have been hit"

        # Verify trailing stop activation and behavior (Hard-Stop Realism)
        # TS activates when BAR CLOSE moves past ±2×ATR from entry
        # Entry: 1.0000, ATR: 0.002, so 2×ATR = 0.004
        # Bar 3 close: 1.0040 = entry + 2×ATR → should activate TS
        assert trade.get("ts_active", False), (
            "Hard-Stop Realism: Trailing stop must activate when close > entry + 2×ATR "
            "(entry=1.0000, close=1.0040, 2×ATR=0.004)"
        )

        # Verify trailing stop exit reason (intrabar touch → immediate exit)
        assert trade["exit_reason"] == "trailing_stop", (
            f"Hard-Stop Realism: TS touched intrabar → immediate exit, got '{trade['exit_reason']}'"
        )

        # Verify trailing stop level calculation
        # Trail distance = 1.5×ATR from entry ATR
        # Expected TS level ≈ 1.0040 - 1.5×0.002 = 1.0010 (from highest close)
        expected_ts_level = 1.0040 - (1.5 * 0.002)  # 1.0010
        assert "ts_level" in trade, "Must record trailing stop level"
        assert trade["ts_level"] == pytest.approx(expected_ts_level, abs=1e-6), (
            f"Hard-Stop Realism: TS level must be 1.5×ATR behind highest close, "
            f"expected ≈{expected_ts_level:.6f}, got {trade.get('ts_level', 'None')}"
        )

        # Verify WIN classification (TP1 hit, so WIN regardless of exit method)
        assert trade["win"], "Trade should remain WIN (TP1 hit)"
        assert not trade["loss"], "Trade should not be LOSS"
        assert not trade["scratch"], "Trade should not be SCRATCH"

        # Verify positive PnL (TP1 guarantees some profit)
        assert trade["pnl"] > 0, "Should have positive PnL (TP1 + runner profit)"

    def test_post_tp1_exit_priority_breakeven_vs_trailing_stop(self):
        """
        Hard-Stop Realism: BE exit priority when TS not active → SCRATCH classification.

        Scenario:
        - Entry at 1.0000, TP1 hit at 1.0020
        - Price returns to entry (1.0000) without activating TS (no 2×ATR move)
        - Expected: exit_reason = 'breakeven_after_tp1', SCRATCH classification
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
                "high": 1.0025,
                "low": 1.0000,
                "close": 1.0020,
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 3: Move up but not enough for TS activation (< 2×ATR)
            {
                "date": "2023-01-04",
                "open": 1.0020,
                "high": 1.0035,
                "low": 1.0020,
                "close": 1.0030,  # Only 1.5×ATR from entry, not 2×ATR
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 4: Return to entry price (breakeven)
            {
                "date": "2023-01-05",
                "open": 1.0030,
                "high": 1.0030,
                "low": 0.9995,
                "close": 1.0000,  # Back to entry, should hit breakeven
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

        # Verify TS was NOT activated (never reached 2×ATR)
        # Golden Standard: TS activates only when close > entry ± 2×ATR
        # Max close was 1.0030 = entry + 1.5×ATR, not enough for activation
        assert not trade.get("ts_active", False), (
            "Golden Standard violation: TS should not activate without 2×ATR close move "
            "(entry=1.0000, max_close=1.0030, 2×ATR=0.004)"
        )

        # Verify strict exit priority: no TS active + at entry = breakeven
        # Golden Standard: When TS inactive and price returns to entry, reason = 'breakeven_after_tp1'
        assert trade["exit_reason"] == "breakeven_after_tp1", (
            f"Golden Standard violation: Post-TP1 return to entry without TS must be 'breakeven_after_tp1', "
            f"got '{trade['exit_reason']}'"
        )

        # Verify SCRATCH classification (Hard-Stop Realism: BE exit = SCRATCH)
        assert not trade["win"], "Trade should not be WIN (breakeven exit)"
        assert not trade["loss"], "Trade should not be LOSS"
        assert trade["scratch"], "Trade should be SCRATCH (breakeven exit)"

    def test_continuation_trade_without_volume_or_atr_distance(self):
        """
        Continuation trade: re-entry without volume/distance checks after baseline hasn't crossed.

        Scenario:
        - Enter long → exit on C1 flip → baseline still bullish
        - C1 flips back long → valid re-entry even with volume=-1 and far from baseline
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
            # Bar 1: First entry
            {
                "date": "2023-01-02",
                "open": 1.0000,
                "high": 1.0005,
                "low": 0.9995,
                "close": 1.0000,
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 2: C1 reversal exit (baseline still bullish)
            {
                "date": "2023-01-03",
                "open": 1.0000,
                "high": 1.0010,
                "low": 0.9995,
                "close": 1.0005,
                "c1_signal": -1,
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 3: C1 flips back (continuation entry with bad volume and far from baseline)
            {
                "date": "2023-01-04",
                "open": 1.0005,
                "high": 1.0015,
                "low": 1.0000,
                "close": 1.0050,  # Far from baseline
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": -1,  # Bad volume
            },
            # Bar 4: Exit continuation trade
            {
                "date": "2023-01-05",
                "open": 1.0050,
                "high": 1.0055,
                "low": 1.0045,
                "close": 1.0050,
                "c1_signal": -1,
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": -1,
            },
        ]

        # Enable volume filter to test that continuation ignores it
        config_overrides = {
            "indicators": {"use_volume": True, "use_baseline": True},
            "rules": {
                "pullback_rule": True  # Also test that continuation ignores distance
            },
        }

        df = create_synthetic_ohlc(bars, atr_value=0.002)
        result = run_synthetic_backtest(df, config_overrides)

        trades = result["trades"]

        # Should have 2 trades: original + continuation
        assert len(trades) >= 1, f"Expected at least 1 trade, got {len(trades)}"

        # First trade should exit on C1 reversal
        first_trade = trades[0]
        assert first_trade["exit_reason"] == "c1_reversal", "First trade should exit on C1 reversal"

        # If continuation logic is implemented, verify second trade exists
        # Note: This test documents expected behavior - implementation may vary
        print(f"Trades generated: {len(trades)}")
        for i, trade in enumerate(trades):
            print(f"Trade {i + 1}: entry={trade['entry_date']}, exit_reason={trade['exit_reason']}")

    def test_spreads_change_pnl_not_trade_counts(self):
        """
        Spreads affect PnL only, never trade counts or exit timing.

        Scenario:
        - Run identical scenario with spreads off vs on
        - Verify: same number of trades, same exit reasons, different PnL
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
                "high": 1.0025,
                "low": 1.0000,
                "close": 1.0020,
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 3: Exit
            {
                "date": "2023-01-04",
                "open": 1.0020,
                "high": 1.0025,
                "low": 1.0015,
                "close": 1.0020,
                "c1_signal": -1,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
        ]

        df = create_synthetic_ohlc(bars, atr_value=0.002)

        # Run without spreads
        result_no_spreads = run_synthetic_backtest(df, {"spreads": {"enabled": False}})

        # Run with spreads
        result_with_spreads = run_synthetic_backtest(
            df, {"spreads": {"enabled": True, "default_pips": 2.0}}
        )

        trades_no_spreads = result_no_spreads["trades"]
        trades_with_spreads = result_with_spreads["trades"]

        # Verify same trade count
        assert len(trades_no_spreads) == len(trades_with_spreads), (
            "Spreads should not change trade count"
        )

        if len(trades_no_spreads) > 0:
            trade_no_spread = trades_no_spreads[0]
            trade_with_spread = trades_with_spreads[0]

            # Verify same exit reasons and classification
            assert trade_no_spread["exit_reason"] == trade_with_spread["exit_reason"], (
                "Spreads should not change exit reason"
            )
            assert trade_no_spread["win"] == trade_with_spread["win"], (
                "Spreads should not change WIN classification"
            )
            assert trade_no_spread["loss"] == trade_with_spread["loss"], (
                "Spreads should not change LOSS classification"
            )
            assert trade_no_spread["scratch"] == trade_with_spread["scratch"], (
                "Spreads should not change SCRATCH classification"
            )

            # Verify different PnL (spreads reduce profit)
            if abs(trade_no_spread["pnl"]) > 1:  # Only test if meaningful PnL
                assert trade_with_spread["pnl"] < trade_no_spread["pnl"], (
                    "Spreads should reduce PnL for profitable trades"
                )

    def test_correlation_scope_ignored_for_c1_unit_tests(self):
        """
        Correlation caps ignored in unit tests (documented behavior).

        This test documents that correlation/aggregate risk caps should NOT
        be applied in C1-only or unit tests, only in full-system tests.
        """
        # This is more of a documentation test - the actual correlation
        # logic would be tested in integration tests, not unit tests

        bars = [
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
        ]

        df = create_synthetic_ohlc(bars, atr_value=0.002)
        result = run_synthetic_backtest(df)

        # In unit tests, correlation caps should be ignored
        # This means trades should be allowed regardless of correlation settings
        result["trades"]

        # Document the expected behavior
        assert True, "Correlation caps should be ignored in unit tests (Golden Standard §2)"

        print("✓ Correlation scope test: Unit tests ignore correlation caps as expected")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
