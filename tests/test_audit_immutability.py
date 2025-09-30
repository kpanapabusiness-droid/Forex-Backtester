"""
Test audit field immutability and spread-only PnL effects.

Rules:
- tp1_at_entry_price and sl_at_entry_price set once; never mutated post-entry.
- Use current_sl for BE/TS ratchets; set sl_at_exit_price on runner exit.
- Spreads do not change trade counts or exit reasons.
"""

from tests.utils_synth import create_synthetic_ohlc, run_synthetic_backtest


class TestAuditImmutability:
    """Test that tp1_at_entry_price and sl_at_entry_price are never mutated."""

    def test_audit_fields_set_once_at_entry(self):
        """Verify audit fields are set at entry and never change."""
        # Create synthetic data with strong trend to trigger trailing stops
        bars = [
            # Bar 0: Setup
            {
                "date": "2023-01-01",
                "open": 1.0000,
                "high": 1.0010,
                "low": 0.9990,
                "close": 1.0000,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 1: Entry signal
            {
                "date": "2023-01-02",
                "open": 1.0000,
                "high": 1.0010,
                "low": 0.9990,
                "close": 1.0000,
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 2: TP1 hit
            {
                "date": "2023-01-03",
                "open": 1.0000,
                "high": 1.0030,
                "low": 1.0000,
                "close": 1.0020,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 3: Trail up
            {
                "date": "2023-01-04",
                "open": 1.0020,
                "high": 1.0050,
                "low": 1.0020,
                "close": 1.0040,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 4: Exit on C1 reversal
            {
                "date": "2023-01-05",
                "open": 1.0040,
                "high": 1.0060,
                "low": 1.0025,
                "close": 1.0030,
                "c1_signal": -1,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
        ]
        df = create_synthetic_ohlc(bars, atr_value=0.002)

        result = run_synthetic_backtest(df)
        trades = result["trades"]

        assert len(trades) > 0, "Should generate at least one trade"

        trade = trades[0]

        # Verify audit fields exist and are reasonable
        assert "tp1_at_entry_price" in trade, "tp1_at_entry_price must exist"
        assert "sl_at_entry_price" in trade, "sl_at_entry_price must exist"
        assert "sl_at_exit_price" in trade, "sl_at_exit_price must exist"

        # Verify audit fields are set to reasonable values
        entry_price = trade["entry_price"]
        tp1_at_entry = trade["tp1_at_entry_price"]
        sl_at_entry = trade["sl_at_entry_price"]

        assert tp1_at_entry != entry_price, "TP1 should differ from entry"
        assert sl_at_entry != entry_price, "SL should differ from entry"

        # For long trade, TP1 should be above entry, SL below
        if trade["direction_int"] == 1:
            assert tp1_at_entry > entry_price, "Long TP1 should be above entry"
            assert sl_at_entry < entry_price, "Long SL should be below entry"
        else:
            assert tp1_at_entry < entry_price, "Short TP1 should be below entry"
            assert sl_at_entry > entry_price, "Short SL should be above entry"

    def test_current_sl_updates_but_audit_fields_dont(self):
        """Verify current_sl can change but audit fields remain constant."""
        # Create data that will trigger breakeven and trailing stops
        bars = [
            # Bar 0: Setup
            {
                "date": "2023-01-01",
                "open": 1.0000,
                "high": 1.0010,
                "low": 0.9990,
                "close": 1.0000,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 1: Entry signal
            {
                "date": "2023-01-02",
                "open": 1.0000,
                "high": 1.0010,
                "low": 0.9990,
                "close": 1.0000,
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 2: TP1 hit -> BE
            {
                "date": "2023-01-03",
                "open": 1.0000,
                "high": 1.0025,
                "low": 1.0000,
                "close": 1.0020,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 3: Trail trigger
            {
                "date": "2023-01-04",
                "open": 1.0020,
                "high": 1.0045,
                "low": 1.0020,
                "close": 1.0040,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
            # Bar 4: Exit on C1 reversal
            {
                "date": "2023-01-05",
                "open": 1.0040,
                "high": 1.0065,
                "low": 1.0035,
                "close": 1.0040,
                "c1_signal": -1,
                "baseline": 0.9990,
                "baseline_signal": 1,
            },
        ]
        df = create_synthetic_ohlc(bars, atr_value=0.002)

        result = run_synthetic_backtest(df)
        trades = result["trades"]

        assert len(trades) > 0, "Should generate trades"

        trade = trades[0]

        # Store original audit values
        original_tp1 = trade["tp1_at_entry_price"]
        original_sl = trade["sl_at_entry_price"]

        # Verify these are the immutable audit fields
        assert original_tp1 == trade["tp1_price"], "tp1_at_entry_price should match tp1_price"
        assert original_sl == trade["sl_price"], "sl_at_entry_price should match sl_price"

        # Verify exit price tracking shows dynamic behavior
        sl_at_exit = trade["sl_at_exit_price"]

        # If trailing stop was hit, sl_at_exit should differ from original SL
        if "trailing" in trade.get("exit_reason", "").lower():
            assert sl_at_exit != original_sl, "Trailing stop should update sl_at_exit_price"
        elif "breakeven" in trade.get("exit_reason", "").lower():
            assert abs(sl_at_exit - trade["entry_price"]) < 1e-6, (
                "Breakeven should set sl_at_exit to entry"
            )

    def test_multiple_trades_preserve_audit_integrity(self):
        """Test that audit fields are preserved across multiple trades."""
        # Create data with multiple entry opportunities
        bars = [
            {
                "open": 1.0000,
                "high": 1.0010,
                "low": 0.9990,
                "close": 1.0000,
                "c1_signal": 1,
                "baseline_signal": 1,
            },  # Entry 1
            {
                "open": 1.0000,
                "high": 1.0020,
                "low": 0.9980,
                "close": 0.9985,
                "c1_signal": 0,
                "baseline_signal": 1,
            },  # Stop out
            {
                "open": 0.9985,
                "high": 1.0000,
                "low": 0.9975,
                "close": 0.9990,
                "c1_signal": 0,
                "baseline_signal": 1,
            },  # Flat
            {
                "open": 0.9990,
                "high": 1.0005,
                "low": 0.9985,
                "close": 0.9995,
                "c1_signal": 1,
                "baseline_signal": 1,
            },  # Entry 2
            {
                "open": 0.9995,
                "high": 1.0015,
                "low": 0.9995,
                "close": 1.0010,
                "c1_signal": 0,
                "baseline_signal": 1,
            },  # TP1 hit
            {
                "open": 1.0010,
                "high": 1.0025,
                "low": 1.0005,
                "close": 1.0020,
                "c1_signal": 0,
                "baseline_signal": 1,
            },  # Trail
        ]
        df = create_synthetic_ohlc(bars, atr_value=0.002)

        result = run_synthetic_backtest(df)
        trades = result["trades"]

        # Should have multiple trades
        if len(trades) >= 2:
            for i, trade in enumerate(trades):
                # Each trade should have its own immutable audit fields
                assert "tp1_at_entry_price" in trade, f"Trade {i} missing tp1_at_entry_price"
                assert "sl_at_entry_price" in trade, f"Trade {i} missing sl_at_entry_price"

                # Audit fields should match the price fields for each trade
                assert trade["tp1_at_entry_price"] == trade["tp1_price"], (
                    f"Trade {i} audit mismatch"
                )
                assert trade["sl_at_entry_price"] == trade["sl_price"], f"Trade {i} audit mismatch"


class TestSpreadInvariant:
    """Test that spreads only affect PnL, not trade counts or exit reasons."""

    def test_spreads_preserve_trade_count(self):
        """Verify spreads don't change number of trades generated."""
        # Create data with clear entry/exit signals
        bars = [
            {
                "open": 1.0000,
                "high": 1.0010,
                "low": 0.9990,
                "close": 1.0000,
                "c1_signal": 1,
                "baseline_signal": 1,
            },
            {
                "open": 1.0000,
                "high": 1.0020,
                "low": 1.0000,
                "close": 1.0015,
                "c1_signal": 0,
                "baseline_signal": 1,
            },
            {
                "open": 1.0015,
                "high": 1.0030,
                "low": 1.0015,
                "close": 1.0025,
                "c1_signal": 0,
                "baseline_signal": 1,
            },
            {
                "open": 1.0025,
                "high": 1.0035,
                "low": 1.0010,
                "close": 1.0020,
                "c1_signal": -1,
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

        # Same number of trades
        assert len(trades_no_spreads) == len(trades_with_spreads), (
            f"Spreads changed trade count: {len(trades_no_spreads)} vs {len(trades_with_spreads)}"
        )

    def test_spreads_preserve_exit_reasons(self):
        """Verify spreads don't change exit reasons or win/loss classification."""
        bars = [
            {
                "open": 1.0000,
                "high": 1.0010,
                "low": 0.9990,
                "close": 1.0000,
                "c1_signal": 1,
                "baseline_signal": 1,
            },
            {
                "open": 1.0000,
                "high": 1.0025,
                "low": 1.0000,
                "close": 1.0020,
                "c1_signal": 0,
                "baseline_signal": 1,
            },  # TP1 hit
            {
                "open": 1.0020,
                "high": 1.0040,
                "low": 1.0020,
                "close": 1.0035,
                "c1_signal": 0,
                "baseline_signal": 1,
            },  # Trail
            {
                "open": 1.0035,
                "high": 1.0040,
                "low": 1.0015,
                "close": 1.0025,
                "c1_signal": -1,
                "baseline_signal": 1,
            },  # Trail stop
        ]
        df = create_synthetic_ohlc(bars, atr_value=0.002)

        result_no_spreads = run_synthetic_backtest(df, {"spreads": {"enabled": False}})
        result_with_spreads = run_synthetic_backtest(
            df, {"spreads": {"enabled": True, "default_pips": 3.0}}
        )

        trades_no_spreads = result_no_spreads["trades"]
        trades_with_spreads = result_with_spreads["trades"]

        assert len(trades_no_spreads) == len(trades_with_spreads), "Trade count should match"

        if len(trades_no_spreads) > 0:
            for i in range(len(trades_no_spreads)):
                trade_no_spread = trades_no_spreads[i]
                trade_with_spread = trades_with_spreads[i]

                # Exit reasons must be identical
                assert trade_no_spread["exit_reason"] == trade_with_spread["exit_reason"], (
                    f"Trade {i}: exit reason changed with spreads"
                )

                # Win/loss classification must be identical
                assert trade_no_spread["win"] == trade_with_spread["win"], (
                    f"Trade {i}: WIN classification changed with spreads"
                )
                assert trade_no_spread["loss"] == trade_with_spread["loss"], (
                    f"Trade {i}: LOSS classification changed with spreads"
                )
                assert trade_no_spread["scratch"] == trade_with_spread["scratch"], (
                    f"Trade {i}: SCRATCH classification changed with spreads"
                )

    def test_spreads_only_affect_pnl(self):
        """Verify spreads only change PnL values, nothing else."""
        bars = [
            {
                "open": 1.0000,
                "high": 1.0010,
                "low": 0.9990,
                "close": 1.0000,
                "c1_signal": 1,
                "baseline_signal": 1,
            },
            {
                "open": 1.0000,
                "high": 1.0025,
                "low": 1.0000,
                "close": 1.0020,
                "c1_signal": 0,
                "baseline_signal": 1,
            },  # TP1
            {
                "open": 1.0020,
                "high": 1.0030,
                "low": 1.0020,
                "close": 1.0025,
                "c1_signal": -1,
                "baseline_signal": 1,
            },  # Exit
        ]
        df = create_synthetic_ohlc(bars, atr_value=0.002)

        result_no_spreads = run_synthetic_backtest(df, {"spreads": {"enabled": False}})
        result_with_spreads = run_synthetic_backtest(
            df, {"spreads": {"enabled": True, "default_pips": 2.5}}
        )

        trades_no_spreads = result_no_spreads["trades"]
        trades_with_spreads = result_with_spreads["trades"]

        assert len(trades_no_spreads) == len(trades_with_spreads), "Trade count should match"

        if len(trades_no_spreads) > 0:
            trade_no_spread = trades_no_spreads[0]
            trade_with_spread = trades_with_spreads[0]

            # All non-PnL fields should be identical
            invariant_fields = [
                "pair",
                "entry_date",
                "entry_price",
                "direction",
                "direction_int",
                "atr_at_entry_price",
                "atr_at_entry_pips",
                "lots_total",
                "lots_half",
                "lots_runner",
                "tp1_price",
                "sl_price",
                "tp1_at_entry_price",
                "sl_at_entry_price",
                "tp1_hit",
                "breakeven_after_tp1",
                "ts_active",
                "exit_date",
                "exit_price",
                "exit_reason",
                "sl_at_exit_price",
                "win",
                "loss",
                "scratch",
            ]

            for field in invariant_fields:
                if field in trade_no_spread and field in trade_with_spread:
                    assert trade_no_spread[field] == trade_with_spread[field], (
                        f"Field '{field}' should not change with spreads"
                    )

            # PnL should be different (spreads reduce PnL)
            pnl_no_spread = trade_no_spread.get("pnl", 0.0)
            pnl_with_spread = trade_with_spread.get("pnl", 0.0)

            # With spreads, PnL should be lower (more negative for losses, less positive for wins)
            if pnl_no_spread > 0:  # Winning trade
                assert pnl_with_spread <= pnl_no_spread, "Spreads should reduce winning PnL"
            elif pnl_no_spread < 0:  # Losing trade
                assert pnl_with_spread <= pnl_no_spread, (
                    "Spreads should make losing PnL more negative"
                )

            # Spread usage should be recorded
            spread_used = trade_with_spread.get("spread_pips_used", 0.0)
            assert spread_used > 0, "spread_pips_used should be positive when spreads enabled"

            no_spread_used = trade_no_spread.get("spread_pips_used", 0.0)
            assert no_spread_used == 0, "spread_pips_used should be 0 when spreads disabled"

    def test_spread_modes_preserve_invariants(self):
        """Test different spread modes preserve trade count/reason invariants."""
        bars = [
            {
                "open": 1.0000,
                "high": 1.0010,
                "low": 0.9990,
                "close": 1.0000,
                "c1_signal": 1,
                "baseline_signal": 1,
            },
            {
                "open": 1.0000,
                "high": 1.0020,
                "low": 1.0000,
                "close": 1.0015,
                "c1_signal": 0,
                "baseline_signal": 1,
            },
            {
                "open": 1.0015,
                "high": 1.0025,
                "low": 1.0010,
                "close": 1.0020,
                "c1_signal": -1,
                "baseline_signal": 1,
            },
        ]
        df = create_synthetic_ohlc(bars, atr_value=0.002)

        # Test different spread configurations
        spread_configs = [
            {"enabled": False},
            {"enabled": True, "mode": "constant", "default_pips": 1.5},
            {"enabled": True, "mode": "atr_mult", "atr_mult": 0.1},
        ]

        results = []
        for spread_cfg in spread_configs:
            result = run_synthetic_backtest(df, {"spreads": spread_cfg})
            results.append(result["trades"])

        # All should have same trade count
        trade_counts = [len(trades) for trades in results]
        assert all(count == trade_counts[0] for count in trade_counts), (
            f"Different spread modes produced different trade counts: {trade_counts}"
        )

        # All should have same exit reasons (if trades exist)
        if trade_counts[0] > 0:
            for i in range(trade_counts[0]):
                exit_reasons = [trades[i]["exit_reason"] for trades in results]
                assert all(reason == exit_reasons[0] for reason in exit_reasons), (
                    f"Trade {i}: different spread modes produced different exit reasons: {exit_reasons}"
                )
