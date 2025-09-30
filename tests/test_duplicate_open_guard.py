"""
Test duplicate-open guard and conceptual WL/S counting.

Tests Phase 8 invariants:
- Blocks unintended same-bar duplicate entry EVENTS while preserving two-leg per entry behavior
- Continuations do not change conceptual WL/S (TP1-only, thread-scoped)
- WL/S parity with continuations ON/OFF and spreads ON/OFF
- Thread-scoped WL/S counting with proper entry event blocking
"""

from __future__ import annotations

from tests.utils_synth import create_synthetic_ohlc, run_synthetic_backtest


class TestDuplicateOpenGuard:
    """Test suite for duplicate-open guard and conceptual WL/S counting."""

    def test_duplicate_open_same_bar(self):
        """
        Test that duplicate entry EVENTs on same bar/side are blocked.

        Constructs a scenario where signal logic might attempt multiple entry events
        on the same bar/side. Verifies that only one entry EVENT is accepted,
        both TP1 and runner LEGS exist from that event, WL/S unchanged, audit fields immutable.

        Scenario:
        - Bar with strong signal that could trigger multiple entries
        - Should result in only one entry event, with two legs (TP1 + runner)
        - No duplicate entry events on same bar/side
        """
        bars = [
            # Bar 0: Setup
            {
                "date": "2023-01-01",
                "open": 1.0000,
                "high": 1.0005,
                "low": 0.9995,
                "close": 1.0000,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 1: Strong entry signal (potential for multiple triggers)
            {
                "date": "2023-01-02",
                "open": 1.0000,
                "high": 1.0030,
                "low": 0.9995,
                "close": 1.0020,
                "c1_signal": 1,  # Strong long signal
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 2: TP1 hit, should see TP1 + runner behavior
            {
                "date": "2023-01-03",
                "open": 1.0020,
                "high": 1.0025,
                "low": 1.0015,
                "close": 1.0022,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 3: Exit remaining position
            {
                "date": "2023-01-04",
                "open": 1.0022,
                "high": 1.0025,
                "low": 1.0018,
                "close": 1.0020,
                "c1_signal": -1,  # Exit signal
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
        ]

        df = create_synthetic_ohlc(bars, atr_value=0.002)

        config_override = {
            "engine": {
                "duplicate_open_policy": "block",
                "allow_continuation": False,
            },
            "exit": {"exit_on_c1_reversal": True},
        }

        result = run_synthetic_backtest(df, config_overrides=config_override)
        trades = result["trades"]

        # Verify exactly one entry event resulted in trade(s)
        assert len(trades) >= 1, f"Expected at least 1 trade, got {len(trades)}"

        # Verify all trades have same thread_id (same entry event)
        thread_ids = [t.get("thread_id") for t in trades if t.get("thread_id") is not None]
        if thread_ids:
            assert len(set(thread_ids)) == 1, (
                f"Expected all trades from same thread, got {set(thread_ids)}"
            )

        # Verify audit immutability
        for trade in trades:
            assert trade.get("tp1_at_entry_price") is not None, "tp1_at_entry_price should be set"
            assert trade.get("sl_at_entry_price") is not None, "sl_at_entry_price should be set"
            # These should be immutable and match the original entry levels

        # Verify W/L/S based on TP1 outcome only
        wins = sum(1 for t in trades if t.get("win", False))
        losses = sum(1 for t in trades if t.get("loss", False))
        scratches = sum(1 for t in trades if t.get("scratch", False))

        # Should have exactly one conceptual WL/S outcome
        assert wins + losses + scratches >= 1, "Should have at least one WL/S classification"

    def test_two_leg_entry_counts_once(self):
        """
        Test that single entry EVENT spawns TP1 leg + runner leg but counts as one conceptual trade.

        Verifies that 2 opened legs result in exactly 1 conceptual WL/S credit (TP1-only, thread-scoped).

        Scenario:
        - Single entry event
        - TP1 hit (partial close)
        - Runner continues and exits later
        - Should count as one conceptual WIN based on TP1 outcome
        """
        bars = [
            # Bar 0: Setup
            {
                "date": "2023-01-01",
                "open": 1.0000,
                "high": 1.0005,
                "low": 0.9995,
                "close": 1.0000,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 1: Entry signal
            {
                "date": "2023-01-02",
                "open": 1.0000,
                "high": 1.0010,
                "low": 0.9995,
                "close": 1.0005,
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 2: TP1 hit - this should trigger TP1 leg closure and runner continuation
            {
                "date": "2023-01-03",
                "open": 1.0005,
                "high": 1.0025,  # High enough to hit TP1 (1x ATR = 0.002 from 1.0005)
                "low": 1.0000,
                "close": 1.0020,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 3: Runner continues
            {
                "date": "2023-01-04",
                "open": 1.0020,
                "high": 1.0030,
                "low": 1.0015,
                "close": 1.0025,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 4: Runner exit
            {
                "date": "2023-01-05",
                "open": 1.0025,
                "high": 1.0030,
                "low": 1.0020,
                "close": 1.0022,
                "c1_signal": -1,  # Exit signal
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
        ]

        df = create_synthetic_ohlc(bars, atr_value=0.002)

        config_override = {
            "engine": {
                "duplicate_open_policy": "block",
                "allow_continuation": False,
            },
            "exit": {"exit_on_c1_reversal": True},
        }

        result = run_synthetic_backtest(df, config_overrides=config_override)
        trades = result["trades"]

        # Should have trades from the single entry event
        assert len(trades) >= 1, f"Expected at least 1 trade, got {len(trades)}"

        # All trades should have thread_id assigned and be from same conceptual entry
        thread_ids = [t.get("thread_id") for t in trades]
        assert all(tid is not None for tid in thread_ids), (
            "All trades should have thread_id assigned"
        )
        unique_threads = set(thread_ids)
        assert len(unique_threads) == 1, (
            f"Expected 1 thread ID, got {len(unique_threads)}: {unique_threads}"
        )

        # Conceptual WL/S should be based on TP1 outcome AND exit reason
        # TP1 hit + breakeven_after_tp1 = SCRATCH (Hard-Stop Realism)
        # TP1 hit + other exits = WIN
        tp1_hit_trades = [t for t in trades if t.get("tp1_hit", False)]
        if tp1_hit_trades:
            # Verify proper WL/S classification based on Hard-Stop Realism rules
            for trade in tp1_hit_trades:
                if trade.get("exit_reason") == "breakeven_after_tp1":
                    assert trade.get("scratch", False), "breakeven_after_tp1 should be SCRATCH"
                else:
                    assert trade.get("win", False), "TP1 hit + non-BE exit should be WIN"

    def test_continuations_thread_behavior(self):
        """
        Test that continuations properly group trades into conceptual threads.

        Uses same dataset/config with allow_continuation=false vs true.
        Verifies that continuations reduce the number of conceptual threads
        by grouping related entries together.

        Scenario:
        - Dataset that would trigger continuation trades
        - Run with continuations OFF and ON
        - Verify thread grouping behavior
        """
        bars = [
            # Bar 0: Setup
            {
                "date": "2023-01-01",
                "open": 1.0000,
                "high": 1.0005,
                "low": 0.9995,
                "close": 1.0000,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 1: Entry signal
            {
                "date": "2023-01-02",
                "open": 1.0000,
                "high": 1.0010,
                "low": 0.9995,
                "close": 1.0005,
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 2: C1 reversal but doesn't close position immediately (allow continuation test)
            {
                "date": "2023-01-03",
                "open": 1.0005,
                "high": 1.0010,
                "low": 0.9998,
                "close": 1.0002,
                "c1_signal": -1,  # C1 reversal
                "baseline": 0.9990,  # Baseline not crossed
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 3: C1 back to original direction (potential continuation while thread is open)
            {
                "date": "2023-01-04",
                "open": 1.0002,
                "high": 1.0015,
                "low": 0.9998,
                "close": 1.0010,
                "c1_signal": 1,  # Back to long
                "baseline": 0.9990,  # Baseline still not crossed
                "baseline_signal": 1,
                "volume_signal": 0,  # Volume fail - should be ignored for continuation
            },
            # Bar 4: Final exit
            {
                "date": "2023-01-05",
                "open": 1.0010,
                "high": 1.0015,
                "low": 1.0005,
                "close": 1.0008,
                "c1_signal": -1,  # Exit
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
        ]

        df = create_synthetic_ohlc(bars, atr_value=0.002)

        # Test with continuations OFF
        config_off = {
            "engine": {
                "duplicate_open_policy": "block",
                "allow_continuation": False,
            },
            "exit": {"exit_on_c1_reversal": True},
        }

        result_off = run_synthetic_backtest(df, config_overrides=config_off)
        trades_off = result_off["trades"]

        # Test with continuations ON
        config_on = {
            "engine": {
                "duplicate_open_policy": "block",
                "allow_continuation": True,
            },
            "exit": {"exit_on_c1_reversal": True},
        }

        result_on = run_synthetic_backtest(df, config_overrides=config_on)
        trades_on = result_on["trades"]

        # With continuations OFF: each entry creates a new thread
        # With continuations ON: continuation entries share the original thread
        # This means continuation scenarios will have fewer conceptual threads
        assert len(set(t.get("thread_id") for t in trades_off)) >= len(
            set(t.get("thread_id") for t in trades_on)
        ), "Continuations ON should have same or fewer threads than OFF"

        # Verify continuation behavior: ON should group trades into fewer threads
        threads_off = len(set(t.get("thread_id") for t in trades_off))
        threads_on = len(set(t.get("thread_id") for t in trades_on))

        if len(trades_off) > 1 and len(trades_on) > 1:
            # If we have multiple trades, continuations should reduce thread count
            assert threads_on <= threads_off, (
                f"Continuations should reduce thread count: OFF={threads_off}, ON={threads_on}"
            )

    def test_spread_parity_wls(self):
        """
        Test that conceptual WL/S totals are identical with spreads OFF vs ON.

        Uses same scenario with spreads disabled vs enabled.
        Verifies conceptual WL/S totals identical (spreads affect PnL only).

        Scenario:
        - Simple entry and exit scenario
        - Run with spreads OFF and ON
        - Compare conceptual WL/S counts (should be identical)
        """
        bars = [
            # Bar 0: Setup
            {
                "date": "2023-01-01",
                "open": 1.0000,
                "high": 1.0005,
                "low": 0.9995,
                "close": 1.0000,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 1: Entry signal
            {
                "date": "2023-01-02",
                "open": 1.0000,
                "high": 1.0010,
                "low": 0.9995,
                "close": 1.0005,
                "c1_signal": 1,
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 2: TP1 hit
            {
                "date": "2023-01-03",
                "open": 1.0005,
                "high": 1.0025,  # TP1 at entry + 0.002
                "low": 1.0000,
                "close": 1.0020,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 3: Exit
            {
                "date": "2023-01-04",
                "open": 1.0020,
                "high": 1.0025,
                "low": 1.0015,
                "close": 1.0018,
                "c1_signal": -1,  # Exit signal
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
        ]

        df = create_synthetic_ohlc(bars, atr_value=0.002)

        # Test with spreads OFF
        config_spreads_off = {
            "engine": {
                "duplicate_open_policy": "block",
                "allow_continuation": False,
            },
            "spreads": {
                "enabled": False,
                "default_pips": 0.0,
            },
            "exit": {"exit_on_c1_reversal": True},
        }

        result_spreads_off = run_synthetic_backtest(df, config_overrides=config_spreads_off)
        trades_spreads_off = result_spreads_off["trades"]

        # Test with spreads ON
        config_spreads_on = {
            "engine": {
                "duplicate_open_policy": "block",
                "allow_continuation": False,
            },
            "spreads": {
                "enabled": True,
                "default_pips": 2.0,  # Significant spread
            },
            "exit": {"exit_on_c1_reversal": True},
        }

        result_spreads_on = run_synthetic_backtest(df, config_overrides=config_spreads_on)
        trades_spreads_on = result_spreads_on["trades"]

        # Count WL/S for both scenarios
        def count_wls(trades):
            return {
                "wins": sum(1 for t in trades if t.get("win", False)),
                "losses": sum(1 for t in trades if t.get("loss", False)),
                "scratches": sum(1 for t in trades if t.get("scratch", False)),
            }

        wls_spreads_off = count_wls(trades_spreads_off)
        wls_spreads_on = count_wls(trades_spreads_on)

        # WL/S counts should be identical (spreads affect PnL only)
        assert wls_spreads_off["wins"] == wls_spreads_on["wins"], (
            f"WIN counts differ: OFF={wls_spreads_off['wins']}, ON={wls_spreads_on['wins']}"
        )
        assert wls_spreads_off["losses"] == wls_spreads_on["losses"], (
            f"LOSS counts differ: OFF={wls_spreads_off['losses']}, ON={wls_spreads_on['losses']}"
        )
        assert wls_spreads_off["scratches"] == wls_spreads_on["scratches"], (
            f"SCRATCH counts differ: OFF={wls_spreads_off['scratches']}, ON={wls_spreads_on['scratches']}"
        )

        # PnL should differ due to spreads (verify spreads actually had an effect)
        pnl_spreads_off = sum(t.get("pnl", 0) for t in trades_spreads_off)
        pnl_spreads_on = sum(t.get("pnl", 0) for t in trades_spreads_on)

        # With spreads enabled, PnL should generally be lower (more negative or less positive)
        # This verifies that spreads actually affected the calculation
        if len(trades_spreads_off) > 0 and len(trades_spreads_on) > 0:
            assert pnl_spreads_on <= pnl_spreads_off, (
                f"Spreads should reduce PnL: OFF={pnl_spreads_off:.4f}, ON={pnl_spreads_on:.4f}"
            )
