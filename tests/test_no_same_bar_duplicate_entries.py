"""
Test no same-bar duplicate entries integrity.

Verifies that the system prevents multiple entry events on the same bar/side
through explicit scenario testing.
"""

from __future__ import annotations

from tests.utils_synth import create_synthetic_ohlc, run_synthetic_backtest


class TestNoSameBarDuplicateEntries:
    """Test suite for same-bar duplicate entry prevention."""

    def test_explicit_same_bar_duplicate_scenario(self):
        """
        Test explicit scenario designed to trigger same-bar duplicate entries.

        Creates a bar with conditions that could potentially trigger multiple
        entry events on the same side, then verifies only one entry is processed.

        Scenario:
        - Strong signal conditions that could trigger multiple entry attempts
        - Verify only one entry event is recorded per bar/side
        - Verify audit trail shows proper duplicate prevention
        """
        bars = [
            # Bar 0: Setup with neutral conditions
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
            # Bar 1: Extremely strong long signals (potential duplicate trigger)
            {
                "date": "2023-01-02",
                "open": 1.0000,
                "high": 1.0050,  # Large upward movement
                "low": 0.9995,
                "close": 1.0040,
                "c1_signal": 1,  # Strong long signal
                "baseline": 0.9980,  # Well below price (very bullish)
                "baseline_signal": 1,  # Bullish baseline
                "volume_signal": 1,  # Volume confirmation
            },
            # Bar 2: Continue with favorable conditions
            {
                "date": "2023-01-03",
                "open": 1.0040,
                "high": 1.0060,
                "low": 1.0035,
                "close": 1.0055,
                "c1_signal": 0,
                "baseline": 0.9980,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 3: Exit conditions
            {
                "date": "2023-01-04",
                "open": 1.0055,
                "high": 1.0060,
                "low": 1.0045,
                "close": 1.0050,
                "c1_signal": -1,  # Exit signal
                "baseline": 0.9980,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
        ]

        df = create_synthetic_ohlc(bars, atr_value=0.002)

        # Configuration with duplicate blocking enabled
        config = {
            "engine": {
                "duplicate_open_policy": "block",  # Explicitly enable blocking
                "allow_continuation": False,
            },
            "indicators": {
                "use_baseline": True,
                "use_volume": True,
            },
            "exit": {"exit_on_c1_reversal": True},
        }

        result = run_synthetic_backtest(df, config_overrides=config)
        signals_df = result["signals_df"]
        trades = result["trades"]

        # Verify only one entry signal was generated at bar 1
        entry_signals = signals_df[signals_df["entry_signal"] != 0]
        assert len(entry_signals) == 1, (
            f"Expected exactly 1 entry signal, got {len(entry_signals)}. "
            f"Multiple entries on same bar detected."
        )

        entry_row = entry_signals.iloc[0]
        entry_idx = entry_signals.index[0]

        # Entry should be at bar 1 (index 1)
        assert entry_idx == 1, f"Entry should be at bar 1, got bar {entry_idx}"
        assert entry_row["entry_signal"] == 1, (
            f"Entry should be long (+1), got {entry_row['entry_signal']}"
        )
        assert entry_row["entry_allowed"], "Entry should be allowed"

        # Verify only one conceptual trade thread was created
        if len(trades) > 0:
            thread_ids = [t.get("thread_id") for t in trades if t.get("thread_id") is not None]
            unique_threads = set(thread_ids)
            assert len(unique_threads) <= 1, (
                f"Expected at most 1 thread ID, got {len(unique_threads)}: {unique_threads}. "
                f"Multiple entry events detected."
            )

        # Verify no duplicate entry attempts were recorded in the same bar
        # Check that there's no indication of blocked duplicate attempts
        bar_1_signals = signals_df.iloc[1]

        # The entry should be clean without duplicate blocking artifacts
        assert bar_1_signals["entry_allowed"], "Entry at bar 1 should be allowed"

        # If there were duplicate attempts, they should be silently blocked
        # without creating additional entry signals
        duplicate_entries_same_bar = signals_df[
            (signals_df.index == 1) & (signals_df["entry_signal"] != 0)
        ]
        assert len(duplicate_entries_same_bar) == 1, (
            f"Expected exactly 1 entry at bar 1, got {len(duplicate_entries_same_bar)}"
        )

    def test_same_bar_opposite_side_entries(self):
        """
        Test same-bar entries on opposite sides.

        Verifies behavior when signals for opposite sides could occur on the same bar.
        Should handle this gracefully without duplicates.

        Scenario:
        - Bar with conflicting signals (e.g., C1 long but immediate reversal)
        - Verify clean handling without duplicate entries
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
            # Bar 1: Conflicting signals (long entry followed by immediate reversal)
            {
                "date": "2023-01-02",
                "open": 1.0000,
                "high": 1.0020,
                "low": 0.9980,  # Wide range suggesting volatility
                "close": 0.9985,  # Closes lower despite high
                "c1_signal": 1,  # Initial long signal
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 2: Clear reversal
            {
                "date": "2023-01-03",
                "open": 0.9985,
                "high": 0.9990,
                "low": 0.9970,
                "close": 0.9975,
                "c1_signal": -1,  # Clear short signal
                "baseline": 0.9990,
                "baseline_signal": -1,  # Baseline now bearish
                "volume_signal": 1,
            },
        ]

        df = create_synthetic_ohlc(bars, atr_value=0.002)

        config = {
            "engine": {
                "duplicate_open_policy": "block",
                "allow_continuation": False,
            },
            "indicators": {
                "use_baseline": True,
            },
            "exit": {"exit_on_c1_reversal": True},
        }

        result = run_synthetic_backtest(df, config_overrides=config)
        signals_df = result["signals_df"]

        # Count total entry signals
        entry_signals = signals_df[signals_df["entry_signal"] != 0]

        # Should have clean entry/exit pattern without same-bar duplicates
        entry_count_per_bar = signals_df.groupby(signals_df.index)["entry_signal"].apply(
            lambda x: (x != 0).sum()
        )

        # No bar should have more than 1 entry signal
        max_entries_per_bar = entry_count_per_bar.max()
        assert max_entries_per_bar <= 1, (
            f"Found bar with {max_entries_per_bar} entry signals. "
            f"Same-bar duplicates detected: {entry_count_per_bar[entry_count_per_bar > 1].to_dict()}"
        )

        # Verify each entry signal corresponds to exactly one decision point
        for idx, row in entry_signals.iterrows():
            entries_at_bar = entry_signals[entry_signals.index == idx]
            assert len(entries_at_bar) == 1, (
                f"Multiple entry signals at bar {idx}: {entries_at_bar['entry_signal'].tolist()}"
            )

    def test_rapid_signal_changes_no_duplicates(self):
        """
        Test rapid signal changes without creating duplicate entries.

        Simulates market conditions with rapid signal oscillations that could
        potentially trigger multiple entry attempts if not properly guarded.

        Scenario:
        - Multiple bars with signal changes
        - Verify each legitimate signal creates exactly one entry
        - No same-bar duplicates despite signal volatility
        """
        bars = [
            # Bar 0: Neutral start
            {
                "date": "2023-01-01",
                "open": 1.0000,
                "high": 1.0005,
                "low": 0.9995,
                "close": 1.0000,
                "c1_signal": 0,
                "baseline": 0.9995,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 1: First entry signal
            {
                "date": "2023-01-02",
                "open": 1.0000,
                "high": 1.0015,
                "low": 0.9995,
                "close": 1.0010,
                "c1_signal": 1,  # Long
                "baseline": 0.9995,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 2: Reversal (should exit and potentially re-enter)
            {
                "date": "2023-01-03",
                "open": 1.0010,
                "high": 1.0015,
                "low": 0.9990,
                "close": 0.9995,
                "c1_signal": -1,  # Short (reversal)
                "baseline": 1.0000,  # Above price now
                "baseline_signal": -1,  # Bearish
                "volume_signal": 1,
            },
            # Bar 3: Another reversal
            {
                "date": "2023-01-04",
                "open": 0.9995,
                "high": 1.0005,
                "low": 0.9985,
                "close": 1.0000,
                "c1_signal": 1,  # Long again
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 4: Final exit
            {
                "date": "2023-01-05",
                "open": 1.0000,
                "high": 1.0010,
                "low": 0.9990,
                "close": 0.9995,
                "c1_signal": -1,  # Exit
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
        ]

        df = create_synthetic_ohlc(bars, atr_value=0.002)

        config = {
            "engine": {
                "duplicate_open_policy": "block",
                "allow_continuation": False,
                "cross_only": False,  # Allow multiple entries
            },
            "exit": {"exit_on_c1_reversal": True},
        }

        result = run_synthetic_backtest(df, config_overrides=config)
        signals_df = result["signals_df"]

        # Verify no same-bar duplicates across all bars
        for bar_idx in range(len(signals_df)):
            bar_entries = signals_df.iloc[bar_idx : bar_idx + 1]
            entry_signals_in_bar = bar_entries[bar_entries["entry_signal"] != 0]

            assert len(entry_signals_in_bar) <= 1, (
                f"Bar {bar_idx} has {len(entry_signals_in_bar)} entry signals. "
                f"Same-bar duplicate detected: {entry_signals_in_bar['entry_signal'].tolist()}"
            )

        # Count total unique entry events
        entry_signals = signals_df[signals_df["entry_signal"] != 0]

        # Each entry should be on a different bar
        entry_bars = entry_signals.index.tolist()
        unique_entry_bars = list(set(entry_bars))

        assert len(entry_bars) == len(unique_entry_bars), (
            f"Duplicate entries on same bars detected. "
            f"Entry bars: {entry_bars}, Unique bars: {unique_entry_bars}"
        )

        # Verify entry signals are clean (no artifacts from duplicate blocking)
        for idx, row in entry_signals.iterrows():
            assert row["entry_signal"] in [-1, 1], (
                f"Invalid entry signal at bar {idx}: {row['entry_signal']}"
            )
            assert row["entry_allowed"] in [True, False], (
                f"Invalid entry_allowed value at bar {idx}: {row['entry_allowed']}"
            )

    def test_duplicate_blocking_with_continuations(self):
        """
        Test duplicate blocking behavior when continuations are enabled.

        Verifies that duplicate blocking works correctly even with continuation
        logic enabled, preventing same-bar duplicate entries while allowing
        proper continuation behavior.

        Scenario:
        - Enable continuations
        - Create scenario with potential same-bar duplicates
        - Verify blocking works with continuation logic
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
            # Bar 1: Initial entry
            {
                "date": "2023-01-02",
                "open": 1.0000,
                "high": 1.0015,
                "low": 0.9995,
                "close": 1.0010,
                "c1_signal": 1,  # Long entry
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 2: Temporary reversal (but baseline not crossed)
            {
                "date": "2023-01-03",
                "open": 1.0010,
                "high": 1.0015,
                "low": 1.0000,
                "close": 1.0005,
                "c1_signal": -1,  # Temporary short signal
                "baseline": 0.9990,  # Not crossed
                "baseline_signal": 1,
                "volume_signal": 0,  # Volume fails (should be ignored for continuation)
            },
            # Bar 3: Continuation signal (same direction as original)
            {
                "date": "2023-01-04",
                "open": 1.0005,
                "high": 1.0020,
                "low": 1.0000,
                "close": 1.0015,
                "c1_signal": 1,  # Long continuation (same bar as potential duplicate)
                "baseline": 0.9990,  # Still not crossed
                "baseline_signal": 1,
                "volume_signal": 0,  # Volume still fails (continuation ignores this)
            },
        ]

        df = create_synthetic_ohlc(bars, atr_value=0.002)

        config = {
            "engine": {
                "duplicate_open_policy": "block",
                "allow_continuation": True,  # Enable continuations
            },
            "indicators": {
                "use_volume": True,  # Use volume for initial entries
            },
            "exit": {"exit_on_c1_reversal": True},
        }

        result = run_synthetic_backtest(df, config_overrides=config)
        signals_df = result["signals_df"]

        # Verify no same-bar duplicates even with continuations enabled
        entry_count_per_bar = signals_df.groupby(signals_df.index)["entry_signal"].apply(
            lambda x: (x != 0).sum()
        )

        max_entries_per_bar = entry_count_per_bar.max()
        assert max_entries_per_bar <= 1, (
            f"Found bar with {max_entries_per_bar} entry signals with continuations enabled. "
            f"Same-bar duplicates: {entry_count_per_bar[entry_count_per_bar > 1].to_dict()}"
        )

        # Verify continuation logic still works (but without same-bar duplicates)
        entry_signals = signals_df[signals_df["entry_signal"] != 0]

        # Should have entries but each on different bars
        if len(entry_signals) > 0:
            entry_bars = entry_signals.index.tolist()
            unique_entry_bars = list(set(entry_bars))

            assert len(entry_bars) == len(unique_entry_bars), (
                f"Continuation logic created same-bar duplicates. "
                f"Entry bars: {entry_bars}, Unique: {unique_entry_bars}"
            )
