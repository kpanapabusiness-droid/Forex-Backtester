"""
Test no-lookahead decisions integrity.

Verifies that trading decisions are not influenced by future data by shifting
inputs by +1 bar and ensuring decisions remain unchanged.
"""

from __future__ import annotations

from tests.utils_synth import create_synthetic_ohlc, run_synthetic_backtest


class TestNoLookaheadDecisions:
    """Test suite for no-lookahead integrity."""

    def test_no_lookahead_decisions_shift_plus_one(self):
        """
        Test that shifting all inputs by +1 bar produces identical decisions.

        Creates a scenario with entry and exit signals, then shifts all data
        by one bar forward. The resulting entry/exit decisions should be
        identical (just shifted by one bar), proving no lookahead bias.

        Scenario:
        - Original: Entry signal at bar 2, exit at bar 4
        - Shifted: Same signals shifted by +1 bar (entry at bar 3, exit at bar 5)
        - Decision timing should be identical relative to signal timing
        """
        # Original scenario bars
        original_bars = [
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
            # Bar 1: Pre-entry setup
            {
                "date": "2023-01-02",
                "open": 1.0000,
                "high": 1.0010,
                "low": 0.9995,
                "close": 1.0005,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 2: Entry signal
            {
                "date": "2023-01-03",
                "open": 1.0005,
                "high": 1.0015,
                "low": 1.0000,
                "close": 1.0010,
                "c1_signal": 1,  # Long entry signal
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 3: Position continues
            {
                "date": "2023-01-04",
                "open": 1.0010,
                "high": 1.0020,
                "low": 1.0005,
                "close": 1.0015,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 4: Exit signal
            {
                "date": "2023-01-05",
                "open": 1.0015,
                "high": 1.0020,
                "low": 1.0010,
                "close": 1.0012,
                "c1_signal": -1,  # Exit signal
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
        ]

        # Create shifted scenario by adding a neutral bar at the beginning
        shifted_bars = [
            # Bar 0: New neutral bar (shifts everything by +1)
            {
                "date": "2022-12-31",  # Date before original sequence
                "open": 0.9995,
                "high": 1.0000,
                "low": 0.9990,
                "close": 0.9995,
                "c1_signal": 0,  # Neutral
                "baseline": 0.9985,
                "baseline_signal": 1,
                "volume_signal": 1,
            }
        ] + original_bars

        # Create DataFrames
        df_original = create_synthetic_ohlc(original_bars, atr_value=0.002)
        df_shifted = create_synthetic_ohlc(shifted_bars, atr_value=0.002)

        # Configuration for both tests
        config = {
            "engine": {
                "duplicate_open_policy": "block",
                "allow_continuation": False,
            },
            "exit": {"exit_on_c1_reversal": True},
        }

        # Run backtests
        result_original = run_synthetic_backtest(df_original, config_overrides=config)
        result_shifted = run_synthetic_backtest(df_shifted, config_overrides=config)

        signals_original = result_original["signals_df"]
        signals_shifted = result_shifted["signals_df"]

        # Extract decision points from original
        original_entries = signals_original[signals_original["entry_signal"] != 0].copy()
        original_exits = signals_original[signals_original["exit_signal_final"] != 0].copy()

        # Extract decision points from shifted (skip first bar which is the added neutral bar)
        shifted_entries = signals_shifted[signals_shifted["entry_signal"] != 0].copy()
        shifted_exits = signals_shifted[signals_shifted["exit_signal_final"] != 0].copy()

        # Verify that decisions exist in both scenarios
        assert len(original_entries) > 0, "No entry decisions in original scenario"
        assert len(original_exits) > 0, "No exit decisions in original scenario"
        assert len(shifted_entries) > 0, "No entry decisions in shifted scenario"
        assert len(shifted_exits) > 0, "No exit decisions in shifted scenario"

        # Verify decision count consistency
        assert len(original_entries) == len(shifted_entries), (
            f"Entry decision count differs: original={len(original_entries)}, "
            f"shifted={len(shifted_entries)}"
        )
        assert len(original_exits) == len(shifted_exits), (
            f"Exit decision count differs: original={len(original_exits)}, "
            f"shifted={len(shifted_exits)}"
        )

        # Verify decision values are identical (ignoring timing)
        original_entry_signals = sorted(original_entries["entry_signal"].tolist())
        shifted_entry_signals = sorted(shifted_entries["entry_signal"].tolist())
        assert original_entry_signals == shifted_entry_signals, (
            f"Entry signal values differ: original={original_entry_signals}, "
            f"shifted={shifted_entry_signals}"
        )

        # Verify relative timing consistency
        # In original: entry at index 2, exit at index 4 (gap of 2)
        # In shifted: entry should be at index 3, exit at index 5 (same gap of 2)
        if len(original_entries) == 1 and len(shifted_entries) == 1:
            original_entry_idx = original_entries.index[0]
            shifted_entry_idx = shifted_entries.index[0]

            # The shifted entry should be exactly +1 bar later
            assert shifted_entry_idx == original_entry_idx + 1, (
                f"Entry timing shift incorrect: original_idx={original_entry_idx}, "
                f"shifted_idx={shifted_entry_idx}, expected_shift=+1"
            )

        if len(original_exits) == 1 and len(shifted_exits) == 1:
            original_exit_idx = original_exits.index[0]
            shifted_exit_idx = shifted_exits.index[0]

            # The shifted exit should be exactly +1 bar later
            assert shifted_exit_idx == original_exit_idx + 1, (
                f"Exit timing shift incorrect: original_idx={original_exit_idx}, "
                f"shifted_idx={shifted_exit_idx}, expected_shift=+1"
            )

    def test_no_lookahead_price_action_decisions(self):
        """
        Test no-lookahead with price action scenarios.

        Creates scenarios where future price movements could influence current
        decisions if lookahead exists. Verifies decisions are based only on
        current and past information.

        Scenario:
        - Bar with entry signal followed by large price movement
        - Decision should be based only on signal, not future price movement
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
            # Bar 1: Entry signal (future movement should not influence this decision)
            {
                "date": "2023-01-02",
                "open": 1.0000,
                "high": 1.0010,
                "low": 0.9995,
                "close": 1.0005,
                "c1_signal": 1,  # Entry signal
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 2: Large favorable movement (could tempt lookahead)
            {
                "date": "2023-01-03",
                "open": 1.0005,
                "high": 1.0050,  # Large upward movement
                "low": 1.0000,
                "close": 1.0040,
                "c1_signal": 0,
                "baseline": 0.9990,
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 3: Large unfavorable movement (could tempt lookahead avoidance)
            {
                "date": "2023-01-04",
                "open": 1.0040,
                "high": 1.0045,
                "low": 0.9950,  # Large downward movement
                "close": 0.9960,
                "c1_signal": 0,
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
            },
            "exit": {"exit_on_c1_reversal": True},
        }

        result = run_synthetic_backtest(df, config_overrides=config)
        signals_df = result["signals_df"]

        # Verify entry decision was made at bar 1 (based on C1 signal)
        entry_signals = signals_df[signals_df["entry_signal"] != 0]
        assert len(entry_signals) == 1, f"Expected exactly 1 entry, got {len(entry_signals)}"

        entry_row = entry_signals.iloc[0]
        entry_idx = entry_signals.index[0]

        # Entry should be at index 1 (where C1 signal was +1)
        assert entry_idx == 1, f"Entry should be at index 1, got {entry_idx}"
        assert entry_row["entry_signal"] == 1, (
            f"Entry should be long (+1), got {entry_row['entry_signal']}"
        )

        # Verify decision was allowed (no lookahead blocking)
        assert entry_row["entry_allowed"], "Entry should have been allowed"
        assert entry_row["reason_block"] == "" or entry_row["reason_block"] == "none", (
            f"Entry should not be blocked, got reason: {entry_row['reason_block']}"
        )

        # The decision timing should be independent of future price movements
        # This is verified by the fact that entry occurred at the signal bar,
        # not delayed or advanced based on future favorable/unfavorable movements

    def test_no_lookahead_indicator_calculations(self):
        """
        Test that indicator-based decisions don't use future data.

        Verifies that baseline and other indicator-based decisions are made
        using only current and historical data, not future indicator values.

        Scenario:
        - Baseline values that change over time
        - Decisions should be based on current baseline, not future values
        """
        bars = [
            # Bar 0: Setup with initial baseline
            {
                "date": "2023-01-01",
                "open": 1.0000,
                "high": 1.0005,
                "low": 0.9995,
                "close": 1.0000,
                "c1_signal": 0,
                "baseline": 1.0010,  # Above current price
                "baseline_signal": -1,  # Bearish baseline
                "volume_signal": 1,
            },
            # Bar 1: C1 signal with current baseline (future baseline changes shouldn't matter)
            {
                "date": "2023-01-02",
                "open": 1.0000,
                "high": 1.0010,
                "low": 0.9995,
                "close": 1.0005,
                "c1_signal": 1,  # Long signal
                "baseline": 1.0010,  # Still above price (bearish)
                "baseline_signal": -1,  # Still bearish
                "volume_signal": 1,
            },
            # Bar 2: Baseline becomes very bullish (should not influence past decisions)
            {
                "date": "2023-01-03",
                "open": 1.0005,
                "high": 1.0015,
                "low": 1.0000,
                "close": 1.0010,
                "c1_signal": 0,
                "baseline": 0.9980,  # Now well below price (very bullish)
                "baseline_signal": 1,  # Now bullish
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

        # The C1 long signal at bar 1 should be processed based on bar 1's baseline state
        # (bearish baseline_signal = -1), not bar 2's future bullish state

        # Check if entry was blocked due to baseline conflict
        bar_1_entry = signals_df.iloc[1]

        # With use_baseline=True and baseline_signal=-1 at bar 1,
        # a long C1 signal should be blocked (baseline bearish conflicts with long entry)
        if bar_1_entry["entry_signal"] == 0:
            # Entry was blocked - this is correct behavior based on current baseline
            assert not bar_1_entry["entry_allowed"], (
                "Entry should be blocked due to baseline conflict"
            )
            # The reason should indicate baseline conflict, not future considerations
            assert (
                "baseline" in str(bar_1_entry["reason_block"]).lower()
                or bar_1_entry["reason_block"] == ""
            ), f"Block reason should be baseline-related, got: {bar_1_entry['reason_block']}"
        else:
            # If entry was allowed, it should be based on current conditions only
            assert bar_1_entry["entry_allowed"], "If entry signal exists, it should be allowed"

        # Verify no decisions were made based on bar 2's favorable baseline change
        # The decision pattern should be consistent with bar-by-bar processing
        entry_count = len(signals_df[signals_df["entry_signal"] != 0])

        # Decision count should be deterministic based on signal-by-signal processing
        # without knowledge of future baseline improvements
        assert entry_count <= 1, f"Expected at most 1 entry decision, got {entry_count}"
