"""
Test that continuations preserve WL/S invariants.

Tests that when allow_continuation affects threading behavior,
the conceptual WL/S counts should remain constant when evaluated
at the thread level (not individual trade level).
"""

from __future__ import annotations

from tests.utils_synth import create_synthetic_ohlc, run_synthetic_backtest


class TestContinuationsWLSInvariant:
    """Test that continuations preserve conceptual WL/S invariants."""

    def test_continuations_preserve_thread_wls_count(self):
        """
        Test that continuations within a thread don't create additional WL/S credits.

        Scenario: When continuations are enabled, multiple trades within the same
        conceptual thread should still count as one WL/S outcome based on TP1.
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
            # Bar 2: Exit first position
            {
                "date": "2023-01-03",
                "open": 1.0005,
                "high": 1.0010,
                "low": 0.9998,
                "close": 1.0002,
                "c1_signal": -1,  # C1 reversal - close position
                "baseline": 0.9990,  # Baseline not crossed
                "baseline_signal": 1,
                "volume_signal": 1,
            },
            # Bar 3: Continuation entry (same thread if continuations enabled)
            {
                "date": "2023-01-04",
                "open": 1.0002,
                "high": 1.0015,
                "low": 0.9998,
                "close": 1.0010,
                "c1_signal": 1,  # Back to long - potential continuation
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

        config_with_continuations = {
            "engine": {
                "duplicate_open_policy": "block",
                "allow_continuation": True,
            },
            "exit": {"exit_on_c1_reversal": True},
        }

        result = run_synthetic_backtest(df, config_overrides=config_with_continuations)
        trades = result["trades"]

        # Should have trades from both entries
        assert len(trades) >= 2, f"Expected at least 2 trades, got {len(trades)}"

        # Count threads and WL/S outcomes
        thread_outcomes = {}
        for trade in trades:
            thread_id = trade.get("thread_id")
            if thread_id is not None:
                if thread_id not in thread_outcomes:
                    thread_outcomes[thread_id] = {
                        "trades": [],
                        "tp1_hit": False,
                        "win": False,
                        "loss": False,
                        "scratch": False,
                    }

                thread_outcomes[thread_id]["trades"].append(trade)

                # Thread-level TP1 tracking: if any trade in thread hit TP1
                if trade.get("tp1_hit", False):
                    thread_outcomes[thread_id]["tp1_hit"] = True

                # Track individual trade WL/S
                if trade.get("win", False):
                    thread_outcomes[thread_id]["win"] = True
                elif trade.get("loss", False):
                    thread_outcomes[thread_id]["loss"] = True
                elif trade.get("scratch", False):
                    thread_outcomes[thread_id]["scratch"] = True

        # Verify that multiple trades within the same thread
        # don't create multiple conceptual WL/S credits
        total_conceptual_outcomes = 0
        for thread_id, outcome in thread_outcomes.items():
            # Each thread should contribute exactly one conceptual outcome
            wls_count = sum([outcome["win"], outcome["loss"], outcome["scratch"]])
            assert wls_count <= 1, f"Thread {thread_id} has multiple WL/S outcomes: {outcome}"

            if wls_count == 1:
                total_conceptual_outcomes += 1

        # Should have exactly one conceptual outcome per thread
        assert total_conceptual_outcomes == len(thread_outcomes), (
            f"Expected {len(thread_outcomes)} conceptual outcomes, got {total_conceptual_outcomes}"
        )

        # Verify that if there's a thread with multiple trades,
        # the thread outcome is determined correctly (TP1-based)
        for thread_id, outcome in thread_outcomes.items():
            if len(outcome["trades"]) > 1:
                # This thread had continuations
                print(
                    f"Thread {thread_id} has {len(outcome['trades'])} trades (continuation scenario)"
                )

                # The thread's WL/S should be based on whether TP1 was hit
                # in any trade within the thread
                if outcome["tp1_hit"]:
                    # If TP1 was hit, should be WIN (unless breakeven exit)
                    # Golden Standard: TP1 hit = WIN regardless of runner outcome
                    assert outcome["win"], "Thread with TP1 hit should be WIN (Golden Standard)"
                else:
                    # No TP1 hit - should be LOSS or SCRATCH based on exit reason
                    has_sl = any(
                        "stoploss" in str(t.get("exit_reason", "")).lower()
                        for t in outcome["trades"]
                    )
                    if has_sl:
                        assert outcome["loss"], "Thread with SL (no TP1) should be LOSS"
                    else:
                        assert outcome["scratch"], (
                            "Thread with system exit (no TP1) should be SCRATCH"
                        )
