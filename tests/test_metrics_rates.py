# tests/test_metrics_rates.py
# Unit tests for compute_rates function

import pytest

from analytics.metrics import compute_rates


class TestMetricsRates:
    """Test win/loss/scratch rate calculations."""

    @pytest.mark.parametrize(
        "total_trades,wins,losses,scratches,expected",
        [
            # Case 1: Normal case with all trade types
            (
                100,
                30,
                20,
                50,
                {
                    "win_rate_ns": 0.6,  # 30/(30+20) = 0.6
                    "loss_rate_ns": 0.4,  # 20/(30+20) = 0.4
                    "scratch_rate_tot": 0.5,  # 50/100 = 0.5
                    "win_rate": 0.6,
                    "loss_rate": 0.4,
                    "scratch_rate": 0.5,
                },
            ),
            # Case 2: Zero non-scratch edge case
            (
                10,
                0,
                0,
                10,
                {
                    "win_rate_ns": 0.0,
                    "loss_rate_ns": 0.0,
                    "scratch_rate_tot": 1.0,  # 10/10 = 1.0
                    "win_rate": 0.0,
                    "loss_rate": 0.0,
                    "scratch_rate": 1.0,
                },
            ),
            # Case 3: No scratches
            (
                50,
                30,
                20,
                0,
                {
                    "win_rate_ns": 0.6,  # 30/(30+20) = 0.6
                    "loss_rate_ns": 0.4,  # 20/(30+20) = 0.4
                    "scratch_rate_tot": 0.0,  # 0/50 = 0.0
                    "win_rate": 0.6,
                    "loss_rate": 0.4,
                    "scratch_rate": 0.0,
                },
            ),
            # Case 4: All wins, no losses or scratches
            (
                25,
                25,
                0,
                0,
                {
                    "win_rate_ns": 1.0,  # 25/(25+0) = 1.0
                    "loss_rate_ns": 0.0,  # 0/(25+0) = 0.0
                    "scratch_rate_tot": 0.0,  # 0/25 = 0.0
                    "win_rate": 1.0,
                    "loss_rate": 0.0,
                    "scratch_rate": 0.0,
                },
            ),
            # Case 5: Zero trades edge case
            (
                0,
                0,
                0,
                0,
                {
                    "win_rate_ns": 0.0,
                    "loss_rate_ns": 0.0,
                    "scratch_rate_tot": 0.0,
                    "win_rate": 0.0,
                    "loss_rate": 0.0,
                    "scratch_rate": 0.0,
                },
            ),
        ],
    )
    def test_compute_rates(self, total_trades, wins, losses, scratches, expected):
        """Test compute_rates with various scenarios."""
        result = compute_rates(total_trades, wins, losses, scratches)

        # Check all expected keys are present
        for key in expected:
            assert key in result, f"Missing key: {key}"
            assert result[key] == pytest.approx(expected[key], abs=1e-6), (
                f"Mismatch for {key}: expected {expected[key]}, got {result[key]}"
            )

    def test_compute_rates_consistency(self):
        """Test that win_rate equals win_rate_ns and loss_rate equals loss_rate_ns."""
        result = compute_rates(100, 30, 20, 50)

        assert result["win_rate"] == result["win_rate_ns"]
        assert result["loss_rate"] == result["loss_rate_ns"]
        assert result["scratch_rate"] == result["scratch_rate_tot"]

    def test_compute_rates_sum_to_one(self):
        """Test that win_rate + loss_rate = 1.0 for non-scratch trades."""
        result = compute_rates(100, 30, 20, 50)

        # Win rate + loss rate should sum to 1.0 (they're based on non-scratch denominator)
        assert result["win_rate"] + result["loss_rate"] == pytest.approx(1.0, abs=1e-6)
        assert result["win_rate_ns"] + result["loss_rate_ns"] == pytest.approx(1.0, abs=1e-6)

    def test_negative_inputs_handled(self):
        """Test that negative inputs are handled gracefully."""
        # Should not crash, negative values get coerced to 0 by max()
        result = compute_rates(-10, -5, -3, -2)

        assert result["win_rate"] == 0.0
        assert result["loss_rate"] == 0.0
        assert result["scratch_rate"] == 0.0
