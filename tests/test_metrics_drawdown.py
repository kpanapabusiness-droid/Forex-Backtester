# tests/test_metrics_drawdown.py
# Unit tests for compute_max_drawdown_pct (canonical max drawdown)

import pandas as pd
import pytest

from analytics.metrics import compute_max_drawdown_pct, max_drawdown_pct


class TestComputeMaxDrawdownPct:
    """Canonical: peak = cummax(E), dd = (E - peak)/peak, max_dd_pct = dd.min() * 100."""

    def test_equity_100_110_105_120_90_gives_minus_25(self):
        equity = pd.Series([100.0, 110.0, 105.0, 120.0, 90.0])
        got = compute_max_drawdown_pct(equity)
        assert got == pytest.approx(-25.0, abs=0.01)

    def test_empty_series_returns_zero(self):
        assert compute_max_drawdown_pct(pd.Series(dtype=float)) == 0.0

    def test_single_value_returns_zero(self):
        assert compute_max_drawdown_pct(pd.Series([100.0])) == 0.0

    def test_monotonic_up_returns_zero(self):
        equity = pd.Series([100.0, 110.0, 120.0])
        assert compute_max_drawdown_pct(equity) == 0.0

    def test_fold_dds_minus_5_minus_12_worst_is_minus_12(self):
        equity_a = pd.Series([100.0, 95.0])
        equity_b = pd.Series([100.0, 88.0])
        assert compute_max_drawdown_pct(equity_a) == pytest.approx(-5.0, abs=0.01)
        assert compute_max_drawdown_pct(equity_b) == pytest.approx(-12.0, abs=0.01)


class TestMaxDrawdownPctDataFrame:
    """max_drawdown_pct(equity_df) uses compute_max_drawdown_pct on df['equity']."""

    def test_dataframe_with_equity_column(self):
        df = pd.DataFrame({"date": [1, 2, 3, 4, 5], "equity": [100.0, 110.0, 105.0, 120.0, 90.0]})
        got = max_drawdown_pct(df)
        assert got == pytest.approx(-25.0, abs=0.01)
