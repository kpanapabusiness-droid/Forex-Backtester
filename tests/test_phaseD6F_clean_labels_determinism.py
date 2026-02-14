"""
Phase D-6F: Determinism tests for clean opportunity labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from labels.clean_opportunity import compute_clean_labels_for_pair


def _make_toy_ohlc(n: int) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    base = np.cumsum(np.random.randn(n) * 0.5) + 1.0
    return pd.DataFrame({
        "date": dates,
        "open": base,
        "high": base + np.abs(np.random.randn(n)) * 0.01,
        "low": base - np.abs(np.random.randn(n)) * 0.01,
        "close": base + np.random.randn(n) * 0.005,
        "volume": np.zeros(n),
    })


def test_determinism_same_input_twice() -> None:
    """Run compute twice on same toy df; assert identical results."""
    df = _make_toy_ohlc(100)
    out1 = compute_clean_labels_for_pair(
        df, "TEST",
        date_start="2019-01-01", date_end="2026-01-01",
        atr_period=14,
    )
    out2 = compute_clean_labels_for_pair(
        df, "TEST",
        date_start="2019-01-01", date_end="2026-01-01",
        atr_period=14,
    )
    pd.testing.assert_frame_equal(out1, out2)
