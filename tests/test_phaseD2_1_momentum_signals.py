"""
Phase D-2.1 Momentum Signals — tests using synthetic data only.

Validates: 3-bar signal correctness, two-row-per-date contract,
deterministic sort, warmup behavior. No real CSVs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_signal_correctness_rising_fires_long() -> None:
    """3-bar rising sequence at known index must fire long=1, short=0."""
    from scripts.phaseD2_generate_momentum_signals import _momentum_3bar_signals

    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=6, freq="D"),
        "close": [1.0, 1.1, 1.2, 1.15, 1.25, 1.3],
    })
    out = _momentum_3bar_signals(df, pair="EUR_USD", signal_name="momentum_3bar")

    rows_by_date = out.set_index(["date", "direction"])["signal"].unstack()
    assert rows_by_date.loc[pd.Timestamp("2020-01-03"), "long"] == 1
    assert rows_by_date.loc[pd.Timestamp("2020-01-03"), "short"] == 0

    assert rows_by_date.loc[pd.Timestamp("2020-01-06"), "long"] == 1
    assert rows_by_date.loc[pd.Timestamp("2020-01-06"), "short"] == 0


def test_signal_correctness_falling_fires_short() -> None:
    """3-bar falling sequence must fire short=1, long=0."""
    from scripts.phaseD2_generate_momentum_signals import _momentum_3bar_signals

    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5, freq="D"),
        "close": [1.2, 1.1, 1.0, 0.95, 0.9],
    })
    out = _momentum_3bar_signals(df, pair="GBP_USD", signal_name="momentum_3bar")

    rows_by_date = out.set_index(["date", "direction"])["signal"].unstack()
    assert rows_by_date.loc[pd.Timestamp("2020-01-03"), "short"] == 1
    assert rows_by_date.loc[pd.Timestamp("2020-01-03"), "long"] == 0

    assert rows_by_date.loc[pd.Timestamp("2020-01-05"), "short"] == 1
    assert rows_by_date.loc[pd.Timestamp("2020-01-05"), "long"] == 0


def test_signal_no_fire_when_not_monotonic() -> None:
    """Non-monotonic 3-bar sequence must produce signal=0 for both."""
    from scripts.phaseD2_generate_momentum_signals import _momentum_3bar_signals

    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5, freq="D"),
        "close": [1.0, 1.1, 1.0, 1.1, 1.0],
    })
    out = _momentum_3bar_signals(df, pair="AUD_USD", signal_name="momentum_3bar")

    rows_by_date = out.set_index(["date", "direction"])["signal"].unstack()
    for d in [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-04"), pd.Timestamp("2020-01-05")]:
        assert rows_by_date.loc[d, "long"] == 0
        assert rows_by_date.loc[d, "short"] == 0


def test_two_rows_per_date_contract() -> None:
    """For N dates, output must have 2N rows with both directions per date."""
    from scripts.phaseD2_generate_momentum_signals import _momentum_3bar_signals

    n = 10
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "close": np.linspace(1.0, 1.5, n),
    })
    out = _momentum_3bar_signals(df, pair="USD_JPY", signal_name="momentum_3bar")

    assert len(out) == 2 * n
    for d in df["date"].unique():
        subset = out[out["date"] == d]
        assert len(subset) == 2
        assert set(subset["direction"]) == {"long", "short"}


def test_deterministic_sort_order() -> None:
    """Output sorted by (pair, date, direction) with long before short."""
    from scripts.phaseD2_generate_momentum_signals import _momentum_3bar_signals

    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5, freq="D"),
        "close": [1.0, 1.1, 1.2, 1.15, 1.1],
    })
    out = _momentum_3bar_signals(df, pair="EUR_GBP", signal_name="momentum_3bar")

    assert list(out.columns) == ["pair", "date", "direction", "signal", "signal_name"]
    dirs = out["direction"].tolist()
    for i in range(0, len(dirs), 2):
        assert dirs[i] == "long"
        assert dirs[i + 1] == "short"
    assert out["pair"].nunique() == 1
    reordered = out.sort_values(["pair", "date", "direction"])
    pd.testing.assert_frame_equal(out, reordered)


def test_warmup_first_two_bars_signal_zero() -> None:
    """First two dates cannot fire (need t-2); assert signal=0 for both directions."""
    from scripts.phaseD2_generate_momentum_signals import _momentum_3bar_signals

    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5, freq="D"),
        "close": [1.0, 1.1, 1.2, 1.3, 1.4],
    })
    out = _momentum_3bar_signals(df, pair="CHF_JPY", signal_name="momentum_3bar")

    rows_by_date = out.set_index(["date", "direction"])["signal"].unstack()
    d0 = pd.Timestamp("2020-01-01")
    d1 = pd.Timestamp("2020-01-02")
    assert rows_by_date.loc[d0, "long"] == 0
    assert rows_by_date.loc[d0, "short"] == 0
    assert rows_by_date.loc[d1, "long"] == 0
    assert rows_by_date.loc[d1, "short"] == 0

    d2 = pd.Timestamp("2020-01-03")
    assert rows_by_date.loc[d2, "long"] == 1
    assert rows_by_date.loc[d2, "short"] == 0


def test_config_validation_date_window() -> None:
    """Config must enforce date window 2019-01-01 → 2026-01-01."""
    from scripts.phaseD2_generate_momentum_signals import _require_phaseD2_1_config

    base = {
        "pairs": ["EUR_USD"],
        "timeframe": "D1",
        "date_range": {"start": "2019-01-01", "end": "2026-01-01"},
        "data": {"dir": "data/daily"},
        "outputs": {"dir": "results/phaseD2/signals"},
        "signal_name": "momentum_3bar",
    }
    _require_phaseD2_1_config(base)

    bad = {**base, "date_range": {"start": "2020-01-01", "end": "2025-12-31"}}
    with pytest.raises(ValueError, match="locked to 2019-01-01"):
        _require_phaseD2_1_config(bad)


def test_config_validation_timeframe() -> None:
    """Config must enforce timeframe in D1 or D only."""
    from scripts.phaseD2_generate_momentum_signals import _require_phaseD2_1_config

    base = {
        "pairs": ["EUR_USD"],
        "timeframe": "D1",
        "date_range": {"start": "2019-01-01", "end": "2026-01-01"},
        "data": {"dir": "data/daily"},
        "outputs": {"dir": "results/phaseD2/signals"},
        "signal_name": "momentum_3bar",
    }
    bad = {**base, "timeframe": "H1"}
    with pytest.raises(ValueError, match=r"D1.*D|timeframe"):
        _require_phaseD2_1_config(bad)


def test_config_validation_outputs_phaseD2() -> None:
    """outputs.dir must live under results/phaseD2."""
    from scripts.phaseD2_generate_momentum_signals import _require_phaseD2_1_config

    base = {
        "pairs": ["EUR_USD"],
        "timeframe": "D1",
        "date_range": {"start": "2019-01-01", "end": "2026-01-01"},
        "data": {"dir": "data/daily"},
        "outputs": {"dir": "results/phaseD2/signals"},
        "signal_name": "momentum_3bar",
    }
    bad = {**base, "outputs": {"dir": "results/other/signals"}}
    with pytest.raises(ValueError, match="phaseD2"):
        _require_phaseD2_1_config(bad)
