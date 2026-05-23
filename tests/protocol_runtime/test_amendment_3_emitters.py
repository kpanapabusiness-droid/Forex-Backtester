"""Unit tests for Amendment 3 engine emission helpers:

  - core.wfo.chained_dd.compute_chained_max_dd_from_continuous_equity
  - core.runners._fold_stats_helpers.compute_per_day_max_dd
  - core.wfo.holdout_rerun.rescale_arch_config_risk

Together with the gate-logic tests in test_amended_gates.py these
cover all six Amendment 3 emission sub-items from the dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from core.architectures.a1_system_level_filter import A1Config
from core.runners._fold_stats_helpers import compute_per_day_max_dd
from core.wfo.chained_dd import compute_chained_max_dd_from_continuous_equity
from core.wfo.holdout_rerun import rescale_arch_config_risk


# ── compute_chained_max_dd_from_continuous_equity ───────────────────


def test_chained_max_dd_simple_drawdown() -> None:
    """Equity rises to 110, falls to 90, recovers to 100. Peak-to-trough
    DD = (110 - 90) / 110 ≈ 18.18%."""
    idx = pd.date_range("2020-01-01", periods=5, freq="1D", tz="UTC")
    equity = pd.Series([100.0, 110.0, 95.0, 90.0, 100.0], index=idx)
    dd = compute_chained_max_dd_from_continuous_equity(equity)
    assert dd == pytest.approx(20.0 / 110.0, abs=1e-9)


def test_chained_max_dd_no_drawdown() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="1D", tz="UTC")
    equity = pd.Series([100.0, 105.0, 110.0, 115.0], index=idx)
    dd = compute_chained_max_dd_from_continuous_equity(equity)
    assert dd == 0.0


def test_chained_max_dd_empty_returns_zero() -> None:
    assert compute_chained_max_dd_from_continuous_equity(pd.Series(dtype=float)) == 0.0
    assert compute_chained_max_dd_from_continuous_equity(None) == 0.0


def test_chained_max_dd_handles_nan() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="1D", tz="UTC")
    equity = pd.Series([100.0, np.nan, 90.0, 95.0], index=idx)
    dd = compute_chained_max_dd_from_continuous_equity(equity)
    assert dd == pytest.approx(10.0 / 100.0, abs=1e-9)


# ── compute_per_day_max_dd ──────────────────────────────────────────


def test_per_day_max_dd_basic_three_days() -> None:
    """Three calendar days; each day intra-day high → low DD computed."""
    idx = pd.DatetimeIndex(
        [
            "2020-01-01 00:00", "2020-01-01 12:00", "2020-01-01 23:00",
            "2020-01-02 00:00", "2020-01-02 12:00",
            "2020-01-03 00:00", "2020-01-03 06:00",
        ],
        tz="UTC",
    )
    # Day 1: starts 100, dips to 90 (10% DD)
    # Day 2: starts 95, no DD
    # Day 3: starts 98, dips to 96 (2.04% DD)
    equity = pd.Series([100.0, 90.0, 95.0,  95.0, 100.0,  98.0, 96.0], index=idx)
    df = compute_per_day_max_dd(equity)
    assert len(df) == 3
    # Day 1
    day1 = df.iloc[0]
    assert day1["day_start_equity"] == 100.0
    assert day1["day_max_dd_base_pct"] == pytest.approx(0.10, abs=1e-9)
    # Day 2: starts 95, min 95 → 0
    day2 = df.iloc[1]
    assert day2["day_max_dd_base_pct"] == 0.0
    # Day 3: starts 98, min 96 → 2/98
    day3 = df.iloc[2]
    assert day3["day_max_dd_base_pct"] == pytest.approx(2.0 / 98.0, abs=1e-9)


def test_per_day_max_dd_empty_input() -> None:
    df = compute_per_day_max_dd(pd.Series(dtype=float))
    assert df.empty
    assert set(df.columns) == {
        "date", "pair_set", "day_start_equity",
        "day_max_dd_base_pct", "n_trades_open_start_of_day",
    }


def test_per_day_max_dd_pair_set_label() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="1D", tz="UTC")
    equity = pd.Series([100.0, 95.0, 98.0], index=idx)
    df = compute_per_day_max_dd(equity, pair_set="all_28")
    assert (df["pair_set"] == "all_28").all()


# ── rescale_arch_config_risk ────────────────────────────────────────


def test_rescale_arch_config_simple_a1() -> None:
    cfg = A1Config(config_id="a1_base", risk_pct=0.005)
    scaled = rescale_arch_config_risk(cfg, k_scale=2.0)
    assert scaled.risk_pct == pytest.approx(0.010, abs=1e-9)
    # Should NOT mutate original (frozen dataclass)
    assert cfg.risk_pct == 0.005


def test_rescale_arch_config_id_includes_scaled_risk() -> None:
    cfg = A1Config(config_id="a1_base", risk_pct=0.005)
    scaled = rescale_arch_config_risk(cfg, k_scale=1.5)
    # New config_id reflects the scaled risk (1.5 * 0.005 = 0.0075)
    assert "0.0075" in scaled.config_id


def test_rescale_arch_config_rejects_nonfinite_k() -> None:
    cfg = A1Config(config_id="x", risk_pct=0.005)
    with pytest.raises(ValueError):
        rescale_arch_config_risk(cfg, k_scale=float("inf"))
    with pytest.raises(ValueError):
        rescale_arch_config_risk(cfg, k_scale=0)
    with pytest.raises(ValueError):
        rescale_arch_config_risk(cfg, k_scale=-1.0)


def test_rescale_arch_config_rejects_non_dataclass() -> None:
    with pytest.raises(TypeError):
        rescale_arch_config_risk("not a dataclass", k_scale=1.0)


def test_rescale_arch_config_rejects_missing_risk_pct() -> None:
    @dataclass(frozen=True)
    class _Minimal:
        config_id: str

    with pytest.raises(TypeError):
        rescale_arch_config_risk(_Minimal(config_id="x"), k_scale=1.0)
