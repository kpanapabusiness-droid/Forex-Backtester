"""Tests for the Arc 7 v3.0.1 SignalModule.

Coverage:
  - Protocol conformance (`isinstance(module, SignalModule)`)
  - Trigger fires on a hand-built fixture where all six conditions hold
  - Trigger does NOT fire when C4 (bullish reclaim) is violated
  - No-lookahead spot-check: perturbing forward bars does not change the
    signal mask at bar t (Arc 9 lesson — producer-level)
  - Locked v0.1 params
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.arc.signal_protocol import SignalModule
from core.sim.panel import Panel
from core.strategies.liquidity_sweep_reclaim_long import (
    LiquiditySweepReclaimLongSignal,
    LSRSignalParams,
)


def _build_pair_df(
    n: int = 200, seed: int = 0
) -> pd.DataFrame:
    """Build a synthetic H4 pair_df with deterministic mild trend + spread."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    base = 1.1000 + np.cumsum(rng.normal(0, 0.0008, size=n))
    high = base + np.abs(rng.normal(0, 0.0008, size=n))
    low = base - np.abs(rng.normal(0, 0.0008, size=n))
    opn = base + rng.normal(0, 0.0004, size=n)
    close = base + rng.normal(0, 0.0004, size=n)
    spread = 0.00005
    return pd.DataFrame({
        "open_bid": opn, "high_bid": high, "low_bid": low, "close_bid": close,
        "open_ask": opn + spread, "high_ask": high + spread,
        "low_ask": low + spread, "close_ask": close + spread,
        "volume": rng.integers(100, 1000, size=n).astype(float),
        "spread_close": np.full(n, spread, dtype=float),
        "bid_ask_data_quality": ["ok"] * n,
    }, index=idx)


def test_protocol_conformance() -> None:
    assert isinstance(LiquiditySweepReclaimLongSignal(), SignalModule)


def test_signal_fires_when_all_six_conditions_hold() -> None:
    """Construct a bar at index 30 where all six trigger conditions hold."""
    n = 50
    idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    high = np.full(n, 1.1050)
    low = np.full(n, 1.1010)
    opn = np.full(n, 1.1020)
    close = np.full(n, 1.1040)
    sweep_bar = 30
    # depth ~ 3× ATR threshold so the magnitude check has clear margin
    low[sweep_bar] = 1.0980     # swing_low_20 = 1.1010 → depth = 0.0030 ≥ 0.25 × ATR(~0.0041)
    close[sweep_bar] = 1.1030    # reclaim strength = (0.0020)/(0.0030) = 0.667 ≥ 0.5
    opn[sweep_bar] = 1.0985      # close > open
    high[sweep_bar] = 1.1035
    spread = 0.00005
    df = pd.DataFrame({
        "open_bid": opn, "high_bid": high, "low_bid": low, "close_bid": close,
        "open_ask": opn + spread, "high_ask": high + spread,
        "low_ask": low + spread, "close_ask": close + spread,
        "volume": np.full(n, 100.0),
        "spread_close": np.full(n, spread),
        "bid_ask_data_quality": ["ok"] * n,
    }, index=idx)
    panel = Panel.from_frames({"FOOBAR": df}, tf="H4")
    sig = LiquiditySweepReclaimLongSignal()
    ev = sig.evaluate({"H4": panel})
    mask = ev.per_pair["FOOBAR"].signal_mask
    assert bool(mask.iloc[sweep_bar]), (
        f"Expected signal at index {sweep_bar}; "
        f"got mask={list(mask[mask].index)} (any={int(mask.sum())})"
    )


def test_signal_does_not_fire_when_close_below_open() -> None:
    """Violation of C4 (bullish reclaim) must suppress the signal."""
    n = 50
    idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    low = np.full(n, 1.1010)
    high = np.full(n, 1.1050)
    opn = np.full(n, 1.1020)
    close = np.full(n, 1.1040)
    sweep_bar = 30
    low[sweep_bar] = 1.0980
    close[sweep_bar] = 1.1015  # reclaim but below open
    opn[sweep_bar] = 1.1018    # close < open → C4 fails
    spread = 0.00005
    df = pd.DataFrame({
        "open_bid": opn, "high_bid": high, "low_bid": low, "close_bid": close,
        "open_ask": opn + spread, "high_ask": high + spread,
        "low_ask": low + spread, "close_ask": close + spread,
        "volume": np.full(n, 100.0),
        "spread_close": np.full(n, spread),
        "bid_ask_data_quality": ["ok"] * n,
    }, index=idx)
    panel = Panel.from_frames({"FOOBAR": df}, tf="H4")
    sig = LiquiditySweepReclaimLongSignal()
    ev = sig.evaluate({"H4": panel})
    mask = ev.per_pair["FOOBAR"].signal_mask
    assert not bool(mask.iloc[sweep_bar])


def test_no_lookahead_perturbation() -> None:
    """Perturbing bars AFTER signal must not change the signal mask at the
    signal bar — producer-level Arc 9 lesson check."""
    df = _build_pair_df(n=300, seed=1)
    panel = Panel.from_frames({"FOOBAR": df}, tf="H4")
    sig = LiquiditySweepReclaimLongSignal()
    ev_orig = sig.evaluate({"H4": panel})
    mask_orig = ev_orig.per_pair["FOOBAR"].signal_mask.copy()

    if mask_orig.sum() == 0:
        pytest.skip("synthetic data produced no signals; no perturbation point to test")

    first_sig = int(np.flatnonzero(mask_orig.values)[0])
    df_pert = df.copy()
    rng = np.random.default_rng(42)
    for col in ("open_bid", "high_bid", "low_bid", "close_bid",
                "open_ask", "high_ask", "low_ask", "close_ask"):
        arr = df_pert[col].values.copy()
        arr[first_sig + 1 :] += rng.normal(0, 0.001, size=len(arr) - first_sig - 1)
        df_pert[col] = arr
    panel_pert = Panel.from_frames({"FOOBAR": df_pert}, tf="H4")
    ev_pert = sig.evaluate({"H4": panel_pert})
    mask_pert = ev_pert.per_pair["FOOBAR"].signal_mask
    assert bool(mask_orig.iloc[first_sig]) == bool(mask_pert.iloc[first_sig])
    assert (
        mask_orig.iloc[: first_sig + 1].equals(mask_pert.iloc[: first_sig + 1])
    ), "Perturbing forward bars changed signal mask at/before first_sig — lookahead!"


def test_params_locked_at_v01() -> None:
    p = LSRSignalParams()
    assert p.swing_window == 20
    assert p.atr_period == 14
    assert p.magnitude_atr_mult == 0.25
    assert p.reclaim_strength_min == 0.5
    assert p.refractory_bars == 20
