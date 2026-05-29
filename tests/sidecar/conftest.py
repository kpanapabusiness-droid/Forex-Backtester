"""Shared fixtures for deployment/sidecar tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ── Synthetic OHLC fixtures ───────────────────────────────────────────


def _synth_h4_panel(n_bars: int = 400, start_iso: str = "2026-01-01T00:00:00") -> pd.DataFrame:
    """Generate a deterministic UTC-anchored H4 panel.

    The seeded RNG produces a stable price walk; values are not meant to
    reproduce any historical market — only to exercise the signal
    module's plumbing.
    """
    rng = np.random.default_rng(seed=42)
    start = datetime.fromisoformat(start_iso).replace(tzinfo=None)
    dates = [start + timedelta(hours=4 * i) for i in range(n_bars)]
    base = 1.10 + np.cumsum(rng.normal(0, 0.0008, size=n_bars))
    spread = rng.uniform(0.0008, 0.0020, size=n_bars)
    close = base
    opens = close - rng.normal(0, 0.0004, size=n_bars)
    high = np.maximum(opens, close) + spread / 2.0
    low = np.minimum(opens, close) - spread / 2.0
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def _synth_d1_panel(start_iso: str = "2026-01-01T00:00:00", n_bars: int = 80) -> pd.DataFrame:
    """Generate a deterministic UTC-anchored D1 panel."""
    rng = np.random.default_rng(seed=43)
    start = datetime.fromisoformat(start_iso).replace(tzinfo=None) - timedelta(days=10)
    dates = [start + timedelta(days=i) for i in range(n_bars)]
    base = 1.10 + np.cumsum(rng.normal(0, 0.0030, size=n_bars))
    high = base + rng.uniform(0.0010, 0.0050, size=n_bars)
    low = base - rng.uniform(0.0010, 0.0050, size=n_bars)
    opens = base - rng.normal(0, 0.0010, size=n_bars)
    closes = base + rng.normal(0, 0.0010, size=n_bars)
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": high,
            "low": low,
            "close": closes,
        }
    )


@pytest.fixture
def synth_h4() -> pd.DataFrame:
    return _synth_h4_panel()


@pytest.fixture
def synth_d1() -> pd.DataFrame:
    return _synth_d1_panel()


# ── Fake MT5 module ──────────────────────────────────────────────────


class FakeMt5:
    """Test-double matching the ``Mt5Module`` Protocol."""

    TIMEFRAME_H4 = 16388  # MT5 actual constant value; pin so tests catch drift
    TIMEFRAME_D1 = 16408

    def __init__(
        self,
        *,
        h4_panel: pd.DataFrame | None = None,
        d1_panel: pd.DataFrame | None = None,
        init_returns: bool = True,
        last_error_value: tuple[int, str] = (0, "ok"),
    ) -> None:
        self.h4_panel = h4_panel if h4_panel is not None else _synth_h4_panel()
        self.d1_panel = d1_panel if d1_panel is not None else _synth_d1_panel()
        self._init_returns = init_returns
        self._last_error = last_error_value
        self.initialize_calls = 0
        self.shutdown_calls = 0
        self.copy_calls: list[tuple[str, int, int, int]] = []
        # Capture (args, kwargs) of every initialize() call so tests can
        # assert how --mt5-path / login / etc. are threaded into initialize.
        self.init_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def initialize(self, *args: Any, **kwargs: Any) -> bool:
        self.initialize_calls += 1
        self.init_calls.append((args, kwargs))
        return self._init_returns

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def last_error(self) -> tuple[int, str]:
        return self._last_error

    def copy_rates_from_pos(
        self, symbol: str, timeframe: int, start_pos: int, count: int
    ) -> Any:
        self.copy_calls.append((symbol, timeframe, start_pos, count))
        if timeframe == self.TIMEFRAME_H4:
            panel = self.h4_panel.tail(count).reset_index(drop=True)
        elif timeframe == self.TIMEFRAME_D1:
            panel = self.d1_panel.tail(count).reset_index(drop=True)
        else:
            return None
        # MT5 returns a numpy structured array with epoch-seconds 'time'.
        # We synthesise that shape from our panel.
        records = []
        for row in panel.itertuples(index=False):
            t = int(row.date.timestamp() if hasattr(row.date, "timestamp") else
                    pd.Timestamp(row.date).tz_localize("UTC").timestamp())
            records.append(
                (t, float(row.open), float(row.high), float(row.low), float(row.close), 0, 0, 0)
            )
        dtype = np.dtype(
            [
                ("time", "i8"),
                ("open", "f8"),
                ("high", "f8"),
                ("low", "f8"),
                ("close", "f8"),
                ("tick_volume", "i8"),
                ("spread", "i4"),
                ("real_volume", "i8"),
            ]
        )
        return np.array(records, dtype=dtype)


@pytest.fixture
def fake_mt5() -> FakeMt5:
    return FakeMt5()


# ── Sidecar root with directory scaffold ─────────────────────────────


@pytest.fixture
def sidecar_root(tmp_path: Path) -> Path:
    """Create a sidecar root with all expected subdirs."""
    root = tmp_path / "sidecar"
    (root / "signals_out").mkdir(parents=True)
    (root / "signals_processed").mkdir(parents=True)
    (root / "signals_failed").mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    return root


# ── Winning config fixture ───────────────────────────────────────────


@pytest.fixture
def winning_config_path(tmp_path: Path) -> Path:
    """Write a minimal valid winning_config.yaml mirroring the UTC rerun's keys."""
    content = """
arc_name: l_arc_10_v3.0.2
verdict: PASS-DEPLOYABLE
boundary_convention: utc

signal:
  name: dlr_d1_swing_low_rejection_long
  module: signals.lchar_dlr_long
  version: v0.1

direction: long

architecture:
  name: A1
  variant: system_level_filter

stop_loss:
  type: atr_multiple
  atr_period: 14
  multiplier: 3.5
  anchor: mid
  reference: entry_price

exit_policy:
  name: sl_partial_close_1r_runner_trail
  partial_close_at: 1.0
  partial_close_fraction: 0.5
  runner_trail_atr_below_peak: 1.0
  update_frequency: bar_close

time_exit:
  max_bars: 240

timeframes:
  primary: H4
  anchor: D1

pairs:
  - EURUSD
  - GBPUSD

risk:
  r_safe_pct: 0.004336

fills:
  entry_long: open_ask
  spread_source: histdata_m1_bid_ask
"""
    p = tmp_path / "winning_config.yaml"
    p.write_text(content.strip() + "\n", encoding="utf-8")
    return p


__all__ = ("FakeMt5",)
