"""Tests for arc_discovery_02 Amendment A — 240-bar time exit.

The time exit fires AT THE OPEN of the bar at offset == time_exit_bars from
entry. It runs BEFORE the SL / trail checks of that bar, so a trade reaching
the cap exits at the cap-anniversary bar's open_bid regardless of what
happens intra-bar afterwards. ``exit_reason="time_exit"``. Final R computed
on the open_bid fill via the same `(exit - entry) / sl_distance` formula
as every other exit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.discovery.pool_simulator import DiscoveryExitConfig, simulate_pair_pool


def _ohlc(rows: list[dict]) -> pd.DataFrame:
    """Build a v3-schema OHLC DataFrame from row dicts."""
    idx = pd.DatetimeIndex(
        [pd.Timestamp(r["ts"], tz="UTC") for r in rows], name="timestamp_utc"
    )
    return pd.DataFrame(
        {
            "open_bid": [r.get("open_bid", r["open_ask"]) for r in rows],
            "high_bid": [r["high_bid"] for r in rows],
            "low_bid": [r["low_bid"] for r in rows],
            "close_bid": [r["close_bid"] for r in rows],
            "open_ask": [r["open_ask"] for r in rows],
            "high_ask": [r.get("high_ask", r["high_bid"]) for r in rows],
            "low_ask": [r.get("low_ask", r["low_bid"]) for r in rows],
            "close_ask": [r.get("close_ask", r["close_bid"]) for r in rows],
            "volume": [r.get("volume", 1.0) for r in rows],
            "spread_close": [
                r.get("close_ask", r["close_bid"]) - r["close_bid"] for r in rows
            ],
            "bid_ask_data_quality": ["ok"] * len(rows),
        },
        index=idx,
    )


def _signal_at(idx, bar):
    arr = np.zeros(len(idx), dtype=bool)
    arr[bar] = True
    return pd.Series(arr, index=idx)


def _atr(idx, value=1.0):
    return pd.Series([value] * len(idx), index=idx, dtype="float64")


def _cfg_with_time_exit(n_bars: int = 240) -> DiscoveryExitConfig:
    return DiscoveryExitConfig(
        initial_sl_atr_mult=2.0,
        trail_activation_atr_mult=4.0,
        trail_distance_atr_mult=2.0,
        primary_tf_warmup_bars=0,
        time_exit_bars=n_bars,
    )


def test_time_exit_fires_at_open_of_cap_anniversary_bar():
    """Trade drifts sideways well above SL but below activation → time exit
    fires at off == time_exit_bars at THAT bar's open_bid."""
    # Build a 12-bar series: signal at bar 0, entry at bar 1, time exit at bar 5
    # (time_exit_bars = 4 → exit at off=4, which is bar 5 in absolute index)
    bars = []
    bars.append({"ts": "2020-01-01T00", "open_ask": 99.0, "high_bid": 100.0,
                 "low_bid": 98.5, "close_bid": 99.0})
    # Entry @ bar 1 open_ask = 100.0; sl_price = 98.0; activation = 104.0
    bars.append({"ts": "2020-01-01T01", "open_ask": 100.0, "high_bid": 100.5,
                 "low_bid": 99.5, "close_bid": 100.2})
    # Sideways drift bars 2-4 (no SL, no activation)
    for i in range(2, 5):
        bars.append({"ts": f"2020-01-01T0{i}", "open_ask": 100.5,
                     "high_bid": 100.6, "low_bid": 99.5, "close_bid": 100.0})
    # Bar 5 (off=4): time exit fires at THIS bar's open_bid = 101.5.
    bars.append({"ts": "2020-01-01T05", "open_ask": 101.5, "high_bid": 101.7,
                 "low_bid": 100.0, "close_bid": 100.5, "open_bid": 101.5})
    # Bars 6-7 to prove we don't run past the cap.
    bars.append({"ts": "2020-01-01T06", "open_ask": 100.5, "high_bid": 100.7,
                 "low_bid": 99.5, "close_bid": 100.0})
    bars.append({"ts": "2020-01-01T07", "open_ask": 100.5, "high_bid": 100.7,
                 "low_bid": 99.5, "close_bid": 100.0})

    df = _ohlc(bars)
    trigger = _signal_at(df.index, 0)
    atr = _atr(df.index, 1.0)
    trades, _, _, _ = simulate_pair_pool(
        "EURUSD", df, trigger, atr, _cfg_with_time_exit(n_bars=4)
    )
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "time_exit"
    # entry 100.0, exit at bar 5 open_bid = 101.5 → final_r = (101.5-100)/2 = +0.75R
    assert t.exit_price == 101.5
    assert abs(t.final_r - 0.75) < 1e-9
    assert t.bars_held == 4   # off=4
    assert t.activated_trail is False


def test_time_exit_does_not_fire_when_sl_hits_first():
    """If hard SL fires before the time-cap anniversary, exit_reason=hard_sl
    (not time_exit) — SL has higher priority than time exit."""
    bars = []
    bars.append({"ts": "2020-01-01T00", "open_ask": 99.0, "high_bid": 100.0,
                 "low_bid": 98.5, "close_bid": 99.0})
    bars.append({"ts": "2020-01-01T01", "open_ask": 100.0, "high_bid": 100.5,
                 "low_bid": 99.5, "close_bid": 100.2})
    # Bar 2 (off=1): low_bid 97.5 → SL @ 98.0 fires
    bars.append({"ts": "2020-01-01T02", "open_ask": 100.1, "high_bid": 100.3,
                 "low_bid": 97.5, "close_bid": 98.6})
    # Bars 3-5 just to fill the time-exit horizon (time_exit_bars=4)
    for i in range(3, 7):
        bars.append({"ts": f"2020-01-01T0{i}", "open_ask": 100.0,
                     "high_bid": 100.5, "low_bid": 99.5, "close_bid": 100.0})
    df = _ohlc(bars)
    trigger = _signal_at(df.index, 0)
    atr = _atr(df.index, 1.0)
    trades, _, _, _ = simulate_pair_pool(
        "EURUSD", df, trigger, atr, _cfg_with_time_exit(n_bars=4)
    )
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "hard_sl"
    assert t.exit_price == 98.0
    assert t.bars_held == 1


def test_time_exit_does_not_fire_when_trail_hits_first():
    """Trail exit (pending from bar B-1's close) fires at bar B's open_bid
    before the time exit check (priority 1 > priority 1.5)."""
    bars = []
    bars.append({"ts": "2020-01-01T00", "open_ask": 99.0, "high_bid": 100.0,
                 "low_bid": 98.5, "close_bid": 99.0})
    bars.append({"ts": "2020-01-01T01", "open_ask": 100.0, "high_bid": 101.0,
                 "low_bid": 99.5, "close_bid": 100.5})
    # Bar 2 (off=1): close 104.5 (>= 104.0) → arm trail; trail_price = 102.5
    bars.append({"ts": "2020-01-01T02", "open_ask": 100.5, "high_bid": 104.6,
                 "low_bid": 100.0, "close_bid": 104.5})
    # Bar 3 (off=2): close 102.0 < trail 102.5 → trail-hit-queued
    bars.append({"ts": "2020-01-01T03", "open_ask": 104.5, "high_bid": 104.6,
                 "low_bid": 102.0, "close_bid": 102.0, "open_bid": 104.0})
    # Bar 4 (off=3): exits at open_bid 103.5 → trail exit
    bars.append({"ts": "2020-01-01T04", "open_ask": 103.5, "high_bid": 103.7,
                 "low_bid": 103.0, "close_bid": 103.5, "open_bid": 103.5})
    # Bar 5 (off=4): if time exit had priority it would fire here, but the
    # trail exit already fired at off=3.
    bars.append({"ts": "2020-01-01T05", "open_ask": 105.0, "high_bid": 105.2,
                 "low_bid": 104.0, "close_bid": 104.5})
    df = _ohlc(bars)
    trigger = _signal_at(df.index, 0)
    atr = _atr(df.index, 1.0)
    trades, _, _, _ = simulate_pair_pool(
        "EURUSD", df, trigger, atr, _cfg_with_time_exit(n_bars=4)
    )
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "trail"
    assert t.bars_held == 3


def test_time_exit_disabled_when_None_preserves_arc_01_behaviour():
    """time_exit_bars=None (default) → no time exit, end_of_data exit fires."""
    bars = []
    bars.append({"ts": "2020-01-01T00", "open_ask": 99.0, "high_bid": 100.0,
                 "low_bid": 98.5, "close_bid": 99.0})
    bars.append({"ts": "2020-01-01T01", "open_ask": 100.0, "high_bid": 100.5,
                 "low_bid": 99.5, "close_bid": 100.0})
    # 20 sideways bars; no SL or trail
    for i in range(20):
        ts = pd.Timestamp("2020-01-01T02", tz="UTC") + pd.Timedelta(hours=i)
        bars.append({"ts": ts.isoformat(), "open_ask": 100.5,
                     "high_bid": 100.6, "low_bid": 99.5, "close_bid": 100.0})
    df = _ohlc(bars)
    trigger = _signal_at(df.index, 0)
    atr = _atr(df.index, 1.0)
    cfg = DiscoveryExitConfig(
        initial_sl_atr_mult=2.0,
        trail_activation_atr_mult=4.0,
        trail_distance_atr_mult=2.0,
        primary_tf_warmup_bars=0,
        # time_exit_bars=None (default)
    )
    trades, _, _, _ = simulate_pair_pool("EURUSD", df, trigger, atr, cfg)
    assert len(trades) == 1
    assert trades[0].exit_reason == "end_of_data"
    assert trades[0].bars_held == 20  # ran to last bar (idx 21 - entry 1)
