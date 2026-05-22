"""Tests for core.discovery.pool_simulator — locked exit semantics.

Synthetic OHLC bars are crafted to exercise each exit path independently:
  - Hard SL hit (intra-bar low_bid <= sl_price)
  - Trail-not-armed -> trade rides to next event
  - Trail arm at +4xATR close
  - Trail ratchet on higher close
  - Trail hit at close -> NEXT bar open_bid fill
  - End of data -> exit at last close_bid
  - No time exit (trade runs forever absent SL/trail trigger)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from core.discovery.pool_simulator import (
    DiscoveryExitConfig,
    simulate_pair_pool,
)


def _ohlc(rows: list[dict]) -> pd.DataFrame:
    """Build a v3-schema OHLC DataFrame from row dicts.

    Each row must specify ``ts``, ``open_ask``, ``open_bid``, ``high_bid``,
    ``low_bid``, ``close_bid``. Optional keys default to the same as their
    bid counterpart (ask side mirrors bid for these tests).
    """
    idx = pd.DatetimeIndex(
        [pd.Timestamp(r["ts"], tz="UTC") for r in rows], name="timestamp_utc"
    )
    out = pd.DataFrame(
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
    return out


def _cfg(warmup: int = 0) -> DiscoveryExitConfig:
    return DiscoveryExitConfig(
        initial_sl_atr_mult=2.0,
        trail_activation_atr_mult=4.0,
        trail_distance_atr_mult=2.0,
        primary_tf_warmup_bars=warmup,
    )


def _signal_at(idx: pd.DatetimeIndex, bar: int) -> pd.Series:
    arr = np.zeros(len(idx), dtype=bool)
    arr[bar] = True
    return pd.Series(arr, index=idx)


def _atr(idx: pd.DatetimeIndex, value: float = 1.0) -> pd.Series:
    return pd.Series([value] * len(idx), index=idx, dtype="float64")


def test_hard_sl_fires_intra_bar():
    """Bar 2's low_bid drops below sl_price (98.0). Trade exits at sl_price."""
    df = _ohlc([
        # ts, open_ask, high_bid, low_bid, close_bid
        {"ts": "2020-01-01T00", "open_ask": 99.0,  "high_bid": 100.0, "low_bid": 98.5, "close_bid": 99.0},
        {"ts": "2020-01-01T01", "open_ask": 100.0, "high_bid": 100.5, "low_bid": 99.5, "close_bid": 100.2},
        {"ts": "2020-01-01T02", "open_ask": 100.1, "high_bid": 100.3, "low_bid": 97.5, "close_bid": 98.6},
    ])
    trigger = _signal_at(df.index, 0)
    atr = _atr(df.index, 1.0)
    trades, _ = simulate_pair_pool("EURUSD", df, trigger, atr, _cfg())
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "hard_sl"
    # entry_price = 100.0 (bar 1 open_ask); sl_price = 98.0
    assert t.entry_price == 100.0
    assert t.sl_at_entry_price == 98.0
    assert t.exit_price == 98.0
    assert t.final_r == -1.0  # exit at exactly 1R loss
    assert t.activated_trail is False


def test_trail_arms_then_ratchets_then_hits():
    """Walk-up sequence: arm at +4xATR close, ratchet on higher close, exit on dip below trail."""
    df = _ohlc([
        # Bar 0 = signal
        {"ts": "2020-01-01T00", "open_ask": 99.0,  "high_bid": 100.0, "low_bid": 98.5, "close_bid": 99.0},
        # Bar 1 = entry @ 100.0
        {"ts": "2020-01-01T01", "open_ask": 100.0, "high_bid": 101.0, "low_bid": 99.5, "close_bid": 100.5},
        # Bar 2: close 103.5 (still < activation 104.0); no arm
        {"ts": "2020-01-01T02", "open_ask": 100.5, "high_bid": 103.6, "low_bid": 100.0, "close_bid": 103.5},
        # Bar 3: close 104.5 (>= activation 104.0) -> arm; trail = 104.5 - 2.0 = 102.5
        {"ts": "2020-01-01T03", "open_ask": 103.6, "high_bid": 104.6, "low_bid": 103.0, "close_bid": 104.5},
        # Bar 4: close 106.0 -> ratchet; new trail = 106.0 - 2.0 = 104.0
        {"ts": "2020-01-01T04", "open_ask": 104.5, "high_bid": 106.5, "low_bid": 104.0, "close_bid": 106.0},
        # Bar 5: close 103.0 (below trail=104.0) -> trail hit at close. Exit at NEXT bar open_bid.
        {"ts": "2020-01-01T05", "open_ask": 106.1, "high_bid": 106.2, "low_bid": 103.0, "close_bid": 103.0,
         "open_bid": 106.0},
        # Bar 6: exit fill at open_bid = 105.5
        {"ts": "2020-01-01T06", "open_ask": 105.6, "high_bid": 105.7, "low_bid": 105.4, "close_bid": 105.5,
         "open_bid": 105.5},
    ])
    trigger = _signal_at(df.index, 0)
    atr = _atr(df.index, 1.0)
    trades, _ = simulate_pair_pool("EURUSD", df, trigger, atr, _cfg())
    assert len(trades) == 1
    t = trades[0]
    assert t.activated_trail is True
    assert t.exit_reason == "trail"
    # entry 100.0, exit at bar 6 open_bid = 105.5 -> final_r = (105.5 - 100.0) / 2.0 = +2.75R
    assert t.exit_price == 105.5
    assert abs(t.final_r - 2.75) < 1e-9


def test_trail_not_armed_before_activation_threshold():
    """Sequence stays below entry+4xATR -> trail never arms; only SL or end-of-data can exit."""
    df = _ohlc([
        {"ts": "2020-01-01T00", "open_ask": 99.0,  "high_bid": 100.0, "low_bid": 98.5, "close_bid": 99.0},
        {"ts": "2020-01-01T01", "open_ask": 100.0, "high_bid": 101.0, "low_bid": 99.5, "close_bid": 100.5},
        # Walk up to just below activation (103.9 < 104.0)
        {"ts": "2020-01-01T02", "open_ask": 100.5, "high_bid": 103.9, "low_bid": 100.0, "close_bid": 103.5},
        # Then drift sideways
        {"ts": "2020-01-01T03", "open_ask": 103.5, "high_bid": 103.9, "low_bid": 102.5, "close_bid": 103.0},
        {"ts": "2020-01-01T04", "open_ask": 103.0, "high_bid": 103.5, "low_bid": 102.0, "close_bid": 102.5},
    ])
    trigger = _signal_at(df.index, 0)
    atr = _atr(df.index, 1.0)
    trades, _ = simulate_pair_pool("EURUSD", df, trigger, atr, _cfg())
    assert len(trades) == 1
    t = trades[0]
    assert t.activated_trail is False
    # No SL hit (lowest low_bid was 99.5 > sl_price 98.0). End-of-data exit.
    assert t.exit_reason == "end_of_data"
    # final_r = (102.5 - 100.0) / 2.0 = +1.25R
    assert abs(t.final_r - 1.25) < 1e-9


def test_no_time_exit_runs_to_end_of_data():
    """Long sideways drift with no SL hit -> exit only at end of data (no time exit)."""
    bars = []
    bars.append({"ts": "2020-01-01T00", "open_ask": 99.0, "high_bid": 100.0, "low_bid": 98.5, "close_bid": 99.0})
    bars.append({"ts": "2020-01-01T01", "open_ask": 100.0, "high_bid": 100.5, "low_bid": 99.5, "close_bid": 100.0})
    # 1000 sideways bars, each stays close to 100.5 with low above SL.
    for i in range(1000):
        ts = pd.Timestamp("2020-01-01T02", tz="UTC") + pd.Timedelta(hours=i)
        bars.append({
            "ts": ts.isoformat(),
            "open_ask": 100.5, "high_bid": 100.6, "low_bid": 99.5, "close_bid": 100.0,
        })
    df = _ohlc(bars)
    trigger = _signal_at(df.index, 0)
    atr = _atr(df.index, 1.0)
    trades, _ = simulate_pair_pool("EURUSD", df, trigger, atr, _cfg())
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "end_of_data"
    # 1002 bars total: signal@0, entry@1, then 1000 forward bars (idx 2..1001).
    # exit_idx = 1001 (last bar), entry_idx = 1 -> bars_held = 1000.
    assert t.bars_held == 1000


def test_deterministic_two_runs():
    """Same inputs -> identical trade list across two runs."""
    df = _ohlc([
        {"ts": "2020-01-01T00", "open_ask": 99.0,  "high_bid": 100.0, "low_bid": 98.5, "close_bid": 99.0},
        {"ts": "2020-01-01T01", "open_ask": 100.0, "high_bid": 101.0, "low_bid": 99.5, "close_bid": 100.5},
        {"ts": "2020-01-01T02", "open_ask": 100.5, "high_bid": 104.6, "low_bid": 100.0, "close_bid": 104.5},
        {"ts": "2020-01-01T03", "open_ask": 104.5, "high_bid": 106.5, "low_bid": 104.0, "close_bid": 106.0},
        {"ts": "2020-01-01T04", "open_ask": 106.0, "high_bid": 106.2, "low_bid": 103.0, "close_bid": 103.0,
         "open_bid": 106.0},
        {"ts": "2020-01-01T05", "open_ask": 106.1, "high_bid": 106.2, "low_bid": 105.4, "close_bid": 105.5,
         "open_bid": 105.5},
    ])
    trigger = _signal_at(df.index, 0)
    atr = _atr(df.index, 1.0)
    a, _ = simulate_pair_pool("EURUSD", df, trigger, atr, _cfg())
    b, _ = simulate_pair_pool("EURUSD", df, trigger, atr, _cfg())
    assert len(a) == len(b)
    for ta, tb in zip(a, b):
        assert ta.to_dict() == tb.to_dict()


def test_warmup_excludes_early_signals():
    """primary_tf_warmup_bars=5 excludes signal indices 0-4."""
    df = _ohlc([
        {"ts": f"2020-01-01T0{i}", "open_ask": 100.0, "high_bid": 100.5, "low_bid": 99.5, "close_bid": 100.0}
        for i in range(8)
    ])
    # Trigger at bar 2 (excluded by warmup=5)
    trigger = _signal_at(df.index, 2)
    atr = _atr(df.index, 1.0)
    trades, _ = simulate_pair_pool("EURUSD", df, trigger, atr, _cfg(warmup=5))
    assert trades == []
