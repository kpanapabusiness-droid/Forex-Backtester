"""Tests for the PR-E.1.6 trail mechanics rewrite.

Verifies the EA pattern: trail activates on bid_close, exit fires when
bar_close ≤ trail_level, fill at NEXT bar's open_bid (not intra-bar
wick on the next bar).
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.sim.account import Account, Direction, ExposureRules
from core.sim.multipair_backtester import MultiPairBacktester, Order
from core.sim.panel import Panel
from core.sim.trailing_stop import TrailManager


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def _panel_from_rows(pair: str, rows: list[tuple]) -> Panel:
    """Build a 1-pair Panel from (ts, ob, hb, lb, cb, oa, ha, la, ca) rows."""
    idx = pd.DatetimeIndex([r[0] for r in rows], tz="UTC")
    df = pd.DataFrame(
        {
            "open_bid": [r[1] for r in rows],
            "high_bid": [r[2] for r in rows],
            "low_bid": [r[3] for r in rows],
            "close_bid": [r[4] for r in rows],
            "open_ask": [r[5] for r in rows],
            "high_ask": [r[6] for r in rows],
            "low_ask": [r[7] for r in rows],
            "close_ask": [r[8] for r in rows],
            "volume": [1] * len(rows),
            "spread_close": [r[8] - r[4] for r in rows],
            "bid_ask_data_quality": ["ok"] * len(rows),
        },
        index=idx,
    )
    df.index.name = "timestamp_utc"
    return Panel.from_frames({pair: df}, tf="H4")


def test_trail_reads_bid_close_not_mid() -> None:
    """Trail update should read close_bid, not (close_bid + close_ask)/2.

    Construct a bar where close_bid = 1.10 and close_ask = 1.12. If trail
    activation reads mid (1.11), trail activates; if it reads bid (1.10),
    trail doesn't (threshold = 1.11). Verifies bid-side reading.
    """
    pair = "EURUSD"
    # Activation threshold at 1.11 (entry 1.10 + 2.0 × ATR 0.005 = 1.11)
    bar_with_wide_spread = (
        "2026-01-01 04:00:00",
        1.10,
        1.12,
        1.09,
        1.10,  # bid: o, h, l, c=1.10 — JUST below activation 1.11
        1.10,
        1.12,
        1.09,
        1.12,  # ask: c=1.12 — mid=1.11 would activate
    )
    panel = _panel_from_rows(pair, [bar_with_wide_spread])

    acct = Account(starting_balance=100_000.0, exposure=ExposureRules(max_concurrent_per_pair=1))
    tm = TrailManager()
    pos = acct.open(
        pair=pair,
        direction=Direction.LONG,
        entry_time=_ts("2026-01-01 00:00"),
        entry_price=1.10,
        size=10_000,
        sl_price=1.09,
    )
    tm.register(pos, atr_at_entry=0.005)

    bar = panel.pair_dfs[pair].iloc[0]
    tm.update_all_at_close({pair: bar}, acct)
    state = tm.get(pos.position_id)
    # Bid close 1.10 < threshold 1.11 → trail must NOT have activated
    assert state.activated is False, (
        "Trail activated on bid_close=1.10 (threshold=1.11) — should have read bid, not mid"
    )


def test_trail_exit_at_close_filled_next_bar_open() -> None:
    """Trail fires when bar_close ≤ trail_level; fills at next bar's open_bid.

    Sequence:
      Bar 1: activate trail (close above entry + 2×ATR), set max_close.
      Bar 2: bid_close drops below trail_level → queue pending close.
      Bar 3: pending close fills at this bar's open_bid (NOT intra-bar low).
    """
    pair = "EURUSD"
    # entry = 1.10, ATR = 0.005 → activation 1.11, trail 1.10 (max_close 1.115 − 1.5×0.005 = 1.1075)
    # Bar 1: bid close 1.115 (activates trail; sets trail level = 1.115 - 0.0075 = 1.1075)
    # Bar 2: bid close 1.105 — BELOW trail level 1.1075 → queue close
    # Bar 3: open_bid 1.108 — this is where the fill should land (NOT bar 3's low_bid)
    rows = [
        ("2026-01-01 00:00:00", 1.110, 1.118, 1.109, 1.115, 1.110, 1.119, 1.110, 1.116),
        ("2026-01-01 04:00:00", 1.115, 1.116, 1.103, 1.105, 1.116, 1.117, 1.104, 1.106),
        ("2026-01-01 08:00:00", 1.108, 1.112, 1.100, 1.110, 1.109, 1.113, 1.101, 1.111),
    ]
    panel = _panel_from_rows(pair, rows)

    def strategy_open_on_first_bar(t, snapshot, account):
        if t != _ts("2026-01-01 00:00:00") or account.closed_trades or account.open_positions:
            return []
        return [
            Order(
                pair=pair,
                direction=Direction.LONG,
                size=10_000,
                sl_price=1.090,
                atr_at_entry=0.005,
                trail_activation_atr=2.0,
                trail_distance_atr=1.5,
            )
        ]

    # Pre-open a position at 1.10 manually so the trail can engage on bar 1
    acct = Account(starting_balance=100_000.0, exposure=ExposureRules(max_concurrent_per_pair=1))
    tm = TrailManager()
    pos = acct.open(
        pair=pair,
        direction=Direction.LONG,
        entry_time=_ts("2025-12-31 20:00"),
        entry_price=1.10,
        size=10_000,
        sl_price=1.090,
    )
    tm.register(pos, atr_at_entry=0.005)

    bt = MultiPairBacktester(
        panel=panel,
        account=acct,
        strategy=lambda t, s, a: [],  # no new orders
        trail_manager=tm,
    )
    bt.run()

    # The trade should have closed on bar 3 at open_bid = 1.108
    assert len(acct.closed_trades) == 1
    trade = acct.closed_trades[0]
    assert trade.exit_reason == "trailing_stop"
    assert trade.exit_price == pytest.approx(1.108), (
        f"Expected trail exit at bar 3 open_bid (1.108), got {trade.exit_price}"
    )
    # Confirm the exit timestamp is bar 3
    assert trade.exit_time == _ts("2026-01-01 08:00:00")


def test_no_intra_bar_wick_exit_when_trail_active() -> None:
    """Trail-active positions don't exit on intra-bar low_bid touching trail.

    The original (pre-PR-E.1.6) v3 design fired SL on next-bar wick. EA
    pattern: only bar-close triggers, no intra-bar trail SL.
    """
    pair = "EURUSD"
    # Trail level will end up at 1.1075. Bar 2 has low_bid = 1.105 (well below
    # trail) but close_bid = 1.115 (well above trail). Under EA pattern, NO exit.
    rows = [
        ("2026-01-01 00:00:00", 1.110, 1.118, 1.109, 1.115, 1.111, 1.119, 1.110, 1.116),
        # Bar 2: deep wick (low 1.105) but close back to 1.115 — should NOT exit
        ("2026-01-01 04:00:00", 1.115, 1.118, 1.105, 1.115, 1.116, 1.119, 1.106, 1.116),
    ]
    panel = _panel_from_rows(pair, rows)

    acct = Account(starting_balance=100_000.0, exposure=ExposureRules(max_concurrent_per_pair=1))
    tm = TrailManager()
    pos = acct.open(
        pair=pair,
        direction=Direction.LONG,
        entry_time=_ts("2025-12-31 20:00"),
        entry_price=1.10,
        size=10_000,
        sl_price=1.090,
    )
    tm.register(pos, atr_at_entry=0.005)

    bt = MultiPairBacktester(
        panel=panel,
        account=acct,
        strategy=lambda t, s, a: [],
        trail_manager=tm,
    )
    bt.run()

    # No trail exit — bar 2's low touched 1.105 but close was 1.115 above trail
    assert len(acct.closed_trades) == 0
    assert len(acct.open_positions) == 1
