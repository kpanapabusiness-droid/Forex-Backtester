#!/usr/bin/env python
"""End-to-end smoke test for the SOLE gate engine, ``MultiPairBacktester``.

Proves the bar-walking pipeline runs clean on the clean-base reset: build a
tiny synthetic bid/ask panel, fire one long trade through the runner-trail
exit policy, and confirm a trade closes and the take-the-loss invariant holds
(a stop on/before the +1R partial = -1R, full size, no partial).

Runs on numpy/pandas only (CI-safe). Exit 0 = pass, non-zero = fail.

    python scripts/smoke_test_multipair.py
"""

from __future__ import annotations

import sys

import pandas as pd

from core.sim.account import Account, Direction, ExposureRules
from core.sim.exit_policy_manager import ExitPolicyManager
from core.sim.multipair_backtester import MultiPairBacktester, Order
from core.sim.panel import Panel


def _bar(t: str, *, o: float, h: float, lo: float, c: float) -> dict:
    return {
        "timestamp_utc": t,
        "open_bid": o, "open_ask": o,
        "high_bid": h, "high_ask": h,
        "low_bid": lo, "low_ask": lo,
        "close_bid": c, "close_ask": c,
        "spread_close": 0.0,
        "bid_ask_data_quality": "ok",
    }


def _panel(rows: list[dict]) -> Panel:
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp_utc")
    return Panel.from_frames({"EURUSD": df}, tf="H1")


def _one_long(t_target: str, **kw):
    target = pd.Timestamp(t_target, tz="UTC")
    fired = {"done": False}

    def strategy(t, snapshot, account):
        if fired["done"] or t != target:
            return []
        fired["done"] = True
        return [Order(pair="EURUSD", direction=Direction.LONG, size=10_000.0, **kw)]

    return strategy


def _run(rows):
    bt = MultiPairBacktester(
        panel=_panel(rows),
        account=Account(starting_balance=100_000.0,
                        exposure=ExposureRules(max_concurrent_per_pair=1)),
        strategy=_one_long("2026-01-01 00:00", sl_price=1.098,
                           atr_at_entry=0.001, sl_atr_mult=2.0,
                           exit_policy="sl_partial_close_1r_runner_trail"),
        exit_policy_manager=ExitPolicyManager(),
    )
    return bt.run()


def main() -> int:
    # 1) A clean winner: +1R partial then runner trails out.
    win = _run([
        _bar("2026-01-01 00:00", o=1.099, h=1.100, lo=1.098, c=1.099),
        _bar("2026-01-01 01:00", o=1.100, h=1.101, lo=1.099, c=1.100),
        _bar("2026-01-01 02:00", o=1.100, h=1.102, lo=1.099, c=1.101),
        _bar("2026-01-01 03:00", o=1.101, h=1.103, lo=1.100, c=1.1015),
        _bar("2026-01-01 04:00", o=1.1015, h=1.1020, lo=1.1010, c=1.101),
        _bar("2026-01-01 05:00", o=1.101, h=1.102, lo=1.100, c=1.101),
    ])
    assert win.n_trades == 2, f"expected partial+runner, got {win.n_trades}"
    legs = list(win.closed_trades)
    assert legs[0].exit_reason == "partial_close_1r"
    assert legs[1].exit_reason == "runner_trail_stop"

    # 2) Take-the-loss: a same-bar stop+partial bar resolves to a single -1R stop.
    loss = _run([
        _bar("2026-01-01 00:00", o=1.099, h=1.100, lo=1.098, c=1.099),
        _bar("2026-01-01 01:00", o=1.100, h=1.101, lo=1.099, c=1.100),
        _bar("2026-01-01 02:00", o=1.100, h=1.102, lo=1.097, c=1.099),  # +1R high AND stop low
        _bar("2026-01-01 03:00", o=1.099, h=1.100, lo=1.099, c=1.099),
    ])
    assert loss.n_trades == 1, f"take-the-loss: expected one stop leg, got {loss.n_trades}"
    leg = loss.closed_trades[0]
    assert leg.exit_reason == "stop_loss"
    assert leg.parent_position_id is None, "partial must not fire on a same-bar stop"
    realised_r = (leg.exit_price - leg.entry_price) / 0.002
    assert abs(realised_r + 1.0) < 1e-9, f"same-bar stop must be -1R, got {realised_r:.6f}"

    print("SMOKE OK: MultiPairBacktester ran end-to-end; "
          "winner=partial+runner, same-bar stop=take-the-loss(-1R).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
