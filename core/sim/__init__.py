"""Simulation primitives for the v3.0 backtester.

Public surface (PR-B):
    Direction         enum {LONG, SHORT}
    Order, Position   dataclasses
    Account           single-account state across all pairs
    Panel             multi-pair OHLC panel
    fill_*            bar-level fill primitives (entry, market exit, SL/TP)
    MultiPairBacktester  driver loop

Strategy plug-in: a strategy callable receives ``(t, snapshot, account)``
and returns a list of ``Order``s. The driver applies exposure rules and
fills entries at next-bar open per L_PROTOCOL §1.
"""

from core.sim.account import Account, Direction, ExposureRules, Position
from core.sim.fill import (
    long_entry_fill_price,
    long_exit_market_price,
    long_sl_triggered,
    long_tp_triggered,
    short_entry_fill_price,
    short_exit_market_price,
    short_sl_triggered,
    short_tp_triggered,
)
from core.sim.multipair_backtester import MultiPairBacktester, Order, RunResult
from core.sim.panel import Panel

__all__ = [
    "Account",
    "Direction",
    "ExposureRules",
    "Position",
    "Panel",
    "MultiPairBacktester",
    "Order",
    "RunResult",
    "long_entry_fill_price",
    "long_exit_market_price",
    "long_sl_triggered",
    "long_tp_triggered",
    "short_entry_fill_price",
    "short_exit_market_price",
    "short_sl_triggered",
    "short_tp_triggered",
]
