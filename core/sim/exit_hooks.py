"""Exit-hook architecture for the v3 multipair backtester.

The driver's built-in exits are intra-bar SL/TP only. Arcs that need
signal-driven exits (e.g. KH-24's ``kijun_d1`` exit when D1 close
crosses below D1 Kijun) register an ``ExitPredicate`` that the driver
consults at bar close, after SL/TP intra-bar checks but before
mark-to-market.

A predicate is a callable:

    ExitPredicate(position, snapshot, t) -> ExitDecision | None

where ``ExitDecision`` carries the fill price + exit reason, or
``None`` means "no exit on this bar". The first triggering predicate
wins.

Generic enough that future arcs can register their own — KH-24 is the
first user. Predicates that need extra state (e.g. cached panel TF
snapshots) close over it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from core.sim.account import Position


@dataclass(frozen=True)
class ExitDecision:
    """A predicate's "yes, exit this position on this bar" verdict."""

    fill_price: float
    exit_reason: str  # e.g. "kijun_d1", "time_exit", "regime_flip"


ExitPredicate = Callable[
    [Position, dict[str, pd.Series | None], pd.Timestamp],
    ExitDecision | None,
]


def evaluate_predicates(
    predicates: list[ExitPredicate],
    position: Position,
    snapshot: dict[str, pd.Series | None],
    t: pd.Timestamp,
) -> ExitDecision | None:
    """Run predicates in order; return first triggering ExitDecision or None.

    Order is determined by the caller — predicates registered first take
    priority. KH-24 registers ``kijun_d1`` after the trail-stop hook so
    a trail-stop hit on the same bar wins.
    """
    for pred in predicates:
        decision = pred(position, snapshot, t)
        if decision is not None:
            return decision
    return None
