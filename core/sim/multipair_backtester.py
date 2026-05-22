"""Multi-pair simultaneous backtester driver.

Bar-by-bar loop over a ``Panel``'s union timestamp index. At each bar:

  1. Check SL/TP exits on every open position (intra-bar, against the
     current bar's bid/ask high/low per ``core.sim.fill``).
  2. ``mark_to_market`` the account using close-mid prices.
  3. Hand the bar to the strategy callable for new orders.
  4. Apply exposure rules; entries that pass fill at the *next* bar's
     ``open_ask`` (long) or ``open_bid`` (short) per L_PROTOCOL §1.

Cost accounting is bid/ask spread only (paid implicitly via entry/exit
fill prices); commission/swap haircuts are applied at deployment-gate
time per L_PROTOCOL Appendix B and are out of scope for the driver.

Determinism contract:
  - Pairs iterated in ``sorted(panel.pairs)`` order at every decision
    point.
  - Open positions checked in ``sorted(position_id)`` order.
  - Strategy callable is expected to return orders in a stable order
    (the driver does not reorder).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from core.sim.account import Account, ClosedTrade, Direction, Position
from core.sim.fill import (
    long_entry_fill_price,
    long_sl_triggered,
    long_tp_triggered,
    short_entry_fill_price,
    short_sl_triggered,
    short_tp_triggered,
)
from core.sim.panel import Panel
from core.spread.real_spread import is_tradable_bar


@dataclass(frozen=True)
class Order:
    """A candidate trade emitted by the strategy at bar t.

    Fills happen at bar t+1's open (long: open_ask, short: open_bid).
    SL/TP, if provided, are absolute prices in the pair's quote currency.
    """

    pair: str
    direction: Direction
    size: float
    sl_price: float | None = None
    tp_price: float | None = None


# Strategy signature: callable(t, snapshot, account) -> list[Order]
StrategyFn = Callable[[pd.Timestamp, dict[str, pd.Series | None], Account], list[Order]]


@dataclass(frozen=True)
class RunResult:
    final_balance: float
    n_trades: int
    n_open_at_end: int
    equity_curve: pd.Series
    max_drawdown_pct: float
    closed_trades: tuple[ClosedTrade, ...]


@dataclass
class MultiPairBacktester:
    """Bar-by-bar driver over a multi-pair Panel.

    Usage::

        bt = MultiPairBacktester(panel=p, account=acct, strategy=my_strategy,
                                 sl_first=True)
        result = bt.run()

    ``sl_first=True`` (default) applies the SL → TP intra-bar priority
    (conservative). Set False for TP-first.
    """

    panel: Panel
    account: Account
    strategy: StrategyFn
    sl_first: bool = True

    # internal: deferred entries from prior bar awaiting fill at next-bar open
    _pending: list[Order] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._pending = []

    # ── exit checks ─────────────────────────────────────────────────
    def _check_exits(self, t: pd.Timestamp, snapshot: dict[str, pd.Series | None]) -> None:
        """Close any positions whose SL/TP fired this bar."""
        for pos_id in sorted(self.account._open.keys()):  # noqa: SLF001
            pos = self.account._open.get(pos_id)  # noqa: SLF001
            if pos is None:
                continue
            bar = snapshot.get(pos.pair)
            if bar is None or not bool(is_tradable_bar(bar.to_frame().T).iloc[0]):
                continue
            sl_hit, sl_px = False, float("nan")
            tp_hit, tp_px = False, float("nan")
            if pos.direction is Direction.LONG:
                if pos.sl_price is not None:
                    sl_hit, sl_px = long_sl_triggered(bar, pos.sl_price)
                if pos.tp_price is not None:
                    tp_hit, tp_px = long_tp_triggered(bar, pos.tp_price)
            else:
                if pos.sl_price is not None:
                    sl_hit, sl_px = short_sl_triggered(bar, pos.sl_price)
                if pos.tp_price is not None:
                    tp_hit, tp_px = short_tp_triggered(bar, pos.tp_price)

            # Intra-bar priority
            if self.sl_first:
                if sl_hit:
                    self.account.close(pos_id, t, sl_px, "stop_loss")
                    continue
                if tp_hit:
                    self.account.close(pos_id, t, tp_px, "take_profit")
            else:
                if tp_hit:
                    self.account.close(pos_id, t, tp_px, "take_profit")
                    continue
                if sl_hit:
                    self.account.close(pos_id, t, sl_px, "stop_loss")

    # ── entry fills (deferred from prior bar) ───────────────────────
    def _fill_pending_entries(
        self, t: pd.Timestamp, snapshot: dict[str, pd.Series | None]
    ) -> list[Position]:
        """Fill any orders queued on the prior bar against this bar's open.

        Exposure check is re-applied at fill time (account state may have
        changed since the order was emitted).
        """
        filled: list[Position] = []
        for order in self._pending:
            bar = snapshot.get(order.pair)
            if bar is None or not bool(is_tradable_bar(bar.to_frame().T).iloc[0]):
                continue  # untradable bar: drop the order silently
            if not self.account.exposure_check(order.pair):
                continue
            if order.direction is Direction.LONG:
                fill_px = long_entry_fill_price(bar)
            else:
                fill_px = short_entry_fill_price(bar)
            pos = self.account.open(
                pair=order.pair,
                direction=order.direction,
                entry_time=t,
                entry_price=fill_px,
                size=order.size,
                sl_price=order.sl_price,
                tp_price=order.tp_price,
            )
            filled.append(pos)
        self._pending = []
        return filled

    # ── mark-to-market helper ───────────────────────────────────────
    @staticmethod
    def _close_mid(snapshot: dict[str, pd.Series | None]) -> dict[str, float]:
        marks: dict[str, float] = {}
        for pair, bar in snapshot.items():
            if bar is None:
                continue
            cb = bar["close_bid"]
            ca = bar["close_ask"]
            if pd.isna(cb) or pd.isna(ca):
                continue
            marks[pair] = float((cb + ca) / 2.0)
        return marks

    # ── per-bar processing ──────────────────────────────────────────
    def _process_bar(self, t: pd.Timestamp, snapshot: dict[str, pd.Series | None]) -> None:
        # 1. fill any entries pending from prior bar
        self._fill_pending_entries(t, snapshot)
        # 2. check exits on currently open positions
        self._check_exits(t, snapshot)
        # 3. mark to market
        self.account.mark_to_market(t, self._close_mid(snapshot))
        # 4. ask the strategy for new orders
        orders = self.strategy(t, snapshot, self.account)
        if not orders:
            return
        # 5. queue for next bar (exposure re-checked at fill time)
        # Pre-check now to drop obvious caps already breached.
        for order in orders:
            if order.pair not in self.panel.pair_dfs:
                raise KeyError(f"Strategy order for unknown pair {order.pair!r}")
            if not self.account.exposure_check(order.pair):
                continue
            self._pending.append(order)

    # ── driver ──────────────────────────────────────────────────────
    def run(self) -> RunResult:
        for t, snapshot in self.panel.iter_bars():
            self._process_bar(t, snapshot)
        return RunResult(
            final_balance=self.account.balance,
            n_trades=len(self.account.closed_trades),
            n_open_at_end=len(self.account.open_positions),
            equity_curve=self.account.equity_curve(),
            max_drawdown_pct=self.account.max_drawdown_pct,
            closed_trades=self.account.closed_trades,
        )
