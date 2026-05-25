"""Single-account state across all pairs (v3.0 multi-pair simulation).

Per CC_06 Task 6 and L_PROTOCOL §1, the v3 backtester maintains a single
account state across all 28 pairs concurrently:

  - one balance / equity curve
  - one drawdown calculation
  - exposure rules enforced at the account level
    (``max_concurrent_per_currency``, ``max_concurrent_total``)

This module owns:

  - ``Direction`` enum (LONG | SHORT)
  - ``Position`` dataclass (open trade record — frozen, size = size-at-open)
  - ``ExposureRules`` config block
  - ``Account`` runtime state with open(), close(), partial_close(),
    mark_to_market(), exposure_check(), and equity/dd accessors

PnL convention: position size is in *units of base currency* (one
"unit"-sized long EURUSD = profit_in_quote per pip on a 1-pip move of
0.0001). Sizing in lots is a higher-layer concern; this module deals in
absolute units and the price-delta PnL that follows.

Cost accounting: entry and exit fill prices passed in have *already*
accounted for bid/ask spread (the caller used ``core.sim.fill``).
Commission and swap are not modelled here — they're an aggregate haircut
at the deployment-gate evaluation per L_PROTOCOL Appendix B's cost-model
section.

Partial-fill semantics
----------------------
``Position`` is immutable (``size`` is the size at open). When a policy
like ``sl_partial_close_1r_runner_trail`` reduces the live position size
mid-trade, Account maintains a shadow ``_current_sizes: dict[int, float]``
keyed by ``position_id``. Read via ``current_size_of(position_id)``;
mutate only via ``partial_close(...)``. Mark-to-market, full close PnL,
and the trade-log ``ClosedTrade.size`` all use the current (possibly
reduced) size, not the size at open.

Exposure caps still count a partial-closed position as 1 (not a
fractional value) until it is fully closed — the position remains in
``_open`` and continues to occupy its exposure-cap slot.

Multi-leg closes are recorded as distinct ``ClosedTrade`` entries with
``parent_position_id`` set to the shared ``position_id``; consumers may
group by ``position_id`` to reconstruct the full close-out sequence. A
trade with ``parent_position_id is None`` is a standalone (single-leg)
full close.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import pandas as pd


class Direction(Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.LONG else -1


@dataclass(frozen=True)
class Position:
    """Open position record. Immutable once opened.

    ``size`` is in units of base currency (positive for both long and
    short — direction carries the sign).
    """

    position_id: int
    pair: str
    direction: Direction
    entry_time: pd.Timestamp
    entry_price: float
    size: float  # positive units of base
    sl_price: float | None = None
    tp_price: float | None = None

    @property
    def base_currency(self) -> str:
        return self.pair[:3]

    @property
    def quote_currency(self) -> str:
        return self.pair[3:6]

    def pnl_at(self, mark_price: float) -> float:
        """Mark-to-market PnL at ``mark_price`` in quote-currency units."""
        return self.direction.sign * (mark_price - self.entry_price) * self.size


@dataclass(frozen=True)
class ClosedTrade:
    """Closed trade record — what hits the equity curve / trade ledger.

    ``size`` is the size actually closed on this leg (not the position's
    size at open, which may have been reduced by prior partial closes).

    ``parent_position_id``:
      * ``None`` for any standalone full close — a position opened, then
        closed once with the full size. The historical default.
      * ``= position_id`` for any leg of a multi-leg close-out (every
        partial leg AND the final leg both carry this). Consumers can
        ``filter(parent_position_id is None)`` to get whole-position
        closes only, or ``groupby(position_id)`` to reconstruct
        multi-leg sequences.
    """

    position_id: int
    pair: str
    direction: Direction
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    size: float
    pnl: float
    exit_reason: str  # "market" | "stop_loss" | "take_profit" | "time_exit" | etc.
    parent_position_id: int | None = None


@dataclass(frozen=True)
class ExposureRules:
    """Account-level exposure caps applied before opening any new position.

    A None value means uncapped. ``max_concurrent_per_currency`` counts a
    position once per currency (both base and quote).
    """

    max_concurrent_total: int | None = None
    max_concurrent_per_currency: int | None = None
    max_concurrent_per_pair: int | None = 1  # 1 = no doubling on the same pair


@dataclass
class Account:
    """Single-account state across all pairs.

    The backtester driver calls (in order at each bar t):
      1. ``mark_to_market(snapshot)``  -> updates equity / drawdown
      2. close-position decisions made by the driver via ``close(...)``
      3. ``exposure_check(...)`` for any candidate new positions
      4. ``open(...)`` for accepted candidates

    State is plain Python — no pandas DataFrames retained in-flight to
    keep the per-bar loop fast. Equity and drawdown series are appended
    one timestamp at a time.
    """

    starting_balance: float
    exposure: ExposureRules = field(default_factory=ExposureRules)

    _balance: float = field(init=False)
    _next_position_id: int = field(init=False, default=1)
    _open: dict[int, Position] = field(init=False, default_factory=dict)
    _closed: list[ClosedTrade] = field(init=False, default_factory=list)
    _equity_history: list[tuple[pd.Timestamp, float]] = field(init=False, default_factory=list)
    _peak_equity: float = field(init=False, default=0.0)
    _max_drawdown_pct: float = field(init=False, default=0.0)
    # Shadow of live position sizes after partial closes. Absent entries
    # mean the position is at its original ``Position.size`` (i.e. never
    # partial-closed). Removed on full close.
    _current_sizes: dict[int, float] = field(init=False, default_factory=dict)
    # Position ids that have ever had a partial close. Used so the
    # final-leg ``close()`` knows to record ``parent_position_id`` even
    # if ``_current_sizes`` has already been popped on full-out.
    _partial_history: set[int] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        self._balance = float(self.starting_balance)
        self._peak_equity = float(self.starting_balance)

    # ── basic accessors ─────────────────────────────────────────────
    @property
    def balance(self) -> float:
        return self._balance

    @property
    def open_positions(self) -> tuple[Position, ...]:
        return tuple(self._open.values())

    @property
    def closed_trades(self) -> tuple[ClosedTrade, ...]:
        return tuple(self._closed)

    @property
    def max_drawdown_pct(self) -> float:
        return self._max_drawdown_pct

    def equity_curve(self) -> pd.Series:
        """Time-indexed equity curve (one point per ``mark_to_market`` call)."""
        if not self._equity_history:
            return pd.Series([], dtype="float64", name="equity")
        idx, vals = zip(*self._equity_history)
        return pd.Series(vals, index=pd.DatetimeIndex(idx, name="timestamp_utc"), name="equity")

    # ── core operations ─────────────────────────────────────────────
    def open(
        self,
        pair: str,
        direction: Direction,
        entry_time: pd.Timestamp,
        entry_price: float,
        size: float,
        sl_price: float | None = None,
        tp_price: float | None = None,
    ) -> Position:
        """Open a position. Exposure check is the caller's job (see
        ``exposure_check``); ``open`` itself does not gate on caps.
        """
        pos = Position(
            position_id=self._next_position_id,
            pair=pair,
            direction=direction,
            entry_time=entry_time,
            entry_price=float(entry_price),
            size=float(size),
            sl_price=None if sl_price is None else float(sl_price),
            tp_price=None if tp_price is None else float(tp_price),
        )
        self._open[pos.position_id] = pos
        self._next_position_id += 1
        return pos

    def close(
        self,
        position_id: int,
        exit_time: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
    ) -> ClosedTrade:
        """Close an open position; realise PnL on the *current* size.

        For positions that have been partial-closed earlier, the close
        uses the shadow ``_current_sizes`` value (not ``Position.size``).
        For never-partialled positions this is identical to the original
        full-size close.

        Multi-leg closes (i.e. positions with prior partial closes) have
        ``parent_position_id`` populated on the recorded ClosedTrade.
        """
        if position_id not in self._open:
            raise KeyError(f"No open position with id={position_id}")
        pos = self._open.pop(position_id)
        current_size = self._current_sizes.pop(position_id, pos.size)
        had_partials = position_id in self._partial_history
        self._partial_history.discard(position_id)
        pnl = pos.direction.sign * (float(exit_price) - pos.entry_price) * current_size
        self._balance += pnl
        trade = ClosedTrade(
            position_id=pos.position_id,
            pair=pos.pair,
            direction=pos.direction,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            exit_time=exit_time,
            exit_price=float(exit_price),
            size=current_size,
            pnl=pnl,
            exit_reason=exit_reason,
            parent_position_id=pos.position_id if had_partials else None,
        )
        self._closed.append(trade)
        return trade

    def partial_close(
        self,
        position_id: int,
        exit_time: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
        size_to_close: float,
    ) -> ClosedTrade:
        """Realise PnL on ``size_to_close`` units; leave the rest open.

        ``size_to_close`` must be strictly positive AND strictly less
        than the current size — otherwise the caller should use
        :meth:`close` (full close). Failing loud here keeps partial-close
        manager bugs surfaceable.

        Records a ClosedTrade with ``parent_position_id = position_id``
        marking this as one leg of a multi-leg close. The matching final
        :meth:`close` for the same position will also have
        ``parent_position_id`` set (see :meth:`close`).
        """
        if position_id not in self._open:
            raise KeyError(f"No open position with id={position_id}")
        pos = self._open[position_id]
        current_size = self._current_sizes.get(position_id, pos.size)
        size_to_close = float(size_to_close)
        if size_to_close <= 0.0:
            raise ValueError(
                f"partial_close size_to_close must be > 0; got {size_to_close}"
            )
        if size_to_close >= current_size:
            raise ValueError(
                f"partial_close size_to_close={size_to_close} >= current_size="
                f"{current_size} for position_id={position_id}; use close() for "
                "full-out"
            )
        pnl = pos.direction.sign * (float(exit_price) - pos.entry_price) * size_to_close
        self._balance += pnl
        new_size = current_size - size_to_close
        self._current_sizes[position_id] = new_size
        self._partial_history.add(position_id)
        trade = ClosedTrade(
            position_id=pos.position_id,
            pair=pos.pair,
            direction=pos.direction,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            exit_time=exit_time,
            exit_price=float(exit_price),
            size=size_to_close,
            pnl=pnl,
            exit_reason=exit_reason,
            parent_position_id=pos.position_id,
        )
        self._closed.append(trade)
        return trade

    def current_size_of(self, position_id: int) -> float:
        """Live size of an open position (post any partial closes).

        Raises KeyError for unknown / already-fully-closed positions.
        """
        if position_id not in self._open:
            raise KeyError(f"No open position with id={position_id}")
        pos = self._open[position_id]
        return self._current_sizes.get(position_id, pos.size)

    def mark_to_market(self, t: pd.Timestamp, marks: dict[str, float]) -> float:
        """Update equity using current bid/ask close-mid for open positions.

        ``marks[pair]`` is the mark price for that pair at time t — by
        convention the close-mid ``(close_bid + close_ask) / 2``. Pairs
        with no mark (e.g. weekend bars) carry the position at its prior
        entry-time price for that step (caller's responsibility to
        provide marks or omit them).

        Returns the new equity. Also updates peak / max-drawdown
        trackers and appends the equity point.

        Unrealised PnL uses the *current* (possibly partial-reduced)
        size for each position.
        """
        unrealised = 0.0
        for pos in self._open.values():
            mark = marks.get(pos.pair)
            if mark is None:
                continue
            size = self._current_sizes.get(pos.position_id, pos.size)
            unrealised += pos.direction.sign * (mark - pos.entry_price) * size
        equity = self._balance + unrealised
        self._equity_history.append((t, equity))
        if equity > self._peak_equity:
            self._peak_equity = equity
        if self._peak_equity > 0:
            dd_pct = max(0.0, (self._peak_equity - equity) / self._peak_equity)
            if dd_pct > self._max_drawdown_pct:
                self._max_drawdown_pct = dd_pct
        return equity

    # ── exposure gating ─────────────────────────────────────────────
    def _currency_concurrency(self) -> Counter[str]:
        """How many open positions touch each currency."""
        c: Counter[str] = Counter()
        for pos in self._open.values():
            c[pos.base_currency] += 1
            c[pos.quote_currency] += 1
        return c

    def _pair_concurrency(self) -> Counter[str]:
        c: Counter[str] = Counter()
        for pos in self._open.values():
            c[pos.pair] += 1
        return c

    def exposure_check(self, pair: str) -> bool:
        """True iff a new position on ``pair`` would respect all caps.

        Tests, in order:
          1. ``max_concurrent_total`` — total open count
          2. ``max_concurrent_per_pair`` — count on this exact pair
          3. ``max_concurrent_per_currency`` — count on either currency
             of the pair (the new position would increment both)
        """
        rules = self.exposure
        if rules.max_concurrent_total is not None:
            if len(self._open) >= rules.max_concurrent_total:
                return False
        if rules.max_concurrent_per_pair is not None:
            if self._pair_concurrency().get(pair, 0) >= rules.max_concurrent_per_pair:
                return False
        if rules.max_concurrent_per_currency is not None:
            cc = self._currency_concurrency()
            base, quote = pair[:3], pair[3:6]
            if cc.get(base, 0) >= rules.max_concurrent_per_currency:
                return False
            if cc.get(quote, 0) >= rules.max_concurrent_per_currency:
                return False
        return True

    # ── batch utilities ─────────────────────────────────────────────
    def open_positions_for(self, pair: str) -> Iterable[Position]:
        for pos in self._open.values():
            if pos.pair == pair:
                yield pos
