"""5ers reset-floor balance accounting + risk-per-trade sizing.

The 5ers prop firm tracks a "reset floor" balance separate from the
running equity. Concretely:

  - Account starts at ``starting_balance`` (e.g. $100,000).
  - Floor = MAX(prior floor, current balance) at each daily close
    (UTC midnight) — ratchets up on winning days, never down on losing
    days.
  - Per-trade risk is computed off the FLOOR, not equity. KH-24 uses
    1.0% of floor (L_arc convention is 0.5%, configurable).

Position size in **units of base currency** for a long entry is::

    risk_amount_quote = floor * risk_pct
    units = risk_amount_quote / (entry_price - sl_price)

This treats the floor as denominated in QUOTE currency for the
purposes of sizing. For USD-quote pairs (EURUSD, GBPUSD, etc.) the
floor IS USD-equivalent. For non-USD-quote pairs (USDJPY, EURGBP,
AUDCAD) the floor is treated as quote-currency-denominated — this is
a documented simplification; full cross-rate conversion lives in a
later PR. KH-24 anchor reproduction in PR-E.2 will verify whether the
simplification meaningfully shifts the published numbers.

The reset-floor manager attaches to a regular ``Account`` and is
queried by the strategy when computing position sizes. It also
exposes ``update_at_day_close(t, balance)`` for the driver to call at
each daily boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ResetFloorAccount:
    """Floor-tracking layer over a balance series.

    Usage::

        floor = ResetFloorAccount(starting_balance=100_000)
        # ... at each daily UTC midnight, after mark-to-market:
        floor.update_at_day_close(t, account.balance)
        # ... when sizing a trade:
        size = floor.risk_size(entry_price=1.10, sl_price=1.098)
    """

    starting_balance: float
    risk_pct: float = 0.01  # 1% per trade (KH-24 convention)
    _floor: float = field(init=False)
    _last_day_seen: pd.Timestamp | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._floor = float(self.starting_balance)

    @property
    def floor(self) -> float:
        return self._floor

    def update_at_day_close(self, t: pd.Timestamp, balance: float) -> bool:
        """Ratchet the floor up if ``balance`` exceeded prior floor.

        Returns True iff the floor moved this call. Idempotent within
        the same calendar day — only the FIRST call per UTC day
        updates the floor (the daily close).
        """
        day = pd.Timestamp(t).normalize()
        if self._last_day_seen is not None and day <= self._last_day_seen:
            return False
        self._last_day_seen = day
        if balance > self._floor:
            self._floor = float(balance)
            return True
        return False

    def risk_size(
        self,
        entry_price: float,
        sl_price: float,
        risk_pct: float | None = None,
    ) -> float:
        """Compute position size in units of base currency.

        Returns ``floor × risk_pct / |entry_price − sl_price|``. Raises
        ``ValueError`` if ``entry_price == sl_price`` (zero risk
        distance — caller should never hit this).
        """
        rp = float(risk_pct if risk_pct is not None else self.risk_pct)
        sl_distance = abs(float(entry_price) - float(sl_price))
        if sl_distance == 0:
            raise ValueError(
                f"Cannot size trade with zero SL distance (entry={entry_price}, sl={sl_price})"
            )
        return (self._floor * rp) / sl_distance
