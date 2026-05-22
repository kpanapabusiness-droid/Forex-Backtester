"""Live-balance risk sizing — matches the deployed KH-24 EA's CalcLots.

The EA reads ``AccountInfoDouble(ACCOUNT_BALANCE)`` at signal time and
sizes the trade as a fixed risk percentage of that **live** balance —
which compounds with realised PnL (winning streaks size up; drawdowns
size down).

This module replaces ``ResetFloorAccount`` for KH-24. The reset-floor
model (5ers ratchet, no compounding on losing days) remains available
in ``core/sim/risk/reset_floor.py`` for L-arc work that uses the 5ers
floor convention.

Sizing formula::

    risk_amount_quote = account.balance × risk_pct
    units             = risk_amount_quote / |entry_price − sl_price|

For USD-quote pairs (EURUSD, GBPUSD, etc.) ``account.balance`` is
USD-denominated and the realised loss on SL hit equals ``risk_amount``.
For non-USD-quote pairs (USDJPY, AUDCAD, EURGBP) the formula treats
the balance as quote-currency-denominated — a documented simplification
inherited from PR-E.1 (full USD cross-rate conversion lives in a later
PR). The EA implicitly handles this via MT5's ``SymbolInfoDouble(
SYMBOL_TRADE_TICK_VALUE)`` which returns account-currency-per-pip; the
v3 simplification will diverge slightly on non-USD-quote pairs.

See ``docs/dispatches/kh24_ea_full_diff.md`` Section E.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.sim.account import Account


@dataclass
class LiveBalanceRisk:
    """Live-balance position sizing — KH-24's EA convention.

    Usage::

        risk = LiveBalanceRisk(risk_pct=0.01)
        # at signal time:
        size = risk.risk_size(account, entry_price=1.10, sl_price=1.098)

    The account is consulted at each call — sizing compounds with the
    account's realised PnL automatically.
    """

    risk_pct: float = 0.01  # 1% per trade (KH-24 EA RiskPercent default)

    def risk_size(
        self,
        account: Account,
        entry_price: float,
        sl_price: float,
        risk_pct: float | None = None,
    ) -> float:
        """Return position size in units of base currency.

        ``floor × risk_pct / |entry − sl|`` but with ``floor`` replaced
        by ``account.balance`` (live, compounding).
        """
        rp = float(risk_pct if risk_pct is not None else self.risk_pct)
        sl_distance = abs(float(entry_price) - float(sl_price))
        if sl_distance == 0:
            raise ValueError(
                f"Cannot size trade with zero SL distance (entry={entry_price}, sl={sl_price})"
            )
        return (float(account.balance) * rp) / sl_distance
