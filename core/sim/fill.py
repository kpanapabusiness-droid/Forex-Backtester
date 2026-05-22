"""Bar-level fill primitives — pure functions of a single bar.

Conventions:
    Long entry  → fills at ``open_ask``  (buying — pay the ask)
    Long exit   → fills at ``close_bid`` (selling — hit the bid)
    Long SL     → triggered when ``low_bid <= sl_price``; fills at sl_price
    Long TP     → triggered when ``high_bid >= tp_price``; fills at tp_price
    Short entry → fills at ``open_bid``  (selling — hit the bid)
    Short exit  → fills at ``close_ask`` (buying back — pay the ask)
    Short SL    → triggered when ``high_ask >= sl_price``; fills at sl_price
    Short TP    → triggered when ``low_ask <= tp_price``; fills at tp_price

Entry uses the *next* bar's open (see ``MultiPairBacktester``); these
primitives operate on the bar they're given.

Intra-bar SL/TP priority is *not* decided here — the caller picks the
priority (``"sl_first" | "tp_first"``) and consults both predicates.
The default convention through the rest of the v3 engine is ``sl_first``
(conservative).

Each function returns ``(triggered: bool, fill_price: float)``. For
entry/market-exit, ``triggered`` is always True (the bar is the trigger);
the price is the realised fill.
"""

from __future__ import annotations

import pandas as pd

# ────────────────────────────────────────────────────────────────────────
# Long side
# ────────────────────────────────────────────────────────────────────────


def long_entry_fill_price(bar: pd.Series) -> float:
    """Fill at next-bar open ask. Caller passes the entry bar directly."""
    return float(bar["open_ask"])


def long_exit_market_price(bar: pd.Series) -> float:
    """Market exit of a long: sell at close_bid."""
    return float(bar["close_bid"])


def long_sl_triggered(bar: pd.Series, sl_price: float) -> tuple[bool, float]:
    """Long SL fires if the bid touched ``sl_price`` during the bar.

    The bid is the reference because that's where a long would be filled
    on a stop-sell.
    """
    if bar["low_bid"] <= sl_price:
        return True, float(sl_price)
    return False, float("nan")


def long_tp_triggered(bar: pd.Series, tp_price: float) -> tuple[bool, float]:
    """Long TP fires if the bid reached ``tp_price`` during the bar."""
    if bar["high_bid"] >= tp_price:
        return True, float(tp_price)
    return False, float("nan")


# ────────────────────────────────────────────────────────────────────────
# Short side
# ────────────────────────────────────────────────────────────────────────


def short_entry_fill_price(bar: pd.Series) -> float:
    """Short entry: sell at the bid (open_bid of the entry bar)."""
    return float(bar["open_bid"])


def short_exit_market_price(bar: pd.Series) -> float:
    """Market exit of a short: buy back at close_ask."""
    return float(bar["close_ask"])


def short_sl_triggered(bar: pd.Series, sl_price: float) -> tuple[bool, float]:
    """Short SL fires if the ask reached ``sl_price`` (stop-buy reference)."""
    if bar["high_ask"] >= sl_price:
        return True, float(sl_price)
    return False, float("nan")


def short_tp_triggered(bar: pd.Series, tp_price: float) -> tuple[bool, float]:
    """Short TP fires if the ask touched ``tp_price``."""
    if bar["low_ask"] <= tp_price:
        return True, float(tp_price)
    return False, float("nan")
