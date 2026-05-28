"""Per-trade spread-multiplier extra-cost calculator.

The recorded per-trade spread (in price units, ``spread_close_at_entry +
spread_close_at_exit``) IS already deducted via HistData bid/ask in
``simulate_path``. To stress-test the strategy against wider broker spreads,
this primitive computes the EXTRA cost from widening by a multiplier:

    extra_spread_price = (spread_entry + spread_exit) × (mult − 1)

Per dispatch §3.3:
- NO floor. ``configs/spread_floors_5ers.yaml`` does NOT exist and is NOT used.
- Zero-spread trades (data gaps): cost remains 0 — do NOT floor them. Driver
  must count + report (surface if >1% of pool).

Convert to R in the caller: ``extra_spread_r = extra_spread_price / sl_distance_price``.

Pure post-hoc R-adjustment primitive. Does NOT modify simulate_path.
"""

from __future__ import annotations


def compute_extra_spread_price(
    spread_at_entry_price: float,
    spread_at_exit_price: float,
    mult: float,
) -> float:
    """Extra spread cost in PRICE units from widening recorded spread by ``mult``.

    At ``mult == 1.0``: returns 0 (no widening).
    At ``mult > 1.0``: returns ``(spread_entry + spread_exit) × (mult − 1)``.
    At ``mult < 1.0``: not meaningful for stress-testing; treated as 0 by clamp.

    Args
    ----
    spread_at_entry_price : float
        Recorded HistData spread at entry, in price units (ask - bid).
    spread_at_exit_price : float
        Recorded HistData spread at exit, in price units.
    mult : float
        Multiplier; ``effective_spread = recorded_spread × mult``.

    Returns
    -------
    Extra spread cost in price units. Always non-negative.
    """
    if mult <= 1.0:
        return 0.0
    spread_total = spread_at_entry_price + spread_at_exit_price
    if spread_total <= 0:
        # Zero-spread trades (data gaps) — no widening cost per dispatch §3.3.
        return 0.0
    return spread_total * (mult - 1.0)
