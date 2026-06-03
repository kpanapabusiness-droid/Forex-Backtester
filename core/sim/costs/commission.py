"""Per-trade commission (round-turn) calculator.

Standard ECN broker convention: flat USD per standard lot, charged once per
closed trade (entry-leg + exit-leg combined as a single round-turn).

5ers default per dispatch §3.2: $4.00 / lot RT, applied on the ORIGINAL lot
(not reduced after TP1 partial close — commission is for the full round-turn
of the position, not per-leg).

Gate default note: the gate cost model (``core.sim.costs.model.CostModel``)
passes the FundedNext rate ``rate_per_lot_rt=5.0`` explicitly — the higher
cost, as the conservative bound. This primitive's ``$4.00`` default (5ers) is
left unchanged; only the caller's override differs, so the cost math here is
untouched.

Pure post-hoc R-adjustment primitive. Does NOT modify simulate_path. Designed
for the deferred-haircut model documented at core/sim/account.py:23-28.
"""

from __future__ import annotations


def compute_commission_usd(
    lots_original: float,
    rate_per_lot_rt: float = 4.0,
) -> float:
    """Round-turn commission in USD.

    Per dispatch §3.2: $4 USD/lot round-turn, full original lot, once per closed
    trade. Scales linearly with lot size.

    Args
    ----
    lots_original : float
        Position size at entry, in standard lots (100,000 base units).
    rate_per_lot_rt : float
        USD per lot round-turn. Default 4.0 (5ers convention).

    Returns
    -------
    Total commission in USD (always non-negative — commission is a cost).
    """
    if lots_original < 0:
        raise ValueError(
            f"compute_commission_usd: lots_original must be >= 0, got {lots_original}"
        )
    return rate_per_lot_rt * lots_original
