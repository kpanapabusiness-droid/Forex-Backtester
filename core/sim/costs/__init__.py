"""Per-trade cost primitives for the L_PROTOCOL Appendix B deferred-haircut model.

These are pure post-hoc R-adjustments computed on closed-trade fields.
NONE of them imports or modifies `core.sim.exit_policies.simulate_path`, the
`Account` model, or any other engine internals. The deferred-haircut placement
is intentional and documented at `core/sim/account.py:23-28`.

Public API:
    from core.sim.costs import (
        compute_swap_usd,
        compute_commission_usd,
        compute_slippage_pips,
        compute_extra_spread_price,
        rollover_instants_utc,
    )
"""

from core.sim.costs.commission import compute_commission_usd
from core.sim.costs.slippage import compute_slippage_pips
from core.sim.costs.spread_multiplier import compute_extra_spread_price
from core.sim.costs.swap import compute_swap_usd, rollover_instants_utc

__all__ = [
    "compute_swap_usd",
    "rollover_instants_utc",
    "compute_commission_usd",
    "compute_slippage_pips",
    "compute_extra_spread_price",
]
