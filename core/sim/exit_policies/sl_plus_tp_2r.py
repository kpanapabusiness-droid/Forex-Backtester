"""``sl_plus_tp_2r`` — SL + fixed TP at +2R.

Reference: [scripts/l_arc_10_v3/step_5.py:153-161][] — path simulator
returns +2.0R when MFE first crosses +2R (and TP is reached before SL),
else falls through to SL / time-exit semantics.

Canonical engine wiring: ``apply_to_order`` sets ``tp_price = entry +
2 × R_atr``. The existing intra-bar TP infrastructure
([core/sim/fill.py:long_tp_triggered][] / ``short_tp_triggered``)
handles trigger detection (long: ``bar.high_bid >= tp_price``; short:
``bar.low_ask <= tp_price``) and fill at ``tp_price``. Realised R on
trigger = ``(tp_price − entry_price) / R_atr = 2.0`` exactly.

Under PR #189 worst-case fills, the bid-anchored trigger means
canonical engine fires TP slightly LATER than the reference's
mid-anchored trigger (bid lags mid by ~half-spread). Under mid-only
parity fixtures this collapses to byte-identity.

This policy has no per-bar state — the intra-bar TP fire handles
everything.
"""

from __future__ import annotations

from typing import Any, Mapping

from core.sim.account import Direction
from core.sim.exit_policies._base import (
    ExitPolicy,
    ExitPolicyContext,
    ExitPolicyState,
    NullPolicyState,
)


class SlPlusTp2RPolicy(ExitPolicy):
    """SL + take-profit at +2R from entry."""

    name = "sl_plus_tp_2r"
    tp_r: float = 2.0

    def apply_to_order(self, ctx: ExitPolicyContext) -> Mapping[str, Any]:
        if ctx.direction is Direction.LONG:
            tp_price = ctx.entry_price + self.tp_r * ctx.r_atr
        else:
            tp_price = ctx.entry_price - self.tp_r * ctx.r_atr
        return {"tp_price": float(tp_price)}

    def make_state(self, ctx: ExitPolicyContext) -> ExitPolicyState:
        return NullPolicyState()


__all__ = ("SlPlusTp2RPolicy",)
