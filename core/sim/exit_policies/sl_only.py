"""``sl_only`` — baseline policy. The entry-time SL is the only exit
(plus any signal-driven exit predicates registered separately by the
SignalModule, e.g. KH-24's ``kijun_d1``).

Reference: [scripts/l_arc_10_v3/step_5.py:143-151][] — path simulator
returns -1R on SL breach, else time-exit at end of held window. In the
canonical engine, the SL hit is handled by the existing intra-bar SL
infrastructure ([core/sim/fill.py:long_sl_triggered][]); time-exit is
handled by the existing ``Order.time_exit_bars`` mechanism (when
configured). This policy itself is a no-op — it serves as the
explicit "no extra exit beyond SL" baseline.
"""

from __future__ import annotations

from core.sim.exit_policies._base import (
    ExitPolicy,
    ExitPolicyContext,
    ExitPolicyState,
    NullPolicyState,
)


class SlOnlyPolicy(ExitPolicy):
    """Baseline: SL is the only exit. No-op state, no-op evaluators."""

    name = "sl_only"

    def make_state(self, ctx: ExitPolicyContext) -> ExitPolicyState:
        return NullPolicyState()


__all__ = ("SlOnlyPolicy",)
