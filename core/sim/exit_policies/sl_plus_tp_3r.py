"""``sl_plus_tp_3r`` — SL + fixed TP at +3R.

Reference: [scripts/l_arc_10_v3/step_5.py:163-171][]. Identical
mechanics to ``sl_plus_tp_2r`` with the threshold moved to +3R.
"""

from __future__ import annotations

from core.sim.exit_policies.sl_plus_tp_2r import SlPlusTp2RPolicy


class SlPlusTp3RPolicy(SlPlusTp2RPolicy):
    """SL + take-profit at +3R from entry."""

    name = "sl_plus_tp_3r"
    tp_r: float = 3.0


__all__ = ("SlPlusTp3RPolicy",)
