"""``sl_plus_trailing_swing`` — SL + swing-low trail, activated at +1R MFE.

Reference: [scripts/l_arc_10_v3/step_5.py:195-217][]:

  active = False
  prev_low = -inf
  for i in range(n):
      if new_mfe_at[i] >= 1.0:
          active = True
      if active:
          if i > 0 and isfinite(new_close_at[i - 1]):
              prev_low = max(prev_low, min(new_close_at[i - 1], 0.0))
          if new_close_at[i] <= prev_low:
              trail_exit_i = i; break

In R-frame: activate when MFE ≥ +1R; once active, the trail level
is the running max of ``min(previous_close_r, 0)``, i.e. the highest
"break-even-or-worse close" seen so far. Exit when the current close
drops to or below this swing-low level. SL preempts trail when SL
breach happens at or before the trail-exit bar.

Note: ``min(prev_close_r, 0)`` caps the candidate at break-even (R=0
= entry price). The trail level never exceeds entry — by design, the
policy promotes a give-back-but-don't-let-it-go-negative protection.

Canonical engine wiring (long):

* Activation: ``bar.high_bid - entry_price >= R_atr``.
* Each post-activation bar uses the PREVIOUS bar's ``close_bid``,
  capped at ``entry_price`` (the break-even level), as the candidate
  trail level. Running max retained across bars.
* Hit: ``bar.close_bid <= trail_level``. Bar-close eval, fills at
  next-bar open_bid.
* Pre-activation ``close_bid`` of the bar JUST BEFORE activation is
  included in the running-max (matches reference's bar i-1
  contribution on activation bar i).

Short side mirrors with ask-anchoring and a max-of-running-min
inverted at entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.sim.account import Account, Direction, Position
from core.sim.exit_policies._base import (
    ExitAction,
    ExitPolicy,
    ExitPolicyContext,
    ExitPolicyDecision,
    ExitPolicyState,
)


@dataclass
class TrailingSwingState(ExitPolicyState):
    activated: bool = False
    # Long: running max of (min(prev_close_bid, entry_price)); NaN means
    # never updated. Short: running min of (max(prev_close_ask, entry_price)).
    trail_level: float = field(default=float("nan"))
    # The prior bar's close_bid (long) / close_ask (short), retained so
    # next bar's evaluation can use it as the trail candidate.
    prev_close: float = field(default=float("nan"))


class SlPlusTrailingSwingPolicy(ExitPolicy):
    """SL + swing-low (capped at entry) trail, activated at +1R MFE."""

    name = "sl_plus_trailing_swing"

    def make_state(self, ctx: ExitPolicyContext) -> ExitPolicyState:
        return TrailingSwingState()

    def evaluate_at_close(
        self,
        position: Position,
        bar: pd.Series,
        state: ExitPolicyState,
        ctx: ExitPolicyContext,
        account: Account,
    ) -> ExitPolicyDecision | None:
        assert isinstance(state, TrailingSwingState)
        r_atr = ctx.r_atr
        entry = ctx.entry_price

        if ctx.direction is Direction.LONG:
            # Activation
            if not state.activated:
                high_bid = bar.get("high_bid")
                if high_bid is not None and not pd.isna(high_bid):
                    if float(high_bid) - entry >= r_atr:
                        state.activated = True
            # Ratchet using prior bar's close_bid, capped at entry
            decision: ExitPolicyDecision | None = None
            if state.activated and not pd.isna(state.prev_close):
                candidate = min(state.prev_close, entry)
                if pd.isna(state.trail_level) or candidate > state.trail_level:
                    state.trail_level = candidate
            # Hit detection (only if trail level established)
            if state.activated and not pd.isna(state.trail_level):
                close_bid = bar.get("close_bid")
                if close_bid is not None and not pd.isna(close_bid):
                    if float(close_bid) <= state.trail_level:
                        decision = ExitPolicyDecision(
                            action=ExitAction.FULL_CLOSE,
                            exit_reason="trailing_stop_swing",
                            timing="at_close",
                        )
            # Always retain this bar's close for next-bar candidate
            close_bid = bar.get("close_bid")
            if close_bid is not None and not pd.isna(close_bid):
                state.prev_close = float(close_bid)
            return decision

        # SHORT
        if not state.activated:
            low_ask = bar.get("low_ask")
            if low_ask is not None and not pd.isna(low_ask):
                if entry - float(low_ask) >= r_atr:
                    state.activated = True
        decision = None
        if state.activated and not pd.isna(state.prev_close):
            candidate = max(state.prev_close, entry)
            if pd.isna(state.trail_level) or candidate < state.trail_level:
                state.trail_level = candidate
        if state.activated and not pd.isna(state.trail_level):
            close_ask = bar.get("close_ask")
            if close_ask is not None and not pd.isna(close_ask):
                if float(close_ask) >= state.trail_level:
                    decision = ExitPolicyDecision(
                        action=ExitAction.FULL_CLOSE,
                        exit_reason="trailing_stop_swing",
                        timing="at_close",
                    )
        close_ask = bar.get("close_ask")
        if close_ask is not None and not pd.isna(close_ask):
            state.prev_close = float(close_ask)
        return decision


__all__ = ("SlPlusTrailingSwingPolicy", "TrailingSwingState")
