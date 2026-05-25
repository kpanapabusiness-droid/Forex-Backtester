"""``sl_plus_trailing_atr`` — SL + 1R-below-peak-MFE trail, activated at +1R.

Reference: [scripts/l_arc_10_v3/step_5.py:173-193][]:

  active = False
  trail_r = -1.0
  for i in range(n):
      if new_mfe_at[i] >= 1.0:
          active = True
          trail_r = max(trail_r, new_mfe_at[i] - 1.0)
      if active and new_close_at[i] <= trail_r:
          trail_exit_i = i
          break

In R-frame: activate when MFE ≥ +1R; trail at ``peak_mfe − 1R``;
exit when close ≤ trail level. SL preempts trail when SL breach
happens at or before the trail-exit bar (reference §187).

Canonical engine wiring:

* Activation: MFE crossed +1R since entry. Detected via
  ``bar.high_bid - entry_price >= R_atr`` for long; symmetric for
  short (bar.low_ask). High-derived, matching reference's
  ``new_mfe_at`` which is high-derived.
* Trail anchor: peak ``bar.high_bid`` for long (bid-anchored — the
  exit reference price). For short, trough ``bar.low_ask``.
* Trail level: peak − ``R_atr`` for long; trough + ``R_atr`` for short.
* Trail-hit detection: ``bar.close_bid <= trail_level`` for long;
  ``bar.close_ask >= trail_level`` for short. Bar-close evaluation;
  queues full close at next-bar open (long: ``open_bid``).
* SL preemption is automatic via the existing intra-bar SL infrastructure.

Distinct from KH-24's ``TrailManager`` ([core/sim/trailing_stop.py][]):
the latter has activation at ``+activation_atr_mult × ATR`` (default
2.0) and trail distance ``trail_atr_mult × ATR`` (default 1.5),
mid-close-based ratchet, bid-close hit. This canonical policy uses
+1R activation and 1R trail (i.e. ``R_atr`` distance — equivalent to
``sl_atr_mult × ATR``), bid-anchored peak for long. Different beast,
different config knobs, separate manager.
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
class TrailingAtrState(ExitPolicyState):
    activated: bool = False
    # Long: max(bid_high) since activation; short: min(ask_low) since
    # activation. NaN means uninitialised.
    peak_price: float = field(default=float("nan"))


def _long_mfe_crossed_one_r(bar: pd.Series, entry_price: float, r_atr: float) -> bool:
    high_bid = bar.get("high_bid")
    if high_bid is None or pd.isna(high_bid):
        return False
    return float(high_bid) - entry_price >= r_atr


def _short_mfe_crossed_one_r(bar: pd.Series, entry_price: float, r_atr: float) -> bool:
    low_ask = bar.get("low_ask")
    if low_ask is None or pd.isna(low_ask):
        return False
    return entry_price - float(low_ask) >= r_atr


class SlPlusTrailingAtrPolicy(ExitPolicy):
    """SL + 1R-below-peak-MFE trail, activated at +1R MFE."""

    name = "sl_plus_trailing_atr"

    def make_state(self, ctx: ExitPolicyContext) -> ExitPolicyState:
        return TrailingAtrState()

    def evaluate_at_close(
        self,
        position: Position,
        bar: pd.Series,
        state: ExitPolicyState,
        ctx: ExitPolicyContext,
        account: Account,
    ) -> ExitPolicyDecision | None:
        assert isinstance(state, TrailingAtrState)
        r_atr = ctx.r_atr
        entry = ctx.entry_price

        if ctx.direction is Direction.LONG:
            # Activation check
            if not state.activated:
                if _long_mfe_crossed_one_r(bar, entry, r_atr):
                    state.activated = True
                    state.peak_price = float(bar["high_bid"])
                else:
                    return None
            else:
                # Ratchet peak
                high_bid = bar.get("high_bid")
                if high_bid is not None and not pd.isna(high_bid):
                    high_bid_f = float(high_bid)
                    if pd.isna(state.peak_price) or high_bid_f > state.peak_price:
                        state.peak_price = high_bid_f
            # Hit detection
            if pd.isna(state.peak_price):
                return None
            trail_level = state.peak_price - r_atr
            close_bid = bar.get("close_bid")
            if close_bid is None or pd.isna(close_bid):
                return None
            if float(close_bid) <= trail_level:
                return ExitPolicyDecision(
                    action=ExitAction.FULL_CLOSE,
                    exit_reason="trailing_stop_atr",
                    timing="at_close",
                )
            return None

        # SHORT
        if not state.activated:
            if _short_mfe_crossed_one_r(bar, entry, r_atr):
                state.activated = True
                state.peak_price = float(bar["low_ask"])
            else:
                return None
        else:
            low_ask = bar.get("low_ask")
            if low_ask is not None and not pd.isna(low_ask):
                low_ask_f = float(low_ask)
                if pd.isna(state.peak_price) or low_ask_f < state.peak_price:
                    state.peak_price = low_ask_f
        if pd.isna(state.peak_price):
            return None
        trail_level = state.peak_price + r_atr
        close_ask = bar.get("close_ask")
        if close_ask is None or pd.isna(close_ask):
            return None
        if float(close_ask) >= trail_level:
            return ExitPolicyDecision(
                action=ExitAction.FULL_CLOSE,
                exit_reason="trailing_stop_atr",
                timing="at_close",
            )
        return None


__all__ = ("SlPlusTrailingAtrPolicy", "TrailingAtrState")
