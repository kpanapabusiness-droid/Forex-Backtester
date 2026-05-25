"""``sl_partial_close_1r_runner_trail`` — Arc 10's load-bearing exit.

Reference: [scripts/l_arc_10_v3/step_5.py:219-250][]:

  tp1_i = first_at_least(new_mfe_at, 1.0)   # MFE (high) first ≥ +1R
  if tp1_i < 0:
      # never reached → falls through to SL / time-exit semantics
      ...
  half_r = 1.0
  trail_r = 0.0
  trail_exit_i = -1
  for i in range(tp1_i, n):
      if isfinite(new_mfe_at[i]):
          trail_r = max(trail_r, new_mfe_at[i] - 1.0)
      if isfinite(new_close_at[i]) and new_close_at[i] <= trail_r and i > tp1_i:
          trail_exit_i = i; break
  if sl_breach >= 0 and sl_breach > tp1_i and (trail_exit_i < 0 or sl_breach <= trail_exit_i):
      runner_r = -1.0
  elif trail_exit_i >= 0:
      runner_r = float(new_close_at[trail_exit_i])
  else:
      runner_r = float(new_close_at[end_held])
  final_r = 0.5 * half_r + 0.5 * runner_r

Semantics (long):

* **Stage 1 — pre-tp1.** Position at full size, original SL binding.
  When MFE (bar high) first crosses +1R → fire PARTIAL close (50%) at
  the +1R price level (intra-bar trigger, fill AT tp1_price). Mirrors
  the existing intra-bar TP infrastructure mechanism.
* **Stage 2 — runner phase (post-tp1).** Remaining 50% open.
  - Original SL still binds (intra-bar SL infrastructure handles).
  - Trail anchor: ``peak_mfe_since_entry`` = running max of
    ``bar.high_bid``. The reference uses peak MFE over the entire
    path (initialised before tp1_i); canonical engine mirrors that
    by ratcheting peak from bar 0, not from tp1.
  - Trail level: ``peak_high_bid − R_atr`` (i.e. 1R below peak).
  - Trail-hit detection: ``bar.close_bid <= trail_level`` AND the
    current bar is strictly AFTER the tp1 bar (reference's
    ``i > tp1_i`` constraint — same-bar partial + trail-exit is
    forbidden). Queues a runner full-close at next-bar open_bid.

Short side mirrors with ask-anchored trough peak + +1R-above-trough
trail level.

Wire interactions:

* The driver's existing intra-bar SL check fires on ``bar.low_bid <=
  sl_price`` (long); when the SL fires after a partial, the
  remaining 50% closes at ``sl_price`` (size = current_size after
  partial). The ``parent_position_id`` linkage on the multi-leg
  ClosedTrade lets consumers reconstruct the full close sequence.
* The driver's existing exit_predicates and trail_manager run
  independently. If both fire on the same bar as the partial-close
  manager, precedence per [PROTOCOL_RUNTIME.md §8b][] (trail wins
  vs predicate; intra-bar SL wins over both). This policy operates
  on the same bar as the partial fire intra-bar — by design
  (partial first, then SL if SL was breached on the SAME bar as
  partial trigger, which the reference does not handle — the
  reference's ``sl_breach > tp1_i`` means same-bar SL is ignored).
* For determinism, the partial fires BEFORE the intra-bar SL check
  on the same bar (matches reference's loop-order: tp1 detection
  scans MFE first, then SL is only evaluated against
  ``sl_breach > tp1_i``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

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
class PartialCloseRunnerTrailState(ExitPolicyState):
    tp1_fired: bool = False
    # Bar index of the tp1 fire (used to enforce reference's
    # ``i > tp1_i`` constraint on trail exit). Set to the integer
    # ordinal of the bar within the position's life.
    tp1_bar_ordinal: int | None = None
    # Long: running max of bar.high_bid since entry; short: running min
    # of bar.low_ask. Ratcheted from bar 0 (NOT from tp1) to match
    # reference's path-wide peak.
    peak_price: float = field(default=float("nan"))
    # Bar ordinal counter (incremented on each evaluate_at_close call).
    bar_ordinal: int = 0


class SlPartialClose1RRunnerTrailPolicy(ExitPolicy):
    """Close 50% at +1R; runner trails at 1R below path-peak MFE."""

    name = "sl_partial_close_1r_runner_trail"
    partial_fraction: float = 0.5

    # No apply_to_order: tp1 fires via state machine intra-bar, not via
    # the engine's tp_price field (because tp_price fires a FULL close,
    # not partial). Keeping it state-driven is the simplest path.
    def apply_to_order(self, ctx: ExitPolicyContext) -> Mapping[str, Any]:
        return {}

    def make_state(self, ctx: ExitPolicyContext) -> ExitPolicyState:
        return PartialCloseRunnerTrailState()

    # ── intra-bar: detect +1R cross, fire partial ────────────────────
    def evaluate_intrabar(
        self,
        position: Position,
        bar: pd.Series,
        state: ExitPolicyState,
        ctx: ExitPolicyContext,
    ) -> ExitPolicyDecision | None:
        assert isinstance(state, PartialCloseRunnerTrailState)
        if state.tp1_fired:
            return None
        r_atr = ctx.r_atr
        entry = ctx.entry_price
        if ctx.direction is Direction.LONG:
            high_bid = bar.get("high_bid")
            if high_bid is None or pd.isna(high_bid):
                return None
            tp1_price = entry + r_atr
            if float(high_bid) >= tp1_price:
                state.tp1_fired = True
                state.tp1_bar_ordinal = state.bar_ordinal
                return ExitPolicyDecision(
                    action=ExitAction.PARTIAL_CLOSE,
                    exit_reason="partial_close_1r",
                    timing="intrabar",
                    fill_price=tp1_price,
                    partial_fraction=self.partial_fraction,
                )
            return None
        # SHORT
        low_ask = bar.get("low_ask")
        if low_ask is None or pd.isna(low_ask):
            return None
        tp1_price = entry - r_atr
        if float(low_ask) <= tp1_price:
            state.tp1_fired = True
            state.tp1_bar_ordinal = state.bar_ordinal
            return ExitPolicyDecision(
                action=ExitAction.PARTIAL_CLOSE,
                exit_reason="partial_close_1r",
                timing="intrabar",
                fill_price=tp1_price,
                partial_fraction=self.partial_fraction,
            )
        return None

    # ── at close: ratchet peak, runner-trail hit detection ───────────
    def evaluate_at_close(
        self,
        position: Position,
        bar: pd.Series,
        state: ExitPolicyState,
        ctx: ExitPolicyContext,
        account: Account,
    ) -> ExitPolicyDecision | None:
        assert isinstance(state, PartialCloseRunnerTrailState)
        r_atr = ctx.r_atr
        entry = ctx.entry_price
        # Always ratchet path-wide peak (per reference: peak runs from bar 0)
        if ctx.direction is Direction.LONG:
            high_bid = bar.get("high_bid")
            if high_bid is not None and not pd.isna(high_bid):
                hb = float(high_bid)
                if pd.isna(state.peak_price) or hb > state.peak_price:
                    state.peak_price = hb
        else:
            low_ask = bar.get("low_ask")
            if low_ask is not None and not pd.isna(low_ask):
                la = float(low_ask)
                if pd.isna(state.peak_price) or la < state.peak_price:
                    state.peak_price = la

        decision: ExitPolicyDecision | None = None
        # Runner trail-hit only post-tp1 AND strictly AFTER the tp1 bar
        # (reference: i > tp1_i). Compare bar_ordinal > tp1_bar_ordinal.
        if (
            state.tp1_fired
            and state.tp1_bar_ordinal is not None
            and state.bar_ordinal > state.tp1_bar_ordinal
            and not pd.isna(state.peak_price)
        ):
            if ctx.direction is Direction.LONG:
                trail_level = state.peak_price - r_atr
                # Reference uses close_r (mid-anchored); canonical uses
                # close_bid (worst-case fill realism). Under mid-only
                # fixture they're identical.
                close_bid = bar.get("close_bid")
                if close_bid is not None and not pd.isna(close_bid):
                    if float(close_bid) <= trail_level:
                        decision = ExitPolicyDecision(
                            action=ExitAction.FULL_CLOSE,
                            exit_reason="runner_trail_stop",
                            timing="at_close",
                        )
            else:
                trail_level = state.peak_price + r_atr
                close_ask = bar.get("close_ask")
                if close_ask is not None and not pd.isna(close_ask):
                    if float(close_ask) >= trail_level:
                        decision = ExitPolicyDecision(
                            action=ExitAction.FULL_CLOSE,
                            exit_reason="runner_trail_stop",
                            timing="at_close",
                        )

        # Advance bar ordinal AFTER trail-hit eval (so the SAME bar as
        # tp1 has bar_ordinal == tp1_bar_ordinal, blocking same-bar
        # trail-exit; next bar has bar_ordinal == tp1_bar_ordinal + 1,
        # unlocking trail-exit).
        state.bar_ordinal += 1
        return decision


__all__ = (
    "SlPartialClose1RRunnerTrailPolicy",
    "PartialCloseRunnerTrailState",
)
