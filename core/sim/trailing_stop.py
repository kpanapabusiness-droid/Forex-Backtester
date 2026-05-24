"""Trailing-stop state machine for the v3 multipair backtester.

Generic enough for any arc; KH-24 is the first user. Activation /
trail-distance / update-frequency are config inputs — KH-24's
``2.0 × ATR(14)`` activation and ``1.5 × ATR(14)`` trail are passed in
at trade open, not hardcoded.

Design:

  - ``Position`` (PR-B) stays immutable. Trailing state lives in a
    parallel ``dict[position_id, TrailState]`` on the Account, keyed by
    position_id. ``Account.open_trail(...)`` registers it at entry.
  - Updates fire at bar close only (NOT intra-bar) — the canonical
    KH-24 convention.
  - On activation (close crosses entry + activation_atr_mult × ATR),
    the trailing logic starts tracking ``highest_close_since_activation``
    and the effective stop is ``highest_close − trail_atr_mult × ATR``.
  - Until activated, the original fixed SL remains in force (set on
    ``Position.sl_price``).
  - When the trail rises above the original fixed SL, the effective
    stop becomes the trail level — but the trail never lowers below
    the most-recent trail level (ratchet-only).

Long-only in this PR per the dispatch's "Long-only for now; short
symmetric implementation deferred".

The driver calls ``trail_update_at_close(snapshot, account)`` once at
the end of each bar (after exits + mark-to-market, before next-bar
entries). The trail-state's ``current_sl_price`` then drives intra-bar
SL checks on the *next* bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.sim.account import Account, Direction, Position


@dataclass
class TrailState:
    """Trailing-stop runtime state for one long position.

    ``current_sl_price`` is what the driver actually uses for SL
    triggers; on registration it equals the position's original
    ``sl_price``, and after activation it ratchets up with the
    highest close.
    """

    position_id: int
    entry_price: float
    atr_at_entry: float
    activation_atr_mult: float = 2.0
    trail_atr_mult: float = 1.5
    activated: bool = False
    highest_close_since_activation: float = float("nan")
    current_sl_price: float = float("nan")  # tracks the live SL

    @property
    def activation_close_threshold(self) -> float:
        return self.entry_price + self.activation_atr_mult * self.atr_at_entry

    def update_at_close(self, close_price: float) -> bool:
        """Process a bar-close price; ratchet trail if needed.

        Returns True if ``current_sl_price`` was updated this bar
        (useful for logging / determinism asserts).
        """
        if not self.activated:
            if close_price >= self.activation_close_threshold:
                self.activated = True
                self.highest_close_since_activation = close_price
                proposed_trail = close_price - self.trail_atr_mult * self.atr_at_entry
                if proposed_trail > self.current_sl_price:
                    self.current_sl_price = proposed_trail
                    return True
            return False

        # Already activated: ratchet on new highs only
        if close_price > self.highest_close_since_activation:
            self.highest_close_since_activation = close_price
            proposed_trail = close_price - self.trail_atr_mult * self.atr_at_entry
            if proposed_trail > self.current_sl_price:
                self.current_sl_price = proposed_trail
                return True
        return False


class TrailManager:
    """Owns per-position trail states for one Account.

    Account doesn't know about trails; TrailManager is the adapter
    that's registered with the driver as an exit-hook source.
    """

    def __init__(self) -> None:
        self._states: dict[int, TrailState] = {}

    def register(
        self,
        position: Position,
        atr_at_entry: float,
        activation_atr_mult: float = 2.0,
        trail_atr_mult: float = 1.5,
    ) -> TrailState:
        if position.direction is not Direction.LONG:
            raise NotImplementedError("Trailing stop is long-only in PR-E.1")
        if position.sl_price is None:
            raise ValueError("Trailing requires a starting fixed SL")
        state = TrailState(
            position_id=position.position_id,
            entry_price=position.entry_price,
            atr_at_entry=float(atr_at_entry),
            activation_atr_mult=activation_atr_mult,
            trail_atr_mult=trail_atr_mult,
            current_sl_price=float(position.sl_price),
        )
        self._states[position.position_id] = state
        return state

    def deregister(self, position_id: int) -> None:
        self._states.pop(position_id, None)

    def get(self, position_id: int) -> TrailState | None:
        return self._states.get(position_id)

    def update_all_at_close(
        self, snapshot: dict[str, pd.Series | None], account: Account
    ) -> dict[int, float]:
        """Update every registered trail using the bar's MID close.

        Signal-parity convention (PR #187, supersedes PR-E.1.6 bid-side
        trail): activation + ratchet operate on mid-close so that trail
        behaviour is venue-independent. Hit detection (see
        ``trail_exit_triggers_at_close``) still uses bid-side close per
        worst-case-fill realism. The EA must be updated to read mid in
        a parallel deployment PR.

        Returns ``{position_id: new_sl_price}`` for positions whose
        trail moved this bar.
        """
        updates: dict[int, float] = {}
        for pos_id in sorted(self._states):
            state = self._states[pos_id]
            pos = account._open.get(pos_id)  # noqa: SLF001
            if pos is None:
                self.deregister(pos_id)
                continue
            bar = snapshot.get(pos.pair)
            if bar is None or pd.isna(bar.get("close_bid")) or pd.isna(bar.get("close_ask")):
                continue
            close_mid = (float(bar["close_bid"]) + float(bar["close_ask"])) / 2.0
            if state.update_at_close(close_mid):
                updates[pos_id] = state.current_sl_price
        return updates

    def trail_exit_triggers_at_close(
        self, snapshot: dict[str, pd.Series | None], account: Account
    ) -> dict[int, float]:
        """Identify positions whose trail's activated AND bid close ≤ trail.

        Signal-parity convention (PR #187): activation/ratchet uses mid
        (see ``update_all_at_close``); hit uses bid for worst-case-fill
        realism — long exits when its bid falls to the trail level.

        Returns ``{position_id: trail_level_when_triggered}``. Caller
        decides what fill price to use.
        """
        triggers: dict[int, float] = {}
        for pos_id in sorted(self._states):
            state = self._states[pos_id]
            if not state.activated:
                continue
            pos = account._open.get(pos_id)  # noqa: SLF001
            if pos is None:
                continue
            bar = snapshot.get(pos.pair)
            if bar is None or pd.isna(bar.get("close_bid")):
                continue
            if float(bar["close_bid"]) <= state.current_sl_price:
                triggers[pos_id] = state.current_sl_price
        return triggers

    def effective_sl(self, position: Position) -> float | None:
        """Return the live SL price for ``position``, including trail updates.

        Falls back to ``position.sl_price`` when no trail is registered.
        """
        state = self._states.get(position.position_id)
        if state is None:
            return position.sl_price
        return state.current_sl_price
