"""Per-position state holder for canonical exit policies.

Analogue of :class:`core.sim.trailing_stop.TrailManager`. The driver
holds one ``ExitPolicyManager`` instance per backtest run and:

  * Calls :meth:`register` at trade fill time for any Order carrying a
    policy spec — the manager constructs per-position
    :class:`ExitPolicyState` via ``policy.make_state(ctx)``.
  * Calls :meth:`evaluate_intrabar_for_all` once per bar, BEFORE the
    existing intra-bar SL/TP check. Used by policies whose triggers
    fire at a specific intra-bar price level (currently only
    ``sl_partial_close_1r_runner_trail``).
  * Calls :meth:`evaluate_at_close_for_all` once per bar, AFTER the
    trail-manager ratchet. Used by trailing-style policies and the
    runner-trail portion of partial-close.
  * Calls :meth:`deregister` on full close.

Same-bar SL preemption
----------------------
When a policy fires a PARTIAL_CLOSE intra-bar, the manager records the
position id in :attr:`positions_with_intrabar_partial_this_bar`. The
driver uses this set to SKIP the existing intra-bar SL/TP check for
that position THIS bar — matching the reference's
``sl_breach > tp1_i`` constraint for ``sl_partial_close_1r_runner_trail``
(the partial fires on the bar's high BEFORE the bar's low is
considered for SL purposes; runner survives the same-bar low).

The set is cleared at the START of each :meth:`evaluate_intrabar_for_all`
call, so it is per-bar transient.

Determinism
-----------
Positions are iterated in sorted(position_id) order at every decision
point — same convention as ``MultiPairBacktester`` and ``TrailManager``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.sim.account import Account, Position
from core.sim.exit_policies._base import (
    ExitAction,
    ExitPolicy,
    ExitPolicyContext,
    ExitPolicyDecision,
    ExitPolicyState,
)


@dataclass
class _PolicyRegistration:
    """Internal record. Not part of the public surface."""

    policy: ExitPolicy
    state: ExitPolicyState
    context: ExitPolicyContext


class ExitPolicyManager:
    """Owns per-position exit-policy state for one Account.

    The driver registers a policy at trade fill, evaluates each bar
    via the two ``evaluate_*_for_all`` hooks, and deregisters on full
    close. The manager is stateless across runs — a fresh instance is
    constructed per backtest.
    """

    def __init__(self) -> None:
        self._regs: dict[int, _PolicyRegistration] = {}
        # Per-bar transient set: positions that had an intra-bar
        # PARTIAL_CLOSE this bar. Cleared at the start of each
        # evaluate_intrabar_for_all call. Read by the driver to
        # suppress intra-bar SL/TP on the same bar.
        self.positions_with_intrabar_partial_this_bar: set[int] = set()

    # ── lifecycle ───────────────────────────────────────────────────
    def register(
        self,
        position: Position,
        policy: ExitPolicy,
        atr_at_entry: float,
        sl_atr_mult: float,
    ) -> None:
        """Attach ``policy`` to ``position``; build per-position state.

        Idempotent on the position_id key: re-registering replaces the
        prior registration. Per-bar transient set is NOT cleared (the
        register call itself does not represent a bar boundary).
        """
        ctx = ExitPolicyContext(
            entry_price=float(position.entry_price),
            atr_at_entry=float(atr_at_entry),
            sl_atr_mult=float(sl_atr_mult),
            direction=position.direction,
        )
        state = policy.make_state(ctx)
        self._regs[position.position_id] = _PolicyRegistration(
            policy=policy,
            state=state,
            context=ctx,
        )

    def deregister(self, position_id: int) -> None:
        """Remove all state for ``position_id``. Idempotent."""
        self._regs.pop(position_id, None)
        self.positions_with_intrabar_partial_this_bar.discard(position_id)

    # ── accessors ───────────────────────────────────────────────────
    def is_registered(self, position_id: int) -> bool:
        return position_id in self._regs

    def get_state(self, position_id: int) -> ExitPolicyState | None:
        reg = self._regs.get(position_id)
        return reg.state if reg is not None else None

    def get_context(self, position_id: int) -> ExitPolicyContext | None:
        reg = self._regs.get(position_id)
        return reg.context if reg is not None else None

    def get_policy_name(self, position_id: int) -> str | None:
        reg = self._regs.get(position_id)
        return reg.policy.name if reg is not None else None

    def has_intrabar_partial_this_bar(self, position_id: int) -> bool:
        """True iff this position fired a partial close this bar.

        Driver uses this to suppress intra-bar SL/TP on the same bar.
        """
        return position_id in self.positions_with_intrabar_partial_this_bar

    # ── per-bar evaluation ──────────────────────────────────────────
    def evaluate_intrabar_for_all(
        self,
        snapshot: dict[str, pd.Series | None],
        account: Account,
    ) -> dict[int, ExitPolicyDecision]:
        """Evaluate every registered policy's intra-bar hook.

        Returns ``{position_id: decision}`` for positions whose
        policy fired a decision THIS bar. Driver acts on these
        immediately at ``decision.fill_price`` (for PARTIAL_CLOSE)
        or via full close (for FULL_CLOSE — unusual at intra-bar
        timing; currently not used by any registered policy).

        Side effects:
          * Clears :attr:`positions_with_intrabar_partial_this_bar`
            at the start of the call.
          * Repopulates it for positions whose policy emitted a
            PARTIAL_CLOSE decision this bar.
        """
        self.positions_with_intrabar_partial_this_bar = set()
        decisions: dict[int, ExitPolicyDecision] = {}
        for pos_id in sorted(self._regs):
            reg = self._regs[pos_id]
            # noqa: SLF001 — manager peers at account internals (parity
            # with TrailManager). Public surface is account.open_positions
            # but that returns a tuple copy; we need the dict for lookup.
            pos = account._open.get(pos_id)  # noqa: SLF001
            if pos is None:
                # Position closed by some other path (e.g. external close).
                # Don't deregister here — driver owns deregistration on
                # full close. Just skip.
                continue
            bar = snapshot.get(pos.pair)
            if bar is None:
                continue
            decision = reg.policy.evaluate_intrabar(pos, bar, reg.state, reg.context)
            if decision is None:
                continue
            decisions[pos_id] = decision
            if decision.action is ExitAction.PARTIAL_CLOSE:
                self.positions_with_intrabar_partial_this_bar.add(pos_id)
        return decisions

    def evaluate_at_close_for_all(
        self,
        snapshot: dict[str, pd.Series | None],
        account: Account,
    ) -> dict[int, ExitPolicyDecision]:
        """Evaluate every registered policy's at-close hook.

        Returns ``{position_id: decision}`` for positions whose
        policy fired this bar. Driver queues each FULL_CLOSE as a
        ``_pending_closes`` entry (fills at next-bar open).

        Does NOT mutate the intrabar-partial set.
        """
        decisions: dict[int, ExitPolicyDecision] = {}
        for pos_id in sorted(self._regs):
            reg = self._regs[pos_id]
            pos = account._open.get(pos_id)  # noqa: SLF001
            if pos is None:
                continue
            bar = snapshot.get(pos.pair)
            if bar is None:
                continue
            decision = reg.policy.evaluate_at_close(
                pos, bar, reg.state, reg.context, account
            )
            if decision is not None:
                decisions[pos_id] = decision
        return decisions


__all__ = ("ExitPolicyManager",)
