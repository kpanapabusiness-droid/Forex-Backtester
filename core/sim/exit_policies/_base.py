"""Canonical exit-policy primitives for the v3 multipair backtester.

A *policy* is a self-contained recipe for "what happens to a position
after entry, beyond the original entry-time SL". The canonical engine
registry holds policies that are reusable across architectures
(A1..A6) — Step 5's per-arc `_apply_exit_policy` hand-rolled
simulators are now thin wrappers over this registry (per the
sl_partial_close_runner_trail PR's per-arc migration phase).

Policy lifecycle

  1. ``apply_to_order(ctx)`` is called at Order construction time. The
     policy may modify Order kwargs — e.g. TP-style policies set
     ``tp_price`` so the existing intra-bar TP infrastructure
     ([core.sim.fill.long_tp_triggered][]) handles fire / fill at the
     TP level. Most policies need nothing here and return ``{}``.

  2. ``make_state(ctx)`` is called at trade fill time by the
     ``ExitPolicyManager`` (analogue of ``TrailManager``). The
     returned ``ExitPolicyState`` carries per-position runtime state
     (e.g. ``tp1_fired``, ``peak_mfe_since_tp1``).

  3. ``evaluate_intrabar(...)`` is called by the driver each bar
     between the existing intra-bar SL/TP check and the bar-close
     evaluation. Used by policies whose triggers fire intra-bar at a
     specific price level — currently only
     ``sl_partial_close_1r_runner_trail`` (partial close at +1R).

  4. ``evaluate_at_close(...)`` is called by the driver each bar
     after intra-bar evaluation and after the existing trail-manager
     ratchet, before mark-to-market. Used by trailing-style policies
     (``sl_plus_trailing_atr``, ``sl_plus_trailing_swing``) and the
     runner-trail portion of ``sl_partial_close_1r_runner_trail``.

  5. The driver queues any emitted ``ExitPolicyDecision`` with the
     appropriate fill semantics:
       * ``timing == "intrabar"``: fills NOW at ``fill_price`` (long
         partial close at +1R fills at the +1R price level, mirroring
         the existing intra-bar TP fill convention).
       * ``timing == "at_close"``: queues for next-bar open fill
         (long: ``open_bid``; short: ``open_ask``), mirroring the
         existing trail-manager queue-and-fill pattern.

R-frame convention

  ``R_atr = sl_atr_mult × atr_at_entry`` — the size of 1R in price
  units. Policies anchor their TP / trail thresholds in R-units
  (e.g. ``+1R`` is ``entry_price + R_atr`` for a long); the registry
  is therefore SL-multiplier-agnostic. The R-frame matches the
  reference path simulator at [scripts/l_arc_10_v3/step_5.py:99-253][].

Determinism

  Policies are pure: ``apply_to_order`` and ``evaluate_*`` are
  deterministic functions of their inputs. Per-position state is
  isolated per ``ExitPolicyState`` instance; no shared mutable state
  across positions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import pandas as pd

from core.sim.account import Account, Direction, Position


class ExitAction(Enum):
    FULL_CLOSE = "full_close"
    PARTIAL_CLOSE = "partial_close"


@dataclass(frozen=True)
class ExitPolicyContext:
    """Per-position inputs the policy needs at register time.

    All prices are in the pair's quote currency. ``atr_at_entry`` is
    mid-anchored per PR #189 §15.1. ``entry_price`` is the actual
    fill price (open_ask for long, open_bid for short) so policies
    can compute absolute SL/TP/trail levels off the realised entry.
    """

    entry_price: float
    atr_at_entry: float
    sl_atr_mult: float
    direction: Direction

    @property
    def r_atr(self) -> float:
        """One R in price units."""
        return self.sl_atr_mult * self.atr_at_entry


@dataclass(frozen=True)
class ExitPolicyDecision:
    """A policy's "yes, act on this position" verdict.

    ``timing == "intrabar"``: driver acts NOW on the current bar at
    ``fill_price`` (must be set). Mirrors intra-bar TP semantics.

    ``timing == "at_close"``: driver queues a close at next-bar open
    (``fill_price`` ignored — long fills at ``open_bid``, short at
    ``open_ask``). Mirrors trail-manager queue-and-fill semantics.

    ``partial_fraction`` is the fraction of CURRENT size to close
    (must be strictly in (0, 1) for ``PARTIAL_CLOSE``; ignored for
    ``FULL_CLOSE``).
    """

    action: ExitAction
    exit_reason: str
    timing: str  # "intrabar" | "at_close"
    fill_price: float | None = None
    partial_fraction: float = 1.0


class ExitPolicyState(ABC):
    """Per-position runtime state. Subclassed per policy.

    Stateless policies (``sl_only``, the TP-only policies) use the
    trivial :class:`NullPolicyState`.
    """


@dataclass
class NullPolicyState(ExitPolicyState):
    """Trivial state for stateless policies."""


class ExitPolicy(ABC):
    """Abstract base for canonical exit policies.

    Subclasses must set the ``name`` class attribute (registry key)
    and implement :meth:`make_state`. The four lifecycle hooks
    (``apply_to_order``, ``evaluate_intrabar``, ``evaluate_at_close``,
    ``make_state``) have permissive defaults so most policies only
    override the one or two they actually use.
    """

    name: str = ""

    def apply_to_order(self, ctx: ExitPolicyContext) -> Mapping[str, Any]:
        """Return Order-kwargs to merge at construction. Default: none."""
        return {}

    def evaluate_intrabar(
        self,
        position: Position,
        bar: pd.Series,
        state: ExitPolicyState,
        ctx: ExitPolicyContext,
    ) -> ExitPolicyDecision | None:
        """Per-bar intra-bar evaluation. Default: never fire."""
        return None

    def evaluate_at_close(
        self,
        position: Position,
        bar: pd.Series,
        state: ExitPolicyState,
        ctx: ExitPolicyContext,
        account: Account,
    ) -> ExitPolicyDecision | None:
        """Per-bar bar-close evaluation. Default: never fire."""
        return None

    @abstractmethod
    def make_state(self, ctx: ExitPolicyContext) -> ExitPolicyState:
        """Construct fresh per-position state at trade fill."""


__all__ = (
    "ExitAction",
    "ExitPolicy",
    "ExitPolicyContext",
    "ExitPolicyDecision",
    "ExitPolicyState",
    "NullPolicyState",
)
