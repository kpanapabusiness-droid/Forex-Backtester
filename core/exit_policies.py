"""Pipeline D1 archetype exit policies — PR 2 of 2.

When :class:`core.d1_pipeline.D1Hook` accepts a trade at bar offset ``t``
the winning archetype's :class:`ExitPolicy` is installed on the trade and
takes over from the engine's standard exit cascade:

* :meth:`ExitPolicy.apply_at_accept` fires once at bar ``t`` — typically
  replaces the pre-t 2.0 × ATR(14) stop with the archetype-specific stop.
* :meth:`ExitPolicy.update_per_bar` fires every subsequent bar — ratchets
  the stop in response to path progress (MFE-lock, trail, etc.).

Only the policies named in L_ARC_PROTOCOL v2.0 §11 with a concrete
consumer are implemented. PR 2 ships row 2 (Stepwise climber) for Arc 4
cluster 1. Other rows are deferred until they have a concrete consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class ExitPolicy:
    """Abstract base. Subclasses implement archetype-specific exit logic.

    Both methods mutate ``trade`` in place — the engine reads
    ``trade["sl_px"]`` after each call. ``apply_at_accept`` is invoked
    once, on the bar at which :class:`~core.d1_pipeline.D1Hook` returned
    :class:`~core.d1_pipeline.ApplyPolicy`. ``update_per_bar`` is invoked
    on every subsequent bar of the trade.

    Policies are stateless apart from what they write into ``trade`` —
    instances are shared across trades, so any per-trade bookkeeping
    must live on the trade dict.
    """

    def apply_at_accept(
        self,
        trade: dict[str, Any],
        entry_px: float,
        atr_at_entry: float,
    ) -> None:
        raise NotImplementedError

    def update_per_bar(
        self,
        trade: dict[str, Any],
        bar_path_row: Mapping[str, float],
        entry_px: float,
        atr_at_entry: float,
    ) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class StepwiseClimberPolicy(ExitPolicy):
    """§11 row 2 — MFE-lock + trail-from-peak.

    Semantics (long-side; short-side mirrors with sign flips):

    * **Pre-t SL** (engine-owned, before this policy is installed): entry
      − 2.0 × ATR(14).
    * **apply_at_accept** (bar ``t``): replace SL with
      entry − ``archetype_sl_r`` × ``r_in_atr`` × ATR. For Arc 4 cluster 1
      this is a *loosening* (entry − 3.0 × ATR).
    * **MFE-lock** (one-shot, fires the first bar at which
      ``mfe_so_far_r >= mfe_lock_r``): records the fire bar in
      ``trade["mfe_lock_fired_bar"]``. Once fired, trail is armed AND
      used in the same-bar SL check (matches the Step 5 simulator's
      Step 4 ``effective_sl = max(current_sl, trail_stop)`` semantics —
      see ``scripts/l_arc_4/step5_stability.py:269-318``).
    * **Effective SL post-lock**: max(current_sl, entry_px, trail_stop)
      for longs (min for shorts), where
      ``trail_stop = entry + (mfe_so_far_r − trail_from_high_r) ×
      r_in_atr × ATR``. At the lock-fire bar with MFE = 1R this yields
      entry + 0.25R; subsequent new MFE highs ratchet upward only.

    Subsequent drawdown in close_r does not reverse the lock (because
    mfe_so_far_r is a running max). Engine ordering is MFE-first: the
    bar_path append updates mfe_so_far_r from bar high before the hook
    fires, so the lock can detect the same bar's high and the trail
    same-bar SL is checked against bar low in the Priority-1 SL test.

    Fields
    ------
    archetype_sl_r : R-frame multiple for the post-t SL. Cluster 1 uses 1.0
        (i.e. SL = entry − 1R = entry − 3 × ATR with ``r_in_atr = 3.0``).
    r_in_atr : the R-unit in ATR multiples. 1R = ``r_in_atr`` × ATR(14).
    mfe_lock_r : MFE level (in R) at which the BE-lock fires. Cluster 1: 1.0.
    trail_from_high_r : trail distance from peak MFE (in R). Cluster 1: 0.75.

    Direction
    ---------
    The policy reads ``trade.get("signal_direction", "long")`` to handle
    short trades. Arc 4 is long-only, so the short path is exercised by
    tests only — the engine never calls into it under current configs.
    """

    archetype_sl_r: float
    r_in_atr: float
    mfe_lock_r: float
    trail_from_high_r: float

    # ------------------------------------------------------------------
    def apply_at_accept(
        self,
        trade: dict[str, Any],
        entry_px: float,
        atr_at_entry: float,
    ) -> None:
        sl_offset = float(self.archetype_sl_r) * float(self.r_in_atr) * float(atr_at_entry)
        if _is_long(trade):
            trade["sl_px"] = float(entry_px) - sl_offset
        else:
            trade["sl_px"] = float(entry_px) + sl_offset
        trade["mfe_lock_fired_bar"] = None
        trade["trail_active_from_bar"] = int(trade.get("bar_path", [{}])[-1].get("bar_offset", 0)) \
            if trade.get("bar_path") else 0

    # ------------------------------------------------------------------
    def update_per_bar(
        self,
        trade: dict[str, Any],
        bar_path_row: Mapping[str, float],
        entry_px: float,
        atr_at_entry: float,
    ) -> None:
        mfe_r = float(bar_path_row["mfe_so_far_r"])
        bar_offset = int(bar_path_row.get("bar_offset", -1))
        r_size_px = float(self.r_in_atr) * float(atr_at_entry)
        long = _is_long(trade)
        current_sl = float(trade["sl_px"])

        # Lock check: fires the first bar at which MFE reaches mfe_lock_r.
        if trade.get("mfe_lock_fired_bar") is None and mfe_r >= float(self.mfe_lock_r):
            trade["mfe_lock_fired_bar"] = bar_offset

        # Effective SL post-lock = max(static SL, BE floor, trail_stop)
        # for longs (min for shorts). Trail fires on the lock-fire bar
        # itself — same-bar arm + use, matching Step 5's effective_sl
        # semantics (scripts/l_arc_4/step5_stability.py:294-298).
        if trade.get("mfe_lock_fired_bar") is not None:
            trail_offset_px = (mfe_r - float(self.trail_from_high_r)) * r_size_px
            if long:
                trail_stop = float(entry_px) + trail_offset_px
                current_sl = max(current_sl, float(entry_px), trail_stop)
            else:
                trail_stop = float(entry_px) - trail_offset_px
                current_sl = min(current_sl, float(entry_px), trail_stop)

        trade["sl_px"] = float(current_sl)


def _is_long(trade: Mapping[str, Any]) -> bool:
    """Default-long unless the trade dict explicitly says ``signal_direction='short'``."""
    return str(trade.get("signal_direction", "long")).lower() != "short"


def build_exit_policy(spec: Mapping[str, Any]) -> ExitPolicy:
    """Factory — instantiates the policy named by ``spec["type"]``.

    Fails loud on unknown types so a typo in YAML doesn't silently fall
    through to the legacy cascade.
    """
    kind = str(spec.get("type", "")).strip()
    if kind == "stepwise_climber":
        return StepwiseClimberPolicy(
            archetype_sl_r=float(spec["archetype_sl_r"]),
            r_in_atr=float(spec["r_in_atr"]),
            mfe_lock_r=float(spec["mfe_lock_r"]),
            trail_from_high_r=float(spec["trail_from_high_r"]),
        )
    if not kind:
        raise ValueError("exit_policy spec is missing 'type'")
    raise ValueError(
        f"unknown exit_policy type {kind!r} — PR 2 implements only "
        "'stepwise_climber' (Arc 4 cluster 1). Other §11 rows are "
        "deferred until they have a concrete consumer."
    )


__all__ = (
    "ExitPolicy",
    "StepwiseClimberPolicy",
    "build_exit_policy",
)
