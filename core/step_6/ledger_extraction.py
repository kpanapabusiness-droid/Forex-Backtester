"""Extract the Top-1 closed-trade ledger + fold assignments for Step 6.

Bridges the orchestrator's per-(config_id, fold_id) StrategyResult side-
channel and the holdout result tuple to the flat (trade_ledger,
fold_assignments) shape the §6.3 spread P&L decomposition diagnostic
consumes.

Public surface:

  * :func:`build_top_1_trade_ledger` — flatten ``ClosedTrade`` objects
    into a DataFrame with the extended bid+ask schema preserved.
  * :func:`build_top_1_fold_assignments` — emit ``(leg_id, fold_id)``
    mapping aligned to the row order of the ledger.
  * :func:`extract_top_1_ledger_bundle` — convenience that returns
    ``(trade_ledger, fold_assignments, holdout_fold_id)`` for the Top-1
    amended candidate.

Returns ``(None, None, None)`` cleanly when the inputs are insufficient
(no Top-1, no StrategyResults for it, etc.). Callers should treat None
as "diagnostic skips gracefully" — never raise.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

import pandas as pd


def _closed_trade_to_row(trade: Any) -> dict[str, Any]:
    """Flatten a :class:`core.sim.account.ClosedTrade` to a row dict.

    Avoids importing the engine's account module so this helper can be
    invoked from any context. Uses dataclass introspection when possible
    and falls back to ``vars`` / attribute scraping.
    """
    if is_dataclass(trade):
        row = asdict(trade)
    else:
        row = {k: getattr(trade, k) for k in dir(trade) if not k.startswith("_")}
    # Direction is an Enum on ClosedTrade; serialise its name (sign) for
    # ledger consumers that expect a plain string.
    direction = row.get("direction")
    if direction is not None and hasattr(direction, "name"):
        row["direction"] = direction.name.lower()
    elif direction is not None and hasattr(direction, "value"):
        row["direction"] = str(direction.value)
    return row


def build_top_1_trade_ledger(
    top_1_config_id: str,
    strategy_results: Mapping[str, Mapping[int, Any]],
    holdout_results: tuple,
) -> pd.DataFrame | None:
    """Flatten Top-1 ``ClosedTrade`` records from every fold + holdout.

    ``strategy_results`` is the orchestrator's
    ``_last_strategy_results`` map: ``{config_id: {fold_id: StrategyResult}}``.
    ``holdout_results`` is the orchestrator's ``holdout`` tuple — each
    entry must have a matching ``config_id`` and carry the holdout
    StrategyResult (only the FoldStats survives the public API today;
    full holdout closed_trades require a side-channel — see
    :func:`extract_top_1_ledger_bundle`).

    Returns ``None`` when no per-fold StrategyResults exist for the
    Top-1 candidate.
    """
    fold_results = strategy_results.get(top_1_config_id) if strategy_results else None
    if not fold_results:
        return None

    rows: list[dict[str, Any]] = []
    for fold_id, sr in fold_results.items():
        closed = getattr(sr, "closed_trades", None) or ()
        for t in closed:
            rows.append(_closed_trade_to_row(t))

    # Holdout — appended if discoverable via the matching config_id.
    for h in holdout_results or ():
        if getattr(h, "config_id", None) != top_1_config_id:
            continue
        # `run_holdout` returns FoldStats wrappers today, not
        # StrategyResults — closed_trades may live on a side-channel
        # attribute. Both shapes accommodated.
        for attr in ("closed_trades", "trades"):
            closed = getattr(h, attr, None)
            if closed:
                for t in closed:
                    rows.append(_closed_trade_to_row(t))
                break

    if not rows:
        return None

    return pd.DataFrame(rows).reset_index(drop=True)


def build_top_1_fold_assignments(
    top_1_config_id: str,
    strategy_results: Mapping[str, Mapping[int, Any]],
    holdout_results: tuple,
    holdout_fold_id: int | None,
) -> pd.DataFrame | None:
    """Emit (leg_id, fold_id) mapping aligned to ledger row order.

    Row order must match :func:`build_top_1_trade_ledger`: per-fold trades
    in fold_id order, then holdout trades (assigned ``holdout_fold_id``).
    """
    fold_results = strategy_results.get(top_1_config_id) if strategy_results else None
    if not fold_results:
        return None

    rows: list[dict[str, Any]] = []
    leg = 0
    for fold_id, sr in fold_results.items():
        for _ in (getattr(sr, "closed_trades", None) or ()):
            rows.append({"leg_id": leg, "fold_id": int(fold_id)})
            leg += 1

    if holdout_fold_id is not None:
        for h in holdout_results or ():
            if getattr(h, "config_id", None) != top_1_config_id:
                continue
            for attr in ("closed_trades", "trades"):
                closed = getattr(h, attr, None)
                if closed:
                    for _ in closed:
                        rows.append({"leg_id": leg, "fold_id": int(holdout_fold_id)})
                        leg += 1
                    break

    if not rows:
        return None
    return pd.DataFrame(rows)


def extract_top_1_ledger_bundle(
    top_1_config_id: str | None,
    strategy_results: Mapping[str, Mapping[int, Any]] | None,
    holdout_results: tuple,
    *,
    holdout_fold_id: int | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, int | None]:
    """Return (trade_ledger, fold_assignments, holdout_fold_id) for Top-1.

    Returns ``(None, None, None)`` when the bundle cannot be reconstructed
    (no top_1 yet, orchestrator did not record strategy results, etc.).
    The caller treats this as "diagnostic skips gracefully".
    """
    if top_1_config_id is None or not strategy_results:
        return None, None, holdout_fold_id
    ledger = build_top_1_trade_ledger(
        top_1_config_id, strategy_results, holdout_results,
    )
    if ledger is None:
        return None, None, holdout_fold_id
    fa = build_top_1_fold_assignments(
        top_1_config_id, strategy_results, holdout_results, holdout_fold_id,
    )
    return ledger, fa, holdout_fold_id


__all__ = (
    "build_top_1_fold_assignments",
    "build_top_1_trade_ledger",
    "extract_top_1_ledger_bundle",
)
