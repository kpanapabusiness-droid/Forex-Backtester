"""Post-hoc path replay for canonical exit policies.

The canonical policies in this package (sl_only, sl_plus_tp_*, sl_plus_trailing_*,
sl_partial_close_1r_runner_trail) define live bar-by-bar behaviour for the
v3 multipair backtester. This module provides the SAME semantics applied
to *recorded* per-trade path data (the fast Step 5 approximation that
walks ``mae_so_far_r / mfe_so_far_r / close_r / is_held`` columns in
R-units relative to a base SL of 2.0×ATR).

Two execution paths, ONE definition of each policy:

  * **Live (bar-by-bar):** ``core/sim/multipair_backtester.py`` +
    ``ExitPolicyManager`` driven by the policy classes' evaluate_intrabar /
    evaluate_at_close hooks. Uses bid/ask worst-case fills.
  * **Replay (this module):** ``simulate_path(name, path_rows, sl_mult)``
    walks recorded path data with the same SL rescaling, threshold, and
    sequencing as the live engine — but in R-units off a single
    mid-anchored price stream (no bid/ask). Fast (Python loop over bars).

Reference origin: scripts/l_arc_10_v3/step_5.py:_apply_exit_policy:99-253
(committed pre-canonical-registry). Per-arc scripts (Arc 10, Arc 8) MUST
import from this module rather than re-implementing — the dispatch's
"hand-rolled per-arc exit logic is creating silent drift" mandate.

Path-data schema (the recorded `bar_path` per trade, written by Step 1
in scripts/l_arc_*/step_1.py):

  * ``bar_offset`` (int): 0-indexed bar offset from entry (entry bar = 0).
  * ``mae_so_far_r`` (float): running-min low (in R-units of base SL=2.0×ATR).
  * ``mfe_so_far_r`` (float): running-max high (R-units).
  * ``close_r`` (float): bar close (R-units relative to entry).
  * ``is_held`` (int 0/1, optional): 1 while position held; 0 after time-exit.

SL rescaling under a new ``sl_mult``:
    scale = 2.0 / sl_mult            (R-frame scale factor)
    sl_threshold_old = -(sl_mult / 2.0)  (SL line in old R-units)
    new_mfe_at = mfe * scale
    new_close_at = close * scale
SL-breach detection scans ``mae`` (old R-units) against ``sl_threshold_old``.

Tolerance contract (per chat answer Q5 — mid-only-parity test):
  * Per-trade R: ±0.01R between replay and reference hand-rolled.
  * Per-fold ROI: ±0.5%.
  * Equity sliding band: ±0.5%.
  * Pool size: byte-identical on the trade-admission axis (entries
    admitted; "trades completed" may differ for partial-close configs).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def _first_at_least(arr: np.ndarray, k: float) -> int:
    """First index where ``arr[i] >= k`` (NaN-safe). -1 if never."""
    mask = np.isfinite(arr) & (arr >= k)
    idx = np.where(mask)[0]
    return int(idx[0]) if idx.size > 0 else -1


def _held_end(is_held: np.ndarray, n: int) -> int:
    """Index of last bar where is_held == 1; falls back to n-1."""
    held_idx = np.where(is_held == 1)[0]
    return int(held_idx[-1]) if held_idx.size > 0 else n - 1


def _prep(
    path_rows: pd.DataFrame, sl_mult: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, float]:
    """Sort + extract arrays + scale factors. Returns:
    bar_offsets, mae, new_mfe_at, new_close_at, is_held, n, sl_breach, scale
    """
    p = path_rows.sort_values("bar_offset")
    bar_offsets = p["bar_offset"].to_numpy(dtype=int)
    mae = p["mae_so_far_r"].to_numpy()
    mfe = p["mfe_so_far_r"].to_numpy()
    close = p["close_r"].to_numpy()
    is_held = (
        p["is_held"].to_numpy()
        if "is_held" in p.columns
        else np.ones(len(p), dtype=int)
    )

    scale = 2.0 / sl_mult
    sl_threshold_old = -(sl_mult / 2.0)
    new_mfe_at = mfe * scale
    new_close_at = close * scale

    sl_breach = -1
    for i, m in enumerate(mae):
        if np.isfinite(m) and m <= sl_threshold_old:
            sl_breach = i
            break

    n = len(p)
    return bar_offsets, mae, new_mfe_at, new_close_at, is_held, n, sl_breach, scale


# ────────────────────────────────────────────────────────────────────────
# Per-policy simulators (one-to-one with the canonical registry names)
# ────────────────────────────────────────────────────────────────────────


def simulate_sl_only(
    trade_row: pd.Series, path_rows: pd.DataFrame, sl_mult: float
) -> tuple[float, int]:
    if path_rows.empty:
        return (
            float(trade_row.get("final_r", 0.0)) * (2.0 / sl_mult),
            int(trade_row.get("bars_held", 0)),
        )
    bar_offsets, _, _, new_close_at, is_held, n, sl_breach, _ = _prep(path_rows, sl_mult)
    if sl_breach >= 0:
        return -1.0, int(bar_offsets[sl_breach] + 1)
    end_i = _held_end(is_held, n)
    return float(new_close_at[end_i]), int(bar_offsets[end_i] + 1)


def _simulate_sl_plus_tp_at_r(
    trade_row: pd.Series, path_rows: pd.DataFrame, sl_mult: float, tp_r: float
) -> tuple[float, int]:
    if path_rows.empty:
        return (
            float(trade_row.get("final_r", 0.0)) * (2.0 / sl_mult),
            int(trade_row.get("bars_held", 0)),
        )
    bar_offsets, _, new_mfe_at, new_close_at, is_held, n, sl_breach, _ = _prep(
        path_rows, sl_mult
    )
    tp_i = _first_at_least(new_mfe_at, tp_r)
    if tp_i >= 0 and (sl_breach < 0 or tp_i <= sl_breach):
        return tp_r, int(bar_offsets[tp_i] + 1)
    if sl_breach >= 0:
        return -1.0, int(bar_offsets[sl_breach] + 1)
    end_i = _held_end(is_held, n)
    return float(new_close_at[end_i]), int(bar_offsets[end_i] + 1)


def simulate_sl_plus_tp_2r(
    trade_row: pd.Series, path_rows: pd.DataFrame, sl_mult: float
) -> tuple[float, int]:
    return _simulate_sl_plus_tp_at_r(trade_row, path_rows, sl_mult, 2.0)


def simulate_sl_plus_tp_3r(
    trade_row: pd.Series, path_rows: pd.DataFrame, sl_mult: float
) -> tuple[float, int]:
    return _simulate_sl_plus_tp_at_r(trade_row, path_rows, sl_mult, 3.0)


def simulate_sl_plus_trailing_atr(
    trade_row: pd.Series, path_rows: pd.DataFrame, sl_mult: float
) -> tuple[float, int]:
    """Trail at 1R below peak MFE; activate at MFE >= 1R. SL preempts when
    sl_breach <= trail_exit_i.

    Reference: scripts/l_arc_10_v3/step_5.py:173-193.
    """
    if path_rows.empty:
        return (
            float(trade_row.get("final_r", 0.0)) * (2.0 / sl_mult),
            int(trade_row.get("bars_held", 0)),
        )
    bar_offsets, _, new_mfe_at, new_close_at, is_held, n, sl_breach, _ = _prep(
        path_rows, sl_mult
    )
    trail_r = -1.0
    active = False
    trail_exit_i = -1
    for i in range(n):
        if not np.isfinite(new_mfe_at[i]):
            continue
        if new_mfe_at[i] >= 1.0:
            active = True
            trail_r = max(trail_r, new_mfe_at[i] - 1.0)
        if active and np.isfinite(new_close_at[i]) and new_close_at[i] <= trail_r:
            trail_exit_i = i
            break
    if sl_breach >= 0 and (trail_exit_i < 0 or sl_breach <= trail_exit_i):
        return -1.0, int(bar_offsets[sl_breach] + 1)
    if trail_exit_i >= 0:
        return float(new_close_at[trail_exit_i]), int(bar_offsets[trail_exit_i] + 1)
    end_i = _held_end(is_held, n)
    return float(new_close_at[end_i]), int(bar_offsets[end_i] + 1)


def simulate_sl_plus_trailing_swing(
    trade_row: pd.Series, path_rows: pd.DataFrame, sl_mult: float
) -> tuple[float, int]:
    """Trail at running max(min(prev_close, 0)). Activate at MFE >= 1R. SL
    preempts when sl_breach <= trail_exit_i.

    Reference: scripts/l_arc_10_v3/step_5.py:195-217.
    """
    if path_rows.empty:
        return (
            float(trade_row.get("final_r", 0.0)) * (2.0 / sl_mult),
            int(trade_row.get("bars_held", 0)),
        )
    bar_offsets, _, new_mfe_at, new_close_at, is_held, n, sl_breach, _ = _prep(
        path_rows, sl_mult
    )
    active = False
    prev_low = -np.inf
    trail_exit_i = -1
    for i in range(n):
        if not np.isfinite(new_mfe_at[i]):
            continue
        if new_mfe_at[i] >= 1.0:
            active = True
        if active:
            if i > 0 and np.isfinite(new_close_at[i - 1]):
                prev_low = max(prev_low, min(new_close_at[i - 1], 0.0))
            if np.isfinite(new_close_at[i]) and new_close_at[i] <= prev_low:
                trail_exit_i = i
                break
    if sl_breach >= 0 and (trail_exit_i < 0 or sl_breach <= trail_exit_i):
        return -1.0, int(bar_offsets[sl_breach] + 1)
    if trail_exit_i >= 0:
        return float(new_close_at[trail_exit_i]), int(bar_offsets[trail_exit_i] + 1)
    end_i = _held_end(is_held, n)
    return float(new_close_at[end_i]), int(bar_offsets[end_i] + 1)


def simulate_sl_partial_close_1r_runner_trail(
    trade_row: pd.Series, path_rows: pd.DataFrame, sl_mult: float
) -> tuple[float, int]:
    """Close 50% at +1R, trail rest at 1R below path-peak MFE.

    Reference: scripts/l_arc_10_v3/step_5.py:219-250 (Arc 10 PASS-DEPLOYABLE
    closure's load-bearing exit).
    """
    if path_rows.empty:
        return (
            float(trade_row.get("final_r", 0.0)) * (2.0 / sl_mult),
            int(trade_row.get("bars_held", 0)),
        )
    bar_offsets, _, new_mfe_at, new_close_at, is_held, n, sl_breach, _ = _prep(
        path_rows, sl_mult
    )
    tp1_i = _first_at_least(new_mfe_at, 1.0)
    if tp1_i < 0:
        if sl_breach >= 0:
            return -1.0, int(bar_offsets[sl_breach] + 1)
        end_i = _held_end(is_held, n)
        return float(new_close_at[end_i]), int(bar_offsets[end_i] + 1)
    half_r = 1.0
    trail_r = 0.0
    trail_exit_i = -1
    for i in range(tp1_i, n):
        if np.isfinite(new_mfe_at[i]):
            trail_r = max(trail_r, new_mfe_at[i] - 1.0)
        if (
            np.isfinite(new_close_at[i])
            and new_close_at[i] <= trail_r
            and i > tp1_i
        ):
            trail_exit_i = i
            break
    if (
        sl_breach >= 0
        and sl_breach > tp1_i
        and (trail_exit_i < 0 or sl_breach <= trail_exit_i)
    ):
        runner_r = -1.0
    elif trail_exit_i >= 0:
        runner_r = float(new_close_at[trail_exit_i])
    else:
        end_i = _held_end(is_held, n)
        runner_r = float(new_close_at[end_i])
    final_r = 0.5 * half_r + 0.5 * runner_r
    if trail_exit_i >= 0:
        last_bar = trail_exit_i
    elif sl_breach >= 0:
        last_bar = sl_breach
    else:
        last_bar = n - 1
    return float(final_r), int(bar_offsets[last_bar] + 1)


# ────────────────────────────────────────────────────────────────────────
# Dispatcher
# ────────────────────────────────────────────────────────────────────────


_SIMULATORS: dict[str, Callable[[pd.Series, pd.DataFrame, float], tuple[float, int]]] = {
    "sl_only": simulate_sl_only,
    "sl_plus_tp_2r": simulate_sl_plus_tp_2r,
    "sl_plus_tp_3r": simulate_sl_plus_tp_3r,
    "sl_plus_trailing_atr": simulate_sl_plus_trailing_atr,
    "sl_plus_trailing_swing": simulate_sl_plus_trailing_swing,
    "sl_partial_close_1r_runner_trail": simulate_sl_partial_close_1r_runner_trail,
}


def simulate_path(
    policy_name: str,
    trade_row: pd.Series,
    path_rows: pd.DataFrame,
    sl_mult: float,
) -> tuple[float, int]:
    """Replay a canonical exit policy over recorded path data.

    Dispatches to the per-policy simulator; falls back to ``sl_only``
    for unknown names (matches the legacy hand-rolled fallthrough at
    scripts/l_arc_10_v3/step_5.py:252-253).
    """
    fn = _SIMULATORS.get(policy_name, simulate_sl_only)
    return fn(trade_row, path_rows, sl_mult)


def available_path_simulators() -> tuple[str, ...]:
    """Sorted tuple of policies with a registered path simulator."""
    return tuple(sorted(_SIMULATORS))


__all__ = (
    "simulate_path",
    "simulate_sl_only",
    "simulate_sl_plus_tp_2r",
    "simulate_sl_plus_tp_3r",
    "simulate_sl_plus_trailing_atr",
    "simulate_sl_plus_trailing_swing",
    "simulate_sl_partial_close_1r_runner_trail",
    "available_path_simulators",
    "simulate_pool_approximation",
)


# ────────────────────────────────────────────────────────────────────────
# Pool-level approximation (legacy Arc 8 fast path)
# ────────────────────────────────────────────────────────────────────────


def simulate_pool_approximation(
    final_r_sl: float,
    mfe_r_sl: float,
    mae_r_sl: float,
    policy_name: str,
) -> float:
    """Pool-level approximation: cap final_r when mfe crossed the TP threshold.

    A coarser approximation than :func:`simulate_path` — operates on
    already-rescaled per-trade scalars rather than the full recorded path.
    Used historically by Arc 8 [scripts/l_arc_8/run_step5_wfo.py:74-88][]
    where bar-by-bar path replay was out of compute budget.

    Assumes "TP fires before any pull-back to SL when mfe >= threshold" —
    strictly optimistic vs. the path-based simulator. Both Arc 8 closure
    §6 and this docstring flag the approximation. Higher-fidelity arcs
    should prefer :func:`simulate_path`.

    Supported policies:
      * ``sl_only``           — pass-through
      * ``sl_plus_tp_2r``     — cap at +2R if mfe >= 2
      * ``sl_plus_tp_3r``     — cap at +3R if mfe >= 3

    Unknown policy names fall through to pass-through (matches legacy
    Arc 8 behaviour).
    """
    if policy_name == "sl_only":
        return float(final_r_sl)
    if policy_name == "sl_plus_tp_2r":
        return 2.0 if mfe_r_sl >= 2.0 else float(final_r_sl)
    if policy_name == "sl_plus_tp_3r":
        return 3.0 if mfe_r_sl >= 3.0 else float(final_r_sl)
    return float(final_r_sl)
