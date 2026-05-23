"""Chained max DD across the full IS + holdout trajectory at r_base.

Per Amendment 3 §"Chained max DD measurement": "folds concatenated
chronologically into one continuous equity curve, peak-to-trough across
the whole curve."

Two reconstruction modes are available; this PR uses the stitching mode
by default and documents the chat-preferred full-window-sim mode as a
v3.0.2 follow-up:

  1. **Equity stitching** (`stitch_per_fold_oos_equity`, the default).
     Takes per-fold OOS equity series (each from an independent fold
     sim that starts at starting_balance and runs through the fold's
     IS+OOS window) plus the holdout's OOS equity series. Each
     successive fold's series is rescaled multiplicatively so its
     starting point matches the prior fold's ending point. Result is
     ONE continuous-looking equity curve over the full IS+holdout
     window.

     LIMITATION: per chat directive Q6, this is "multiplicative
     chaining" — it assumes fold independence (each fold's return
     stream is treated as a multiplier on prior fold's ending
     balance). The assumption holds approximately under reset-floor
     sizing but is imperfect because fold k+1's IS sim restarts the
     account, so fold k+1's OOS state is not exactly continuous with
     fold k's OOS-end.

  2. **Full-window sim** (TODO v3.0.2 follow-up). Run ONE sim spanning
     the full IS+holdout window per top-K candidate, with no per-fold
     resets. Produces a TRUE continuous-equity curve. Per chat
     directive Q6 this is the gold standard. Deferred to a follow-up
     PR per the combined-engine PR's scope contract (concern: ~20%
     wall-clock overhead per top-K candidate; A3/A4 require
     classifier-selection at fold boundaries which adds complexity).

Both modes feed the same downstream computation:
:func:`compute_chained_max_dd_from_continuous_equity` (peak-to-trough on
the resulting series).
"""

from __future__ import annotations

import pandas as pd


def compute_chained_max_dd_from_continuous_equity(equity: pd.Series) -> float:
    """Peak-to-trough drawdown as positive decimal fraction.

    Equivalent to ``core.runners._fold_stats_helpers.max_drawdown_pct``
    but documented separately at this module path so the Amendment 3
    artefact emission has a single canonical reference.

    Returns 0.0 on empty or all-NaN equity.
    """
    if equity is None or len(equity) == 0:
        return 0.0
    series = equity.dropna()
    if len(series) == 0:
        return 0.0
    cmax = series.cummax()
    dd = (cmax - series) / cmax
    return float(dd.max()) if len(dd) else 0.0


def stitch_per_fold_oos_equity(
    per_fold_equity: list[pd.Series],
    *,
    starting_balance: float,
) -> pd.Series:
    """Stitch per-fold OOS equity curves into one continuous-equity proxy.

    Each ``per_fold_equity[i]`` is the i'th fold's OOS-window equity
    series from an INDEPENDENT sim (each ran with its own account that
    started at ``starting_balance`` and went through the fold's IS+OOS
    window). To produce one continuous-looking curve, each successive
    fold's series is multiplicatively rescaled so its first sample
    equals the prior fold's last sample.

    Returns one ``pd.Series`` covering all timestamps in
    ``per_fold_equity`` concatenated in input order. Empty fold series
    are skipped. Returns an empty Series if all folds are empty.

    Limitation: see the module docstring. This is multiplicative
    chaining; the v3.0.2 full-window-sim follow-up replaces it.
    """
    if not per_fold_equity:
        return pd.Series(dtype=float)
    pieces: list[pd.Series] = []
    running_anchor = float(starting_balance)
    for series in per_fold_equity:
        if series is None or len(series) == 0:
            continue
        clean = series.dropna()
        if len(clean) == 0:
            continue
        # Rescale so this fold's first sample equals running_anchor
        first = float(clean.iloc[0])
        if first <= 0:
            continue
        scale = running_anchor / first
        rescaled = clean * scale
        # Drop overlap with prior piece if any (avoid duplicate timestamps)
        if pieces:
            last_ts = pieces[-1].index[-1]
            rescaled = rescaled.loc[rescaled.index > last_ts]
            if len(rescaled) == 0:
                continue
        pieces.append(rescaled)
        running_anchor = float(rescaled.iloc[-1])
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces).sort_index()


__all__ = (
    "compute_chained_max_dd_from_continuous_equity",
    "stitch_per_fold_oos_equity",
)
