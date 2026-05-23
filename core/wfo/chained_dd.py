"""Chained max DD across the full IS + holdout trajectory at r_base.

Per Amendment 3 §"Chained max DD measurement": "folds concatenated
chronologically into one continuous equity curve, peak-to-trough across
the whole curve."

Per chat directive Q6: use a FULL-WINDOW continuous sim per top-K
candidate (not multiplicative chaining). The continuous sim is run by
the orchestrator; this module's job is just the DD computation on the
resulting equity series.

Multiplicative chaining (the alternative considered) assumes fold
independence which doesn't hold under reset-floor sizing — sequencing
of fold P&L matters because the starting capital of each fold is
slightly different from prior fold's ending capital. The continuous
sim captures sequencing correctly.
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


__all__ = ("compute_chained_max_dd_from_continuous_equity",)
