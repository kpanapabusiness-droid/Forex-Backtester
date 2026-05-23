"""FoldStats construction helpers shared across architectures.

The same conversion logic — equity curve restricted to the OOS window,
peak-to-trough max DD, 5%-day breach count, period ROI, ratio — lives
in :mod:`core.wfo.fold_runner` already. This module exposes the
internal helpers as a stable API for architecture-level callers, so
KH24FoldRunner can be refactored to use them without circular imports.

L_PROTOCOL §3 gate inputs (FoldStats fields):
  fold_id, n_trades, roi_pct, max_dd_pct, days_breaching_daily_5pct,
  roi_dd_ratio
"""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from core.sim.multipair_backtester import RunResult
from core.wfo.folds import Fold
from core.wfo.gates import FoldStats


def slice_equity_to_oos(equity: pd.Series, fold: Fold) -> pd.Series:
    """Restrict ``equity`` to the fold's OOS window inclusive."""
    if len(equity) == 0:
        return equity
    oos_start_ts = pd.Timestamp(fold.oos_start, tz="UTC")
    oos_end_ts = (
        pd.Timestamp(fold.oos_end, tz="UTC")
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    )
    return equity.loc[oos_start_ts:oos_end_ts]


def max_drawdown_pct(equity: pd.Series) -> float:
    """Peak-to-trough drawdown as a positive percent."""
    if len(equity) == 0:
        return 0.0
    cmax = equity.cummax()
    dd = (cmax - equity) / cmax
    return float(dd.max()) if len(dd) else 0.0


def count_daily_5pct_breaches(equity: pd.Series) -> int:
    """5ers 5% daily-DD breaches: any UTC day where equity dropped > 5%
    from the day's opening equity."""
    if len(equity) == 0:
        return 0
    daily = equity.resample("1D").agg(["first", "min"]).dropna()
    if len(daily) == 0:
        return 0
    return int(((daily["first"] - daily["min"]) / daily["first"] > 0.05).sum())


def filter_oos_trades(closed_trades: Iterable, fold: Fold) -> tuple:
    """Restrict closed trades to those entered within the OOS window."""
    oos_start_ts = pd.Timestamp(fold.oos_start, tz="UTC")
    oos_end_ts = (
        pd.Timestamp(fold.oos_end, tz="UTC")
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    )
    return tuple(
        t for t in closed_trades
        if oos_start_ts <= pd.Timestamp(t.entry_time) <= oos_end_ts
    )


def build_fold_stats_from_run(
    *,
    fold: Fold,
    run_result: RunResult,
    starting_balance: float,
) -> FoldStats:
    """Convert a RunResult into a FoldStats restricted to fold's OOS window.

    Trade count uses OOS-only entries; ROI uses OOS-restricted equity;
    DD uses OOS-restricted equity; daily-breach count uses OOS days.
    """
    equity = slice_equity_to_oos(run_result.equity_curve, fold)
    if len(equity) == 0:
        return FoldStats(
            fold_id=fold.fold_id,
            n_trades=0,
            roi_pct=0.0,
            max_dd_pct=0.0,
            days_breaching_daily_5pct=0,
            roi_dd_ratio=0.0,
        )
    period_roi = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    dd = max_drawdown_pct(equity)
    breaches = count_daily_5pct_breaches(equity)
    if dd > 0:
        ratio = period_roi / dd
    else:
        ratio = float("inf") if period_roi > 0 else 0.0
    if math.isinf(ratio):
        ratio = 999.0
    oos_trades = filter_oos_trades(run_result.closed_trades, fold)
    return FoldStats(
        fold_id=fold.fold_id,
        n_trades=len(oos_trades),
        roi_pct=period_roi,
        max_dd_pct=dd,
        days_breaching_daily_5pct=breaches,
        roi_dd_ratio=ratio,
    )


def compute_per_day_max_dd(
    equity: pd.Series,
    *,
    pair_set: str = "unknown",
) -> pd.DataFrame:
    """Per-day max-DD series at r_base — Amendment 3 §"Daily DD measurement".

    For each UTC trading day in ``equity.index``, computes:

      - ``date`` (UTC day, datetime.date)
      - ``pair_set`` (label, useful for multi-arc registry rows)
      - ``day_start_equity`` (first equity sample of that day —
        the day's 00:00-UTC reference per Amendment 3 §"Day-start
        equity definition"; NOT the reset-floor sizing baseline)
      - ``day_max_dd_base_pct`` (``(day_start_equity - day_min_equity)
        / day_start_equity`` as decimal fraction; 0.05 = 5%)
      - ``n_trades_open_start_of_day`` (placeholder 0 — caller can
        post-fill from account state if needed; not load-bearing for
        the gate logic)

    Per Amendment 3 §"Day-start equity definition": this is the
    REFERENCE for daily DD scaling. The verdict logic in
    ``core.wfo.amended_gates.count_daily_breaches_at_scaled_risk``
    multiplies each row's ``day_max_dd_base_pct`` by ``k`` and counts
    days at-or-above the 5% breach threshold.

    Boundary: UTC broker-day (locked per Amendment 3 §"Boundary").
    """
    if equity is None or len(equity) == 0:
        return pd.DataFrame(columns=[
            "date", "pair_set", "day_start_equity",
            "day_max_dd_base_pct", "n_trades_open_start_of_day",
        ])
    s = equity.dropna()
    if len(s) == 0:
        return pd.DataFrame(columns=[
            "date", "pair_set", "day_start_equity",
            "day_max_dd_base_pct", "n_trades_open_start_of_day",
        ])

    # Group by UTC calendar day. Use .first() / .min() to pick the
    # day's opening equity + the intra-day low.
    df = s.to_frame(name="equity")
    df["date"] = df.index.tz_convert("UTC").date if hasattr(df.index, "tz_convert") else df.index.date

    by_day = (
        df.groupby("date")["equity"]
        .agg(day_start_equity="first", day_min_equity="min")
        .reset_index()
    )
    # DD as positive decimal fraction; clamp negative to 0 (shouldn't
    # happen but defensive).
    by_day["day_max_dd_base_pct"] = (
        (by_day["day_start_equity"] - by_day["day_min_equity"])
        / by_day["day_start_equity"]
    ).clip(lower=0.0)
    by_day["pair_set"] = pair_set
    by_day["n_trades_open_start_of_day"] = 0  # placeholder; see docstring

    return by_day[[
        "date", "pair_set", "day_start_equity",
        "day_max_dd_base_pct", "n_trades_open_start_of_day",
    ]].reset_index(drop=True)


__all__ = (
    "slice_equity_to_oos",
    "max_drawdown_pct",
    "count_daily_5pct_breaches",
    "compute_per_day_max_dd",
    "filter_oos_trades",
    "build_fold_stats_from_run",
)
