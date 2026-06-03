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

from core.sim.costs.model import CostModel, apply_cost_model
from core.sim.multipair_backtester import RunResult
from core.time_utils.session_boundary import (
    SUPPORTED_CONVENTIONS,
    utc_to_eet_trading_day,
)
from core.wfo.folds import Fold
from core.wfo.gates import FoldStats

# Per Amendment 6 (supersedes Amendment 3 §"Boundary"): daily-DD
# measurement uses the EET broker trading day post-PR-189. KH-24 and
# legacy callers may opt back to UTC via boundary_convention="utc".
_EET_TZ: str = "Europe/Athens"

# FundedNext is the gate default cost profile (1.5x spread, $5/lot RT, 0.5
# pip/fill slippage, swaps off) — the higher cost, as the conservative bound.
# This is the chokepoint that resolves HONEST_ENGINE_SWEEP.md Part C: every
# FoldStats-producing gate path nets broker costs by default. A cost-free
# FoldStats requires an EXPLICIT CostModel.zero() — never the silent default.
_DEFAULT_FUNDEDNEXT: CostModel = CostModel.fundednext()


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
    cost_model: CostModel = _DEFAULT_FUNDEDNEXT,
) -> FoldStats:
    """Convert a RunResult into a FoldStats restricted to fold's OOS window.

    Trade count uses OOS-only entries; ROI uses OOS-restricted equity;
    DD uses OOS-restricted equity; daily-breach count uses OOS days.

    Broker costs (``cost_model``, FundedNext by default) are netted at this
    gate-scoring layer: the engine output is gross, and ``apply_cost_model``
    rebuilds a net equity curve by debiting each closed position's cost at its
    final exit bar (so net ROI / DD / daily-breach are all faithful). Pass
    ``CostModel.zero()`` for an EXPLICIT cost-free FoldStats — there is no
    silent cost-free default.
    """
    costed = apply_cost_model(run_result, cost_model)
    equity = slice_equity_to_oos(costed.net_equity, fold)
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
    boundary_convention: str = "5ers_eet",
) -> pd.DataFrame:
    """Per-day max-DD series at r_base — Amendment 6 §"Daily DD measurement".

    For each trading day in ``equity.index``, computes:

      - ``date`` (broker-local trading-day calendar date, ``datetime.date``;
        EET-local under ``"5ers_eet"``, UTC under ``"utc"``)
      - ``pair_set`` (label, useful for multi-arc registry rows)
      - ``day_start_equity`` (first equity sample of that day —
        the day's open reference per Amendment 6 §"Day-start equity
        definition"; NOT the reset-floor sizing baseline)
      - ``day_max_dd_base_pct`` (``(day_start_equity - day_min_equity)
        / day_start_equity`` as decimal fraction; 0.05 = 5%)
      - ``n_trades_open_start_of_day`` (placeholder 0 — caller can
        post-fill from account state if needed; not load-bearing for
        the gate logic)

    Per Amendment 6 (supersedes Amendment 3 §"Boundary"): the boundary
    is the EET broker trading day, matching 5ers' actual daily-DD reset
    boundary. ``boundary_convention="utc"`` is retained for KH-24
    anchor compatibility (byte-identical to pre-Amendment-6 output).

    The verdict logic in
    ``core.wfo.amended_gates.count_daily_breaches_at_scaled_risk``
    multiplies each row's ``day_max_dd_base_pct`` by ``k`` and counts
    days at-or-above the 5% breach threshold.
    """
    if boundary_convention not in SUPPORTED_CONVENTIONS:
        raise ValueError(
            f"Unsupported boundary_convention {boundary_convention!r}; "
            f"expected one of {SUPPORTED_CONVENTIONS}"
        )

    empty_cols = [
        "date", "pair_set", "day_start_equity",
        "day_max_dd_base_pct", "n_trades_open_start_of_day",
    ]
    if equity is None or len(equity) == 0:
        return pd.DataFrame(columns=empty_cols)
    s = equity.dropna()
    if len(s) == 0:
        return pd.DataFrame(columns=empty_cols)

    df = s.to_frame(name="equity")

    # Trading-day-start key per row (tz-aware UTC timestamp), then map
    # to a broker-local datetime.date label so audit reports read in
    # the same timezone as the boundary.
    if hasattr(df.index, "tz_convert"):
        idx_utc = df.index.tz_convert("UTC")
    else:
        idx_utc = df.index
    day_keys = utc_to_eet_trading_day(idx_utc, convention=boundary_convention)
    if boundary_convention == "utc":
        df["date"] = day_keys.date
    else:
        # day_keys are at EET midnight expressed in UTC. Convert to EET
        # local so the date label reads as the EET calendar day.
        df["date"] = day_keys.tz_convert(_EET_TZ).date

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

    return by_day[empty_cols].reset_index(drop=True)


__all__ = (
    "slice_equity_to_oos",
    "max_drawdown_pct",
    "count_daily_5pct_breaches",
    "compute_per_day_max_dd",
    "filter_oos_trades",
    "build_fold_stats_from_run",
)
