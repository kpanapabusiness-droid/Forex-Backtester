"""KH-24 WFO fold runner — RETAINED for back-compatibility with
:mod:`scripts.anchor.run_anchor` and the existing anchor reproduction
path.

CC_07 (L_PROTOCOL v3.0 runtime) prefers the generic
:class:`core.runners.arc_fold_runner.ArcFoldRunner` for new arcs;
KH-24 itself is now expressible as an A1 config via
:class:`core.strategies.kh24.signal_module.KH24SignalModule` +
:class:`core.architectures.a1_system_level_filter.A1Config`. See
``docs/PROTOCOL_RUNTIME.md`` §"KH-24 as A1" for the equivalence.

This module preserves the direct ``MultiPairBacktester`` invocation
path (no A1 indirection) so the anchor regression test can confirm
``KH24FoldRunner == A1(KH24SignalModule, KH24-as-A1-config)`` produces
byte-identical fold stats. Once the regression check passes, future
arcs should use ArcFoldRunner; KH24FoldRunner is frozen as a baseline.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import pandas as pd

from core.sim.multipair_backtester import MultiPairBacktester, RunResult
from core.sim.panel import Panel
from core.strategies.kh24.kh24 import KH24Config, build_kh24_runtime
from core.wfo.folds import Fold
from core.wfo.gates import FoldStats


@dataclass(frozen=True)
class FoldRunResult:
    """Wrapper around RunResult + the FoldStats the orchestrator consumes."""

    fold: Fold
    config_id: str
    run_result: RunResult
    fold_stats: FoldStats


def _slice_panel_by_dates(panel: Panel, start: date, end: date) -> Panel:
    """Slice every pair_df in the panel to ``start..end`` (inclusive)."""
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    sliced: dict[str, pd.DataFrame] = {}
    for pair, df in panel.pair_dfs.items():
        sliced[pair] = df.loc[start_ts:end_ts]
    return Panel.from_frames(sliced, tf=panel.tf, boundary_convention=panel.boundary_convention)


def _equity_to_daily_returns(equity: pd.Series) -> pd.Series:
    """Resample equity to daily returns. Used for Sharpe + daily-DD checks."""
    if len(equity) == 0:
        return pd.Series([], dtype="float64")
    daily_eq = equity.resample("1D").last().ffill()
    daily_ret = daily_eq.pct_change().fillna(0.0)
    return daily_ret


def _max_drawdown_pct(equity: pd.Series) -> float:
    """Peak-to-trough drawdown as a positive percent."""
    if len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    drawdown = (running_max - equity) / running_max
    return float(drawdown.max())


def _count_daily_5pct_breaches(equity: pd.Series) -> int:
    """5ers 5% daily-DD breaches: any UTC day where equity dropped > 5% from
    the day's open."""
    if len(equity) == 0:
        return 0
    daily = equity.resample("1D").agg(["first", "min"])
    daily = daily.dropna()
    if len(daily) == 0:
        return 0
    daily_dd = (daily["first"] - daily["min"]) / daily["first"]
    return int((daily_dd > 0.05).sum())


def _annualised_roi(equity: pd.Series) -> float:
    """ROI annualised: ((end / start) ^ (365 / span_days)) - 1."""
    if len(equity) < 2:
        return 0.0
    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    if start_val <= 0 or end_val <= 0:
        return 0.0
    span_days = (equity.index[-1] - equity.index[0]).days
    if span_days <= 0:
        return 0.0
    return (end_val / start_val) ** (365.0 / span_days) - 1.0


def _build_fold_stats(fold: Fold, result: RunResult, starting_balance: float) -> FoldStats:
    """Convert a RunResult into the FoldStats the §3 gate consumes."""
    equity = result.equity_curve
    if len(equity) == 0:
        return FoldStats(
            fold_id=fold.fold_id,
            n_trades=0,
            roi_pct=0.0,
            max_dd_pct=0.0,
            days_breaching_daily_5pct=0,
            roi_dd_ratio=0.0,
        )

    # ROI over the fold window — simple end/start
    period_roi = float(equity.iloc[-1] / starting_balance - 1.0)
    max_dd = _max_drawdown_pct(equity)
    breaches = _count_daily_5pct_breaches(equity)
    ratio = period_roi / max_dd if max_dd > 0 else (float("inf") if period_roi > 0 else 0.0)
    if math.isinf(ratio):
        # Cap infinity at a sentinel so downstream gates don't choke
        ratio = 999.0
    return FoldStats(
        fold_id=fold.fold_id,
        n_trades=len(result.closed_trades),
        roi_pct=period_roi,
        max_dd_pct=max_dd,
        days_breaching_daily_5pct=breaches,
        roi_dd_ratio=ratio,
    )


@dataclass
class KH24FoldRunner:
    """Adapter that produces ``fold_runner(fold, config) -> FoldStats``
    callables for the KH-24 strategy.

    Caller supplies the three panels (H4 + D1 + H1) at construction
    time; the runner slices them per-fold and rebuilds a fresh
    ``KH24Runtime`` for each (fold, config) pair.
    """

    panel_h4: Panel
    panel_d1: Panel
    panel_h1: Panel

    def __call__(self, fold: Fold, config: KH24Config) -> FoldStats:
        # Slice each panel to the fold's OOS window (IS warmup is built
        # into the per-pair feature/signal precompute — H4 ATR(14) needs
        # ~14 bars of history; we slice generously to include a bit of
        # warmup before OOS).
        warmup_days = 30
        slice_start = fold.oos_start - pd.Timedelta(days=warmup_days).to_pytimedelta()
        slice_end = fold.oos_end

        h4 = _slice_panel_by_dates(self.panel_h4, slice_start, slice_end)
        d1 = _slice_panel_by_dates(self.panel_d1, slice_start, slice_end)
        h1 = _slice_panel_by_dates(self.panel_h1, slice_start, slice_end)

        runtime = build_kh24_runtime(h4, d1, h1, config=config)
        bt = MultiPairBacktester(
            panel=h4,
            account=runtime.account,
            strategy=runtime.strategy,
            trail_manager=runtime.trail_manager,
            exit_predicates=runtime.exit_predicates,
        )
        result = bt.run()
        return _build_fold_stats(fold, result, starting_balance=config.starting_balance)


def make_kh24_fold_runner(
    panel_h4: Panel, panel_d1: Panel, panel_h1: Panel
) -> Callable[[Fold, KH24Config], FoldStats]:
    """Convenience constructor — returns a callable suitable for
    ``run_search(structure, candidates, fold_runner=...)``."""
    return KH24FoldRunner(panel_h4=panel_h4, panel_d1=panel_d1, panel_h1=panel_h1)
