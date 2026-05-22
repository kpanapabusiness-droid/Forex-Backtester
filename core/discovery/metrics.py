"""Per-rule metric computation from a trade pool.

Computes the metrics required by the dispatch's ``full_search_log``:

    pool_size, mean_r, std_r, sharpe_lo, r_p25, r_p50, r_p75, win_rate,
    mean_bars_held, t_stat, p_value, n_pairs_with_trades, n_trail_activated

``sharpe_lo`` is Lo's small-sample-corrected Sharpe ratio (per Lo 2002),
included for diagnostic transparency even though ranking is by raw mean R
(per dispatch Override 1).

``t_stat`` / ``p_value`` use a two-sided one-sample t-test of mean R
against zero. The p-value drives Bonferroni accounting in
``core.discovery.bonferroni``.

Determinism: pure functions of the trade list. No RNG, no global state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import stats as scipy_stats

from core.discovery.pool_simulator import TradeRow


@dataclass(frozen=True)
class RuleMetrics:
    """Per-rule summary statistics."""

    pool_size: int
    mean_r: float
    std_r: float
    sharpe_lo: float
    r_p25: float
    r_p50: float
    r_p75: float
    win_rate: float
    mean_bars_held: float
    t_stat: float
    p_value: float
    n_pairs_with_trades: int
    n_trail_activated: int

    def to_dict(self) -> dict:
        return {
            "pool_size": self.pool_size,
            "mean_r": self.mean_r,
            "std_r": self.std_r,
            "sharpe_lo": self.sharpe_lo,
            "r_p25": self.r_p25,
            "r_p50": self.r_p50,
            "r_p75": self.r_p75,
            "win_rate": self.win_rate,
            "mean_bars_held": self.mean_bars_held,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "n_pairs_with_trades": self.n_pairs_with_trades,
            "n_trail_activated": self.n_trail_activated,
        }


def empty_metrics() -> RuleMetrics:
    """Sentinel for rules with insufficient pool — every numeric field NaN."""
    nan = float("nan")
    return RuleMetrics(
        pool_size=0,
        mean_r=nan,
        std_r=nan,
        sharpe_lo=nan,
        r_p25=nan,
        r_p50=nan,
        r_p75=nan,
        win_rate=nan,
        mean_bars_held=nan,
        t_stat=nan,
        p_value=nan,
        n_pairs_with_trades=0,
        n_trail_activated=0,
    )


def compute_rule_metrics(trades: Sequence[TradeRow]) -> RuleMetrics:
    """Compute the full metric set from a sequence of trades.

    Edge cases:
      * Empty trade list -> :func:`empty_metrics`
      * Single trade -> std=0, sharpe=NaN, t-stat=NaN (degenerate)
    """
    if not trades:
        return empty_metrics()
    n = len(trades)
    r = np.fromiter((t.final_r for t in trades), dtype="float64", count=n)
    bars = np.fromiter((t.bars_held for t in trades), dtype="float64", count=n)

    pool_size = int(n)
    mean_r = float(np.mean(r))
    # ddof=1 for sample std (consistent with t-test variance)
    std_r = float(np.std(r, ddof=1)) if n > 1 else 0.0
    r_p25, r_p50, r_p75 = (float(x) for x in np.quantile(r, [0.25, 0.50, 0.75]))
    win_rate = float(np.mean(r > 0.0))
    mean_bars_held = float(np.mean(bars))
    n_pairs = len({t.pair for t in trades})
    n_trail = int(sum(1 for t in trades if t.activated_trail))

    # Lo small-sample-corrected Sharpe (Lo 2002):
    #   sharpe_lo = mean / std * sqrt(n) / sqrt(n - 1)
    # Reduces to Sharpe at large n. NaN for n <= 1.
    if n > 1 and std_r > 0.0:
        sharpe_raw = mean_r / std_r
        sharpe_lo = sharpe_raw * math.sqrt(n / (n - 1.0))
    else:
        sharpe_lo = float("nan")

    # Two-sided one-sample t-test (mean vs 0).
    if n > 1 and std_r > 0.0:
        # scipy_stats.ttest_1samp is deterministic for fixed input.
        t_result = scipy_stats.ttest_1samp(r, popmean=0.0)
        t_stat = float(t_result.statistic)
        p_value = float(t_result.pvalue)
    else:
        t_stat = float("nan")
        p_value = float("nan")

    return RuleMetrics(
        pool_size=pool_size,
        mean_r=mean_r,
        std_r=std_r,
        sharpe_lo=sharpe_lo,
        r_p25=r_p25,
        r_p50=r_p50,
        r_p75=r_p75,
        win_rate=win_rate,
        mean_bars_held=mean_bars_held,
        t_stat=t_stat,
        p_value=p_value,
        n_pairs_with_trades=n_pairs,
        n_trail_activated=n_trail,
    )


__all__ = ("RuleMetrics", "empty_metrics", "compute_rule_metrics")
