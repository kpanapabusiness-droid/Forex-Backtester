"""A5 — Portfolio composition architecture.

Per L_PROTOCOL §2 Step 5: combines ≥ 2 candidate-cluster strategies'
admit-only economics into a portfolio with combined account state.
Only applies when Step 3 surfaces 2+ candidate clusters.

A5 doesn't run a new backtest — it consumes a list of already-computed
:class:`StrategyResult` objects (each from A1/A2/A6 for a single
candidate cluster) and combines them:

  - Closed trades: union (sorted by entry_time)
  - Equity curve: shared starting balance; per-bar contributions
    summed across constituents
  - DD: recomputed on the combined equity
  - Daily breaches: recomputed on combined equity

The portfolio inherits the worst constituent's FoldStats fields where
applicable; ROI / DD / ratio come from the combined curve.

This is admit-only economics — per L_PROTOCOL the constituents are
assumed to be the cluster-specific signal flows; the combined portfolio
is the sum of those flows. No double-counting because each constituent
already restricts to its own cluster's admitted trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from core.architectures._protocol import StrategyResult
from core.runners._fold_stats_helpers import (
    build_fold_stats_from_run,
    count_daily_5pct_breaches,
    max_drawdown_pct,
)
from core.sim.multipair_backtester import RunResult
from core.wfo.folds import Fold
from core.wfo.gates import FoldStats


@dataclass(frozen=True)
class A5Config:
    """A5 portfolio composition: list of constituent StrategyResults."""

    config_id: str
    constituents: tuple[StrategyResult, ...]
    starting_balance: float = 100_000.0


@dataclass(frozen=True)
class A5Architecture:
    architecture_name: str = "A5"

    def run(
        self,
        *,
        arch_config: A5Config,
        config_id: str,
        fold: Fold | None = None,
        # The protocol's other kwargs are unused but kept for interface
        # uniformity; A5 doesn't need them.
        signal_evaluation=None,
        panels=None,
        run_context=None,
    ) -> StrategyResult:
        if not arch_config.constituents:
            raise ValueError("A5 requires at least one constituent StrategyResult")
        fold_resolved = fold or arch_config.constituents[0].fold

        # Union of closed trades across constituents
        all_trades = []
        for c in arch_config.constituents:
            all_trades.extend(c.closed_trades)
        # Sort by entry_time then position_id for determinism
        all_trades.sort(key=lambda t: (pd.Timestamp(t.entry_time), int(t.position_id)))
        all_trades_tuple = tuple(all_trades)

        # Combined equity: sum of (constituent equity - starting balance) +
        # combined starting balance. Each constituent ran on the same fold
        # so their equity timestamps overlap; we align on a union index
        # forward-filled per constituent then sum the deltas.
        union_index = None
        for c in arch_config.constituents:
            if len(c.equity_curve) == 0:
                continue
            if union_index is None:
                union_index = c.equity_curve.index
            else:
                union_index = union_index.union(c.equity_curve.index)
        if union_index is None or len(union_index) == 0:
            equity = pd.Series([], dtype="float64", name="equity")
        else:
            sb = float(arch_config.starting_balance)
            n_constituents = len(arch_config.constituents)
            per_constituent_sb = sb / n_constituents
            combined = pd.Series(0.0, index=union_index)
            for c in arch_config.constituents:
                eq = c.equity_curve.reindex(union_index).ffill()
                # Convert to PnL relative to constituent's starting balance,
                # then sum. Constituent's starting balance is recovered from
                # eq.iloc[0] (first equity point); approximate but consistent.
                if len(eq.dropna()) == 0:
                    continue
                first = float(eq.dropna().iloc[0])
                delta = (eq - first).fillna(0.0)
                combined = combined + delta
            equity = (combined + sb).rename("equity")

        # Build a synthetic RunResult so downstream code stays uniform
        run_result = RunResult(
            final_balance=float(equity.iloc[-1]) if len(equity) else float(arch_config.starting_balance),
            n_trades=len(all_trades_tuple),
            n_open_at_end=0,
            equity_curve=equity,
            max_drawdown_pct=max_drawdown_pct(equity),
            closed_trades=all_trades_tuple,
        )
        fold_stats = build_fold_stats_from_run(
            fold=fold_resolved,
            run_result=run_result,
            starting_balance=arch_config.starting_balance,
        )
        return StrategyResult(
            architecture=self.architecture_name,
            config_id=config_id,
            fold=fold_resolved,
            run_result=run_result,
            fold_stats=fold_stats,
            equity_curve=equity,
            closed_trades=all_trades_tuple,
            metadata={
                "n_constituents": len(arch_config.constituents),
                "constituent_architectures": tuple(c.architecture for c in arch_config.constituents),
                "constituent_config_ids": tuple(c.config_id for c in arch_config.constituents),
            },
        )


__all__ = ("A5Config", "A5Architecture")
