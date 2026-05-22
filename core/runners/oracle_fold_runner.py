"""Oracle WFO fold runner — upper-bound establishment.

Per L_PROTOCOL §2 Step 5 item 5: "for each candidate cluster, run WFO
using true-label cluster membership (no classifier). Establishes the
upper bound on what's possible if entry filtering were free. Reported
alongside real WFO for each architecture."

The oracle runner restricts trades to those whose Step 2 cluster_id
matches the target candidate cluster. Since the cluster label is
post-hoc (computed from the realised forward path), this is
NON-DEPLOYABLE — it's a diagnostic showing the ceiling on what
filtering could achieve.

Implementation: wraps A1 with an extra filter rule "trade.cluster_id ==
candidate_cluster_id". The runtime looks up the cluster_id by
(pair, signal_time) from the cluster_assignments dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from core.arc.signal_protocol import SignalEvaluation
from core.architectures._protocol import StrategyResult
from core.architectures.a1_system_level_filter import (
    A1Architecture,
    A1Config,
    A1RunContext,
)
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.panel import Panel
from core.wfo.folds import Fold
from core.wfo.gates import FoldStats


@dataclass
class OracleFoldRunner:
    """WFO fold runner that admits only trades belonging to the target
    cluster (true-label, post-hoc).

    Construct with:
      - SignalEvaluation
      - Panels
      - cluster_assignments: per-trade cluster_id from Step 2
      - candidate_cluster_id: the cluster the oracle admits

    Call signature matches ArcFoldRunner: (fold, A1Config) -> FoldStats.
    """

    signal_evaluation: SignalEvaluation
    panels: Mapping[str, Panel]
    cluster_assignments: pd.DataFrame  # trade_id, cluster_id
    candidate_cluster_id: int
    trades: pd.DataFrame  # Step 1 trades — to map (pair, signal_time) -> cluster_id
    last_result: StrategyResult | None = None

    def __post_init__(self) -> None:
        # Build per-pair, per-signal_time -> cluster_id lookup
        merged = self.trades.merge(
            self.cluster_assignments[["trade_id", "cluster_id"]],
            on="trade_id",
        )
        # Per-trade lookup as a frozen dict of (pair, signal_time) -> cluster_id
        self._lookup: dict[tuple[str, pd.Timestamp], int] = {}
        for _, r in merged.iterrows():
            key = (str(r["pair"]), pd.Timestamp(r["signal_time"]))
            self._lookup[key] = int(r["cluster_id"])

    def __call__(self, fold: Fold, arch_config: A1Config) -> FoldStats:
        # Build a per-trade-features lookup with synthetic "cluster_id" feature
        # so A1's filter_rules can gate on it.
        feats = {
            key: {"_oracle_cluster_id": float(cid)}
            for key, cid in self._lookup.items()
        }
        # Inject an "_oracle_cluster_id == candidate" filter rule into the config
        oracle_rule = ("_oracle_cluster_id", "ge", float(self.candidate_cluster_id))
        oracle_rule_le = ("_oracle_cluster_id", "le", float(self.candidate_cluster_id))
        cfg_with_oracle = A1Config(
            config_id=arch_config.config_id + f"_oracle_c{self.candidate_cluster_id}",
            sl_atr_mult=arch_config.sl_atr_mult,
            trail_enabled=arch_config.trail_enabled,
            trail_activation_atr=arch_config.trail_activation_atr,
            trail_distance_atr=arch_config.trail_distance_atr,
            filter_rules=(oracle_rule, oracle_rule_le),
            risk_pct=arch_config.risk_pct,
            starting_balance=arch_config.starting_balance,
            max_concurrent_total=arch_config.max_concurrent_total,
            max_concurrent_per_pair=arch_config.max_concurrent_per_pair,
            max_concurrent_per_currency=arch_config.max_concurrent_per_currency,
        )
        runner = ArcFoldRunner(
            architecture=A1Architecture(),
            signal_evaluation=self.signal_evaluation,
            panels=self.panels,
            run_context=A1RunContext(per_trade_features=feats),
        )
        stats = runner(fold, cfg_with_oracle)
        self.last_result = runner.last_result
        return stats


__all__ = ("OracleFoldRunner",)
