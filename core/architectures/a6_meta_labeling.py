"""A6 — Meta-labeling architecture.

Per L_PROTOCOL §2 Step 5 + Amendment 2:
  - Same classifier as A2 (Step 4's best)
  - Confidence mapped to position size (NOT admit/reject):
      prob < lower            -> 0.0x  (skip)
      lower <= prob < upper   -> 0.5x  (half size)
      prob >= upper           -> 1.0x  (full size)
  - Default (lower, upper) = (0.4, 0.6)
  - Step 5 threshold sweep tests {(0.3,0.5), (0.4,0.6), (0.5,0.7)}

Sizing channel: chat decision §6.4 — :class:`Order` has a
``risk_multiplier`` field; the driver multiplies position size by it
at fill time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.arc.signal_protocol import SignalEvaluation
from core.architectures._protocol import StrategyResult
from core.architectures.a1_system_level_filter import (
    A1RunContext,
    _slice_equity_to_oos,
    _slice_panels_to_fold,
)
from core.runners._fold_stats_helpers import build_fold_stats_from_run
from core.sim.account import Account, Direction, ExposureRules
from core.sim.exit_hooks import ExitPredicate
from core.sim.multipair_backtester import MultiPairBacktester, Order, StrategyFn
from core.sim.panel import Panel
from core.sim.risk.live_balance import LiveBalanceRisk
from core.sim.trailing_stop import TrailManager
from core.wfo.folds import Fold

ClassifierLike = Any

# Three threshold pairs per Amendment 2 sweep
THRESHOLD_PAIRS: tuple[tuple[float, float], ...] = (
    (0.3, 0.5),
    (0.4, 0.6),
    (0.5, 0.7),
)


@dataclass(frozen=True)
class A6Config:
    """A6 layers meta-labeling sizing over A1 mechanics."""

    config_id: str
    classifier: ClassifierLike
    lower_threshold: float
    upper_threshold: float
    classifier_feature_order: tuple[str, ...]
    sl_atr_mult: float = 2.0
    trail_enabled: bool = True
    trail_activation_atr: float = 2.0
    trail_distance_atr: float = 1.5
    risk_pct: float = 0.005
    starting_balance: float = 100_000.0
    max_concurrent_total: int | None = None
    max_concurrent_per_pair: int | None = 1
    max_concurrent_per_currency: int | None = 2
    # Amendment 3 §"Sizing convention"
    sizing_convention: str = "reset_floor"   # "reset_floor" | "equity_pct"


def _confidence_to_multiplier(
    prob: float, lower: float, upper: float
) -> float:
    if prob < lower:
        return 0.0
    if prob < upper:
        return 0.5
    return 1.0


def _build_a6_strategy(
    signal_eval: SignalEvaluation,
    panels: Mapping[str, Panel],
    cfg: A6Config,
    ctx: A1RunContext,
    account: Account,
    risk: LiveBalanceRisk,
) -> StrategyFn:
    primary_panel = panels[signal_eval.primary_tf]
    per_pair = signal_eval.per_pair
    series_cache: dict[str, dict[str, pd.Series]] = {}
    for pair, state in per_pair.items():
        df = primary_panel.pair_dfs.get(pair)
        if df is None:
            continue
        mask = state.signal_mask.reindex(df.index).fillna(False).astype(bool)
        atr = state.atr.reindex(df.index)
        gates = {
            k: v.reindex(df.index).fillna(False).astype(bool)
            for k, v in state.additional_gates.items()
        }
        series_cache[pair] = {"mask": mask, "atr": atr, **gates}

    feat_keys = cfg.classifier_feature_order

    def strategy(t, snapshot, acct):  # type: ignore[no-untyped-def]
        orders: list[Order] = []
        for pair in sorted(series_cache):
            cached = series_cache[pair]
            mask = cached["mask"]
            if t not in mask.index or not bool(mask.loc[t]):
                continue
            gate_ok = True
            for k, ser in cached.items():
                if k in ("mask", "atr"):
                    continue
                if not bool(ser.loc[t]):
                    gate_ok = False
                    break
            if not gate_ok:
                continue
            if ctx.per_trade_features is None:
                continue
            feats = ctx.per_trade_features.get((pair, t))
            if feats is None:
                continue
            try:
                row = np.array([float(feats[k]) for k in feat_keys], dtype=np.float64).reshape(1, -1)
            except (KeyError, TypeError, ValueError):
                continue
            if not np.all(np.isfinite(row)):
                continue
            try:
                proba = float(cfg.classifier.predict_proba(row)[0, 1])
            except Exception:
                continue
            mult = _confidence_to_multiplier(proba, cfg.lower_threshold, cfg.upper_threshold)
            if mult <= 0.0:
                continue
            atr = float(cached["atr"].loc[t])
            if not (atr > 0):
                continue
            bar = snapshot.get(pair)
            if bar is None:
                continue
            entry_proxy = float(bar["close_ask"])
            sl_price = entry_proxy - cfg.sl_atr_mult * atr
            if sl_price <= 0:
                continue
            base_size = risk.risk_size(
                acct, entry_price=entry_proxy, sl_price=sl_price, risk_pct=cfg.risk_pct
            )
            orders.append(Order(
                pair=pair,
                direction=Direction.LONG,
                size=base_size,
                sl_price=sl_price,
                tp_price=None,
                atr_at_entry=atr,
                trail_activation_atr=cfg.trail_activation_atr,
                trail_distance_atr=cfg.trail_distance_atr,
                risk_multiplier=mult,
            ))
        return orders

    return strategy


@dataclass(frozen=True)
class A6Architecture:
    architecture_name: str = "A6"

    def run(
        self,
        *,
        signal_evaluation: SignalEvaluation,
        panels: Mapping[str, Panel],
        fold: Fold,
        arch_config: A6Config,
        config_id: str,
        run_context: A1RunContext | None = None,
    ) -> StrategyResult:
        ctx = run_context or A1RunContext()
        sliced = _slice_panels_to_fold(panels, fold)
        primary = sliced[signal_evaluation.primary_tf]
        account = Account(
            starting_balance=arch_config.starting_balance,
            exposure=ExposureRules(
                max_concurrent_total=arch_config.max_concurrent_total,
                max_concurrent_per_pair=arch_config.max_concurrent_per_pair,
                max_concurrent_per_currency=arch_config.max_concurrent_per_currency,
            ),
        )
        risk = LiveBalanceRisk(risk_pct=arch_config.risk_pct)
        trail_manager = TrailManager() if arch_config.trail_enabled else None
        exit_predicates: list[ExitPredicate] = []
        for pair in sorted(signal_evaluation.per_pair):
            ep = signal_evaluation.per_pair[pair].exit_predicate
            if ep is not None:
                exit_predicates.append(ep)
        strategy = _build_a6_strategy(
            signal_eval=signal_evaluation,
            panels=sliced,
            cfg=arch_config,
            ctx=ctx,
            account=account,
            risk=risk,
        )
        bt = MultiPairBacktester(
            panel=primary,
            account=account,
            strategy=strategy,
            trail_manager=trail_manager,
            exit_predicates=tuple(exit_predicates),
        )
        run_result = bt.run()
        fold_stats = build_fold_stats_from_run(
            fold=fold,
            run_result=run_result,
            starting_balance=arch_config.starting_balance,
        )
        equity_oos = _slice_equity_to_oos(run_result.equity_curve, fold)
        return StrategyResult(
            architecture=self.architecture_name,
            config_id=config_id,
            fold=fold,
            run_result=run_result,
            fold_stats=fold_stats,
            equity_curve=equity_oos,
            closed_trades=run_result.closed_trades,
            metadata={
                "lower_threshold": arch_config.lower_threshold,
                "upper_threshold": arch_config.upper_threshold,
                "classifier_type": type(arch_config.classifier).__name__,
            },
        )


__all__ = ("THRESHOLD_PAIRS", "A6Config", "A6Architecture")
