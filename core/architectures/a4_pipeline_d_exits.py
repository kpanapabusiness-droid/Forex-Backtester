"""A4 — Pipeline D (differentiated exits) architecture.

Per L_PROTOCOL §2 Step 5 + Amendment 2:
  - NEW classifier per fold
  - Target: "final R > 0" (binary, profitable trade outcome)
  - Features: path-so-far at each bar post-entry (same schema as A3)
  - Decision: at each bar post-entry, if classifier confidence < exit
    threshold, queue exit at next-bar open
  - Tested exit thresholds: {0.3, 0.4, 0.5}
  - Initial SL still binds (classifier exit does NOT override SL — the
    SL fires first if both would trigger same bar)

Implementation: A4 uses an :class:`ExitPredicate` that consults the
per-fold classifier at each bar. The predicate returns an exit decision
when confidence drops below the threshold.

Per chat decision: the predicate path is the cleanest channel — it
hooks into MultiPairBacktester's existing exit_predicates without
modifying the driver. SL precedence is preserved by the driver's
intra-bar exits-first ordering.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from core.arc.signal_protocol import SignalEvaluation
from core.architectures._path_classifier import (
    PathClassifierFit,
    predict_admit,
)
from core.architectures._protocol import StrategyResult
from core.architectures.a1_system_level_filter import (
    A1Config,
    A1RunContext,
    _build_a1_strategy,
    _slice_equity_to_oos,
    _slice_panels_to_fold,
)
from core.runners._fold_stats_helpers import build_fold_stats_from_run
from core.sim.account import Account, Direction, ExposureRules, Position
from core.sim.exit_hooks import ExitDecision, ExitPredicate
from core.sim.multipair_backtester import MultiPairBacktester
from core.sim.panel import Panel
from core.sim.risk.live_balance import LiveBalanceRisk
from core.sim.trailing_stop import TrailManager
from core.wfo.folds import Fold

DEFAULT_EXIT_THRESHOLDS = (0.3, 0.4, 0.5)


@dataclass(frozen=True)
class A4Config:
    """A4 differentiated-exits config.

    ``exit_threshold`` is the confidence floor; below it, a bar-close
    exit is queued.

    ``classifier_fit`` and ``per_trade_entry_features`` are DEPRECATED
    single-fit paths kept for backwards-compat with synthetic tests +
    direct-construction drivers. Prefer threading per-fold data via
    :class:`A1RunContext` (``path_classifier_fits`` keyed by
    ``fold.fold_id``, ``per_trade_entry_features`` shared across
    folds). The deprecated fields emit a DeprecationWarning at
    runtime; per chat directive Q2 they are scheduled for removal
    after 2 closed arcs use the new path successfully.
    """

    config_id: str
    classifier_fit: PathClassifierFit | None = None  # DEPRECATED
    exit_threshold: float = 0.4
    sl_atr_mult: float = 2.0
    trail_enabled: bool = True
    trail_activation_atr: float = 2.0
    trail_distance_atr: float = 1.5
    risk_pct: float = 0.005
    starting_balance: float = 100_000.0
    max_concurrent_total: int | None = None
    max_concurrent_per_pair: int | None = 1
    max_concurrent_per_currency: int | None = 2
    per_trade_entry_features: Mapping[tuple[str, pd.Timestamp], Mapping[str, float]] | None = None  # DEPRECATED
    # Amendment 3 §"Sizing convention"
    sizing_convention: str = "reset_floor"   # "reset_floor" | "equity_pct"


@dataclass
class _A4ExitPredicate:
    """Exit predicate that consults the per-fold path-classifier.

    For each open position, at bar close: compute path-so-far features
    relative to the position's entry; query the classifier; if
    confidence drops below ``exit_threshold``, return an exit decision.
    """

    pair: str
    pair_df: pd.DataFrame
    classifier_fit: PathClassifierFit
    exit_threshold: float
    entry_features_by_signal_time: Mapping[pd.Timestamp, Mapping[str, float]]
    # SignalModule sets the entry-time anchor; we need to recover it
    # to look up entry-features. Convention: position.entry_time is the
    # bar AFTER the signal bar (next-bar-open fill). Signal time =
    # entry_time - one primary-TF bar.
    primary_tf_index: pd.Index

    def __call__(
        self, position: Position, snapshot: dict[str, pd.Series | None], t: pd.Timestamp
    ) -> ExitDecision | None:
        if position.pair != self.pair or position.direction is not Direction.LONG:
            return None
        # Recover signal_time = bar preceding entry_time on this pair
        entry_ts = pd.Timestamp(position.entry_time)
        idx_arr = self.primary_tf_index.get_indexer([entry_ts])
        if len(idx_arr) == 0 or idx_arr[0] <= 0:
            return None
        sig_idx = int(idx_arr[0]) - 1
        sig_t = self.primary_tf_index[sig_idx]
        entry_feats = self.entry_features_by_signal_time.get(sig_t)
        if entry_feats is None:
            return None
        # Path-so-far up to current bar t
        t_arr = self.primary_tf_index.get_indexer([t])
        if len(t_arr) == 0 or t_arr[0] < 0:
            return None
        t_idx = int(t_arr[0])
        entry_idx = int(idx_arr[0])
        if t_idx < entry_idx:
            return None
        sl_distance = float(position.entry_price) - float(position.sl_price or 0)
        if sl_distance <= 0:
            return None
        # Re-use the same path-so-far computation as A3
        from core.architectures.a3_pipeline_de import _path_features_so_far
        path_feats = _path_features_so_far(
            self.pair_df,
            signal_idx=sig_idx,
            n_defer=t_idx - sig_idx,
            sl_anchor_price=float(position.entry_price),
            sl_distance=sl_distance,
        )
        combined = {**entry_feats, **path_feats}
        admit, proba = predict_admit(
            self.classifier_fit, combined, threshold_override=None
        )
        # We want "exit if confidence DROPS BELOW threshold". The
        # classifier predicts P(final_r > 0). High proba = let it run.
        # Low proba = exit.
        if proba < self.exit_threshold:
            # fill_price is recorded but the driver's PR-E.1.6 path
            # discards it and fills at next-bar open_bid (long exits).
            # Provide a sane value anyway for any consumer that reads it.
            bar = snapshot.get(self.pair)
            fp = float(bar["close_bid"]) if bar is not None else float(position.entry_price)
            return ExitDecision(fill_price=fp, exit_reason="pipeline_d_exit")
        return None


@dataclass(frozen=True)
class A4Architecture:
    architecture_name: str = "A4"

    def run(
        self,
        *,
        signal_evaluation: SignalEvaluation,
        panels: Mapping[str, Panel],
        fold: Fold,
        arch_config: A4Config,
        config_id: str,
        run_context: A1RunContext | None = None,
    ) -> StrategyResult:
        # Resolve classifier_fit + entry_features_lookup via run_context
        # (canonical, per-fold) or arch_config (deprecated single-fit).
        classifier_fit = _resolve_a4_classifier_fit(arch_config, run_context, fold)
        entry_features_lookup = _resolve_a4_entry_features(arch_config, run_context)

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

        # Build per-pair exit predicates from the classifier + entry features
        entry_features_by_signal_time_per_pair: dict[str, dict[pd.Timestamp, Mapping[str, float]]] = {}
        if entry_features_lookup is not None:
            for (pair, sig_t), feats in entry_features_lookup.items():
                entry_features_by_signal_time_per_pair.setdefault(pair, {})[sig_t] = feats

        a4_predicates: list[ExitPredicate] = []
        for pair in sorted(signal_evaluation.per_pair):
            df = primary.pair_dfs.get(pair)
            if df is None:
                continue
            a4_predicates.append(_A4ExitPredicate(
                pair=pair,
                pair_df=df,
                classifier_fit=classifier_fit,
                exit_threshold=arch_config.exit_threshold,
                entry_features_by_signal_time=entry_features_by_signal_time_per_pair.get(pair, {}),
                primary_tf_index=df.index,
            ))
        # Append signal-class exit predicates as well
        for pair in sorted(signal_evaluation.per_pair):
            ep = signal_evaluation.per_pair[pair].exit_predicate
            if ep is not None:
                a4_predicates.append(ep)

        # Strategy is identical to A1 — A4 differs only in exit side
        a1_like_cfg = A1Config(
            config_id=arch_config.config_id,
            sl_atr_mult=arch_config.sl_atr_mult,
            trail_enabled=arch_config.trail_enabled,
            trail_activation_atr=arch_config.trail_activation_atr,
            trail_distance_atr=arch_config.trail_distance_atr,
            filter_rules=(),
            risk_pct=arch_config.risk_pct,
            starting_balance=arch_config.starting_balance,
            max_concurrent_total=arch_config.max_concurrent_total,
            max_concurrent_per_pair=arch_config.max_concurrent_per_pair,
            max_concurrent_per_currency=arch_config.max_concurrent_per_currency,
        )
        strategy = _build_a1_strategy(
            signal_eval=signal_evaluation,
            panels=sliced,
            cfg=a1_like_cfg,
            ctx=run_context or A1RunContext(),
            account=account,
            risk=risk,
        )
        bt = MultiPairBacktester(
            panel=primary,
            account=account,
            strategy=strategy,
            trail_manager=trail_manager,
            exit_predicates=tuple(a4_predicates),
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
                "exit_threshold": arch_config.exit_threshold,
                "classifier_fit_auc": classifier_fit.fit_auc,
            },
        )


def _resolve_a4_classifier_fit(
    cfg: "A4Config",
    ctx: A1RunContext | None,
    fold: Fold,
) -> PathClassifierFit:
    """Return the PathClassifierFit for A4 at ``fold``.

    Mirror of :func:`core.architectures.a3_pipeline_de._resolve_a3_classifier_fit`.
    """
    if ctx is not None and ctx.path_classifier_fits is not None:
        fit = ctx.path_classifier_fits.get(fold.fold_id)
        if fit is not None:
            return fit  # type: ignore[return-value]
    if cfg.classifier_fit is not None:
        warnings.warn(
            "A4Config.classifier_fit is deprecated; pass per-fold fits "
            "via A1RunContext.path_classifier_fits (keyed by fold_id). "
            "Single-fit support will be removed after 2 closed arcs use "
            "the new path successfully (per chat directive Q2).",
            DeprecationWarning,
            stacklevel=3,
        )
        return cfg.classifier_fit
    raise RuntimeError(
        f"A4 requires a path-classifier fit at fold {fold.fold_id}: pass "
        f"via run_context.path_classifier_fits[{fold.fold_id}] (canonical) "
        f"or arch_config.classifier_fit (deprecated)"
    )


def _resolve_a4_entry_features(
    cfg: "A4Config",
    ctx: A1RunContext | None,
) -> Mapping[tuple[str, pd.Timestamp], Mapping[str, float]] | None:
    """Return the entry-features lookup for A4. Mirror of A3 helper."""
    if ctx is not None and ctx.per_trade_entry_features is not None:
        return ctx.per_trade_entry_features
    if cfg.per_trade_entry_features is not None:
        warnings.warn(
            "A4Config.per_trade_entry_features is deprecated; pass via "
            "A1RunContext.per_trade_entry_features. Will be removed "
            "after 2 closed arcs use the new path (chat directive Q2).",
            DeprecationWarning,
            stacklevel=3,
        )
        return cfg.per_trade_entry_features
    return None


__all__ = (
    "DEFAULT_EXIT_THRESHOLDS",
    "A4Config",
    "A4Architecture",
)
