"""A3 — Pipeline DE (deferred entry) architecture.

Per L_PROTOCOL §2 Step 5 + Amendment 2:
  - NEW classifier per fold (trained on IS pool, evaluated on OOS)
  - Target: cluster membership (same as Step 4)
  - Features: path-so-far at bar N ∈ {3, 5} (per Appendix B)
  - Decision: wait N bars after signal, evaluate path-so-far features,
    enter if classifier prob >= threshold, else cancel
  - Initial SL applies once entered

Per-fold retrain is enforced — the runtime instantiates a fresh
classifier from IS data at each fold's start.

A3 implementation note: the entry deferral creates a different
fill-time vs A1. A3 "waits N bars", which in v3 simulator semantics
means:

  - At signal bar t: register a deferred-entry candidate.
  - At bar t+N (close): evaluate path-so-far features (close_r,
    mfe_so_far_r, mae_so_far_r, bar count, velocity).
  - If admitted: emit Order at bar t+N (fills at bar t+N+1 open).
  - Else: drop candidate.

The simulator already fills orders at next-bar open. A3 produces the
order at the *deferred* bar rather than the signal bar; everything
else is identical to A1.
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
    A1RunContext,
    _slice_equity_to_oos,
    _slice_panels_to_fold,
)
from core.features_path_so_far import (
    PATH_FEATURE_KEYS,
)
from core.runners._fold_stats_helpers import build_fold_stats_from_run
from core.sim.account import Account, Direction, ExposureRules
from core.sim.exit_hooks import ExitPredicate
from core.sim.exit_policy_manager import ExitPolicyManager
from core.sim.multipair_backtester import MultiPairBacktester, Order, StrategyFn
from core.sim.panel import Panel
from core.sim.risk.live_balance import LiveBalanceRisk
from core.sim.trailing_stop import TrailManager
from core.wfo.folds import Fold

DEFAULT_N_DEFER_VALUES = (3, 5)


@dataclass(frozen=True)
class A3Config:
    """A3 deferred-entry config.

    ``n_defer`` is the bar count between signal and decision (in {3, 5}
    per Appendix B). ``threshold_override`` is optional; otherwise the
    fit's AUC-best threshold is used.

    ``classifier_fit`` and ``per_trade_entry_features`` are DEPRECATED
    single-fit paths kept for backwards-compat with synthetic tests +
    direct-construction drivers. Prefer threading per-fold data via
    :class:`A1RunContext` (``path_classifier_fits`` keyed by
    ``fold.fold_id``, ``per_trade_entry_features`` shared across folds).
    The deprecated fields emit a DeprecationWarning at runtime; per chat
    directive Q2 they are scheduled for removal after 2 closed arcs use
    the new path successfully.
    """

    config_id: str
    n_defer: int
    classifier_fit: PathClassifierFit | None = None  # DEPRECATED; use run_context.path_classifier_fits
    threshold_override: float | None = None
    sl_atr_mult: float = 2.0
    trail_enabled: bool = True
    trail_activation_atr: float = 2.0
    trail_distance_atr: float = 1.5
    risk_pct: float = 0.005
    starting_balance: float = 100_000.0
    max_concurrent_total: int | None = None
    max_concurrent_per_pair: int | None = 1
    max_concurrent_per_currency: int | None = 2
    # DEPRECATED; use run_context.per_trade_entry_features
    per_trade_entry_features: Mapping[tuple[str, pd.Timestamp], Mapping[str, float]] | None = None
    # Amendment 3 §"Sizing convention"
    sizing_convention: str = "reset_floor"   # "reset_floor" | "equity_pct"
    # Canonical exit-policy (see core.sim.exit_policies). None preserves
    # prior behaviour.
    exit_policy: str | None = None


def _path_features_so_far(
    pair_df: pd.DataFrame,
    signal_idx: int,
    n_defer: int,
    sl_anchor_price: float,
    sl_distance: float,
) -> dict[str, float]:
    """Compute path-so-far features at bar (signal_idx + n_defer).

    The simulator treats bar signal_idx+1 as the would-be entry; we
    compute path-so-far relative to that "tentative entry price" =
    open_ask(signal_idx+1). The classifier decides at signal_idx +
    n_defer whether the path-so-far features admit entry at
    signal_idx + n_defer + 1.

    Returns a dict matching PATH_FEATURE_KEYS. Bars used: from
    signal_idx+1 up to signal_idx+n_defer (inclusive).
    """
    entry_idx = signal_idx + 1
    decide_idx = signal_idx + n_defer
    n = len(pair_df)
    if entry_idx >= n or decide_idx >= n:
        return {k: float("nan") for k in PATH_FEATURE_KEYS}
    entry_price = float(pair_df["open_ask"].iat[entry_idx])
    closes_bid = pair_df["close_bid"].values
    highs_bid = pair_df["high_bid"].values
    lows_bid = pair_df["low_bid"].values
    # close_r at each held bar
    close_r_series: list[float] = []
    mfe_so_far = 0.0
    mae_so_far = 0.0
    for bidx in range(entry_idx, decide_idx + 1):
        close_r = (closes_bid[bidx] - entry_price) / sl_distance
        mfe_bar = (highs_bid[bidx] - entry_price) / sl_distance
        mae_bar = (lows_bid[bidx] - entry_price) / sl_distance
        if mfe_bar > mfe_so_far:
            mfe_so_far = mfe_bar
        if mae_bar < mae_so_far:
            mae_so_far = mae_bar
        close_r_series.append(float(close_r))
    close_r_at_t = close_r_series[-1] if close_r_series else 0.0
    bars_in_profit = sum(1 for c in close_r_series if c > 0)
    local_peaks = 0
    prev_mfe = None
    running_mfe = 0.0
    for bidx in range(entry_idx, decide_idx + 1):
        m = (highs_bid[bidx] - entry_price) / sl_distance
        if m > running_mfe:
            running_mfe = m
            if prev_mfe is not None and running_mfe > prev_mfe:
                local_peaks += 1
        prev_mfe = running_mfe
    in_profit_closes = [c for c in close_r_series if c > 0]
    if in_profit_closes:
        monotone = 1
        for prev, cur in zip(in_profit_closes, in_profit_closes[1:]):
            if cur >= prev:
                monotone += 1
        monotonicity = monotone / max(1, len(in_profit_closes))
    else:
        monotonicity = 0.0
    velocity = mfe_so_far / max(1, n_defer)
    return {
        "close_r_at_t": float(close_r_at_t),
        "mfe_so_far_r_at_t": float(mfe_so_far),
        "mae_so_far_r_at_t": float(mae_so_far),
        "bars_in_profit_at_t": float(bars_in_profit),
        "local_peaks_so_far_at_t": float(local_peaks),
        "monotonicity_so_far_at_t": float(monotonicity),
        "velocity_first_t": float(velocity),
    }


def _build_a3_strategy(
    signal_eval: SignalEvaluation,
    panels: Mapping[str, Panel],
    cfg: A3Config,
    account: Account,
    risk: LiveBalanceRisk,
    *,
    classifier_fit: PathClassifierFit,
    entry_features_lookup: Mapping[tuple[str, pd.Timestamp], Mapping[str, float]] | None,
) -> StrategyFn:
    """A3 emits orders at signal+n_defer rather than signal — deferred entry.

    ``classifier_fit`` and ``entry_features_lookup`` are resolved
    upstream by :meth:`A3Architecture.run` from either ``run_context``
    (canonical orchestrator path, per-fold) or ``arch_config``
    (deprecated single-fit path; emits a DeprecationWarning at the
    callsite).
    """
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

    n_defer = cfg.n_defer

    def strategy(t, snapshot, acct):  # type: ignore[no-untyped-def]
        orders: list[Order] = []
        for pair in sorted(series_cache):
            cached = series_cache[pair]
            mask = cached["mask"]
            df = primary_panel.pair_dfs[pair]
            # Look back n_defer bars for a signal that's now ready to decide
            sig_idx_list = df.index.get_indexer([t])
            if len(sig_idx_list) == 0 or sig_idx_list[0] < 0:
                continue
            t_idx = int(sig_idx_list[0])
            signal_bar_idx = t_idx - n_defer
            if signal_bar_idx < 0:
                continue
            sig_t = df.index[signal_bar_idx]
            if not bool(mask.iloc[signal_bar_idx]):
                continue
            # AND gates at the signal bar
            gate_ok = True
            for k, ser in cached.items():
                if k in ("mask", "atr"):
                    continue
                if not bool(ser.iloc[signal_bar_idx]):
                    gate_ok = False
                    break
            if not gate_ok:
                continue
            atr = float(cached["atr"].iloc[signal_bar_idx])
            if not (atr > 0):
                continue
            # Tentative entry price + SL (computed once at signal bar)
            entry_proxy = float(df["close_ask"].iat[signal_bar_idx])
            sl_price = entry_proxy - cfg.sl_atr_mult * atr
            sl_distance = entry_proxy - sl_price
            if sl_distance <= 0:
                continue
            # Build classifier feature vector
            entry_feats = (
                entry_features_lookup.get((pair, sig_t))
                if entry_features_lookup is not None
                else None
            )
            if entry_feats is None:
                continue
            path_feats = _path_features_so_far(
                df, signal_bar_idx, n_defer, entry_proxy, sl_distance
            )
            combined = {**entry_feats, **path_feats}
            admit, _proba = predict_admit(
                classifier_fit,
                combined,
                threshold_override=cfg.threshold_override,
            )
            if not admit:
                continue
            bar_now = snapshot.get(pair)
            if bar_now is None:
                continue
            # Use current bar's close_ask as the *re-anchored* entry proxy
            # so the order fills at this bar's next-bar open. This matches
            # "wait N bars, then enter at next-bar open" semantics.
            new_entry_proxy = float(bar_now["close_ask"])
            new_sl_price = new_entry_proxy - cfg.sl_atr_mult * atr
            if new_sl_price <= 0:
                continue
            size = risk.risk_size(
                acct,
                entry_price=new_entry_proxy,
                sl_price=new_sl_price,
                risk_pct=cfg.risk_pct,
            )
            orders.append(Order(
                pair=pair,
                direction=Direction.LONG,
                size=size,
                sl_price=new_sl_price,
                tp_price=None,
                atr_at_entry=atr,
                trail_activation_atr=cfg.trail_activation_atr,
                trail_distance_atr=cfg.trail_distance_atr,
                exit_policy=cfg.exit_policy,
                sl_atr_mult=cfg.sl_atr_mult if cfg.exit_policy else None,
            ))
        return orders

    return strategy


@dataclass(frozen=True)
class A3Architecture:
    architecture_name: str = "A3"

    def run(
        self,
        *,
        signal_evaluation: SignalEvaluation,
        panels: Mapping[str, Panel],
        fold: Fold,
        arch_config: A3Config,
        config_id: str,
        run_context: A1RunContext | None = None,
    ) -> StrategyResult:
        # Resolve classifier_fit + entry_features_lookup with the
        # run_context (canonical, per-fold) path preferred over the
        # deprecated arch_config (single-fit) path.
        ctx = run_context
        classifier_fit = _resolve_a3_classifier_fit(arch_config, ctx, fold)
        entry_features_lookup = _resolve_a3_entry_features(arch_config, ctx)

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
        exit_policy_manager = (
            ExitPolicyManager() if arch_config.exit_policy is not None else None
        )
        exit_predicates: list[ExitPredicate] = []
        for pair in sorted(signal_evaluation.per_pair):
            ep = signal_evaluation.per_pair[pair].exit_predicate
            if ep is not None:
                exit_predicates.append(ep)
        strategy = _build_a3_strategy(
            signal_eval=signal_evaluation,
            panels=sliced,
            cfg=arch_config,
            account=account,
            risk=risk,
            classifier_fit=classifier_fit,
            entry_features_lookup=entry_features_lookup,
        )
        bt = MultiPairBacktester(
            panel=primary,
            account=account,
            strategy=strategy,
            trail_manager=trail_manager,
            exit_predicates=tuple(exit_predicates),
            exit_policy_manager=exit_policy_manager,
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
                "n_defer": arch_config.n_defer,
                "classifier_fit_auc": classifier_fit.fit_auc,
                "threshold_used": (
                    arch_config.threshold_override
                    if arch_config.threshold_override is not None
                    else classifier_fit.threshold
                ),
                "exit_policy": arch_config.exit_policy,
            },
        )


def _resolve_a3_classifier_fit(
    cfg: "A3Config",
    ctx: A1RunContext | None,
    fold: Fold,
) -> PathClassifierFit:
    """Return the PathClassifierFit for A3 at ``fold``.

    Preference order:
      1. ``ctx.path_classifier_fits[fold.fold_id]`` (canonical
         orchestrator path; per-fold retrained classifier).
      2. ``cfg.classifier_fit`` (deprecated single-fit path; emits a
         DeprecationWarning).

    Raises ``RuntimeError`` if neither source supplies a fit.
    """
    if ctx is not None and ctx.path_classifier_fits is not None:
        fit = ctx.path_classifier_fits.get(fold.fold_id)
        if fit is not None:
            return fit  # type: ignore[return-value]
    if cfg.classifier_fit is not None:
        warnings.warn(
            "A3Config.classifier_fit is deprecated; pass per-fold fits "
            "via A1RunContext.path_classifier_fits (keyed by fold_id). "
            "Single-fit support will be removed after 2 closed arcs use "
            "the new path successfully (per chat directive Q2).",
            DeprecationWarning,
            stacklevel=3,
        )
        return cfg.classifier_fit
    raise RuntimeError(
        f"A3 requires a path-classifier fit at fold {fold.fold_id}: pass "
        f"via run_context.path_classifier_fits[{fold.fold_id}] (canonical) "
        f"or arch_config.classifier_fit (deprecated)"
    )


def _resolve_a3_entry_features(
    cfg: "A3Config",
    ctx: A1RunContext | None,
) -> Mapping[tuple[str, pd.Timestamp], Mapping[str, float]] | None:
    """Return the entry-features lookup for A3.

    Preference order:
      1. ``ctx.per_trade_entry_features`` (canonical orchestrator path).
      2. ``cfg.per_trade_entry_features`` (deprecated; emits a
         DeprecationWarning).
    Returns ``None`` if neither source supplies the lookup — A3's
    strategy then rejects every signal (entry-features required).
    """
    if ctx is not None and ctx.per_trade_entry_features is not None:
        return ctx.per_trade_entry_features
    if cfg.per_trade_entry_features is not None:
        warnings.warn(
            "A3Config.per_trade_entry_features is deprecated; pass via "
            "A1RunContext.per_trade_entry_features. Will be removed "
            "after 2 closed arcs use the new path (chat directive Q2).",
            DeprecationWarning,
            stacklevel=3,
        )
        return cfg.per_trade_entry_features
    return None


__all__ = (
    "DEFAULT_N_DEFER_VALUES",
    "A3Config",
    "A3Architecture",
)
