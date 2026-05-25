"""A1 — System-level filter architecture.

Per L_PROTOCOL §2 Step 5: base signal + optional rule-based filter from
Step 4 + exposure cap + searched SL/exit policy. No classifier.

KH-24 IS an A1 instantiation: the signal is KH24SignalModule (carrying
C1-C9 + H1 CIR + kijun_d1 exit predicate); the A1Config layers on
SL=2.0×ATR, trail (activation=2.0, distance=1.5), per-currency
exposure cap=2, live-balance 1% risk. The anchor test confirms
equivalence.

Other arcs instantiate A1 with their own SignalModule + filter rules
(optional list of column-threshold predicates) + exposure / SL / exit
config. The runtime wires:

  1. Per-pair signal + atr lookup
  2. AND with every additional gate from the signal module
  3. AND with any external rule-based filter from the config
  4. Build SL = signal_bar.close_ask − sl_atr_mult × ATR
  5. Size from the configured risk model
  6. Emit Order at signal-bar close (driver fills at next-bar open)

Trail and signal-class exit predicates pass through to
MultiPairBacktester unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from core.arc.signal_protocol import SignalEvaluation
from core.architectures._protocol import StrategyResult
from core.sim.account import Account, Direction, ExposureRules
from core.sim.exit_hooks import ExitPredicate
from core.sim.exit_policy_manager import ExitPolicyManager
from core.sim.multipair_backtester import MultiPairBacktester, Order, StrategyFn
from core.sim.panel import Panel
from core.sim.risk.live_balance import LiveBalanceRisk
from core.sim.trailing_stop import TrailManager
from core.wfo.folds import Fold

# A filter predicate maps (feature_value_per_pair_at_t) -> bool. v3.0 keeps
# the filter language tiny: a tuple of (feature_name, "ge"|"le", threshold).
# Rule semantics: rule passes iff feature(t) >= threshold (ge) or <=
# threshold (le). All rules ANDed together.
FilterRule = tuple[str, str, float]


@dataclass(frozen=True)
class A1Config:
    """Architecture-level config for A1.

    The signal layer is supplied separately (via SignalEvaluation); A1
    only adds the system-level wrapper around it.
    """

    config_id: str
    sl_atr_mult: float = 2.0
    # Trailing-stop params. Set ``trail_enabled=False`` to disable.
    trail_enabled: bool = True
    trail_activation_atr: float = 2.0
    trail_distance_atr: float = 1.5
    # External filter rules from Step 4 (None / empty list = no filter).
    # Per-trade features must be available in ``per_trade_features``
    # (a DataFrame indexed by entry_time or signal_time per pair).
    filter_rules: tuple[FilterRule, ...] = ()
    # Sizing model — live_balance (KH-24 default) or fixed
    risk_pct: float = 0.005
    starting_balance: float = 100_000.0
    # Exposure caps
    max_concurrent_total: int | None = None
    max_concurrent_per_pair: int | None = 1
    max_concurrent_per_currency: int | None = 2
    # Hold-bars cap as a fallback time exit (None = use signal exit only)
    time_exit_bars: int | None = None
    # Amendment 3 §"Sizing convention": linear DD scaling holds ONLY
    # under reset-floor sizing. equity_pct sizing FAILs the scalability
    # gate by default; chat must approve a separate scaling treatment
    # via ``ArcConfig.accept_equity_pct = True``.
    sizing_convention: str = "reset_floor"   # "reset_floor" | "equity_pct"
    # Canonical exit-policy name from core.sim.exit_policies registry.
    # None (default) preserves prior behaviour: SL + optional trail +
    # signal-class exit predicates only. When set, the architecture
    # instantiates an ExitPolicyManager and the driver registers the
    # policy at trade fill; the manager's per-bar hooks then handle
    # TP / trailing / partial-close lifecycle per
    # [docs/PROTOCOL_RUNTIME.md §8c][]. KH-24 uses None (its trail +
    # kijun_d1 path is unchanged).
    exit_policy: str | None = None


@dataclass(frozen=True)
class A1RunContext:
    """Optional precomputed per-trade feature lookup for filter rules
    plus A3/A4 per-fold classifier orchestration data.

    The same context object is shared across every fold of a WFO run.
    Each architecture reads only the fields it needs; A1 / A5 ignore
    everything but ``per_trade_features``.

    Fields
    ------
    per_trade_features
        Map ``(pair, signal_time) -> feature dict`` consumed by A1
        filter rules and A2 / A6 classifier admit / sizing gates.
        ``None`` means no external rules are applied (filter_rules
        ignored). Provided by the orchestrator after Step 1 produces
        the per-trade feature matrix.

    per_trade_entry_features
        Same shape as ``per_trade_features`` but populated with
        entry-time features for A3 / A4 path-classifier inference
        (the 8 ``ENTRY_FEATURE_KEYS`` from ``core.features_path_so_far``).
        ``None`` means callers must supply these via the deprecated
        ``A3Config.per_trade_entry_features`` /
        ``A4Config.per_trade_entry_features`` field (emits a
        DeprecationWarning at run-time).

    path_classifier_fits
        Map ``fold_id -> PathClassifierFit`` carrying the per-fold-
        trained path-classifier for A3 / A4. Keyed by ``Fold.fold_id``.
        Built once by the orchestrator (via
        ``core.steps.path_classifier_per_fold.build_path_classifier_fits_per_fold``)
        and reused across the WFO search loop. ``None`` means callers
        must supply a single fit via the deprecated
        ``A3Config.classifier_fit`` / ``A4Config.classifier_fit``
        field (emits a DeprecationWarning at run-time).

    Deprecation: the per-fold path is the canonical way to wire A3 / A4
    from the orchestrator. The single-fit fields on ``A3Config`` /
    ``A4Config`` remain available for backwards compatibility with
    synthetic tests and direct-construction drivers, but warn on use.
    Per chat directive Q2 they are scheduled for removal after 2
    closed arcs use the new path successfully.
    """

    per_trade_features: Mapping[tuple[str, pd.Timestamp], Mapping[str, float]] | None = None
    per_trade_entry_features: Mapping[tuple[str, pd.Timestamp], Mapping[str, float]] | None = None
    # Forward type hint to break a circular import — the actual type is
    # core.architectures._path_classifier.PathClassifierFit but importing
    # it here would pull RF defaults into the A1 module's import graph.
    path_classifier_fits: Mapping[int, object] | None = None


def _evaluate_filter_rules(
    rules: Sequence[FilterRule],
    feats: Mapping[str, float] | None,
) -> bool:
    """Return True iff every rule passes against ``feats``.

    Missing feature, NaN, or feats=None -> False (conservative — reject).
    """
    if not rules:
        return True
    if feats is None:
        return False
    for name, op, threshold in rules:
        v = feats.get(name)
        if v is None or pd.isna(v):
            return False
        if op == "ge":
            if not (v >= threshold):
                return False
        elif op == "le":
            if not (v <= threshold):
                return False
        else:
            raise ValueError(f"unknown filter op {op!r}; expected ge|le")
    return True


def _build_a1_strategy(
    signal_eval: SignalEvaluation,
    panels: Mapping[str, Panel],
    cfg: A1Config,
    ctx: A1RunContext,
    account: Account,
    risk: LiveBalanceRisk,
) -> StrategyFn:
    """Construct the StrategyFn closure for A1 over the fold's panels."""
    primary_panel = panels[signal_eval.primary_tf]
    per_pair = signal_eval.per_pair

    # Reindex per-pair series onto the fold-sliced primary panel
    series_cache: dict[str, dict[str, pd.Series]] = {}
    for pair, state in per_pair.items():
        df = primary_panel.pair_dfs.get(pair)
        if df is None:
            continue
        mask = state.signal_mask.reindex(df.index).fillna(False).astype(bool)
        atr = state.atr.reindex(df.index)
        gates: dict[str, pd.Series] = {}
        for k, v in state.additional_gates.items():
            gates[k] = v.reindex(df.index).fillna(False).astype(bool)
        series_cache[pair] = {"mask": mask, "atr": atr, **gates}

    def strategy(
        t: pd.Timestamp,
        snapshot: dict[str, pd.Series | None],
        acct: Account,
    ) -> list[Order]:
        orders: list[Order] = []
        for pair in sorted(series_cache):
            cached = series_cache[pair]
            mask = cached["mask"]
            if t not in mask.index:
                continue
            if not bool(mask.loc[t]):
                continue
            # AND any additional gates
            gate_ok = True
            for k, ser in cached.items():
                if k in ("mask", "atr"):
                    continue
                if not bool(ser.loc[t]):
                    gate_ok = False
                    break
            if not gate_ok:
                continue
            # External rule filter
            if cfg.filter_rules:
                feats = (
                    ctx.per_trade_features.get((pair, t))
                    if ctx.per_trade_features is not None
                    else None
                )
                if not _evaluate_filter_rules(cfg.filter_rules, feats):
                    continue
            # ATR + entry proxy
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
            size = risk.risk_size(
                acct, entry_price=entry_proxy, sl_price=sl_price, risk_pct=cfg.risk_pct
            )
            orders.append(
                Order(
                    pair=pair,
                    direction=Direction.LONG,
                    size=size,
                    sl_price=sl_price,
                    tp_price=None,
                    atr_at_entry=atr,
                    trail_activation_atr=cfg.trail_activation_atr,
                    trail_distance_atr=cfg.trail_distance_atr,
                    exit_policy=cfg.exit_policy,
                    sl_atr_mult=cfg.sl_atr_mult if cfg.exit_policy else None,
                )
            )
        return orders

    return strategy


def _slice_panels_to_fold(
    panels: Mapping[str, Panel], fold: Fold, warmup_days: int = 60
) -> dict[str, Panel]:
    """Slice each TF panel to [oos_start - warmup, oos_end] for the fold."""
    out: dict[str, Panel] = {}
    start_ts = pd.Timestamp(fold.oos_start, tz="UTC") - pd.Timedelta(days=warmup_days)
    end_ts = pd.Timestamp(fold.oos_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    for tf, panel in panels.items():
        sliced = {p: df.loc[start_ts:end_ts] for p, df in panel.pair_dfs.items()}
        out[tf] = Panel.from_frames(sliced, tf=panel.tf, boundary_convention=panel.boundary_convention)
    return out


@dataclass(frozen=True)
class A1Architecture:
    """A1 implementation conforming to :class:`core.architectures._protocol.Architecture`."""

    architecture_name: str = "A1"

    def run(
        self,
        *,
        signal_evaluation: SignalEvaluation,
        panels: Mapping[str, Panel],
        fold: Fold,
        arch_config: A1Config,
        config_id: str,
        run_context: A1RunContext | None = None,
    ) -> StrategyResult:
        ctx = run_context or A1RunContext()
        # Re-evaluate the signal on the fold-sliced primary panel? No — the
        # arc-pool builder already evaluated against the full panel. For
        # WFO folds we restrict via timestamp masking inside the strategy
        # closure. Slicing the panel still happens for the driver loop.
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
        # Canonical exit-policy manager only when this config selects one
        exit_policy_manager = (
            ExitPolicyManager() if arch_config.exit_policy is not None else None
        )

        # Signal-class-inherent exit predicates from the SignalModule
        exit_predicates: list[ExitPredicate] = []
        for pair in sorted(signal_evaluation.per_pair):
            ep = signal_evaluation.per_pair[pair].exit_predicate
            if ep is not None:
                exit_predicates.append(ep)

        strategy = _build_a1_strategy(
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
            exit_policy_manager=exit_policy_manager,
        )
        run_result = bt.run()

        # Build FoldStats restricting equity to OOS window
        from core.runners._fold_stats_helpers import build_fold_stats_from_run
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
                "sl_atr_mult": arch_config.sl_atr_mult,
                "trail_enabled": arch_config.trail_enabled,
                "n_filter_rules": len(arch_config.filter_rules),
                "exit_policy": arch_config.exit_policy,
                "exposure": (
                    arch_config.max_concurrent_total,
                    arch_config.max_concurrent_per_pair,
                    arch_config.max_concurrent_per_currency,
                ),
            },
        )


def _slice_equity_to_oos(equity: pd.Series, fold: Fold) -> pd.Series:
    if len(equity) == 0:
        return equity
    oos_start_ts = pd.Timestamp(fold.oos_start, tz="UTC")
    oos_end_ts = pd.Timestamp(fold.oos_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return equity.loc[oos_start_ts:oos_end_ts]


__all__ = (
    "FilterRule",
    "A1Config",
    "A1RunContext",
    "A1Architecture",
)
