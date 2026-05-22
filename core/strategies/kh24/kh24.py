"""KH-24 strategy assembly: signal + filters + trail + kijun_d1 + risk.

Ties together every component into a ``StrategyFn`` callable that the
``MultiPairBacktester`` driver consumes. Caller flow:

    bt_config = KH24Config(...)
    runtime = build_kh24_runtime(panel_h4, panel_d1, panel_h1, bt_config)
    bt = MultiPairBacktester(
        panel=panel_h4,
        account=runtime.account,
        strategy=runtime.strategy,
        trail_manager=runtime.trail_manager,
        exit_predicates=runtime.exit_predicates,
    )
    result = bt.run()

The runtime owns:
  - The reset-floor accounting layer (consulted on every entry for size)
  - The trail manager (registers a TrailState for every new long)
  - The exit predicates (one ``kijun_d1`` predicate per pair)
  - The per-pair pre-computed signal masks + ATR / Kijun / lag-1 D1 series

Signal evaluation is precomputed ONCE per pair (vectorised) before
``run()`` — at runtime the strategy callable just does timestamp lookups.

Causal lineage: every component above is independently audited (signal
in tests/test_kh24_signal.py, D1 regime in tests/test_kh24_d1_regime.py,
H1 CIR in tests/test_kh24_h1_cir.py, kijun_d1 in tests/test_kh24_kijun_d1.py,
trail in tests/test_trailing_stop.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.sim.account import Account, Direction, ExposureRules
from core.sim.exit_hooks import ExitPredicate
from core.sim.multipair_backtester import Order, StrategyFn
from core.sim.panel import Panel
from core.sim.risk.live_balance import LiveBalanceRisk
from core.sim.trailing_stop import TrailManager
from core.strategies.kh24.exits.kijun_d1 import make_kijun_d1_exit_predicate
from core.strategies.kh24.filters.h1_cir import H1CIRParams, evaluate_h1_cir
from core.strategies.kh24.signal import KH24SignalParams, evaluate_kh24_signal


@dataclass(frozen=True)
class KH24Config:
    """Locked KH-24 deployment configuration.

    Matches the live EA (EA/KH24_EA.mq5) and the published lineage in
    ARC_HISTORY.md. Tests / probes can override; default values are
    the production lock.
    """

    signal: KH24SignalParams = field(default_factory=KH24SignalParams)
    h1_cir: H1CIRParams = field(default_factory=H1CIRParams)
    # Note: there is no separate D1 regime filter in this assembly. The EA's
    # ``EvalSignal`` enforces the D1 regime check via signal conditions
    # C8 (prev D1 close > prev D1 Kijun) and C9 (close ≤ Kijun + 1×ATR).
    # The standalone ``core.strategies.kh24.filters.d1_regime`` module is
    # retained for future arcs that want the gate without the bundled
    # C1-C7 conditions, but it's not used here. See PR-E.1.5 diff doc
    # Section C for the bisect evidence (Step 1 == Step 2 byte-identical).
    # SL = entry - sl_atr_mult × ATR(14) at entry
    sl_atr_mult: float = 2.0
    # Trailing stop: activate at +trail_activation_atr × ATR; trail behind highest close
    # by trail_distance_atr × ATR (bar-close updates only)
    trail_activation_atr: float = 2.0
    trail_distance_atr: float = 1.5
    # Risk: 1% of LIVE balance per trade — matches EA's
    # AccountInfoDouble(ACCOUNT_BALANCE) × RiskPercent/100 convention
    # (compounds with realised PnL). Per PR-E.1.6 diff doc Section E.
    risk_pct: float = 0.01
    starting_balance: float = 100_000.0
    # Exposure (per PR-E.1.6 diff doc Section F):
    #   - Per-currency cap = 2 (matches EA's ExposureCap=2 applied PER currency
    #     via CountCurrencyExposure)
    #   - Per-pair cap = 1 (implicit in EA via FindPosition gate)
    #   - NO total cap (EA does not impose one; total open positions can exceed 2
    #     as long as no single currency reaches the per-currency cap)
    # Prior misconfiguration (max_concurrent_total=2) was wildly more restrictive
    # than the EA and accounted for a large fraction of trade-count drift.
    exposure: ExposureRules = field(
        default_factory=lambda: ExposureRules(
            max_concurrent_total=None,
            max_concurrent_per_pair=1,
            max_concurrent_per_currency=2,
        )
    )


@dataclass
class _PerPairState:
    """Precomputed per-pair series the strategy reads at runtime."""

    signal_mask: pd.Series  # bool, indexed by H4 timestamp
    atr_h4: pd.Series  # float
    h1_cir: pd.Series  # bool
    exit_predicate: ExitPredicate
    h4_open_ask: pd.Series  # for entry-price computation


@dataclass
class KH24Runtime:
    """Pre-assembled runtime for the KH-24 strategy.

    ``strategy`` is the ``StrategyFn`` to hand to MultiPairBacktester;
    ``account``, ``trail_manager``, and ``exit_predicates`` are the
    same objects the driver should be constructed with.

    ``risk`` is the live-balance sizing layer (matches EA's
    ``AccountInfoDouble(ACCOUNT_BALANCE)`` convention). Sizing is
    consulted at signal time and reads the account's current balance.
    """

    account: Account
    risk: LiveBalanceRisk
    trail_manager: TrailManager
    exit_predicates: tuple[ExitPredicate, ...]
    strategy: StrategyFn
    config: KH24Config
    per_pair: dict[str, _PerPairState]


def _precompute_pair(
    pair: str,
    df_h4: pd.DataFrame,
    df_d1: pd.DataFrame,
    df_h1: pd.DataFrame,
    cfg: KH24Config,
) -> _PerPairState:
    """Run the signal + filters once per pair; return a runtime cache."""
    sig_result = evaluate_kh24_signal(df_h4, df_d1, params=cfg.signal)
    cir = evaluate_h1_cir(df_h4, df_h1, params=cfg.h1_cir)
    return _PerPairState(
        signal_mask=pd.Series(sig_result.signal_mask, index=df_h4.index),
        atr_h4=pd.Series(sig_result.atr_h4, index=df_h4.index),
        h1_cir=pd.Series(cir, index=df_h4.index),
        exit_predicate=make_kijun_d1_exit_predicate(
            pair, df_h4, df_d1, kijun_period=cfg.signal.d1_kijun_period
        ),
        h4_open_ask=df_h4["open_ask"].copy(),
    )


def build_kh24_runtime(
    panel_h4: Panel,
    panel_d1: Panel,
    panel_h1: Panel,
    config: KH24Config | None = None,
) -> KH24Runtime:
    """Precompute everything KH-24 needs; return a runnable runtime bundle.

    Three panels are required — same pair set in all three. The H4
    panel is the driver's iteration axis; D1 + H1 are auxiliary
    feeds for filters / exits.
    """
    cfg = config or KH24Config()
    if set(panel_h4.pairs) != set(panel_d1.pairs) or set(panel_h4.pairs) != set(panel_h1.pairs):
        raise ValueError("H4, D1, H1 panels must share the same pair set")

    per_pair: dict[str, _PerPairState] = {}
    for pair in panel_h4.pairs:
        per_pair[pair] = _precompute_pair(
            pair,
            panel_h4.pair_dfs[pair],
            panel_d1.pair_dfs[pair],
            panel_h1.pair_dfs[pair],
            cfg,
        )

    account = Account(starting_balance=cfg.starting_balance, exposure=cfg.exposure)
    risk = LiveBalanceRisk(risk_pct=cfg.risk_pct)
    trail_manager = TrailManager()
    exit_predicates = tuple(per_pair[p].exit_predicate for p in sorted(per_pair))

    def strategy(
        t: pd.Timestamp,
        snapshot: dict[str, pd.Series | None],
        acct: Account,
    ) -> list[Order]:
        orders: list[Order] = []
        for pair in sorted(per_pair):  # deterministic order
            state = per_pair[pair]
            if t not in state.signal_mask.index:
                continue
            # Read precomputed booleans at this H4 timestamp.
            # (No separate D1 regime gate — signal C8+C9 enforce identical logic.
            # See PR-E.1.5 diff doc Section C.)
            if not bool(state.signal_mask.loc[t]):
                continue
            if not bool(state.h1_cir.loc[t]):
                continue
            bar = snapshot.get(pair)
            if bar is None:
                continue
            atr_val = state.atr_h4.loc[t]
            if pd.isna(atr_val) or atr_val <= 0:
                continue
            # SL anchored to the realised entry price post-fill is a
            # future fix (PR-E.1.6 diff doc Section H DEFERRED). For now,
            # use this bar's close_ask as a proxy for next-bar open_ask
            # entry. KH-24 EA pattern: SL = entry - 2*ATR(14).
            entry_proxy = float(bar["close_ask"])
            sl_price = entry_proxy - cfg.sl_atr_mult * float(atr_val)
            # Size from LIVE balance (compounds with PnL) — matches EA's
            # AccountInfoDouble(ACCOUNT_BALANCE) × RiskPercent/100.
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
                    atr_at_entry=float(atr_val),
                    trail_activation_atr=cfg.trail_activation_atr,
                    trail_distance_atr=cfg.trail_distance_atr,
                )
            )
        return orders

    return KH24Runtime(
        account=account,
        risk=risk,
        trail_manager=trail_manager,
        exit_predicates=exit_predicates,
        strategy=strategy,
        config=cfg,
        per_pair=per_pair,
    )
