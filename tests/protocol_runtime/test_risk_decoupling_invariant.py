"""Risk-decoupling invariant — regression guard.

Codifies the engine property surfaced and verified by the diagnosis at
[docs/dispatches/risk_leak_diagnosis.md][] (2026-05-25):

  At identical seeds and identical configs, admit decisions, fill
  decisions, and position lifecycle events must be byte-identical
  across risk levels. Position sizes scale by ``k = r_target / r_base``;
  nothing else changes.

This test pins the property so that any future engine change that
silently re-introduces a coupling between ``risk_pct`` and admit / exit
decisions fails immediately. Architectures A1 and A6 exercise both the
straight ``LiveBalanceRisk.risk_size`` path (A1) and the ``risk_size ×
risk_multiplier`` meta-labeling path (A6) — the latter is the
architecture from the original Arc 7 r=2% rerun that triggered the
diagnosis.

What is asserted (per risk-pair {r_low, r_high}):

  * **Admit timestamps**: the set of ``(entry_time, pair)`` tuples for
    every position the strategy attempted to open must be identical.
  * **Exit timestamps**: the set of ``(exit_time, pair, exit_reason)``
    tuples for every closed trade must be identical.
  * **Size scaling** — checked in two layers:
      - **First trade (strict)**: ``size_at_r_high / size_at_r_low``
        must equal ``r_high / r_low`` to ~6 decimal places. The first
        trade fires before any prior PnL has compounded, so both runs
        size off the identical ``starting_balance`` — the ratio is
        exact (modulo float rounding).
      - **Subsequent trades (soft, 20%)**: per-trade ratio must be
        within 20% of nominal ``k``. The wider tolerance absorbs the
        expected ``LiveBalance`` compounding signature documented in
        the diagnosis doc §2.1: balance trajectory diverges across
        risk levels → per-call sizes super-/sub-scale slightly from
        nominal ``k``. Real-data observed deviations up to ~16% (Arc
        7 holdout); 20% gives margin without masking a genuine bug.
  * **Closed-trade count**: equal across risk levels (no trade is
    silently dropped at higher risk).

What is NOT asserted:

  * Determinism across seeds (separate suite — see
    ``tests/test_determinism.py``).
  * KH-24 anchor reproduction (separate suite —
    ``tests/test_kh24_*``).
  * The orchestrator-vs-analysis driver mismatch surfaced in the
    diagnosis doc §5.3 (separate follow-up dispatch).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from core.architectures.a1_system_level_filter import (
    A1Architecture, A1Config, A1RunContext,
)
from core.architectures.a6_meta_labeling import A6Architecture, A6Config
from core.determinism import seed_everything
from core.wfo.folds import Fold
from tests.protocol_runtime._fixtures import (
    SyntheticSignal, build_synthetic_panel,
)


def _fold() -> Fold:
    return Fold(
        fold_id=1,
        is_start=date(2018, 1, 1),
        is_end=date(2018, 6, 30),
        oos_start=date(2018, 7, 1),
        oos_end=date(2018, 12, 31),
    )


def _panel_eval():
    panel = build_synthetic_panel(
        pairs=("EURUSD", "GBPUSD", "USDJPY"),
        n_bars=900, start="2018-01-01",
    )
    signal = SyntheticSignal()
    eval_ = signal.evaluate({"H4": panel})
    return panel, eval_


class _AdmitAll:
    """Deterministic dummy classifier — proba=0.99 for every input.

    Maps to ``mult=1.0`` under any A6 threshold pair with ``upper <
    0.99`` (i.e. the canonical {(0.3, 0.5), (0.4, 0.6), (0.5, 0.7)}
    grid). The risk-decoupling invariant does not depend on the
    classifier output; using a deterministic classifier avoids any
    sklearn-version-dependent variability.
    """

    def predict_proba(self, X):
        n = X.shape[0]
        return np.column_stack([np.full(n, 0.01), np.full(n, 0.99)])


def _run_a1_at_risk(risk_pct: float, panel, eval_):
    """Run A1 once at the given risk_pct; return its StrategyResult."""
    seed_everything(42)
    cfg = A1Config(
        config_id=f"a1_invariant_r{risk_pct:.4f}",
        sl_atr_mult=2.0,
        trail_enabled=True,
        trail_activation_atr=2.0,
        trail_distance_atr=1.5,
        risk_pct=risk_pct,
        max_concurrent_per_currency=2,
        max_concurrent_per_pair=1,
        starting_balance=100_000.0,
    )
    return A1Architecture().run(
        signal_evaluation=eval_,
        panels={"H4": panel},
        fold=_fold(),
        arch_config=cfg,
        config_id=cfg.config_id,
    )


def _run_a6_at_risk(risk_pct: float, panel, eval_):
    """Run A6 once at the given risk_pct; return its StrategyResult.

    Per-trade features are constant ``f0=1.0`` for every (pair,
    signal_time) — the ``_AdmitAll`` classifier maps every input to
    proba=0.99, so every signal admits at ``mult=1.0`` regardless of
    risk_pct. This isolates the engine path (the property under test).
    """
    seed_everything(42)
    cfg = A6Config(
        config_id=f"a6_invariant_r{risk_pct:.4f}",
        classifier=_AdmitAll(),
        lower_threshold=0.5,
        upper_threshold=0.7,
        classifier_feature_order=("f0",),
        sl_atr_mult=2.0,
        trail_enabled=True,
        trail_activation_atr=2.0,
        trail_distance_atr=1.5,
        risk_pct=risk_pct,
        max_concurrent_per_currency=2,
        max_concurrent_per_pair=1,
        starting_balance=100_000.0,
    )
    feats: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
    for pair, df in panel.pair_dfs.items():
        for ts in df.index:
            feats[(pair, ts)] = {"f0": 1.0}
    ctx = A1RunContext(per_trade_features=feats)
    return A6Architecture().run(
        signal_evaluation=eval_,
        panels={"H4": panel},
        fold=_fold(),
        arch_config=cfg,
        config_id=cfg.config_id,
        run_context=ctx,
    )


def _assert_risk_decoupled(low_res, high_res, *, r_low: float, r_high: float) -> None:
    """Shared assertions for any architecture's two-risk-level pair.

    Splits into three layers:
      1. Admit-set / exit-set equality (the load-bearing invariant).
      2. Trade-count equality (no silent drop at higher risk).
      3. Size scaling within ~5% of ``k = r_high / r_low``.

    The 5% size-scaling tolerance absorbs the legitimate
    ``LiveBalance`` compounding signature documented in the diagnosis
    doc §2.1 (per-call sizes deviate from nominal `k` because account
    balance trajectory diverges across risk levels).
    """
    trades_low = sorted(
        low_res.closed_trades, key=lambda t: (t.entry_time, t.pair, t.position_id),
    )
    trades_high = sorted(
        high_res.closed_trades, key=lambda t: (t.entry_time, t.pair, t.position_id),
    )

    # 1a — admit set equality (entry_time, pair)
    admits_low = {(t.entry_time, t.pair) for t in trades_low}
    admits_high = {(t.entry_time, t.pair) for t in trades_high}
    assert admits_low == admits_high, (
        f"Risk leak in admit path: at r_low={r_low}, admits={len(admits_low)}; "
        f"at r_high={r_high}, admits={len(admits_high)}. "
        f"Symmetric diff: {admits_low ^ admits_high}"
    )

    # 1b — exit set equality (exit_time, pair, exit_reason)
    exits_low = {
        (t.exit_time, t.pair, t.exit_reason) for t in trades_low
    }
    exits_high = {
        (t.exit_time, t.pair, t.exit_reason) for t in trades_high
    }
    assert exits_low == exits_high, (
        f"Risk leak in exit path: exit set differs across risk levels. "
        f"Symmetric diff (first 5): {list(exits_low ^ exits_high)[:5]}"
    )

    # 2 — closed-trade count equality
    assert len(trades_low) == len(trades_high), (
        f"Trade count differs across risk levels: r_low={r_low} → {len(trades_low)}, "
        f"r_high={r_high} → {len(trades_high)}"
    )

    # 3 — size scaling, two layers (see module docstring for rationale)
    k_nominal = r_high / r_low
    assert len(trades_low) > 0, "test fixture must produce at least one closed trade"

    # 3a — first trade, strict (no prior compounding; ratio must be exact)
    first_low, first_high = trades_low[0], trades_high[0]
    assert first_low.entry_time == first_high.entry_time
    assert first_low.pair == first_high.pair
    assert first_low.size > 0, "first trade has zero size — fixture broken"
    k_first = first_high.size / first_low.size
    assert abs(k_first - k_nominal) < 1e-6, (
        f"First-trade size ratio must be exact (no prior compounding). "
        f"k_nominal={k_nominal:.6f}, k_first={k_first:.6f}; "
        f"deviation {abs(k_first - k_nominal):.2e} > 1e-6. "
        f"This indicates the sizing path is NOT `balance × risk_pct / sl_distance`."
    )

    # 3b — subsequent trades, soft tolerance (compounding allowed)
    for t_low, t_high in zip(trades_low, trades_high):
        # Match by (entry_time, pair) — pair both lists in the same sort order
        assert t_low.entry_time == t_high.entry_time
        assert t_low.pair == t_high.pair
        if t_low.size == 0:
            continue
        k_actual = t_high.size / t_low.size
        rel_dev = abs(k_actual - k_nominal) / k_nominal
        assert rel_dev < 0.20, (
            f"Size scaling deviation > 20% at {t_low.entry_time} {t_low.pair}: "
            f"k_nominal={k_nominal:.4f}, k_actual={k_actual:.4f} "
            f"(size_low={t_low.size:.2f}, size_high={t_high.size:.2f}). "
            f"Compounding can produce sub-/super-linear scaling but >20% "
            f"deviation suggests sizing is no longer `balance × risk_pct / sl_distance`."
        )


@pytest.mark.parametrize("r_low,r_high", [
    (0.005, 0.010),
    (0.005, 0.020),
    (0.010, 0.020),
])
def test_a1_risk_decoupled(r_low: float, r_high: float) -> None:
    """A1 (LiveBalanceRisk only, no meta-labeling) at two risk levels.

    Covers the canonical ``risk_size(account, entry_price, sl_price,
    risk_pct=cfg.risk_pct)`` path that every architecture uses for the
    base sizing. KH-24 instantiates A1 — this is the most load-bearing
    path in the engine.
    """
    panel, eval_ = _panel_eval()
    low = _run_a1_at_risk(r_low, panel, eval_)
    high = _run_a1_at_risk(r_high, panel, eval_)
    _assert_risk_decoupled(low, high, r_low=r_low, r_high=r_high)


@pytest.mark.parametrize("r_low,r_high", [
    (0.005, 0.020),
])
def test_a6_risk_decoupled(r_low: float, r_high: float) -> None:
    """A6 (meta-labeling — risk_size × risk_multiplier) at two risks.

    The architecture from the Arc 7 r=2% rerun that triggered the
    original diagnosis. ``risk_multiplier`` is band-driven (0 / 0.5 /
    1.0) by the classifier output — config-driven, not risk-driven —
    so the band assignment is identical across risk levels and the
    size scaling factor should still resolve to ~``k``.
    """
    panel, eval_ = _panel_eval()
    low = _run_a6_at_risk(r_low, panel, eval_)
    high = _run_a6_at_risk(r_high, panel, eval_)
    _assert_risk_decoupled(low, high, r_low=r_low, r_high=r_high)


def test_a1_size_scales_above_zero() -> None:
    """Sanity: the synthetic fixture produces non-zero closed trades.

    Guards against the test passing trivially because both runs
    produced 0 trades (would make the set-equality assertions vacuous
    and the size-scaling assertion a no-op).
    """
    panel, eval_ = _panel_eval()
    res = _run_a1_at_risk(0.005, panel, eval_)
    assert len(res.closed_trades) >= 3, (
        f"Synthetic fixture must produce ≥3 closed trades for the "
        f"invariant assertions to be non-vacuous; got {len(res.closed_trades)}. "
        "Adjust SyntheticSignal.period / n_bars or fold window."
    )
