"""Locked regression: FundedNext costs are netted at the gate-scoring layer.

Resolves HONEST_ENGINE_SWEEP.md Part C (FAIL): the cost primitives were
orphaned, so every gate verdict scored on raw bid/ask. This pins the wiring:

  - ``apply_cost_model`` computes per-position cost from the GROSS ledger using
    the existing primitives (commission $5/lot RT, slippage 0.5 pip/fill x
    n_fills, spread 1.5x extra), with the correct unit conversions.
  - n_fills tracks the ACTUAL SL-honest leg structure (3 if the +1R partial
    fired, else 2).
  - ``build_fold_stats_from_run`` nets by DEFAULT — a cost-free FoldStats
    requires an EXPLICIT ``CostModel.zero()``.
  - swaps OFF; NaN bid/ask -> zero spread (no equity poisoning); deterministic.

Engine-level pnl/equity stay GROSS (Account is never touched), so the
take-the-loss invariant and determinism fixtures are unaffected.

Numpy/pandas + stdlib only -> runs under CI's ``pytest -m "not research"``.
"""

from __future__ import annotations

import hashlib
from datetime import date

import numpy as np
import pandas as pd
import pytest

from core.runners._fold_stats_helpers import build_fold_stats_from_run
from core.sim.account import ClosedTrade, Direction
from core.sim.costs.model import CostModel, apply_cost_model
from core.sim.multipair_backtester import RunResult
from core.wfo.folds import Fold

_PIP = 0.0001  # EURUSD


def _leg(
    *,
    position_id: int,
    size: float,
    entry_price: float,
    exit_price: float,
    pnl: float,
    exit_time: str,
    parent_position_id: int | None = None,
    entry_spread: float = _PIP,
    exit_spread: float = _PIP,
    pair: str = "EURUSD",
) -> ClosedTrade:
    """A gross ClosedTrade leg with symmetric bid/ask around the fill prices."""
    return ClosedTrade(
        position_id=position_id,
        pair=pair,
        direction=Direction.LONG,
        entry_time=pd.Timestamp("2020-01-01", tz="UTC"),
        entry_price=entry_price,
        exit_time=pd.Timestamp(exit_time, tz="UTC"),
        exit_price=exit_price,
        size=size,
        pnl=pnl,
        exit_reason="x",
        parent_position_id=parent_position_id,
        entry_bid=entry_price - entry_spread / 2.0,
        entry_ask=entry_price + entry_spread / 2.0,
        exit_bid=exit_price - exit_spread / 2.0,
        exit_ask=exit_price + exit_spread / 2.0,
        sl_price=None,
    )


def _run_result(legs: tuple[ClosedTrade, ...], equity: pd.Series | None = None) -> RunResult:
    if equity is None:
        equity = pd.Series([], dtype="float64", name="equity")
    return RunResult(
        final_balance=float(equity.iloc[-1]) if len(equity) else 100_000.0,
        n_trades=len(legs),
        n_open_at_end=0,
        equity_curve=equity,
        max_drawdown_pct=0.0,
        closed_trades=legs,
    )


# ── cost math: a clean-win partial (3 fills) ─────────────────────────────


def test_partial_win_three_fills_exact_haircut() -> None:
    """+1R partial + runner = 2 legs => 3 fills; exact commission+slippage+spread."""
    legs = (
        _leg(position_id=1, parent_position_id=1, size=5_000.0,
             entry_price=1.10000, exit_price=1.10200, pnl=10.0,
             exit_time="2020-01-02"),
        _leg(position_id=1, parent_position_id=1, size=5_000.0,
             entry_price=1.10000, exit_price=1.10100, pnl=5.0,
             exit_time="2020-01-03"),
    )
    costed = apply_cost_model(_run_result(legs), CostModel.fundednext())
    assert len(costed.breakdown) == 1
    row = costed.breakdown.iloc[0]

    # original_size 10_000 = 0.1 lot; partial fired -> 3 fills.
    assert row["n_legs"] == 2
    assert row["n_fills"] == 3
    assert row["original_size"] == pytest.approx(10_000.0)
    # commission: 0.1 lot x $5 = $0.50
    assert row["commission"] == pytest.approx(0.5)
    # slippage: 0.5 pip x 3 fills = 1.5 pip = 0.00015 price x 10_000 = $1.50
    assert row["slippage"] == pytest.approx(1.5)
    # spread: per leg (1pip+1pip)x(1.5-1)=0.0001 price x 5_000 = $0.50 ; x2 legs = $1.00
    assert row["spread"] == pytest.approx(1.0)
    assert row["total_cost"] == pytest.approx(3.0)
    assert row["gross_pnl"] == pytest.approx(15.0)
    assert row["net_pnl"] == pytest.approx(12.0)
    assert costed.cost_total == pytest.approx(3.0)
    assert costed.net_pnl_total == pytest.approx(12.0)


# ── cost math: a take-the-loss (2 fills) ─────────────────────────────────


def test_take_the_loss_two_fills_exact_haircut() -> None:
    """Stop before any partial = 1 leg => 2 fills; costs still applied."""
    legs = (
        _leg(position_id=7, parent_position_id=None, size=10_000.0,
             entry_price=1.10000, exit_price=1.09800, pnl=-20.0,
             exit_time="2020-01-02"),
    )
    costed = apply_cost_model(_run_result(legs), CostModel.fundednext())
    row = costed.breakdown.iloc[0]
    assert row["n_legs"] == 1
    assert row["n_fills"] == 2  # entry + stop, partial never fired
    # commission 0.5 ; slippage 0.5pip x2 = 0.0001 x 10_000 = 1.0 ; spread 0.0001 x 10_000 = 1.0
    assert row["commission"] == pytest.approx(0.5)
    assert row["slippage"] == pytest.approx(1.0)
    assert row["spread"] == pytest.approx(1.0)
    assert row["total_cost"] == pytest.approx(2.5)
    assert row["gross_pnl"] == pytest.approx(-20.0)
    assert row["net_pnl"] == pytest.approx(-22.5)  # a loss is made worse, never flattered


# ── default-on guarantee at the gate-scoring layer ───────────────────────


def _fold() -> Fold:
    return Fold(
        fold_id=1,
        is_start=date(2019, 12, 1),
        is_end=date(2019, 12, 31),
        oos_start=date(2020, 1, 1),
        oos_end=date(2020, 1, 31),
    )


def _gate_inputs() -> RunResult:
    idx = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="UTC") for d in ("2020-01-02", "2020-01-03", "2020-01-04")],
        name="timestamp_utc",
    )
    equity = pd.Series([100_000.0, 100_010.0, 100_015.0], index=idx, name="equity")
    legs = (
        _leg(position_id=1, parent_position_id=1, size=5_000.0,
             entry_price=1.10000, exit_price=1.10200, pnl=10.0, exit_time="2020-01-03"),
        _leg(position_id=1, parent_position_id=1, size=5_000.0,
             entry_price=1.10000, exit_price=1.10100, pnl=5.0, exit_time="2020-01-03"),
    )
    return _run_result(legs, equity)


def test_gate_run_cannot_be_cost_free_by_default() -> None:
    """build_fold_stats_from_run with no cost_model nets FundedNext costs."""
    rr = _gate_inputs()
    default_fs = build_fold_stats_from_run(fold=_fold(), run_result=rr, starting_balance=100_000.0)
    explicit_fn = build_fold_stats_from_run(
        fold=_fold(), run_result=rr, starting_balance=100_000.0, cost_model=CostModel.fundednext()
    )
    cost_free = build_fold_stats_from_run(
        fold=_fold(), run_result=rr, starting_balance=100_000.0, cost_model=CostModel.zero()
    )
    # The default IS FundedNext.
    assert default_fs.roi_pct == pytest.approx(explicit_fn.roi_pct)
    # Costs strictly reduce ROI vs an explicit cost-free run.
    assert default_fs.roi_pct < cost_free.roi_pct
    # n_trades (count) is unchanged by costs.
    assert default_fs.n_trades == cost_free.n_trades


def test_zero_model_equals_gross() -> None:
    rr = _gate_inputs()
    costed = apply_cost_model(rr, CostModel.zero())
    pd.testing.assert_series_equal(costed.net_equity, rr.equity_curve)
    assert costed.cost_total == pytest.approx(0.0)


# ── robustness: NaN bid/ask, swaps off, determinism ──────────────────────


def test_nan_bid_ask_yields_zero_spread_and_finite_net() -> None:
    """Synthetic/legacy panels carry NaN bid/ask -> zero spread, finite equity."""
    leg = ClosedTrade(
        position_id=3, pair="EURUSD", direction=Direction.LONG,
        entry_time=pd.Timestamp("2020-01-01", tz="UTC"), entry_price=1.10000,
        exit_time=pd.Timestamp("2020-01-02", tz="UTC"), exit_price=1.10100,
        size=10_000.0, pnl=10.0, exit_reason="x", parent_position_id=None,
        entry_bid=float("nan"), entry_ask=float("nan"),
        exit_bid=float("nan"), exit_ask=float("nan"), sl_price=None,
    )
    idx = pd.DatetimeIndex([pd.Timestamp("2020-01-02", tz="UTC")], name="timestamp_utc")
    equity = pd.Series([100_010.0], index=idx, name="equity")
    costed = apply_cost_model(_run_result((leg,), equity), CostModel.fundednext())
    row = costed.breakdown.iloc[0]
    assert row["spread"] == pytest.approx(0.0)        # NaN bid/ask -> 0, not NaN
    assert row["commission"] == pytest.approx(0.5)    # still applied
    assert row["slippage"] == pytest.approx(1.0)
    assert bool(np.isfinite(costed.net_equity).all())  # equity not poisoned


def test_swaps_enabled_is_rejected() -> None:
    """Swaps are intentionally OFF; a swaps-on model must fail loud, not half-wire."""
    with pytest.raises(NotImplementedError):
        apply_cost_model(_run_result(()), CostModel(swaps_enabled=True))


def _sha(series: pd.Series) -> str:
    return hashlib.sha256(
        series.to_csv(lineterminator="\n", float_format="%.10g").encode("utf-8")
    ).hexdigest()


def test_determinism_two_run_identity_with_costs_on() -> None:
    rr = _gate_inputs()
    a = apply_cost_model(rr, CostModel.fundednext())
    b = apply_cost_model(rr, CostModel.fundednext())
    assert _sha(a.net_equity) == _sha(b.net_equity)
