"""End-to-end KH-24 strategy integration tests.

Runs the full assembled KH-24 strategy (signal + filters + trail +
kijun_d1 + reset-floor risk) against a synthetic Panel + the
MultiPairBacktester driver. Tests focus on:

  - The strategy callable runs to completion without crashing
  - Trailing stop registers on every long entry
  - Exit predicates registered correctly per pair
  - Two-run determinism preserved
  - Single-pair single-month + multi-pair single-month sanity runs
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.sim.multipair_backtester import MultiPairBacktester
from core.sim.panel import Panel
from core.strategies.kh24.kh24 import KH24Config, build_kh24_runtime
from tests.fixtures.histdata_mini.build import FixtureSpec, build_fixture


@pytest.fixture
def mini_root(tmp_path: Path) -> Path:
    """Two months of synthetic M1 — enough to aggregate H4 + D1 + H1."""
    return build_fixture(
        tmp_path / "histdata",
        FixtureSpec(
            pairs=("EURUSD", "GBPUSD", "USDJPY"),
            months=("201001", "201002"),
            minutes_per_month=1440 * 28,  # ~28 days of M1 per month
        ),
    )


@pytest.fixture
def panels(mini_root: Path, tmp_path: Path) -> tuple[Panel, Panel, Panel]:
    cache = tmp_path / "cache"
    pairs = ["EURUSD", "GBPUSD", "USDJPY"]
    h4 = Panel.from_pairs(pairs, "H4", histdata_root=mini_root, cache_root=cache)
    d1 = Panel.from_pairs(pairs, "D1", histdata_root=mini_root, cache_root=cache)
    h1 = Panel.from_pairs(pairs, "H1", histdata_root=mini_root, cache_root=cache)
    return h4, d1, h1


def test_kh24_runtime_builds_without_error(panels: tuple[Panel, Panel, Panel]) -> None:
    h4, d1, h1 = panels
    runtime = build_kh24_runtime(h4, d1, h1)
    assert runtime.config.signal.atr_period == 14
    assert runtime.account.starting_balance == 100_000.0
    # Per PR-E.1.6 diff doc Section F: EA caps per-currency at 2, no total cap.
    assert runtime.config.exposure.max_concurrent_total is None
    assert runtime.config.exposure.max_concurrent_per_currency == 2
    assert runtime.config.exposure.max_concurrent_per_pair == 1
    # One exit predicate per pair
    assert len(runtime.exit_predicates) == len(h4.pairs)


def test_kh24_runtime_rejects_pair_mismatch(panels: tuple[Panel, Panel, Panel]) -> None:
    h4, d1, h1 = panels
    # Slice H1 to a smaller pair set
    h1_subset = Panel.from_frames({"EURUSD": h1.pair_dfs["EURUSD"]}, tf="H1")
    with pytest.raises(ValueError, match="same pair set"):
        build_kh24_runtime(h4, d1, h1_subset)


def test_kh24_e2e_single_pair_no_crash(mini_root: Path, tmp_path: Path) -> None:
    """Smallest possible end-to-end run: 1 pair, synthetic data, full strategy."""
    cache = tmp_path / "cache"
    h4 = Panel.from_pairs(["EURUSD"], "H4", histdata_root=mini_root, cache_root=cache)
    d1 = Panel.from_pairs(["EURUSD"], "D1", histdata_root=mini_root, cache_root=cache)
    h1 = Panel.from_pairs(["EURUSD"], "H1", histdata_root=mini_root, cache_root=cache)

    runtime = build_kh24_runtime(h4, d1, h1)
    bt = MultiPairBacktester(
        panel=h4,
        account=runtime.account,
        strategy=runtime.strategy,
        trail_manager=runtime.trail_manager,
        exit_predicates=runtime.exit_predicates,
    )
    result = bt.run()
    # Sanity: runs to completion, equity curve produced
    assert len(result.equity_curve) == len(h4.timestamps)


def test_kh24_e2e_multi_pair_no_crash(
    panels: tuple[Panel, Panel, Panel],
) -> None:
    """Three pairs, full strategy, run to completion."""
    h4, d1, h1 = panels
    runtime = build_kh24_runtime(h4, d1, h1)
    bt = MultiPairBacktester(
        panel=h4,
        account=runtime.account,
        strategy=runtime.strategy,
        trail_manager=runtime.trail_manager,
        exit_predicates=runtime.exit_predicates,
    )
    result = bt.run()
    # Equity curve has one point per H4 bar
    assert len(result.equity_curve) == len(h4.timestamps)


def test_kh24_e2e_two_run_determinism(
    panels: tuple[Panel, Panel, Panel],
) -> None:
    """Two runs from identical inputs produce identical equity curves."""
    h4, d1, h1 = panels

    def go():
        runtime = build_kh24_runtime(h4, d1, h1)
        bt = MultiPairBacktester(
            panel=h4,
            account=runtime.account,
            strategy=runtime.strategy,
            trail_manager=runtime.trail_manager,
            exit_predicates=runtime.exit_predicates,
        )
        return bt.run()

    a = go()
    b = go()
    pd.testing.assert_series_equal(a.equity_curve, b.equity_curve)
    assert a.final_balance == b.final_balance
    assert a.n_trades == b.n_trades
    # Closed-trade ledger byte-identical
    assert len(a.closed_trades) == len(b.closed_trades)
    for ta, tb in zip(a.closed_trades, b.closed_trades):
        assert ta == tb


def test_kh24_exposure_cap_respected(
    panels: tuple[Panel, Panel, Panel],
) -> None:
    """End-of-run open positions never exceed max_concurrent_total=2."""
    h4, d1, h1 = panels
    runtime = build_kh24_runtime(h4, d1, h1)
    bt = MultiPairBacktester(
        panel=h4,
        account=runtime.account,
        strategy=runtime.strategy,
        trail_manager=runtime.trail_manager,
        exit_predicates=runtime.exit_predicates,
    )
    bt.run()
    assert len(runtime.account.open_positions) <= 2


def test_kh24_config_locked_defaults() -> None:
    """KH24Config defaults match the published lineage spec."""
    cfg = KH24Config()
    assert cfg.signal.atr_period == 14
    assert cfg.signal.kijun_period == 26
    assert cfg.signal.long_body_threshold == 0.5
    assert cfg.signal.long_close_position_max == 0.24
    assert cfg.signal.c5_distance_cap_atr == 1.0
    assert cfg.signal.c6_depth_bars == 10
    assert cfg.signal.c6_depth_threshold == 0.5
    assert cfg.h1_cir.threshold == 0.28
    assert cfg.sl_atr_mult == 2.0
    assert cfg.trail_activation_atr == 2.0
    assert cfg.trail_distance_atr == 1.5
    assert cfg.risk_pct == 0.01
    assert cfg.starting_balance == 100_000.0
    # PR-E.1.6 §F: per-currency cap=2 matches EA; no total cap
    assert cfg.exposure.max_concurrent_total is None
    assert cfg.exposure.max_concurrent_per_currency == 2
    assert cfg.exposure.max_concurrent_per_pair == 1
