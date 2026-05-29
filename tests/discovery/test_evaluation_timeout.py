"""Tests for arc_discovery_02 Amendment C — bar-iteration cap (evaluation_timeout).

The cap is a deterministic counter: total bar-iterations summed across pairs
within one rule's evaluation. When the count reaches the budget at a TRADE
BOUNDARY (start of next trade in same pair), the simulator returns
``aborted=True`` and the search loop records ``evaluation_timeout=True`` for
that rule.

These tests use very small iteration budgets so the cap fires reliably on
synthetic data without needing realistic compute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.discovery.causal_filter import clean_feature_pool
from core.discovery.grammar import GrammarConfig
from core.discovery.pool_simulator import DiscoveryExitConfig, simulate_pair_pool
from core.discovery.quantile_grid import build_quantile_grid
from core.discovery.random_search import PairFixture, SearchConfig, run_search
from tests.discovery.test_pool_simulator import _atr, _ohlc, _signal_at


def test_iteration_budget_aborts_pair_mid_run():
    """A pair with many trades + low iteration budget aborts before completing."""
    # Build 60 bars; trigger on every other bar from 0-40 to generate ~20 trades.
    bars = []
    for i in range(60):
        ts = pd.Timestamp("2020-01-01T00", tz="UTC") + pd.Timedelta(hours=i)
        bars.append({
            "ts": ts.isoformat(),
            "open_ask": 100.0 + (i * 0.01),
            "high_bid": 100.5 + (i * 0.01),
            "low_bid": 99.5 + (i * 0.01),
            "close_bid": 100.0 + (i * 0.01),
        })
    df = _ohlc(bars)
    # Trigger on every-other-bar (15 triggers)
    arr = np.zeros(len(df.index), dtype=bool)
    arr[0:30:2] = True
    trigger = pd.Series(arr, index=df.index)
    atr = _atr(df.index, 1.0)

    cfg = DiscoveryExitConfig(
        initial_sl_atr_mult=2.0,
        trail_activation_atr_mult=4.0,
        trail_distance_atr_mult=2.0,
        primary_tf_warmup_bars=0,
        time_exit_bars=10,    # bound each trade at 10 bars
    )

    # Budget = 50 iterations: enough for ~5 trades × 10 bars each, then abort.
    trades, _, iters, aborted = simulate_pair_pool(
        "EURUSD", df, trigger, atr, cfg, iteration_budget=50
    )
    # Should have aborted (15 triggers, each ~10 iterations → ~150 total > 50)
    assert aborted is True
    # Some trades completed before the abort
    assert len(trades) > 0
    # iterations_consumed should be >= budget (we abort AT or above budget)
    assert iters >= 50


def test_iteration_budget_None_means_no_cap():
    """iteration_budget=None preserves arc_discovery_01 behaviour — runs to completion."""
    bars = []
    for i in range(40):
        ts = pd.Timestamp("2020-01-01T00", tz="UTC") + pd.Timedelta(hours=i)
        bars.append({
            "ts": ts.isoformat(), "open_ask": 100.0, "high_bid": 100.5,
            "low_bid": 99.5, "close_bid": 100.0,
        })
    df = _ohlc(bars)
    trigger = _signal_at(df.index, 0)
    atr = _atr(df.index, 1.0)
    cfg = DiscoveryExitConfig(
        initial_sl_atr_mult=2.0, trail_activation_atr_mult=4.0,
        trail_distance_atr_mult=2.0, primary_tf_warmup_bars=0,
        time_exit_bars=None,
    )
    trades, _, iters, aborted = simulate_pair_pool(
        "EURUSD", df, trigger, atr, cfg, iteration_budget=None
    )
    assert aborted is False
    assert len(trades) == 1
    assert iters > 0  # we did simulate something


def test_search_log_row_for_timeout_has_correct_flags():
    """When a rule times out in the search loop, its log row has
    evaluation_timeout=True, pool_floor_pass=False, NaN metrics."""
    # Build a fixture with abundant trades; set tiny iteration_budget_per_rule.
    n_bars = 1000
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open_bid": [1.20] * n_bars,
            "high_bid": [1.21] * n_bars,
            "low_bid": [1.19] * n_bars,
            "close_bid": [1.20] * n_bars,
            "open_ask": [1.2001] * n_bars,
            "high_ask": [1.2101] * n_bars,
            "low_ask": [1.1901] * n_bars,
            "close_ask": [1.2001] * n_bars,
            "volume": [1.0] * n_bars,
            "spread_close": [0.0001] * n_bars,
            "bid_ask_data_quality": ["ok"] * n_bars,
        },
        index=idx,
    )
    feat = pd.DataFrame({"feat_a": np.arange(n_bars, dtype="float64")}, index=idx)
    atr = pd.Series([0.001] * n_bars, index=idx, name="atr_14")
    fixtures = [PairFixture(pair="EURUSD", pair_df=df, feature_matrix=feat, atr_series=atr)]
    grid = build_quantile_grid({"EURUSD": feat}, quantiles=(0.10, 0.25, 0.50, 0.75, 0.90))
    lineage_df = pd.DataFrame(
        [{"name": "feat_a", "causal_lineage": "clean", "feature_class": "synthetic"}]
    )
    cfg = SearchConfig(
        n_rules=3,
        random_seed=42,
        pool_floor=10,
        grammar_cfg=GrammarConfig(max_atoms_per_rule=1, allow_not=False),
        exit_cfg=DiscoveryExitConfig(
            initial_sl_atr_mult=2.0,
            trail_activation_atr_mult=4.0,
            trail_distance_atr_mult=2.0,
            primary_tf_warmup_bars=0,
            time_exit_bars=200,  # each trade caps at 200 bars
        ),
        iteration_budget_per_rule=10,  # tiny → all rules will time out
    )
    pool = clean_feature_pool(lineage_df)
    res = run_search(
        fixtures=fixtures, grid=grid, lineage_df=lineage_df, cfg=cfg,
        feature_pool=pool, progress_every=0,
    )
    # All rules timed out
    timeout_rows = [r for r in res.log_rows if r.get("evaluation_timeout") is True]
    assert len(timeout_rows) >= 1
    for r in timeout_rows:
        assert r["pool_floor_pass"] is False
        assert r["pool_size"] == 0
        assert r["mean_r"] is None
        assert r["p_value"] is None
        assert r["iterations_consumed"] >= 10


def test_aggregate_wallclock_cap_halts_at_rule_boundary():
    """total_wallclock_cap_hours fires at rule boundary; not all rules processed."""
    # Use a 0-second cap to trigger HALT immediately.
    n_bars = 200
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open_bid": [1.20] * n_bars,
            "high_bid": [1.21] * n_bars,
            "low_bid": [1.19] * n_bars,
            "close_bid": [1.20] * n_bars,
            "open_ask": [1.2001] * n_bars,
            "high_ask": [1.2101] * n_bars,
            "low_ask": [1.1901] * n_bars,
            "close_ask": [1.2001] * n_bars,
            "volume": [1.0] * n_bars,
            "spread_close": [0.0001] * n_bars,
            "bid_ask_data_quality": ["ok"] * n_bars,
        },
        index=idx,
    )
    feat = pd.DataFrame({"feat_a": np.arange(n_bars, dtype="float64")}, index=idx)
    atr = pd.Series([0.001] * n_bars, index=idx, name="atr_14")
    fixtures = [PairFixture(pair="EURUSD", pair_df=df, feature_matrix=feat, atr_series=atr)]
    grid = build_quantile_grid({"EURUSD": feat}, quantiles=(0.10, 0.25, 0.50, 0.75, 0.90))
    lineage_df = pd.DataFrame(
        [{"name": "feat_a", "causal_lineage": "clean", "feature_class": "synthetic"}]
    )
    # Wall-clock budget of effectively zero → first rule-boundary check halts immediately.
    cfg = SearchConfig(
        n_rules=50,
        random_seed=42,
        pool_floor=10,
        grammar_cfg=GrammarConfig(max_atoms_per_rule=1, allow_not=False),
        exit_cfg=DiscoveryExitConfig(
            initial_sl_atr_mult=2.0, trail_activation_atr_mult=4.0,
            trail_distance_atr_mult=2.0, primary_tf_warmup_bars=0,
            time_exit_bars=100,
        ),
        total_wallclock_cap_hours=1e-9,  # effectively zero
    )
    pool = clean_feature_pool(lineage_df)
    res = run_search(
        fixtures=fixtures, grid=grid, lineage_df=lineage_df, cfg=cfg,
        feature_pool=pool, progress_every=0,
    )
    assert res.halted_at_aggregate_cap is True
    # Some rules may have started before the first cap check; the rest are skipped.
    assert res.rules_run < cfg.n_rules
