"""Forward-geometry computations — KH-24 v2.0 Step 3."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.arc_kh24_v2.step3._forward_geometry import (
    PERCENTILES,
    _percentiles,
    _t_stat,
    frac_wrong_way_definition_a,
    frac_wrong_way_definition_b,
    pct_peak_and_collapse,
    split_by_cluster,
)


def test_percentiles_match_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=1000)
    out = _percentiles(x)
    for q in PERCENTILES:
        assert abs(out[q] - float(np.percentile(x, q))) < 1e-12


def test_percentiles_empty_input():
    out = _percentiles(np.array([], dtype=np.float64))
    for q in PERCENTILES:
        assert np.isnan(out[q])


def test_t_stat_formula():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, std, t = _t_stat(x)
    # mean = 3.0, std (ddof=1) = sqrt(((1-3)^2+(2-3)^2+(3-3)^2+(4-3)^2+(5-3)^2)/4) = sqrt(10/4)
    expected_std = float(np.std(x, ddof=1))
    expected_t = mean / (expected_std / np.sqrt(len(x)))
    assert abs(mean - 3.0) < 1e-12
    assert abs(std - expected_std) < 1e-12
    assert abs(t - expected_t) < 1e-12


def test_t_stat_zero_std():
    mean, std, t = _t_stat(np.array([1.5, 1.5, 1.5]))
    assert mean == 1.5
    assert std == 0.0
    # mean != 0, std = 0 → t is +/- inf
    assert np.isinf(t)


def test_frac_wrong_way_def_a():
    final_r = np.array([0.5, -0.6, -0.4, -1.0, 2.0])
    # final_r <= -0.5: -0.6, -1.0 → 2/5
    assert abs(frac_wrong_way_definition_a(final_r) - 0.4) < 1e-12


def test_frac_wrong_way_def_b_never_reached_profit():
    # Trade that never reaches mfe>0.5: wrong-way by def B.
    paths = pd.DataFrame(
        [
            {"trade_id": "T1", "bar_offset": 0, "mfe_so_far_r": 0.1, "mae_so_far_r": -0.2},
            {"trade_id": "T1", "bar_offset": 1, "mfe_so_far_r": 0.3, "mae_so_far_r": -0.5},
            {"trade_id": "T1", "bar_offset": 2, "mfe_so_far_r": 0.3, "mae_so_far_r": -1.05},
        ]
    )
    assert frac_wrong_way_definition_b(paths) == 1.0


def test_frac_wrong_way_def_b_mae_before_profit():
    # T1: mfe>0.5 reached at bar 2, but mae<=-1 happened at bar 1 (before).
    paths = pd.DataFrame(
        [
            {"trade_id": "T1", "bar_offset": 0, "mfe_so_far_r": 0.1, "mae_so_far_r": -0.2},
            {"trade_id": "T1", "bar_offset": 1, "mfe_so_far_r": 0.2, "mae_so_far_r": -1.05},
            {"trade_id": "T1", "bar_offset": 2, "mfe_so_far_r": 0.6, "mae_so_far_r": -1.05},
        ]
    )
    assert frac_wrong_way_definition_b(paths) == 1.0


def test_frac_wrong_way_def_b_clean_winner():
    # T1: mfe>0.5 at bar 1, mae never <= -1 before that.
    paths = pd.DataFrame(
        [
            {"trade_id": "T1", "bar_offset": 0, "mfe_so_far_r": 0.1, "mae_so_far_r": -0.1},
            {"trade_id": "T1", "bar_offset": 1, "mfe_so_far_r": 0.6, "mae_so_far_r": -0.4},
        ]
    )
    assert frac_wrong_way_definition_b(paths) == 0.0


def test_pct_peak_and_collapse():
    trades = pd.DataFrame(
        [
            # Early-peak + collapsed: ttp 0.1, mfe 2.0, final -1.0 → collapsed
            {"trade_id": "T1", "mfe_r": 2.0, "final_r": -1.0},
            # Early-peak but NOT collapsed: ttp 0.2, mfe 2.0, final 1.5 → above 0.5*2.0=1.0
            {"trade_id": "T2", "mfe_r": 2.0, "final_r": 1.5},
            # NOT early-peak: ttp 0.8
            {"trade_id": "T3", "mfe_r": 2.0, "final_r": -1.0},
        ]
    )
    features = pd.DataFrame(
        [
            {"trade_id": "T1", "time_to_peak_mfe_relative": 0.10},
            {"trade_id": "T2", "time_to_peak_mfe_relative": 0.20},
            {"trade_id": "T3", "time_to_peak_mfe_relative": 0.80},
        ]
    )
    # Only T1 qualifies.
    assert abs(pct_peak_and_collapse(trades, features) - 1 / 3) < 1e-12


def test_split_by_cluster_partitions_trades_and_paths():
    trades = pd.DataFrame(
        [
            {
                "trade_id": "T1",
                "final_r": 0.1,
                "mfe_r": 0.2,
                "mae_r": -0.1,
                "bars_held": 3,
                "pair": "X",
            },
            {
                "trade_id": "T2",
                "final_r": 0.5,
                "mfe_r": 0.6,
                "mae_r": -0.2,
                "bars_held": 5,
                "pair": "X",
            },
            {
                "trade_id": "T3",
                "final_r": -1.0,
                "mfe_r": 0.0,
                "mae_r": -1.0,
                "bars_held": 2,
                "pair": "X",
            },
        ]
    )
    paths = pd.DataFrame(
        [
            {
                "trade_id": tid,
                "bar_offset": i,
                "close_r": 0.0,
                "mfe_so_far_r": 0.0,
                "mae_so_far_r": 0.0,
            }
            for tid in ["T1", "T2", "T3"]
            for i in range(2)
        ]
    )
    clusters = pd.DataFrame(
        [
            {"trade_id": "T1", "cluster_id": 0},
            {"trade_id": "T2", "cluster_id": 0},
            {"trade_id": "T3", "cluster_id": 1},
        ]
    )
    out = split_by_cluster(trades, paths, clusters)
    assert len(out) == 2
    assert out[0].cluster_id == 0
    assert sorted(out[0].trade_ids) == ["T1", "T2"]
    assert out[1].cluster_id == 1
    assert out[1].trade_ids == ["T3"]
