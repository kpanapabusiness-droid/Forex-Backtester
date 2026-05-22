"""Tests for Step 2 clustering."""

from __future__ import annotations

from core.arc.arc_pool_builder import ArcPoolConfig, build_arc_pool
from core.steps.step_2_clustering import (
    PATH_FEATURE_COLS,
    compute_path_features,
    run_step_2,
    step_2_sha256,
)
from tests.protocol_runtime._fixtures import build_synthetic_arc_pool_inputs


def _build_step1_pool():
    signal, panels = build_synthetic_arc_pool_inputs(n_bars=800)
    pool = build_arc_pool(signal, panels, ArcPoolConfig(arc_name="t", hold_bars=24))
    return pool


def test_compute_path_features_returns_expected_columns() -> None:
    pool = _build_step1_pool()
    feats = compute_path_features(pool.trades, pool.paths)
    assert set(PATH_FEATURE_COLS).issubset(feats.columns)
    assert len(feats) > 0


def test_run_step_2_picks_k_in_range() -> None:
    pool = _build_step1_pool()
    res = run_step_2(pool.trades, pool.paths)
    assert res.k_selected in (1, 2, 3, 4, 5, 6)
    if res.k_selected > 1:
        assert len(res.cluster_summary) == res.k_selected


def test_run_step_2_assignments_match_pool() -> None:
    pool = _build_step1_pool()
    res = run_step_2(pool.trades, pool.paths)
    pool_ids = set(int(t) for t in pool.trades["trade_id"])
    assign_ids = set(int(t) for t in res.cluster_assignments["trade_id"])
    assert assign_ids.issubset(pool_ids)


def test_run_step_2_determinism() -> None:
    pool = _build_step1_pool()
    res_1 = run_step_2(pool.trades, pool.paths)
    res_2 = run_step_2(pool.trades, pool.paths)
    assert step_2_sha256(res_1) == step_2_sha256(res_2)
