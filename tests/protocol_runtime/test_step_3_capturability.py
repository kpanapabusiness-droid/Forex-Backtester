"""Tests for Step 3 capturability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.arc.arc_pool_builder import ArcPoolConfig, build_arc_pool
from core.steps.step_2_clustering import run_step_2
from core.steps.step_3_capturability import (
    SL_MULT_SWEEP,
    run_step_3,
    step_3_sha256,
)
from tests.protocol_runtime._fixtures import build_synthetic_arc_pool_inputs


def _build_for_step_3():
    signal, panels = build_synthetic_arc_pool_inputs(n_bars=800)
    pool = build_arc_pool(signal, panels, ArcPoolConfig(arc_name="t", hold_bars=24))
    s2 = run_step_2(pool.trades, pool.paths)
    return pool, s2


def test_step_3_runs_to_completion() -> None:
    pool, s2 = _build_for_step_3()
    s3 = run_step_3(
        pool.trades, pool.paths, s2.cluster_assignments,
        declared_sl_mult=2.0, cluster_centroids=s2.centroids,
    )
    assert len(s3.per_cluster) == s2.k_selected
    assert "reach_1r" in s3.capturability_csv.columns
    assert "wrong_way_pp" in s3.capturability_csv.columns


def test_step_3_candidate_flag_threshold() -> None:
    pool, s2 = _build_for_step_3()
    s3 = run_step_3(
        pool.trades, pool.paths, s2.cluster_assignments,
        declared_sl_mult=2.0, cluster_centroids=s2.centroids,
    )
    for c in s3.per_cluster:
        if c.is_candidate:
            assert c.reach_1r >= 0.50
            assert c.wrong_way_pp <= 0.30
            assert c.mfe_p50 >= 1.5


def test_step_3_sweep_covers_all_multipliers() -> None:
    pool, s2 = _build_for_step_3()
    s3 = run_step_3(
        pool.trades, pool.paths, s2.cluster_assignments,
        declared_sl_mult=2.0, cluster_centroids=s2.centroids,
    )
    if s3.per_cluster:
        sweep = s3.per_cluster[0].sl_sweep
        assert set(sweep) == set(SL_MULT_SWEEP)


def test_step_3_determinism() -> None:
    pool, s2 = _build_for_step_3()
    a = run_step_3(
        pool.trades, pool.paths, s2.cluster_assignments,
        declared_sl_mult=2.0, cluster_centroids=s2.centroids,
    )
    b = run_step_3(
        pool.trades, pool.paths, s2.cluster_assignments,
        declared_sl_mult=2.0, cluster_centroids=s2.centroids,
    )
    assert step_3_sha256(a) == step_3_sha256(b)
