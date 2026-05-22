"""Tests for Step 4 extraction.

Uses synthetic feature matrix with a known signal-to-noise structure
so AUC is predictable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.steps.step_4_extraction import run_step_4, step_4_sha256


def _synthetic_step_4_inputs(n: int = 400, seed: int = 42):
    rng = np.random.default_rng(seed)
    # Two features: f_signal is correlated with cluster membership, f_noise isn't
    cluster_id = (np.arange(n) % 2)  # 50/50 split, time-ordered
    f_signal = cluster_id + rng.normal(0, 0.5, n)  # AUC ~ 0.85
    f_noise = rng.normal(0, 1, n)  # AUC ~ 0.50
    f_clean = cluster_id * 0.5 + rng.normal(0, 1, n)  # weak signal
    timestamps = pd.date_range("2018-01-01", periods=n, freq="4h", tz="UTC")
    trades = pd.DataFrame({
        "trade_id": np.arange(1, n + 1),
        "entry_time": timestamps,
    })
    fm = pd.DataFrame({
        "trade_id": np.arange(1, n + 1),
        "f_signal": f_signal,
        "f_noise": f_noise,
        "f_clean": f_clean,
    })
    assignments = pd.DataFrame({
        "trade_id": np.arange(1, n + 1),
        "cluster_id": cluster_id,
    })
    return trades, fm, assignments


def test_step_4_runs_on_synthetic_data() -> None:
    trades, fm, assignments = _synthetic_step_4_inputs()
    res = run_step_4(trades, fm, assignments, candidate_cluster_ids=(1,))
    assert len(res.per_cluster) == 1
    ce = res.per_cluster[0]
    # f_signal should be the dominant feature
    top_feature = ce.feature_importance.iloc[0]["feature"]
    assert top_feature in ("f_signal", "f_clean")  # f_clean is also informative


def test_step_4_auc_above_chance() -> None:
    trades, fm, assignments = _synthetic_step_4_inputs(n=600)
    res = run_step_4(trades, fm, assignments, candidate_cluster_ids=(1,))
    assert len(res.per_cluster) == 1
    # Mean AUC across folds should be clearly above 0.5
    assert res.per_cluster[0].best_classifier_mean_auc > 0.6


def test_step_4_lineage_exclusion() -> None:
    trades, fm, assignments = _synthetic_step_4_inputs()
    lineage = pd.DataFrame({
        "name": ["f_signal", "f_noise", "f_clean"],
        "causal_lineage": ["clean", "suspect", "clean"],
    })
    res = run_step_4(
        trades, fm, assignments,
        feature_lineage=lineage,
        candidate_cluster_ids=(1,),
    )
    assert len(res.per_cluster) == 1
    ce = res.per_cluster[0]
    assert "f_noise" in ce.excluded_features
    assert "f_signal" in ce.used_features


def test_step_4_determinism() -> None:
    trades, fm, assignments = _synthetic_step_4_inputs()
    a = run_step_4(trades, fm, assignments, candidate_cluster_ids=(1,))
    b = run_step_4(trades, fm, assignments, candidate_cluster_ids=(1,))
    assert step_4_sha256(a) == step_4_sha256(b)
