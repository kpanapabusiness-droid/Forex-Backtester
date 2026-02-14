"""
Phase D-6F.1: Tests for clean geometry analysis.
"""
from __future__ import annotations

import pandas as pd
import pytest

from analytics.phaseD6F1_clean_geometry import (
    compute_per_pair_prob_matrix,
    compute_per_pair_quantiles,
    compute_pooled_prob_matrix,
    compute_pooled_quantiles,
    compute_pooled_stability_prob_matrix,
    run_geometry_analysis,
)


def _make_synthetic_df() -> pd.DataFrame:
    """Synthetic clean labels with known geometry."""
    return pd.DataFrame({
        "pair": ["PAIR_A"] * 6 + ["PAIR_B"] * 4,
        "date": pd.to_datetime([
            "2020-01-01", "2020-06-01", "2020-12-31",
            "2021-01-01", "2021-06-01", "2021-12-31",
            "2022-06-01", "2022-12-31",
            "2023-01-01", "2023-06-01",
        ]),
        "valid_atr": [True] * 10,
        "valid_ref": [True] * 10,
        "valid_h40": [True] * 10,
        "clean_mfe_long_x1_h40": [0.5, 1.2, 2.0, 3.0, 4.0, 7.0, 1.0, 2.5, 5.0, 0.3],
        "clean_mfe_long_x2_h40": [0.3, 0.8, 1.5, 2.0, 3.0, 6.0, 0.5, 1.0, 4.0, 0.0],
        "clean_mfe_long_x3_h40": [0.0, 0.5, 1.0, 1.2, 2.0, 5.0, 0.0, 0.5, 3.0, 0.0],
        "clean_mfe_short_x1_h40": [1.0, 0.5, 2.5, 0.8, 1.5, 2.0, 0.7, 1.2, 0.4, 0.6],
        "clean_mfe_short_x2_h40": [0.5, 0.0, 1.5, 0.3, 0.8, 1.0, 0.2, 0.6, 0.0, 0.0],
        "clean_mfe_short_x3_h40": [0.0, 0.0, 0.8, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    })


def test_probability_table_correctness() -> None:
    """Known hits for thresholds."""
    df = _make_synthetic_df()
    prob = compute_pooled_prob_matrix(df)
    assert not prob.empty
    r = prob.set_index(["direction", "x", "y"])
    long_x1 = df["clean_mfe_long_x1_h40"].dropna()
    n_valid = len(long_x1)
    assert n_valid == 10
    n_hit_y1 = (long_x1 >= 1.0).sum()
    assert r.loc[("long", 1, 1.0), "n_hit"] == n_hit_y1
    assert r.loc[("long", 1, 1.0), "n_valid"] == n_valid
    assert r.loc[("long", 1, 1.0), "rate"] == pytest.approx(n_hit_y1 / n_valid, rel=1e-5)


def test_per_pair_equals_pooled_when_one_pair() -> None:
    """Per-pair equals pooled when only one pair."""
    df = _make_synthetic_df()
    single = df[df["pair"] == "PAIR_A"].copy()
    pooled = compute_pooled_prob_matrix(single)
    per_pair = compute_per_pair_prob_matrix(single)
    assert len(per_pair["pair"].unique()) == 1
    for _, row_pp in per_pair.iterrows():
        match = pooled[
            (pooled["direction"] == row_pp["direction"])
            & (pooled["x"] == row_pp["x"])
            & (pooled["y"] == row_pp["y"])
        ]
        assert len(match) == 1
        assert match.iloc[0]["n_valid"] == row_pp["n_valid"]
        assert match.iloc[0]["n_hit"] == row_pp["n_hit"]
        assert match.iloc[0]["rate"] == pytest.approx(row_pp["rate"], rel=1e-5)


def test_stability_split_expected_rates() -> None:
    """Split into discovery/validation yields expected rates."""
    df = _make_synthetic_df()
    stab = compute_pooled_stability_prob_matrix(df, discovery_end="2022-12-31")
    assert not stab.empty
    disc_dates = df[df["date"] <= "2022-12-31"]
    val_dates = df[df["date"] > "2022-12-31"]
    assert len(disc_dates) == 8
    assert len(val_dates) == 2
    r = stab[(stab["direction"] == "long") & (stab["x"] == 1) & (stab["y"] == 1.0)].iloc[0]
    disc_vals = disc_dates["clean_mfe_long_x1_h40"].dropna()
    val_vals = val_dates["clean_mfe_long_x1_h40"].dropna()
    exp_rate_disc = (disc_vals >= 1.0).mean() if len(disc_vals) else 0
    exp_rate_val = (val_vals >= 1.0).mean() if len(val_vals) else 0
    assert r["rate_discovery"] == pytest.approx(exp_rate_disc, rel=1e-5)
    assert r["rate_validation"] == pytest.approx(exp_rate_val, rel=1e-5)


def test_determinism_same_output_twice() -> None:
    """Calling compute twice yields identical DataFrames."""
    df = _make_synthetic_df()
    a1 = compute_pooled_prob_matrix(df)
    a2 = compute_pooled_prob_matrix(df)
    pd.testing.assert_frame_equal(a1, a2)
    b1 = compute_pooled_quantiles(df)
    b2 = compute_pooled_quantiles(df)
    pd.testing.assert_frame_equal(b1, b2)
    c1 = compute_pooled_stability_prob_matrix(df)
    c2 = compute_pooled_stability_prob_matrix(df)
    pd.testing.assert_frame_equal(c1, c2)


def test_run_geometry_analysis_writes_files(tmp_path) -> None:
    """run_geometry_analysis writes expected CSVs."""
    df = _make_synthetic_df()
    paths = run_geometry_analysis(df, out_dir=tmp_path)
    assert "pooled_prob_matrix" in paths
    assert "per_pair_prob_matrix" in paths
    assert "pooled_quantiles" in paths
    assert "per_pair_quantiles" in paths
    assert "pooled_stability_prob_matrix" in paths
    assert "per_pair_stability_prob_matrix" in paths
    for p in paths.values():
        assert p.exists()
        loaded = pd.read_csv(p)
        assert not loaded.empty


def test_per_pair_quantiles_columns() -> None:
    """Per-pair quantiles has expected stat columns."""
    df = _make_synthetic_df()
    q = compute_per_pair_quantiles(df)
    expected = {"pair", "direction", "x", "n", "mean", "std", "p50", "p75", "p90", "p95", "p99", "max"}
    assert expected.issubset(set(q.columns))
