"""Tests for core/features/pipeline.py and the feature registry.

Covers:
  - Every class registers expected features (count + lineage tags)
  - Pipeline computes all features on a synthetic pair_df without error
  - Output schema is deterministic (sorted columns, indexed by pair_df.index)
  - Features-needing-panel emit NaN if no panel provided
  - Cross-pair features work when a real Panel is supplied
  - Lookahead spot-check: feature values at time t do not change when
    pair_df is truncated to t (no future-bar influence)
  - Two-run determinism on the feature matrix sha256
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data.aggregator import aggregate
from core.features.lineage import CausalLineage
from core.features.pipeline import compute_feature_matrix, feature_lineage_dataframe
from core.features.registry import all_specs
from core.sim.panel import Panel
from tests.fixtures.histdata_mini.build import FixtureSpec, build_fixture


@pytest.fixture
def mini_root(tmp_path: Path) -> Path:
    # 1440 minutes/month × 2 months = 2880 M1 bars per pair → ~576 M5 bars
    # enough rows for trailing-100 features and the lookahead spot-check.
    return build_fixture(
        tmp_path / "histdata",
        FixtureSpec(
            pairs=("EURUSD", "GBPUSD", "USDJPY"),
            months=("201001", "201002"),
            minutes_per_month=1440,
        ),
    )


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def pair_df(mini_root: Path, cache_root: Path) -> pd.DataFrame:
    return aggregate("EURUSD", "M5", histdata_root=mini_root, cache_root=cache_root)


@pytest.fixture
def panel(mini_root: Path, cache_root: Path) -> Panel:
    return Panel.from_pairs(
        ["EURUSD", "GBPUSD", "USDJPY"],
        "M5",
        histdata_root=mini_root,
        cache_root=cache_root,
    )


# ── registry / lineage shape ─────────────────────────────────────────


def test_registry_has_expected_classes() -> None:
    classes = {s.feature_class for s in all_specs()}
    assert classes >= {
        "price_geometry",
        "session",
        "vol_regime",
        "distance",
        "spread_regime",
        "multi_tf",
        "cross_pair",
    }


def test_clean_lineage_is_majority() -> None:
    """Most features should be clean; suspect is reserved for cross-pair."""
    n_clean = sum(1 for s in all_specs() if s.lineage is CausalLineage.CLEAN)
    n_total = len(all_specs())
    assert n_clean / n_total > 0.5


def test_all_cross_pair_features_are_suspect_or_clean() -> None:
    """Cross-pair features default to suspect pending Step 6 audit."""
    cp = [s for s in all_specs() if s.feature_class == "cross_pair"]
    assert cp, "no cross_pair features registered"
    for s in cp:
        assert s.lineage in (CausalLineage.SUSPECT, CausalLineage.CLEAN)


def test_feature_names_unique() -> None:
    names = [s.name for s in all_specs()]
    assert len(names) == len(set(names))


def test_lineage_dataframe_has_required_columns() -> None:
    df = feature_lineage_dataframe()
    # Column renamed `lineage` → `causal_lineage` per L_PROTOCOL §1
    # terminology (matches _filter_lineage in core.steps.step_4_extraction
    # and clean_feature_pool in core.discovery.causal_filter).
    assert set(df.columns) == {
        "name", "feature_class", "causal_lineage", "needs_panel", "description",
    }
    # Sorted by name
    assert list(df["name"]) == sorted(df["name"])


# ── pipeline smoke ───────────────────────────────────────────────────


def test_pipeline_runs_on_synthetic_pair_df(pair_df: pd.DataFrame) -> None:
    result = compute_feature_matrix("EURUSD", pair_df)
    assert result.matrix.index.equals(pair_df.index)
    # All registered features present (panel-needing features still get a column, all-NaN)
    assert set(result.matrix.columns) == {s.name for s in all_specs()}


def test_pipeline_columns_are_sorted(pair_df: pd.DataFrame) -> None:
    result = compute_feature_matrix("EURUSD", pair_df)
    assert list(result.matrix.columns) == sorted(result.matrix.columns)


def test_panel_only_features_nan_without_panel(pair_df: pd.DataFrame) -> None:
    """Cross-pair + multi-tf features should be all-NaN when panel=None."""
    result = compute_feature_matrix("EURUSD", pair_df, panel=None)
    panel_dependent = [s.name for s in all_specs() if s.needs_panel]
    for col in panel_dependent:
        assert result.matrix[col].isna().all(), f"{col} should be NaN without panel"


def test_panel_dependent_features_compute_with_panel(pair_df: pd.DataFrame, panel: Panel) -> None:
    """Cross-pair features should produce non-NaN values when a panel is supplied."""
    result = compute_feature_matrix("EURUSD", pair_df, panel=panel)
    # At least one cross-pair feature should produce some non-NaN value somewhere
    cp_columns = [s.name for s in all_specs() if s.feature_class == "cross_pair"]
    any_nonna = False
    for col in cp_columns:
        if result.matrix[col].notna().any():
            any_nonna = True
            break
    assert any_nonna, "no cross-pair feature produced any non-NaN value"


def test_pipeline_subset_by_names(pair_df: pd.DataFrame) -> None:
    result = compute_feature_matrix("EURUSD", pair_df, names=["hour_of_day", "day_of_week"])
    assert list(result.matrix.columns) == ["day_of_week", "hour_of_day"]
    assert len(result.lineage) == 2


# ── lookahead spot-check ─────────────────────────────────────────────


def test_lookahead_spotcheck_5_random_trades(pair_df: pd.DataFrame, panel: Panel) -> None:
    """Truncate pair_df at five random timestamps and confirm features at those
    timestamps don't change vs the full-history compute. If any do, that's
    lookahead.

    Cross-pair and multi-TF features that need a panel are exempted from the
    truncation arm because truncating only EURUSD would unrealistically
    deprive the panel — we cover those via panel=None / panel-supplied
    parity in other tests.
    """
    full = compute_feature_matrix("EURUSD", pair_df, panel=panel)
    panel_dependent = {s.name for s in all_specs() if s.needs_panel}

    rng = np.random.default_rng(seed=42)
    n = len(pair_df)
    # Pick 5 random row positions far enough in for trailing windows to fill,
    # but excluding the last bar.
    idxs = rng.choice(range(105, n - 1), size=5, replace=False)
    for i in idxs:
        truncated = pair_df.iloc[: i + 1].copy()
        truncated_features = compute_feature_matrix("EURUSD", truncated, panel=panel)
        ts = pair_df.index[i]
        for col in full.matrix.columns:
            if col in panel_dependent:
                continue
            full_val = full.matrix.at[ts, col]
            trunc_val = truncated_features.matrix.at[ts, col]
            if pd.isna(full_val) and pd.isna(trunc_val):
                continue
            assert full_val == trunc_val, (
                f"Lookahead in {col!r} at {ts}: full={full_val!r} vs trunc={trunc_val!r}"
            )


# ── determinism ───────────────────────────────────────────────────────


def test_pipeline_two_run_determinism(pair_df: pd.DataFrame, panel: Panel) -> None:
    """Two runs of compute_feature_matrix produce equal DataFrames."""
    a = compute_feature_matrix("EURUSD", pair_df, panel=panel)
    b = compute_feature_matrix("EURUSD", pair_df, panel=panel)
    pd.testing.assert_frame_equal(a.matrix, b.matrix)


def test_pipeline_sha256_stable(pair_df: pd.DataFrame, panel: Panel) -> None:
    """CSV serialisation of the feature matrix has stable sha256 across runs."""

    def sha() -> str:
        result = compute_feature_matrix("EURUSD", pair_df, panel=panel)
        csv = result.matrix.to_csv(lineterminator="\n", float_format="%.10g")
        return hashlib.sha256(csv.encode("utf-8")).hexdigest()

    assert sha() == sha()


# ── error handling ───────────────────────────────────────────────────


def test_pipeline_rejects_unknown_feature_name(pair_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="No registered feature named"):
        compute_feature_matrix("EURUSD", pair_df, names=["bogus_feature_xyz"])
