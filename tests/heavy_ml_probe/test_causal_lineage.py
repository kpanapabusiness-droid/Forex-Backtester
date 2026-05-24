"""Tests for core.heavy_ml_probe.causal_lineage.

Per dispatch §6 #2 + spec §"Per-feature causal lineage": the gate is
pre-evaluation. Features without a clean lineage tag are excluded from
the training column set BEFORE AutoML touches the data. These tests
exercise:

  * clean features pass
  * suspect / unverified / unknown / excluded-class features are rejected
    with the right reason code
  * the gate result accounts for every input column
  * the markdown summary is stable + grep-able
  * the gate is pure (no side effects on the input lineage_df)
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from core.heavy_ml_probe.causal_lineage import (
    DEFAULT_ACCEPTED_LINEAGE,
    REASON_EXCLUDED_CLASS,
    REASON_NON_CLEAN_LINEAGE,
    REASON_UNKNOWN_FEATURE,
    LineageGateResult,
    filter_training_columns,
    lineage_summary_markdown,
)

# Matches the schema produced by core.features.pipeline.feature_lineage_dataframe().
LINEAGE_DF = pd.DataFrame(
    [
        {"name": "atr_14", "causal_lineage": "clean", "feature_class": "price_geometry"},
        {"name": "session_london", "causal_lineage": "clean", "feature_class": "session"},
        {"name": "kijun_26_distance", "causal_lineage": "clean", "feature_class": "price_geometry"},
        {"name": "experimental_x", "causal_lineage": "clean", "feature_class": "experimental"},
        {"name": "dxy_state", "causal_lineage": "suspect", "feature_class": "cross_asset"},
        {"name": "raw_volume_rank", "causal_lineage": "unverified", "feature_class": "volume"},
    ]
)


def test_accepts_all_clean_features():
    cols = ["atr_14", "session_london", "kijun_26_distance"]
    res = filter_training_columns(cols, LINEAGE_DF)
    assert isinstance(res, LineageGateResult)
    assert res.n_input_columns == 3
    assert res.n_accepted == 3
    assert res.n_rejected == 0
    assert res.accepted_features == tuple(sorted(cols))


def test_rejects_suspect_lineage():
    cols = ["atr_14", "dxy_state"]
    res = filter_training_columns(cols, LINEAGE_DF)
    assert res.n_accepted == 1
    assert res.accepted_features == ("atr_14",)
    assert res.n_rejected == 1
    [rej] = res.rejected
    assert rej["feature"] == "dxy_state"
    assert rej["reason"] == REASON_NON_CLEAN_LINEAGE
    assert "suspect" in rej["detail"]


def test_rejects_unverified_lineage():
    cols = ["atr_14", "raw_volume_rank"]
    res = filter_training_columns(cols, LINEAGE_DF)
    assert res.n_rejected == 1
    [rej] = res.rejected
    assert rej["feature"] == "raw_volume_rank"
    assert rej["reason"] == REASON_NON_CLEAN_LINEAGE
    assert "unverified" in rej["detail"]


def test_rejects_unknown_feature_with_clear_reason():
    cols = ["atr_14", "totally_made_up_feature"]
    res = filter_training_columns(cols, LINEAGE_DF)
    assert res.n_accepted == 1
    assert res.n_rejected == 1
    [rej] = res.rejected
    assert rej["feature"] == "totally_made_up_feature"
    assert rej["reason"] == REASON_UNKNOWN_FEATURE


def test_exclude_class_overrides_clean_lineage():
    """Even a clean-tagged feature is rejected if its class is excluded."""
    cols = ["atr_14", "experimental_x"]
    res = filter_training_columns(
        cols, LINEAGE_DF, exclude_classes=("experimental",)
    )
    assert res.n_accepted == 1
    assert res.accepted_features == ("atr_14",)
    [rej] = res.rejected
    assert rej["feature"] == "experimental_x"
    assert rej["reason"] == REASON_EXCLUDED_CLASS
    assert "experimental" in rej["detail"]


def test_mixed_pool_accounts_for_every_column():
    cols = [
        "atr_14",                # clean -> accept
        "dxy_state",             # suspect -> reject
        "raw_volume_rank",       # unverified -> reject
        "totally_made_up",       # unknown -> reject
        "session_london",        # clean -> accept
    ]
    res = filter_training_columns(cols, LINEAGE_DF)
    assert res.n_input_columns == 5
    assert res.n_accepted == 2
    assert res.n_rejected == 3
    assert set(res.accepted_features) == {"atr_14", "session_london"}
    reasons = {r["feature"]: r["reason"] for r in res.rejected}
    assert reasons["dxy_state"] == REASON_NON_CLEAN_LINEAGE
    assert reasons["raw_volume_rank"] == REASON_NON_CLEAN_LINEAGE
    assert reasons["totally_made_up"] == REASON_UNKNOWN_FEATURE


def test_duplicate_columns_dedup():
    """Duplicate column names in the input are deduplicated before gating."""
    cols = ["atr_14", "atr_14", "session_london"]
    res = filter_training_columns(cols, LINEAGE_DF)
    assert res.n_input_columns == 2
    assert res.n_accepted == 2


def test_accepted_lineage_default_is_clean_only():
    assert DEFAULT_ACCEPTED_LINEAGE == ("clean",)


def test_gate_is_pure_does_not_mutate_lineage_df():
    snapshot = copy.deepcopy(LINEAGE_DF)
    _ = filter_training_columns(["atr_14"], LINEAGE_DF)
    pd.testing.assert_frame_equal(LINEAGE_DF, snapshot)


def test_missing_required_column_raises():
    bad = pd.DataFrame([{"name": "x", "causal_lineage": "clean"}])  # no feature_class
    with pytest.raises(ValueError, match="missing required columns"):
        filter_training_columns(["x"], bad)


def test_lineage_summary_markdown_renders_stably():
    res = filter_training_columns(
        ["atr_14", "dxy_state", "raw_volume_rank", "totally_made_up"],
        LINEAGE_DF,
    )
    md1 = lineage_summary_markdown(res)
    md2 = lineage_summary_markdown(res)
    assert md1 == md2  # deterministic
    # Spot-check expected sections render
    assert "# Causal lineage gate — heavy_ml_probe" in md1
    assert "Accepted (clean): **1**" in md1
    assert "Rejected: **3**" in md1
    assert "Rejection breakdown" in md1
    assert REASON_NON_CLEAN_LINEAGE in md1
    assert REASON_UNKNOWN_FEATURE in md1


def test_lineage_summary_markdown_when_no_rejections():
    res = filter_training_columns(["atr_14"], LINEAGE_DF)
    md = lineage_summary_markdown(res)
    assert "Accepted (clean): **1**" in md
    assert "Rejected: **0**" in md
    # When nothing was rejected, the detail tables should not appear.
    assert "Rejection breakdown" not in md
    assert "Rejected features (sorted)" not in md


def test_accepts_clean_features_with_lineage_case_insensitive():
    """Real-world lineage_df values use lowercase, but the gate
    should not be case-sensitive (defensive against producer drift)."""
    upper = pd.DataFrame([
        {"name": "x", "causal_lineage": "CLEAN", "feature_class": "Price_Geometry"},
    ])
    res = filter_training_columns(["x"], upper)
    assert res.n_accepted == 1
    assert res.accepted_features == ("x",)
