"""Tests for core.discovery.causal_filter."""

from __future__ import annotations

import pandas as pd

from core.discovery.causal_filter import (
    CausalCheckResult,
    check_rule_causal,
    clean_feature_pool,
)
from core.discovery.grammar import Atom, Combinator, Node, Op, RuleSpec


LINEAGE_DF = pd.DataFrame(
    [
        {"name": "atr_14", "lineage": "clean", "feature_class": "price_geometry"},
        {"name": "session_london", "lineage": "clean", "feature_class": "session"},
        {"name": "dxy_state", "lineage": "suspect", "feature_class": "cross_asset"},
        {"name": "raw_volume_rank", "lineage": "unverified", "feature_class": "volume"},
        {"name": "experimental_x", "lineage": "clean", "feature_class": "experimental"},
    ]
)


def _spec_with_features(features: list[str]) -> RuleSpec:
    """Build a single-AND chain of atoms over the requested features."""
    atoms = [Atom(feature=f, op=Op.GT, quantile=0.50) for f in features]
    root = atoms[0]
    for atom in atoms[1:]:
        root = Node(combinator=Combinator.AND, left=root, right=atom)
    return RuleSpec(rule_id=0, root=root, n_atoms=len(atoms))


def test_clean_feature_pool_default():
    pool = clean_feature_pool(LINEAGE_DF)
    # All clean-tagged features (regardless of class) come through.
    assert "atr_14" in pool
    assert "session_london" in pool
    assert "experimental_x" in pool
    assert "dxy_state" not in pool
    assert "raw_volume_rank" not in pool
    # Sorted output.
    assert list(pool) == sorted(pool)


def test_clean_pool_with_exclude_classes():
    pool = clean_feature_pool(LINEAGE_DF, exclude_classes=("experimental",))
    assert "experimental_x" not in pool
    assert "atr_14" in pool


def test_passes_when_all_clean():
    spec = _spec_with_features(["atr_14", "session_london"])
    res = check_rule_causal(spec, LINEAGE_DF)
    assert res.passed is True
    assert res.rejected_features == ()


def test_rejects_on_suspect_feature():
    spec = _spec_with_features(["atr_14", "dxy_state"])
    res = check_rule_causal(spec, LINEAGE_DF)
    assert res.passed is False
    assert "dxy_state" in res.rejected_features
    assert "non_clean_lineage" in res.reason


def test_rejects_on_unverified_feature():
    spec = _spec_with_features(["raw_volume_rank"])
    res = check_rule_causal(spec, LINEAGE_DF)
    assert res.passed is False
    assert "raw_volume_rank" in res.rejected_features


def test_rejects_on_unknown_feature():
    spec = _spec_with_features(["missing_feature"])
    res = check_rule_causal(spec, LINEAGE_DF)
    assert res.passed is False
    assert "missing_feature" in res.rejected_features
    assert "unknown_feature" in res.reason


def test_rejects_on_excluded_class():
    spec = _spec_with_features(["experimental_x"])
    res = check_rule_causal(spec, LINEAGE_DF, exclude_classes=("experimental",))
    assert res.passed is False
    assert "experimental_x" in res.rejected_features
    assert "excluded_class" in res.reason
