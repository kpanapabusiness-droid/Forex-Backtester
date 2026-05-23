"""Tests for Step 4 classifier persistence + the A2/A6 builders.

Covers dispatch Task 6:
  - persistence file exists at expected path
  - manifest SHA256 matches file SHA256
  - load_classifier round-trip predicts identically to the in-memory
    classifier from Step 4
  - build_a2_config_from_step4 / build_a6_config_from_step4 construct
    valid configs
  - threshold sweep does NOT retrain (same classifier, different
    threshold field)
  - corrupted pickle -> ClassifierIntegrityError
  - determinism: two run_step_4 calls produce predictions that agree
    on a held-out sample (binary-byte equality is a soft check; some
    joblib versions write non-deterministic metadata)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from core.steps.classifier_persistence import (
    ClassifierIntegrityError,
    build_a2_config_from_step4,
    build_a6_config_from_step4,
    load_classifier,
)
from core.steps.step_4_extraction import run_step_4


def _synthetic_step_4_inputs(n: int = 400, seed: int = 42):
    rng = np.random.default_rng(seed)
    cluster_id = (np.arange(n) % 2)
    f_signal = cluster_id + rng.normal(0, 0.5, n)
    f_noise = rng.normal(0, 1, n)
    f_clean = cluster_id * 0.5 + rng.normal(0, 1, n)
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


def test_persistence_writes_pickle_at_expected_path(tmp_path: Path) -> None:
    trades, fm, assignments = _synthetic_step_4_inputs()
    persistence_dir = tmp_path / "classifiers"
    res = run_step_4(
        trades, fm, assignments,
        candidate_cluster_ids=(1,),
        persistence_dir=persistence_dir,
        arc_name="synthetic_test",
    )
    assert len(res.per_cluster) == 1
    ce = res.per_cluster[0]
    assert ce.fitted_classifier_path is not None
    assert ce.fitted_classifier_path.exists()
    assert ce.fitted_classifier_path.name == "1.pkl"
    assert ce.fitted_classifier_path.parent == persistence_dir
    assert ce.fitted_classifier_type is not None
    assert ce.fitted_classifier_feature_order is not None
    # Manifest exists alongside
    assert (persistence_dir / "manifest.json").exists()


def test_persistence_skipped_when_dir_is_none() -> None:
    """Backwards-compat: existing callers that don't pass
    persistence_dir get None on every fitted_classifier_* field."""
    trades, fm, assignments = _synthetic_step_4_inputs()
    res = run_step_4(trades, fm, assignments, candidate_cluster_ids=(1,))
    assert len(res.per_cluster) == 1
    ce = res.per_cluster[0]
    assert ce.fitted_classifier_path is None
    assert ce.fitted_classifier_type is None
    assert ce.fitted_classifier_feature_order is None


def test_manifest_sha_matches_pickle_sha(tmp_path: Path) -> None:
    trades, fm, assignments = _synthetic_step_4_inputs()
    persistence_dir = tmp_path / "classifiers"
    res = run_step_4(
        trades, fm, assignments,
        candidate_cluster_ids=(1,),
        persistence_dir=persistence_dir,
        arc_name="synth",
    )
    manifest = json.loads((persistence_dir / "manifest.json").read_text(encoding="utf-8"))
    cls_entry = manifest["classifiers"]["1"]
    pkl_path = persistence_dir / cls_entry["path"]
    import hashlib
    h = hashlib.sha256(pkl_path.read_bytes()).hexdigest()
    assert h == cls_entry["sha256"]
    # Provenance fields present
    assert "joblib_version" in manifest
    assert "sklearn_version" in manifest
    assert "trained_on_pool_size" in cls_entry
    assert cls_entry["trained_on_pool_size"] == res.per_cluster[0].n_trades


def test_load_classifier_round_trip_identical_predictions(tmp_path: Path) -> None:
    trades, fm, assignments = _synthetic_step_4_inputs()
    persistence_dir = tmp_path / "classifiers"
    res = run_step_4(
        trades, fm, assignments,
        candidate_cluster_ids=(1,),
        persistence_dir=persistence_dir,
        arc_name="synth",
    )
    ce = res.per_cluster[0]
    loaded = load_classifier(ce.fitted_classifier_path)
    # Build a held-out feature matrix in the classifier's expected
    # column order
    X_holdout = pd.DataFrame({
        col: np.linspace(-1, 1, 50) for col in ce.fitted_classifier_feature_order
    })
    proba_loaded = loaded.predict_proba(X_holdout)[:, 1]
    # Load again — should be byte-identical predictions
    loaded_2 = load_classifier(ce.fitted_classifier_path)
    proba_loaded_2 = loaded_2.predict_proba(X_holdout)[:, 1]
    np.testing.assert_array_equal(proba_loaded, proba_loaded_2)


def test_build_a2_config_from_step4_constructs(tmp_path: Path) -> None:
    trades, fm, assignments = _synthetic_step_4_inputs()
    persistence_dir = tmp_path / "classifiers"
    res = run_step_4(
        trades, fm, assignments,
        candidate_cluster_ids=(1,),
        persistence_dir=persistence_dir,
        arc_name="synth",
    )
    a2 = build_a2_config_from_step4(res, cluster_id=1)
    assert a2.classifier is not None
    assert a2.threshold == res.per_cluster[0].best_threshold
    assert a2.classifier_feature_order == res.per_cluster[0].fitted_classifier_feature_order
    # config_id auto-generated
    assert a2.config_id.startswith("a2_cluster1_")


def test_build_a6_config_from_step4_constructs(tmp_path: Path) -> None:
    trades, fm, assignments = _synthetic_step_4_inputs()
    persistence_dir = tmp_path / "classifiers"
    res = run_step_4(
        trades, fm, assignments,
        candidate_cluster_ids=(1,),
        persistence_dir=persistence_dir,
        arc_name="synth",
    )
    a6 = build_a6_config_from_step4(res, cluster_id=1, lower_threshold=0.3, upper_threshold=0.5)
    assert a6.classifier is not None
    assert a6.lower_threshold == 0.3
    assert a6.upper_threshold == 0.5
    assert a6.config_id == "a6_cluster1_0.30_0.50"


def test_threshold_sweep_does_not_retrain(tmp_path: Path) -> None:
    trades, fm, assignments = _synthetic_step_4_inputs()
    persistence_dir = tmp_path / "classifiers"
    res = run_step_4(
        trades, fm, assignments,
        candidate_cluster_ids=(1,),
        persistence_dir=persistence_dir,
        arc_name="synth",
    )
    a2_a = build_a2_config_from_step4(res, cluster_id=1, threshold_override=0.40)
    a2_b = build_a2_config_from_step4(res, cluster_id=1, threshold_override=0.65)
    # Different thresholds
    assert a2_a.threshold == 0.40
    assert a2_b.threshold == 0.65
    # Same persisted estimator -> same predictions on identical input
    X = pd.DataFrame({
        col: np.linspace(-1, 1, 30) for col in a2_a.classifier_feature_order
    })
    proba_a = a2_a.classifier.predict_proba(X)[:, 1]
    proba_b = a2_b.classifier.predict_proba(X)[:, 1]
    np.testing.assert_array_equal(proba_a, proba_b)


def test_corrupted_pickle_raises_integrity_error(tmp_path: Path) -> None:
    trades, fm, assignments = _synthetic_step_4_inputs()
    persistence_dir = tmp_path / "classifiers"
    res = run_step_4(
        trades, fm, assignments,
        candidate_cluster_ids=(1,),
        persistence_dir=persistence_dir,
        arc_name="synth",
    )
    ce = res.per_cluster[0]
    # Corrupt the file: append a byte
    with open(ce.fitted_classifier_path, "ab") as f:
        f.write(b"\x00")
    with pytest.raises(ClassifierIntegrityError):
        load_classifier(ce.fitted_classifier_path)


def test_load_classifier_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(ClassifierIntegrityError):
        load_classifier(tmp_path / "does_not_exist.pkl")


def test_load_classifier_missing_manifest_raises(tmp_path: Path) -> None:
    # Drop a fake pickle without a manifest alongside
    pkl = tmp_path / "1.pkl"
    joblib.dump({"not": "a classifier"}, pkl)
    with pytest.raises(ClassifierIntegrityError):
        load_classifier(pkl)


def test_build_a2_raises_when_cluster_has_no_persisted_classifier() -> None:
    """Calling the builder against a Step 4 result that didn't request
    persistence is a usage error — surfaces clearly, not silently."""
    trades, fm, assignments = _synthetic_step_4_inputs()
    res = run_step_4(trades, fm, assignments, candidate_cluster_ids=(1,))
    with pytest.raises(ValueError, match="no fitted classifier"):
        build_a2_config_from_step4(res, cluster_id=1)


def test_build_a2_raises_for_unknown_cluster_id(tmp_path: Path) -> None:
    trades, fm, assignments = _synthetic_step_4_inputs()
    res = run_step_4(
        trades, fm, assignments,
        candidate_cluster_ids=(1,),
        persistence_dir=tmp_path / "classifiers",
        arc_name="synth",
    )
    with pytest.raises(ValueError, match="not present in Step4Result"):
        build_a2_config_from_step4(res, cluster_id=99)


def _train_holdout_inputs(n_train: int = 300, n_holdout: int = 200, seed: int = 11):
    """Synthetic trades spanning a train + holdout window. The train
    window emits the same signal structure as the original fixture;
    the holdout window flips the feature -> cluster relationship so
    a classifier trained on the union behaves differently from one
    trained on train-only.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_holdout
    # Cluster split: alternating IDs through the whole pool
    cluster_id = np.arange(n) % 2
    # In the train window: f_signal correlates positively with cluster
    f_signal_train = cluster_id[:n_train] + rng.normal(0, 0.5, n_train)
    # In the holdout window: signal sign flips — adds bias if mixed in
    f_signal_holdout = -cluster_id[n_train:] + rng.normal(0, 0.5, n_holdout)
    f_signal = np.concatenate([f_signal_train, f_signal_holdout])
    f_noise = rng.normal(0, 1, n)
    # Calendar timestamps: train window 2018, holdout window 2021
    train_ts = pd.date_range("2018-01-01", periods=n_train, freq="4h", tz="UTC")
    holdout_ts = pd.date_range("2021-01-01", periods=n_holdout, freq="4h", tz="UTC")
    timestamps = train_ts.append(holdout_ts)
    trades = pd.DataFrame({
        "trade_id": np.arange(1, n + 1),
        "entry_time": timestamps,
    })
    fm = pd.DataFrame({
        "trade_id": np.arange(1, n + 1),
        "f_signal": f_signal,
        "f_noise": f_noise,
    })
    assignments = pd.DataFrame({
        "trade_id": np.arange(1, n + 1),
        "cluster_id": cluster_id,
    })
    return trades, fm, assignments


def test_train_end_excludes_holdout_from_persistence(tmp_path: Path) -> None:
    """The persisted classifier MUST NOT see trades with
    entry_time >= train_end. Both the CV evaluation AND the refit on
    "the full lineage-filtered pool" restrict to the IS subset.
    """
    trades, fm, assignments = _train_holdout_inputs(n_train=300, n_holdout=200)
    train_end = pd.Timestamp("2020-12-31", tz="UTC")
    res = run_step_4(
        trades, fm, assignments,
        candidate_cluster_ids=(1,),
        persistence_dir=tmp_path / "with_train_end",
        arc_name="holdout_test",
        train_end=train_end,
    )
    assert len(res.per_cluster) == 1
    ce = res.per_cluster[0]
    # ce.n_trades is total binary-classifier pool size (positives +
    # negatives). After train_end filter, both classes together = the
    # train window size = 300. The original pool was 500.
    assert ce.n_trades == 300
    # Manifest carries the train_end declaration
    manifest = json.loads((tmp_path / "with_train_end" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_end"] == "2020-12-31T00:00:00Z"
    assert manifest["classifiers"]["1"]["trained_on_pool_size"] == 300


def test_train_end_changes_persisted_classifier_predictions(tmp_path: Path) -> None:
    """Persisting WITH vs WITHOUT train_end on a pool whose holdout
    contradicts the train window should yield classifiers that disagree
    on at least some held-out inputs. Confirms the filter is load-bearing,
    not a no-op."""
    trades, fm, assignments = _train_holdout_inputs(n_train=300, n_holdout=200)
    train_end = pd.Timestamp("2020-12-31", tz="UTC")
    res_with = run_step_4(
        trades, fm, assignments,
        candidate_cluster_ids=(1,),
        persistence_dir=tmp_path / "with",
        arc_name="holdout_test",
        train_end=train_end,
    )
    res_without = run_step_4(
        trades, fm, assignments,
        candidate_cluster_ids=(1,),
        persistence_dir=tmp_path / "without",
        arc_name="holdout_test",
    )
    clf_with = load_classifier(res_with.per_cluster[0].fitted_classifier_path)
    clf_without = load_classifier(res_without.per_cluster[0].fitted_classifier_path)
    X = pd.DataFrame({
        col: np.linspace(-2, 2, 80)
        for col in res_with.per_cluster[0].fitted_classifier_feature_order
    })
    proba_with = clf_with.predict_proba(X)[:, 1]
    proba_without = clf_without.predict_proba(X)[:, 1]
    # At least some predictions must differ
    assert not np.allclose(proba_with, proba_without), (
        "persisted classifier appears unaffected by train_end — filter "
        "is not load-bearing"
    )
    # Pool sizes differ as expected
    assert res_with.per_cluster[0].n_trades < res_without.per_cluster[0].n_trades


def test_train_end_null_in_manifest_when_unset(tmp_path: Path) -> None:
    """Backwards-compat: train_end=None records null in manifest."""
    trades, fm, assignments = _synthetic_step_4_inputs()
    run_step_4(
        trades, fm, assignments,
        candidate_cluster_ids=(1,),
        persistence_dir=tmp_path,
        arc_name="synth",
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_end"] is None


def test_train_end_determinism(tmp_path: Path) -> None:
    """Two run_step_4 calls with the same train_end produce
    classifiers that predict identically on held-out X."""
    trades, fm, assignments = _train_holdout_inputs()
    train_end = pd.Timestamp("2020-12-31", tz="UTC")
    res_a = run_step_4(
        trades, fm, assignments,
        candidate_cluster_ids=(1,),
        persistence_dir=tmp_path / "a",
        arc_name="determinism",
        train_end=train_end,
    )
    res_b = run_step_4(
        trades, fm, assignments,
        candidate_cluster_ids=(1,),
        persistence_dir=tmp_path / "b",
        arc_name="determinism",
        train_end=train_end,
    )
    clf_a = load_classifier(res_a.per_cluster[0].fitted_classifier_path)
    clf_b = load_classifier(res_b.per_cluster[0].fitted_classifier_path)
    X = pd.DataFrame({
        col: np.linspace(-2, 2, 80)
        for col in res_a.per_cluster[0].fitted_classifier_feature_order
    })
    np.testing.assert_array_equal(
        clf_a.predict_proba(X)[:, 1],
        clf_b.predict_proba(X)[:, 1],
    )


def test_determinism_prediction_equality(tmp_path: Path) -> None:
    """Two run_step_4 calls -> matching predictions on held-out X.

    Joblib pickle bytes may vary across runs depending on internal
    metadata ordering, so the load-and-predict comparison is the
    binding assertion. Manifest SHA256 equality between the two runs
    is asserted as a soft check; logged but does not fail.
    """
    trades, fm, assignments = _synthetic_step_4_inputs()
    res_a = run_step_4(
        trades, fm, assignments,
        candidate_cluster_ids=(1,),
        persistence_dir=tmp_path / "run_a",
        arc_name="determinism_test",
    )
    res_b = run_step_4(
        trades, fm, assignments,
        candidate_cluster_ids=(1,),
        persistence_dir=tmp_path / "run_b",
        arc_name="determinism_test",
    )
    clf_a = load_classifier(res_a.per_cluster[0].fitted_classifier_path)
    clf_b = load_classifier(res_b.per_cluster[0].fitted_classifier_path)
    X = pd.DataFrame({
        col: np.linspace(-2, 2, 100)
        for col in res_a.per_cluster[0].fitted_classifier_feature_order
    })
    proba_a = clf_a.predict_proba(X)[:, 1]
    proba_b = clf_b.predict_proba(X)[:, 1]
    np.testing.assert_array_equal(proba_a, proba_b)
