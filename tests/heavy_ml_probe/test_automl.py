"""Tests for core.heavy_ml_probe.automl + the PR-B pipeline integration.

Per dispatch §6: covers end-to-end run, evaluation-cap respect,
determinism, NaN-AUC propagation, holdout-guard, and lineage-rejects-all
fast-fail.

Pool fixtures stay small (≤400 rows, ≤5 folds, ≤10 max_iter) so the
full file runs in well under the dispatch §4 ceiling (60s) — synthetic
pools exist to verify mechanics, not to stress FLAML.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import-guard FLAML — heavy dep; if missing the test module is skipped.
# Imports below are intentionally module-level (noqa E402) so the
# importorskip can gate the entire module.
flaml = pytest.importorskip("flaml")  # noqa: F841 — kept for diagnostics if a future test asserts FLAML version

from core.heavy_ml_probe.automl import (  # noqa: E402
    DOCUMENTED_FLAML_2_6_0_ESTIMATORS,
    AllFeaturesRejected,
    AutoMLResult,
    HoldoutGuardViolation,
    compute_budget_markdown,
    importance_to_dataframe,
    leaderboard_to_dataframe,
    run_automl,
)
from core.heavy_ml_probe.io import sha256_file  # noqa: E402
from core.heavy_ml_probe.pipeline import (  # noqa: E402
    AUTOML_REQUIRED_COLUMNS,
    load_config,
    run_pipeline,
    stable_payload_sha256,
)

DEFAULT_CONFIG_PATH = Path("configs/heavy_ml_probe/default.yaml")

# Small problem sizes to keep the file fast. n_folds=5 + max_iter=10
# keeps single-cluster wall-clock under a second on this hardware.
TEST_N_FOLDS = 5
TEST_MAX_ITER = 10
TEST_PERMUTATION_REPEATS = 3  # smaller than default to save synthetic-pool time
N_TRADES = 400


# ── Synthetic data fixtures ──────────────────────────────────────────


def _synthetic_pool(
    *,
    n: int = N_TRADES,
    seed: int = 1,
    last_entry: pd.Timestamp = pd.Timestamp("2020-12-15", tz="UTC"),
    include_y: bool = True,
    include_entry_time: bool = True,
    target_strength: float = 1.0,
    single_class_y: bool = False,
) -> pd.DataFrame:
    """Build a small classification pool with a learnable target.

    Default target: ``y = (a + 0.5*b + noise) > 0`` where noise is
    scaled by ``1 - target_strength``. ``target_strength=1.0`` →
    near-perfect; ``0.0`` → pure noise.

    ``entry_time`` is monotonically increasing over the trade window
    ending at ``last_entry`` so TimeSeriesSplit slices the trades
    causally.

    ``single_class_y=True`` overrides ``y`` to all-zeros; useful for
    NaN-propagation tests.
    """
    rng = np.random.default_rng(seed)
    n_features = 4
    feats = rng.standard_normal((n, n_features))
    df = pd.DataFrame({
        "a": feats[:, 0],
        "b": feats[:, 1],
        "c": feats[:, 2],
        "d": feats[:, 3],
    })
    if include_y:
        if single_class_y:
            df["y"] = 0
        else:
            noise = rng.standard_normal(n) * (1.0 - target_strength)
            df["y"] = ((df["a"] + 0.5 * df["b"] + noise) > 0).astype(int)
    if include_entry_time:
        # Even per-hour spacing ending at last_entry
        end = pd.Timestamp(last_entry)
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        idx = pd.date_range(end=end, periods=n, freq="h")
        df["entry_time"] = idx
    df["trade_id"] = np.arange(n)
    return df


def _synthetic_lineage_df() -> pd.DataFrame:
    """Four feature columns marked clean; metadata columns marked
    suspect so the lineage gate filters them out of the training matrix."""
    return pd.DataFrame([
        {"name": "trade_id", "causal_lineage": "suspect", "feature_class": "meta"},
        {"name": "a", "causal_lineage": "clean", "feature_class": "price_geometry"},
        {"name": "b", "causal_lineage": "clean", "feature_class": "price_geometry"},
        {"name": "c", "causal_lineage": "clean", "feature_class": "price_geometry"},
        {"name": "d", "causal_lineage": "clean", "feature_class": "price_geometry"},
        {"name": "entry_time", "causal_lineage": "suspect", "feature_class": "meta"},
        {"name": "y", "causal_lineage": "suspect", "feature_class": "meta"},
    ])


def _used_features(pool: pd.DataFrame) -> tuple[str, ...]:
    """Pretend the lineage gate let only the 4 feature columns through."""
    return tuple(c for c in pool.columns if c in {"a", "b", "c", "d"})


# ── Core: run_automl directly ────────────────────────────────────────


def test_run_automl_smoke():
    """End-to-end run on a learnable target — AutoML produces a non-NaN
    aggregate AUC and the artefact-shape helpers don't crash."""
    pool = _synthetic_pool()
    result = run_automl(
        pool=pool,
        used_features=_used_features(pool),
        train_end=pd.Timestamp("2021-01-01", tz="UTC"),
        n_folds=TEST_N_FOLDS,
        max_iter_per_fold=TEST_MAX_ITER,
        permutation_repeats=TEST_PERMUTATION_REPEATS,
    )
    assert isinstance(result, AutoMLResult)
    assert result.n_folds_total == TEST_N_FOLDS
    assert result.n_folds_valid == TEST_N_FOLDS  # learnable target → every fold valid
    assert np.isfinite(result.auc_mean)
    # The target is learnable, so AUC should be materially above 0.5;
    # be generous (0.6) to absorb FLAML small-pool wobble.
    assert result.auc_mean > 0.6, f"expected AUC > 0.6, got {result.auc_mean:.4f}"
    # Smoke-check artefact helpers
    lb = leaderboard_to_dataframe(result)
    imp = importance_to_dataframe(result)
    md = compute_budget_markdown(result, train_end=pd.Timestamp("2021-01-01", tz="UTC"))
    assert not lb.empty
    assert {"fold", "estimator", "best_loss", "auc_val", "auc_train",
            "modelcount", "max_iter"} <= set(lb.columns)
    assert {"feature", "importance_mean", "importance_std", "n_folds_present"} <= set(imp.columns)
    assert "compute budget audit" in md.lower()


# ── §3a: modelcount respects max_iter ─────────────────────────────────


def test_modelcount_does_not_exceed_max_iter():
    """Per dispatch §3a + PR-B empirical: modelcount ≤ max_iter per fold."""
    pool = _synthetic_pool()
    result = run_automl(
        pool=pool, used_features=_used_features(pool),
        train_end=pd.Timestamp("2021-01-01", tz="UTC"),
        n_folds=TEST_N_FOLDS, max_iter_per_fold=TEST_MAX_ITER,
        permutation_repeats=TEST_PERMUTATION_REPEATS,
    )
    for fr in result.fold_results:
        assert fr.modelcount <= fr.max_iter, (
            f"fold {fr.fold}: modelcount={fr.modelcount} > max_iter={fr.max_iter}"
        )
    # Total over folds should equal the per-fold sum.
    assert result.total_modelcount == sum(fr.modelcount for fr in result.fold_results)


def test_modelcount_matches_max_iter_on_learnable_target():
    """On a real (non-edge) problem with `max_iter=10`, FLAML 2.6.0 runs
    the full budget. Documents the §3a empirical: ratio == 1.0."""
    pool = _synthetic_pool(target_strength=0.5)  # noisier so FLAML doesn't early-stop
    result = run_automl(
        pool=pool, used_features=_used_features(pool),
        train_end=pd.Timestamp("2021-01-01", tz="UTC"),
        n_folds=TEST_N_FOLDS, max_iter_per_fold=TEST_MAX_ITER,
        permutation_repeats=TEST_PERMUTATION_REPEATS,
    )
    # On this small + noisy problem, each fold should hit the cap.
    for fr in result.fold_results:
        assert fr.modelcount == fr.max_iter, (
            f"fold {fr.fold}: modelcount={fr.modelcount} != max_iter={fr.max_iter} "
            "— FLAML budget accounting drift; review PR-B log §3a"
        )


# ── §3b: NaN-AUC propagation on single-class folds ────────────────────


def test_nan_auc_propagates_on_single_class_training():
    """Force a single-class fold and verify NaN propagates without
    poisoning the aggregate."""
    # Build a pool whose first 80% is all y=0 and last 20% is mostly
    # y=1. The earliest TimeSeriesSplit fold(s) train on the all-zero
    # slice → single-class training → AutoML skipped → NaN.
    n = 400
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
        "c": rng.standard_normal(n),
        "d": rng.standard_normal(n),
        "trade_id": np.arange(n),
        "entry_time": pd.date_range(end="2020-12-15", periods=n, freq="h", tz="UTC"),
    })
    # First 320 trades all y=0; last 80 are 50/50
    y = np.zeros(n, dtype=int)
    cutoff = int(0.8 * n)
    y[cutoff:] = (rng.random(n - cutoff) > 0.5).astype(int)
    df["y"] = y

    # Suppress the "single-class training" UserWarning we emit on purpose.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = run_automl(
            pool=df, used_features=("a", "b", "c", "d"),
            train_end=pd.Timestamp("2021-01-01", tz="UTC"),
            n_folds=TEST_N_FOLDS, max_iter_per_fold=TEST_MAX_ITER,
            permutation_repeats=TEST_PERMUTATION_REPEATS,
        )
    # At least one fold should have been skipped / NaN.
    nan_folds = [fr for fr in result.fold_results if not np.isfinite(fr.auc_val)]
    assert len(nan_folds) >= 1, "expected at least one single-class fold to NaN out"
    # n_folds_valid reflects the live count.
    assert result.n_folds_valid == result.n_folds_total - len(nan_folds)
    # auc_mean must be finite (nanmean) as long as at least one fold was valid.
    if result.n_folds_valid > 0:
        assert np.isfinite(result.auc_mean)
    else:
        assert not np.isfinite(result.auc_mean)


def test_nan_auc_aggregate_uses_nanmean():
    """Trivial unit-check on the aggregation logic: when every fold is
    NaN, auc_mean is NaN; when some are valid, auc_mean ignores the
    NaNs (nanmean semantics)."""
    pool_all_zero = _synthetic_pool(single_class_y=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # nanmean-of-all-nan
            result = run_automl(
                pool=pool_all_zero, used_features=_used_features(pool_all_zero),
                train_end=pd.Timestamp("2021-01-01", tz="UTC"),
                n_folds=TEST_N_FOLDS, max_iter_per_fold=TEST_MAX_ITER,
                permutation_repeats=TEST_PERMUTATION_REPEATS,
            )
    # Every fold should NaN out (single-class training across the board).
    assert result.n_folds_valid == 0
    assert not np.isfinite(result.auc_mean)
    for fr in result.fold_results:
        assert not np.isfinite(fr.auc_val)


# ── Guards: holdout + all-features-rejected ──────────────────────────


def test_holdout_guard_violation():
    """Pool with an entry_time inside the holdout window must raise
    HoldoutGuardViolation before any AutoML call."""
    pool = _synthetic_pool(last_entry=pd.Timestamp("2022-06-01", tz="UTC"))
    with pytest.raises(HoldoutGuardViolation, match="holdout-guard violated"):
        run_automl(
            pool=pool, used_features=_used_features(pool),
            train_end=pd.Timestamp("2021-01-01", tz="UTC"),
            n_folds=TEST_N_FOLDS, max_iter_per_fold=TEST_MAX_ITER,
            permutation_repeats=TEST_PERMUTATION_REPEATS,
        )


def test_holdout_guard_at_boundary_raises():
    """entry_time exactly equal to train_end must also be rejected
    (strict less-than per L_PROTOCOL §1)."""
    boundary = pd.Timestamp("2021-01-01", tz="UTC")
    pool = _synthetic_pool(last_entry=boundary)
    with pytest.raises(HoldoutGuardViolation):
        run_automl(
            pool=pool, used_features=_used_features(pool),
            train_end=boundary,
            n_folds=TEST_N_FOLDS, max_iter_per_fold=TEST_MAX_ITER,
            permutation_repeats=TEST_PERMUTATION_REPEATS,
        )


def test_all_features_rejected_fast_fail():
    pool = _synthetic_pool()
    with pytest.raises(AllFeaturesRejected, match="empty design matrix"):
        run_automl(
            pool=pool, used_features=(),  # nothing past the gate
            train_end=pd.Timestamp("2021-01-01", tz="UTC"),
            n_folds=TEST_N_FOLDS, max_iter_per_fold=TEST_MAX_ITER,
            permutation_repeats=TEST_PERMUTATION_REPEATS,
        )


def test_missing_required_column_raises():
    pool = _synthetic_pool(include_y=False)
    with pytest.raises(ValueError, match="must contain 'y' column"):
        run_automl(
            pool=pool, used_features=_used_features(pool),
            train_end=pd.Timestamp("2021-01-01", tz="UTC"),
            n_folds=TEST_N_FOLDS, max_iter_per_fold=TEST_MAX_ITER,
            permutation_repeats=TEST_PERMUTATION_REPEATS,
        )


def test_non_binary_target_raises():
    pool = _synthetic_pool()
    pool["y"] = pool["y"] + 2  # values {2, 3}, not {0, 1}
    with pytest.raises(ValueError, match="target must be binary"):
        run_automl(
            pool=pool, used_features=_used_features(pool),
            train_end=pd.Timestamp("2021-01-01", tz="UTC"),
            n_folds=TEST_N_FOLDS, max_iter_per_fold=TEST_MAX_ITER,
            permutation_repeats=TEST_PERMUTATION_REPEATS,
        )


# ── Determinism: two runs → identical artefacts ───────────────────────


def test_two_run_determinism_automl_aggregates():
    """Same input → same auc_mean, modelcount, leaderboard, importance."""
    pool = _synthetic_pool()
    feats = _used_features(pool)
    end = pd.Timestamp("2021-01-01", tz="UTC")
    kwargs = dict(
        train_end=end, n_folds=TEST_N_FOLDS,
        max_iter_per_fold=TEST_MAX_ITER,
        permutation_repeats=TEST_PERMUTATION_REPEATS,
    )
    r1 = run_automl(pool=pool.copy(), used_features=feats, **kwargs)
    r2 = run_automl(pool=pool.copy(), used_features=feats, **kwargs)
    assert r1.auc_mean == r2.auc_mean
    assert r1.auc_std == r2.auc_std
    assert r1.total_modelcount == r2.total_modelcount
    # Leaderboards: same content
    pd.testing.assert_frame_equal(
        r1.leaderboard.reset_index(drop=True),
        r2.leaderboard.reset_index(drop=True),
    )
    # Importance: same content
    pd.testing.assert_frame_equal(
        r1.importance.reset_index(drop=True),
        r2.importance.reset_index(drop=True),
    )


# ── FLAML estimator-set capture ──────────────────────────────────────


def test_estimator_list_recorded_per_fold():
    """The actual FLAML 2.6.0 estimator list is captured so cross-env
    audit catches drift if it ever shows up."""
    pool = _synthetic_pool()
    result = run_automl(
        pool=pool, used_features=_used_features(pool),
        train_end=pd.Timestamp("2021-01-01", tz="UTC"),
        n_folds=TEST_N_FOLDS, max_iter_per_fold=TEST_MAX_ITER,
        permutation_repeats=TEST_PERMUTATION_REPEATS,
    )
    for fr in result.fold_results:
        # FLAML defaults can drift across minor versions; the documented
        # set is the FLAML 2.6.0 baseline. We assert non-empty + a couple
        # of always-present learners rather than equality so a benign
        # FLAML upgrade doesn't break the test.
        assert len(fr.estimator_list) > 0
        assert {"lgbm", "rf"}.issubset(set(fr.estimator_list))
    # The constant lists the documented baseline; test stays for audit.
    assert "lgbm" in DOCUMENTED_FLAML_2_6_0_ESTIMATORS


# ── Pipeline integration: artefacts + manifest + determinism ─────────


def _pipeline_pool(tmp_path: Path) -> Path:
    pool = _synthetic_pool()
    p = tmp_path / "pool.parquet"
    pool.to_parquet(p, compression="snappy", index=False)
    return p


def _override_config(tmp_path: Path) -> Path:
    """Write a config that overrides the heavy production defaults
    (n_folds=11, max_iter=1000) with test-fast values."""
    import yaml
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["automl"]["n_folds"] = TEST_N_FOLDS
    raw["automl"]["max_iter_per_fold"] = TEST_MAX_ITER
    p = tmp_path / "test_config.yaml"
    p.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    return p


@pytest.fixture
def pipeline_run(tmp_path):
    """Two pipeline runs against the same pool + override config."""
    pool_path = _pipeline_pool(tmp_path)
    cfg_path = _override_config(tmp_path)
    lineage = _synthetic_lineage_df()

    def _run(label: str):
        out_root = tmp_path / label
        cfg = load_config(
            cfg_path,
            arc_name="test_arc",
            cluster_id=0,
            pool_path=pool_path,
            output_root=out_root,
        )
        return run_pipeline(cfg, lineage_df=lineage)

    return _run


def test_pipeline_writes_full_automl_artefact_set(pipeline_run):
    r = pipeline_run("run1")
    # AutoML stage ran (pool has entry_time + y)
    assert r.automl_skip_reason == "ok"
    assert r.automl_result is not None
    assert r.automl_leaderboard_path is not None and r.automl_leaderboard_path.exists()
    assert r.automl_importance_path is not None and r.automl_importance_path.exists()
    assert r.compute_budget_path is not None and r.compute_budget_path.exists()
    # Manifest lists at minimum the four AutoML artefacts. PR-C/D add
    # meta-label + survival when the pool carries those schemas — this
    # test's pool doesn't, so only the AutoML set + the always-present
    # stub_summary + aggregate compute_budget_used should appear.
    payload = json.loads(r.step4_manifest_path.read_text(encoding="utf-8"))
    required = {
        "stub_summary",
        "automl_leaderboard",
        "automl_feature_importance",
        "compute_budget_used",
    }
    assert required <= set(payload["artefacts"].keys())
    # Each path is recorded relative to step4_dir with forward slashes.
    for name in ("automl_leaderboard", "automl_feature_importance",
                 "compute_budget_used"):
        entry = payload["artefacts"][name]
        assert "\\" not in entry["path"]
        # SHA256 matches the on-disk file.
        on_disk = r.step4_manifest_path.parent / entry["path"]
        assert entry["sha256"] == sha256_file(on_disk)

    # Manifest's automl extras block
    automl_block = payload["automl"]
    assert automl_block["skip_reason"] == "ok"
    assert automl_block["n_folds_total"] == TEST_N_FOLDS
    assert automl_block["n_folds_valid"] >= 1
    assert automl_block["total_modelcount"] >= 1
    assert automl_block["n_features_used"] == 4


def test_pipeline_automl_two_run_determinism(pipeline_run):
    r1 = pipeline_run("run1")
    r2 = pipeline_run("run2")
    # All four artefact sha256s identical across runs
    for path_attr in ("stub_summary_path", "automl_leaderboard_path",
                      "automl_importance_path", "compute_budget_path"):
        p1 = getattr(r1, path_attr)
        p2 = getattr(r2, path_attr)
        assert sha256_file(p1) == sha256_file(p2), f"{path_attr} differs across runs"
    # Manifest stable payload identical.
    assert stable_payload_sha256(r1.step4_manifest_path) == stable_payload_sha256(
        r2.step4_manifest_path
    )


def test_pipeline_holdout_guard_violation_surfaces(tmp_path):
    """A pool whose max entry_time is inside the holdout window must
    raise HoldoutGuardViolation. The pipeline writes a partial manifest
    recording the skip reason before re-raising."""
    pool = _synthetic_pool(last_entry=pd.Timestamp("2022-06-01", tz="UTC"))
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path, compression="snappy", index=False)
    cfg_path = _override_config(tmp_path)
    cfg = load_config(
        cfg_path, arc_name="x", cluster_id=0,
        pool_path=pool_path, output_root=tmp_path / "out",
    )
    with pytest.raises(HoldoutGuardViolation):
        run_pipeline(cfg, lineage_df=_synthetic_lineage_df())
    # Partial manifest was emitted.
    mp = cfg.step4_dir / "manifest.json"
    assert mp.exists()
    payload = json.loads(mp.read_text(encoding="utf-8"))
    assert payload["automl"]["skip_reason"] == "holdout_guard_violation"


def test_pipeline_skip_reason_missing_entry_time(tmp_path):
    """PR-A's synthetic pool has no entry_time — pipeline should skip
    AutoML with the right reason rather than crash."""
    pool = _synthetic_pool(include_entry_time=False, include_y=False)
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path, compression="snappy", index=False)
    cfg_path = _override_config(tmp_path)
    cfg = load_config(
        cfg_path, arc_name="x", cluster_id=0,
        pool_path=pool_path, output_root=tmp_path / "out",
    )
    r = run_pipeline(cfg, lineage_df=_synthetic_lineage_df())
    assert r.automl_skip_reason == "missing_entry_time"
    assert r.automl_result is None
    # When AutoML skips, only the always-present pipeline artefacts
    # land: stub_summary + the PR-E aggregate compute_budget_used.md.
    payload = json.loads(r.step4_manifest_path.read_text(encoding="utf-8"))
    assert set(payload["artefacts"].keys()) == {"stub_summary", "compute_budget_used"}
    assert payload["automl"]["skip_reason"] == "missing_entry_time"


def test_pipeline_skip_reason_missing_target(tmp_path):
    pool = _synthetic_pool(include_y=False)
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path, compression="snappy", index=False)
    cfg_path = _override_config(tmp_path)
    cfg = load_config(
        cfg_path, arc_name="x", cluster_id=0,
        pool_path=pool_path, output_root=tmp_path / "out",
    )
    r = run_pipeline(cfg, lineage_df=_synthetic_lineage_df())
    assert r.automl_skip_reason == "missing_y"
    assert r.automl_result is None


def test_pipeline_skip_reason_no_clean_features(tmp_path):
    """When the lineage gate rejects every column, AutoML skips with
    reason `no_clean_features` (and does NOT raise — the pipeline
    surfaces the situation through the manifest)."""
    pool = _synthetic_pool()
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path, compression="snappy", index=False)
    cfg_path = _override_config(tmp_path)
    cfg = load_config(
        cfg_path, arc_name="x", cluster_id=0,
        pool_path=pool_path, output_root=tmp_path / "out",
    )
    # Mark every feature suspect so the gate rejects them all
    all_suspect = _synthetic_lineage_df().assign(causal_lineage="suspect")
    r = run_pipeline(cfg, lineage_df=all_suspect)
    assert r.automl_skip_reason == "no_clean_features"
    assert r.automl_result is None


def test_pool_required_columns_constant_matches_implementation():
    assert AUTOML_REQUIRED_COLUMNS == ("entry_time", "y")
