"""Tests for core.heavy_ml_probe.meta_labeling + PR-C pipeline integration.

Per dispatch §6: covers target construction (incl. edge cases per §1),
end-to-end run, classifier persistence + sha256 manifest, two-run
determinism, threshold-sweep mean-R math correctness, single-class
fold NaN propagation (reuses PR-B pattern), and HALT-loud pool-schema
validation.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# Import-guard FLAML — heavy dep; if missing the test module is skipped.
flaml = pytest.importorskip("flaml")  # noqa: F841

from core.heavy_ml_probe.io import sha256_file  # noqa: E402
from core.heavy_ml_probe.meta_labeling import (  # noqa: E402
    DEFAULT_THRESHOLD_SWEEP,
    META_LABEL_TARGET_COL,
    REQUIRED_POOL_COLUMNS,
    MetaLabelResult,
    PoolSchemaError,
    build_meta_label_target,
    persist_fold_classifiers,
    run_meta_labeling,
    stable_classifier_manifest_sha256,
    threshold_sweep,
)
from core.heavy_ml_probe.pipeline import (  # noqa: E402
    load_config,
    run_pipeline,
    stable_payload_sha256,
)

DEFAULT_CONFIG_PATH = Path("configs/heavy_ml_probe/default.yaml")

# Small scale for fast tests (same convention as test_automl.py).
TEST_N_FOLDS = 5
TEST_MAX_ITER = 10
TEST_PERMUTATION_REPEATS = 3
N_TRADES = 400


# ── Synthetic data ──────────────────────────────────────────────────


def _meta_label_pool(
    *,
    n: int = N_TRADES,
    seed: int = 1,
    last_entry: pd.Timestamp = pd.Timestamp("2020-12-15", tz="UTC"),
    target_strength: float = 0.7,
) -> pd.DataFrame:
    """Build a pool with the meta-label schema + a target the
    classifier can learn from.

    Synthetic semantics:

      * 4 features (a, b, c, d) plus entry_time + trade_id.
      * ``reached_1r_proxy = sigmoid(a + 0.5*b) > 0.5`` with optional
        noise — this drives the meta-label target via the schema columns.
      * Each trade gets a ``bars_to_1r_mfe`` (NaN if not reached),
        ``bars_held``, ``exit_reason`` ("sl"/"time_exit"/"tp"), and
        ``final_r`` proportional to whether the trade reached +1R.
    """
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n, 4))
    df = pd.DataFrame({
        "a": feats[:, 0],
        "b": feats[:, 1],
        "c": feats[:, 2],
        "d": feats[:, 3],
    })

    end = pd.Timestamp(last_entry)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    df["entry_time"] = pd.date_range(end=end, periods=n, freq="h")
    df["trade_id"] = np.arange(n)

    # Probability of reaching +1R MFE before SL — driven by (a, b).
    noise = rng.standard_normal(n) * (1.0 - target_strength)
    score = df["a"] + 0.5 * df["b"] + noise
    reached = (score > 0).values  # roughly 50/50

    bars_held = rng.integers(low=3, high=30, size=n)
    bars_to_1r_mfe = np.where(
        reached,
        rng.integers(low=1, high=np.maximum(bars_held, 2), size=n),
        np.nan,
    )
    exit_reasons = np.where(
        reached & (rng.random(n) > 0.3),
        "tp",
        np.where(reached, "time_exit", "sl"),
    )
    # Realised R: reached → uniform[0.5, 3.0]; not reached → uniform[-1.0, 0.5]
    final_r = np.where(
        reached,
        rng.uniform(0.5, 3.0, size=n),
        rng.uniform(-1.0, 0.5, size=n),
    )

    df["bars_to_1r_mfe"] = bars_to_1r_mfe
    df["bars_held"] = bars_held
    df["exit_reason"] = exit_reasons
    df["final_r"] = final_r
    # Vanilla AutoML target so the pipeline's PR-B stage also runs in
    # the end-to-end test. Production pools have both; testing only the
    # meta-label half would leave AutoML skipping with "missing_y" and
    # the artefact-set assertion would only see 3 instead of 6 files.
    df["y"] = reached.astype(int)
    return df


def _meta_label_lineage_df() -> pd.DataFrame:
    """Four feature columns marked clean; everything else marked
    suspect so the lineage gate filters them out."""
    return pd.DataFrame([
        {"name": "a", "causal_lineage": "clean", "feature_class": "price_geometry"},
        {"name": "b", "causal_lineage": "clean", "feature_class": "price_geometry"},
        {"name": "c", "causal_lineage": "clean", "feature_class": "price_geometry"},
        {"name": "d", "causal_lineage": "clean", "feature_class": "price_geometry"},
        {"name": "trade_id", "causal_lineage": "suspect", "feature_class": "meta"},
        {"name": "entry_time", "causal_lineage": "suspect", "feature_class": "meta"},
        {"name": "bars_to_1r_mfe", "causal_lineage": "suspect", "feature_class": "meta"},
        {"name": "bars_held", "causal_lineage": "suspect", "feature_class": "meta"},
        {"name": "exit_reason", "causal_lineage": "suspect", "feature_class": "meta"},
        {"name": "final_r", "causal_lineage": "suspect", "feature_class": "meta"},
    ])


# ── Target construction (dispatch §1 edge cases) ────────────────────


def test_target_reached_before_close():
    """Trade reaches +1R strictly before close → target = 1."""
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": 3.0,
        "bars_held": 10.0,
        "exit_reason": "tp",
        "final_r": 2.0,
    }])
    y = build_meta_label_target(pool)
    assert y.tolist() == [1]


def test_target_never_reached_time_exit():
    """Trade never reaches +1R, exits at time-exit → target = 0."""
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": np.nan,
        "bars_held": 30.0,
        "exit_reason": "time_exit",
        "final_r": -0.2,
    }])
    y = build_meta_label_target(pool)
    assert y.tolist() == [0]


def test_target_sl_on_entry_bar():
    """Trade hits SL on entry bar (MFE never reached +1R) → target = 0."""
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": np.nan,
        "bars_held": 1.0,
        "exit_reason": "sl",
        "final_r": -1.0,
    }])
    y = build_meta_label_target(pool)
    assert y.tolist() == [0]


def test_target_same_bar_tie_sl_wins():
    """Per dispatch §1: +1R and SL on the same bar → SL wins → target = 0."""
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": 5.0,
        "bars_held": 5.0,
        "exit_reason": "sl",
        "final_r": -1.0,
    }])
    y = build_meta_label_target(pool)
    assert y.tolist() == [0]


def test_target_same_bar_tie_non_sl_counts_as_reached():
    """If MFE and time-exit fall on same bar (not SL), the +1R event
    happened before the (non-adverse) close → target = 1."""
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": 7.0,
        "bars_held": 7.0,
        "exit_reason": "time_exit",
        "final_r": 1.2,
    }])
    y = build_meta_label_target(pool)
    assert y.tolist() == [1]


def test_target_exact_1r_on_first_bar():
    """Trade hits exactly +1R on bar 1, then moves to close at +1.2R
    over 10 bars → target = 1 (reached strictly before close)."""
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": 1.0,
        "bars_held": 10.0,
        "exit_reason": "time_exit",
        "final_r": 1.2,
    }])
    y = build_meta_label_target(pool)
    assert y.tolist() == [1]


def test_target_case_insensitive_sl_exit_reason():
    """Exit-reason comparison is case-insensitive — 'SL' / 'Sl' / 'sl'
    all trigger the same-bar tie-break."""
    for er in ("sl", "SL", "Sl", "sL"):
        pool = pd.DataFrame([{
            "bars_to_1r_mfe": 5.0, "bars_held": 5.0,
            "exit_reason": er, "final_r": -1.0,
        }])
        y = build_meta_label_target(pool)
        assert y.tolist() == [0], f"exit_reason={er!r} should trigger SL tie-break"


def test_target_same_bar_hard_sl_is_loss():
    """HONEST_ENGINE_SWEEP.md FLAG-D2 regression: the pool simulators emit
    ``exit_reason='hard_sl'`` (not ``'sl'``). A same-bar +1R/SL tie with a
    ``hard_sl`` exit MUST label 0 (LOSS) — the old ``== 'sl'`` compare let it
    through as a WIN."""
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": 5.0, "bars_held": 5.0,
        "exit_reason": "hard_sl", "final_r": -1.0,
    }])
    assert build_meta_label_target(pool).tolist() == [0]


def test_target_vectorised_over_pool():
    """Mixed cases vectorise correctly — incl. the producer-native
    ``hard_sl`` spelling alongside the legacy ``sl``."""
    pool = pd.DataFrame([
        {"bars_to_1r_mfe": 2.0, "bars_held": 5.0, "exit_reason": "tp", "final_r": 2.0},          # 1
        {"bars_to_1r_mfe": np.nan, "bars_held": 10.0, "exit_reason": "sl", "final_r": -1.0},      # 0
        {"bars_to_1r_mfe": 5.0, "bars_held": 5.0, "exit_reason": "sl", "final_r": -1.0},          # 0 (tie + SL)
        {"bars_to_1r_mfe": 5.0, "bars_held": 5.0, "exit_reason": "hard_sl", "final_r": -1.0},     # 0 (tie + hard_sl)
        {"bars_to_1r_mfe": 5.0, "bars_held": 5.0, "exit_reason": "time_exit", "final_r": 1.0},    # 1 (tie + non-SL)
        {"bars_to_1r_mfe": np.nan, "bars_held": 30.0, "exit_reason": "time_exit", "final_r": 0.3},  # 0
    ])
    y = build_meta_label_target(pool)
    assert y.tolist() == [1, 0, 0, 0, 1, 0]


def test_target_halts_loud_on_missing_column():
    """Per dispatch §1 last paragraph: schema mismatches HALT loudly."""
    pool = pd.DataFrame([{"bars_to_1r_mfe": 1.0, "bars_held": 2.0}])  # no exit_reason / final_r
    with pytest.raises(PoolSchemaError, match="requires pool columns"):
        build_meta_label_target(pool)


def test_target_halts_loud_on_nan_bars_held():
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": 1.0, "bars_held": np.nan,
        "exit_reason": "sl", "final_r": -1.0,
    }])
    with pytest.raises(ValueError, match="bars_held column has NaN"):
        build_meta_label_target(pool)


def test_target_rejects_non_default_threshold_override():
    """Sensitivity probes need an upstream pre-computed column; we
    refuse silent column substitution per docstring."""
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": 5.0, "bars_held": 10.0,
        "exit_reason": "tp", "final_r": 1.0,
    }])
    with pytest.raises(ValueError, match="not supported in"):
        build_meta_label_target(pool, mfe_r_threshold=0.5)


# ── Threshold sweep math (dispatch §3) ──────────────────────────────


def test_threshold_sweep_hand_computed_mean_r():
    """Synthetic 6-trade pool with hand-computed kept/dropped mean R."""
    # trade_ids 0..5; OOF preds chosen so threshold 0.5 splits cleanly
    oof = pd.DataFrame([
        {"trade_id": 0, "fold": 1, "y_true": 1, "y_pred_proba": 0.9},
        {"trade_id": 1, "fold": 1, "y_true": 1, "y_pred_proba": 0.7},
        {"trade_id": 2, "fold": 2, "y_true": 1, "y_pred_proba": 0.6},
        {"trade_id": 3, "fold": 2, "y_true": 0, "y_pred_proba": 0.4},
        {"trade_id": 4, "fold": 3, "y_true": 0, "y_pred_proba": 0.2},
        {"trade_id": 5, "fold": 3, "y_true": 0, "y_pred_proba": 0.1},
    ])
    realised_r = pd.Series(
        [2.0, 1.5, 1.0, -0.5, -1.0, -1.0],
        index=[0, 1, 2, 3, 4, 5],
    )
    sweep = threshold_sweep(oof, realised_r, thresholds=(0.5,))
    [row] = sweep.to_dict("records")
    assert row["threshold"] == 0.5
    # At threshold 0.5: kept = trades 0,1,2 (all y_true=1); dropped = 3,4,5 (all y_true=0)
    assert row["n_trades_kept"] == 3
    assert row["n_trades_dropped"] == 3
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(1.0)
    assert row["f1"] == pytest.approx(1.0)
    # Mean R kept = (2.0 + 1.5 + 1.0) / 3 = 1.5
    assert row["mean_r_kept_set"] == pytest.approx(1.5)
    # Mean R dropped = (-0.5 + -1.0 + -1.0) / 3 ≈ -0.8333
    assert row["mean_r_dropped_set"] == pytest.approx(-0.8333333333, rel=1e-6)
    # Edge lift = kept - dropped = 1.5 - (-0.8333) ≈ 2.3333
    assert row["edge_lift_r"] == pytest.approx(2.3333333333, rel=1e-6)


def test_threshold_sweep_empty_kept_set():
    """Threshold higher than any prediction → kept set empty → NaN mean R."""
    oof = pd.DataFrame([
        {"trade_id": 0, "fold": 1, "y_true": 0, "y_pred_proba": 0.1},
        {"trade_id": 1, "fold": 1, "y_true": 1, "y_pred_proba": 0.2},
    ])
    realised_r = pd.Series([0.5, 1.0], index=[0, 1])
    sweep = threshold_sweep(oof, realised_r, thresholds=(0.99,))
    [row] = sweep.to_dict("records")
    assert row["n_trades_kept"] == 0
    assert row["n_trades_dropped"] == 2
    assert not np.isfinite(row["mean_r_kept_set"])
    assert row["mean_r_dropped_set"] == pytest.approx(0.75)
    assert not np.isfinite(row["edge_lift_r"])


def test_threshold_sweep_missing_required_oof_columns():
    bad_oof = pd.DataFrame([{"trade_id": 0, "y_true": 1}])  # missing y_pred_proba
    with pytest.raises(ValueError, match="requires OOF columns"):
        threshold_sweep(bad_oof, pd.Series(dtype=float), thresholds=(0.5,))


# ── End-to-end run + persistence ────────────────────────────────────


def _override_config(tmp_path: Path) -> Path:
    import yaml
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["automl"]["n_folds"] = TEST_N_FOLDS
    raw["automl"]["max_iter_per_fold"] = TEST_MAX_ITER
    p = tmp_path / "test_config.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return p


@pytest.fixture
def pipeline_run(tmp_path):
    pool = _meta_label_pool()
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path, compression="snappy", index=False)
    cfg_path = _override_config(tmp_path)
    lineage = _meta_label_lineage_df()

    def _run(label: str):
        out_root = tmp_path / label
        cfg = load_config(
            cfg_path, arc_name="test_arc", cluster_id=0,
            pool_path=pool_path, output_root=out_root,
        )
        return run_pipeline(cfg, lineage_df=lineage)

    return _run


def test_pipeline_writes_meta_label_artefacts(pipeline_run):
    r = pipeline_run("run1")
    assert r.meta_label_skip_reason == "ok"
    assert r.meta_label_result is not None
    assert r.meta_label_results_path is not None and r.meta_label_results_path.exists()
    assert (
        r.meta_label_classifier_manifest_path is not None
        and r.meta_label_classifier_manifest_path.exists()
    )

    # Manifest should list at minimum the 6 meta-label / AutoML artefacts.
    # PR-D adds survival_model_results + survival_classifier_manifest
    # when the pool also carries the survival schema (which the
    # test_meta_labeling fixture does — same MFE columns).
    payload = json.loads(r.step4_manifest_path.read_text(encoding="utf-8"))
    required = {
        "stub_summary",
        "automl_leaderboard",
        "automl_feature_importance",
        "compute_budget_used",
        "meta_label_results",
        "meta_label_classifier_manifest",
    }
    assert required <= set(payload["artefacts"].keys())
    # Meta-label extras block
    ml = payload["meta_label"]
    assert ml["skip_reason"] == "ok"
    assert ml["n_folds_total"] == TEST_N_FOLDS
    assert ml["target_distribution"]["0"] + ml["target_distribution"]["1"] == N_TRADES
    assert ml["n_thresholds_swept"] == len(DEFAULT_THRESHOLD_SWEEP)
    assert isinstance(ml["positive_rate"], float)


def test_pipeline_classifier_manifest_lists_every_fold(pipeline_run):
    r = pipeline_run("run1")
    cm = json.loads(r.meta_label_classifier_manifest_path.read_text(encoding="utf-8"))
    assert cm["sub_protocol"] == "heavy_ml_probe"
    assert cm["stage"] == "meta_label"
    assert cm["cluster_id"] == 0
    assert len(cm["folds"]) == TEST_N_FOLDS
    assert cm["n_folds_persisted"] >= 1
    # Every persisted fold has a valid sha256 + on-disk file
    persisted = [f for f in cm["folds"] if f["path"] is not None]
    for f in persisted:
        on_disk = r.meta_label_classifier_manifest_path.parent / f["path"]
        assert on_disk.exists()
        assert sha256_file(on_disk) == f["sha256"]
        # Confirm joblib can load + the loaded object exposes predict_proba
        clf = joblib.load(on_disk)
        assert hasattr(clf, "predict_proba")
        assert hasattr(clf, "get_params")


def test_pipeline_meta_label_results_csv_columns(pipeline_run):
    r = pipeline_run("run1")
    df = pd.read_csv(r.meta_label_results_path)
    expected_cols = {
        "threshold", "precision", "recall", "f1",
        "n_trades_kept", "n_trades_dropped",
        "mean_r_kept_set", "mean_r_dropped_set", "edge_lift_r",
        "target_n_pos", "target_n_neg", "target_positive_rate",
        "aggregate_oof_auc_mean", "aggregate_oof_auc_std",
        "n_folds_valid", "n_folds_total",
    }
    assert expected_cols <= set(df.columns)
    assert len(df) == len(DEFAULT_THRESHOLD_SWEEP)
    assert list(df["threshold"]) == list(DEFAULT_THRESHOLD_SWEEP)


def test_pipeline_meta_label_edge_lift_on_learnable_target(pipeline_run):
    """Sanity check: at threshold 0.5, kept-set mean R should exceed
    dropped-set mean R on a learnable synthetic target."""
    r = pipeline_run("run1")
    df = pd.read_csv(r.meta_label_results_path)
    # Pick the 0.5 row (it's deterministic across runs)
    row = df[df["threshold"] == 0.5].iloc[0]
    if row["n_trades_kept"] > 0 and row["n_trades_dropped"] > 0:
        # The synthetic target is learnable; kept set should out-perform.
        # Soft assertion — small synthetic pool can wobble, so just
        # require positive edge lift, not a specific magnitude.
        assert row["edge_lift_r"] > 0, (
            f"expected positive edge lift on learnable synthetic; got "
            f"kept={row['mean_r_kept_set']:.4f} vs dropped="
            f"{row['mean_r_dropped_set']:.4f} (lift={row['edge_lift_r']:.4f})"
        )


# ── Determinism ──────────────────────────────────────────────────────


def test_pipeline_meta_label_two_run_determinism(pipeline_run):
    """All seven on-disk artefacts byte-identical across two runs;
    classifier-manifest stable-payload sha256 matches."""
    r1 = pipeline_run("run1")
    r2 = pipeline_run("run2")
    for path_attr in (
        "stub_summary_path",
        "automl_leaderboard_path", "automl_importance_path",
        "compute_budget_path",
        "meta_label_results_path",
    ):
        p1 = getattr(r1, path_attr)
        p2 = getattr(r2, path_attr)
        assert sha256_file(p1) == sha256_file(p2), f"{path_attr} differs across runs"

    # Top-level manifest stable payload
    assert stable_payload_sha256(r1.step4_manifest_path) == stable_payload_sha256(
        r2.step4_manifest_path
    )
    # Classifier-manifest stable payload (timestamp + sha256s of pickles)
    cm1 = stable_classifier_manifest_sha256(r1.meta_label_classifier_manifest_path)
    cm2 = stable_classifier_manifest_sha256(r2.meta_label_classifier_manifest_path)
    assert cm1 == cm2


def _params_equal_nan_aware(p1: dict, p2: dict) -> bool:
    """Compare two sklearn ``get_params()`` dicts with NaN-aware
    equality. Required because XGBoost's params include
    ``'missing': nan``, and Python's ``nan != nan``."""
    if set(p1.keys()) != set(p2.keys()):
        return False
    for k in p1:
        v1, v2 = p1[k], p2[k]
        if isinstance(v1, float) and isinstance(v2, float):
            if np.isnan(v1) and np.isnan(v2):
                continue  # both NaN → treat as equal
        if v1 != v2:
            return False
    return True


def test_pipeline_classifier_pickle_params_match_across_runs(pipeline_run):
    """Per dispatch §6 #7: joblib pickle bytes can vary across joblib
    versions, but the get_params() snapshot of the loaded classifier
    should be identical across two runs at the same joblib version.

    NaN-aware comparison handles XGBoost's ``'missing': nan`` param
    (Python's ``nan != nan`` would otherwise produce false negatives).
    """
    r1 = pipeline_run("run1")
    r2 = pipeline_run("run2")
    cm1 = json.loads(r1.meta_label_classifier_manifest_path.read_text(encoding="utf-8"))
    cm2 = json.loads(r2.meta_label_classifier_manifest_path.read_text(encoding="utf-8"))
    persisted1 = [f for f in cm1["folds"] if f["path"] is not None]
    persisted2 = [f for f in cm2["folds"] if f["path"] is not None]
    assert len(persisted1) == len(persisted2)
    for f1, f2 in zip(persisted1, persisted2):
        clf1 = joblib.load(r1.meta_label_classifier_manifest_path.parent / f1["path"])
        clf2 = joblib.load(r2.meta_label_classifier_manifest_path.parent / f2["path"])
        assert _params_equal_nan_aware(clf1.get_params(), clf2.get_params()), (
            f"params differ for fold {f1['fold_id']} "
            f"({f1['classifier_type']}): "
            f"{clf1.get_params()!r} vs {clf2.get_params()!r}"
        )


# ── Skip-path tests ─────────────────────────────────────────────────


def test_pipeline_skip_meta_label_when_pool_missing_columns(tmp_path):
    """Pool without meta-label columns → AutoML runs, meta-labeling
    skips with the right reason."""
    # Same synthetic pool from test_automl.py: has entry_time + y but no
    # meta-label columns.
    rng = np.random.default_rng(2)
    n = 200
    df = pd.DataFrame({
        "a": rng.standard_normal(n), "b": rng.standard_normal(n),
        "c": rng.standard_normal(n), "d": rng.standard_normal(n),
        "trade_id": np.arange(n),
        "entry_time": pd.date_range(end="2020-12-15", periods=n, freq="h", tz="UTC"),
    })
    df["y"] = ((df["a"] + 0.5 * df["b"]) > 0).astype(int)
    pool_path = tmp_path / "pool.parquet"
    df.to_parquet(pool_path, compression="snappy", index=False)

    cfg_path = _override_config(tmp_path)
    lineage = pd.DataFrame([
        {"name": "a", "causal_lineage": "clean", "feature_class": "x"},
        {"name": "b", "causal_lineage": "clean", "feature_class": "x"},
        {"name": "c", "causal_lineage": "clean", "feature_class": "x"},
        {"name": "d", "causal_lineage": "clean", "feature_class": "x"},
        {"name": "trade_id", "causal_lineage": "suspect", "feature_class": "meta"},
        {"name": "entry_time", "causal_lineage": "suspect", "feature_class": "meta"},
        {"name": "y", "causal_lineage": "suspect", "feature_class": "meta"},
    ])
    cfg = load_config(
        cfg_path, arc_name="x", cluster_id=0,
        pool_path=pool_path, output_root=tmp_path / "out",
    )
    r = run_pipeline(cfg, lineage_df=lineage)
    assert r.automl_skip_reason == "ok"
    assert r.meta_label_skip_reason.startswith("missing_meta_label_columns")
    assert r.meta_label_result is None
    # Manifest reflects the skip
    payload = json.loads(r.step4_manifest_path.read_text(encoding="utf-8"))
    assert payload["meta_label"]["skip_reason"].startswith("missing_meta_label_columns")


def test_pipeline_skip_meta_label_when_pool_missing_entry_time(tmp_path):
    """Pool without entry_time → AutoML AND meta-labeling both skip
    independently (meta-labeling does NOT cascade from AutoML — they're
    parallel stages with independent preconditions)."""
    rng = np.random.default_rng(3)
    n = 50
    df = pd.DataFrame({
        "a": rng.standard_normal(n), "b": rng.standard_normal(n),
        "trade_id": np.arange(n),
    })
    pool_path = tmp_path / "pool.parquet"
    df.to_parquet(pool_path, compression="snappy", index=False)
    cfg_path = _override_config(tmp_path)
    lineage = pd.DataFrame([
        {"name": "a", "causal_lineage": "clean", "feature_class": "x"},
        {"name": "b", "causal_lineage": "clean", "feature_class": "x"},
        {"name": "trade_id", "causal_lineage": "suspect", "feature_class": "meta"},
    ])
    cfg = load_config(
        cfg_path, arc_name="x", cluster_id=0,
        pool_path=pool_path, output_root=tmp_path / "out",
    )
    r = run_pipeline(cfg, lineage_df=lineage)
    assert r.automl_skip_reason == "missing_entry_time"
    # Meta-labeling skips for the same reason — pool lacks entry_time
    # (which both stages need independently).
    assert r.meta_label_skip_reason == "missing_entry_time"


# ── run_meta_labeling direct unit tests (HALT loud) ─────────────────


def test_run_meta_labeling_halts_loud_on_pool_schema(tmp_path):
    pool = pd.DataFrame([
        {"trade_id": i, "entry_time": pd.Timestamp("2020-01-01", tz="UTC")
         + pd.Timedelta(hours=i),
         "a": float(i), "b": float(-i)}
        for i in range(50)
    ])
    with pytest.raises(PoolSchemaError, match="requires pool columns"):
        run_meta_labeling(
            pool=pool, used_features=("a", "b"),
            train_end=pd.Timestamp("2021-01-01", tz="UTC"),
            arc_name="x", cluster_id=0,
            classifiers_dir=tmp_path / "clf",
            n_folds=TEST_N_FOLDS, max_iter_per_fold=TEST_MAX_ITER,
        )


def test_run_meta_labeling_halts_loud_on_missing_final_r(tmp_path):
    pool = _meta_label_pool(n=120).drop(columns=["final_r"])
    with pytest.raises(PoolSchemaError, match="final_r"):
        run_meta_labeling(
            pool=pool, used_features=("a", "b", "c", "d"),
            train_end=pd.Timestamp("2021-01-01", tz="UTC"),
            arc_name="x", cluster_id=0,
            classifiers_dir=tmp_path / "clf",
            n_folds=TEST_N_FOLDS, max_iter_per_fold=TEST_MAX_ITER,
        )


def test_run_meta_labeling_end_to_end(tmp_path):
    """Direct unit test (no pipeline orchestrator) for the
    run_meta_labeling public API."""
    pool = _meta_label_pool()
    result = run_meta_labeling(
        pool=pool, used_features=("a", "b", "c", "d"),
        train_end=pd.Timestamp("2021-01-01", tz="UTC"),
        arc_name="test", cluster_id=0,
        classifiers_dir=tmp_path / "clf",
        n_folds=TEST_N_FOLDS, max_iter_per_fold=TEST_MAX_ITER,
        permutation_repeats=TEST_PERMUTATION_REPEATS,
    )
    assert isinstance(result, MetaLabelResult)
    assert result.skip_reason == "ok"
    # OOF predictions span all trades (TimeSeriesSplit covers folds 2..N)
    oof = result.automl_result.oof_predictions
    assert not oof.empty
    assert set(oof.columns) == {"trade_id", "fold", "y_true", "y_pred_proba"}
    # Threshold sweep produced default-grid rows
    assert len(result.threshold_sweep) == len(DEFAULT_THRESHOLD_SWEEP)
    # Classifier manifest written
    assert result.classifier_manifest_path.exists()


# ── Single-class fold NaN propagation (PR-B pattern) ─────────────────


def test_meta_label_nan_auc_on_single_class_folds(tmp_path):
    """Build a meta-label pool whose target is mostly all-zero in
    early folds → single-class training → meta-label AUC NaN for those
    folds, aggregate uses nanmean."""
    n = 400
    rng = np.random.default_rng(7)
    end = pd.Timestamp("2020-12-15", tz="UTC")
    df = pd.DataFrame({
        "a": rng.standard_normal(n), "b": rng.standard_normal(n),
        "c": rng.standard_normal(n), "d": rng.standard_normal(n),
        "trade_id": np.arange(n),
        "entry_time": pd.date_range(end=end, periods=n, freq="h"),
    })
    # First 80% of trades never reach 1R; last 20% half do
    bars_to_1r = np.full(n, np.nan)
    cutoff = int(0.8 * n)
    bars_to_1r[cutoff:] = np.where(
        rng.random(n - cutoff) > 0.5,
        rng.integers(low=1, high=10, size=n - cutoff),
        np.nan,
    )
    bars_held = rng.integers(low=3, high=30, size=n)
    df["bars_to_1r_mfe"] = bars_to_1r
    df["bars_held"] = bars_held
    df["exit_reason"] = "time_exit"
    df["final_r"] = np.where(~np.isnan(bars_to_1r), 1.5, -0.5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = run_meta_labeling(
            pool=df, used_features=("a", "b", "c", "d"),
            train_end=pd.Timestamp("2021-01-01", tz="UTC"),
            arc_name="x", cluster_id=0,
            classifiers_dir=tmp_path / "clf",
            n_folds=TEST_N_FOLDS, max_iter_per_fold=TEST_MAX_ITER,
            permutation_repeats=TEST_PERMUTATION_REPEATS,
        )
    ar = result.automl_result
    nan_folds = [fr for fr in ar.fold_results if not np.isfinite(fr.auc_val)]
    # At least one early fold should NaN out
    assert len(nan_folds) >= 1
    # n_folds_valid + n_folds_nan == n_folds_total
    assert ar.n_folds_valid == ar.n_folds_total - len(nan_folds)


# ── persist_fold_classifiers direct test ─────────────────────────────


def test_persist_fold_classifiers_skips_none_fitted(tmp_path):
    """Folds whose fitted_estimator is None still appear in the
    manifest (with path: null) — full audit trail per dispatch §4."""
    # Build a result with a mix of fitted + skipped folds via the public
    # API on a target that produces ≥1 NaN fold.
    pool = _meta_label_pool()
    # Force a single-class early fold by clobbering target generation
    pool.loc[: int(0.8 * len(pool)), "bars_to_1r_mfe"] = np.nan
    pool.loc[: int(0.8 * len(pool)), "exit_reason"] = "sl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = run_meta_labeling(
            pool=pool, used_features=("a", "b", "c", "d"),
            train_end=pd.Timestamp("2021-01-01", tz="UTC"),
            arc_name="x", cluster_id=0,
            classifiers_dir=tmp_path / "clf",
            n_folds=TEST_N_FOLDS, max_iter_per_fold=TEST_MAX_ITER,
            permutation_repeats=TEST_PERMUTATION_REPEATS,
        )
    cm = json.loads(result.classifier_manifest_path.read_text(encoding="utf-8"))
    # All N folds present in manifest
    assert len(cm["folds"]) == TEST_N_FOLDS
    # joblib version captured
    assert cm["joblib_version"] == joblib.__version__


def test_persist_fold_classifiers_handles_empty_result(tmp_path):
    """Directly persisting an empty AutoMLResult still writes a manifest."""
    from core.heavy_ml_probe.automl import AutoMLResult
    empty = AutoMLResult(
        fold_results=(), leaderboard=pd.DataFrame(),
        importance=pd.DataFrame(), used_features=(),
        n_folds_total=0, n_folds_valid=0,
        auc_mean=float("nan"), auc_std=float("nan"),
        total_modelcount=0, total_fit_wall_seconds=0.0,
        flaml_version="", metric="roc_auc",
    )
    path = persist_fold_classifiers(
        empty, tmp_path / "clf", arc_name="x", cluster_id=0,
    )
    assert path.exists()
    cm = json.loads(path.read_text(encoding="utf-8"))
    assert cm["folds"] == []
    assert cm["n_folds_persisted"] == 0


# ── Pool-schema constants ────────────────────────────────────────────


def test_required_pool_columns_constant_locked():
    """Public constant matches the dispatch §1 schema requirement."""
    assert REQUIRED_POOL_COLUMNS == (
        "bars_to_1r_mfe", "bars_held", "exit_reason", "final_r",
    )


def test_meta_label_target_col_constant_locked():
    assert META_LABEL_TARGET_COL == "y_meta_label"
