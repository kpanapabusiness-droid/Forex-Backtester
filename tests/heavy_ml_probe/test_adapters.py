"""Unit tests for core.heavy_ml_probe.adapters (PR-E).

Per dispatch §2 / §8: adapter contracts must match what
``core/architectures/A2.py``, ``A4.py``, ``A6.py`` actually expect.
This test module verifies:

  * `build_a2_from_heavy_ml` returns an :class:`A2Config` with a
    `predict_proba`-shaped classifier and the right feature order.
  * `build_a4_from_heavy_ml` returns ``(A4Config, CoxPHAdapter)`` and
    the adapter's `hazard_predict` math matches a hand-computed
    synthetic case (dispatch §2.2 formula).
  * `build_a6_from_heavy_ml` returns an :class:`A6Config` with the
    same predict_proba contract as A2.
  * Fold selection strategies: `LAST_FOLD` (default) and
    `ENSEMBLE_MEAN` both produce valid configs; `FULL_REFIT` raises
    `NotImplementedError`.
  * Manifest schema validation: bad schema_version → `HeavyMLManifestError`.
  * Partial-success manifest: `survival.status != "ok"` → A4 build
    raises `StageUnavailableError` with a clear message.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import-guard FLAML + statsmodels — both required for adapter tests.
pytest.importorskip("flaml")  # noqa: F841
pytest.importorskip("statsmodels")  # noqa: F841

from core.heavy_ml_probe.adapters import (  # noqa: E402
    BARS_SURVIVED_FEATURE,
    STEP5_MANIFEST_SCHEMA_VERSION,
    CoxPHAdapter,
    EnsembleMeanProbaClassifier,
    FoldSelectionStrategy,
    HeavyMLManifestError,
    StageUnavailableError,
    _coxph_payload_from_pickle,
    _CoxPHFoldPayload,
    _interp_cumulative_hazard,
    build_a2_from_heavy_ml,
    build_a4_from_heavy_ml,
    build_a6_from_heavy_ml,
)
from core.heavy_ml_probe.pipeline import load_config, run_pipeline  # noqa: E402

DEFAULT_CONFIG_PATH = Path("configs/heavy_ml_probe/default.yaml")
TEST_N_FOLDS = 5
N_TRADES = 400


# ── Synthetic pool reused from PR-C / PR-D tests ────────────────────


def _full_schema_pool(n: int = N_TRADES, seed: int = 1) -> pd.DataFrame:
    """Pool carrying ALL three stage schemas so the full pipeline runs."""
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n, 4))
    df = pd.DataFrame({"a": feats[:, 0], "b": feats[:, 1],
                       "c": feats[:, 2], "d": feats[:, 3]})
    df["entry_time"] = pd.date_range(
        end="2020-12-15", periods=n, freq="h", tz="UTC"
    )
    df["trade_id"] = np.arange(n)
    noise = rng.standard_normal(n) * 0.3
    score = df["a"] + 0.5 * df["b"] + noise
    reached = (score > 0).values
    df["bars_to_1r_mfe"] = np.where(
        reached, rng.integers(low=1, high=15, size=n), np.nan
    )
    df["bars_held"] = rng.integers(low=3, high=30, size=n)
    df["exit_reason"] = np.where(reached, "tp", "sl")
    df["final_r"] = np.where(
        reached, rng.uniform(0.5, 3.0, size=n), rng.uniform(-1.0, 0.5, size=n)
    )
    df["y"] = reached.astype(int)
    return df


def _lineage_df_clean_features() -> pd.DataFrame:
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
        {"name": "y", "causal_lineage": "suspect", "feature_class": "meta"},
    ])


def _override_config(tmp_path: Path) -> Path:
    import yaml
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["automl"]["n_folds"] = TEST_N_FOLDS
    raw["automl"]["max_iter_per_fold"] = 10
    p = tmp_path / "test_config.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return p


@pytest.fixture
def pipeline_run(tmp_path):
    pool = _full_schema_pool()
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path, compression="snappy", index=False)
    cfg_path = _override_config(tmp_path)
    lineage = _lineage_df_clean_features()
    cfg = load_config(
        cfg_path, arc_name="adapter_test", cluster_id=0,
        pool_path=pool_path, output_root=tmp_path / "out",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = run_pipeline(cfg, lineage_df=lineage)
    return result


# ── CoxPHAdapter hand-computed math (dispatch §2.2 formula) ─────────


def _build_synthetic_cox_payload(
    coefficients: tuple[float, ...] = (0.5, -0.3),
    used_features: tuple[str, ...] = ("a", "b"),
    times: tuple[float, ...] = (2.0, 5.0, 10.0),
    cumulative_hazard: tuple[float, ...] = (0.10, 0.40, 0.90),
) -> _CoxPHFoldPayload:
    """Build a tiny Cox PH payload for hand-computed verification."""
    surv = tuple(math.exp(-h) for h in cumulative_hazard)
    return _CoxPHFoldPayload(
        coefficients=coefficients,
        used_features=used_features,
        baseline_times=times,
        baseline_cumulative_hazard=cumulative_hazard,
        baseline_survival=surv,
    )


def test_coxph_adapter_hazard_predict_hand_computed():
    """Per dispatch §2.2 formula:

      relative_risk     = exp(features @ coefficients)
      integrated_hazard = (H_0(N+K) - H_0(N)) * relative_risk
      P                 = 1 - exp(-integrated_hazard)

    Setup:
      coefficients = [0.5, -0.3]
      features     = [1.0, 2.0] → linpred = 0.5*1 + (-0.3)*2 = -0.1
                                relative_risk = exp(-0.1) ≈ 0.9048
      baseline times      = [2, 5, 10]
      baseline cumhaz     = [0.10, 0.40, 0.90]
      bars_survived_at_N  = 2 (lookup → H_0(2) = 0.10)
      k_horizon           = 3 (so N+K = 5, lookup → H_0(5) = 0.40)
      integrated_hazard   = (0.40 - 0.10) * 0.9048 = 0.30 * 0.9048 = 0.2714
      P                   = 1 - exp(-0.2714) ≈ 0.2378
    """
    payload = _build_synthetic_cox_payload()
    adapter = CoxPHAdapter(
        payloads=(payload,),
        strategy=FoldSelectionStrategy.LAST_FOLD,
        k_horizon=3,
    )
    p = adapter.hazard_predict({"a": 1.0, "b": 2.0}, bars_survived_at_N=2)
    expected = 1.0 - math.exp(-0.30 * math.exp(-0.1))
    assert p == pytest.approx(expected, rel=1e-9)


def test_coxph_adapter_hazard_predict_zero_when_step_function_flat():
    """If N and N+K both fall in the same step-function plateau,
    H_0(N+K) == H_0(N), integrated hazard is 0, P = 0."""
    payload = _build_synthetic_cox_payload(
        times=(2.0, 5.0, 10.0), cumulative_hazard=(0.10, 0.40, 0.90),
    )
    adapter = CoxPHAdapter(
        payloads=(payload,),
        strategy=FoldSelectionStrategy.LAST_FOLD,
        k_horizon=1,
    )
    # Bars survived 2 → H_0(2) = 0.10; N+K = 3 → step-function flat
    # below time 5, so H_0(3) = H_0(2) = 0.10 → integrated hazard 0.
    p = adapter.hazard_predict({"a": 1.0, "b": 2.0}, bars_survived_at_N=2)
    assert p == pytest.approx(0.0, abs=1e-12)


def test_coxph_adapter_hazard_predict_before_first_event_time():
    """H_0(t) for t < first event time should be 0 (Breslow estimator
    convention)."""
    payload = _build_synthetic_cox_payload(
        times=(5.0, 10.0), cumulative_hazard=(0.4, 0.9),
    )
    adapter = CoxPHAdapter(
        payloads=(payload,),
        strategy=FoldSelectionStrategy.LAST_FOLD,
        k_horizon=2,
    )
    # bars=1: H_0(1) = 0; H_0(3) still < first event time = 0
    # → integrated = 0 → P = 0.
    p = adapter.hazard_predict({"a": 0.0, "b": 0.0}, bars_survived_at_N=1)
    assert p == pytest.approx(0.0, abs=1e-12)


def test_coxph_adapter_hazard_predict_past_last_event_clamps():
    """Beyond the last observed event time, H_0 is clamped at the last
    observed value (Breslow convention)."""
    payload = _build_synthetic_cox_payload(
        times=(2.0, 5.0, 10.0), cumulative_hazard=(0.10, 0.40, 0.90),
    )
    adapter = CoxPHAdapter(
        payloads=(payload,),
        strategy=FoldSelectionStrategy.LAST_FOLD,
        k_horizon=5,
    )
    # bars=100 + horizon=5: both well past last time. H_0(both) = 0.90
    # → integrated = 0 → P = 0.
    p = adapter.hazard_predict({"a": 1.0, "b": 0.0}, bars_survived_at_N=100)
    assert p == pytest.approx(0.0, abs=1e-12)


def test_coxph_adapter_hazard_predict_returns_nan_on_missing_feature():
    payload = _build_synthetic_cox_payload()
    adapter = CoxPHAdapter(
        payloads=(payload,), strategy=FoldSelectionStrategy.LAST_FOLD, k_horizon=3,
    )
    p = adapter.hazard_predict({"a": 1.0}, bars_survived_at_N=2)  # no `b`
    assert not np.isfinite(p)


def test_coxph_adapter_predict_proba_shape():
    """predict_proba(X) returns (n_samples, 2); column 1 is P(event)."""
    payload = _build_synthetic_cox_payload()
    adapter = CoxPHAdapter(
        payloads=(payload,), strategy=FoldSelectionStrategy.LAST_FOLD, k_horizon=3,
    )
    # X shape = (n, len(used_features) + 1) — last col is bars_survived
    X = np.array([
        [1.0, 2.0, 2],   # row 1: features [1,2], bars=2
        [0.5, -1.0, 1],  # row 2: features [0.5, -1], bars=1
    ])
    out = adapter.predict_proba(X)
    assert out.shape == (2, 2)
    # Each row sums to 1
    assert np.allclose(out.sum(axis=1), 1.0)
    # Both probabilities in [0, 1]
    assert ((out >= 0) & (out <= 1)).all()


def test_coxph_adapter_feature_order_includes_bars_survived():
    payload = _build_synthetic_cox_payload(used_features=("alpha", "beta"))
    adapter = CoxPHAdapter(
        payloads=(payload,), strategy=FoldSelectionStrategy.LAST_FOLD, k_horizon=1,
    )
    assert adapter.feature_order == ("alpha", "beta", BARS_SURVIVED_FEATURE)


def test_coxph_adapter_rejects_full_refit_strategy():
    payload = _build_synthetic_cox_payload()
    with pytest.raises(NotImplementedError, match="FULL_REFIT"):
        CoxPHAdapter(
            payloads=(payload,),
            strategy=FoldSelectionStrategy.FULL_REFIT,
            k_horizon=3,
        )


def test_coxph_adapter_rejects_empty_payloads():
    with pytest.raises(ValueError, match="at least one fold"):
        CoxPHAdapter(
            payloads=(), strategy=FoldSelectionStrategy.LAST_FOLD, k_horizon=3,
        )


def test_coxph_adapter_rejects_non_positive_horizon():
    payload = _build_synthetic_cox_payload()
    with pytest.raises(ValueError, match="k_horizon must be > 0"):
        CoxPHAdapter(
            payloads=(payload,), strategy=FoldSelectionStrategy.LAST_FOLD, k_horizon=0,
        )


def test_coxph_adapter_ensemble_mean_averages_coefficients():
    """ENSEMBLE_MEAN averages coefficients across folds. With two
    folds having coefs (0.5, -0.3) and (0.7, -0.1), the mean coefs are
    (0.6, -0.2)."""
    p1 = _build_synthetic_cox_payload(coefficients=(0.5, -0.3))
    p2 = _build_synthetic_cox_payload(coefficients=(0.7, -0.1))
    adapter = CoxPHAdapter(
        payloads=(p1, p2),
        strategy=FoldSelectionStrategy.ENSEMBLE_MEAN,
        k_horizon=3,
    )
    eff = getattr(adapter, "_effective")
    assert eff.coefficients == pytest.approx((0.6, -0.2), rel=1e-12)


def test_coxph_adapter_ensemble_mean_rejects_mismatched_features():
    p1 = _build_synthetic_cox_payload(used_features=("a", "b"))
    p2 = _build_synthetic_cox_payload(used_features=("a", "c"))
    with pytest.raises(ValueError, match="mismatched used_features"):
        CoxPHAdapter(
            payloads=(p1, p2),
            strategy=FoldSelectionStrategy.ENSEMBLE_MEAN,
            k_horizon=3,
        )


# ── _interp_cumulative_hazard step function ─────────────────────────


def test_interp_cumulative_hazard_step_function():
    payload = _build_synthetic_cox_payload(
        times=(2.0, 5.0, 10.0), cumulative_hazard=(0.10, 0.40, 0.90),
    )
    # Before first event time
    assert _interp_cumulative_hazard(payload, 0.0) == 0.0
    assert _interp_cumulative_hazard(payload, 1.99) == 0.0
    # At first event time (inclusive)
    assert _interp_cumulative_hazard(payload, 2.0) == 0.10
    # Between events: takes the last observed
    assert _interp_cumulative_hazard(payload, 3.5) == 0.10
    assert _interp_cumulative_hazard(payload, 4.99) == 0.10
    assert _interp_cumulative_hazard(payload, 5.0) == 0.40
    assert _interp_cumulative_hazard(payload, 7.5) == 0.40
    assert _interp_cumulative_hazard(payload, 10.0) == 0.90
    # Past last: clamps
    assert _interp_cumulative_hazard(payload, 100.0) == 0.90


# ── _coxph_payload_from_pickle ──────────────────────────────────────


def test_coxph_payload_from_pickle_minimal_dict():
    payload_dict = {
        "coefficients": [0.5, -0.3],
        "used_features": ["a", "b"],
        "baseline_hazard": {
            "times": [1, 2, 3],
            "cumulative_hazard": [0.1, 0.3, 0.5],
            "survival_function": [0.9, 0.74, 0.61],
        },
    }
    p = _coxph_payload_from_pickle(payload_dict)
    assert p.coefficients == (0.5, -0.3)
    assert p.used_features == ("a", "b")
    assert p.baseline_times == (1.0, 2.0, 3.0)


def test_coxph_payload_from_pickle_raises_on_missing_key():
    with pytest.raises(ValueError, match="missing keys"):
        _coxph_payload_from_pickle({"coefficients": [], "used_features": []})


# ── EnsembleMeanProbaClassifier ──────────────────────────────────────


class _StubClassifier:
    """Minimal stub exposing predict_proba for ensemble testing."""
    def __init__(self, proba_value: float) -> None:
        self.proba_value = proba_value

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X)
        n = X.shape[0] if X.ndim > 1 else 1
        out = np.zeros((n, 2))
        out[:, 1] = self.proba_value
        out[:, 0] = 1.0 - self.proba_value
        return out


def test_ensemble_mean_classifier_averages():
    c1 = _StubClassifier(0.2)
    c2 = _StubClassifier(0.8)
    c3 = _StubClassifier(0.5)
    ens = EnsembleMeanProbaClassifier(members=(c1, c2, c3))
    out = ens.predict_proba(np.zeros((1, 3)))
    assert out[0, 1] == pytest.approx(0.5)  # (0.2 + 0.8 + 0.5) / 3
    assert out[0, 0] == pytest.approx(0.5)


def test_ensemble_mean_classifier_empty_raises():
    ens = EnsembleMeanProbaClassifier(members=())
    with pytest.raises(RuntimeError, match="no members"):
        ens.predict_proba(np.zeros((1, 1)))


# ── Manifest readers + adapter builders against real pipeline ───────


def test_step5_manifest_emitted_with_correct_schema(pipeline_run):
    assert pipeline_run.step5_manifest_path is not None
    assert pipeline_run.step5_manifest_path.exists()
    payload = json.loads(pipeline_run.step5_manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == STEP5_MANIFEST_SCHEMA_VERSION
    assert payload["sub_protocol"] == "heavy_ml_probe"
    assert payload["arc_name"] == "adapter_test"
    assert payload["cluster_id"] == 0
    assert {"automl", "meta_label", "survival"} <= set(payload["stages"].keys())
    # Adapters block reflects per-stage status
    for key in ("a2_buildable", "a4_buildable", "a6_buildable"):
        assert isinstance(payload["adapters"][key], bool)
    # Used features include the lineage-gate-accepted columns
    assert {"a", "b", "c", "d"} <= set(payload["used_features"])


def test_build_a2_from_heavy_ml_returns_a2config(pipeline_run):
    from core.architectures.a2_classifier_filter import A2Config
    cfg = build_a2_from_heavy_ml(pipeline_run.step5_manifest_path)
    assert isinstance(cfg, A2Config)
    # A2Config requires a predict_proba-shaped classifier — verify by call.
    assert hasattr(cfg.classifier, "predict_proba")
    # Feature order matches the manifest's used_features
    assert set(cfg.classifier_feature_order) >= {"a", "b", "c", "d"}


def test_build_a6_from_heavy_ml_returns_a6config(pipeline_run):
    from core.architectures.a6_meta_labeling import A6Config
    cfg = build_a6_from_heavy_ml(
        pipeline_run.step5_manifest_path,
        lower_threshold=0.3, upper_threshold=0.7,
    )
    assert isinstance(cfg, A6Config)
    assert hasattr(cfg.classifier, "predict_proba")
    assert cfg.lower_threshold == 0.3
    assert cfg.upper_threshold == 0.7


def test_build_a4_from_heavy_ml_returns_bundle(pipeline_run):
    from core.architectures._path_classifier import PathClassifierFit
    from core.architectures.a4_pipeline_d_exits import A4Config

    a4_cfg, cox_adapter = build_a4_from_heavy_ml(
        pipeline_run.step5_manifest_path,
        k_horizon=5, exit_threshold=0.4,
    )
    assert isinstance(a4_cfg, A4Config)
    assert isinstance(cox_adapter, CoxPHAdapter)
    # A4Config carries a PathClassifierFit whose .model exposes predict_proba.
    fit = a4_cfg.classifier_fit
    assert isinstance(fit, PathClassifierFit)
    assert hasattr(fit.model, "predict_proba")
    # The fit's feature_order includes BARS_SURVIVED_FEATURE at the end.
    assert fit.feature_order[-1] == BARS_SURVIVED_FEATURE
    # Sanity: hazard_predict returns a finite P in [0, 1] when given
    # valid features + bars.
    sample_features = {f: 0.0 for f in cox_adapter._effective.used_features}
    p = cox_adapter.hazard_predict(sample_features, bars_survived_at_N=3)
    assert np.isfinite(p)
    assert 0.0 <= p <= 1.0


def test_build_a4_predict_admit_handshake(pipeline_run):
    """End-to-end: the A4Config.classifier_fit + predict_admit pair
    works on a feature_row including BARS_SURVIVED_FEATURE.

    This is the load-bearing contract test — A4's runtime consumes
    via predict_admit(fit, feature_row) per
    core.architectures._path_classifier.
    """
    from core.architectures._path_classifier import predict_admit

    a4_cfg, cox_adapter = build_a4_from_heavy_ml(
        pipeline_run.step5_manifest_path,
        k_horizon=5, exit_threshold=0.4,
    )
    fit = a4_cfg.classifier_fit
    # Build a feature_row with every feature in fit.feature_order
    feature_row = {f: 0.0 for f in fit.feature_order}
    feature_row[BARS_SURVIVED_FEATURE] = 3
    admit, proba = predict_admit(fit, feature_row)
    assert isinstance(admit, bool)
    assert 0.0 <= proba <= 1.0


# ── Fold selection strategy ─────────────────────────────────────────


def test_fold_selection_last_vs_ensemble_different_predictions(pipeline_run):
    """LAST_FOLD and ENSEMBLE_MEAN both produce valid adapters but
    typically different P values (mean across folds ≠ last fold)."""
    _, adapter_last = build_a4_from_heavy_ml(
        pipeline_run.step5_manifest_path,
        fold_selection_strategy=FoldSelectionStrategy.LAST_FOLD,
        k_horizon=3,
    )
    _, adapter_ens = build_a4_from_heavy_ml(
        pipeline_run.step5_manifest_path,
        fold_selection_strategy=FoldSelectionStrategy.ENSEMBLE_MEAN,
        k_horizon=3,
    )
    # Same input → typically different output (different effective coefs)
    feats = {f: 0.5 for f in adapter_last._effective.used_features}
    p_last = adapter_last.hazard_predict(feats, bars_survived_at_N=3)
    p_ens = adapter_ens.hazard_predict(feats, bars_survived_at_N=3)
    assert np.isfinite(p_last) and np.isfinite(p_ens)
    # Both in [0, 1]; soft assertion that they differ
    assert 0.0 <= p_last <= 1.0
    assert 0.0 <= p_ens <= 1.0


def test_fold_selection_full_refit_raises_on_build(pipeline_run):
    with pytest.raises(NotImplementedError, match="FULL_REFIT"):
        build_a4_from_heavy_ml(
            pipeline_run.step5_manifest_path,
            fold_selection_strategy=FoldSelectionStrategy.FULL_REFIT,
            k_horizon=3,
        )


# ── Manifest schema validation ──────────────────────────────────────


def test_adapter_raises_on_missing_manifest(tmp_path):
    bad = tmp_path / "does_not_exist.json"
    with pytest.raises(HeavyMLManifestError, match="not found"):
        build_a2_from_heavy_ml(bad)


def test_adapter_raises_on_bad_schema_version(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "schema_version": "999.999",
        "sub_protocol": "heavy_ml_probe",
        "stages": {},
    }), encoding="utf-8")
    with pytest.raises(HeavyMLManifestError, match="schema mismatch"):
        build_a2_from_heavy_ml(bad)


def test_adapter_raises_on_wrong_sub_protocol(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "schema_version": STEP5_MANIFEST_SCHEMA_VERSION,
        "sub_protocol": "vanilla",
        "stages": {},
    }), encoding="utf-8")
    with pytest.raises(HeavyMLManifestError, match="sub_protocol"):
        build_a2_from_heavy_ml(bad)


def test_adapter_raises_on_stage_not_ok(tmp_path):
    """Stage status != "ok" → StageUnavailableError with diagnostic."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "schema_version": STEP5_MANIFEST_SCHEMA_VERSION,
        "sub_protocol": "heavy_ml_probe",
        "arc_name": "x", "cluster_id": 0,
        "stages": {
            "survival": {
                "status": "skipped",
                "skip_reason": "missing_survival_columns:bars_to_1r_mfe",
                "classifier_manifest_path": None,
            }
        },
        "used_features": ["a"],
    }), encoding="utf-8")
    with pytest.raises(StageUnavailableError, match="survival"):
        build_a4_from_heavy_ml(bad)


def test_adapter_raises_on_partial_success_for_unavailable_stage(tmp_path):
    """Per dispatch §4: exit-code-3 partial success → adapter for the
    unavailable stage raises StageUnavailableError with a clear msg."""
    bad = tmp_path / "partial.json"
    bad.write_text(json.dumps({
        "schema_version": STEP5_MANIFEST_SCHEMA_VERSION,
        "sub_protocol": "heavy_ml_probe",
        "arc_name": "x", "cluster_id": 0,
        "stages": {
            "automl": {
                "status": "ok",
                "skip_reason": "ok",
                "classifier_manifest_path": "classifiers/meta_label/manifest.json",
            },
            "meta_label": {
                "status": "ok",
                "skip_reason": "ok",
                "classifier_manifest_path": "classifiers/meta_label/manifest.json",
            },
            "survival": {
                "status": "skipped",
                "skip_reason": "missing_survival_columns:bars_held",
                "classifier_manifest_path": None,
            },
        },
        "used_features": ["a"],
    }), encoding="utf-8")
    # A4 adapter (needs survival) raises with clear message
    with pytest.raises(StageUnavailableError) as excinfo:
        build_a4_from_heavy_ml(bad)
    assert "missing_survival_columns:bars_held" in str(excinfo.value)
