"""Tests for core.heavy_ml_probe.survival + PR-D pipeline integration.

Per dispatch §10: covers target construction (event, duration,
censoring), concordance index correctness, end-to-end PHReg fit + model
persistence, two-run determinism, minimum-N warning (n < 200), and
convergence-failure NaN propagation.

Test pool scale matches PR-B/PR-C: n=400, n_folds=5. Cox PH fit cost is
typically lower than FLAML AutoML so wall-clock should stay well under
the dispatch §4 60s ceiling per-test.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# Import-guard statsmodels — heavy dep; if missing the test module is
# skipped. Per dispatch §1: install verified to clean cp314 wheel.
statsmodels = pytest.importorskip("statsmodels")  # noqa: F841

from core.heavy_ml_probe.io import sha256_file  # noqa: E402
from core.heavy_ml_probe.meta_labeling import PoolSchemaError  # noqa: E402
from core.heavy_ml_probe.metrics import concordance  # noqa: E402
from core.heavy_ml_probe.pipeline import (  # noqa: E402
    load_config,
    run_pipeline,
    stable_payload_sha256,
)
from core.heavy_ml_probe.survival import (  # noqa: E402
    MIN_N_WARN,
    SURVIVAL_REQUIRED_POOL_COLUMNS,
    SurvivalResult,
    build_survival_target,
    persist_fold_models,
    run_survival,
)

DEFAULT_CONFIG_PATH = Path("configs/heavy_ml_probe/default.yaml")

TEST_N_FOLDS = 5
N_TRADES = 400


# ── Target construction (dispatch §3 edge cases) ────────────────────


def test_target_event_before_close():
    """Reached +1R before close → event=1, duration=bars_to_1r."""
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": 3.0, "bars_held": 10.0, "exit_reason": "tp",
    }])
    d, e = build_survival_target(pool)
    assert d.tolist() == [3]
    assert e.tolist() == [1]


def test_target_never_reached_censored_at_time_exit():
    """Never reached +1R, time-exited → event=0, duration=bars_held."""
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": np.nan, "bars_held": 30.0, "exit_reason": "time_exit",
    }])
    d, e = build_survival_target(pool)
    assert d.tolist() == [30]
    assert e.tolist() == [0]


def test_target_sl_on_entry_bar_censored():
    """SL on entry bar (MFE never reached +1R) → event=0, duration=1
    (clipped from 0 to keep Cox PH happy)."""
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": np.nan, "bars_held": 1.0, "exit_reason": "sl",
    }])
    d, e = build_survival_target(pool)
    assert d.tolist() == [1]
    assert e.tolist() == [0]


def test_target_same_bar_sl_tie_censored():
    """+1R and SL on same bar → SL wins → event=0 (censored at that bar)."""
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": 5.0, "bars_held": 5.0, "exit_reason": "sl",
    }])
    d, e = build_survival_target(pool)
    assert d.tolist() == [5]
    assert e.tolist() == [0]


def test_target_same_bar_hard_sl_censored():
    """HONEST_ENGINE_SWEEP.md FLAG-D2: the simulators emit ``'hard_sl'``.
    A same-bar +1R/SL tie with a ``hard_sl`` exit is SL-first → event=0
    (censored), not a spurious event=1."""
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": 5.0, "bars_held": 5.0, "exit_reason": "hard_sl",
    }])
    d, e = build_survival_target(pool)
    assert d.tolist() == [5]
    assert e.tolist() == [0]


def test_target_same_bar_non_sl_event():
    """+1R and time-exit on same bar (not SL) → event=1 (reached before
    non-adverse close)."""
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": 7.0, "bars_held": 7.0, "exit_reason": "time_exit",
    }])
    d, e = build_survival_target(pool)
    assert d.tolist() == [7]
    assert e.tolist() == [1]


def test_target_zero_bars_clipped_to_one():
    """Cox PH requires duration > 0; a 0-bar trade gets clipped to 1."""
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": np.nan, "bars_held": 0.0, "exit_reason": "sl",
    }])
    d, e = build_survival_target(pool)
    assert d.tolist() == [1]
    assert e.tolist() == [0]


def test_target_vectorised_mixed():
    pool = pd.DataFrame([
        {"bars_to_1r_mfe": 2.0,   "bars_held": 5.0,  "exit_reason": "tp"},
        {"bars_to_1r_mfe": np.nan,"bars_held": 10.0, "exit_reason": "sl"},
        {"bars_to_1r_mfe": 5.0,   "bars_held": 5.0,  "exit_reason": "sl"},
        {"bars_to_1r_mfe": 5.0,   "bars_held": 5.0,  "exit_reason": "time_exit"},
        {"bars_to_1r_mfe": np.nan,"bars_held": 30.0, "exit_reason": "time_exit"},
    ])
    d, e = build_survival_target(pool)
    assert d.tolist() == [2, 10, 5, 5, 30]
    assert e.tolist() == [1, 0, 0, 1, 0]


def test_target_halts_loud_on_missing_column():
    pool = pd.DataFrame([{"bars_to_1r_mfe": 1.0, "bars_held": 2.0}])  # no exit_reason
    with pytest.raises(PoolSchemaError, match="requires pool columns"):
        build_survival_target(pool)


def test_target_halts_loud_on_nan_bars_held():
    pool = pd.DataFrame([{
        "bars_to_1r_mfe": 1.0, "bars_held": np.nan, "exit_reason": "sl",
    }])
    with pytest.raises(ValueError, match="bars_held column has NaN"):
        build_survival_target(pool)


def test_target_sl_case_insensitive():
    """SL tie-break is case-insensitive (matches PR-C convention)."""
    for er in ("sl", "SL", "Sl", "sL"):
        pool = pd.DataFrame([{
            "bars_to_1r_mfe": 5.0, "bars_held": 5.0, "exit_reason": er,
        }])
        _, e = build_survival_target(pool)
        assert e.tolist() == [0], f"exit_reason={er!r} should trigger SL tie-break"


# ── Concordance from-scratch correctness (dispatch §4) ──────────────


def test_concordance_perfectly_ranked():
    """Higher predicted risk → smaller observed time → concordant
    pairs only → C = 1.0."""
    event = np.array([1, 1, 1, 1])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    risk = np.array([4.0, 3.0, 2.0, 1.0])  # inverse rank to time
    assert concordance(event, time, risk) == pytest.approx(1.0)


def test_concordance_perfectly_reversed():
    """Predicted risk inversely related to observed event order →
    discordant pairs only → C = 0.0."""
    event = np.array([1, 1, 1, 1])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    risk = np.array([1.0, 2.0, 3.0, 4.0])  # same rank as time
    assert concordance(event, time, risk) == pytest.approx(0.0)


def test_concordance_random_close_to_half():
    """Random risk vs random time should be ~0.5 over a reasonable
    sample. Soft bound to absorb finite-sample wobble."""
    rng = np.random.default_rng(42)
    n = 200
    event = (rng.random(n) > 0.3).astype(int)
    time = rng.uniform(1, 100, size=n)
    risk = rng.standard_normal(n)
    c = concordance(event, time, risk)
    assert 0.40 <= c <= 0.60, f"random concordance should be near 0.5, got {c:.4f}"


def test_concordance_hand_computed_with_censoring():
    """Hand-computed 3-trade case with one censored.

    trades: (event=1, time=1, risk=10), (event=0, time=5, risk=8), (event=1, time=3, risk=5)
    Comparable pairs (i with event=1 and time_i < time_j):
      i=0, j=1: t=1 < t=5, risk_i=10 > risk_j=8 → concordant
      i=0, j=2: t=1 < t=3, risk_i=10 > risk_j=5 → concordant
      i=2, j=1: t=3 < t=5, risk_i=5  < risk_j=8 → discordant
    concordance = (2 + 0) / 3 = 0.6667
    """
    event = np.array([1, 0, 1])
    time = np.array([1.0, 5.0, 3.0])
    risk = np.array([10.0, 8.0, 5.0])
    assert concordance(event, time, risk) == pytest.approx(2.0 / 3.0)


def test_concordance_with_ties():
    """Tied predicted risks contribute 0.5 each.

    trades: (event=1, time=1, risk=5), (event=1, time=2, risk=5), (event=1, time=3, risk=1)
    Comparable pairs:
      i=0, j=1: t=1 < t=2, risk_i=5 == risk_j=5 → tied
      i=0, j=2: t=1 < t=3, risk_i=5 >  risk_j=1 → concordant
      i=1, j=2: t=2 < t=3, risk_i=5 >  risk_j=1 → concordant
    concordance = (2 + 0.5*1) / 3 = 2.5 / 3
    """
    event = np.array([1, 1, 1])
    time = np.array([1.0, 2.0, 3.0])
    risk = np.array([5.0, 5.0, 1.0])
    assert concordance(event, time, risk) == pytest.approx(2.5 / 3.0)


def test_concordance_no_comparable_pairs_returns_nan():
    """All-censored validation slice → no comparable pairs → NaN."""
    event = np.array([0, 0, 0])
    time = np.array([1.0, 2.0, 3.0])
    risk = np.array([1.0, 2.0, 3.0])
    assert np.isnan(concordance(event, time, risk))


def test_concordance_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        concordance(np.array([1, 0]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))


def test_concordance_non_finite_dropped():
    """Trades with NaN risk/time are dropped before pair enumeration."""
    event = np.array([1, 1, 1])
    time = np.array([1.0, 2.0, np.nan])
    risk = np.array([5.0, 1.0, 99.0])
    # After dropping the NaN-time row: 2 trades, both events, t=1 < t=2,
    # risk=5 > risk=1 → concordant. C = 1/1 = 1.0
    assert concordance(event, time, risk) == pytest.approx(1.0)


# ── End-to-end Cox PH (run_survival direct) ──────────────────────────


def _survival_pool(
    *,
    n: int = N_TRADES,
    seed: int = 1,
    last_entry: pd.Timestamp = pd.Timestamp("2020-12-15", tz="UTC"),
    target_strength: float = 0.7,
) -> pd.DataFrame:
    """Synthetic pool with a learnable Cox PH target.

    The signal: trades with high `a + 0.5*b` reach +1R faster than
    those with low values. Cox PH should recover positive coefficient
    on `a` and a smaller positive coefficient on `b`.
    """
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n, 4))
    df = pd.DataFrame({
        "a": feats[:, 0], "b": feats[:, 1],
        "c": feats[:, 2], "d": feats[:, 3],
    })
    end = pd.Timestamp(last_entry)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    df["entry_time"] = pd.date_range(end=end, periods=n, freq="h")
    df["trade_id"] = np.arange(n)
    noise = rng.standard_normal(n) * (1.0 - target_strength)
    score = df["a"] + 0.5 * df["b"] + noise
    reached = (score > 0).values
    bars_held = rng.integers(low=3, high=30, size=n)
    df["bars_to_1r_mfe"] = np.where(
        reached, rng.integers(low=1, high=15, size=n), np.nan
    )
    df["bars_held"] = bars_held
    df["exit_reason"] = np.where(reached, "tp", "sl")
    return df


def _used_features() -> tuple[str, ...]:
    return ("a", "b", "c", "d")


def test_run_survival_smoke(tmp_path):
    """End-to-end Cox PH run produces valid concordance + non-empty
    coefficient table + persisted manifest."""
    pool = _survival_pool()
    with pytest.warns(UserWarning):  # small-N warning fires on early folds
        result = run_survival(
            pool=pool, used_features=_used_features(),
            train_end=pd.Timestamp("2021-01-01", tz="UTC"),
            arc_name="smoke", cluster_id=0,
            classifiers_dir=tmp_path / "clf",
            n_folds=TEST_N_FOLDS,
        )
    assert isinstance(result, SurvivalResult)
    assert result.n_folds_total == TEST_N_FOLDS
    assert result.n_folds_valid >= 1
    # Learnable target → concordance materially above 0.5
    assert result.concordance_mean > 0.6, (
        f"expected concordance > 0.6, got {result.concordance_mean:.4f}"
    )
    # Coefficient table populated
    assert not result.coefficients_long.empty
    assert {"fold", "feature", "coefficient", "p_value", "std_err",
            "concordance", "n_train", "n_train_event",
            "convergence_warning"} <= set(result.coefficients_long.columns)


def test_run_survival_recovers_positive_a_coefficient(tmp_path):
    """The synthetic generator builds the target as monotone-increasing
    in `a`, so Cox PH should recover a positive coefficient for `a`
    on at least one fold."""
    pool = _survival_pool()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = run_survival(
            pool=pool, used_features=_used_features(),
            train_end=pd.Timestamp("2021-01-01", tz="UTC"),
            arc_name="x", cluster_id=0,
            classifiers_dir=tmp_path / "clf",
            n_folds=TEST_N_FOLDS,
        )
    a_coefs = result.coefficients_long[
        result.coefficients_long["feature"] == "a"
    ]["coefficient"].dropna()
    assert (a_coefs > 0).any(), (
        f"expected at least one positive `a` coefficient; got {a_coefs.tolist()}"
    )


def test_run_survival_persists_baseline_hazard(tmp_path):
    """Per dispatch §8: PR-E needs baseline hazard to recover absolute
    survival probability. The per-fold pickle must include it."""
    pool = _survival_pool()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = run_survival(
            pool=pool, used_features=_used_features(),
            train_end=pd.Timestamp("2021-01-01", tz="UTC"),
            arc_name="x", cluster_id=0,
            classifiers_dir=tmp_path / "clf",
            n_folds=TEST_N_FOLDS,
        )
    cm = json.loads(result.classifier_manifest_path.read_text(encoding="utf-8"))
    persisted = [f for f in cm["folds"] if f["path"] is not None]
    assert persisted, "expected at least one persisted fold"
    for f in persisted:
        on_disk = result.classifier_manifest_path.parent / f["path"]
        payload = joblib.load(on_disk)
        assert set(payload.keys()) >= {
            "results", "baseline_hazard", "used_features", "coefficients",
        }
        bh = payload["baseline_hazard"]
        assert set(bh.keys()) == {
            "stratum", "times", "cumulative_hazard", "survival_function",
        }
        # Non-empty (we had events in training)
        assert len(bh["times"]) > 0
        assert len(bh["cumulative_hazard"]) == len(bh["times"])
        # Cumulative hazard is non-decreasing
        ch = np.asarray(bh["cumulative_hazard"])
        assert np.all(np.diff(ch) >= -1e-9), (
            "cumulative hazard should be non-decreasing"
        )
        # Survival function is non-increasing and within [0, 1]
        s = np.asarray(bh["survival_function"])
        assert np.all((s >= -1e-9) & (s <= 1 + 1e-9))
        assert np.all(np.diff(s) <= 1e-9)


# ── Determinism (Cox PH is deterministic; no seed needed) ───────────


def test_run_survival_two_run_determinism(tmp_path):
    """Same input → byte-identical CSV + identical coefficient pickles.

    Cox PH has no randomness; statsmodels' optimiser is deterministic
    given the same starting point. Two consecutive runs should produce
    coefficients that match to numerical precision.
    """
    pool = _survival_pool()
    feats = _used_features()
    end = pd.Timestamp("2021-01-01", tz="UTC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r1 = run_survival(
            pool=pool.copy(), used_features=feats, train_end=end,
            arc_name="x", cluster_id=0, classifiers_dir=tmp_path / "r1",
            n_folds=TEST_N_FOLDS,
        )
        r2 = run_survival(
            pool=pool.copy(), used_features=feats, train_end=end,
            arc_name="x", cluster_id=0, classifiers_dir=tmp_path / "r2",
            n_folds=TEST_N_FOLDS,
        )
    # Concordance + coefficient table match byte-for-byte
    assert r1.concordance_mean == r2.concordance_mean
    assert r1.concordance_std == r2.concordance_std
    pd.testing.assert_frame_equal(
        r1.coefficients_long.reset_index(drop=True),
        r2.coefficients_long.reset_index(drop=True),
    )
    # Manifest stable payload (no timestamps in sidecar) matches
    assert sha256_file(r1.classifier_manifest_path) == sha256_file(r2.classifier_manifest_path)
    # Per-fold pickles match — Cox PH coefficients reproducible
    cm1 = json.loads(r1.classifier_manifest_path.read_text(encoding="utf-8"))
    cm2 = json.loads(r2.classifier_manifest_path.read_text(encoding="utf-8"))
    persisted1 = [f for f in cm1["folds"] if f["path"] is not None]
    persisted2 = [f for f in cm2["folds"] if f["path"] is not None]
    assert len(persisted1) == len(persisted2)
    for f1, f2 in zip(persisted1, persisted2):
        p1 = joblib.load(r1.classifier_manifest_path.parent / f1["path"])
        p2 = joblib.load(r2.classifier_manifest_path.parent / f2["path"])
        # Pickle params should be numerically identical (Cox PH is
        # deterministic; statsmodels uses analytic optimisation).
        np.testing.assert_array_equal(p1["coefficients"], p2["coefficients"])
        np.testing.assert_array_equal(
            p1["baseline_hazard"]["cumulative_hazard"],
            p2["baseline_hazard"]["cumulative_hazard"],
        )


# ── Minimum-N warning (dispatch §6) ─────────────────────────────────


def test_run_survival_min_n_warning_fires_and_completes(tmp_path):
    """n < 200 → UserWarning fires per fold, run completes with
    convergence_warning=True on those folds."""
    # Small pool: 80 trades, 3 folds → each train slice well below 200
    pool = _survival_pool(n=80)
    assert len(pool) < MIN_N_WARN
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = run_survival(
            pool=pool, used_features=_used_features(),
            train_end=pd.Timestamp("2021-01-01", tz="UTC"),
            arc_name="x", cluster_id=0,
            classifiers_dir=tmp_path / "clf",
            n_folds=3,
        )
    min_n_warnings = [
        w for w in caught
        if "MIN_N_WARN" in str(w.message) or "n_train=" in str(w.message)
    ]
    assert min_n_warnings, "expected at least one MIN_N_WARN UserWarning"
    # Run completed; every fold flagged convergence_warning=True
    # because n_train < MIN_N_WARN
    rows = result.coefficients_long
    if not rows.empty:
        assert rows["convergence_warning"].all()


def test_run_survival_no_events_fold_skipped(tmp_path):
    """A fold whose training slice has zero events emits a
    'skipped:no_events_in_training_slice' fit_message and NaN
    concordance (no crash)."""
    # Construct a pool where the first N trades all censored, last N all events
    n = 300
    rng = np.random.default_rng(11)
    feats = rng.standard_normal((n, 3))
    df = pd.DataFrame({"a": feats[:, 0], "b": feats[:, 1], "c": feats[:, 2]})
    df["entry_time"] = pd.date_range(
        end="2020-12-15", periods=n, freq="h", tz="UTC"
    )
    df["trade_id"] = np.arange(n)
    df["bars_held"] = rng.integers(low=3, high=30, size=n)
    # First 90% all censored (no +1R), last 10% all events
    cutoff = int(0.9 * n)
    bars_to_1r = np.full(n, np.nan)
    bars_to_1r[cutoff:] = rng.integers(low=1, high=10, size=n - cutoff)
    df["bars_to_1r_mfe"] = bars_to_1r
    df["exit_reason"] = np.where(np.isnan(bars_to_1r), "time_exit", "tp")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = run_survival(
            pool=df, used_features=("a", "b", "c"),
            train_end=pd.Timestamp("2021-01-01", tz="UTC"),
            arc_name="x", cluster_id=0,
            classifiers_dir=tmp_path / "clf",
            n_folds=5,
        )
    # At least one early fold should have been skipped
    skipped = [
        fr for fr in result.fold_results
        if "no_events_in_training_slice" in fr.fit_message
    ]
    assert len(skipped) >= 1
    for fr in skipped:
        assert not fr.fit_succeeded
        assert not np.isfinite(fr.concordance)


# ── Pipeline integration ────────────────────────────────────────────


def _override_config(tmp_path: Path) -> Path:
    import yaml
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["automl"]["n_folds"] = TEST_N_FOLDS
    raw["automl"]["max_iter_per_fold"] = 10
    p = tmp_path / "test_config.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return p


def _full_pipeline_lineage_df() -> pd.DataFrame:
    """Mark feature cols clean, metadata suspect."""
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
        {"name": "y", "causal_lineage": "suspect", "feature_class": "meta"},
    ])


@pytest.fixture
def pipeline_run(tmp_path):
    """Pool carries survival schema (bars_to_1r_mfe + bars_held +
    exit_reason). No final_r / y → meta-label + AutoML skip cleanly."""
    pool = _survival_pool()
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path, compression="snappy", index=False)
    cfg_path = _override_config(tmp_path)
    lineage = _full_pipeline_lineage_df()

    def _run(label: str):
        out_root = tmp_path / label
        cfg = load_config(
            cfg_path, arc_name="test_arc", cluster_id=0,
            pool_path=pool_path, output_root=out_root,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return run_pipeline(cfg, lineage_df=lineage)

    return _run


def test_pipeline_writes_survival_artefacts(pipeline_run):
    r = pipeline_run("run1")
    assert r.survival_skip_reason == "ok"
    assert r.survival_result is not None
    assert r.survival_results_path is not None and r.survival_results_path.exists()
    assert (
        r.survival_classifier_manifest_path is not None
        and r.survival_classifier_manifest_path.exists()
    )

    # Manifest must include survival artefacts
    payload = json.loads(r.step4_manifest_path.read_text(encoding="utf-8"))
    assert {"survival_model_results", "survival_classifier_manifest"} <= set(
        payload["artefacts"].keys()
    )
    sb = payload["survival"]
    assert sb["skip_reason"] == "ok"
    assert sb["n_folds_total"] == TEST_N_FOLDS
    assert sb["statsmodels_version"]
    assert isinstance(sb["concordance_mean"], float)


def test_pipeline_survival_csv_columns(pipeline_run):
    r = pipeline_run("run1")
    df = pd.read_csv(r.survival_results_path)
    expected = {
        "fold", "feature", "coefficient", "p_value", "std_err",
        "concordance", "n_train", "n_train_event", "convergence_warning",
    }
    assert expected <= set(df.columns)
    # One row per (fold, feature)
    assert len(df) == TEST_N_FOLDS * 4  # 4 features


def test_pipeline_survival_two_run_determinism(pipeline_run):
    r1 = pipeline_run("run1")
    r2 = pipeline_run("run2")
    # Survival CSV + classifier manifest byte-identical
    assert sha256_file(r1.survival_results_path) == sha256_file(r2.survival_results_path)
    assert sha256_file(r1.survival_classifier_manifest_path) == sha256_file(
        r2.survival_classifier_manifest_path
    )
    # Top-level manifest stable payload matches
    assert stable_payload_sha256(r1.step4_manifest_path) == stable_payload_sha256(
        r2.step4_manifest_path
    )


def test_pipeline_skip_survival_when_missing_columns(tmp_path):
    """Pool with no survival columns → AutoML may still run; survival
    skips with the right reason."""
    rng = np.random.default_rng(3)
    n = 100
    df = pd.DataFrame({
        "a": rng.standard_normal(n), "b": rng.standard_normal(n),
        "trade_id": np.arange(n),
        "entry_time": pd.date_range(
            end="2020-12-15", periods=n, freq="h", tz="UTC"
        ),
    })
    df["y"] = (df["a"] > 0).astype(int)
    pool_path = tmp_path / "pool.parquet"
    df.to_parquet(pool_path, compression="snappy", index=False)
    cfg_path = _override_config(tmp_path)
    lineage = pd.DataFrame([
        {"name": "a", "causal_lineage": "clean", "feature_class": "x"},
        {"name": "b", "causal_lineage": "clean", "feature_class": "x"},
        {"name": "trade_id", "causal_lineage": "suspect", "feature_class": "meta"},
        {"name": "entry_time", "causal_lineage": "suspect", "feature_class": "meta"},
        {"name": "y", "causal_lineage": "suspect", "feature_class": "meta"},
    ])
    cfg = load_config(
        cfg_path, arc_name="x", cluster_id=0,
        pool_path=pool_path, output_root=tmp_path / "out",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = run_pipeline(cfg, lineage_df=lineage)
    assert r.survival_skip_reason.startswith("missing_survival_columns")
    assert r.survival_result is None


# ── persist_fold_models direct tests ────────────────────────────────


def test_persist_fold_models_handles_empty(tmp_path):
    """Empty SurvivalResult still writes a manifest."""
    empty = SurvivalResult(
        fold_results=(), coefficients_long=pd.DataFrame(),
        used_features=(),
        n_folds_total=0, n_folds_valid=0,
        concordance_mean=float("nan"), concordance_std=float("nan"),
        total_n_events=0, statsmodels_version="0.14.6",
    )
    path = persist_fold_models(empty, tmp_path / "clf", arc_name="x", cluster_id=0)
    assert path.exists()
    cm = json.loads(path.read_text(encoding="utf-8"))
    assert cm["folds"] == []
    assert cm["n_folds_persisted"] == 0
    assert cm["stage"] == "survival"


# ── Module constants locked ─────────────────────────────────────────


def test_required_columns_constant_locked():
    assert SURVIVAL_REQUIRED_POOL_COLUMNS == (
        "bars_to_1r_mfe", "bars_held", "exit_reason",
    )


def test_min_n_warn_constant_locked():
    assert MIN_N_WARN == 200
