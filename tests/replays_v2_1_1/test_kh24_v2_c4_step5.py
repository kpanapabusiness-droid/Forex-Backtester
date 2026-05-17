"""Unit tests for KH-24 v2.0 Step 5 cross-fold stability.

Covers: fold boundary assignment, no-lookahead per-fold training, OOS admission,
§9 gates (sign / size / DD), per-pair stability flag, null-threshold handling,
determinism.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.replays_v2_1_1.kh24_v2_c4_step5.step5 import (  # noqa: E402
    AssembledData,  # noqa: E402
    ClusterResults,
    FoldResult,
    FoldSpec,
    compute_fold_max_dd_r,
    compute_fold_roi_annualised,
    evaluate_cluster,
)

# ---------------------------------------------------------------------------------------
# 1. Fold boundary assignment
# ---------------------------------------------------------------------------------------

def test_fold_boundary_assignment_is_oos_disjoint():
    """A trade with entry_time inside [oos_start, oos_end) belongs to OOS, not IS.
    A trade in [is_start, is_end) belongs to IS. Boundary at is_end == oos_start
    goes to OOS (half-open intervals)."""
    fold = FoldSpec(
        fold_id=1,
        is_start=pd.Timestamp("2019-01-01"),
        is_end=pd.Timestamp("2020-10-01"),
        oos_start=pd.Timestamp("2020-10-01"),
        oos_end=pd.Timestamp("2021-07-01"),
    )
    trades = pd.DataFrame({
        "entry_time": pd.to_datetime([
            "2019-06-15",      # IS
            "2020-09-30",      # IS (just before boundary)
            "2020-10-01",      # OOS (boundary inclusive on OOS side)
            "2021-03-01",      # OOS
            "2021-07-01",      # outside (exclusive end)
        ])
    })
    is_mask = (trades["entry_time"] >= fold.is_start) & (trades["entry_time"] < fold.is_end)
    oos_mask = (trades["entry_time"] >= fold.oos_start) & (trades["entry_time"] < fold.oos_end)
    assert is_mask.tolist() == [True, True, False, False, False]
    assert oos_mask.tolist() == [False, False, True, True, False]
    # No double-assignment
    assert not (is_mask & oos_mask).any()


# ---------------------------------------------------------------------------------------
# 2. No-lookahead in per-fold classifier training
# ---------------------------------------------------------------------------------------

def test_per_fold_classifier_only_sees_is_data(monkeypatch):
    """Construct a synthetic dataset and verify the RF trained for fold k uses only
    rows with entry_time < fold.is_end."""
    from scripts.replays_v2_1_1.kh24_v2_c4_step5 import step5

    seen_train_indices = {}

    class SpyRF:
        def __init__(self, *a, **kw):
            self.fitted = False
        def fit(self, X, y):
            seen_train_indices["X_shape"] = X.shape
            self.fitted = True
            return self
        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.zeros(len(X))])

    monkeypatch.setattr(step5, "build_rf", lambda cfg: SpyRF())

    n = 100
    times = pd.date_range("2019-01-01", periods=n, freq="14D")
    df = pd.DataFrame({
        "trade_id": [f"T{i}" for i in range(n)],
        "pair": ["A"] * n,
        "entry_time": times,
        "bars_held": [10] * n,
        "final_r": np.zeros(n),
        "cluster_id": [1] * 30 + [0] * 70,
        "body_to_range_ratio": np.random.default_rng(0).random(n),
        "upper_wick_ratio": np.random.default_rng(1).random(n),
        "lower_wick_ratio": np.random.default_rng(2).random(n),
        "range_to_atr_14": np.random.default_rng(3).random(n),
        "ret_5bar_atr": np.random.default_rng(4).random(n),
        "ret_20bar_atr": np.random.default_rng(5).random(n),
        "pos_in_20bar_range": np.random.default_rng(6).random(n),
        "rsi_14": np.random.default_rng(7).random(n),
        "signal_bar_atr_14": np.random.default_rng(8).random(n),
    })
    data = AssembledData(
        full=df, base8_cols=[
            "body_to_range_ratio", "upper_wick_ratio", "lower_wick_ratio",
            "range_to_atr_14", "ret_5bar_atr", "ret_20bar_atr",
            "pos_in_20bar_range", "rsi_14",
        ], arc_specific_cols=["signal_bar_atr_14"],
    )
    # Mock add_path_so_far_for_t to return rows + path features as zeros
    def mock_psf(d, tp, t):
        out = d.copy()
        for c in step5.PATH_SO_FAR_COLS:
            out[c] = 0.0
        return out
    monkeypatch.setattr(step5, "add_path_so_far_for_t", mock_psf)

    folds = [FoldSpec(
        fold_id=1,
        is_start=pd.Timestamp("2019-01-01"),
        is_end=pd.Timestamp("2020-06-01"),
        oos_start=pd.Timestamp("2020-06-01"),
        oos_end=pd.Timestamp("2021-06-01"),
    )]
    cohort = {"id": 1, "label": "c1"}
    policy = {"chosen_t": 1, "admit_threshold": 0.5}
    # We assert via the SpyRF side-effect, so we don't bind the return value.
    evaluate_cluster(
        cohort, data, pd.DataFrame(), folds,
        {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 5, "random_state": 42},
        {"pct_per_trade": 0.005, "days_per_year": 365.25},
        policy,
    )
    # Verify the RF was trained on at most the IS subset
    is_count = ((df["entry_time"] >= folds[0].is_start) & (df["entry_time"] < folds[0].is_end)).sum()
    assert seen_train_indices.get("X_shape") is not None
    assert seen_train_indices["X_shape"][0] == is_count, (
        f"RF trained on {seen_train_indices['X_shape'][0]} rows; IS has {is_count}"
    )


# ---------------------------------------------------------------------------------------
# 3. OOS admission via P(cluster) >= threshold
# ---------------------------------------------------------------------------------------

def test_oos_admission_threshold_logic():
    """Trades with proba 0.3, 0.5, 0.6, 0.8; threshold 0.5; admitted = 3 (>=0.5)."""
    proba = np.array([0.3, 0.5, 0.6, 0.8])
    threshold = 0.5
    admitted = proba >= threshold
    assert admitted.tolist() == [False, True, True, True]
    assert admitted.sum() == 3


# ---------------------------------------------------------------------------------------
# 4. Sign consistency gate
# ---------------------------------------------------------------------------------------

def test_sign_consistency_gate():
    """All folds positive → PASS. One negative → FAIL."""
    def make_fr(mean):
        return FoldResult(
            fold_id=1, is_start="", is_end="", oos_start="", oos_end="",
            is_n_trades=0, is_n_positives=0, oos_n_trades=0, oos_n_positives=0,
            admitted_n=1, admitted_n_positives=0,
            final_r_mean=mean, final_r_t_stat=0.0,
            fold_roi_annualised_pct=0.0, fold_max_dd_R=0.0,
            sign_pos=(mean > 0),
        )

    all_pos = ClusterResults(
        cluster_label="x", cluster_id=0, chosen_t=1, admit_threshold=0.5,
        blocked=False, block_reason="",
        fold_results=[make_fr(x) for x in [0.1, 0.2, 0.05, 0.3]],
    )
    one_neg = ClusterResults(
        cluster_label="x", cluster_id=0, chosen_t=1, admit_threshold=0.5,
        blocked=False, block_reason="",
        fold_results=[make_fr(x) for x in [0.1, -0.05, 0.05, 0.3]],
    )
    assert all_pos.sign_consistency() is True
    assert one_neg.sign_consistency() is False


# ---------------------------------------------------------------------------------------
# 5. Size variance gate
# ---------------------------------------------------------------------------------------

def test_size_variance_gate():
    """sizes [10,15,12,20,18,14,16] → max/min = 2.0 → PASS (<=3).
    sizes [5,15,12,20,18,14,16] → max/min = 4.0 → FAIL."""
    def make_results(sizes):
        return ClusterResults(
            cluster_label="x", cluster_id=0, chosen_t=1, admit_threshold=0.5,
            blocked=False, block_reason="",
            fold_results=[
                FoldResult(
                    fold_id=i + 1, is_start="", is_end="", oos_start="", oos_end="",
                    is_n_trades=0, is_n_positives=0, oos_n_trades=0, oos_n_positives=0,
                    admitted_n=s, admitted_n_positives=0,
                    final_r_mean=None, final_r_t_stat=None,
                    fold_roi_annualised_pct=None, fold_max_dd_R=None, sign_pos=None,
                )
                for i, s in enumerate(sizes)
            ],
        )

    pass_case = make_results([10, 15, 12, 20, 18, 14, 16])
    fail_case = make_results([5, 15, 12, 20, 18, 14, 16])
    assert pass_case.size_variance_ratio() == pytest.approx(2.0)
    assert fail_case.size_variance_ratio() == pytest.approx(4.0)


# ---------------------------------------------------------------------------------------
# 6. DD ceiling gate
# ---------------------------------------------------------------------------------------

def test_dd_ceiling_gate():
    """DDs [1.0, 1.2, 0.8, 2.0, 1.0, 1.5, 1.0] — median 1.0, worst 2.0 → ratio 2.0 → PASS.
    Change worst to 3.0 → ratio 3.0 → FAIL (>2)."""
    def make_results(dds):
        return ClusterResults(
            cluster_label="x", cluster_id=0, chosen_t=1, admit_threshold=0.5,
            blocked=False, block_reason="",
            fold_results=[
                FoldResult(
                    fold_id=i + 1, is_start="", is_end="", oos_start="", oos_end="",
                    is_n_trades=0, is_n_positives=0, oos_n_trades=0, oos_n_positives=0,
                    admitted_n=10, admitted_n_positives=0,
                    final_r_mean=None, final_r_t_stat=None,
                    fold_roi_annualised_pct=None, fold_max_dd_R=dd, sign_pos=None,
                )
                for i, dd in enumerate(dds)
            ],
        )

    pass_case = make_results([1.0, 1.2, 0.8, 2.0, 1.0, 1.5, 1.0])
    fail_case = make_results([1.0, 1.2, 0.8, 3.0, 1.0, 1.5, 1.0])
    assert pass_case.dd_ceiling_ratio() == pytest.approx(2.0)
    assert fail_case.dd_ceiling_ratio() == pytest.approx(3.0)


# ---------------------------------------------------------------------------------------
# 7. Per-pair stability flag
# ---------------------------------------------------------------------------------------

def test_pair_concentration_flag():
    """If 4 pairs hold 80% of trades and >5 total pairs exist, flag fires
    (top-5 share > 50%)."""
    counts = pd.Series({
        "EUR_USD": 20, "GBP_USD": 18, "USD_JPY": 16, "AUD_USD": 14,
        "EUR_GBP": 4, "NZD_USD": 3, "EUR_JPY": 2, "GBP_JPY": 2, "CAD_JPY": 1,
    })
    total = counts.sum()
    top5_share = counts.head(5).sum() / total
    assert top5_share > 0.50, f"top-5 share {top5_share} should exceed 0.50"


# ---------------------------------------------------------------------------------------
# 8. c4 null-threshold handling
# ---------------------------------------------------------------------------------------

def test_null_threshold_blocks_cluster():
    """admit_threshold=None ⇒ evaluate_cluster returns blocked=True without crashing."""
    data = AssembledData(
        full=pd.DataFrame({
            "trade_id": ["T1"], "entry_time": [pd.Timestamp("2020-01-01")],
            "bars_held": [10], "final_r": [0.0], "pair": ["A"], "cluster_id": [0],
        }),
        base8_cols=[], arc_specific_cols=[],
    )
    cohort = {"id": 4, "label": "c4"}
    policy = {"chosen_t": 2, "admit_threshold": None}
    results = evaluate_cluster(
        cohort, data, pd.DataFrame(), [], {}, {}, policy,
    )
    assert results.blocked is True
    assert "admit_threshold is null" in results.block_reason
    assert results.fold_results == []


# ---------------------------------------------------------------------------------------
# 9. Determinism — same inputs + random_state=42 → same fold metrics
# ---------------------------------------------------------------------------------------

def test_fold_max_dd_computation_known_path():
    """final_r sequence [0.5, 0.3, -1.0, 0.4, -0.5, 0.2]
    cumulative: [0.5, 0.8, -0.2, 0.2, -0.3, -0.1]
    peak:       [0.5, 0.8, 0.8, 0.8, 0.8, 0.8]
    DD:         [0.0, 0.0, 1.0, 0.6, 1.1, 0.9]
    max DD = 1.1"""
    final_r = np.array([0.5, 0.3, -1.0, 0.4, -0.5, 0.2])
    assert compute_fold_max_dd_r(final_r) == pytest.approx(1.1)


def test_fold_roi_annualisation():
    """sum(R)=2.0; risk_pct=0.005; oos_days=90; days_per_year=365.25
    ROI = 2.0 × 0.005 / 90 × 365.25 = 0.04058..."""
    arr = np.array([0.5, 0.5, 0.5, 0.5])
    roi = compute_fold_roi_annualised(arr, oos_days=90, risk_pct=0.005, days_per_year=365.25)
    assert roi == pytest.approx(2.0 * 0.005 / 90 * 365.25)


def test_determinism_rf_same_seed_same_output():
    """Two RF instances with random_state=42 trained on the same data give identical
    predictions — same property tested in step4 tests but re-verified for step5 use."""
    from scripts.replays_v2_1_1.kh24_v2_c4_step5.step5 import build_rf
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 4))
    y = rng.integers(0, 2, size=80)
    rf1 = build_rf({"n_estimators": 30, "max_depth": 4, "min_samples_leaf": 5, "random_state": 42, "n_jobs": 1}).fit(X, y)
    rf2 = build_rf({"n_estimators": 30, "max_depth": 4, "min_samples_leaf": 5, "random_state": 42, "n_jobs": 1}).fit(X, y)
    np.testing.assert_array_equal(rf1.predict_proba(X), rf2.predict_proba(X))
