"""Clustering integrity, standardization, archetype-labelling — Step 2."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.arc_kh24_v2.step2._archetype import match_archetype
from scripts.arc_kh24_v2.step2._cluster import (
    GATE_MIN_CLUSTER_SIZE,
    GATE_MIN_SILHOUETTE,
    evaluate_gate,
    run_kmeans_sweep,
    select_k,
)
from scripts.arc_kh24_v2.step2._features import FEATURE_COLUMNS


def _synthetic_features(seed: int = 42) -> pd.DataFrame:
    """Three well-separated clusters in 4-D feature space; easy gate pass."""
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [0.20, 2.0, 0.05, 0.20],  # cluster A: low mono, few peaks, early
            [0.60, 15.0, 0.15, 0.80],  # cluster B: stepwise-ish
            [0.15, 25.0, 1.20, 0.50],  # cluster C: random-walk-ish
        ]
    )
    rows = []
    for i, c in enumerate(centers):
        for j in range(80):
            noise = rng.normal(0.0, [0.02, 0.5, 0.02, 0.02])
            row = c + noise
            rows.append(
                {
                    "trade_id": f"T_{i:02d}_{j:03d}",
                    FEATURE_COLUMNS[0]: float(row[0]),
                    FEATURE_COLUMNS[1]: float(row[1]),
                    FEATURE_COLUMNS[2]: float(row[2]),
                    FEATURE_COLUMNS[3]: float(row[3]),
                }
            )
    return pd.DataFrame(rows)


def test_standardization_inverse_round_trip():
    feats = _synthetic_features()
    results, scaler = run_kmeans_sweep(feats, ks=(3,))
    res = results[3]
    # inverse_transform(centroids_std) should equal centroids_unstd exactly
    back = scaler.inverse_transform(res.centroids_std)
    np.testing.assert_allclose(back, res.centroids_unstd, rtol=0, atol=1e-12)


def test_three_well_separated_clusters_pass_gate():
    feats = _synthetic_features()
    results, _ = run_kmeans_sweep(feats, ks=(3, 4, 5))
    g3 = evaluate_gate(results[3])
    assert g3.passes
    assert g3.silhouette >= GATE_MIN_SILHOUETTE
    assert g3.min_cluster_size >= GATE_MIN_CLUSTER_SIZE


def test_k_selection_prefers_higher_silhouette_then_smaller_k():
    feats = _synthetic_features()
    results, _ = run_kmeans_sweep(feats, ks=(3, 4, 5))
    gates = {k: evaluate_gate(r) for k, r in results.items()}
    sel = select_k(gates)
    # Best silhouette on well-separated 3-cluster data should be K=3 itself.
    assert sel == 3


def test_kmeans_determinism_two_runs_identical_labels():
    feats = _synthetic_features()
    r1, _ = run_kmeans_sweep(feats, ks=(3, 5))
    r2, _ = run_kmeans_sweep(feats, ks=(3, 5))
    np.testing.assert_array_equal(r1[3].labels, r2[3].labels)
    np.testing.assert_array_equal(r1[5].labels, r2[5].labels)
    np.testing.assert_allclose(r1[3].centroids_std, r2[3].centroids_std, rtol=0, atol=0)
    np.testing.assert_allclose(r1[5].centroids_std, r2[5].centroids_std, rtol=0, atol=0)


# ----- Archetype labelling determinism + correctness -----


def test_monotone_ascent_match():
    lbl = match_archetype(monotonicity=0.70, local_peaks=3.0, pullback=0.10, time_to_peak_rel=0.80)
    assert lbl.label == "monotone_ascent"
    assert lbl.method == "centroid_match"


def test_stepwise_climber_match():
    lbl = match_archetype(monotonicity=0.60, local_peaks=15.0, pullback=0.30, time_to_peak_rel=0.85)
    assert lbl.label == "stepwise_climber"
    assert lbl.method == "centroid_match"


def test_random_walk_match():
    lbl = match_archetype(monotonicity=0.20, local_peaks=12.0, pullback=1.50, time_to_peak_rel=0.40)
    assert lbl.label == "random_walk"
    assert lbl.method == "centroid_match"


def test_early_peak_family_provisional():
    lbl = match_archetype(monotonicity=0.40, local_peaks=10.0, pullback=0.50, time_to_peak_rel=0.15)
    assert lbl.label == "early_peak_family"
    assert lbl.method == "provisional_step3"


def test_unresolved_no_match():
    # Boundary centroid that matches no rule and not the early_peak prefix.
    lbl = match_archetype(monotonicity=0.45, local_peaks=10.0, pullback=0.40, time_to_peak_rel=0.55)
    assert lbl.label.startswith("unresolved_")
    assert lbl.method == "unresolved"


def test_labelling_is_deterministic_same_input_same_label():
    a = match_archetype(0.501, 8.40, 0.222, 0.459)
    b = match_archetype(0.501, 8.40, 0.222, 0.459)
    assert a == b
    assert a.label == b.label  # same hex suffix → reproducible
