"""v2.0 archetype diagnostic — Step 2: k-means clustering of path-shape features.

Per dataset, K in {3, 4, 5, 6, 7}. Pre-standardise via StandardScaler, then
KMeans(random_state=42, n_init=10, max_iter=300). Centroids returned in
original feature units.

Outputs:
  clusters_K<k>.csv         trade_id, archetype_id
  centroids_K<k>.csv        archetype_id, monotonicity_centroid,
                            local_peaks_centroid, pullback_magnitude_centroid,
                            time_to_peak_relative_centroid
  silhouette_K<k>.txt       float (silhouette score)

Sanity flag: any archetype containing >90% of trades is flagged as
"failed_clustering" for that (dataset, K).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from scripts.v2_0_diagnostic.path_features import FEATURE_COLS

K_VALUES = (3, 4, 5, 6, 7)


def cluster_one(
    features: pd.DataFrame, k: int, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, float, bool]:
    """Run one (dataset, K) clustering.

    Returns:
      assignments — DataFrame[trade_id, archetype_id] sorted by trade_id input order
      centroids   — DataFrame in original feature units
      silhouette  — float (silhouette_score, computed on standardised features)
      failed      — bool, True if any archetype >90% of trades
    """
    X_raw = features[list(FEATURE_COLS)].astype("float64").to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    km = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    labels = km.fit_predict(X)

    # Silhouette score: sub-sample for very large datasets to keep runtime
    # bounded; subsample uses the same random_state for determinism.
    if len(X) > 20000:
        sil = float(silhouette_score(
            X, labels, sample_size=20000, random_state=random_state, metric="euclidean"
        ))
    else:
        sil = float(silhouette_score(X, labels, metric="euclidean"))

    # Centroids back to original units.
    centroid_std = km.cluster_centers_
    centroid_raw = scaler.inverse_transform(centroid_std)
    centroids = pd.DataFrame(centroid_raw, columns=[
        "monotonicity_centroid",
        "local_peaks_centroid",
        "pullback_magnitude_centroid",
        "time_to_peak_relative_centroid",
    ])
    centroids.insert(0, "archetype_id", np.arange(k, dtype=np.int32))

    assignments = pd.DataFrame({
        "trade_id":     features["trade_id"].values,
        "archetype_id": labels.astype(np.int32),
    })

    # Sanity: failed clustering = any archetype >90% of pool.
    counts = pd.Series(labels).value_counts(normalize=True)
    failed = bool((counts > 0.90).any())

    return assignments, centroids, sil, failed


def run_for_dataset(features: pd.DataFrame) -> dict[int, dict]:
    """Run all K values for one dataset, returning a dict keyed by K."""
    out: dict[int, dict] = {}
    for k in K_VALUES:
        a, c, sil, failed = cluster_one(features, k)
        out[k] = {
            "assignments": a,
            "centroids":   c,
            "silhouette":  sil,
            "failed":      failed,
        }
    return out
