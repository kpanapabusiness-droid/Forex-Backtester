"""Clustering for L Arc 1 Step 3 Phase A.

k-means: scikit-learn KMeans with explicit seed + n_init=10.
Hierarchical Ward: scipy linkage on a 10k subsample, then propagate
to all rows via nearest centroid of the subsample's cluster means.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score

from . import _common as C


def fit_hdbscan(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """v1.1 Amendment 3: HDBSCAN clustering.

    min_cluster_size = ceil(0.05 * n_pool); min_samples = 50.
    Returns label vector where -1 = noise.
    """
    n = X.shape[0]
    min_cs = max(50, int(np.ceil(C.HDBSCAN_MIN_CLUSTER_SIZE_FRAC * n)))
    clf = HDBSCAN(
        min_cluster_size=min_cs,
        min_samples=C.HDBSCAN_MIN_SAMPLES,
        cluster_selection_method="eom",
        allow_single_cluster=False,
    )
    labels = clf.fit_predict(X)
    return labels.astype(int)


def fit_kmeans(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
    return km.fit_predict(X)


def fit_hierarchical_ward(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fit Ward linkage on a subsample, propagate to all rows.

    Decision documented in PHASE_L_ARC_1_STEP3 §schema_notes.
    """
    n = X.shape[0]
    sub_n = min(C.WARD_SUBSAMPLE, n)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=sub_n, replace=False))

    Z = linkage(X[idx], method="ward")
    sub_labels = fcluster(Z, t=k, criterion="maxclust")
    sub_labels = sub_labels - 1  # 0-indexed

    # Compute cluster centroids on the subsample
    centroids = np.array([
        X[idx][sub_labels == j].mean(axis=0)
        for j in range(k)
    ])

    # Assign every row to its nearest centroid (Euclidean, on standardised X)
    # Vectorised distance
    diffs = X[:, None, :] - centroids[None, :, :]
    dists = np.einsum("ijk,ijk->ij", diffs, diffs)
    labels = np.argmin(dists, axis=1)
    return labels


def silhouette_sample(X: np.ndarray, labels: np.ndarray, seed: int,
                      sample_size: int = 5000) -> float:
    n = X.shape[0]
    if sample_size >= n:
        return float(silhouette_score(X, labels))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=sample_size, replace=False))
    return float(silhouette_score(X[idx], labels[idx]))


def cluster_size_distribution(labels: np.ndarray) -> dict:
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    return {int(u): {"n": int(c), "frac": float(c / total)} for u, c in zip(unique, counts)}


def ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    return float(adjusted_rand_score(labels_a, labels_b))
