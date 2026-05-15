"""Clustering for L Arc 3 Step 3 Phase A (mirrors arc 1).

k-means: scikit-learn KMeans with explicit seed + n_init=10.
Hierarchical Ward: scipy linkage on a 10k subsample, then propagate to all rows
via nearest centroid of the subsample's cluster means.
HDBSCAN: sklearn.cluster.HDBSCAN, min_cluster_size = max(50, ceil(0.05 * n_pool)),
min_samples=50, EOM selection.
"""

# ruff: noqa: E402, E701, E702, F841, I001, F401
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score

from . import _common as C


def fit_hdbscan(X: np.ndarray, seed: int = 0) -> np.ndarray:
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
    n = X.shape[0]
    sub_n = min(C.WARD_SUBSAMPLE, n)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=sub_n, replace=False))

    Z = linkage(X[idx], method="ward")
    sub_labels = fcluster(Z, t=k, criterion="maxclust")
    sub_labels = sub_labels - 1  # 0-indexed

    centroids = np.array([X[idx][sub_labels == j].mean(axis=0) for j in range(k)])

    diffs = X[:, None, :] - centroids[None, :, :]
    dists = np.einsum("ijk,ijk->ij", diffs, diffs)
    labels = np.argmin(dists, axis=1)
    return labels


def silhouette_sample(
    X: np.ndarray, labels: np.ndarray, seed: int, sample_size: int = 5000
) -> float:
    n = X.shape[0]
    if sample_size >= n:
        try:
            return float(silhouette_score(X, labels))
        except Exception:
            return float("nan")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=sample_size, replace=False))
    try:
        return float(silhouette_score(X[idx], labels[idx]))
    except Exception:
        return float("nan")


def cluster_size_distribution(labels: np.ndarray) -> dict:
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    return {int(u): {"n": int(c), "frac": float(c / total)} for u, c in zip(unique, counts)}


def ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    return float(adjusted_rand_score(labels_a, labels_b))
