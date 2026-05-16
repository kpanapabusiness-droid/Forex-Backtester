"""KMeans clustering + §6 gate evaluation for KH-24 v2.0 Step 2.

Hyperparameters EXACT per §6:
    KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=300)
    StandardScaler fit on the full pool (single pass; not per-fold)

Silhouette computed on the standardized features (same space as KMeans).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from scripts.arc_kh24_v2.step2._features import FEATURE_COLUMNS

KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300

K_SWEEP: tuple[int, ...] = (3, 4, 5, 6, 7)

GATE_MIN_SILHOUETTE = 0.30
GATE_MAX_CLUSTER_FRACTION = 0.90
GATE_MIN_CLUSTER_SIZE = 30


class ClusterResult(NamedTuple):
    k: int
    labels: np.ndarray  # shape (n_trades,)
    silhouette: float
    centroids_std: np.ndarray  # shape (k, 4) — in standardized space
    centroids_unstd: np.ndarray  # shape (k, 4) — in original feature space
    cluster_sizes: np.ndarray  # shape (k,)
    cluster_fractions: np.ndarray  # shape (k,)


class GateOutcome(NamedTuple):
    k: int
    silhouette: float
    silhouette_ok: bool
    max_cluster_fraction: float
    cluster_fraction_ok: bool
    min_cluster_size: int
    cluster_size_ok: bool
    passes: bool


def run_kmeans_sweep(
    features_df: pd.DataFrame, ks: tuple[int, ...] = K_SWEEP
) -> tuple[dict[int, ClusterResult], StandardScaler]:
    """Fit StandardScaler + KMeans for each K and return per-K results.

    The scaler is fit ONCE on the full pool and shared across all K (per §6).
    """
    X = features_df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(X)
    X_std = scaler.transform(X)

    results: dict[int, ClusterResult] = {}
    n = len(features_df)
    for k in ks:
        km = KMeans(
            n_clusters=k,
            random_state=KMEANS_RANDOM_STATE,
            n_init=KMEANS_N_INIT,
            max_iter=KMEANS_MAX_ITER,
        )
        labels = km.fit_predict(X_std)
        # silhouette requires at least 2 unique labels and n_samples > k.
        sil = float(silhouette_score(X_std, labels)) if k > 1 and n > k else float("nan")
        # Recompute centroids deterministically from the labels (rather than
        # using km.cluster_centers_ directly): sklearn KMeans uses OpenMP
        # internally, and its stored cluster_centers_ can vary by a few ulps
        # across runs when threaded reductions reorder additions. The labels
        # ARE deterministic, so mean(X[labels==cid]) is bit-stable.
        centroids_std = np.stack(
            [X_std[labels == cid].mean(axis=0) for cid in range(k)], axis=0
        ).astype(np.float64)
        centroids_unstd = scaler.inverse_transform(centroids_std)
        sizes = np.bincount(labels, minlength=k).astype(np.int64)
        fractions = sizes.astype(np.float64) / float(n)
        results[k] = ClusterResult(
            k=k,
            labels=labels,
            silhouette=sil,
            centroids_std=centroids_std,
            centroids_unstd=centroids_unstd,
            cluster_sizes=sizes,
            cluster_fractions=fractions,
        )
    return results, scaler


def evaluate_gate(res: ClusterResult) -> GateOutcome:
    sil_ok = res.silhouette >= GATE_MIN_SILHOUETTE
    max_frac = float(res.cluster_fractions.max())
    frac_ok = max_frac <= GATE_MAX_CLUSTER_FRACTION
    # §6 wording: "no cluster > 90%". 90% exactly is the boundary; treat
    # strict > 90% as failure, so max == 0.90 passes. Use comparison directly.
    frac_ok = max_frac <= GATE_MAX_CLUSTER_FRACTION + 1e-12
    min_size = int(res.cluster_sizes.min())
    size_ok = min_size >= GATE_MIN_CLUSTER_SIZE
    return GateOutcome(
        k=res.k,
        silhouette=res.silhouette,
        silhouette_ok=sil_ok,
        max_cluster_fraction=max_frac,
        cluster_fraction_ok=frac_ok,
        min_cluster_size=min_size,
        cluster_size_ok=size_ok,
        passes=bool(sil_ok and frac_ok and size_ok),
    )


def select_k(gates: dict[int, GateOutcome]) -> int | None:
    """Pick the highest-silhouette passing K; tie-break = smaller K (parsimony)."""
    passing = [(k, g.silhouette) for k, g in gates.items() if g.passes]
    if not passing:
        return None
    # Sort: silhouette desc, then K asc.
    passing.sort(key=lambda kv: (-kv[1], kv[0]))
    return passing[0][0]
