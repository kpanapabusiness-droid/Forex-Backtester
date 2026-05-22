"""Step 2 — path-shape clustering per L_PROTOCOL §2 Step 2.

Reads the Step 1 paths table + trades table; computes path-shape
features per trade; runs KMeans K∈{2..6}; reports silhouette per K;
picks the K with highest silhouette; assigns shape-tag labels to each
cluster centroid; emits the cluster_assignments + cluster_summary
artefacts.

Path-shape features per trade (computed from the per-bar paths table):

  - monotonicity:        fraction of in-profit bars where close_r >=
                         previous in-profit close_r
  - local_peaks:         count of bar-to-bar mfe_so_far_r increases
  - mfe_p50_proxy:       median close_r over the trade's path (proxy
                         used at clustering only; Step 3 computes true
                         mfe distribution)
  - time_to_peak_rel:    bar_offset of max mfe_so_far_r divided by
                         bars_held (∈ [0, 1])
  - wrong_way_first:     1.0 if mae_so_far_r reached -0.95R before
                         mfe_so_far_r reached +0.95R, else 0.0

Determinism: ``random_state=42``, ``n_init=10``, ``max_iter=300``,
StandardScaler with no centering or scaling subtleties.
"""

from __future__ import annotations

from dataclasses import dataclass

import hashlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from core.steps._shape_tags import (
    ClusterCentroid,
    TAG_ORDER,
    assign_shape_tag,
)

K_RANGE = (2, 3, 4, 5, 6)
PATH_FEATURE_COLS = (
    "monotonicity",
    "local_peaks",
    "mfe_p50_proxy",
    "time_to_peak_rel",
    "wrong_way_first",
)


@dataclass(frozen=True)
class Step2Result:
    """Output of :func:`run_step_2`."""

    cluster_assignments: pd.DataFrame  # trade_id, cluster_id, k_selected
    cluster_summary: pd.DataFrame  # cluster_id, n_trades, shape_tag, centroid_features
    silhouette_per_k: dict[int, float]
    k_selected: int
    centroids: dict[int, ClusterCentroid]
    summary_md: str


def compute_path_features(
    trades: pd.DataFrame, paths: pd.DataFrame
) -> pd.DataFrame:
    """Compute per-trade path-shape features.

    Returns a DataFrame keyed on trade_id with PATH_FEATURE_COLS.
    Missing or single-bar trades are filled with NaN; the caller drops
    them before clustering.
    """
    if len(paths) == 0 or len(trades) == 0:
        return pd.DataFrame(columns=("trade_id",) + PATH_FEATURE_COLS)

    rows: list[dict] = []
    for tid, group in paths.groupby("trade_id", sort=True):
        group = group.sort_values("bar_offset")
        close_r = group["close_r"].values
        mfe = group["mfe_so_far_r"].values
        mae = group["mae_so_far_r"].values
        n_bars = len(group)
        if n_bars < 2:
            continue
        # monotonicity (in-profit bars)
        in_profit = close_r > 0
        if in_profit.sum() >= 2:
            ip_closes = close_r[in_profit]
            monotone = 1 + np.sum(np.diff(ip_closes) >= 0)
            monotonicity = monotone / max(1, len(ip_closes))
        elif in_profit.sum() == 1:
            monotonicity = 1.0 / 1.0
        else:
            monotonicity = 0.0
        # local peaks
        local_peaks = int(np.sum(np.diff(mfe) > 0))
        # mfe p50 proxy
        mfe_p50_proxy = float(np.median(close_r))
        # ttp_rel
        ttp_rel = int(np.argmax(mfe)) / max(1, n_bars - 1)
        # wrong_way_first
        wrong_way_first = 0.0
        for i in range(n_bars):
            if mae[i] <= -0.95:
                wrong_way_first = 1.0
                break
            if mfe[i] >= 0.95:
                wrong_way_first = 0.0
                break
        rows.append({
            "trade_id": int(tid),
            "monotonicity": float(monotonicity),
            "local_peaks": float(local_peaks),
            "mfe_p50_proxy": mfe_p50_proxy,
            "time_to_peak_rel": float(ttp_rel),
            "wrong_way_first": float(wrong_way_first),
        })
    return pd.DataFrame(rows, columns=("trade_id",) + PATH_FEATURE_COLS)


def _cluster_at_k(features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit KMeans at ``k``. Returns (labels, centroids, silhouette)."""
    if features.shape[0] < k:
        return np.full(features.shape[0], -1, dtype=int), np.empty((0, features.shape[1])), float("nan")
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
        max_iter=300,
        algorithm="lloyd",
    )
    labels = km.fit_predict(features)
    # Silhouette requires >=2 clusters and at least n_samples > n_clusters.
    if features.shape[0] > k:
        sil = float(silhouette_score(features, labels))
    else:
        sil = float("nan")
    return labels, km.cluster_centers_, sil


def run_step_2(
    trades: pd.DataFrame,
    paths: pd.DataFrame,
    *,
    k_range: tuple[int, ...] = K_RANGE,
) -> Step2Result:
    """Run Step 2 end-to-end on Step 1 outputs.

    Single-cluster fallback (per L_PROTOCOL §2 Step 2 Failure
    Diagnostics): if every K gives silhouette < 0.30, the result still
    completes — cluster_assignments has a single cluster_id=0 for all
    trades and k_selected=1.
    """
    features_df = compute_path_features(trades, paths)
    if len(features_df) < max(k_range):
        # Not enough trades to cluster meaningfully; single-cluster fallback
        assignments = pd.DataFrame(
            {
                "trade_id": features_df["trade_id"].astype(int),
                "cluster_id": np.zeros(len(features_df), dtype=int),
                "k_selected": 1,
            }
        )
        summary = pd.DataFrame(
            [{
                "cluster_id": 0,
                "n_trades": len(features_df),
                "shape_tag": "unclassified",
                "monotonicity": float(features_df["monotonicity"].mean()) if len(features_df) else 0.0,
                "local_peaks": float(features_df["local_peaks"].mean()) if len(features_df) else 0.0,
                "mfe_p50_proxy": float(features_df["mfe_p50_proxy"].mean()) if len(features_df) else 0.0,
                "time_to_peak_rel": float(features_df["time_to_peak_rel"].mean()) if len(features_df) else 0.0,
                "wrong_way_first": float(features_df["wrong_way_first"].mean()) if len(features_df) else 0.0,
            }]
        )
        return Step2Result(
            cluster_assignments=assignments,
            cluster_summary=summary,
            silhouette_per_k={},
            k_selected=1,
            centroids={
                0: ClusterCentroid(
                    cluster_id=0,
                    monotonicity=float(summary["monotonicity"].iloc[0]),
                    local_peaks=float(summary["local_peaks"].iloc[0]),
                    mfe_p50=float(summary["mfe_p50_proxy"].iloc[0]),
                    time_to_peak_rel=float(summary["time_to_peak_rel"].iloc[0]),
                    wrong_way_pp=float(summary["wrong_way_first"].iloc[0]),
                )
            },
            summary_md=_render_summary_md({}, 1, summary),
        )

    raw = features_df[list(PATH_FEATURE_COLS)].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw)

    silhouettes: dict[int, float] = {}
    labels_per_k: dict[int, np.ndarray] = {}
    centroids_scaled_per_k: dict[int, np.ndarray] = {}
    for k in k_range:
        labels, centroids_k, sil = _cluster_at_k(scaled, k)
        silhouettes[k] = sil
        labels_per_k[k] = labels
        centroids_scaled_per_k[k] = centroids_k

    # Select K = argmax silhouette
    finite_sils = {k: s for k, s in silhouettes.items() if np.isfinite(s)}
    if finite_sils:
        k_selected = max(finite_sils, key=lambda k: finite_sils[k])
    else:
        k_selected = k_range[0]

    labels = labels_per_k[k_selected]
    # Unscale centroids back into feature space
    centroids_unscaled = scaler.inverse_transform(centroids_scaled_per_k[k_selected])

    # Reorder clusters by mean mfe_p50_proxy descending (stable IDs across runs)
    cluster_means = []
    for cid in range(k_selected):
        mask = labels == cid
        if mask.sum() == 0:
            cluster_means.append((cid, -float("inf")))
            continue
        cluster_means.append((cid, float(features_df["mfe_p50_proxy"].values[mask].mean())))
    cluster_means.sort(key=lambda kv: -kv[1])
    remap = {old: new for new, (old, _) in enumerate(cluster_means)}
    new_labels = np.array([remap[c] for c in labels])
    new_centroids = np.zeros_like(centroids_unscaled)
    for old, new in remap.items():
        new_centroids[new] = centroids_unscaled[old]

    assignments = pd.DataFrame(
        {
            "trade_id": features_df["trade_id"].astype(int).values,
            "cluster_id": new_labels.astype(int),
            "k_selected": k_selected,
        }
    ).sort_values("trade_id").reset_index(drop=True)

    summary_rows = []
    centroid_map: dict[int, ClusterCentroid] = {}
    for new_cid in range(k_selected):
        mask = new_labels == new_cid
        n = int(mask.sum())
        if n == 0:
            continue
        mono = float(features_df["monotonicity"].values[mask].mean())
        peaks = float(features_df["local_peaks"].values[mask].mean())
        mfe = float(features_df["mfe_p50_proxy"].values[mask].mean())
        ttp = float(features_df["time_to_peak_rel"].values[mask].mean())
        ww = float(features_df["wrong_way_first"].values[mask].mean())
        cc = ClusterCentroid(
            cluster_id=new_cid,
            monotonicity=mono,
            local_peaks=peaks,
            mfe_p50=mfe,
            time_to_peak_rel=ttp,
            wrong_way_pp=ww,
        )
        centroid_map[new_cid] = cc
        summary_rows.append({
            "cluster_id": new_cid,
            "n_trades": n,
            "shape_tag": assign_shape_tag(cc),
            "monotonicity": mono,
            "local_peaks": peaks,
            "mfe_p50_proxy": mfe,
            "time_to_peak_rel": ttp,
            "wrong_way_first": ww,
        })
    summary = pd.DataFrame(summary_rows)
    summary_md = _render_summary_md(silhouettes, k_selected, summary)

    return Step2Result(
        cluster_assignments=assignments,
        cluster_summary=summary,
        silhouette_per_k=silhouettes,
        k_selected=k_selected,
        centroids=centroid_map,
        summary_md=summary_md,
    )


def _render_summary_md(
    silhouettes: dict[int, float], k_selected: int, summary: pd.DataFrame
) -> str:
    lines = ["# Step 2 — Clustering Summary", ""]
    if silhouettes:
        lines.append("## Silhouette per K")
        lines.append("")
        lines.append("| K | silhouette |")
        lines.append("|---:|---:|")
        for k, s in sorted(silhouettes.items()):
            lines.append(f"| {k} | {s:.4f} |" if np.isfinite(s) else f"| {k} | nan |")
        lines.append("")
    lines.append(f"**K selected:** {k_selected}")
    lines.append("")
    lines.append("## Per-cluster summary")
    lines.append("")
    lines.append(
        "| cluster | n | shape_tag | mono | local_peaks | mfe_p50_proxy | ttp_rel | ww_first |"
    )
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {int(r['cluster_id'])} | {int(r['n_trades'])} | {r['shape_tag']} | "
            f"{r['monotonicity']:.3f} | {r['local_peaks']:.2f} | "
            f"{r['mfe_p50_proxy']:.3f} | {r['time_to_peak_rel']:.3f} | "
            f"{r['wrong_way_first']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def step_2_sha256(result: Step2Result) -> str:
    """Deterministic hash of the cluster assignments for determinism tests."""
    payload = result.cluster_assignments.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = (
    "K_RANGE",
    "PATH_FEATURE_COLS",
    "Step2Result",
    "compute_path_features",
    "run_step_2",
    "step_2_sha256",
)
