"""Arc 8 Step 2 — path-shape clustering.

Per L_PROTOCOL §2 Step 2 + dispatch §"Step 2":

  - Path features per trade: mono, peaks, ttp_rel, drawdown depth,
    recovery shape (+ supporting scalars derived from the forward path).
  - K in {2, 3, 4, 5, 6}; KMeans default. Report silhouette per K.
  - Pick highest silhouette as primary; keep all K for diagnostics.
  - Assign shape-tag label per trade per quartile rules.

Reads ``step_1/pool.parquet`` + ``step_1/paths.parquet``. Writes
``step_2/cluster_assignments.parquet``, ``cluster_summary.md``,
``cluster_metrics.csv``, ``path_features.parquet``, ``manifest.json``.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from core.determinism import seed_everything, write_text_deterministic
from core.manifest import write_manifest
from scripts.l_arc_8.shared import RESULTS_ROOT

STEP_DIR: Path = RESULTS_ROOT / "step_2"
K_SWEEP: tuple[int, ...] = (2, 3, 4, 5, 6)
RANDOM_STATE: int = 42

PATH_FEATURE_COLS: tuple[str, ...] = (
    "mono",
    "n_local_peaks",
    "ttp_rel",
    "dd_depth",
    "recovery_score",
    "mfe_r",
    "mae_r",
    "final_r",
    "bars_held",
)
# Subset used for clustering (avoid mixing magnitudes that dominate distance).
# Keep dimensionless / shape-only features. mfe_r/mae_r/final_r normalized via StandardScaler.
CLUSTERING_COLS: tuple[str, ...] = (
    "mono",
    "n_local_peaks",
    "ttp_rel",
    "dd_depth",
    "recovery_score",
)


def _path_feats_per_trade(paths_df: pd.DataFrame, trade_pool: pd.DataFrame) -> pd.DataFrame:
    """Compute shape features per trade from the long-format paths frame."""
    feats: list[dict] = []
    pool_indexed = trade_pool.set_index("trade_id")
    # Pre-sort once for groupby determinism.
    paths_sorted = paths_df.sort_values(["trade_id", "bar_idx"])
    for tid, grp in paths_sorted.groupby("trade_id", sort=True):
        close_r = grp["close_r"].to_numpy()
        low_r = grp["low_r"].to_numpy()
        high_r = grp["high_r"].to_numpy()
        n = len(close_r)
        if n == 0:
            continue
        # mono — fraction of bar-to-bar increases (excluding ties)
        diffs = np.diff(close_r)
        mono = float((diffs > 0).sum() / max(1, len(diffs)))
        # local peaks on close_r (strict 3-bar local maxima)
        n_peaks = 0
        for k in range(1, n - 1):
            if close_r[k] > close_r[k - 1] and close_r[k] > close_r[k + 1]:
                n_peaks += 1
        # time-to-peak — bar of MFE (use high_r so peak captures intra-bar high)
        peak_bar = int(np.argmax(high_r))
        ttp_rel = float(peak_bar / max(1, n - 1))
        # drawdown depth from running peak (low_r vs running max of close_r)
        running_peak = np.maximum.accumulate(close_r)
        dd_depth = float(np.min(low_r - running_peak))  # most-negative
        # Pool final_r/mfe_r/mae_r used directly: they account for SL exits +
        # time exits at close_bid in original R units. Path-derived MFE/MAE
        # were used in earlier iterations but the pool values are canonical.
        pool_row = pool_indexed.loc[tid]
        pool_final = float(pool_row["final_r"])
        pool_mfe = float(pool_row["mfe_r"])
        pool_mae = float(pool_row["mae_r"])
        if pool_mfe > 0.05:
            recovery = pool_final / pool_mfe
        else:
            recovery = 0.0
        feats.append({
            "trade_id": int(tid),
            "mono": mono,
            "n_local_peaks": int(n_peaks),
            "ttp_rel": ttp_rel,
            "dd_depth": dd_depth,
            "recovery_score": recovery,
            "mfe_r": pool_mfe,
            "mae_r": pool_mae,
            "final_r": pool_final,
            "bars_held": int(pool_row["bars_held"]),
        })
    return pd.DataFrame(feats).sort_values("trade_id").reset_index(drop=True)


def _shape_tag(row: pd.Series, q: dict) -> str:
    """Assign archetype label per quartile rules.

    Rules:
      - 'V-shape recovery'      if dd_depth <= q.dd_q25 AND recovery_score >= q.rec_q75 AND final_r > 0
      - 'Stepwise climber'      if mono >= q.mono_q75 AND n_local_peaks >= q.peaks_q50 AND final_r > 0
      - 'Monotonic up'          if mono >= q.mono_q75 AND n_local_peaks <= q.peaks_q25 AND final_r > 0
      - 'Monotonic down'        if mono <= q.mono_q25 AND final_r <= 0
      - 'Bimodal'               if n_local_peaks >= q.peaks_q75 AND mfe_r >= q.mfe_q50
      - 'Choppy'                if mono <= q.mono_q25 AND n_local_peaks >= q.peaks_q50
      - 'Mixed'                 otherwise

    These are derived from the path features; intentionally permit multiple
    label candidates (first match wins below). Step 5 architecture-selection
    uses the per-cluster modal tag.
    """
    if row["dd_depth"] <= q["dd_q25"] and row["recovery_score"] >= q["rec_q75"] and row["final_r"] > 0:
        return "V-shape recovery"
    if row["mono"] >= q["mono_q75"] and row["n_local_peaks"] >= q["peaks_q50"] and row["final_r"] > 0:
        return "Stepwise climber"
    if row["mono"] >= q["mono_q75"] and row["n_local_peaks"] <= q["peaks_q25"] and row["final_r"] > 0:
        return "Monotonic up"
    if row["mono"] <= q["mono_q25"] and row["final_r"] <= 0:
        return "Monotonic down"
    if row["n_local_peaks"] >= q["peaks_q75"] and row["mfe_r"] >= q["mfe_q50"]:
        return "Bimodal"
    if row["mono"] <= q["mono_q25"] and row["n_local_peaks"] >= q["peaks_q50"]:
        return "Choppy"
    return "Mixed"


def main() -> Path:
    seed_everything(42)
    t0 = time.perf_counter()
    STEP_DIR.mkdir(parents=True, exist_ok=True)
    step1 = RESULTS_ROOT / "step_1"
    pool = pd.read_parquet(step1 / "pool.parquet")
    paths = pd.read_parquet(step1 / "paths.parquet")
    print(f"[step2] pool: {len(pool)} trades, paths: {len(paths)} rows")

    print("[step2] Computing path features per trade...")
    feats = _path_feats_per_trade(paths, pool)
    feats_path = STEP_DIR / "path_features.parquet"
    feats.to_parquet(feats_path, engine="pyarrow", compression="snappy", index=False)
    print(f"[step2] path_features: {feats.shape} -> {feats_path}")

    # Quartile cache for shape tags
    q = {
        "mono_q25": float(feats["mono"].quantile(0.25)),
        "mono_q75": float(feats["mono"].quantile(0.75)),
        "peaks_q25": float(feats["n_local_peaks"].quantile(0.25)),
        "peaks_q50": float(feats["n_local_peaks"].quantile(0.50)),
        "peaks_q75": float(feats["n_local_peaks"].quantile(0.75)),
        "dd_q25": float(feats["dd_depth"].quantile(0.25)),
        "rec_q75": float(feats["recovery_score"].quantile(0.75)),
        "mfe_q50": float(feats["mfe_r"].quantile(0.50)),
    }
    feats["shape_tag"] = feats.apply(lambda r: _shape_tag(r, q), axis=1)
    print(f"[step2] quartile cache: {q}")
    print(f"[step2] shape_tag value counts:\n{feats['shape_tag'].value_counts()}")

    # Standardise clustering features
    X = feats[list(CLUSTERING_COLS)].to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    print(f"[step2] Sweeping K in {K_SWEEP}...")
    sweep_rows = []
    cluster_assignments_per_k: dict[int, np.ndarray] = {}
    inertia_per_k: dict[int, float] = {}
    silhouette_per_k: dict[int, float] = {}
    for k in K_SWEEP:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
        labels = km.fit_predict(Xs)
        sil = float(silhouette_score(Xs, labels, sample_size=min(len(Xs), 5000), random_state=RANDOM_STATE))
        cluster_assignments_per_k[k] = labels
        inertia_per_k[k] = float(km.inertia_)
        silhouette_per_k[k] = sil
        sweep_rows.append({"k": k, "silhouette": sil, "inertia": float(km.inertia_)})
        print(f"[step2]   K={k}: silhouette={sil:.4f} inertia={km.inertia_:.2f}")

    sweep_df = pd.DataFrame(sweep_rows).sort_values("k").reset_index(drop=True)
    sweep_path = STEP_DIR / "silhouette_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False, lineterminator="\n")

    # Pick primary K = argmax silhouette
    primary_k = int(sweep_df.loc[sweep_df["silhouette"].idxmax(), "k"])
    print(f"[step2] Primary K = {primary_k} (max silhouette)")
    primary_labels = cluster_assignments_per_k[primary_k]

    # Per-cluster outcome summary (use primary K)
    assignments = feats.copy()
    for k, labs in cluster_assignments_per_k.items():
        assignments[f"cluster_K{k}"] = labs
    assignments["cluster_primary"] = primary_labels
    assignments_path = STEP_DIR / "cluster_assignments.parquet"
    assignments.to_parquet(assignments_path, engine="pyarrow", compression="snappy", index=False)

    # Cluster metrics
    metric_rows = []
    for cid in sorted(np.unique(primary_labels)):
        mask = primary_labels == cid
        size = int(mask.sum())
        share = size / len(primary_labels)
        cluster_feats = feats.loc[mask]
        modal_tag = cluster_feats["shape_tag"].value_counts().index[0]
        modal_share = cluster_feats["shape_tag"].value_counts().iloc[0] / size
        metric_rows.append({
            "cluster_id": int(cid),
            "size": size,
            "share": share,
            "modal_shape_tag": modal_tag,
            "modal_tag_share": float(modal_share),
            "mono_mean": float(cluster_feats["mono"].mean()),
            "n_peaks_mean": float(cluster_feats["n_local_peaks"].mean()),
            "ttp_rel_mean": float(cluster_feats["ttp_rel"].mean()),
            "dd_depth_mean": float(cluster_feats["dd_depth"].mean()),
            "recovery_score_mean": float(cluster_feats["recovery_score"].mean()),
            "mean_r": float(cluster_feats["final_r"].mean()),
            "p25_r": float(cluster_feats["final_r"].quantile(0.25)),
            "p50_r": float(cluster_feats["final_r"].quantile(0.50)),
            "p75_r": float(cluster_feats["final_r"].quantile(0.75)),
            "mfe_p50": float(cluster_feats["mfe_r"].quantile(0.50)),
            "mae_p50": float(cluster_feats["mae_r"].quantile(0.50)),
            "bars_held_mean": float(cluster_feats["bars_held"].mean()),
        })
    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = STEP_DIR / "cluster_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, lineterminator="\n")

    # Summary markdown
    summary = _build_summary_md(
        feats=feats, primary_k=primary_k, sweep_df=sweep_df,
        metrics_df=metrics_df, quartiles=q, silhouette_per_k=silhouette_per_k,
    )
    summary_path = STEP_DIR / "cluster_summary.md"
    write_text_deterministic(summary_path, summary)

    write_manifest(
        STEP_DIR / "manifest.json",
        artefacts=[feats_path, sweep_path, assignments_path, metrics_path, summary_path],
    )

    elapsed = time.perf_counter() - t0
    print(f"[step2] DONE in {elapsed:.1f}s — primary K={primary_k}, silhouette={silhouette_per_k[primary_k]:.4f}")
    return STEP_DIR


def _build_summary_md(
    feats: pd.DataFrame,
    primary_k: int,
    sweep_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    quartiles: dict,
    silhouette_per_k: dict,
) -> str:
    lines = [
        "# Arc 8 — Step 2 Cluster Summary",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}Z_",
        "",
        f"- **Trades clustered:** {len(feats):,}",
        f"- **K sweep:** {list(sweep_df['k'])}",
        f"- **Primary K (max silhouette):** {primary_k}",
        f"- **Primary silhouette:** {silhouette_per_k[primary_k]:.4f}",
        "",
        "## Silhouette by K",
        "",
        "| K | Silhouette | Inertia |",
        "|---:|---:|---:|",
    ]
    for _, row in sweep_df.iterrows():
        lines.append(f"| {int(row['k'])} | {row['silhouette']:.4f} | {row['inertia']:.2f} |")

    lines += [
        "",
        "## Per-cluster characterisation (primary K)",
        "",
        "| Cluster | Size | Share | Modal tag (share) | Mono | Peaks | TTP_rel | DD_depth | Recovery | Mean R | p50 R | MFE p50 | MAE p50 | Bars held |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in metrics_df.iterrows():
        lines.append(
            f"| {int(r['cluster_id'])} "
            f"| {int(r['size']):,} "
            f"| {r['share']:.2%} "
            f"| {r['modal_shape_tag']} ({r['modal_tag_share']:.2%}) "
            f"| {r['mono_mean']:.3f} "
            f"| {r['n_peaks_mean']:.2f} "
            f"| {r['ttp_rel_mean']:.3f} "
            f"| {r['dd_depth_mean']:.3f} "
            f"| {r['recovery_score_mean']:.3f} "
            f"| {r['mean_r']:+.3f} "
            f"| {r['p50_r']:+.3f} "
            f"| {r['mfe_p50']:.3f} "
            f"| {r['mae_p50']:+.3f} "
            f"| {r['bars_held_mean']:.1f} |"
        )

    lines += [
        "",
        "## Shape-tag value counts (across full pool)",
        "",
        "| Tag | Count | Share |",
        "|---|---:|---:|",
    ]
    vc = feats["shape_tag"].value_counts()
    for tag, n in vc.items():
        lines.append(f"| {tag} | {int(n):,} | {n / len(feats):.2%} |")

    lines += [
        "",
        "## Quartile cache (drives shape-tag rules)",
        "",
        "```",
        json.dumps(quartiles, indent=2),
        "```",
        "",
        "Notes:",
        "- Clustering features (5): mono, n_local_peaks, ttp_rel, dd_depth, recovery_score. Standardised via StandardScaler before KMeans.",
        "- All K values reported above; primary K used for Step 3 capturability sweep.",
        "- If max silhouette below 0.30, see L_PROTOCOL §2 Step 2 failure-diagnostics path (continue with single-cluster assignment, flag in this report).",
        "",
    ]
    if silhouette_per_k[primary_k] < 0.30:
        lines.append("**FLAG:** Primary silhouette below 0.30 floor. Continuing per protocol — Step 3 will work with the K-cluster split but downstream confidence reduced.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
