"""Arc 10 v3.0 — Step 2 clustering on forward-path geometry.

Per L_PROTOCOL v3.0 §2 Step 2. Reads results/l_arc_10/step_1/pool.parquet
and clusters trades on path-shape features. Selects K with highest
silhouette; reports all K diagnostically.

Outputs:
    results/l_arc_10/step_2/cluster_assignments.parquet
    results/l_arc_10/step_2/cluster_summary.md
    results/l_arc_10/step_2/manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from core.determinism import RANDOM_STATE, seed_everything  # noqa: E402
from scripts.l_arc_10_v3._common import (  # noqa: E402
    load_config,
    sha256_file,
    write_manifest,
)

# Path-shape features used for clustering (per §2 Step 2)
CLUSTER_FEATURES = [
    "path_mono",
    "path_peaks",
    "path_ttp_rel",
    "path_drawdown_depth_r",
    "path_recovery_ratio",
]

K_RANGE = (2, 3, 4, 5, 6)


def _archetype_label(centroid: dict) -> str:
    """Assign a shape-tag archetype per quartile rules (per L_PROTOCOL §2 Step 2)."""
    mono = centroid.get("path_mono", np.nan)
    peaks = centroid.get("path_peaks", np.nan)
    ttp = centroid.get("path_ttp_rel", np.nan)
    dd = centroid.get("path_drawdown_depth_r", np.nan)
    rec = centroid.get("path_recovery_ratio", np.nan)
    # wrong_way_pp is available via centroid but not currently used in label rules.

    # Rules — match the qualitative archetypes in L_PROTOCOL §2 Step 2:
    #   V-shape recovery — high recovery ratio, moderate drawdown, low ttp_rel
    #   Stepwise climber — high mono, low peaks
    #   Monotonic up — very high mono, ~0 peaks
    #   Monotonic down — very negative mono
    #   Bimodal — 2+ peaks, moderate mono
    #   Choppy — high peaks, near-zero mono
    if np.isfinite(mono) and mono >= 0.85 and np.isfinite(peaks) and peaks <= 1:
        return "monotonic_up"
    if np.isfinite(mono) and mono <= -0.5:
        return "monotonic_down"
    if (
        np.isfinite(rec) and rec >= 0.6
        and np.isfinite(dd) and dd >= 0.4
        and np.isfinite(ttp) and ttp >= 0.4
    ):
        return "v_shape_recovery"
    if np.isfinite(mono) and mono >= 0.3 and np.isfinite(peaks) and peaks <= 2:
        return "stepwise_climber"
    if np.isfinite(peaks) and peaks >= 3:
        if np.isfinite(mono) and abs(mono) < 0.2:
            return "choppy"
        return "bimodal"
    return "unclassified"


def _cluster_summary_row(grp: pd.DataFrame, k: int, cluster_id: int) -> dict:
    n = int(len(grp))
    if n == 0:
        return dict(K=k, cluster_id=cluster_id, n=0)
    final_r = grp["final_r"].to_numpy()
    mfe_r = grp["mfe_r"].to_numpy()
    mae_r = grp["mae_r"].to_numpy()
    return dict(
        K=k,
        cluster_id=cluster_id,
        n=n,
        share_of_pool=float(n) / max(int(grp.attrs.get("pool_size", n)), 1),
        mean_r=float(np.nanmean(final_r)),
        p25_r=float(np.nanpercentile(final_r, 25)),
        p50_r=float(np.nanpercentile(final_r, 50)),
        p75_r=float(np.nanpercentile(final_r, 75)),
        mfe_p50=float(np.nanpercentile(mfe_r, 50)),
        mfe_p75=float(np.nanpercentile(mfe_r, 75)),
        mae_p50=float(np.nanpercentile(mae_r, 50)),
        bars_held_p50=float(np.nanpercentile(grp["bars_held"].to_numpy(), 50)),
        # Centroid in feature space (pre-scaling)
        **{f"centroid_{f}": float(np.nanmean(grp[f].to_numpy())) for f in CLUSTER_FEATURES},
        wrong_way_pp=float(np.nanmean(grp["path_wrong_way_first"].to_numpy())),
    )


def run(cfg_path: Path, *, write_manifest_flag: bool = True) -> dict:
    seed_everything(RANDOM_STATE)
    cfg = load_config(cfg_path)

    pool_path = REPO_ROOT / cfg["output"]["results_dir"] / cfg["output"]["pool_parquet"]
    pool = pd.read_parquet(pool_path)
    pool_size = int(len(pool))

    # Arc root derived from Step 1 output_dir (parent). Arc 10 v3.0.2 sets
    # results_dir=results/l_arc_10_v3.0.2/step_1, so arc_root=results/l_arc_10_v3.0.2.
    # Arc 10 v3.0 baseline (results_dir=results/l_arc_10/step_1) byte-identically
    # resolves arc_root=results/l_arc_10 — pre-change semantics preserved.
    arc_root = REPO_ROOT / Path(cfg["output"]["results_dir"]).parent
    out_dir = arc_root / "step_2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build feature matrix; drop rows with any NaN
    X_df = pool[CLUSTER_FEATURES].copy()
    valid_mask = X_df.notna().all(axis=1).to_numpy()
    X_valid = X_df.loc[valid_mask].to_numpy()
    if X_valid.shape[0] < 50:
        raise RuntimeError(
            f"Too few valid rows for clustering ({X_valid.shape[0]}); "
            f"Step 1 path features are degenerate."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    per_k_results: dict[int, dict] = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels_valid = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels_valid))
        per_k_results[k] = dict(
            silhouette=sil,
            labels_valid=labels_valid.tolist(),
            inertia=float(km.inertia_),
        )

    # Pick K with max silhouette
    best_k = max(per_k_results, key=lambda k: per_k_results[k]["silhouette"])

    # Re-fit at best K to attach labels (deterministic since same seed)
    km_best = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    labels_best_valid = km_best.fit_predict(X_scaled)
    # Project back to full pool with -1 for invalid rows
    labels_full = np.full(pool_size, -1, dtype=int)
    labels_full[valid_mask] = labels_best_valid

    # Also run all K and store labels in the same projection
    all_labels = {}
    for k in K_RANGE:
        km_k = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labs_k = km_k.fit_predict(X_scaled)
        full_k = np.full(pool_size, -1, dtype=int)
        full_k[valid_mask] = labs_k
        all_labels[k] = full_k

    # Build assignments dataframe
    assignments = pool[["trade_id", "pair", "signal_bar_time"]].copy()
    for k in K_RANGE:
        assignments[f"cluster_K{k}"] = all_labels[k]
    assignments["cluster_primary"] = labels_full
    assignments["primary_K"] = best_k

    # Per-cluster summary at best_k
    cluster_rows = []
    for cid in range(best_k):
        grp = pool.iloc[np.where(labels_full == cid)[0]]
        grp.attrs["pool_size"] = pool_size
        row = _cluster_summary_row(grp, best_k, cid)
        cluster_rows.append(row)

    # Assign archetype labels
    for row in cluster_rows:
        centroid = {f: row.get(f"centroid_{f}") for f in CLUSTER_FEATURES}
        centroid["wrong_way_pp"] = row.get("wrong_way_pp")
        row["archetype"] = _archetype_label(centroid)

    # Attach archetype back to assignments
    arch_map = {row["cluster_id"]: row["archetype"] for row in cluster_rows}
    assignments["archetype_primary"] = assignments["cluster_primary"].map(lambda c: arch_map.get(c, "unclassified"))

    # Write artefacts
    assignments_path = out_dir / "cluster_assignments.parquet"
    assignments.to_parquet(assignments_path, engine="pyarrow", compression="snappy", index=False)

    # Markdown summary
    lines = []
    lines.append("# Arc 10 v3.0 — Step 2 Clustering Summary\n\n")
    lines.append(f"- Pool size: {pool_size}  (valid for clustering: {int(valid_mask.sum())})\n")
    lines.append(f"- Features: {CLUSTER_FEATURES}\n")
    lines.append(f"- K tested: {list(K_RANGE)}; primary K = **{best_k}** (highest silhouette)\n\n")
    lines.append("## Silhouette per K\n\n| K | Silhouette | Inertia |\n|---:|---:|---:|\n")
    for k in K_RANGE:
        marker = " ◀ primary" if k == best_k else ""
        lines.append(f"| {k} | {per_k_results[k]['silhouette']:.4f} | {per_k_results[k]['inertia']:.2f} |{marker}\n")

    lines.append(f"\n## Cluster outcomes (K={best_k})\n\n")
    lines.append(
        "| Cluster | Archetype | n | share | mean_r | p25 | p50 | p75 | mfe_p50 | mfe_p75 | mae_p50 | ww_pp | "
        + " | ".join(f"c_{f}" for f in CLUSTER_FEATURES)
        + " |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        + "|".join(["---:"] * len(CLUSTER_FEATURES))
        + "|\n"
    )
    for row in sorted(cluster_rows, key=lambda r: -r["n"]):
        lines.append(
            f"| c{row['cluster_id']} | {row['archetype']} | {row['n']} | "
            f"{row['share_of_pool']:.3f} | {row['mean_r']:.3f} | {row['p25_r']:.3f} | "
            f"{row['p50_r']:.3f} | {row['p75_r']:.3f} | {row['mfe_p50']:.3f} | "
            f"{row['mfe_p75']:.3f} | {row['mae_p50']:.3f} | {row['wrong_way_pp']:.3f} | "
            + " | ".join(f"{row[f'centroid_{f}']:.3f}" for f in CLUSTER_FEATURES)
            + " |\n"
        )

    summary_md = "".join(lines)
    summary_path = out_dir / "cluster_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8", newline="\n")

    # Manifest
    manifest = dict(
        arc_name="l_arc_10",
        step="step_2",
        protocol_version="v3.0",
        pool_size=pool_size,
        valid_for_clustering=int(valid_mask.sum()),
        K_tested=list(K_RANGE),
        primary_K=best_k,
        silhouette_per_K={k: per_k_results[k]["silhouette"] for k in K_RANGE},
        cluster_summary=cluster_rows,
        sha256=dict(
            cluster_assignments=sha256_file(assignments_path),
            cluster_summary_md=sha256_file(summary_path),
            pool_input=sha256_file(pool_path),
        ),
        env=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__),
        determinism=dict(random_state=RANDOM_STATE, n_jobs=1, line_terminator="\\n"),
        run_timestamp_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )
    try:
        import sklearn  # noqa
        manifest["env"]["sklearn"] = sklearn.__version__
    except Exception:
        manifest["env"]["sklearn"] = "not_installed"

    manifest_path = out_dir / "manifest.json"
    if write_manifest_flag:
        write_manifest(manifest_path, manifest)

    print(f"[step_2] best_K={best_k} silhouette={per_k_results[best_k]['silhouette']:.4f}", flush=True)
    print(f"  archetypes: {[(r['cluster_id'], r['archetype'], r['n']) for r in cluster_rows]}", flush=True)
    return manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Arc 10 v3.0 — Step 2 clustering")
    p.add_argument("-c", "--config", required=True, type=Path)
    args = p.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
