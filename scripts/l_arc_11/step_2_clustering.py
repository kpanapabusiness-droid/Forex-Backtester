"""Arc 11 — Step 2 path-shape clustering (L_PROTOCOL v3.0 §2 Step 2).

Inputs:
  - results/l_arc_11/step_1/pool.parquet
  - results/l_arc_11/step_1/trades_paths.parquet

Outputs:
  - results/l_arc_11/step_2/path_features.parquet
  - results/l_arc_11/step_2/cluster_assignments.parquet  (per K)
  - results/l_arc_11/step_2/cluster_summary.md
  - results/l_arc_11/step_2/silhouettes.csv
  - results/l_arc_11/step_2/manifest.json

Mechanics per protocol:
  - Path features: mono (monotonicity), peaks (count), ttp_rel,
    drawdown depth, recovery shape.
  - K in {2,3,4,5,6}; KMeans; silhouette per K.
  - Pick best K by silhouette as primary.
  - Assign shape-tag label per quartile rules.
  - Per-cluster outcome distribution.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.determinism import RANDOM_STATE, seed_everything
from scripts.l_arc_11.common import (
    REPO_ROOT,
    load_config,
    results_root,
    sha256_file,
    write_manifest,
)


def _log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[arc_11 step_2 {ts}] {msg}", flush=True)


# ─── Path-feature extraction ──────────────────────────────────────────


def compute_path_features(paths_df: pd.DataFrame, pool_df: pd.DataFrame) -> pd.DataFrame:
    """Compute one row per trade with path-shape features.

    Features:
      - mono: monotonicity of cumulative MFE (fraction of bars where MFE
        does not decrease vs prior bar) - already true by construction
        of mfe_so_far, so we use close_r monotonicity instead
      - close_r_mono: fraction of held bars where close_r is non-decreasing
      - peaks: number of local maxima in close_r series during hold
      - ttp_rel: time-to-peak MFE / bars_held
      - drawdown_depth: maximum drawdown from running peak in close_r (R units)
      - recovery_shape: post-drawdown recovery fraction (max close_r after
        deepest DD bar / max close_r before)
      - final_r: outcome R (carried for diagnostics)
    """
    held = paths_df[paths_df["is_held"] == 1].copy()
    rows = []
    for trade_id, g in held.groupby("trade_id"):
        g = g.sort_values("bar_offset")
        if len(g) < 2:
            continue
        close_r = g["close_r"].to_numpy(dtype=float)
        mfe = g["mfe_so_far_r"].to_numpy(dtype=float)
        mae = g["mae_so_far_r"].to_numpy(dtype=float)
        bars_held = len(g)

        # close_r monotonicity (non-decreasing fraction)
        diffs = np.diff(close_r)
        close_r_mono = float(np.mean(diffs >= -1e-12))

        # Peaks: local maxima in close_r (strict)
        peaks = 0
        for i in range(1, len(close_r) - 1):
            if close_r[i] > close_r[i - 1] and close_r[i] > close_r[i + 1]:
                peaks += 1

        # Time-to-peak MFE (relative)
        ttp_idx = int(np.argmax(mfe))
        ttp_rel = ttp_idx / max(bars_held - 1, 1)

        # Drawdown depth: max drop from running peak in close_r
        running_max = np.maximum.accumulate(close_r)
        dd = running_max - close_r
        dd_depth = float(dd.max())

        # Recovery shape: max close_r after deepest-DD bar / max close_r overall
        # (>=1 means full recovery or higher; <1 means failed recovery)
        dd_idx = int(np.argmax(dd))
        if dd_idx < len(close_r) - 1:
            post_max = float(close_r[dd_idx + 1:].max())
        else:
            post_max = float(close_r[dd_idx])
        overall_max = float(close_r.max())
        recovery_shape = post_max / overall_max if overall_max > 0 else 1.0

        rows.append(
            {
                "trade_id": int(trade_id),
                "bars_held": bars_held,
                "close_r_mono": close_r_mono,
                "peaks": peaks,
                "ttp_rel": ttp_rel,
                "drawdown_depth": dd_depth,
                "recovery_shape": recovery_shape,
                "mfe_r_final": float(mfe[-1]),
                "mae_r_final": float(mae[-1]),
                "close_r_final": float(close_r[-1]),
            }
        )
    out = pd.DataFrame(rows)
    # Merge in pool's final_r + pair + signal_time
    if not pool_df.empty:
        keep = pool_df[["trade_id", "pair", "signal_time", "final_r", "exit_reason"]]
        out = out.merge(keep, on="trade_id", how="left")
    return out


def shape_tag(row: pd.Series) -> str:
    """Quartile-rule shape tag.

    - V-shape recovery: drawdown_depth high (top quartile) AND recovery_shape > 0.7
    - Stepwise climber: peaks >= 3 AND close_r_mono > 0.55 AND final_r > 0
    - Bimodal: peaks == 2 (two distinct local maxima)
    - Monotonic up: close_r_mono > 0.7 AND final_r > 0
    - Monotonic down: close_r_mono < 0.3 AND final_r < 0
    - Choppy: peaks >= 5 AND close_r_mono between 0.4 and 0.6
    - else: mixed
    """
    if row["drawdown_depth"] > 0.8 and row["recovery_shape"] > 0.7:
        return "v_shape_recovery"
    if row["peaks"] >= 3 and row["close_r_mono"] > 0.55 and row["final_r"] > 0:
        return "stepwise_climber"
    if row["peaks"] == 2:
        return "bimodal"
    if row["close_r_mono"] > 0.70 and row["final_r"] > 0:
        return "monotonic_up"
    if row["close_r_mono"] < 0.30 and row["final_r"] < 0:
        return "monotonic_down"
    if row["peaks"] >= 5 and 0.40 <= row["close_r_mono"] <= 0.60:
        return "choppy"
    return "mixed"


# ─── Clustering ──────────────────────────────────────────────────────


CLUSTER_FEATURES = [
    "close_r_mono",
    "peaks",
    "ttp_rel",
    "drawdown_depth",
    "recovery_shape",
]


def run_clustering(path_feats: pd.DataFrame, k_values: list[int]) -> tuple[pd.DataFrame, dict, dict]:
    if path_feats.empty:
        return pd.DataFrame(), {}, {}
    X = path_feats[CLUSTER_FEATURES].copy().to_numpy(dtype=float)
    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    silhouettes: dict[int, float] = {}
    assignments_by_k: dict[int, np.ndarray] = {}

    for k in k_values:
        if k > len(X):
            continue
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(Xs)
        try:
            sil = float(silhouette_score(Xs, labels))
        except Exception:
            sil = float("nan")
        silhouettes[k] = sil
        assignments_by_k[k] = labels
        _log(f"  K={k} silhouette={sil:.4f}")

    if not silhouettes:
        return pd.DataFrame(), {}, {}

    best_k = max(silhouettes, key=lambda k: silhouettes[k] if np.isfinite(silhouettes[k]) else -np.inf)
    out = path_feats.copy()
    for k, labels in assignments_by_k.items():
        out[f"cluster_k{k}"] = labels
    out["cluster_best"] = assignments_by_k[best_k]
    out["best_k"] = best_k
    return out, silhouettes, {"best_k": best_k}


# ─── Per-cluster outcome summary ────────────────────────────────────


def cluster_outcomes(path_feats: pd.DataFrame, best_k: int) -> pd.DataFrame:
    if path_feats.empty:
        return pd.DataFrame()
    g = path_feats.groupby("cluster_best")
    rows = []
    for cid, sub in g:
        rows.append(
            {
                "cluster": int(cid),
                "n": int(len(sub)),
                "mean_R": float(sub["final_r"].mean()),
                "p25_R": float(sub["final_r"].quantile(0.25)),
                "p50_R": float(sub["final_r"].quantile(0.50)),
                "p75_R": float(sub["final_r"].quantile(0.75)),
                "mfe_p50": float(sub["mfe_r_final"].quantile(0.50)),
                "mae_p50": float(sub["mae_r_final"].quantile(0.50)),
                "bars_held_mean": float(sub["bars_held"].mean()),
                "drawdown_depth_mean": float(sub["drawdown_depth"].mean()),
                "recovery_shape_mean": float(sub["recovery_shape"].mean()),
                "peaks_mean": float(sub["peaks"].mean()),
                "close_r_mono_mean": float(sub["close_r_mono"].mean()),
                "ttp_rel_mean": float(sub["ttp_rel"].mean()),
                "dominant_shape_tag": sub["shape_tag"].mode().iloc[0] if "shape_tag" in sub.columns else "",
            }
        )
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


# ─── Step 2 driver ──────────────────────────────────────────────────


def run(cfg: dict) -> dict:
    seed_everything(RANDOM_STATE)

    results_dir = results_root(cfg)
    step1_dir = results_dir / "step_1"
    step2_dir = results_dir / "step_2"
    step2_dir.mkdir(parents=True, exist_ok=True)

    pool_path = step1_dir / "pool.parquet"
    paths_path = step1_dir / "trades_paths.parquet"

    if not pool_path.exists() or not paths_path.exists():
        raise RuntimeError(f"Step 1 outputs missing: {pool_path} or {paths_path}")

    _log("Loading Step 1 pool and paths")
    pool_df = pd.read_parquet(pool_path)
    paths_df = pd.read_parquet(paths_path)
    _log(f"  pool n={len(pool_df)}, paths rows={len(paths_df)}")

    _log("Computing path features per trade")
    path_feats = compute_path_features(paths_df, pool_df)
    _log(f"  path features for {len(path_feats)} trades")

    _log("Tagging shape per protocol quartile rules")
    if not path_feats.empty:
        path_feats["shape_tag"] = path_feats.apply(shape_tag, axis=1)

    k_values = list(cfg["step_2"]["k_values"])
    _log(f"Running KMeans for K in {k_values}")
    clustered, silhouettes, meta = run_clustering(path_feats, k_values)

    best_k = meta.get("best_k", None)
    _log(f"Best K by silhouette: {best_k}")

    # Per-cluster outcomes (best K)
    outcomes_df = cluster_outcomes(clustered, best_k) if best_k is not None else pd.DataFrame()

    # Persist
    pf_path = step2_dir / "path_features.parquet"
    ca_path = step2_dir / "cluster_assignments.parquet"
    sil_path = step2_dir / "silhouettes.csv"
    summary_path = step2_dir / "cluster_summary.md"
    outcomes_path = step2_dir / "cluster_outcomes.csv"

    if not path_feats.empty:
        path_feats.to_parquet(pf_path, engine="pyarrow", compression="snappy", index=False)
    if not clustered.empty:
        clustered.to_parquet(ca_path, engine="pyarrow", compression="snappy", index=False)
    pd.DataFrame(
        [{"k": k, "silhouette": v} for k, v in sorted(silhouettes.items())]
    ).to_csv(sil_path, index=False, lineterminator="\n")
    if not outcomes_df.empty:
        outcomes_df.to_csv(outcomes_path, index=False, lineterminator="\n")

    # Markdown summary
    lines = ["# Arc 11 v3.0 — Step 2 Cluster Summary", "", "Per L_PROTOCOL §2 Step 2.", ""]
    lines.append(f"## Silhouette per K")
    for k in sorted(silhouettes.keys()):
        sil = silhouettes[k]
        marker = "  ← primary" if k == best_k else ""
        lines.append(f"- K={k}: silhouette = {sil:.4f}{marker}")
    lines.append("")
    lines.append(f"Primary K = **{best_k}**")
    if best_k is not None and all(
        s < float(cfg["step_2"]["silhouette_min_diagnostic"])
        for s in silhouettes.values()
    ):
        lines.append("")
        lines.append("> FLAG: all silhouettes < 0.30 — continuing per protocol but flagging path-shape variance as diagnostic concern.")
    lines.append("")

    if not outcomes_df.empty:
        lines.append(f"## Per-cluster outcomes (K={best_k})")
        lines.append("")
        lines.append("| cluster | n | mean_R | p50_R | mfe_p50 | mae_p50 | bars_held | dd_depth | recov | peaks | mono | ttp_rel | dominant_tag |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for _, r in outcomes_df.iterrows():
            lines.append(
                f"| {int(r['cluster'])} | {int(r['n'])} | {r['mean_R']:.3f} | {r['p50_R']:.3f} | "
                f"{r['mfe_p50']:.3f} | {r['mae_p50']:.3f} | {r['bars_held_mean']:.1f} | "
                f"{r['drawdown_depth_mean']:.3f} | {r['recovery_shape_mean']:.3f} | "
                f"{r['peaks_mean']:.2f} | {r['close_r_mono_mean']:.3f} | {r['ttp_rel_mean']:.3f} | "
                f"{r['dominant_shape_tag']} |"
            )
        lines.append("")

    # Shape-tag distribution
    if not path_feats.empty:
        lines.append(f"## Shape-tag distribution (full pool)")
        st = path_feats["shape_tag"].value_counts().sort_index()
        for tag, n in st.items():
            lines.append(f"- {tag}: {int(n)}")
        lines.append("")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    manifest = {
        "step": 2,
        "n_trades_clustered": int(len(path_feats)),
        "k_values_tested": list(silhouettes.keys()),
        "silhouettes": {int(k): float(v) for k, v in silhouettes.items()},
        "best_k": int(best_k) if best_k is not None else None,
        "cluster_outcomes": outcomes_df.to_dict(orient="records") if not outcomes_df.empty else [],
        "sha256": {
            "path_features": sha256_file(pf_path) if pf_path.exists() else "",
            "cluster_assignments": sha256_file(ca_path) if ca_path.exists() else "",
            "silhouettes": sha256_file(sil_path),
            "cluster_outcomes": sha256_file(outcomes_path) if outcomes_path.exists() else "",
        },
        "run_timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "env": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    write_manifest(step2_dir / "manifest.json", manifest)
    _log(f"Step 2 complete: best_k={best_k}, n_clustered={len(path_feats)}")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/wfo_l_arc_11.yaml")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
