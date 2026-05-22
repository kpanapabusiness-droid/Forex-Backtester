"""Arc 2 Per-Fold Cluster Breakdown — open question on regime-vs-trade-structure.

Phase: l6_arc2_per_fold_clusters (additive; descriptive only per L6.0 v1.1 §14.5).

Reads (read-only):
- results/l6/arc2/trajectory_appendix/cluster_assignments.csv
- results/l6/arc2/trajectory_appendix/shape_features.csv
Both must hash to the values recorded in run_manifest.txt of the trajectory
appendix. HALT on mismatch.

Writes:
- results/l6/arc2/trajectory_appendix/per_fold_clusters/

Open question: are k-means clusters concentrated in specific Arc 2 WFO OOS
folds, or are they distributed evenly across folds (i.e., capture genuinely
regime-independent trade-level structure)?

Analyses:
- A1: Cross-tabs (cluster x fold) at k in {4, 6, 8, 10}; counts, pct_of_fold,
  pct_of_cluster. 12 CSV outputs.
- A2: Per-(cluster, fold) outcome stats at k=6 (long format).
- A3: Independence diagnostics — chi-square + Cramer's V for (cluster x fold),
  with (cluster x pair) and (cluster x month_of_year) as comparators at k=6.
- A4: Per-cluster signed contribution to fold-level pooled mean r_at_t240.

Determinism: deterministic sort (signal_idx asc), CSV LF line endings, pinned
matplotlib metadata, double-run byte-identicality check.

§14.5: report is empirical only. Action-shaped patterns append to
../CANDIDATE_HYPOTHESES_DRAFT.md by hand, not by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Headless / deterministic matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SOURCE_DATE_EPOCH", "1577836800")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats as sps  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# LOCKED CONSTANTS
# ---------------------------------------------------------------------------

NUMPY_SEED: int = 20260512

APPENDIX_DIR: Path = REPO_ROOT / "results" / "l6" / "arc2" / "trajectory_appendix"
INPUT_CLUSTERS: Path = APPENDIX_DIR / "cluster_assignments.csv"
INPUT_SHAPE: Path = APPENDIX_DIR / "shape_features.csv"

OUTPUT_DIR: Path = APPENDIX_DIR / "per_fold_clusters"
CHARTS_DIR: Path = OUTPUT_DIR / "charts"

# Expected hashes from the parent appendix manifest.
EXPECTED_INPUT_HASHES: Dict[str, str] = {
    "cluster_assignments.csv": "c82f7208ece085e804bc7c66c741d91e5004c604463f8dbeff03381aacf7ecbe",
    "shape_features.csv": "6920c990628dd2d769cfbc56bd006391f6b9a24beea64f52f5bda9da02d44aaa",
}

# Fold OOS boundaries (LOCKED from PHASE_L6_ARC2_RESULT.md §5 / lines 148-154).
# Left-closed, right-open: signal at boundary belongs to the later fold.
FOLDS: Tuple[Tuple[int, str, str], ...] = (
    (1, "2020-10-01", "2021-07-01"),
    (2, "2021-07-01", "2022-04-01"),
    (3, "2022-04-01", "2023-01-01"),
    (4, "2023-01-01", "2023-10-01"),
    (5, "2023-10-01", "2024-07-01"),
    (6, "2024-07-01", "2025-04-01"),
    (7, "2025-04-01", "2026-01-01"),
)

# WFO per-fold reference numbers (LOCKED from PHASE_L6_ARC2_RESULT.md §5)
# (fold_id, n_taken, mean_R, sl_hit_rate, roi_pct, max_dd_pct)
WFO_REFERENCE: Tuple[Tuple[int, int, float, float, float, float], ...] = (
    (1, 541, -0.176, 0.786, -71.2, 78.8),
    (2, 621, -0.055, 0.784, -73.4, 79.9),
    (3, 607, -0.088, 0.771, -55.5, 64.1),
    (4, 614, -0.183, 0.788, -76.5, 83.2),
    (5, 504, 0.303, 0.734, 217.7, 44.9),
    (6, 599, 0.034, 0.738, -2.3, 72.4),
    (7, 507, 0.090, 0.710, 33.4, 40.5),
)

K_VALUES: Tuple[int, ...] = (4, 6, 8, 10)
K_PRIMARY: int = 6


# ---------------------------------------------------------------------------
# Filesystem helpers (deterministic writes).
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png_pixel_hash(path: Path) -> str:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return hashlib.sha256(data).hexdigest()
    out = hashlib.sha256()
    pos = 8
    while pos < len(data):
        if pos + 8 > len(data):
            break
        length = int.from_bytes(data[pos : pos + 4], "big")
        chunk_type = data[pos + 4 : pos + 8]
        chunk_data = data[pos + 8 : pos + 8 + length]
        if chunk_type == b"IDAT":
            out.update(chunk_data)
        pos = pos + 8 + length + 4
    return out.hexdigest()


def _write_csv(
    df: pd.DataFrame, path: Path, *, float_format: str = "%.10g", index: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    df.to_csv(buf, index=index, lineterminator="\n", float_format=float_format)
    path.write_bytes(buf.getvalue().encode("utf-8"))


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Software": "matplotlib", "Creation Time": "2020-01-01T00:00:00Z"}
    fig.savefig(path, format="png", dpi=120, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def _fmt_float(v: float, decimals: int = 4) -> str:
    if not np.isfinite(v):
        return "NaN"
    return f"{v:.{decimals}f}"


# ---------------------------------------------------------------------------
# Load + hash-verify.
# ---------------------------------------------------------------------------


def load_inputs() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load cluster_assignments + shape_features; verify hashes; return joined df."""
    actual_hashes: Dict[str, str] = {
        "cluster_assignments.csv": _sha256_file(INPUT_CLUSTERS),
        "shape_features.csv": _sha256_file(INPUT_SHAPE),
    }
    for fname, expected in EXPECTED_INPUT_HASHES.items():
        actual = actual_hashes[fname]
        if actual != expected:
            raise RuntimeError(
                f"Input hash mismatch for {fname}: expected {expected}, got {actual}. HALT."
            )
    clusters = pd.read_csv(INPUT_CLUSTERS, low_memory=False)
    shape = pd.read_csv(INPUT_SHAPE, low_memory=False)
    shape["signal_time"] = pd.to_datetime(shape["signal_time"])

    # Keep only the columns we need. (Artifact column name is `actual_taken`,
    # which the prompt's input spec calls `taken` — artifact wins.)
    keep_shape_cols = [
        "signal_idx",
        "signal_time",
        "pair",
        "actual_taken",
        "peak_mfe_r",
        "peak_mae_r",
        "r_at_t240",
    ]
    shape = shape[keep_shape_cols].copy()
    shape = shape.rename(columns={"actual_taken": "taken"})
    keep_cluster_cols = ["signal_idx", "km_k4", "km_k6", "km_k8", "km_k10"]
    clusters = clusters[keep_cluster_cols].copy()

    joined = pd.merge(shape, clusters, on="signal_idx", how="left", validate="one_to_one")
    joined = joined.sort_values("signal_idx", kind="mergesort").reset_index(drop=True)
    return joined, actual_hashes


# ---------------------------------------------------------------------------
# Fold assignment.
# ---------------------------------------------------------------------------


def assign_folds(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Assign each row to a fold_id ∈ {1..7} based on signal_time. Drop rows
    outside the [2020-10-01, 2026-01-01) window with reason 'out_of_fold_window'.
    Returns (df_with_fold_col, drop_meta).
    """
    fold_id = np.zeros(len(df), dtype=np.int32)  # 0 = out-of-window
    ts = df["signal_time"].values
    for fid, start, end in FOLDS:
        start_ts = np.datetime64(pd.Timestamp(start))
        end_ts = np.datetime64(pd.Timestamp(end))
        mask = (ts >= start_ts) & (ts < end_ts)
        fold_id[mask] = fid
    df = df.copy()
    df["fold_id"] = fold_id

    drop_mask = df["fold_id"] == 0
    drop_count = int(drop_mask.sum())
    drops = df.loc[drop_mask, ["signal_idx", "signal_time", "pair"]].copy()
    drops["reason"] = "out_of_fold_window"

    kept = df.loc[~drop_mask].copy()
    meta = {
        "drop_count": drop_count,
        "drop_rows": drops.to_dict(orient="records"),
        "n_kept": int(len(kept)),
        "n_per_fold": {int(f): int((kept["fold_id"] == f).sum()) for f in range(1, 8)},
    }
    return kept, meta


# ---------------------------------------------------------------------------
# Analysis 1 — Cross-tabs at each k.
# ---------------------------------------------------------------------------


def crosstabs_at_k(df: pd.DataFrame, k: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    col = f"km_k{k}"
    sub = df.loc[df[col] >= 1].copy()  # exclude -1 sentinels (NaN-cluster rows)
    counts = pd.crosstab(sub[col], sub["fold_id"], dropna=False)
    # Ensure all clusters 1..k and folds 1..7 appear, even if empty.
    counts = counts.reindex(index=range(1, k + 1), columns=range(1, 8), fill_value=0)
    counts.index.name = "cluster_id"
    counts.columns.name = "fold_id"
    counts["Total"] = counts.sum(axis=1)
    counts.loc["Total"] = counts.sum(axis=0)

    raw = counts.drop(index="Total", columns="Total")
    col_totals = raw.sum(axis=0)
    row_totals = raw.sum(axis=1)
    int(raw.sum().sum())

    pct_of_fold = raw.divide(col_totals.replace(0, np.nan), axis=1)
    pct_of_fold["Total_check"] = pct_of_fold.sum(axis=1).fillna(0)
    pct_of_fold.loc["sum_check"] = pct_of_fold.sum(axis=0).fillna(0)

    pct_of_cluster = raw.divide(row_totals.replace(0, np.nan), axis=0)
    pct_of_cluster["Total_check"] = pct_of_cluster.sum(axis=1).fillna(0)
    pct_of_cluster.loc["sum_check"] = pct_of_cluster.sum(axis=0).fillna(0)
    return counts.reset_index(), pct_of_fold.reset_index(), pct_of_cluster.reset_index()


def analysis_1(df: pd.DataFrame) -> Dict[int, Dict[str, pd.DataFrame]]:
    out: Dict[int, Dict[str, pd.DataFrame]] = {}
    for k in K_VALUES:
        counts_df, pof_df, poc_df = crosstabs_at_k(df, k)
        out[k] = {"counts": counts_df, "pct_of_fold": pof_df, "pct_of_cluster": poc_df}
    return out


# ---------------------------------------------------------------------------
# Analysis 2 — Per-(cluster, fold) outcome stats at k=6.
# ---------------------------------------------------------------------------


def analysis_2(df: pd.DataFrame, k: int = K_PRIMARY) -> pd.DataFrame:
    col = f"km_k{k}"
    sub = df.loc[df[col] >= 1].copy()
    rows: List[Dict[str, Any]] = []
    for cluster in range(1, k + 1):
        for fold in range(1, 8):
            cell = sub.loc[(sub[col] == cluster) & (sub["fold_id"] == fold)]
            n = len(cell)
            base = {"cluster_id": cluster, "fold_id": fold, "n": n}
            if n == 0:
                rows.extend(
                    [
                        {**base, "metric": "r_at_t240", "stat": s, "value": float("nan")}
                        for s in ("mean", "std", "p25", "p50", "p75")
                    ]
                )
                rows.append({**base, "metric": "peak_mfe_r", "stat": "mean", "value": float("nan")})
                rows.append({**base, "metric": "peak_mae_r", "stat": "mean", "value": float("nan")})
                rows.append(
                    {**base, "metric": "pct_taken", "stat": "fraction", "value": float("nan")}
                )
                continue
            r240 = cell["r_at_t240"].to_numpy(dtype=np.float64)
            r240_finite = r240[np.isfinite(r240)]
            if r240_finite.size:
                rows.append(
                    {
                        **base,
                        "metric": "r_at_t240",
                        "stat": "mean",
                        "value": float(np.mean(r240_finite)),
                    }
                )
                rows.append(
                    {
                        **base,
                        "metric": "r_at_t240",
                        "stat": "std",
                        "value": float(np.std(r240_finite, ddof=1))
                        if r240_finite.size > 1
                        else float("nan"),
                    }
                )
                p25, p50, p75 = np.percentile(r240_finite, [25, 50, 75], method="linear")
                rows.append({**base, "metric": "r_at_t240", "stat": "p25", "value": float(p25)})
                rows.append({**base, "metric": "r_at_t240", "stat": "p50", "value": float(p50)})
                rows.append({**base, "metric": "r_at_t240", "stat": "p75", "value": float(p75)})
            else:
                for s in ("mean", "std", "p25", "p50", "p75"):
                    rows.append({**base, "metric": "r_at_t240", "stat": s, "value": float("nan")})
            pmfe = cell["peak_mfe_r"].to_numpy(dtype=np.float64)
            pmae = cell["peak_mae_r"].to_numpy(dtype=np.float64)
            rows.append(
                {**base, "metric": "peak_mfe_r", "stat": "mean", "value": float(np.nanmean(pmfe))}
            )
            rows.append(
                {**base, "metric": "peak_mae_r", "stat": "mean", "value": float(np.nanmean(pmae))}
            )
            taken = cell["taken"].astype(bool).to_numpy()
            rows.append(
                {**base, "metric": "pct_taken", "stat": "fraction", "value": float(np.mean(taken))}
            )
    out = (
        pd.DataFrame(rows)
        .sort_values(by=["cluster_id", "fold_id", "metric", "stat"], kind="mergesort")
        .reset_index(drop=True)
    )
    return out


# ---------------------------------------------------------------------------
# Analysis 3 — Independence diagnostics.
# ---------------------------------------------------------------------------


def _cramers_v_and_chi2(table: np.ndarray) -> Tuple[float, float, int, float]:
    """Return (chi2, p_value, dof, cramers_v) for a 2D contingency table.
    Cramer's V = sqrt(chi2 / (n * min(r-1, c-1)))."""
    table = np.asarray(table, dtype=np.float64)
    if table.sum() == 0:
        return float("nan"), float("nan"), 0, float("nan")
    chi2, p, dof, _ = sps.chi2_contingency(table, correction=False)
    n = table.sum()
    r, c = table.shape
    denom = n * (min(r, c) - 1) if min(r, c) > 1 else 1
    v = float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan")
    return float(chi2), float(p), int(dof), v


def analysis_3(df: pd.DataFrame) -> Dict[str, Any]:
    """Chi-square + Cramer's V for (cluster × fold), (cluster × pair), (cluster × month) at k=6."""
    col = f"km_k{K_PRIMARY}"
    sub = df.loc[df[col] >= 1].copy()

    # cluster × fold
    cf = pd.crosstab(sub[col], sub["fold_id"], dropna=False).to_numpy()
    chi2_cf, p_cf, dof_cf, v_cf = _cramers_v_and_chi2(cf)

    # cluster × pair
    cp = pd.crosstab(sub[col], sub["pair"], dropna=False).to_numpy()
    chi2_cp, p_cp, dof_cp, v_cp = _cramers_v_and_chi2(cp)

    # cluster × month
    sub = sub.copy()
    sub["month"] = sub["signal_time"].dt.month
    cm = pd.crosstab(sub[col], sub["month"], dropna=False).to_numpy()
    chi2_cm, p_cm, dof_cm, v_cm = _cramers_v_and_chi2(cm)

    return {
        "cluster_fold": {
            "chi2": chi2_cf,
            "p": p_cf,
            "dof": dof_cf,
            "cramers_v": v_cf,
            "n_rows": int(cf.shape[0]),
            "n_cols": int(cf.shape[1]),
        },
        "cluster_pair": {
            "chi2": chi2_cp,
            "p": p_cp,
            "dof": dof_cp,
            "cramers_v": v_cp,
            "n_rows": int(cp.shape[0]),
            "n_cols": int(cp.shape[1]),
        },
        "cluster_month": {
            "chi2": chi2_cm,
            "p": p_cm,
            "dof": dof_cm,
            "cramers_v": v_cm,
            "n_rows": int(cm.shape[0]),
            "n_cols": int(cm.shape[1]),
        },
    }


def cramers_v_label(v: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    if v < 0.1:
        return "negligible"
    if v < 0.2:
        return "weak"
    if v < 0.3:
        return "moderate"
    return "strong"


# ---------------------------------------------------------------------------
# Analysis 4 — Cluster contribution to fold-level pooled mean R.
# ---------------------------------------------------------------------------


def analysis_4(df: pd.DataFrame, k: int = K_PRIMARY) -> Tuple[pd.DataFrame, Dict[int, float]]:
    """Per (fold, cluster), compute the cluster's signed contribution to the
    fold's pooled mean r_at_t240.

    contribution = mean_r_cell × (n_cell / n_fold), so sum over clusters = pooled mean.
    cumulative_share = cumulative |contribution| / sum(|contributions|) sorted by abs desc.
    """
    col = f"km_k{k}"
    sub = df.loc[df[col] >= 1].copy()
    rows: List[Dict[str, Any]] = []
    fold_pooled_mean: Dict[int, float] = {}

    for fold in range(1, 8):
        fold_sub = sub.loc[sub["fold_id"] == fold]
        n_fold = len(fold_sub)
        if n_fold == 0:
            fold_pooled_mean[fold] = float("nan")
            continue
        r240_fold = fold_sub["r_at_t240"].to_numpy(dtype=np.float64)
        r240_fold = r240_fold[np.isfinite(r240_fold)]
        fold_pooled = float(np.mean(r240_fold)) if r240_fold.size else float("nan")
        fold_pooled_mean[fold] = fold_pooled

        cluster_contribs: List[Dict[str, Any]] = []
        for cluster in range(1, k + 1):
            cell = fold_sub.loc[fold_sub[col] == cluster]
            n_cell = len(cell)
            r240 = cell["r_at_t240"].to_numpy(dtype=np.float64)
            r240 = r240[np.isfinite(r240)]
            mean_r = float(np.mean(r240)) if r240.size else 0.0
            weight = (n_cell / n_fold) if n_fold > 0 else 0.0
            contrib = mean_r * weight
            cluster_contribs.append(
                {
                    "fold_id": fold,
                    "cluster_id": cluster,
                    "n_cell": n_cell,
                    "weight_n_over_fold": weight,
                    "mean_r_cell": mean_r,
                    "contribution_to_mean_r": contrib,
                }
            )

        # Sort by |contribution| desc to compute cumulative share.
        cluster_contribs.sort(key=lambda r: abs(r["contribution_to_mean_r"]), reverse=True)
        total_abs = sum(abs(r["contribution_to_mean_r"]) for r in cluster_contribs) or 1.0
        cum = 0.0
        for rank, r in enumerate(cluster_contribs, start=1):
            cum += abs(r["contribution_to_mean_r"])
            r["abs_rank"] = rank
            r["cumulative_share"] = cum / total_abs
            rows.append(r)

    out = (
        pd.DataFrame(rows)
        .sort_values(by=["fold_id", "abs_rank"], kind="mergesort")
        .reset_index(drop=True)
    )
    return out, fold_pooled_mean


# ---------------------------------------------------------------------------
# Charts.
# ---------------------------------------------------------------------------


def _cluster_palette(k: int) -> List[Tuple[float, float, float, float]]:
    cmap = plt.cm.tab10
    return [cmap(i % 10) for i in range(k)]


def chart_cluster_composition_per_fold(df: pd.DataFrame, k: int, path: Path) -> None:
    col = f"km_k{k}"
    sub = df.loc[df[col] >= 1]
    table = pd.crosstab(sub["fold_id"], sub[col], dropna=False)
    table = table.reindex(index=range(1, 8), columns=range(1, k + 1), fill_value=0)
    row_totals = table.sum(axis=1).replace(0, np.nan)
    pct = table.divide(row_totals, axis=0)

    fig, ax = plt.subplots(figsize=(11, 6))
    palette = _cluster_palette(k)
    bottom = np.zeros(len(pct))
    folds = pct.index.tolist()
    for i, cluster in enumerate(pct.columns):
        vals = pct[cluster].to_numpy(dtype=np.float64)
        ax.bar(
            folds,
            vals,
            bottom=bottom,
            color=palette[i],
            label=f"cluster {cluster}",
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += vals
    ax.set_xlabel("fold_id")
    ax.set_ylabel("proportion of fold")
    ax.set_title(f"Cluster composition per fold (k={k}, km full)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8, ncol=2, bbox_to_anchor=(1.18, 1.0))
    _save_figure(fig, path)


def chart_fold_composition_per_cluster(df: pd.DataFrame, k: int, path: Path) -> None:
    col = f"km_k{k}"
    sub = df.loc[df[col] >= 1]
    table = pd.crosstab(sub[col], sub["fold_id"], dropna=False)
    table = table.reindex(index=range(1, k + 1), columns=range(1, 8), fill_value=0)
    row_totals = table.sum(axis=1).replace(0, np.nan)
    pct = table.divide(row_totals, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    fold_palette = plt.cm.viridis(np.linspace(0.05, 0.95, 7))
    bottom = np.zeros(len(pct))
    clusters = pct.index.tolist()
    for i, fold in enumerate(pct.columns):
        vals = pct[fold].to_numpy(dtype=np.float64)
        ax.bar(
            clusters,
            vals,
            bottom=bottom,
            color=fold_palette[i],
            label=f"fold {fold}",
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += vals
    ax.set_xlabel("cluster_id")
    ax.set_ylabel("proportion of cluster")
    ax.set_title(f"Fold composition per cluster (k={k}, km full)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8, ncol=1, bbox_to_anchor=(1.18, 1.0))
    _save_figure(fig, path)


def chart_mean_r_heatmap(outcomes: pd.DataFrame, k: int, path: Path) -> None:
    """Heatmap of mean_r_at_t240 per (cluster, fold), annotated with value and N."""
    pivot_mean = (
        outcomes[(outcomes["metric"] == "r_at_t240") & (outcomes["stat"] == "mean")]
        .pivot(index="cluster_id", columns="fold_id", values="value")
        .reindex(index=range(1, k + 1), columns=range(1, 8))
    )
    pivot_n = (
        outcomes[outcomes["metric"] == "r_at_t240"]
        .groupby(["cluster_id", "fold_id"], as_index=False)["n"]
        .max()
        .pivot(index="cluster_id", columns="fold_id", values="n")
        .reindex(index=range(1, k + 1), columns=range(1, 8), fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    arr = pivot_mean.to_numpy(dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", norm=norm, origin="upper")
    ax.set_xticks(range(7))
    ax.set_xticklabels([str(f) for f in range(1, 8)])
    ax.set_yticks(range(k))
    ax.set_yticklabels([str(c) for c in range(1, k + 1)])
    ax.set_xlabel("fold_id")
    ax.set_ylabel("cluster_id")
    ax.set_title(f"Mean r_at_t240 by (cluster, fold) (k={k}, km full)")
    plt.colorbar(im, ax=ax, label="mean r_at_t240 (R)")
    for i in range(k):
        for j in range(7):
            mean_v = arr[i, j]
            n_v = int(pivot_n.iat[i, j]) if not np.isnan(pivot_n.iat[i, j]) else 0
            txt = f"{mean_v:+.2f}\nN={n_v}" if np.isfinite(mean_v) else "n/a"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")
    _save_figure(fig, path)


def chart_cluster_contribution_to_fold_mean(contrib_df: pd.DataFrame, k: int, path: Path) -> None:
    """Per fold, a stacked bar showing each cluster's signed contribution to pooled mean R."""
    fig, ax = plt.subplots(figsize=(11, 6))
    palette = _cluster_palette(k)
    folds = sorted(contrib_df["fold_id"].unique().tolist())
    # Separate positive and negative contributions for stacking.
    pos_bottom = np.zeros(len(folds))
    neg_bottom = np.zeros(len(folds))
    bar_positions = np.arange(len(folds))
    for cluster in range(1, k + 1):
        vals = []
        for fold in folds:
            cell = contrib_df[
                (contrib_df["fold_id"] == fold) & (contrib_df["cluster_id"] == cluster)
            ]
            vals.append(float(cell["contribution_to_mean_r"].iloc[0]) if len(cell) else 0.0)
        vals = np.asarray(vals)
        pos_heights = np.where(vals > 0, vals, 0.0)
        neg_heights = np.where(vals < 0, vals, 0.0)
        if (pos_heights != 0).any():
            ax.bar(
                bar_positions,
                pos_heights,
                bottom=pos_bottom,
                color=palette[cluster - 1],
                label=f"cluster {cluster}",
                edgecolor="white",
                linewidth=0.5,
            )
            pos_bottom = pos_bottom + pos_heights
        if (neg_heights != 0).any():
            ax.bar(
                bar_positions,
                neg_heights,
                bottom=neg_bottom,
                color=palette[cluster - 1],
                edgecolor="white",
                linewidth=0.5,
            )
            neg_bottom = neg_bottom + neg_heights
    ax.axhline(0.0, color="black", lw=0.6)
    # Plot pooled mean as a marker.
    pooled = []
    for fold in folds:
        cell = contrib_df[contrib_df["fold_id"] == fold]
        pooled.append(float(cell["contribution_to_mean_r"].sum()))
    ax.plot(bar_positions, pooled, "ko-", lw=1.5, ms=6, label="pooled mean (sum)")
    ax.set_xticks(bar_positions)
    ax.set_xticklabels([str(f) for f in folds])
    ax.set_xlabel("fold_id")
    ax.set_ylabel("contribution to fold pooled mean r_at_t240 (R)")
    ax.set_title(f"Cluster contributions to fold-level pooled mean R (k={k}, km full)")
    # Dedupe legend labels.
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict()
    for h, lbl in zip(handles, labels):
        if lbl not in by_label:
            by_label[lbl] = h
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper right",
        fontsize=8,
        ncol=2,
        bbox_to_anchor=(1.18, 1.0),
    )
    _save_figure(fig, path)


# ---------------------------------------------------------------------------
# Independence diagnostics markdown.
# ---------------------------------------------------------------------------


def write_independence_md(diag: Dict[str, Any], path: Path) -> None:
    lines: List[str] = [
        "# Arc 2 Per-Fold Clusters — Independence Diagnostics",
        "",
        "Tables: 2D contingency tables of cluster (km_k6) against fold, pair, and month_of_year.",
        "Chi-square is computed without continuity correction. Cramer's V is the effect-size measure",
        "appropriate for large samples (with N ~= 41,538, the chi-square p-value is dominated by",
        "sample size — Cramer's V is the meaningful quantity).",
        "",
        "Interpretation of Cramer's V (Cohen, broadly applied):",
        "",
        "| range | interpretation |",
        "|-------|----------------|",
        "| 0.0–0.1 | negligible association |",
        "| 0.1–0.2 | weak association |",
        "| 0.2–0.3 | moderate association |",
        "| 0.3+ | strong association |",
        "",
        "## Diagnostics",
        "",
        "| pairing | rows | cols | dof | chi2 | p_value | Cramer's V | interpretation |",
        "|---------|------|------|-----|------|---------|------------|----------------|",
    ]
    for key, label in (
        ("cluster_fold", "cluster × fold (primary)"),
        ("cluster_pair", "cluster × pair (28 pairs)"),
        ("cluster_month", "cluster × month_of_year (12 months)"),
    ):
        d = diag[key]
        lines.append(
            f"| {label} | {d['n_rows']} | {d['n_cols']} | {d['dof']} | "
            f"{d['chi2']:.2f} | {d['p']:.4g} | {_fmt_float(d['cramers_v'])} | {cramers_v_label(d['cramers_v'])} |"
        )
    lines.append("")
    lines.append(
        "**Reading guide.** The (cluster × fold) Cramer's V is the open-question quantity."
    )
    lines.append(
        "If it materially exceeds the (cluster × pair) and (cluster × month) comparators, the "
        "cluster structure is more strongly associated with fold (i.e., time regime) than with "
        "pair or seasonal month — consistent with the clusters acting partially as regime "
        "detectors. If it is similar or smaller, the cluster structure is roughly regime-"
        "independent in the same sense as it is pair- or month-independent."
    )
    lines.append("")
    _write_text("\n".join(lines) + "\n", path)


# ---------------------------------------------------------------------------
# Report writer.
# ---------------------------------------------------------------------------


def write_report(
    *,
    actual_hashes: Dict[str, str],
    n_total_input: int,
    fold_meta: Dict[str, Any],
    crosstabs: Dict[int, Dict[str, pd.DataFrame]],
    outcomes_df: pd.DataFrame,
    diag: Dict[str, Any],
    contrib_df: pd.DataFrame,
    pooled_mean_per_fold: Dict[int, float],
    path: Path,
) -> None:
    lines: List[str] = [
        "# Arc 2 Per-Fold Cluster Breakdown — Empirical Report",
        "",
        "Phase: l6_arc2_per_fold_clusters (supplementary, descriptive only per L6.0 v1.1 §14.5).",
        "Open question: are k=6 clusters concentrated in specific Arc 2 WFO OOS folds, or are they",
        "distributed roughly evenly across folds?",
        "",
        "## 1. Inputs and hashes",
        "",
        f"- cluster_assignments.csv sha256: {actual_hashes['cluster_assignments.csv']}",
        f"- shape_features.csv sha256: {actual_hashes['shape_features.csv']}",
        f"- N rows merged from shape_features: {n_total_input}",
        "",
        "## 2. Fold assignment summary",
        "",
        f"- N kept (in [2020-10-01, 2026-01-01) window): {fold_meta['n_kept']}",
        f"- N dropped (out_of_fold_window): {fold_meta['drop_count']}",
        "",
        "Per-fold counts (ex-ante, full population — taken + non-taken):",
        "",
        "| fold | n_ex_ante | n_taken_wfo (reference) | mean_R_taken (reference) | roi_pct (reference) |",
        "|------|-----------|--------------------------|---------------------------|----------------------|",
    ]
    for fid, n_taken, mean_R, _, roi, _ in WFO_REFERENCE:
        n_xa = fold_meta["n_per_fold"][fid]
        lines.append(
            f"| {fid} | {n_xa} | {n_taken} | {_fmt_float(mean_R)} | {_fmt_float(roi, 2)} |"
        )
    lines.append("")

    lines.append("## 3. Cross-tab tables at k=6 (km full)")
    lines.append("")
    lines.append(
        "All cross-tabs at k=4, k=6, k=8, k=10 are persisted as CSV files in this directory."
    )
    lines.append("The primary table (k=6) is shown below.")
    lines.append("")
    lines.append("### 3.1 Counts (cluster × fold)")
    lines.append("")
    counts_df = crosstabs[K_PRIMARY]["counts"]
    lines.append("| cluster_id | f1 | f2 | f3 | f4 | f5 | f6 | f7 | Total |")
    lines.append("|------------|----|----|----|----|----|----|----|-------|")
    for _, row in counts_df.iterrows():
        cid = row["cluster_id"]
        cid_s = "Total" if cid == "Total" else str(int(cid))
        cells = [str(int(row[f])) for f in list(range(1, 8))]
        total = str(int(row["Total"]))
        lines.append(f"| {cid_s} | " + " | ".join(cells) + f" | {total} |")
    lines.append("")
    lines.append("### 3.2 Pct of fold (column proportions; sum to 1.0 per fold)")
    lines.append("")
    pof_df = crosstabs[K_PRIMARY]["pct_of_fold"]
    lines.append("| cluster_id | f1 | f2 | f3 | f4 | f5 | f6 | f7 | sum_row |")
    lines.append("|------------|----|----|----|----|----|----|----|---------|")
    for _, row in pof_df.iterrows():
        cid = row["cluster_id"]
        cid_s = "sum_check" if cid == "sum_check" else str(int(cid))
        cells = [_fmt_float(float(row[f])) for f in list(range(1, 8))]
        sum_row = _fmt_float(float(row["Total_check"]))
        lines.append(f"| {cid_s} | " + " | ".join(cells) + f" | {sum_row} |")
    lines.append("")
    lines.append("### 3.3 Pct of cluster (row proportions; sum to 1.0 per cluster)")
    lines.append("")
    poc_df = crosstabs[K_PRIMARY]["pct_of_cluster"]
    lines.append("| cluster_id | f1 | f2 | f3 | f4 | f5 | f6 | f7 | sum_row |")
    lines.append("|------------|----|----|----|----|----|----|----|---------|")
    for _, row in poc_df.iterrows():
        cid = row["cluster_id"]
        cid_s = "sum_check" if cid == "sum_check" else str(int(cid))
        cells = [_fmt_float(float(row[f])) for f in list(range(1, 8))]
        sum_row = _fmt_float(float(row["Total_check"]))
        lines.append(f"| {cid_s} | " + " | ".join(cells) + f" | {sum_row} |")
    lines.append("")

    lines.append("## 4. Per-(cluster, fold) outcome stats at k=6")
    lines.append("")
    lines.append("`mean_r_at_t240` per cell (clusters × folds):")
    lines.append("")
    lines.append("| cluster | f1 | f2 | f3 | f4 | f5 | f6 | f7 |")
    lines.append("|---------|----|----|----|----|----|----|----|")
    means = outcomes_df[(outcomes_df["metric"] == "r_at_t240") & (outcomes_df["stat"] == "mean")]
    for cluster in range(1, K_PRIMARY + 1):
        row_means = []
        for fold in range(1, 8):
            cell = means[(means["cluster_id"] == cluster) & (means["fold_id"] == fold)]
            row_means.append(_fmt_float(float(cell["value"].iloc[0])) if len(cell) else "NaN")
        lines.append(f"| {cluster} | " + " | ".join(row_means) + " |")
    lines.append("")
    lines.append(
        "Full long-format table (mean, std, p25, p50, p75 of r_at_t240, plus mean of peak_mfe_r / peak_mae_r and pct_taken) in `cluster_fold_outcomes_k6.csv`."
    )
    lines.append("")

    lines.append("## 5. Independence diagnostics")
    lines.append("")
    lines.append("Full table in `independence_diagnostics.md`. Summary:")
    lines.append("")
    lines.append("| pairing | dof | chi2 | p_value | Cramer's V | interpretation |")
    lines.append("|---------|-----|------|---------|------------|----------------|")
    for key, label in (
        ("cluster_fold", "cluster × fold (primary)"),
        ("cluster_pair", "cluster × pair"),
        ("cluster_month", "cluster × month_of_year"),
    ):
        d = diag[key]
        lines.append(
            f"| {label} | {d['dof']} | {d['chi2']:.2f} | {d['p']:.4g} | "
            f"{_fmt_float(d['cramers_v'])} | {cramers_v_label(d['cramers_v'])} |"
        )
    lines.append("")
    lines.append(
        "Reading: Cramer's V is the effect-size measure; the chi-square p-value at N~=41k is "
        "uninformative on its own because it is almost certainly small."
    )
    lines.append("")

    lines.append("## 6. Cluster contribution to fold-level pooled mean r_at_t240")
    lines.append("")
    lines.append("For each fold, the cluster's signed contribution to the fold's pooled mean is")
    lines.append("`mean_r_at_t240 (within cell) × n_cell / n_fold`. Summed over clusters in a")
    lines.append("fold, these recover the pooled mean. Sorted within each fold by |contribution|")
    lines.append("descending; `cumulative_share` is the running fraction of total |contribution|.")
    lines.append("")
    lines.append("Per-fold pooled mean r_at_t240 (ex-ante population):")
    lines.append("")
    lines.append(
        "| fold | n_ex_ante (in cluster set) | pooled_mean_r_at_t240 | wfo_mean_R_taken (reference) |"
    )
    lines.append(
        "|------|------------------------------|-------------------------|--------------------------------|"
    )
    for fid in range(1, 8):
        contrib_fold = contrib_df[contrib_df["fold_id"] == fid]
        n_fold = int(contrib_fold["n_cell"].sum())
        pooled = pooled_mean_per_fold.get(fid, float("nan"))
        wfo_mean = next((m for f, _, m, _, _, _ in WFO_REFERENCE if f == fid), float("nan"))
        lines.append(f"| {fid} | {n_fold} | {_fmt_float(pooled)} | {_fmt_float(wfo_mean)} |")
    lines.append("")
    lines.append("Per-fold dominant contributors (top-3 by |contribution|):")
    lines.append("")
    lines.append(
        "| fold | rank | cluster | n_cell | weight | mean_r_cell | contribution | cumulative_share |"
    )
    lines.append(
        "|------|------|---------|--------|--------|-------------|--------------|-------------------|"
    )
    for fid in range(1, 8):
        contrib_fold = contrib_df[contrib_df["fold_id"] == fid].sort_values(
            "abs_rank", kind="mergesort"
        )
        for _, r in contrib_fold.head(3).iterrows():
            lines.append(
                f"| {int(r['fold_id'])} | {int(r['abs_rank'])} | {int(r['cluster_id'])} | "
                f"{int(r['n_cell'])} | {_fmt_float(float(r['weight_n_over_fold']))} | "
                f"{_fmt_float(float(r['mean_r_cell']))} | "
                f"{_fmt_float(float(r['contribution_to_mean_r']))} | "
                f"{_fmt_float(float(r['cumulative_share']))} |"
            )
    lines.append("")
    lines.append("Full long-format table in `cluster_contribution_per_fold.csv`.")
    lines.append("")

    lines.append("## 7. Observations")
    lines.append("")
    # Empirical observations only — no prescription. Sentences derived from the data.
    # Highlight the structural facts the prompt asked about.
    diag_cf = diag["cluster_fold"]
    diag_cp = diag["cluster_pair"]
    diag_cm = diag["cluster_month"]
    lines.append(
        f"- Cramer's V for (cluster × fold) at k=6 = {_fmt_float(diag_cf['cramers_v'])} "
        f"({cramers_v_label(diag_cf['cramers_v'])} association). "
        f"Comparators: (cluster × pair) = {_fmt_float(diag_cp['cramers_v'])} "
        f"({cramers_v_label(diag_cp['cramers_v'])}), "
        f"(cluster × month_of_year) = {_fmt_float(diag_cm['cramers_v'])} "
        f"({cramers_v_label(diag_cm['cramers_v'])})."
    )
    # Compute concentration extreme observations from pct_of_cluster table.
    poc = crosstabs[K_PRIMARY]["pct_of_cluster"].copy()
    poc_clean = poc[
        poc["cluster_id"].apply(
            lambda v: isinstance(v, (int, np.integer)) or (isinstance(v, str) and v != "sum_check")
        )
    ]
    poc_clean = poc_clean[poc_clean["cluster_id"] != "sum_check"]
    # Find each cluster's most-concentrated fold.
    for _, row in poc_clean.iterrows():
        try:
            cid = int(row["cluster_id"])
        except (TypeError, ValueError):
            continue
        fold_pcts = [(int(f), float(row[f])) for f in range(1, 8) if pd.notna(row[f])]
        if not fold_pcts:
            continue
        top_fold, top_pct = max(fold_pcts, key=lambda x: x[1])
        bot_fold, bot_pct = min(fold_pcts, key=lambda x: x[1])
        cluster_total = int(
            crosstabs[K_PRIMARY]["counts"][crosstabs[K_PRIMARY]["counts"]["cluster_id"] == cid][
                "Total"
            ].iloc[0]
        )
        top_n = int(round(top_pct * cluster_total))
        bot_n = int(round(bot_pct * cluster_total))
        lines.append(
            f"- Cluster {cid} (k=6, n={cluster_total}): highest membership share in fold {top_fold} "
            f"({_fmt_float(top_pct)} ~= n={top_n}); lowest in fold {bot_fold} "
            f"({_fmt_float(bot_pct)} ~= n={bot_n})."
        )
    # Per-fold pooled mean r_at_t240 (ex-ante) vs WFO reference.
    for fid in range(1, 8):
        pooled = pooled_mean_per_fold.get(fid, float("nan"))
        wfo_mean = next((m for f, _, m, _, _, _ in WFO_REFERENCE if f == fid), float("nan"))
        lines.append(
            f"- Fold {fid}: ex-ante pooled mean r_at_t240 = {_fmt_float(pooled)}; "
            f"WFO taken-trade mean R (reference, with execution costs) = {_fmt_float(wfo_mean)}."
        )
    # Top contributor per fold.
    for fid in range(1, 8):
        contrib_fold = contrib_df[contrib_df["fold_id"] == fid].sort_values(
            "abs_rank", kind="mergesort"
        )
        if len(contrib_fold) == 0:
            continue
        top = contrib_fold.iloc[0]
        lines.append(
            f"- Fold {fid}: top contributor to pooled mean is cluster {int(top['cluster_id'])} "
            f"with contribution {_fmt_float(float(top['contribution_to_mean_r']))} R/signal "
            f"(weight {_fmt_float(float(top['weight_n_over_fold']))}, "
            f"mean_r_cell {_fmt_float(float(top['mean_r_cell']))}); "
            f"cumulative_share at rank 1 = {_fmt_float(float(top['cumulative_share']))}."
        )
    lines.append("")

    lines.append("## 8. Out of scope items observed")
    lines.append("")
    lines.append(
        "- Per-pair × fold cluster breakdown: not produced (would multiply table count by 28)."
    )
    lines.append(
        "- IS-period cluster assignments: not produced; trajectory appendix's clustering is"
    )
    lines.append("  on the full ex-ante population, which has both IS and OOS rows by construction")
    lines.append(
        "  (signals_features.csv labels by fold_disposition). This analysis uses signal_time"
    )
    lines.append("  alone for fold assignment.")
    lines.append("- Hierarchical (subsample) cluster labels: only the km full assignments are used")
    lines.append("  here. The hier_k* columns in cluster_assignments.csv have most entries marked")
    lines.append("  -1 (not in the 8000-row subsample), so per-fold breakdown would be biased by")
    lines.append("  the subsampling.")
    lines.append("- Per-(cluster, fold) cell distribution histograms: only summary stats are")
    lines.append("  produced. Full per-cell histograms would multiply output volume by ~25.")
    lines.append("")

    lines.append("## §14.5 discipline")
    lines.append("")
    lines.append(
        "This report is empirical observations only. No filter, exit, or sizing recommendations "
        "are made here. Action-shaped patterns (if any) appear in "
        "`../CANDIDATE_HYPOTHESES_DRAFT.md` (draft, not committed to the registry)."
    )
    lines.append("")

    _write_text("\n".join(lines) + "\n", path)


# ---------------------------------------------------------------------------
# Manifest writer.
# ---------------------------------------------------------------------------


def write_manifest(
    *,
    out_paths: List[Path],
    actual_input_hashes: Dict[str, str],
    fold_meta: Dict[str, Any],
    diag: Dict[str, Any],
    prior_hashes: Optional[Dict[str, str]],
    run_ordinal: int,
) -> Dict[str, str]:
    hashes: "OrderedDict[str, str]" = OrderedDict()
    pixel_hashes: "OrderedDict[str, str]" = OrderedDict()
    for p in out_paths:
        rel = p.relative_to(OUTPUT_DIR).as_posix()
        hashes[rel] = _sha256_file(p)
        if p.suffix.lower() == ".png":
            pixel_hashes[rel] = _png_pixel_hash(p)

    lines: List[str] = [
        f"Arc 2 Per-Fold Clusters — run_manifest (run #{run_ordinal})",
        "Generated: (suppressed in deterministic mode below)",
        "",
        "## Operational decisions",
        "",
        "- Fold boundaries from PHASE_L6_ARC2_RESULT.md §5 (lines 148-154); left-closed, right-open.",
        "- Cluster assignments from trajectory appendix's k-means (full kept N); km_k* values of -1",
        "  indicate the 256 NaN-feature rows excluded from clustering (also excluded here).",
        "- Hier_* columns (subsample only) are NOT used for cross-tabs — would be biased by",
        "  the 8000-row subsampling.",
        "- Cramer's V computed without continuity correction (Yates correction = False), consistent",
        "  with chi2_contingency default for tables larger than 2x2.",
        "- Per-fold pooled mean r_at_t240 is computed on the ex-ante full population (taken +",
        "  non-taken). WFO reference numbers from §5 (taken-only) are comparators only.",
        "",
        "## Inputs (read-only)",
        "",
        f"- cluster_assignments.csv sha256: {actual_input_hashes['cluster_assignments.csv']}",
        f"- shape_features.csv sha256: {actual_input_hashes['shape_features.csv']}",
        "",
        "## Fold assignment",
        "",
        f"- N kept: {fold_meta['n_kept']}",
        f"- N dropped (out_of_fold_window): {fold_meta['drop_count']}",
    ]
    for fid in range(1, 8):
        lines.append(f"- Fold {fid}: n_ex_ante = {fold_meta['n_per_fold'][fid]}")
    lines.append("")
    lines.append("## Independence diagnostics (k=6)")
    lines.append("")
    for key, label in (
        ("cluster_fold", "cluster x fold"),
        ("cluster_pair", "cluster x pair"),
        ("cluster_month", "cluster x month_of_year"),
    ):
        d = diag[key]
        lines.append(
            f"- {label}: chi2 = {d['chi2']:.2f}, dof = {d['dof']}, "
            f"p = {d['p']:.4g}, Cramer's V = {d['cramers_v']:.4f} ({cramers_v_label(d['cramers_v'])})"
        )
    lines.append("")
    lines.append("## Determinism config")
    lines.append("")
    lines.append(f"- NUMPY_SEED = {NUMPY_SEED}")
    lines.append("- Sort order: rows sorted by signal_idx ascending after the merge.")
    lines.append("- CSV: utf-8, LF line endings, float_format='%.10g'.")
    lines.append(
        "- Matplotlib: Agg backend, metadata pinned (Software='matplotlib', Creation Time='2020-01-01')."
    )
    lines.append("")
    lines.append(f"## Output file sha256 (run #{run_ordinal})")
    lines.append("")
    for rel, h in hashes.items():
        lines.append(f"{h}  {rel}")
    lines.append("")
    if pixel_hashes:
        lines.append(f"## PNG pixel-only sha256 (run #{run_ordinal}) — metadata-stripped fallback")
        lines.append("")
        for rel, h in pixel_hashes.items():
            lines.append(f"{h}  {rel}")
        lines.append("")

    if prior_hashes is not None:
        lines.append("## Output file sha256 (run #1 — prior)")
        lines.append("")
        for rel in hashes.keys():
            prior = prior_hashes.get(rel, "MISSING")
            lines.append(f"{prior}  {rel}")
        lines.append("")
        lines.append("## Byte-identicality vs prior run")
        lines.append("")
        all_match = True
        for rel, h in hashes.items():
            prior = prior_hashes.get(rel)
            ok = prior == h
            all_match = all_match and ok
            lines.append(f"- {rel}: {'IDENTICAL' if ok else 'DIVERGED'}")
        lines.append("")
        lines.append(
            f"Overall: {'PASS - byte-identical across runs' if all_match else 'FAIL - some files diverged'}"
        )
        lines.append("")

    _write_text("\n".join(lines) + "\n", OUTPUT_DIR / "run_manifest.txt")
    return dict(hashes)


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------


def run_pipeline(
    *, run_ordinal: int, prior_hashes: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[run #{run_ordinal}] Loading + hash-verifying inputs ...", flush=True)
    joined, actual_hashes = load_inputs()
    n_total_input = len(joined)

    print(f"[run #{run_ordinal}] Assigning folds ...", flush=True)
    df, fold_meta = assign_folds(joined)
    if fold_meta["drop_count"] > 10:
        raise RuntimeError(
            f"BLOCKER: {fold_meta['drop_count']} signals fall outside the fold window — "
            "investigate before proceeding."
        )

    print(f"[run #{run_ordinal}] Analysis 1 — cross-tabs at k in {{4,6,8,10}} ...", flush=True)
    ct_by_k = analysis_1(df)

    print(f"[run #{run_ordinal}] Analysis 2 — per-(cluster, fold) outcomes at k=6 ...", flush=True)
    outcomes_df = analysis_2(df, k=K_PRIMARY)

    print(f"[run #{run_ordinal}] Analysis 3 — independence diagnostics ...", flush=True)
    diag = analysis_3(df)

    print(
        f"[run #{run_ordinal}] Analysis 4 — cluster contributions to fold pooled mean R ...",
        flush=True,
    )
    contrib_df, pooled_mean = analysis_4(df, k=K_PRIMARY)

    print(f"[run #{run_ordinal}] Writing outputs ...", flush=True)
    out_paths: List[Path] = []
    for k in K_VALUES:
        for stat_key, fname in (
            ("counts", f"cluster_fold_counts_k{k}.csv"),
            ("pct_of_fold", f"cluster_fold_pct_of_fold_k{k}.csv"),
            ("pct_of_cluster", f"cluster_fold_pct_of_cluster_k{k}.csv"),
        ):
            p = OUTPUT_DIR / fname
            _write_csv(ct_by_k[k][stat_key], p)
            out_paths.append(p)
    _write_csv(outcomes_df, OUTPUT_DIR / "cluster_fold_outcomes_k6.csv")
    out_paths.append(OUTPUT_DIR / "cluster_fold_outcomes_k6.csv")
    _write_csv(contrib_df, OUTPUT_DIR / "cluster_contribution_per_fold.csv")
    out_paths.append(OUTPUT_DIR / "cluster_contribution_per_fold.csv")

    write_independence_md(diag, OUTPUT_DIR / "independence_diagnostics.md")
    out_paths.append(OUTPUT_DIR / "independence_diagnostics.md")

    chart_cluster_composition_per_fold(
        df, K_PRIMARY, CHARTS_DIR / "cluster_composition_per_fold.png"
    )
    out_paths.append(CHARTS_DIR / "cluster_composition_per_fold.png")
    chart_fold_composition_per_cluster(
        df, K_PRIMARY, CHARTS_DIR / "fold_composition_per_cluster.png"
    )
    out_paths.append(CHARTS_DIR / "fold_composition_per_cluster.png")
    chart_mean_r_heatmap(outcomes_df, K_PRIMARY, CHARTS_DIR / "mean_r_heatmap_cluster_fold.png")
    out_paths.append(CHARTS_DIR / "mean_r_heatmap_cluster_fold.png")
    chart_cluster_contribution_to_fold_mean(
        contrib_df, K_PRIMARY, CHARTS_DIR / "cluster_contribution_to_fold_mean.png"
    )
    out_paths.append(CHARTS_DIR / "cluster_contribution_to_fold_mean.png")

    print(f"[run #{run_ordinal}] Writing report ...", flush=True)
    write_report(
        actual_hashes=actual_hashes,
        n_total_input=n_total_input,
        fold_meta=fold_meta,
        crosstabs=ct_by_k,
        outcomes_df=outcomes_df,
        diag=diag,
        contrib_df=contrib_df,
        pooled_mean_per_fold=pooled_mean,
        path=OUTPUT_DIR / "per_fold_clusters_report.md",
    )
    out_paths.append(OUTPUT_DIR / "per_fold_clusters_report.md")

    print(f"[run #{run_ordinal}] Writing run_manifest.txt ...", flush=True)
    hashes = write_manifest(
        out_paths=out_paths,
        actual_input_hashes=actual_hashes,
        fold_meta=fold_meta,
        diag=diag,
        prior_hashes=prior_hashes,
        run_ordinal=run_ordinal,
    )
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description="Arc 2 Per-Fold Cluster Breakdown")
    parser.add_argument("--mode", choices=("once", "verify"), default="verify")
    args = parser.parse_args()

    np.random.seed(NUMPY_SEED)

    if args.mode == "once":
        run_pipeline(run_ordinal=1)
        return 0

    run1 = run_pipeline(run_ordinal=1)
    run2 = run_pipeline(run_ordinal=2, prior_hashes=run1)

    all_match = True
    for rel, h in run1.items():
        if run2.get(rel) != h:
            all_match = False
            print(f"  DIVERGED: {rel}", flush=True)
    if all_match:
        print("[verify] All outputs are byte-identical across the two runs.", flush=True)
    else:
        print("[verify] Some outputs diverged.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
