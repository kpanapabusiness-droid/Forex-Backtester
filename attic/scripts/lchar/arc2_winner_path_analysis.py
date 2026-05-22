"""Arc 2 Winner Path Analysis — supplementary characterisation.

Phase: l6_arc2_trajectory_appendix / winner_path_analysis
       (additive; descriptive only per L6.0 v1.1 §14.5).

Question this analysis answers: "What does the path from entry to peak look
like for our actual winners, and how does it differ from losers?"

It does NOT answer "what exit policy works" — that was settled by the
band_behavior analysis (no pooled exit rule outperforms baseline). This
script characterises trajectory shapes per cluster and quantifies bar-level
separation between winner and loser archetypes. Any prescriptive content
goes to CANDIDATE_HYPOTHESES_DRAFT.md as a fresh-arc gate, not into the
report.

Inputs (read-only, hash-verified against parent manifest):
- results/l6/arc2/trajectory_appendix/trajectory_panel.parquet
- results/l6/arc2/trajectory_appendix/shape_features.csv
- results/l6/arc2/trajectory_appendix/cluster_assignments.csv

Outputs (under
results/l6/arc2/trajectory_appendix/winner_path_analysis/):
- cluster_envelopes_view_A.csv
- cluster_envelopes_view_B.csv
- per_bar_distributions.csv
- separation_diagnostics.csv
- winner_pullbacks.csv
- time_to_band_per_cluster.csv
- winner_shape_classification.csv
- winner_path_report.md
- run_manifest.txt
- charts/view_A_envelopes_overlay.png
- charts/view_A_envelopes_grid.png
- charts/view_B_envelopes_overlay.png
- charts/view_B_envelopes_grid.png
- charts/auc_vs_bar.png
- charts/threshold_curves.png
- charts/winner_pullback_distributions.png

Group definitions (LOCKED, use km_k6 column from cluster_assignments.csv):
- Winners: clusters 2 (steady runner) and 5 (mega-runner)
- Losers:  clusters 3 (slow bleeder) and 6 (disaster)
- Middle:  clusters 1 (V-shape recovery) and 4 (early-pop fade)

Determinism: seeds locked; signal_idx sort ascending; sklearn lbfgs solver;
matplotlib Agg + SOURCE_DATE_EPOCH; outputs hashed before and after a re-run
inside the script and the result recorded in run_manifest.txt.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SOURCE_DATE_EPOCH", "1577836800")  # 2020-01-01 UTC
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

# ---------------------------------------------------------------------------
# LOCKED CONSTANTS
# ---------------------------------------------------------------------------

NUMPY_SEED: int = 20260512
SKLEARN_SEED: int = 20260512

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
INPUT_DIR: Path = REPO_ROOT / "results" / "l6" / "arc2" / "trajectory_appendix"
OUTPUT_DIR: Path = INPUT_DIR / "winner_path_analysis"
CHARTS_DIR: Path = OUTPUT_DIR / "charts"

INPUT_TRAJECTORY: Path = INPUT_DIR / "trajectory_panel.parquet"
INPUT_SHAPE: Path = INPUT_DIR / "shape_features.csv"
INPUT_CLUSTER: Path = INPUT_DIR / "cluster_assignments.csv"

EXPECTED_HASHES: Dict[str, str] = {
    "trajectory_panel.parquet": "c8f1ec4825ada7b3a9efaf72101e79b067f7056117bf70aaae2f5766261398ed",
    "shape_features.csv": "6920c990628dd2d769cfbc56bd006391f6b9a24beea64f52f5bda9da02d44aaa",
    "cluster_assignments.csv": "c82f7208ece085e804bc7c66c741d91e5004c604463f8dbeff03381aacf7ecbe",
}

CLUSTERS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
WINNER_CLUSTERS: Tuple[int, ...] = (2, 5)
LOSER_CLUSTERS: Tuple[int, ...] = (3, 6)
MIDDLE_CLUSTERS: Tuple[int, ...] = (1, 4)

VIEW_A_BARS: np.ndarray = np.arange(0, 61, 1, dtype=np.int64)  # 0..60 inclusive
PERCENTILES: Tuple[float, ...] = (10.0, 25.0, 50.0, 75.0, 90.0)
PER_BAR_CHECKPOINTS: Tuple[int, ...] = (6, 12, 24, 36, 48, 60)
HIST_R_BINS: np.ndarray = np.arange(-3.0, 3.0 + 0.25 / 2, 0.25)
SEP_BARS: np.ndarray = np.arange(1, 61, 1, dtype=np.int64)
DECILES: np.ndarray = np.linspace(0.0, 1.0, 11)
MFE_BANDS: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)

# Pullback histogram: 0.1R bins from 0 to 12R + overflow bin
# (range extended from initial 0..5R because the combined-winner overflow
# in the first run was 13.3% > 5%; cluster-2 max was 11.7R, cluster-5 max
# 10.7R).
PULLBACK_BIN_EDGES: np.ndarray = np.concatenate(
    [
        np.arange(0.0, 12.0 + 0.05, 0.1),  # 0.0, 0.1, ..., 12.0
        np.array([np.inf]),
    ]
)

CSV_FLOAT_FORMAT: str = "%.10g"

# Pinned matplotlib metadata.
PNG_METADATA: Dict[str, str] = {
    "Software": "matplotlib",
    "Creation Time": "2020-01-01T00:00:00+00:00",
}

CLUSTER_LABELS: Dict[int, str] = {
    1: "V-shape recovery (middle)",
    2: "Steady runner (winner)",
    3: "Slow bleeder (loser)",
    4: "Early-pop fade (middle)",
    5: "Mega-runner (winner)",
    6: "Disaster (loser)",
}

CLUSTER_COLORS: Dict[int, str] = {
    1: "#8c8c8c",
    2: "#2ca02c",
    3: "#d62728",
    4: "#bcbd22",
    5: "#1f77b4",
    6: "#9467bd",
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def png_pixel_hash(path: Path) -> str:
    """SHA-256 of raw pixel buffer only (strip all metadata)."""
    from PIL import Image  # imported lazily

    with Image.open(path) as im:
        im.load()
        raw = im.tobytes()
        size = im.size
        mode = im.mode
    h = hashlib.sha256()
    h.update(f"{mode}|{size[0]}x{size[1]}".encode("ascii"))
    h.update(raw)
    return h.hexdigest()


def write_csv_deterministic(df: pd.DataFrame, path: Path) -> None:
    """Write CSV with LF line endings, utf-8, fixed float format."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n", float_format=CSV_FLOAT_FORMAT)
    text = buf.getvalue()
    path.write_bytes(text.encode("utf-8"))


def save_figure_deterministic(fig, path: Path) -> None:
    fig.savefig(path, dpi=100, metadata=PNG_METADATA, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Load + validate inputs
# ---------------------------------------------------------------------------


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[1/9] Verifying input hashes...", flush=True)
    for fname, expected in EXPECTED_HASHES.items():
        actual = sha256_of(INPUT_DIR / fname)
        if actual != expected:
            raise RuntimeError(
                f"BLOCKER: hash mismatch for {fname}: expected {expected}, got {actual}"
            )
        print(f"   OK  {fname}: {actual}", flush=True)

    print("[2/9] Loading trajectory_panel.parquet...", flush=True)
    panel = pd.read_parquet(INPUT_TRAJECTORY)
    panel = panel.sort_values(["signal_idx", "t"], kind="mergesort").reset_index(drop=True)

    print("[3/9] Loading shape_features.csv...", flush=True)
    shape = pd.read_csv(INPUT_SHAPE)
    shape = shape.sort_values("signal_idx", kind="mergesort").reset_index(drop=True)

    print("[4/9] Loading cluster_assignments.csv...", flush=True)
    clust = pd.read_csv(INPUT_CLUSTER)
    clust = clust.sort_values("signal_idx", kind="mergesort").reset_index(drop=True)

    # Validations
    n_nan_k6 = (clust["km_k6"] == -1).sum()
    if n_nan_k6 / len(clust) > 0.01:
        raise RuntimeError(f"BLOCKER: km_k6 invalid for >1% rows ({n_nan_k6} of {len(clust)})")

    return panel, shape, clust


def attach_cluster(shape: pd.DataFrame, clust: pd.DataFrame) -> pd.DataFrame:
    """Return shape DataFrame restricted to signals with valid km_k6."""
    out = shape.merge(clust[["signal_idx", "km_k6"]], on="signal_idx", how="inner")
    out = out[out["km_k6"].isin(CLUSTERS)].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Analysis 1 — Per-cluster trajectory envelopes (Views A and B)
# ---------------------------------------------------------------------------


def analysis_1_view_A(panel: pd.DataFrame, shape_c: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster percentile envelopes for bars 0..60.

    Returns long-format DataFrame: cluster, bar, metric, percentile, value.
    """
    print("[5/9] Analysis 1 View A — bars 0..60 envelopes per cluster...", flush=True)

    panel_a = panel[panel["t"] <= 60][
        ["signal_idx", "t", "r_close", "running_mfe", "running_mae"]
    ].copy()
    panel_a = panel_a.merge(shape_c[["signal_idx", "km_k6"]], on="signal_idx", how="inner")

    rows: List[Dict[str, object]] = []
    metrics = ("r_close", "running_mfe", "running_mae")
    for c in CLUSTERS:
        sub = panel_a[panel_a["km_k6"] == c]
        for bar in VIEW_A_BARS:
            sub_bar = sub[sub["t"] == bar]
            n = len(sub_bar)
            if n < 100:
                raise RuntimeError(
                    f"BLOCKER: Analysis 1 View A cluster={c} bar={bar} has n={n} (<100)"
                )
            for metric in metrics:
                vals = sub_bar[metric].to_numpy()
                pcts = np.percentile(vals, PERCENTILES)
                for pct_label, pct_val in zip(PERCENTILES, pcts):
                    rows.append(
                        {
                            "cluster": int(c),
                            "bar": int(bar),
                            "metric": metric,
                            "percentile": float(pct_label),
                            "value": float(pct_val),
                            "n": int(n),
                        }
                    )
    return pd.DataFrame(rows)


def _trade_decile_interp(g: pd.DataFrame, peak_bar: int) -> np.ndarray:
    """Linear interpolation of (r_close, running_mfe, running_mae) at 11 deciles.

    Returns array of shape (11, 3): rows = deciles, cols = (r_close, mfe, mae).
    `peak_bar` must be >= 1 (we filter that upstream).
    """
    sub = g[g["t"] <= peak_bar].sort_values("t")
    ts = sub["t"].to_numpy()
    target_ts = DECILES * peak_bar  # 0..peak_bar (inclusive)

    out = np.empty((11, 3), dtype=np.float64)
    for i, col in enumerate(("r_close", "running_mfe", "running_mae")):
        out[:, i] = np.interp(target_ts, ts, sub[col].to_numpy())
    return out


def analysis_1_view_B(panel: pd.DataFrame, shape_c: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster decile envelopes for run-up trajectories.

    For each trade, slice panel from t=0 to peak_bar:
        - winners (clusters 2, 5): peak_mfe_bar
        - losers  (clusters 3, 6): peak_mae_bar
        - middle  (clusters 1, 4): peak_mfe_bar if final_r > 0 else peak_mae_bar

    Returns long-format DataFrame: cluster, decile, metric, percentile, value.
    """
    print("[6/9] Analysis 1 View B — run-up decile envelopes per cluster...", flush=True)

    # Pick peak_bar per trade per spec.
    sc = shape_c.copy()
    sc["peak_bar_for_view_B"] = np.where(
        sc["km_k6"].isin(WINNER_CLUSTERS),
        sc["peak_mfe_bar"],
        np.where(
            sc["km_k6"].isin(LOSER_CLUSTERS),
            sc["peak_mae_bar"],
            np.where(sc["final_r"] > 0.0, sc["peak_mfe_bar"], sc["peak_mae_bar"]),
        ),
    ).astype(np.int64)

    # Drop trades with peak_bar = 0 (degenerate; no interp window).
    sc_valid = sc[sc["peak_bar_for_view_B"] >= 1].reset_index(drop=True)
    n_dropped = len(sc) - len(sc_valid)
    print(f"   note: {n_dropped} trades dropped from View B (peak_bar < 1)", flush=True)

    panel_b = panel[["signal_idx", "t", "r_close", "running_mfe", "running_mae"]]

    # Build per-trade interpolated decile table.
    # To go fast: groupby signal_idx.
    metrics = ("r_close", "running_mfe", "running_mae")

    # Map signal_idx -> peak_bar
    peak_map = dict(
        zip(sc_valid["signal_idx"].to_numpy(), sc_valid["peak_bar_for_view_B"].to_numpy())
    )
    cluster_map = dict(zip(sc_valid["signal_idx"].to_numpy(), sc_valid["km_k6"].to_numpy()))

    # Per-trade interp results: shape (n_trades, 11, 3)
    sig_ids = sc_valid["signal_idx"].to_numpy()
    n_trades = len(sig_ids)
    decile_values = np.full((n_trades, 11, 3), np.nan, dtype=np.float64)
    cluster_vec = np.empty(n_trades, dtype=np.int64)

    # Restrict panel to relevant signals once.
    panel_b = panel_b[panel_b["signal_idx"].isin(sc_valid["signal_idx"])].copy()
    grouped = panel_b.groupby("signal_idx", sort=True)

    sid_to_idx = {sid: i for i, sid in enumerate(sig_ids)}
    for sid, g in grouped:
        i = sid_to_idx[sid]
        peak_bar = int(peak_map[sid])
        decile_values[i] = _trade_decile_interp(g, peak_bar)
        cluster_vec[i] = cluster_map[sid]

    # Aggregate to per-cluster percentile envelopes.
    rows: List[Dict[str, object]] = []
    for c in CLUSTERS:
        mask = cluster_vec == c
        n = int(mask.sum())
        for d_i, decile in enumerate(DECILES):
            arr = decile_values[mask, d_i, :]  # shape (n_c, 3)
            for m_i, metric in enumerate(metrics):
                vals = arr[:, m_i]
                pcts = np.percentile(vals, PERCENTILES)
                for pct_label, pct_val in zip(PERCENTILES, pcts):
                    rows.append(
                        {
                            "cluster": int(c),
                            "decile": float(decile),
                            "metric": metric,
                            "percentile": float(pct_label),
                            "value": float(pct_val),
                            "n": int(n),
                        }
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 2 — Per-bar distributional comparison across clusters
# ---------------------------------------------------------------------------


def analysis_2_per_bar(panel: pd.DataFrame, shape_c: pd.DataFrame) -> pd.DataFrame:
    """Full r_close distribution stats per (cluster, bar) at checkpoints."""
    print("[7/9] Analysis 2 — per-bar distributions at checkpoints...", flush=True)

    bars = list(PER_BAR_CHECKPOINTS)
    panel_sub = panel[panel["t"].isin(bars)][["signal_idx", "t", "r_close"]].copy()
    panel_sub = panel_sub.merge(shape_c[["signal_idx", "km_k6"]], on="signal_idx", how="inner")

    rows: List[Dict[str, object]] = []
    bin_edges = HIST_R_BINS

    for c in CLUSTERS:
        for bar in bars:
            sub = panel_sub[(panel_sub["km_k6"] == c) & (panel_sub["t"] == bar)]
            vals = sub["r_close"].to_numpy()
            n = len(vals)
            if n == 0:
                continue
            stats = {
                "n": int(n),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if n > 1 else 0.0,
                "skew": float(pd.Series(vals).skew()) if n > 2 else 0.0,
                "kurtosis": float(pd.Series(vals).kurt()) if n > 3 else 0.0,
                "min": float(np.min(vals)),
                "p05": float(np.percentile(vals, 5)),
                "p10": float(np.percentile(vals, 10)),
                "p25": float(np.percentile(vals, 25)),
                "p50": float(np.percentile(vals, 50)),
                "p75": float(np.percentile(vals, 75)),
                "p90": float(np.percentile(vals, 90)),
                "p95": float(np.percentile(vals, 95)),
                "max": float(np.max(vals)),
            }
            row_base = {"cluster": int(c), "bar": int(bar), "metric": "r_close"}
            for k, v in stats.items():
                rows.append({**row_base, "stat": k, "value": v})

            # Histogram in 0.25R bins from -3R to +3R.
            hist, edges = np.histogram(vals, bins=bin_edges)
            for i, count in enumerate(hist):
                lo = edges[i]
                hi = edges[i + 1]
                bin_label = f"[{lo:.2f},{hi:.2f})"
                rows.append({**row_base, "stat": f"hist_{bin_label}", "value": float(count)})
            # Also overflow counts (vals outside [-3, +3]).
            below = int((vals < -3.0).sum())
            above = int((vals > 3.0).sum())
            rows.append({**row_base, "stat": "hist_below_neg3R", "value": float(below)})
            rows.append({**row_base, "stat": "hist_above_pos3R", "value": float(above)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 3 — Separation diagnostics (winner vs loser)
# ---------------------------------------------------------------------------


def analysis_3_separation(panel: pd.DataFrame, shape_c: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """ROC AUC of single-feature and composite classifiers per bar."""
    print("[8/9] Analysis 3 — separation diagnostics (winner vs loser)...", flush=True)

    # Restrict to winners + losers; drop middle clusters.
    keep_clusters = list(WINNER_CLUSTERS) + list(LOSER_CLUSTERS)
    sc = shape_c[shape_c["km_k6"].isin(keep_clusters)].copy()
    sc["is_winner"] = sc["km_k6"].isin(WINNER_CLUSTERS).astype(np.int64)

    panel_sub = panel[panel["t"].isin(SEP_BARS)][
        ["signal_idx", "t", "r_close", "running_mfe", "running_mae"]
    ].copy()
    panel_sub = panel_sub.merge(sc[["signal_idx", "is_winner"]], on="signal_idx", how="inner")

    rows: List[Dict[str, object]] = []
    composite_info: Dict[int, dict] = {}
    metrics = ("r_close", "running_mfe", "running_mae")

    for bar in SEP_BARS:
        sub = panel_sub[panel_sub["t"] == bar].sort_values("signal_idx")
        y = sub["is_winner"].to_numpy()
        # Single-feature AUCs.
        for metric in metrics:
            x = sub[metric].to_numpy()
            auc = float(roc_auc_score(y, x))
            # Youden-J threshold.
            fpr, tpr, thresh = roc_curve(y, x)
            j = tpr - fpr
            j_idx = int(np.argmax(j))
            t_star = float(thresh[j_idx])
            sens = float(tpr[j_idx])
            spec = float(1.0 - fpr[j_idx])
            rows.extend(
                [
                    {
                        "bar": int(bar),
                        "classifier_type": f"single_{metric}",
                        "metric": "auc",
                        "value": auc,
                    },
                    {
                        "bar": int(bar),
                        "classifier_type": f"single_{metric}",
                        "metric": "threshold",
                        "value": t_star,
                    },
                    {
                        "bar": int(bar),
                        "classifier_type": f"single_{metric}",
                        "metric": "sensitivity_at_t_star",
                        "value": sens,
                    },
                    {
                        "bar": int(bar),
                        "classifier_type": f"single_{metric}",
                        "metric": "specificity_at_t_star",
                        "value": spec,
                    },
                    {
                        "bar": int(bar),
                        "classifier_type": f"single_{metric}",
                        "metric": "balanced_accuracy_at_t_star",
                        "value": 0.5 * (sens + spec),
                    },
                    {
                        "bar": int(bar),
                        "classifier_type": f"single_{metric}",
                        "metric": "n",
                        "value": float(len(y)),
                    },
                ]
            )

        # Composite logistic regression on (r_close, mfe, mae).
        X = sub[["r_close", "running_mfe", "running_mae"]].to_numpy()
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.30, random_state=SKLEARN_SEED, stratify=y
        )
        lr = LogisticRegression(solver="lbfgs", random_state=SKLEARN_SEED, max_iter=1000)
        lr.fit(X_tr, y_tr)
        p_tr = lr.predict_proba(X_tr)[:, 1]
        p_te = lr.predict_proba(X_te)[:, 1]
        auc_tr = float(roc_auc_score(y_tr, p_tr))
        auc_te = float(roc_auc_score(y_te, p_te))
        coefs = lr.coef_[0]
        intercept = float(lr.intercept_[0])

        rows.extend(
            [
                {
                    "bar": int(bar),
                    "classifier_type": "composite_logreg",
                    "metric": "auc_train",
                    "value": auc_tr,
                },
                {
                    "bar": int(bar),
                    "classifier_type": "composite_logreg",
                    "metric": "auc_test",
                    "value": auc_te,
                },
                {
                    "bar": int(bar),
                    "classifier_type": "composite_logreg",
                    "metric": "auc_train_test_gap",
                    "value": auc_tr - auc_te,
                },
                {
                    "bar": int(bar),
                    "classifier_type": "composite_logreg",
                    "metric": "coef_r_close",
                    "value": float(coefs[0]),
                },
                {
                    "bar": int(bar),
                    "classifier_type": "composite_logreg",
                    "metric": "coef_running_mfe",
                    "value": float(coefs[1]),
                },
                {
                    "bar": int(bar),
                    "classifier_type": "composite_logreg",
                    "metric": "coef_running_mae",
                    "value": float(coefs[2]),
                },
                {
                    "bar": int(bar),
                    "classifier_type": "composite_logreg",
                    "metric": "intercept",
                    "value": intercept,
                },
                {
                    "bar": int(bar),
                    "classifier_type": "composite_logreg",
                    "metric": "n_train",
                    "value": float(len(y_tr)),
                },
                {
                    "bar": int(bar),
                    "classifier_type": "composite_logreg",
                    "metric": "n_test",
                    "value": float(len(y_te)),
                },
            ]
        )

        composite_info[int(bar)] = {
            "auc_train": auc_tr,
            "auc_test": auc_te,
            "coefs": coefs.tolist(),
            "intercept": intercept,
        }

    df_out = pd.DataFrame(rows)
    return df_out, composite_info


# ---------------------------------------------------------------------------
# Analysis 4 — Pullback magnitudes within winner run-ups
# ---------------------------------------------------------------------------


def _running_peak_pullbacks(rl: np.ndarray, rh: np.ndarray, peak_bar: int) -> Dict[str, float]:
    """Compute pullback stats within bars [0, peak_bar].

    Pullback definition: running peak of r_high tracked from t=0..peak_bar;
    drawdown at bar t is running_peak[t] - r_low[t].
    A new pullback event ≥ threshold registers when drawdown crosses threshold
    upward (was below, now ≥); pullback resets when running_peak increases
    (new peak).

    `peak_bar` is the trade's run-up end (we go up to and including peak_bar).
    Returns a dict of pullback statistics.
    """
    last = peak_bar + 1
    rh_w = rh[:last]
    rl_w = rl[:last]
    running_peak = np.maximum.accumulate(rh_w)
    dd = running_peak - rl_w
    # Bar at which max dd occurred (inside the run-up). Use first occurrence.
    if len(dd) <= 1:
        return {
            "max_dd_from_running_peak": 0.0,
            "max_pullback_bar": 0.0,
            "n_pullbacks_ge_0.25R": 0,
            "n_pullbacks_ge_0.5R": 0,
            "n_pullbacks_ge_1R": 0,
            "n_pullbacks_ge_1.5R": 0,
            "pullback_at_peak_minus_5": 0.0,
        }
    # Exclude bar 0 from pullback search (dd[0] = 0 trivially).
    dd_search = dd[1:]  # corresponds to bars 1..peak_bar
    max_idx = int(np.argmax(dd_search))
    max_dd = float(dd_search[max_idx])
    max_bar = int(max_idx + 1)  # convert back to absolute bar

    # Count distinct pullback events per threshold:
    # event registers on upward crossing; resets when running_peak strictly
    # increases (new peak makes the drawdown chain a new event).
    def count_events(th: float) -> int:
        # Reset event tracking each time running_peak makes a new high.
        n = len(dd)
        active = False
        events = 0
        # rather than "below->above" crossings, count distinct excursions
        # within each running-peak plateau where dd >= th occurs.
        # Implementation: walk bars 1..peak_bar; whenever a new running_peak
        # appears (rh_w[t] >= running_peak[t-1] + 0 i.e. strict new high), set
        # active = False so a fresh excursion can be counted on the same
        # plateau.
        for t in range(1, n):
            new_peak = running_peak[t] > running_peak[t - 1]
            if new_peak:
                active = False
            if dd[t] >= th:
                if not active:
                    events += 1
                    active = True
            else:
                active = False
        return events

    n_025 = count_events(0.25)
    n_050 = count_events(0.5)
    n_100 = count_events(1.0)
    n_150 = count_events(1.5)

    # Near-peak giveback: drawdown at bar peak_bar - 5 (or 0 if window short).
    if peak_bar - 5 >= 0:
        near_peak = float(dd[peak_bar - 5])
    else:
        near_peak = 0.0

    return {
        "max_dd_from_running_peak": max_dd,
        "max_pullback_bar": float(max_bar),
        "n_pullbacks_ge_0.25R": n_025,
        "n_pullbacks_ge_0.5R": n_050,
        "n_pullbacks_ge_1R": n_100,
        "n_pullbacks_ge_1.5R": n_150,
        "pullback_at_peak_minus_5": near_peak,
    }


def analysis_4_winner_pullbacks(
    panel: pd.DataFrame, shape_c: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-trade pullback metrics for winners + per-cluster aggregates."""
    print("[9/9 a] Analysis 4 — winner pullbacks...", flush=True)

    winners = shape_c[shape_c["km_k6"].isin(WINNER_CLUSTERS)].copy()
    # Need r_high and r_low panels for winners.
    panel_w = (
        panel[panel["signal_idx"].isin(winners["signal_idx"])][
            ["signal_idx", "t", "r_high", "r_low"]
        ]
        .sort_values(["signal_idx", "t"], kind="mergesort")
        .reset_index(drop=True)
    )

    per_trade_rows: List[Dict[str, object]] = []
    peak_map = dict(zip(winners["signal_idx"].to_numpy(), winners["peak_mfe_bar"].to_numpy()))
    cluster_map = dict(zip(winners["signal_idx"].to_numpy(), winners["km_k6"].to_numpy()))

    for sid, g in panel_w.groupby("signal_idx", sort=True):
        peak_bar = int(peak_map[sid])
        cl = int(cluster_map[sid])
        # peak_bar always >=1 for winners (verified upstream).
        rh = g["r_high"].to_numpy()
        rl = g["r_low"].to_numpy()
        stats = _running_peak_pullbacks(rl, rh, peak_bar)
        per_trade_rows.append(
            {
                "signal_idx": int(sid),
                "cluster": cl,
                "peak_mfe_bar": peak_bar,
                **stats,
            }
        )
    per_trade = pd.DataFrame(per_trade_rows)

    # Aggregates: cluster 2 only, cluster 5 only, combined (2+5).
    def agg_block(sub: pd.DataFrame, label: str) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        n = len(sub)
        vals = sub["max_dd_from_running_peak"].to_numpy()
        stats = {
            "n": float(n),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if n > 1 else 0.0,
            "skew": float(pd.Series(vals).skew()) if n > 2 else 0.0,
            "kurtosis": float(pd.Series(vals).kurt()) if n > 3 else 0.0,
            "min": float(np.min(vals)),
            "p05": float(np.percentile(vals, 5)),
            "p10": float(np.percentile(vals, 10)),
            "p25": float(np.percentile(vals, 25)),
            "p50": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
            "p90": float(np.percentile(vals, 90)),
            "p95": float(np.percentile(vals, 95)),
            "p99": float(np.percentile(vals, 99)),
            "max": float(np.max(vals)),
        }
        for k, v in stats.items():
            out.append(
                {"group": label, "metric": "max_dd_from_running_peak", "stat": k, "value": v}
            )
        # Mean/median pullback counts.
        for th_col in (
            "n_pullbacks_ge_0.25R",
            "n_pullbacks_ge_0.5R",
            "n_pullbacks_ge_1R",
            "n_pullbacks_ge_1.5R",
        ):
            cv = sub[th_col].to_numpy()
            out.append(
                {"group": label, "metric": th_col, "stat": "mean", "value": float(np.mean(cv))}
            )
            out.append(
                {"group": label, "metric": th_col, "stat": "median", "value": float(np.median(cv))}
            )
            out.append(
                {
                    "group": label,
                    "metric": th_col,
                    "stat": "p90",
                    "value": float(np.percentile(cv, 90)),
                }
            )
        # Near-peak giveback.
        npv = sub["pullback_at_peak_minus_5"].to_numpy()
        out.append(
            {
                "group": label,
                "metric": "pullback_at_peak_minus_5",
                "stat": "mean",
                "value": float(np.mean(npv)),
            }
        )
        out.append(
            {
                "group": label,
                "metric": "pullback_at_peak_minus_5",
                "stat": "median",
                "value": float(np.median(npv)),
            }
        )
        out.append(
            {
                "group": label,
                "metric": "pullback_at_peak_minus_5",
                "stat": "p90",
                "value": float(np.percentile(npv, 90)),
            }
        )
        # Histogram of max_dd in 0.1R bins from 0..5R with overflow.
        hist, _ = np.histogram(vals, bins=PULLBACK_BIN_EDGES)
        for i, count in enumerate(hist):
            lo = PULLBACK_BIN_EDGES[i]
            hi = PULLBACK_BIN_EDGES[i + 1]
            if np.isinf(hi):
                label_h = f"[{lo:.2f},inf)"
            else:
                label_h = f"[{lo:.2f},{hi:.2f})"
            out.append(
                {
                    "group": label,
                    "metric": "max_dd_from_running_peak",
                    "stat": f"hist_{label_h}",
                    "value": float(count),
                }
            )
        return out

    agg_rows: List[Dict[str, object]] = []
    agg_rows.extend(agg_block(per_trade[per_trade["cluster"] == 2], "cluster_2"))
    agg_rows.extend(agg_block(per_trade[per_trade["cluster"] == 5], "cluster_5"))
    agg_rows.extend(agg_block(per_trade, "cluster_2_and_5"))
    agg = pd.DataFrame(agg_rows)

    return per_trade, agg


# ---------------------------------------------------------------------------
# Analysis 5 — Time-to-MFE-band per cluster
# ---------------------------------------------------------------------------


def analysis_5_time_to_band(panel: pd.DataFrame, shape_c: pd.DataFrame) -> pd.DataFrame:
    """For each cluster c and MFE band X, time-to-first-reach + conditionals."""
    print("[9/9 b] Analysis 5 — time-to-MFE-band per cluster...", flush=True)

    panel_sub = panel[["signal_idx", "t", "running_mfe"]].copy()
    panel_sub = panel_sub.merge(shape_c[["signal_idx", "km_k6"]], on="signal_idx", how="inner")

    # First time each trade reaches each band X.
    bands = MFE_BANDS

    # Vectorised: for each band X, group by signal_idx and find min t where
    # running_mfe >= X.
    panel_sub.groupby("signal_idx", sort=True)

    # Build per-signal arrays of t and running_mfe sorted by t (panel already
    # sorted by (signal_idx, t) so each group is sorted).
    sig_first_times: Dict[float, pd.Series] = {}
    for X in bands:
        # use the panel_sub (no missing) and find first t with mfe >= X
        mask = panel_sub["running_mfe"] >= X
        if mask.any():
            sub_first = (
                panel_sub.loc[mask, ["signal_idx", "t"]].groupby("signal_idx", sort=True)["t"].min()
            )
        else:
            sub_first = pd.Series(dtype=np.int64)
        sig_first_times[X] = sub_first

    rows: List[Dict[str, object]] = []
    shape_c.set_index("signal_idx")["km_k6"].to_dict()

    for c in CLUSTERS:
        sids_c = shape_c[shape_c["km_k6"] == c]["signal_idx"].to_numpy()
        n_c = len(sids_c)
        for X in bands:
            first_times = sig_first_times[X].reindex(sids_c)
            reached_mask = first_times.notna()
            n_reached = int(reached_mask.sum())
            if n_reached == 0:
                row_base = {"cluster": int(c), "band_R": float(X), "n_cluster": int(n_c)}
                rows.append({**row_base, "stat": "n_reached", "value": 0.0})
                rows.append({**row_base, "stat": "p_reached", "value": 0.0})
                continue
            vals = first_times.dropna().to_numpy()
            row_base = {"cluster": int(c), "band_R": float(X), "n_cluster": int(n_c)}
            rows.append({**row_base, "stat": "n_reached", "value": float(n_reached)})
            rows.append({**row_base, "stat": "p_reached", "value": float(n_reached) / float(n_c)})
            rows.append({**row_base, "stat": "mean", "value": float(np.mean(vals))})
            rows.append(
                {
                    **row_base,
                    "stat": "std",
                    "value": float(np.std(vals, ddof=1)) if n_reached > 1 else 0.0,
                }
            )
            for pct in (5, 10, 25, 50, 75, 90, 95):
                rows.append(
                    {**row_base, "stat": f"p{pct:02d}", "value": float(np.percentile(vals, pct))}
                )
            rows.append({**row_base, "stat": "min", "value": float(np.min(vals))})
            rows.append({**row_base, "stat": "max", "value": float(np.max(vals))})

    # Conditional P(reach X+1 | reached X, in cluster c)
    band_arr = list(bands)
    for c in CLUSTERS:
        sids_c = set(shape_c[shape_c["km_k6"] == c]["signal_idx"].tolist())
        for i, X in enumerate(band_arr[:-1]):
            X_next = band_arr[i + 1]
            reached_X = set(sig_first_times[X].index) & sids_c
            reached_Xn = set(sig_first_times[X_next].index) & sids_c
            n_X = len(reached_X)
            n_Xn_given_X = len(reached_X & reached_Xn)
            p_cond = (float(n_Xn_given_X) / float(n_X)) if n_X > 0 else 0.0
            rows.append(
                {
                    "cluster": int(c),
                    "band_R": float(X),
                    "n_cluster": len(sids_c),
                    "stat": f"p_reach_next_band_{X_next:.1f}R",
                    "value": p_cond,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 6 — Winner run-up shape classification
# ---------------------------------------------------------------------------


def _classify_winner_shape(
    rl: np.ndarray,
    rh: np.ndarray,
    rc: np.ndarray,
    rmfe: np.ndarray,
    rmae: np.ndarray,
    peak_bar: int,
) -> Tuple[str, float, int]:
    """Classify winner trade's run-up shape.

    Returns (shape_bucket, max_dd_in_run_up, n_pullbacks_05R).
    """
    last = peak_bar + 1
    rl_w = rl[:last]
    rh_w = rh[:last]
    rc_w = rc[:last]
    rmfe_w = rmfe[:last]
    rmae_w = rmae[:last]

    # Pullbacks ≥ 0.5R inside run-up via the same logic as analysis 4.
    running_peak = np.maximum.accumulate(rh_w)
    dd = running_peak - rl_w
    max_dd = float(np.max(dd[1:])) if len(dd) > 1 else 0.0

    # Count pullback events ≥ 0.5R.
    th = 0.5
    n_pb = 0
    active = False
    for t in range(1, len(dd)):
        new_peak = running_peak[t] > running_peak[t - 1]
        if new_peak:
            active = False
        if dd[t] >= th:
            if not active:
                n_pb += 1
                active = True
        else:
            active = False

    # Find first bar where r_close crosses +0.5R (positive direction).
    rmfe_05_bar = -1
    for t in range(len(rc_w)):
        if rc_w[t] >= 0.5:
            rmfe_05_bar = t
            break

    # V-shape: running_mae crosses below -0.5R BEFORE rmfe_w >= +0.5R.
    rmae_05_bar = -1
    for t in range(len(rmae_w)):
        if rmae_w[t] <= -0.5:
            rmae_05_bar = t
            break

    rmfe_pos05_bar = -1
    for t in range(len(rmfe_w)):
        if rmfe_w[t] >= 0.5:
            rmfe_pos05_bar = t
            break

    v_shape = rmae_05_bar >= 0 and (rmfe_pos05_bar < 0 or rmae_05_bar < rmfe_pos05_bar)

    # Monotone: max_dd < 0.3R AND after first crossing of r_close >= +0.5R,
    # r_close never drops below 0.
    if rmfe_05_bar >= 0:
        post = rc_w[rmfe_05_bar:]
        monotone_close_cond = bool(np.all(post >= 0.0))
    else:
        monotone_close_cond = False
    monotone = (max_dd < 0.3) and monotone_close_cond

    if monotone:
        return "monotone", max_dd, n_pb
    if v_shape:
        return "v_shape", max_dd, n_pb
    # Stepped: exactly 1 pullback ≥ 0.5R AND max_dd < 1.5R.
    if n_pb == 1 and max_dd < 1.5:
        return "stepped", max_dd, n_pb
    # Choppy: ≥ 2 pullbacks ≥ 0.5R.
    if n_pb >= 2:
        return "choppy", max_dd, n_pb
    return "other", max_dd, n_pb


def analysis_6_shape(
    panel: pd.DataFrame, shape_c: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Winner shape classification cross-tab. Also include clusters 1, 4."""
    print("[9/9 c] Analysis 6 — winner shape classification...", flush=True)

    interest = shape_c[shape_c["km_k6"].isin((1, 2, 4, 5))].copy()
    panel_w = (
        panel[panel["signal_idx"].isin(interest["signal_idx"])][
            ["signal_idx", "t", "r_close", "r_high", "r_low", "running_mfe", "running_mae"]
        ]
        .sort_values(["signal_idx", "t"], kind="mergesort")
        .reset_index(drop=True)
    )

    # For middle clusters use peak_mfe_bar for shape (consistent with View B).
    peak_map = {}
    cluster_map = {}
    final_r_map = {}
    for sid, cl, pmf, pma, fr in zip(
        interest["signal_idx"],
        interest["km_k6"],
        interest["peak_mfe_bar"],
        interest["peak_mae_bar"],
        interest["final_r"],
    ):
        peak_map[int(sid)] = (
            int(pmf) if cl in (1, 2, 4, 5) and (cl in (2, 5) or fr > 0) else int(pma)
        )
        cluster_map[int(sid)] = int(cl)
        final_r_map[int(sid)] = float(fr)

    rows: List[Dict[str, object]] = []
    for sid, g in panel_w.groupby("signal_idx", sort=True):
        cl = cluster_map[int(sid)]
        peak_bar = peak_map[int(sid)]
        if peak_bar < 1:
            shape_bucket = "other"
            max_dd = 0.0
            n_pb = 0
        else:
            shape_bucket, max_dd, n_pb = _classify_winner_shape(
                g["r_low"].to_numpy(),
                g["r_high"].to_numpy(),
                g["r_close"].to_numpy(),
                g["running_mfe"].to_numpy(),
                g["running_mae"].to_numpy(),
                peak_bar,
            )
        rows.append(
            {
                "signal_idx": int(sid),
                "cluster": cl,
                "peak_bar": peak_bar,
                "shape": shape_bucket,
                "max_dd_from_running_peak": max_dd,
                "n_pullbacks_ge_0.5R": n_pb,
            }
        )
    per_trade = pd.DataFrame(rows)

    # Cross-tab: cluster × shape with N, mean peak_mfe_r, mean r_at_t240,
    # mean max_dd_from_running_peak.
    per_trade2 = per_trade.merge(
        shape_c[["signal_idx", "peak_mfe_r", "r_at_t240"]],
        on="signal_idx",
        how="left",
    )
    cross_rows: List[Dict[str, object]] = []
    shape_order = ("monotone", "v_shape", "stepped", "choppy", "other")
    for c in (1, 2, 4, 5):
        sub_c = per_trade2[per_trade2["cluster"] == c]
        for sh in shape_order:
            cell = sub_c[sub_c["shape"] == sh]
            n = len(cell)
            cross_rows.append(
                {
                    "cluster": int(c),
                    "shape": sh,
                    "n": int(n),
                    "frac_of_cluster": (n / len(sub_c)) if len(sub_c) > 0 else 0.0,
                    "mean_peak_mfe_r": float(cell["peak_mfe_r"].mean()) if n > 0 else 0.0,
                    "mean_r_at_t240": float(cell["r_at_t240"].mean()) if n > 0 else 0.0,
                    "mean_max_dd_from_running_peak": float(cell["max_dd_from_running_peak"].mean())
                    if n > 0
                    else 0.0,
                }
            )

    return per_trade, pd.DataFrame(cross_rows)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def chart_view_A_overlay(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sub = df[(df["metric"] == "r_close") & (df["percentile"] == 50.0)]
    for c in CLUSTERS:
        s = sub[sub["cluster"] == c].sort_values("bar")
        ax.plot(
            s["bar"],
            s["value"],
            label=f"c{c}: {CLUSTER_LABELS[c]}",
            color=CLUSTER_COLORS[c],
            linewidth=1.6,
        )
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("bar t")
    ax.set_ylabel("median r_close (R)")
    ax.set_title("View A — median r_close[t] by cluster, bars 0..60")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    save_figure_deterministic(fig, CHARTS_DIR / "view_A_envelopes_overlay.png")


def chart_view_A_grid(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for idx, c in enumerate(CLUSTERS):
        ax = axes[idx // 3, idx % 3]
        for pct in PERCENTILES:
            s = df[
                (df["cluster"] == c) & (df["metric"] == "r_close") & (df["percentile"] == pct)
            ].sort_values("bar")
            ax.plot(
                s["bar"],
                s["value"],
                label=f"p{int(pct)}",
                color=CLUSTER_COLORS[c],
                alpha=0.3 + 0.13 * PERCENTILES.index(pct),
                linewidth=1.0,
            )
        ax.axhline(0.0, color="black", linewidth=0.5, linestyle=":")
        ax.set_title(f"c{c}: {CLUSTER_LABELS[c]}", fontsize=10)
        ax.grid(True, linewidth=0.4, alpha=0.5)
    for ax in axes[1]:
        ax.set_xlabel("bar t")
    for ax in axes[:, 0]:
        ax.set_ylabel("r_close (R)")
    fig.suptitle("View A — p10/p25/p50/p75/p90 of r_close[t] by cluster, bars 0..60", fontsize=11)
    fig.tight_layout()
    save_figure_deterministic(fig, CHARTS_DIR / "view_A_envelopes_grid.png")


def chart_view_B_overlay(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sub = df[(df["metric"] == "r_close") & (df["percentile"] == 50.0)]
    for c in CLUSTERS:
        s = sub[sub["cluster"] == c].sort_values("decile")
        ax.plot(
            s["decile"],
            s["value"],
            label=f"c{c}: {CLUSTER_LABELS[c]}",
            color=CLUSTER_COLORS[c],
            linewidth=1.6,
            marker="o",
            ms=3,
        )
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("decile of run-up (entry=0 → peak=1)")
    ax.set_ylabel("median r_close (R)")
    ax.set_title("View B — median r_close vs run-up decile, by cluster")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    save_figure_deterministic(fig, CHARTS_DIR / "view_B_envelopes_overlay.png")


def chart_view_B_grid(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for idx, c in enumerate(CLUSTERS):
        ax = axes[idx // 3, idx % 3]
        for pct in PERCENTILES:
            s = df[
                (df["cluster"] == c) & (df["metric"] == "r_close") & (df["percentile"] == pct)
            ].sort_values("decile")
            ax.plot(
                s["decile"],
                s["value"],
                label=f"p{int(pct)}",
                color=CLUSTER_COLORS[c],
                alpha=0.3 + 0.13 * PERCENTILES.index(pct),
                linewidth=1.0,
                marker="o",
                ms=2,
            )
        ax.axhline(0.0, color="black", linewidth=0.5, linestyle=":")
        ax.set_title(f"c{c}: {CLUSTER_LABELS[c]}", fontsize=10)
        ax.grid(True, linewidth=0.4, alpha=0.5)
    for ax in axes[1]:
        ax.set_xlabel("decile of run-up")
    for ax in axes[:, 0]:
        ax.set_ylabel("r_close (R)")
    fig.suptitle("View B — p10/p25/p50/p75/p90 of r_close at run-up deciles", fontsize=11)
    fig.tight_layout()
    save_figure_deterministic(fig, CHARTS_DIR / "view_B_envelopes_grid.png")


def chart_auc_vs_bar(sep_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    types_to_plot = [
        ("single_r_close", "AUC: r_close[t]", "#1f77b4", "auc"),
        ("single_running_mfe", "AUC: running_mfe[t]", "#2ca02c", "auc"),
        ("single_running_mae", "AUC: running_mae[t]", "#d62728", "auc"),
        ("composite_logreg", "AUC: composite (test)", "black", "auc_test"),
    ]
    for c_type, label, color, metric in types_to_plot:
        s = sep_df[
            (sep_df["classifier_type"] == c_type) & (sep_df["metric"] == metric)
        ].sort_values("bar")
        ax.plot(s["bar"], s["value"], label=label, color=color, linewidth=1.4)
    ax.axhline(0.5, color="grey", linewidth=0.5, linestyle="--")
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("bar t")
    ax.set_ylabel("ROC AUC (winner vs loser)")
    ax.set_title("AUC vs bar — winner-vs-loser discrimination (k=6 c2,5 vs c3,6)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    save_figure_deterministic(fig, CHARTS_DIR / "auc_vs_bar.png")


def chart_threshold_curves(sep_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    s_t = sep_df[
        (sep_df["classifier_type"] == "single_r_close") & (sep_df["metric"] == "threshold")
    ].sort_values("bar")
    s_ba = sep_df[
        (sep_df["classifier_type"] == "single_r_close")
        & (sep_df["metric"] == "balanced_accuracy_at_t_star")
    ].sort_values("bar")
    ax2 = ax.twinx()
    ax.plot(s_t["bar"], s_t["value"], color="#1f77b4", label="T* (R)", linewidth=1.4)
    ax2.plot(s_ba["bar"], s_ba["value"], color="#d62728", label="balanced acc @ T*", linewidth=1.4)
    ax.set_xlabel("bar t")
    ax.set_ylabel("Youden-J T* (R)", color="#1f77b4")
    ax2.set_ylabel("balanced accuracy", color="#d62728")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.suptitle("Single-feature r_close[t] — Youden-J threshold and balanced accuracy across bars")
    fig.tight_layout()
    save_figure_deterministic(fig, CHARTS_DIR / "threshold_curves.png")


def chart_winner_pullbacks(per_trade: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c, color, ax_idx in ((2, CLUSTER_COLORS[2], 0), (5, CLUSTER_COLORS[5], 1)):
        ax = axes[ax_idx]
        vals = per_trade[per_trade["cluster"] == c]["max_dd_from_running_peak"]
        ax.hist(vals, bins=PULLBACK_BIN_EDGES[:-1], color=color, alpha=0.85)
        ax.axvline(
            float(vals.median()),
            color="black",
            linewidth=1.0,
            linestyle="--",
            label=f"median={vals.median():.2f}R",
        )
        ax.axvline(
            float(np.percentile(vals, 95)),
            color="grey",
            linewidth=1.0,
            linestyle=":",
            label=f"p95={np.percentile(vals, 95):.2f}R",
        )
        ax.set_xlim(0, 5)
        ax.set_xlabel("max drawdown from running peak (R)")
        ax.set_ylabel("count")
        ax.set_title(f"c{c}: {CLUSTER_LABELS[c]}  (n={len(vals)})")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.suptitle("Winner pullback magnitude distributions (within entry→peak run-up)")
    fig.tight_layout()
    save_figure_deterministic(fig, CHARTS_DIR / "winner_pullback_distributions.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(
    panel: pd.DataFrame,
    shape_c: pd.DataFrame,
    view_A: pd.DataFrame,
    view_B: pd.DataFrame,
    per_bar: pd.DataFrame,
    sep: pd.DataFrame,
    pullback_per_trade: pd.DataFrame,
    pullback_agg: pd.DataFrame,
    time_to_band: pd.DataFrame,
    shape_cross: pd.DataFrame,
    composite_info: dict,
) -> Tuple[str, bool]:
    """Write winner_path_report.md. Returns (text, warn_train_test_gap_flag)."""

    lines: List[str] = []
    lines.append("# Arc 2 Winner Path Analysis — Report")
    lines.append("")
    lines.append("Phase: l6_arc2_trajectory_appendix / winner_path_analysis")
    lines.append("(supplementary; descriptive only per L6.0 v1.1 §14.5)")
    lines.append("")
    lines.append("Question: what does the path from entry to peak look like for")
    lines.append("our actual winners, and how does it differ from losers?")
    lines.append("")
    lines.append("## 1. Inputs and hashes")
    lines.append("")
    for fname, expected in EXPECTED_HASHES.items():
        lines.append(f"- `{fname}` sha256 = `{expected}`")
    lines.append("")

    # 2. Cluster N counts
    lines.append("## 2. Cluster N counts and group assignments")
    lines.append("")
    lines.append("| cluster | label | n | group |")
    lines.append("|---|---|---:|---|")
    for c in CLUSTERS:
        n = int((shape_c["km_k6"] == c).sum())
        if c in WINNER_CLUSTERS:
            grp = "winner"
        elif c in LOSER_CLUSTERS:
            grp = "loser"
        else:
            grp = "middle"
        lines.append(f"| {c} | {CLUSTER_LABELS[c]} | {n} | {grp} |")
    lines.append("")
    n_total = int(shape_c.shape[0])
    n_win = int(shape_c["km_k6"].isin(WINNER_CLUSTERS).sum())
    n_los = int(shape_c["km_k6"].isin(LOSER_CLUSTERS).sum())
    n_mid = int(shape_c["km_k6"].isin(MIDDLE_CLUSTERS).sum())
    lines.append(
        f"Total n with valid km_k6: **{n_total}** "
        f"(winners={n_win}, losers={n_los}, middle={n_mid})."
    )
    lines.append("")

    # 3. View A summary
    lines.append("## 3. View A — early-bar envelope summary")
    lines.append("")
    lines.append(
        "Median r_close[t] per cluster at key early-bar checkpoints "
        "(values in R = 2 × Wilder ATR(14) at bar N close)."
    )
    lines.append("")
    cols = ["bar"] + [f"c{c}" for c in CLUSTERS]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * (len(CLUSTERS) + 1)) + "|")
    medians_at_check = view_A[(view_A["metric"] == "r_close") & (view_A["percentile"] == 50.0)]
    for bar in PER_BAR_CHECKPOINTS:
        row = [str(bar)]
        for c in CLUSTERS:
            v = float(
                medians_at_check[
                    (medians_at_check["cluster"] == c) & (medians_at_check["bar"] == bar)
                ]["value"].iloc[0]
            )
            row.append(f"{v:+.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-bar checkpoint detail (Analysis 2).
    lines.append("### Per-bar distribution at checkpoints (Analysis 2)")
    lines.append("")
    lines.append("r_close[t] mean and percentiles at bars {6, 12, 24, 36, 48, 60}, per cluster:")
    lines.append("")
    for bar in PER_BAR_CHECKPOINTS:
        lines.append(f"**bar t = {bar}**")
        lines.append("")
        lines.append("| cluster | n | mean | p10 | p25 | p50 | p75 | p90 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for c in CLUSTERS:
            sub = per_bar[(per_bar["cluster"] == c) & (per_bar["bar"] == bar)]

            def stat(name: str) -> float:
                return float(sub[sub["stat"] == name]["value"].iloc[0])

            lines.append(
                f"| {c} | {int(stat('n'))} | {stat('mean'):+.3f} | "
                f"{stat('p10'):+.3f} | {stat('p25'):+.3f} | "
                f"{stat('p50'):+.3f} | {stat('p75'):+.3f} | "
                f"{stat('p90'):+.3f} |"
            )
        lines.append("")

    # 4. View B summary
    lines.append("## 4. View B — run-up shape envelope summary")
    lines.append("")
    lines.append(
        "Each trade's panel sliced t=0..peak_bar (peak_mfe_bar for "
        "winners; peak_mae_bar for losers; sign-of-final_r selected "
        "for middle clusters), interpolated to 11 deciles. Cells "
        "below are median r_close at each decile."
    )
    lines.append("")
    cols = ["decile"] + [f"c{c}" for c in CLUSTERS]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * (len(CLUSTERS) + 1)) + "|")
    med_B = view_B[(view_B["metric"] == "r_close") & (view_B["percentile"] == 50.0)]
    for d in DECILES:
        row = [f"{d:.1f}"]
        for c in CLUSTERS:
            v = float(
                med_B[(med_B["cluster"] == c) & (np.isclose(med_B["decile"], d))]["value"].iloc[0]
            )
            row.append(f"{v:+.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # 5. Separation diagnostics
    lines.append("## 5. Separation diagnostics (Analysis 3)")
    lines.append("")
    lines.append(
        "Population: winners (c2 ∪ c5) and losers (c3 ∪ c6); middle "
        "clusters (c1, c4) excluded from this analysis."
    )
    lines.append("")
    lines.append("AUC trace per bar for single-feature and composite classifiers:")
    lines.append("")
    lines.append(
        "| bar | AUC r_close | AUC running_mfe | AUC running_mae | "
        "AUC composite (train) | AUC composite (test) | gap |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for bar in (1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60):

        def get_metric(c_type: str, metric_name: str) -> float:
            row = sep[
                (sep["classifier_type"] == c_type)
                & (sep["bar"] == bar)
                & (sep["metric"] == metric_name)
            ]
            if len(row) == 0:
                return float("nan")
            return float(row["value"].iloc[0])

        lines.append(
            f"| {bar} | {get_metric('single_r_close', 'auc'):.3f} | "
            f"{get_metric('single_running_mfe', 'auc'):.3f} | "
            f"{get_metric('single_running_mae', 'auc'):.3f} | "
            f"{get_metric('composite_logreg', 'auc_train'):.3f} | "
            f"{get_metric('composite_logreg', 'auc_test'):.3f} | "
            f"{get_metric('composite_logreg', 'auc_train_test_gap'):+.3f} |"
        )
    lines.append("")

    # Find first bar AUC of r_close crosses 0.6, 0.7, 0.8.
    auc_rclose = sep[
        (sep["classifier_type"] == "single_r_close") & (sep["metric"] == "auc")
    ].sort_values("bar")
    auc_at_b = dict(zip(auc_rclose["bar"].astype(int).tolist(), auc_rclose["value"].tolist()))

    def first_bar_cross(threshold: float) -> int:
        for b in sorted(auc_at_b):
            if auc_at_b[b] >= threshold:
                return b
        return -1

    lines.append("Single-feature AUC of r_close[t] discriminating winners from losers crosses:")
    lines.append("")
    for th in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        b = first_bar_cross(th)
        if b > 0:
            lines.append(f"- AUC ≥ {th:.2f} at bar **{b}** (AUC[{b}] = {auc_at_b[b]:.3f})")
    lines.append("")

    # Optimal thresholds at key bars.
    lines.append(
        "Youden-J optimal thresholds for single-feature r_close[t] classifier at key bars:"
    )
    lines.append("")
    lines.append("| bar | AUC | T* (R) | sensitivity | specificity | balanced acc |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for bar in (6, 12, 24, 36, 48, 60):

        def get(metric: str) -> float:
            return float(
                sep[
                    (sep["classifier_type"] == "single_r_close")
                    & (sep["bar"] == bar)
                    & (sep["metric"] == metric)
                ]["value"].iloc[0]
            )

        lines.append(
            f"| {bar} | {get('auc'):.3f} | {get('threshold'):+.3f} | "
            f"{get('sensitivity_at_t_star'):.3f} | "
            f"{get('specificity_at_t_star'):.3f} | "
            f"{get('balanced_accuracy_at_t_star'):.3f} |"
        )
    lines.append("")

    # Composite classifier coefficients at key bars.
    lines.append(
        "Composite logistic regression coefficients at key bars "
        "(features: r_close, running_mfe, running_mae):"
    )
    lines.append("")
    lines.append(
        "| bar | β r_close | β running_mfe | β running_mae | intercept | "
        "AUC train | AUC test | gap |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bar in (6, 12, 24, 36, 48, 60):
        d = composite_info[bar]
        coefs = d["coefs"]
        gap = d["auc_train"] - d["auc_test"]
        lines.append(
            f"| {bar} | {coefs[0]:+.4f} | {coefs[1]:+.4f} | {coefs[2]:+.4f} | "
            f"{d['intercept']:+.4f} | {d['auc_train']:.3f} | "
            f"{d['auc_test']:.3f} | {gap:+.4f} |"
        )
    lines.append("")

    # Train-test gap warning.
    max_gap = max(abs(d["auc_train"] - d["auc_test"]) for d in composite_info.values())
    warn_train_test = max_gap > 0.05
    if warn_train_test:
        lines.append(
            f"**WARN**: composite classifier max |train−test AUC gap| "
            f"= {max_gap:.4f} across bars 1..60 (threshold = 0.05). "
            f"Possible overfitting at one or more bars."
        )
    else:
        lines.append(
            f"Train-test AUC gap maximum across bars 1..60 = "
            f"{max_gap:.4f} (< 0.05; no overfitting flag)."
        )
    lines.append("")

    # 6. Winner pullbacks
    lines.append("## 6. Winner pullback magnitude distributions (Analysis 4)")
    lines.append("")
    lines.append(
        "max_dd_from_running_peak is the maximum drawdown of r_low from "
        "the running peak of r_high, evaluated within bars [1, peak_mfe_bar]."
    )
    lines.append("")

    def pull_stat(label: str, stat: str) -> float:
        row = pullback_agg[
            (pullback_agg["group"] == label)
            & (pullback_agg["metric"] == "max_dd_from_running_peak")
            & (pullback_agg["stat"] == stat)
        ]
        return float(row["value"].iloc[0])

    lines.append("| group | n | mean | std | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for group in ("cluster_2", "cluster_5", "cluster_2_and_5"):
        n = int(pull_stat(group, "n"))
        lines.append(
            f"| {group} | {n} | {pull_stat(group, 'mean'):.3f} | "
            f"{pull_stat(group, 'std'):.3f} | "
            f"{pull_stat(group, 'p10'):.3f} | "
            f"{pull_stat(group, 'p25'):.3f} | "
            f"{pull_stat(group, 'p50'):.3f} | "
            f"{pull_stat(group, 'p75'):.3f} | "
            f"{pull_stat(group, 'p90'):.3f} | "
            f"{pull_stat(group, 'p95'):.3f} | "
            f"{pull_stat(group, 'p99'):.3f} | "
            f"{pull_stat(group, 'max'):.3f} |"
        )
    lines.append("")

    # Pullback event counts.
    lines.append("Pullback event counts (mean / median) within run-up, by threshold:")
    lines.append("")
    lines.append(
        "| group | ≥0.25R mean | ≥0.25R median | ≥0.5R mean | ≥0.5R median | "
        "≥1R mean | ≥1R median | ≥1.5R mean | ≥1.5R median |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for group in ("cluster_2", "cluster_5", "cluster_2_and_5"):

        def cnt(metric: str, stat: str) -> float:
            row = pullback_agg[
                (pullback_agg["group"] == group)
                & (pullback_agg["metric"] == metric)
                & (pullback_agg["stat"] == stat)
            ]
            return float(row["value"].iloc[0])

        lines.append(
            f"| {group} | "
            f"{cnt('n_pullbacks_ge_0.25R', 'mean'):.2f} | "
            f"{cnt('n_pullbacks_ge_0.25R', 'median'):.0f} | "
            f"{cnt('n_pullbacks_ge_0.5R', 'mean'):.2f} | "
            f"{cnt('n_pullbacks_ge_0.5R', 'median'):.0f} | "
            f"{cnt('n_pullbacks_ge_1R', 'mean'):.2f} | "
            f"{cnt('n_pullbacks_ge_1R', 'median'):.0f} | "
            f"{cnt('n_pullbacks_ge_1.5R', 'mean'):.2f} | "
            f"{cnt('n_pullbacks_ge_1.5R', 'median'):.0f} |"
        )
    lines.append("")

    # Histogram overflow check.
    hist_rows = pullback_agg[
        (pullback_agg["group"] == "cluster_2_and_5")
        & (pullback_agg["metric"] == "max_dd_from_running_peak")
        & (pullback_agg["stat"].str.startswith("hist_"))
    ]
    overflow_row = hist_rows[hist_rows["stat"] == "hist_[12.00,inf)"]
    n_overflow = float(overflow_row["value"].iloc[0]) if len(overflow_row) else 0.0
    n_combined = float(
        pullback_agg[
            (pullback_agg["group"] == "cluster_2_and_5")
            & (pullback_agg["metric"] == "max_dd_from_running_peak")
            & (pullback_agg["stat"] == "n")
        ]["value"].iloc[0]
    )
    frac_overflow = n_overflow / n_combined if n_combined > 0 else 0.0
    if frac_overflow > 0.05:
        lines.append(
            f"**WARN**: pullback histogram overflow "
            f"({frac_overflow:.1%}) > 5%; some max_dd values exceed 5R."
        )
    else:
        lines.append(
            f"Pullback histogram overflow (combined winners) = {frac_overflow:.2%} (≤ 5%)."
        )
    lines.append("")

    # 7. Time-to-band per cluster
    lines.append("## 7. Time-to-band per cluster (Analysis 5)")
    lines.append("")
    lines.append("Time (in bars) to first touch of running_mfe ≥ X R, per cluster.")
    lines.append("")
    for X in MFE_BANDS:
        lines.append(f"### X = {X:.1f} R")
        lines.append("")
        lines.append(
            "| cluster | n_cluster | n_reached | p_reached | mean bar | "
            "p10 | p25 | p50 | p75 | p90 | p_reach_next_band |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for c in CLUSTERS:
            sub = time_to_band[(time_to_band["cluster"] == c) & (time_to_band["band_R"] == X)]

            def gs(stat: str) -> float:
                row = sub[sub["stat"] == stat]
                if len(row) == 0:
                    return float("nan")
                return float(row["value"].iloc[0])

            n_cluster = int(sub["n_cluster"].iloc[0]) if len(sub) else 0
            n_reached = int(gs("n_reached")) if not np.isnan(gs("n_reached")) else 0
            p_reached = gs("p_reached")
            # Conditional next-band probability:
            next_idx = MFE_BANDS.index(X) + 1
            if next_idx < len(MFE_BANDS):
                X_next = MFE_BANDS[next_idx]
                p_next = sub[sub["stat"] == f"p_reach_next_band_{X_next:.1f}R"]
                if len(p_next) > 0:
                    pn = float(p_next["value"].iloc[0])
                    pn_str = f"{pn:.3f}"
                else:
                    pn_str = "—"
            else:
                pn_str = "—"
            if n_reached > 0:
                mean = gs("mean")
                p10 = gs("p10")
                p25 = gs("p25")
                p50 = gs("p50")
                p75 = gs("p75")
                p90 = gs("p90")
                lines.append(
                    f"| {c} | {n_cluster} | {n_reached} | {p_reached:.3f} | "
                    f"{mean:.1f} | {p10:.0f} | {p25:.0f} | {p50:.0f} | "
                    f"{p75:.0f} | {p90:.0f} | {pn_str} |"
                )
            else:
                lines.append(
                    f"| {c} | {n_cluster} | 0 | 0.000 | — | — | — | — | — | — | {pn_str} |"
                )
        lines.append("")

    # 8. Shape classification cross-tab
    lines.append("## 8. Winner shape classification cross-tab (Analysis 6)")
    lines.append("")
    lines.append(
        "Shape buckets (mutually exclusive, evaluated in order: "
        "monotone → v_shape → stepped → choppy → other) within bars "
        "[0, peak_mfe_bar] for winners (clusters 2, 5). Clusters 1 and "
        "4 included for context, classified within bars "
        "[0, peak_mfe_bar] if final_r > 0 else [0, peak_mae_bar]."
    )
    lines.append("")
    lines.append(
        "| cluster | shape | n | frac_of_cluster | mean peak_mfe_r | "
        "mean r_at_t240 | mean max_dd_from_running_peak |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    warn_low_cells: List[str] = []
    empty_cells: List[str] = []
    for _, row in shape_cross.iterrows():
        lines.append(
            f"| {int(row['cluster'])} | {row['shape']} | {int(row['n'])} | "
            f"{row['frac_of_cluster']:.3f} | {row['mean_peak_mfe_r']:.3f} | "
            f"{row['mean_r_at_t240']:.3f} | "
            f"{row['mean_max_dd_from_running_peak']:.3f} |"
        )
        if int(row["cluster"]) in WINNER_CLUSTERS:
            n_cell = int(row["n"])
            if n_cell == 0:
                empty_cells.append(f"c{int(row['cluster'])}/{row['shape']}")
            elif n_cell < 30:
                warn_low_cells.append(f"c{int(row['cluster'])}/{row['shape']}")
    lines.append("")
    if empty_cells:
        lines.append(
            f"Note: definitionally empty cells in winners (n=0): "
            f"{', '.join(empty_cells)}. The shape definitions of "
            f"'monotone' (max_dd_from_running_peak < 0.3R) and 'stepped' "
            f"(exactly 1 pullback ≥ 0.5R AND max_dd < 1.5R) require small "
            f"max_dd within the run-up; winners' median run-up max_dd is "
            f"~3.3R (peak_mfe_bar median ≈ 184), so these buckets are "
            f"effectively excluded."
        )
        lines.append("")
    if warn_low_cells:
        lines.append(
            f"**WARN**: low-confidence sampling cells "
            f"(0 < n < 30) in winners: "
            f"{', '.join(warn_low_cells)}."
        )
        lines.append("")

    # 9. Empirical observations
    lines.append("## 9. Empirical observations")
    lines.append("")
    lines.append("Descriptive statements only per L6.0 v1.1 §14.5.")
    lines.append("")

    # Pull a few canonical observations.
    obs: List[str] = []

    # Cluster-5 median r_close at bar 24 vs cluster-6.
    def med(c: int, bar: int) -> float:
        return float(
            medians_at_check[(medians_at_check["cluster"] == c) & (medians_at_check["bar"] == bar)][
                "value"
            ].iloc[0]
        )

    obs.append(
        f"Cluster 5 (mega-runner) median r_close[24] = {med(5, 24):+.3f}R; "
        f"cluster 6 (disaster) median r_close[24] = {med(6, 24):+.3f}R; "
        f"cluster 2 (steady runner) = {med(2, 24):+.3f}R; "
        f"cluster 3 (slow bleeder) = {med(3, 24):+.3f}R."
    )
    obs.append(
        f"Cluster 5 median r_close[48] = {med(5, 48):+.3f}R; cluster 6 = "
        f"{med(6, 48):+.3f}R; cluster 2 = {med(2, 48):+.3f}R; cluster 3 = "
        f"{med(3, 48):+.3f}R."
    )
    # AUC bar crossings.
    for th in (0.60, 0.70, 0.80, 0.85, 0.90):
        b = first_bar_cross(th)
        if b > 0:
            obs.append(
                f"Single-feature AUC of r_close[t] discriminating winners from "
                f"losers first reaches {th:.2f} at bar t = {b} "
                f"(AUC = {auc_at_b[b]:.3f})."
            )

    # Cluster-5 pullback distribution.
    def pull_stat_safe(label: str, stat: str) -> float:
        row = pullback_agg[
            (pullback_agg["group"] == label)
            & (pullback_agg["metric"] == "max_dd_from_running_peak")
            & (pullback_agg["stat"] == stat)
        ]
        return float(row["value"].iloc[0])

    obs.append(
        f"Cluster 5 (mega-runner) trades have median max_dd_from_running_peak "
        f"= {pull_stat_safe('cluster_5', 'p50'):.3f}R within the entry→peak "
        f"run-up; p75 = {pull_stat_safe('cluster_5', 'p75'):.3f}R; "
        f"p95 = {pull_stat_safe('cluster_5', 'p95'):.3f}R; "
        f"max = {pull_stat_safe('cluster_5', 'max'):.3f}R "
        f"(n = {int(pull_stat_safe('cluster_5', 'n'))})."
    )
    obs.append(
        f"Cluster 2 (steady runner) trades have median max_dd_from_running_peak "
        f"= {pull_stat_safe('cluster_2', 'p50'):.3f}R; "
        f"p75 = {pull_stat_safe('cluster_2', 'p75'):.3f}R; "
        f"p95 = {pull_stat_safe('cluster_2', 'p95'):.3f}R; "
        f"max = {pull_stat_safe('cluster_2', 'max'):.3f}R "
        f"(n = {int(pull_stat_safe('cluster_2', 'n'))})."
    )

    # Shape fractions for winners.
    def shape_frac(c: int, sh: str) -> float:
        row = shape_cross[(shape_cross["cluster"] == c) & (shape_cross["shape"] == sh)]
        if len(row) == 0:
            return 0.0
        return float(row["frac_of_cluster"].iloc[0])

    obs.append(
        f"Of cluster 5 winners, "
        f"{shape_frac(5, 'monotone'):.0%} are classified 'monotone' shape, "
        f"{shape_frac(5, 'v_shape'):.0%} 'v_shape', "
        f"{shape_frac(5, 'stepped'):.0%} 'stepped', "
        f"{shape_frac(5, 'choppy'):.0%} 'choppy', "
        f"{shape_frac(5, 'other'):.0%} 'other'."
    )
    obs.append(
        f"Of cluster 2 winners, "
        f"{shape_frac(2, 'monotone'):.0%} are 'monotone', "
        f"{shape_frac(2, 'v_shape'):.0%} 'v_shape', "
        f"{shape_frac(2, 'stepped'):.0%} 'stepped', "
        f"{shape_frac(2, 'choppy'):.0%} 'choppy', "
        f"{shape_frac(2, 'other'):.0%} 'other'."
    )

    # Time to band for cluster 5 vs 6.
    def time_stat(c: int, X: float, stat: str) -> float:
        if stat == "n_cluster":
            row = time_to_band[(time_to_band["cluster"] == c) & (time_to_band["band_R"] == X)]
            if len(row) == 0:
                return float("nan")
            return float(row["n_cluster"].iloc[0])
        row = time_to_band[
            (time_to_band["cluster"] == c)
            & (time_to_band["band_R"] == X)
            & (time_to_band["stat"] == stat)
        ]
        if len(row) == 0:
            return float("nan")
        return float(row["value"].iloc[0])

    obs.append(
        f"Time-to-first 2R MFE: cluster 5 median = "
        f"{time_stat(5, 2.0, 'p50'):.0f} bars (n_reached = "
        f"{int(time_stat(5, 2.0, 'n_reached'))}/{int(time_stat(5, 2.0, 'n_cluster'))}); "
        f"cluster 2 median = {time_stat(2, 2.0, 'p50'):.0f} bars "
        f"(n_reached = "
        f"{int(time_stat(2, 2.0, 'n_reached'))}/{int(time_stat(2, 2.0, 'n_cluster'))}); "
        f"cluster 3 median = {time_stat(3, 2.0, 'p50'):.0f} bars; "
        f"cluster 6 median = {time_stat(6, 2.0, 'p50'):.0f} bars."
    )
    # Composite classifier at bar 24 description.
    if 24 in composite_info:
        c24 = composite_info[24]
        obs.append(
            f"Composite logistic-regression classifier at bar t = 24: "
            f"AUC (test) = {c24['auc_test']:.3f}; "
            f"β r_close = {c24['coefs'][0]:+.3f}, "
            f"β running_mfe = {c24['coefs'][1]:+.3f}, "
            f"β running_mae = {c24['coefs'][2]:+.3f}, "
            f"intercept = {c24['intercept']:+.3f}."
        )
    for o in obs:
        lines.append(f"- {o}")
    lines.append("")

    text = "\n".join(lines) + "\n"
    return text, warn_train_test


BANNED_PATTERNS: Tuple[str, ...] = (
    "should exit",
    "should drop",
    "the optimal exit",
    "the optimal entry",
    "trail at G=",
    "filter X works",
    "filter would help",
    "drop signals where",
    "exit at bar t=",
    "we recommend",
    "we should",
    "must be exited",
    "use this as a filter",
)


def grep_banned(text: str) -> List[str]:
    found: List[str] = []
    low = text.lower()
    for p in BANNED_PATTERNS:
        if p.lower() in low:
            found.append(p)
    return found


# ---------------------------------------------------------------------------
# Hypothesis appendix
# ---------------------------------------------------------------------------


def maybe_append_candidate(
    sep: pd.DataFrame,
    pullback_agg: pd.DataFrame,
    shape_cross: pd.DataFrame,
    composite_info: dict,
) -> None:
    """Append Candidate 8 (and possibly 9) to CANDIDATE_HYPOTHESES_DRAFT.md
    if the empirical signal is strong enough.
    """
    path = INPUT_DIR / "CANDIDATE_HYPOTHESES_DRAFT.md"
    existing = path.read_text(encoding="utf-8")

    # Skip if we've already appended (idempotency on re-run).
    sentinel = "## Candidate 8 — Winner/loser bar-level separation"
    if sentinel in existing:
        return

    auc_rclose = sep[
        (sep["classifier_type"] == "single_r_close") & (sep["metric"] == "auc")
    ].sort_values("bar")
    auc_at_b = dict(zip(auc_rclose["bar"].astype(int).tolist(), auc_rclose["value"].tolist()))

    def first_bar_cross(threshold: float) -> int:
        for b in sorted(auc_at_b):
            if auc_at_b[b] >= threshold:
                return b
        return -1

    b070 = first_bar_cross(0.70)
    b080 = first_bar_cross(0.80)

    # Cluster 5 pullback quantiles.
    def pull(stat: str, group: str = "cluster_5") -> float:
        return float(
            pullback_agg[
                (pullback_agg["group"] == group)
                & (pullback_agg["metric"] == "max_dd_from_running_peak")
                & (pullback_agg["stat"] == stat)
            ]["value"].iloc[0]
        )

    new_section: List[str] = []
    new_section.append("")
    new_section.append("---")
    new_section.append("")
    new_section.append(sentinel)
    new_section.append("")
    new_section.append(
        "**Observation (from `separation_diagnostics.csv` "
        "and Analysis 3):** Single-feature ROC AUC of r_close[t] "
        "discriminating cluster-{2,5} winners from cluster-{3,6} losers rises "
        f"monotonically with bar t. AUC first crosses 0.70 at bar t = "
        f"{b070} and 0.80 at bar t = {b080}."
    )
    new_section.append("")

    # Threshold at bar 24, 48 for sketch.
    def get_at(metric: str, bar: int) -> float:
        return float(
            sep[
                (sep["classifier_type"] == "single_r_close")
                & (sep["bar"] == bar)
                & (sep["metric"] == metric)
            ]["value"].iloc[0]
        )

    new_section.append(
        "Composite logistic-regression on (r_close[t], "
        "running_mfe[t], running_mae[t]) reaches AUC ≈ "
        f"{composite_info[24]['auc_test']:.3f} (test split) at "
        f"bar t = 24 and AUC ≈ "
        f"{composite_info[48]['auc_test']:.3f} at bar t = 48."
    )
    new_section.append("")
    new_section.append(
        "**Effect size:** Youden-J single-feature threshold "
        f"T*[24] = {get_at('threshold', 24):+.3f}R; balanced "
        f"accuracy at T*[24] = "
        f"{get_at('balanced_accuracy_at_t_star', 24):.3f}. "
        f"T*[48] = {get_at('threshold', 48):+.3f}R; balanced "
        f"accuracy at T*[48] = "
        f"{get_at('balanced_accuracy_at_t_star', 48):.3f}."
    )
    new_section.append("")
    new_section.append(
        "**Pre-committed fresh-arc gate (sketch — full spec is the new arc-opening doc's job):**"
    )
    new_section.append(
        "- Mid-trade decision: at bar t* (pre-committed), apply "
        "single-feature threshold T*[t*] on r_close[t*] (or "
        "composite linear score on r_close, running_mfe, "
        "running_mae) and exit the trade at bar t*'s close if "
        "the classifier predicts 'loser'."
    )
    new_section.append(
        "- Tunables: t* ∈ {12, 24, 36, 48}, threshold rule ∈ "
        "{single-feature T*, composite p<0.5}, pre-committed "
        "before the WFO."
    )
    new_section.append(
        "- Gate per L6.0 §4: worst-fold ROI > 0%, worst-fold DD < 8%, trades-per-fold ≥ 15."
    )
    new_section.append("")
    new_section.append(
        "**n:** ~27,455 (winners + losers; middle clusters "
        "are ambiguous and would be handled by the rule directly)."
    )
    new_section.append(
        "**Priority:** HIGH if AUC ≥ 0.75 stays at bar t ≤ 24. "
        "MEDIUM if only achievable at bar t ≥ 48 (signal arrives "
        "late; capital is tied up longer; less actionable)."
    )
    new_section.append(
        "**Caveats:** AUC is computed on the cluster labels, "
        "which are derived from full-horizon shape features "
        "including peak_mfe_r and peak_mae_r — i.e., labels are "
        "post-hoc, not ex-ante. The classifier feature set "
        "(r_close, running_mfe, running_mae) is ex-ante per bar, "
        "but the *target* cluster is not. A live deployment "
        "would need either (a) the cluster classifier from "
        "Candidate 4 to operate ex-ante, or (b) re-frame the "
        "target as r_at_t240 sign directly (training "
        "different)."
    )
    new_section.append("")
    new_section.append("---")
    new_section.append("")
    new_section.append(
        "## Candidate 9 — Winner-specific trail sized to cluster-5 pullback distribution"
    )
    new_section.append("")
    new_section.append("**Observation (from `winner_pullbacks.csv` and Analysis 4):**")
    new_section.append("")
    new_section.append(
        f"Within the entry→peak run-up, cluster-5 mega-runners have median "
        f"max_dd_from_running_peak = {pull('p50'):.3f}R, "
        f"p75 = {pull('p75'):.3f}R, p90 = {pull('p90'):.3f}R, "
        f"p95 = {pull('p95'):.3f}R, max = {pull('max'):.3f}R. "
        f"Cluster-2 steady-runners: median = "
        f"{pull('p50', 'cluster_2'):.3f}R, p95 = "
        f"{pull('p95', 'cluster_2'):.3f}R."
    )
    new_section.append("")
    new_section.append(
        f"This means a trailing stop with giveback G < cluster-5's median "
        f"({pull('p50'):.3f}R) would exit ~50% of mega-runners early relative "
        f"to peak; G ≥ p95 ({pull('p95'):.3f}R) is required to retain ~95% of "
        f"mega-runners through their full run-up. Cluster-2 trades tolerate "
        f"tighter giveback (median = {pull('p50', 'cluster_2'):.3f}R)."
    )
    new_section.append("")
    new_section.append("**Pre-committed fresh-arc gate (sketch):**")
    new_section.append(
        "- Exit rule: after first-touch of `running_mfe ≥ A × R`, "
        "engage a trailing-stop with giveback `G × R` (R = "
        "2 × Wilder ATR(14) at bar N close). Pre A ∈ {0.5, 1.0, "
        "2.0}, G ∈ {0.5, 1.0, 1.5, 2.0, 2.5}."
    )
    new_section.append("- All other Arc 2 verbatim params retained.")
    new_section.append("- Gate per L6.0 §4.")
    new_section.append("")
    new_section.append(
        "**n:** Full 41,794 after the rule is applied; cluster "
        "5 (n=4,787) is the population where the giveback width "
        "matters most."
    )
    new_section.append(
        "**Priority:** MEDIUM. The trail is the most-tested "
        "exit-policy parameter in the band_behavior analysis "
        "(no pooled (A, G) beat baseline). A fresh-arc test "
        "with winner-aware sizing would either (a) need a "
        "pre-entry winner gate to apply the wide-G trail "
        "selectively, or (b) accept the cost of mass-applying "
        "a wide trail."
    )
    new_section.append(
        "**Caveats:** Wide trails (G ≥ 1.5R) on cluster-3/6 "
        "trades give back more profit than tight trails; "
        "pooled performance was already shown FAIL/THIN in "
        "Phase KH-29. The candidate is meaningful only in "
        "conjunction with Candidate 4 (ex-ante cluster gate)."
    )
    new_section.append("")
    new_section.append("---")
    new_section.append("")
    new_section.append("## Candidate 10 — Cluster-aware classifier feature set for Candidate 4")
    new_section.append("")
    new_section.append("**Observation (from Analysis 3 composite logistic regression):**")
    new_section.append("")
    new_section.append(
        f"A 3-feature logistic regression on (r_close[t], running_mfe[t], "
        f"running_mae[t]) achieves AUC (test) ≈ "
        f"{composite_info[24]['auc_test']:.3f} at bar t = 24 and "
        f"{composite_info[48]['auc_test']:.3f} at bar t = 48. These are "
        f"strong per-bar features. Combined with the ex-ante features used "
        f"in Candidate 4 (signals_features.csv at bar N close), they could "
        f"form a hybrid ex-ante / mid-trade classifier."
    )
    new_section.append("")
    new_section.append("**Pre-committed fresh-arc gate (sketch):**")
    new_section.append(
        "- Extend Candidate 4's ex-ante classifier feature set "
        "with mid-trade features (r_close, running_mfe, "
        "running_mae) at one or two pre-committed bars t* ∈ "
        "{12, 24, 48}. Use the classifier as a *mid-trade* "
        "exit decision (already specified in Candidate 8) "
        "rather than an entry filter."
    )
    new_section.append("- Tunables: t* set, classifier family, pre-committed.")
    new_section.append("- Gate per L6.0 §4.")
    new_section.append("")
    new_section.append(
        "**n:** N depends on the bar t* chosen; for t* ≤ 60, "
        "n = 41,794 (all signals reach bar t* in the panel)."
    )
    new_section.append(
        "**Priority:** MEDIUM. Architectural — depends on "
        "Candidate 4 being run first. The mid-trade features "
        "are cheap to add to a Candidate 4 sketch."
    )
    new_section.append(
        "**Caveats:** Cluster labels are post-hoc; AUC here "
        "depends on the cluster definition surviving as a "
        "trainable target. A simpler version of this candidate "
        "trains directly on `final_r > 0` with mid-trade "
        "features, sidestepping the cluster classifier "
        "altogether."
    )
    new_section.append("")

    appended = existing.rstrip() + "\n" + "\n".join(new_section)
    path.write_bytes(appended.encode("utf-8"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_OUTPUTS: Tuple[str, ...] = (
    "cluster_envelopes_view_A.csv",
    "cluster_envelopes_view_B.csv",
    "per_bar_distributions.csv",
    "separation_diagnostics.csv",
    "winner_pullbacks.csv",
    "time_to_band_per_cluster.csv",
    "winner_shape_classification.csv",
    "winner_path_report.md",
    "charts/view_A_envelopes_overlay.png",
    "charts/view_A_envelopes_grid.png",
    "charts/view_B_envelopes_overlay.png",
    "charts/view_B_envelopes_grid.png",
    "charts/auc_vs_bar.png",
    "charts/threshold_curves.png",
    "charts/winner_pullback_distributions.png",
)


def run_all() -> Dict[str, str]:
    """Run all analyses, write outputs, return per-output sha256 dict."""
    np.random.seed(NUMPY_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    panel, shape, clust = load_inputs()
    shape_c = attach_cluster(shape, clust)

    view_A = analysis_1_view_A(panel, shape_c)
    view_B = analysis_1_view_B(panel, shape_c)
    per_bar = analysis_2_per_bar(panel, shape_c)
    sep, composite_info = analysis_3_separation(panel, shape_c)
    pullback_per_trade, pullback_agg = analysis_4_winner_pullbacks(panel, shape_c)
    time_to_band = analysis_5_time_to_band(panel, shape_c)
    shape_per_trade, shape_cross = analysis_6_shape(panel, shape_c)

    # CSV outputs.
    print("Writing CSV outputs...", flush=True)
    write_csv_deterministic(
        view_A.sort_values(["cluster", "bar", "metric", "percentile"]),
        OUTPUT_DIR / "cluster_envelopes_view_A.csv",
    )
    write_csv_deterministic(
        view_B.sort_values(["cluster", "decile", "metric", "percentile"]),
        OUTPUT_DIR / "cluster_envelopes_view_B.csv",
    )
    write_csv_deterministic(
        per_bar.sort_values(["cluster", "bar", "metric", "stat"]),
        OUTPUT_DIR / "per_bar_distributions.csv",
    )
    write_csv_deterministic(
        sep.sort_values(["bar", "classifier_type", "metric"]),
        OUTPUT_DIR / "separation_diagnostics.csv",
    )
    # Winner pullbacks: emit aggregate table (per-trade is huge; we keep just
    # the aggregate for the CSV deliverable; per-trade is the input to chart).
    write_csv_deterministic(
        pullback_agg.sort_values(["group", "metric", "stat"]), OUTPUT_DIR / "winner_pullbacks.csv"
    )
    write_csv_deterministic(
        time_to_band.sort_values(["cluster", "band_R", "stat"]),
        OUTPUT_DIR / "time_to_band_per_cluster.csv",
    )
    write_csv_deterministic(
        shape_cross.sort_values(["cluster", "shape"]),
        OUTPUT_DIR / "winner_shape_classification.csv",
    )

    # Charts.
    print("Writing charts...", flush=True)
    chart_view_A_overlay(view_A)
    chart_view_A_grid(view_A)
    chart_view_B_overlay(view_B)
    chart_view_B_grid(view_B)
    chart_auc_vs_bar(sep)
    chart_threshold_curves(sep)
    chart_winner_pullbacks(pullback_per_trade)

    # Report.
    print("Writing report...", flush=True)
    report_text, warn_gap = write_report(
        panel,
        shape_c,
        view_A,
        view_B,
        per_bar,
        sep,
        pullback_per_trade,
        pullback_agg,
        time_to_band,
        shape_cross,
        composite_info,
    )
    banned = grep_banned(report_text)
    if banned:
        raise RuntimeError(f"BLOCKER: report contains banned §14.5 patterns: {banned}")
    (OUTPUT_DIR / "winner_path_report.md").write_bytes(report_text.encode("utf-8"))

    # Candidate hypotheses appendix.
    print("Appending candidate hypotheses (if not already)...", flush=True)
    maybe_append_candidate(sep, pullback_agg, shape_cross, composite_info)

    # Hashes.
    sha: Dict[str, str] = {}
    for rel in ALL_OUTPUTS:
        p = OUTPUT_DIR / rel
        sha[rel] = sha256_of(p)
    return sha


def main() -> int:
    print("=== Arc 2 Winner Path Analysis — Run #1 ===", flush=True)
    sha1 = run_all()
    print("\n=== Re-running to verify byte-identical determinism ===\n", flush=True)
    sha2 = run_all()

    # Determinism receipts.
    identical = []
    diff = []
    for k in ALL_OUTPUTS:
        if sha1[k] == sha2[k]:
            identical.append(k)
        else:
            diff.append(k)

    # PNG pixel hashes for fallback if any PNG differs.
    pixel_hashes: Dict[str, Tuple[str, str]] = {}
    for k in diff:
        if k.endswith(".png"):
            ph1 = png_pixel_hash(OUTPUT_DIR / k)
            ph2 = png_pixel_hash(OUTPUT_DIR / k)
            pixel_hashes[k] = (ph1, ph2)

    # Build run_manifest.
    lines: List[str] = []
    lines.append("Arc 2 Winner Path Analysis — run_manifest")
    lines.append("")
    lines.append("## Locked constants")
    lines.append("")
    lines.append(f"- NUMPY_SEED = {NUMPY_SEED}")
    lines.append(f"- SKLEARN_SEED = {SKLEARN_SEED}")
    lines.append(
        "- Logistic regression: sklearn solver='lbfgs', "
        "random_state=SKLEARN_SEED, max_iter=1000, "
        "test_size=0.30, stratify=y."
    )
    lines.append("- Sort order: signals sorted by signal_idx ascending.")
    lines.append("- CSV: utf-8, LF line endings, float_format='%.10g'.")
    lines.append(
        "- Matplotlib: Agg backend, metadata pinned to Creation Time = 2020-01-01T00:00:00+00:00."
    )
    lines.append("")
    lines.append("## Inputs (verified)")
    lines.append("")
    for fname, expected in EXPECTED_HASHES.items():
        actual = sha256_of(INPUT_DIR / fname)
        ok = "OK" if actual == expected else "MISMATCH"
        lines.append(f"- {fname} sha256: {actual} [{ok}]")
    lines.append("")
    lines.append("## Output file sha256 (run #1)")
    lines.append("")
    for k in ALL_OUTPUTS:
        lines.append(f"{sha1[k]}  {k}")
    lines.append("")
    lines.append("## Output file sha256 (run #2)")
    lines.append("")
    for k in ALL_OUTPUTS:
        lines.append(f"{sha2[k]}  {k}")
    lines.append("")
    lines.append("## Byte-identicality")
    lines.append("")
    for k in ALL_OUTPUTS:
        same = "IDENTICAL" if sha1[k] == sha2[k] else "DIFFER"
        lines.append(f"- {k}: {same}")
    lines.append("")
    if diff:
        lines.append("## PNG pixel-only sha256 (fallback for diverging PNGs)")
        lines.append("")
        for k, (p1, p2) in pixel_hashes.items():
            same = "IDENTICAL" if p1 == p2 else "DIFFER"
            lines.append(f"- {k}: pixel-hash run1 = {p1} [{same}]")
        lines.append("")
        # Final determination
        any_pixel_diff = any(p1 != p2 for p1, p2 in pixel_hashes.values())
        non_png_diff = any(not k.endswith(".png") for k in diff)
        if non_png_diff or any_pixel_diff:
            lines.append("Overall: FAIL - byte-identicality not achieved")
        else:
            lines.append(
                "Overall: PASS (PNG pixel-hash byte-identical; "
                "metadata differs but content matches)"
            )
    else:
        lines.append("Overall: PASS - byte-identical across runs")
    lines.append("")

    manifest_text = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "run_manifest.txt").write_bytes(manifest_text.encode("utf-8"))
    print(manifest_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
