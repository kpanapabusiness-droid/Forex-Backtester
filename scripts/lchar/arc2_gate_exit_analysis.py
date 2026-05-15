"""Arc 2 Gate Exit Simulation — final descriptive analysis before Arc 2 close.

Phase: l6_arc2_gate_exit_analysis (additive; descriptive only per L6.0 v1.1 §14.5).

Reads (read-only):
- results/l6/arc2/trajectory_appendix/trajectory_panel.parquet
- results/l6/arc2/trajectory_appendix/shape_features.csv
- results/l6/arc2/trajectory_appendix/cluster_assignments.csv
All three must hash to the values recorded in the parent appendix manifest.

Writes:
- results/l6/arc2/trajectory_appendix/gate_exit_analysis/

Four exit-rule variants are simulated on the full N = 41,794 ex-ante population,
plus two reference cells (verbatim Arc 2 / no time exit) for band_behavior
parity verification (HALT on > 0.005R deviation).

Variants:
- V1 single-bar gate: 5 t* x 5 T = 25 cells. At bar t* exit if r_close[t*] < T.
- V2 composite classifier: 5 t* cells. Logistic regression on
  (r_close, running_mfe, running_mae) trained per t* with target r_at_t240 > 0.
- V3 two-bar consistency: 4 (t1, t2) cells. Exit at t2 if both bars failed gate.
- V4 MFE-conditional skip: 25 cells. Same grid as V1 but skip the gate if
  running_mfe[t*] >= 1R (the trade has proven itself).

Plus:
- Per-cluster contribution table for top-5 cells of each variant and the 2
  reference cells.
- Per-fold breakdown for every cell (4 variants + 2 refs = 61 cells).
- SL-wick analysis (Candidate 3b probe): classify intrabar SL hits as wicks vs
  close-below; simulate close-confirmed SL variant.

Determinism: numpy seed locked; sort signal_idx asc; logistic regression via
scipy.optimize.minimize (L-BFGS-B) with deterministic init; manual AUC via
Mann-Whitney. Two-run byte-identicality verified.

§14.5: report is empirical only. Any unmissable cell triggers a manual append
to ../CANDIDATE_HYPOTHESES_DRAFT.md under Candidate 8.
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

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SOURCE_DATE_EPOCH", "1577836800")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402
from scipy import stats as sps  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# LOCKED CONSTANTS
# ---------------------------------------------------------------------------

NUMPY_SEED: int = 20260512
SKLEARN_SEED: int = 20260512  # used as random_state for train/test split + classifier init

APPENDIX_DIR: Path = REPO_ROOT / "results" / "l6" / "arc2" / "trajectory_appendix"
INPUT_PANEL: Path = APPENDIX_DIR / "trajectory_panel.parquet"
INPUT_SHAPE: Path = APPENDIX_DIR / "shape_features.csv"
INPUT_CLUSTERS: Path = APPENDIX_DIR / "cluster_assignments.csv"

OUTPUT_DIR: Path = APPENDIX_DIR / "gate_exit_analysis"
CHARTS_DIR: Path = OUTPUT_DIR / "charts"

EXPECTED_INPUT_HASHES: Dict[str, str] = {
    "trajectory_panel.parquet":
        "c8f1ec4825ada7b3a9efaf72101e79b067f7056117bf70aaae2f5766261398ed",
    "shape_features.csv":
        "6920c990628dd2d769cfbc56bd006391f6b9a24beea64f52f5bda9da02d44aaa",
    "cluster_assignments.csv":
        "c82f7208ece085e804bc7c66c741d91e5004c604463f8dbeff03381aacf7ecbe",
}

# Reference parity targets from band_behavior (allow tolerance 0.005R).
BAND_BEHAVIOR_REF_MEAN_R: Dict[str, float] = {
    "verbatim_arc2": 0.0028,
    "no_time_exit": 0.0205,
}
REF_TOLERANCE: float = 0.005

# Fold OOS boundaries.
FOLDS: Tuple[Tuple[int, str, str], ...] = (
    (1, "2020-10-01", "2021-07-01"),
    (2, "2021-07-01", "2022-04-01"),
    (3, "2022-04-01", "2023-01-01"),
    (4, "2023-01-01", "2023-10-01"),
    (5, "2023-10-01", "2024-07-01"),
    (6, "2024-07-01", "2025-04-01"),
    (7, "2025-04-01", "2026-01-01"),
)

# Grids.
V1_T_STARS: Tuple[int, ...] = (12, 24, 36, 48, 60)
V1_THRESHOLDS: Tuple[float, ...] = (-0.5, -0.25, 0.0, 0.25, 0.5)
V2_T_STARS: Tuple[int, ...] = (12, 24, 36, 48, 60)
V3_PAIRS: Tuple[Tuple[int, int], ...] = ((12, 24), (24, 48), (12, 36), (24, 36))
V3_T1: float = 0.0
V3_T2: float = 0.0
V4_T_STARS: Tuple[int, ...] = (12, 24, 36, 48, 60)
V4_THRESHOLDS: Tuple[float, ...] = (-0.5, -0.25, 0.0, 0.25, 0.5)
V4_MFE_SKIP_LEVEL: float = 1.0

DIST_PERCENTILES: Tuple[float, ...] = (1, 5, 10, 25, 50, 75, 90, 95, 99)
HIST_BIN_EDGES_DEFAULT: np.ndarray = np.concatenate([
    [-np.inf],
    np.round(np.arange(-3.0, 12.0 + 0.25 / 2, 0.25), 4),
    [np.inf],
])

EXIT_REASONS = ("sl", "gate", "time_240")


# ---------------------------------------------------------------------------
# Filesystem helpers.
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
        length = int.from_bytes(data[pos:pos + 4], "big")
        chunk_type = data[pos + 4:pos + 8]
        chunk_data = data[pos + 8:pos + 8 + length]
        if chunk_type == b"IDAT":
            out.update(chunk_data)
        pos = pos + 8 + length + 4
    return out.hexdigest()


def _write_csv(df: pd.DataFrame, path: Path, *, float_format: str = "%.10g", index: bool = False) -> None:
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
# Distributional spec.
# ---------------------------------------------------------------------------


def describe_distribution(values: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    out: Dict[str, Any] = {"n": n}
    if n == 0:
        for k in ("mean", "std", "skew", "excess_kurt", "min", "max"):
            out[k] = float("nan")
        for p in DIST_PERCENTILES:
            out[f"p{int(p)}"] = float("nan")
        out["histogram_bin_edges"] = HIST_BIN_EDGES_DEFAULT.tolist()
        out["histogram_counts"] = [0] * (len(HIST_BIN_EDGES_DEFAULT) - 1)
        return out
    out["mean"] = float(np.mean(arr))
    out["std"] = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
    out["skew"] = float(sps.skew(arr, bias=False)) if n > 2 else float("nan")
    out["excess_kurt"] = float(sps.kurtosis(arr, fisher=True, bias=False)) if n > 3 else float("nan")
    out["min"] = float(np.min(arr))
    out["max"] = float(np.max(arr))
    pcts = np.percentile(arr, list(DIST_PERCENTILES), method="linear")
    for p, v in zip(DIST_PERCENTILES, pcts):
        out[f"p{int(p)}"] = float(v)
    counts, _ = np.histogram(arr, bins=HIST_BIN_EDGES_DEFAULT)
    out["histogram_bin_edges"] = HIST_BIN_EDGES_DEFAULT.tolist()
    out["histogram_counts"] = counts.astype(int).tolist()
    return out


# ---------------------------------------------------------------------------
# Hand-rolled logistic regression + AUC.
# ---------------------------------------------------------------------------


def _logistic_loss_and_grad(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    w = theta[:-1]
    b = theta[-1]
    z = X @ w + b
    # log(1+exp(z)) - y*z, computed stably:
    log1pexp = np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
    loss = float(np.mean(log1pexp - y * z))
    p = expit(z)
    diff = p - y
    grad_w = X.T @ diff / len(y)
    grad_b = float(np.mean(diff))
    grad = np.concatenate([grad_w, [grad_b]])
    return loss, grad


def fit_logistic_lbfgs(
    X: np.ndarray, y: np.ndarray, *, max_iter: int = 500, tol: float = 1e-7
) -> Tuple[np.ndarray, float, bool]:
    """Fit a logistic regression via L-BFGS-B; returns (w, b, converged)."""
    theta0 = np.zeros(X.shape[1] + 1, dtype=np.float64)
    res = minimize(
        _logistic_loss_and_grad, theta0,
        args=(X, y), jac=True, method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": tol, "ftol": tol},
    )
    w = res.x[:-1]
    b = float(res.x[-1])
    return w, b, bool(res.success)


def predict_proba_logistic(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return expit(X @ w + b)


def compute_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC: rank-based, handles ties correctly via average ranks."""
    pos_mask = y == 1
    neg_mask = y == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = sps.rankdata(scores, method="average")
    sum_ranks_pos = float(ranks[pos_mask].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ---------------------------------------------------------------------------
# Load inputs.
# ---------------------------------------------------------------------------


def load_inputs() -> Tuple[
    Dict[int, Dict[str, np.ndarray]],
    Dict[int, int],   # first_sl_bar per signal_idx (0 if never)
    Dict[int, float], # r_at_t240 per signal_idx
    Dict[int, int],   # n_post_bars per signal_idx (panel length excluding t=0)
    Dict[int, int],   # cluster_id per signal_idx (km_k6; -1 for NaN-cluster)
    Dict[int, int],   # fold_id per signal_idx
    Dict[int, str],   # pair per signal_idx
    Dict[str, str],   # actual hashes
    int,              # N total
]:
    actual_hashes: Dict[str, str] = {
        "trajectory_panel.parquet": _sha256_file(INPUT_PANEL),
        "shape_features.csv": _sha256_file(INPUT_SHAPE),
        "cluster_assignments.csv": _sha256_file(INPUT_CLUSTERS),
    }
    for fname, expected in EXPECTED_INPUT_HASHES.items():
        actual = actual_hashes[fname]
        if actual != expected:
            raise RuntimeError(
                f"BLOCKER: input hash mismatch for {fname}: expected {expected}, got {actual}."
            )

    print("  Reading parquet ...", flush=True)
    panel_tbl = pq.read_table(INPUT_PANEL)
    panel = panel_tbl.to_pandas()
    panel = panel.sort_values(["signal_idx", "t"], kind="mergesort").reset_index(drop=True)

    print("  Reading shape_features ...", flush=True)
    shape = pd.read_csv(INPUT_SHAPE, low_memory=False)
    shape["signal_time"] = pd.to_datetime(shape["signal_time"])

    print("  Reading cluster_assignments ...", flush=True)
    clusters = pd.read_csv(INPUT_CLUSTERS, low_memory=False)[["signal_idx", "km_k6"]]

    shape = shape.merge(clusters, on="signal_idx", how="left")

    # Assign folds.
    ts = shape["signal_time"].values
    fold = np.zeros(len(shape), dtype=np.int32)
    for fid, start, end in FOLDS:
        s = np.datetime64(pd.Timestamp(start))
        e = np.datetime64(pd.Timestamp(end))
        fold[(ts >= s) & (ts < e)] = fid
    shape["fold_id"] = fold

    # Build per-signal arrays.
    sig_to_arr: Dict[int, Dict[str, np.ndarray]] = {}
    first_sl_bar: Dict[int, int] = {}
    r_at_t240: Dict[int, float] = {}
    n_post_bars: Dict[int, int] = {}

    print("  Grouping panel by signal_idx (this can take a minute) ...", flush=True)
    grouped = panel.groupby("signal_idx", sort=False)
    for s_idx, grp in grouped:
        t = grp["t"].to_numpy(dtype=np.int32)
        keep = t >= 1
        if not keep.any():
            continue
        rc = grp["r_close"].to_numpy(dtype=np.float64)[keep]
        rh = grp["r_high"].to_numpy(dtype=np.float64)[keep]
        rl = grp["r_low"].to_numpy(dtype=np.float64)[keep]
        rmfe = grp["running_mfe"].to_numpy(dtype=np.float64)[keep]
        rmae = grp["running_mae"].to_numpy(dtype=np.float64)[keep]
        t_arr = t[keep]
        sig_to_arr[int(s_idx)] = {
            "t": t_arr,
            "r_close": rc,
            "r_high": rh,
            "r_low": rl,
            "running_mfe": rmfe,
            "running_mae": rmae,
        }
        # first_sl_bar = first t (1-indexed) where r_low <= -1.0.
        sl_mask = rl <= -1.0
        if sl_mask.any():
            first_sl_bar[int(s_idx)] = int(t_arr[int(np.argmax(sl_mask))])
        else:
            first_sl_bar[int(s_idx)] = 0  # 0 = never SL'd
        n_post_bars[int(s_idx)] = int(t_arr[-1])
        # r_at_t240: r_close at t=240 if available.
        if t_arr[-1] >= 240:
            r_at_t240[int(s_idx)] = float(rc[239])  # t=240 is at index 239 (since t_arr starts at 1)
        else:
            r_at_t240[int(s_idx)] = float("nan")

    cluster_id_map: Dict[int, int] = dict(zip(
        shape["signal_idx"].astype(int).tolist(),
        shape["km_k6"].fillna(-1).astype(int).tolist(),
    ))
    fold_id_map: Dict[int, int] = dict(zip(
        shape["signal_idx"].astype(int).tolist(),
        shape["fold_id"].astype(int).tolist(),
    ))
    pair_map: Dict[int, str] = dict(zip(
        shape["signal_idx"].astype(int).tolist(),
        shape["pair"].astype(str).tolist(),
    ))

    n_total = len(sig_to_arr)
    return (
        sig_to_arr,
        first_sl_bar,
        r_at_t240,
        n_post_bars,
        cluster_id_map,
        fold_id_map,
        pair_map,
        actual_hashes,
        n_total,
    )


# ---------------------------------------------------------------------------
# Simulation primitives.
# ---------------------------------------------------------------------------


def _last_available_r_close(rc: np.ndarray, t_target: int) -> Tuple[int, float]:
    """Return (exit_bar, r_close) at min(t_target, len(rc)) for time exit."""
    last_idx = min(t_target, len(rc)) - 1  # 0-based index, 1-based t
    return last_idx + 1, float(rc[last_idx])


def simulate_v1(
    sigs_sorted: List[int],
    sig_to_arr: Dict[int, Dict[str, np.ndarray]],
    first_sl_bar: Dict[int, int],
    t_star: int,
    T: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V1 single-bar gate. Returns (exit_t, exit_reason_idx, realized_r) per signal."""
    n = len(sigs_sorted)
    exit_t = np.zeros(n, dtype=np.int32)
    reason_idx = np.zeros(n, dtype=np.int32)  # 0=sl, 1=gate, 2=time_240
    realized = np.zeros(n, dtype=np.float64)
    for i, sig_idx in enumerate(sigs_sorted):
        a = sig_to_arr[sig_idx]
        rc = a["r_close"]
        sl_bar = first_sl_bar[sig_idx]
        n_bars = len(rc)
        if sl_bar > 0 and sl_bar <= t_star:
            exit_t[i] = sl_bar
            reason_idx[i] = 0
            realized[i] = -1.0
            continue
        if t_star <= n_bars and rc[t_star - 1] < T:
            exit_t[i] = t_star
            reason_idx[i] = 1
            realized[i] = float(rc[t_star - 1])
            continue
        # gate passed (or unavailable). SL after t_star?
        if sl_bar > t_star:
            exit_t[i] = sl_bar
            reason_idx[i] = 0
            realized[i] = -1.0
            continue
        # time exit
        et, rcv = _last_available_r_close(rc, 240)
        exit_t[i] = et
        reason_idx[i] = 2
        realized[i] = rcv
    return exit_t, reason_idx, realized


def simulate_v3(
    sigs_sorted: List[int],
    sig_to_arr: Dict[int, Dict[str, np.ndarray]],
    first_sl_bar: Dict[int, int],
    t1: int,
    t2: int,
    T1: float,
    T2: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V3 two-bar consistency. Exit at t2 iff r_close[t1] < T1 AND r_close[t2] < T2."""
    n = len(sigs_sorted)
    exit_t = np.zeros(n, dtype=np.int32)
    reason_idx = np.zeros(n, dtype=np.int32)
    realized = np.zeros(n, dtype=np.float64)
    for i, sig_idx in enumerate(sigs_sorted):
        a = sig_to_arr[sig_idx]
        rc = a["r_close"]
        sl_bar = first_sl_bar[sig_idx]
        n_bars = len(rc)
        # SL within [1, t2]?
        if sl_bar > 0 and sl_bar <= t2:
            exit_t[i] = sl_bar
            reason_idx[i] = 0
            realized[i] = -1.0
            continue
        # Both bar conditions met?
        t1_idx = t1 - 1
        t2_idx = t2 - 1
        v1 = rc[t1_idx] if t1 <= n_bars else float("inf")  # if missing, gate cannot fire
        v2 = rc[t2_idx] if t2 <= n_bars else float("inf")
        if v1 < T1 and v2 < T2:
            exit_t[i] = t2
            reason_idx[i] = 1
            realized[i] = float(v2)
            continue
        # gate didn't fire. SL after t2?
        if sl_bar > t2:
            exit_t[i] = sl_bar
            reason_idx[i] = 0
            realized[i] = -1.0
            continue
        et, rcv = _last_available_r_close(rc, 240)
        exit_t[i] = et
        reason_idx[i] = 2
        realized[i] = rcv
    return exit_t, reason_idx, realized


def simulate_v4(
    sigs_sorted: List[int],
    sig_to_arr: Dict[int, Dict[str, np.ndarray]],
    first_sl_bar: Dict[int, int],
    t_star: int,
    T: float,
    mfe_skip_level: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """V4 MFE-conditional skip. Returns (exit_t, reason_idx, realized_r, skipped_by_mfe)."""
    n = len(sigs_sorted)
    exit_t = np.zeros(n, dtype=np.int32)
    reason_idx = np.zeros(n, dtype=np.int32)
    realized = np.zeros(n, dtype=np.float64)
    skipped = np.zeros(n, dtype=np.int8)
    for i, sig_idx in enumerate(sigs_sorted):
        a = sig_to_arr[sig_idx]
        rc = a["r_close"]
        rmfe = a["running_mfe"]
        sl_bar = first_sl_bar[sig_idx]
        n_bars = len(rc)
        if sl_bar > 0 and sl_bar <= t_star:
            exit_t[i] = sl_bar
            reason_idx[i] = 0
            realized[i] = -1.0
            continue
        t_idx = t_star - 1
        if t_star <= n_bars and rmfe[t_idx] >= mfe_skip_level:
            skipped[i] = 1
            # Skip gate: continue to SL or time.
            if sl_bar > t_star:
                exit_t[i] = sl_bar
                reason_idx[i] = 0
                realized[i] = -1.0
                continue
            et, rcv = _last_available_r_close(rc, 240)
            exit_t[i] = et
            reason_idx[i] = 2
            realized[i] = rcv
            continue
        if t_star <= n_bars and rc[t_idx] < T:
            exit_t[i] = t_star
            reason_idx[i] = 1
            realized[i] = float(rc[t_idx])
            continue
        if sl_bar > t_star:
            exit_t[i] = sl_bar
            reason_idx[i] = 0
            realized[i] = -1.0
            continue
        et, rcv = _last_available_r_close(rc, 240)
        exit_t[i] = et
        reason_idx[i] = 2
        realized[i] = rcv
    return exit_t, reason_idx, realized, skipped


def simulate_reference(
    sigs_sorted: List[int],
    sig_to_arr: Dict[int, Dict[str, np.ndarray]],
    first_sl_bar: Dict[int, int],
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mode: 'verbatim_arc2' = SL + close at t=120; 'no_time_exit' = SL + close at t=240."""
    n = len(sigs_sorted)
    horizon = 120 if mode == "verbatim_arc2" else 240
    exit_t = np.zeros(n, dtype=np.int32)
    reason_idx = np.zeros(n, dtype=np.int32)  # 0=sl, 2=time
    realized = np.zeros(n, dtype=np.float64)
    for i, sig_idx in enumerate(sigs_sorted):
        a = sig_to_arr[sig_idx]
        rc = a["r_close"]
        sl_bar = first_sl_bar[sig_idx]
        if sl_bar > 0 and sl_bar <= horizon:
            exit_t[i] = sl_bar
            reason_idx[i] = 0
            realized[i] = -1.0
            continue
        et, rcv = _last_available_r_close(rc, horizon)
        exit_t[i] = et
        reason_idx[i] = 2
        realized[i] = rcv
    return exit_t, reason_idx, realized


# ---------------------------------------------------------------------------
# Cell stats emission.
# ---------------------------------------------------------------------------


def cell_stats_rows(
    exit_t: np.ndarray, reason_idx: np.ndarray, realized: np.ndarray, base: Dict[str, Any]
) -> List[Dict[str, Any]]:
    n_total = int(realized.size)
    rows: List[Dict[str, Any]] = [{**base, "stat": "N", "value": n_total}]
    for k, reason in enumerate(EXIT_REASONS):
        c = int(np.sum(reason_idx == k))
        rows.append({**base, "stat": f"n_{reason}", "value": c})
        rows.append({**base, "stat": f"p_{reason}", "value": (c / n_total) if n_total else float("nan")})
    desc = describe_distribution(realized)
    for s in ("mean", "std", "skew", "excess_kurt", "min", "max"):
        rows.append({**base, "stat": f"{s}_realized_r", "value": desc[s]})
    for p in DIST_PERCENTILES:
        rows.append({**base, "stat": f"p{int(p)}_realized_r", "value": desc[f"p{int(p)}"]})
    rows.append({**base, "stat": "p_realized_r_gt_0", "value": float(np.mean(realized > 0.0))})
    rows.append({**base, "stat": "p_realized_r_gt_1", "value": float(np.mean(realized > 1.0))})
    rows.append({**base, "stat": "mean_exit_bar", "value": float(np.mean(exit_t))})
    edges = desc["histogram_bin_edges"]
    counts = desc["histogram_counts"]
    for i, c in enumerate(counts):
        lo = edges[i]
        hi = edges[i + 1]
        lo_s = "-inf" if not np.isfinite(lo) else f"{lo:.4f}"
        hi_s = "inf" if not np.isfinite(hi) else f"{hi:.4f}"
        rows.append({**base, "stat": f"hist[{lo_s},{hi_s})", "value": int(c)})
    return rows


# ---------------------------------------------------------------------------
# Per-cluster + per-fold breakdowns.
# ---------------------------------------------------------------------------


def per_cluster_rows(
    base: Dict[str, Any],
    sigs_sorted: List[int],
    cluster_id_map: Dict[int, int],
    realized: np.ndarray,
) -> List[Dict[str, Any]]:
    cluster_arr = np.array([cluster_id_map.get(s, -1) for s in sigs_sorted], dtype=np.int32)
    n_total = int(realized.size)
    pooled_mean = float(np.mean(realized))
    rows: List[Dict[str, Any]] = [{**base, "cluster": "POOLED", "stat": "pooled_mean_r", "value": pooled_mean}]
    for cluster_id in sorted(set(cluster_arr.tolist())):
        mask = cluster_arr == cluster_id
        n_cluster = int(mask.sum())
        if n_cluster == 0:
            continue
        cluster_mean = float(np.mean(realized[mask]))
        contrib = (n_cluster / n_total) * cluster_mean
        rows.append({**base, "cluster": int(cluster_id), "stat": "n", "value": n_cluster})
        rows.append({**base, "cluster": int(cluster_id), "stat": "mean_r", "value": cluster_mean})
        rows.append({**base, "cluster": int(cluster_id), "stat": "contribution_to_pooled_mean", "value": contrib})
    return rows


def per_fold_rows(
    base: Dict[str, Any],
    sigs_sorted: List[int],
    fold_id_map: Dict[int, int],
    realized: np.ndarray,
) -> List[Dict[str, Any]]:
    fold_arr = np.array([fold_id_map.get(s, 0) for s in sigs_sorted], dtype=np.int32)
    rows: List[Dict[str, Any]] = []
    worst_mean = float("inf")
    for fold in range(1, 8):
        mask = fold_arr == fold
        n_fold = int(mask.sum())
        if n_fold == 0:
            rows.append({**base, "fold": fold, "stat": "n", "value": 0})
            rows.append({**base, "fold": fold, "stat": "mean_r", "value": float("nan")})
            rows.append({**base, "fold": fold, "stat": "dd_proxy", "value": float("nan")})
            continue
        r_fold = realized[mask]
        mean_r = float(np.mean(r_fold))
        worst_mean = min(worst_mean, mean_r)
        # DD proxy: cumulative sum of R values × 0.01 (1% sizing), peak-to-trough.
        cumsum = np.cumsum(r_fold * 0.01)
        running_peak = np.maximum.accumulate(cumsum)
        dd = float(np.min(cumsum - running_peak))
        rows.append({**base, "fold": fold, "stat": "n", "value": n_fold})
        rows.append({**base, "fold": fold, "stat": "mean_r", "value": mean_r})
        rows.append({**base, "fold": fold, "stat": "dd_proxy", "value": dd})
    rows.append({**base, "fold": "WORST", "stat": "worst_fold_mean_r", "value": worst_mean})
    return rows


# ---------------------------------------------------------------------------
# Variant 2 — composite logistic regression at each t*.
# ---------------------------------------------------------------------------


def variant_2_pipeline(
    sigs_sorted: List[int],
    sig_to_arr: Dict[int, Dict[str, np.ndarray]],
    first_sl_bar: Dict[int, int],
    r_at_t240: Dict[int, float],
    cluster_id_map: Dict[int, int],
    fold_id_map: Dict[int, int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, Dict[str, float]]]:
    """For each t*, train logistic regression on (rc, mfe, mae)[t*] -> r_at_t240>0,
    apply gate p_winner<0.5, and emit cell stats + per-cluster + per-fold rows."""
    rng = np.random.default_rng(SKLEARN_SEED)
    grid_rows: List[Dict[str, Any]] = []
    cluster_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    classifier_meta: Dict[int, Dict[str, float]] = {}

    for t_star in V2_T_STARS:
        t_idx = t_star - 1
        # Build feature matrix on signals where panel reaches t_star AND r_at_t240 is finite.
        feat_rows: List[Tuple[int, float, float, float, float]] = []
        for s in sigs_sorted:
            arrs = sig_to_arr[s]
            if len(arrs["r_close"]) <= t_idx:
                continue
            target = r_at_t240.get(s, float("nan"))
            if not np.isfinite(target):
                continue
            feat_rows.append((
                s,
                float(arrs["r_close"][t_idx]),
                float(arrs["running_mfe"][t_idx]),
                float(arrs["running_mae"][t_idx]),
                float(target),
            ))
        if not feat_rows:
            continue
        feat_arr = np.array(feat_rows, dtype=np.float64)
        sig_ids = feat_arr[:, 0].astype(np.int64)
        X_raw = feat_arr[:, 1:4]
        y = (feat_arr[:, 4] > 0.0).astype(np.float64)

        # Standardise.
        means = X_raw.mean(axis=0)
        stds = X_raw.std(axis=0, ddof=0)
        stds = np.where(stds > 1e-12, stds, 1.0)
        X_std = (X_raw - means) / stds

        # Train/test split 70/30 deterministically.
        n = len(X_std)
        perm = rng.permutation(n)
        n_train = int(round(n * 0.70))
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
        X_train, y_train = X_std[train_idx], y[train_idx]
        X_test, y_test = X_std[test_idx], y[test_idx]

        w, b, converged = fit_logistic_lbfgs(X_train, y_train)
        if not converged:
            raise RuntimeError(f"BLOCKER: classifier did not converge at t*={t_star}")

        score_train = predict_proba_logistic(X_train, w, b)
        score_test = predict_proba_logistic(X_test, w, b)
        auc_train = compute_auc(score_train, y_train)
        auc_test = compute_auc(score_test, y_test)
        auc_gap = auc_train - auc_test
        classifier_meta[t_star] = {
            "auc_train": auc_train,
            "auc_test": auc_test,
            "auc_gap": auc_gap,
            "coef_r_close": float(w[0]),
            "coef_running_mfe": float(w[1]),
            "coef_running_mae": float(w[2]),
            "intercept": float(b),
            "feature_mean_r_close": float(means[0]),
            "feature_mean_running_mfe": float(means[1]),
            "feature_mean_running_mae": float(means[2]),
            "feature_std_r_close": float(stds[0]),
            "feature_std_running_mfe": float(stds[1]),
            "feature_std_running_mae": float(stds[2]),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_excluded_no_target": int(len(sigs_sorted) - n),
        }

        # Apply gate to the FULL population (in-sample for descriptive purposes).
        exit_t = np.zeros(len(sigs_sorted), dtype=np.int32)
        reason_idx = np.zeros(len(sigs_sorted), dtype=np.int32)
        realized = np.zeros(len(sigs_sorted), dtype=np.float64)
        for i, s in enumerate(sigs_sorted):
            arrs = sig_to_arr[s]
            rc = arrs["r_close"]
            rmfe = arrs["running_mfe"]
            rmae = arrs["running_mae"]
            sl_bar = first_sl_bar[s]
            n_bars = len(rc)
            if sl_bar > 0 and sl_bar <= t_star:
                exit_t[i] = sl_bar
                reason_idx[i] = 0
                realized[i] = -1.0
                continue
            if t_star > n_bars:
                # Panel ended before t_star, fall through to time exit on last bar.
                if sl_bar > 0 and sl_bar <= n_bars:
                    exit_t[i] = sl_bar
                    reason_idx[i] = 0
                    realized[i] = -1.0
                else:
                    et, rcv = _last_available_r_close(rc, 240)
                    exit_t[i] = et
                    reason_idx[i] = 2
                    realized[i] = rcv
                continue
            # Score this signal at t_star.
            feat = np.array([rc[t_star - 1], rmfe[t_star - 1], rmae[t_star - 1]], dtype=np.float64)
            feat_std = (feat - means) / stds
            p_winner = float(expit(feat_std @ w + b))
            if p_winner < 0.5:
                exit_t[i] = t_star
                reason_idx[i] = 1
                realized[i] = float(rc[t_star - 1])
                continue
            if sl_bar > t_star:
                exit_t[i] = sl_bar
                reason_idx[i] = 0
                realized[i] = -1.0
                continue
            et, rcv = _last_available_r_close(rc, 240)
            exit_t[i] = et
            reason_idx[i] = 2
            realized[i] = rcv

        base = {"variant": "V2_composite", "cell_id": f"V2_t{t_star}", "t_star": t_star}
        grid_rows.extend(cell_stats_rows(exit_t, reason_idx, realized, base))
        # Classifier meta rows in the same CSV.
        for k, v in classifier_meta[t_star].items():
            grid_rows.append({**base, "stat": f"classifier__{k}", "value": v})
        cluster_rows.extend(per_cluster_rows(base, sigs_sorted, cluster_id_map, realized))
        fold_rows.extend(per_fold_rows(base, sigs_sorted, fold_id_map, realized))

    return grid_rows, cluster_rows, fold_rows, classifier_meta


# ---------------------------------------------------------------------------
# SL-wick analysis (Candidate 3b probe).
# ---------------------------------------------------------------------------


def sl_wick_analysis(
    sigs_sorted: List[int],
    sig_to_arr: Dict[int, Dict[str, np.ndarray]],
    first_sl_bar: Dict[int, int],
    cluster_id_map: Dict[int, int],
    fold_id_map: Dict[int, int],
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Classify SL hits as wicks (r_close[t_sl] > -1) vs close-below; simulate close-confirmed SL."""
    analysis_rows: List[Dict[str, Any]] = []

    # Pooled stats + per-cluster breakdown.
    sl_hit_total = 0
    wick_total = 0
    close_below_total = 0
    subsequent_peak_mfe_wicks: List[float] = []
    final_r_wicks: List[float] = []
    recover_count_wicks = 0
    reach_neg1_close_after_wick = 0
    per_cluster_counts: Dict[int, Dict[str, int]] = {}

    for s in sigs_sorted:
        sl_bar = first_sl_bar[s]
        cluster = cluster_id_map.get(s, -1)
        if cluster not in per_cluster_counts:
            per_cluster_counts[cluster] = {"sl_hit": 0, "wick": 0, "close_below": 0}
        if sl_bar == 0:
            continue
        sl_hit_total += 1
        per_cluster_counts[cluster]["sl_hit"] += 1
        arrs = sig_to_arr[s]
        rc = arrs["r_close"]
        rmfe = arrs["running_mfe"]
        idx_sl = sl_bar - 1
        if idx_sl >= len(rc):
            continue
        if rc[idx_sl] > -1.0:
            wick_total += 1
            per_cluster_counts[cluster]["wick"] += 1
            # Subsequent trajectory: bars > t_sl.
            tail_rc = rc[idx_sl + 1:]
            tail_mfe = rmfe[idx_sl + 1:]
            if tail_mfe.size:
                subsequent_peak_mfe_wicks.append(float(np.max(tail_mfe)))
            if tail_rc.size:
                # final r_at_t240 if held: last r_close.
                final_r_wicks.append(float(tail_rc[-1]))
                if np.any(tail_rc > 0.0):
                    recover_count_wicks += 1
                if np.any(tail_rc <= -1.0):
                    reach_neg1_close_after_wick += 1
        else:
            close_below_total += 1
            per_cluster_counts[cluster]["close_below"] += 1

    p_wick = (wick_total / sl_hit_total) if sl_hit_total else float("nan")
    p_recover = (recover_count_wicks / wick_total) if wick_total else float("nan")
    p_reach_neg1_close_after = (reach_neg1_close_after_wick / wick_total) if wick_total else float("nan")

    analysis_rows.append({"scope": "POOLED", "stat": "n_sl_hit", "value": sl_hit_total})
    analysis_rows.append({"scope": "POOLED", "stat": "n_wick", "value": wick_total})
    analysis_rows.append({"scope": "POOLED", "stat": "n_close_below_sl", "value": close_below_total})
    analysis_rows.append({"scope": "POOLED", "stat": "p_wick_given_sl_hit", "value": p_wick})
    analysis_rows.append({"scope": "POOLED", "stat": "p_recover_r_close_gt_0_after_wick", "value": p_recover})
    analysis_rows.append({"scope": "POOLED", "stat": "p_close_below_neg1_after_wick", "value": p_reach_neg1_close_after})

    desc_peak = describe_distribution(np.asarray(subsequent_peak_mfe_wicks, dtype=np.float64))
    desc_final = describe_distribution(np.asarray(final_r_wicks, dtype=np.float64))
    for label, d in (("subsequent_peak_mfe_among_wicks", desc_peak), ("final_r_among_wicks", desc_final)):
        analysis_rows.append({"scope": "POOLED", "stat": f"{label}_n", "value": d["n"]})
        for s in ("mean", "std", "min", "max"):
            analysis_rows.append({"scope": "POOLED", "stat": f"{label}_{s}", "value": d[s]})
        for p in DIST_PERCENTILES:
            analysis_rows.append({"scope": "POOLED", "stat": f"{label}_p{int(p)}", "value": d[f"p{int(p)}"]})

    for cid, counts in sorted(per_cluster_counts.items()):
        analysis_rows.append({"scope": f"cluster_{cid}", "stat": "n_sl_hit", "value": counts["sl_hit"]})
        analysis_rows.append({"scope": f"cluster_{cid}", "stat": "n_wick", "value": counts["wick"]})
        analysis_rows.append({"scope": f"cluster_{cid}", "stat": "n_close_below_sl", "value": counts["close_below"]})
        if counts["sl_hit"]:
            analysis_rows.append({
                "scope": f"cluster_{cid}", "stat": "p_wick_given_sl_hit",
                "value": counts["wick"] / counts["sl_hit"],
            })

    analysis_df = pd.DataFrame(analysis_rows)

    # Close-confirmed SL variant: SL only if r_close <= -1.0 (not intrabar low).
    n = len(sigs_sorted)
    exit_t = np.zeros(n, dtype=np.int32)
    reason_idx = np.zeros(n, dtype=np.int32)
    realized = np.zeros(n, dtype=np.float64)
    for i, s in enumerate(sigs_sorted):
        arrs = sig_to_arr[s]
        rc = arrs["r_close"]
        # Find first bar where r_close <= -1.0.
        close_sl_mask = rc <= -1.0
        if close_sl_mask.any():
            idx = int(np.argmax(close_sl_mask))
            exit_t[i] = idx + 1
            reason_idx[i] = 0
            realized[i] = float(rc[idx])  # close at the bar that fired the SL
            continue
        et, rcv = _last_available_r_close(rc, 240)
        exit_t[i] = et
        reason_idx[i] = 2
        realized[i] = rcv

    # Simulation summary rows.
    sim_rows = cell_stats_rows(exit_t, reason_idx, realized,
                                {"variant": "ref_close_confirmed_sl", "cell_id": "close_confirmed_sl"})
    sim_rows.extend(per_fold_rows({"variant": "ref_close_confirmed_sl", "cell_id": "close_confirmed_sl"},
                                   sigs_sorted, fold_id_map, realized))
    sim_df = pd.DataFrame(sim_rows)

    return analysis_df, sim_df, realized


# ---------------------------------------------------------------------------
# Charts.
# ---------------------------------------------------------------------------


def chart_heatmap(
    values: np.ndarray,
    n_per_cell: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    path: Path,
    *,
    diverging: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    finite = values[np.isfinite(values)]
    if diverging and finite.size:
        vmax = float(np.max(np.abs(finite)))
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax) if vmax > 0 else None
        im = ax.imshow(values, aspect="auto", cmap="RdYlGn", norm=norm, origin="upper")
    else:
        im = ax.imshow(values, aspect="auto", cmap="viridis", origin="upper")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=cbar_label)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            n = int(n_per_cell[i, j]) if not np.isnan(n_per_cell[i, j]) else 0
            txt = f"{v:+.3f}\nN={n}" if np.isfinite(v) else f"NaN\nN={n}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="black")
    _save_figure(fig, path)


def chart_per_fold_heatmap(
    per_fold_df: pd.DataFrame, top_cells: List[str], path: Path
) -> None:
    """For top cells, show (cell × fold) mean R."""
    folds = list(range(1, 8))
    z = np.full((len(top_cells), len(folds)), np.nan)
    n_arr = np.full((len(top_cells), len(folds)), 0)
    for i, cell in enumerate(top_cells):
        for j, fold in enumerate(folds):
            sub = per_fold_df[
                (per_fold_df["cell_id"] == cell)
                & (per_fold_df["fold"] == fold)
                & (per_fold_df["stat"] == "mean_r")
            ]
            if len(sub):
                try:
                    z[i, j] = float(sub["value"].iloc[0])
                except (TypeError, ValueError):
                    pass
            sub_n = per_fold_df[
                (per_fold_df["cell_id"] == cell)
                & (per_fold_df["fold"] == fold)
                & (per_fold_df["stat"] == "n")
            ]
            if len(sub_n):
                try:
                    n_arr[i, j] = int(float(sub_n["value"].iloc[0]))
                except (TypeError, ValueError):
                    pass
    fig, ax = plt.subplots(figsize=(11, max(5, 0.55 * len(top_cells))))
    finite = z[np.isfinite(z)]
    if finite.size:
        vmax = float(np.max(np.abs(finite)))
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax) if vmax > 0 else None
        im = ax.imshow(z, aspect="auto", cmap="RdYlGn", norm=norm, origin="upper")
    else:
        im = ax.imshow(z, aspect="auto", cmap="viridis", origin="upper")
    ax.set_xticks(range(7))
    ax.set_xticklabels([f"f{f}" for f in folds])
    ax.set_yticks(range(len(top_cells)))
    ax.set_yticklabels(top_cells)
    ax.set_xlabel("fold_id")
    ax.set_ylabel("cell")
    ax.set_title("Per-fold mean realized R for top cells")
    plt.colorbar(im, ax=ax, label="mean realized R")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z[i, j]
            n = int(n_arr[i, j])
            txt = f"{v:+.3f}\nN={n}" if np.isfinite(v) else "n/a"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)
    _save_figure(fig, path)


def chart_two_bar_comparison(grid_v1: pd.DataFrame, grid_v3: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    means_v3 = []
    means_v1_t1 = []
    means_v1_t2 = []
    for (t1, t2) in V3_PAIRS:
        v3 = grid_v3[(grid_v3["cell_id"] == f"V3_t{t1}_t{t2}") & (grid_v3["stat"] == "mean_realized_r")]
        v1_t1 = grid_v1[(grid_v1["cell_id"] == f"V1_t{t1}_T0") & (grid_v1["stat"] == "mean_realized_r")]
        v1_t2 = grid_v1[(grid_v1["cell_id"] == f"V1_t{t2}_T0") & (grid_v1["stat"] == "mean_realized_r")]
        labels.append(f"({t1},{t2})")
        means_v3.append(float(v3["value"].iloc[0]) if len(v3) else float("nan"))
        means_v1_t1.append(float(v1_t1["value"].iloc[0]) if len(v1_t1) else float("nan"))
        means_v1_t2.append(float(v1_t2["value"].iloc[0]) if len(v1_t2) else float("nan"))
    x = np.arange(len(labels))
    width = 0.27
    ax.bar(x - width, means_v1_t1, width, label="V1 single-bar @ t1 (T=0)", color="C0")
    ax.bar(x, means_v3, width, label="V3 two-bar (T1=T2=0)", color="C1")
    ax.bar(x + width, means_v1_t2, width, label="V1 single-bar @ t2 (T=0)", color="C2")
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("(t1, t2) pair")
    ax.set_ylabel("mean realized R")
    ax.set_title("Two-bar consistency vs single-bar at the same anchors")
    ax.legend(fontsize=8)
    _save_figure(fig, path)


def chart_sl_wick_recovery(realized_cc: np.ndarray, realized_intrabar: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.arange(-3.0, 12.0 + 0.25, 0.25)
    ax.hist(realized_cc, bins=bins, alpha=0.5, color="C1", label="close-confirmed SL variant")
    ax.hist(realized_intrabar, bins=bins, alpha=0.5, color="C0", label="intrabar SL (no time exit)")
    ax.axvline(0.0, color="black", lw=0.5)
    ax.axvline(-1.0, color="red", lw=0.5, linestyle="--")
    ax.set_xlabel("realized R")
    ax.set_ylabel("count")
    ax.set_title("Realized R: close-confirmed SL vs intrabar SL (no time exit reference)")
    ax.legend(fontsize=9)
    _save_figure(fig, path)


# ---------------------------------------------------------------------------
# Report writer.
# ---------------------------------------------------------------------------


def write_report(
    *,
    actual_hashes: Dict[str, str],
    n_total: int,
    fold_counts: Dict[int, int],
    grid_v1: pd.DataFrame,
    grid_v2: pd.DataFrame,
    grid_v3: pd.DataFrame,
    grid_v4: pd.DataFrame,
    grid_refs: pd.DataFrame,
    per_cluster_df: pd.DataFrame,
    per_fold_df: pd.DataFrame,
    sl_wick_df: pd.DataFrame,
    sl_wick_sim_df: pd.DataFrame,
    classifier_meta: Dict[int, Dict[str, float]],
    ref_parity_ok: Dict[str, bool],
    path: Path,
) -> None:
    lines: List[str] = [
        "# Arc 2 Gate Exit Simulation — Empirical Report",
        "",
        "Phase: l6_arc2_gate_exit_analysis (supplementary, descriptive only per L6.0 v1.1 §14.5).",
        "",
        "## 1. Inputs and hashes",
        "",
        f"- trajectory_panel.parquet sha256: {actual_hashes['trajectory_panel.parquet']}",
        f"- shape_features.csv sha256: {actual_hashes['shape_features.csv']}",
        f"- cluster_assignments.csv sha256: {actual_hashes['cluster_assignments.csv']}",
        f"- N total in panel: {n_total}",
        "",
        "## 2. Conventions and fold mapping",
        "",
        "- 1R = 2 * Wilder ATR(14) at bar N close (`atr_1h_wilder_at_n`).",
        "- t=0 = entry price = open of bar N+1; per-bar metrics from the parent panel.",
        "- Bar-order: SL (intrabar r_low <= -1R) first, gate (r_close threshold) second,",
        "  time exit at t=240 last.",
        "",
        "Per-fold N (signal_time-based assignment):",
        "",
        "| fold | n_ex_ante |",
        "|------|-----------|",
    ]
    for f in range(1, 8):
        lines.append(f"| {f} | {fold_counts[f]} |")
    lines.append("")

    lines.append("## 3. Reference cells — band_behavior parity check")
    lines.append("")
    lines.append("| reference | observed_mean_r | band_behavior_target | delta | within_tolerance |")
    lines.append("|-----------|-----------------|----------------------|-------|-------------------|")
    for ref_mode in ("verbatim_arc2", "no_time_exit"):
        obs = grid_refs[(grid_refs["cell_id"] == ref_mode) & (grid_refs["stat"] == "mean_realized_r")]
        obs_v = float(obs["value"].iloc[0]) if len(obs) else float("nan")
        target = BAND_BEHAVIOR_REF_MEAN_R[ref_mode]
        delta = obs_v - target
        ok = ref_parity_ok[ref_mode]
        lines.append(
            f"| {ref_mode} | {_fmt_float(obs_v)} | {_fmt_float(target)} | "
            f"{_fmt_float(delta)} | {'PASS' if ok else 'FAIL'} |"
        )
    lines.append("")
    lines.append(f"Reference parity tolerance: |delta| <= {REF_TOLERANCE}R. Both PASS implies "
                 "methodology consistency with band_behavior.")
    lines.append("")

    lines.append("## 4. Variant 1 — single-bar gate grid (5 t* x 5 T = 25 cells)")
    lines.append("")
    lines.append("Mean realized R per cell:")
    lines.append("")
    header = ["t*\\T"] + [f"T={T:g}" for T in V1_THRESHOLDS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for t in V1_T_STARS:
        row = [f"t*={t}"]
        for T in V1_THRESHOLDS:
            cell_id = f"V1_t{t}_T{T:g}"
            r = grid_v1[(grid_v1["cell_id"] == cell_id) & (grid_v1["stat"] == "mean_realized_r")]
            row.append(_fmt_float(float(r["value"].iloc[0])) if len(r) else "NaN")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("P(realized_r > 0) per cell:")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for t in V1_T_STARS:
        row = [f"t*={t}"]
        for T in V1_THRESHOLDS:
            cell_id = f"V1_t{t}_T{T:g}"
            r = grid_v1[(grid_v1["cell_id"] == cell_id) & (grid_v1["stat"] == "p_realized_r_gt_0")]
            row.append(_fmt_float(float(r["value"].iloc[0])) if len(r) else "NaN")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Exit composition per cell (n_sl / n_gate / n_time_240):")
    lines.append("")
    lines.append("| cell | n_sl | n_gate | n_time_240 |")
    lines.append("|------|------|--------|------------|")
    for t in V1_T_STARS:
        for T in V1_THRESHOLDS:
            cell_id = f"V1_t{t}_T{T:g}"
            def gv(stat: str) -> str:
                r = grid_v1[(grid_v1["cell_id"] == cell_id) & (grid_v1["stat"] == stat)]
                return str(int(float(r["value"].iloc[0]))) if len(r) else "NaN"
            lines.append(f"| {cell_id} | {gv('n_sl')} | {gv('n_gate')} | {gv('n_time_240')} |")
    lines.append("")
    lines.append("Charts: `charts/gate_meanR_heatmap.png`, `charts/gate_winrate_heatmap.png`.")
    lines.append("")

    lines.append("## 5. Variant 2 — composite classifier")
    lines.append("")
    lines.append("Logistic regression on standardised (r_close[t*], running_mfe[t*], running_mae[t*]),")
    lines.append("target = r_at_t240 > 0, 70/30 train/test split (seeded), L-BFGS-B solver, intercept fitted.")
    lines.append("Gate rule: at bar t*, exit at r_close[t*] if predict_proba < 0.5; otherwise hold.")
    lines.append("")
    lines.append("| t* | n_train | n_test | AUC_train | AUC_test | gap | coef_r_close | coef_running_mfe | coef_running_mae | intercept | mean_R | P(R>0) |")
    lines.append("|----|---------|--------|-----------|----------|-----|---------------|--------------------|--------------------|-----------|--------|--------|")
    for t in V2_T_STARS:
        cm = classifier_meta.get(t, {})
        cell_id = f"V2_t{t}"
        r_mean = grid_v2[(grid_v2["cell_id"] == cell_id) & (grid_v2["stat"] == "mean_realized_r")]
        r_win = grid_v2[(grid_v2["cell_id"] == cell_id) & (grid_v2["stat"] == "p_realized_r_gt_0")]
        mean_v = _fmt_float(float(r_mean["value"].iloc[0])) if len(r_mean) else "NaN"
        win_v = _fmt_float(float(r_win["value"].iloc[0])) if len(r_win) else "NaN"
        lines.append(
            f"| {t} | {int(cm.get('n_train', 0))} | {int(cm.get('n_test', 0))} | "
            f"{_fmt_float(cm.get('auc_train', float('nan')))} | "
            f"{_fmt_float(cm.get('auc_test', float('nan')))} | "
            f"{_fmt_float(cm.get('auc_gap', float('nan')))} | "
            f"{_fmt_float(cm.get('coef_r_close', float('nan')))} | "
            f"{_fmt_float(cm.get('coef_running_mfe', float('nan')))} | "
            f"{_fmt_float(cm.get('coef_running_mae', float('nan')))} | "
            f"{_fmt_float(cm.get('intercept', float('nan')))} | "
            f"{mean_v} | {win_v} |"
        )
    lines.append("")
    lines.append("Train/test AUC gap should be < 0.05 for overfit warning to clear; values above that threshold are flagged in run_manifest.txt diagnostics.")
    lines.append("")

    lines.append("## 6. Variant 3 — two-bar consistency (T1 = T2 = 0)")
    lines.append("")
    lines.append("| (t1, t2) | mean_R (V3) | mean_R (V1@t1, T=0) | mean_R (V1@t2, T=0) | n_gate (V3) | n_sl (V3) |")
    lines.append("|----------|-------------|----------------------|----------------------|--------------|------------|")
    for t1, t2 in V3_PAIRS:
        cell_id = f"V3_t{t1}_t{t2}"
        def grab(df: pd.DataFrame, cid: str, stat: str) -> str:
            r = df[(df["cell_id"] == cid) & (df["stat"] == stat)]
            if not len(r):
                return "NaN"
            try:
                return _fmt_float(float(r["value"].iloc[0]))
            except (TypeError, ValueError):
                return str(r["value"].iloc[0])
        def grab_n(df: pd.DataFrame, cid: str, stat: str) -> str:
            r = df[(df["cell_id"] == cid) & (df["stat"] == stat)]
            return str(int(float(r["value"].iloc[0]))) if len(r) else "NaN"
        lines.append(
            f"| ({t1},{t2}) | {grab(grid_v3, cell_id, 'mean_realized_r')} | "
            f"{grab(grid_v1, f'V1_t{t1}_T0', 'mean_realized_r')} | "
            f"{grab(grid_v1, f'V1_t{t2}_T0', 'mean_realized_r')} | "
            f"{grab_n(grid_v3, cell_id, 'n_gate')} | "
            f"{grab_n(grid_v3, cell_id, 'n_sl')} |"
        )
    lines.append("")
    lines.append("Chart: `charts/two_bar_comparison.png`.")
    lines.append("")

    lines.append("## 7. Variant 4 — MFE-conditional skip")
    lines.append("")
    lines.append("Rule: at bar t*, if running_mfe[t*] >= 1R skip the gate (continue); otherwise apply V1 gate.")
    lines.append("")
    lines.append("Mean realized R per cell (delta vs V1 same cell shown in parentheses):")
    lines.append("")
    header = ["t*\\T"] + [f"T={T:g}" for T in V4_THRESHOLDS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for t in V4_T_STARS:
        row = [f"t*={t}"]
        for T in V4_THRESHOLDS:
            v4_id = f"V4_t{t}_T{T:g}"
            v1_id = f"V1_t{t}_T{T:g}"
            v4 = grid_v4[(grid_v4["cell_id"] == v4_id) & (grid_v4["stat"] == "mean_realized_r")]
            v1 = grid_v1[(grid_v1["cell_id"] == v1_id) & (grid_v1["stat"] == "mean_realized_r")]
            v4_v = float(v4["value"].iloc[0]) if len(v4) else float("nan")
            v1_v = float(v1["value"].iloc[0]) if len(v1) else float("nan")
            delta = v4_v - v1_v
            row.append(f"{_fmt_float(v4_v)} ({delta:+.3f})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Chart: `charts/gate_meanR_with_mfe_skip.png`.")
    lines.append("")

    lines.append("## 8. Per-cluster contribution table (top-5 cells per variant + 2 reference cells)")
    lines.append("")
    lines.append("Full table in `gate_cell_per_cluster.csv`. Headline: per-cell pooled mean R decomposed by cluster.")
    lines.append("")
    lines.append("Top contributors are shown per variant in the CSV; mean R per cluster differs only weakly across cells of the same variant (the rule sees the same population modulo gate timing).")
    lines.append("")

    lines.append("## 9. Per-fold breakdown — the headline gate criterion")
    lines.append("")
    lines.append("Worst-fold mean R per cell. The L6.0 §4 condition 1 requires worst-fold ROI > 0% across all 7 folds. Worst-fold mean R > 0 is a (much looser) proxy.")
    lines.append("")
    # Find cells where worst_fold_mean_r > 0.
    worst = per_fold_df[per_fold_df["fold"] == "WORST"].copy()
    worst["worst_mean_r"] = worst["value"].astype(float)
    surviving = worst[worst["worst_mean_r"] > 0.0].sort_values("worst_mean_r", ascending=False)
    if len(surviving) == 0:
        lines.append("**No cell has worst-fold mean R > 0.** All cells fail the proxy criterion on at least one fold.")
    else:
        lines.append("Cells with worst-fold mean R > 0:")
        lines.append("")
        lines.append("| variant | cell_id | worst_fold_mean_r |")
        lines.append("|---------|---------|---------------------|")
        for _, r in surviving.iterrows():
            lines.append(f"| {r['variant']} | {r['cell_id']} | {_fmt_float(float(r['worst_mean_r']))} |")
    lines.append("")
    lines.append("Top 10 cells by worst-fold mean R (regardless of sign):")
    lines.append("")
    top10 = worst.sort_values("worst_mean_r", ascending=False).head(10)
    lines.append("| variant | cell_id | worst_fold_mean_r |")
    lines.append("|---------|---------|---------------------|")
    for _, r in top10.iterrows():
        lines.append(f"| {r['variant']} | {r['cell_id']} | {_fmt_float(float(r['worst_mean_r']))} |")
    lines.append("")
    lines.append("Chart: `charts/gate_per_fold_heatmap.png` — top-10 cells × 7 folds.")
    lines.append("")

    lines.append("## 10. SL wick analysis (Candidate 3b probe)")
    lines.append("")
    pooled = sl_wick_df[sl_wick_df["scope"] == "POOLED"]
    def grab_pooled(stat: str) -> str:
        r = pooled[pooled["stat"] == stat]
        if not len(r):
            return "NaN"
        try:
            return _fmt_float(float(r["value"].iloc[0]))
        except (TypeError, ValueError):
            return str(r["value"].iloc[0])
    lines.append(f"- N(SL hit) = {int(float(pooled[pooled['stat'] == 'n_sl_hit']['value'].iloc[0]))}")
    lines.append(f"- N(wick) = {int(float(pooled[pooled['stat'] == 'n_wick']['value'].iloc[0]))}")
    lines.append(f"- N(close-below SL) = {int(float(pooled[pooled['stat'] == 'n_close_below_sl']['value'].iloc[0]))}")
    lines.append(f"- P(wick | SL hit) = {grab_pooled('p_wick_given_sl_hit')}")
    lines.append(f"- Among wicks: P(recover r_close > 0R within bars (t_sl, 240]) = {grab_pooled('p_recover_r_close_gt_0_after_wick')}")
    lines.append(f"- Among wicks: P(close-below -1R confirmed after wick) = {grab_pooled('p_close_below_neg1_after_wick')}")
    lines.append("")
    lines.append("Per-cluster wick rate:")
    lines.append("")
    lines.append("| cluster | n_sl_hit | n_wick | p_wick |")
    lines.append("|---------|----------|--------|--------|")
    for cid in sorted(set(sl_wick_df["scope"].tolist())):
        if not cid.startswith("cluster_"):
            continue
        sub = sl_wick_df[sl_wick_df["scope"] == cid]
        def gv(s: str) -> str:
            r = sub[sub["stat"] == s]
            if not len(r):
                return "NaN"
            try:
                return _fmt_float(float(r["value"].iloc[0]))
            except (TypeError, ValueError):
                return str(r["value"].iloc[0])
        lines.append(f"| {cid} | {gv('n_sl_hit')} | {gv('n_wick')} | {gv('p_wick_given_sl_hit')} |")
    lines.append("")
    lines.append("Close-confirmed SL variant (SL only if r_close <= -1R, no intrabar trigger):")
    lines.append("")
    cc = sl_wick_sim_df[sl_wick_sim_df["cell_id"] == "close_confirmed_sl"]
    def grab_cc(stat: str) -> str:
        r = cc[cc["stat"] == stat]
        if not len(r):
            return "NaN"
        try:
            return _fmt_float(float(r["value"].iloc[0]))
        except (TypeError, ValueError):
            return str(r["value"].iloc[0])
    lines.append(f"- N = {int(float(cc[cc['stat'] == 'N']['value'].iloc[0]))}")
    lines.append(f"- mean_realized_r = {grab_cc('mean_realized_r')}")
    lines.append(f"- std_realized_r = {grab_cc('std_realized_r')}")
    lines.append(f"- P(R > 0) = {grab_cc('p_realized_r_gt_0')}")
    lines.append(f"- P(R > 1) = {grab_cc('p_realized_r_gt_1')}")
    lines.append(f"- mean exit bar = {grab_cc('mean_exit_bar')}")
    lines.append("")
    lines.append("Per-fold mean R (close-confirmed SL):")
    lines.append("")
    lines.append("| fold | n | mean_R | dd_proxy |")
    lines.append("|------|---|--------|----------|")
    for fold in range(1, 8):
        for_fold = sl_wick_sim_df[(sl_wick_sim_df["cell_id"] == "close_confirmed_sl") & (sl_wick_sim_df["fold"] == fold)]
        def fold_gv(s: str) -> str:
            r = for_fold[for_fold["stat"] == s]
            if not len(r):
                return "NaN"
            try:
                return _fmt_float(float(r["value"].iloc[0]))
            except (TypeError, ValueError):
                return str(r["value"].iloc[0])
        n_str = "NaN"
        n_row = for_fold[for_fold["stat"] == "n"]
        if len(n_row):
            n_str = str(int(float(n_row["value"].iloc[0])))
        lines.append(f"| {fold} | {n_str} | {fold_gv('mean_r')} | {fold_gv('dd_proxy')} |")
    lines.append("")
    lines.append("Chart: `charts/sl_wick_recovery.png` (realized R histograms: close-confirmed vs intrabar).")
    lines.append("")

    lines.append("## 11. Empirical observations")
    lines.append("")
    # Pull headline numbers for the observations.
    best_v1 = grid_v1[grid_v1["stat"] == "mean_realized_r"].copy()
    best_v1["value_f"] = best_v1["value"].astype(float)
    if len(best_v1):
        top = best_v1.sort_values("value_f", ascending=False).head(3)
        for _, r in top.iterrows():
            lines.append(
                f"- V1 top cell `{r['cell_id']}`: pooled mean realized R = {_fmt_float(r['value_f'])}. "
                "Worst-fold mean R: see Section 9."
            )
    best_v4 = grid_v4[grid_v4["stat"] == "mean_realized_r"].copy()
    best_v4["value_f"] = best_v4["value"].astype(float)
    if len(best_v4):
        top = best_v4.sort_values("value_f", ascending=False).head(3)
        for _, r in top.iterrows():
            lines.append(
                f"- V4 top cell `{r['cell_id']}`: pooled mean realized R = {_fmt_float(r['value_f'])} "
                "(MFE-skip variant)."
            )
    for t in V2_T_STARS:
        cm = classifier_meta.get(t, {})
        if not cm:
            continue
        cell_id = f"V2_t{t}"
        r_mean = grid_v2[(grid_v2["cell_id"] == cell_id) & (grid_v2["stat"] == "mean_realized_r")]
        if len(r_mean):
            lines.append(
                f"- V2 t*={t}: AUC test = {_fmt_float(cm.get('auc_test', float('nan')))}, "
                f"gap = {_fmt_float(cm.get('auc_gap', float('nan')))}, "
                f"pooled mean realized R = {_fmt_float(float(r_mean['value'].iloc[0]))}."
            )
    # Close-confirmed SL net effect.
    cc_mean = sl_wick_sim_df[(sl_wick_sim_df["cell_id"] == "close_confirmed_sl") & (sl_wick_sim_df["stat"] == "mean_realized_r")]
    ref_no_time = grid_refs[(grid_refs["cell_id"] == "no_time_exit") & (grid_refs["stat"] == "mean_realized_r")]
    if len(cc_mean) and len(ref_no_time):
        cc_v = float(cc_mean["value"].iloc[0])
        nt_v = float(ref_no_time["value"].iloc[0])
        lines.append(
            f"- Close-confirmed SL variant: mean realized R = {_fmt_float(cc_v)} vs no_time_exit "
            f"intrabar-SL reference {_fmt_float(nt_v)} (delta = {_fmt_float(cc_v - nt_v)})."
        )
    # Worst-fold survivors.
    if len(surviving) == 0:
        lines.append(
            "- Zero cells across V1/V2/V3/V4 achieve worst-fold mean R > 0. The strongest cell by "
            "worst-fold metric is the top row of the Section 9 'top 10' table."
        )
    else:
        lines.append(
            f"- {len(surviving)} cell(s) achieve worst-fold mean R > 0: see Section 9 'Cells with worst-fold mean R > 0'."
        )
    lines.append("")
    lines.append("Reading guide. The gate is a mid-trade exit conditional on r_close[t*] < threshold. "
                 "Variants V1/V3/V4 use fixed thresholds; V2 uses a learned threshold via logistic regression. "
                 "The headline graduation criterion for Candidate 8 (per the open task prompt) is: pooled "
                 "mean R > +0.10R, worst-fold mean R > +0.05R, and P(R>0) > 0.45 net of est. cost. The "
                 "cells in this run that meet that bar (if any) are flagged in Section 9.")
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
# Manifest.
# ---------------------------------------------------------------------------


def write_manifest(
    *,
    out_paths: List[Path],
    actual_input_hashes: Dict[str, str],
    n_total: int,
    fold_counts: Dict[int, int],
    classifier_meta: Dict[int, Dict[str, float]],
    ref_parity_ok: Dict[str, bool],
    ref_observed: Dict[str, float],
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
        f"Arc 2 Gate Exit Simulation — run_manifest (run #{run_ordinal})",
        "Generated: (suppressed in deterministic mode below)",
        "",
        "## Operational decisions",
        "",
        "- Hand-rolled logistic regression via scipy.optimize.minimize L-BFGS-B (sklearn unavailable).",
        "  Solver: 500 max iter, gtol=ftol=1e-7. Init theta=zeros (deterministic).",
        "- AUC via Mann-Whitney rank statistic with scipy.stats.rankdata 'average' method.",
        "- 70/30 train/test split per t* via numpy.random.default_rng(SKLEARN_SEED).permutation(N).",
        "- Bar-order: SL (intrabar r_low<=-1R) first, gate (r_close vs T) second, time exit last.",
        "- Reference cells: verbatim_arc2 = SL throughout + close at t=120; no_time_exit = SL + close at t=240.",
        "  Both verified against band_behavior values within 0.005R tolerance.",
        "- Clamped panels (n_post_bars < 240): time exit uses r_close at last available bar.",
        "- Per-fold dd_proxy: cumulative R * 0.01 (1% sizing assumption), peak-to-trough min. NOT the",
        "  L6.0 §4 condition 2 metric — that requires full equity-curve simulation.",
        "",
        "## Inputs (read-only, all hashes verified against parent appendix)",
        "",
        f"- trajectory_panel.parquet sha256: {actual_input_hashes['trajectory_panel.parquet']}",
        f"- shape_features.csv sha256: {actual_input_hashes['shape_features.csv']}",
        f"- cluster_assignments.csv sha256: {actual_input_hashes['cluster_assignments.csv']}",
        f"- N total: {n_total}",
        "",
        "## Reference parity vs band_behavior",
        "",
    ]
    for ref_mode in ("verbatim_arc2", "no_time_exit"):
        target = BAND_BEHAVIOR_REF_MEAN_R[ref_mode]
        obs = ref_observed.get(ref_mode, float("nan"))
        delta = obs - target
        lines.append(
            f"- {ref_mode}: observed mean R = {obs:.6f}, target {target:.4f}, "
            f"delta = {delta:+.6f}, "
            f"{'PASS' if ref_parity_ok[ref_mode] else 'FAIL'}"
        )
    lines.append("")
    lines.append("## Per-fold N counts")
    lines.append("")
    for f in range(1, 8):
        lines.append(f"- Fold {f}: n = {fold_counts[f]}")
    lines.append("")
    lines.append("## V2 classifier diagnostics")
    lines.append("")
    for t in V2_T_STARS:
        cm = classifier_meta.get(t, {})
        if not cm:
            continue
        warn = " WARN-overfit" if cm.get("auc_gap", 0) > 0.05 else ""
        lines.append(
            f"- t*={t}: AUC train = {cm.get('auc_train', float('nan')):.4f}, "
            f"AUC test = {cm.get('auc_test', float('nan')):.4f}, "
            f"gap = {cm.get('auc_gap', float('nan')):+.4f}{warn}, "
            f"n_train = {cm.get('n_train', 0)}, n_test = {cm.get('n_test', 0)}"
        )
    lines.append("")
    lines.append("## Determinism config")
    lines.append("")
    lines.append(f"- NUMPY_SEED = {NUMPY_SEED}")
    lines.append(f"- SKLEARN_SEED = {SKLEARN_SEED}")
    lines.append("- Sort: sigs_sorted = sorted(signal_idx). Per-cluster / per-fold iteration sorted.")
    lines.append("- CSV: utf-8, LF, float_format='%.10g'.")
    lines.append("- Matplotlib: Agg backend, pinned metadata.")
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
            ok = (prior == h)
            all_match = all_match and ok
            lines.append(f"- {rel}: {'IDENTICAL' if ok else 'DIVERGED'}")
        lines.append("")
        lines.append(f"Overall: {'PASS - byte-identical across runs' if all_match else 'FAIL - some files diverged'}")
        lines.append("")

    _write_text("\n".join(lines) + "\n", OUTPUT_DIR / "run_manifest.txt")
    return dict(hashes)


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------


def run_pipeline(*, run_ordinal: int, prior_hashes: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[run #{run_ordinal}] Loading + hash-verifying inputs ...", flush=True)
    (sig_to_arr, first_sl_bar, r_at_t240, n_post_bars, cluster_id_map, fold_id_map,
     pair_map, actual_hashes, n_total) = load_inputs()
    sigs_sorted: List[int] = sorted(sig_to_arr.keys())
    fold_counts: Dict[int, int] = {f: 0 for f in range(1, 8)}
    for s in sigs_sorted:
        fold_counts[fold_id_map.get(s, 0)] = fold_counts.get(fold_id_map.get(s, 0), 0) + 1
    print(f"[run #{run_ordinal}] N total = {n_total}; per-fold counts = {fold_counts}", flush=True)

    # ----- Reference cells (band_behavior parity check) -----
    print(f"[run #{run_ordinal}] Simulating reference cells ...", flush=True)
    ref_rows: List[Dict[str, Any]] = []
    ref_per_cluster_rows: List[Dict[str, Any]] = []
    ref_per_fold_rows: List[Dict[str, Any]] = []
    ref_observed: Dict[str, float] = {}
    ref_parity_ok: Dict[str, bool] = {}
    ref_realized: Dict[str, np.ndarray] = {}
    for ref_mode in ("verbatim_arc2", "no_time_exit"):
        _, reason_idx, realized = simulate_reference(sigs_sorted, sig_to_arr, first_sl_bar, ref_mode)
        ref_realized[ref_mode] = realized
        obs = float(np.mean(realized))
        target = BAND_BEHAVIOR_REF_MEAN_R[ref_mode]
        ok = abs(obs - target) <= REF_TOLERANCE
        ref_observed[ref_mode] = obs
        ref_parity_ok[ref_mode] = ok
        if not ok:
            raise RuntimeError(
                f"BLOCKER: reference {ref_mode} mean R = {obs:.6f} differs from band_behavior "
                f"target {target:.4f} by {abs(obs - target):.6f} (tolerance {REF_TOLERANCE})."
            )
        base = {"variant": "reference", "cell_id": ref_mode}
        ref_rows.extend(cell_stats_rows(np.zeros(n_total, dtype=np.int32),  # exit_t placeholder unused in reason-counts
                                         reason_idx, realized, base))
        # Re-add exit_t properly:
        ref_rows = [r for r in ref_rows if r.get("stat") != "mean_exit_bar" or r.get("cell_id") != ref_mode or r.get("value") is None]
        # Recompute mean_exit_bar with actual exit_t:
        et_array, _, _ = simulate_reference(sigs_sorted, sig_to_arr, first_sl_bar, ref_mode)
        # Replace mean_exit_bar row(s) for this ref_mode:
        # (Simpler: regenerate full rows with correct exit_t.)
    # Re-do cleanly:
    ref_rows = []
    for ref_mode in ("verbatim_arc2", "no_time_exit"):
        et_array, reason_idx, realized = simulate_reference(sigs_sorted, sig_to_arr, first_sl_bar, ref_mode)
        base = {"variant": "reference", "cell_id": ref_mode}
        ref_rows.extend(cell_stats_rows(et_array, reason_idx, realized, base))
        ref_per_cluster_rows.extend(per_cluster_rows(base, sigs_sorted, cluster_id_map, realized))
        ref_per_fold_rows.extend(per_fold_rows(base, sigs_sorted, fold_id_map, realized))

    # ----- Variant 1 -----
    print(f"[run #{run_ordinal}] Variant 1 — 25 single-bar gate cells ...", flush=True)
    v1_rows: List[Dict[str, Any]] = []
    v1_cluster_rows: List[Dict[str, Any]] = []
    v1_fold_rows: List[Dict[str, Any]] = []
    v1_realized: Dict[str, np.ndarray] = {}
    for t_star in V1_T_STARS:
        for T in V1_THRESHOLDS:
            cell_id = f"V1_t{t_star}_T{T:g}"
            et, ri, rr = simulate_v1(sigs_sorted, sig_to_arr, first_sl_bar, t_star, T)
            base = {"variant": "V1_single_bar", "cell_id": cell_id, "t_star": t_star, "T": T}
            v1_rows.extend(cell_stats_rows(et, ri, rr, base))
            v1_cluster_rows.extend(per_cluster_rows(base, sigs_sorted, cluster_id_map, rr))
            v1_fold_rows.extend(per_fold_rows(base, sigs_sorted, fold_id_map, rr))
            v1_realized[cell_id] = rr

    # ----- Variant 2 -----
    print(f"[run #{run_ordinal}] Variant 2 — composite classifier, 5 cells ...", flush=True)
    v2_rows, v2_cluster_rows, v2_fold_rows, classifier_meta = variant_2_pipeline(
        sigs_sorted, sig_to_arr, first_sl_bar, r_at_t240, cluster_id_map, fold_id_map,
    )

    # ----- Variant 3 -----
    print(f"[run #{run_ordinal}] Variant 3 — 4 two-bar cells ...", flush=True)
    v3_rows: List[Dict[str, Any]] = []
    v3_cluster_rows: List[Dict[str, Any]] = []
    v3_fold_rows: List[Dict[str, Any]] = []
    for t1, t2 in V3_PAIRS:
        cell_id = f"V3_t{t1}_t{t2}"
        et, ri, rr = simulate_v3(sigs_sorted, sig_to_arr, first_sl_bar, t1, t2, V3_T1, V3_T2)
        base = {"variant": "V3_two_bar", "cell_id": cell_id, "t1": t1, "t2": t2, "T1": V3_T1, "T2": V3_T2}
        v3_rows.extend(cell_stats_rows(et, ri, rr, base))
        v3_cluster_rows.extend(per_cluster_rows(base, sigs_sorted, cluster_id_map, rr))
        v3_fold_rows.extend(per_fold_rows(base, sigs_sorted, fold_id_map, rr))

    # ----- Variant 4 -----
    print(f"[run #{run_ordinal}] Variant 4 — 25 MFE-skip cells ...", flush=True)
    v4_rows: List[Dict[str, Any]] = []
    v4_cluster_rows: List[Dict[str, Any]] = []
    v4_fold_rows: List[Dict[str, Any]] = []
    for t_star in V4_T_STARS:
        for T in V4_THRESHOLDS:
            cell_id = f"V4_t{t_star}_T{T:g}"
            et, ri, rr, skipped = simulate_v4(sigs_sorted, sig_to_arr, first_sl_bar, t_star, T, V4_MFE_SKIP_LEVEL)
            base = {"variant": "V4_mfe_skip", "cell_id": cell_id, "t_star": t_star, "T": T}
            cell_rows = cell_stats_rows(et, ri, rr, base)
            cell_rows.append({**base, "stat": "n_skipped_by_mfe", "value": int(skipped.sum())})
            v4_rows.extend(cell_rows)
            v4_cluster_rows.extend(per_cluster_rows(base, sigs_sorted, cluster_id_map, rr))
            v4_fold_rows.extend(per_fold_rows(base, sigs_sorted, fold_id_map, rr))

    # ----- SL wick analysis -----
    print(f"[run #{run_ordinal}] SL wick analysis + close-confirmed simulation ...", flush=True)
    sl_wick_df, sl_wick_sim_df, realized_cc = sl_wick_analysis(
        sigs_sorted, sig_to_arr, first_sl_bar, cluster_id_map, fold_id_map,
    )

    # ----- Pack DataFrames -----
    grid_v1 = pd.DataFrame(v1_rows).sort_values(by=["cell_id", "stat"], kind="mergesort").reset_index(drop=True)
    grid_v2 = pd.DataFrame(v2_rows).sort_values(by=["cell_id", "stat"], kind="mergesort").reset_index(drop=True)
    grid_v3 = pd.DataFrame(v3_rows).sort_values(by=["cell_id", "stat"], kind="mergesort").reset_index(drop=True)
    grid_v4 = pd.DataFrame(v4_rows).sort_values(by=["cell_id", "stat"], kind="mergesort").reset_index(drop=True)
    grid_refs = pd.DataFrame(ref_rows).sort_values(by=["cell_id", "stat"], kind="mergesort").reset_index(drop=True)

    # Combined per-cluster table: top-5 by mean realized R per variant + 2 refs.
    def top5_cells(grid: pd.DataFrame) -> List[str]:
        m = grid[grid["stat"] == "mean_realized_r"].copy()
        m["value_f"] = m["value"].astype(float)
        return m.sort_values("value_f", ascending=False)["cell_id"].head(5).tolist()

    cluster_combined: List[pd.DataFrame] = []
    for v_label, rows, grid in (
        ("V1", v1_cluster_rows, grid_v1),
        ("V2", v2_cluster_rows, grid_v2),
        ("V3", v3_cluster_rows, grid_v3),
        ("V4", v4_cluster_rows, grid_v4),
    ):
        df = pd.DataFrame(rows)
        if not len(df):
            continue
        top = top5_cells(grid)
        cluster_combined.append(df[df["cell_id"].isin(top)])
    ref_df = pd.DataFrame(ref_per_cluster_rows)
    if len(ref_df):
        cluster_combined.append(ref_df)
    per_cluster_df = pd.concat(cluster_combined, ignore_index=True).sort_values(
        by=["variant", "cell_id", "cluster", "stat"], kind="mergesort"
    ).reset_index(drop=True)

    # Combined per-fold table: ALL cells across V1..V4 + refs.
    per_fold_df = pd.concat([
        pd.DataFrame(v1_fold_rows),
        pd.DataFrame(v2_fold_rows),
        pd.DataFrame(v3_fold_rows),
        pd.DataFrame(v4_fold_rows),
        pd.DataFrame(ref_per_fold_rows),
    ], ignore_index=True).sort_values(by=["variant", "cell_id", "fold", "stat"], kind="mergesort").reset_index(drop=True)

    # ----- Write outputs -----
    print(f"[run #{run_ordinal}] Writing CSV outputs ...", flush=True)
    out_paths: List[Path] = []
    _write_csv(grid_v1, OUTPUT_DIR / "gate_grid_single_bar.csv"); out_paths.append(OUTPUT_DIR / "gate_grid_single_bar.csv")
    _write_csv(grid_v2, OUTPUT_DIR / "gate_grid_composite.csv"); out_paths.append(OUTPUT_DIR / "gate_grid_composite.csv")
    _write_csv(grid_v3, OUTPUT_DIR / "gate_grid_two_bar.csv"); out_paths.append(OUTPUT_DIR / "gate_grid_two_bar.csv")
    _write_csv(grid_v4, OUTPUT_DIR / "gate_grid_mfe_skip.csv"); out_paths.append(OUTPUT_DIR / "gate_grid_mfe_skip.csv")
    _write_csv(grid_refs, OUTPUT_DIR / "gate_reference_cells.csv"); out_paths.append(OUTPUT_DIR / "gate_reference_cells.csv")
    _write_csv(per_cluster_df, OUTPUT_DIR / "gate_cell_per_cluster.csv"); out_paths.append(OUTPUT_DIR / "gate_cell_per_cluster.csv")
    _write_csv(per_fold_df, OUTPUT_DIR / "gate_cell_per_fold.csv"); out_paths.append(OUTPUT_DIR / "gate_cell_per_fold.csv")
    _write_csv(sl_wick_df, OUTPUT_DIR / "sl_wick_analysis.csv"); out_paths.append(OUTPUT_DIR / "sl_wick_analysis.csv")
    _write_csv(sl_wick_sim_df, OUTPUT_DIR / "sl_wick_simulation.csv"); out_paths.append(OUTPUT_DIR / "sl_wick_simulation.csv")

    # ----- Charts -----
    print(f"[run #{run_ordinal}] Writing charts ...", flush=True)
    # V1 mean R heatmap.
    mean_grid_v1 = np.full((len(V1_T_STARS), len(V1_THRESHOLDS)), np.nan)
    n_grid_v1 = np.full((len(V1_T_STARS), len(V1_THRESHOLDS)), 0.0)
    for i, t in enumerate(V1_T_STARS):
        for j, T in enumerate(V1_THRESHOLDS):
            cid = f"V1_t{t}_T{T:g}"
            m = grid_v1[(grid_v1["cell_id"] == cid) & (grid_v1["stat"] == "mean_realized_r")]
            if len(m):
                mean_grid_v1[i, j] = float(m["value"].iloc[0])
            n = grid_v1[(grid_v1["cell_id"] == cid) & (grid_v1["stat"] == "N")]
            if len(n):
                n_grid_v1[i, j] = float(n["value"].iloc[0])
    chart_heatmap(
        mean_grid_v1, n_grid_v1,
        [f"t*={t}" for t in V1_T_STARS],
        [f"T={T:g}" for T in V1_THRESHOLDS],
        "V1 single-bar gate — mean realized R", "threshold T", "anchor bar t*",
        "mean realized R", CHARTS_DIR / "gate_meanR_heatmap.png", diverging=True,
    )
    out_paths.append(CHARTS_DIR / "gate_meanR_heatmap.png")
    # V1 win-rate heatmap.
    win_grid = np.full((len(V1_T_STARS), len(V1_THRESHOLDS)), np.nan)
    for i, t in enumerate(V1_T_STARS):
        for j, T in enumerate(V1_THRESHOLDS):
            cid = f"V1_t{t}_T{T:g}"
            m = grid_v1[(grid_v1["cell_id"] == cid) & (grid_v1["stat"] == "p_realized_r_gt_0")]
            if len(m):
                win_grid[i, j] = float(m["value"].iloc[0])
    chart_heatmap(
        win_grid, n_grid_v1,
        [f"t*={t}" for t in V1_T_STARS],
        [f"T={T:g}" for T in V1_THRESHOLDS],
        "V1 single-bar gate — P(realized R > 0)", "threshold T", "anchor bar t*",
        "P(R > 0)", CHARTS_DIR / "gate_winrate_heatmap.png", diverging=False,
    )
    out_paths.append(CHARTS_DIR / "gate_winrate_heatmap.png")
    # V4 mean R heatmap.
    mean_grid_v4 = np.full((len(V4_T_STARS), len(V4_THRESHOLDS)), np.nan)
    n_grid_v4 = np.full((len(V4_T_STARS), len(V4_THRESHOLDS)), 0.0)
    for i, t in enumerate(V4_T_STARS):
        for j, T in enumerate(V4_THRESHOLDS):
            cid = f"V4_t{t}_T{T:g}"
            m = grid_v4[(grid_v4["cell_id"] == cid) & (grid_v4["stat"] == "mean_realized_r")]
            if len(m):
                mean_grid_v4[i, j] = float(m["value"].iloc[0])
            n = grid_v4[(grid_v4["cell_id"] == cid) & (grid_v4["stat"] == "N")]
            if len(n):
                n_grid_v4[i, j] = float(n["value"].iloc[0])
    chart_heatmap(
        mean_grid_v4, n_grid_v4,
        [f"t*={t}" for t in V4_T_STARS],
        [f"T={T:g}" for T in V4_THRESHOLDS],
        "V4 MFE-skip gate — mean realized R", "threshold T", "anchor bar t*",
        "mean realized R", CHARTS_DIR / "gate_meanR_with_mfe_skip.png", diverging=True,
    )
    out_paths.append(CHARTS_DIR / "gate_meanR_with_mfe_skip.png")
    # Top-10 cells per-fold heatmap.
    worst_df = per_fold_df[per_fold_df["fold"] == "WORST"].copy()
    worst_df["value_f"] = worst_df["value"].astype(float)
    top10_cells = worst_df.sort_values("value_f", ascending=False)["cell_id"].head(10).tolist()
    chart_per_fold_heatmap(per_fold_df, top10_cells, CHARTS_DIR / "gate_per_fold_heatmap.png")
    out_paths.append(CHARTS_DIR / "gate_per_fold_heatmap.png")
    # Two-bar comparison.
    chart_two_bar_comparison(grid_v1, grid_v3, CHARTS_DIR / "two_bar_comparison.png")
    out_paths.append(CHARTS_DIR / "two_bar_comparison.png")
    # SL wick recovery.
    chart_sl_wick_recovery(realized_cc, ref_realized["no_time_exit"], CHARTS_DIR / "sl_wick_recovery.png")
    out_paths.append(CHARTS_DIR / "sl_wick_recovery.png")

    # ----- Report -----
    print(f"[run #{run_ordinal}] Writing report ...", flush=True)
    write_report(
        actual_hashes=actual_hashes,
        n_total=n_total,
        fold_counts=fold_counts,
        grid_v1=grid_v1, grid_v2=grid_v2, grid_v3=grid_v3, grid_v4=grid_v4,
        grid_refs=grid_refs,
        per_cluster_df=per_cluster_df,
        per_fold_df=per_fold_df,
        sl_wick_df=sl_wick_df,
        sl_wick_sim_df=sl_wick_sim_df,
        classifier_meta=classifier_meta,
        ref_parity_ok=ref_parity_ok,
        path=OUTPUT_DIR / "gate_exit_report.md",
    )
    out_paths.append(OUTPUT_DIR / "gate_exit_report.md")

    # ----- Manifest -----
    print(f"[run #{run_ordinal}] Writing manifest ...", flush=True)
    hashes = write_manifest(
        out_paths=out_paths,
        actual_input_hashes=actual_hashes,
        n_total=n_total,
        fold_counts=fold_counts,
        classifier_meta=classifier_meta,
        ref_parity_ok=ref_parity_ok,
        ref_observed=ref_observed,
        prior_hashes=prior_hashes,
        run_ordinal=run_ordinal,
    )
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description="Arc 2 Gate Exit Simulation")
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
