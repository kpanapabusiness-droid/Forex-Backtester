"""Arc 2 Band Behavior Follow-up — three descriptive analyses on top of the appendix.

Phase: l6_arc2_band_behavior (additive; descriptive only per L6.0 v1.1 §14.5).

Reads (read-only):
- results/l6/arc2/trajectory_appendix/trajectory_panel.parquet
- results/l6/arc2/trajectory_appendix/shape_features.csv
Both must hash to the values recorded in run_manifest.txt (Arc 2 trajectory
appendix). HALT on mismatch.

Writes:
- results/l6/arc2/trajectory_appendix/band_behavior/

Analyses:
- A. Per-band outcomes — for each MFE band X in {1..7}R, distributions of
  peak_mfe_r, r_at_t240, giveback_from_peak, subsequent_mae/mfe after first
  touch, r_at_first_X_touch_plus_{24,48,120}, bars_from_first_X_touch_to_peak,
  bars_from_peak_to_t240, and P(reach X+1|X), P(reach X+2|X).
- B. Intra-trade givebacks — max-dd-from-running-peak after / before first
  touch at each X in {1..5}R, plus per-trade pullback event counts at
  thresholds {0.5, 1.0, 1.5, 2.0}R via a simple zigzag detector.
- C. Trail simulation grid — for each (X, G) with X in {1..5}R and
  G in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}R, walk bars 1..240 applying
  (hard SL -1R intrabar) + (trail activates at first running_mfe >= X, level =
  peak - G, tighten-then-check ordering). Plus 3 reference cells: verbatim
  Arc 2 (SL + t=120 close), no-time-exit (SL + r_close[240]), hard TP=2R.
- D. First-touch 2R-up subpopulation — trades with first_touch_2r_dir == +1.
  Outcome distributions overall and per (c05,c1,c2) cell where c2 = up.

R-multiple convention: 1R = 2 x Wilder ATR(14) at bar N close. R denominator
unchanged from the parent appendix (already encoded in panel/shape).

Determinism: numpy seed locked; deterministic sort order; CSV LF endings;
matplotlib Agg with pinned metadata; two consecutive runs assert byte-
identicality (PNG pixel-only fallback documented per PNG metadata quirks).

§14.5 discipline: band_behavior_report.md is empirical only. Action-shaped
observations append to ../CANDIDATE_HYPOTHESES_DRAFT.md by hand, not by this
script.
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
import matplotlib.pyplot as plt  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402
from scipy import stats as sps  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# LOCKED CONSTANTS
# ---------------------------------------------------------------------------

NUMPY_SEED: int = 20260512

APPENDIX_DIR: Path = REPO_ROOT / "results" / "l6" / "arc2" / "trajectory_appendix"
INPUT_PANEL: Path = APPENDIX_DIR / "trajectory_panel.parquet"
INPUT_SHAPE: Path = APPENDIX_DIR / "shape_features.csv"

OUTPUT_DIR: Path = APPENDIX_DIR / "band_behavior"
CHARTS_DIR: Path = OUTPUT_DIR / "charts"

# Expected input hashes (from the parent appendix's run_manifest.txt).
EXPECTED_INPUT_HASHES: Dict[str, str] = {
    "trajectory_panel.parquet":
        "c8f1ec4825ada7b3a9efaf72101e79b067f7056117bf70aaae2f5766261398ed",
    "shape_features.csv":
        "6920c990628dd2d769cfbc56bd006391f6b9a24beea64f52f5bda9da02d44aaa",
}

TRAJ_HORIZON: int = 240
ANALYSIS_A_BANDS: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
ANALYSIS_B_BANDS: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)
TRAIL_X_GRID: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)
TRAIL_G_GRID: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
PULLBACK_THRESHOLDS: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)

# Distributional spec (same as parent appendix; histograms widened for giveback).
DIST_PERCENTILES: Tuple[float, ...] = (1, 5, 10, 25, 50, 75, 90, 95, 99)
HIST_BIN_EDGES_DEFAULT: np.ndarray = np.concatenate([
    [-np.inf],
    np.round(np.arange(-3.0, 12.0 + 0.25 / 2, 0.25), 4),
    [np.inf],
])
# Giveback is non-negative; widen to 0..15R.
HIST_BIN_EDGES_GIVEBACK: np.ndarray = np.concatenate([
    np.round(np.arange(0.0, 15.0 + 0.25 / 2, 0.25), 4),
    [np.inf],
])


# ---------------------------------------------------------------------------
# Filesystem helpers (deterministic writes; mirrors the parent appendix).
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


def _write_csv(df: pd.DataFrame, path: Path, *, float_format: str = "%.10g") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n", float_format=float_format)
    path.write_bytes(buf.getvalue().encode("utf-8"))


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Software": "matplotlib", "Creation Time": "2020-01-01T00:00:00Z"}
    fig.savefig(path, format="png", dpi=120, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Distributional spec helpers.
# ---------------------------------------------------------------------------


def describe_distribution(
    values: np.ndarray,
    label: str,
    *,
    hist_edges: np.ndarray = HIST_BIN_EDGES_DEFAULT,
) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    out: Dict[str, Any] = {"label": label, "n": n}
    if n == 0:
        for k in ("mean", "std", "skew", "excess_kurt", "min", "max"):
            out[k] = float("nan")
        for p in DIST_PERCENTILES:
            out[f"p{int(p)}"] = float("nan")
        out["histogram_bin_edges"] = hist_edges.tolist()
        out["histogram_counts"] = [0] * (len(hist_edges) - 1)
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
    counts, _ = np.histogram(arr, bins=hist_edges)
    out["histogram_bin_edges"] = hist_edges.tolist()
    out["histogram_counts"] = counts.astype(int).tolist()
    return out


def _fmt_float(v: float, decimals: int = 4) -> str:
    if not np.isfinite(v):
        return "NaN"
    return f"{v:.{decimals}f}"


def render_distribution_md(desc: Dict[str, Any], *, header_level: int = 3) -> str:
    lvl = "#" * header_level
    lines: List[str] = [f"{lvl} {desc['label']}  (N = {desc['n']})", ""]
    lines.append("Moments:")
    lines.append("")
    lines.append("| mean | std | skew | excess_kurt | min | max |")
    lines.append("|------|-----|------|-------------|-----|-----|")
    lines.append(
        f"| {_fmt_float(desc['mean'])} | {_fmt_float(desc['std'])} | "
        f"{_fmt_float(desc['skew'])} | {_fmt_float(desc['excess_kurt'])} | "
        f"{_fmt_float(desc['min'])} | {_fmt_float(desc['max'])} |"
    )
    lines.append("")
    lines.append("Percentiles:")
    lines.append("")
    pct_hdr = " | ".join([f"p{int(p)}" for p in DIST_PERCENTILES])
    pct_sep = " | ".join(["---"] * len(DIST_PERCENTILES))
    pct_row = " | ".join([_fmt_float(desc[f"p{int(p)}"]) for p in DIST_PERCENTILES])
    lines.append(f"| {pct_hdr} |")
    lines.append(f"| {pct_sep} |")
    lines.append(f"| {pct_row} |")
    lines.append("")
    lines.append("Histogram:")
    lines.append("")
    lines.append("| bin_lo | bin_hi | count |")
    lines.append("|--------|--------|-------|")
    edges = desc["histogram_bin_edges"]
    counts = desc["histogram_counts"]
    for i, c in enumerate(counts):
        lo = edges[i]
        hi = edges[i + 1]
        lo_s = "-inf" if not np.isfinite(lo) else f"{lo:.2f}"
        hi_s = "inf" if not np.isfinite(hi) else f"{hi:.2f}"
        lines.append(f"| {lo_s} | {hi_s} | {c} |")
    lines.append("")
    return "\n".join(lines)


def desc_to_long_rows(
    desc: Dict[str, Any], extra_cols: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Render a distribution descriptor as long-format rows (one per stat)."""
    rows: List[Dict[str, Any]] = []
    base = {**extra_cols, "metric_label": desc["label"], "n": desc["n"]}
    for stat in ("mean", "std", "skew", "excess_kurt", "min", "max"):
        rows.append({**base, "stat": stat, "value": desc[stat]})
    for p in DIST_PERCENTILES:
        rows.append({**base, "stat": f"p{int(p)}", "value": desc[f"p{int(p)}"]})
    edges = desc["histogram_bin_edges"]
    counts = desc["histogram_counts"]
    for i, c in enumerate(counts):
        lo = edges[i]
        hi = edges[i + 1]
        lo_s = "-inf" if not np.isfinite(lo) else f"{lo:.4f}"
        hi_s = "inf" if not np.isfinite(hi) else f"{hi:.4f}"
        rows.append({**base, "stat": f"hist[{lo_s},{hi_s})", "value": float(c)})
    return rows


# ---------------------------------------------------------------------------
# Input loading + hash verify.
# ---------------------------------------------------------------------------


def load_inputs() -> Tuple[Dict[int, Dict[str, np.ndarray]], pd.DataFrame, Dict[str, str]]:
    """Load panel + shape; verify hashes; return per-signal arrays and shape df."""
    actual_hashes: Dict[str, str] = {
        "trajectory_panel.parquet": _sha256_file(INPUT_PANEL),
        "shape_features.csv": _sha256_file(INPUT_SHAPE),
    }
    for fname, expected in EXPECTED_INPUT_HASHES.items():
        actual = actual_hashes[fname]
        if actual != expected:
            raise RuntimeError(
                f"Input hash mismatch for {fname}: expected {expected}, got {actual}. "
                "HALT — the upstream appendix outputs have changed; re-verify before proceeding."
            )

    panel_table = pq.read_table(INPUT_PANEL)
    panel = panel_table.to_pandas()
    panel = panel.sort_values(["signal_idx", "t"], kind="mergesort").reset_index(drop=True)

    shape = pd.read_csv(INPUT_SHAPE, low_memory=False)
    shape = shape.sort_values("signal_idx", kind="mergesort").reset_index(drop=True)

    # Build per-signal arrays: dict[int signal_idx] -> {'t', 'r_close', 'r_high', 'r_low',
    #                                                   'running_mfe', 'running_mae'}
    sig_to_arr: Dict[int, Dict[str, np.ndarray]] = {}
    for s_idx, grp in panel.groupby("signal_idx", sort=True):
        # t in panel ranges 0..n_post_bars; we extract bars 1..n (drop t=0 anchor for the run).
        t_arr = grp["t"].to_numpy(dtype=np.int32)
        keep = t_arr >= 1
        if not keep.any():
            continue
        sig_to_arr[int(s_idx)] = {
            "t": t_arr[keep],
            "r_close": grp["r_close"].to_numpy(dtype=np.float64)[keep],
            "r_high": grp["r_high"].to_numpy(dtype=np.float64)[keep],
            "r_low": grp["r_low"].to_numpy(dtype=np.float64)[keep],
            "running_mfe": grp["running_mfe"].to_numpy(dtype=np.float64)[keep],
            "running_mae": grp["running_mae"].to_numpy(dtype=np.float64)[keep],
        }
    return sig_to_arr, shape, actual_hashes


# ---------------------------------------------------------------------------
# Analysis A — Per-band outcomes.
# ---------------------------------------------------------------------------


def _first_idx_ge(arr: np.ndarray, threshold: float) -> Optional[int]:
    mask = arr >= threshold
    if not mask.any():
        return None
    return int(np.argmax(mask))


def analysis_a(
    sig_to_arr: Dict[int, Dict[str, np.ndarray]],
    shape: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Per-band metric distributions. Returns (long_df, [descriptors])."""
    long_rows: List[Dict[str, Any]] = []
    descriptors: List[Dict[str, Any]] = []

    shape_idx = shape.set_index("signal_idx")
    peak_mfe_lookup = shape_idx["peak_mfe_r"].to_dict()
    r240_lookup = shape_idx["r_at_t240"].to_dict()
    peak_mfe_bar_lookup = shape_idx["peak_mfe_bar"].to_dict()

    chain_probs: List[Dict[str, Any]] = []

    for X in ANALYSIS_A_BANDS:
        # Subpopulation: trades with peak_mfe_r >= X.
        per_trade: Dict[str, List[float]] = {
            "peak_mfe_r": [],
            "r_at_t240": [],
            "giveback_from_peak": [],
            "subsequent_mae_after_first_X_touch": [],
            "subsequent_max_mfe_after_first_X_touch": [],
            "r_at_first_X_touch_plus_24": [],
            "r_at_first_X_touch_plus_48": [],
            "r_at_first_X_touch_plus_120": [],
            "bars_from_first_X_touch_to_peak": [],
            "bars_from_peak_to_t240": [],
        }
        n_reached_next1 = 0
        n_reached_next2 = 0

        for sig_idx, arrs in sig_to_arr.items():
            run_mfe = arrs["running_mfe"]
            r_close = arrs["r_close"]
            r_high = arrs["r_high"]
            r_low = arrs["r_low"]
            peak_mfe_r = peak_mfe_lookup.get(sig_idx, np.nan)
            if not np.isfinite(peak_mfe_r) or peak_mfe_r < X:
                continue
            t_x_idx = _first_idx_ge(run_mfe, X)  # 0-based position in arrs (corresponds to t=t_x_idx+1)
            if t_x_idx is None:
                continue
            t_x_bar = int(t_x_idx + 1)  # 1-based bar

            per_trade["peak_mfe_r"].append(float(peak_mfe_r))

            r240 = r240_lookup.get(sig_idx, np.nan)
            per_trade["r_at_t240"].append(float(r240))

            giveback = float(peak_mfe_r - r240) if np.isfinite(r240) else np.nan
            per_trade["giveback_from_peak"].append(giveback)

            sub_mae = float(np.min(r_low[t_x_idx:]))
            per_trade["subsequent_mae_after_first_X_touch"].append(sub_mae)

            sub_mfe = float(np.max(r_high[t_x_idx:]))
            per_trade["subsequent_max_mfe_after_first_X_touch"].append(sub_mfe)

            def r_at_offset(offset_bars: int) -> float:
                target_bar = t_x_bar + offset_bars  # 1-based
                target_idx = target_bar - 1  # 0-based in arrs
                if target_idx < len(r_close):
                    return float(r_close[target_idx])
                return np.nan

            per_trade["r_at_first_X_touch_plus_24"].append(r_at_offset(24))
            per_trade["r_at_first_X_touch_plus_48"].append(r_at_offset(48))
            per_trade["r_at_first_X_touch_plus_120"].append(r_at_offset(120))

            peak_bar = int(peak_mfe_bar_lookup.get(sig_idx, t_x_bar))
            per_trade["bars_from_first_X_touch_to_peak"].append(float(peak_bar - t_x_bar))
            per_trade["bars_from_peak_to_t240"].append(float(TRAJ_HORIZON - peak_bar))

            # Chain probs (relative to this subpop).
            if peak_mfe_r >= X + 1.0:
                n_reached_next1 += 1
            if peak_mfe_r >= X + 2.0:
                n_reached_next2 += 1

        n_sub = len(per_trade["peak_mfe_r"])
        if n_sub == 0:
            continue

        for metric_name, vals in per_trade.items():
            arr = np.asarray(vals, dtype=np.float64)
            desc = describe_distribution(arr, f"X={X:g}R | {metric_name}")
            descriptors.append(desc)
            for row in desc_to_long_rows(desc, {"band_R": X, "metric": metric_name}):
                long_rows.append(row)

        chain_probs.append({
            "band_R": X,
            "n_reached": n_sub,
            "p_reach_X_plus_1_given_X": (n_reached_next1 / n_sub) if n_sub else float("nan"),
            "p_reach_X_plus_2_given_X": (n_reached_next2 / n_sub) if n_sub else float("nan"),
            "n_reached_X_plus_1": n_reached_next1,
            "n_reached_X_plus_2": n_reached_next2,
        })

    long_df = pd.DataFrame(long_rows)
    long_df = long_df.sort_values(by=["band_R", "metric", "stat"], kind="mergesort").reset_index(drop=True)
    chain_df = pd.DataFrame(chain_probs).sort_values("band_R", kind="mergesort").reset_index(drop=True)
    return long_df, descriptors, chain_df


# ---------------------------------------------------------------------------
# Analysis B — Intra-trade givebacks + pullback counts.
# ---------------------------------------------------------------------------


def analysis_b_givebacks(
    sig_to_arr: Dict[int, Dict[str, np.ndarray]],
    shape: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Intra-trade max drawdown-from-running-peak distributions after / before first X touch."""
    shape_idx = shape.set_index("signal_idx")
    peak_mfe_lookup = shape_idx["peak_mfe_r"].to_dict()

    long_after_rows: List[Dict[str, Any]] = []
    long_before_rows: List[Dict[str, Any]] = []
    after_descs: List[Dict[str, Any]] = []
    before_descs: List[Dict[str, Any]] = []

    for X in ANALYSIS_B_BANDS:
        after_vals: List[float] = []
        before_vals: List[float] = []
        for sig_idx, arrs in sig_to_arr.items():
            peak_mfe_r = peak_mfe_lookup.get(sig_idx, np.nan)
            if not np.isfinite(peak_mfe_r) or peak_mfe_r < X:
                continue
            run_mfe = arrs["running_mfe"]
            r_high = arrs["r_high"]
            r_low = arrs["r_low"]
            t_x_idx = _first_idx_ge(run_mfe, X)
            if t_x_idx is None:
                continue

            # After first X touch.
            after_high = r_high[t_x_idx:]
            after_low = r_low[t_x_idx:]
            if after_high.size:
                running_peak_after = np.maximum.accumulate(after_high)
                dd_after = running_peak_after - after_low
                after_vals.append(float(np.max(dd_after)))
            else:
                after_vals.append(0.0)

            # Before first X touch (bars 1..t_x_idx exclusive; i.e., array indices 0..t_x_idx-1).
            before_high = r_high[:t_x_idx]
            before_low = r_low[:t_x_idx]
            if before_high.size:
                running_peak_before = np.maximum.accumulate(before_high)
                # Pre-pad with 0 to honor "peak from entry" (which starts at 0 at t=0).
                running_peak_before = np.maximum(running_peak_before, 0.0)
                dd_before = running_peak_before - before_low
                before_vals.append(float(np.max(dd_before)))
            else:
                before_vals.append(0.0)

        after_arr = np.asarray(after_vals, dtype=np.float64)
        before_arr = np.asarray(before_vals, dtype=np.float64)
        a_desc = describe_distribution(
            after_arr, f"X={X:g}R | max_dd_from_peak_after_X",
            hist_edges=HIST_BIN_EDGES_GIVEBACK,
        )
        b_desc = describe_distribution(
            before_arr, f"X={X:g}R | max_dd_from_peak_before_X",
            hist_edges=HIST_BIN_EDGES_GIVEBACK,
        )
        after_descs.append(a_desc)
        before_descs.append(b_desc)
        for row in desc_to_long_rows(a_desc, {"band_R": X}):
            long_after_rows.append(row)
        for row in desc_to_long_rows(b_desc, {"band_R": X}):
            long_before_rows.append(row)

    after_df = pd.DataFrame(long_after_rows).sort_values(
        by=["band_R", "stat"], kind="mergesort"
    ).reset_index(drop=True)
    before_df = pd.DataFrame(long_before_rows).sort_values(
        by=["band_R", "stat"], kind="mergesort"
    ).reset_index(drop=True)
    return after_df, before_df, after_descs, before_descs


def _count_pullback_events(
    r_high: np.ndarray, r_low: np.ndarray, threshold: float
) -> int:
    """Zigzag pullback counter.

    Walks the bars maintaining a running extreme. In 'up' state, the extreme
    is the peak r_high; a pullback event registers when (peak - r_low[t]) >=
    threshold. Then state flips to 'down' and the trough becomes r_low[t].
    In 'down' state, when (r_high[t] - trough) >= threshold/2, state flips
    back to 'up' (no event registered for recovery). Only down-moves count.
    """
    n = len(r_high)
    if n == 0:
        return 0
    pull_t = threshold
    recover_t = threshold / 2.0
    direction = "up"
    extreme = float(r_high[0])
    count = 0
    for t in range(n):
        if direction == "up":
            if r_high[t] > extreme:
                extreme = float(r_high[t])
            elif extreme - r_low[t] >= pull_t:
                count += 1
                direction = "down"
                extreme = float(r_low[t])
        else:
            if r_low[t] < extreme:
                extreme = float(r_low[t])
            elif r_high[t] - extreme >= recover_t:
                direction = "up"
                extreme = float(r_high[t])
    return count


def analysis_b_pullbacks(
    sig_to_arr: Dict[int, Dict[str, np.ndarray]],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Per-trade pullback event counts at 4 thresholds, plus aggregate stats."""
    sigs_sorted = sorted(sig_to_arr.keys())
    per_trade_rows: List[Dict[str, Any]] = []
    by_threshold: Dict[float, List[int]] = {t: [] for t in PULLBACK_THRESHOLDS}
    for sig_idx in sigs_sorted:
        arrs = sig_to_arr[sig_idx]
        r_high = arrs["r_high"]
        r_low = arrs["r_low"]
        row: Dict[str, Any] = {"signal_idx": int(sig_idx)}
        for thr in PULLBACK_THRESHOLDS:
            c = _count_pullback_events(r_high, r_low, thr)
            row[f"n_pullbacks_ge_{thr:g}R"] = int(c)
            by_threshold[thr].append(c)
        per_trade_rows.append(row)
    per_trade_df = pd.DataFrame(per_trade_rows).sort_values(
        "signal_idx", kind="mergesort"
    ).reset_index(drop=True)

    descriptors: List[Dict[str, Any]] = []
    for thr, vals in by_threshold.items():
        arr = np.asarray(vals, dtype=np.float64)
        desc = describe_distribution(arr, f"n_pullbacks_ge_{thr:g}R (per trade)")
        descriptors.append(desc)
    return per_trade_df, descriptors


# ---------------------------------------------------------------------------
# Analysis C — Trail simulation grid.
# ---------------------------------------------------------------------------


def _simulate_trail(
    r_high: np.ndarray, r_low: np.ndarray, r_close: np.ndarray, X: float, G: float
) -> Tuple[int, str, float]:
    """Walks bars 1..N applying SL-throughout + trail-after-X.

    Order on each bar:
      1) SL check (r_low <= -1.0) — exit at -1.0
      2) If trail_active: tighten peak from r_high; trail_level = peak - G;
         check r_low <= trail_level — exit at trail_level
      3) Else: if r_high >= X, activate trail (peak = r_high[t], trail = peak - G)
    Time exit: if reaches end without exit, close at r_close[end].
    """
    n = len(r_high)
    trail_active = False
    peak = 0.0
    trail_level = -np.inf
    for t in range(n):
        if r_low[t] <= -1.0:
            return (t + 1, "sl", -1.0)
        if trail_active:
            if r_high[t] > peak:
                peak = float(r_high[t])
            trail_level = peak - G
            if r_low[t] <= trail_level:
                return (t + 1, "trail", float(trail_level))
        else:
            if r_high[t] >= X:
                trail_active = True
                peak = float(r_high[t])
                trail_level = peak - G
    if trail_active:
        return (n, "time_240", float(r_close[n - 1]))
    return (n, "never_reached_X", float(r_close[n - 1]))


def _simulate_reference(
    r_high: np.ndarray, r_low: np.ndarray, r_close: np.ndarray, mode: str
) -> Tuple[int, str, float]:
    """Reference exit rules:

    mode = 'verbatim_arc2'   -> SL -1R intrabar; if no SL by t=120, close at r_close[119].
    mode = 'no_time_exit'    -> SL -1R intrabar; if no SL by t=240, close at r_close[239].
    mode = 'hard_tp_2R'      -> SL -1R + first-touch +2R TP intrabar (SL-first); else r_close[239].
    """
    n = len(r_high)
    horizon = 120 if mode == "verbatim_arc2" else n
    horizon = min(horizon, n)
    for t in range(horizon):
        if r_low[t] <= -1.0:
            return (t + 1, "sl", -1.0)
        if mode == "hard_tp_2R" and r_high[t] >= 2.0:
            return (t + 1, "tp_2R", 2.0)
    exit_t = horizon
    return (exit_t, "time_exit", float(r_close[horizon - 1]))


def analysis_c(
    sig_to_arr: Dict[int, Dict[str, np.ndarray]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[float, float], np.ndarray]]:
    sigs_sorted = sorted(sig_to_arr.keys())

    grid_rows: List[Dict[str, Any]] = []
    realized_r_by_cell: Dict[Tuple[float, float], np.ndarray] = {}

    for X in TRAIL_X_GRID:
        for G in TRAIL_G_GRID:
            results: List[Tuple[int, str, float]] = []
            for sig_idx in sigs_sorted:
                a = sig_to_arr[sig_idx]
                results.append(_simulate_trail(a["r_high"], a["r_low"], a["r_close"], X, G))
            ts = np.asarray([r[0] for r in results], dtype=np.float64)
            reasons = np.asarray([r[1] for r in results], dtype=object)
            realized = np.asarray([r[2] for r in results], dtype=np.float64)
            realized_r_by_cell[(X, G)] = realized

            n_total = int(realized.size)
            counts = {k: int(np.sum(reasons == k)) for k in ("sl", "trail", "time_240", "never_reached_X")}

            base = {"X": X, "G": G}
            grid_rows.append({**base, "stat": "N", "value": n_total})
            for reason, c in counts.items():
                grid_rows.append({**base, "stat": f"n_{reason}", "value": c})
                grid_rows.append({
                    **base, "stat": f"p_{reason}",
                    "value": (c / n_total) if n_total else float("nan"),
                })
            grid_rows.append({**base, "stat": "mean_realized_r", "value": float(np.mean(realized))})
            grid_rows.append({
                **base, "stat": "std_realized_r",
                "value": float(np.std(realized, ddof=1)) if n_total > 1 else float("nan"),
            })
            grid_rows.append({
                **base, "stat": "skew_realized_r",
                "value": float(sps.skew(realized, bias=False)) if n_total > 2 else float("nan"),
            })
            grid_rows.append({
                **base, "stat": "excess_kurt_realized_r",
                "value": float(sps.kurtosis(realized, fisher=True, bias=False)) if n_total > 3 else float("nan"),
            })
            grid_rows.append({**base, "stat": "min_realized_r", "value": float(np.min(realized))})
            grid_rows.append({**base, "stat": "max_realized_r", "value": float(np.max(realized))})
            pcts = np.percentile(realized, list(DIST_PERCENTILES), method="linear")
            for p, v in zip(DIST_PERCENTILES, pcts):
                grid_rows.append({**base, "stat": f"p{int(p)}_realized_r", "value": float(v)})
            grid_rows.append({
                **base, "stat": "p_realized_r_gt_0",
                "value": float(np.mean(realized > 0.0)),
            })
            grid_rows.append({
                **base, "stat": "p_realized_r_gt_1",
                "value": float(np.mean(realized > 1.0)),
            })
            grid_rows.append({
                **base, "stat": "p_realized_r_gt_2",
                "value": float(np.mean(realized > 2.0)),
            })
            grid_rows.append({**base, "stat": "mean_exit_bar", "value": float(np.mean(ts))})
            # Histogram counts (long-format with bin labels embedded in stat name).
            hist_counts, _ = np.histogram(realized, bins=HIST_BIN_EDGES_DEFAULT)
            for i, c in enumerate(hist_counts):
                lo = HIST_BIN_EDGES_DEFAULT[i]
                hi = HIST_BIN_EDGES_DEFAULT[i + 1]
                lo_s = "-inf" if not np.isfinite(lo) else f"{lo:.4f}"
                hi_s = "inf" if not np.isfinite(hi) else f"{hi:.4f}"
                grid_rows.append({**base, "stat": f"hist[{lo_s},{hi_s})", "value": int(c)})

    grid_df = pd.DataFrame(grid_rows).sort_values(by=["X", "G", "stat"], kind="mergesort").reset_index(drop=True)

    # Reference cells.
    ref_rows: List[Dict[str, Any]] = []
    for ref_mode in ("verbatim_arc2", "no_time_exit", "hard_tp_2R"):
        results = []
        for sig_idx in sigs_sorted:
            a = sig_to_arr[sig_idx]
            results.append(_simulate_reference(a["r_high"], a["r_low"], a["r_close"], ref_mode))
        ts = np.asarray([r[0] for r in results], dtype=np.float64)
        reasons = np.asarray([r[1] for r in results], dtype=object)
        realized = np.asarray([r[2] for r in results], dtype=np.float64)
        n_total = int(realized.size)
        unique_reasons = sorted(set(reasons.tolist()))
        base = {"reference": ref_mode}
        ref_rows.append({**base, "stat": "N", "value": n_total})
        for reason in unique_reasons:
            c = int(np.sum(reasons == reason))
            ref_rows.append({**base, "stat": f"n_{reason}", "value": c})
            ref_rows.append({
                **base, "stat": f"p_{reason}",
                "value": (c / n_total) if n_total else float("nan"),
            })
        ref_rows.append({**base, "stat": "mean_realized_r", "value": float(np.mean(realized))})
        ref_rows.append({
            **base, "stat": "std_realized_r",
            "value": float(np.std(realized, ddof=1)) if n_total > 1 else float("nan"),
        })
        ref_rows.append({
            **base, "stat": "skew_realized_r",
            "value": float(sps.skew(realized, bias=False)) if n_total > 2 else float("nan"),
        })
        ref_rows.append({
            **base, "stat": "excess_kurt_realized_r",
            "value": float(sps.kurtosis(realized, fisher=True, bias=False)) if n_total > 3 else float("nan"),
        })
        ref_rows.append({**base, "stat": "min_realized_r", "value": float(np.min(realized))})
        ref_rows.append({**base, "stat": "max_realized_r", "value": float(np.max(realized))})
        pcts = np.percentile(realized, list(DIST_PERCENTILES), method="linear")
        for p, v in zip(DIST_PERCENTILES, pcts):
            ref_rows.append({**base, "stat": f"p{int(p)}_realized_r", "value": float(v)})
        ref_rows.append({**base, "stat": "p_realized_r_gt_0", "value": float(np.mean(realized > 0.0))})
        ref_rows.append({**base, "stat": "p_realized_r_gt_1", "value": float(np.mean(realized > 1.0))})
        ref_rows.append({**base, "stat": "p_realized_r_gt_2", "value": float(np.mean(realized > 2.0))})
        ref_rows.append({**base, "stat": "mean_exit_bar", "value": float(np.mean(ts))})
        hist_counts, _ = np.histogram(realized, bins=HIST_BIN_EDGES_DEFAULT)
        for i, c in enumerate(hist_counts):
            lo = HIST_BIN_EDGES_DEFAULT[i]
            hi = HIST_BIN_EDGES_DEFAULT[i + 1]
            lo_s = "-inf" if not np.isfinite(lo) else f"{lo:.4f}"
            hi_s = "inf" if not np.isfinite(hi) else f"{hi:.4f}"
            ref_rows.append({**base, "stat": f"hist[{lo_s},{hi_s})", "value": int(c)})

    ref_df = pd.DataFrame(ref_rows).sort_values(by=["reference", "stat"], kind="mergesort").reset_index(drop=True)

    return grid_df, ref_df, realized_r_by_cell


# ---------------------------------------------------------------------------
# Analysis D — First-touch 2R-up subpopulation.
# ---------------------------------------------------------------------------


def analysis_d(
    sig_to_arr: Dict[int, Dict[str, np.ndarray]],
    shape: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Distributions for the subpopulation where first_touch_2r_dir == +1."""
    sub_mask = shape["first_touch_2r_dir"] == 1
    sub = shape.loc[sub_mask].copy()
    n_sub = int(len(sub))
    rows: List[Dict[str, Any]] = []

    def cell_code(v: Any) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "never"
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return "never"
        return "up" if iv > 0 else "down"

    sub["_c05"] = sub["first_touch_05r_dir"].apply(cell_code)
    sub["_c1"] = sub["first_touch_1r_dir"].apply(cell_code)
    sub["_c2"] = sub["first_touch_2r_dir"].apply(cell_code)

    # Overall distributions.
    overall_descs: List[Dict[str, Any]] = [
        describe_distribution(sub["r_at_t240"].to_numpy(np.float64), "all_2r_up | r_at_t240"),
        describe_distribution(sub["peak_mfe_r"].to_numpy(np.float64), "all_2r_up | peak_mfe_r"),
    ]
    for d in overall_descs:
        for row in desc_to_long_rows(d, {"cell": "ALL", "metric": d["label"].split("| ")[-1]}):
            rows.append(row)

    # Time-to-2R touch (first_touch arr from shape OR re-derive). shape has
    # time_to_first_2r_mfe column.
    t_to_2r = sub["time_to_first_2r_mfe"].to_numpy(np.float64)
    d_t2r = describe_distribution(t_to_2r, "all_2r_up | time_to_first_2r_mfe")
    overall_descs.append(d_t2r)
    for row in desc_to_long_rows(d_t2r, {"cell": "ALL", "metric": "time_to_first_2r_mfe"}):
        rows.append(row)

    # Time-from-2R-touch-to-peak: peak_mfe_bar - time_to_first_2r_mfe (where both finite).
    t_to_peak_from_2r = sub["peak_mfe_bar"].to_numpy(np.float64) - t_to_2r
    d_tp = describe_distribution(t_to_peak_from_2r, "all_2r_up | bars_from_first_2r_touch_to_peak")
    overall_descs.append(d_tp)
    for row in desc_to_long_rows(d_tp, {"cell": "ALL", "metric": "bars_from_first_2r_touch_to_peak"}):
        rows.append(row)

    # Conditional chain P(reach Y | reached 2R first up) for Y in {3..7}.
    chain_rows: List[Dict[str, Any]] = []
    for Y in (3.0, 4.0, 5.0, 6.0, 7.0):
        n_Y = int(np.sum(sub["peak_mfe_r"] >= Y))
        chain_rows.append({
            "cell": "ALL",
            "target_R": Y,
            "n_reached": n_Y,
            "p_reached_given_2r_up": (n_Y / n_sub) if n_sub else float("nan"),
        })
        # Conditional r_at_t240 given reached Y.
        sub_Y = sub.loc[sub["peak_mfe_r"] >= Y]
        d_cond = describe_distribution(
            sub_Y["r_at_t240"].to_numpy(np.float64),
            f"all_2r_up & reached_{Y:g}R | r_at_t240",
        )
        for row in desc_to_long_rows(d_cond, {"cell": "ALL", "metric": f"r_at_t240 | reached_{Y:g}R"}):
            rows.append(row)
        overall_descs.append(d_cond)

    # Per-cell (c05, c1, c2=up).
    per_cell_descs: List[Dict[str, Any]] = []
    cells_seen: List[Tuple[str, str, str]] = []
    grouped = sub.groupby(["_c05", "_c1", "_c2"], sort=True)
    cell_summary_rows: List[Dict[str, Any]] = []
    for (c05, c1, c2), grp in grouped:
        if c2 != "up":
            continue
        cell_label = f"({c05},{c1},{c2})"
        cells_seen.append((c05, c1, c2))
        n_cell = int(len(grp))
        mean_peak_mfe = float(np.nanmean(grp["peak_mfe_r"].to_numpy(np.float64)))
        d1 = describe_distribution(
            grp["r_at_t240"].to_numpy(np.float64),
            f"{cell_label} | r_at_t240",
        )
        d2 = describe_distribution(
            grp["peak_mfe_r"].to_numpy(np.float64),
            f"{cell_label} | peak_mfe_r",
        )
        per_cell_descs.extend([d1, d2])
        for row in desc_to_long_rows(d1, {"cell": cell_label, "metric": "r_at_t240"}):
            rows.append(row)
        for row in desc_to_long_rows(d2, {"cell": cell_label, "metric": "peak_mfe_r"}):
            rows.append(row)
        cell_summary_rows.append({
            "cell": cell_label,
            "c05": c05,
            "c1": c1,
            "c2": c2,
            "n": n_cell,
            "mean_peak_mfe_r": mean_peak_mfe,
            "mean_r_at_t240": float(np.nanmean(grp["r_at_t240"].to_numpy(np.float64))),
            "p_final_r_gt_0": float(np.mean(grp["r_at_t240"].to_numpy(np.float64) > 0.0)),
        })

    df = pd.DataFrame(rows).sort_values(by=["cell", "metric", "stat"], kind="mergesort").reset_index(drop=True)
    meta: Dict[str, Any] = {
        "n_sub": n_sub,
        "expected_n_from_crosstab_sum": 21851,
        "overall_descs": overall_descs,
        "per_cell_descs": per_cell_descs,
        "cell_summary": cell_summary_rows,
        "chain_rows": chain_rows,
    }
    return df, meta


# ---------------------------------------------------------------------------
# Charts.
# ---------------------------------------------------------------------------


def chart_trail_heatmap(
    grid_df: pd.DataFrame, stat_name: str, title: str, path: Path
) -> None:
    Xs = list(TRAIL_X_GRID)
    Gs = list(TRAIL_G_GRID)
    z = np.full((len(Xs), len(Gs)), np.nan)
    nz = np.full((len(Xs), len(Gs)), 0, dtype=int)
    for i, X in enumerate(Xs):
        for j, G in enumerate(Gs):
            cell = grid_df[(grid_df["X"] == X) & (grid_df["G"] == G) & (grid_df["stat"] == stat_name)]
            n_cell = grid_df[(grid_df["X"] == X) & (grid_df["G"] == G) & (grid_df["stat"] == "N")]
            if len(cell):
                z[i, j] = float(cell["value"].iloc[0])
            if len(n_cell):
                nz[i, j] = int(float(n_cell["value"].iloc[0]))
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(z, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(Gs)))
    ax.set_xticklabels([f"{g:g}" for g in Gs])
    ax.set_yticks(range(len(Xs)))
    ax.set_yticklabels([f"{x:g}" for x in Xs])
    ax.set_xlabel("trail distance G (R)")
    ax.set_ylabel("activation X (R)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=stat_name)
    for i in range(len(Xs)):
        for j in range(len(Gs)):
            txt = f"{z[i, j]:.3f}\n(N={nz[i, j]})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black")
    _save_figure(fig, path)


def chart_per_band_giveback(per_band_descriptors_by_X: Dict[float, Dict[str, Any]], path: Path) -> None:
    """7-panel grid of giveback_from_peak distributions, one per X band."""
    Xs = list(ANALYSIS_A_BANDS)
    n_rows = int(np.ceil(len(Xs) / 2))
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(12, 3 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    for i, X in enumerate(Xs):
        ax = axes[i]
        desc = per_band_descriptors_by_X.get(X)
        if desc is None:
            ax.set_visible(False)
            continue
        edges = np.asarray(desc["histogram_bin_edges"], dtype=np.float64)
        counts = np.asarray(desc["histogram_counts"], dtype=np.int64)
        # plot only finite-edge bars.
        finite_edges = np.isfinite(edges)
        finite_pairs = []
        for k in range(len(counts)):
            if finite_edges[k] and finite_edges[k + 1]:
                center = 0.5 * (edges[k] + edges[k + 1])
                finite_pairs.append((center, counts[k]))
        if finite_pairs:
            centers = [p[0] for p in finite_pairs]
            heights = [p[1] for p in finite_pairs]
            ax.bar(centers, heights, width=0.25, color="C0", alpha=0.6)
        ax.set_title(f"X={X:g}R | giveback_from_peak  (N={desc['n']})", fontsize=9)
        ax.set_xlabel("R")
        ax.set_ylabel("count")
    for j in range(len(Xs), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Giveback from peak distribution per MFE band", fontsize=11)
    _save_figure(fig, path)


def chart_intra_trade_giveback_overlay(after_descs: List[Dict[str, Any]], path: Path) -> None:
    """Overlay normalised distributions of max_dd_from_peak_after_X across X bands."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(after_descs)))
    for desc, color in zip(after_descs, colors):
        edges = np.asarray(desc["histogram_bin_edges"], dtype=np.float64)
        counts = np.asarray(desc["histogram_counts"], dtype=np.float64)
        # Use finite edges only for plot.
        centers = []
        density = []
        n_total = counts.sum() if counts.sum() else 1
        for k in range(len(counts)):
            lo, hi = edges[k], edges[k + 1]
            if np.isfinite(lo) and np.isfinite(hi):
                centers.append(0.5 * (lo + hi))
                density.append(counts[k] / n_total)
        ax.plot(centers, density, label=desc["label"].split("|")[0].strip(), color=color, lw=1.0)
    ax.set_xlabel("max_dd_from_peak_after_X (R)")
    ax.set_ylabel("density (per-bin fraction)")
    ax.set_title("Intra-trade max drawdown from running peak after first X touch")
    ax.legend(fontsize=8, loc="upper right")
    _save_figure(fig, path)


def chart_pullback_counts(descriptors: List[Dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for desc in descriptors:
        edges = np.asarray(desc["histogram_bin_edges"], dtype=np.float64)
        counts = np.asarray(desc["histogram_counts"], dtype=np.float64)
        centers = []
        heights = []
        n_total = counts.sum() if counts.sum() else 1
        for k in range(len(counts)):
            lo, hi = edges[k], edges[k + 1]
            if np.isfinite(lo) and np.isfinite(hi):
                centers.append(0.5 * (lo + hi))
                heights.append(counts[k] / n_total)
        ax.plot(centers, heights, label=desc["label"], lw=1.0)
    ax.set_xlabel("n pullback events per trade (R)")
    ax.set_ylabel("density (per-bin fraction)")
    ax.set_title("Per-trade pullback event count distributions by threshold")
    ax.legend(fontsize=8, loc="upper right")
    _save_figure(fig, path)


# ---------------------------------------------------------------------------
# Report writer.
# ---------------------------------------------------------------------------


def write_report(
    *,
    actual_hashes: Dict[str, str],
    n_total: int,
    chain_df: pd.DataFrame,
    a_descriptors: List[Dict[str, Any]],
    b_after_descs: List[Dict[str, Any]],
    b_before_descs: List[Dict[str, Any]],
    b_pullback_descs: List[Dict[str, Any]],
    grid_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    analysis_d_meta: Dict[str, Any],
    path: Path,
) -> None:
    lines: List[str] = [
        "# Arc 2 Band Behavior Follow-up — Empirical Report",
        "",
        "Phase: l6_arc2_band_behavior (supplementary, descriptive only per L6.0 v1.1 §14.5).",
        "",
        "## Inputs (read-only, hash-verified against parent appendix manifest)",
        "",
        f"- trajectory_panel.parquet sha256: {actual_hashes['trajectory_panel.parquet']}",
        f"- shape_features.csv sha256: {actual_hashes['shape_features.csv']}",
        f"- Population: N = {n_total} (from shape_features rows / panel signal_idx unique count).",
        "",
        "## Conventions",
        "",
        "- 1R = 2 x Wilder ATR(14) at bar N close (`atr_1h_wilder_at_n`, same as parent appendix).",
        "- Long-only; t=0 = entry price = open of bar N+1; per-bar metrics r_close, r_high, r_low,",
        "  running_mfe, running_mae as defined in the parent appendix.",
        "- Virtual trajectory — ignores actual Arc 2 SL / time exits.",
        "- Trail simulation uses r_high for peak update and r_low for trail trigger (intrabar).",
        "  Hard SL is intrabar r_low <= -1R. Bar-order on each step: SL check first, then trail",
        "  tighten-then-check.",
        "",
        "## Analysis A — Per-band outcomes",
        "",
        "For each MFE band X in {1..7}R, the subpopulation is trades with peak_mfe_r >= X.",
        "Distribution tables (full moments + percentiles + extremes + histograms) for each metric",
        "are in `per_band_outcomes.csv` (long format). Conditional chain probabilities below.",
        "",
        "| band_R | n_reached | P(reach X+1 \\| reached X) | P(reach X+2 \\| reached X) |",
        "|--------|-----------|---------------------------|---------------------------|",
    ]
    for r in chain_df.itertuples(index=False):
        lines.append(
            f"| {_fmt_float(r.band_R, 2)} | {int(r.n_reached)} | "
            f"{_fmt_float(r.p_reach_X_plus_1_given_X)} | {_fmt_float(r.p_reach_X_plus_2_given_X)} |"
        )
    lines.append("")
    lines.append("Summary mean/p50/p90 across selected metrics per band:")
    lines.append("")
    lines.append("| band_R | metric | mean | p50 | p90 | N |")
    lines.append("|--------|--------|------|-----|-----|---|")
    metrics_of_interest = (
        "peak_mfe_r",
        "r_at_t240",
        "giveback_from_peak",
        "subsequent_mae_after_first_X_touch",
        "subsequent_max_mfe_after_first_X_touch",
        "r_at_first_X_touch_plus_24",
        "r_at_first_X_touch_plus_120",
        "bars_from_first_X_touch_to_peak",
    )
    by_label = {d["label"]: d for d in a_descriptors}
    for X in ANALYSIS_A_BANDS:
        for m in metrics_of_interest:
            d = by_label.get(f"X={X:g}R | {m}")
            if d is None:
                continue
            lines.append(
                f"| {_fmt_float(X, 2)} | {m} | {_fmt_float(d['mean'])} | "
                f"{_fmt_float(d['p50'])} | {_fmt_float(d['p90'])} | {d['n']} |"
            )
    lines.append("")
    lines.append("Full per-metric histograms in `per_band_outcomes.csv`. See `charts/per_band_giveback_distributions.png` for giveback shapes.")
    lines.append("")

    lines.append("## Analysis B — Intra-trade pullback / giveback distributions")
    lines.append("")
    lines.append("`intra_trade_givebacks_after_X.csv` records the max drawdown from running peak measured AFTER the first touch of X.")
    lines.append("`intra_trade_givebacks_before_X.csv` records the same measured on the run up to first touch (max drawdown from running peak in bars [1, t_X-1]).")
    lines.append("Histogram range for both: 0R..15R (giveback is non-negative), 0.25R bins, with overflow.")
    lines.append("")
    lines.append("| band_R | metric | mean | p50 | p90 | p99 | max | N |")
    lines.append("|--------|--------|------|-----|-----|-----|-----|---|")
    for d in b_after_descs:
        x_str = d["label"].split("|")[0].replace("X=", "").strip()
        lines.append(
            f"| {x_str} | max_dd_after | {_fmt_float(d['mean'])} | "
            f"{_fmt_float(d['p50'])} | {_fmt_float(d['p90'])} | {_fmt_float(d['p99'])} | "
            f"{_fmt_float(d['max'])} | {d['n']} |"
        )
    for d in b_before_descs:
        x_str = d["label"].split("|")[0].replace("X=", "").strip()
        lines.append(
            f"| {x_str} | max_dd_before | {_fmt_float(d['mean'])} | "
            f"{_fmt_float(d['p50'])} | {_fmt_float(d['p90'])} | {_fmt_float(d['p99'])} | "
            f"{_fmt_float(d['max'])} | {d['n']} |"
        )
    lines.append("")
    lines.append("Per-trade pullback event count distributions at thresholds {0.5, 1.0, 1.5, 2.0}R (zigzag detector: registers a pullback when running_peak - r_low >= threshold; recovers when next r_high - trough >= threshold/2). Per-trade counts in `pullback_counts.csv`.")
    lines.append("")
    lines.append("| threshold | mean | p50 | p90 | p99 | max | N |")
    lines.append("|-----------|------|-----|-----|-----|-----|---|")
    for d in b_pullback_descs:
        thr_str = d["label"].split("(")[0].strip()
        lines.append(
            f"| {thr_str} | {_fmt_float(d['mean'])} | {_fmt_float(d['p50'])} | "
            f"{_fmt_float(d['p90'])} | {_fmt_float(d['p99'])} | {_fmt_float(d['max'])} | {d['n']} |"
        )
    lines.append("")

    lines.append("## Analysis C — Trail simulation grid")
    lines.append("")
    lines.append("Rule: SL at -1R intrabar throughout; trail activates the first bar running_mfe >= X; trail_level = peak - G; tighten-then-check on each subsequent bar.")
    lines.append("Grid: X in {1..5}R x G in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}R = 30 cells.")
    lines.append("Full per-cell distribution (moments + percentiles + histogram) in `trail_simulation_grid.csv` (long format).")
    lines.append("")
    lines.append("Mean realized R per cell:")
    lines.append("")
    header_cells = ["X \\ G"] + [f"G={g:g}" for g in TRAIL_G_GRID]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
    for X in TRAIL_X_GRID:
        row = [f"X={X:g}"]
        for G in TRAIL_G_GRID:
            cell = grid_df[(grid_df["X"] == X) & (grid_df["G"] == G) & (grid_df["stat"] == "mean_realized_r")]
            row.append(_fmt_float(float(cell["value"].iloc[0])) if len(cell) else "NaN")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("P(realized_r > 0) per cell:")
    lines.append("")
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
    for X in TRAIL_X_GRID:
        row = [f"X={X:g}"]
        for G in TRAIL_G_GRID:
            cell = grid_df[(grid_df["X"] == X) & (grid_df["G"] == G) & (grid_df["stat"] == "p_realized_r_gt_0")]
            row.append(_fmt_float(float(cell["value"].iloc[0])) if len(cell) else "NaN")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Reference cells (verbatim Arc 2, no-time-exit, hard-TP-2R) in `trail_simulation_reference.csv`.")
    lines.append("")
    lines.append("| reference | N | mean_R | std_R | p50_R | P(R>0) | mean_exit_bar |")
    lines.append("|-----------|---|--------|-------|-------|--------|---------------|")
    for ref_mode in ("verbatim_arc2", "no_time_exit", "hard_tp_2R"):
        def gv(stat: str) -> str:
            r = ref_df[(ref_df["reference"] == ref_mode) & (ref_df["stat"] == stat)]
            if not len(r):
                return "NaN"
            try:
                return _fmt_float(float(r["value"].iloc[0]))
            except (TypeError, ValueError):
                return str(r["value"].iloc[0])
        n_str = "NaN"
        n_row = ref_df[(ref_df["reference"] == ref_mode) & (ref_df["stat"] == "N")]
        if len(n_row):
            n_str = str(int(float(n_row["value"].iloc[0])))
        lines.append(
            f"| {ref_mode} | {n_str} | {gv('mean_realized_r')} | {gv('std_realized_r')} | "
            f"{gv('p50_realized_r')} | {gv('p_realized_r_gt_0')} | {gv('mean_exit_bar')} |"
        )
    lines.append("")
    lines.append("Charts: `charts/trail_simulation_heatmap_meanR.png` and `charts/trail_simulation_heatmap_winrate.png`.")
    lines.append("")
    lines.append("Cost caveat: realized R above is gross of execution cost. Real-execution spread + slippage at trail trigger is typically -0.05 to -0.15R per trade and would shift every cell downward by approximately that amount.")
    lines.append("")

    lines.append("## Analysis D — First-touch 2R-up subpopulation")
    lines.append("")
    lines.append(
        f"Subpopulation: trades with `first_touch_2r_dir == +1` in shape_features.csv. "
        f"N = {analysis_d_meta['n_sub']} (crosstab sum of c2=up cells: "
        f"{analysis_d_meta['expected_n_from_crosstab_sum']})."
    )
    if analysis_d_meta["n_sub"] != analysis_d_meta["expected_n_from_crosstab_sum"]:
        lines.append("")
        lines.append(
            f"Discrepancy of {analysis_d_meta['n_sub'] - analysis_d_meta['expected_n_from_crosstab_sum']} "
            "between the first_touch_2r_dir==+1 mask on shape_features.csv and the c2='up' "
            "crosstab cell sum. Cause: the crosstab's 'never' encoding for first_touch_dir==NaN "
            "vs the mask interprets NaN as 'not +1' i.e. exclude. The crosstab-sum quoted in the "
            "prompt is the exclusive count; the mask used here matches that definition."
        )
    lines.append("")
    lines.append("Per-cell summary (c2=up cells only):")
    lines.append("")
    lines.append("| cell | n | mean_peak_mfe_r | mean_r_at_t240 | P(r_at_t240 > 0) |")
    lines.append("|------|---|------------------|-----------------|-------------------|")
    for r in analysis_d_meta["cell_summary"]:
        lines.append(
            f"| {r['cell']} | {r['n']} | {_fmt_float(r['mean_peak_mfe_r'])} | "
            f"{_fmt_float(r['mean_r_at_t240'])} | {_fmt_float(r['p_final_r_gt_0'])} |"
        )
    lines.append("")
    lines.append("Conditional chain (whole 2R-up subpop): P(reach Y | reached 2R up first):")
    lines.append("")
    lines.append("| target_R | n_reached | P(reached \\| 2r_up_first) |")
    lines.append("|----------|-----------|----------------------------|")
    for r in analysis_d_meta["chain_rows"]:
        lines.append(
            f"| {_fmt_float(r['target_R'], 2)} | {int(r['n_reached'])} | "
            f"{_fmt_float(r['p_reached_given_2r_up'])} |"
        )
    lines.append("")
    lines.append("Full distributions per cell in `first_touch_2r_up_subpopulation.csv`.")
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


def write_trail_summary_md(grid_df: pd.DataFrame, ref_df: pd.DataFrame, path: Path) -> None:
    """Standalone summary markdown for the trail grid (Analysis C deliverable)."""
    lines: List[str] = [
        "# Arc 2 Trail Simulation Summary  (Analysis C)",
        "",
        "Rule: hard SL at -1R intrabar throughout; trail activates the first bar running_mfe >= X; "
        "trail_level = peak - G; on subsequent bars tighten peak from r_high THEN check r_low <= trail_level.",
        "",
        "Grid: X in {1, 2, 3, 4, 5}R x G in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}R = 30 cells.",
        "",
        "Each cell's full distribution (moments + percentiles + histogram) in `trail_simulation_grid.csv`.",
        "",
        "## Mean realized R",
        "",
    ]
    header_cells = ["X \\ G"] + [f"G={g:g}" for g in TRAIL_G_GRID]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
    for X in TRAIL_X_GRID:
        row = [f"X={X:g}"]
        for G in TRAIL_G_GRID:
            cell = grid_df[(grid_df["X"] == X) & (grid_df["G"] == G) & (grid_df["stat"] == "mean_realized_r")]
            row.append(_fmt_float(float(cell["value"].iloc[0])) if len(cell) else "NaN")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## P(realized_r > 0)")
    lines.append("")
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
    for X in TRAIL_X_GRID:
        row = [f"X={X:g}"]
        for G in TRAIL_G_GRID:
            cell = grid_df[(grid_df["X"] == X) & (grid_df["G"] == G) & (grid_df["stat"] == "p_realized_r_gt_0")]
            row.append(_fmt_float(float(cell["value"].iloc[0])) if len(cell) else "NaN")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## P(realized_r > 1)")
    lines.append("")
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
    for X in TRAIL_X_GRID:
        row = [f"X={X:g}"]
        for G in TRAIL_G_GRID:
            cell = grid_df[(grid_df["X"] == X) & (grid_df["G"] == G) & (grid_df["stat"] == "p_realized_r_gt_1")]
            row.append(_fmt_float(float(cell["value"].iloc[0])) if len(cell) else "NaN")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Exit-reason composition per cell")
    lines.append("")
    lines.append("| X | G | N | n_sl | n_trail | n_time_240 | n_never_reached_X |")
    lines.append("|---|---|---|------|---------|------------|--------------------|")
    for X in TRAIL_X_GRID:
        for G in TRAIL_G_GRID:
            def gv(stat: str) -> str:
                r = grid_df[(grid_df["X"] == X) & (grid_df["G"] == G) & (grid_df["stat"] == stat)]
                if not len(r):
                    return "NaN"
                try:
                    return str(int(float(r["value"].iloc[0])))
                except (TypeError, ValueError):
                    return str(r["value"].iloc[0])
            lines.append(
                f"| {X:g} | {G:g} | {gv('N')} | {gv('n_sl')} | {gv('n_trail')} | "
                f"{gv('n_time_240')} | {gv('n_never_reached_X')} |"
            )
    lines.append("")
    lines.append("## Reference cells")
    lines.append("")
    lines.append("| reference | N | mean_R | std_R | p50_R | P(R>0) | P(R>1) | mean_exit_bar |")
    lines.append("|-----------|---|--------|-------|-------|--------|--------|---------------|")
    for ref_mode in ("verbatim_arc2", "no_time_exit", "hard_tp_2R"):
        def gv(stat: str) -> str:
            r = ref_df[(ref_df["reference"] == ref_mode) & (ref_df["stat"] == stat)]
            if not len(r):
                return "NaN"
            try:
                return _fmt_float(float(r["value"].iloc[0]))
            except (TypeError, ValueError):
                return str(r["value"].iloc[0])
        n_row = ref_df[(ref_df["reference"] == ref_mode) & (ref_df["stat"] == "N")]
        n_str = str(int(float(n_row["value"].iloc[0]))) if len(n_row) else "NaN"
        lines.append(
            f"| {ref_mode} | {n_str} | {gv('mean_realized_r')} | {gv('std_realized_r')} | "
            f"{gv('p50_realized_r')} | {gv('p_realized_r_gt_0')} | {gv('p_realized_r_gt_1')} | "
            f"{gv('mean_exit_bar')} |"
        )
    lines.append("")
    lines.append("Cost caveat: gross R only. Real execution would include spread at trail trigger; "
                 "the parent appendix's spread modeling produces approximately -0.05 to -0.15 R per trade.")
    lines.append("")
    _write_text("\n".join(lines) + "\n", path)


# ---------------------------------------------------------------------------
# Manifest writer.
# ---------------------------------------------------------------------------


def write_manifest(
    *,
    out_paths: List[Path],
    actual_input_hashes: Dict[str, str],
    n_total: int,
    analysis_d_meta: Dict[str, Any],
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
        f"Arc 2 Band Behavior Follow-up — run_manifest (run #{run_ordinal})",
        "Generated: (suppressed in deterministic mode below)",
        "",
        "## Operational decisions",
        "",
        "- Input hashes verified against parent appendix's run_manifest.txt; HALT on mismatch.",
        "- Trail simulation order: SL first (intrabar r_low <= -1R), then trail tighten-then-check",
        "  (update peak from r_high, then check r_low <= peak - G). Per the prompt's explicit",
        "  step ordering.",
        "- Reference cells: verbatim_arc2 walks bars 1..120 (closes at r_close[119] if no SL);",
        "  no_time_exit walks bars 1..240 (closes at r_close[239]); hard_tp_2R walks 1..240 with",
        "  SL-first then TP (r_high >= 2.0).",
        "- Pullback detector: zigzag, threshold S registers an event when running_peak - r_low >= S;",
        "  state flips to 'down'; flips back to 'up' when r_high - trough >= S/2 (no event for",
        "  recovery; only down-moves counted).",
        "- Analysis D subpopulation defined by shape_features.csv first_touch_2r_dir == +1 (exclusive",
        "  of NaN), matching the c2='up' crosstab definition.",
        "",
        "## Inputs (read-only)",
        "",
        f"- trajectory_panel.parquet sha256: {actual_input_hashes['trajectory_panel.parquet']}",
        f"- shape_features.csv sha256: {actual_input_hashes['shape_features.csv']}",
        f"- N total in panel + shape: {n_total}",
        "",
        "## Determinism config",
        "",
        f"- NUMPY_SEED = {NUMPY_SEED}",
        "- Sort order: per-signal arrays iterated in sorted(signal_idx) order; all aggregations",
        "  use mergesort on grouped indices.",
        "- CSV: utf-8, LF line endings, float_format='%.10g'.",
        "- Matplotlib: Agg backend, metadata pinned (Software='matplotlib', Creation Time='2020-01-01').",
        "",
        "## Diagnostics",
        "",
        f"- Analysis D subpopulation N = {analysis_d_meta['n_sub']} "
        f"(crosstab c2=up sum: {analysis_d_meta['expected_n_from_crosstab_sum']}; "
        f"delta = {analysis_d_meta['n_sub'] - analysis_d_meta['expected_n_from_crosstab_sum']})",
        "",
        f"## Output file sha256 (run #{run_ordinal})",
        "",
    ]
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
        lines.append(f"Overall: {'PASS - byte-identical across runs' if all_match else 'FAIL - some files diverged (PNG pixel-only fallbacks above)'}")
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
    sig_to_arr, shape, actual_hashes = load_inputs()
    n_total = len(sig_to_arr)
    print(f"[run #{run_ordinal}] Loaded {n_total} signals.", flush=True)

    print(f"[run #{run_ordinal}] Analysis A — per-band outcomes ...", flush=True)
    a_long, a_descs, chain_df = analysis_a(sig_to_arr, shape)
    print(f"[run #{run_ordinal}] Analysis B — intra-trade givebacks + pullback counts ...", flush=True)
    b_after_df, b_before_df, b_after_descs, b_before_descs = analysis_b_givebacks(sig_to_arr, shape)
    b_pullback_df, b_pullback_descs = analysis_b_pullbacks(sig_to_arr)

    print(f"[run #{run_ordinal}] Analysis C — trail simulation grid ...", flush=True)
    grid_df, ref_df, _ = analysis_c(sig_to_arr)

    print(f"[run #{run_ordinal}] Analysis D — first-touch 2R-up subpopulation ...", flush=True)
    d_df, d_meta = analysis_d(sig_to_arr, shape)

    # Build a giveback-descriptor lookup keyed by X (from Analysis A).
    giveback_by_X: Dict[float, Dict[str, Any]] = {}
    for d in a_descs:
        # label format: "X={X}R | metric"
        try:
            parts = d["label"].split("|")
            x_part = parts[0].strip()  # "X=1R"
            metric_part = parts[1].strip()
            if metric_part != "giveback_from_peak":
                continue
            x_val = float(x_part.replace("X=", "").rstrip("R"))
            giveback_by_X[x_val] = d
        except (ValueError, IndexError):
            continue

    print(f"[run #{run_ordinal}] Writing outputs ...", flush=True)
    out_paths: List[Path] = []
    _write_csv(a_long, OUTPUT_DIR / "per_band_outcomes.csv")
    out_paths.append(OUTPUT_DIR / "per_band_outcomes.csv")
    _write_csv(b_after_df, OUTPUT_DIR / "intra_trade_givebacks_after_X.csv")
    out_paths.append(OUTPUT_DIR / "intra_trade_givebacks_after_X.csv")
    _write_csv(b_before_df, OUTPUT_DIR / "intra_trade_givebacks_before_X.csv")
    out_paths.append(OUTPUT_DIR / "intra_trade_givebacks_before_X.csv")
    _write_csv(b_pullback_df, OUTPUT_DIR / "pullback_counts.csv")
    out_paths.append(OUTPUT_DIR / "pullback_counts.csv")
    _write_csv(grid_df, OUTPUT_DIR / "trail_simulation_grid.csv")
    out_paths.append(OUTPUT_DIR / "trail_simulation_grid.csv")
    _write_csv(ref_df, OUTPUT_DIR / "trail_simulation_reference.csv")
    out_paths.append(OUTPUT_DIR / "trail_simulation_reference.csv")
    write_trail_summary_md(grid_df, ref_df, OUTPUT_DIR / "trail_simulation_summary.md")
    out_paths.append(OUTPUT_DIR / "trail_simulation_summary.md")
    _write_csv(d_df, OUTPUT_DIR / "first_touch_2r_up_subpopulation.csv")
    out_paths.append(OUTPUT_DIR / "first_touch_2r_up_subpopulation.csv")

    # Charts.
    chart_trail_heatmap(
        grid_df, "mean_realized_r", "Trail grid mean realized R",
        CHARTS_DIR / "trail_simulation_heatmap_meanR.png",
    )
    out_paths.append(CHARTS_DIR / "trail_simulation_heatmap_meanR.png")
    chart_trail_heatmap(
        grid_df, "p_realized_r_gt_0", "Trail grid P(realized R > 0)",
        CHARTS_DIR / "trail_simulation_heatmap_winrate.png",
    )
    out_paths.append(CHARTS_DIR / "trail_simulation_heatmap_winrate.png")
    chart_per_band_giveback(giveback_by_X, CHARTS_DIR / "per_band_giveback_distributions.png")
    out_paths.append(CHARTS_DIR / "per_band_giveback_distributions.png")
    chart_intra_trade_giveback_overlay(b_after_descs, CHARTS_DIR / "intra_trade_giveback_after_X.png")
    out_paths.append(CHARTS_DIR / "intra_trade_giveback_after_X.png")
    chart_pullback_counts(b_pullback_descs, CHARTS_DIR / "pullback_count_distributions.png")
    out_paths.append(CHARTS_DIR / "pullback_count_distributions.png")

    print(f"[run #{run_ordinal}] Writing band_behavior_report.md ...", flush=True)
    write_report(
        actual_hashes=actual_hashes,
        n_total=n_total,
        chain_df=chain_df,
        a_descriptors=a_descs,
        b_after_descs=b_after_descs,
        b_before_descs=b_before_descs,
        b_pullback_descs=b_pullback_descs,
        grid_df=grid_df,
        ref_df=ref_df,
        analysis_d_meta=d_meta,
        path=OUTPUT_DIR / "band_behavior_report.md",
    )
    out_paths.append(OUTPUT_DIR / "band_behavior_report.md")

    print(f"[run #{run_ordinal}] Writing run_manifest.txt ...", flush=True)
    hashes = write_manifest(
        out_paths=out_paths,
        actual_input_hashes=actual_hashes,
        n_total=n_total,
        analysis_d_meta=d_meta,
        prior_hashes=prior_hashes,
        run_ordinal=run_ordinal,
    )
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description="Arc 2 Band Behavior Follow-up")
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
        print("[verify] Some outputs diverged; check run_manifest.txt for PNG pixel-only fallback hashes.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
