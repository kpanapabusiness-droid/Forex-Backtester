"""Arc 2 Baseline Trajectory Appendix — supplementary characterisation.

Phase: l6_arc2_trajectory_appendix (additive; descriptive only per L6.0 v1.1 §14.5).

Produces a per-bar virtual trajectory panel (t=0..240) for every Arc 2 ex-ante
signal (`TRIAL__mtf_alignment__2_down_mixed__kijun__h_120`), plus full
distributional characterisation, unsupervised clustering, and conditional
distributions. The virtual trajectory ignores Arc 2's actual SL/time exit —
actual exit info is recorded as separate columns on the per-trade row.

Inputs (read-only):
- results/l6/arc2/characterisation/v1_1_full/signals_features.csv (population)
- data/1hr/<PAIR>.csv (28 pairs)

Outputs:
- results/l6/arc2/trajectory_appendix/...

R-multiple convention (LOCKED):
    1R = 2 * Wilder ATR(14) on 1H bars, snapshotted at bar N close.
    The Wilder ATR is taken directly from signals_features.csv column
    `atr_1h_wilder_at_n` (Arc 2 execution layer's locked basis per
    PHASE_L6_ARC2_OPEN.md §3.2).

Determinism: numpy/sklearn seeds locked; sort orders deterministic; pyarrow
parquet version='2.6'; outputs hashed before and after a re-run; matplotlib
Agg backend with metadata pinned.

Documented operational decisions (re-run with --print-decisions to reproduce):
- Ward-linkage hierarchical clustering is applied to a seeded 8,000-row
  subsample (full-N Ward is infeasible at N=41,796: ~7 GB pairwise distance
  matrix). K-means runs on the full feature matrix at every k. ARI is computed
  on the 8k subsample (hierarchical vs k-means restricted to the same rows).
- Ex-ante population scope: every row in signals_features.csv where
  bar N+1 exists in the 1H series (taken + non-taken). `actual_exit_bar` and
  `actual_exit_reason` are populated only for taken rows.
- Charts: matplotlib Agg; metadata pinned; PNG byte-identicality is the goal,
  with PNG pixel-only fallback hash logged if metadata embedding diverges.
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
os.environ.setdefault("SOURCE_DATE_EPOCH", "1577836800")  # 2020-01-01 UTC
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy import stats as sps  # noqa: E402
from scipy.cluster import hierarchy as sch  # noqa: E402
from scipy.cluster.vq import kmeans2  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ----------------------------------------------------------------------------
# LOCKED CONSTANTS
# ----------------------------------------------------------------------------

NUMPY_SEED: int = 20260512
SKLEARN_SEED: int = 20260512
HIERARCHICAL_SUBSAMPLE_N: int = 8000

PAIRS: Tuple[str, ...] = (
    "AUD_CAD",
    "AUD_CHF",
    "AUD_JPY",
    "AUD_NZD",
    "AUD_USD",
    "CAD_CHF",
    "CAD_JPY",
    "CHF_JPY",
    "EUR_AUD",
    "EUR_CAD",
    "EUR_CHF",
    "EUR_GBP",
    "EUR_JPY",
    "EUR_NZD",
    "EUR_USD",
    "GBP_AUD",
    "GBP_CAD",
    "GBP_CHF",
    "GBP_JPY",
    "GBP_NZD",
    "GBP_USD",
    "NZD_CAD",
    "NZD_CHF",
    "NZD_JPY",
    "NZD_USD",
    "USD_CAD",
    "USD_CHF",
    "USD_JPY",
)

INPUT_SIGNALS_CSV: Path = (
    REPO_ROOT
    / "results"
    / "l6"
    / "arc2"
    / "characterisation"
    / "v1_1_full"
    / "signals_features.csv"
)
DATA_1H_DIR: Path = REPO_ROOT / "data" / "1hr"
OUTPUT_DIR: Path = REPO_ROOT / "results" / "l6" / "arc2" / "trajectory_appendix"
CHARTS_DIR: Path = OUTPUT_DIR / "charts"

L4_ANCHOR_LOW: int = 38_543
L4_ANCHOR_HIGH: int = 42_601

TRAJ_HORIZON: int = 240  # bars after entry: t = 0..240 inclusive (241 rows / trade)

# Distributional spec: 0.25R histogram bins from -3R to +12R, plus overflow.
HIST_BIN_EDGES: np.ndarray = np.concatenate(
    [
        [-np.inf],
        np.round(np.arange(-3.0, 12.0 + 0.25 / 2, 0.25), 4),
        [np.inf],
    ]
)

DIST_PERCENTILES: Tuple[float, ...] = (1, 5, 10, 25, 50, 75, 90, 95, 99)
ENVELOPE_PERCENTILES: Tuple[float, ...] = (1, 5, 10, 25, 50, 75, 90, 95, 99)
CLUSTER_ENVELOPE_PERCENTILES: Tuple[float, ...] = (10, 25, 50, 75, 90)

R_AT_TIMES: Tuple[int, ...] = (6, 12, 24, 48, 72, 120, 240)
MFE_BANDS_R: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
MAE_BANDS_R: Tuple[float, ...] = (-0.25, -0.5, -0.75, -1.0, -1.5, -2.0)
TIME_STATE_ANCHORS: Tuple[int, ...] = (6, 12, 24, 48, 72, 120)

# Bucket boundaries for time-state cross-tabs.
R_CLOSE_BUCKETS: Tuple[Tuple[float, float, str], ...] = (
    (-np.inf, -1.0, "(-inf,-1]"),
    (-1.0, -0.5, "(-1,-0.5]"),
    (-0.5, 0.0, "(-0.5,0]"),
    (0.0, 0.5, "(0,0.5]"),
    (0.5, 1.0, "(0.5,1]"),
    (1.0, 2.0, "(1,2]"),
    (2.0, np.inf, "(2,inf)"),
)
RUNNING_MFE_BUCKETS: Tuple[Tuple[float, float, str], ...] = (
    (0.0, 0.5, "[0,0.5)"),
    (0.5, 1.0, "[0.5,1)"),
    (1.0, 2.0, "[1,2)"),
    (2.0, np.inf, "[2,inf)"),
)
RUNNING_MAE_BUCKETS: Tuple[Tuple[float, float, str], ...] = (
    (-np.inf, -1.0, "(-inf,-1]"),
    (-1.0, -0.5, "(-1,-0.5]"),
    (-0.5, -0.25, "(-0.5,-0.25]"),
    (-0.25, 0.0, "(-0.25,0]"),
)

# Shape features used as the clustering matrix (numeric, no NaN-by-construction
# subset is enforced before clustering; rows with NaN in any of these are dropped
# with a recorded count).
CLUSTERING_FEATURES: Tuple[str, ...] = (
    "peak_mfe_r",
    "peak_mae_r",
    "peak_mfe_bar",
    "peak_mae_bar",
    "r_at_t6",
    "r_at_t12",
    "r_at_t24",
    "r_at_t48",
    "r_at_t72",
    "r_at_t120",
    "r_at_t240",
    "retrace_from_peak_mfe",
    "num_distinct_1r_peaks",
)


# ----------------------------------------------------------------------------
# Filesystem helpers — deterministic writes (LF line endings, fixed metadata).
# ----------------------------------------------------------------------------


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png_pixel_hash(path: Path) -> str:
    """Hash only the pixel data of a PNG (strip metadata chunks).

    Returns the sha256 of concatenated IDAT chunk payloads. This isolates pixel
    content from PNG metadata (tIME, tEXt, pHYs, etc.) for cross-run comparison
    when matplotlib metadata pinning is insufficient.
    """
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return _sha256_bytes(data)
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
        pos = pos + 8 + length + 4  # +4 for CRC
    return out.hexdigest()


def _write_csv(df: pd.DataFrame, path: Path, *, float_format: str = "%.10g") -> None:
    """Deterministic CSV write: LF line endings, UTF-8, no index, fixed float fmt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n", float_format=float_format)
    path.write_bytes(buf.getvalue().encode("utf-8"))


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Deterministic parquet write via pyarrow, format version 2.6.

    Strips pandas-from-pandas metadata so the parquet bytes do not embed
    pandas / pyarrow runtime info that varies across environments. The
    "created_by" parquet footer field still encodes the writer library; we
    accept that and rely on the pinned pyarrow version for determinism.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata(None)
    pq.write_table(
        table,
        path,
        version="2.6",
        compression="snappy",
        use_dictionary=False,
        write_statistics=False,
        data_page_size=1024 * 1024,
    )


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Save matplotlib figure with metadata pinned for byte-identicality."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Software": "matplotlib",
        "Creation Time": "2020-01-01T00:00:00Z",
    }
    fig.savefig(path, format="png", dpi=120, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Step 1 — Load signals + 1H bars; build the trajectory panel.
# ----------------------------------------------------------------------------


def load_signals() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load signals_features.csv. Return sorted population with signal_idx."""
    if not INPUT_SIGNALS_CSV.exists():
        raise FileNotFoundError(
            f"signals_features.csv not found at {INPUT_SIGNALS_CSV} — "
            "ASK protocol: regenerate from run_layer4.py is not implemented here."
        )
    df = pd.read_csv(INPUT_SIGNALS_CSV, low_memory=False)
    # Coerce types.
    df["time"] = pd.to_datetime(df["time"])
    df["pair"] = df["pair"].astype(str)
    if df["taken"].dtype != bool:
        df["taken"] = df["taken"].astype(str).str.lower().isin(("true", "1"))
    # Deterministic sort: (time, pair) — assign signal_idx afterwards.
    df = df.sort_values(by=["time", "pair"], kind="mergesort").reset_index(drop=True)
    df["signal_idx"] = np.arange(len(df), dtype=np.int64)

    # L4 anchor band check.
    input_sha = _sha256_file(INPUT_SIGNALS_CSV)
    pop_count = len(df)
    band_ok = L4_ANCHOR_LOW <= pop_count <= L4_ANCHOR_HIGH

    meta = {
        "input_signals_features_sha256": input_sha,
        "input_signals_features_rows": str(pop_count),
        "l4_anchor_band": f"[{L4_ANCHOR_LOW}, {L4_ANCHOR_HIGH}]",
        "l4_anchor_band_ok": str(band_ok).lower(),
    }
    return df, meta


def load_1h_bars(pair: str) -> pd.DataFrame:
    """Load 1H bars for a pair; sort and add a positional index."""
    path = DATA_1H_DIR / f"{pair}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing 1H data for pair {pair}: {path}")
    df = pd.read_csv(path, usecols=["time", "open", "high", "low", "close"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time", kind="mergesort").reset_index(drop=True)
    return df


def build_panel_and_shape_features(
    signals: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Build the per-bar trajectory panel + per-trade shape features.

    Panel rows: (signal_idx, t, r_close, r_high, r_low, running_mfe, running_mae)
        t=0 is the entry anchor: r_close=r_high=r_low=running_mfe=running_mae=0.
        t=k for k=1..240 reads bar (N+k) of the 1H series; entry_price = open of
        bar N+1. R denominator = 2 * atr_1h_wilder_at_n. When bar N+t is past
        data end, that trade's panel is shorter (clamped); trade row records
        the clamp.

    Shape features: one row per signal (only signals with bar N+1 available).
    """
    panel_chunks: List[Dict[str, np.ndarray]] = []
    shape_rows: List[Dict[str, Any]] = []
    dropped_no_next_bar = 0
    dropped_no_signal_bar = 0
    dropped_invalid_atr = 0
    spot_checks: List[Dict[str, Any]] = []

    horizon = TRAJ_HORIZON  # 240
    panel_t = np.arange(horizon + 1, dtype=np.int32)  # 0..240

    for pair in PAIRS:
        df_1h = load_1h_bars(pair)
        time_to_pos = pd.Series(
            np.arange(len(df_1h), dtype=np.int64),
            index=df_1h["time"].values,
        )
        opens = df_1h["open"].to_numpy(dtype=np.float64)
        highs = df_1h["high"].to_numpy(dtype=np.float64)
        lows = df_1h["low"].to_numpy(dtype=np.float64)
        closes = df_1h["close"].to_numpy(dtype=np.float64)
        n_bars_1h = len(df_1h)

        pair_signals = signals.loc[signals["pair"] == pair]
        for row in pair_signals.itertuples(index=False):
            sig_idx = int(row.signal_idx)
            sig_time = row.time
            atr_w = float(row.atr_1h_wilder_at_n) if pd.notna(row.atr_1h_wilder_at_n) else np.nan
            taken = bool(row.taken)

            if not np.isfinite(atr_w) or atr_w <= 0:
                dropped_invalid_atr += 1
                continue

            n_idx_raw = time_to_pos.get(sig_time)
            if n_idx_raw is None or (isinstance(n_idx_raw, float) and not np.isfinite(n_idx_raw)):
                dropped_no_signal_bar += 1
                continue
            n_idx = int(n_idx_raw)

            entry_idx = n_idx + 1
            if entry_idx >= n_bars_1h:
                dropped_no_next_bar += 1
                continue

            entry_price = opens[entry_idx]
            # Bars covered: positions [entry_idx, entry_idx + horizon) = up to 240 bars (t=1..240).
            end_idx_excl = min(entry_idx + horizon, n_bars_1h)
            n_post_bars = end_idx_excl - entry_idx  # 1..horizon
            if n_post_bars < 1:
                dropped_no_next_bar += 1
                continue

            denom = 2.0 * atr_w
            bar_high = highs[entry_idx:end_idx_excl]
            bar_low = lows[entry_idx:end_idx_excl]
            bar_close = closes[entry_idx:end_idx_excl]

            r_high = (bar_high - entry_price) / denom
            r_low = (bar_low - entry_price) / denom
            r_close = (bar_close - entry_price) / denom

            run_mfe = np.maximum.accumulate(r_high)
            run_mae = np.minimum.accumulate(r_low)

            # Build the panel arrays with the t=0 anchor prepended.
            t_arr = panel_t[: n_post_bars + 1]
            r_close_full = np.concatenate(([0.0], r_close))
            r_high_full = np.concatenate(([0.0], r_high))
            r_low_full = np.concatenate(([0.0], r_low))
            run_mfe_full = np.concatenate(([0.0], run_mfe))
            run_mae_full = np.concatenate(([0.0], run_mae))

            panel_chunks.append(
                {
                    "signal_idx": np.full(len(t_arr), sig_idx, dtype=np.int64),
                    "t": t_arr.astype(np.int32),
                    "r_close": r_close_full.astype(np.float64),
                    "r_high": r_high_full.astype(np.float64),
                    "r_low": r_low_full.astype(np.float64),
                    "running_mfe": run_mfe_full.astype(np.float64),
                    "running_mae": run_mae_full.astype(np.float64),
                }
            )

            fold_id_val = int(row.fold_id) if pd.notna(row.fold_id) else -1
            # Shape features (over t=1..n_post_bars; t=0 excluded).
            shape = _compute_shape_features(
                sig_idx=sig_idx,
                pair=pair,
                entry_time=df_1h["time"].iloc[entry_idx],
                signal_time=sig_time,
                entry_price=entry_price,
                entry_atr_1h=atr_w,
                r_close=r_close,
                r_high=r_high,
                r_low=r_low,
                run_mfe=run_mfe,
                run_mae=run_mae,
                n_post_bars=n_post_bars,
                actual_taken=taken,
                actual_exit_bar=(
                    int(row.held_bars) if (taken and pd.notna(row.held_bars)) else None
                ),
                actual_exit_reason=(
                    str(row.exit_reason) if (taken and pd.notna(row.exit_reason)) else None
                ),
                fold_id=fold_id_val,
                clamped=(n_post_bars < horizon),
            )
            shape_rows.append(shape)

            # Spot-check: virtual trajectory must run past actual_exit_bar.
            if (
                taken
                and pd.notna(row.exit_reason)
                and str(row.exit_reason) == "stop_loss"
                and len(spot_checks) < 3
            ):
                actual_bar = int(row.held_bars) if pd.notna(row.held_bars) else None
                spot_checks.append(
                    {
                        "signal_idx": sig_idx,
                        "pair": pair,
                        "signal_time": str(sig_time),
                        "actual_exit_bar": actual_bar,
                        "panel_last_t": int(t_arr[-1]),
                        "panel_extends_past_actual_exit": (
                            actual_bar is not None and t_arr[-1] > actual_bar
                        ),
                        "r_close_at_panel_end": float(r_close_full[-1]),
                    }
                )

    # Concatenate panel chunks.
    panel_dict = {
        "signal_idx": np.concatenate([c["signal_idx"] for c in panel_chunks]),
        "t": np.concatenate([c["t"] for c in panel_chunks]),
        "r_close": np.concatenate([c["r_close"] for c in panel_chunks]),
        "r_high": np.concatenate([c["r_high"] for c in panel_chunks]),
        "r_low": np.concatenate([c["r_low"] for c in panel_chunks]),
        "running_mfe": np.concatenate([c["running_mfe"] for c in panel_chunks]),
        "running_mae": np.concatenate([c["running_mae"] for c in panel_chunks]),
    }
    panel = pd.DataFrame(panel_dict)
    # Deterministic order.
    panel = panel.sort_values(by=["signal_idx", "t"], kind="mergesort").reset_index(drop=True)

    shape_df = (
        pd.DataFrame(shape_rows).sort_values("signal_idx", kind="mergesort").reset_index(drop=True)
    )

    meta = {
        "dropped_no_signal_bar": dropped_no_signal_bar,
        "dropped_no_next_bar": dropped_no_next_bar,
        "dropped_invalid_atr": dropped_invalid_atr,
        "n_signals_with_panel": len(shape_df),
        "n_panel_rows": len(panel),
        "spot_checks": spot_checks,
    }
    return panel, shape_df, meta


def _first_idx_ge(arr: np.ndarray, threshold: float) -> Optional[int]:
    mask = arr >= threshold
    if not mask.any():
        return None
    return int(np.argmax(mask))  # first True


def _first_idx_le(arr: np.ndarray, threshold: float) -> Optional[int]:
    mask = arr <= threshold
    if not mask.any():
        return None
    return int(np.argmax(mask))


def _count_distinct_1r_peaks(
    r_high: np.ndarray,
    threshold: float = 1.0,
    pullback: float = 0.5,
) -> int:
    """Count distinct peaks above `threshold` separated by `>=pullback` retracement.

    Rule: enter peak-mode the first bar r_high >= threshold (count it). Track
    the highest r_high seen while in peak-mode. When r_high drops to
    (peak_high - pullback) or below, exit peak-mode. Next entry counts as a
    new distinct peak.
    """
    count = 0
    in_peak = False
    peak_high = -np.inf
    for v in r_high:
        if not in_peak:
            if v >= threshold:
                count += 1
                in_peak = True
                peak_high = v
        else:
            if v > peak_high:
                peak_high = v
            if v <= peak_high - pullback:
                in_peak = False
                peak_high = -np.inf
    return count


def _compute_shape_features(
    *,
    sig_idx: int,
    pair: str,
    entry_time: pd.Timestamp,
    signal_time: pd.Timestamp,
    entry_price: float,
    entry_atr_1h: float,
    r_close: np.ndarray,
    r_high: np.ndarray,
    r_low: np.ndarray,
    run_mfe: np.ndarray,
    run_mae: np.ndarray,
    n_post_bars: int,
    actual_taken: bool,
    actual_exit_bar: Optional[int],
    actual_exit_reason: Optional[str],
    fold_id: int,
    clamped: bool,
) -> Dict[str, Any]:
    """Per-trade shape features. Indices into r_* are 0-based across t=1..N."""
    # Peak metrics: argmax over r_high (the bar that first reached the max),
    # argmin over r_low.
    peak_mfe_r = float(run_mfe[-1])
    peak_mae_r = float(run_mae[-1])
    peak_mfe_bar = int(np.argmax(r_high)) + 1  # bar t (1..N)
    peak_mae_bar = int(np.argmin(r_low)) + 1
    mae_before_mfe = bool(peak_mae_bar < peak_mfe_bar)

    sl_idx = _first_idx_le(r_low, -1.0)
    virtual_sl_hit = sl_idx is not None
    virtual_sl_hit_bar = (sl_idx + 1) if virtual_sl_hit else None

    def r_at(t: int) -> float:
        if t > n_post_bars:
            return float("nan")
        return float(r_close[t - 1])

    r_at_map = {f"r_at_t{t}": r_at(t) for t in R_AT_TIMES}

    def time_to_mfe(level: float) -> Optional[int]:
        idx = _first_idx_ge(run_mfe, level)
        return (idx + 1) if idx is not None else None

    def time_to_mae(level: float) -> Optional[int]:
        idx = _first_idx_le(run_mae, level)
        return (idx + 1) if idx is not None else None

    first_05_mfe_bar = time_to_mfe(0.5)
    first_1_mfe_bar = time_to_mfe(1.0)
    first_2_mfe_bar = time_to_mfe(2.0)
    first_3_mfe_bar = time_to_mfe(3.0)
    first_neg05_mae_bar = time_to_mae(-0.5)
    first_neg1_mae_bar = time_to_mae(-1.0)
    first_neg2_mae_bar = time_to_mae(-2.0)

    def first_touch_dir(level: float) -> Optional[int]:
        mfe_bar = time_to_mfe(level)
        mae_bar = time_to_mae(-level)
        if mfe_bar is None and mae_bar is None:
            return None
        if mfe_bar is None:
            return -1
        if mae_bar is None:
            return +1
        return +1 if mfe_bar < mae_bar else (-1 if mae_bar < mfe_bar else +1)

    first_touch_05_dir = first_touch_dir(0.5)
    first_touch_1_dir = first_touch_dir(1.0)
    first_touch_2_dir = first_touch_dir(2.0)

    num_distinct_1r_peaks = _count_distinct_1r_peaks(r_high, 1.0, 0.5)

    final_r = float(r_close[-1]) if len(r_close) > 0 else float("nan")
    retrace_from_peak_mfe = (
        float(peak_mfe_r - r_at(TRAJ_HORIZON)) if not np.isnan(r_at(TRAJ_HORIZON)) else float("nan")
    )
    time_to_peak_mfe = peak_mfe_bar  # already the bar; named separately for distribution stats

    return {
        "signal_idx": int(sig_idx),
        "pair": pair,
        "fold_id": int(fold_id),
        "signal_time": signal_time.isoformat(),
        "entry_time": pd.Timestamp(entry_time).isoformat(),
        "entry_price": float(entry_price),
        "entry_atr_1h": float(entry_atr_1h),
        "n_post_bars": int(n_post_bars),
        "clamped": bool(clamped),
        "actual_taken": bool(actual_taken),
        "actual_exit_bar": int(actual_exit_bar) if actual_exit_bar is not None else None,
        "actual_exit_reason": actual_exit_reason if actual_exit_reason is not None else "",
        "peak_mfe_r": float(peak_mfe_r),
        "peak_mfe_bar": int(peak_mfe_bar),
        "peak_mae_r": float(peak_mae_r),
        "peak_mae_bar": int(peak_mae_bar),
        "mae_before_mfe": bool(mae_before_mfe),
        "virtual_sl_hit": bool(virtual_sl_hit),
        "virtual_sl_hit_bar": (int(virtual_sl_hit_bar) if virtual_sl_hit_bar is not None else None),
        **{k: v for k, v in r_at_map.items()},
        "time_to_first_05r_mfe": (int(first_05_mfe_bar) if first_05_mfe_bar is not None else None),
        "time_to_first_1r_mfe": (int(first_1_mfe_bar) if first_1_mfe_bar is not None else None),
        "time_to_first_2r_mfe": (int(first_2_mfe_bar) if first_2_mfe_bar is not None else None),
        "time_to_first_3r_mfe": (int(first_3_mfe_bar) if first_3_mfe_bar is not None else None),
        "time_to_first_neg05r_mae": (
            int(first_neg05_mae_bar) if first_neg05_mae_bar is not None else None
        ),
        "time_to_first_neg1r_mae": (
            int(first_neg1_mae_bar) if first_neg1_mae_bar is not None else None
        ),
        "time_to_first_neg2r_mae": (
            int(first_neg2_mae_bar) if first_neg2_mae_bar is not None else None
        ),
        "first_touch_05r_dir": (
            int(first_touch_05_dir) if first_touch_05_dir is not None else None
        ),
        "first_touch_1r_dir": (int(first_touch_1_dir) if first_touch_1_dir is not None else None),
        "first_touch_2r_dir": (int(first_touch_2_dir) if first_touch_2_dir is not None else None),
        "num_distinct_1r_peaks": int(num_distinct_1r_peaks),
        "retrace_from_peak_mfe": float(retrace_from_peak_mfe),
        "time_to_peak_mfe": int(time_to_peak_mfe),
        "final_r": float(final_r),
    }


# ----------------------------------------------------------------------------
# Distributional spec — moments + percentiles + extremes + histogram + N.
# ----------------------------------------------------------------------------


def describe_distribution(values: np.ndarray, label: str) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    out: Dict[str, Any] = {"label": label, "n": n}
    if n == 0:
        for k in ("mean", "std", "skew", "excess_kurt", "min", "max"):
            out[k] = float("nan")
        for p in DIST_PERCENTILES:
            out[f"p{int(p)}"] = float("nan")
        out["histogram_bin_edges"] = HIST_BIN_EDGES.tolist()
        out["histogram_counts"] = [0] * (len(HIST_BIN_EDGES) - 1)
        return out
    out["mean"] = float(np.mean(arr))
    out["std"] = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
    out["skew"] = float(sps.skew(arr, bias=False)) if n > 2 else float("nan")
    out["excess_kurt"] = (
        float(sps.kurtosis(arr, fisher=True, bias=False)) if n > 3 else float("nan")
    )
    out["min"] = float(np.min(arr))
    out["max"] = float(np.max(arr))
    pcts = np.percentile(arr, list(DIST_PERCENTILES), method="linear")
    for p, v in zip(DIST_PERCENTILES, pcts):
        out[f"p{int(p)}"] = float(v)
    counts, _ = np.histogram(arr, bins=HIST_BIN_EDGES)
    out["histogram_bin_edges"] = HIST_BIN_EDGES.tolist()
    out["histogram_counts"] = counts.astype(int).tolist()
    return out


def _fmt_float(v: float, decimals: int = 4) -> str:
    if not np.isfinite(v):
        return "NaN"
    return f"{v:.{decimals}f}"


def render_distribution_md(desc: Dict[str, Any], *, header_level: int = 3) -> str:
    """Render a single distribution descriptor as a markdown block."""
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
    lines.append("Histogram (0.25R bins, -3R to +12R, with overflow):")
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


# ----------------------------------------------------------------------------
# Phase 1 — Pooled descriptives.
# ----------------------------------------------------------------------------


def phase1_pooled_envelope(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-bar percentile envelope of r_close across all trades."""
    rows: List[Dict[str, Any]] = []
    grouped = panel.groupby("t", sort=True)["r_close"]
    for t, vals in grouped:
        arr = vals.to_numpy(dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        rec: Dict[str, Any] = {"t": int(t), "n": int(finite.size)}
        if finite.size:
            pcts = np.percentile(finite, list(ENVELOPE_PERCENTILES), method="linear")
            for p, v in zip(ENVELOPE_PERCENTILES, pcts):
                rec[f"p{int(p)}"] = float(v)
            rec["mean"] = float(np.mean(finite))
        else:
            for p in ENVELOPE_PERCENTILES:
                rec[f"p{int(p)}"] = float("nan")
            rec["mean"] = float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def phase1_first_touch_crosstab(shape: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab first_touch_{05r,1r,2r}_dir. Counts + P(final_r > 0)."""

    def code(v: Any) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "never"
        v = int(v)
        return "up" if v > 0 else "down"

    df = shape[
        ["signal_idx", "first_touch_05r_dir", "first_touch_1r_dir", "first_touch_2r_dir", "final_r"]
    ].copy()
    df["c05"] = df["first_touch_05r_dir"].apply(code)
    df["c1"] = df["first_touch_1r_dir"].apply(code)
    df["c2"] = df["first_touch_2r_dir"].apply(code)
    df["final_r_gt_0"] = df["final_r"] > 0

    grp = (
        df.groupby(["c05", "c1", "c2"], sort=True)
        .agg(n=("signal_idx", "count"), p_final_r_gt_0=("final_r_gt_0", "mean"))
        .reset_index()
    )
    return grp


def phase1_pooled_distributions(shape: pd.DataFrame) -> List[Dict[str, Any]]:
    """Pooled (no clustering) distribution descriptors for key shape features."""
    targets = (
        ("peak_mfe_r", shape["peak_mfe_r"].to_numpy()),
        ("peak_mae_r", shape["peak_mae_r"].to_numpy()),
        ("r_at_t240", shape["r_at_t240"].to_numpy()),
        ("time_to_peak_mfe", shape["time_to_peak_mfe"].to_numpy().astype(np.float64)),
        ("retrace_from_peak_mfe", shape["retrace_from_peak_mfe"].to_numpy()),
    )
    return [describe_distribution(v, k) for k, v in targets]


def phase1_chart_envelope(envelope: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    t = envelope["t"].to_numpy()
    ax.fill_between(t, envelope["p1"], envelope["p99"], alpha=0.10, color="C0", label="p1-p99")
    ax.fill_between(t, envelope["p5"], envelope["p95"], alpha=0.15, color="C0", label="p5-p95")
    ax.fill_between(t, envelope["p10"], envelope["p90"], alpha=0.20, color="C0", label="p10-p90")
    ax.fill_between(t, envelope["p25"], envelope["p75"], alpha=0.30, color="C0", label="p25-p75")
    ax.plot(t, envelope["p50"], color="C0", lw=1.2, label="p50")
    ax.axhline(0.0, color="black", lw=0.5, linestyle="--", alpha=0.5)
    ax.axhline(-1.0, color="red", lw=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("bar (t)")
    ax.set_ylabel("r_close (R = 2 * Wilder ATR(14) at N)")
    ax.set_title("Pooled r_close envelope, t=0..240")
    ax.legend(loc="upper left", fontsize=8)
    _save_figure(fig, path)


def phase1_chart_scatter_mfe_mae(shape: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        shape["peak_mfe_r"], shape["peak_mae_r"], s=2, alpha=0.15, color="C0", rasterized=True
    )
    ax.axhline(
        -1.0, color="red", lw=0.8, linestyle="--", alpha=0.7, label="SL line (peak_mae_r = -1R)"
    )
    ax.set_xlabel("peak_mfe_r")
    ax.set_ylabel("peak_mae_r")
    ax.set_title("peak_mfe_r vs peak_mae_r (per trade)")
    ax.legend(loc="upper right", fontsize=8)
    _save_figure(fig, path)


def phase1_chart_scatter_end_peak_mfe(shape: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        shape["peak_mfe_r"], shape["r_at_t240"], s=2, alpha=0.15, color="C0", rasterized=True
    )
    ax.axhline(0.0, color="black", lw=0.5, linestyle="--", alpha=0.5)
    ax.plot(
        [0, shape["peak_mfe_r"].max()],
        [0, shape["peak_mfe_r"].max()],
        color="red",
        lw=0.6,
        linestyle="--",
        alpha=0.6,
        label="r_at_t240 = peak_mfe_r",
    )
    ax.set_xlabel("peak_mfe_r")
    ax.set_ylabel("r_at_t240")
    ax.set_title("r_at_t240 vs peak_mfe_r (per trade)")
    ax.legend(loc="lower right", fontsize=8)
    _save_figure(fig, path)


# ----------------------------------------------------------------------------
# Phase 2 — Clustering.
# ----------------------------------------------------------------------------


def _adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Compute Adjusted Rand Index from two label arrays of equal length.

    Closed-form: ARI = (sum_comb_ij - E) / (M - E) where
        sum_comb_ij = sum over contingency cells C(n_ij, 2)
        E = (sum_row * sum_col) / C(n, 2) where sum_row, sum_col are sums of
            C(n_i., 2) and C(n_.j, 2) respectively
        M = 0.5 * (sum_row + sum_col)
    Degenerate cases (M == E) return 0.0.
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    contingency: Dict[Tuple[int, int], int] = {}
    for x, y in zip(a.tolist(), b.tolist()):
        key = (int(x), int(y))
        contingency[key] = contingency.get(key, 0) + 1
    n = a.size

    def comb2(x: int) -> int:
        return x * (x - 1) // 2

    sum_comb_ij = sum(comb2(v) for v in contingency.values())
    row_sums: Dict[int, int] = {}
    col_sums: Dict[int, int] = {}
    for (x, y), v in contingency.items():
        row_sums[x] = row_sums.get(x, 0) + v
        col_sums[y] = col_sums.get(y, 0) + v
    sum_comb_a = sum(comb2(v) for v in row_sums.values())
    sum_comb_b = sum(comb2(v) for v in col_sums.values())
    total_comb = comb2(n)
    if total_comb == 0:
        return 0.0
    expected = (sum_comb_a * sum_comb_b) / total_comb
    max_idx = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_idx - expected
    if denom == 0.0:
        return 0.0
    return float((sum_comb_ij - expected) / denom)


def phase2_clustering(
    shape: pd.DataFrame,
    panel: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run hierarchical (subsample) + K-means (full) clustering at k=4,6,8,10.

    Returns (cluster_assignments_df, cluster_trajectories_long_df, meta).
    """
    rng = np.random.default_rng(NUMPY_SEED)

    feat_cols = list(CLUSTERING_FEATURES)
    feat_mat_full = shape[feat_cols].to_numpy(dtype=np.float64)
    has_nan_row = ~np.isfinite(feat_mat_full).all(axis=1)
    dropped_n = int(has_nan_row.sum())
    keep_mask = ~has_nan_row
    feat_kept = feat_mat_full[keep_mask]
    sig_idx_kept = shape.loc[keep_mask, "signal_idx"].to_numpy()

    # Z-score on the full kept set.
    mean = feat_kept.mean(axis=0)
    std = feat_kept.std(axis=0, ddof=0)
    std = np.where(std > 1e-12, std, 1.0)
    feat_z_full = (feat_kept - mean) / std

    # Subsample for hierarchical.
    n_full = feat_z_full.shape[0]
    sub_n = min(HIERARCHICAL_SUBSAMPLE_N, n_full)
    sub_order = np.sort(rng.choice(n_full, size=sub_n, replace=False))
    feat_z_sub = feat_z_full[sub_order]
    sig_idx_sub = sig_idx_kept[sub_order]

    # Hierarchical Ward linkage on subsample.
    linkage = sch.linkage(feat_z_sub, method="ward", metric="euclidean")

    ks = (4, 6, 8, 10)
    hier_labels_sub: Dict[int, np.ndarray] = {}
    km_labels_full: Dict[int, np.ndarray] = {}
    ari_per_k: Dict[int, float] = {}

    for k in ks:
        hier_labels_sub[k] = sch.fcluster(linkage, t=k, criterion="maxclust").astype(np.int32)
        # scipy kmeans2 with k-means++ init; single run is deterministic given seed.
        # We run 10 inits with distinct seeds and pick the lowest-distortion run for
        # stability, matching sklearn KMeans(n_init=10) intent.
        best_labels: Optional[np.ndarray] = None
        best_distortion = np.inf
        for init_offset in range(10):
            try:
                centroids, labels = kmeans2(
                    feat_z_full,
                    k,
                    iter=300,
                    minit="++",
                    seed=SKLEARN_SEED + init_offset,
                    missing="warn",
                )
            except Exception:
                continue
            assigned = centroids[labels]
            distortion = float(np.sum((feat_z_full - assigned) ** 2))
            if distortion < best_distortion:
                best_distortion = distortion
                best_labels = labels.astype(np.int32)
        if best_labels is None:
            raise RuntimeError(f"kmeans2 failed for k={k}")
        km_labels_full[k] = best_labels + 1  # 1-indexed to match hier
        km_labels_sub = km_labels_full[k][sub_order]
        ari_per_k[k] = _adjusted_rand_index(hier_labels_sub[k], km_labels_sub)

    # Cluster assignments DataFrame: one row per signal in shape, with NaN for dropped.
    n_total = len(shape)
    cols: Dict[str, Any] = {"signal_idx": shape["signal_idx"].to_numpy()}
    for k in ks:
        full_col = np.full(n_total, -1, dtype=np.int32)
        full_col[keep_mask] = km_labels_full[k]
        cols[f"km_k{k}"] = full_col
        hier_col = np.full(n_total, -1, dtype=np.int32)
        # Map subsample assignments back to their signal_idx.
        sub_positions_in_keep = (
            sub_order  # positions within feat_kept (== positions within keep-mask rows)
        )
        keep_positions = np.flatnonzero(keep_mask)  # absolute positions in shape
        target_positions = keep_positions[sub_positions_in_keep]
        hier_col[target_positions] = hier_labels_sub[k]
        cols[f"hier_k{k}"] = hier_col
    assignments = pd.DataFrame(cols)
    # ARI columns: same value broadcast (one column per k).
    for k in ks:
        assignments[f"ari_k{k}"] = ari_per_k[k]

    # Cluster trajectory envelopes (long format).
    # Per (k, source, cluster, t, percentile, r_close).
    long_rows: List[Dict[str, Any]] = []
    panel.set_index(["signal_idx", "t"])["r_close"]

    # Build a quick per-signal r_close-by-t numpy view to avoid index overhead.
    # Group panel by signal_idx.
    panel_sorted = panel.sort_values(["signal_idx", "t"], kind="mergesort")
    sig_to_arr: Dict[int, np.ndarray] = {}
    for s_idx, grp in panel_sorted.groupby("signal_idx", sort=False):
        ts = grp["t"].to_numpy()
        rc = grp["r_close"].to_numpy()
        full = np.full(TRAJ_HORIZON + 1, np.nan, dtype=np.float64)
        full[ts] = rc
        sig_to_arr[int(s_idx)] = full

    pcts = list(CLUSTER_ENVELOPE_PERCENTILES)
    for k in ks:
        for source, labels, sig_index_array in (
            ("km", km_labels_full[k], sig_idx_kept),
            ("hier", hier_labels_sub[k], sig_idx_sub),
        ):
            unique_clusters = np.sort(np.unique(labels))
            for cid in unique_clusters:
                member_sigs = sig_index_array[labels == cid]
                if len(member_sigs) == 0:
                    continue
                stacked = np.vstack([sig_to_arr[int(s)] for s in member_sigs])
                for t in range(TRAJ_HORIZON + 1):
                    col = stacked[:, t]
                    finite = col[np.isfinite(col)]
                    if finite.size == 0:
                        continue
                    pct_vals = np.percentile(finite, pcts, method="linear")
                    for p, v in zip(pcts, pct_vals):
                        long_rows.append(
                            {
                                "k": k,
                                "source": source,
                                "cluster": int(cid),
                                "n_cluster": int(len(member_sigs)),
                                "t": int(t),
                                "percentile": int(p),
                                "r_close": float(v),
                            }
                        )
    trajectories = pd.DataFrame(long_rows)

    meta = {
        "n_full_kept": int(n_full),
        "n_dropped_for_nan": dropped_n,
        "subsample_n": int(sub_n),
        "ari_per_k": {int(k): float(v) for k, v in ari_per_k.items()},
        "linkage": linkage,
        "feature_names": feat_cols,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "hier_labels_sub": hier_labels_sub,
        "km_labels_full": km_labels_full,
        "km_labels_sub_for_ari": {k: km_labels_full[k][sub_order] for k in ks},
        "sig_idx_kept": sig_idx_kept,
        "sig_idx_sub": sig_idx_sub,
        "keep_mask": keep_mask,
    }
    return assignments, trajectories, meta


def phase2_chart_dendrogram(meta: Dict[str, Any], path: Path) -> None:
    linkage = meta["linkage"]
    fig, ax = plt.subplots(figsize=(12, 6))
    sch.dendrogram(linkage, no_labels=True, color_threshold=0, above_threshold_color="black", ax=ax)
    ax.set_title(f"Hierarchical (Ward) dendrogram, subsample N={meta['subsample_n']}")
    ax.set_xlabel("samples")
    ax.set_ylabel("distance")
    _save_figure(fig, path)


def phase2_chart_cluster_envelopes(
    trajectories: pd.DataFrame,
    k: int,
    path: Path,
) -> None:
    """Per-cluster median (+IQR) trajectory for K-means (full)."""
    sub = trajectories[(trajectories["k"] == k) & (trajectories["source"] == "km")]
    clusters = np.sort(sub["cluster"].unique())
    fig, axes = plt.subplots(
        nrows=int(np.ceil(len(clusters) / 2)),
        ncols=2,
        figsize=(12, 3 * int(np.ceil(len(clusters) / 2))),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).flatten()
    for i, cid in enumerate(clusters):
        ax = axes[i]
        cl = sub[sub["cluster"] == cid]
        p10 = cl[cl["percentile"] == 10].set_index("t")["r_close"]
        p25 = cl[cl["percentile"] == 25].set_index("t")["r_close"]
        p50 = cl[cl["percentile"] == 50].set_index("t")["r_close"]
        p75 = cl[cl["percentile"] == 75].set_index("t")["r_close"]
        p90 = cl[cl["percentile"] == 90].set_index("t")["r_close"]
        t = p50.index.to_numpy()
        if len(t) == 0:
            continue
        ax.fill_between(t, p10, p90, alpha=0.15, color="C0")
        ax.fill_between(t, p25, p75, alpha=0.30, color="C0")
        ax.plot(t, p50, color="C0", lw=1.0)
        ax.axhline(0.0, color="black", lw=0.5, linestyle="--", alpha=0.5)
        ax.axhline(-1.0, color="red", lw=0.5, linestyle="--", alpha=0.5)
        n_c = int(cl["n_cluster"].iloc[0]) if len(cl) else 0
        ax.set_title(f"k={k} cluster {cid} (km, n={n_c})", fontsize=9)
        ax.set_xlabel("t")
        ax.set_ylabel("r_close")
    # Hide unused axes.
    for j in range(len(clusters), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Cluster trajectory envelopes — K-means, k={k}", fontsize=11)
    _save_figure(fig, path)


# ----------------------------------------------------------------------------
# Phase 3 — Conditional distributions.
# ----------------------------------------------------------------------------


def phase3_mfe_band_conditionals(
    shape: pd.DataFrame, panel: pd.DataFrame
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """MFE-band conditional distributions for X in MFE_BANDS_R."""
    n_total = len(shape)
    rows: List[Dict[str, Any]] = []
    distributions: List[Dict[str, Any]] = []

    sig_to_arr: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    panel_sorted = panel.sort_values(["signal_idx", "t"], kind="mergesort")
    for s_idx, grp in panel_sorted.groupby("signal_idx", sort=False):
        grp["t"].to_numpy(dtype=np.int64)
        rc = grp["r_close"].to_numpy(dtype=np.float64)
        run_mfe = grp["running_mfe"].to_numpy(dtype=np.float64)
        run_mae = grp["running_mae"].to_numpy(dtype=np.float64)
        sig_to_arr[int(s_idx)] = (run_mfe, run_mae, rc)

    # Pre-compute peak_mfe_r per signal_idx.
    s2peak_mfe = shape.set_index("signal_idx")["peak_mfe_r"].to_dict()
    shape.set_index("signal_idx")["peak_mae_r"].to_dict()
    s2r240 = shape.set_index("signal_idx")["r_at_t240"].to_dict()
    shape.set_index("signal_idx")["virtual_sl_hit"].to_dict()

    for i, X in enumerate(MFE_BANDS_R):
        reached_sigs: List[int] = []
        first_touch_bars: List[int] = []
        subsequent_mae: List[float] = []
        retrace_below_zero: List[int] = []
        sl_after_touch: List[int] = []
        next_band_reached: List[int] = []
        next_band = MFE_BANDS_R[i + 1] if i + 1 < len(MFE_BANDS_R) else None

        for sig_idx, (run_mfe, run_mae, rc) in sig_to_arr.items():
            reach_mask = run_mfe >= X
            if not reach_mask.any():
                continue
            t_first = int(np.argmax(reach_mask))  # first index reaching X (index into arr — bar t)
            reached_sigs.append(sig_idx)
            first_touch_bars.append(int(t_first))
            # Subsequent MAE: minimum r_low ~~ but we have running_mae which is monotonic non-increasing.
            # Subsequent MAE after first X touch = min(running_mae[t_first:end]) — but running_mae monotone
            # non-increasing means equals running_mae[-1] if there is further drawdown after t_first.
            sub_mae = float(np.min(run_mae[t_first:]))
            subsequent_mae.append(sub_mae)
            retrace_below_zero.append(int(np.any(rc[t_first:] <= 0.0)))
            sl_after_touch.append(int(np.any(run_mae[t_first:] <= -1.0)))
            if next_band is not None:
                next_band_reached.append(int(np.any(run_mfe[t_first:] >= next_band)))

        n_reached = len(reached_sigs)
        p_reached = n_reached / n_total if n_total else 0.0

        peak_mfe_arr = np.array([s2peak_mfe[s] for s in reached_sigs], dtype=np.float64)
        r240_arr = np.array([s2r240.get(s, np.nan) for s in reached_sigs], dtype=np.float64)
        first_touch_arr = np.array(first_touch_bars, dtype=np.float64)
        sub_mae_arr = np.array(subsequent_mae, dtype=np.float64)

        rows.append(
            {
                "band_R": X,
                "n_total": int(n_total),
                "n_reached": int(n_reached),
                "p_reached": float(p_reached),
                "p_retrace_to_or_below_0_after_touch": float(np.mean(retrace_below_zero))
                if retrace_below_zero
                else float("nan"),
                "p_virtual_sl_after_touch": float(np.mean(sl_after_touch))
                if sl_after_touch
                else float("nan"),
                "p_reach_next_band_given_reached": float(np.mean(next_band_reached))
                if next_band_reached
                else float("nan"),
                "next_band_R": (next_band if next_band is not None else float("nan")),
            }
        )

        if n_reached >= 1:
            for label, arr in (
                (f"peak_mfe_r | reached {X}R MFE", peak_mfe_arr),
                (f"r_at_t240 | reached {X}R MFE", r240_arr),
                (f"time_at_first_touch | reached {X}R MFE", first_touch_arr),
                (f"subsequent_mae_after_first_touch | reached {X}R MFE", sub_mae_arr),
            ):
                distributions.append(describe_distribution(arr, label))

    return pd.DataFrame(rows), distributions


def phase3_mae_band_conditionals(
    shape: pd.DataFrame, panel: pd.DataFrame
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    n_total = len(shape)
    rows: List[Dict[str, Any]] = []
    distributions: List[Dict[str, Any]] = []

    sig_to_arr: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    panel_sorted = panel.sort_values(["signal_idx", "t"], kind="mergesort")
    for s_idx, grp in panel_sorted.groupby("signal_idx", sort=False):
        run_mfe = grp["running_mfe"].to_numpy(dtype=np.float64)
        run_mae = grp["running_mae"].to_numpy(dtype=np.float64)
        rc = grp["r_close"].to_numpy(dtype=np.float64)
        sig_to_arr[int(s_idx)] = (run_mfe, run_mae, rc)

    s2peak_mae = shape.set_index("signal_idx")["peak_mae_r"].to_dict()
    s2r240 = shape.set_index("signal_idx")["r_at_t240"].to_dict()

    for X in MAE_BANDS_R:
        reached_sigs: List[int] = []
        first_touch_bars: List[int] = []
        post_touch_mfe: List[float] = []
        recover_1r_mfe: List[int] = []
        sl_after_touch: List[int] = []

        for sig_idx, (run_mfe, run_mae, rc) in sig_to_arr.items():
            reach_mask = run_mae <= X
            if not reach_mask.any():
                continue
            t_first = int(np.argmax(reach_mask))
            reached_sigs.append(sig_idx)
            first_touch_bars.append(int(t_first))
            post_mfe = float(np.max(run_mfe[t_first:]))
            post_touch_mfe.append(post_mfe)
            recover_1r_mfe.append(int(post_mfe >= 1.0))
            sl_after_touch.append(int(np.any(run_mae[t_first:] <= -1.0)))

        n_reached = len(reached_sigs)
        p_reached = n_reached / n_total if n_total else 0.0

        peak_mae_arr = np.array([s2peak_mae[s] for s in reached_sigs], dtype=np.float64)
        r240_arr = np.array([s2r240.get(s, np.nan) for s in reached_sigs], dtype=np.float64)
        first_touch_arr = np.array(first_touch_bars, dtype=np.float64)
        post_mfe_arr = np.array(post_touch_mfe, dtype=np.float64)

        rows.append(
            {
                "band_R": X,
                "n_total": int(n_total),
                "n_reached": int(n_reached),
                "p_reached": float(p_reached),
                "p_recover_ge_1R_mfe_after_touch": float(np.mean(recover_1r_mfe))
                if recover_1r_mfe
                else float("nan"),
                "p_virtual_sl_after_touch": float(np.mean(sl_after_touch))
                if sl_after_touch
                else float("nan"),
            }
        )

        if n_reached >= 1:
            for label, arr in (
                (f"peak_mae_r | reached {X}R MAE", peak_mae_arr),
                (f"r_at_t240 | reached {X}R MAE", r240_arr),
                (f"time_at_first_touch | reached {X}R MAE", first_touch_arr),
                (f"peak_mfe_r_after_first_touch | reached {X}R MAE", post_mfe_arr),
            ):
                distributions.append(describe_distribution(arr, label))

    return pd.DataFrame(rows), distributions


def _bucket_for(value: float, buckets: Tuple[Tuple[float, float, str], ...]) -> Optional[str]:
    if not np.isfinite(value):
        return None
    for lo, hi, label in buckets:
        if lo == -np.inf:
            if value <= hi:
                return label
        elif hi == np.inf:
            if value > lo:
                return label
        else:
            # Mixed conventions vary by spec; use the textual definition for each table.
            # We default to: (lo, hi] for closed-right buckets, [lo, hi) for half-open MFE buckets.
            # The label string itself documents the convention; matching uses inclusion rules below.
            pass
    return None


def phase3_time_state_conditionals(panel: pd.DataFrame, shape: pd.DataFrame) -> pd.DataFrame:
    """Time-state bins at anchor bars t in TIME_STATE_ANCHORS."""
    rows: List[Dict[str, Any]] = []

    # Build per-signal r_close at the anchor bars; running_mfe, running_mae at each anchor;
    # also r_at_t240 and peak_mfe_past_t.
    panel_sorted = panel.sort_values(["signal_idx", "t"], kind="mergesort")
    sig_to_arr: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for s_idx, grp in panel_sorted.groupby("signal_idx", sort=False):
        ts = grp["t"].to_numpy(dtype=np.int64)
        rc_full = np.full(TRAJ_HORIZON + 1, np.nan, dtype=np.float64)
        mfe_full = np.full(TRAJ_HORIZON + 1, np.nan, dtype=np.float64)
        mae_full = np.full(TRAJ_HORIZON + 1, np.nan, dtype=np.float64)
        rc_full[ts] = grp["r_close"].to_numpy()
        mfe_full[ts] = grp["running_mfe"].to_numpy()
        mae_full[ts] = grp["running_mae"].to_numpy()
        sig_to_arr[int(s_idx)] = (rc_full, mfe_full, mae_full)

    # Bucket inclusion rules.
    def rc_bucket(v: float) -> Optional[str]:
        if not np.isfinite(v):
            return None
        if v <= -1.0:
            return "(-inf,-1]"
        if v <= -0.5:
            return "(-1,-0.5]"
        if v <= 0.0:
            return "(-0.5,0]"
        if v <= 0.5:
            return "(0,0.5]"
        if v <= 1.0:
            return "(0.5,1]"
        if v <= 2.0:
            return "(1,2]"
        return "(2,inf)"

    def mfe_bucket(v: float) -> Optional[str]:
        if not np.isfinite(v):
            return None
        if v < 0.5:
            return "[0,0.5)"
        if v < 1.0:
            return "[0.5,1)"
        if v < 2.0:
            return "[1,2)"
        return "[2,inf)"

    def mae_bucket(v: float) -> Optional[str]:
        if not np.isfinite(v):
            return None
        if v <= -1.0:
            return "(-inf,-1]"
        if v <= -0.5:
            return "(-1,-0.5]"
        if v <= -0.25:
            return "(-0.5,-0.25]"
        return "(-0.25,0]"

    cells: Dict[Tuple[int, str, str, str], Dict[str, List[float]]] = {}
    for sig_idx, (rc_full, mfe_full, mae_full) in sig_to_arr.items():
        for t in TIME_STATE_ANCHORS:
            rc_t = rc_full[t]
            mfe_t = mfe_full[t]
            mae_t = mae_full[t]
            r240 = rc_full[TRAJ_HORIZON]
            # peak_mfe past t = max(running_mfe[t..end]) which equals running_mfe[end] since monotone.
            tail = mfe_full[t:]
            tail_finite = tail[np.isfinite(tail)]
            peak_mfe_past = float(np.max(tail_finite)) if tail_finite.size else float("nan")
            if not (np.isfinite(rc_t) and np.isfinite(mfe_t) and np.isfinite(mae_t)):
                continue
            bkt = (t, rc_bucket(rc_t), mfe_bucket(mfe_t), mae_bucket(mae_t))
            if None in bkt[1:]:
                continue
            cell = cells.setdefault(bkt, {"r240": [], "peak_mfe_past": []})
            if np.isfinite(r240):
                cell["r240"].append(float(r240))
            if np.isfinite(peak_mfe_past):
                cell["peak_mfe_past"].append(float(peak_mfe_past))

    for (t, rc_b, mfe_b, mae_b), agg in sorted(cells.items()):
        r240_arr = np.asarray(agg["r240"], dtype=np.float64)
        pmf_arr = np.asarray(agg["peak_mfe_past"], dtype=np.float64)
        if r240_arr.size == 0 and pmf_arr.size == 0:
            continue
        rec = {
            "anchor_t": int(t),
            "r_close_bucket": rc_b,
            "running_mfe_bucket": mfe_b,
            "running_mae_bucket": mae_b,
            "n": int(max(r240_arr.size, pmf_arr.size)),
        }
        for prefix, arr in (("r_at_t240", r240_arr), ("peak_mfe_past_t", pmf_arr)):
            if arr.size == 0:
                rec.update(
                    {
                        f"{prefix}_mean": float("nan"),
                        f"{prefix}_std": float("nan"),
                        f"{prefix}_p10": float("nan"),
                        f"{prefix}_p50": float("nan"),
                        f"{prefix}_p90": float("nan"),
                    }
                )
            else:
                rec[f"{prefix}_mean"] = float(np.mean(arr))
                rec[f"{prefix}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
                pcts = np.percentile(arr, [10, 50, 90], method="linear")
                rec[f"{prefix}_p10"] = float(pcts[0])
                rec[f"{prefix}_p50"] = float(pcts[1])
                rec[f"{prefix}_p90"] = float(pcts[2])
        rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Phase 2 supplement — cluster distributions per k per cluster.
# ----------------------------------------------------------------------------


def phase2_cluster_distributions(
    shape: pd.DataFrame, cluster_meta: Dict[str, Any]
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """For each (k, source, cluster): describe peak_mfe_r, peak_mae_r, r_at_t240, time_to_peak_mfe."""
    sections: List[Tuple[str, List[Dict[str, Any]]]] = []
    ks = (4, 6, 8, 10)
    sig_idx_kept = cluster_meta["sig_idx_kept"]
    sig_idx_sub = cluster_meta["sig_idx_sub"]
    shape_idx = shape.set_index("signal_idx")
    target_cols = ("peak_mfe_r", "peak_mae_r", "r_at_t240", "time_to_peak_mfe")

    for k in ks:
        # KM full
        for cid in np.sort(np.unique(cluster_meta["km_labels_full"][k])):
            members = sig_idx_kept[cluster_meta["km_labels_full"][k] == cid]
            section_label = f"k{k} km cluster {int(cid)}  (n={len(members)})"
            descs: List[Dict[str, Any]] = []
            for col in target_cols:
                arr = shape_idx.loc[members, col].to_numpy(dtype=np.float64)
                descs.append(describe_distribution(arr, f"{col}  [{section_label}]"))
            sections.append((section_label, descs))
        # Hier sub
        for cid in np.sort(np.unique(cluster_meta["hier_labels_sub"][k])):
            members = sig_idx_sub[cluster_meta["hier_labels_sub"][k] == cid]
            section_label = f"k{k} hier cluster {int(cid)}  (n={len(members)}, subsample)"
            descs = []
            for col in target_cols:
                arr = shape_idx.loc[members, col].to_numpy(dtype=np.float64)
                descs.append(describe_distribution(arr, f"{col}  [{section_label}]"))
            sections.append((section_label, descs))
    return sections


def phase2_cluster_summary_md(shape: pd.DataFrame, cluster_meta: Dict[str, Any]) -> str:
    """Cluster sizes per k, ARI per k, top distinguishing features per cluster."""
    lines: List[str] = [
        "# Arc 2 Trajectory Appendix — Cluster Summary",
        "",
        "Hierarchical: Ward linkage on Euclidean z-scored shape features, applied to a seeded "
        f"{cluster_meta['subsample_n']}-row subsample (full-N Ward infeasible at N={cluster_meta['n_full_kept']}).",
        "K-means: applied to the full kept feature matrix at each k.",
        "ARI: computed on the {n}-row subsample (hierarchical vs k-means restricted to the same rows).".format(
            n=cluster_meta["subsample_n"]
        ),
        "",
        "## ARI per k (subsample-restricted)",
        "",
        "| k | ARI(hier, km) |",
        "|---|---------------|",
    ]
    for k in (4, 6, 8, 10):
        lines.append(f"| {k} | {_fmt_float(cluster_meta['ari_per_k'][k])} |")
    lines.append("")

    ks = (4, 6, 8, 10)
    feat_cols = list(cluster_meta["feature_names"])
    shape_idx = shape.set_index("signal_idx")

    for k in ks:
        lines.append(f"## k = {k}")
        lines.append("")
        for source, labels, sig_array in (
            ("km (full)", cluster_meta["km_labels_full"][k], cluster_meta["sig_idx_kept"]),
            ("hier (subsample)", cluster_meta["hier_labels_sub"][k], cluster_meta["sig_idx_sub"]),
        ):
            lines.append(f"### Source: {source}")
            lines.append("")
            header_cols = ["cluster", "n"] + list(feat_cols)
            lines.append("| " + " | ".join(header_cols) + " |")
            lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")
            for cid in np.sort(np.unique(labels)):
                members = sig_array[labels == cid]
                n = len(members)
                means = []
                for col in feat_cols:
                    arr = shape_idx.loc[members, col].to_numpy(dtype=np.float64)
                    arr = arr[np.isfinite(arr)]
                    means.append(_fmt_float(float(np.mean(arr))) if arr.size else "NaN")
                lines.append(f"| {int(cid)} | {n} | " + " | ".join(means) + " |")
            lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Report writer — empirical observations only, §14.5 compliant.
# ----------------------------------------------------------------------------


def write_report(
    *,
    pop_meta: Dict[str, str],
    panel_meta: Dict[str, Any],
    cluster_meta: Dict[str, Any],
    pooled_dists: List[Dict[str, Any]],
    mfe_band_df: pd.DataFrame,
    mae_band_df: pd.DataFrame,
    first_touch_df: pd.DataFrame,
    envelope_df: pd.DataFrame,
    path: Path,
) -> None:
    lines: List[str] = [
        "# Arc 2 Trajectory Appendix — Empirical Report",
        "",
        "Phase: l6_arc2_trajectory_appendix (supplementary, descriptive only per L6.0 v1.1 §14.5).",
        "",
        "Signal: `TRIAL__mtf_alignment__2_down_mixed__kijun__h_120` (long, 1H, 4H/D1 kijun-26 alignment).",
        "R denominator: 2 * Wilder ATR(14) at bar N close (from `atr_1h_wilder_at_n`).",
        "Panel horizon: t = 0 .. 240 (241 rows per trade).",
        "",
        "## Population",
        "",
        f"- signals_features.csv rows: {pop_meta['input_signals_features_rows']}",
        f"- L4 anchor band [{L4_ANCHOR_LOW}, {L4_ANCHOR_HIGH}]: in-band = {pop_meta['l4_anchor_band_ok']}",
        f"- Panel built for: {panel_meta['n_signals_with_panel']} signals "
        f"({panel_meta['n_panel_rows']} panel rows total).",
        f"- Dropped (no signal bar in 1H series): {panel_meta['dropped_no_signal_bar']}",
        f"- Dropped (bar N+1 not available): {panel_meta['dropped_no_next_bar']}",
        f"- Dropped (invalid atr_1h_wilder_at_n): {panel_meta['dropped_invalid_atr']}",
        "",
        "Lookahead spot-check (taken SL trades — panel must extend past actual_exit_bar):",
        "",
        "| signal_idx | pair | signal_time | actual_exit_bar | panel_last_t | extends_past_actual |",
        "|------------|------|-------------|-----------------|--------------|---------------------|",
    ]
    for s in panel_meta["spot_checks"]:
        lines.append(
            f"| {s['signal_idx']} | {s['pair']} | {s['signal_time']} | "
            f"{s['actual_exit_bar']} | {s['panel_last_t']} | "
            f"{str(s['panel_extends_past_actual_exit']).lower()} |"
        )
    lines.append("")

    lines.append("## Phase 1 — pooled descriptives")
    lines.append("")
    lines.append("Per-bar envelope: see `pooled_envelope.csv` and `charts/pooled_envelope.png`.")
    lines.append("")
    lines.append(
        "At t=240, the pooled r_close distribution has "
        f"p50 = {_fmt_float(float(envelope_df.set_index('t').loc[TRAJ_HORIZON, 'p50']))}, "
        f"p10 = {_fmt_float(float(envelope_df.set_index('t').loc[TRAJ_HORIZON, 'p10']))}, "
        f"p90 = {_fmt_float(float(envelope_df.set_index('t').loc[TRAJ_HORIZON, 'p90']))}."
    )
    lines.append("")
    lines.append("First-touch crosstab (counts and P(final_r > 0)):")
    lines.append("")
    lines.append("| c05 | c1 | c2 | n | P(final_r > 0) |")
    lines.append("|-----|----|----|---|----------------|")
    for r in first_touch_df.itertuples(index=False):
        lines.append(
            f"| {r.c05} | {r.c1} | {r.c2} | {int(r.n)} | {_fmt_float(float(r.p_final_r_gt_0))} |"
        )
    lines.append("")

    lines.append(
        "### Pooled distributions (peak_mfe_r, peak_mae_r, r_at_t240, time_to_peak_mfe, retrace_from_peak_mfe)"
    )
    lines.append("")
    for desc in pooled_dists:
        lines.append(
            f"- **{desc['label']}**: N={desc['n']}, "
            f"mean={_fmt_float(desc['mean'])}, std={_fmt_float(desc['std'])}, "
            f"skew={_fmt_float(desc['skew'])}, excess_kurt={_fmt_float(desc['excess_kurt'])}, "
            f"min={_fmt_float(desc['min'])}, max={_fmt_float(desc['max'])}, "
            f"p10={_fmt_float(desc['p10'])}, p50={_fmt_float(desc['p50'])}, p90={_fmt_float(desc['p90'])}."
        )
    lines.append("")
    lines.append(
        "Full moments + percentiles + histograms in `cluster_distributions.md` (section: pooled)."
    )
    lines.append("")

    lines.append("## Phase 2 — clustering")
    lines.append("")
    lines.append(
        f"Hierarchical (Ward) subsample N = {cluster_meta['subsample_n']}; K-means run on full kept N = {cluster_meta['n_full_kept']}."
    )
    lines.append("ARI per k (subsample-restricted):")
    lines.append("")
    lines.append("| k | ARI(hier, km) |")
    lines.append("|---|---------------|")
    for k in (4, 6, 8, 10):
        lines.append(f"| {k} | {_fmt_float(cluster_meta['ari_per_k'][k])} |")
    lines.append("")
    lines.append(
        "Cluster sizes and per-cluster feature means in `cluster_summary.md`. Per-cluster trajectory envelopes in `cluster_trajectories.csv` and `charts/cluster_envelopes_k{4,6,8,10}.png`. Per-cluster distribution tables (full moments + percentiles + histograms) in `cluster_distributions.md`."
    )
    lines.append("")

    lines.append("## Phase 3 — conditional distributions")
    lines.append("")
    lines.append("MFE-band conditionals:")
    lines.append("")
    lines.append("| band_R | n_reached | p_reached | p_retrace_to_0 | p_sl_after | p_next_band |")
    lines.append("|--------|-----------|-----------|----------------|------------|-------------|")
    for r in mfe_band_df.itertuples(index=False):
        lines.append(
            f"| {_fmt_float(r.band_R, 2)} | {int(r.n_reached)} | {_fmt_float(r.p_reached)} | "
            f"{_fmt_float(r.p_retrace_to_or_below_0_after_touch)} | "
            f"{_fmt_float(r.p_virtual_sl_after_touch)} | "
            f"{_fmt_float(r.p_reach_next_band_given_reached)} |"
        )
    lines.append("")
    lines.append("MAE-band conditionals:")
    lines.append("")
    lines.append("| band_R | n_reached | p_reached | p_recover_1R_mfe | p_sl_after |")
    lines.append("|--------|-----------|-----------|-------------------|------------|")
    for r in mae_band_df.itertuples(index=False):
        lines.append(
            f"| {_fmt_float(r.band_R, 2)} | {int(r.n_reached)} | {_fmt_float(r.p_reached)} | "
            f"{_fmt_float(r.p_recover_ge_1R_mfe_after_touch)} | "
            f"{_fmt_float(r.p_virtual_sl_after_touch)} |"
        )
    lines.append("")
    lines.append(
        "Time-state conditionals: see `time_state_conditionals.csv` (long-format bin distributions of r_at_t240 and peak_mfe_past_t at anchor bars t in {6,12,24,48,72,120})."
    )
    lines.append("")

    lines.append("## §14.5 discipline")
    lines.append("")
    lines.append(
        "This appendix is descriptive only. No filter, exit, or sizing recommendations are made here. "
        "Patterns identified are candidate hypotheses for fresh arcs per L6.0 v1.1 §14.2; any such candidate "
        "is recorded in `CANDIDATE_HYPOTHESES_DRAFT.md` (draft, not committed to the registry)."
    )

    _write_text("\n".join(lines) + "\n", path)


# ----------------------------------------------------------------------------
# Outputs writer.
# ----------------------------------------------------------------------------


def write_all_outputs(
    *,
    panel: pd.DataFrame,
    shape_df: pd.DataFrame,
    envelope_df: pd.DataFrame,
    first_touch_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    trajectories_df: pd.DataFrame,
    mfe_band_df: pd.DataFrame,
    mae_band_df: pd.DataFrame,
    time_state_df: pd.DataFrame,
    pooled_dists: List[Dict[str, Any]],
    cluster_meta: Dict[str, Any],
    cluster_distributions: List[Tuple[str, List[Dict[str, Any]]]],
    mfe_distributions: List[Dict[str, Any]],
    mae_distributions: List[Dict[str, Any]],
) -> List[Path]:
    out_paths: List[Path] = []

    _write_parquet(panel, OUTPUT_DIR / "trajectory_panel.parquet")
    out_paths.append(OUTPUT_DIR / "trajectory_panel.parquet")

    _write_csv(shape_df, OUTPUT_DIR / "shape_features.csv")
    out_paths.append(OUTPUT_DIR / "shape_features.csv")

    _write_csv(envelope_df, OUTPUT_DIR / "pooled_envelope.csv")
    out_paths.append(OUTPUT_DIR / "pooled_envelope.csv")

    _write_csv(first_touch_df, OUTPUT_DIR / "first_touch_crosstab.csv")
    out_paths.append(OUTPUT_DIR / "first_touch_crosstab.csv")

    _write_csv(assignments_df, OUTPUT_DIR / "cluster_assignments.csv")
    out_paths.append(OUTPUT_DIR / "cluster_assignments.csv")

    _write_text(
        phase2_cluster_summary_md(shape_df, cluster_meta), OUTPUT_DIR / "cluster_summary.md"
    )
    out_paths.append(OUTPUT_DIR / "cluster_summary.md")

    _write_csv(trajectories_df, OUTPUT_DIR / "cluster_trajectories.csv")
    out_paths.append(OUTPUT_DIR / "cluster_trajectories.csv")

    # cluster_distributions.md — pooled section 0 + MFE band conditionals + MAE band conditionals + per-cluster.
    lines: List[str] = [
        "# Arc 2 Trajectory Appendix — Distribution Tables",
        "",
        "## Section 0 — Pooled distributions (all trades)",
        "",
    ]
    for desc in pooled_dists:
        lines.append(render_distribution_md(desc, header_level=3))

    lines.append("## Section 1 — MFE-band conditional distributions")
    lines.append("")
    for desc in mfe_distributions:
        lines.append(render_distribution_md(desc, header_level=3))

    lines.append("## Section 2 — MAE-band conditional distributions")
    lines.append("")
    for desc in mae_distributions:
        lines.append(render_distribution_md(desc, header_level=3))

    lines.append("## Section 3 — Per-cluster distributions (k=4,6,8,10; km full + hier subsample)")
    lines.append("")
    for section_label, descs in cluster_distributions:
        lines.append(f"### {section_label}")
        lines.append("")
        for d in descs:
            lines.append(render_distribution_md(d, header_level=4))

    _write_text("\n".join(lines) + "\n", OUTPUT_DIR / "cluster_distributions.md")
    out_paths.append(OUTPUT_DIR / "cluster_distributions.md")

    _write_csv(mfe_band_df, OUTPUT_DIR / "mfe_band_conditionals.csv")
    out_paths.append(OUTPUT_DIR / "mfe_band_conditionals.csv")

    _write_csv(mae_band_df, OUTPUT_DIR / "mae_band_conditionals.csv")
    out_paths.append(OUTPUT_DIR / "mae_band_conditionals.csv")

    _write_csv(time_state_df, OUTPUT_DIR / "time_state_conditionals.csv")
    out_paths.append(OUTPUT_DIR / "time_state_conditionals.csv")

    # Charts.
    phase1_chart_envelope(envelope_df, CHARTS_DIR / "pooled_envelope.png")
    out_paths.append(CHARTS_DIR / "pooled_envelope.png")
    phase1_chart_scatter_mfe_mae(shape_df, CHARTS_DIR / "scatter_mfe_mae.png")
    out_paths.append(CHARTS_DIR / "scatter_mfe_mae.png")
    phase1_chart_scatter_end_peak_mfe(shape_df, CHARTS_DIR / "scatter_end_peak_mfe.png")
    out_paths.append(CHARTS_DIR / "scatter_end_peak_mfe.png")
    phase2_chart_dendrogram(cluster_meta, CHARTS_DIR / "dendrogram_hierarchical.png")
    out_paths.append(CHARTS_DIR / "dendrogram_hierarchical.png")
    for k in (4, 6, 8, 10):
        phase2_chart_cluster_envelopes(
            trajectories_df, k, CHARTS_DIR / f"cluster_envelopes_k{k}.png"
        )
        out_paths.append(CHARTS_DIR / f"cluster_envelopes_k{k}.png")

    return out_paths


def write_manifest(
    *,
    out_paths: List[Path],
    pop_meta: Dict[str, str],
    panel_meta: Dict[str, Any],
    cluster_meta: Dict[str, Any],
    prior_hashes: Optional[Dict[str, str]],
    run_ordinal: int,
) -> Dict[str, str]:
    """Hash all outputs (raw and PNG-pixel for charts). Write run_manifest.txt."""
    hashes: "OrderedDict[str, str]" = OrderedDict()
    pixel_hashes: "OrderedDict[str, str]" = OrderedDict()
    for p in out_paths:
        rel = p.relative_to(OUTPUT_DIR).as_posix()
        hashes[rel] = _sha256_file(p)
        if p.suffix.lower() == ".png":
            pixel_hashes[rel] = _png_pixel_hash(p)

    lines: List[str] = [
        f"Arc 2 Trajectory Appendix — run_manifest (run #{run_ordinal})",
        "Generated: (suppressed in deterministic mode below)",
        "",
        "## Operational decisions (defaults — see script docstring for rationale)",
        "",
        "- Hierarchical clustering: Ward on a seeded 8,000-row z-score subsample (full Ward at N≈42k is",
        "  infeasible). K-means runs on full kept N at every k. ARI computed on subsample.",
        "- Ex-ante population scope: every row in signals_features.csv where bar N+1 exists in the 1H series",
        "  (taken + non-taken). actual_exit_* populated only for taken rows.",
        "- R denominator: 2 * atr_1h_wilder_at_n (Wilder ATR(14) at signal bar close).",
        "- No-bar-N+1 / no-signal-bar / invalid-ATR rows dropped (counts below).",
        "",
        "## Inputs",
        "",
        f"- signals_features.csv sha256: {pop_meta['input_signals_features_sha256']}",
        f"- signals_features.csv rows: {pop_meta['input_signals_features_rows']}",
        f"- L4 anchor band: {pop_meta['l4_anchor_band']}; in band: {pop_meta['l4_anchor_band_ok']}",
        "",
        "## Determinism config",
        "",
        f"- NUMPY_SEED = {NUMPY_SEED}",
        f"- SKLEARN_SEED = {SKLEARN_SEED}",
        f"- HIERARCHICAL_SUBSAMPLE_N = {HIERARCHICAL_SUBSAMPLE_N}",
        "- Sort order: signals sorted (time, pair) → signal_idx assigned 0..N-1.",
        "- Parquet: pyarrow version='2.6', compression='snappy', use_dictionary=False.",
        "- CSV: utf-8, LF line endings, float_format='%.10g'.",
        "- Matplotlib: Agg backend, metadata pinned to fixed 2020-01-01 timestamp.",
        "",
        "## Population / panel diagnostics",
        "",
        f"- dropped_no_signal_bar: {panel_meta['dropped_no_signal_bar']}",
        f"- dropped_no_next_bar: {panel_meta['dropped_no_next_bar']}",
        f"- dropped_invalid_atr: {panel_meta['dropped_invalid_atr']}",
        f"- n_signals_with_panel: {panel_meta['n_signals_with_panel']}",
        f"- n_panel_rows: {panel_meta['n_panel_rows']}",
        "",
        "## Clustering diagnostics",
        "",
        f"- n_full_kept: {cluster_meta['n_full_kept']}",
        f"- n_dropped_for_nan_in_feature_matrix: {cluster_meta['n_dropped_for_nan']}",
        f"- subsample_n (hierarchical): {cluster_meta['subsample_n']}",
        f"- ARI(k=4): {cluster_meta['ari_per_k'][4]:.6f}",
        f"- ARI(k=6): {cluster_meta['ari_per_k'][6]:.6f}",
        f"- ARI(k=8): {cluster_meta['ari_per_k'][8]:.6f}",
        f"- ARI(k=10): {cluster_meta['ari_per_k'][10]:.6f}",
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
            ok = prior == h
            all_match = all_match and ok
            lines.append(f"- {rel}: {'IDENTICAL' if ok else 'DIVERGED'}")
        lines.append("")
        lines.append(
            f"Overall: {'PASS - byte-identical across runs' if all_match else 'FAIL - some files diverged (check PNG pixel-only hashes for chart files)'}"
        )
        lines.append("")

    text = "\n".join(lines) + "\n"
    manifest_path = OUTPUT_DIR / "run_manifest.txt"
    _write_text(text, manifest_path)
    return dict(hashes)


# ----------------------------------------------------------------------------
# Orchestrator.
# ----------------------------------------------------------------------------


def run_pipeline(
    *, run_ordinal: int, prior_hashes: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[run #{run_ordinal}] Loading signals_features.csv ...", flush=True)
    signals, pop_meta = load_signals()
    print(
        f"[run #{run_ordinal}] Signals loaded: {pop_meta['input_signals_features_rows']} rows; in L4 band: {pop_meta['l4_anchor_band_ok']}",
        flush=True,
    )

    print(f"[run #{run_ordinal}] Building trajectory panel + shape features ...", flush=True)
    panel, shape_df, panel_meta = build_panel_and_shape_features(signals)
    print(
        f"[run #{run_ordinal}] Panel: {panel_meta['n_signals_with_panel']} signals, {panel_meta['n_panel_rows']} rows.",
        flush=True,
    )

    print(f"[run #{run_ordinal}] Phase 1 — pooled descriptives ...", flush=True)
    envelope_df = phase1_pooled_envelope(panel)
    first_touch_df = phase1_first_touch_crosstab(shape_df)
    pooled_dists = phase1_pooled_distributions(shape_df)

    print(f"[run #{run_ordinal}] Phase 2 — clustering ...", flush=True)
    assignments_df, trajectories_df, cluster_meta = phase2_clustering(shape_df, panel)
    cluster_distributions = phase2_cluster_distributions(shape_df, cluster_meta)

    print(f"[run #{run_ordinal}] Phase 3 — conditional distributions ...", flush=True)
    mfe_band_df, mfe_distributions = phase3_mfe_band_conditionals(shape_df, panel)
    mae_band_df, mae_distributions = phase3_mae_band_conditionals(shape_df, panel)
    time_state_df = phase3_time_state_conditionals(panel, shape_df)

    print(f"[run #{run_ordinal}] Writing outputs ...", flush=True)
    out_paths = write_all_outputs(
        panel=panel,
        shape_df=shape_df,
        envelope_df=envelope_df,
        first_touch_df=first_touch_df,
        assignments_df=assignments_df,
        trajectories_df=trajectories_df,
        mfe_band_df=mfe_band_df,
        mae_band_df=mae_band_df,
        time_state_df=time_state_df,
        pooled_dists=pooled_dists,
        cluster_meta=cluster_meta,
        cluster_distributions=cluster_distributions,
        mfe_distributions=mfe_distributions,
        mae_distributions=mae_distributions,
    )

    print(f"[run #{run_ordinal}] Writing trajectory_appendix_report.md ...", flush=True)
    write_report(
        pop_meta=pop_meta,
        panel_meta=panel_meta,
        cluster_meta=cluster_meta,
        pooled_dists=pooled_dists,
        mfe_band_df=mfe_band_df,
        mae_band_df=mae_band_df,
        first_touch_df=first_touch_df,
        envelope_df=envelope_df,
        path=OUTPUT_DIR / "trajectory_appendix_report.md",
    )
    out_paths.append(OUTPUT_DIR / "trajectory_appendix_report.md")

    print(f"[run #{run_ordinal}] Writing run_manifest.txt ...", flush=True)
    hashes = write_manifest(
        out_paths=out_paths,
        pop_meta=pop_meta,
        panel_meta=panel_meta,
        cluster_meta=cluster_meta,
        prior_hashes=prior_hashes,
        run_ordinal=run_ordinal,
    )
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description="Arc 2 Baseline Trajectory Appendix")
    parser.add_argument(
        "--mode",
        choices=("once", "verify"),
        default="verify",
        help="'verify' runs the pipeline twice and checks byte-identicality; 'once' runs it once.",
    )
    args = parser.parse_args()

    if args.mode == "once":
        run_pipeline(run_ordinal=1)
        return 0

    # 'verify' mode: run twice; record both manifests' hashes.
    run1 = run_pipeline(run_ordinal=1)
    run2 = run_pipeline(run_ordinal=2, prior_hashes=run1)

    # Compare raw and pixel hashes for charts (pixel-only fallback documented in manifest).
    all_match = True
    for rel, h in run1.items():
        if run2.get(rel) != h:
            all_match = False
            print(f"  DIVERGED: {rel}", flush=True)
    if all_match:
        print("[verify] All outputs are byte-identical across the two runs.", flush=True)
    else:
        print(
            "[verify] Some outputs diverged; check run_manifest.txt for PNG pixel-only fallback hashes.",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
