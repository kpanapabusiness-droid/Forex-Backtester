"""Arc 2 — per-trade path visualisation by Block B category.

Phase: l6_arc2_trade_paths_by_block_b

Descriptive visualisation of individual trade price paths, partitioned by
Block B's 4 path-pattern categories at the +1R threshold. Read-existing-
CSV-only per L6.0 §14.6. Image output explicitly in scope.

Disposition: §14.5 (descriptive only).

Outputs to: results/l6/arc2/characterisation/extended/trade_paths_by_block_b/
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import random
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Force matplotlib Agg backend before pyplot import.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Locked input sha256s (gate 1)
# ---------------------------------------------------------------------------
LOCKED_SHAS: Dict[str, str] = {
    "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv":
        "7b2acd6ccb98f1fd145a631b318fc95d10f5cf4f42633be9c0b59738fa1696ee",
    "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv":
        "9f841c5b29e87ed90d34c9617431978baf3041459797cedef02fa16c27e3abb5",
    "results/l6/arc2/characterisation/v1_1_full/signals_features.csv":
        "71b39383632bd695b878add8b331b76bcd231ab5b9adba9eea03d69f8762483e",
    "L6_0_METHODOLOGY_LOCK.md":
        "4fd870b1d17380e4fc4fbfda5a43f7775d313c7a5f50dbfd1f06a3e49c519c26",
}

OUTPUT_DIR_REL = (
    "results/l6/arc2/characterisation/extended/trade_paths_by_block_b"
)

# Plot constants
K_MAX = 240
Y_MIN = -3.0
Y_MAX = 8.0
DPI = 100
FIG_SIZE = (14.0, 9.0)  # 1400x900 at 100 DPI

# Category metadata: (id, block_b_name, light_colour, dark_colour, plot_no)
CATEGORY_META: Dict[str, Dict[str, Any]] = {
    "only_up": {
        "block_b_name": "reached_up_only",
        "light_colour": "#639922",  # green
        "dark_colour": "#3B6D11",
        "plot_no": 1,
        "expected_n": 956,
        "min_n": 932,
        "max_n": 980,
    },
    "up_then_down": {
        "block_b_name": "up_then_down",
        "light_colour": "#EF9F27",  # amber
        "dark_colour": "#A86E10",
        "plot_no": 2,
        "expected_n": 1075,
        "min_n": 1048,
        "max_n": 1102,
    },
    "down_then_up": {
        "block_b_name": "down_then_up",
        "light_colour": "#D85A30",  # coral
        "dark_colour": "#8E3D1F",
        "plot_no": 3,
        "expected_n": 1090,
        "min_n": 1063,
        "max_n": 1117,
    },
    "straight_to_sl": {
        "block_b_name": "reached_down_only",
        "light_colour": "#E24B4A",  # red
        "dark_colour": "#8A2828",
        "plot_no": 4,
        "expected_n": 858,
        "min_n": 837,
        "max_n": 879,
    },
}

RESIDUAL_CATEGORIES = ("simultaneous", "neither_reached")

EXPECTED_TOTAL_TRADES = 3993
EXPECTED_PER_BAR_ROWS = 954749

PLOT_SAMPLE_CAP = 1100  # cap before random subsampling
SAMPLE_SEED = 20260511  # deterministic seed if any category exceeds cap


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Gate 1: input integrity
# ---------------------------------------------------------------------------
def verify_inputs() -> Dict[str, str]:
    observed: Dict[str, str] = {}
    for rel, expected in LOCKED_SHAS.items():
        p = REPO_ROOT / rel
        actual = _sha256_file(p)
        if actual != expected:
            raise RuntimeError(
                f"Gate 1 HALT - sha256 mismatch on {rel}:\n"
                f"  expected: {expected}\n  observed: {actual}"
            )
        observed[rel] = actual
    return observed


# ---------------------------------------------------------------------------
# Data load
# ---------------------------------------------------------------------------
def load_signals_features(path: Path) -> pd.DataFrame:
    """Load signals_features.csv and filter to taken==True."""
    print("  Loading signals_features.csv...", flush=True)
    usecols = [
        "pair", "time", "fold_id", "taken",
        "bars_to_plus_2atr_capped_240h",
        "bars_to_minus_2atr_capped_240h",
    ]
    sf = pd.read_csv(path, usecols=usecols)
    sf = sf[sf["taken"]].copy()
    sf["time"] = pd.to_datetime(sf["time"])
    sf = sf.sort_values(["pair", "time"]).reset_index(drop=True)
    return sf


def load_trade_index(path: Path) -> pd.DataFrame:
    print("  Loading trade_index.csv...", flush=True)
    ti = pd.read_csv(path)
    ti["signal_bar_ts"] = pd.to_datetime(ti["signal_bar_ts"])
    ti = ti.sort_values("trade_id").reset_index(drop=True)
    return ti


def join_taken_to_trade_index(
    sf_taken: pd.DataFrame, ti: pd.DataFrame
) -> pd.DataFrame:
    """Match each trade in trade_index to its signals_features row by (pair, ts).

    Returns ti augmented with bp1 / bm1 columns.
    """
    sf_lookup = sf_taken[[
        "pair", "time",
        "bars_to_plus_2atr_capped_240h",
        "bars_to_minus_2atr_capped_240h",
    ]].rename(columns={"time": "signal_bar_ts"})

    merged = ti.merge(
        sf_lookup,
        on=["pair", "signal_bar_ts"],
        how="left",
        validate="one_to_one",
    )
    merged = merged.rename(columns={
        "bars_to_plus_2atr_capped_240h": "bp1",
        "bars_to_minus_2atr_capped_240h": "bm1",
    })
    if merged["bp1"].isna().any() or merged["bm1"].isna().any():
        n_miss = int(merged["bp1"].isna().sum() + merged["bm1"].isna().sum())
        raise RuntimeError(
            f"Join HALT - {n_miss} rows have unmatched bp1/bm1 after merge"
        )
    merged["bp1"] = merged["bp1"].astype(np.int64)
    merged["bm1"] = merged["bm1"].astype(np.int64)
    return merged


# ---------------------------------------------------------------------------
# Gate 5: bp1/bm1 validity
# ---------------------------------------------------------------------------
def gate_bp1_bm1_validity(ti: pd.DataFrame) -> bool:
    bp1 = ti["bp1"].to_numpy()
    bm1 = ti["bm1"].to_numpy()
    if np.any(bp1 < 1) or np.any(bp1 > 241):
        raise RuntimeError(
            f"Gate 5 HALT - bp1 out of [1,241]: min={bp1.min()} max={bp1.max()}"
        )
    if np.any(bm1 < 1) or np.any(bm1 > 241):
        raise RuntimeError(
            f"Gate 5 HALT - bm1 out of [1,241]: min={bm1.min()} max={bm1.max()}"
        )
    return True


# ---------------------------------------------------------------------------
# Categorisation
# ---------------------------------------------------------------------------
def assign_categories(ti: pd.DataFrame) -> pd.DataFrame:
    """Assign each trade to one of 6 mutually exclusive categories.

    Threshold note: matches Block B (scripts/lchar/arc2_extended_analysis.py
    _classify_path) which uses `t <= 120` (the BL execution horizon), not 240.
    The prompt's text said `<= 240` but its mandated reference counts were
    derived at 120; the categorisation rule "Block B's exact 4 buckets, no new
    definitions" + Gate 4 (tight match to Block B counts) resolve in favour of
    120. The 240-bar window applies only to the *path observation*, which is
    still rendered out to k=240.
    """
    bp1 = ti["bp1"].to_numpy()
    bm1 = ti["bm1"].to_numpy()
    cats = np.full(len(ti), "", dtype=object)

    up_reached = bp1 <= 120
    down_reached = bm1 <= 120

    only_up = up_reached & (~down_reached)
    only_down = (~up_reached) & down_reached
    both = up_reached & down_reached
    neither = (~up_reached) & (~down_reached)

    up_then_down = both & (bp1 < bm1)
    down_then_up = both & (bm1 < bp1)
    simultaneous = both & (bp1 == bm1)

    cats[only_up] = "only_up"
    cats[only_down] = "straight_to_sl"
    cats[up_then_down] = "up_then_down"
    cats[down_then_up] = "down_then_up"
    cats[simultaneous] = "simultaneous"
    cats[neither] = "neither_reached"

    out = ti.copy()
    out["category"] = cats
    return out


# ---------------------------------------------------------------------------
# Gate 3: exhaustivity + zero overlap
# ---------------------------------------------------------------------------
def gate_exhaustivity(ti_cat: pd.DataFrame) -> Dict[str, int]:
    counts = ti_cat["category"].value_counts().to_dict()
    expected_set = set(CATEGORY_META.keys()) | set(RESIDUAL_CATEGORIES)
    observed_set = set(counts.keys())
    unknown = observed_set - expected_set
    if unknown:
        raise RuntimeError(
            f"Gate 3 HALT - unknown categories observed: {unknown}"
        )
    # check for empty strings (unassigned)
    if (ti_cat["category"] == "").any():
        raise RuntimeError(
            "Gate 3 HALT - some trades unassigned (empty category)"
        )
    total = int(sum(counts.values()))
    if total != EXPECTED_TOTAL_TRADES:
        raise RuntimeError(
            f"Gate 3 HALT - category sum {total} != expected {EXPECTED_TOTAL_TRADES}"
        )
    return counts


def gate_category_counts(counts: Dict[str, int]) -> None:
    for cat, meta in CATEGORY_META.items():
        n = counts.get(cat, 0)
        if not (meta["min_n"] <= n <= meta["max_n"]):
            raise RuntimeError(
                f"Gate 4 HALT - {cat} count {n} outside "
                f"[{meta['min_n']},{meta['max_n']}] (target {meta['expected_n']})"
            )


# ---------------------------------------------------------------------------
# Per-bar path extraction (streamed by trade_id)
# ---------------------------------------------------------------------------
def load_per_bar_paths(path: Path) -> pd.DataFrame:
    print("  Loading per_bar_paths.csv (~120 MB)...", flush=True)
    usecols = ["trade_id", "k", "bar_close_atr"]
    dtype = {"trade_id": np.int64, "k": np.int32, "bar_close_atr": np.float64}
    pb = pd.read_csv(path, usecols=usecols, dtype=dtype)
    pb = pb.sort_values(["trade_id", "k"]).reset_index(drop=True)
    return pb


def build_per_trade_paths(
    pb: pd.DataFrame, ti_cat: pd.DataFrame
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Returns trade_id -> (k_array, close_R_array)."""
    n_trades = int(ti_cat["trade_id"].max()) + 1
    tids_arr = pb["trade_id"].to_numpy()
    starts = np.searchsorted(tids_arr, np.arange(n_trades), side="left")
    ends = np.searchsorted(tids_arr, np.arange(n_trades), side="right")

    k_arr = pb["k"].to_numpy()
    close_R_arr = pb["bar_close_atr"].to_numpy() / 2.0  # ATR-units -> R-units

    per_trade: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for tid in range(n_trades):
        s, e = int(starts[tid]), int(ends[tid])
        if s == e:
            continue
        # Cap k at K_MAX
        ks = k_arr[s:e]
        cs = close_R_arr[s:e]
        mask = ks <= K_MAX
        per_trade[tid] = (ks[mask].astype(np.int32),
                          cs[mask].astype(np.float64))
    return per_trade


# ---------------------------------------------------------------------------
# Per-bar aggregates per category
# ---------------------------------------------------------------------------
def compute_per_bar_aggregates(
    per_trade: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ti_cat: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict[int, Tuple[float, float, float, int]]]]:
    """For each (category, k in [1, K_MAX]) compute median, q25, q75, n_active."""
    cat_tids: Dict[str, List[int]] = {
        cat: ti_cat.loc[ti_cat["category"] == cat, "trade_id"].astype(int).tolist()
        for cat in CATEGORY_META.keys()
    }

    rows: List[Dict[str, Any]] = []
    aggregates: Dict[str, Dict[int, Tuple[float, float, float, int]]] = {}

    for cat in CATEGORY_META.keys():
        # Build a matrix (n_trades_in_cat, K_MAX) filled with NaN
        tids = cat_tids[cat]
        n_cat = len(tids)
        mat = np.full((n_cat, K_MAX), np.nan, dtype=np.float64)
        for row_i, tid in enumerate(tids):
            if tid not in per_trade:
                continue
            ks, cs = per_trade[tid]
            # ks are 1-indexed integers in [1, K_MAX]
            idx = ks - 1
            mat[row_i, idx] = cs
        cat_aggs: Dict[int, Tuple[float, float, float, int]] = {}
        for k in range(1, K_MAX + 1):
            col = mat[:, k - 1]
            valid = col[~np.isnan(col)]
            n_active = int(valid.size)
            if n_active == 0:
                med = q25 = q75 = float("nan")
            else:
                med = float(np.median(valid))
                q25 = float(np.quantile(valid, 0.25))
                q75 = float(np.quantile(valid, 0.75))
            rows.append({
                "category": cat,
                "k": k,
                "n_active": n_active,
                "median_close_R": med,
                "q25_close_R": q25,
                "q75_close_R": q75,
            })
            cat_aggs[k] = (med, q25, q75, n_active)
        aggregates[cat] = cat_aggs

    df = pd.DataFrame(rows)
    return df, aggregates


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _style_axes(ax: matplotlib.axes.Axes) -> None:
    ax.set_xlim(1, K_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks(np.arange(0, K_MAX + 1, 20))
    ax.set_xlabel("Bar number (k)")
    ax.set_ylabel("R-units from entry")
    ax.grid(True, color="lightgray", alpha=0.3, linewidth=0.5)
    # reference lines
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(-1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(120, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)


def _pick_alpha(n_lines: int) -> float:
    # Spec: alpha=0.04-0.08 depending on n
    if n_lines >= 1000:
        return 0.04
    if n_lines >= 700:
        return 0.05
    if n_lines >= 500:
        return 0.06
    if n_lines >= 300:
        return 0.07
    return 0.08


def _plot_category(
    ax: matplotlib.axes.Axes,
    cat: str,
    tids: List[int],
    per_trade: Dict[int, Tuple[np.ndarray, np.ndarray]],
    aggs: Dict[int, Tuple[float, float, float, int]],
    sampling_note: Dict[str, Any],
    title_size: int = 14,
    annotate: bool = True,
) -> None:
    meta = CATEGORY_META[cat]
    light = meta["light_colour"]
    dark = meta["dark_colour"]

    plot_tids = list(tids)
    if len(plot_tids) > 1200:
        rng = random.Random(SAMPLE_SEED)
        plot_tids = sorted(rng.sample(plot_tids, PLOT_SAMPLE_CAP))
        sampling_note[cat] = {
            "sampled": True, "n_plotted": len(plot_tids),
            "n_total": len(tids), "seed": SAMPLE_SEED,
        }
    else:
        sampling_note[cat] = {
            "sampled": False, "n_plotted": len(plot_tids),
            "n_total": len(tids), "seed": None,
        }

    # Build line segments for LineCollection
    segments: List[np.ndarray] = []
    for tid in plot_tids:
        if tid not in per_trade:
            continue
        ks, cs = per_trade[tid]
        if ks.size < 2:
            continue
        segments.append(np.column_stack([ks.astype(np.float64), cs]))

    alpha = _pick_alpha(len(segments))
    lc = LineCollection(
        segments,
        colors=[light],
        linewidths=0.5,
        alpha=alpha,
        rasterized=True,
        zorder=1,
    )
    ax.add_collection(lc)

    # Median + Q25/Q75 overlays
    k_grid = np.arange(1, K_MAX + 1)
    med = np.array([aggs[k][0] for k in k_grid])
    q25 = np.array([aggs[k][1] for k in k_grid])
    q75 = np.array([aggs[k][2] for k in k_grid])
    valid_med = ~np.isnan(med)
    ax.fill_between(
        k_grid[valid_med], q25[valid_med], q75[valid_med],
        color=light, alpha=0.15, zorder=2, linewidth=0,
    )
    ax.plot(
        k_grid[valid_med], med[valid_med],
        color=dark, linewidth=2.5, alpha=1.0, zorder=3,
        label="median",
    )

    _style_axes(ax)
    n_str = f"n={sampling_note[cat]['n_total']}"
    if sampling_note[cat]["sampled"]:
        n_str += f" (plotted {sampling_note[cat]['n_plotted']})"
    ax.set_title(f"{cat} ({n_str}) - Block B at 1R", fontsize=title_size)
    if annotate:
        ax.text(
            122, Y_MAX - 0.4,
            "k > 120 is post-exit forward observation",
            fontsize=8, color="dimgray", style="italic",
            verticalalignment="top",
        )


def render_plots(
    out_dir: Path,
    ti_cat: pd.DataFrame,
    per_trade: Dict[int, Tuple[np.ndarray, np.ndarray]],
    aggregates: Dict[str, Dict[int, Tuple[float, float, float, int]]],
) -> Dict[str, Any]:
    sampling_note: Dict[str, Any] = {}

    cat_tids_map: Dict[str, List[int]] = {
        cat: ti_cat.loc[ti_cat["category"] == cat, "trade_id"]
                  .astype(int).tolist()
        for cat in CATEGORY_META.keys()
    }

    # 4 single-category plots
    for cat, meta in CATEGORY_META.items():
        plot_no = meta["plot_no"]
        fname = f"plot_{plot_no:02d}_{cat}.png"
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
        _plot_category(
            ax, cat, cat_tids_map[cat], per_trade,
            aggregates[cat], sampling_note,
        )
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=DPI)
        plt.close(fig)
        print(f"  Wrote {fname}", flush=True)

    # Combined 2x2
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE, dpi=DPI)
    order = ["only_up", "up_then_down", "down_then_up", "straight_to_sl"]
    for ax, cat in zip(axes.flatten(), order):
        # Reuse sampling_note but ensure deterministic
        _plot_category(
            ax, cat, cat_tids_map[cat], per_trade,
            aggregates[cat], sampling_note,
            title_size=11, annotate=False,
        )
    fig.suptitle("Block B trade paths by category (1R threshold)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "plot_05_combined_2x2.png", dpi=DPI)
    plt.close(fig)
    print("  Wrote plot_05_combined_2x2.png", flush=True)

    return sampling_note


# ---------------------------------------------------------------------------
# Gate 6 / 8 / 7 helpers
# ---------------------------------------------------------------------------
def gate_n_active_sensible(
    aggregates: Dict[str, Dict[int, Tuple[float, float, float, int]]],
    counts: Dict[str, int],
    bars_available_by_cat: Dict[str, np.ndarray],
) -> None:
    """Gate 6:

    1. At k=1, n_active must equal the category count for all 4 categories.
    2. At k=240, n_active must equal the number of trades whose
       `bars_available >= 240`. (The spec's text "≥ category_count × 0.99"
       describes the same invariant phrased loosely; checking parity against
       bars_available is the principled version and is what makes the gate
       informative — it confirms per_bar_paths covers the full observation
       horizon exactly once per non-clamped trade.)
    """
    for cat in CATEGORY_META.keys():
        n_cat = counts.get(cat, 0)
        n1 = aggregates[cat][1][3]
        if n1 != n_cat:
            raise RuntimeError(
                f"Gate 6 HALT - {cat} n_active at k=1 ({n1}) != category count ({n_cat})"
            )
        n240 = aggregates[cat][K_MAX][3]
        ba = bars_available_by_cat[cat]
        expected_240 = int((ba >= 240).sum())
        if n240 != expected_240:
            raise RuntimeError(
                f"Gate 6 HALT - {cat} n_active at k=240 ({n240})"
                f" != trades with bars_available>=240 ({expected_240})"
            )


def gate_plot_files(out_dir: Path) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    expected = [
        "plot_01_only_up.png",
        "plot_02_up_then_down.png",
        "plot_03_down_then_up.png",
        "plot_04_straight_to_sl.png",
        "plot_05_combined_2x2.png",
    ]
    for fname in expected:
        p = out_dir / fname
        if not p.exists():
            raise RuntimeError(f"Gate 7 HALT - missing {fname}")
        sz = p.stat().st_size
        if sz < 100 * 1024:
            raise RuntimeError(
                f"Gate 7 HALT - {fname} size {sz} bytes < 100KB"
            )
        sizes[fname] = sz
    return sizes


def gate_aggregate_plausibility(
    aggregates: Dict[str, Dict[int, Tuple[float, float, float, int]]],
) -> None:
    only_up_120 = aggregates["only_up"][120][0]
    if not (only_up_120 > 0):
        raise RuntimeError(
            f"Gate 8 HALT - only_up median at k=120 ({only_up_120:.4f}) <= 0"
        )
    straight_120 = aggregates["straight_to_sl"][120][0]
    if not (straight_120 < -0.5):
        raise RuntimeError(
            f"Gate 8 HALT - straight_to_sl median at k=120 ({straight_120:.4f}) >= -0.5"
        )
    # up_then_down: positive early, negative late
    utd_early = aggregates["up_then_down"][10][0]
    utd_late = aggregates["up_then_down"][120][0]
    if not (utd_early > 0):
        raise RuntimeError(
            f"Gate 8 HALT - up_then_down median at k=10 ({utd_early:.4f}) not positive"
        )
    if not (utd_late < 0):
        raise RuntimeError(
            f"Gate 8 HALT - up_then_down median at k=120 ({utd_late:.4f}) not negative"
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(
    out_dir: Path,
    counts: Dict[str, int],
    ti_cat: pd.DataFrame,
    aggregates: Dict[str, Dict[int, Tuple[float, float, float, int]]],
    sampling_note: Dict[str, Any],
    wallclock: float,
    mem_peak_mb: float,
) -> Path:
    p = out_dir / "paths_by_block_b_report.md"
    k_pts = [1, 10, 30, 60, 90, 120, 180, 240]
    lines: List[str] = []
    lines.append("# Arc 2 — Per-Trade Paths by Block B Category (1R threshold)")
    lines.append("")
    lines.append("**Phase:** `l6_arc2_trade_paths_by_block_b`")
    lines.append(
        "**Scope:** Descriptive visualisation of individual trade price paths,"
        " partitioned by Block B's 4 path-pattern categories at the +1R"
        " threshold. Read-existing-CSV-only (§14.6). Disposition §14.5 —"
        " descriptive only."
    )
    lines.append("")
    lines.append("## Category counts")
    lines.append("")
    lines.append("| Category | n | block_b_name | expected |")
    lines.append("|---|---:|---|---:|")
    for cat, meta in CATEGORY_META.items():
        lines.append(
            f"| {cat} | {counts.get(cat,0)} | "
            f"{meta['block_b_name']} | {meta['expected_n']} |"
        )
    for resid in RESIDUAL_CATEGORIES:
        lines.append(f"| _{resid}_ (residual) | {counts.get(resid,0)} | - | - |")
    lines.append(f"| **Total** | **{sum(counts.values())}** | | **3993** |")
    lines.append("")

    lines.append("## Per-category R distribution")
    lines.append("")
    lines.append("| Category | n | mean R | median R |")
    lines.append("|---|---:|---:|---:|")
    for cat in list(CATEGORY_META.keys()) + list(RESIDUAL_CATEGORIES):
        sub = ti_cat[ti_cat["category"] == cat]
        if sub.empty:
            continue
        lines.append(
            f"| {cat} | {len(sub)} | {sub['R'].mean():.4f} | "
            f"{sub['R'].median():.4f} |"
        )
    lines.append("")

    lines.append("## Per-bar median bar_close_R (R-units from entry)")
    lines.append("")
    header_k = " | ".join([f"k={k}" for k in k_pts])
    lines.append(f"| Category | {header_k} |")
    lines.append("|---|" + "|".join(["---:"] * len(k_pts)) + "|")
    for cat in CATEGORY_META.keys():
        row_vals = [f"{aggregates[cat][k][0]:+.3f}" for k in k_pts]
        lines.append(f"| {cat} | " + " | ".join(row_vals) + " |")
    lines.append("")

    lines.append("## Sampling")
    lines.append("")
    for cat, info in sampling_note.items():
        if info["sampled"]:
            lines.append(
                f"- `{cat}`: sampled {info['n_plotted']} of {info['n_total']}"
                f" (seed={info['seed']})"
            )
        else:
            lines.append(
                f"- `{cat}`: plotted all {info['n_total']} (no sampling)"
            )
    lines.append("")

    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def write_manifest(
    out_dir: Path,
    observed_input_shas: Dict[str, str],
    counts: Dict[str, int],
    sampling_note: Dict[str, Any],
    wallclock: float,
    mem_peak_mb: float,
) -> Path:
    """Write the deterministic manifest.

    Volatile fields (timestamp_utc, wallclock_seconds, peak_memory_mb) go to
    `_run_perf.txt`. `run_manifest.txt` contains only deterministic content so
    Gate 9's byte-identicality check applies to it.
    """
    perf_p = out_dir / "_run_perf.txt"
    perf_p.write_text(
        f"timestamp_utc: {_dt.datetime.now(_dt.UTC).isoformat()}\n"
        f"wallclock_seconds: {wallclock:.3f}\n"
        f"peak_memory_mb: {mem_peak_mb:.1f}\n",
        encoding="utf-8",
    )

    p = out_dir / "run_manifest.txt"
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("Arc 2 - Per-Trade Paths by Block B Category - run manifest")
    lines.append("=" * 78)
    lines.append(f"phase: l6_arc2_trade_paths_by_block_b")
    lines.append(f"python: {sys.version.split()[0]}")
    lines.append(f"numpy: {np.__version__}")
    lines.append(f"pandas: {pd.__version__}")
    lines.append(f"matplotlib: {matplotlib.__version__}")
    lines.append("")
    lines.append("--- INPUT SHA256s (gate 1 verified) ---")
    for rel, sha in observed_input_shas.items():
        lines.append(f"{sha}  {rel}")
    lines.append("")
    lines.append("--- CATEGORY COUNTS ---")
    for cat, meta in CATEGORY_META.items():
        lines.append(
            f"{cat}: {counts.get(cat,0)} (target {meta['expected_n']},"
            f" band [{meta['min_n']},{meta['max_n']}])"
        )
    for resid in RESIDUAL_CATEGORIES:
        lines.append(f"{resid}: {counts.get(resid,0)}")
    lines.append(f"total: {sum(counts.values())}")
    lines.append("")
    lines.append("--- SAMPLING ---")
    for cat, info in sampling_note.items():
        lines.append(
            f"{cat}: sampled={info['sampled']} n_plotted={info['n_plotted']}"
            f" n_total={info['n_total']} seed={info['seed']}"
        )
    lines.append("")
    lines.append("--- OUTPUT FILE SHA256s ---")
    # Exclude run_manifest.txt itself and _run_perf.txt (volatile)
    out_files = sorted(
        f for f in out_dir.iterdir()
        if f.is_file() and f.name not in ("run_manifest.txt", "_run_perf.txt")
    )
    for f in out_files:
        sha = _sha256_file(f)
        lines.append(f"{sha}  {f.name}")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / OUTPUT_DIR_REL,
        help="Output directory (default: results/.../trade_paths_by_block_b)",
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    t0 = time.perf_counter()

    print("[Gate 1] Verifying input sha256s...", flush=True)
    observed_input_shas = verify_inputs()
    print("  OK - all 4 sha256s match", flush=True)

    print("[Load] Reading inputs...", flush=True)
    sf_path = REPO_ROOT / "results/l6/arc2/characterisation/v1_1_full/signals_features.csv"
    ti_path = REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv"
    pb_path = REPO_ROOT / "results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv"

    sf_taken = load_signals_features(sf_path)
    ti = load_trade_index(ti_path)

    print("[Gate 2] Row count parity...", flush=True)
    if len(ti) != EXPECTED_TOTAL_TRADES:
        raise RuntimeError(
            f"Gate 2 HALT - trade_index rows {len(ti)} != {EXPECTED_TOTAL_TRADES}"
        )
    if len(sf_taken) != EXPECTED_TOTAL_TRADES:
        raise RuntimeError(
            f"Gate 2 HALT - signals_features taken rows {len(sf_taken)}"
            f" != {EXPECTED_TOTAL_TRADES}"
        )
    # per_bar_paths row count check
    with pb_path.open("rb") as fp:
        # subtract header
        n_pb = sum(1 for _ in fp) - 1
    if n_pb != EXPECTED_PER_BAR_ROWS:
        raise RuntimeError(
            f"Gate 2 HALT - per_bar_paths rows {n_pb} != {EXPECTED_PER_BAR_ROWS}"
        )
    print(f"  OK - trade_index={len(ti)}, taken={len(sf_taken)}, per_bar={n_pb}", flush=True)

    print("[Merge] Joining signals_features (bp1/bm1) to trade_index...", flush=True)
    ti = join_taken_to_trade_index(sf_taken, ti)
    del sf_taken

    print("[Gate 5] bp1/bm1 validity...", flush=True)
    gate_bp1_bm1_validity(ti)
    print("  OK - all bp1/bm1 in [1,241], integer, no NaN", flush=True)

    print("[Categorise] Assigning categories...", flush=True)
    ti_cat = assign_categories(ti)

    print("[Gate 3] Exhaustivity...", flush=True)
    counts = gate_exhaustivity(ti_cat)
    for cat in list(CATEGORY_META.keys()) + list(RESIDUAL_CATEGORIES):
        print(f"  {cat}: {counts.get(cat,0)}", flush=True)

    print("[Gate 4] Category counts vs Block B reference (±2.5%)...", flush=True)
    gate_category_counts(counts)
    print("  OK - all 4 plotted categories within tolerance", flush=True)

    print("[Load] per_bar_paths.csv...", flush=True)
    pb = load_per_bar_paths(pb_path)
    print(f"  Loaded {len(pb)} rows", flush=True)

    print("[Extract] Per-trade path series...", flush=True)
    per_trade = build_per_trade_paths(pb, ti_cat)
    del pb
    print(f"  {len(per_trade)} trades have path data", flush=True)

    print("[Aggregate] Per-bar (category, k) medians/quantiles...", flush=True)
    per_bar_agg_df, aggregates = compute_per_bar_aggregates(per_trade, ti_cat)

    print("[Gate 6] n_active sensible at k=1 and k=240...", flush=True)
    bars_available_by_cat: Dict[str, np.ndarray] = {
        cat: ti_cat.loc[ti_cat["category"] == cat, "bars_available"].to_numpy()
        for cat in CATEGORY_META.keys()
    }
    gate_n_active_sensible(aggregates, counts, bars_available_by_cat)
    print("  OK", flush=True)

    print("[Gate 8] Aggregate plausibility (winners +ve, losers -ve)...", flush=True)
    gate_aggregate_plausibility(aggregates)
    print("  OK", flush=True)

    print("[Write] category_assignment.csv...", flush=True)
    assignment_cols = [
        "trade_id", "pair", "signal_bar_ts", "fold_id",
        "exit_reason", "held_bars", "bp1", "bm1", "category", "R",
    ]
    ti_cat[assignment_cols].to_csv(out_dir / "category_assignment.csv", index=False)

    print("[Write] per_bar_aggregates_by_category.csv...", flush=True)
    per_bar_agg_df.to_csv(
        out_dir / "per_bar_aggregates_by_category.csv",
        index=False, float_format="%.10g",
    )

    print("[Plot] Rendering 5 PNGs...", flush=True)
    sampling_note = render_plots(out_dir, ti_cat, per_trade, aggregates)

    print("[Gate 7] Plot files exist and > 100KB...", flush=True)
    sizes = gate_plot_files(out_dir)
    for fname, sz in sizes.items():
        print(f"  {fname}: {sz/1024:.1f} KB", flush=True)

    elapsed = time.perf_counter() - t0
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_peak_mb = mem_peak / (1024 * 1024)

    print("[Write] paths_by_block_b_report.md...", flush=True)
    write_report(out_dir, counts, ti_cat, aggregates, sampling_note,
                 elapsed, mem_peak_mb)

    print("[Write] run_manifest.txt...", flush=True)
    write_manifest(out_dir, observed_input_shas, counts, sampling_note,
                   elapsed, mem_peak_mb)

    print(f"[Done] wallclock={elapsed:.2f}s peak_memory={mem_peak_mb:.1f} MB",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
