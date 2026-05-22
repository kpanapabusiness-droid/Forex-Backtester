"""Arc 4 — Step 3 capturability characterisation + SL sweep.

L_ARC_PROTOCOL v2.1.1 §7. Per cluster (from Step 2 chosen K=4), sweep candidate
SLs in {0.5, 1.0, 1.5, 2.0, 3.0, 4.0} × ATR_signal_TF, re-impose hypothetical
SL on the full bar_path (held bars + forward observation bars), compute pre-peak
path-quality metrics + full-window forward-geometry + distribution-shape
classification, and select the SL whose configuration maximises the v2.1.1
capturability composite among SLs passing the §2 floors.

Inputs:
  - results/l_arc_4/step1/trades_all.csv       (entry_price, atr_14_at_signal, ...)
  - results/l_arc_4/step1/trades_paths.csv     (open/high/low/close per bar offset 0..240)
  - results/l_arc_4/step2/clusters_K4.csv      (trade_id → cluster_id)
  - results/l_arc_4/step2/centroids_K4.csv     (per-cluster path-shape centroids)

Outputs (results/l_arc_4/step3/):
  - sl_sweep_cluster_{0..3}.csv          per-SL gate metrics + composite
  - distribution_cluster_{0..3}.csv       fwd_mfe distribution + shape stats
  - pct_peak_and_collapse_cluster_{0..3}.csv  per-trade drawdown_from_peak
  - archetype_summaries.csv               per cluster selected-SL identity + label
  - capturability_pass_list.csv           surviving clusters only
  - step3_diagnostics.md                  full report

Implementation notes:
- SL re-imposition uses bar LOW (not close) — matches Step 1's intrabar SL fire.
- MFE per bar uses bar HIGH (matches §7's max(high_so_far) − entry_price formulation).
  Note: this differs from Step 1's mfe_so_far_r which uses close. The two diverge
  intra-bar but agree at bar boundaries on close-only days. §7's SL sweep
  explicitly specifies the high-based MFE.
- Truncation: at first bar B where low ≤ entry − X × ATR, trade closes at SL price,
  final_r_at_X = −1.0. mfe_so_far_r includes bar B (assumption: MFE before MAE
  intra-bar). If never hit within bars 0..240, truncate at 240,
  final_r_at_X = (close_at_240 − entry_price) / (X × ATR). The prompt's literal
  formula `close_at_240 / (X × ATR)` is interpreted as a typo (entry_price
  subtraction is required for the R-multiple to be well-defined).
- Forward observation bars (is_held=0) are used identically to held bars for SL
  re-imposition — both contain real-market OHLC.

Determinism: two-run byte-identical sha256 across all output files (CSVs only;
the markdown diagnostic embeds the sha256 set inline). Log both runs.

Usage:
  py scripts/l_arc_4/step3_capturability.py -c configs/l_arc_4.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANDIDATE_SL_MULTS: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
PATH_BARS: int = 241  # bar_offset 0..240 inclusive (must match Step 1 forward_window_bars + 1)

# §2 hard floors (locked, within-arc unchangeable per §1.8)
FLOOR_MONO_PRE_PEAK: float = 0.55
FLOOR_FWD_MFE_P50: float = 1.5
FLOOR_FRAC_REACH_1R: float = 0.70
FLOOR_FRAC_WRONG_WAY: float = 0.30
FLOOR_SIZE_FRACTION: float = 0.10
ADMITTED_SHAPE_TAGS = ("tight_unimodal", "heavy_right_tail", "bimodal_separated")

# §7 capturability composite
COMPOSITE_TIE_TOLERANCE: float = 0.02

# Shape classification thresholds (§7 + arc-open prompt)
DIP_P_BIMODAL_MAX: float = 0.05
DIP_P_UNIMODAL_MIN: float = 0.10
DIP_P_HEAVYTAIL_MIN: float = 0.05
TIGHT_IQR_MAX_R: float = 2.0
HEAVYTAIL_P95_TO_P75_RATIO: float = 3.0
HEAVYTAIL_P50_MAX_R: float = 1.5
BIMODAL_MIN_MODE_MASS: float = 0.20
BIMODAL_MODE_SEPARATION_R: float = 1.0

# pct_peak_and_collapse threshold (post-peak retracement, info-only at §7)
PCT_PC_DRAWDOWN_FRACTION: float = 0.50
PCT_PC_MIN_PEAK_R: float = 1.0

# Cap-binding warn threshold (matches §5 auto-extend rule)
CAP_BIND_WARN_PCT: float = 0.20

# Near-miss tolerance: 10% of the rule threshold value (per arc-open prompt
# "single feature off by ≤ 10% of feature range" — interpreted as 10% of the
# threshold, the most defensible operational reading given non-stationary
# data ranges).
NEAR_MISS_RELATIVE: float = 0.10


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_trades_all(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "atr_14_at_signal" not in df.columns:
        raise ValueError(
            f"{path.name} missing required column 'atr_14_at_signal' "
            f"(Step 1 schema mismatch)"
        )
    if "entry_price" not in df.columns:
        raise ValueError(f"{path.name} missing required column 'entry_price'")
    if "trade_id" not in df.columns:
        raise ValueError(f"{path.name} missing required column 'trade_id'")
    df = df.sort_values("trade_id").reset_index(drop=True)
    return df


def _load_paths_2d(
    path: Path, expected_n_trades: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (trade_ids_sorted, opens, highs, lows, closes) where each OHLC
    array has shape (n_trades, PATH_BARS).
    """
    df = pd.read_csv(path)
    required = {"trade_id", "bar_offset", "open", "high", "low", "close", "is_held"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    df = df.sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)
    n_rows = len(df)
    if n_rows != expected_n_trades * PATH_BARS:
        raise ValueError(
            f"{path.name} has {n_rows} rows; expected {expected_n_trades} × {PATH_BARS} = "
            f"{expected_n_trades * PATH_BARS}"
        )
    trade_ids_per_row = df["trade_id"].to_numpy()
    bar_offsets_per_row = df["bar_offset"].to_numpy()
    # Sanity: every trade has bar_offsets 0..240 contiguous.
    first_offsets = bar_offsets_per_row.reshape(expected_n_trades, PATH_BARS)
    if not np.all(first_offsets == np.arange(PATH_BARS)[None, :]):
        raise ValueError(
            f"{path.name} bar_offsets are not uniformly 0..{PATH_BARS - 1} per trade"
        )
    trade_ids_sorted = trade_ids_per_row.reshape(expected_n_trades, PATH_BARS)[:, 0].astype(int)
    opens = df["open"].to_numpy(dtype=float).reshape(expected_n_trades, PATH_BARS)
    highs = df["high"].to_numpy(dtype=float).reshape(expected_n_trades, PATH_BARS)
    lows = df["low"].to_numpy(dtype=float).reshape(expected_n_trades, PATH_BARS)
    closes = df["close"].to_numpy(dtype=float).reshape(expected_n_trades, PATH_BARS)
    return trade_ids_sorted, opens, highs, lows, closes


def _load_clusters(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "cluster_id" not in df.columns:
        raise ValueError(f"{path.name} missing required column 'cluster_id'")
    df = df.sort_values("trade_id").reset_index(drop=True)
    return df


def _load_centroids(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values("cluster_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-trade SL re-imposition + metrics
# ---------------------------------------------------------------------------


@dataclass
class _SLPassMetrics:
    """Per-cluster per-SL metric block."""

    sl_atr_mult: float
    r_per_trade_mean_price: float          # mean of X × ATR (price units, descriptive)
    n_trades: int
    cap_bind_count: int                    # trades that ran to bar 240 without SL hit
    cap_bind_pct: float

    mono_pre_peak_centroid: float          # cluster mean of per-trade monotonicity_pre_peak
    frac_wrong_way_pre_peak: float
    fwd_mfe_p50: float
    fwd_mfe_p75: float
    fwd_mfe_p95: float
    fwd_mfe_mean: float
    frac_reach_1R: float
    frac_reach_2R: float
    final_r_mean: float
    final_r_p5: float
    final_r_p25: float
    final_r_p50: float
    final_r_p75: float
    final_r_p95: float
    final_r_t_stat: float

    pct_peak_and_collapse: float

    # Distribution shape
    dip_p_value: float
    dip_statistic: float
    iqr_r: float
    mode_count: int
    mode_loc_1: float
    mode_loc_2: float
    mode_mass_1: float
    mode_mass_2: float
    min_mode_mass: float
    mode_separation_r: float
    shape_tag: str

    # §2 gate
    gate_A_mono: bool
    gate_B_mfe: bool
    gate_C_reach: bool
    gate_D_wrong_way: bool
    gate_E_size: bool
    gate_F_shape: bool
    size_fraction: float
    all_pass: bool

    # Composite (only meaningful when all_pass=True)
    capturability_composite: float
    peak_mfe_atr_units: float  # for tiebreaker 1 — physical-distance MFE p50

    # Raw fwd_mfe array — kept around for downstream writes (set externally).
    fwd_mfe_arr: np.ndarray = field(default_factory=lambda: np.array([]))
    drawdown_from_peak_arr: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_mfe_arr: np.ndarray = field(default_factory=lambda: np.array([]))


def _per_trade_monotonicity_pre_peak(
    close_r_subset: np.ndarray, peak_mfe_bar_subset: np.ndarray
) -> np.ndarray:
    """For each trade i, compute monotonicity_ratio_in_profit on bars
    0..peak_mfe_bar[i] inclusive of close_r_subset[i].

    Vectorised-friendly per-trade loop (cluster sizes ≤ ~4k → fast).
    """
    n = close_r_subset.shape[0]
    out = np.zeros(n, dtype=float)
    for i in range(n):
        end_inclusive = int(peak_mfe_bar_subset[i]) + 1
        if end_inclusive <= 0:
            out[i] = 0.0
            continue
        cl = close_r_subset[i, :end_inclusive]
        in_profit = cl > 0.0
        n_in_profit = int(in_profit.sum())
        if n_in_profit <= 1:
            out[i] = 0.0
            continue
        in_profit_closes = cl[in_profit]
        gte = in_profit_closes[1:] >= in_profit_closes[:-1]
        out[i] = float(gte.sum() / gte.size)
    return out


def _classify_shape(
    fwd_mfe: np.ndarray,
) -> Tuple[str, float, float, float, int, float, float, float, float, float, float]:
    """Classify the fwd_mfe distribution per §7 shape rules.

    Returns (shape_tag, dip_p, dip_stat, iqr_r, mode_count, mode_loc_1,
    mode_loc_2, mode_mass_1, mode_mass_2, min_mode_mass, mode_separation_r).
    """
    import diptest
    from scipy.signal import find_peaks
    from scipy.stats import gaussian_kde

    arr = np.asarray(fwd_mfe, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        # Hartigan dip needs > 3 distinct values to be meaningful — defer.
        return ("unclassified", float("nan"), float("nan"), float("nan"), 0,
                float("nan"), float("nan"), float("nan"), float("nan"),
                float("nan"), float("nan"))

    p25 = float(np.percentile(arr, 25))
    p50 = float(np.percentile(arr, 50))
    p75 = float(np.percentile(arr, 75))
    p95 = float(np.percentile(arr, 95))
    iqr = p75 - p25

    # diptest can raise on degenerate inputs (all identical). Guard.
    if np.unique(arr).size < 2:
        return ("unclassified", float("nan"), float("nan"), iqr, 1,
                float(arr[0]), float("nan"), 1.0, 0.0, 0.0, 0.0)
    try:
        dip_stat, dip_p = diptest.diptest(arr)
        dip_stat = float(dip_stat)
        dip_p = float(dip_p)
    except Exception:
        dip_stat, dip_p = float("nan"), float("nan")

    # Mode detection via KDE (deterministic given Silverman bandwidth).
    mode_count = 0
    mode_loc_1 = float("nan")
    mode_loc_2 = float("nan")
    mode_mass_1 = float("nan")
    mode_mass_2 = float("nan")
    min_mode_mass = float("nan")
    mode_separation_r = float("nan")
    try:
        if np.std(arr) > 0:
            kde = gaussian_kde(arr, bw_method="silverman")
            x_grid = np.linspace(arr.min(), arr.max(), 500)
            density = kde(x_grid)
            peaks_idx, _ = find_peaks(density)
            mode_count = int(peaks_idx.size)
            if peaks_idx.size >= 1:
                # Sort detected peaks by density descending.
                heights = density[peaks_idx]
                sort = np.argsort(-heights)
                top = peaks_idx[sort][:2]
                if top.size == 1:
                    mode_loc_1 = float(x_grid[top[0]])
                    mode_mass_1 = 1.0
                    mode_mass_2 = 0.0
                    min_mode_mass = 0.0
                    mode_separation_r = 0.0
                else:
                    loc_a = float(x_grid[top[0]])
                    loc_b = float(x_grid[top[1]])
                    if loc_a <= loc_b:
                        mode_loc_1, mode_loc_2 = loc_a, loc_b
                    else:
                        mode_loc_1, mode_loc_2 = loc_b, loc_a
                    midpoint = (mode_loc_1 + mode_loc_2) / 2.0
                    mass_left = float(np.mean(arr < midpoint))
                    mass_right = float(np.mean(arr >= midpoint))
                    mode_mass_1 = mass_left
                    mode_mass_2 = mass_right
                    min_mode_mass = min(mass_left, mass_right)
                    mode_separation_r = mode_loc_2 - mode_loc_1
    except Exception:
        pass  # mode stats remain NaN

    # Classification with priority: bimodal_separated > tight_unimodal >
    # heavy_right_tail > scattered > unclassified.
    is_bimodal_sep = (
        not math.isnan(dip_p)
        and dip_p < DIP_P_BIMODAL_MAX
        and not math.isnan(min_mode_mass)
        and min_mode_mass >= BIMODAL_MIN_MODE_MASS
        and not math.isnan(mode_separation_r)
        and mode_separation_r >= BIMODAL_MODE_SEPARATION_R
    )
    is_tight_unimodal = (
        not math.isnan(dip_p)
        and dip_p >= DIP_P_UNIMODAL_MIN
        and iqr <= TIGHT_IQR_MAX_R
    )
    is_heavy_right_tail = (
        not math.isnan(dip_p)
        and dip_p >= DIP_P_HEAVYTAIL_MIN
        and (p75 > 0 and p95 >= HEAVYTAIL_P95_TO_P75_RATIO * p75)
        and p50 <= HEAVYTAIL_P50_MAX_R
    )
    is_scattered = (
        not math.isnan(dip_p)
        and dip_p < DIP_P_BIMODAL_MAX
        and (
            math.isnan(mode_separation_r)
            or mode_separation_r < BIMODAL_MODE_SEPARATION_R
            or (not math.isnan(min_mode_mass) and min_mode_mass < BIMODAL_MIN_MODE_MASS)
        )
    )

    if is_bimodal_sep:
        tag = "bimodal_separated"
    elif is_tight_unimodal:
        tag = "tight_unimodal"
    elif is_heavy_right_tail:
        tag = "heavy_right_tail"
    elif is_scattered:
        tag = "scattered"
    else:
        tag = "unclassified"

    return (
        tag,
        dip_p,
        dip_stat,
        iqr,
        mode_count,
        mode_loc_1,
        mode_loc_2,
        mode_mass_1,
        mode_mass_2,
        min_mode_mass if not math.isnan(min_mode_mass) else 0.0,
        mode_separation_r if not math.isnan(mode_separation_r) else 0.0,
    )


def _evaluate_sl_for_cluster(
    sl_mult: float,
    cluster_entry_price: np.ndarray,    # (n_c,)
    cluster_atr: np.ndarray,            # (n_c,)
    cluster_opens: np.ndarray,          # (n_c, 241)
    cluster_highs: np.ndarray,
    cluster_lows: np.ndarray,
    cluster_closes: np.ndarray,
    pool_size: int,
) -> _SLPassMetrics:
    """Evaluate one (cluster, SL) configuration. Vectorised across trades."""
    n_c = int(cluster_entry_price.size)
    sl_distance = sl_mult * cluster_atr               # (n_c,)
    sl_price = cluster_entry_price - sl_distance      # (n_c,) long
    r_unit = sl_distance                              # (n_c,) — 1R = X × ATR per trade

    # Per-bar hit mask (low ≤ sl_price).
    hit_mask = cluster_lows <= sl_price[:, None]       # (n_c, 241)
    hit_any = hit_mask.any(axis=1)
    # argmax returns first True position; for trades with no hit it returns 0 —
    # use hit_any to set those to PATH_BARS - 1 (truncate at bar 240).
    first_hit_bar = np.argmax(hit_mask, axis=1)
    truncate_at = np.where(hit_any, first_hit_bar, PATH_BARS - 1).astype(int)

    # R-frame conversion of close/high/low to new R-units.
    inv_r = 1.0 / r_unit
    close_r = (cluster_closes - cluster_entry_price[:, None]) * inv_r[:, None]
    high_r = (cluster_highs - cluster_entry_price[:, None]) * inv_r[:, None]
    low_r = (cluster_lows - cluster_entry_price[:, None]) * inv_r[:, None]

    # Build mfe_so_far_r as running max of high_r, but ignore bars after truncate.
    cummax_high_r = np.maximum.accumulate(high_r, axis=1)
    bar_idx = np.arange(PATH_BARS)
    trunc_mask = bar_idx[None, :] <= truncate_at[:, None]
    # Mask values after truncation with -inf so they don't enter argmax/max.
    mfe_seq = np.where(trunc_mask, cummax_high_r, -np.inf)

    # peak_mfe_r per trade.
    peak_mfe_r = np.max(mfe_seq, axis=1)  # finite since trunc_mask covers bar 0
    # peak_mfe_bar — first bar achieving the peak (within truncation window).
    eps = 1e-12
    # Comparison must be float-aware.
    is_peak = (mfe_seq >= (peak_mfe_r[:, None] - eps))
    # Restrict argmax to trunc bars only (we already masked beyond-trunc to −inf).
    peak_mfe_bar = np.argmax(is_peak, axis=1)

    # Pre-peak window: bars 0..peak_mfe_bar inclusive.
    prepeak_mask = bar_idx[None, :] <= peak_mfe_bar[:, None]

    # frac_wrong_way_pre_peak (Def C): per-trade, min(low_r) in pre-peak ≤ −1.
    low_r_pre = np.where(prepeak_mask, low_r, np.inf)
    mae_pre_peak_r = np.min(low_r_pre, axis=1)
    wrong_way_pre_peak = (mae_pre_peak_r <= -1.0)

    # monotonicity_pre_peak per trade (loop on close_r, peak_mfe_bar).
    mono_pre_peak = _per_trade_monotonicity_pre_peak(close_r, peak_mfe_bar)

    # final_r_at_X.
    final_r = np.where(hit_any, -1.0, close_r[:, PATH_BARS - 1])

    # fwd_mfe_h240 = peak_mfe_r (max MFE within truncated path).
    fwd_mfe = peak_mfe_r

    # Aggregations.
    mono_centroid = float(np.mean(mono_pre_peak))
    fww_pp = float(np.mean(wrong_way_pre_peak))
    fwd_mfe_p50 = float(np.percentile(fwd_mfe, 50))
    fwd_mfe_p75 = float(np.percentile(fwd_mfe, 75))
    fwd_mfe_p95 = float(np.percentile(fwd_mfe, 95))
    fwd_mfe_mean = float(np.mean(fwd_mfe))
    frac_reach_1r = float(np.mean(fwd_mfe >= 1.0))
    frac_reach_2r = float(np.mean(fwd_mfe >= 2.0))
    final_r_mean = float(np.mean(final_r))
    final_r_p5 = float(np.percentile(final_r, 5))
    final_r_p25 = float(np.percentile(final_r, 25))
    final_r_p50 = float(np.percentile(final_r, 50))
    final_r_p75 = float(np.percentile(final_r, 75))
    final_r_p95 = float(np.percentile(final_r, 95))
    std = float(np.std(final_r, ddof=1)) if n_c > 1 else 0.0
    final_r_t = float(final_r_mean / (std / math.sqrt(n_c))) if std > 0 else 0.0

    cap_bind_count = int((~hit_any).sum())
    cap_bind_pct = cap_bind_count / n_c if n_c > 0 else 0.0

    # pct_peak_and_collapse.
    drawdown_from_peak = peak_mfe_r - final_r
    pc_flagged = (drawdown_from_peak >= PCT_PC_DRAWDOWN_FRACTION * peak_mfe_r) & (
        peak_mfe_r >= PCT_PC_MIN_PEAK_R
    )
    pct_pc = float(np.mean(pc_flagged))

    # Distribution shape on fwd_mfe.
    (
        shape_tag,
        dip_p,
        dip_stat,
        iqr,
        mode_count,
        mode_loc_1,
        mode_loc_2,
        mode_mass_1,
        mode_mass_2,
        min_mode_mass,
        mode_sep_r,
    ) = _classify_shape(fwd_mfe)

    # Gate evaluation.
    size_fraction = n_c / pool_size if pool_size > 0 else 0.0
    gate_A = mono_centroid >= FLOOR_MONO_PRE_PEAK
    gate_B = fwd_mfe_p50 >= FLOOR_FWD_MFE_P50
    gate_C = frac_reach_1r >= FLOOR_FRAC_REACH_1R
    gate_D = fww_pp <= FLOOR_FRAC_WRONG_WAY
    gate_E = size_fraction >= FLOOR_SIZE_FRACTION
    gate_F = shape_tag in ADMITTED_SHAPE_TAGS
    all_pass = bool(gate_A and gate_B and gate_C and gate_D and gate_E and gate_F)

    composite = (
        (mono_centroid - FLOOR_MONO_PRE_PEAK)
        + (frac_reach_1r - FLOOR_FRAC_REACH_1R)
        + (FLOOR_FRAC_WRONG_WAY - fww_pp)
    )

    return _SLPassMetrics(
        sl_atr_mult=float(sl_mult),
        r_per_trade_mean_price=float(np.mean(r_unit)),
        n_trades=int(n_c),
        cap_bind_count=cap_bind_count,
        cap_bind_pct=float(cap_bind_pct),
        mono_pre_peak_centroid=mono_centroid,
        frac_wrong_way_pre_peak=fww_pp,
        fwd_mfe_p50=fwd_mfe_p50,
        fwd_mfe_p75=fwd_mfe_p75,
        fwd_mfe_p95=fwd_mfe_p95,
        fwd_mfe_mean=fwd_mfe_mean,
        frac_reach_1R=frac_reach_1r,
        frac_reach_2R=frac_reach_2r,
        final_r_mean=final_r_mean,
        final_r_p5=final_r_p5,
        final_r_p25=final_r_p25,
        final_r_p50=final_r_p50,
        final_r_p75=final_r_p75,
        final_r_p95=final_r_p95,
        final_r_t_stat=final_r_t,
        pct_peak_and_collapse=pct_pc,
        dip_p_value=dip_p,
        dip_statistic=dip_stat,
        iqr_r=iqr,
        mode_count=int(mode_count),
        mode_loc_1=mode_loc_1,
        mode_loc_2=mode_loc_2,
        mode_mass_1=mode_mass_1,
        mode_mass_2=mode_mass_2,
        min_mode_mass=min_mode_mass,
        mode_separation_r=mode_sep_r,
        shape_tag=shape_tag,
        gate_A_mono=gate_A,
        gate_B_mfe=gate_B,
        gate_C_reach=gate_C,
        gate_D_wrong_way=gate_D,
        gate_E_size=gate_E,
        gate_F_shape=gate_F,
        size_fraction=float(size_fraction),
        all_pass=all_pass,
        capturability_composite=float(composite),
        peak_mfe_atr_units=float(fwd_mfe_p50 * sl_mult),
        fwd_mfe_arr=fwd_mfe,
        drawdown_from_peak_arr=drawdown_from_peak,
        peak_mfe_arr=peak_mfe_r,
    )


def _full_window_mono_centroid(
    cluster_entry_price: np.ndarray,
    cluster_atr: np.ndarray,
    cluster_closes: np.ndarray,
    sl_mult: float,
    truncate_at: np.ndarray,
) -> float:
    """Diagnostic-only: compute monotonicity_ratio_in_profit over the FULL
    truncated window (bars 0..truncate_at inclusive) — i.e. the v2.0 metric.

    Used in the pre-peak vs full-window mono comparison.
    """
    sl_distance = sl_mult * cluster_atr
    inv_r = 1.0 / sl_distance
    close_r = (cluster_closes - cluster_entry_price[:, None]) * inv_r[:, None]
    n = close_r.shape[0]
    out = np.zeros(n, dtype=float)
    for i in range(n):
        end_inclusive = int(truncate_at[i]) + 1
        cl = close_r[i, :end_inclusive]
        in_profit = cl > 0.0
        n_in_profit = int(in_profit.sum())
        if n_in_profit <= 1:
            out[i] = 0.0
            continue
        ipc = cl[in_profit]
        gte = ipc[1:] >= ipc[:-1]
        out[i] = float(gte.sum() / gte.size)
    return float(np.mean(out))


# ---------------------------------------------------------------------------
# Archetype mapping (Step 3)
# ---------------------------------------------------------------------------


@dataclass
class ArchetypeAssignment:
    cluster_id: int
    archetype_label: str
    archetype_row: str
    candidate_rows: List[int]
    candidate_labels: List[str]
    near_miss_rows: List[int]
    assignment_certainty: str  # "clean" | "boundary" | "near-miss" | "unclassified"
    reason: str
    bimodal_split_also_active: bool


def _row_centroid_match(
    mono: float, peaks: float, pullback: float, ttp: float, near_miss: bool
) -> Dict[int, Tuple[bool, List[str]]]:
    """Return {row: (matches, unmet_list)} for §11 rows 1-6 evaluated on the
    four-feature centroid. If near_miss=True, thresholds are relaxed by
    NEAR_MISS_RELATIVE × threshold (e.g. mono ≥ 0.55 → ≥ 0.495).

    Rows with non-centroid constraints (3, 4 → forward outcome; 5 → not
    expressible from centroid) have their centroid-feature parts evaluated;
    forward-outcome / non-centroid parts are deferred / reported as the
    cluster's "remaining" requirements.
    """
    def slack(threshold: float) -> float:
        return abs(threshold) * NEAR_MISS_RELATIVE if near_miss else 0.0

    out: Dict[int, Tuple[bool, List[str]]] = {}

    # Row 1 — Monotone ascent
    unmet: List[str] = []
    if not (mono >= 0.55 - slack(0.55)):
        unmet.append(f"mono {mono:.3f} < 0.55")
    if not (peaks <= 4 + slack(4)):
        unmet.append(f"local_peaks {peaks:.2f} > 4")
    if not (ttp >= 0.50 - slack(0.50)):
        unmet.append(f"time_to_peak_rel {ttp:.3f} < 0.50")
    out[1] = (len(unmet) == 0, unmet)

    # Row 2 — Stepwise climber
    unmet = []
    if not (mono >= 0.50 - slack(0.50)):
        unmet.append(f"mono {mono:.3f} < 0.50")
    if not ((5 - slack(5)) <= peaks <= (30 + slack(30))):
        unmet.append(f"local_peaks {peaks:.2f} not in [5, 30]")
    if not (pullback <= 0.5 + slack(0.5)):
        unmet.append(f"pullback {pullback:.3f} > 0.5R")
    if not (ttp >= 0.50 - slack(0.50)):
        unmet.append(f"time_to_peak_rel {ttp:.3f} < 0.50")
    out[2] = (len(unmet) == 0, unmet)

    # Row 3 — Early-peak hold (centroid part only: ttp ≤ 0.30)
    unmet = []
    if not (ttp <= 0.30 + slack(0.30)):
        unmet.append(f"time_to_peak_rel {ttp:.3f} > 0.30")
    out[3] = (len(unmet) == 0, unmet)

    # Row 4 — Peak-and-collapse (centroid part only: ttp ≤ 0.30)
    unmet = []
    if not (ttp <= 0.30 + slack(0.30)):
        unmet.append(f"time_to_peak_rel {ttp:.3f} > 0.30")
    out[4] = (len(unmet) == 0, unmet)

    # Row 5 — V-shape recovery: not centroid-matchable from 4-feature centroid
    out[5] = (False, ["row 5 requires MAE timing + peak position — not centroid-matchable"])

    # Row 6 — Random walk
    unmet = []
    if not (peaks >= 8 - slack(8)):
        unmet.append(f"local_peaks {peaks:.2f} < 8")
    if not (mono <= 0.30 + slack(0.30)):
        unmet.append(f"mono {mono:.3f} > 0.30")
    if not (pullback >= 1.0 - slack(1.0)):
        unmet.append(f"pullback {pullback:.3f} < 1.0R")
    out[6] = (len(unmet) == 0, unmet)

    return out


_ROW_LABELS: Dict[int, str] = {
    1: "Monotone ascent",
    2: "Stepwise climber",
    3: "Early-peak hold",
    4: "Peak-and-collapse",
    5: "V-shape recovery",
    6: "Random walk",
}


def _assign_archetype(
    cluster_id: int,
    centroid: Dict[str, float],
    sel_metrics: Optional[_SLPassMetrics],
    sel_passed: bool,
) -> ArchetypeAssignment:
    """Apply Step 3 archetype mapping.

    Disambiguation rules:
      - Cluster 2 (Step 2 boundary on rows 3/4): use Step 3 pct_peak_and_collapse:
        < 0.30 → Early-peak hold; ≥ 0.50 → Peak-and-collapse; in-between → flag.
      - Other clusters: re-evaluate §11 rows 1-6 against centroid (strict first;
        if none match, near-miss with 10% relative tolerance).
      - bimodal_split_also_active: True iff selected SL's shape_tag is
        bimodal_separated AND the cluster also matches a base row.
    """
    mono = centroid["mono"]
    peaks = centroid["local_peaks"]
    pullback = centroid["pullback"]
    ttp = centroid["time_to_peak"]

    # Cluster 2 — known Step 2 row-3/4 boundary.
    # Use Step 3 pct_pc only if §2 admitted the cluster.
    if cluster_id == 2 and sel_passed and sel_metrics is not None:
        pct_pc = sel_metrics.pct_peak_and_collapse
        if pct_pc < 0.30:
            base_row = 3
            base_label = "Early-peak hold"
            reason = (
                f"row-3/4 boundary disambiguated via Step 3 pct_peak_and_collapse "
                f"= {pct_pc:.2%} < 0.30 → row 3 (Early-peak hold)"
            )
            certainty = "clean"
        elif pct_pc >= 0.50:
            base_row = 4
            base_label = "Peak-and-collapse"
            reason = (
                f"row-3/4 boundary disambiguated via Step 3 pct_peak_and_collapse "
                f"= {pct_pc:.2%} ≥ 0.50 → row 4 (Peak-and-collapse)"
            )
            certainty = "clean"
        else:
            base_row = 0
            base_label = "Early-peak hold + Peak-and-collapse (ambiguous)"
            reason = (
                f"row-3/4 ambiguity zone: pct_peak_and_collapse {pct_pc:.2%} "
                f"∈ [0.30, 0.50) — empirical capture-ratio test deferred"
            )
            certainty = "boundary"
        bimodal_also = sel_metrics.shape_tag == "bimodal_separated" and base_row > 0
        return ArchetypeAssignment(
            cluster_id=cluster_id,
            archetype_label=base_label,
            archetype_row=str(base_row) if base_row > 0 else "3/4",
            candidate_rows=[3, 4],
            candidate_labels=["Early-peak hold", "Peak-and-collapse"],
            near_miss_rows=[],
            assignment_certainty=certainty,
            reason=reason,
            bimodal_split_also_active=bimodal_also,
        )

    # Strict-centroid match against rows 1-6 (centroid-feature parts only).
    strict = _row_centroid_match(mono, peaks, pullback, ttp, near_miss=False)
    strict_rows = [r for r, (m, _) in strict.items() if m]
    # Rows 3 and 4 share only the centroid part `ttp ≤ 0.30`. When both
    # strict-match, the centroid-feature is met but disambiguation needs
    # Step 3 pct_peak_and_collapse — same as Cluster 2 path.
    if set(strict_rows) == {3, 4} and sel_passed and sel_metrics is not None:
        pct_pc = sel_metrics.pct_peak_and_collapse
        if pct_pc < 0.30:
            return ArchetypeAssignment(
                cluster_id=cluster_id,
                archetype_label="Early-peak hold",
                archetype_row="3",
                candidate_rows=[3, 4],
                candidate_labels=["Early-peak hold", "Peak-and-collapse"],
                near_miss_rows=[],
                assignment_certainty="clean",
                reason=(
                    f"rows 3+4 centroid-matched; pct_peak_and_collapse "
                    f"= {pct_pc:.2%} < 0.30 → row 3"
                ),
                bimodal_split_also_active=sel_metrics.shape_tag == "bimodal_separated",
            )
        if pct_pc >= 0.50:
            return ArchetypeAssignment(
                cluster_id=cluster_id,
                archetype_label="Peak-and-collapse",
                archetype_row="4",
                candidate_rows=[3, 4],
                candidate_labels=["Early-peak hold", "Peak-and-collapse"],
                near_miss_rows=[],
                assignment_certainty="clean",
                reason=(
                    f"rows 3+4 centroid-matched; pct_peak_and_collapse "
                    f"= {pct_pc:.2%} ≥ 0.50 → row 4"
                ),
                bimodal_split_also_active=sel_metrics.shape_tag == "bimodal_separated",
            )
    if strict_rows:
        rows_str = "+".join(str(r) for r in strict_rows)
        labels = [_ROW_LABELS[r] for r in strict_rows]
        certainty = "clean" if len(strict_rows) == 1 else "boundary"
        bimodal_also = bool(
            sel_metrics is not None and sel_metrics.shape_tag == "bimodal_separated"
        )
        return ArchetypeAssignment(
            cluster_id=cluster_id,
            archetype_label=" + ".join(labels),
            archetype_row=rows_str,
            candidate_rows=strict_rows,
            candidate_labels=labels,
            near_miss_rows=[],
            assignment_certainty=certainty,
            reason=(
                "strict centroid match"
                if len(strict_rows) == 1
                else f"multiple strict matches: {labels}"
            ),
            bimodal_split_also_active=bimodal_also,
        )

    # Strict failed: try 10% near-miss tolerance.
    relaxed = _row_centroid_match(mono, peaks, pullback, ttp, near_miss=True)
    near_rows = [r for r, (m, _) in relaxed.items() if m]
    if near_rows:
        labels = [_ROW_LABELS[r] for r in near_rows]
        # If both rows 3 and 4 are near-misses, use pct_pc to disambiguate
        # at near-miss certainty (same logic as above).
        bimodal_also = bool(
            sel_metrics is not None and sel_metrics.shape_tag == "bimodal_separated"
        )
        return ArchetypeAssignment(
            cluster_id=cluster_id,
            archetype_label=" + ".join(labels) + " (near-miss)",
            archetype_row="+".join(str(r) for r in near_rows),
            candidate_rows=[],
            candidate_labels=[],
            near_miss_rows=near_rows,
            assignment_certainty="near-miss",
            reason=(
                f"strict centroid match failed; relaxed by ±{NEAR_MISS_RELATIVE:.0%} of "
                f"each row threshold → {labels}"
            ),
            bimodal_split_also_active=bimodal_also,
        )

    return ArchetypeAssignment(
        cluster_id=cluster_id,
        archetype_label="unclassified",
        archetype_row="",
        candidate_rows=[],
        candidate_labels=[],
        near_miss_rows=[],
        assignment_certainty="unclassified",
        reason="no §11 row centroid-feature part is satisfied, even with 10% near-miss tolerance — chat-level disambiguation",
        bimodal_split_also_active=False,
    )


# ---------------------------------------------------------------------------
# CSV writers (deterministic)
# ---------------------------------------------------------------------------


def _fmt_g(x: Any) -> Any:
    if x is None or (isinstance(x, float) and (math.isnan(x) or not math.isfinite(x))):
        return ""
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return int(bool(x))
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.10g}"
    return x


def _write_sl_sweep(rows: List[Dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, na_rep="", lineterminator="\n")


def _write_distribution(rows: List[Dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, na_rep="", lineterminator="\n")


def _write_archetype_summaries(rows: List[Dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, na_rep="", lineterminator="\n")


def _write_capturability_pass_list(rows: List[Dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, na_rep="", lineterminator="\n")


def _write_pct_pc(rows: List[Dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, na_rep="", lineterminator="\n")


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Sweep + selection per cluster
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    cluster_id: int
    centroid: Dict[str, float]
    size: int
    size_fraction: float
    sweep: Dict[float, _SLPassMetrics]
    selected_sl: Optional[float]
    selection_reason: str
    full_window_mono_at_selected: Optional[float]
    pre_peak_minus_full_window_mono: Optional[float]
    archetype: ArchetypeAssignment


def _select_sl(sweep: Dict[float, _SLPassMetrics]) -> Tuple[Optional[float], str]:
    """Apply §7 v2.1.1 SL selection rule. Return (selected_sl, reason)."""
    passing = [(sl, m) for sl, m in sweep.items() if m.all_pass]
    if not passing:
        # Build a reason summary listing the failed floors at each SL.
        per_sl = []
        for sl, m in sorted(sweep.items()):
            fails = []
            if not m.gate_A_mono:
                fails.append(f"A:mono {m.mono_pre_peak_centroid:.3f}<{FLOOR_MONO_PRE_PEAK}")
            if not m.gate_B_mfe:
                fails.append(f"B:mfe_p50 {m.fwd_mfe_p50:.3f}<{FLOOR_FWD_MFE_P50}")
            if not m.gate_C_reach:
                fails.append(f"C:reach_1R {m.frac_reach_1R:.3f}<{FLOOR_FRAC_REACH_1R}")
            if not m.gate_D_wrong_way:
                fails.append(f"D:wrong_way {m.frac_wrong_way_pre_peak:.3f}>{FLOOR_FRAC_WRONG_WAY}")
            if not m.gate_E_size:
                fails.append(f"E:size {m.size_fraction:.3f}<{FLOOR_SIZE_FRACTION}")
            if not m.gate_F_shape:
                fails.append(f"F:shape {m.shape_tag}")
            per_sl.append(f"SL={sl}: {','.join(fails) if fails else 'pass'}")
        return None, f"no SL passes §2 floors. Detail: {'; '.join(per_sl)}"

    # Sort passing by composite descending.
    passing.sort(key=lambda t: -t[1].capturability_composite)
    top = passing[0]
    top_sl, top_m = top
    # Tiebreaker 1 candidates: within COMPOSITE_TIE_TOLERANCE of top composite.
    near = [
        (sl, m)
        for sl, m in passing
        if (top_m.capturability_composite - m.capturability_composite) <= COMPOSITE_TIE_TOLERANCE
    ]
    if len(near) > 1:
        # Tiebreaker 1: larger peak_mfe in ATR units (fwd_mfe_p50 × sl_mult).
        near.sort(key=lambda t: -t[1].peak_mfe_atr_units)
        atr_top = near[0]
        atr_top_value = atr_top[1].peak_mfe_atr_units
        atr_near = [
            t for t in near if abs(t[1].peak_mfe_atr_units - atr_top_value) < 1e-9
        ]
        if len(atr_near) > 1:
            # Tiebreaker 2: smaller SL (parsimony).
            atr_near.sort(key=lambda t: t[0])
            sel = atr_near[0]
            reason = (
                f"composite-max within {COMPOSITE_TIE_TOLERANCE} tolerance ties on "
                f"peak_mfe ATR-units; smaller SL (parsimony) selected — "
                f"composite={sel[1].capturability_composite:.4f} "
                f"peak_mfe_atr={sel[1].peak_mfe_atr_units:.3f}"
            )
            return float(sel[0]), reason
        sel = atr_top
        reason = (
            f"composite-max within {COMPOSITE_TIE_TOLERANCE} tolerance; tiebreaker 1 "
            f"(larger peak_mfe ATR-units) selected — "
            f"composite={sel[1].capturability_composite:.4f} "
            f"peak_mfe_atr={sel[1].peak_mfe_atr_units:.3f}"
        )
        return float(sel[0]), reason
    sel = top
    reason = (
        f"unique composite max = {sel[1].capturability_composite:.4f} "
        f"(margin over next-best ≥ {COMPOSITE_TIE_TOLERANCE})"
    )
    return float(sel[0]), reason


# ---------------------------------------------------------------------------
# Diagnostics markdown
# ---------------------------------------------------------------------------


def _pp_table_sweep(cluster: ClusterResult) -> List[str]:
    lines: List[str] = []
    lines.append(
        "| SL ×ATR | mono_pp | wrong_way_pp | mfe_p50 | reach_1R | reach_2R | "
        "shape_tag | dip_p | composite | peak_mfe_ATR | A | B | C | D | E | F | all_pass |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|---|---|---|---|---|"
    )
    for sl in sorted(cluster.sweep.keys()):
        m = cluster.sweep[sl]
        def tick(b: bool) -> str:
            return "✓" if b else "✗"
        lines.append(
            f"| {sl} | {m.mono_pre_peak_centroid:.3f} | {m.frac_wrong_way_pre_peak:.3f} | "
            f"{m.fwd_mfe_p50:.3f} | {m.frac_reach_1R:.3f} | {m.frac_reach_2R:.3f} | "
            f"{m.shape_tag} | {m.dip_p_value:.4g} | {m.capturability_composite:+.4f} | "
            f"{m.peak_mfe_atr_units:.3f} | {tick(m.gate_A_mono)} | {tick(m.gate_B_mfe)} | "
            f"{tick(m.gate_C_reach)} | {tick(m.gate_D_wrong_way)} | {tick(m.gate_E_size)} | "
            f"{tick(m.gate_F_shape)} | {'PASS' if m.all_pass else 'FAIL'} |"
        )
    return lines


def _pp_table_selected(cluster: ClusterResult) -> List[str]:
    if cluster.selected_sl is None:
        return [
            "| Cluster | Size | Size fraction | Selected SL | mono_pp | reach_1R | "
            "wrong_way_pp | mfe_p50 | shape_tag | composite | §2 | Archetype |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|",
            f"| {cluster.cluster_id} | {cluster.size} | {cluster.size_fraction:.4f} | "
            f"FAIL | — | — | — | — | — | — | FAIL | {cluster.archetype.archetype_label} |",
        ]
    m = cluster.sweep[cluster.selected_sl]
    return [
        "| Cluster | Size | Size fraction | Selected SL | mono_pp | reach_1R | "
        "wrong_way_pp | mfe_p50 | shape_tag | composite | §2 | Archetype |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|",
        f"| {cluster.cluster_id} | {cluster.size} | {cluster.size_fraction:.4f} | "
        f"{cluster.selected_sl} × ATR | {m.mono_pre_peak_centroid:.3f} | "
        f"{m.frac_reach_1R:.3f} | {m.frac_wrong_way_pre_peak:.3f} | {m.fwd_mfe_p50:.3f} | "
        f"{m.shape_tag} | {m.capturability_composite:+.4f} | PASS | "
        f"{cluster.archetype.archetype_label} |",
    ]


def write_diagnostics(
    out_path: Path,
    cluster_results: List[ClusterResult],
    pool_size: int,
    sha_run1: Dict[str, str],
    sha_run2: Optional[Dict[str, str]],
    determinism_gate: str,
    config_paths: Dict[str, str],
    config_shas: Dict[str, str],
) -> str:
    survivors = [c for c in cluster_results if c.selected_sl is not None]
    arc_pass = len(survivors) >= 1
    arc_disp = "PASS" if arc_pass else "FAIL"

    lines: List[str] = []
    lines.append("# Arc 4 — Step 3 capturability + SL sweep diagnostics")
    lines.append("")
    lines.append("Protocol: `L_ARC_PROTOCOL.md` v2.1.1 §7 (SL sweep, capturability composite, bimodal_separated test)")
    lines.append(
        "Signal:   `TRIAL__univariate_extreme__bar_range_top_decile__neg__h_001` "
        "(LCHAR_TOPN_REGISTRY.md Entry 4)"
    )
    lines.append("")

    # Headline.
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"{len(cluster_results)} clusters tested under {len(CANDIDATE_SL_MULTS)} "
        f"candidate SLs × ATR. **{len(survivors)} cluster(s) pass §2 conjunctively** "
        f"({'; '.join(f'cluster {c.cluster_id}: SL={c.selected_sl}×ATR, {c.archetype.archetype_label}' for c in survivors) if survivors else 'no survivors'}). "
        f"Arc-level §7 gate disposition (≥ 1 cluster passes): **{arc_disp}**. "
        f"Determinism: **{determinism_gate}**."
    )
    lines.append("")

    # Per-cluster SL sweep tables.
    for c in cluster_results:
        lines.append(f"## Cluster {c.cluster_id} — SL sweep")
        lines.append("")
        lines.append(
            f"Path-shape centroid: mono = {c.centroid['mono']:.4f}, "
            f"local_peaks = {c.centroid['local_peaks']:.2f}, "
            f"pullback = {c.centroid['pullback']:.4f}, "
            f"time_to_peak = {c.centroid['time_to_peak']:.4f}. "
            f"Size: {c.size} ({c.size_fraction:.2%} of pool)."
        )
        lines.append("")
        lines.extend(_pp_table_sweep(c))
        lines.append("")
        if c.selected_sl is not None:
            lines.append(f"**Selected SL: {c.selected_sl} × ATR** — {c.selection_reason}")
        else:
            lines.append(f"**No SL passes §2** — {c.selection_reason}")
        lines.append("")

    # Per-cluster selected-SL identity.
    lines.append("## Per-cluster selected-SL identity")
    lines.append("")
    for c in cluster_results:
        lines.extend(_pp_table_selected(c))
        lines.append("")

    # Pre-peak vs full-window mono comparison (v2.1.1 rescue test).
    lines.append("## Pre-peak vs full-window monotonicity (v2.1.1 rescue test)")
    lines.append("")
    lines.append(
        "Per-cluster comparison at the selected SL (or at SL=2.0×ATR for clusters "
        "that fail §2, to give a comparable baseline). A positive delta = the "
        "v2.1.1 pre-peak metric is more permissive than the v2.0 full-window metric "
        "and rescues clusters whose post-peak retracement dragged down the v2.0 "
        "monotonicity centroid."
    )
    lines.append("")
    lines.append("| Cluster | SL used | pre-peak mono | full-window mono | delta (pp − fw) |")
    lines.append("|---:|---:|---:|---:|---:|")
    for c in cluster_results:
        sl_used = c.selected_sl if c.selected_sl is not None else 2.0
        m = c.sweep[sl_used]
        fw = c.full_window_mono_at_selected
        if fw is None:
            lines.append(
                f"| {c.cluster_id} | {sl_used} × ATR | {m.mono_pre_peak_centroid:.4f} | "
                f"— | — |"
            )
        else:
            delta = m.mono_pre_peak_centroid - fw
            lines.append(
                f"| {c.cluster_id} | {sl_used} × ATR | {m.mono_pre_peak_centroid:.4f} | "
                f"{fw:.4f} | {delta:+.4f} |"
            )
    lines.append("")

    # Cap-binding behaviour per SL per cluster.
    lines.append("## Cap-binding behaviour (trade never hit SL within 240 bars)")
    lines.append("")
    lines.append(
        f"Per §5 auto-extend rule, pool-level cap-binding > {CAP_BIND_WARN_PCT:.0%} "
        "flags forward-window extension."
    )
    lines.append("")
    sl_cols = " | ".join(f"{sl}× ATR" for sl in CANDIDATE_SL_MULTS)
    lines.append(f"| Cluster | {sl_cols} |")
    lines.append("|---|" + "---:|" * len(CANDIDATE_SL_MULTS))
    for c in cluster_results:
        cells = []
        for sl in CANDIDATE_SL_MULTS:
            m = c.sweep[sl]
            tag = "⚠" if m.cap_bind_pct > CAP_BIND_WARN_PCT else ""
            cells.append(f"{m.cap_bind_pct:.1%}{tag}")
        lines.append(f"| {c.cluster_id} | " + " | ".join(cells) + " |")
    # Pool-level totals.
    pool_cells = []
    for sl in CANDIDATE_SL_MULTS:
        pool_cb = sum(c.sweep[sl].cap_bind_count for c in cluster_results)
        pool_pct = pool_cb / pool_size
        tag = "⚠" if pool_pct > CAP_BIND_WARN_PCT else ""
        pool_cells.append(f"{pool_pct:.1%}{tag}")
    lines.append("| **Pool** | " + " | ".join(pool_cells) + " |")
    lines.append("")

    # Distribution shape per cluster at selected SL.
    lines.append("## Distribution-shape detail at selected SL")
    lines.append("")
    lines.append(
        "| Cluster | SL used | dip_p | dip_stat | IQR_R | mode_count | "
        "mode_loc_1 | mode_loc_2 | min_mode_mass | mode_sep_R | shape_tag |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for c in cluster_results:
        sl_used = c.selected_sl if c.selected_sl is not None else 2.0
        m = c.sweep[sl_used]
        lines.append(
            f"| {c.cluster_id} | {sl_used} × ATR | {m.dip_p_value:.4g} | "
            f"{m.dip_statistic:.4g} | {m.iqr_r:.3f} | {m.mode_count} | "
            f"{m.mode_loc_1:.3f} | {m.mode_loc_2:.3f} | "
            f"{m.min_mode_mass:.3f} | {m.mode_separation_r:.3f} | {m.shape_tag} |"
        )
    lines.append("")

    # Archetype assignments + Step 2 reconciliation.
    lines.append("## Archetype assignments (Step 3 mapping)")
    lines.append("")
    lines.append(
        "| Cluster | Archetype | Row(s) | Certainty | Bimodal split also active | Reason |"
    )
    lines.append("|---:|---|---|---|---|---|")
    for c in cluster_results:
        a = c.archetype
        lines.append(
            f"| {c.cluster_id} | {a.archetype_label} | {a.archetype_row or '—'} | "
            f"{a.assignment_certainty} | "
            f"{'yes' if a.bimodal_split_also_active else 'no'} | {a.reason} |"
        )
    lines.append("")

    lines.append("### Step 2 → Step 3 archetype reconciliation")
    lines.append("")
    lines.append(
        "Step 2 reported 0 clean / 1 boundary / 3 unclassified. Step 3 re-evaluation "
        "with forward-geometry + 10% near-miss tolerance:"
    )
    lines.append("")
    for c in cluster_results:
        a = c.archetype
        lines.append(
            f"- **Cluster {c.cluster_id}** (Step 2: unclassified or boundary) → "
            f"Step 3: **{a.archetype_label}** ({a.assignment_certainty})."
        )
    lines.append("")

    # Per-cluster pct_peak_and_collapse summary (informational; not gated).
    lines.append("## pct_peak_and_collapse (informational, post-peak; not gated by §2)")
    lines.append("")
    lines.append("Drawdown_from_peak distribution at each cluster's selected SL.")
    lines.append("")
    lines.append(
        "| Cluster | SL used | pct_peak_and_collapse | drawdown_from_peak p50 | "
        "drawdown_from_peak p95 |"
    )
    lines.append("|---:|---:|---:|---:|---:|")
    for c in cluster_results:
        sl_used = c.selected_sl if c.selected_sl is not None else 2.0
        m = c.sweep[sl_used]
        dd = m.drawdown_from_peak_arr
        if dd.size == 0:
            lines.append(f"| {c.cluster_id} | {sl_used} × ATR | — | — | — |")
            continue
        lines.append(
            f"| {c.cluster_id} | {sl_used} × ATR | {m.pct_peak_and_collapse:.2%} | "
            f"{float(np.percentile(dd, 50)):.3f} | {float(np.percentile(dd, 95)):.3f} |"
        )
    lines.append("")

    # Determinism table.
    lines.append("## Determinism")
    lines.append("")
    lines.append("Two-run byte-identical sha256 over all output CSVs:")
    lines.append("")
    lines.append("| File | Run 1 sha256 | Run 2 sha256 | Match |")
    lines.append("|---|---|---|---|")
    for fname in sorted(sha_run1.keys()):
        s1 = sha_run1[fname]
        s2 = sha_run2.get(fname) if sha_run2 else None
        if s2 is None:
            match = "—"
        else:
            match = "PASS" if s1 == s2 else "FAIL"
        lines.append(f"| `{fname}` | `{s1}` | `{s2 or '—'}` | {match} |")
    lines.append("")
    lines.append(f"**Determinism: {determinism_gate}**")
    lines.append("")

    # Configs.
    lines.append("## Config / input sha256s")
    lines.append("")
    for label, path in config_paths.items():
        lines.append(f"- `{label}` (`{path}`) — `{config_shas[label]}`")
    lines.append("")

    # Cross-arc observations (informational — no calibration changes).
    lines.append("## Cross-arc observations")
    lines.append("")
    lines.append(
        "Recorded but not acted on — protocol governance requires a separate "
        "calibration document + KH-24 anchor recheck + chat approval before "
        "any floor/threshold change."
    )
    lines.append("")
    for c in cluster_results:
        if c.selected_sl is None:
            sweeps = list(c.sweep.values())
            # Identify which gate killed the cluster across SLs.
            never_a = all(not m.gate_A_mono for m in sweeps)
            never_b = all(not m.gate_B_mfe for m in sweeps)
            never_c = all(not m.gate_C_reach for m in sweeps)
            never_d = all(not m.gate_D_wrong_way for m in sweeps)
            never_f = all(not m.gate_F_shape for m in sweeps)
            killers: List[str] = []
            if never_a:
                killers.append("§2-A (mono_pre_peak)")
            if never_b:
                killers.append("§2-B (fwd_mfe_p50)")
            if never_c:
                killers.append("§2-C (frac_reach_1R)")
            if never_d:
                killers.append("§2-D (frac_wrong_way_pre_peak)")
            if never_f:
                killers.append("§2-F (shape_tag admit)")
            if killers:
                lines.append(
                    f"- Cluster {c.cluster_id} dies on: "
                    f"{', '.join(killers)} — never satisfied across any candidate SL."
                )
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return arc_disp


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _env_dict() -> Dict[str, str]:
    try:
        import diptest as _dt  # type: ignore

        dt_ver = getattr(_dt, "__version__", "unknown")
    except Exception:
        dt_ver = "not_installed"
    try:
        import sklearn  # type: ignore

        sk_ver = sklearn.__version__
    except Exception:
        sk_ver = "not_installed"
    try:
        import scipy  # type: ignore

        sp_ver = scipy.__version__
    except Exception:
        sp_ver = "not_installed"
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sk_ver,
        "scipy": sp_ver,
        "diptest": dt_ver,
    }


def run_once(
    trades_csv: Path,
    paths_csv: Path,
    clusters_csv: Path,
    centroids_csv: Path,
    out_dir: Path,
) -> Tuple[List[ClusterResult], Dict[str, str], int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = _load_trades_all(trades_csv)
    n_trades = int(len(trades))
    entry_price = trades["entry_price"].to_numpy(dtype=float)
    atr_14 = trades["atr_14_at_signal"].to_numpy(dtype=float)
    trade_ids = trades["trade_id"].to_numpy(dtype=int)

    paths_trade_ids, opens, highs, lows, closes = _load_paths_2d(paths_csv, n_trades)
    if not np.array_equal(trade_ids, paths_trade_ids):
        raise ValueError(
            "trade_id ordering mismatch between trades_all.csv and trades_paths.csv "
            "(both must be sorted ascending by trade_id and identical sets)"
        )

    clusters = _load_clusters(clusters_csv)
    if not np.array_equal(trade_ids, clusters["trade_id"].to_numpy(dtype=int)):
        raise ValueError("trade_id ordering mismatch between trades_all.csv and clusters_K4.csv")
    cluster_id_per_trade = clusters["cluster_id"].to_numpy(dtype=int)

    centroids = _load_centroids(centroids_csv)
    centroid_map: Dict[int, Dict[str, float]] = {}
    for _, row in centroids.iterrows():
        cid = int(row["cluster_id"])
        centroid_map[cid] = {
            "mono": float(row["mono_raw"]),
            "local_peaks": float(row["local_peaks_raw"]),
            "pullback": float(row["pullback_raw"]),
            "time_to_peak": float(row["time_to_peak_raw"]),
        }

    pool_size = int(n_trades)
    sha_files: Dict[str, str] = {}

    cluster_ids_sorted = sorted(centroid_map.keys())
    cluster_results: List[ClusterResult] = []

    for cid in cluster_ids_sorted:
        mask = cluster_id_per_trade == cid
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        c_entry = entry_price[idx]
        c_atr = atr_14[idx]
        c_opens = opens[idx]
        c_highs = highs[idx]
        c_lows = lows[idx]
        c_closes = closes[idx]
        sweep: Dict[float, _SLPassMetrics] = {}
        for sl_mult in CANDIDATE_SL_MULTS:
            sweep[float(sl_mult)] = _evaluate_sl_for_cluster(
                sl_mult=float(sl_mult),
                cluster_entry_price=c_entry,
                cluster_atr=c_atr,
                cluster_opens=c_opens,
                cluster_highs=c_highs,
                cluster_lows=c_lows,
                cluster_closes=c_closes,
                pool_size=pool_size,
            )
        selected_sl, sel_reason = _select_sl(sweep)

        # Full-window mono diagnostic at selected SL (or SL=2.0 if FAIL).
        sl_for_fw = selected_sl if selected_sl is not None else 2.0
        sl_dist_at = sl_for_fw * c_atr
        sl_price_at = c_entry - sl_dist_at
        hit_mask = c_lows <= sl_price_at[:, None]
        hit_any = hit_mask.any(axis=1)
        first_hit_bar = np.argmax(hit_mask, axis=1)
        truncate_at = np.where(hit_any, first_hit_bar, PATH_BARS - 1).astype(int)
        full_window_mono = _full_window_mono_centroid(
            c_entry, c_atr, c_closes, sl_for_fw, truncate_at
        )

        # Archetype assignment (uses selected SL's pct_pc + shape_tag if passed).
        sel_metrics = sweep[selected_sl] if selected_sl is not None else None
        arche = _assign_archetype(
            cluster_id=cid,
            centroid=centroid_map[cid],
            sel_metrics=sel_metrics,
            sel_passed=selected_sl is not None,
        )

        pre_peak_at = sweep[sl_for_fw].mono_pre_peak_centroid
        cluster_results.append(
            ClusterResult(
                cluster_id=cid,
                centroid=centroid_map[cid],
                size=int(idx.size),
                size_fraction=float(idx.size / pool_size),
                sweep=sweep,
                selected_sl=selected_sl,
                selection_reason=sel_reason,
                full_window_mono_at_selected=full_window_mono,
                pre_peak_minus_full_window_mono=pre_peak_at - full_window_mono,
                archetype=arche,
            )
        )

        # Write per-cluster sweep CSV.
        sweep_rows: List[Dict[str, Any]] = []
        for sl_mult in CANDIDATE_SL_MULTS:
            m = sweep[float(sl_mult)]
            sweep_rows.append(
                {
                    "sl_atr_mult": _fmt_g(m.sl_atr_mult),
                    "R_per_atr_unit": _fmt_g(m.sl_atr_mult),  # by definition R = X × ATR
                    "n_trades": int(m.n_trades),
                    "cap_bind_count": int(m.cap_bind_count),
                    "cap_bind_pct": _fmt_g(m.cap_bind_pct),
                    "mono_pre_peak_centroid": _fmt_g(m.mono_pre_peak_centroid),
                    "frac_wrong_way_pre_peak": _fmt_g(m.frac_wrong_way_pre_peak),
                    "fwd_mfe_h240_p50": _fmt_g(m.fwd_mfe_p50),
                    "fwd_mfe_h240_p75": _fmt_g(m.fwd_mfe_p75),
                    "fwd_mfe_h240_p95": _fmt_g(m.fwd_mfe_p95),
                    "fwd_mfe_h240_mean": _fmt_g(m.fwd_mfe_mean),
                    "frac_reach_1R": _fmt_g(m.frac_reach_1R),
                    "frac_reach_2R": _fmt_g(m.frac_reach_2R),
                    "final_r_mean": _fmt_g(m.final_r_mean),
                    "final_r_p5": _fmt_g(m.final_r_p5),
                    "final_r_p25": _fmt_g(m.final_r_p25),
                    "final_r_p50": _fmt_g(m.final_r_p50),
                    "final_r_p75": _fmt_g(m.final_r_p75),
                    "final_r_p95": _fmt_g(m.final_r_p95),
                    "final_r_t_stat": _fmt_g(m.final_r_t_stat),
                    "dip_p_value": _fmt_g(m.dip_p_value),
                    "dip_statistic": _fmt_g(m.dip_statistic),
                    "iqr_r": _fmt_g(m.iqr_r),
                    "mode_count": int(m.mode_count),
                    "mode_loc_1": _fmt_g(m.mode_loc_1),
                    "mode_loc_2": _fmt_g(m.mode_loc_2),
                    "min_mode_mass": _fmt_g(m.min_mode_mass),
                    "mode_separation_r": _fmt_g(m.mode_separation_r),
                    "shape_tag": m.shape_tag,
                    "gate_A_mono": int(m.gate_A_mono),
                    "gate_B_mfe": int(m.gate_B_mfe),
                    "gate_C_reach": int(m.gate_C_reach),
                    "gate_D_wrong_way": int(m.gate_D_wrong_way),
                    "gate_E_size": int(m.gate_E_size),
                    "gate_F_shape": int(m.gate_F_shape),
                    "size_fraction": _fmt_g(m.size_fraction),
                    "all_pass": int(m.all_pass),
                    "capturability_composite": _fmt_g(m.capturability_composite),
                    "peak_mfe_atr_units": _fmt_g(m.peak_mfe_atr_units),
                    "pct_peak_and_collapse": _fmt_g(m.pct_peak_and_collapse),
                }
            )
        sweep_path = out_dir / f"sl_sweep_cluster_{cid}.csv"
        _write_sl_sweep(sweep_rows, sweep_path)
        sha_files[f"sl_sweep_cluster_{cid}.csv"] = _file_sha256(sweep_path)

        # Distribution CSV (at selected SL, or SL=2.0 for FAIL).
        m_sel = sweep[sl_for_fw]
        fwd = m_sel.fwd_mfe_arr
        mass_bands = []
        if fwd.size > 0:
            mass_bands = [
                ("0_to_0_5R", float(np.mean((fwd >= 0.0) & (fwd < 0.5)))),
                ("0_5_to_1R", float(np.mean((fwd >= 0.5) & (fwd < 1.0)))),
                ("1_to_2R", float(np.mean((fwd >= 1.0) & (fwd < 2.0)))),
                ("2_to_5R", float(np.mean((fwd >= 2.0) & (fwd < 5.0)))),
                ("gt_5R", float(np.mean(fwd >= 5.0))),
                ("lt_0R", float(np.mean(fwd < 0.0))),
            ]
        dist_rows = [
            {
                "cluster_id": cid,
                "sl_atr_mult_used": _fmt_g(sl_for_fw),
                "n_trades": int(m_sel.n_trades),
                "shape_tag": m_sel.shape_tag,
                "dip_p_value": _fmt_g(m_sel.dip_p_value),
                "dip_statistic": _fmt_g(m_sel.dip_statistic),
                "iqr_r": _fmt_g(m_sel.iqr_r),
                "mode_count": int(m_sel.mode_count),
                "mode_loc_1": _fmt_g(m_sel.mode_loc_1),
                "mode_loc_2": _fmt_g(m_sel.mode_loc_2),
                "mode_mass_1": _fmt_g(m_sel.mode_mass_1),
                "mode_mass_2": _fmt_g(m_sel.mode_mass_2),
                "min_mode_mass": _fmt_g(m_sel.min_mode_mass),
                "mode_separation_r": _fmt_g(m_sel.mode_separation_r),
                **{f"mass_{name}": _fmt_g(v) for name, v in mass_bands},
                "fwd_mfe_p5": _fmt_g(float(np.percentile(fwd, 5))) if fwd.size else "",
                "fwd_mfe_p25": _fmt_g(float(np.percentile(fwd, 25))) if fwd.size else "",
                "fwd_mfe_p50": _fmt_g(m_sel.fwd_mfe_p50),
                "fwd_mfe_p75": _fmt_g(m_sel.fwd_mfe_p75),
                "fwd_mfe_p95": _fmt_g(m_sel.fwd_mfe_p95),
                "fwd_mfe_max": _fmt_g(float(np.max(fwd))) if fwd.size else "",
            }
        ]
        dist_path = out_dir / f"distribution_cluster_{cid}.csv"
        _write_distribution(dist_rows, dist_path)
        sha_files[f"distribution_cluster_{cid}.csv"] = _file_sha256(dist_path)

        # pct_peak_and_collapse per-cluster CSV.
        dd = m_sel.drawdown_from_peak_arr
        peak = m_sel.peak_mfe_arr
        pc_rows: List[Dict[str, Any]] = []
        if dd.size > 0:
            pc_rows = [
                {
                    "cluster_id": cid,
                    "sl_atr_mult_used": _fmt_g(sl_for_fw),
                    "n_trades": int(dd.size),
                    "pct_peak_and_collapse": _fmt_g(m_sel.pct_peak_and_collapse),
                    "drawdown_from_peak_mean": _fmt_g(float(np.mean(dd))),
                    "drawdown_from_peak_p5": _fmt_g(float(np.percentile(dd, 5))),
                    "drawdown_from_peak_p25": _fmt_g(float(np.percentile(dd, 25))),
                    "drawdown_from_peak_p50": _fmt_g(float(np.percentile(dd, 50))),
                    "drawdown_from_peak_p75": _fmt_g(float(np.percentile(dd, 75))),
                    "drawdown_from_peak_p95": _fmt_g(float(np.percentile(dd, 95))),
                    "peak_mfe_p50": _fmt_g(float(np.percentile(peak, 50))),
                    "peak_mfe_p95": _fmt_g(float(np.percentile(peak, 95))),
                }
            ]
        pc_path = out_dir / f"pct_peak_and_collapse_cluster_{cid}.csv"
        _write_pct_pc(pc_rows, pc_path)
        sha_files[f"pct_peak_and_collapse_cluster_{cid}.csv"] = _file_sha256(pc_path)

    # archetype_summaries.csv
    summary_rows: List[Dict[str, Any]] = []
    for c in cluster_results:
        passed = c.selected_sl is not None
        sl_used = c.selected_sl if passed else float("nan")
        m = c.sweep[c.selected_sl] if passed else None
        summary_rows.append(
            {
                "cluster_id": c.cluster_id,
                "size": c.size,
                "size_fraction": _fmt_g(c.size_fraction),
                "selected_SL": _fmt_g(sl_used),
                "selection_reason": c.selection_reason,
                "mono_pre_peak_centroid": _fmt_g(m.mono_pre_peak_centroid) if m else "",
                "frac_wrong_way_pre_peak": _fmt_g(m.frac_wrong_way_pre_peak) if m else "",
                "fwd_mfe_h240_p50": _fmt_g(m.fwd_mfe_p50) if m else "",
                "frac_reach_1R": _fmt_g(m.frac_reach_1R) if m else "",
                "frac_reach_2R": _fmt_g(m.frac_reach_2R) if m else "",
                "pct_peak_and_collapse": _fmt_g(m.pct_peak_and_collapse) if m else "",
                "shape_tag": m.shape_tag if m else "",
                "capturability_composite": _fmt_g(m.capturability_composite) if m else "",
                "peak_mfe_atr_units": _fmt_g(m.peak_mfe_atr_units) if m else "",
                "archetype_label": c.archetype.archetype_label,
                "archetype_row": c.archetype.archetype_row,
                "candidate_rows": ";".join(str(r) for r in c.archetype.candidate_rows),
                "assignment_certainty": c.archetype.assignment_certainty,
                "bimodal_split_also_active": int(c.archetype.bimodal_split_also_active),
                "pre_peak_mono_at_selected": _fmt_g(m.mono_pre_peak_centroid) if m else _fmt_g(
                    c.sweep[2.0].mono_pre_peak_centroid
                ),
                "full_window_mono_at_selected": _fmt_g(c.full_window_mono_at_selected),
                "delta_pre_peak_vs_full_window": _fmt_g(c.pre_peak_minus_full_window_mono),
            }
        )
    summaries_path = out_dir / "archetype_summaries.csv"
    _write_archetype_summaries(summary_rows, summaries_path)
    sha_files["archetype_summaries.csv"] = _file_sha256(summaries_path)

    # capturability_pass_list.csv (survivors only)
    pass_rows: List[Dict[str, Any]] = []
    for c in cluster_results:
        if c.selected_sl is None:
            continue
        m = c.sweep[c.selected_sl]
        pass_rows.append(
            {
                "cluster_id": c.cluster_id,
                "selected_SL": _fmt_g(c.selected_sl),
                "archetype_label": c.archetype.archetype_label,
                "primary_row": c.archetype.archetype_row,
                "candidate_rows": ";".join(str(r) for r in c.archetype.candidate_rows),
                "bimodal_split_also_active": int(c.archetype.bimodal_split_also_active),
                "size": c.size,
                "size_fraction": _fmt_g(c.size_fraction),
                "mono_pre_peak": _fmt_g(m.mono_pre_peak_centroid),
                "fwd_mfe_p50": _fmt_g(m.fwd_mfe_p50),
                "frac_reach_1R": _fmt_g(m.frac_reach_1R),
                "frac_wrong_way_pre_peak": _fmt_g(m.frac_wrong_way_pre_peak),
                "shape_tag": m.shape_tag,
                "capturability_composite": _fmt_g(m.capturability_composite),
            }
        )
    pass_path = out_dir / "capturability_pass_list.csv"
    _write_capturability_pass_list(pass_rows, pass_path)
    sha_files["capturability_pass_list.csv"] = _file_sha256(pass_path)

    return cluster_results, sha_files, pool_size


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Arc 4 Step 3 capturability + SL sweep (L_ARC_PROTOCOL v2.1.1 §7)."
    )
    p.add_argument(
        "-c",
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs" / "l_arc_4.yaml",
    )
    p.add_argument(
        "--trades-csv",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--paths-csv",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--clusters-csv",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step2" / "clusters_K4.csv",
    )
    p.add_argument(
        "--centroids-csv",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step2" / "centroids_K4.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step3",
    )
    p.add_argument(
        "--no-determinism-check",
        action="store_true",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.config = args.config.resolve()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    step1_dir = _REPO_ROOT / cfg["output"]["results_dir"]
    trades_csv = (args.trades_csv or (step1_dir / cfg["output"]["trades_csv"])).resolve()
    paths_csv = (args.paths_csv or (step1_dir / cfg["output"]["paths_csv"])).resolve()
    clusters_csv = args.clusters_csv.resolve()
    centroids_csv = args.centroids_csv.resolve()
    out_dir = args.out_dir.resolve()

    print("[l_arc_4 step3] === RUN 1 ===", file=sys.stderr)
    cluster_results, sha_run1, pool_size = run_once(
        trades_csv, paths_csv, clusters_csv, centroids_csv, out_dir
    )

    sha_run2: Optional[Dict[str, str]] = None
    determinism_gate = "N/A"
    if not args.no_determinism_check:
        print("[l_arc_4 step3] === RUN 2 (determinism) ===", file=sys.stderr)
        _, sha_run2, _ = run_once(trades_csv, paths_csv, clusters_csv, centroids_csv, out_dir)
        matched = all(
            sha_run1.get(k) == sha_run2.get(k) for k in sorted(set(sha_run1) | set(sha_run2))
        )
        determinism_gate = "PASS" if matched else "FAIL"

    config_paths = {
        "configs/l_arc_4.yaml": str(args.config.relative_to(_REPO_ROOT)),
        f"{cfg['output']['results_dir']}/trades_all.csv": str(trades_csv.relative_to(_REPO_ROOT)),
        f"{cfg['output']['results_dir']}/trades_paths.csv": str(paths_csv.relative_to(_REPO_ROOT)),
        "results/l_arc_4/step2/clusters_K4.csv": str(clusters_csv.relative_to(_REPO_ROOT)),
        "results/l_arc_4/step2/centroids_K4.csv": str(centroids_csv.relative_to(_REPO_ROOT)),
    }
    config_shas = {
        label: _file_sha256(_REPO_ROOT / Path(path)) for label, path in config_paths.items()
    }

    diag_path = out_dir / "step3_diagnostics.md"
    arc_disp = write_diagnostics(
        diag_path,
        cluster_results,
        pool_size,
        sha_run1,
        sha_run2,
        determinism_gate,
        config_paths,
        config_shas,
    )

    survivors = [c for c in cluster_results if c.selected_sl is not None]
    print(
        f"[l_arc_4 step3] DONE pool={pool_size} clusters={len(cluster_results)} "
        f"survivors={len(survivors)} arc_disp={arc_disp} determinism={determinism_gate}",
        file=sys.stderr,
    )
    print(f"[l_arc_4 step3] diagnostics → {diag_path}", file=sys.stderr)

    (out_dir / "step3_env.json").write_text(
        json.dumps(
            {
                "env": _env_dict(),
                "args": {
                    "config": str(args.config),
                    "trades_csv": str(trades_csv),
                    "paths_csv": str(paths_csv),
                    "clusters_csv": str(clusters_csv),
                    "centroids_csv": str(centroids_csv),
                    "out_dir": str(out_dir),
                    "no_determinism_check": bool(args.no_determinism_check),
                },
                "arc_disposition": arc_disp,
                "determinism_gate": determinism_gate,
                "survivor_count": len(survivors),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    return 0 if (arc_disp == "PASS" and determinism_gate in ("PASS", "N/A")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
