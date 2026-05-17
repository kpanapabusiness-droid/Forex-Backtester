"""Open-18 Replay #2: KH-24 v2.0 c4 under L_ARC_PROTOCOL v2.1.1.

Step 3 capturability re-analysis on existing trades_paths.csv. SL sweep, pre-peak
metrics, MFE-based shape_tag with bimodal_separated test, capturability composite
selection. Dry-run mode evaluates one cluster, writes nothing to disk; default mode
writes per-cluster outputs to output_dir.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import gaussian_kde, mode
from diptest import diptest


# --- Path truncation under candidate SL -------------------------------------------------

def truncate_path(
    mae_so_far_r: np.ndarray,
    close_r_threshold: float,
) -> tuple[int, bool]:
    """Find SL-hit bar in path; return (truncation_idx, sl_hit).

    truncation_idx: index of last bar to keep (inclusive). If SL never hit, returns
    last index of input. If SL hit at bar B, returns B (SL-hit bar is included).
    sl_hit: True if first crossing of threshold was found within path.
    """
    if len(mae_so_far_r) == 0:
        return 0, False
    hits = np.where(mae_so_far_r <= close_r_threshold)[0]
    if len(hits) == 0:
        return len(mae_so_far_r) - 1, False
    return int(hits[0]), True


def peak_mfe_bar_in_truncated(mfe_so_far_r: np.ndarray) -> int:
    """Bar offset of max mfe_so_far_r within the truncated path. argmax returns first
    occurrence on ties, which is the correct behaviour for the earliest peak.
    """
    if len(mfe_so_far_r) == 0:
        return 0
    return int(np.argmax(mfe_so_far_r))


# --- Pre-peak metrics ------------------------------------------------------------------

def monotonicity_pre_peak(close_r_pre_peak: np.ndarray) -> float:
    """Among in-profit bars (close_r > 0) in 0..peak_mfe_bar, fraction with
    close_r >= previous in-profit bar. Returns 0.0 if zero in-profit bars (matches
    v2.0 monotonicity_ratio_in_profit edge case).
    """
    in_profit = close_r_pre_peak[close_r_pre_peak > 0]
    if len(in_profit) == 0:
        return 0.0
    if len(in_profit) == 1:
        return 1.0
    diffs = np.diff(in_profit)
    non_decreasing = int(np.sum(diffs >= 0))
    return non_decreasing / (len(in_profit) - 1)


def local_peaks_pre_peak(mfe_so_far_r_pre_peak: np.ndarray) -> int:
    """Count bars in 0..peak_mfe_bar where mfe_so_far_r strictly increased vs prior bar.
    """
    if len(mfe_so_far_r_pre_peak) < 2:
        return 0
    return int(np.sum(np.diff(mfe_so_far_r_pre_peak) > 0))


def wrong_way_pre_peak_hit(
    mae_so_far_r_pre_peak: np.ndarray,
    close_r_threshold: float,
) -> bool:
    """True if any bar in 0..peak_mfe_bar has mae_so_far_r <= -1R_candidate (i.e.
    <= candidate SL threshold). §17 Def C.
    """
    if len(mae_so_far_r_pre_peak) == 0:
        return False
    return bool(np.any(mae_so_far_r_pre_peak <= close_r_threshold))


# --- MFE-based shape_tag (v2.1.1) -----------------------------------------------------

@dataclass
class ShapeTagResult:
    tag: str
    p95_over_p50: float | None
    dip_p: float | None
    mode_locations: list[float] = field(default_factory=list)
    mode_masses: list[float] = field(default_factory=list)
    mode_separation: float | None = None


def _kde_modes(values: np.ndarray, n_grid: int = 512) -> tuple[list[float], list[float]]:
    """Find KDE modes (local maxima of KDE evaluated on a dense grid). Returns
    (mode_locations, mode_masses_within_half_distance_to_neighbour). For pre-mode-mass
    computation: each mode's mass is the fraction of points within half the distance
    to the nearest neighbouring mode. If only one mode, mass is 1.0.
    """
    if len(values) < 5 or np.std(values) == 0:
        # Degenerate; one mode at mean
        return [float(np.mean(values))], [1.0]

    kde = gaussian_kde(values)
    grid = np.linspace(values.min(), values.max(), n_grid)
    density = kde(grid)
    # Local maxima: strictly greater than both neighbours
    if len(density) < 3:
        return [float(grid[int(np.argmax(density))])], [1.0]
    is_peak = (density[1:-1] > density[:-2]) & (density[1:-1] > density[2:])
    peak_idx = np.where(is_peak)[0] + 1
    if len(peak_idx) == 0:
        # Edge-only maxima; pick global max
        return [float(grid[int(np.argmax(density))])], [1.0]
    mode_locs = [float(grid[i]) for i in peak_idx]
    mode_locs.sort()

    if len(mode_locs) == 1:
        return mode_locs, [1.0]

    masses: list[float] = []
    for i, loc in enumerate(mode_locs):
        # Determine half-distance window using nearest neighbour
        neighbours = [m for j, m in enumerate(mode_locs) if j != i]
        nearest_dist = min(abs(loc - m) for m in neighbours)
        half = nearest_dist / 2.0
        lo, hi = loc - half, loc + half
        mass = float(np.mean((values >= lo) & (values <= hi)))
        masses.append(mass)
    return mode_locs, masses


def classify_shape_tag(
    peak_mfe_distribution: np.ndarray,
    bimodal_cfg: dict,
) -> ShapeTagResult:
    """v2.1.1 shape_tag classifier on MFE distribution (not final_r — avoids c4's
    v2.0 censoring artifact).
    Categories: tight_unimodal | heavy_right_tail | bimodal_separated | scattered |
    unclassified.
    """
    n = len(peak_mfe_distribution)
    if n < 5:
        return ShapeTagResult(tag="unclassified", p95_over_p50=None, dip_p=None)

    vals = np.asarray(peak_mfe_distribution, dtype=float)
    p50 = float(np.percentile(vals, 50))
    p95 = float(np.percentile(vals, 95))
    ratio = (p95 / p50) if p50 > 0 else (math.inf if p95 > 0 else 0.0)

    # bimodal_separated test
    dip_p: float | None = None
    mode_locs: list[float] = []
    mode_masses: list[float] = []
    mode_sep: float | None = None
    is_bimodal_sep = False
    try:
        _, dip_p_val = diptest(vals)
        dip_p = float(dip_p_val)
    except Exception:
        dip_p = None

    if dip_p is not None and dip_p < bimodal_cfg["hartigan_dip_p_max"]:
        mode_locs, mode_masses = _kde_modes(vals)
        if len(mode_locs) >= 2:
            # Sort by location and take the two most-separated adjacent modes by mass
            # Operational interpretation: use min mass across all detected modes; use
            # max separation between any two modes (the "two structurally distinct modes"
            # the dip test is signalling).
            min_mass = float(min(mode_masses)) if mode_masses else 0.0
            mode_sep = float(max(mode_locs) - min(mode_locs))
            if (
                min_mass >= bimodal_cfg["min_mode_mass"]
                and mode_sep >= bimodal_cfg["min_mode_separation_r"]
            ):
                is_bimodal_sep = True

    if is_bimodal_sep:
        return ShapeTagResult(
            tag="bimodal_separated",
            p95_over_p50=ratio,
            dip_p=dip_p,
            mode_locations=mode_locs,
            mode_masses=mode_masses,
            mode_separation=mode_sep,
        )

    # Non-bimodal: distinguish tight_unimodal / heavy_right_tail / scattered
    # Use dip_p >= 0.05 to indicate "not significantly multimodal" (i.e. unimodal-ish).
    is_unimodal_likely = (dip_p is None) or (dip_p >= 0.05)
    std = float(np.std(vals))
    if is_unimodal_likely:
        if ratio <= 2.0:
            tag = "tight_unimodal"
        elif ratio > 3.0:
            tag = "heavy_right_tail"
        else:
            tag = "unclassified"
    else:
        # Dip suggests multimodality but bimodal_separated test failed (mode mass or
        # separation): scattered.
        tag = "scattered" if std > 1.0 else "unclassified"

    return ShapeTagResult(
        tag=tag,
        p95_over_p50=ratio,
        dip_p=dip_p,
        mode_locations=mode_locs,
        mode_masses=mode_masses,
        mode_separation=mode_sep,
    )


# --- Per-cluster SL sweep --------------------------------------------------------------

@dataclass
class TradeMetrics:
    trade_id: str
    truncation_idx: int
    sl_hit: bool
    peak_mfe_bar: int
    peak_mfe_r_candidate: float
    final_r_candidate: float
    mono_pre_peak: float
    local_peaks_pre_peak: int
    wrong_way_pre_peak: bool


@dataclass
class SLEvalResult:
    sl_atr: float
    threshold_r: float
    n_trades: int
    mono_pre_peak_centroid: float
    local_peaks_centroid: float
    fwd_mfe_h240_p25: float
    fwd_mfe_h240_p50: float
    fwd_mfe_h240_p75: float
    fwd_mfe_h240_p95: float
    frac_reach_1R: float
    frac_reach_2R: float
    frac_wrong_way_all_time: float
    frac_wrong_way_pre_peak: float
    cap_binding_rate: float
    final_r_mean: float
    final_r_t_stat: float
    final_r_p25: float
    final_r_p50: float
    final_r_p75: float
    size_count: int
    size_fraction_of_pool: float
    shape_tag: ShapeTagResult
    floors_passed: dict
    local_peaks_floor_status: str  # "PASS" | "FAIL" | "DEFERRED"
    composite: float | None
    peak_mfe_in_atr_units_mean: float


def evaluate_sl_for_cluster(
    cluster_paths: pd.DataFrame,
    sl_atr: float,
    atr_units_per_R_at_simulation: float,
    pool_size: int,
    floors: dict,
    bimodal_cfg: dict,
    local_peaks_relaxed_max: float,
) -> SLEvalResult:
    """Sweep one candidate SL across a cluster's paths."""
    close_r_threshold = -sl_atr / atr_units_per_R_at_simulation
    r_scale = atr_units_per_R_at_simulation / sl_atr  # convert close_r (stored R) to R_candidate

    trade_metrics: list[TradeMetrics] = []
    for trade_id, sub in cluster_paths.groupby("trade_id", sort=False):
        sub = sub.sort_values("bar_offset")
        close_r = sub["close_r"].to_numpy()
        mfe = sub["mfe_so_far_r"].to_numpy()
        mae = sub["mae_so_far_r"].to_numpy()

        trunc_idx, sl_hit = truncate_path(mae, close_r_threshold)
        close_r_t = close_r[: trunc_idx + 1]
        mfe_t = mfe[: trunc_idx + 1]
        mae_t = mae[: trunc_idx + 1]

        peak_bar = peak_mfe_bar_in_truncated(mfe_t)
        peak_mfe_r_cand = float(mfe_t[peak_bar] * r_scale) if len(mfe_t) else 0.0

        if sl_hit:
            final_r_cand = -1.0
        else:
            final_r_cand = float(close_r_t[-1] * r_scale) if len(close_r_t) else 0.0

        pre_peak_slice = slice(0, peak_bar + 1)
        mono = monotonicity_pre_peak(close_r_t[pre_peak_slice])
        lp = local_peaks_pre_peak(mfe_t[pre_peak_slice])
        ww = wrong_way_pre_peak_hit(mae_t[pre_peak_slice], close_r_threshold)

        trade_metrics.append(TradeMetrics(
            trade_id=str(trade_id),
            truncation_idx=trunc_idx,
            sl_hit=sl_hit,
            peak_mfe_bar=peak_bar,
            peak_mfe_r_candidate=peak_mfe_r_cand,
            final_r_candidate=final_r_cand,
            mono_pre_peak=mono,
            local_peaks_pre_peak=lp,
            wrong_way_pre_peak=ww,
        ))

    n = len(trade_metrics)
    if n == 0:
        raise ValueError("No trades in cluster slice")

    mono_centroid = float(np.mean([m.mono_pre_peak for m in trade_metrics]))
    lp_centroid = float(np.mean([m.local_peaks_pre_peak for m in trade_metrics]))
    peak_mfe_dist = np.array([m.peak_mfe_r_candidate for m in trade_metrics])
    final_r_dist = np.array([m.final_r_candidate for m in trade_metrics])

    p25, p50, p75, p95 = (float(np.percentile(peak_mfe_dist, q)) for q in (25, 50, 75, 95))
    frac_reach_1R = float(np.mean(peak_mfe_dist >= 1.0))
    frac_reach_2R = float(np.mean(peak_mfe_dist >= 2.0))
    cap_bind = float(np.mean([not m.sl_hit for m in trade_metrics]))
    # frac_wrong_way_all_time: anywhere on full (truncated) path the MAE hit threshold.
    # Equivalent to sl_hit for the truncated path since truncation = first hit. So this
    # is the same as the sl_hit rate; we report it for transparency.
    frac_ww_all = float(np.mean([m.sl_hit for m in trade_metrics]))
    frac_ww_pre = float(np.mean([m.wrong_way_pre_peak for m in trade_metrics]))

    final_r_mean = float(np.mean(final_r_dist))
    final_r_std = float(np.std(final_r_dist, ddof=1)) if n > 1 else 0.0
    t_stat = final_r_mean / (final_r_std / math.sqrt(n)) if final_r_std > 0 else 0.0
    fr_p25, fr_p50, fr_p75 = (float(np.percentile(final_r_dist, q)) for q in (25, 50, 75))

    peak_mfe_atr_mean = float(np.mean(peak_mfe_dist * sl_atr))  # back to ATR units

    shape = classify_shape_tag(peak_mfe_dist, bimodal_cfg)

    size_fraction = n / pool_size

    floors_passed = {
        "mono": mono_centroid >= floors["monotonicity_pre_peak_min"],
        "fwd_mfe_p50": p50 >= floors["fwd_mfe_h240_p50_min"],
        "frac_reach_1R": frac_reach_1R >= floors["frac_reach_1R_min"],
        "frac_wrong_way_pre_peak": frac_ww_pre <= floors["frac_wrong_way_pre_peak_max"],
        "shape_tag": shape.tag in floors["shape_tag_admitted"],
        "size_fraction": size_fraction >= floors["size_fraction_min"],
    }

    # local_peaks ceiling: defer for c4 (centroid 30.94 > Stepwise's 30). Report
    # counterfactual under relaxed ceiling.
    if lp_centroid <= 30.0:
        local_peaks_status = "PASS"
    elif lp_centroid <= local_peaks_relaxed_max:
        local_peaks_status = "DEFERRED"
    else:
        local_peaks_status = "FAIL"

    numerical_floors_all_pass = all(floors_passed.values())
    composite: float | None = None
    if numerical_floors_all_pass:
        composite = (
            (mono_centroid - floors["monotonicity_pre_peak_min"])
            + (frac_reach_1R - floors["frac_reach_1R_min"])
            + (floors["frac_wrong_way_pre_peak_max"] - frac_ww_pre)
        )

    return SLEvalResult(
        sl_atr=sl_atr,
        threshold_r=close_r_threshold,
        n_trades=n,
        mono_pre_peak_centroid=mono_centroid,
        local_peaks_centroid=lp_centroid,
        fwd_mfe_h240_p25=p25,
        fwd_mfe_h240_p50=p50,
        fwd_mfe_h240_p75=p75,
        fwd_mfe_h240_p95=p95,
        frac_reach_1R=frac_reach_1R,
        frac_reach_2R=frac_reach_2R,
        frac_wrong_way_all_time=frac_ww_all,
        frac_wrong_way_pre_peak=frac_ww_pre,
        cap_binding_rate=cap_bind,
        final_r_mean=final_r_mean,
        final_r_t_stat=t_stat,
        final_r_p25=fr_p25,
        final_r_p50=fr_p50,
        final_r_p75=fr_p75,
        size_count=n,
        size_fraction_of_pool=size_fraction,
        shape_tag=shape,
        floors_passed=floors_passed,
        local_peaks_floor_status=local_peaks_status,
        composite=composite,
        peak_mfe_in_atr_units_mean=peak_mfe_atr_mean,
    )


# --- Selection rule --------------------------------------------------------------------

def select_sl(
    sweep_results: list[SLEvalResult],
    tiebreaker_tolerance: float,
) -> tuple[SLEvalResult | None, str]:
    """Pick the composite-maximising SL among those that pass all numerical floors.
    Returns (selected_result_or_None, tiebreaker_note).
    """
    eligible = [r for r in sweep_results if r.composite is not None]
    if not eligible:
        return None, "no candidate passed all numerical floors"

    eligible_sorted = sorted(eligible, key=lambda r: r.composite, reverse=True)
    top = eligible_sorted[0]
    contenders = [r for r in eligible_sorted if (top.composite - r.composite) <= tiebreaker_tolerance]
    if len(contenders) == 1:
        return top, "no tiebreaker invoked"

    # Tiebreaker 1: higher peak_mfe in ATR units
    contenders.sort(key=lambda r: r.peak_mfe_in_atr_units_mean, reverse=True)
    best_atr = contenders[0].peak_mfe_in_atr_units_mean
    tb1_set = [r for r in contenders if abs(r.peak_mfe_in_atr_units_mean - best_atr) < 1e-9]
    if len(tb1_set) == 1:
        return tb1_set[0], f"tiebreaker 1 (peak_mfe_ATR) — {len(contenders)} within {tiebreaker_tolerance}"

    # Tiebreaker 2: smaller SL
    tb1_set.sort(key=lambda r: r.sl_atr)
    return tb1_set[0], "tiebreaker 2 (smaller SL)"


# --- Driver ----------------------------------------------------------------------------

def load_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repo_root = Path(__file__).resolve().parents[3]
    paths_cfg = cfg["inputs"]
    trades_all = pd.read_csv(repo_root / paths_cfg["trades_all"])
    trades_paths = pd.read_csv(repo_root / paths_cfg["trades_paths"])
    clusters = pd.read_csv(repo_root / paths_cfg["clusters"])
    centroids = pd.read_csv(repo_root / paths_cfg["centroids"])
    return trades_all, trades_paths, clusters, centroids


def render_dryrun_stdout(
    cluster_label: str,
    centroid_row: pd.Series,
    cluster_size: int,
    pool_size: int,
    sweep: list[SLEvalResult],
    selected: SLEvalResult | None,
    tiebreaker_note: str,
    v20_mono_full_window: float,
    v20_shape_tag: str,
    relaxed_max: float,
) -> None:
    v20_local_peaks = float(centroid_row["centroid_local_peaks"])
    print(f"=== Replay #2 — KH-24 v2.0 c4 under v2.1.1 (DRY-RUN) ===\n")
    print(f"Cluster: {cluster_label}")
    print(f"Pool: n={cluster_size}, size_fraction={cluster_size / pool_size:.3f}")
    print(
        f"v2.0 centroid: mono={centroid_row['centroid_monotonicity']:.3f}, "
        f"peaks={centroid_row['centroid_local_peaks']:.2f}, "
        f"pullback={centroid_row['centroid_pullback']:.3f}, "
        f"ttp_rel={centroid_row['centroid_time_to_peak_rel']:.3f}"
    )
    print(f"v2.0 label: unresolved_11fee4a0 (near stepwise)\n")

    print("--- SL sweep ---\n")
    for r in sweep:
        shape = r.shape_tag
        shape_extras = ""
        if shape.dip_p is not None:
            modes_str = (
                f", modes at {[round(x, 3) for x in shape.mode_locations]} "
                f"mass {[round(x, 3) for x in shape.mode_masses]}, "
                f"separation {shape.mode_separation:.2f}"
                if shape.mode_locations and shape.mode_separation is not None
                else ""
            )
            shape_extras = f"  (hartigan p={shape.dip_p:.3f}{modes_str})"
        floors_summary = (
            f"mono [{('PASS' if r.floors_passed['mono'] else 'FAIL')} margin "
            f"{r.mono_pre_peak_centroid - 0.55:+.3f}] | "
            f"mfe_p50 [{('PASS' if r.floors_passed['fwd_mfe_p50'] else 'FAIL')} {r.fwd_mfe_h240_p50:.2f}R] | "
            f"reach_1R [{('PASS' if r.floors_passed['frac_reach_1R'] else 'FAIL')} {r.frac_reach_1R:.3f}] | "
            f"wrong_way [{('PASS' if r.floors_passed['frac_wrong_way_pre_peak'] else 'FAIL')} {r.frac_wrong_way_pre_peak:.3f}] | "
            f"shape [{('PASS' if r.floors_passed['shape_tag'] else 'FAIL')} {shape.tag}] | "
            f"size [{('PASS' if r.floors_passed['size_fraction'] else 'FAIL')} {r.size_fraction_of_pool:.3f}] | "
            f"local_peaks [{r.local_peaks_floor_status} centroid {r.local_peaks_centroid:.2f}]"
        )
        comp_str = f"{r.composite:.3f}" if r.composite is not None else "N/A (numerical floor failed)"
        print(f"SL={r.sl_atr} × ATR (threshold close_r = {r.threshold_r:.3f}):")
        print(f"  mono_pre_peak_centroid: {r.mono_pre_peak_centroid:.3f}")
        print(f"  local_peaks_centroid: {r.local_peaks_centroid:.2f}")
        print(f"  fwd_mfe_h240_p50: {r.fwd_mfe_h240_p50:.2f} R_candidate")
        print(f"  fwd_mfe_h240_p95: {r.fwd_mfe_h240_p95:.2f} R_candidate")
        print(f"  frac_reach_1R: {r.frac_reach_1R:.3f}")
        print(f"  frac_reach_2R: {r.frac_reach_2R:.3f}")
        print(f"  frac_wrong_way_pre_peak: {r.frac_wrong_way_pre_peak:.3f}")
        print(f"  cap_binding_rate: {r.cap_binding_rate:.3f}")
        print(f"  shape_tag: {shape.tag}{shape_extras}")
        print(f"  floors: {floors_summary}")
        print(f"  composite: {comp_str}\n")

    print("--- Selection ---\n")
    if selected is None:
        print("Composite-maximising SL: NONE — no candidate passes all numerical floors.")
        print(f"Tiebreaker invoked: n/a — {tiebreaker_note}")
    else:
        print(
            f"Composite-maximising SL (excluding categorical local_peaks floor): "
            f"{selected.sl_atr} × ATR, composite {selected.composite:.3f}"
        )
        print(f"Tiebreaker invoked: {tiebreaker_note}\n")

    print("--- v2.0 vs v2.1.1 comparison (c4 only) ---\n")
    if selected is not None:
        delta = selected.mono_pre_peak_centroid - v20_mono_full_window
        print(f"mono full-window v2.0:    {v20_mono_full_window:.3f}")
        print(f"mono pre-peak v2.1.1:     {selected.mono_pre_peak_centroid:.3f}  (delta {delta:+.3f})\n")
        print(f"shape_tag v2.0 (final_r): {v20_shape_tag}")
        print(f"shape_tag v2.1.1 (MFE):   {selected.shape_tag.tag}\n")
    else:
        print(f"mono full-window v2.0:    {v20_mono_full_window:.3f}")
        print("mono pre-peak v2.1.1:     (no SL passed numerical floors; see per-SL above)\n")
        print(f"shape_tag v2.0 (final_r): {v20_shape_tag}")
        print("shape_tag v2.1.1 (MFE):   (per-SL, see sweep above)\n")

    print("--- Flags for chat ---\n")
    deferred_sls = [r for r in sweep if r.local_peaks_floor_status == "DEFERRED"]
    if deferred_sls:
        sl_list = ", ".join(f"{r.sl_atr}×" for r in deferred_sls)
        print(
            f"[ ] local_peaks ceiling: {cluster_label} v2.0 centroid {v20_local_peaks:.2f} "
            f"outside Stepwise 5-30 range. Counterfactual: under ceiling ≤ {relaxed_max:g}, "
            f"local_peaks floor DEFERRED for SLs: {sl_list}. Per-SL pre-peak local_peaks "
            f"centroids vary (see sweep above). Decision deferred per LIVE doc Issue #2."
        )
    else:
        print(
            f"[ ] local_peaks ceiling: {cluster_label} v2.0 centroid {v20_local_peaks:.2f} — "
            f"all SLs pass under relaxed ceiling ≤ {relaxed_max:g}."
        )
    nan_sweep = any(
        not math.isfinite(x)
        for r in sweep
        for x in (
            r.mono_pre_peak_centroid, r.local_peaks_centroid, r.fwd_mfe_h240_p50,
            r.frac_reach_1R, r.frac_wrong_way_pre_peak,
        )
    )
    if nan_sweep:
        print("[ ] WARNING: NaN/inf detected in sweep metrics — HALT and surface.")
    print()
    print("=== END DRY-RUN ===")


def write_cluster_outputs(
    output_dir: Path,
    cluster_label: str,
    sweep: list[SLEvalResult],
    selected: SLEvalResult | None,
    tiebreaker_note: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in sweep:
        rows.append({
            "cluster": cluster_label,
            "sl_atr": r.sl_atr,
            "n_trades": r.n_trades,
            "size_fraction": r.size_fraction_of_pool,
            "mono_pre_peak_centroid": r.mono_pre_peak_centroid,
            "local_peaks_centroid": r.local_peaks_centroid,
            "fwd_mfe_h240_p25": r.fwd_mfe_h240_p25,
            "fwd_mfe_h240_p50": r.fwd_mfe_h240_p50,
            "fwd_mfe_h240_p75": r.fwd_mfe_h240_p75,
            "fwd_mfe_h240_p95": r.fwd_mfe_h240_p95,
            "frac_reach_1R": r.frac_reach_1R,
            "frac_reach_2R": r.frac_reach_2R,
            "frac_wrong_way_pre_peak": r.frac_wrong_way_pre_peak,
            "frac_wrong_way_all_time": r.frac_wrong_way_all_time,
            "cap_binding_rate": r.cap_binding_rate,
            "final_r_mean": r.final_r_mean,
            "final_r_t_stat": r.final_r_t_stat,
            "final_r_p25": r.final_r_p25,
            "final_r_p50": r.final_r_p50,
            "final_r_p75": r.final_r_p75,
            "shape_tag": r.shape_tag.tag,
            "shape_dip_p": r.shape_tag.dip_p,
            "shape_p95_over_p50": r.shape_tag.p95_over_p50,
            "local_peaks_status": r.local_peaks_floor_status,
            "composite": r.composite,
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"{cluster_label}_sl_sweep.csv", index=False)

    selection_row = {
        "cluster": cluster_label,
        "selected_sl_atr": selected.sl_atr if selected else None,
        "selected_composite": selected.composite if selected else None,
        "tiebreaker_note": tiebreaker_note,
    }
    pd.DataFrame([selection_row]).to_csv(output_dir / f"{cluster_label}_selected_sl.csv", index=False)


def run(cfg_path: Path, dry_run: bool, target_cluster: str | None) -> int:
    cfg = yaml.safe_load(cfg_path.read_text())
    trades_all, trades_paths, clusters, centroids = load_inputs(cfg)

    if cfg["atr_column"] not in trades_all.columns:
        print(
            f"ERROR: configured atr_column '{cfg['atr_column']}' not in trades_all.csv "
            f"header. Available: {list(trades_all.columns)}",
            file=sys.stderr,
        )
        return 2

    pool_size = trades_all["trade_id"].nunique()
    sl_candidates = cfg["sl_sweep_candidates_atr"]
    floors = cfg["floors"]
    bimodal_cfg = cfg["bimodal_separated_test"]
    relaxed_max = cfg.get("local_peaks_stepwise_relaxed_max", 35.0)
    tb_tol = cfg["composite_tiebreaker_tolerance"]
    output_dir = Path(__file__).resolve().parents[3] / cfg["output_dir"]

    cluster_labels = sorted(clusters["cluster_id"].unique())
    label_map = {cid: f"c{cid}" for cid in cluster_labels}

    if target_cluster is not None:
        match = [cid for cid, lbl in label_map.items() if lbl == target_cluster]
        if not match:
            print(f"ERROR: cluster '{target_cluster}' not in clusters file. Have: "
                  f"{list(label_map.values())}", file=sys.stderr)
            return 2
        run_ids = match
    else:
        run_ids = list(cluster_labels)

    for cid in run_ids:
        cluster_label = label_map[cid]
        cluster_trade_ids = set(clusters.loc[clusters["cluster_id"] == cid, "trade_id"])
        cluster_paths = trades_paths[trades_paths["trade_id"].isin(cluster_trade_ids)]
        centroid_row = centroids.loc[centroids["cluster_id"] == cid].iloc[0]
        cluster_size = len(cluster_trade_ids)

        sweep_results: list[SLEvalResult] = []
        for sl in sl_candidates:
            res = evaluate_sl_for_cluster(
                cluster_paths=cluster_paths,
                sl_atr=float(sl),
                atr_units_per_R_at_simulation=float(cfg["atr_units_per_R_at_simulation"]),
                pool_size=pool_size,
                floors=floors,
                bimodal_cfg=bimodal_cfg,
                local_peaks_relaxed_max=float(relaxed_max),
            )
            sweep_results.append(res)

        selected, tb_note = select_sl(sweep_results, tb_tol)

        if dry_run:
            # KH-24 v2.0 c4 final-r shape was "scattered"; full-window mono 0.530
            render_dryrun_stdout(
                cluster_label=cluster_label,
                centroid_row=centroid_row,
                cluster_size=cluster_size,
                pool_size=pool_size,
                sweep=sweep_results,
                selected=selected,
                tiebreaker_note=tb_note,
                v20_mono_full_window=0.530,
                v20_shape_tag="scattered",
                relaxed_max=float(relaxed_max),
            )
        else:
            write_cluster_outputs(output_dir, cluster_label, sweep_results, selected, tb_note)
            print(f"Wrote: {output_dir}/{cluster_label}_sl_sweep.csv (selected: "
                  f"{selected.sl_atr if selected else 'NONE'})")

    return 0


def main() -> int:
    # Force UTF-8 stdout so non-ASCII chars in dryrun output (×, ≤, etc.) survive
    # Windows cp1252 default.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print to stdout, write nothing to disk.")
    parser.add_argument("--cluster", type=str, default=None,
                        help="Restrict to one cluster label (e.g. 'c4').")
    args = parser.parse_args()
    return run(args.config, args.dry_run, args.cluster)


if __name__ == "__main__":
    raise SystemExit(main())
