"""Shape-tag detection for v2.1.1 §7.

Categorisation priority:
  no_magnitude → bimodal_separated → heavy_right_tail → tight_unimodal → scattered → unclassified

Compared to v2.0 (prior arc_2_redo step3), v2.1.1 replaces the heuristic "bimodal"
detector with a strict three-part conjunctive test:
  1. Hartigan dip-statistic p < 0.05 (via `diptest` Python lib)
  2. Min-mode-mass ≥ 0.20 (each detected mode carries ≥ 20% of mass)
  3. Mode separation ≥ 1.0 in candidate-SL R units

tight_unimodal / heavy_right_tail / scattered / no_magnitude thresholds inherit
verbatim from arc_2_redo step3 config for cross-comparability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ShapeTagDecision:
    tag: str
    detail: str
    # bimodal_separated sub-diagnostics (None if not applicable):
    dip_stat: float | None = None
    dip_p: float | None = None
    mode_locs: List[float] = field(default_factory=list)
    mode_masses: List[float] = field(default_factory=list)
    mode_separation: float | None = None
    bimodal_sep_admits: bool = False


def _detect_modes_from_histogram(
    fwd_mfe: np.ndarray, n_bins: int, upper_q: float
) -> tuple[List[float], List[float], List[int]]:
    """Returns (mode_centers, mode_masses, mode_bin_indices) of top-2 local-max
    peaks in the smoothed positive-MFE histogram. Used for separation / mass.

    Negatives clipped to 0; upper bound = max(p_upper_q, p75 × 1.5).
    """
    if fwd_mfe.size == 0:
        return [], [], []
    p75 = float(np.percentile(fwd_mfe, 75))
    upper = max(float(np.percentile(fwd_mfe, upper_q * 100)), p75 * 1.5)
    if upper <= 0:
        return [], [], []
    clipped = np.clip(fwd_mfe, 0.0, upper)
    hist, edges = np.histogram(clipped, bins=n_bins, range=(0.0, upper))
    centers = (edges[:-1] + edges[1:]) / 2
    n = fwd_mfe.size
    hist_mass = hist / max(n, 1)
    smooth = np.convolve(hist_mass, np.ones(3) / 3, mode="same")
    peaks: List[int] = []
    for i in range(1, n_bins - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]:
            peaks.append(i)
    peaks = [i for i in peaks if smooth[i] >= 0.02]
    # Sort by descending mass, keep top 2.
    peaks.sort(key=lambda i: -smooth[i])
    top2 = sorted(peaks[:2])

    if len(top2) < 2:
        return (
            [float(centers[i]) for i in top2],
            [float(smooth[i]) for i in top2],
            top2,
        )

    a, b = top2
    # Estimate mode mass: sum hist_mass over the bin range from valley-left to
    # valley-right of each peak. Simpler proxy: use ±2-bin neighborhood mass.
    # For separation test we use ±k neighborhood mass for each mode.
    def _mode_mass(peak_idx: int) -> float:
        lo = max(0, peak_idx - 2)
        hi = min(n_bins, peak_idx + 3)
        return float(hist_mass[lo:hi].sum())

    mass_a = _mode_mass(a)
    mass_b = _mode_mass(b)
    return [float(centers[a]), float(centers[b])], [mass_a, mass_b], [a, b]


def detect_shape_tag_v2_1_1(
    fwd_mfe: np.ndarray, cfg: dict
) -> ShapeTagDecision:
    """v2.1.1 §7 shape-tag detection. fwd_mfe in candidate-SL R units."""
    from diptest import diptest
    from scipy import stats as sps

    st = cfg["shape_tag"]
    n_bins = int(st["histogram_bins"])
    upper_q = float(st["histogram_range_quantile"])
    no_mag_p75_max = float(st["no_magnitude_p75_max"])
    iqr_over_range_max = float(st["tight_unimodal_iqr_over_range_max"])
    skew_min = float(st["heavy_right_tail_skew_min"])
    p95_p50_min = float(st["heavy_right_tail_p95_over_p50_min"])
    scattered_max_bin_mass = float(st["scattered_max_bin_mass_max"])
    dip_p_max = float(st["bimodal_dip_p_max"])
    min_mode_mass = float(st["bimodal_min_mode_mass"])
    mode_sep_min = float(st["bimodal_mode_separation_min_r"])

    if fwd_mfe.size == 0:
        return ShapeTagDecision("unclassified", "empty sample")

    p25, p50, p75, p95, p99 = np.percentile(fwd_mfe, [25, 50, 75, 95, 99])

    # 1) no_magnitude
    if p75 < no_mag_p75_max:
        return ShapeTagDecision(
            "no_magnitude", f"p75={p75:.4f} < {no_mag_p75_max:.4f}"
        )

    # 2) bimodal_separated — three-part conjunctive test
    # Run dip on the positive portion (consistent with mode detection range).
    upper = max(float(np.percentile(fwd_mfe, upper_q * 100)), p75 * 1.5)
    pos = np.clip(fwd_mfe, 0.0, upper)
    try:
        dip_stat, dip_p = diptest(pos)
    except Exception as e:
        dip_stat, dip_p = float("nan"), float("nan")
        bimodal_err = f"diptest failed: {e}"
    else:
        bimodal_err = ""

    mode_locs, mode_masses, _ = _detect_modes_from_histogram(fwd_mfe, n_bins, upper_q)
    sep = (
        abs(mode_locs[1] - mode_locs[0]) if len(mode_locs) == 2 else float("nan")
    )
    min_mass = min(mode_masses) if len(mode_masses) == 2 else float("nan")

    sub1 = (dip_p == dip_p) and (dip_p < dip_p_max)  # not NaN and below threshold
    sub2 = (len(mode_masses) == 2) and (min_mass >= min_mode_mass)
    sub3 = (len(mode_locs) == 2) and (sep >= mode_sep_min)

    if sub1 and sub2 and sub3:
        return ShapeTagDecision(
            "bimodal_separated",
            f"dip_p={dip_p:.4g}<{dip_p_max}, min_mode_mass={min_mass:.3f}>={min_mode_mass}, "
            f"sep={sep:.3f}R>={mode_sep_min}R; modes @ {mode_locs[0]:.2f}R, {mode_locs[1]:.2f}R",
            dip_stat=float(dip_stat) if dip_stat == dip_stat else None,
            dip_p=float(dip_p) if dip_p == dip_p else None,
            mode_locs=list(mode_locs),
            mode_masses=list(mode_masses),
            mode_separation=float(sep) if sep == sep else None,
            bimodal_sep_admits=True,
        )

    # Bimodal_separated diagnostics carried forward even if not admitted, for
    # downstream reporting.
    bimodal_diag = ShapeTagDecision(
        "",
        "",
        dip_stat=float(dip_stat) if dip_stat == dip_stat else None,
        dip_p=float(dip_p) if dip_p == dip_p else None,
        mode_locs=list(mode_locs),
        mode_masses=list(mode_masses),
        mode_separation=float(sep) if sep == sep else None,
        bimodal_sep_admits=False,
    )

    # 3) heavy_right_tail
    skew = float(sps.skew(fwd_mfe, bias=False, nan_policy="omit"))
    p95_over_p50 = p95 / p50 if p50 > 0 else float("inf")
    if skew >= skew_min and p95_over_p50 >= p95_p50_min:
        bimodal_diag.tag = "heavy_right_tail"
        bimodal_diag.detail = (
            f"skew={skew:.2f}>={skew_min}, p95/p50={p95_over_p50:.2f}>={p95_p50_min}"
        )
        return bimodal_diag

    # 4) tight_unimodal
    p1, p99 = np.percentile(fwd_mfe, [1, 99])
    rng = p99 - p1
    iqr = p75 - p25
    iqr_over_range = iqr / rng if rng > 0 else 0.0
    if iqr_over_range < iqr_over_range_max:
        bimodal_diag.tag = "tight_unimodal"
        bimodal_diag.detail = (
            f"IQR/range={iqr_over_range:.3f} < {iqr_over_range_max:.2f}"
        )
        return bimodal_diag

    # 5) scattered
    upper2 = max(float(np.percentile(fwd_mfe, upper_q * 100)), p75 * 1.5)
    if upper2 > 0:
        clipped = np.clip(fwd_mfe, 0.0, upper2)
        hist, _ = np.histogram(clipped, bins=n_bins, range=(0.0, upper2))
        hist_mass = hist / max(int(fwd_mfe.size), 1)
        max_mass = float(hist_mass.max())
    else:
        max_mass = 0.0

    if max_mass < scattered_max_bin_mass:
        bimodal_diag.tag = "scattered"
        bimodal_diag.detail = (
            f"max histogram bin mass {max_mass:.4f} < {scattered_max_bin_mass:.2f}"
        )
        return bimodal_diag

    # Default: unclassified
    bimodal_diag.tag = "unclassified"
    bimodal_diag.detail = (
        f"skew={skew:.2f}, IQR/range={iqr_over_range:.3f}, max_bin_mass={max_mass:.4f}"
    )
    return bimodal_diag
