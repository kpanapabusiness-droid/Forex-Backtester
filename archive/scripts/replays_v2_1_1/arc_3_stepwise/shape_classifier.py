"""Arc 3 v2.0 shape_tag classifier — lifted verbatim from
scripts/arc_3/step3_capturability.py (function `_classify_shape_tag` and helpers).

DO NOT EDIT the algorithmic logic. Lifted to a separate module so this Open-18
replay can reuse the byte-equivalent v2.0 classifier without taking a dependency
on the Arc 3 closure script (which is historical and read-only).

The v2.1.1 bimodal_separated test is applied as a layer ON TOP of this classifier
in step3.py — this module is the v2.0 baseline only.

Returns one of:
  no_magnitude, tight_unimodal, heavy_right_tail, bimodal, scattered, unclassified.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.stats as sps


def _pct(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def classify_shape_tag(final_r: np.ndarray) -> Tuple[str, Dict[str, Any]]:
    """Return (shape_tag, decision_log).

    Document-order check: no_magnitude -> tight_unimodal -> heavy_right_tail
    -> bimodal -> scattered -> unclassified.
    """
    log: Dict[str, Any] = {}
    if final_r.size < 3:
        log["reason"] = "insufficient_sample"
        return "unclassified", log

    p25, p50, p75, p95 = (
        _pct(final_r, 25),
        _pct(final_r, 50),
        _pct(final_r, 75),
        _pct(final_r, 95),
    )
    iqr = p75 - p25
    skew = float(sps.skew(final_r, bias=False))
    std = float(final_r.std(ddof=1))
    log.update(
        {
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p95": p95,
            "iqr": iqr,
            "skew": skew,
            "std": std,
        }
    )

    # 1. no_magnitude: p75 <= 0.5R.
    if p75 <= 0.5:
        log["no_magnitude_p75_le_0.5R"] = True
        return "no_magnitude", log

    # 2. tight_unimodal: IQR <= 1.5R AND |skew| <= 1.0.
    if iqr <= 1.5 and abs(skew) <= 1.0:
        log["tight_unimodal_iqr_le_1.5_skew_abs_le_1.0"] = True
        return "tight_unimodal", log

    # 3. heavy_right_tail: p95/p75 >= 2.5 AND skew >= 1.0.
    if p75 > 0:
        ratio_95_75 = p95 / p75
        log["p95_over_p75"] = float(ratio_95_75)
        if ratio_95_75 >= 2.5 and skew >= 1.0:
            log["heavy_right_tail_ratio_ge_2.5_skew_ge_1.0"] = True
            return "heavy_right_tail", log

    # 4. bimodal: KDE valley test (v2.0 only used KDE — Hartigan dip is v2.1.1
    # additional gate, applied as overlay in step3.py).
    try:
        kde = sps.gaussian_kde(final_r)
        x_grid = np.linspace(final_r.min(), final_r.max(), 512)
        dens = kde(x_grid)
        peaks: List[int] = []
        for i in range(1, len(dens) - 1):
            if dens[i] > dens[i - 1] and dens[i] > dens[i + 1]:
                peaks.append(i)
        log["kde_peak_count"] = len(peaks)
        if len(peaks) >= 2:
            peaks_sorted = sorted(peaks, key=lambda i: -dens[i])[:2]
            i1, i2 = sorted(peaks_sorted)
            valley_dens = float(np.min(dens[i1 : i2 + 1]))
            mode_dens_min = float(min(dens[i1], dens[i2]))
            mode_separation = abs(float(x_grid[i2]) - float(x_grid[i1]))
            valley_depth_pct = (
                (mode_dens_min - valley_dens) / mode_dens_min if mode_dens_min > 0 else 0.0
            )
            log["mode_separation_R"] = mode_separation
            log["valley_depth_pct"] = float(valley_depth_pct)
            if mode_separation >= 0.5 and valley_depth_pct >= 0.10:
                log["bimodal_decision"] = "fire"
                return "bimodal", log
            else:
                log["bimodal_decision"] = "separation_or_valley_insufficient"
    except Exception as e:
        log["kde_error"] = str(e)

    # 5. scattered: std >= 2.5R.
    if std >= 2.5:
        log["scattered_std_ge_2.5"] = True
        return "scattered", log

    log["fall_through"] = True
    return "unclassified", log
