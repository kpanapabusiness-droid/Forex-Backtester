"""Distribution-shape (shape_tag) classifier per §7.

shape_tag ∈ {no_magnitude, tight_unimodal, heavy_right_tail, bimodal,
              scattered, unclassified}.

Mode detection: gaussian KDE (Scott's bandwidth) evaluated on a fixed
200-point grid, then `scipy.signal.find_peaks` with prominence
threshold = 10% of the global max density. Both are deterministic for
fixed inputs.

Classification priority:
  1. no_magnitude       — p95(final_r) < 0.5  (no meaningful win tail)
  2. bimodal            — >= 2 modes with at least one pair separated by >= 1R
                          AND secondary mode mass >= 30% of primary mode mass
  3. heavy_right_tail   — secondary mode below the 30% threshold (i.e.,
                          dominant single mode + right-tail bumps); OR a
                          single mode with skew > 1.0
  4. scattered          — std > 2.0 and no clean unimodal/bimodal structure
  5. tight_unimodal     — single mode, std < 1.0, |skew| <= 1.0
  6. unclassified       — otherwise

Note: §2's "Internal consistency" criterion requires shape_tag ∈
{tight_unimodal, heavy_right_tail}. §14 records KH-24 K=4 archetype 3 as
"bimodal (right-mode dominates)" — but it still passes §2 per §14's
narrative. The "right-mode dominates" qualifier is what this classifier
captures via the 30% secondary-mass threshold: a tall primary mode plus a
much smaller right-tail bump is classified as `heavy_right_tail` (passes
§2), not `bimodal` (fails §2). True bimodal (both modes >= 30% mass)
fails §2 and would route to the §11 Split-exit variant.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, skew

KDE_GRID_POINTS = 200
PROMINENCE_FACTOR = 0.10
BIMODAL_SEPARATION_R = 1.0
BIMODAL_SECONDARY_MASS_FRACTION = 0.30


class ShapeResult(NamedTuple):
    tag: str
    stats: dict


def _mode_density_at(kde: gaussian_kde, x: float) -> float:
    return float(kde(np.array([x]))[0])


def _detect_modes(values: np.ndarray) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Return (mode positions, KDE grid, KDE density on grid).

    Returns ([], [], []) when there aren't enough points for a stable KDE.
    """
    if len(values) < 5:
        return [], np.array([]), np.array([])
    if float(values.std(ddof=1)) == 0.0:
        # KDE is undefined for zero-variance data.
        return [], np.array([]), np.array([])
    kde = gaussian_kde(values)
    grid = np.linspace(values.min(), values.max(), KDE_GRID_POINTS)
    density = kde(grid)
    if density.max() == 0:
        return [], grid, density
    prominence = PROMINENCE_FACTOR * float(density.max())
    peak_idx, _ = find_peaks(density, prominence=prominence)
    return [float(grid[i]) for i in peak_idx], grid, density


def classify_shape(final_r: np.ndarray) -> ShapeResult:
    """Classify the per-trade final_r distribution into a §7 shape_tag.

    Returns ShapeResult with the tag and a stats dict that records every
    numeric input used in the decision so the call is auditable.
    """
    n = len(final_r)
    if n == 0:
        return ShapeResult("unclassified", {"n": 0})

    mean = float(np.mean(final_r))
    std = float(np.std(final_r, ddof=1)) if n > 1 else 0.0
    sk = float(skew(final_r, bias=False)) if n > 2 else 0.0
    p95 = float(np.percentile(final_r, 95))

    modes, grid, density = _detect_modes(final_r)
    n_modes = len(modes)

    # Compute secondary-mode mass ratio when there are at least two modes.
    secondary_ratio: float | None = None
    max_pair_separation = 0.0
    if n_modes >= 2 and density.size:
        # Density at each mode position is the local maximum value.
        mode_densities = []
        for m in modes:
            # Find nearest grid index
            idx = int(np.argmin(np.abs(grid - m)))
            mode_densities.append(float(density[idx]))
        sorted_md = sorted(mode_densities, reverse=True)
        if sorted_md[0] > 0:
            secondary_ratio = sorted_md[1] / sorted_md[0]
        sorted_modes = sorted(modes)
        for i in range(len(sorted_modes) - 1):
            sep = sorted_modes[i + 1] - sorted_modes[i]
            if sep > max_pair_separation:
                max_pair_separation = sep

    stats = {
        "n": n,
        "mean": mean,
        "std": std,
        "skew": sk,
        "p95": p95,
        "n_modes_detected": n_modes,
        "modes": modes,
        "secondary_to_primary_mass_ratio": secondary_ratio,
        "max_pair_separation": max_pair_separation,
    }

    # Rule 1: no magnitude
    if p95 < 0.5:
        return ShapeResult("no_magnitude", stats)

    # Rule 2/3: multi-mode handling
    if n_modes >= 2 and max_pair_separation >= BIMODAL_SEPARATION_R:
        # If the secondary mode carries >= 30% of primary mass, it's true bimodal.
        if secondary_ratio is not None and secondary_ratio >= BIMODAL_SECONDARY_MASS_FRACTION:
            return ShapeResult("bimodal", stats)
        # Otherwise: dominant primary mode with smaller right/left-tail bumps.
        # If the secondary is to the right of the primary (typical right-tail
        # shape), classify as heavy_right_tail.
        sorted_md_pairs = sorted(
            zip(modes, [float(density[int(np.argmin(np.abs(grid - m)))]) for m in modes]),
            key=lambda kv: -kv[1],
        )
        primary_pos = sorted_md_pairs[0][0]
        secondary_pos = sorted_md_pairs[1][0]
        if secondary_pos > primary_pos:
            return ShapeResult("heavy_right_tail", stats)
        # Secondary is to the LEFT — still effectively a single dominant mode
        # with a left-skewed shoulder; fall through to single-mode logic.

    # Rule 4/5/6: single-mode (or treated-as-single) classification
    if n_modes <= 1 or (
        n_modes >= 2
        and (secondary_ratio is None or secondary_ratio < BIMODAL_SECONDARY_MASS_FRACTION)
    ):
        if sk > 1.0:
            return ShapeResult("heavy_right_tail", stats)
        if std < 1.0 and abs(sk) <= 1.0:
            return ShapeResult("tight_unimodal", stats)
        if std > 2.0:
            return ShapeResult("scattered", stats)
        return ShapeResult("unclassified", stats)

    return ShapeResult("unclassified", stats)
