"""Per-feature quantile threshold grid.

The grammar's atoms compare a feature against a quantile (e.g.
``atr_14 > p75``). At evaluation time we need the concrete numeric
threshold corresponding to each (feature, quantile) pair. The
``QuantileGrid`` computes these once from a training-window feature
matrix, then caches them for the search.

Pooling discipline:

  * Quantiles are computed from the POOLED feature distribution across
    all 28 pairs within the training window. This is the same pool the
    rule search evaluates against, so quantile thresholds and trigger
    rates are mutually consistent.
  * NaN values are excluded from quantile computation (a feature whose
    rolling window hasn't filled yet should not depress the p10).
  * Quantiles are reported via ``numpy.nanquantile`` with
    ``method='linear'`` (default) for cross-platform reproducibility.

Determinism:

  * Same input matrix + same quantile grid -> byte-identical thresholds.
  * The matrix concatenation order is sorted by pair name; per-pair
    rows preserve their input order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuantileGrid:
    """Per-(feature, quantile) numeric threshold table.

    ``thresholds[feature][quantile]`` -> float threshold value, or NaN
    if the feature's column had < ``min_non_nan`` non-null observations
    (such features are unusable and should have been filtered out
    upstream — they're flagged here for diagnostic completeness).
    """

    quantiles: tuple[float, ...]
    thresholds: Mapping[str, Mapping[float, float]]
    n_observations: Mapping[str, int]   # non-NaN count per feature

    def get(self, feature: str, quantile: float) -> float:
        """Look up the threshold for (feature, quantile). KeyError if unknown."""
        col = self.thresholds[feature]
        if quantile not in col:
            raise KeyError(
                f"quantile {quantile!r} not in grid for feature {feature!r} "
                f"(grid has: {sorted(col)})"
            )
        return col[quantile]

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(sorted(self.thresholds.keys()))

    def features_with_data(self, min_non_nan: int = 100) -> tuple[str, ...]:
        """Features whose non-NaN count meets the threshold."""
        return tuple(
            sorted(f for f, n in self.n_observations.items() if n >= min_non_nan)
        )


def build_quantile_grid(
    feature_matrices: Mapping[str, pd.DataFrame],
    quantiles: Iterable[float],
    feature_names: Iterable[str] | None = None,
) -> QuantileGrid:
    """Compute the quantile grid from a per-pair feature matrix dict.

    Parameters
    ----------
    feature_matrices
        ``{pair: matrix}`` — each matrix has features as columns. The
        index can differ per pair (different timestamps); we pool by
        column across pairs.
    quantiles
        Quantile points to compute (e.g. ``(0.10, 0.25, 0.50, 0.75, 0.90)``).
    feature_names
        Optional restriction to a subset of columns. Default: union
        across all pairs.

    Returns
    -------
    :class:`QuantileGrid` with thresholds per (feature, quantile).
    """
    if not feature_matrices:
        raise ValueError("feature_matrices is empty")
    q_tuple = tuple(quantiles)
    for q in q_tuple:
        if not (0.0 < q < 1.0):
            raise ValueError(f"quantile {q} must be in (0, 1)")

    # Union of feature names across pairs — deterministic ordering.
    if feature_names is None:
        names: set[str] = set()
        for m in feature_matrices.values():
            names.update(m.columns)
        feature_names_sorted = tuple(sorted(names))
    else:
        feature_names_sorted = tuple(sorted(set(feature_names)))

    # Pair iteration order is sorted to remove dict-order dependence.
    pairs_sorted = tuple(sorted(feature_matrices.keys()))

    thresholds: dict[str, dict[float, float]] = {}
    n_obs: dict[str, int] = {}

    for feat in feature_names_sorted:
        # Concatenate this feature's column across pairs (sorted order).
        pieces: list[np.ndarray] = []
        for p in pairs_sorted:
            m = feature_matrices[p]
            if feat not in m.columns:
                continue
            col = m[feat].to_numpy(dtype="float64", copy=False)
            pieces.append(col)
        if not pieces:
            thresholds[feat] = {q: float("nan") for q in q_tuple}
            n_obs[feat] = 0
            continue
        joined = np.concatenate(pieces)
        finite = joined[np.isfinite(joined)]
        n_obs[feat] = int(finite.size)
        if finite.size == 0:
            thresholds[feat] = {q: float("nan") for q in q_tuple}
            continue
        # numpy.quantile is deterministic given finite, fixed-order input.
        qs = np.quantile(finite, q_tuple, method="linear")
        thresholds[feat] = {q: float(qs[i]) for i, q in enumerate(q_tuple)}

    return QuantileGrid(
        quantiles=q_tuple,
        thresholds=thresholds,
        n_observations=n_obs,
    )


__all__ = ("QuantileGrid", "build_quantile_grid")
