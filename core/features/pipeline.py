"""Feature pipeline orchestrator.

Computes the full Step-1 feature matrix for a (pair, pair_df) by walking
every registered ``FeatureSpec`` and concatenating the producer outputs
into a single DataFrame. A sidecar lineage table records the
``causal_lineage`` tag per feature so Step 6 can drive its audit.

Public:

    compute_feature_matrix(pair, pair_df, panel=None, names=None) -> (matrix, lineage_df)
    feature_lineage_dataframe() -> pd.DataFrame   # all-features lineage

The orchestrator imports every feature class module to populate the
registry — callers don't need to import them individually.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Triggers feature registration via side-effect imports — keep this list
# in sync with whatever classes exist under core.features.
from core.features import (  # noqa: F401  (registration side-effects)
    cross_pair,
    distance,
    multi_tf,
    price_geometry,
    session,
    spread_regime,
    vol_regime,
)
from core.features.lineage import FeatureSpec
from core.features.registry import all_specs, get


@dataclass(frozen=True)
class FeatureMatrix:
    """Result of ``compute_feature_matrix``.

    ``matrix`` is a DataFrame indexed by pair_df.index with one column per
    requested feature, sorted alphabetically. ``lineage`` is a small
    DataFrame describing each column's lineage / class / inputs — Step 6
    consumes this directly.
    """

    matrix: pd.DataFrame
    lineage: pd.DataFrame


def feature_lineage_dataframe(names: list[str] | None = None) -> pd.DataFrame:
    """Return one row per registered (or named) feature with lineage info."""
    if names is None:
        specs = all_specs()
    else:
        specs = tuple(get(n) for n in sorted(set(names)))
    rows = [
        {
            "name": s.name,
            "feature_class": s.feature_class,
            "lineage": s.lineage.value,
            "needs_panel": s.needs_panel,
            "description": s.description,
        }
        for s in specs
    ]
    df = pd.DataFrame(
        rows, columns=["name", "feature_class", "lineage", "needs_panel", "description"]
    )
    return df.sort_values("name").reset_index(drop=True)


def compute_feature_matrix(
    pair: str,
    pair_df: pd.DataFrame,
    panel=None,
    names: list[str] | None = None,
) -> FeatureMatrix:
    """Compute the requested feature columns for ``pair_df``.

    Parameters
    ----------
    pair : str
        Pair name. Stored in ``pair_df.attrs['pair']`` so producers that
        need pip-size or pair-identity can look it up.
    pair_df : pd.DataFrame
        UTC-indexed bid+ask OHLC frame (schema from
        ``core.data.histdata_loader.M1_COLUMNS`` or an aggregated TF).
    panel : optional
        The v3 ``Panel`` from ``core.sim.panel``. Required for any
        feature with ``spec.needs_panel = True``; if absent, those
        features emit NaN columns and are flagged in the lineage report.
    names : list[str] | None
        Optional subset of feature names. Default: all registered.

    Returns
    -------
    FeatureMatrix
        ``.matrix`` — DataFrame indexed by pair_df.index, columns sorted
        alphabetically by feature name (deterministic).
        ``.lineage`` — small DataFrame with one row per included feature.
    """
    pair_df = pair_df.copy()
    pair_df.attrs["pair"] = pair

    if names is None:
        specs: tuple[FeatureSpec, ...] = all_specs()
    else:
        specs = tuple(get(n) for n in sorted(set(names)))

    columns: dict[str, pd.Series] = {}
    for spec in specs:
        if spec.needs_panel and panel is None:
            # Emit NaN column rather than crash — Step 6 reads lineage
            # and surfaces the missing-panel case as suspect.
            columns[spec.name] = pd.Series(float("nan"), index=pair_df.index, name=spec.name)
            continue
        out = spec.producer(pair_df, panel=panel)
        if not isinstance(out, pd.Series):
            raise TypeError(f"Feature {spec.name!r} returned non-Series: {type(out)}")
        # Align by index and rename — defensive against producer returning a Series with mismatched index
        out = out.reindex(pair_df.index)
        columns[spec.name] = out.astype("float64", errors="ignore").rename(spec.name)

    matrix = pd.DataFrame(columns)[sorted(columns)]
    matrix.index = pair_df.index
    lineage_df = feature_lineage_dataframe([s.name for s in specs])
    return FeatureMatrix(matrix=matrix, lineage=lineage_df)
