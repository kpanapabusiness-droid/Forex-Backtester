"""Pre-evaluation causal-lineage gate for heavy_ml_probe.

Per ``docs/sub_protocols/heavy_ml_probe.md`` v1.0 §"Per-feature causal
lineage" + dispatch §6 #2: every feature consumed by AutoML must carry
a ``causal_lineage == "clean"`` tag from Step 1 (set at ``FeatureSpec``
registration time, see ``core/features/lineage.py``). Features tagged
``suspect`` or ``unverified`` are excluded from the training set
*before* the AutoML run starts — not flagged after.

This module mirrors ``core/discovery/causal_filter.py`` so the same
filtering vocabulary appears across sub-protocols. The lineage source
of truth is ``core.features.pipeline.feature_lineage_dataframe()``;
this gate calls into it (or accepts the DataFrame directly for
testability) and returns:

  * the cleaned column list to pass to AutoML
  * a rejection log (per-feature reason) for the manifest + audit trail

Q3 resolution from build intent: lineage tags live on FeatureSpec, NOT
embedded per-row in Step 1's pool parquet. Heavy ML's training matrix is
columns × trades; the gate filters at column-level only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

# Lineage values that are accepted into training by default. Per spec
# §"Per-feature causal lineage" only ``clean`` features pass. Tests may
# override (e.g. to assert a hypothetical multi-tag policy) but
# production callers should not.
DEFAULT_ACCEPTED_LINEAGE: tuple[str, ...] = ("clean",)

# Reason strings recorded in the rejection log. Stable + grep-able.
REASON_NON_CLEAN_LINEAGE: str = "non_clean_lineage"
REASON_EXCLUDED_CLASS: str = "excluded_feature_class"
REASON_UNKNOWN_FEATURE: str = "unknown_feature"
REASON_MISSING_FROM_MATRIX: str = "missing_from_training_matrix"


@dataclass(frozen=True)
class LineageGateResult:
    """Outcome of running the pre-evaluation lineage gate.

    Attributes
    ----------
    accepted_features
        Sorted tuple of feature names cleared for training.
    rejected
        List of ``{"feature": <name>, "reason": <code>, "detail":
        <str>}`` dicts in feature-name order. Empty when nothing was
        rejected.
    n_input_columns
        Count of columns presented to the gate (post lineage_df join
        but before filtering). Drives the rejection-fraction metric in
        the manifest.
    """

    accepted_features: tuple[str, ...]
    rejected: tuple[dict, ...]
    n_input_columns: int

    @property
    def n_accepted(self) -> int:
        return len(self.accepted_features)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)


def _normalise_lineage_df(lineage_df: pd.DataFrame) -> pd.DataFrame:
    """Light shape-check on the lineage DataFrame.

    Required columns: ``name``, ``causal_lineage``, ``feature_class``.
    These match ``core.features.pipeline.feature_lineage_dataframe()``'s
    output exactly. Other columns are ignored.

    Raises ``ValueError`` with a precise diagnostic on missing columns;
    callers passing a hand-built DataFrame in tests get a clear error
    rather than a downstream KeyError.
    """
    required = {"name", "causal_lineage", "feature_class"}
    missing = required - set(lineage_df.columns)
    if missing:
        raise ValueError(
            f"lineage_df missing required columns: {sorted(missing)}; "
            f"present: {sorted(lineage_df.columns)}"
        )
    return lineage_df


def filter_training_columns(
    training_columns: Iterable[str],
    lineage_df: pd.DataFrame,
    *,
    accepted_lineage: Iterable[str] = DEFAULT_ACCEPTED_LINEAGE,
    exclude_classes: Iterable[str] = (),
) -> LineageGateResult:
    """Pre-evaluation gate. Returns the column subset cleared for training.

    Parameters
    ----------
    training_columns
        Iterable of feature column names present in the training matrix.
    lineage_df
        Lineage table. Must have columns ``name``, ``causal_lineage``,
        ``feature_class`` (see ``core.features.pipeline``).
    accepted_lineage
        Lineage values that pass the gate. Default ``("clean",)`` per
        spec §"Per-feature causal lineage".
    exclude_classes
        Optional feature_class values to exclude regardless of lineage
        (e.g. ``("spread_regime",)`` if a class is known leaky).
    """
    lineage_df = _normalise_lineage_df(lineage_df)
    accepted_set = {s.lower() for s in accepted_lineage}
    exclude_set = {s.lower() for s in exclude_classes}

    cols = sorted(set(training_columns))
    by_name = lineage_df.set_index("name", drop=False)
    accepted: list[str] = []
    rejected: list[dict] = []

    for col in cols:
        if col not in by_name.index:
            rejected.append({
                "feature": col,
                "reason": REASON_UNKNOWN_FEATURE,
                "detail": "no lineage row registered for this column",
            })
            continue
        row = by_name.loc[col]
        # by_name.loc returns a Series for unique index hits and a
        # DataFrame for duplicates. The registry guarantees unique
        # names, but for defensive parsing we always take the first row.
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        lineage_val = str(row["causal_lineage"]).lower()
        feat_class = str(row["feature_class"]).lower()
        if feat_class in exclude_set:
            rejected.append({
                "feature": col,
                "reason": REASON_EXCLUDED_CLASS,
                "detail": f"feature_class={feat_class}",
            })
            continue
        if lineage_val not in accepted_set:
            rejected.append({
                "feature": col,
                "reason": REASON_NON_CLEAN_LINEAGE,
                "detail": f"lineage={lineage_val}",
            })
            continue
        accepted.append(col)

    return LineageGateResult(
        accepted_features=tuple(sorted(accepted)),
        rejected=tuple(rejected),
        n_input_columns=len(cols),
    )


def lineage_summary_markdown(result: LineageGateResult) -> str:
    """Render a human-readable summary suitable for inclusion in the
    Step 4 ``compute_budget_used.md`` or a dedicated lineage report.

    Stable formatting for two-run sha256 determinism: feature lists
    sorted, no timestamps embedded.
    """
    lines: list[str] = []
    lines.append("# Causal lineage gate — heavy_ml_probe")
    lines.append("")
    lines.append(
        f"- Input columns: **{result.n_input_columns}**"
    )
    lines.append(
        f"- Accepted (clean): **{result.n_accepted}**"
    )
    lines.append(
        f"- Rejected: **{result.n_rejected}**"
    )
    lines.append("")
    if result.rejected:
        # Aggregate by reason
        from collections import Counter
        reasons = Counter(r["reason"] for r in result.rejected)
        lines.append("## Rejection breakdown")
        lines.append("")
        for reason, count in sorted(reasons.items()):
            lines.append(f"- `{reason}`: {count}")
        lines.append("")
        lines.append("## Rejected features (sorted)")
        lines.append("")
        lines.append("| Feature | Reason | Detail |")
        lines.append("|---|---|---|")
        for r in sorted(result.rejected, key=lambda x: (x["reason"], x["feature"])):
            lines.append(
                f"| {r['feature']} | {r['reason']} | {r['detail']} |"
            )
        lines.append("")
    return "\n".join(lines)


__all__ = (
    "DEFAULT_ACCEPTED_LINEAGE",
    "REASON_NON_CLEAN_LINEAGE",
    "REASON_EXCLUDED_CLASS",
    "REASON_UNKNOWN_FEATURE",
    "REASON_MISSING_FROM_MATRIX",
    "LineageGateResult",
    "filter_training_columns",
    "lineage_summary_markdown",
)
