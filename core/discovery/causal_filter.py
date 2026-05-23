"""Causal-lineage filter: reject rules touching SUSPECT or UNVERIFIED features.

Per dispatch Task 2 and signal_discovery_probe.md v1.0 §"Causal audit
constraint": only ``CausalLineage.CLEAN`` features may participate in
the search. Rules using SUSPECT or UNVERIFIED features are rejected
PRE-EVALUATION — they never touch the data, so they don't count toward
the Bonferroni denominator (see core.discovery.bonferroni).

The clean-feature pool is computed once at search startup from
``core.features.pipeline.feature_lineage_dataframe()``; rules are then
either filtered at generation time (cheaper) or post-filtered (cleaner
audit trail). This module supports both modes:

  * ``clean_feature_pool(lineage_df)`` — list of feature names whose
    lineage is CLEAN. Pass this as the grammar's ``feature_pool`` to
    generate-time-restrict the search.
  * ``check_rule_causal(spec, lineage_df) -> (passed, reason)`` — used
    by the search loop to log rejection reasons even when generate-time
    filtering already prevented bad rules (defensive double-check).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from core.discovery.grammar import RuleSpec


@dataclass(frozen=True)
class CausalCheckResult:
    """Outcome of running the causal filter on one rule."""

    passed: bool
    rejected_features: tuple[str, ...]   # features causing rejection (sorted)
    reason: str                          # human-readable reason


def clean_feature_pool(
    lineage_df: pd.DataFrame,
    accepted: Iterable[str] = ("clean",),
    exclude_classes: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return the sorted tuple of feature names with accepted lineage.

    ``lineage_df`` must have columns ``name``, ``causal_lineage``, ``feature_class``
    (the shape produced by ``core.features.pipeline.feature_lineage_dataframe``).
    """
    required = {"name", "causal_lineage", "feature_class"}
    missing = required - set(lineage_df.columns)
    if missing:
        raise ValueError(f"lineage_df missing columns: {sorted(missing)}")
    accepted_set = {s.lower() for s in accepted}
    exclude_set = {s.lower() for s in exclude_classes}
    keep = lineage_df[
        lineage_df["causal_lineage"].str.lower().isin(accepted_set)
        & ~lineage_df["feature_class"].str.lower().isin(exclude_set)
    ]
    return tuple(sorted(keep["name"].tolist()))


def check_rule_causal(
    spec: RuleSpec,
    lineage_df: pd.DataFrame,
    accepted: Iterable[str] = ("clean",),
    exclude_classes: Iterable[str] = (),
) -> CausalCheckResult:
    """Confirm every feature in ``spec`` is accepted by the causal filter.

    Returns ``passed=True`` only if EVERY feature is in the accepted set
    AND none is in an excluded class. The rejected feature list is the
    sorted set of features that failed; ``reason`` summarises why.
    """
    accepted_set = {s.lower() for s in accepted}
    exclude_set = {s.lower() for s in exclude_classes}
    by_name = lineage_df.set_index("name")
    features = spec.features_used()
    bad: list[str] = []
    bad_classes: list[str] = []
    missing: list[str] = []
    for f in features:
        if f not in by_name.index:
            missing.append(f)
            continue
        row = by_name.loc[f]
        lineage_value = str(row["causal_lineage"]).lower()
        feat_class = str(row["feature_class"]).lower()
        if lineage_value not in accepted_set:
            bad.append(f)
        elif feat_class in exclude_set:
            bad_classes.append(f)
    if not (bad or bad_classes or missing):
        return CausalCheckResult(
            passed=True, rejected_features=(), reason="clean_lineage_passed"
        )
    rejected = tuple(sorted(set(bad) | set(bad_classes) | set(missing)))
    parts: list[str] = []
    if bad:
        parts.append(f"non_clean_lineage={sorted(bad)}")
    if bad_classes:
        parts.append(f"excluded_class={sorted(bad_classes)}")
    if missing:
        parts.append(f"unknown_feature={sorted(missing)}")
    reason = "; ".join(parts)
    return CausalCheckResult(
        passed=False, rejected_features=rejected, reason=reason
    )


__all__ = ("CausalCheckResult", "clean_feature_pool", "check_rule_causal")
