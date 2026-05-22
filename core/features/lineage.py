"""Causal-lineage tagging for v3.0 features.

Every Step-1 feature carries a provisional ``CausalLineage`` tag at
construction time. Step 6 (lazy, deployment-only) runs a producer-level
audit that may upgrade or downgrade the tag based on actual code review.

Tag semantics (per L_PROTOCOL §2 Step 4 & §2 Step 6 + dispatch §5):

  clean       — verified-by-construction: uses only OHLC / volume from
                bars closed strictly before the signal bar; no indicators
                with hidden lookback, no rolling joins from future panels.

  suspect     — plausible lookahead risk; needs Step 6 audit before
                deployment. Features that aggregate cross-panel data
                where alignment is non-trivial fall here by default.

  unverified  — no causal review has happened yet. New features land
                here until at minimum a docstring justification exists.
                ``clean`` requires a unit test that spot-checks 5 random
                trades.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class CausalLineage(Enum):
    CLEAN = "clean"
    SUSPECT = "suspect"
    UNVERIFIED = "unverified"


# A feature producer takes (pair_df, panel, **inputs) and returns a Series
# aligned to pair_df's DatetimeIndex.
FeatureProducer = Callable[..., pd.Series]


@dataclass(frozen=True)
class FeatureSpec:
    """One feature: name, producer, lineage tag, and documentation hooks.

    ``inputs`` is a free-form dict carried alongside the feature for
    documentation (passed to ``compute_feature_matrix`` if needed). It
    typically names the panel columns / TF / window length the producer
    consumes — drives the lookahead audit in Step 6.
    """

    name: str
    producer: FeatureProducer
    lineage: CausalLineage
    feature_class: str  # "price_geometry" | "session" | "cross_pair" | etc.
    description: str
    inputs: dict[str, Any] = field(default_factory=dict)
    needs_panel: bool = False  # True if producer requires the multi-pair Panel

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FeatureSpec.name must be non-empty")
        if not callable(self.producer):
            raise TypeError(f"producer for {self.name!r} must be callable")
