"""v3.0 Step-1 feature engineering.

Per L_PROTOCOL §2 Step 1, every Step-1 feature is:

  - Computable from data closed strictly before signal-bar open (no lookahead)
  - Tagged with a ``causal_lineage`` ∈ {clean, suspect, unverified}
  - Documented in ``docs/features_reference.md``

The feature pipeline is config-driven (no hardcoded enables). The orchestrator
(``core.features.pipeline.compute_feature_matrix``) walks the registered
feature specs, runs each producer against the (pair_df, panel) input, and
returns a single DataFrame indexed by signal timestamps.

Public surface:

    CausalLineage              # enum
    FeatureSpec                # (name, producer, lineage, inputs, etc.)
    register, get, all_specs   # registry helpers
    compute_feature_matrix     # the orchestrator
"""

from core.features.lineage import CausalLineage, FeatureSpec
from core.features.registry import all_specs, get, register

__all__ = [
    "CausalLineage",
    "FeatureSpec",
    "register",
    "get",
    "all_specs",
]
