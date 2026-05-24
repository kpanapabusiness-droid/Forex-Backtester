"""heavy_ml_probe — Sub-protocol layered on L_PROTOCOL v3.0 overseer.

Implements ``docs/sub_protocols/heavy_ml_probe.md`` v1.0. Replaces
vanilla Step 4 with AutoML (FLAML) inside CV, adds a meta-labeling
target ("reach +1R MFE before SL"), and trains Cox PH + Random Survival
Forest variants for A4 Pipeline D consumption. Step 5 augmentation hook
allows A2 / A4 / A6 to consume heavy-ML-trained components.

This package is invoked when an arc's ``ARC_OPEN.md`` declares
``sub_protocol: heavy_ml_probe``. Vanilla Step 1 / 2 / 3 / 6 + the §3
deployment gates remain unchanged — heavy_ml_probe only overrides Step
4 and augments Step 5 per the sub-protocol spec.

PR-A scope (this commit): scaffolding + causal-lineage pre-evaluation
gate + deterministic IO + sha256 manifest writer + metric stubs +
orchestration skeleton + CLI stub. AutoML / meta-label / survival
modules land in PR-B / PR-C / PR-D respectively.

See:
  - L_PROTOCOL.md §0 (project goal) + §2 Step 4-5 + §3 gates + §5 (sub-protocol mechanism)
  - docs/sub_protocols/heavy_ml_probe.md (v1.0 spec)
  - docs/sub_protocols/signal_discovery_probe.md (sibling reference)
  - docs/dispatches/heavy_ml_probe_build_intent.md (build plan)
  - configs/heavy_ml_probe/default.yaml (locked invocation parameters)
"""

from __future__ import annotations

__version__ = "0.1.0"  # PR-A scaffolding; bumps per PR per build plan
