"""arc_discovery_01 — Signal Discovery Probe (L_PROTOCOL v3.0 / signal_discovery_probe v1.0).

Random-search rule discovery over the v3 clean-lineage feature space. Each
rule is a boolean AND/OR/NOT tree of up to 5 atoms (``feature OP threshold``)
where threshold is drawn from a per-feature quantile grid. 10,000 rules are
evaluated deterministically (random_state=42); each rule's trade pool is
simulated under a single locked exit policy (initial SL 2.0xATR + trail
activation at +2.0R close-based + trail 2.0xATR + no time exit). Ranking is
by mean R per rule; Bonferroni significance is reported alongside.

The package is internal to one arc but built reusable — the grammar /
rule-engine / pool-simulator components are signal-class-agnostic and can
seed future discovery arcs.

See:
  - L_PROTOCOL.md §0 (project goal) + §2 Step 1 + §5 (sub-protocol mechanism)
  - docs/sub_protocols/signal_discovery_probe.md (v1.0)
  - configs/arc_discovery_01.yaml (this arc's locked parameters)
  - results/arc_discovery_01/ARC_OPEN.md (open-doc + dispatch resolutions)
"""

from __future__ import annotations
