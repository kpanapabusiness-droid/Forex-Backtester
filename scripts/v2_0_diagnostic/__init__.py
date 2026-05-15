"""v2.0 archetype clustering diagnostic.

Path-shape clustering on KH-24 + Arc 1 + Arc 2 (three weak datasets).
Outcome-blind feature engineering, then per-archetype characterisation +
predictability AUC + overlap with existing dual-gate clusters. Central
output is `results/v2_0_diagnostic/v2_0_evidence_flags.csv` — every
(dataset, K, archetype) row evaluated against first-pass priors.

Diagnostic scope only — no protocol revision, no floor-setting, no verdicts.
"""
