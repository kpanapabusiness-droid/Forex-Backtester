"""Arc 8 v3.0.2 — corrected retry under canonical 5ers_eet boundary convention.

Mirrors `scripts/l_arc_8/` layout with v3.0.2 patches:
- ``boundary_convention="5ers_eet"`` end-to-end at every subsystem (Step 1
  panels + multi_tf aux + downstream daily-DD bucketing). Replaces the
  prior `_halted` branch's UTC convention (UTC-vs-EET non-comparability
  per Amendment 6 + intent doc §0).
- PR-#208 canonical W1 producer (no `merge_asof` lookahead).
- PR-#185 holdout-window training filter (Step 4 `train_end=2021-01-01`).
- Multi-TF aux panel restored (d1/w1 attached pre-feature-matrix; closes
  Arc 8 original's all-NaN multi_tf feature gap).
- CC_18 canonical exit-policy registry at Step 5.
- Amendment 3 evaluation DEFERRED (pending engine risk-decoupling fix —
  see closure deferral block; Steps 1-5 at r_base=0.5% unaffected).
- Amendment 5 four-gate + Amendment 5.1 Gate 4 admission at dispatch time.
- Amendment 6 EET daily-DD boundary native (panels are 5ers_eet).
- Amendment 3.1 r_max-as-cap semantics applied at Step 5 amended gates.
"""
