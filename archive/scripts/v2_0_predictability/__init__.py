"""v2.0 predictability investigation.

Tests whether the predictability wall (logistic AUC < 0.60 from PR #129)
is real or an artifact, via three angles:

  A) Random Forest on the same 8-feature entry set
  B) Expanded feature set (Arc 2 catalogue / KH-24 derived)
  C) Path-so-far features observed at t > 0 bars after entry

Scope: target archetypes from PR #129 that pass relaxed clean+meaty+size
criteria (skip Arc 1; bars_held=1 issue makes Arc 1 path-shape clusters
invalid). Computation + reporting only — no protocol revision.
"""
