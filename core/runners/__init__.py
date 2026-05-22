"""Per-architecture fold runners for L_PROTOCOL v3.0 Step 5.

  - :mod:`core.runners.arc_fold_runner` — generic architecture-aware
    fold runner; replaces the KH-24-specific KH24FoldRunner for new arcs.
  - :mod:`core.runners.oracle_fold_runner` — runs the upper-bound oracle
    WFO using true-label cluster membership (no classifier).
  - :mod:`core.runners._fold_stats_helpers` — shared utilities for
    converting RunResult equity curves into FoldStats.

KH24FoldRunner (at ``core/wfo/fold_runner.py``) is retained as a thin
regression path; it delegates to A1 + ArcFoldRunner internally.
"""
