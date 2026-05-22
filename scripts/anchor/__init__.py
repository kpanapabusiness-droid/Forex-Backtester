"""KH-24 anchor reproduction scripts (PR-E.2).

Two run modes:
  - Mode A: 7-fold rolling Oct 2020 → Jan 2026 (apples-to-apples vs
    published KH-24 lineage)
  - Mode B: 11-fold expanding-IS 2010-2020 + one-shot holdout
    2021-most-recent-complete-month (v3.0 baseline)

Both modes share ``run_anchor.run(...)`` — the mode is chosen by the
WFO ``structure`` argument.
"""
