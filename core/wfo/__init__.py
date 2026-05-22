"""Walk-forward optimisation for the v3.0 backtester.

Public API:

    Fold                            # immutable fold dataclass (IS + OOS bounds)
    build_v3_folds()                # 11-fold expanding-IS 2010-2020 + holdout
    build_kh24_anchor_folds()       # 7-fold rolling 2020-10-01 → 2026-01-01
    Verdict                         # PASS_DEPLOYABLE | PASS_VIABLE | FAIL
    classify_fold_stats()           # §3 gate application

The orchestrator (``core.wfo.orchestrator``) drives per-fold backtests and
top-K selection; the holdout one-shot is locked from search per L_PROTOCOL
§2 Step 5.
"""

from core.wfo.folds import (
    Fold,
    WfoStructure,
    build_kh24_anchor_folds,
    build_v3_folds,
)
from core.wfo.gates import Verdict, classify_fold_stats

__all__ = [
    "Fold",
    "WfoStructure",
    "build_v3_folds",
    "build_kh24_anchor_folds",
    "Verdict",
    "classify_fold_stats",
]
