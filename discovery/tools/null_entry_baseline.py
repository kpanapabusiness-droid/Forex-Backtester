"""Random-entry NULL baseline — discovery EXPERIMENT tool (BUILT).

The council-mandated soundness control: does the real signal beat RANDOM entry
at the same fire-rate, under the same SL/exit and the same engine? It isolates
"does WHEN you enter (the signal) carry information" from "does the exit/sizing
machinery alone make money." A real edge must clear its own random-entry null.

Design (per discovery/TOOL_REGISTRY.md, "first expected BUILT entry"):
  - The EXPERIMENT part is ONLY the random-mask generation: per pair, place the
    SAME NUMBER of fires as the real signal at random eligible bar positions
    (>= warmup, with a next bar to fill), deterministic via np.random.default_rng.
    ATR is reused from the real evaluation unchanged, so SL geometry is identical
    and only entry TIMING is randomized.
  - The MEASUREMENT stays 100% canonical: the random SignalEvaluation is scored
    through the LOCKED ArcFoldRunner -> A1Architecture -> MultiPairBacktester ->
    build_fold_stats_from_run (FundedNext costs netted). This tool NEVER scores a
    trade itself (anti-Arc-10: geometry/eligibility only, no P&L realization).

Usage:
    from discovery.tools.null_entry_baseline import build_null_signal_evaluation
    null_eval = build_null_signal_evaluation(real_eval, seed=42, warmup=100)
    runner = ArcFoldRunner(architecture=A1Architecture(),
                           signal_evaluation=null_eval, panels={"H4": panel})
    stats = run_config_over_folds(runner, folds, cfg)   # canonical scoring

Created by: arc 1000 (chat 1000-1999).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.arc.signal_protocol import PerPairSignalState, SignalEvaluation


def build_null_signal_evaluation(
    real_eval: SignalEvaluation,
    *,
    seed: int = 42,
    warmup: int = 100,
) -> SignalEvaluation:
    """Return a random-entry SignalEvaluation matched to ``real_eval``.

    For every pair, draw the same number of fires as the real mask, uniformly
    at random over eligible positions ``[warmup, n-2]`` (so a next bar exists to
    fill, mirroring the pool builder's entry_idx = s+1 < n rule). ATR and the
    primary TF are reused unchanged; only entry timing is randomized.

    Deterministic: the RNG is seeded once and advanced per pair in sorted order,
    so the same (real_eval, seed) always yields byte-identical masks.

    Direction-aware (arc 2013): the per-pair ``direction`` and the eval-level
    ``direction`` are carried through unchanged, so a SHORT signal's null is a
    SHORT random entry (a fair same-side baseline). Longs are byte-identical —
    ``direction`` defaults to ``Direction.LONG``.
    """
    rng = np.random.default_rng(seed)
    per_pair: dict[str, PerPairSignalState] = {}
    for pair in sorted(real_eval.per_pair):
        state = real_eval.per_pair[pair]
        mask = state.signal_mask
        n = len(mask)
        n_fires = int(mask.to_numpy().sum())
        new_mask = np.zeros(n, dtype=bool)
        hi = n - 1  # need entry bar s+1 < n  ->  s <= n-2  ->  exclusive hi=n-1
        lo = min(warmup, hi)
        n_eligible = max(hi - lo, 0)
        if n_fires > 0 and n_eligible > 0:
            k = min(n_fires, n_eligible)
            picks = rng.choice(np.arange(lo, hi), size=k, replace=False)
            new_mask[picks] = True
        per_pair[pair] = PerPairSignalState(
            signal_mask=pd.Series(new_mask, index=mask.index),
            atr=state.atr,
            direction=state.direction,
        )
    return SignalEvaluation(
        primary_tf=real_eval.primary_tf,
        per_pair=per_pair,
        signal_name=f"null_random_entry_seed{seed}",
        causal_lineage="clean",
        direction=real_eval.direction,
    )


__all__ = ("build_null_signal_evaluation",)
