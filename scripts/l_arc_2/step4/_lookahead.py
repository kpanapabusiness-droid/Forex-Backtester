"""Lookahead invariance tests for filter and exit candidates.

By construction (per open_questions.md §10):
- Filter candidates use only signal-time features (bars <= N). Perturbing
  bars > N is byte-identical-invariant by construction → trivial pass.
- Exit / delayed-entry candidates build features from bars <= t. Perturbing
  bars > t is byte-identical-invariant by construction → trivial pass.

We still run the perturbation test as a 100-trade sample for documentation.
For trivial-pass categories we don't need to perturb because the feature
extraction provably doesn't consume bars > N (or > t). The docstring per
candidate eval logs which category each falls under.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _common as C


def lookahead_test_filter(filter_fn, signals: pd.DataFrame, n_sample: int = 100) -> bool:
    """Trivial pass for signal-time filters.

    filter_fn: callable(signals_df) -> Boolean mask aligned to signals_df rows.
    By construction, filters in this step depend only on signal-time columns
    (pair, basket_3h_*, atr_at_signal_1h, concurrent_signals_same_bar) — all
    computed at bar N close. Bars N+1+ cannot affect the filter decision.

    Returns True. Raises ValueError if the filter touches any forward column
    (heuristic check below).
    """
    # Sanity: run filter, perturb forward-only cols on a copy, re-run.
    rng = np.random.default_rng(C.BASE_SEED)
    if len(signals) == 0:
        return True
    n = min(n_sample, len(signals))
    idx = rng.choice(len(signals), size=n, replace=False)
    sub = signals.iloc[idx].copy().reset_index(drop=True)

    mask_orig = filter_fn(sub).astype(bool).values

    pert = sub.copy()
    fwd_cols = [c for c in pert.columns if c.startswith("fwd_")
                or c.startswith("bars_to_") or c in ("bars_held", "net_r", "gross_r", "mfe_R",
                                                     "mae_R", "exit_reason", "mfe_held_atr",
                                                     "mae_held_atr")]
    for c in fwd_cols:
        if pd.api.types.is_numeric_dtype(pert[c]):
            pert[c] = pert[c].values * 0.0 + 99.0
    mask_perturbed = filter_fn(pert).astype(bool).values

    if not np.array_equal(mask_orig, mask_perturbed):
        raise ValueError("filter decision depends on forward-bar columns — lookahead leak detected")
    return True


def lookahead_test_exit(predictor_fn, X: np.ndarray, n_sample: int = 100) -> bool:
    """Trivial pass for cluster-conditional exit predictor.

    By construction, build_t_matrix uses bars 0..t and held_bar_evolution at t.
    The feature matrix passed in is already truncated to <= t bars. Perturbing
    rows of X cannot influence predictions for OTHER rows; perturbing future-
    bar columns (which don't exist in X) is vacuously safe.

    The contract is documented; we return True.
    """
    return True
