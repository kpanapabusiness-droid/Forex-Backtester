"""Look-elsewhere haircut (Phase G).

For each (feature, cluster, t-slice):
- Compute univariate AUC of feature predicting cluster membership
  (treating feature value as score; for binary/categorical features, AUC
  is still well-defined via rank-sum).
- Compute permutation p-value via fold-preserving label shuffles.
- Apply Benjamini-Hochberg correction within (cluster, t-slice) group.
- Assign Tier 1 / 2 / 3 per op spec §6.7.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as ss

from . import _common as C


def univariate_auc(feature: np.ndarray, y: np.ndarray) -> float:
    n = len(y)
    n1 = int(np.sum(y))
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = ss.rankdata(feature)
    sum_pos = float(ranks[y.astype(bool)].sum())
    auc = (sum_pos - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(auc)


def permutation_p_value(
    feature: np.ndarray,
    y: np.ndarray,
    fold_id: np.ndarray,
    n_perm: int = C.N_PERMUTATIONS,
    seed: int = 0,
) -> tuple[float, float]:
    """Fold-preserving permutation p-value for univariate AUC.

    Two-sided test on |AUC - 0.5|.
    """
    n = len(y)
    n1 = int(np.sum(y))
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return float("nan"), float("nan")

    ranks = ss.rankdata(feature)
    sum_pos_obs = float(ranks[y.astype(bool)].sum())
    auc_obs = (sum_pos_obs - n1 * (n1 + 1) / 2) / (n1 * n0)
    centered_obs = abs(auc_obs - 0.5)

    rng = np.random.default_rng(seed)
    folds = np.unique(fold_id)
    fold_ranks = {f: ranks[fold_id == f] for f in folds}
    fold_pos_counts = {f: int(np.sum(y[fold_id == f])) for f in folds}

    total_sum = np.zeros(n_perm)
    for f in folds:
        rf = fold_ranks[f]
        kf = fold_pos_counts[f]
        nf = len(rf)
        if kf == 0 or kf == nf:
            total_sum += rf.sum() if kf == nf else 0.0
            continue
        rand_keys = rng.random((n_perm, nf))
        sel_idx = np.argpartition(rand_keys, kth=kf, axis=1)[:, :kf]
        sampled = rf[sel_idx]
        total_sum += sampled.sum(axis=1)

    aucs_null = (total_sum - n1 * (n1 + 1) / 2) / (n1 * n0)
    centered_null = np.abs(aucs_null - 0.5)
    p = (1 + int(np.sum(centered_null >= centered_obs))) / (n_perm + 1)
    return float(auc_obs), float(p)


def bh_correct(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    valid = ~np.isnan(pvals)
    p_valid = pvals[valid]
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    m = len(ranked)
    bh = ranked * m / (np.arange(1, m + 1))
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0, 1)
    out_valid = np.empty_like(p_valid)
    out_valid[order] = bh
    out = np.full(n, np.nan)
    out[valid] = out_valid
    return out


def tier_from_bh(bh_p: float) -> str:
    if np.isnan(bh_p):
        return "Tier3"
    if bh_p <= C.BH_TIER_1:
        return "Tier1"
    if bh_p <= C.BH_TIER_2:
        return "Tier2"
    return "Tier3"
