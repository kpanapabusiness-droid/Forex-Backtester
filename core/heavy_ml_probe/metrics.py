"""Metric wrappers for heavy_ml_probe.

Coverage:

  * ``auc_roc(y_true, y_score)`` — wraps ``sklearn.metrics.roc_auc_score``.
    Used by AutoML (PR-B) per-fold scoring and by meta-labeling (PR-C)
    threshold sweeps.
  * ``concordance(y_event, y_time, predicted_risk)`` — Harrell's C-index,
    implemented from scratch in PR-D. Used by Cox PH survival evaluation.
  * ``integrated_brier_score(...)`` — deferred per PR-B/D library-block
    decisions (sksurv is blocked on Python 3.14). Stub retained so the
    public surface stays stable; raises ``NotImplementedError`` at call
    time. Will be revisited if RSF is re-enabled.

All metric functions return ``float`` and raise informatively on input
shape mismatches; NaN / non-finite returns are propagated, not masked.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_roc(y_true, y_score) -> float:
    """Binary AUC-ROC. Identical semantics to ``sklearn.metrics.roc_auc_score``.

    Returns ``float('nan')`` when the input set has < 2 classes
    (sklearn raises in that case; we degrade to NaN since per-fold
    AutoML can hit single-class folds for rare events).
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError(
            f"auc_roc shape mismatch: y_true={y_true_arr.shape} y_score={y_score_arr.shape}"
        )
    if len(set(y_true_arr.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true_arr, y_score_arr))


def concordance(y_event, y_time, predicted_risk) -> float:
    """Harrell's C-index — from-scratch implementation (PR-D).

    Per dispatch §4 fallback: lifelines + scikit-survival are blocked
    on Python 3.14, so this implements the canonical pairwise-comparable
    definition directly. Math reference: Harrell, F. E. Jr. (1982),
    "Evaluating the yield of medical tests" (JAMA).

    Definition. For all ordered pairs ``(i, j)`` where trade ``i`` had
    ``event=1`` AND ``time_i < time_j`` (``j`` may be censored OR event):

      * concordant if ``predicted_risk_i  >  predicted_risk_j``
      * tied       if ``predicted_risk_i  ==  predicted_risk_j``
      * discordant if ``predicted_risk_i  <  predicted_risk_j``

    Returns ``(concordant + 0.5 * tied) / (concordant + discordant + tied)``.

    Note: pairs where both trades have ``time_i == time_j`` are
    skipped (no rank-order signal). Pairs where the earlier-time trade
    is censored are also skipped (we cannot know it would have ranked
    against ``j``).

    Parameters
    ----------
    y_event : array-like of {0, 1}
        Event indicator (1 = event observed, 0 = censored).
    y_time : array-like of float
        Observed time (event time if observed, censoring time otherwise).
    predicted_risk : array-like of float
        Higher = higher predicted risk → faster expected event. For Cox
        PH this is ``exp(features @ coefficients)``.

    Returns
    -------
    Concordance index in ``[0, 1]``. Returns ``float('nan')`` when no
    comparable pairs exist (e.g. fold validation slice is all-censored
    or all-same-time).
    """
    y_event_arr = np.asarray(y_event, dtype=int)
    y_time_arr = np.asarray(y_time, dtype=float)
    pred_arr = np.asarray(predicted_risk, dtype=float)
    if not (y_event_arr.shape == y_time_arr.shape == pred_arr.shape):
        raise ValueError(
            f"concordance shape mismatch: y_event={y_event_arr.shape} "
            f"y_time={y_time_arr.shape} predicted_risk={pred_arr.shape}"
        )
    if y_event_arr.ndim != 1:
        raise ValueError(
            f"concordance expects 1-D arrays; got y_event.ndim={y_event_arr.ndim}"
        )
    n = len(y_event_arr)
    if n < 2:
        return float("nan")
    # Drop rows with non-finite predictions/times — they cannot rank.
    finite_mask = (
        np.isfinite(y_time_arr) & np.isfinite(pred_arr)
    )
    if finite_mask.sum() < 2:
        return float("nan")
    e = y_event_arr[finite_mask]
    t = y_time_arr[finite_mask]
    r = pred_arr[finite_mask]
    # Vectorised pair enumeration via broadcasting. O(n^2) memory; fine
    # at synthetic-test scale (n ~ 400) and at validation-fold scale
    # (n ~ a few hundred at typical L-arc pool sizes).
    t_i = t.reshape(-1, 1)
    t_j = t.reshape(1, -1)
    r_i = r.reshape(-1, 1)
    r_j = r.reshape(1, -1)
    e_i = e.reshape(-1, 1)
    # Comparable mask: i had event, t_i < t_j (strict — same-time pairs
    # excluded since they carry no rank signal). j's event status is
    # irrelevant — being censored at time > t_i still gives a valid
    # ordering.
    comparable = (e_i == 1) & (t_i < t_j)
    n_comparable = int(comparable.sum())
    if n_comparable == 0:
        return float("nan")
    # Higher risk → faster event → should rank with smaller time. So
    # for comparable (i, j) with t_i < t_j, concordant means r_i > r_j.
    concordant = int(((r_i > r_j) & comparable).sum())
    tied = int(((r_i == r_j) & comparable).sum())
    discordant = int(((r_i < r_j) & comparable).sum())
    # Sanity: concordant + tied + discordant == n_comparable
    if concordant + tied + discordant != n_comparable:
        # Should not happen given the three branches partition the
        # comparable set; defensive guard against future refactors.
        raise RuntimeError(
            f"concordance internal accounting drift: "
            f"c={concordant}+t={tied}+d={discordant} != comparable={n_comparable}"
        )
    return float((concordant + 0.5 * tied) / n_comparable)


def integrated_brier_score(
    survival_train,
    survival_test,
    survival_predictions,
    times,
) -> float:
    """IBS for survival predictions — DEFERRED per PR-B/D library-block.

    scikit-survival's ``integrated_brier_score`` is the canonical
    implementation but sksurv is blocked on Python 3.14 (transitive dep
    ``ecos`` has no cp314 wheel). Per PR-B flag-1 disposition: RSF is
    deferred; this function stays as a stable public-API stub. Raises
    ``NotImplementedError`` at call time with a pointer to the deferral
    rationale.
    """
    raise NotImplementedError(
        "integrated_brier_score is deferred per heavy_ml_probe PR-B "
        "flag-1 disposition: scikit-survival is blocked on Python 3.14 "
        "(ecos has no cp314 wheel). Will be reinstated when the wheel "
        "ships AND chat re-enables RSF in the spec."
    )


__all__ = (
    "auc_roc",
    "concordance",
    "integrated_brier_score",
)
