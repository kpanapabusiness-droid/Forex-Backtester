"""Metric wrappers for heavy_ml_probe.

Thin wrappers around sklearn / lifelines / scikit-survival so the
sub-protocol's per-model evaluation calls a single canonical surface.
PR-A lands callable stubs that exercise the dependency imports
defensively (so sklearn / lifelines / sksurv availability is detectable
at module import without crashing scaffolding), with real bodies
landing in PR-B/D where the data exists.

Coverage:

  * ``auc_roc(y_true, y_score)`` — wraps ``sklearn.metrics.roc_auc_score``.
    Used by AutoML (PR-B) per-fold scoring and by meta-labeling (PR-C)
    threshold sweeps.
  * ``concordance(y_event, y_time, predicted_risk)`` — wraps lifelines'
    concordance index. Used by Cox PH (PR-D).
  * ``integrated_brier_score(...)`` — wraps scikit-survival's
    ``integrated_brier_score``. Used by Random Survival Forest (PR-D).

All metric functions return ``float`` and raise informatively on input
shape mismatches; NaN / non-finite returns are propagated, not masked.

Lifelines / scikit-survival are imported lazily so the rest of the
sub-protocol can scaffold without them (matches the defensive
``lightgbm`` import pattern in ``core/steps/_classifier_defaults.py``).
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
    """C-index for survival models.

    Wraps ``lifelines.utils.concordance_index``. Imports lifelines
    lazily so PR-A scaffolding does not require the library to be
    installed in environments that won't run the survival path.

    Parameters
    ----------
    y_event : array-like of {0, 1}
        Event indicator (1 = event observed, 0 = censored).
    y_time : array-like of float
        Observed time (event time if observed, censoring time otherwise).
    predicted_risk : array-like of float
        Higher = higher predicted risk.
    """
    try:
        from lifelines.utils import concordance_index
    except ImportError as e:  # pragma: no cover — environment-dependent
        raise ImportError(
            "lifelines is required for concordance(); install via "
            "requirements-dev.txt"
        ) from e

    y_event_arr = np.asarray(y_event)
    y_time_arr = np.asarray(y_time)
    pred_arr = np.asarray(predicted_risk)
    if not (y_event_arr.shape == y_time_arr.shape == pred_arr.shape):
        raise ValueError(
            f"concordance shape mismatch: y_event={y_event_arr.shape} "
            f"y_time={y_time_arr.shape} predicted_risk={pred_arr.shape}"
        )
    # lifelines' concordance_index uses ``higher risk = lower expected
    # survival time``. Pass risk directly; do NOT negate.
    return float(concordance_index(y_time_arr, -pred_arr, y_event_arr))


def integrated_brier_score(
    survival_train,
    survival_test,
    survival_predictions,
    times,
) -> float:
    """IBS for survival predictions.

    Wraps ``sksurv.metrics.integrated_brier_score``. Imports sksurv
    lazily for the same reason as :func:`concordance` above.

    Parameters
    ----------
    survival_train : structured ndarray
        Training survival data in sksurv's ``(event, time)`` structured
        array shape (typically the output of
        ``sksurv.util.Surv.from_arrays``).
    survival_test : structured ndarray
        Test survival data in the same shape.
    survival_predictions : 2-D ndarray of shape (n_test, len(times))
        Predicted survival probabilities at each evaluation time.
    times : 1-D array of float
        Time points at which IBS is evaluated.
    """
    try:
        from sksurv.metrics import integrated_brier_score as _ibs
    except ImportError as e:  # pragma: no cover — environment-dependent
        raise ImportError(
            "scikit-survival is required for integrated_brier_score(); "
            "install via requirements-dev.txt"
        ) from e
    return float(
        _ibs(survival_train, survival_test, survival_predictions, times)
    )


__all__ = (
    "auc_roc",
    "concordance",
    "integrated_brier_score",
)
