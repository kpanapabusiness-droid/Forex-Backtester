"""FLAML AutoML wrapper for heavy_ml_probe Step 4 (PR-B).

Per ``docs/sub_protocols/heavy_ml_probe.md`` v1.0 §"What overrides the
overseer" + dispatch §1:

  * CV: 11-fold ``TimeSeriesSplit`` on the IS slice (2010-01-01 →
    2020-12-31). No random k-fold. Holdout (2021-01-01 → present)
    untouched.
  * Per-fold training: each fold runs its own ``flaml.AutoML.fit()``.
    No "train once globally, evaluate per fold" — guarantees feature
    selection + hyperparameter search happen inside the fold's
    training window only.
  * Budget cap: ``max_iter`` per fold per FLAML's native trial-
    counting. PR-B empirical verification (see
    ``docs/dispatches/heavy_ml_probe_pr_b_log.md`` §3a):
    ``flaml.AutoML.modelcount`` == ``max_iter`` exactly when FLAML
    runs to the cap. Ratio 1.0; locked.
  * Determinism: ``seed=42`` propagated through FLAML + numpy +
    pandas; ``n_jobs=1``; identical input pool → identical artefacts
    across two runs.

  * Estimator set: FLAML's default for binary classification on this
    installation (FLAML 2.6.0):
    ``['lgbm', 'rf', 'xgboost', 'extra_tree', 'xgb_limitdepth', 'sgd',
    'catboost', 'lrl1']``. Captured at runtime into the leaderboard
    artefact for cross-env audit.

Single-class fold AUC handling (dispatch §3b): a fold whose validation
slice contains only one class of the target yields an undefined ROC AUC.
``sklearn.metrics.roc_auc_score`` returns NaN in that case in modern
sklearn (verified at PR-B time); FLAML internally surfaces this as a
sentinel loss but does not crash. This module's contract:

  * Per-fold AUC is recorded as ``numpy.nan`` when the fold is
    single-class. Confirmed by ``test_automl.py``.
  * Aggregate AUC uses ``numpy.nanmean`` so a single bad fold does not
    poison the whole estimate.
  * ``n_folds_valid`` (folds with computable AUC) is reported
    alongside the mean so downstream readers can tell whether the
    aggregate spans all 11 folds or fewer.

Holdout-guard (dispatch §5 #3): ``run_automl`` asserts the input
pool's max ``entry_time`` is strictly less than the configured
``train_end`` cutoff before calling FLAML. Violation raises
:class:`HoldoutGuardViolation`.

Lineage-rejects-all guard (dispatch §5 #2): if the lineage gate
rejects every feature, :class:`AllFeaturesRejected` is raised before
any AutoML call.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

from core.heavy_ml_probe.metrics import auc_roc

# Locked defaults — overridable via PipelineConfig/YAML but baked here
# so internal callers (tests, integration) don't need to thread them
# through every code path.
DEFAULT_N_FOLDS: int = 11
DEFAULT_MAX_ITER_PER_FOLD: int = 1000
DEFAULT_PERMUTATION_REPEATS: int = 5
DEFAULT_RANDOM_STATE: int = 42

# FLAML's default binary-classification estimator list on this install
# (FLAML 2.6.0). Recorded for cross-env audit; the actual list FLAML
# uses at runtime is captured per-fold via ``automl.estimator_list`` so
# we never log a stale value.
DOCUMENTED_FLAML_2_6_0_ESTIMATORS: tuple[str, ...] = (
    "lgbm",
    "rf",
    "xgboost",
    "extra_tree",
    "xgb_limitdepth",
    "sgd",
    "catboost",
    "lrl1",
)


class HoldoutGuardViolation(RuntimeError):
    """Raised when the input pool contains trades on/after ``train_end``.

    Per L_PROTOCOL §1 (non-negotiable) + dispatch §5 #3: the 2021-01-01
    → present holdout window is OFF-LIMITS during training. The guard
    fires BEFORE any FLAML call so no holdout-tainted model ever lands
    on disk.
    """


class AllFeaturesRejected(RuntimeError):
    """Raised when the lineage gate rejects every column in the pool.

    Per dispatch §5 #2: if no clean features remain, fail fast with a
    clear error rather than calling AutoML with an empty design matrix.
    """


# ── Result types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class FoldResult:
    """One fold of the AutoML CV evaluation."""

    fold: int                       # 1..n_folds
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    n_train: int
    n_val: int
    n_val_pos: int                  # positive-class count in val (for single-class diagnostic)
    best_estimator: str             # FLAML's best_estimator name for this fold
    best_loss: float                # FLAML's internal loss for the best config
    auc_val: float                  # OOS AUC on the validation slice (NaN if single-class)
    auc_train: float                # IS AUC on the training slice (NaN if single-class)
    modelcount: int                 # FLAML's automl.modelcount (= actual trials run)
    max_iter: int                   # the cap we passed in
    estimator_list: tuple[str, ...] # FLAML's estimator_list for this fold
    fit_wall_seconds: float


@dataclass(frozen=True)
class AutoMLResult:
    """Aggregated AutoML output across all folds."""

    fold_results: tuple[FoldResult, ...]
    leaderboard: pd.DataFrame       # one row per (fold, estimator) — long-form
    importance: pd.DataFrame        # mean + std permutation importance per feature
    used_features: tuple[str, ...]  # post lineage-gate, in column-order passed to FLAML
    n_folds_total: int
    n_folds_valid: int              # folds with finite val AUC
    auc_mean: float                 # nanmean of per-fold val AUCs
    auc_std: float                  # nanstd of per-fold val AUCs
    total_modelcount: int           # sum of per-fold modelcount values
    total_fit_wall_seconds: float
    flaml_version: str
    metric: str


# ── Internal helpers ─────────────────────────────────────────────────


def _coerce_entry_time(pool: pd.DataFrame, column: str = "entry_time") -> pd.Series:
    """Return ``pool[column]`` parsed as UTC pandas Timestamps.

    Tests pass already-Timestamped columns; production pools come from
    parquet with ISO strings or naive datetimes. Normalise both.
    """
    if column not in pool.columns:
        raise ValueError(
            f"pool must contain {column!r} column for TimeSeriesSplit ordering"
        )
    s = pool[column]
    return pd.to_datetime(s, utc=True)


def _assert_holdout_guard(
    pool: pd.DataFrame,
    train_end: pd.Timestamp,
    *,
    entry_time_col: str = "entry_time",
) -> None:
    """Per L_PROTOCOL §1: max ``entry_time`` in pool must be strictly
    less than ``train_end``."""
    ts = pd.Timestamp(train_end)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    times = _coerce_entry_time(pool, entry_time_col)
    if len(times) == 0:
        return  # empty pool — nothing to guard
    max_ts = times.max()
    if max_ts >= ts:
        raise HoldoutGuardViolation(
            f"holdout-guard violated: pool max entry_time={max_ts.isoformat()} "
            f"is >= train_end={ts.isoformat()}; filter the pool to IS-only "
            f"before invoking AutoML"
        )


def _coerce_target(pool: pd.DataFrame, target_col: str = "y") -> np.ndarray:
    """Return the target column as a binary int ndarray.

    Tests and production pools both label the binary target ``y``. The
    actual target-construction (cluster membership for PR-B; reach-1R
    for PR-C) lives upstream — this module just consumes the array.
    """
    if target_col not in pool.columns:
        raise ValueError(
            f"pool must contain {target_col!r} column (binary target)"
        )
    y = pool[target_col].astype(int).values
    uniq = set(int(v) for v in np.unique(y))
    if not uniq.issubset({0, 1}):
        raise ValueError(
            f"target must be binary (0/1); found {sorted(uniq)}"
        )
    return y


def _select_used_features(
    pool: pd.DataFrame,
    used_features: Sequence[str],
) -> pd.DataFrame:
    """Slice ``pool`` to the lineage-gate-cleared features in column order."""
    if not used_features:
        raise AllFeaturesRejected(
            "lineage gate rejected every feature — cannot fit AutoML on an "
            "empty design matrix. Inspect step_4/heavy_ml/manifest.json's "
            "lineage_gate.rejection_reasons field for the per-feature reasons."
        )
    missing = [c for c in used_features if c not in pool.columns]
    if missing:
        raise ValueError(
            f"pool missing lineage-gate-accepted columns: {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}"
        )
    return pool[list(used_features)].copy()


def _flaml_version() -> str:
    try:
        import flaml  # type: ignore

        return str(flaml.__version__)
    except ImportError:
        return ""


def _build_time_series_folds(
    pool_sorted: pd.DataFrame,
    n_folds: int,
    entry_time_col: str = "entry_time",
) -> list[tuple[int, int, int, int]]:
    """Return ``[(train_start_idx, train_end_idx, val_start_idx, val_end_idx), ...]``.

    Indices are half-open ranges into the time-sorted pool. Mirrors
    ``sklearn.model_selection.TimeSeriesSplit`` exactly so the test
    suite can assert behaviour without a second implementation.
    """
    cv = TimeSeriesSplit(n_splits=n_folds)
    n = len(pool_sorted)
    out: list[tuple[int, int, int, int]] = []
    for train_idx, val_idx in cv.split(np.arange(n)):
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        out.append((int(train_idx[0]), int(train_idx[-1]) + 1,
                    int(val_idx[0]), int(val_idx[-1]) + 1))
    return out


def _run_one_fold(
    *,
    fold_idx: int,
    pool_sorted: pd.DataFrame,
    used_features: Sequence[str],
    y_all: np.ndarray,
    train_s: int,
    train_e: int,
    val_s: int,
    val_e: int,
    times: pd.Series,
    max_iter: int,
    seed: int,
    metric: str,
    n_jobs: int,
    permutation_repeats: int,
) -> tuple[FoldResult, pd.DataFrame, pd.DataFrame]:
    """Train + evaluate one fold. Returns (fold_result, leaderboard_rows,
    importance_rows)."""
    import time

    from flaml import AutoML  # local import — heavy library, lazy load

    X = pool_sorted[list(used_features)].values
    X_train, X_val = X[train_s:train_e], X[val_s:val_e]
    y_train, y_val = y_all[train_s:train_e], y_all[val_s:val_e]
    n_val_pos = int((y_val == 1).sum())

    # Per dispatch §3b: detect single-class fold up front so we can
    # short-circuit the AutoML call. A single-class training fold breaks
    # FLAML's internal CV; a single-class validation fold yields NaN
    # AUC. Skip the fit when training is degenerate.
    n_train_classes = len(np.unique(y_train))
    if n_train_classes < 2:
        warnings.warn(
            f"fold {fold_idx} has single-class training slice "
            f"(n_train={len(y_train)}, n_train_pos={int((y_train == 1).sum())}); "
            f"skipping AutoML fit, AUC will be NaN",
            UserWarning,
            stacklevel=2,
        )
        fr = FoldResult(
            fold=fold_idx,
            train_start=times.iloc[train_s], train_end=times.iloc[train_e - 1],
            val_start=times.iloc[val_s], val_end=times.iloc[val_e - 1],
            n_train=int(train_e - train_s), n_val=int(val_e - val_s),
            n_val_pos=n_val_pos,
            best_estimator="<skipped:single_class_train>",
            best_loss=float("nan"),
            auc_val=float("nan"), auc_train=float("nan"),
            modelcount=0, max_iter=int(max_iter),
            estimator_list=(),
            fit_wall_seconds=0.0,
        )
        return fr, pd.DataFrame(), pd.DataFrame()

    t0 = time.perf_counter()
    automl = AutoML()
    # Per dispatch §1: FLAML's internal eval_method='cv' would do
    # ANOTHER CV inside the training fold. That's expected — FLAML uses
    # its own inner CV to pick hyperparameters; our outer 11-fold split
    # is the OOS evaluation surface. n_splits=3 keeps inner-CV
    # compute bounded.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        automl.fit(
            X_train=X_train, y_train=y_train,
            task="classification",
            metric=metric,
            max_iter=int(max_iter),
            seed=int(seed),
            n_jobs=int(n_jobs),
            eval_method="cv",
            n_splits=3,
            verbose=0,
        )
    fit_wall = time.perf_counter() - t0

    # Per dispatch §3a: automl.modelcount is the actual trial counter.
    modelcount = int(getattr(automl, "modelcount", -1))
    if modelcount < 0:
        # Should not happen on FLAML 2.6.0; defensive fallback.
        warnings.warn(
            f"flaml.AutoML.modelcount unavailable; budget audit will be "
            f"under-reported for fold {fold_idx}",
            UserWarning,
            stacklevel=2,
        )
        modelcount = 0
    if modelcount > max_iter:
        # PR-B HALT trigger per dispatch §3a if ratio ≥ 1.5×; we accept
        # modelcount ≤ max_iter empirically (PR-B log §3a), but
        # surface as a warning if it ever drifts.
        warnings.warn(
            f"fold {fold_idx} modelcount={modelcount} exceeds max_iter={max_iter}; "
            f"FLAML budget accounting may have drifted — review PR-B log §3a",
            UserWarning,
            stacklevel=2,
        )

    # OOS + IS AUCs
    try:
        proba_val = automl.predict_proba(X_val)[:, 1]
    except Exception:
        proba_val = np.full(len(y_val), np.nan)
    try:
        proba_train = automl.predict_proba(X_train)[:, 1]
    except Exception:
        proba_train = np.full(len(y_train), np.nan)
    auc_val = auc_roc(y_val, proba_val) if np.all(np.isfinite(proba_val)) else float("nan")
    auc_train = auc_roc(y_train, proba_train) if np.all(np.isfinite(proba_train)) else float("nan")

    # Leaderboard rows: one per estimator FLAML evaluated. FLAML 2.6.0
    # exposes ``best_loss_per_estimator``; use it as the leaderboard.
    best_per_est: Mapping[str, Any] = dict(
        getattr(automl, "best_loss_per_estimator", {}) or {}
    )
    leaderboard_rows = pd.DataFrame([
        {
            "fold": fold_idx,
            "estimator": str(est),
            "best_loss": float(loss) if loss is not None and np.isfinite(float(loss)) else float("nan"),
            "is_overall_best": (str(est) == str(automl.best_estimator)),
        }
        for est, loss in sorted(best_per_est.items())
    ])

    # Permutation importance on the validation slice (dispatch §2).
    # NaN-safe: if validation is single-class or sklearn raises,
    # importance comes back NaN-filled.
    #
    # sklearn 1.5+'s permutation_importance refuses FLAML's outer
    # AutoML wrapper because the wrapper does not advertise itself as
    # a ClassifierMixin in a way sklearn's `_check_response_method`
    # recognises. We pass the inner fitted estimator
    # (``automl.model.estimator`` — e.g. ``LGBMClassifier``,
    # ``XGBClassifier``, ``RandomForestClassifier``) instead, which IS
    # a proper sklearn-compatible classifier. Predictions made through
    # the inner estimator are byte-identical to the outer wrapper for
    # standard pipelines per FLAML's docs.
    importance_rows = pd.DataFrame()
    inner_estimator = getattr(getattr(automl, "model", None), "estimator", None)
    if len(np.unique(y_val)) >= 2 and inner_estimator is not None:
        try:
            pi = permutation_importance(
                inner_estimator, X_val, y_val,
                n_repeats=int(permutation_repeats),
                random_state=int(seed),
                n_jobs=int(n_jobs),
                scoring="roc_auc",
            )
            importance_rows = pd.DataFrame({
                "fold": fold_idx,
                "feature": list(used_features),
                "importance_mean": pi.importances_mean,
                "importance_std": pi.importances_std,
            })
        except Exception as e:
            warnings.warn(
                f"fold {fold_idx}: permutation_importance raised "
                f"{type(e).__name__}({e}); importance row will be empty",
                UserWarning,
                stacklevel=2,
            )

    fr = FoldResult(
        fold=fold_idx,
        train_start=times.iloc[train_s], train_end=times.iloc[train_e - 1],
        val_start=times.iloc[val_s], val_end=times.iloc[val_e - 1],
        n_train=int(train_e - train_s), n_val=int(val_e - val_s),
        n_val_pos=n_val_pos,
        best_estimator=str(automl.best_estimator),
        best_loss=float(automl.best_loss) if np.isfinite(automl.best_loss) else float("nan"),
        auc_val=float(auc_val),
        auc_train=float(auc_train),
        modelcount=int(modelcount),
        max_iter=int(max_iter),
        estimator_list=tuple(automl.estimator_list or ()),
        fit_wall_seconds=float(fit_wall),
    )
    return fr, leaderboard_rows, importance_rows


# ── Public surface ───────────────────────────────────────────────────


def run_automl(
    pool: pd.DataFrame,
    used_features: Sequence[str],
    *,
    train_end: pd.Timestamp,
    n_folds: int = DEFAULT_N_FOLDS,
    max_iter_per_fold: int = DEFAULT_MAX_ITER_PER_FOLD,
    seed: int = DEFAULT_RANDOM_STATE,
    metric: str = "roc_auc",
    n_jobs: int = 1,
    permutation_repeats: int = DEFAULT_PERMUTATION_REPEATS,
    entry_time_col: str = "entry_time",
    target_col: str = "y",
) -> AutoMLResult:
    """Run heavy_ml_probe AutoML across an 11-fold TimeSeriesSplit.

    Parameters
    ----------
    pool
        Step-1 trade pool. Must contain ``entry_time`` (or
        ``entry_time_col``), the binary target ``y`` (or
        ``target_col``), and every column listed in ``used_features``.
    used_features
        Column order to pass to FLAML. Caller is responsible for
        applying the lineage gate first (see
        :mod:`core.heavy_ml_probe.causal_lineage`).
    train_end
        Holdout-guard cutoff. Max ``entry_time`` in ``pool`` must be
        strictly less than this timestamp; otherwise raises
        :class:`HoldoutGuardViolation`.
    n_folds
        TimeSeriesSplit fold count. Default 11 per spec.
    max_iter_per_fold
        FLAML budget per fold. Default 1000 per spec.

    Returns
    -------
    :class:`AutoMLResult` with per-fold ``FoldResult`` snapshots, a
    long-form leaderboard DataFrame, an importance DataFrame, and
    aggregate AUC + budget stats.

    Raises
    ------
    HoldoutGuardViolation
        If the input pool's max ``entry_time`` >= ``train_end``.
    AllFeaturesRejected
        If ``used_features`` is empty.
    ValueError
        If required columns are missing or the target is non-binary.
    """
    if not used_features:
        raise AllFeaturesRejected(
            "lineage gate rejected every feature — cannot fit AutoML on an "
            "empty design matrix"
        )
    _assert_holdout_guard(pool, train_end, entry_time_col=entry_time_col)

    # Sort by entry_time so TimeSeriesSplit indexing is causal.
    times_all = _coerce_entry_time(pool, entry_time_col)
    pool_sorted = pool.assign(_entry_ts=times_all).sort_values(
        ["_entry_ts"], kind="mergesort"
    ).drop(columns=["_entry_ts"]).reset_index(drop=True)
    times = _coerce_entry_time(pool_sorted, entry_time_col).reset_index(drop=True)

    y_all = _coerce_target(pool_sorted, target_col)
    used_features = tuple(used_features)
    # Validate column presence early — clearer error than FLAML's eventual KeyError.
    _ = _select_used_features(pool_sorted, used_features)

    folds = _build_time_series_folds(pool_sorted, n_folds, entry_time_col)
    if not folds:
        raise ValueError(
            f"TimeSeriesSplit with n_folds={n_folds} produced no folds on "
            f"a pool of size {len(pool_sorted)}; pool too small"
        )

    fold_results: list[FoldResult] = []
    leaderboard_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for fold_idx, (train_s, train_e, val_s, val_e) in enumerate(folds, start=1):
        fr, lb, imp = _run_one_fold(
            fold_idx=fold_idx,
            pool_sorted=pool_sorted,
            used_features=used_features,
            y_all=y_all,
            train_s=train_s, train_e=train_e,
            val_s=val_s, val_e=val_e,
            times=times,
            max_iter=max_iter_per_fold,
            seed=seed,
            metric=metric,
            n_jobs=n_jobs,
            permutation_repeats=permutation_repeats,
        )
        fold_results.append(fr)
        if not lb.empty:
            leaderboard_frames.append(lb)
        if not imp.empty:
            importance_frames.append(imp)

    leaderboard = (
        pd.concat(leaderboard_frames, ignore_index=True)
        if leaderboard_frames
        else pd.DataFrame(columns=["fold", "estimator", "best_loss", "is_overall_best"])
    )
    # Deterministic row order: (fold, estimator)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(
            ["fold", "estimator"], kind="mergesort"
        ).reset_index(drop=True)

    importance = _aggregate_importance(importance_frames, used_features)

    auc_vals = np.array([fr.auc_val for fr in fold_results], dtype=float)
    finite_mask = np.isfinite(auc_vals)
    n_valid = int(finite_mask.sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # nanmean on all-NaN raises a RuntimeWarning
        auc_mean = float(np.nanmean(auc_vals)) if n_valid > 0 else float("nan")
        auc_std = float(np.nanstd(auc_vals, ddof=0)) if n_valid > 0 else float("nan")

    return AutoMLResult(
        fold_results=tuple(fold_results),
        leaderboard=leaderboard,
        importance=importance,
        used_features=tuple(used_features),
        n_folds_total=len(fold_results),
        n_folds_valid=n_valid,
        auc_mean=auc_mean,
        auc_std=auc_std,
        total_modelcount=int(sum(fr.modelcount for fr in fold_results)),
        total_fit_wall_seconds=float(sum(fr.fit_wall_seconds for fr in fold_results)),
        flaml_version=_flaml_version(),
        metric=metric,
    )


def _aggregate_importance(
    importance_frames: list[pd.DataFrame],
    used_features: Sequence[str],
) -> pd.DataFrame:
    """Mean + std per feature across folds. ``n_folds_present`` records
    how many folds contributed a finite importance for that feature."""
    if not importance_frames:
        return pd.DataFrame(columns=[
            "feature", "importance_mean", "importance_std", "n_folds_present"
        ])
    cat = pd.concat(importance_frames, ignore_index=True)
    agg = (
        cat.groupby("feature")["importance_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "importance_mean",
            "std": "importance_std",
            "count": "n_folds_present",
        })
    )
    # Ensure every used feature appears (even if no fold had finite importance).
    present = set(agg["feature"].tolist())
    rows_to_add = [
        {"feature": f, "importance_mean": float("nan"),
         "importance_std": float("nan"), "n_folds_present": 0}
        for f in used_features if f not in present
    ]
    if rows_to_add:
        agg = pd.concat([agg, pd.DataFrame(rows_to_add)], ignore_index=True)
    agg["n_folds_present"] = agg["n_folds_present"].astype(int)
    # Sort by importance_mean desc, feature asc — deterministic.
    agg = agg.sort_values(
        ["importance_mean", "feature"],
        ascending=[False, True], kind="mergesort", na_position="last",
    ).reset_index(drop=True)
    return agg


# ── Artefact-shape helpers (consumed by io.py renderers) ─────────────


def leaderboard_to_dataframe(result: AutoMLResult) -> pd.DataFrame:
    """Long-form leaderboard CSV row layout. Columns:

      fold, estimator, best_loss, is_overall_best, auc_val,
      auc_train, modelcount, max_iter, best_estimator_for_fold

    ``fit_wall_seconds`` is intentionally NOT included — wall-clock is
    not deterministic across runs and would break the two-run
    artefact-byte equality test. Wall-clock stays available on the
    in-memory :class:`AutoMLResult` for log-doc / diagnostic use.
    """
    if not result.fold_results:
        return pd.DataFrame(columns=[
            "fold", "estimator", "best_loss", "is_overall_best",
            "auc_val", "auc_train", "modelcount", "max_iter",
            "best_estimator_for_fold",
        ])
    by_fold = {fr.fold: fr for fr in result.fold_results}
    lb = result.leaderboard.copy()
    if lb.empty:
        # Build a synthetic leaderboard from fold results so callers
        # always get a non-empty CSV when folds ran (even if FLAML did
        # not produce best_loss_per_estimator on this install).
        lb = pd.DataFrame([
            {"fold": fr.fold, "estimator": fr.best_estimator,
             "best_loss": fr.best_loss, "is_overall_best": True}
            for fr in result.fold_results
        ])
    lb["auc_val"] = lb["fold"].map({k: v.auc_val for k, v in by_fold.items()})
    lb["auc_train"] = lb["fold"].map({k: v.auc_train for k, v in by_fold.items()})
    lb["modelcount"] = lb["fold"].map({k: v.modelcount for k, v in by_fold.items()})
    lb["max_iter"] = lb["fold"].map({k: v.max_iter for k, v in by_fold.items()})
    lb["best_estimator_for_fold"] = lb["fold"].map({k: v.best_estimator for k, v in by_fold.items()})
    lb = lb[[
        "fold", "estimator", "best_loss", "is_overall_best",
        "auc_val", "auc_train", "modelcount", "max_iter",
        "best_estimator_for_fold",
    ]]
    return lb.sort_values(["fold", "estimator"], kind="mergesort").reset_index(drop=True)


def importance_to_dataframe(result: AutoMLResult) -> pd.DataFrame:
    """Permutation importance CSV layout. Columns already set by
    :func:`_aggregate_importance`."""
    return result.importance.copy()


def compute_budget_markdown(result: AutoMLResult, *, train_end: pd.Timestamp) -> str:
    """Render the ``compute_budget_used.md`` artefact.

    Stable formatting (no embedded timestamps, no wall-clock) so two-run
    sha256 determinism holds. Wall-clock stays available on the
    in-memory :class:`AutoMLResult` for log-doc / diagnostic use.
    """
    lines: list[str] = []
    lines.append("# heavy_ml_probe — AutoML compute budget audit")
    lines.append("")
    lines.append(f"- FLAML version: `{result.flaml_version}`")
    lines.append(f"- Metric: `{result.metric}`")
    lines.append(f"- Folds total: **{result.n_folds_total}**")
    lines.append(f"- Folds with valid OOS AUC: **{result.n_folds_valid}**")
    if result.n_folds_total > 0:
        max_iter = result.fold_results[0].max_iter
        cap = max_iter * result.n_folds_total
        ratio = (
            float(result.total_modelcount) / float(cap)
            if cap > 0 else float("nan")
        )
        lines.append(f"- `max_iter_per_fold`: **{max_iter}**")
        lines.append(f"- Cap (max_iter × n_folds): **{cap}**")
        lines.append(f"- Total modelcount across folds: **{result.total_modelcount}**")
        lines.append(f"- Consumption ratio (actual/cap): **{ratio:.4f}**")
    lines.append(
        f"- Holdout cutoff (`train_end`): "
        f"**{pd.Timestamp(train_end).strftime('%Y-%m-%dT%H:%M:%SZ')}**"
    )
    lines.append("")
    lines.append(
        "_Per dispatch §3a (PR-B empirical): `flaml.AutoML.modelcount` "
        "equals `max_iter` exactly when FLAML runs to the cap; ratio 1.0. "
        "Locked, proceed. Wall-clock measurements live on the in-memory "
        "`AutoMLResult` (non-deterministic; excluded from artefacts)._"
    )
    lines.append("")
    lines.append("## Per-fold trial counts")
    lines.append("")
    lines.append(
        "| Fold | n_train | n_val | n_val_pos | best_estimator | modelcount | "
        "max_iter | auc_val | auc_train |"
    )
    lines.append("|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for fr in result.fold_results:
        auc_v = "NaN" if not np.isfinite(fr.auc_val) else f"{fr.auc_val:.4f}"
        auc_t = "NaN" if not np.isfinite(fr.auc_train) else f"{fr.auc_train:.4f}"
        lines.append(
            f"| {fr.fold} | {fr.n_train} | {fr.n_val} | {fr.n_val_pos} | "
            f"{fr.best_estimator} | {fr.modelcount} | {fr.max_iter} | "
            f"{auc_v} | {auc_t} |"
        )
    lines.append("")
    lines.append("## Aggregate AUC (val, OOS)")
    lines.append("")
    auc_m = "NaN" if not np.isfinite(result.auc_mean) else f"{result.auc_mean:.4f}"
    auc_s = "NaN" if not np.isfinite(result.auc_std) else f"{result.auc_std:.4f}"
    lines.append(
        f"- `nanmean` AUC: **{auc_m}** (across {result.n_folds_valid} "
        f"valid folds; total {result.n_folds_total})"
    )
    lines.append(f"- `nanstd` AUC: **{auc_s}**")
    lines.append("")
    return "\n".join(lines)


__all__ = (
    "DEFAULT_N_FOLDS",
    "DEFAULT_MAX_ITER_PER_FOLD",
    "DEFAULT_PERMUTATION_REPEATS",
    "DEFAULT_RANDOM_STATE",
    "DOCUMENTED_FLAML_2_6_0_ESTIMATORS",
    "HoldoutGuardViolation",
    "AllFeaturesRejected",
    "FoldResult",
    "AutoMLResult",
    "run_automl",
    "leaderboard_to_dataframe",
    "importance_to_dataframe",
    "compute_budget_markdown",
)
