"""Cox PH survival modelling for heavy_ml_probe Step 4 (PR-D).

Per ``docs/sub_protocols/heavy_ml_probe.md`` v1.0 §"Survival model
variant for Pipeline D" (updated PR-D) + chat resolution Q1 + dispatch
§3:

  * **Target framing:** time-to-+1R-MFE censored at SL hit OR time-exit.
  * **Event indicator:** ``1`` if trade reached +1R MFE before SL/time-exit;
    ``0`` otherwise (censored).
  * **Duration:** number of bars from entry to first of
    ``{+1R MFE, SL hit, time-exit}``.
  * **Same-bar tie-break:** if +1R MFE and SL hit fall on the same bar
    (within OHLC), treat as SL-hit-first (event = 0, censored at that
    bar). Identical tie-break to PR-C's ``build_meta_label_target`` —
    same underlying event modelled differently.

**Library swap from spec:** the original dispatch named `lifelines` for
Cox PH; this module uses ``statsmodels.duration.hazard_regression.PHReg``
instead because `lifelines` is blocked on Python 3.14 (transitive dep
``ecos`` has no cp314 wheel). Modelling semantics are equivalent.

**RSF (Random Survival Forest) deferred** per PR-B flag-1 disposition —
scikit-survival is also blocked on Py 3.14. PR-D ships Cox PH only.

**Concordance** is computed by :func:`core.heavy_ml_probe.metrics.concordance`
(Harrell's C-index, from-scratch implementation per dispatch §4 fallback).

**Pool schema** (PR-C compatible — meta-labeling and survival share the
same +1R event + competing-risk definition):

  * ``bars_to_1r_mfe`` (int, NaN if never reached)
  * ``bars_held`` (int, total bars to close)
  * ``exit_reason`` (str, ``"sl"`` triggers tie-break)
  * ``entry_time`` (datetime, TimeSeriesSplit ordering)
  * ``trade_id`` (int, OOF / persistence keying)

Optional column ``final_r`` is NOT required by survival (unlike
meta-labeling, which uses it for kept/dropped mean R).

Module surface:

  * ``build_survival_target(pool)`` → ``(duration, event)`` ndarrays
    over the pool, vectorised.
  * ``run_survival(pool, used_features, ...)`` → :class:`SurvivalResult`
    with per-fold ``PHRegResults`` snapshots, concordance per fold,
    aggregate, and persistence of all folds for PR-E's A4 adapter.
  * ``persist_fold_models(...)`` writes
    ``<classifiers_dir>/fold_NN.joblib`` plus a sidecar manifest with
    sha256 + per-fold provenance (matches PR-C convention).

Minimum-N discipline (dispatch §6): warn on n < 200, mark
``convergence_warning=True`` in the per-fold output, never silently
skip. Convergence failures (``ConvergenceWarning`` / ``LinAlgError``)
are caught — fold concordance becomes ``NaN`` and the loop continues.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from core.heavy_ml_probe.automl import (
    DEFAULT_N_FOLDS,
    DEFAULT_RANDOM_STATE,
    AllFeaturesRejected,
    _assert_holdout_guard,
    _build_time_series_folds,
    _coerce_entry_time,
    _select_used_features,
)
from core.heavy_ml_probe.io import sha256_file
from core.heavy_ml_probe.labels import (
    SURVIVAL_REQUIRED_POOL_COLUMNS,
    PoolSchemaError,
    _validate_pool_schema,
    build_survival_target,
)
from core.heavy_ml_probe.metrics import concordance

# Locked threshold per dispatch §3. Cox PH is unstable below this n.
MIN_N_WARN: int = 200

# ``SURVIVAL_REQUIRED_POOL_COLUMNS`` + ``build_survival_target`` + the
# take-the-loss same-bar tie-break now live in the sklearn-free
# ``core.heavy_ml_probe.labels`` module (imported above, re-exported via
# ``__all__``). The survival event/duration construction shares its +1R-MFE
# definition and SL/time-exit censoring with the meta-label, including the
# ``"hard_sl"``-aware tie-break.


# ── Result types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class SurvivalFoldResult:
    """One fold of the Cox PH evaluation."""

    fold: int                      # 1..n_folds
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    n_train: int
    n_train_event: int             # event=1 count in training slice
    n_train_censored: int          # event=0 count in training slice
    n_val: int
    n_val_event: int
    n_val_censored: int
    fit_succeeded: bool
    convergence_warning: bool
    concordance: float             # NaN if fit failed or no comparable pairs
    coefficients: tuple[float, ...]  # one per used_feature; () if fit failed
    log_likelihood: float          # NaN if fit failed
    fit_message: str               # human-readable status / failure reason
    fitted_results: Any | None = field(default=None, repr=False)  # PHRegResults; not pickled at result level


@dataclass(frozen=True)
class SurvivalResult:
    """Aggregated Cox PH output across all folds."""

    fold_results: tuple[SurvivalFoldResult, ...]
    coefficients_long: pd.DataFrame   # (fold, feature, coef, p_value, std_err)
    used_features: tuple[str, ...]
    n_folds_total: int
    n_folds_valid: int                # folds where concordance is finite
    concordance_mean: float           # nanmean
    concordance_std: float            # nanstd
    total_n_events: int               # sum of event=1 across folds' training slices
    statsmodels_version: str
    skip_reason: str = "ok"
    classifier_manifest_path: Path | None = None


# ── Internal: PHReg fit + concordance ─────────────────────────────────


def _statsmodels_version() -> str:
    try:
        import statsmodels  # type: ignore

        return str(statsmodels.__version__)
    except ImportError:
        return ""


def _safe_phreg_fit(
    *,
    X: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
) -> tuple[Any | None, bool, str]:
    """Fit PHReg with full error trapping.

    Returns ``(results_or_none, convergence_warning, message)``.

      * On success: ``(results, False, "ok")``.
      * On ConvergenceWarning raised but fit returned: ``(results, True, "converged_with_warning")``.
      * On exception: ``(None, True, f"{ExcType}: {msg}")``.
    """
    from statsmodels.duration.hazard_regression import PHReg
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    try:
        model = PHReg(endog=duration, status=event, exog=X)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            results = model.fit(disp=0)
        warned = any(
            issubclass(w.category, ConvergenceWarning) for w in caught
        )
        return results, bool(warned), ("converged_with_warning" if warned else "ok")
    except Exception as exc:
        return None, True, f"{type(exc).__name__}: {exc}"


def _predicted_risk(results: Any, X: np.ndarray) -> np.ndarray:
    """Per-trade hazard ratio ``exp(X @ params)``.

    Used as the ranking score for concordance. Higher → faster expected
    event → should rank with smaller duration. Computed manually (not
    via ``results.predict(pred_type='hr')``) so we don't depend on
    statsmodels' predict-bunch shape staying stable across versions —
    and so we can return a plain ndarray regardless of how PHReg
    structures its prediction return.

    Per PR-D flag-1 disposition (chat-approved): the direct
    ``exp(X @ params)`` path is the right home for this; the comment
    above is the rationale that should land in spec / log review.
    """
    lp = X @ np.asarray(results.params, dtype=float)
    return np.exp(lp)


def _extract_baseline_cumulative_hazard(results: Any) -> dict[str, list[float]]:
    """Extract the baseline cumulative hazard in a stable shape.

    statsmodels' ``results.baseline_cumulative_hazard`` is a list of
    strata, each stratum being ``[unique_times, cumhaz, survival]``.
    For single-stratum models (default) we have one entry. Returned as
    a plain Python dict for joblib portability (no ndarray dtype
    surprises across runs):

        {
          "stratum": <int>,
          "times": [...],
          "cumulative_hazard": [...],
          "survival_function": [...],
        }

    For multi-stratum models (rare in our use case) only the first
    stratum is returned with a warning. PR-D's invocation does not
    stratify, so this branch never fires in production.
    """
    bch = results.baseline_cumulative_hazard
    if not isinstance(bch, list) or len(bch) == 0:
        return {
            "stratum": 0,
            "times": [],
            "cumulative_hazard": [],
            "survival_function": [],
        }
    if len(bch) > 1:
        warnings.warn(
            f"PHReg returned {len(bch)} strata; persisting first only. "
            "heavy_ml_probe does not stratify Cox PH by default.",
            UserWarning,
            stacklevel=3,
        )
    stratum = bch[0]
    if not isinstance(stratum, (list, tuple)) or len(stratum) < 3:
        return {
            "stratum": 0,
            "times": [],
            "cumulative_hazard": [],
            "survival_function": [],
        }
    times, cumhaz, surv = stratum[0], stratum[1], stratum[2]
    return {
        "stratum": 0,
        "times": [float(t) for t in times],
        "cumulative_hazard": [float(h) for h in cumhaz],
        "survival_function": [float(s) for s in surv],
    }


# ── Per-fold runner ──────────────────────────────────────────────────


def _run_one_fold(
    *,
    fold_idx: int,
    pool_sorted: pd.DataFrame,
    used_features: Sequence[str],
    duration_all: np.ndarray,
    event_all: np.ndarray,
    train_s: int,
    train_e: int,
    val_s: int,
    val_e: int,
    times: pd.Series,
    keep_model: bool,
) -> SurvivalFoldResult:
    X = pool_sorted[list(used_features)].values
    X_train, X_val = X[train_s:train_e], X[val_s:val_e]
    d_train, d_val = duration_all[train_s:train_e], duration_all[val_s:val_e]
    e_train, e_val = event_all[train_s:train_e], event_all[val_s:val_e]

    n_train = int(len(d_train))
    n_train_event = int((e_train == 1).sum())
    n_train_censored = int((e_train == 0).sum())
    n_val = int(len(d_val))
    n_val_event = int((e_val == 1).sum())
    n_val_censored = int((e_val == 0).sum())

    # Cox PH fit requires at least one event in the training slice.
    if n_train_event == 0:
        return SurvivalFoldResult(
            fold=fold_idx,
            train_start=times.iloc[train_s], train_end=times.iloc[train_e - 1],
            val_start=times.iloc[val_s], val_end=times.iloc[val_e - 1],
            n_train=n_train, n_train_event=n_train_event,
            n_train_censored=n_train_censored,
            n_val=n_val, n_val_event=n_val_event, n_val_censored=n_val_censored,
            fit_succeeded=False, convergence_warning=True,
            concordance=float("nan"),
            coefficients=(),
            log_likelihood=float("nan"),
            fit_message="skipped:no_events_in_training_slice",
            fitted_results=None,
        )
    if n_train < MIN_N_WARN:
        warnings.warn(
            f"survival fold {fold_idx}: n_train={n_train} < MIN_N_WARN="
            f"{MIN_N_WARN}; Cox PH coefficients may be unstable. "
            f"convergence_warning flagged on output row.",
            UserWarning,
            stacklevel=2,
        )

    results, conv_warning, msg = _safe_phreg_fit(
        X=X_train, duration=d_train, event=e_train,
    )
    if results is None:
        # Convergence failure → NaN concordance + propagate the loop.
        return SurvivalFoldResult(
            fold=fold_idx,
            train_start=times.iloc[train_s], train_end=times.iloc[train_e - 1],
            val_start=times.iloc[val_s], val_end=times.iloc[val_e - 1],
            n_train=n_train, n_train_event=n_train_event,
            n_train_censored=n_train_censored,
            n_val=n_val, n_val_event=n_val_event, n_val_censored=n_val_censored,
            fit_succeeded=False, convergence_warning=True,
            concordance=float("nan"),
            coefficients=(),
            log_likelihood=float("nan"),
            fit_message=msg,
            fitted_results=None,
        )

    # Minimum-N flag is OR-ed with statsmodels' own convergence warning.
    convergence_warning = bool(conv_warning or n_train < MIN_N_WARN)

    # Concordance on the validation slice
    val_risk = _predicted_risk(results, X_val)
    try:
        c_idx = concordance(e_val, d_val, val_risk)
    except Exception as exc:  # defensive
        warnings.warn(
            f"survival fold {fold_idx}: concordance computation raised "
            f"{type(exc).__name__}({exc}); concordance set to NaN",
            UserWarning, stacklevel=2,
        )
        c_idx = float("nan")

    coefs = tuple(float(c) for c in np.asarray(results.params, dtype=float).tolist())
    try:
        ll = float(results.llf)
    except Exception:
        ll = float("nan")

    return SurvivalFoldResult(
        fold=fold_idx,
        train_start=times.iloc[train_s], train_end=times.iloc[train_e - 1],
        val_start=times.iloc[val_s], val_end=times.iloc[val_e - 1],
        n_train=n_train, n_train_event=n_train_event,
        n_train_censored=n_train_censored,
        n_val=n_val, n_val_event=n_val_event, n_val_censored=n_val_censored,
        fit_succeeded=True, convergence_warning=convergence_warning,
        concordance=float(c_idx),
        coefficients=coefs,
        log_likelihood=ll,
        fit_message=msg,
        fitted_results=(results if keep_model else None),
    )


# ── Persistence ──────────────────────────────────────────────────────


def persist_fold_models(
    result: SurvivalResult,
    classifiers_dir: Path,
    *,
    arc_name: str,
    cluster_id: int,
) -> Path:
    """Persist each fold's ``PHRegResults`` + extracted baseline-hazard
    metadata, plus a sidecar manifest with sha256 + provenance.

    Per dispatch §7: ``classifiers/survival/fold_{NN}.joblib``. Each
    pickle is a dict ``{results, baseline_hazard, used_features,
    coefficients}`` so PR-E's A4 adapter has everything it needs to
    compute ``P(reach +1R in next K bars | survived to N)`` without
    re-fitting.

    No ``generated_at`` timestamp on the sidecar manifest — same
    convention as PR-C ``meta_labeling.persist_fold_classifiers`` (the
    top-level Step-4 manifest carries ``created_at``). Manifest:

        {
          "arc_name": "<arc>",
          "cluster_id": <int>,
          "sub_protocol": "heavy_ml_probe",
          "stage": "survival",
          "joblib_version": "<x.y.z>",
          "statsmodels_version": "<x.y.z>",
          "n_folds_persisted": <int>,
          "folds": [
            {
              "fold_id": <int>,
              "path": "<filename.joblib>" | null,
              "sha256": "<hex>" | null,
              "n_train": <int>,
              "n_train_event": <int>,
              "n_train_censored": <int>,
              "concordance": <float | null>,
              "convergence_warning": <bool>,
              "fit_succeeded": <bool>,
              "fit_message": "<str>"
            },
            ...
          ]
        }
    """
    classifiers_dir = Path(classifiers_dir)
    classifiers_dir.mkdir(parents=True, exist_ok=True)

    folds: list[dict] = []
    for fr in result.fold_results:
        entry: dict = {
            "fold_id": int(fr.fold),
            "n_train": int(fr.n_train),
            "n_train_event": int(fr.n_train_event),
            "n_train_censored": int(fr.n_train_censored),
            "concordance": (
                float(fr.concordance) if np.isfinite(fr.concordance) else None
            ),
            "convergence_warning": bool(fr.convergence_warning),
            "fit_succeeded": bool(fr.fit_succeeded),
            "fit_message": str(fr.fit_message),
        }
        if not fr.fit_succeeded or fr.fitted_results is None:
            entry["path"] = None
            entry["sha256"] = None
        else:
            fname = f"fold_{int(fr.fold):02d}.joblib"
            fpath = classifiers_dir / fname
            payload = {
                "results": fr.fitted_results,
                "baseline_hazard": _extract_baseline_cumulative_hazard(fr.fitted_results),
                "used_features": list(result.used_features),
                "coefficients": list(fr.coefficients),
            }
            joblib.dump(payload, fpath, compress=3)
            entry["path"] = fname
            entry["sha256"] = sha256_file(fpath)
        folds.append(entry)

    manifest_payload = {
        "arc_name": str(arc_name),
        "cluster_id": int(cluster_id),
        "sub_protocol": "heavy_ml_probe",
        "stage": "survival",
        "joblib_version": str(joblib.__version__),
        "statsmodels_version": _statsmodels_version(),
        "n_folds_persisted": int(sum(1 for f in folds if f["path"] is not None)),
        "folds": folds,
    }
    manifest_path = classifiers_dir / "manifest.json"
    import json
    blob = json.dumps(manifest_payload, sort_keys=True, indent=2)
    if not blob.endswith("\n"):
        blob = blob + "\n"
    manifest_path.write_bytes(blob.encode("utf-8"))
    return manifest_path


# ── Top-level orchestrator ───────────────────────────────────────────


def run_survival(
    pool: pd.DataFrame,
    used_features: Sequence[str],
    *,
    train_end: pd.Timestamp,
    arc_name: str,
    cluster_id: int,
    classifiers_dir: Path,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_RANDOM_STATE,  # noqa: ARG001 — Cox PH is deterministic; seed kept for signature parity
    entry_time_col: str = "entry_time",
    trade_id_col: str = "trade_id",
) -> SurvivalResult:
    """Run heavy_ml_probe Cox PH survival across an 11-fold TimeSeriesSplit.

    Sequence:

      1. Validate pool schema (HALT loud on missing columns).
      2. Apply holdout guard via :func:`core.heavy_ml_probe.automl._assert_holdout_guard`.
      3. Sort pool by ``entry_time`` for causal TimeSeriesSplit indexing.
      4. Construct ``(duration, event)`` via :func:`build_survival_target`.
      5. Iterate folds via :func:`_run_one_fold`; trap ConvergenceWarning
         + LinAlgError; never silently skip.
      6. Persist all folds via :func:`persist_fold_models`.

    Raises
    ------
    PoolSchemaError
        If pool lacks any column in :data:`SURVIVAL_REQUIRED_POOL_COLUMNS`
        (HALT-loud per dispatch §9 #7).
    HoldoutGuardViolation
        If pool's max ``entry_time`` >= ``train_end``.
    AllFeaturesRejected
        If ``used_features`` is empty.
    ValueError
        If required columns missing or fold construction degenerate.
    """
    if not used_features:
        raise AllFeaturesRejected(
            "lineage gate rejected every feature — cannot fit Cox PH on an "
            "empty design matrix"
        )
    _validate_pool_schema(pool, SURVIVAL_REQUIRED_POOL_COLUMNS, target="survival")
    if trade_id_col not in pool.columns:
        raise PoolSchemaError(
            f"survival requires {trade_id_col!r} column for fold-result "
            f"persistence keying"
        )
    _assert_holdout_guard(pool, train_end, entry_time_col=entry_time_col)

    # Sort by entry_time for causal indexing
    times_all = _coerce_entry_time(pool, entry_time_col)
    pool_sorted = (
        pool.assign(_entry_ts=times_all)
        .sort_values(["_entry_ts"], kind="mergesort")
        .drop(columns=["_entry_ts"])
        .reset_index(drop=True)
    )
    times = _coerce_entry_time(pool_sorted, entry_time_col).reset_index(drop=True)

    used_features = tuple(used_features)
    _ = _select_used_features(pool_sorted, used_features)

    duration, event = build_survival_target(pool_sorted)
    folds = _build_time_series_folds(pool_sorted, n_folds, entry_time_col)
    if not folds:
        raise ValueError(
            f"TimeSeriesSplit with n_folds={n_folds} produced no folds on "
            f"pool of size {len(pool_sorted)}; pool too small"
        )

    fold_results: list[SurvivalFoldResult] = []
    coef_rows: list[dict] = []
    for fold_idx, (train_s, train_e, val_s, val_e) in enumerate(folds, start=1):
        fr = _run_one_fold(
            fold_idx=fold_idx,
            pool_sorted=pool_sorted,
            used_features=used_features,
            duration_all=duration,
            event_all=event,
            train_s=train_s, train_e=train_e,
            val_s=val_s, val_e=val_e,
            times=times,
            keep_model=True,
        )
        fold_results.append(fr)

        # Long-form coefficients for the CSV. Per-feature columns: coef,
        # p_value, std_err. Pull from results object directly when fit
        # succeeded; emit NaN rows otherwise so every feature appears
        # for every fold (sparse-row-less CSV).
        if fr.fit_succeeded and fr.fitted_results is not None:
            try:
                pvals = np.asarray(fr.fitted_results.pvalues, dtype=float)
            except Exception:
                pvals = np.full(len(used_features), np.nan)
            try:
                bse = np.asarray(fr.fitted_results.bse, dtype=float)
            except Exception:
                bse = np.full(len(used_features), np.nan)
            for j, feat in enumerate(used_features):
                coef_rows.append({
                    "fold": fr.fold,
                    "feature": feat,
                    "coefficient": float(fr.coefficients[j]),
                    "p_value": float(pvals[j]) if np.isfinite(pvals[j]) else float("nan"),
                    "std_err": float(bse[j]) if np.isfinite(bse[j]) else float("nan"),
                    "concordance": float(fr.concordance) if np.isfinite(fr.concordance) else float("nan"),
                    "n_train": int(fr.n_train),
                    "n_train_event": int(fr.n_train_event),
                    "convergence_warning": bool(fr.convergence_warning),
                })
        else:
            for feat in used_features:
                coef_rows.append({
                    "fold": fr.fold,
                    "feature": feat,
                    "coefficient": float("nan"),
                    "p_value": float("nan"),
                    "std_err": float("nan"),
                    "concordance": float("nan"),
                    "n_train": int(fr.n_train),
                    "n_train_event": int(fr.n_train_event),
                    "convergence_warning": True,
                })

    coefficients_long = pd.DataFrame(coef_rows)
    if not coefficients_long.empty:
        coefficients_long = coefficients_long.sort_values(
            ["fold", "feature"], kind="mergesort"
        ).reset_index(drop=True)

    c_vals = np.array([fr.concordance for fr in fold_results], dtype=float)
    n_valid = int(np.isfinite(c_vals).sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # nanmean-of-all-nan RuntimeWarning
        c_mean = float(np.nanmean(c_vals)) if n_valid > 0 else float("nan")
        c_std = float(np.nanstd(c_vals, ddof=0)) if n_valid > 0 else float("nan")

    result = SurvivalResult(
        fold_results=tuple(fold_results),
        coefficients_long=coefficients_long,
        used_features=tuple(used_features),
        n_folds_total=len(fold_results),
        n_folds_valid=n_valid,
        concordance_mean=c_mean,
        concordance_std=c_std,
        total_n_events=int(sum(fr.n_train_event for fr in fold_results)),
        statsmodels_version=_statsmodels_version(),
        skip_reason="ok",
    )

    # Persist all folds + sidecar manifest
    manifest_path = persist_fold_models(
        result, Path(classifiers_dir),
        arc_name=arc_name, cluster_id=cluster_id,
    )
    return SurvivalResult(
        fold_results=result.fold_results,
        coefficients_long=result.coefficients_long,
        used_features=result.used_features,
        n_folds_total=result.n_folds_total,
        n_folds_valid=result.n_folds_valid,
        concordance_mean=result.concordance_mean,
        concordance_std=result.concordance_std,
        total_n_events=result.total_n_events,
        statsmodels_version=result.statsmodels_version,
        skip_reason="ok",
        classifier_manifest_path=manifest_path,
    )


def survival_results_to_dataframe(result: SurvivalResult) -> pd.DataFrame:
    """Long-form survival results CSV per dispatch §7.

    One row per (fold, feature). Columns:
      ``fold, feature, coefficient, p_value, std_err, concordance,
      n_train, n_train_event, convergence_warning``

    Aggregate-AUC-style columns are NOT included per-row (they would be
    redundant across rows within a fold). Aggregate concordance lives
    on the in-memory :class:`SurvivalResult` and in the manifest
    extras block — surfacing it as a CSV header would be more confusing
    than helpful.
    """
    return result.coefficients_long.copy()


__all__ = (
    "MIN_N_WARN",
    "SURVIVAL_REQUIRED_POOL_COLUMNS",
    "SurvivalFoldResult",
    "SurvivalResult",
    "build_survival_target",
    "persist_fold_models",
    "run_survival",
    "survival_results_to_dataframe",
)
