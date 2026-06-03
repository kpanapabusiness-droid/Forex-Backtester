"""Meta-labeling target construction + classifier pipeline for heavy_ml_probe.

Per ``docs/sub_protocols/heavy_ml_probe.md`` v1.0 §"What overrides the
overseer" and dispatch §1: the meta-label target is binary —

  1 = trade reached +1R MFE STRICTLY BEFORE SL hit OR time-exit
  0 = otherwise

Same-bar tie-break (dispatch §1): when +1R MFE and SL hit fall on the
same bar within OHLC, treat as SL-hit-first (conservative). Equivalent
formulation:

  * ``bars_to_1r_mfe`` is NaN     → target = 0  (never reached 1R)
  * ``bars_to_1r_mfe <  bars_held`` → target = 1 (reached strictly before close)
  * ``bars_to_1r_mfe == bars_held``:
        - ``exit_reason == "sl"``     → target = 0 (same-bar tie, SL wins)
        - any other exit reason       → target = 1 (reached 1R same bar
                                                    as a non-adverse close)
  * ``bars_to_1r_mfe >  bars_held`` → target = 0 (defensive; shouldn't
                                                  happen on consistent data)

**Relationship to PR-D survival target (chat resolution Q1):** the
survival framing (``time_to_reach_1r_censored_at_sl_or_time_exit``)
models the same underlying event with time-to-event semantics +
censoring. Meta-labeling here models the event-occurred as a binary
classification problem. Both share the +1R MFE event definition AND
the SL/time-exit competing-risk definition; they differ in what they
model. Reference Q1 in PR-D's survival module when it lands.

**No lookahead in features (still); target is hindsight by design.**
Per dispatch §1: the target uses post-entry MFE / MAE / exit columns
from the closed-trade pool, which is correct — the target is a
hindsight label of what actually happened. The non-negotiable is that
the features consumed by the classifier are ex-ante (enforced by the
lineage gate in PR-A + the holdout cutoff in PR-B).

This module:

  1. ``build_meta_label_target`` — vectorised target construction over
     a pool DataFrame with the required schema (see
     :data:`REQUIRED_POOL_COLUMNS`).
  2. ``run_meta_labeling`` — orchestrates target construction →
     ``run_automl(target_col=..., keep_classifiers=True,
     collect_oof_predictions=True)`` → threshold sweep → classifier
     persistence → manifest emission.
  3. Threshold sweep produces ``meta_label_results.csv`` rows with
     ``precision, recall, f1, n_trades_kept, n_trades_dropped,
     mean_R_kept_set, mean_R_dropped_set`` per threshold in
     :data:`DEFAULT_THRESHOLD_SWEEP`.
  4. Per-fold classifier persistence to
     ``<classifiers_dir>/fold_{N}.joblib`` with sha256-bound manifest
     so PR-E's A6 consumer (``build_a6_config_from_heavy_ml``) can load
     them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from core.heavy_ml_probe.automl import (
    DEFAULT_MAX_ITER_PER_FOLD,
    DEFAULT_N_FOLDS,
    DEFAULT_PERMUTATION_REPEATS,
    DEFAULT_RANDOM_STATE,
    AllFeaturesRejected,
    AutoMLResult,
    HoldoutGuardViolation,
    run_automl,
)
from core.heavy_ml_probe.io import sha256_file
from core.heavy_ml_probe.labels import (
    DEFAULT_MFE_R_THRESHOLD,
    META_LABEL_TARGET_COL,
    REQUIRED_POOL_COLUMNS,
    SL_EXIT_REASON,
    PoolSchemaError,
    _validate_pool_schema,
    build_meta_label_target,
)

# Locked threshold-sweep grid per dispatch §3 (meta-label specific). The
# target name, +1R threshold, required-column schema, the SL tie-break
# anchor, ``PoolSchemaError`` and ``build_meta_label_target`` itself now
# live in the sklearn-free ``core.heavy_ml_probe.labels`` module (imported
# above, re-exported via ``__all__``) so label honesty — including the
# take-the-loss same-bar tie-break — can be tested in CI's minimal env
# without the AutoML / joblib machinery present.
DEFAULT_THRESHOLD_SWEEP: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


# ── Threshold sweep ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ThresholdRow:
    """One row of the meta-label threshold sweep."""

    threshold: float
    precision: float
    recall: float
    f1: float
    n_trades_kept: int
    n_trades_dropped: int
    mean_r_kept_set: float
    mean_r_dropped_set: float
    edge_lift_r: float  # mean_R_kept - mean_R_dropped


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def _f1(precision: float, recall: float) -> float:
    if not (np.isfinite(precision) and np.isfinite(recall)):
        return float("nan")
    if (precision + recall) <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def threshold_sweep(
    oof: pd.DataFrame,
    final_r_by_trade_id: pd.Series,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLD_SWEEP,
    trade_id_col: str = "trade_id",
) -> pd.DataFrame:
    """Compute the threshold sweep over out-of-fold predictions.

    Per dispatch §3:

      * Aggregate all 11 folds' OOF predictions into one prediction
        table (keyed by trade_id).
      * For each threshold, compute precision / recall / f1 against the
        binary target AND the mean realised R for both the kept set
        (probability ≥ threshold) and the dropped set (probability <
        threshold).
      * ``edge_lift_r = mean_r_kept_set - mean_r_dropped_set`` is the
        load-bearing diagnostic — a meta-label that doesn't shift mean
        R isn't adding edge.

    Parameters
    ----------
    oof
        DataFrame with columns ``(trade_id, fold, y_true, y_pred_proba)``
        from :class:`AutoMLResult.oof_predictions`.
    final_r_by_trade_id
        ``Series`` indexed by ``trade_id`` mapping to realised R.
    thresholds
        Iterable of probability thresholds to sweep. Default per dispatch
        §3.

    Returns
    -------
    DataFrame with one row per threshold, columns matching
    :class:`ThresholdRow` field names.
    """
    required = {trade_id_col, "y_true", "y_pred_proba"}
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(
            f"threshold_sweep requires OOF columns {sorted(required)}; "
            f"missing: {sorted(missing)}"
        )

    # If a trade was scored in multiple folds (shouldn't happen with
    # TimeSeriesSplit, but defensive), keep the LAST fold's prediction.
    # Sort by fold ascending → groupby trade_id → tail(1) for stability.
    oof_sorted = oof.sort_values(["fold", trade_id_col], kind="mergesort")
    oof_dedup = (
        oof_sorted
        .groupby(trade_id_col, sort=True)
        .tail(1)
        .reset_index(drop=True)
    )

    # Align realised R to OOF rows. Missing trade_ids land as NaN; the
    # kept/dropped mean R for those trades is excluded via nanmean.
    realised_r = oof_dedup[trade_id_col].map(final_r_by_trade_id).astype(float).values
    y_true = oof_dedup["y_true"].astype(int).values
    y_pred_proba = oof_dedup["y_pred_proba"].astype(float).values
    n_pos = int((y_true == 1).sum())

    rows: list[ThresholdRow] = []
    for t in thresholds:
        kept_mask = y_pred_proba >= float(t)
        dropped_mask = ~kept_mask
        n_kept = int(kept_mask.sum())
        n_dropped = int(dropped_mask.sum())
        tp = int(((kept_mask) & (y_true == 1)).sum())
        precision = _safe_div(tp, n_kept)
        recall = _safe_div(tp, n_pos)
        f1 = _f1(precision, recall)

        # Realised R aggregates (silence the all-NaN nanmean warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_r_kept = (
                float(np.nanmean(realised_r[kept_mask])) if n_kept > 0 else float("nan")
            )
            mean_r_dropped = (
                float(np.nanmean(realised_r[dropped_mask])) if n_dropped > 0 else float("nan")
            )

        edge_lift = (
            mean_r_kept - mean_r_dropped
            if np.isfinite(mean_r_kept) and np.isfinite(mean_r_dropped)
            else float("nan")
        )
        rows.append(ThresholdRow(
            threshold=float(t),
            precision=precision,
            recall=recall,
            f1=f1,
            n_trades_kept=n_kept,
            n_trades_dropped=n_dropped,
            mean_r_kept_set=mean_r_kept,
            mean_r_dropped_set=mean_r_dropped,
            edge_lift_r=edge_lift,
        ))

    df = pd.DataFrame([r.__dict__ for r in rows])
    return df.sort_values("threshold", kind="mergesort").reset_index(drop=True)


# ── Classifier persistence ───────────────────────────────────────────


def persist_fold_classifiers(
    automl_result: AutoMLResult,
    classifiers_dir: Path,
    *,
    arc_name: str,
    cluster_id: int,
) -> Path:
    """Persist each fold's inner sklearn-compatible classifier to disk.

    Requires the upstream ``run_automl(..., keep_classifiers=True)``.
    Writes per-fold joblib pickles + a sidecar manifest with sha256 +
    provenance so PR-E's A6 consumer can load + verify without
    retraining.

    Manifest schema:

        {
          "arc_name": "<arc>",
          "cluster_id": <int>,
          "sub_protocol": "heavy_ml_probe",
          "stage": "meta_label",
          "generated_at": "<UTC ISO>",
          "joblib_version": "<x.y.z>",
          "n_folds_persisted": <int>,
          "folds": [
            {
              "fold_id": <int>,
              "path": "<relative_to_classifiers_dir>",
              "sha256": "<hex>",
              "classifier_type": "<sklearn class name>",
              "n_train": <int>,
              "n_valid": <int>,
              "fold_auc": <float | null>,
              "best_estimator_name": "<FLAML name>"
            },
            ...
          ]
        }

    Returns the manifest path. Folds whose ``fitted_estimator`` is
    ``None`` (single-class training or AutoML skip) are still recorded
    in the manifest with ``path: null`` so the audit trail covers every
    fold.
    """
    classifiers_dir = Path(classifiers_dir)
    classifiers_dir.mkdir(parents=True, exist_ok=True)

    folds: list[dict] = []
    for fr in automl_result.fold_results:
        entry: dict = {
            "fold_id": int(fr.fold),
            "n_train": int(fr.n_train),
            "n_valid": int(fr.n_val),
            "fold_auc": float(fr.auc_val) if np.isfinite(fr.auc_val) else None,
            "best_estimator_name": str(fr.best_estimator),
        }
        if fr.fitted_estimator is None:
            entry["path"] = None
            entry["sha256"] = None
            entry["classifier_type"] = None
        else:
            fname = f"fold_{int(fr.fold):02d}.joblib"
            fpath = classifiers_dir / fname
            joblib.dump(fr.fitted_estimator, fpath, compress=3)
            entry["path"] = fname
            entry["sha256"] = sha256_file(fpath)
            entry["classifier_type"] = type(fr.fitted_estimator).__name__
        folds.append(entry)

    # No ``generated_at`` timestamp: the classifier manifest is a
    # sidecar to the top-level step_4/heavy_ml/manifest.json, which
    # already carries ``created_at``. Omitting the timestamp here keeps
    # the classifier manifest's sha256 deterministic across two runs,
    # which in turn keeps the top-level manifest's
    # ``artefacts.meta_label_classifier_manifest.sha256`` deterministic.
    # (If a future audit needs per-classifier-manifest write time,
    # mtime on disk is the source of truth.)
    manifest_payload = {
        "arc_name": str(arc_name),
        "cluster_id": int(cluster_id),
        "sub_protocol": "heavy_ml_probe",
        "stage": "meta_label",
        "joblib_version": str(joblib.__version__),
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


def stable_classifier_manifest_sha256(manifest_path: Path) -> str:
    """SHA256 of the classifier manifest payload.

    No timestamp field exists in the classifier manifest by design (the
    top-level Step 4 manifest carries ``created_at``; the sidecar
    doesn't duplicate it) — so this function is equivalent to
    ``sha256_file(manifest_path)``. Kept on the public API surface for
    symmetry with :func:`core.heavy_ml_probe.pipeline.stable_payload_sha256`
    and so a future timestamp re-introduction has an obvious place to
    strip it.
    """
    import hashlib
    return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()


# ── Top-level orchestrator ──────────────────────────────────────────


@dataclass(frozen=True)
class MetaLabelResult:
    """Aggregated outcome of one meta-labeling run."""

    automl_result: AutoMLResult
    threshold_sweep: pd.DataFrame
    target_distribution: dict[int, int]  # {0: count, 1: count}
    classifier_manifest_path: Path | None
    skip_reason: str = "ok"

    @property
    def n_total(self) -> int:
        return int(sum(self.target_distribution.values()))

    @property
    def positive_rate(self) -> float:
        n = self.n_total
        return float(self.target_distribution.get(1, 0) / n) if n > 0 else float("nan")


def run_meta_labeling(
    pool: pd.DataFrame,
    used_features: Sequence[str],
    *,
    train_end: pd.Timestamp,
    arc_name: str,
    cluster_id: int,
    classifiers_dir: Path,
    n_folds: int = DEFAULT_N_FOLDS,
    max_iter_per_fold: int = DEFAULT_MAX_ITER_PER_FOLD,
    seed: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = 1,
    permutation_repeats: int = DEFAULT_PERMUTATION_REPEATS,
    thresholds: Sequence[float] = DEFAULT_THRESHOLD_SWEEP,
    entry_time_col: str = "entry_time",
    trade_id_col: str = "trade_id",
    final_r_col: str = "final_r",
    mfe_r_threshold: float = DEFAULT_MFE_R_THRESHOLD,
) -> MetaLabelResult:
    """Run the full meta-labeling stage end-to-end.

    Sequence:

      1. Validate pool schema (HALT loud on missing columns).
      2. Construct the binary target via :func:`build_meta_label_target`.
      3. Inject target into the pool under the locked column name
         :data:`META_LABEL_TARGET_COL` (does NOT mutate caller's pool).
      4. Call :func:`core.heavy_ml_probe.automl.run_automl` with
         ``target_col=META_LABEL_TARGET_COL``, ``keep_classifiers=True``,
         ``collect_oof_predictions=True``.
      5. Run threshold sweep on the OOF predictions vs realised R.
      6. Persist per-fold classifiers + sidecar manifest under
         ``classifiers_dir``.

    Raises
    ------
    PoolSchemaError
        If the pool is missing any column in :data:`REQUIRED_POOL_COLUMNS`
        (HALT-loud per dispatch §1 last paragraph).
    HoldoutGuardViolation
        If the pool has trades on/after ``train_end`` (propagated from
        ``run_automl``).
    AllFeaturesRejected
        If ``used_features`` is empty (propagated).
    """
    # 1. Schema validation up-front
    _validate_pool_schema(pool, REQUIRED_POOL_COLUMNS, target="meta-label")
    if final_r_col not in pool.columns:
        raise PoolSchemaError(
            f"meta-labeling threshold sweep requires {final_r_col!r} column "
            f"(realised R per trade); pool has: {sorted(pool.columns)[:20]}"
        )

    # 2. Target construction (vectorised over the pool)
    target = build_meta_label_target(
        pool, mfe_r_threshold=mfe_r_threshold,
        sl_exit_reason=SL_EXIT_REASON,
    )
    target_distribution = {
        0: int((target == 0).sum()),
        1: int((target == 1).sum()),
    }

    # 3. Inject target. Use .assign so the caller's DataFrame is not
    # mutated; the resulting frame is row-aligned to the original.
    pool_with_target = pool.assign(**{META_LABEL_TARGET_COL: target})

    # 4. Classifier pipeline reusing PR-B's run_automl
    automl_result = run_automl(
        pool=pool_with_target,
        used_features=used_features,
        train_end=train_end,
        n_folds=n_folds,
        max_iter_per_fold=max_iter_per_fold,
        seed=seed,
        metric="roc_auc",
        n_jobs=n_jobs,
        permutation_repeats=permutation_repeats,
        entry_time_col=entry_time_col,
        target_col=META_LABEL_TARGET_COL,
        trade_id_col=trade_id_col,
        keep_classifiers=True,
        collect_oof_predictions=True,
    )

    # 5. Threshold sweep on OOF predictions vs realised R
    final_r_series = pool[[trade_id_col, final_r_col]].set_index(trade_id_col)[final_r_col]
    sweep_df = threshold_sweep(
        automl_result.oof_predictions,
        final_r_series,
        thresholds=thresholds,
        trade_id_col=trade_id_col,
    )

    # 6. Persist per-fold classifiers
    classifier_manifest_path = persist_fold_classifiers(
        automl_result, Path(classifiers_dir),
        arc_name=arc_name, cluster_id=cluster_id,
    )

    return MetaLabelResult(
        automl_result=automl_result,
        threshold_sweep=sweep_df,
        target_distribution=target_distribution,
        classifier_manifest_path=classifier_manifest_path,
        skip_reason="ok",
    )


def meta_label_results_to_dataframe(result: MetaLabelResult) -> pd.DataFrame:
    """Wrap the threshold sweep with diagnostic columns for the CSV.

    Adds per-fold AUC context + the OOF-aggregated positive rate so a
    single CSV reader gets the full meta-label picture without
    cross-referencing other artefacts.
    """
    df = result.threshold_sweep.copy()
    df["target_n_pos"] = result.target_distribution.get(1, 0)
    df["target_n_neg"] = result.target_distribution.get(0, 0)
    df["target_positive_rate"] = result.positive_rate
    df["aggregate_oof_auc_mean"] = result.automl_result.auc_mean
    df["aggregate_oof_auc_std"] = result.automl_result.auc_std
    df["n_folds_valid"] = result.automl_result.n_folds_valid
    df["n_folds_total"] = result.automl_result.n_folds_total
    return df


__all__ = (
    "DEFAULT_MFE_R_THRESHOLD",
    "DEFAULT_THRESHOLD_SWEEP",
    "META_LABEL_TARGET_COL",
    "REQUIRED_POOL_COLUMNS",
    "SL_EXIT_REASON",
    "MetaLabelResult",
    "PoolSchemaError",
    "ThresholdRow",
    "build_meta_label_target",
    "meta_label_results_to_dataframe",
    "persist_fold_classifiers",
    "run_meta_labeling",
    "stable_classifier_manifest_sha256",
    "threshold_sweep",
    # re-exports for upstream callers that only need to import from one place
    "AllFeaturesRejected",
    "HoldoutGuardViolation",
)
