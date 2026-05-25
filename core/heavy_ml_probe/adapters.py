"""Step 5 adapter contracts for heavy_ml_probe (PR-E).

This module is the contract surface between heavy_ml_probe's Step 4
artefacts (AutoML / meta-label / survival classifiers) and the
existing L_PROTOCOL Step 5 architectures (A2 / A4 / A6) defined under
``core/architectures/``. PR-E is **narrow scope** per dispatch §6:
emits the adapters + ``heavy_ml_manifest.json``; downstream Step 5
loop is responsible for invoking the architectures themselves.

Three public builders:

  * :func:`build_a2_from_heavy_ml` — wraps PR-B's FLAML best classifier
    into an :class:`core.architectures.a2_classifier_filter.A2Config`.
    FLAML's inner sklearn-compatible estimator (e.g. ``LGBMClassifier``,
    ``XGBClassifier``) exposes ``predict_proba`` directly — drop-in.
  * :func:`build_a4_from_heavy_ml` — wraps PR-D's Cox PH coefficients +
    baseline cumulative hazard into a :class:`PathClassifierFit`-shaped
    fit consumable by :func:`core.architectures._path_classifier.predict_admit`.
    Per Q5b option B1 (chat resolution): A4's runtime treats this fit
    identically to a vanilla per-fold RandomForest fit. The wrapper's
    ``predict_proba`` reads ``bars_survived`` from a named column and
    computes ``P(reach +1R in next K bars | survived to N)`` from Cox
    PH.
  * :func:`build_a6_from_heavy_ml` — wraps PR-C's meta-label classifier
    into an :class:`core.architectures.a6_meta_labeling.A6Config`.
    Same drop-in pattern as A2 (FLAML inner estimator).

All three builders consume ``step_5/heavy_ml_augmented/heavy_ml_manifest.json``
written by :mod:`core.heavy_ml_probe.pipeline` at the end of a successful
``run_pipeline`` call. The manifest carries per-stage status + paths +
adapter-buildability flags so adapters fail loudly with a clear
message when an upstream stage skipped or failed.

Fold selection (per dispatch §3):

  * ``"last_fold"`` — use fold N (most recent training window). Default
    for production deployment.
  * ``"ensemble_mean"`` — average predictions across all valid folds.
    Useful for sensitivity checks; less principled but smoke-testable.
  * ``"full_refit"`` — NOT IMPLEMENTED in PR-E (raises
    ``NotImplementedError``). Reserved for a future enhancement that
    refits on the entire IS slice rather than using a fold artefact.
"""

from __future__ import annotations

import enum
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

# ── Public manifest schema ───────────────────────────────────────────


HEAVY_ML_MANIFEST_NAME: str = "heavy_ml_manifest.json"

# Locked schema version stamped into the Step 5 manifest. Bumps when
# the consumer-visible contract changes — adapters check the version
# at load time and refuse to consume an unknown schema.
STEP5_MANIFEST_SCHEMA_VERSION: str = "1.0"

# Per-stage status taxonomy persisted into the manifest.
STAGE_STATUS_OK: str = "ok"
STAGE_STATUS_SKIPPED: str = "skipped"
STAGE_STATUS_FAILED: str = "failed"

# Conventional column name carrying "bars survived since entry" in the
# feature row passed to the A4 adapter's predict_proba. A4's runtime
# is responsible for populating this column when it calls predict_admit.
BARS_SURVIVED_FEATURE: str = "_bars_survived"


# ── Fold selection ───────────────────────────────────────────────────


class FoldSelectionStrategy(str, enum.Enum):
    """How to pick the per-fold artefact for deployment."""

    LAST_FOLD = "last_fold"
    ENSEMBLE_MEAN = "ensemble_mean"
    FULL_REFIT = "full_refit"  # not implemented in PR-E


# ── Adapter exceptions ───────────────────────────────────────────────


class HeavyMLManifestError(RuntimeError):
    """Raised when the manifest is missing, malformed, or schema-mismatched."""


class StageUnavailableError(RuntimeError):
    """Raised when an adapter is asked to build from a stage that
    upstream-skipped or upstream-failed.

    Example: ``build_a4_from_heavy_ml`` called against a manifest whose
    ``stages.survival.status != "ok"``. The error message tells the
    caller which stage is unavailable and why.
    """


# ── A4 Cox PH adapter wrapper ────────────────────────────────────────


@dataclass(frozen=True)
class _CoxPHFoldPayload:
    """Snapshot of one fold's persisted Cox PH content.

    Mirrors the dict written by
    :func:`core.heavy_ml_probe.survival.persist_fold_models`. Pulled
    into a frozen dataclass so the adapter's internal state is
    immutable + introspectable in tests.
    """

    coefficients: tuple[float, ...]
    used_features: tuple[str, ...]
    baseline_times: tuple[float, ...]
    baseline_cumulative_hazard: tuple[float, ...]
    baseline_survival: tuple[float, ...]


def _coxph_payload_from_pickle(payload: dict) -> _CoxPHFoldPayload:
    """Lift the joblib-pickled dict from
    :func:`core.heavy_ml_probe.survival.persist_fold_models` into a
    :class:`_CoxPHFoldPayload`. Tolerates the documented dict shape;
    raises ``ValueError`` on any deviation."""
    required = {"coefficients", "used_features", "baseline_hazard"}
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(
            f"Cox PH pickle missing keys {sorted(missing)}; "
            f"present: {sorted(payload.keys())}"
        )
    bh = payload["baseline_hazard"]
    bh_required = {"times", "cumulative_hazard", "survival_function"}
    bh_missing = bh_required - set(bh.keys())
    if bh_missing:
        raise ValueError(
            f"Cox PH baseline_hazard missing keys {sorted(bh_missing)}"
        )
    return _CoxPHFoldPayload(
        coefficients=tuple(float(c) for c in payload["coefficients"]),
        used_features=tuple(str(f) for f in payload["used_features"]),
        baseline_times=tuple(float(t) for t in bh["times"]),
        baseline_cumulative_hazard=tuple(float(h) for h in bh["cumulative_hazard"]),
        baseline_survival=tuple(float(s) for s in bh["survival_function"]),
    )


def _interp_cumulative_hazard(
    payload: _CoxPHFoldPayload,
    t_query: float,
) -> float:
    """Step-function lookup for the baseline cumulative hazard at ``t_query``.

    Cox PH's baseline cumulative hazard is a right-continuous step
    function defined at the unique observed event times. Standard
    convention: for ``t`` between event times, the cumulative hazard
    equals the value at the largest observed event time ≤ ``t``. Before
    the first event time, H_0(t) = 0. After the last event time, the
    Breslow estimator does not extrapolate — we clamp to the last
    observed value.

    Choice: step function (not linear interpolation) preserves Cox PH's
    non-parametric semantics. Linear interpolation between event times
    would introduce a fake assumption that hazard accrues smoothly
    between events.
    """
    if not payload.baseline_times:
        return 0.0
    times = np.asarray(payload.baseline_times, dtype=float)
    cumhaz = np.asarray(payload.baseline_cumulative_hazard, dtype=float)
    if t_query < times[0]:
        return 0.0
    if t_query >= times[-1]:
        return float(cumhaz[-1])
    # searchsorted right gives the insertion point; idx-1 is the largest
    # observed time ≤ t_query.
    idx = int(np.searchsorted(times, t_query, side="right")) - 1
    return float(cumhaz[max(0, idx)])


@dataclass
class CoxPHAdapter:
    """Cox PH wrapper exposing both the dispatch's ``hazard_predict``
    contract AND a sklearn-compatible ``predict_proba`` shim for A4's
    :class:`PathClassifierFit` consumer.

    Adapter math (per PR-D prompt §8 + dispatch §2.2):

      .. code-block::

         H_0(N)         = baseline_cumulative_hazard at time N
         H_0(N+K)       = baseline_cumulative_hazard at time N+K
         relative_risk  = exp(features_at_N @ cox_coefficients)
         integrated_h   = (H_0(N+K) - H_0(N)) * relative_risk
         P(reach 1R in
           next K bars
           | survived N) = 1 - exp(-integrated_h)

    The ``predict_proba(X)`` shim reads ``bars_survived`` from a named
    column (:data:`BARS_SURVIVED_FEATURE`) in ``X`` so that A4's
    runtime — which only knows about ``predict_proba`` — can dispatch
    to the Cox PH machinery without modification. The remaining feature
    columns are the ones the Cox PH was fitted on (per
    ``payload.used_features``).

    When the strategy is :attr:`FoldSelectionStrategy.ENSEMBLE_MEAN`,
    :class:`CoxPHAdapter` averages coefficients + baseline hazards
    across all provided folds before any prediction (mean-coefficient
    semantics; less principled than per-fold-mean-prediction but
    cheaper and well-defined).
    """

    payloads: tuple[_CoxPHFoldPayload, ...]
    strategy: FoldSelectionStrategy
    k_horizon: int

    def __post_init__(self) -> None:
        if not self.payloads:
            raise ValueError("CoxPHAdapter requires at least one fold payload")
        if self.k_horizon <= 0:
            raise ValueError(f"k_horizon must be > 0; got {self.k_horizon}")
        # Effective payload after fold-strategy reduction
        if self.strategy is FoldSelectionStrategy.LAST_FOLD:
            object.__setattr__(self, "_effective", self.payloads[-1])
        elif self.strategy is FoldSelectionStrategy.ENSEMBLE_MEAN:
            object.__setattr__(self, "_effective", _ensemble_mean_payload(self.payloads))
        elif self.strategy is FoldSelectionStrategy.FULL_REFIT:
            raise NotImplementedError(
                "FoldSelectionStrategy.FULL_REFIT is reserved for a future PR. "
                "Use LAST_FOLD (default) or ENSEMBLE_MEAN."
            )
        else:  # pragma: no cover
            raise ValueError(f"unknown fold strategy: {self.strategy!r}")

    # ── Primary dispatch §2.2 contract ───────────────────────────────

    def hazard_predict(
        self,
        features_at_N: Mapping[str, float],
        bars_survived_at_N: int,
    ) -> float:
        """``P(reach +1R MFE in next K bars | survived to bar N)``.

        Per dispatch §2.2 + PR-D log §7.2. Returns NaN if any of the
        used features is missing or non-finite in ``features_at_N`` —
        the A4 predicate treats NaN as a non-decision.
        """
        eff: _CoxPHFoldPayload = getattr(self, "_effective")
        try:
            x = np.array(
                [float(features_at_N[k]) for k in eff.used_features],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            return float("nan")
        if not np.all(np.isfinite(x)):
            return float("nan")
        lp = float(x @ np.asarray(eff.coefficients, dtype=np.float64))
        # Numerical guard: clamp exp argument to avoid overflow.
        relative_risk = math.exp(min(max(lp, -50.0), 50.0))
        h0_n = _interp_cumulative_hazard(eff, float(bars_survived_at_N))
        h0_n_k = _interp_cumulative_hazard(
            eff, float(bars_survived_at_N + self.k_horizon)
        )
        integrated = (h0_n_k - h0_n) * relative_risk
        if integrated < 0:
            # Shouldn't happen (cumhaz is non-decreasing) but defensive.
            return 0.0
        return float(1.0 - math.exp(-integrated))

    # ── sklearn-compatible predict_proba shim ────────────────────────
    #
    # A4's runtime consumes via ``predict_admit(fit, feature_row)``
    # (see ``core/architectures/_path_classifier.py``), which calls
    # ``fit.model.predict_proba(X)`` where X is a (1, n_features)
    # ndarray. This shim:
    #
    #   1. Reads ``bars_survived`` from a named feature column.
    #   2. Delegates the rest of the columns to ``hazard_predict``.
    #
    # Returns the standard sklearn binary-classifier shape
    # ``(n_samples, 2)`` where ``[:, 1]`` is P(event) and ``[:, 0]``
    # is its complement.

    @property
    def feature_order(self) -> tuple[str, ...]:
        """Feature column order expected by :meth:`predict_proba`.

        Always: ``(*used_features, BARS_SURVIVED_FEATURE)``. The
        bars-survived column is appended at the end so A4's existing
        feature-assembly code can splice it in without re-shaping the
        Cox PH coefficient vector.
        """
        eff: _CoxPHFoldPayload = getattr(self, "_effective")
        return tuple(eff.used_features) + (BARS_SURVIVED_FEATURE,)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """sklearn-shape wrapper around :meth:`hazard_predict`.

        Expects ``X.shape == (n_samples, len(self.feature_order))``.
        Last column is interpreted as bars-survived per
        :data:`BARS_SURVIVED_FEATURE`.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        eff: _CoxPHFoldPayload = getattr(self, "_effective")
        n_features = len(eff.used_features)
        if X.shape[1] != n_features + 1:
            raise ValueError(
                f"predict_proba expects shape (n, {n_features + 1}) "
                f"(features + bars_survived); got {X.shape}"
            )
        out = np.zeros((X.shape[0], 2), dtype=np.float64)
        for i in range(X.shape[0]):
            feats = {
                eff.used_features[j]: float(X[i, j]) for j in range(n_features)
            }
            bars = int(X[i, n_features])
            p = self.hazard_predict(feats, bars)
            if not np.isfinite(p):
                # NaN P → default to 0.0 so A4's predicate treats as
                # "no admit" rather than crashing on NaN comparison.
                out[i, 1] = 0.0
            else:
                out[i, 1] = p
            out[i, 0] = 1.0 - out[i, 1]
        return out


def _ensemble_mean_payload(
    payloads: tuple[_CoxPHFoldPayload, ...],
) -> _CoxPHFoldPayload:
    """Average coefficients + baseline hazard across folds.

    Coefficients: simple arithmetic mean across folds (folds with
    different feature counts are rejected at load time, so all
    payloads share ``used_features``).

    Baseline hazard: interpolate each fold's H_0 to a common time grid
    (the union of all observed event times across folds) and average.
    Returns a new payload with the merged grid.
    """
    base = payloads[0]
    for p in payloads[1:]:
        if p.used_features != base.used_features:
            raise ValueError(
                "ensemble_mean: fold payloads have mismatched used_features; "
                f"first={base.used_features}, other={p.used_features}"
            )
    # Mean coefficients
    coefs = np.mean(
        np.stack([np.asarray(p.coefficients, dtype=float) for p in payloads], axis=0),
        axis=0,
    )
    # Common time grid = sorted union of all fold time grids
    all_times = sorted({float(t) for p in payloads for t in p.baseline_times})
    times_arr = np.asarray(all_times, dtype=float)
    if times_arr.size == 0:
        return _CoxPHFoldPayload(
            coefficients=tuple(float(c) for c in coefs),
            used_features=base.used_features,
            baseline_times=(),
            baseline_cumulative_hazard=(),
            baseline_survival=(),
        )
    # Interpolate each fold's H_0 to the common grid using the same
    # step-function semantics as _interp_cumulative_hazard.
    cumhaz_per_fold = []
    for p in payloads:
        cumhaz_per_fold.append(
            np.array([_interp_cumulative_hazard(p, t) for t in times_arr])
        )
    mean_cumhaz = np.mean(np.stack(cumhaz_per_fold, axis=0), axis=0)
    # Survival function from mean cumulative hazard: S(t) = exp(-H(t))
    mean_surv = np.exp(-mean_cumhaz)
    return _CoxPHFoldPayload(
        coefficients=tuple(float(c) for c in coefs),
        used_features=base.used_features,
        baseline_times=tuple(float(t) for t in times_arr),
        baseline_cumulative_hazard=tuple(float(h) for h in mean_cumhaz),
        baseline_survival=tuple(float(s) for s in mean_surv),
    )


# ── A2 / A6 sklearn-classifier wrapper (ensemble path) ──────────────


@dataclass
class EnsembleMeanProbaClassifier:
    """sklearn-shaped wrapper that averages ``predict_proba`` across
    multiple per-fold classifiers.

    Used by :func:`build_a2_from_heavy_ml` and
    :func:`build_a6_from_heavy_ml` when the caller requests
    :attr:`FoldSelectionStrategy.ENSEMBLE_MEAN`. For
    :attr:`FoldSelectionStrategy.LAST_FOLD` the underlying classifier
    is returned directly (no wrapper needed).
    """

    members: tuple[Any, ...]  # each member exposes predict_proba(X)

    def predict_proba(self, X) -> np.ndarray:
        if not self.members:
            raise RuntimeError(
                "EnsembleMeanProbaClassifier has no members; cannot predict"
            )
        proba_stack = np.stack(
            [np.asarray(m.predict_proba(X), dtype=float) for m in self.members],
            axis=0,
        )
        return np.mean(proba_stack, axis=0)


# ── Manifest IO ──────────────────────────────────────────────────────


def _read_manifest(manifest_path: Path) -> dict:
    p = Path(manifest_path)
    if not p.exists():
        raise HeavyMLManifestError(f"heavy_ml manifest not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    expected_schema = STEP5_MANIFEST_SCHEMA_VERSION
    schema = payload.get("schema_version")
    if schema != expected_schema:
        raise HeavyMLManifestError(
            f"heavy_ml manifest schema mismatch: expected "
            f"{expected_schema!r}, got {schema!r}"
        )
    sub_protocol = payload.get("sub_protocol")
    if sub_protocol != "heavy_ml_probe":
        raise HeavyMLManifestError(
            f"manifest sub_protocol must be 'heavy_ml_probe'; got {sub_protocol!r}"
        )
    return payload


def _resolve_path(manifest_path: Path, rel: str) -> Path:
    """Manifest paths are recorded relative to the manifest's parent."""
    return (Path(manifest_path).parent / rel).resolve()


def _assert_stage_ok(manifest: dict, stage: str) -> dict:
    """Look up ``stages.<stage>`` and assert ``status == "ok"``;
    return the stage dict so the caller can use its paths."""
    stages = manifest.get("stages") or {}
    info = stages.get(stage)
    if info is None:
        raise HeavyMLManifestError(
            f"manifest has no `stages.{stage}` entry; cannot build adapter"
        )
    status = info.get("status")
    if status != STAGE_STATUS_OK:
        raise StageUnavailableError(
            f"cannot build adapter from heavy_ml manifest: stage "
            f"{stage!r} status={status!r} (skip_reason="
            f"{info.get('skip_reason', '<unknown>')!r}). The upstream "
            f"pipeline must report `ok` for this stage before adapters "
            f"can consume it."
        )
    return info


def _load_classifier_manifest(
    heavy_ml_manifest: dict,
    stage: str,
    manifest_dir: Path,
) -> tuple[Path, dict]:
    """Locate + read the per-stage classifier manifest emitted by PR-C
    (meta_label) or PR-D (survival). Returns (classifier_dir, parsed_json)."""
    info = _assert_stage_ok(heavy_ml_manifest, stage)
    rel = info.get("classifier_manifest_path")
    if not rel:
        raise HeavyMLManifestError(
            f"manifest stages.{stage}.classifier_manifest_path is empty"
        )
    abs_path = _resolve_path(manifest_dir / "heavy_ml_manifest.json", rel)
    if not abs_path.exists():
        raise HeavyMLManifestError(
            f"classifier manifest for stage {stage!r} not found at {abs_path}"
        )
    return abs_path.parent, json.loads(abs_path.read_text(encoding="utf-8"))


# ── Builders ─────────────────────────────────────────────────────────


def _pick_classifier_folds(
    classifier_dir: Path,
    fold_list: list[dict],
    strategy: FoldSelectionStrategy,
) -> tuple[list[Any], list[dict]]:
    """Load + return ``(classifier_objects, fold_entries)`` per the
    strategy. Returns a list because ENSEMBLE_MEAN needs all valid
    folds; LAST_FOLD returns exactly one element."""
    valid = [
        f for f in fold_list
        if f.get("path") and f.get("sha256")
    ]
    if not valid:
        raise StageUnavailableError(
            "no valid classifier folds in manifest (every fold has "
            "path=null — upstream stage produced no fitted model)"
        )
    if strategy is FoldSelectionStrategy.LAST_FOLD:
        chosen = [valid[-1]]
    elif strategy is FoldSelectionStrategy.ENSEMBLE_MEAN:
        chosen = list(valid)
    elif strategy is FoldSelectionStrategy.FULL_REFIT:
        raise NotImplementedError(
            "FoldSelectionStrategy.FULL_REFIT is reserved for a future PR."
        )
    else:  # pragma: no cover
        raise ValueError(f"unknown fold strategy: {strategy!r}")

    loaded: list[Any] = []
    for f in chosen:
        path = classifier_dir / str(f["path"])
        loaded.append(joblib.load(path))
    return loaded, chosen


def _pick_classifier_object(
    objects: list[Any],
    strategy: FoldSelectionStrategy,
) -> Any:
    """Reduce a multi-fold load to a single predict_proba-shaped
    object. For LAST_FOLD: the single member. For ENSEMBLE_MEAN: an
    :class:`EnsembleMeanProbaClassifier` wrapping all members."""
    if not objects:
        raise StageUnavailableError("empty classifier load")
    if strategy is FoldSelectionStrategy.LAST_FOLD:
        return objects[0]
    if strategy is FoldSelectionStrategy.ENSEMBLE_MEAN:
        return EnsembleMeanProbaClassifier(members=tuple(objects))
    raise NotImplementedError(strategy)


def build_a2_from_heavy_ml(
    heavy_ml_manifest_path: Path,
    *,
    fold_selection_strategy: FoldSelectionStrategy = FoldSelectionStrategy.LAST_FOLD,
    threshold: float | None = None,
    config_id: str | None = None,
    **a2_kwargs: Any,
):
    """Build an :class:`A2Config` from the heavy_ml manifest's AutoML
    classifier.

    AutoML stage's per-fold classifiers are persisted by PR-B
    (``step_4/heavy_ml/classifiers/automl_*``) — at PR-E time PR-B
    actually persists only the fold-wise leaderboard and the inner
    estimator wrapper isn't yet on disk. This adapter therefore
    consumes the meta-label classifier manifest INSTEAD when AutoML
    classifiers aren't separately persisted, since the meta-label
    classifier IS the FLAML AutoML output (PR-C runs the same FLAML
    machine with the meta-label target).

    Implementation note for PR-E narrow scope: per dispatch §2.1 + Q5a,
    A2 consumes a ``predict_proba``-shaped classifier. PR-C's
    meta-label classifiers are FLAML inner estimators (e.g.
    ``LGBMClassifier`` from PR-C log §2.1). When chat wants A2 to
    consume the vanilla-target AutoML output instead, PR-B will need
    a separate persistence path (currently not implemented; this
    adapter intentionally points at the PR-C artefact set).
    """
    from core.architectures.a2_classifier_filter import A2Config

    mp = Path(heavy_ml_manifest_path)
    manifest = _read_manifest(mp)
    classifier_dir, clf_manifest = _load_classifier_manifest(
        manifest, "meta_label", mp.parent,
    )
    objects, _ = _pick_classifier_folds(
        classifier_dir, clf_manifest["folds"], fold_selection_strategy,
    )
    classifier = _pick_classifier_object(objects, fold_selection_strategy)

    # Threshold: prefer explicit override; otherwise default to 0.5
    # per L_PROTOCOL Amendment 2 §"A2" (AUC-best threshold default for
    # vanilla A2; heavy_ml_probe doesn't compute per-cluster AUC-best
    # thresholds at PR-E scope — surface as a future enhancement).
    thr = float(threshold) if threshold is not None else 0.5
    cid = config_id or f"a2_heavy_ml_t{thr:.4f}"

    # Feature order is the classifier's training feature set. For
    # FLAML inner estimators, this is the same column order passed to
    # AutoML.fit(); PR-E stores it in the meta_label manifest extras
    # (n_features_used) but not the column names themselves — the
    # heavy_ml_manifest's top-level `used_features` carries the names.
    used_features = tuple(manifest.get("used_features") or ())
    return A2Config(
        config_id=cid,
        classifier=classifier,
        threshold=thr,
        classifier_feature_order=used_features,
        **a2_kwargs,
    )


def build_a6_from_heavy_ml(
    heavy_ml_manifest_path: Path,
    *,
    fold_selection_strategy: FoldSelectionStrategy = FoldSelectionStrategy.LAST_FOLD,
    lower_threshold: float = 0.4,
    upper_threshold: float = 0.6,
    config_id: str | None = None,
    **a6_kwargs: Any,
):
    """Build an :class:`A6Config` from PR-C's meta-label classifier."""
    from core.architectures.a6_meta_labeling import A6Config

    mp = Path(heavy_ml_manifest_path)
    manifest = _read_manifest(mp)
    classifier_dir, clf_manifest = _load_classifier_manifest(
        manifest, "meta_label", mp.parent,
    )
    objects, _ = _pick_classifier_folds(
        classifier_dir, clf_manifest["folds"], fold_selection_strategy,
    )
    classifier = _pick_classifier_object(objects, fold_selection_strategy)

    used_features = tuple(manifest.get("used_features") or ())
    cid = config_id or f"a6_heavy_ml_{lower_threshold:.2f}_{upper_threshold:.2f}"
    return A6Config(
        config_id=cid,
        classifier=classifier,
        lower_threshold=float(lower_threshold),
        upper_threshold=float(upper_threshold),
        classifier_feature_order=used_features,
        **a6_kwargs,
    )


def build_a4_from_heavy_ml(
    heavy_ml_manifest_path: Path,
    *,
    fold_selection_strategy: FoldSelectionStrategy = FoldSelectionStrategy.LAST_FOLD,
    k_horizon: int = 5,
    exit_threshold: float = 0.4,
    config_id: str | None = None,
    **a4_kwargs: Any,
):
    """Build an :class:`A4Config` from PR-D's Cox PH survival folds.

    Returns a tuple ``(a4_config, cox_adapter)``:

      * ``a4_config`` is the :class:`A4Config` with ``classifier_fit``
        populated by a :class:`PathClassifierFit`-shaped wrapper around
        :class:`CoxPHAdapter`. A4's runtime consumes this via
        ``predict_admit`` per :mod:`core.architectures._path_classifier`.
      * ``cox_adapter`` is the bare :class:`CoxPHAdapter` for direct
        callers (chat-side analysis, tests). It exposes
        ``hazard_predict(features, bars_survived) → P(reach +1R in
        next K bars)`` per dispatch §2.2.

    The bundle pattern (not a single object) is the cleanest
    interpretation of dispatch §2.2's "A4Config object exposing a
    callable hazard_predict" within the narrow scope of PR-E (no
    changes to ``core/architectures/A4.py``). The A4Config stays
    schema-pure; chat-side code that wants to introspect the adapter's
    raw hazard math uses ``cox_adapter`` directly.
    """
    from core.architectures._path_classifier import PathClassifierFit
    from core.architectures.a4_pipeline_d_exits import A4Config

    mp = Path(heavy_ml_manifest_path)
    manifest = _read_manifest(mp)
    classifier_dir, clf_manifest = _load_classifier_manifest(
        manifest, "survival", mp.parent,
    )
    # Load survival per-fold pickles + lift into _CoxPHFoldPayload
    objects, _ = _pick_classifier_folds(
        classifier_dir, clf_manifest["folds"], fold_selection_strategy,
    )
    payloads = tuple(_coxph_payload_from_pickle(o) for o in objects)
    cox_adapter = CoxPHAdapter(
        payloads=payloads,
        strategy=fold_selection_strategy,
        k_horizon=int(k_horizon),
    )

    fit = PathClassifierFit(
        model=cox_adapter,
        threshold=float(exit_threshold),
        fit_auc=float("nan"),  # not available — Cox PH evaluated by concordance
        feature_order=cox_adapter.feature_order,
    )
    cid = config_id or f"a4_heavy_ml_k{int(k_horizon)}_t{exit_threshold:.2f}"
    config = A4Config(
        config_id=cid,
        classifier_fit=fit,
        exit_threshold=float(exit_threshold),
        per_trade_entry_features=None,
        **a4_kwargs,
    )
    return config, cox_adapter


__all__ = (
    "BARS_SURVIVED_FEATURE",
    "HEAVY_ML_MANIFEST_NAME",
    "STAGE_STATUS_FAILED",
    "STAGE_STATUS_OK",
    "STAGE_STATUS_SKIPPED",
    "STEP5_MANIFEST_SCHEMA_VERSION",
    "CoxPHAdapter",
    "EnsembleMeanProbaClassifier",
    "FoldSelectionStrategy",
    "HeavyMLManifestError",
    "StageUnavailableError",
    "build_a2_from_heavy_ml",
    "build_a4_from_heavy_ml",
    "build_a6_from_heavy_ml",
)
