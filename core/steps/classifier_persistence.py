"""Step 4 classifier persistence — loader + A2/A6 config builders.

Step 4 (`core.steps.step_4_extraction.run_step_4`) persists the
best-AUC fitted classifier per candidate cluster to disk when invoked
with a ``persistence_dir`` (see Step 4 module docstring). This module
is the read-side counterpart: SHA256-verified loader plus helpers that
instantiate :class:`A2Config` / :class:`A6Config` from a
:class:`Step4Result` without retraining.

Used by:

  - :mod:`core.arc.arc_orchestrator` to auto-wire A2/A6 architectures
    from Step 4 output at Step 5 dispatch.
  - Arc-local drivers (e.g. ``scripts/l_arc_*/run.py``) that prefer the
    canonical persisted-classifier flow over local prefit workarounds.

The classifier object never lives in memory beyond the request: each
``load_classifier(path)`` re-reads from disk and SHA256-verifies
against the manifest. Threshold sweeps must NOT retrain — they call
``build_a2_config_from_step4(..., threshold_override=t)`` against the
same persisted estimator.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import sklearn

from core.architectures.a2_classifier_filter import A2Config
from core.architectures.a6_meta_labeling import A6Config
from core.steps.step_4_extraction import Step4Result

try:
    import lightgbm as _lightgbm_mod  # type: ignore
    _LGBM_VERSION = str(_lightgbm_mod.__version__)
except ImportError:
    _LGBM_VERSION = ""


class ClassifierIntegrityError(RuntimeError):
    """Raised when a persisted classifier's SHA256 does not match the
    sibling ``manifest.json`` entry. Indicates either pickle corruption
    on disk or a manifest written against a different file."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(persistence_dir: Path) -> dict:
    manifest_path = persistence_dir / "manifest.json"
    if not manifest_path.exists():
        raise ClassifierIntegrityError(
            f"manifest.json not found in {persistence_dir}; was Step 4 "
            f"invoked with persistence_dir set?"
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _warn_on_version_drift(manifest: dict) -> None:
    """Soft signal — pickle compatibility across versions is the
    operator's responsibility. We don't refuse to load on drift; we
    just say so."""
    expected_joblib = manifest.get("joblib_version", "")
    expected_sklearn = manifest.get("sklearn_version", "")
    expected_lgbm = manifest.get("lightgbm_version", "")
    drifts = []
    if expected_joblib and expected_joblib != str(joblib.__version__):
        drifts.append(f"joblib {expected_joblib} -> {joblib.__version__}")
    if expected_sklearn and expected_sklearn != str(sklearn.__version__):
        drifts.append(f"sklearn {expected_sklearn} -> {sklearn.__version__}")
    if expected_lgbm and _LGBM_VERSION and expected_lgbm != _LGBM_VERSION:
        drifts.append(f"lightgbm {expected_lgbm} -> {_LGBM_VERSION}")
    if drifts:
        warnings.warn(
            "Loading Step 4 classifier under different library versions "
            f"than it was fit with: {', '.join(drifts)}. Predictions "
            "may differ from the manifest's recorded AUC.",
            UserWarning,
            stacklevel=3,
        )


def load_classifier(path: Path) -> Any:
    """Load a fitted classifier from a Step 4 persistence path.

    Verifies SHA256 against the sibling ``manifest.json`` before
    loading. Raises :class:`ClassifierIntegrityError` if the manifest
    cannot be read, the path is not listed in the manifest, or the
    file's SHA256 does not match. Emits a :class:`UserWarning` on
    joblib / sklearn / lightgbm version drift relative to the
    manifest's recorded environment; does not refuse the load.
    """
    path = Path(path)
    if not path.exists():
        raise ClassifierIntegrityError(f"classifier file not found: {path}")
    persistence_dir = path.parent
    manifest = _read_manifest(persistence_dir)
    classifiers = manifest.get("classifiers", {})
    # Locate the entry whose "path" matches this filename
    entry = next(
        (e for e in classifiers.values() if e.get("path") == path.name),
        None,
    )
    if entry is None:
        raise ClassifierIntegrityError(
            f"manifest in {persistence_dir} does not list {path.name}; "
            f"known: {sorted(e.get('path', '?') for e in classifiers.values())}"
        )
    actual_sha = _sha256_file(path)
    expected_sha = entry.get("sha256", "")
    if actual_sha != expected_sha:
        raise ClassifierIntegrityError(
            f"SHA256 mismatch for {path}: file={actual_sha[:16]}... "
            f"manifest={expected_sha[:16]}..."
        )
    _warn_on_version_drift(manifest)
    return joblib.load(path)


def _lookup_extraction(step4_result: Step4Result, cluster_id: int):
    """Return the ClusterExtraction for ``cluster_id`` from a
    Step4Result, or raise ValueError if absent."""
    for ce in step4_result.per_cluster:
        if int(ce.cluster_id) == int(cluster_id):
            return ce
    raise ValueError(
        f"cluster_id {cluster_id} not present in Step4Result; "
        f"available: {[int(c.cluster_id) for c in step4_result.per_cluster]}"
    )


def build_a2_config_from_step4(
    step4_result: Step4Result,
    cluster_id: int,
    *,
    config_id: str | None = None,
    threshold_override: float | None = None,
    **a2_kwargs: Any,
) -> A2Config:
    """Build :class:`A2Config` by loading the persisted classifier for
    ``cluster_id`` from Step 4.

    The classifier is loaded once (SHA256-verified) and bound directly
    to the returned config. Threshold sweeps that vary only the
    decision boundary MUST use ``threshold_override`` rather than
    re-fitting — the persisted estimator stays fixed; only the
    threshold field varies. Extra ``a2_kwargs`` pass through to
    :class:`A2Config` (e.g. ``sl_atr_mult``, ``risk_pct``,
    ``max_concurrent_per_pair``); fields not supplied take A2Config's
    defaults.
    """
    ce = _lookup_extraction(step4_result, cluster_id)
    if ce.fitted_classifier_path is None:
        raise ValueError(
            f"cluster {cluster_id} has no fitted classifier (Step 4 was "
            f"run without persistence_dir, or this cluster produced no "
            f"viable best-AUC classifier)"
        )
    classifier = load_classifier(ce.fitted_classifier_path)
    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else float(ce.best_threshold)
    )
    cid = config_id or f"a2_cluster{int(cluster_id)}_t{threshold:.4f}"
    return A2Config(
        config_id=cid,
        classifier=classifier,
        threshold=threshold,
        classifier_feature_order=ce.fitted_classifier_feature_order or (),
        **a2_kwargs,
    )


def build_a6_config_from_step4(
    step4_result: Step4Result,
    cluster_id: int,
    *,
    lower_threshold: float = 0.4,
    upper_threshold: float = 0.6,
    config_id: str | None = None,
    **a6_kwargs: Any,
) -> A6Config:
    """Build :class:`A6Config` by loading the persisted classifier for
    ``cluster_id`` from Step 4.

    Defaults match L_PROTOCOL Amendment 2 A6 spec (lower=0.4,
    upper=0.6). Threshold-pair sweep (the {(0.3,0.5), (0.4,0.6),
    (0.5,0.7)} grid from Amendment 2) is achieved by calling this
    builder three times with the same ``cluster_id`` and different
    ``lower_threshold`` / ``upper_threshold`` arguments — the
    underlying classifier is reused (loaded fresh each call;
    SHA256-verified each time, but no retraining).
    """
    ce = _lookup_extraction(step4_result, cluster_id)
    if ce.fitted_classifier_path is None:
        raise ValueError(
            f"cluster {cluster_id} has no fitted classifier (Step 4 was "
            f"run without persistence_dir, or this cluster produced no "
            f"viable best-AUC classifier)"
        )
    classifier = load_classifier(ce.fitted_classifier_path)
    cid = (
        config_id
        or f"a6_cluster{int(cluster_id)}_{lower_threshold:.2f}_{upper_threshold:.2f}"
    )
    return A6Config(
        config_id=cid,
        classifier=classifier,
        lower_threshold=float(lower_threshold),
        upper_threshold=float(upper_threshold),
        classifier_feature_order=ce.fitted_classifier_feature_order or (),
        **a6_kwargs,
    )


def build_a3_config_from_step4(
    step4_result: Step4Result,
    cluster_id: int,
    *,
    n_defer: int = 5,
    config_id: str | None = None,
    **a3_kwargs: Any,
) -> "A3Config":
    """Build :class:`A3Config` for the orchestrator's auto-spec path.

    A3 does NOT load a classifier from Step 4 — per L_PROTOCOL §2 Step 5
    "Architecture-specific retraining policy", A3 retrains per fold via
    :mod:`core.steps.path_classifier_per_fold`. This builder produces
    the config object with sane defaults; the orchestrator then threads
    per-fold fits via ``A1RunContext.path_classifier_fits``.

    The Step4Result is consumed only for cluster-existence sanity
    checking + config_id naming consistency with A2 / A6 builders.

    ``n_defer`` ∈ {1, 3, 5, 8} per L_PROTOCOL Appendix B.
    ``classifier_fit`` and ``per_trade_entry_features`` are left as
    ``None`` on the returned config — the orchestrator supplies both
    via run_context.
    """
    # Import deferred to break a circular import (a3_pipeline_de imports
    # from a1_system_level_filter which is import-stable; classifier_persistence
    # otherwise stays in a small dependency tree).
    from core.architectures.a3_pipeline_de import A3Config

    _lookup_extraction(step4_result, cluster_id)  # sanity-check existence
    cid = config_id or f"a3_cluster{int(cluster_id)}_n{int(n_defer)}"
    return A3Config(
        config_id=cid,
        n_defer=int(n_defer),
        classifier_fit=None,
        per_trade_entry_features=None,
        **a3_kwargs,
    )


def build_a4_config_from_step4(
    step4_result: Step4Result,
    cluster_id: int,
    *,
    exit_threshold: float = 0.4,
    config_id: str | None = None,
    **a4_kwargs: Any,
) -> "A4Config":
    """Build :class:`A4Config` for the orchestrator's auto-spec path.

    Like A3 (above), A4 retrains per fold — Step4Result is consumed
    only for cluster sanity-check + config_id naming. ``exit_threshold``
    ∈ {0.3, 0.4, 0.5} per L_PROTOCOL Amendment 2 §"A4".
    """
    from core.architectures.a4_pipeline_d_exits import A4Config

    _lookup_extraction(step4_result, cluster_id)
    cid = config_id or f"a4_cluster{int(cluster_id)}_t{exit_threshold:.2f}"
    return A4Config(
        config_id=cid,
        classifier_fit=None,
        exit_threshold=float(exit_threshold),
        per_trade_entry_features=None,
        **a4_kwargs,
    )


__all__ = (
    "ClassifierIntegrityError",
    "load_classifier",
    "build_a2_config_from_step4",
    "build_a3_config_from_step4",
    "build_a4_config_from_step4",
    "build_a6_config_from_step4",
)
