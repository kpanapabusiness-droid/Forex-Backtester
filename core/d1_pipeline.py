"""Pipeline D1 hook for the WFO engine — PR 1 (plumbing + close-at-market)
+ PR 2 (per-archetype exit policies + per-fold classifiers).

At a single bar offset ``t`` (locked across all archetypes for an L arc),
score each archetype's classifier against the path-so-far feature vector
and decide:

* :class:`Close` — none of the archetypes cleared their threshold; the
  trade is closed at the next bar's open (the engine books the realised
  PnL after spread).
* :class:`ApplyPolicy` — at least one archetype cleared; carries the
  winning archetype's :class:`~core.exit_policies.ExitPolicy`. PR 1
  shipped this as an empty no-op; PR 2 makes it real — the engine calls
  ``policy.apply_at_accept(trade, ...)`` at the accept bar and
  ``policy.update_per_bar(trade, ...)`` on every subsequent bar.
* :class:`Hold` — reserved for future use; ``evaluate`` never returns this.

Configuration is loaded from YAML (see :func:`D1Hook.from_yaml_dict`). PR 2
adds two required blocks per archetype: ``per_fold_classifiers`` (mapping
fold id → joblib path) and ``exit_policy`` (type + parameters):

.. code-block:: yaml

    d1_archetypes:
      - label: stepwise_climber_arc4_cluster1
        bar_offset_t: 1
        decision_threshold: 0.1647
        feature_order: [body_to_range_ratio, ..., velocity_first_t]
        per_fold_classifiers:
          F2: artefacts/arc_4/refit_classifiers/fold_2.joblib
          F3: artefacts/arc_4/refit_classifiers/fold_3.joblib
          F4: artefacts/arc_4/refit_classifiers/fold_4.joblib
          F5: artefacts/arc_4/refit_classifiers/fold_5.joblib
          F6: artefacts/arc_4/refit_classifiers/fold_6.joblib
          F7: artefacts/arc_4/refit_classifiers/fold_7.joblib
        exit_policy:
          type: stepwise_climber
          archetype_sl_r: 1.0
          r_in_atr: 3.0
          mfe_lock_r: 1.0
          trail_from_high_r: 0.75

Trades whose fold id is not present in ``per_fold_classifiers`` fall
through with no D1 decision — used in Arc 4 to skip F1 (no training data
before the first OOS window). The set of folds gated is whatever the
YAML lists; missing folds are an explicit skip, not a runtime error.

See L_ARC_PROTOCOL.md §3 (Pipeline D1) and §11 (exit-family map).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from core.exit_policies import ExitPolicy, build_exit_policy
from core.features_path_so_far import ALL_FEATURE_KEYS, build_features_at_t

# ---------------------------------------------------------------------------
# Decision dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Close:
    """Trade is untradeable per all archetypes — close at next bar's open."""

    reason: str  # PR 1: always "d1_untradeable"


@dataclass(frozen=True)
class ApplyPolicy:
    """An archetype's classifier accepted the trade.

    PR 2: carries the winning archetype's :class:`~core.exit_policies.ExitPolicy`.
    The engine calls ``policy.apply_at_accept(trade, entry_px, atr)`` at
    bar ``t``, installs the policy on the trade dict, and on every
    subsequent bar calls ``policy.update_per_bar(trade, bar_path_row,
    entry_px, atr)`` to ratchet SL.

    ``label`` and ``probability`` remain informational.
    """

    policy: ExitPolicy | None = None  # None preserves PR 1 no-op semantics
    label: str = ""
    probability: float = float("nan")


@dataclass(frozen=True)
class Hold:
    """Reserved. Not emitted by D1Hook.evaluate in PR 1."""

    pass


# ---------------------------------------------------------------------------
# Classifier protocol — anything with predict_proba(shape=(n,2)) works.
# ---------------------------------------------------------------------------


class _ClassifierLike(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# Archetype config + hook.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class D1ArchetypeConfig:
    """One archetype's classifier set + decision threshold + feature schema.

    PR 2 schema change: instead of a single ``classifier`` /
    ``classifier_path``, each archetype carries ``per_fold_classifiers``
    — a mapping from fold id (e.g. ``"F2"``) to either a loaded
    classifier object or a joblib path. At evaluate time the trade's
    fold id selects the classifier. Plus an ``exit_policy`` that the
    engine installs on the trade when this archetype wins.

    ``per_fold_classifiers`` values may be:

    * a classifier object (with ``predict_proba``) — used directly,
      typical for tests.
    * a string / :class:`Path` — joblib-loaded by
      :func:`D1Hook.from_yaml_dict`. After loading the cache holds the
      live classifier; the path itself is kept on
      ``per_fold_classifier_paths`` for audit.
    """

    label: str
    feature_order: tuple[str, ...]
    decision_threshold: float
    bar_offset_t: int
    per_fold_classifiers: Mapping[str, Any] = field(default_factory=dict)
    per_fold_classifier_paths: Mapping[str, str] = field(default_factory=dict)
    exit_policy: ExitPolicy | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


class D1Hook:
    """Pipeline D1 hook — load once at engine start, call once per trade
    at the configured bar offset ``t``.

    Constructor validates that:

    * at least one archetype is configured
    * all archetypes share the same ``bar_offset_t`` (PR 1 constraint)
    * each archetype's ``feature_order`` matches the locked schema
    * each archetype's classifier (if attached) reports an expected
      feature count matching the length of ``feature_order``

    All validation failures raise :class:`ValueError` with a clear,
    actionable message.
    """

    def __init__(self, archetypes: Sequence[D1ArchetypeConfig]):
        if not archetypes:
            raise ValueError("D1Hook requires at least one archetype config")

        ts = {a.bar_offset_t for a in archetypes}
        if len(ts) > 1:
            raise ValueError(
                "PR 1 constraint: all archetypes must share bar_offset_t; "
                f"got {sorted(ts)}"
            )
        self._bar_offset_t = int(next(iter(ts)))

        known = set(ALL_FEATURE_KEYS)
        for a in archetypes:
            req = list(a.feature_order)
            if len(req) != len(set(req)):
                raise ValueError(
                    f"archetype '{a.label}': feature_order has duplicate keys"
                )
            req_set = set(req)
            unexpected = sorted(req_set - known)
            missing = sorted(known - req_set)
            if unexpected or missing:
                parts: list[str] = []
                if unexpected:
                    parts.append("unexpected: " + ", ".join(unexpected))
                if missing:
                    parts.append("missing: " + ", ".join(missing))
                raise ValueError(
                    f"archetype '{a.label}': feature_order does not match the "
                    "locked schema; " + "; ".join(parts)
                )

            if not a.per_fold_classifiers:
                raise ValueError(
                    f"archetype '{a.label}': per_fold_classifiers is empty. "
                    "PR 2 requires a per-fold classifier mapping (e.g. "
                    "{'F2': <classifier>, ...}). The PR 1 single-classifier "
                    "schema is no longer supported — see docstring for the "
                    "new YAML format."
                )

            for fold_key, clf in a.per_fold_classifiers.items():
                expected = getattr(clf, "n_features_in_", None)
                if expected is not None and int(expected) != len(req):
                    raise ValueError(
                        f"archetype '{a.label}' fold '{fold_key}': "
                        f"classifier expects {int(expected)} features but "
                        f"feature_order has {len(req)}"
                    )

            if not (0.0 <= float(a.decision_threshold) <= 1.0):
                raise ValueError(
                    f"archetype '{a.label}': decision_threshold "
                    f"{a.decision_threshold} not in [0, 1]"
                )
            if int(a.bar_offset_t) < 1:
                raise ValueError(
                    f"archetype '{a.label}': bar_offset_t must be >= 1; "
                    f"got {a.bar_offset_t}"
                )

        self._archetypes: tuple[D1ArchetypeConfig, ...] = tuple(archetypes)

    # ------------------------------------------------------------------
    @property
    def bar_offset_t(self) -> int:
        """The single ``t`` shared across all archetypes."""
        return self._bar_offset_t

    @property
    def archetypes(self) -> tuple[D1ArchetypeConfig, ...]:
        return self._archetypes

    # ------------------------------------------------------------------
    def evaluate(
        self,
        trade: Mapping[str, Any],
        bar_path_so_far: Sequence[Mapping[str, float]],
        entry_features: Mapping[str, float],
        t: int,
        cached: Mapping[str, Any] | None = None,
        fold_id: str | None = None,
    ) -> Close | ApplyPolicy | Hold:
        """Score every archetype at ``t``; pick max-P clearer, else Close.

        ``fold_id`` selects which per-fold classifier each archetype uses.
        If an archetype has no classifier registered for the trade's
        fold (e.g. Arc 4's F1, where there's no pre-OOS training data)
        that archetype abstains. If *no* archetype has a classifier for
        this fold, returns :class:`Hold` — the engine falls through to
        the legacy cascade. ``trade`` and ``cached`` are accepted for
        API symmetry with future policies that may read trade state.
        """
        del cached

        if int(t) != self._bar_offset_t:
            raise ValueError(
                f"D1Hook.evaluate called at t={t}; configured t={self._bar_offset_t}"
            )

        best_label: str | None = None
        best_p: float = -1.0
        best_policy: ExitPolicy | None = None
        any_classifier_available = False
        for arch in self._archetypes:
            clf = _select_classifier(arch, fold_id)
            if clf is None:
                continue
            any_classifier_available = True
            features = build_features_at_t(
                bar_path_so_far=bar_path_so_far,
                entry_features=entry_features,
                t=int(t),
                feature_order=list(arch.feature_order),
            )
            if not np.isfinite(features).all():
                # Missing warmup or invalid path — abstain on this
                # archetype. (A trade can still be accepted by another
                # archetype with valid features.)
                continue
            p = _score_positive_class(clf, features)
            if p >= float(arch.decision_threshold) and p > best_p:
                best_label = arch.label
                best_p = float(p)
                best_policy = arch.exit_policy

        if best_label is not None:
            return ApplyPolicy(
                policy=best_policy, label=best_label, probability=best_p
            )
        if not any_classifier_available:
            # No archetype has a classifier for this fold. Don't gate
            # the trade — let the engine apply its legacy cascade.
            return Hold()
        return Close(reason="d1_untradeable")

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml_dict(
        cls,
        cfg_block: Sequence[Mapping[str, Any]],
        project_root: Path | None = None,
    ) -> "D1Hook":
        """Build a hook from the ``d1_archetypes`` YAML block.

        Each archetype must carry:

        * ``per_fold_classifiers`` — mapping fold id → joblib path (or
          a pre-loaded classifier object for tests). Paths are resolved
          relative to ``project_root`` when not absolute. joblib import
          is lazy — only triggered when at least one path is present.
        * ``exit_policy`` — block with at least ``type`` plus the
          parameters for that policy (see :mod:`core.exit_policies`).

        The PR 1 single-classifier schema (``classifier_path`` /
        ``classifier`` at the archetype top level) is no longer accepted
        — using it raises a clear migration error.
        """
        if not cfg_block:
            raise ValueError("d1_archetypes block is empty")

        needs_joblib = False
        for item in cfg_block:
            if not isinstance(item, Mapping):
                continue
            for v in (item.get("per_fold_classifiers") or {}).values():
                if isinstance(v, (str, Path)):
                    needs_joblib = True
                    break
            if needs_joblib:
                break
        joblib_load = None
        if needs_joblib:
            try:
                import joblib  # type: ignore
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise ImportError(
                    "d1_archetypes config references per-fold classifier "
                    "paths but joblib is not installed"
                ) from exc
            joblib_load = joblib.load

        archetypes: list[D1ArchetypeConfig] = []
        for raw in cfg_block:
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"d1_archetypes entries must be mappings; got {type(raw).__name__}"
                )
            label = str(raw["label"])
            feature_order = tuple(str(k) for k in raw["feature_order"])
            decision_threshold = float(raw["decision_threshold"])
            bar_offset_t = int(raw["bar_offset_t"])

            if "classifier_path" in raw or (
                "classifier" in raw and "per_fold_classifiers" not in raw
            ):
                raise ValueError(
                    f"archetype '{label}': the PR 1 schema "
                    "('classifier_path' / single 'classifier') is no longer "
                    "supported. PR 2 requires a 'per_fold_classifiers' "
                    "mapping plus an 'exit_policy' block — see the "
                    "core.d1_pipeline module docstring for the new format."
                )

            raw_per_fold = raw.get("per_fold_classifiers")
            if not isinstance(raw_per_fold, Mapping) or not raw_per_fold:
                raise ValueError(
                    f"archetype '{label}': 'per_fold_classifiers' is "
                    "required and must be a non-empty mapping of fold id "
                    "to classifier (or joblib path)"
                )
            per_fold_classifiers: dict[str, Any] = {}
            per_fold_classifier_paths: dict[str, str] = {}
            for fold_key_raw, value in raw_per_fold.items():
                fold_key = str(fold_key_raw)
                if hasattr(value, "predict_proba"):
                    per_fold_classifiers[fold_key] = value
                    continue
                if value is None:
                    raise ValueError(
                        f"archetype '{label}' fold '{fold_key}': "
                        "missing classifier path (value is None)"
                    )
                path_str = str(value)
                per_fold_classifier_paths[fold_key] = path_str
                p = Path(path_str)
                if not p.is_absolute() and project_root is not None:
                    p = project_root / p
                assert joblib_load is not None  # guarded above
                per_fold_classifiers[fold_key] = joblib_load(p)

            raw_policy = raw.get("exit_policy")
            if not isinstance(raw_policy, Mapping):
                raise ValueError(
                    f"archetype '{label}': 'exit_policy' block is required "
                    "and must be a mapping with at least a 'type' key"
                )
            policy = build_exit_policy(raw_policy)

            archetypes.append(
                D1ArchetypeConfig(
                    label=label,
                    feature_order=feature_order,
                    decision_threshold=decision_threshold,
                    bar_offset_t=bar_offset_t,
                    per_fold_classifiers=per_fold_classifiers,
                    per_fold_classifier_paths=per_fold_classifier_paths,
                    exit_policy=policy,
                    extras={
                        k: v
                        for k, v in raw.items()
                        if k
                        not in {
                            "label",
                            "feature_order",
                            "decision_threshold",
                            "bar_offset_t",
                            "per_fold_classifiers",
                            "exit_policy",
                        }
                    },
                )
            )
        return cls(archetypes)


# ---------------------------------------------------------------------------
# Internals.
# ---------------------------------------------------------------------------


def _select_classifier(arch: D1ArchetypeConfig, fold_id: str | None) -> Any:
    """Return the per-fold classifier for ``fold_id``, or ``None`` if absent.

    ``None`` is the signal to abstain (the archetype has no classifier
    registered for the trade's fold — e.g. Arc 4's F1).
    """
    if fold_id is None:
        return None
    return arch.per_fold_classifiers.get(str(fold_id))


def _score_positive_class(classifier: Any, features: np.ndarray) -> float:
    """Return P(class=1) from a binary classifier's ``predict_proba``.

    Accepts the standard sklearn ``predict_proba`` output of shape
    ``(n, 2)``. Single-sample input is reshaped to ``(1, n_features)``.
    """
    X = np.asarray(features, dtype=np.float64).reshape(1, -1)
    proba = classifier.predict_proba(X)
    proba = np.asarray(proba)
    if proba.ndim != 2 or proba.shape[0] != 1 or proba.shape[1] < 2:
        raise ValueError(
            f"classifier.predict_proba returned shape {proba.shape}; "
            "expected (1, 2) for binary classification"
        )
    return float(proba[0, 1])


# ---------------------------------------------------------------------------
# Engine helper.
# ---------------------------------------------------------------------------


def apply_d1_hook_per_bar(
    hook: D1Hook | None,
    trade: dict[str, Any],
    j: int,
    entry_idx: int,
    cached: Mapping[str, Any],
    c_j: float,
    fold_id: str | None = None,
) -> tuple[float | None, int | None, str | None]:
    """Engine integration point — called once per (trade, bar) on the
    per-trade management loop.

    Returns ``(exit_px, exit_bar, exit_reason)`` when the hook decides to
    :class:`Close` the trade at bar ``j``, or ``(None, None, None)`` when
    the engine should fall through to its legacy exit cascade (hook is
    None, bar offset doesn't match, hook returned :class:`Hold`, or
    :class:`ApplyPolicy` installed a policy on the trade).

    On :class:`ApplyPolicy` the policy is installed via
    ``policy.apply_at_accept(trade, entry_px, atr)`` and stored on
    ``trade["exit_policy"]`` so subsequent bars can mutate SL via
    ``policy.update_per_bar``. The trade also gets:

    * ``trade["d1_decision"] = "apply_policy" | "close" | "no_d1"``
    * ``trade["classifier_fold_id"] = fold_id`` (for trades_all.csv)

    Fill resolution for Close mirrors the engine's standard next-bar-open
    pattern. When ``j + 1`` is past end-of-data the fill collapses to bar
    ``j``'s close.
    """
    if hook is None:
        return None, None, None
    bars_held = int(j) - int(entry_idx)
    if bars_held != hook.bar_offset_t:
        return None, None, None
    entry_features = trade.get("entry_features")
    if entry_features is None:
        # Trade was opened without entry features (e.g. KH-16 / KH-17
        # re-entry paths). Fall through to legacy exits.
        return None, None, None
    decision = hook.evaluate(
        trade=trade,
        bar_path_so_far=trade["bar_path"],
        entry_features=entry_features,
        t=bars_held,
        cached=cached,
        fold_id=fold_id,
    )
    trade["classifier_fold_id"] = fold_id
    if isinstance(decision, Close):
        trade["d1_decision"] = "close"
        opens = cached["o"]
        n = len(opens)
        if j + 1 < n:
            return float(opens[j + 1]), int(j + 1), decision.reason
        return float(c_j), int(j), decision.reason
    if isinstance(decision, ApplyPolicy):
        trade["d1_decision"] = "apply_policy"
        trade["d1_archetype_label"] = decision.label
        trade["d1_probability"] = float(decision.probability)
        if decision.policy is not None:
            decision.policy.apply_at_accept(
                trade=trade,
                entry_px=float(trade["entry_px"]),
                atr_at_entry=float(trade["atr"]),
            )
            trade["exit_policy"] = decision.policy
        return None, None, None
    # Hold — no classifier for this fold, no gating, fall through.
    trade["d1_decision"] = "no_d1"
    return None, None, None


__all__ = (
    "Close",
    "ApplyPolicy",
    "Hold",
    "D1ArchetypeConfig",
    "D1Hook",
    "apply_d1_hook_per_bar",
)
