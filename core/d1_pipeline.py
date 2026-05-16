"""Pipeline D1 hook for the WFO engine — PR 1 (plumbing + close-at-market).

At a single bar offset ``t`` (locked across all archetypes for an L arc),
score each archetype's classifier against the path-so-far feature vector
and decide:

* :class:`Close` — none of the archetypes cleared their threshold; the
  trade is closed at the next bar's open (the engine books the realised
  PnL after spread).
* :class:`ApplyPolicy` — at least one archetype cleared; the winning
  archetype's exit policy applies. PR 1 ships this as a no-op so the
  trade continues with the engine's standard exit cascade (SL / trail /
  kijun_d1). PR 2 adds per-archetype SL / trail / TP mutations.
* :class:`Hold` — reserved for future use; PR 1 never returns this from
  :meth:`D1Hook.evaluate`.

Configuration is loaded from YAML (see :func:`D1Hook.from_yaml_dict`):

.. code-block:: yaml

    d1_archetypes:
      - label: stepwise_climber
        classifier_path: artefacts/arc_3/stepwise_climber_D1.joblib
        feature_order: [body_to_range_ratio, ..., velocity_first_t]
        decision_threshold: 0.55
        bar_offset_t: 3

See L_ARC_PROTOCOL.md §3 (Pipeline D1) and §11 (exit-family map).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

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

    PR 1: empty payload — engine continues with standard exits. PR 2 will
    carry SL / trail / TP mutations keyed on the winning archetype.
    """

    label: str = ""  # winning archetype label; informational in PR 1
    probability: float = float("nan")  # winning archetype P; informational in PR 1


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
    """One archetype's classifier + decision threshold + feature schema.

    ``classifier`` can be supplied directly (e.g. a stub in tests) — in
    which case ``classifier_path`` is informational only. When loaded via
    :func:`D1Hook.from_yaml_dict`, ``classifier_path`` is joblib-loaded.
    """

    label: str
    feature_order: tuple[str, ...]
    decision_threshold: float
    bar_offset_t: int
    classifier: Any = None  # _ClassifierLike at runtime
    classifier_path: str | None = None
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

            if a.classifier is not None:
                expected = getattr(a.classifier, "n_features_in_", None)
                if expected is not None and int(expected) != len(req):
                    raise ValueError(
                        f"archetype '{a.label}': classifier expects "
                        f"{int(expected)} features but feature_order has "
                        f"{len(req)}"
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
    ) -> Close | ApplyPolicy | Hold:
        """Score every archetype at ``t``; pick max-P clearer, else Close.

        ``trade`` and ``cached`` are accepted for API symmetry with PR 2 (where
        archetype policies may read trade state); PR 1 ignores them.
        """
        del trade, cached

        if int(t) != self._bar_offset_t:
            raise ValueError(
                f"D1Hook.evaluate called at t={t}; configured t={self._bar_offset_t}"
            )

        best_label: str | None = None
        best_p: float = -1.0
        best_threshold: float = 0.0
        for arch in self._archetypes:
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
            p = _score_positive_class(arch.classifier, features)
            if p >= float(arch.decision_threshold) and p > best_p:
                best_label = arch.label
                best_p = float(p)
                best_threshold = float(arch.decision_threshold)

        if best_label is not None:
            return ApplyPolicy(label=best_label, probability=best_p)

        del best_threshold  # unused in PR 1; PR 2 may surface it on Close.
        return Close(reason="d1_untradeable")

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml_dict(
        cls,
        cfg_block: Sequence[Mapping[str, Any]],
        project_root: Path | None = None,
    ) -> "D1Hook":
        """Build a hook from the ``d1_archetypes`` YAML block.

        ``classifier_path`` is resolved relative to ``project_root`` if
        provided and the path is not absolute. joblib loading is lazy —
        we only import joblib when at least one classifier_path is set.
        """
        if not cfg_block:
            raise ValueError("d1_archetypes block is empty")

        needs_joblib = any(
            isinstance(item, Mapping) and item.get("classifier_path")
            for item in cfg_block
        )
        joblib_load = None
        if needs_joblib:
            try:
                import joblib  # type: ignore
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise ImportError(
                    "d1_archetypes config references classifier_path but "
                    "joblib is not installed"
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
            classifier_path_raw = raw.get("classifier_path")
            classifier_path = (
                str(classifier_path_raw) if classifier_path_raw is not None else None
            )
            classifier = raw.get("classifier")  # tests inject directly
            if classifier is None and classifier_path is not None:
                p = Path(classifier_path)
                if not p.is_absolute() and project_root is not None:
                    p = project_root / p
                assert joblib_load is not None  # guarded above
                classifier = joblib_load(p)

            archetypes.append(
                D1ArchetypeConfig(
                    label=label,
                    feature_order=feature_order,
                    decision_threshold=decision_threshold,
                    bar_offset_t=bar_offset_t,
                    classifier=classifier,
                    classifier_path=classifier_path,
                    extras={
                        k: v
                        for k, v in raw.items()
                        if k
                        not in {
                            "label",
                            "feature_order",
                            "decision_threshold",
                            "bar_offset_t",
                            "classifier_path",
                            "classifier",
                        }
                    },
                )
            )
        return cls(archetypes)


# ---------------------------------------------------------------------------
# Internals.
# ---------------------------------------------------------------------------


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
    trade: Mapping[str, Any],
    j: int,
    entry_idx: int,
    cached: Mapping[str, Any],
    c_j: float,
) -> tuple[float | None, int | None, str | None]:
    """Engine integration point — called once per (trade, bar) on the
    per-trade management loop.

    Returns ``(exit_px, exit_bar, exit_reason)`` when the hook decides to
    Close the trade at bar ``j``, or ``(None, None, None)`` when the engine
    should fall through to its legacy exit cascade (hook is None, bar
    offset doesn't match, or hook returned ApplyPolicy / Hold).

    Fill resolution mirrors the engine's standard next-bar-open pattern
    (see scripts/phase_kgl_v2_4h_wfo.py:1376-1384). When ``j + 1`` is past
    end-of-data the fill collapses to bar ``j``'s close — same as every
    other engine exit branch.
    """
    if hook is None:
        return None, None, None
    bars_held = int(j) - int(entry_idx)
    if bars_held != hook.bar_offset_t:
        return None, None, None
    entry_features = trade.get("entry_features")
    if entry_features is None:
        # Trade was opened without entry features (e.g. KH-16 / KH-17
        # re-entry paths in PR 1). Fall through to legacy exits.
        return None, None, None
    decision = hook.evaluate(
        trade=trade,
        bar_path_so_far=trade["bar_path"],
        entry_features=entry_features,
        t=bars_held,
        cached=cached,
    )
    if isinstance(decision, Close):
        opens = cached["o"]
        n = len(opens)
        if j + 1 < n:
            return float(opens[j + 1]), int(j + 1), decision.reason
        return float(c_j), int(j), decision.reason
    # ApplyPolicy / Hold — PR 1 no-op, fall through to legacy exits.
    return None, None, None


__all__ = (
    "Close",
    "ApplyPolicy",
    "Hold",
    "D1ArchetypeConfig",
    "D1Hook",
    "apply_d1_hook_per_bar",
)
