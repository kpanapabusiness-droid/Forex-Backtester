"""Feature registry — global lookup of declared ``FeatureSpec``s.

Each feature class module (``core.features.session``, ``cross_pair``,
etc.) registers its features at import time via ``register(spec)``. The
pipeline orchestrator queries ``all_specs()`` to compute the full
feature matrix.

Determinism: ``all_specs()`` returns specs sorted by name so output
column ordering is stable across imports.
"""

from __future__ import annotations

from core.features.lineage import FeatureSpec

_REGISTRY: dict[str, FeatureSpec] = {}


def register(spec: FeatureSpec) -> FeatureSpec:
    """Register ``spec`` under its name. Raises ValueError on collision."""
    if spec.name in _REGISTRY:
        existing = _REGISTRY[spec.name]
        raise ValueError(
            f"Feature {spec.name!r} already registered (existing class={existing.feature_class!r})"
        )
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> FeatureSpec:
    """Return the spec for ``name``. Raises KeyError if absent."""
    if name not in _REGISTRY:
        raise KeyError(f"No registered feature named {name!r}")
    return _REGISTRY[name]


def all_specs() -> tuple[FeatureSpec, ...]:
    """Return all registered specs, sorted by name for deterministic ordering."""
    return tuple(sorted(_REGISTRY.values(), key=lambda s: s.name))


def clear() -> None:
    """Test-only: wipe the registry."""
    _REGISTRY.clear()
