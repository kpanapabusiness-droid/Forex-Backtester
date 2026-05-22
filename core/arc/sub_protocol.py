"""Sub-protocol hook (per L_PROTOCOL §5).

A sub-protocol is a named override layer that customises specific steps
of an arc. Examples: ``heavy_ml_probe`` replaces Step 4 default classifiers
with AutoML; ``signal_discovery_probe`` replaces Step 1 signal application
with rule-search.

CC_07 ships the hook but no actual sub-protocols. ``ArcOrchestrator``
consults this module for each step; when no sub-protocol is registered
under the arc's declared name, ``resolve_step_override`` returns None
and the vanilla overseer runs.

Adding a new sub-protocol:

  1. Add a definition file at ``docs/sub_protocols/<name>.md``.
  2. Add a Python module at ``core/sub_protocols/<name>.py`` exposing
     callables for the steps it overrides.
  3. Register it via :func:`register_sub_protocol` (typically at import).

Until step 2 happens, the registry is empty and vanilla overseer runs.
"""

from __future__ import annotations

from typing import Callable, Mapping

# Step override callables get the same arguments the vanilla step would
# get; their return type matches the step's normal output. The
# orchestrator does not introspect them — it just calls them.
StepOverride = Callable[..., object]


_REGISTRY: dict[str, Mapping[str, StepOverride]] = {}


def register_sub_protocol(
    name: str, overrides: Mapping[str, StepOverride]
) -> None:
    """Register a sub-protocol's step overrides.

    ``overrides`` maps step name ("step_1" .. "step_5") to the
    replacement callable. Steps not in the dict fall through to the
    vanilla overseer.

    Raises ValueError if ``name`` is already registered (overwrites
    must go through :func:`unregister_sub_protocol` first).
    """
    if name in _REGISTRY:
        raise ValueError(
            f"sub-protocol {name!r} already registered; unregister first"
        )
    _REGISTRY[name] = dict(overrides)


def unregister_sub_protocol(name: str) -> None:
    """Remove a sub-protocol from the registry. Idempotent."""
    _REGISTRY.pop(name, None)


def resolve_step_override(
    sub_protocol: str | None, step: str
) -> StepOverride | None:
    """Return the override callable for ``step`` under ``sub_protocol``, or None.

    ``sub_protocol=None`` always returns None (vanilla overseer).
    Unknown sub-protocol names raise KeyError so configuration errors
    surface immediately rather than silently falling through to vanilla.
    """
    if sub_protocol is None or sub_protocol == "vanilla":
        return None
    if sub_protocol not in _REGISTRY:
        raise KeyError(
            f"unknown sub-protocol {sub_protocol!r}; "
            f"registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[sub_protocol].get(step)


def registered_sub_protocols() -> tuple[str, ...]:
    """Return the names of every registered sub-protocol."""
    return tuple(sorted(_REGISTRY))


__all__ = (
    "StepOverride",
    "register_sub_protocol",
    "unregister_sub_protocol",
    "resolve_step_override",
    "registered_sub_protocols",
)
