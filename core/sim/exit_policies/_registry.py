"""Factory: name → ExitPolicy instance.

The registry is the single source of truth for the canonical exit-policy
namespace. Architectures and arcs select a policy by name (string in
``arch_config.exit_policy``); the factory returns a fresh policy
instance per call.

Policy instances are stateless (per-position state lives in
:class:`ExitPolicyState` constructed via ``make_state``), so the same
policy instance is safe to share across positions — the factory
returns fresh instances primarily for clean test isolation and to
keep the wiring obvious.
"""

from __future__ import annotations

from core.sim.exit_policies._base import ExitPolicy
from core.sim.exit_policies.sl_only import SlOnlyPolicy
from core.sim.exit_policies.sl_partial_close_1r_runner_trail import (
    SlPartialClose1RRunnerTrailPolicy,
)
from core.sim.exit_policies.sl_plus_tp_2r import SlPlusTp2RPolicy
from core.sim.exit_policies.sl_plus_tp_3r import SlPlusTp3RPolicy
from core.sim.exit_policies.sl_plus_trailing_atr import SlPlusTrailingAtrPolicy
from core.sim.exit_policies.sl_plus_trailing_swing import SlPlusTrailingSwingPolicy

_REGISTRY: dict[str, type[ExitPolicy]] = {
    SlOnlyPolicy.name: SlOnlyPolicy,
    SlPlusTp2RPolicy.name: SlPlusTp2RPolicy,
    SlPlusTp3RPolicy.name: SlPlusTp3RPolicy,
    SlPlusTrailingAtrPolicy.name: SlPlusTrailingAtrPolicy,
    SlPlusTrailingSwingPolicy.name: SlPlusTrailingSwingPolicy,
    SlPartialClose1RRunnerTrailPolicy.name: SlPartialClose1RRunnerTrailPolicy,
}


def build_exit_policy(name: str) -> ExitPolicy:
    """Construct a fresh policy instance by registry name.

    Raises ``KeyError`` with the available-policy list if the name is
    unknown — typo-loud rather than silently falling back to a default.
    """
    try:
        cls = _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"Unknown exit policy {name!r}. Available: {available}"
        ) from None
    return cls()


def available_policies() -> tuple[str, ...]:
    """Sorted tuple of registered policy names."""
    return tuple(sorted(_REGISTRY))


__all__ = ("build_exit_policy", "available_policies")
