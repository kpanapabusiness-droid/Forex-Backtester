"""Canonical exit-policy registry for the v3 multipair backtester.

Public surface:

  * :class:`ExitPolicy` — abstract base for all policies.
  * :class:`ExitPolicyContext` — per-position inputs.
  * :class:`ExitPolicyDecision` — policy verdict at a bar.
  * :class:`ExitAction` — FULL_CLOSE / PARTIAL_CLOSE enum.
  * :class:`ExitPolicyState` — abstract per-position runtime state.
  * :class:`NullPolicyState` — no-op state for stateless policies.
  * :func:`build_exit_policy` — factory keyed by policy name.
  * :func:`available_policies` — sorted tuple of registered names.
  * :func:`simulate_path` — post-hoc path replay (for legacy Step 5
    scripts) with the same semantics as the live engine policies.
  * :func:`available_path_simulators` — sorted tuple of policies with
    a registered path simulator.

Registered policies (alphabetised by name):

  * ``sl_only``
  * ``sl_partial_close_1r_runner_trail``
  * ``sl_plus_tp_2r``
  * ``sl_plus_tp_3r``
  * ``sl_plus_trailing_atr``
  * ``sl_plus_trailing_swing``

See [docs/PROTOCOL_RUNTIME.md §8c][] for the full catalogue and
semantic spec. Reference implementations: [scripts/l_arc_10_v3/step_5.py:99-253][].
"""

from core.sim.exit_policies._base import (
    ExitAction,
    ExitPolicy,
    ExitPolicyContext,
    ExitPolicyDecision,
    ExitPolicyState,
    NullPolicyState,
)
from core.sim.exit_policies._registry import available_policies, build_exit_policy
from core.sim.exit_policies.path_simulate import (
    available_path_simulators,
    simulate_path,
    simulate_pool_approximation,
)
from core.sim.exit_policies.sl_only import SlOnlyPolicy
from core.sim.exit_policies.sl_partial_close_1r_runner_trail import (
    PartialCloseRunnerTrailState,
    SlPartialClose1RRunnerTrailPolicy,
)
from core.sim.exit_policies.sl_plus_tp_2r import SlPlusTp2RPolicy
from core.sim.exit_policies.sl_plus_tp_3r import SlPlusTp3RPolicy
from core.sim.exit_policies.sl_plus_trailing_atr import (
    SlPlusTrailingAtrPolicy,
    TrailingAtrState,
)
from core.sim.exit_policies.sl_plus_trailing_swing import (
    SlPlusTrailingSwingPolicy,
    TrailingSwingState,
)

__all__ = (
    "ExitAction",
    "ExitPolicy",
    "ExitPolicyContext",
    "ExitPolicyDecision",
    "ExitPolicyState",
    "NullPolicyState",
    "PartialCloseRunnerTrailState",
    "SlOnlyPolicy",
    "SlPartialClose1RRunnerTrailPolicy",
    "SlPlusTp2RPolicy",
    "SlPlusTp3RPolicy",
    "SlPlusTrailingAtrPolicy",
    "SlPlusTrailingSwingPolicy",
    "TrailingAtrState",
    "TrailingSwingState",
    "available_path_simulators",
    "available_policies",
    "build_exit_policy",
    "simulate_path",
    "simulate_pool_approximation",
)
