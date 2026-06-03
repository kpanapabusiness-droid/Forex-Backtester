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

Registered policies (alphabetised by name):

  * ``sl_only``
  * ``sl_partial_close_1r_runner_trail``
  * ``sl_plus_tp_2r``
  * ``sl_plus_tp_3r``
  * ``sl_plus_trailing_atr``
  * ``sl_plus_trailing_swing``

See [docs/PROTOCOL_RUNTIME.md §8c][] for the full catalogue and
semantic spec. The post-hoc path-replay scorer was retired 2026-06-02 —
``MultiPairBacktester`` is the sole engine that scores a trade (see
RESET_MANIFEST.md and docs/ARC_10_GATE_FIDELITY_DEFECT.md).
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
    "available_policies",
    "build_exit_policy",
)
