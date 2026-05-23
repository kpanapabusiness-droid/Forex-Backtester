"""Holdout re-run at scaled risk per Amendment 3 §"Engine-side changes" #4.

For each top-K candidate that reaches the gate stage:

  1. Compute ``k_safe`` / ``k_hard`` from worst-fold DD at ``r_base``.
  2. Re-scale the candidate's ``arch_config.risk_pct`` by ``k_safe``
     (and/or ``k_hard``) — multiplies the sizing only; everything else
     in the config is unchanged.
  3. Re-run the holdout fold ONCE per scaled config (one sim per
     scale tier).
  4. Result: ``FoldStats`` at the scaled risk, consumed by
     ``core.wfo.amended_gates.classify_amended_fold_stats``.

This module provides a small generic helper that does the rescaling
via ``dataclasses.replace`` — the supplied ``arch_config`` must be a
frozen dataclass with a ``risk_pct: float`` field (A1Config / A2Config
/ ... / A6Config all conform). Two separate manifest entries get
emitted by the caller (one per scaled holdout sim) per Amendment 3
§5.5 "Determinism".
"""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from typing import Any


def rescale_arch_config_risk(
    arch_config: Any,
    *,
    k_scale: float,
) -> Any:
    """Return a copy of ``arch_config`` with ``risk_pct`` multiplied by ``k_scale``.

    Used by the holdout re-run pipeline to evaluate a candidate at
    ``r_safe`` (``k_scale = k_safe``) or ``r_hard`` (``k_scale = k_hard``).

    Raises ``TypeError`` if ``arch_config`` is not a frozen dataclass
    with a ``risk_pct`` field. Raises ``ValueError`` on non-finite
    ``k_scale``.
    """
    if not (k_scale == k_scale and k_scale > 0 and k_scale != float("inf")):
        raise ValueError(f"k_scale must be finite positive; got {k_scale!r}")
    if not is_dataclass(arch_config):
        raise TypeError(
            f"arch_config must be a dataclass; got {type(arch_config).__name__}"
        )
    if not hasattr(arch_config, "risk_pct"):
        raise TypeError(
            f"{type(arch_config).__name__} must have a 'risk_pct' field"
        )
    base = float(arch_config.risk_pct)
    scaled = base * float(k_scale)
    # ``replace`` works on frozen dataclasses too — returns a fresh
    # instance with the updated field.
    new_config_id = f"{arch_config.config_id}_r{scaled:.4f}" if hasattr(arch_config, "config_id") else None
    kwargs: dict[str, Any] = {"risk_pct": scaled}
    if new_config_id is not None:
        kwargs["config_id"] = new_config_id
    return replace(arch_config, **kwargs)


__all__ = ("rescale_arch_config_risk",)
