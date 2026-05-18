"""SignalAdapter protocol for engine signal pluggability.

Used by scripts/phase_kgl_v2_4h_wfo.py to decouple the signal definition
from the engine. Each L arc / system can supply its own adapter via
``signal_adapter`` / ``signal_adapter_kwargs`` config keys.

Contract:

- ``required_aux_data()``: declares aux data keys ("h1", "d1") the engine
  must load. Unknown keys are rejected loudly at startup.
- ``compute_signal_mask(df, aux)``: returns ``(mask, extras)``. ``mask`` is
  an int/bool array, length len(df), 1 at signal-bar close where an entry
  should fire. ``extras`` is a dict of signal-specific cache data that the
  engine merges into the per-pair cache (e.g. KH-24's d1_dist_ratio_by_bar).
  Returning extras as part of the call (rather than a separate query) keeps
  the cache build single-pass.
- ``compute_signal_atr(df, idx, aux)``: 1R-anchor ATR at signal bar.
- ``compute_entry_features(df, idx, aux)``: signal-bar feature dict consumed
  by the Pipeline D1 hook (see core/d1_pipeline.py).
- ``per_trade_init(trade, df, idx, aux)``: stamp signal-specific fields
  onto the trade dict (e.g. KH-24's kh13_triggered/kh14_triggered/
  first_bar_dir initial values). Called only from the primary entry
  path. Adapters whose features must populate on re-entry paths need
  to extend those paths separately — KH-24 re-entry / KH-17 paths
  preserve inline init for byte-identity.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class SignalAdapter(Protocol):
    """Minimal plug-in signal contract. See module docstring for shape."""

    def required_aux_data(self) -> list[str]:
        ...

    def compute_signal_mask(
        self,
        df_signal_tf: pd.DataFrame,
        aux: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        ...

    def compute_signal_atr(
        self,
        df_signal_tf: pd.DataFrame,
        idx: int,
        aux: dict[str, Any],
    ) -> float:
        ...

    def compute_entry_features(
        self,
        df_signal_tf: pd.DataFrame,
        idx: int,
        aux: dict[str, Any],
    ) -> dict[str, float]:
        ...

    def per_trade_init(
        self,
        trade: dict[str, Any],
        df_signal_tf: pd.DataFrame,
        idx: int,
        aux: dict[str, Any],
    ) -> None:
        ...


ALLOWED_AUX_KEYS: frozenset[str] = frozenset({"h1", "d1"})


def import_class(spec: str) -> type:
    """Load a class from a 'module.path:ClassName' spec string.

    Fails loud at startup with a clear message if the module or class is
    not importable — preferred over a silent fallback to a default adapter.
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ImportError(
            f"signal_adapter spec must be of form 'module.path:ClassName', "
            f"got {spec!r}"
        )
    module_path, class_name = spec.split(":", 1)
    module_path = module_path.strip()
    class_name = class_name.strip()
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"signal_adapter: cannot import module '{module_path}' "
            f"(from spec '{spec}'): {e}"
        ) from e
    if not hasattr(module, class_name):
        raise ImportError(
            f"signal_adapter: class '{class_name}' not found in module "
            f"'{module_path}' (from spec '{spec}')"
        )
    return getattr(module, class_name)


def validate_aux_declaration(declared: list[str], spec: str) -> None:
    """Ensure adapter-declared aux keys are within the recognised set."""
    unknown = [k for k in declared if k not in ALLOWED_AUX_KEYS]
    if unknown:
        raise ValueError(
            f"signal_adapter '{spec}' declared unknown aux keys: {unknown}. "
            f"Allowed: {sorted(ALLOWED_AUX_KEYS)}"
        )
