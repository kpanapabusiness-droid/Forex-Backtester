"""KbExhaustionBarAdapter — SignalAdapter wrapping the KH-24 signal.

KH-24 is a long-running, locked-in-production signal whose mask logic
lives in ``scripts.phase_kgl_v2_4h_wfo._build_signal_series``. To preserve
byte-identical behaviour while exposing the SignalAdapter contract, this
adapter delegates to the engine helper rather than re-implementing the
logic. Future arc adapters (Arc 4+) will be self-contained.

The engine populates its parameter globals (ATR_PERIOD, BODY_THRESH, etc.)
from cfg in ``main()``. Those globals are read by ``_build_signal_series``
at call time, so the adapter doesn't need to thread them through.

Required aux: ``["h1", "d1"]`` — both are used by the KH-24 system
(D1 for C8/C9 regime gates inside the signal mask; H1 for the engine-level
KH-22/24 close-in-range filter at entry time).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.features_path_so_far import build_entry_features_at_signal_bar
from signals.kb_exhaustion_bar import _wilder_atr


class KbExhaustionBarAdapter:
    """Adapter implementing the SignalAdapter contract for KH-24."""

    def __init__(self, **kwargs: Any) -> None:
        # KH-24 sources its parameters from cfg via the engine's module
        # globals (set in main()). kwargs is reserved for future signals
        # but accepted-and-ignored here so the engine can pass an empty
        # ``signal_adapter_kwargs: {}`` block without raising.
        self._extra_kwargs = dict(kwargs)
        # Per-df ATR cache: signal_bar ATR is recomputed many times across
        # a run; cache by df identity to keep compute_signal_atr O(1).
        self._atr_cache: dict[int, np.ndarray] = {}

    def required_aux_data(self) -> list[str]:
        # h1: KH-22/24 close-in-range entry filter (engine-level).
        # d1: C8/C9 regime gates inside the signal mask.
        return ["h1", "d1"]

    def compute_signal_mask(
        self,
        df_signal_tf: pd.DataFrame,
        aux: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Deferred import: the engine imports this adapter at module load,
        # so a top-level import would be circular. When the engine is run
        # as __main__, scripts/phase_kgl_v2_4h_wfo.py aliases its module
        # under the package path so this import returns the SAME module
        # whose globals main() populated (otherwise cfg overrides like
        # NO_VOLUME_FILTER would silently revert to defaults here).
        from scripts.phase_kgl_v2_4h_wfo import _build_signal_series

        d1_filter = aux.get("d1_filter")
        baseline_override = aux.get("baseline_override")
        min_warmup = aux.get("min_warmup")

        sig, n_blocked, n_passed, n_cond9, cond9_dates, d1_dist_ratios = (
            _build_signal_series(
                df_signal_tf,
                d1_filter,
                baseline_override=baseline_override,
                min_warmup=min_warmup,
            )
        )

        extras: dict[str, Any] = {
            "n_d1_blocked":         n_blocked,
            "n_d1_passed":          n_passed,
            "n_cond9_blocked":      n_cond9,
            "cond9_block_dates":    cond9_dates,
            "d1_dist_ratio_by_bar": dict(d1_dist_ratios),
        }
        return sig, extras

    def _get_atr_array(self, df_signal_tf: pd.DataFrame) -> np.ndarray:
        key = id(df_signal_tf)
        arr = self._atr_cache.get(key)
        if arr is None:
            arr = _wilder_atr(df_signal_tf, 14).values
            self._atr_cache[key] = arr
        return arr

    def compute_signal_atr(
        self,
        df_signal_tf: pd.DataFrame,
        idx: int,
        aux: dict[str, Any],
    ) -> float:
        return float(self._get_atr_array(df_signal_tf)[idx])

    def compute_entry_features(
        self,
        df_signal_tf: pd.DataFrame,
        idx: int,
        aux: dict[str, Any],
    ) -> dict[str, float]:
        atr_at_signal = self.compute_signal_atr(df_signal_tf, idx, aux)
        return build_entry_features_at_signal_bar(
            open_arr=df_signal_tf["open"].values.astype(float),
            high_arr=df_signal_tf["high"].values.astype(float),
            low_arr=df_signal_tf["low"].values.astype(float),
            close_arr=df_signal_tf["close"].values.astype(float),
            atr_at_signal_bar=float(atr_at_signal),
            signal_bar_idx=int(idx),
        )

    def per_trade_init(
        self,
        trade: dict[str, Any],
        df_signal_tf: pd.DataFrame,
        idx: int,
        aux: dict[str, Any],
    ) -> None:
        # KH-24-specific trade fields. The engine's primary entry path
        # already initialises these inline (kept for compatibility with
        # the re-entry paths that don't route through per_trade_init);
        # we stamp them again here so the adapter contract is honoured
        # and any field-by-field changes only need to happen in one place.
        entry_idx = idx + 1
        n = len(df_signal_tf)
        if entry_idx < n:
            entry_open = float(df_signal_tf["open"].iloc[entry_idx])
            entry_close = float(df_signal_tf["close"].iloc[entry_idx])
            if entry_close > entry_open:
                fbd = 1
            elif entry_close < entry_open:
                fbd = -1
            else:
                fbd = 0
        else:
            fbd = 0
        trade.setdefault("first_bar_dir", fbd)
        trade.setdefault("kh13_triggered", False)
        trade.setdefault("kh14_triggered", False)
