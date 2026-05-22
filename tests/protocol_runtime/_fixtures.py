"""Shared synthetic fixtures for protocol_runtime tests.

Builds small in-memory panels + signal modules that exercise the runtime
without depending on the real HistData layer (which is absent in CI /
worktrees by default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from core.arc.signal_protocol import PerPairSignalState, SignalEvaluation, SignalModule
from core.sim.panel import Panel


def _synthetic_pair_df(
    n_bars: int = 500, start: str = "2018-01-01", seed: int = 42, pair: str = "EURUSD"
) -> pd.DataFrame:
    """Build a synthetic H4 OHLC frame with deterministic random walk.

    Schema matches the v3 backtester: open/high/low/close per side (bid + ask).
    Spread is constant 2 pips so fill primitives work.
    """
    rng = np.random.default_rng(seed + hash(pair) % 1000)
    n = n_bars
    drift = 0.00001
    sigma = 0.0008
    returns = rng.normal(drift, sigma, n)
    close_mid = 1.10 * np.exp(np.cumsum(returns))
    high_mid = close_mid + rng.uniform(0.0001, 0.0008, n)
    low_mid = close_mid - rng.uniform(0.0001, 0.0008, n)
    open_mid = np.concatenate(([close_mid[0]], close_mid[:-1]))
    spread = 0.0002  # 2 pips
    timestamps = pd.date_range(start=start, periods=n, freq="4h", tz="UTC")
    df = pd.DataFrame({
        "open_bid": open_mid - spread / 2,
        "open_ask": open_mid + spread / 2,
        "high_bid": high_mid - spread / 2,
        "high_ask": high_mid + spread / 2,
        "low_bid": low_mid - spread / 2,
        "low_ask": low_mid + spread / 2,
        "close_bid": close_mid - spread / 2,
        "close_ask": close_mid + spread / 2,
        "volume": rng.integers(100, 1000, n),
        "spread": np.full(n, 20.0),  # 20 points = 2 pips for 5-digit pair
        "bid_ask_data_quality": "ok",
    }, index=timestamps)
    df.index.name = "timestamp_utc"
    return df


def build_synthetic_panel(
    pairs: tuple[str, ...] = ("EURUSD", "GBPUSD"),
    n_bars: int = 500,
    start: str = "2018-01-01",
    seed: int = 42,
) -> Panel:
    """Build a synthetic multi-pair Panel for tests."""
    frames = {pair: _synthetic_pair_df(n_bars, start, seed, pair) for pair in pairs}
    return Panel.from_frames(frames, tf="H4")


@dataclass(frozen=True)
class SyntheticSignal(SignalModule):
    """Synthetic SignalModule that fires on a deterministic schedule.

    Fires every ``period`` bars after ``warmup`` bars. ATR is a constant
    so SL distance is predictable in tests.
    """

    signal_name: str = "synthetic_periodic"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"
    period: int = 20
    warmup: int = 50
    atr_constant: float = 0.0010  # 10 pips fixed ATR

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        primary = panels[self.primary_tf]
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(primary.pairs):
            df = primary.pair_dfs[pair]
            n = len(df)
            mask = np.zeros(n, dtype=bool)
            for i in range(self.warmup, n, self.period):
                mask[i] = True
            atr = np.full(n, self.atr_constant)
            per_pair[pair] = PerPairSignalState(
                signal_mask=pd.Series(mask, index=df.index),
                atr=pd.Series(atr, index=df.index),
            )
        return SignalEvaluation(
            primary_tf=self.primary_tf,
            per_pair=per_pair,
            signal_name=self.signal_name,
            causal_lineage=self.causal_lineage,
        )


def build_synthetic_arc_pool_inputs(
    pairs: tuple[str, ...] = ("EURUSD", "GBPUSD", "USDJPY"),
    n_bars: int = 600,
    seed: int = 42,
):
    """Convenience: returns (signal_module, panels_dict)."""
    panel = build_synthetic_panel(pairs=pairs, n_bars=n_bars, seed=seed)
    signal = SyntheticSignal()
    return signal, {"H4": panel}


__all__ = (
    "build_synthetic_panel",
    "SyntheticSignal",
    "build_synthetic_arc_pool_inputs",
)
