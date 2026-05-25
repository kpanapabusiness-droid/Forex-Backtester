"""Multi-pair OHLC panel for the v3.0 backtester.

A ``Panel`` is a thin facade over ``dict[pair, DataFrame]`` (one DataFrame
per pair, all sharing the same TF and DatetimeIndex schema produced by
``core.data.aggregator.aggregate``). It exposes:

    Panel.pairs              -> tuple[str, ...]
    Panel.timestamps         -> pd.DatetimeIndex   (union across pairs)
    Panel.tf                 -> str
    Panel.snapshot_at(t)     -> dict[pair, pd.Series | None]
    Panel.iter_bars()        -> Iterator[(timestamp, snapshot)]
    Panel.bar_for(pair, t)   -> pd.Series | None

The panel does not own data ingestion — it accepts a dict that the caller
built via ``aggregate`` for each pair, or via the convenience constructor
``Panel.from_pairs(pairs, tf, ...)`` that walks the data layer.

The union timestamp index is the load-bearing iteration axis. Different
pairs may have different bars (weekend coverage, broker outages); the
panel iterates over the *union* so cross-pair features at time t see a
complete picture (some pairs return ``None`` for the bar — the strategy
handles missing).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from core.data.aggregator import aggregate
from core.time_utils.session_boundary import SUPPORTED_CONVENTIONS


@dataclass(frozen=True)
class Panel:
    """Multi-pair OHLC panel at a single TF.

    Construct directly with ``Panel(pair_dfs={...}, tf="H4")`` or via
    ``Panel.from_pairs(pairs=[...], tf="H4")``.

    Invariants asserted at construction:
      - All DataFrames share the same TF (caller-declared).
      - All DataFrames have the canonical column schema produced by
        ``aggregator.aggregate`` / ``histdata_loader.load_m1``.
      - Indices are tz-aware (UTC).

    Boundary convention:
      - ``boundary_convention`` ("utc" | "5ers_eet", default "utc")
        identifies which trading-day boundary the bars in this panel
        were aggregated under. Downstream consumers (distance.py
        prior-session features, _fold_stats_helpers.compute_per_day_max_dd
        daily-DD bucketing) read this to choose the matching session
        bucketing logic. Default ``"utc"`` preserves legacy
        byte-identical behaviour for KH-24 anchor; engine-side default
        post-PR-189 is ``"5ers_eet"`` (set by aggregator-driven
        constructors).
    """

    pair_dfs: Mapping[str, pd.DataFrame]
    tf: str
    boundary_convention: str = "utc"

    def __post_init__(self) -> None:
        if not self.pair_dfs:
            raise ValueError("Panel needs at least one pair")
        if self.boundary_convention not in SUPPORTED_CONVENTIONS:
            raise ValueError(
                f"Unsupported boundary_convention {self.boundary_convention!r}; "
                f"expected one of {SUPPORTED_CONVENTIONS}"
            )
        first_pair = next(iter(self.pair_dfs))
        first_cols = tuple(self.pair_dfs[first_pair].columns)
        for pair, df in self.pair_dfs.items():
            if tuple(df.columns) != first_cols:
                raise ValueError(
                    f"Panel column mismatch on {pair!r}: "
                    f"{tuple(df.columns)} vs reference {first_cols}"
                )
            if df.index.tz is None:
                raise ValueError(f"Panel pair {pair!r} index is tz-naive; expected UTC")

    @property
    def pairs(self) -> tuple[str, ...]:
        return tuple(self.pair_dfs.keys())

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """Union of all pair timestamps, sorted ascending."""
        idx = pd.DatetimeIndex([], tz="UTC")
        for df in self.pair_dfs.values():
            idx = idx.union(df.index)
        return idx

    def bar_for(self, pair: str, t: pd.Timestamp) -> pd.Series | None:
        """Return the bar for ``(pair, t)`` or None if absent."""
        df = self.pair_dfs[pair]
        if t in df.index:
            return df.loc[t]
        return None

    def snapshot_at(self, t: pd.Timestamp) -> dict[str, pd.Series | None]:
        """Snapshot of all pairs at exactly ``t``.

        Returns ``dict[pair, bar | None]``. Pairs with no bar at ``t``
        (weekend / outage) map to ``None``; the strategy is expected to
        skip them.
        """
        return {pair: self.bar_for(pair, t) for pair in self.pair_dfs}

    def iter_bars(self) -> Iterator[tuple[pd.Timestamp, dict[str, pd.Series | None]]]:
        """Iterate the union of timestamps; yield ``(t, snapshot_at(t))``."""
        for t in self.timestamps:
            yield t, self.snapshot_at(t)

    @classmethod
    def from_pairs(
        cls,
        pairs: list[str],
        tf: str,
        histdata_root: Path | str = "data/histdata",
        cache_root: Path | str = "data/cache",
        use_cache: bool = True,
        boundary_convention: str = "utc",
    ) -> "Panel":
        """Build a panel by aggregating each pair from the data layer.

        ``boundary_convention`` is forwarded to ``aggregate`` AND stamped
        onto the Panel so downstream consumers (distance.py,
        compute_per_day_max_dd) see a single source of truth.
        """
        pair_dfs = {
            p: aggregate(
                p,
                tf,
                histdata_root=histdata_root,
                cache_root=cache_root,
                use_cache=use_cache,
                boundary_convention=boundary_convention,
            )
            for p in pairs
        }
        return cls(pair_dfs=pair_dfs, tf=tf, boundary_convention=boundary_convention)

    @classmethod
    def from_frames(
        cls,
        pair_dfs: Mapping[str, pd.DataFrame],
        tf: str,
        boundary_convention: str = "utc",
    ) -> "Panel":
        """Convenience constructor — same as the dataclass call."""
        return cls(
            pair_dfs=dict(pair_dfs),
            tf=tf,
            boundary_convention=boundary_convention,
        )
