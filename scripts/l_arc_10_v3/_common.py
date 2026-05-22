"""Arc 10 v3.0 — shared helpers across Steps 1-5.

Provides:
    - load_config(path)                       — YAML loader
    - load_pair_h4_d1_w1(pair, cfg)           — aggregates per pair, caches via core.data
    - build_panel_with_aux(h4_dfs, d1_dfs, w1_dfs) — feature panel surface
    - bid_view_for_signal(df_h4)              — bid-OHLC + 'date' column for signal module
    - sha256_file(path)
    - write_manifest(path, info)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.data.aggregator import aggregate  # noqa: E402
from core.sim.panel import Panel  # noqa: E402


def load_config(path: Path | str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_pair_tf(pair: str, tf: str, cfg: dict) -> pd.DataFrame:
    """Aggregate a single pair to ``tf`` using the v3 cache."""
    return aggregate(
        pair,
        tf,
        histdata_root=cfg["data"]["histdata_root"],
        cache_root=cfg["data"]["cache_root"],
    )


def bid_view_for_signal(df_tf: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df_tf`` with a 'date' column and plain
    open/high/low/close (bid side) — the schema the DLR signal module expects.

    The v3 aggregator returns a DatetimeIndex-keyed frame with separate
    open_bid / high_bid / low_bid / close_bid + open_ask/... columns. The
    DLR signal module operates on the bid side because the entry-side
    semantics (long fills at ask) live outside the signal.
    """
    out = pd.DataFrame(
        {
            "date": df_tf.index,
            "open": df_tf["open_bid"].to_numpy(),
            "high": df_tf["high_bid"].to_numpy(),
            "low": df_tf["low_bid"].to_numpy(),
            "close": df_tf["close_bid"].to_numpy(),
        }
    )
    return out.reset_index(drop=True)


def window_slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Slice a DatetimeIndex-keyed frame to [start, end] inclusive (end-of-day on end)."""
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]


class FeaturePanel:
    """Lightweight wrapper exposing the Panel surface plus an ``aux`` dict.

    The v3 feature producers in core.features.multi_tf expect
    ``panel.aux["d1"]`` and ``panel.aux["w1"]`` — both Panel instances.
    core.features.cross_pair just needs ``panel.pair_dfs``.

    Panel itself is a frozen dataclass; we wrap it rather than subclass.
    """

    __slots__ = ("_h4", "aux", "pair_dfs", "tf")

    def __init__(self, h4_panel: Panel, d1_panel: Panel, w1_panel: Panel):
        self._h4 = h4_panel
        self.aux = {"d1": d1_panel, "w1": w1_panel}
        self.pair_dfs = h4_panel.pair_dfs
        self.tf = h4_panel.tf

    def snapshot_at(self, t):
        return self._h4.snapshot_at(t)

    def bar_for(self, pair, t):
        return self._h4.bar_for(pair, t)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(path: Path, info: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(info, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(txt, encoding="utf-8", newline="\n")
