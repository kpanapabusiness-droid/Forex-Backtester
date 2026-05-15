"""Data loaders for step 4. Returns wide-format frames keyed by trade_id.

- signals_features: per-trade features + verbatim outcomes.
- cluster_assignments: K2_kmeans renamed to kmeans_K2_cluster_id internally.
- trade_paths (long → wide for bar_offset 0..t): used for held-bar feature
  matrix construction and for simulator (entry price, OHLC, MAE).
- held_bar_evolution/t{t}.csv: held-bar context features at bar t.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import _common as C


PATH_NUMERIC_PER_BAR = (
    "open", "high", "low", "close",
    "cum_logret_from_entry",
    "mfe_to_date_atr", "mae_to_date_atr",
)

HELD_CTX_NUMERIC = (
    "atr_regime_ratio",
    "broker_spread_pips_raw", "broker_spread_pips_floored",
    "cross_pair_dispersion_proxy",
    "basket_cum_logret_USD", "basket_cum_logret_EUR",
    "basket_cum_logret_JPY", "basket_cum_logret_GBP",
)


def load_signals() -> pd.DataFrame:
    return pd.read_csv(C.SIGNALS_CSV)


def load_clusters() -> pd.DataFrame:
    df = pd.read_csv(C.CLUSTER_CSV, usecols=["trade_id", C.CLUSTER_COL_FILE])
    df = df.rename(columns={C.CLUSTER_COL_FILE: C.CLUSTER_COL_INTERNAL})
    return df


def load_signals_with_clusters() -> pd.DataFrame:
    sig = load_signals()
    clu = load_clusters()
    out = sig.merge(clu, on="trade_id", how="left")
    return out


def load_held_ctx(t: int) -> pd.DataFrame:
    p = C.HELD_CTX / f"t{t}.csv"
    df = pd.read_csv(p)
    keep = ["trade_id"] + list(HELD_CTX_NUMERIC)
    out = df[keep].copy()
    # Mark t suffix for downstream feature naming
    out = out.rename(columns={c: f"{c}_t{t}" for c in HELD_CTX_NUMERIC})
    return out


@lru_cache(maxsize=1)
def _load_paths_full() -> pd.DataFrame:
    cols = ["trade_id", "bar_offset", "open", "high", "low", "close",
            "cum_logret_from_entry", "mfe_to_date_atr", "mae_to_date_atr",
            "is_held_bar", "is_forward_bar", "data_end_flag"]
    df = pd.read_csv(C.PATHS_CSV, usecols=cols)
    return df


def load_paths_long(max_offset: int | None = None) -> pd.DataFrame:
    df = _load_paths_full()
    if max_offset is not None:
        df = df[df["bar_offset"] <= max_offset]
    return df.copy()


def load_paths_wide_to_t(t: int, all_trade_ids: np.ndarray) -> pd.DataFrame:
    """Pivot trade_paths to wide format with cols <feature>_b{0..t} per trade."""
    df = load_paths_long(max_offset=t)
    pieces = []
    for off in range(t + 1):
        sub = df[df["bar_offset"] == off].set_index("trade_id")[list(PATH_NUMERIC_PER_BAR)]
        sub = sub.add_suffix(f"_b{off}")
        pieces.append(sub)
    wide = pd.concat(pieces, axis=1)
    wide = wide.reindex(all_trade_ids)
    wide = wide.reset_index().rename(columns={"index": "trade_id"})
    return wide


def load_path_at_offset(offset: int) -> pd.DataFrame:
    """Per-trade row at a specific bar_offset (for simulator)."""
    df = _load_paths_full()
    sub = df[df["bar_offset"] == offset].copy()
    return sub


def trade_ids_active_at_bar(t: int, signals: pd.DataFrame) -> np.ndarray:
    """Trades with bars_held >= t (still open at bar t)."""
    return signals.loc[signals["bars_held"] >= t, "trade_id"].values
