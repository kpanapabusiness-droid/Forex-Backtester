"""Deterministic M1→higher-TF OHLC aggregation with parquet caching.

Aggregates the per-pair M1 cache (see ``core.data.histdata_loader``) into
{M5, M15, M30, H1, H4, D1, W1} bars. Both bid and ask streams are aggregated
independently with standard OHLC rules:

    open  = first   (chronological)
    high  = max
    low   = min
    close = last    (chronological)
    volume = sum

``spread_close`` and ``bid_ask_data_quality`` are recomputed from the
aggregated ``close_ask`` / ``close_bid`` (the per-minute quality flag does not
propagate — a 5-minute bar is its own thing).

TF boundaries (UTC, fixed):
    M5     every  5 minutes anchored to epoch (which is :00 / :05 / :10 ...)
    M15    every 15 minutes anchored to epoch
    M30    every 30 minutes anchored to epoch
    H1     every hour at :00
    H4     every 4 hours at 00:00 / 04:00 / 08:00 / 12:00 / 16:00 / 20:00 UTC
    D1     daily at 00:00 UTC
    W1     weekly starting Monday 00:00 UTC

Determinism guarantees:
    1. Bid and ask streams are aggregated in a fixed order with deterministic
       ``resample`` arguments (``label='left'``, ``closed='left'``).
    2. Output column order is fixed (``M1_COLUMNS``).
    3. Bars where any side has all-NaN OHLC (i.e. no underlying M1 minute) are
       dropped — these correspond to weekend gaps / market closures.

Two-run byte-identical parquet output is the contract (tested in
tests/test_aggregator.py).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.data.cache_keys import (
    CacheMeta,
    cache_valid,
    load_m1_manifest,
    m1_cache_key_for_pair,
    manifest_self_sha256,
    now_iso,
    tf_cache_key,
    write_meta,
)
from core.data.histdata_loader import (
    DQ_NAN_BID_OR_ASK,
    DQ_OK,
    DQ_ZERO_OR_NEG_SPREAD,
    M1_COLUMNS,
    HistDataPaths,
    load_m1,
)

# Mapping from TF label → pandas resample freq string. Anchoring:
#   - sub-hourly + H1: pandas default origin (epoch, midnight UTC) is correct
#   - H4: explicit ``origin='start_day'`` to force 00:00/04:00/... anchors
#   - D1: ``1D`` is non-tick-like → ``origin`` is a no-op; default anchors at
#     midnight UTC which is what we want
#   - W1: ``W-MON`` with label='left' / closed='left' gives Monday 00:00 → Sunday
SUPPORTED_TFS: tuple[str, ...] = ("M5", "M15", "M30", "H1", "H4", "D1", "W1")

_TF_TO_FREQ: dict[str, str] = {
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
    "W1": "W-MON",
}

# Freqs for which pandas honours the ``origin`` keyword (Tick-like).
_TICK_LIKE_FREQS: frozenset[str] = frozenset({"5min", "15min", "30min", "1h", "4h"})

# Per-side OHLC aggregation rules (column → aggregator). Identical for bid and
# ask; rebuilt programmatically to avoid duplication.
_OHLC_AGG: dict[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
}


def _resample_kwargs(freq: str, origin: str) -> dict:
    """Build resample kwargs; ``origin`` is only honoured for tick-like freqs."""
    kw: dict = {"label": "left", "closed": "left"}
    if freq in _TICK_LIKE_FREQS:
        kw["origin"] = origin
    return kw


def _aggregate_side(df: pd.DataFrame, side: str, freq: str, origin: str) -> pd.DataFrame:
    """Resample one side (bid or ask) into ``freq`` bars.

    Returns a DataFrame with columns ``open_<side> high_<side> low_<side> close_<side>``
    and the same DatetimeIndex (left-labelled) as the resample output.
    """
    cols = {f"{k}_{side}": f"{k}_{side}" for k in _OHLC_AGG}
    side_df = df[list(cols)].rename(columns={f"{k}_{side}": k for k in _OHLC_AGG})
    resampled = side_df.resample(freq, **_resample_kwargs(freq, origin)).agg(_OHLC_AGG)
    return resampled.rename(columns={k: f"{k}_{side}" for k in _OHLC_AGG})


def aggregate_m1_to_tf(df_m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate an M1 DataFrame (per ``histdata_loader.M1_COLUMNS``) to ``tf``.

    Parameters
    ----------
    df_m1 : pd.DataFrame
        M1 frame as returned by ``load_m1`` — DatetimeIndex (UTC), columns
        per ``M1_COLUMNS``.
    tf : str
        Target TF, one of ``SUPPORTED_TFS``.

    Returns
    -------
    pd.DataFrame
        Same column ordering as ``M1_COLUMNS``; DatetimeIndex left-labelled
        at the TF boundary; bars with no underlying M1 data are dropped.
    """
    if tf not in SUPPORTED_TFS:
        raise ValueError(f"Unsupported TF {tf!r}; expected one of {SUPPORTED_TFS}")
    freq = _TF_TO_FREQ[tf]
    origin = "start_day" if tf in ("H4",) else "epoch"

    bid = _aggregate_side(df_m1, "bid", freq, origin)
    ask = _aggregate_side(df_m1, "ask", freq, origin)
    volume = df_m1["volume"].resample(freq, **_resample_kwargs(freq, origin)).sum()

    out = pd.concat([bid, ask, volume.rename("volume")], axis=1)
    # Drop bars where either side is entirely NaN (no underlying minutes).
    keep = out[["open_bid", "open_ask"]].notna().all(axis=1)
    out = out.loc[keep].copy()
    out["volume"] = out["volume"].astype("int64")

    out["spread_close"] = out["close_ask"] - out["close_bid"]
    dq = pd.Series(DQ_OK, index=out.index, dtype="object")
    nan_mask = out[["open_bid", "close_bid", "open_ask", "close_ask"]].isna().any(axis=1)
    dq.loc[nan_mask] = DQ_NAN_BID_OR_ASK
    bad_spread = (out["spread_close"] <= 0) & ~nan_mask
    dq.loc[bad_spread] = DQ_ZERO_OR_NEG_SPREAD
    out["bid_ask_data_quality"] = dq

    return out[M1_COLUMNS]


def _tf_cache_parquet(paths: HistDataPaths, tf: str, pair: str) -> Path:
    return paths.cache_root / tf / f"{pair}.parquet"


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=True)


def aggregate(
    pair: str,
    tf: str,
    histdata_root: Path | str = "data/histdata",
    cache_root: Path | str = "data/cache",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load (or compute + cache) aggregated TF bars for ``pair``.

    On a cache hit (parquet exists + sidecar cache_key matches the upstream M1
    key for that pair), the parquet is read directly. Otherwise the M1 layer
    is loaded (which itself may build its cache), aggregated, written to
    ``<cache_root>/<tf>/<pair>.parquet``, and returned.
    """
    if tf not in SUPPORTED_TFS:
        raise ValueError(f"Unsupported TF {tf!r}; expected one of {SUPPORTED_TFS}")

    paths = HistDataPaths(Path(histdata_root), Path(cache_root))
    manifest = load_m1_manifest(paths.m1_manifest_path)
    m1_key = m1_cache_key_for_pair(manifest, pair)
    key = tf_cache_key(m1_key, tf)
    parquet_path = _tf_cache_parquet(paths, tf, pair)

    if use_cache and cache_valid(parquet_path, key):
        return pd.read_parquet(parquet_path)

    df_m1 = load_m1(pair, histdata_root=histdata_root, cache_root=cache_root, use_cache=use_cache)
    df_tf = aggregate_m1_to_tf(df_m1, tf)

    _write_parquet(df_tf, parquet_path)
    write_meta(
        parquet_path,
        CacheMeta(
            pair=pair,
            layer=tf,
            cache_key=key,
            source_m1_manifest_sha256=manifest_self_sha256(paths.m1_manifest_path),
            n_rows=int(len(df_tf)),
            columns=list(df_tf.columns),
            created_at=now_iso(),
        ),
    )
    return df_tf
