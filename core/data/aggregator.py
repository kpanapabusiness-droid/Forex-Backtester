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

Two boundary conventions are supported:

    ``boundary_convention="utc"`` (default) — TF boundaries anchored to UTC:
        M5     every  5 minutes anchored to epoch (:00 / :05 / :10 ...)
        M15    every 15 minutes anchored to epoch
        M30    every 30 minutes anchored to epoch
        H1     every hour at :00 UTC
        H4     every 4 hours at 00:00 / 04:00 / 08:00 / 12:00 / 16:00 / 20:00 UTC
        D1     daily at 00:00 UTC
        W1     weekly starting Monday 00:00 UTC

    ``boundary_convention="5ers_eet"`` — anchored to the 5ers broker EET/EEST
    trading day (EU DST rules via the ``Europe/Athens`` zone). The M1 UTC
    index is converted to EET before resampling, then bar labels are converted
    back to UTC for storage. Anchors in UTC terms (DST-dependent):

        H4 boundaries (EET-local 00/04/.../20):
            EET winter (UTC+2): UTC 22, 02, 06, 10, 14, 18
            EEST summer (UTC+3): UTC 21, 01, 05, 09, 13, 17
        D1: EET 00:00 → UTC 22:00 winter / UTC 21:00 summer
        W1: Monday 00:00 EET

    DST transitions (handled automatically by ``tz_convert``):
        - Spring forward (last Sun in March): EET 03:00 → EEST 04:00. The
          4H bar starting at EET 00:00 that day spans only ~3 wall-clock
          hours but the same 4 UTC hours. No bar is missing.
        - Autumn fall-back (last Sun in October): EEST 04:00 → EET 03:00.
          The EET 02:00-03:00 wall-clock interval occurs twice. UTC index
          is unambiguous; the 4H bar starting at EET 00:00 spans ~5
          wall-clock hours but the same 4 UTC hours. No duplicate bar
          emitted because the UTC index is monotonic.

Cache layout:
    UTC:      <cache_root>/<TF>/<PAIR>.parquet
    5ers EET: <cache_root>/<TF>_5ers_eet/<PAIR>.parquet

The two convention caches coexist; the convention is encoded in both the
directory name and the cache_key sidecar so cross-pollination is impossible.

Determinism guarantees:
    1. Bid and ask streams are aggregated in a fixed order with deterministic
       ``resample`` arguments (``label='left'``, ``closed='left'``).
    2. Output column order is fixed (``M1_COLUMNS``).
    3. Bars where any side has all-NaN OHLC (i.e. no underlying M1 minute) are
       dropped — these correspond to weekend gaps / market closures.

Two-run byte-identical parquet output is the contract (tested in
tests/test_aggregator.py and tests/test_aggregator_5ers_eet.py).
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

# Supported boundary conventions. "utc" is the legacy default; "5ers_eet"
# anchors bars to the 5ers broker EET/EEST trading day.
SUPPORTED_BOUNDARY_CONVENTIONS: tuple[str, ...] = ("utc", "5ers_eet")

# Representative IANA zone for EET/EEST with EU DST rules. 5ers is a Cyprus-
# regulated firm; ``Asia/Nicosia`` follows the same EU rules as
# ``Europe/Athens`` and the two are byte-identical for the 2010+ HistData
# range we care about. Picked Athens for brevity and historical stability.
_EET_TZ: str = "Europe/Athens"

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


def _aggregate_side_per_day(df: pd.DataFrame, side: str, freq: str) -> pd.DataFrame:
    """Per-local-date resample for the 5ers EET sub-day re-anchoring path.

    For H4 under EET, the day starts at local midnight (which is a UTC anchor
    that shifts at DST). Standard ``resample(origin='start_day')`` anchors
    to the first day's midnight only and does NOT re-anchor at DST. To get
    local 00/04/08/12/16/20 bars on every day, we group by local date and
    resample each day independently from its own midnight.
    """
    cols = {f"{k}_{side}": f"{k}_{side}" for k in _OHLC_AGG}
    side_df = df[list(cols)].rename(columns={f"{k}_{side}": k for k in _OHLC_AGG})
    local_dates = pd.Index(side_df.index.date, name="__local_date")
    parts = []
    for _, group in side_df.groupby(local_dates, sort=True):
        parts.append(
            group.resample(freq, label="left", closed="left", origin="start_day").agg(_OHLC_AGG)
        )
    if not parts:
        out = pd.DataFrame(columns=list(_OHLC_AGG.keys()), index=pd.DatetimeIndex([], tz=df.index.tz))
    else:
        out = pd.concat(parts).sort_index()
    return out.rename(columns={k: f"{k}_{side}" for k in _OHLC_AGG})


def _aggregate_volume_per_day(volume: pd.Series, freq: str) -> pd.Series:
    """Per-local-date volume sum for the 5ers EET sub-day path."""
    local_dates = pd.Index(volume.index.date, name="__local_date")
    parts = []
    for _, group in volume.groupby(local_dates, sort=True):
        parts.append(group.resample(freq, label="left", closed="left", origin="start_day").sum())
    if not parts:
        return pd.Series([], dtype="int64", name=volume.name)
    return pd.concat(parts).sort_index()


def aggregate_m1_to_tf(
    df_m1: pd.DataFrame, tf: str, boundary_convention: str = "utc"
) -> pd.DataFrame:
    """Aggregate an M1 DataFrame (per ``histdata_loader.M1_COLUMNS``) to ``tf``.

    Parameters
    ----------
    df_m1 : pd.DataFrame
        M1 frame as returned by ``load_m1`` — DatetimeIndex (UTC), columns
        per ``M1_COLUMNS``.
    tf : str
        Target TF, one of ``SUPPORTED_TFS``.
    boundary_convention : str, default ``"utc"``
        ``"utc"`` (legacy) or ``"5ers_eet"`` (EET/EEST broker day; EU DST
        rules via ``Europe/Athens``).

    Returns
    -------
    pd.DataFrame
        Same column ordering as ``M1_COLUMNS``; DatetimeIndex (UTC)
        left-labelled at the convention-appropriate TF boundary; bars with
        no underlying M1 data are dropped.
    """
    if tf not in SUPPORTED_TFS:
        raise ValueError(f"Unsupported TF {tf!r}; expected one of {SUPPORTED_TFS}")
    if boundary_convention not in SUPPORTED_BOUNDARY_CONVENTIONS:
        raise ValueError(
            f"Unsupported boundary_convention {boundary_convention!r}; "
            f"expected one of {SUPPORTED_BOUNDARY_CONVENTIONS}"
        )
    freq = _TF_TO_FREQ[tf]

    if boundary_convention == "utc":
        origin = "start_day" if tf in ("H4",) else "epoch"
        bid = _aggregate_side(df_m1, "bid", freq, origin)
        ask = _aggregate_side(df_m1, "ask", freq, origin)
        volume = df_m1["volume"].resample(freq, **_resample_kwargs(freq, origin)).sum()
    else:
        # 5ers EET: convert to local tz so anchors are computed in local
        # wall-clock terms.
        df_work = df_m1.copy()
        df_work.index = df_work.index.tz_convert(_EET_TZ)
        # D1 / W1: pandas' built-in resample on tz-aware index does per-day
        # / per-week re-anchoring correctly across DST.
        # Sub-hourly (M5..H1): bin edges are sub-DST-shift so bins coincide
        # with the UTC convention; we fall through to the simple path.
        # H4: pandas does NOT re-anchor sub-day bins per DST day — we go
        # through the per-local-date path to keep bars at local 00/04/.../20.
        if tf == "H4":
            bid = _aggregate_side_per_day(df_work, "bid", freq)
            ask = _aggregate_side_per_day(df_work, "ask", freq)
            volume = _aggregate_volume_per_day(df_work["volume"], freq)
        else:
            origin = "start_day"
            bid = _aggregate_side(df_work, "bid", freq, origin)
            ask = _aggregate_side(df_work, "ask", freq, origin)
            volume = df_work["volume"].resample(freq, **_resample_kwargs(freq, origin)).sum()

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

    if boundary_convention != "utc":
        # Storage convention: UTC-naive bar labels regardless of resample tz.
        out.index = out.index.tz_convert("UTC")

    return out[M1_COLUMNS]


def _tf_cache_parquet(
    paths: HistDataPaths, tf: str, pair: str, boundary_convention: str = "utc"
) -> Path:
    layer = tf if boundary_convention == "utc" else f"{tf}_{boundary_convention}"
    return paths.cache_root / layer / f"{pair}.parquet"


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=True)


def aggregate(
    pair: str,
    tf: str,
    histdata_root: Path | str = "data/histdata",
    cache_root: Path | str = "data/cache",
    use_cache: bool = True,
    boundary_convention: str = "utc",
) -> pd.DataFrame:
    """Load (or compute + cache) aggregated TF bars for ``pair``.

    On a cache hit (parquet exists + sidecar cache_key matches the upstream M1
    key for that pair under the given boundary convention), the parquet is
    read directly. Otherwise the M1 layer is loaded (which itself may build
    its cache), aggregated, written to the convention-appropriate path:

        ``"utc"``      → ``<cache_root>/<tf>/<pair>.parquet``       (legacy)
        ``"5ers_eet"`` → ``<cache_root>/<tf>_5ers_eet/<pair>.parquet``

    and returned.
    """
    if tf not in SUPPORTED_TFS:
        raise ValueError(f"Unsupported TF {tf!r}; expected one of {SUPPORTED_TFS}")
    if boundary_convention not in SUPPORTED_BOUNDARY_CONVENTIONS:
        raise ValueError(
            f"Unsupported boundary_convention {boundary_convention!r}; "
            f"expected one of {SUPPORTED_BOUNDARY_CONVENTIONS}"
        )

    paths = HistDataPaths(Path(histdata_root), Path(cache_root))
    manifest = load_m1_manifest(paths.m1_manifest_path)
    m1_key = m1_cache_key_for_pair(manifest, pair)
    key = tf_cache_key(m1_key, tf, boundary_convention)
    parquet_path = _tf_cache_parquet(paths, tf, pair, boundary_convention)
    layer = tf if boundary_convention == "utc" else f"{tf}_{boundary_convention}"

    if use_cache and cache_valid(parquet_path, key):
        return pd.read_parquet(parquet_path)

    df_m1 = load_m1(pair, histdata_root=histdata_root, cache_root=cache_root, use_cache=use_cache)
    df_tf = aggregate_m1_to_tf(df_m1, tf, boundary_convention=boundary_convention)

    _write_parquet(df_tf, parquet_path)
    write_meta(
        parquet_path,
        CacheMeta(
            pair=pair,
            layer=layer,
            cache_key=key,
            source_m1_manifest_sha256=manifest_self_sha256(paths.m1_manifest_path),
            n_rows=int(len(df_tf)),
            columns=list(df_tf.columns),
            created_at=now_iso(),
        ),
    )
    return df_tf
