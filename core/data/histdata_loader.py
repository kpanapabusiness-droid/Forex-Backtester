"""HistData M1 bid+ask loader with parquet caching.

Reads HistData M1 derived CSVs (one per pair-side-month) and joins them into a
single per-pair time-indexed DataFrame with separate bid and ask OHLC columns.
First load parses CSVs and writes a parquet cache; subsequent loads of the same
pair read the cache directly when the upstream M1 manifest is unchanged.

Layout (from docs/DATA_FOUNDATION.md):

    data/histdata/<PAIR>/m1/bid/<YYYY>/<PAIR>_M1_BID_<YYYYMM>.csv
    data/histdata/<PAIR>/m1/ask/<YYYY>/<PAIR>_M1_ASK_<YYYYMM>.csv
    data/histdata/m1_manifest.json

CSV schema (both sides):

    timestamp_utc,open,high,low,close,volume
    2010-01-03T22:00:00Z,1.4301,1.4304,1.4301,1.4304,6

Output DataFrame schema:

    Index: DatetimeIndex (UTC, named "timestamp_utc")
    Columns:
        open_bid, high_bid, low_bid, close_bid,
        open_ask, high_ask, low_ask, close_ask,
        volume,
        spread_close,
        bid_ask_data_quality

``volume`` is the per-minute tick count (bid.volume == ask.volume by
construction per DATA_FOUNDATION §Aggregation — asserted at load).
``spread_close`` = ``close_ask - close_bid``; non-positive or NaN values
mark the bar in ``bid_ask_data_quality`` per the §1 non-negotiable
"zero-spread bars are a data quality flag, not silently backfilled".

Determinism: rows sorted ascending by timestamp_utc; per-row column order
fixed; parquet written with ``compression='snappy'`` and explicit schema
ordering for two-run reproduction within a tolerance documented in
tests/test_histdata_loader.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.data.cache_keys import (
    CacheMeta,
    cache_valid,
    load_m1_manifest,
    m1_cache_key_for_pair,
    manifest_self_sha256,
    now_iso,
    write_meta,
)

# Canonical column order — used for output DataFrames and parquet schema.
M1_COLUMNS: list[str] = [
    "open_bid",
    "high_bid",
    "low_bid",
    "close_bid",
    "open_ask",
    "high_ask",
    "low_ask",
    "close_ask",
    "volume",
    "spread_close",
    "bid_ask_data_quality",
]

DQ_OK = "ok"
DQ_ZERO_OR_NEG_SPREAD = "zero_or_negative_spread"
DQ_NAN_BID_OR_ASK = "nan_bid_or_ask"


@dataclass(frozen=True)
class HistDataPaths:
    """Resolved roots for the data layer."""

    histdata_root: Path
    cache_root: Path

    @property
    def m1_manifest_path(self) -> Path:
        return self.histdata_root / "m1_manifest.json"

    def m1_cache_parquet(self, pair: str) -> Path:
        return self.cache_root / "m1" / f"{pair}.parquet"

    def pair_root(self, pair: str) -> Path:
        return self.histdata_root / pair

    def side_csv(self, pair: str, side: str, yyyymm: str) -> Path:
        year = yyyymm[:4]
        side_upper = side.upper()
        return (
            self.histdata_root / pair / "m1" / side / year / f"{pair}_M1_{side_upper}_{yyyymm}.csv"
        )


def _enumerate_pair_months(manifest: dict[str, Any], pair: str) -> list[str]:
    """Return sorted YYYYMM strings for which both bid and ask CSVs exist in the manifest.

    The M1 manifest lists per-file entries with relpath like
    ``<PAIR>/m1/bid/<YYYY>/<PAIR>_M1_BID_<YYYYMM>.csv``. A month is loadable iff
    both sides are present; mismatched months (one side missing) are dropped.
    """
    files = manifest["pairs"][pair]["files"]
    bid_months: set[str] = set()
    ask_months: set[str] = set()
    for relpath in files:
        parts = relpath.split("/")
        # parts: [PAIR, "m1", side, YYYY, FILENAME]
        if len(parts) != 5 or parts[1] != "m1":
            continue
        side = parts[2]
        fname = parts[4]
        # filename: <PAIR>_M1_{BID|ASK}_<YYYYMM>.csv
        try:
            yyyymm = fname.split("_")[3].removesuffix(".csv")
        except IndexError:
            continue
        if side == "bid":
            bid_months.add(yyyymm)
        elif side == "ask":
            ask_months.add(yyyymm)
    return sorted(bid_months & ask_months)


def _read_side_csv(path: Path) -> pd.DataFrame:
    """Read one side CSV with locked dtypes and UTC parsing.

    Schema: timestamp_utc,open,high,low,close,volume — the
    ``timestamp_utc`` column is ISO-8601 with a ``Z`` suffix.
    """
    df = pd.read_csv(
        path,
        dtype={
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "int64",
        },
    )
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, format="ISO8601")
    return df


def _read_pair_csvs(paths: HistDataPaths, pair: str, months: list[str]) -> pd.DataFrame:
    """Read all bid+ask CSVs for a pair, join on timestamp, return canonical schema."""
    bid_frames: list[pd.DataFrame] = []
    ask_frames: list[pd.DataFrame] = []
    for yyyymm in months:
        bid_path = paths.side_csv(pair, "bid", yyyymm)
        ask_path = paths.side_csv(pair, "ask", yyyymm)
        if not bid_path.exists():
            raise FileNotFoundError(f"Missing bid CSV: {bid_path}")
        if not ask_path.exists():
            raise FileNotFoundError(f"Missing ask CSV: {ask_path}")
        bid_frames.append(_read_side_csv(bid_path))
        ask_frames.append(_read_side_csv(ask_path))

    bid = pd.concat(bid_frames, ignore_index=True).sort_values("timestamp_utc")
    ask = pd.concat(ask_frames, ignore_index=True).sort_values("timestamp_utc")

    # Inner-merge: keep only minutes present on both sides.
    merged = pd.merge(
        bid.rename(
            columns={
                "open": "open_bid",
                "high": "high_bid",
                "low": "low_bid",
                "close": "close_bid",
                "volume": "volume_bid",
            }
        ),
        ask.rename(
            columns={
                "open": "open_ask",
                "high": "high_ask",
                "low": "low_ask",
                "close": "close_ask",
                "volume": "volume_ask",
            }
        ),
        on="timestamp_utc",
        how="inner",
        validate="one_to_one",
    )

    # Volume invariant: bid and ask tick counts agree per minute by source design.
    # Tolerance: allow tiny mismatches (e.g. file boundary edge cases) but fail
    # loudly if widespread — would indicate aggregation drift.
    mismatched = (merged["volume_bid"] != merged["volume_ask"]).sum()
    if mismatched > 0:
        frac = mismatched / len(merged)
        if frac > 0.001:
            raise ValueError(
                f"bid/ask volume mismatch on {mismatched:,} of {len(merged):,} bars "
                f"({frac:.4%}) for pair {pair!r} — aggregation drift suspected"
            )
    merged["volume"] = merged[["volume_bid", "volume_ask"]].max(axis=1).astype("int64")
    merged = merged.drop(columns=["volume_bid", "volume_ask"])

    merged["spread_close"] = merged["close_ask"] - merged["close_bid"]
    dq = pd.Series(DQ_OK, index=merged.index, dtype="object")
    nan_mask = merged[["open_bid", "close_bid", "open_ask", "close_ask"]].isna().any(axis=1)
    dq.loc[nan_mask] = DQ_NAN_BID_OR_ASK
    bad_spread = (merged["spread_close"] <= 0) & ~nan_mask
    dq.loc[bad_spread] = DQ_ZERO_OR_NEG_SPREAD
    merged["bid_ask_data_quality"] = dq

    merged = merged.set_index("timestamp_utc")
    return merged[M1_COLUMNS]


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a deterministic parquet file.

    Snappy compression, fixed column order. pandas/pyarrow handles schema
    introspection; column ordering is the load-bearing determinism control.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=True)


def load_m1(
    pair: str,
    histdata_root: Path | str = "data/histdata",
    cache_root: Path | str = "data/cache",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load the full M1 bid+ask history for one pair.

    Parameters
    ----------
    pair : str
        Pair symbol, e.g. ``"EURUSD"``.
    histdata_root : Path | str
        Root of the HistData layer (containing ``<PAIR>/m1/...`` and
        ``m1_manifest.json``). Defaults to ``data/histdata``.
    cache_root : Path | str
        Cache root. M1 cache is written to ``<cache_root>/m1/<PAIR>.parquet``.
    use_cache : bool
        If False, ignore any existing cache and re-parse the CSVs (cache is
        still written on success). Default True.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex named ``timestamp_utc`` (UTC); columns ``M1_COLUMNS``.
    """
    paths = HistDataPaths(Path(histdata_root), Path(cache_root))
    manifest = load_m1_manifest(paths.m1_manifest_path)
    cache_key = m1_cache_key_for_pair(manifest, pair)
    parquet_path = paths.m1_cache_parquet(pair)

    if use_cache and cache_valid(parquet_path, cache_key):
        return pd.read_parquet(parquet_path)

    months = _enumerate_pair_months(manifest, pair)
    if not months:
        raise ValueError(f"No paired bid/ask months found for {pair!r}")
    df = _read_pair_csvs(paths, pair, months)

    _write_parquet(df, parquet_path)
    write_meta(
        parquet_path,
        CacheMeta(
            pair=pair,
            layer="m1",
            cache_key=cache_key,
            source_m1_manifest_sha256=manifest_self_sha256(paths.m1_manifest_path),
            n_rows=int(len(df)),
            columns=list(df.columns),
            created_at=now_iso(),
        ),
    )
    return df
