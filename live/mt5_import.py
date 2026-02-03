"""
Load MT5-exported DAILY CSV market data (space-delimited) and positions/history CSVs.
Canonical dataframe per symbol: time, open, high, low, close, spread_points.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REQUIRED_MARKET_COLS = ["open", "high", "low", "close"]
CANONICAL_COLS = ["time", "open", "high", "low", "close", "spread_points"]


def symbol_from_filename(filename: str) -> str:
    """
    Infer 6-letter symbol from filename (e.g. AUDCAD, EURUSD).
    - AUDCAD_Daily.csv -> AUDCAD
    - EURUSD_Daily_202401020000_202602020000.csv -> EURUSD
    - EUR_USD.csv -> EURUSD (underscore removed, first two 3-letter codes).
    """
    stem = Path(filename).stem
    letters = re.sub(r"[^A-Za-z]", "", stem).upper()
    if len(letters) >= 6:
        return (letters[:3] + letters[3:6])[:6]
    return letters[:6] if len(letters) >= 3 else ""


def symbol_to_config_pair(symbol: str) -> str:
    """Convert 6-letter symbol to config pair format (e.g. EURUSD -> EUR_USD)."""
    s = (symbol or "").upper().replace("_", "")[:6]
    if len(s) >= 6:
        return f"{s[:3]}_{s[3:6]}"
    return s


def parse_mt5_date(series: pd.Series) -> pd.Series:
    """Parse MT5 DATE column format YYYY.MM.DD into pandas datetime."""
    return pd.to_datetime(series.astype(str).str.replace(".", "-", regex=False), errors="coerce")


def get_market_export_dates(market_dir: str | Path) -> dict[str, str]:
    """
    Return symbol -> export date (YYYY-MM-DD) in Australia/Melbourne from each
    market CSV file's mtime. Used to treat the last bar as forming when it
    equals the export day (stale mid-candle exports).
    """
    market_path = Path(market_dir)
    if not market_path.is_dir():
        return {}
    melb = ZoneInfo("Australia/Melbourne")
    result: dict[str, str] = {}
    for path in sorted(market_path.glob("*.csv")):
        try:
            stat = path.stat()
        except OSError:
            continue
        sym = symbol_from_filename(path.name)
        if not sym or len(sym) < 6:
            continue
        mtime_utc = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        export_date = mtime_utc.astimezone(melb).strftime("%Y-%m-%d")
        result[sym] = export_date
    return result


def load_market_tsvs(market_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load all *.csv market files from market_dir. MT5 exports are space-delimited with
    DATE, OPEN, HIGH, LOW, CLOSE, TICKVOL, VOL, SPREAD. Return dict symbol -> canonical df.
    Canonical df columns: time, open, high, low, close, spread_points (and volume for schema).
    DATE is parsed as YYYY.MM.DD to pandas datetime.
    """
    market_path = Path(market_dir)
    if not market_path.is_dir():
        return {}

    result: dict[str, pd.DataFrame] = {}
    for path in sorted(market_path.glob("*.csv")):
        try:
            df = pd.read_csv(path, sep=r"\s+", engine="python", on_bad_lines="skip")
        except Exception:
            continue
        if df.empty:
            continue

        # Robust column normalization: ignore case, whitespace, punctuation, and BOM-like chars.
        # This tolerates headers like "<DATE>", "date ", "TICKVOL", "tick_volume", etc.
        cols_norm: dict[str, str] = {}
        for col in df.columns:
            key = re.sub(r"[^a-z]", "", str(col).lower())
            if key and key not in cols_norm:
                cols_norm[key] = col

        def _col(*candidates: str) -> str | None:
            for cand in candidates:
                if cand in cols_norm:
                    return cols_norm[cand]
            return None

        date_col = _col("date", "time", "datetime", "timestamp")
        open_col = _col("open")
        high_col = _col("high")
        low_col = _col("low")
        close_col = _col("close")
        spread_col = _col("spread", "spreadpoints")
        tickvol_col = _col("tickvol", "tickvolume", "vol", "volume")

        if not date_col or not open_col or not high_col or not low_col or not close_col:
            # Fail-closed: malformed header; skip this file.
            continue

        sym = symbol_from_filename(path.name)
        if not sym or len(sym) < 6:
            continue

        out = pd.DataFrame()
        out["time"] = parse_mt5_date(df[date_col].astype(str))
        for k, col in [("open", open_col), ("high", high_col), ("low", low_col), ("close", close_col)]:
            out[k] = pd.to_numeric(df[col], errors="coerce")
        out["spread_points"] = (
            pd.to_numeric(df[spread_col], errors="coerce").fillna(0).astype(int)
            if spread_col
            else 0
        )
        if tickvol_col is not None:
            out["volume"] = pd.to_numeric(df[tickvol_col], errors="coerce").fillna(0)
        else:
            out["volume"] = 0

        out = out.dropna(subset=["time", "open", "high", "low", "close"])
        if out.empty:
            continue
        out = out.sort_values("time").reset_index(drop=True)
        result[sym] = out
    return result


def load_positions_csv(path: str | Path) -> pd.DataFrame:
    """
    Load positions CSV (comma-separated): ticket,symbol,type,lots,open_time,open_price,sl,tp.
    Tolerate empty (headers only). Return dataframe with these columns.
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(
            columns=["ticket", "symbol", "type", "lots", "open_time", "open_price", "sl", "tp"]
        )
    try:
        df = pd.read_csv(p)
        expected = ["ticket", "symbol", "type", "lots", "open_time", "open_price", "sl", "tp"]
        for c in expected:
            if c not in df.columns:
                df[c] = None
        return df[expected] if not df.empty else df
    except Exception:
        return pd.DataFrame(
            columns=["ticket", "symbol", "type", "lots", "open_time", "open_price", "sl", "tp"]
        )


def load_history_csv(path: str | Path) -> pd.DataFrame:
    """
    Load trade history CSV (comma-separated): ticket,symbol,type,open_time,close_time,open_price,close_price,profit.
    Tolerate empty (headers only).
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(
            columns=[
                "ticket", "symbol", "type", "open_time", "close_time",
                "open_price", "close_price", "profit",
            ]
        )
    try:
        df = pd.read_csv(p)
        expected = [
            "ticket", "symbol", "type", "open_time", "close_time",
            "open_price", "close_price", "profit",
        ]
        for c in expected:
            if c not in df.columns:
                df[c] = None
        return df[expected] if not df.empty else df
    except Exception:
        return pd.DataFrame(
            columns=[
                "ticket", "symbol", "type", "open_time", "close_time",
                "open_price", "close_price", "profit",
            ]
        )


def canonical_df_for_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy suitable for engine: date, open, high, low, close, volume
    (from canonical time, open, high, low, close, volume/spread_points).
    """
    out = df.copy()
    if "time" in out.columns and "date" not in out.columns:
        out = out.rename(columns={"time": "date"})
    if "volume" not in out.columns:
        out["volume"] = 0
    required = ["date", "open", "high", "low", "close"]
    if not all(c in out.columns for c in required):
        return pd.DataFrame()
    return out[[c for c in required + ["volume"] if c in out.columns]]
