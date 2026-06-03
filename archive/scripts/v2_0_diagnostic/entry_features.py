"""v2.0 archetype diagnostic — Step 4 (a): cross-dataset entry feature set.

Eight entry-bar features computed strictly from past bars (no lookahead),
joined to trade_id. KH-24 walks 4H bars; Arc 1 / Arc 2 walk 1H bars (native
timeframe for those signals).

Features (signal-bar = the bar that fired the signal; KH-24 = entry_bar - 1,
Arc 1 / Arc 2 = signal_bar_ts):

  body_to_range_ratio  |close - open| / (high - low)
  upper_wick_ratio     (high - max(open, close)) / (high - low)
  lower_wick_ratio     (min(open, close) - low) / (high - low)
  range_to_atr_14      (high - low) / ATR(14) at signal bar
  ret_5bar_atr         (close_N - close_{N-5}) / ATR(14)_N
  ret_20bar_atr        (close_N - close_{N-20}) / ATR(14)_N
  pos_in_20bar_range   (close - min(low last 20)) / (max(high last 20) - min(low last 20))
  rsi_14               Wilder RSI(14) at signal bar

Wilder ATR(14) and RSI(14) are computed on the native bar feed walking
forward; only data with timestamp <= signal_bar is used (no future bars).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAIR_LIST = sorted(p.stem for p in (REPO_ROOT / "data" / "1hr").glob("*.csv"))


def _wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR(period). df has open, high, low, close indexed by time."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Wilder smoothing = EMA with alpha = 1/period (adjust=False).
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr


def _wilder_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder RSI(period) on close."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _features_at_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 8 features at each bar of one pair's bar feed."""
    out = df.copy()
    rng = out["high"] - out["low"]
    safe_rng = rng.replace(0, np.nan)
    body = (out["close"] - out["open"]).abs()
    upper_wick = out["high"] - out[["open", "close"]].max(axis=1)
    lower_wick = out[["open", "close"]].min(axis=1) - out["low"]

    out["body_to_range_ratio"] = body / safe_rng
    out["upper_wick_ratio"]    = upper_wick / safe_rng
    out["lower_wick_ratio"]    = lower_wick / safe_rng

    atr14 = _wilder_atr(out, 14)
    out["range_to_atr_14"] = rng / atr14.replace(0, np.nan)

    close5  = out["close"].shift(5)
    close20 = out["close"].shift(20)
    out["ret_5bar_atr"]  = (out["close"] - close5)  / atr14.replace(0, np.nan)
    out["ret_20bar_atr"] = (out["close"] - close20) / atr14.replace(0, np.nan)

    low20_min  = out["low"].rolling(20, min_periods=20).min()
    high20_max = out["high"].rolling(20, min_periods=20).max()
    rng20 = (high20_max - low20_min).replace(0, np.nan)
    out["pos_in_20bar_range"] = (out["close"] - low20_min) / rng20

    out["rsi_14"] = _wilder_rsi(out, 14)

    return out[[
        "time",
        "body_to_range_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "range_to_atr_14",
        "ret_5bar_atr",
        "ret_20bar_atr",
        "pos_in_20bar_range",
        "rsi_14",
    ]]


def _load_bars(pair: str, timeframe: str) -> pd.DataFrame:
    p = REPO_ROOT / "data" / timeframe / f"{pair}.csv"
    df = pd.read_csv(p, usecols=["time", "open", "high", "low", "close"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def _build_pair_index(pairs: list[str], timeframe: str) -> dict[str, pd.DataFrame]:
    """Pair -> DataFrame with time-indexed precomputed features."""
    out: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        df = _load_bars(pair, timeframe)
        feats = _features_at_bar(df)
        feats = feats.set_index("time")
        out[pair] = feats
    return out


def entry_features_kh24(meta: pd.DataFrame) -> pd.DataFrame:
    """KH-24: walk 4H bars. trade_id = '<PAIR>_<entry_date_str>'.

    Signal bar for KH-24 is entry_date - 4h (one 4H bar before entry).
    """
    # Derive (pair, entry_ts) from trade_id directly (matches load_paths meta).
    trades = pd.read_csv(
        REPO_ROOT / "results" / "kh24" / "trades_all.csv",
        usecols=["trade_id", "pair", "entry_date"],
    )
    trades["entry_ts"]  = pd.to_datetime(trades["entry_date"])
    trades["signal_ts"] = trades["entry_ts"] - pd.Timedelta(hours=4)
    trades = trades[["trade_id", "pair", "signal_ts"]]

    pairs = sorted(trades["pair"].dropna().unique().tolist())
    bar_idx = _build_pair_index(pairs, "4hr")

    rows = []
    for pair, sub in trades.groupby("pair", sort=False):
        feats = bar_idx[pair]
        # Reindex: exact-match join on signal_ts.
        joined = feats.reindex(sub["signal_ts"].values)
        joined.insert(0, "trade_id", sub["trade_id"].values)
        joined.insert(1, "pair", pair)
        joined.insert(2, "entry_date", sub["signal_ts"].values)
        joined = joined.reset_index(drop=True)
        rows.append(joined)
    out = pd.concat(rows, axis=0, ignore_index=True)
    out["trade_id"] = out["trade_id"].astype("string")
    return out


def entry_features_arc(name: str) -> pd.DataFrame:
    """Arc 1 / Arc 2: walk 1H bars. Signal bar from signals_features.signal_bar_ts."""
    sf_path = REPO_ROOT / "results" / f"l_arc_{name[-1]}" / "step2_descriptive" / "signals_features.csv"
    trades = pd.read_csv(sf_path, usecols=["trade_id", "pair", "signal_bar_ts"])
    trades["trade_id"]      = trades["trade_id"].astype("string")
    trades["signal_bar_ts"] = pd.to_datetime(trades["signal_bar_ts"])

    pairs = sorted(trades["pair"].dropna().unique().tolist())
    bar_idx = _build_pair_index(pairs, "1hr")

    rows = []
    for pair, sub in trades.groupby("pair", sort=False):
        feats = bar_idx[pair]
        joined = feats.reindex(sub["signal_bar_ts"].values)
        joined.insert(0, "trade_id",   sub["trade_id"].values)
        joined.insert(1, "pair",       pair)
        joined.insert(2, "entry_date", sub["signal_bar_ts"].values)
        joined = joined.reset_index(drop=True)
        rows.append(joined)
    out = pd.concat(rows, axis=0, ignore_index=True)
    out["trade_id"] = out["trade_id"].astype("string")
    return out


def compute_entry_features(name: str) -> pd.DataFrame:
    if name == "kh24":
        return entry_features_kh24(None)
    return entry_features_arc(name)
