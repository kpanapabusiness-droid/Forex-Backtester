"""Shared I/O + path helpers for L Arc 1 Step 2."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STEP1_DIR = REPO_ROOT / "results" / "l_arc_1" / "step1_verbatim"
STEP2_DIR = REPO_ROOT / "results" / "l_arc_1" / "step2_descriptive"
CONFIG_PATH = REPO_ROOT / "configs" / "wfo_l_arc1_verbatim.yaml"
DATA_1H_DIR = REPO_ROOT / "data" / "1hr"

PAIRS: List[str] = [
    "AUD_CAD", "AUD_CHF", "AUD_JPY", "AUD_NZD", "AUD_USD",
    "CAD_CHF", "CAD_JPY", "CHF_JPY",
    "EUR_AUD", "EUR_CAD", "EUR_CHF", "EUR_GBP", "EUR_JPY", "EUR_NZD", "EUR_USD",
    "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_JPY", "GBP_NZD", "GBP_USD",
    "NZD_CAD", "NZD_CHF", "NZD_JPY", "NZD_USD",
    "USD_CAD", "USD_CHF", "USD_JPY",
]

FORWARD_HORIZON_BARS_DEFAULT: int = 240
FORWARD_HORIZON_BARS_EXTENDED: int = 480
RANDOM_SEED: int = 1234

# Time-exit and SL-distance shadow grids
ENTRY_DELAYS = [1, 2, 3, 5, 10]       # bar_offset; 1 is the verbatim baseline
SL_DISTANCES = [1.0, 1.5, 2.0, 2.5, 3.0]  # ATR multipliers; 2.0 baseline
TIME_EXIT_H = [1, 3, 6, 12, 24, 48, 120, 240]  # 1 baseline
H_GRID = [1, 3, 6, 12, 24, 48, 72, 120, 240, 360, 480]
HEALD_T = [1, 3, 5, 10, 20]
SPREAD_MULT = [0.5, 1.0, 1.5, 2.0]


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


def load_trades_verbatim() -> pd.DataFrame:
    """Load step 1 trades_verbatim.csv with timestamps as Timestamps."""
    path = STEP1_DIR / "trades_verbatim.csv"
    df = pd.read_csv(path)
    for col in ("signal_bar_ts", "entry_bar_ts", "exit_bar_ts",
                "signal_time_utc", "entry_time_utc", "exit_time_utc"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    if "trade_id" not in df.columns:
        df.insert(0, "trade_id", np.arange(len(df), dtype=np.int64))
    return df


def load_signals_log() -> pd.DataFrame:
    path = STEP1_DIR / "signals_log.csv"
    df = pd.read_csv(path)
    df["signal_bar_ts"] = pd.to_datetime(df["signal_bar_ts"])
    return df


def load_pair_1h(pair: str) -> pd.DataFrame:
    """Load a pair's 1H OHLC+spread+volume CSV; time parsed; sorted ascending."""
    path = DATA_1H_DIR / f"{pair}.csv"
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Bit-identical to engine's `_wilder_atr` (Python loop, period-1 simple seed)."""
    n = len(close)
    if n == 0:
        return np.array([], dtype=float)
    prev_close = np.empty(n, dtype=float)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def compute_signal_mask(df_1h: pd.DataFrame, lookback: int = 100, q: float = 0.90,
                        direction: str = "neg", atr_period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """Engine-parity signal mask; returns (signal_fired bool, atr_at_signal float)."""
    close = df_1h["close"].astype(float).values
    open_ = df_1h["open"].astype(float).values
    n = len(close)
    prev_close = np.empty(n, dtype=float)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_return = np.log(close / prev_close)
    abs_log_return = np.abs(log_return)
    threshold = (
        pd.Series(abs_log_return)
        .rolling(window=lookback, min_periods=lookback)
        .quantile(q, interpolation="linear")
        .shift(1)
        .to_numpy()
    )
    high = df_1h["high"].astype(float).values
    low = df_1h["low"].astype(float).values
    atr = wilder_atr(high, low, close, atr_period)
    sign_filter = (close < open_) if direction == "neg" else (close > open_)
    fired = (
        np.isfinite(threshold) & np.isfinite(atr) & np.isfinite(abs_log_return)
        & (abs_log_return > threshold) & sign_filter
    )
    return fired, atr


def ts_index_by_ts(df_1h: pd.DataFrame) -> Dict[pd.Timestamp, int]:
    """Map signal_bar_ts -> row index. Timestamps as np.datetime64 for speed."""
    return {pd.Timestamp(t): i for i, t in enumerate(df_1h["time"].to_numpy())}


def floor_pips_for_pair(spread_floor_yaml: Path, pair: str, points_per_pip: float = 10.0) -> Optional[float]:
    data = yaml.safe_load(spread_floor_yaml.read_text(encoding="utf-8")) or {}
    stats = (data.get("floors") or {}).get(pair)
    if not stats:
        return None
    return float(stats["min_nonzero_spread_native"]) / points_per_pip


def load_all_floors(spread_floor_yaml: Path, points_per_pip: float = 10.0) -> Dict[str, float]:
    data = yaml.safe_load(spread_floor_yaml.read_text(encoding="utf-8")) or {}
    out: Dict[str, float] = {}
    for pair, stats in (data.get("floors") or {}).items():
        out[pair] = float(stats["min_nonzero_spread_native"]) / points_per_pip
    return out
