"""Shared I/O + path helpers for L Arc 2 Step 2.

Arc 2 differs from arc 1 in:
  - Signal: mtf_alignment / 2_down_mixed / kijun / h=120 (categorical state, not univariate-extreme threshold)
  - Trade count: 3,993 (vs arc 1 45,673) due to exposure cap binding heavily at h=120
  - Held window: REAL (mean bars_held ~47.5, max 120), not degenerate at 1 bar
  - trade_paths.csv schema: richer (OHLC, is_held/forward_bar, data_end_flag, bar_offset starts at 0)
  - 4 new pre-signal context features (cum_logret_1h_{24,72,168}, vol_realized_1h_24h)
  - 3 Amendment 4 clustering features (fwd_realized_range_atr, fwd_fraction_time_above_entry,
    fwd_max_consecutive_directional_bars)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STEP1_DIR = REPO_ROOT / "results" / "l_arc_2" / "step1_verbatim"
STEP2_DIR = REPO_ROOT / "results" / "l_arc_2" / "step2_descriptive"
CONFIG_PATH = REPO_ROOT / "configs" / "wfo_l_arc2_verbatim.yaml"
DATA_1H_DIR = REPO_ROOT / "data" / "1hr"
SPREAD_FLOOR_PATH = REPO_ROOT / "configs" / "spread_floors_5ers.yaml"

PAIRS: List[str] = [
    "AUD_CAD", "AUD_CHF", "AUD_JPY", "AUD_NZD", "AUD_USD",
    "CAD_CHF", "CAD_JPY", "CHF_JPY",
    "EUR_AUD", "EUR_CAD", "EUR_CHF", "EUR_GBP", "EUR_JPY", "EUR_NZD", "EUR_USD",
    "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_JPY", "GBP_NZD", "GBP_USD",
    "NZD_CAD", "NZD_CHF", "NZD_JPY", "NZD_USD",
    "USD_CAD", "USD_CHF", "USD_JPY",
]

# Forward-horizon cap. Default = 240 (10 trading days at 1H), extended to 480 if stability fires.
FORWARD_HORIZON_BARS_DEFAULT: int = 240
FORWARD_HORIZON_BARS_EXTENDED: int = 480

# Verbatim time-exit horizon (arc 2 registry entry 2)
VERBATIM_TIME_EXIT_H: int = 120
VERBATIM_SL_ATR_MULT: float = 2.0

# Hash-based seed convention (v1.1 Amendment 11)
def hash_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")


RANDOM_SEED: int = hash_seed("l_arc_2_step2_root")

# Shadow grids (per task spec)
ENTRY_DELAYS = [1, 2, 3, 5, 10]                                 # 1 verbatim baseline
SL_DISTANCES = [1.0, 1.5, 2.0, 2.5, 3.0]                        # 2.0 verbatim
TIME_EXIT_H = [1, 3, 6, 12, 24, 48, 120, 240]                   # 120 verbatim
H_GRID = [1, 3, 6, 12, 24, 48, 72, 120, 240, 360, 480]          # forward-horizon checkpoints
HEALD_T = [1, 3, 5, 10, 20]
SPREAD_MULT = [0.5, 1.0, 1.5, 2.0]

POINTS_PER_PIP = 10.0


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
    """Load step 1 trades_verbatim.csv with timestamps parsed."""
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
    """Load step 1 signals_log.csv (all fires, taken + dropped)."""
    path = STEP1_DIR / "signals_log.csv"
    df = pd.read_csv(path)
    df["signal_bar_ts"] = pd.to_datetime(df["signal_bar_ts"])
    return df


def load_pair_1h(pair: str) -> pd.DataFrame:
    path = DATA_1H_DIR / f"{pair}.csv"
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Bit-identical to engine's `_wilder_atr` (period-1 simple seed, then Wilder)."""
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


def floor_pips_for_pair(spread_floor_yaml: Path, pair: str, points_per_pip: float = POINTS_PER_PIP) -> Optional[float]:
    data = yaml.safe_load(spread_floor_yaml.read_text(encoding="utf-8")) or {}
    stats = (data.get("floors") or {}).get(pair)
    if not stats:
        return None
    return float(stats["min_nonzero_spread_native"]) / points_per_pip


def load_all_floors(spread_floor_yaml: Path = SPREAD_FLOOR_PATH,
                    points_per_pip: float = POINTS_PER_PIP) -> Dict[str, float]:
    data = yaml.safe_load(spread_floor_yaml.read_text(encoding="utf-8")) or {}
    out: Dict[str, float] = {}
    for pair, stats in (data.get("floors") or {}).items():
        out[pair] = float(stats["min_nonzero_spread_native"]) / points_per_pip
    return out


# -- distribution helpers (op spec §11.1: mean/std/skew/kurt/min/p1..p99/max + histogram) --

PERCENTILES: List[int] = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
DIST_COLUMNS: List[str] = (
    ["n", "n_nan", "mean", "std", "skew", "kurt", "min"]
    + [f"p{p}" for p in PERCENTILES]
    + ["max"]
)


def _skew_kurt(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return (float("nan"), float("nan"))
    mu = float(x.mean())
    var = float(x.var(ddof=0))
    if var <= 0:
        return (0.0, -3.0)
    sd = var ** 0.5
    m3 = float(((x - mu) ** 3).mean())
    m4 = float(((x - mu) ** 4).mean())
    return (m3 / (sd ** 3), m4 / (var ** 2) - 3.0)


def describe_distribution(values, *, name: str) -> pd.DataFrame:
    arr = np.asarray(list(values), dtype=float)
    n_total = arr.size
    finite = arr[np.isfinite(arr)]
    n = finite.size
    n_nan = n_total - n
    row = {
        "n": n, "n_nan": n_nan,
        "mean": float(finite.mean()) if n else float("nan"),
        "std": float(finite.std(ddof=1)) if n >= 2 else float("nan"),
        "skew": float("nan"), "kurt": float("nan"),
        "min": float(finite.min()) if n else float("nan"),
        "max": float(finite.max()) if n else float("nan"),
    }
    if n:
        sk, ku = _skew_kurt(finite)
        row["skew"] = sk; row["kurt"] = ku
        pcts = np.percentile(finite, PERCENTILES, method="linear")
        for p, v in zip(PERCENTILES, pcts):
            row[f"p{p}"] = float(v)
    else:
        for p in PERCENTILES:
            row[f"p{p}"] = float("nan")
    return pd.DataFrame([row], index=[name])[DIST_COLUMNS]


def histogram_csv(values, *, bins: int = 50,
                  bin_range: Optional[tuple[float, float]] = None) -> pd.DataFrame:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return pd.DataFrame({"bin_left": [], "bin_right": [], "count": []})
    if bin_range is None:
        bin_range = (float(finite.min()), float(finite.max()))
    if bin_range[1] <= bin_range[0]:
        bin_range = (bin_range[0], bin_range[0] + 1e-9)
    counts, edges = np.histogram(finite, bins=bins, range=bin_range)
    return pd.DataFrame({
        "bin_left": edges[:-1], "bin_right": edges[1:],
        "count": counts.astype(np.int64),
    })


def write_distribution(values, out_path: Path, *, metric_name: str,
                       degenerate: bool = False, degenerate_reason: str = "",
                       bins: int = 50, hist_path: Optional[Path] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(list(values), dtype=float)
    dist = describe_distribution(arr, name=metric_name)
    csv_body = dist.to_csv(index=True, index_label="metric", lineterminator="\n")
    with out_path.open("w", encoding="utf-8", newline="") as f:
        if degenerate:
            f.write(f"# degenerate_by_construction: true | reason: {degenerate_reason}\n")
        f.write(csv_body)
    if hist_path is not None:
        hist = histogram_csv(arr, bins=bins)
        hist.to_csv(hist_path, index=False, lineterminator="\n")


def write_per_fold_distribution(df: pd.DataFrame, value_col: str, fold_col: str,
                                out_path: Path, metric_name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[pd.DataFrame] = []
    for fid in sorted(df[fold_col].unique()):
        sub = df[df[fold_col] == fid][value_col].to_numpy()
        rows.append(describe_distribution(sub, name=f"fold_{int(fid)}"))
    pool = describe_distribution(df[value_col].to_numpy(), name="pool")
    out = pd.concat(rows + [pool], axis=0)
    csv_body = out.to_csv(index=True, index_label=f"{metric_name}__fold", lineterminator="\n")
    out_path.write_text(csv_body, encoding="utf-8")
