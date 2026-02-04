# tests/_indicator_contracts.py — shared test helpers for indicator contract auditing.
# Pure-test module: no production code changes.

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Module paths used by production (from core/backtester.py, scripts/batch_sweeper.py, core/backtester_helpers.py)
_CONFIRMATION_MODULE = "indicators.confirmation_funcs"
_VOLUME_MODULE = "indicators.volume_funcs"
_EXIT_MODULE = "indicators.exit_funcs"
_BASELINE_MODULE = "indicators.baseline_funcs"


def load_test_df() -> pd.DataFrame:
    """
    Load a deterministic OHLCV DataFrame from data/daily/.
    Prefers EUR_USD.csv if present; else the first .csv (sorted).
    Parses datetime, sets index, sorts, clips to 2019-01-01..2026-01-01, keeps last ~2200 rows.
    """
    from core.utils import normalize_ohlcv_schema

    data_dir = _REPO_ROOT / "data" / "daily"
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"data/daily not found at {data_dir}. Add data/daily/ with OHLCV CSVs for tests that need real data."
        )
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No .csv files in {data_dir}. Add at least one OHLCV CSV (e.g. EUR_USD.csv) for tests that need real data."
        )
    path = None
    for p in csvs:
        if p.name.upper() == "EUR_USD.CSV":
            path = p
            break
    if path is None:
        path = csvs[0]
    df = pd.read_csv(path)
    df = normalize_ohlcv_schema(df)
    date_col = "date"
    if date_col not in df.columns:
        date_col = next((c for c in df.columns if c.lower() in ("date", "time", "datetime")), None)
    if date_col is None:
        raise ValueError("No date/time column found after normalize_ohlcv_schema")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col)
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()
    start = pd.Timestamp("2019-01-01")
    end = pd.Timestamp("2026-01-01")
    df = df.loc[(df.index >= start) & (df.index <= end)]
    df = df.tail(2200)
    cols_lower = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols_lower)
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns (after normalize): {missing}")
    return df


def nan_rate_after_warmup(series: pd.Series, warmup: int) -> float:
    """Fraction of NaN in series after the first `warmup` elements."""
    if warmup <= 0:
        s = series
    else:
        s = series.iloc[warmup:]
    if len(s) == 0:
        return 0.0
    return float(s.isna().sum() / len(s))


def max_nan_streak_after_warmup(series: pd.Series, warmup: int) -> int:
    """Length of the longest consecutive run of NaN in series after the first `warmup` elements."""
    if warmup <= 0:
        s = series
    else:
        s = series.iloc[warmup:]
    if len(s) == 0:
        return 0
    is_na = s.isna()
    groups = is_na.ne(is_na.shift()).cumsum()
    na_groups = groups[is_na]
    if len(na_groups) == 0:
        return 0
    return int(na_groups.groupby(na_groups).size().max())


def unique_non_nan_values_after_warmup(series: pd.Series, warmup: int) -> np.ndarray:
    """Sorted array of unique non-NaN values in series after the first `warmup` elements."""
    if warmup <= 0:
        s = series
    else:
        s = series.iloc[warmup:]
    uniq = s.dropna().unique()
    return np.sort(uniq)


def run_causality_probe(
    fn: Callable[..., pd.DataFrame],
    df: pd.DataFrame,
    call_kwargs: Dict[str, Any],
    signal_col: str,
    future_window: int = 20,
) -> None:
    """
    Assert that the indicator does not use future bars: compute signal on df,
    mutate only the last `future_window` bars of OHLC with small deterministic offsets,
    recompute, and assert signals before the window are identical.
    """
    assert len(df) > future_window, (
        f"run_causality_probe needs len(df) > future_window (got {len(df)} and {future_window})"
    )
    ohlc = ["open", "high", "low", "close"]
    for c in ohlc:
        if c not in df.columns:
            raise ValueError(f"DataFrame must have column '{c}' for causality probe")
    df1 = df.copy()
    out1 = fn(df1, **call_kwargs)
    if not isinstance(out1, pd.DataFrame) or signal_col not in out1.columns:
        raise ValueError(f"Indicator must return a DataFrame with column '{signal_col}'")
    df2 = df.copy()
    np.random.seed(42)
    offsets = np.random.RandomState(42).randn(future_window, 4) * 1e-5
    for i, col in enumerate(ohlc):
        df2.iloc[-future_window:, df2.columns.get_loc(col)] += offsets[:, i]
    out2 = fn(df2, **call_kwargs)
    before = slice(None, -future_window)
    s1 = out1[signal_col].iloc[before]
    s2 = out2[signal_col].iloc[before]
    pd.testing.assert_series_equal(
        s1.reset_index(drop=True),
        s2.reset_index(drop=True),
        check_names=True,
        check_exact=False,
    )


def _discover_by_prefix(module_name: str, prefix: str) -> List[Tuple[str, Callable]]:
    """Import module and return (name, callable) for all callables whose name starts with prefix."""
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Cannot import {module_name}: {e}") from e
    pairs: List[Tuple[str, Callable]] = []
    for name, obj in inspect.getmembers(mod, callable):
        if name.startswith(prefix) and not name.startswith("_"):
            pairs.append((name, obj))
    return sorted(pairs, key=lambda x: x[0])


def _registry_if_exists(module_name: str, registry_attr: str = "REGISTRY"):
    """Return (name, callable) list from module.REGISTRY if present, else None."""
    try:
        mod = importlib.import_module(module_name)
        reg = getattr(mod, registry_attr, None)
        if reg is None or not callable(getattr(reg, "items", None)):
            return None
        return [(k, v) for k, v in reg.items() if callable(v)]
    except Exception:
        return None


def discover_c1_functions() -> List[Tuple[str, Callable]]:
    """
    Discover C1 confirmation functions from indicators.confirmation_funcs.
    Priority: registry if present; else all c1_* callables; else raise.
    """
    reg = _registry_if_exists(_CONFIRMATION_MODULE)
    if reg is not None:
        return sorted(reg, key=lambda x: x[0])
    pairs = _discover_by_prefix(_CONFIRMATION_MODULE, "c1_")
    if not pairs:
        raise ValueError(
            f"No C1 functions found in {_CONFIRMATION_MODULE}. "
            "Expected c1_* callables or a REGISTRY dict."
        )
    return pairs


def discover_volume_functions() -> List[Tuple[str, Callable]]:
    """
    Discover volume functions from indicators.volume_funcs.
    Priority: registry if present; else all volume_* callables; else raise.
    """
    reg = _registry_if_exists(_VOLUME_MODULE)
    if reg is not None:
        return sorted(reg, key=lambda x: x[0])
    pairs = _discover_by_prefix(_VOLUME_MODULE, "volume_")
    if not pairs:
        raise ValueError(
            f"No volume functions found in {_VOLUME_MODULE}. "
            "Expected volume_* callables or a REGISTRY dict."
        )
    return pairs


def discover_exit_functions() -> List[Tuple[str, Callable]]:
    """
    Discover exit functions from indicators.exit_funcs.
    Priority: registry if present; else all exit_* callables; else raise.
    """
    reg = _registry_if_exists(_EXIT_MODULE)
    if reg is not None:
        return sorted(reg, key=lambda x: x[0])
    pairs = _discover_by_prefix(_EXIT_MODULE, "exit_")
    if not pairs:
        raise ValueError(
            f"No exit functions found in {_EXIT_MODULE}. "
            "Expected exit_* callables or a REGISTRY dict."
        )
    return pairs


def discover_baseline_functions() -> List[Tuple[str, Callable]]:
    """
    Discover baseline functions from indicators.baseline_funcs.
    Priority: registry if present; else all baseline_* callables; else raise.
    """
    reg = _registry_if_exists(_BASELINE_MODULE)
    if reg is not None:
        return sorted(reg, key=lambda x: x[0])
    pairs = _discover_by_prefix(_BASELINE_MODULE, "baseline_")
    if not pairs:
        raise ValueError(
            f"No baseline functions found in {_BASELINE_MODULE}. "
            "Expected baseline_* callables or a REGISTRY dict."
        )
    return pairs
