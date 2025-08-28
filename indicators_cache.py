# indicators_cache.py — v1.9.6 (stable)

from __future__ import annotations

import hashlib
import json  # ✅ required for dumps in hashing/logs
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

"""
v1.9.6 — Indicator output caching utilities.

Key:
    (pair, timeframe, indicator_role, indicator_name, params_hash, data_hash, scope_key)

- params_hash is a sorted JSON of params, hashed.
- data_hash is a stable SHA256 over OHLCV arrays + index.
- Stores single-file artifact per key in cache/<role>/<name>/<pair>/<key>.(parquet|feather)

Public helpers:
    compute_data_hash(df)
    compute_params_hash(params_dict)
    cache_key_parts(...)
    load_from_cache(...)
    save_to_cache(...)
    purge_cache(...)
    describe_cache(...)

Integrates in backtester.apply_indicators().
"""

# Hidden index column used only for feather (since it doesn't preserve index)
_INDEX_COL = "__index__"

# --- Shared confirmation resolver (C1 and C2 use the same pool) ---
import importlib


def _resolve_params(cfg: dict, fq_candidates: list[str]) -> dict:
    """
    Fetch params from cfg['indicator_params'] using the first matching FQ name,
    else return {}. This lets 'c2_x' reuse 'c1_x' params if only those are defined.
    """
    params_map = cfg.get("indicator_params") or {}
    for fq in fq_candidates:
        if fq in params_map:
            return params_map[fq] or {}
    return {}


# =========================
#  Hashing helpers
# =========================


def _human_size(nbytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    x = float(nbytes)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.1f} {units[i]}"


def compute_data_hash(df: pd.DataFrame) -> str:
    """
    Stable hash of price arrays + index. Tolerates missing volume and NaNs.
    - Normalizes index to string (timezone/type-safe)
    - Converts NaN/inf to None so JSON is canonical
    - Rounds floats (10 dp) to reduce insignificant noise
    """
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    if not cols:
        raise ValueError("Data hash requires at least one of: open, high, low, close, volume")

    # Round floats for stability; coerce non-finite to None
    arr = (
        df[cols]
        .astype(float)
        .round(10)
        .replace([np.inf, -np.inf], np.nan)
        .where(lambda x: pd.notna(x), None)
        .to_numpy()
        .tolist()
    )

    payload = {
        "index": df.index.astype(str).tolist(),  # string form avoids tz dtype surprises
        "data": arr,
        "cols": cols,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def compute_params_hash(params: Dict | None) -> str:
    """
    Stable hash of indicator params. Handles exotic values via default=str.
    """
    norm = json.dumps(
        params or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


# =========================
#  Key building
# =========================


def cache_key_parts(
    pair: str,
    timeframe: str,
    role: str,  # "c1", "c2", "baseline", "volume", "exit"
    name: str,  # e.g. "twiggs_money_flow"
    params_hash: str,
    data_hash: str,
    scope_key: Optional[str] = None,
) -> Tuple[Path, str]:
    scope = scope_key.strip().replace("/", "_") if scope_key else "global"
    key = f"{pair}__{timeframe}__{role}__{name}__{params_hash}__{data_hash}__{scope}"
    # Directory layout keeps things tidy and easy to purge by role/name/pair
    return Path(role) / name / pair, key


def _artifact_path(cache_dir: Path, parts_path: Path, key: str, fmt: str) -> Path:
    ext = ".parquet" if fmt == "parquet" else ".feather"
    return cache_dir / parts_path / f"{key}{ext}"


# =========================
#  Load / Save
# =========================


def load_from_cache(
    cache_dir: str | Path, fmt: str, parts_path: Path, key: str
) -> Optional[pd.DataFrame]:
    path = _artifact_path(Path(cache_dir), parts_path, key, fmt)
    if not path.exists():
        return None
    if fmt == "parquet":
        return pd.read_parquet(path)  # preserves index
    else:
        df = pd.read_feather(path)
        # For feather, restore index if we stored it
        if _INDEX_COL in df.columns:
            idx = pd.to_datetime(df[_INDEX_COL], errors="ignore")
            df = df.drop(columns=[_INDEX_COL])
            if isinstance(idx, pd.DatetimeIndex):
                df = df.set_index(idx)
            else:
                df = df.set_index(pd.Index(idx, name=None))
        return df


def save_to_cache(
    cache_dir: str | Path, fmt: str, parts_path: Path, key: str, df: pd.DataFrame
) -> Path:
    outdir = Path(cache_dir) / parts_path
    outdir.mkdir(parents=True, exist_ok=True)
    path = _artifact_path(Path(cache_dir), parts_path, key, fmt)

    if fmt == "parquet":
        # Keep index; compress to save space
        df.to_parquet(path, index=True, compression="snappy")
    else:
        # Feather cannot store index; tuck it into a hidden column so we can restore it on load
        tmp = df.copy()
        tmp[_INDEX_COL] = df.index.astype(str)
        tmp.reset_index(drop=True).to_feather(path)
    return path


# =========================
#  Maintenance
# =========================


def _iter_cache_files(cache_dir: str | Path) -> Iterable[Path]:
    root = Path(cache_dir)
    if not root.exists():
        return []
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def purge_cache(
    cache_dir: str | Path,
    *,
    roles: Optional[Iterable[str]] = None,
    names: Optional[Iterable[str]] = None,
    pairs: Optional[Iterable[str]] = None,
) -> int:
    """
    Delete cached artifacts selectively. Returns number of files removed.
    Examples:
        purge_cache("cache")                       # all
        purge_cache("cache", roles=["c1","c2"])
        purge_cache("cache", names=["twiggs_money_flow"])
        purge_cache("cache", pairs=["EUR_USD"])
    """
    base = Path(cache_dir)
    if not base.exists():
        return 0

    removed = 0
    roles_glob = list(roles) if roles else ["*"]
    names_glob = list(names) if names else ["*"]
    pairs_glob = list(pairs) if pairs else ["*"]

    for r in roles_glob:
        for n in names_glob:
            for p in pairs_glob:
                d = base / r / n / p
                if not d.exists():
                    continue
                for f in d.glob("*.*"):
                    try:
                        f.unlink()
                        removed += 1
                    except Exception:
                        pass
    return removed


def describe_cache(cache_dir: str | Path) -> Dict[str, Any]:
    """
    Basic stats for the cache directory.
    Returns: {"files": int, "bytes": int, "human": "X MB"}
    """
    base = Path(cache_dir)
    if not base.exists():
        return {"files": 0, "bytes": 0, "human": "0 B"}

    total = 0
    count = 0
    for f in _iter_cache_files(base):
        try:
            total += f.stat().st_size
            count += 1
        except Exception:
            pass

    return {"files": count, "bytes": total, "human": _human_size(total)}


# --- Resolver helpers (single, canonical versions) ---
# --- Resolver + caller helpers (canonical, no duplicates) ---
from typing import Callable


def _resolve_confirm_func(short_name: str, role: str = "c1") -> Tuple[str, Callable]:
    """
    Confirmation resolver (C1/C2 share the c1_* pool).
    Tries: <role>_<short> -> c1_<short> -> <short>
    Returns: (fully_qualified_name, callable)
    """
    mod = importlib.import_module("indicators.confirmation_funcs")
    short = (str(short_name) or "").strip()
    candidates = [f"{role}_{short}", f"c1_{short}", short]
    for name in candidates:
        if hasattr(mod, name):
            return f"indicators.confirmation_funcs.{name}", getattr(mod, name)
    raise ImportError(
        f"Function '{candidates[0]}' not found in module 'indicators.confirmation_funcs'. Tried {candidates}."
    )


def _resolve_indicator_func(
    role: str, short_name: str, verbose: bool = False
) -> Tuple[Optional[str], Optional[Callable]]:
    """
    Generic resolver for non-confirmation roles (baseline/volume/exit).
    Uses backtester_helpers._resolve_indicator_func if available; else local fallback.
    """
    # Prefer centralized helper if your project has it
    try:
        from backtester_helpers import _resolve_indicator_func as _bh_resolve_indicator_func

        return _bh_resolve_indicator_func(role, short_name, verbose)
    except Exception:
        pass

    role_to_module = {
        "baseline": "indicators.baseline_funcs",
        "volume": "indicators.volume_funcs",
        "exit": "indicators.exit_funcs",
    }
    mod_name = role_to_module.get(role)
    if not mod_name:
        if verbose:
            print(f"⚠️ No module mapping for role '{role}'")
        return (None, None)

    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        if verbose:
            print(f"❌ import {mod_name} failed: {e}")
        return (None, None)

    short = (str(short_name) or "").strip()
    for cand in (f"{role}_{short}", short):  # e.g., baseline_ema, ema
        if hasattr(mod, cand):
            return (f"{mod_name}.{cand}", getattr(mod, cand))
    if verbose:
        print(f"❌ {mod_name}: none of {[f'{role}_{short}', short]} found")
    return (None, None)


def _call_indicator(
    func: Optional[Callable], df: pd.DataFrame, params: dict, signal_col: str
) -> pd.DataFrame:
    """
    Unified call wrapper.
    - If backtester_helpers._call_indicator exists, use it (project canonical).
    - Else, call func(df_copy, signal_col=..., **params) if supported.
      Fallbacks if signature differs, ensuring signal_col exists.
    """
    # Prefer centralized helper if present
    try:
        from backtester_helpers import _call_indicator as _bh_call_indicator
    except Exception:
        _bh_call_indicator = None

    if _bh_call_indicator is not None:
        return _bh_call_indicator(func, df, params, signal_col)

    if func is None:
        # Nothing to do, ensure column exists
        if signal_col not in df.columns:
            df = df.copy()
            df[signal_col] = 0
        return df

    dfc = df.copy()
    try:
        # Preferred convention in this project
        out = func(dfc, signal_col=signal_col, **(params or {}))
    except TypeError:
        # Try without the explicit keyword if the function has a different signature
        out = func(dfc, **(params or {}))

    if isinstance(out, pd.DataFrame):
        return out
    elif out is None:
        # Indicator mutated in-place or returned nothing; ensure col exists
        if signal_col not in dfc.columns:
            dfc[signal_col] = 0
        return dfc
    else:
        # Indicator returned a Series/array/scalar: attach as the signal column
        dfc[signal_col] = out
        return dfc


def _params_for(full_func_name: str, cfg: dict) -> dict:
    return (cfg.get("indicator_params") or {}).get(full_func_name, {}) or {}


def apply_indicators_with_cache(df: pd.DataFrame, pair: str, cfg: dict) -> pd.DataFrame:
    """
    Applies indicators for roles (c1, c2, baseline, volume, exit) with caching.

    FIX (v1.9.8): C2 confirmation resolves from the same pool as C1 (indicators.confirmation_funcs.c1_*).
    If an indicator ignores `signal_col` and writes to 'c1_signal' only, we synthesize 'c2_signal'
    by copying 'c1_signal' so downstream logic never KeyErrors.
    """
    cache_cfg = cfg.get("cache") or {}
    cache_on = cache_cfg.get("enabled", True)
    cache_dir = cache_cfg.get("dir", "cache")
    cache_fmt = cache_cfg.get("format", "parquet")
    scope_key = cache_cfg.get("scope_key")
    timeframe = cfg.get("timeframe", "D")
    verbose = (cfg.get("tracking") or {}).get("verbose_logs", False)

    roles_filter = set((cfg.get("cache") or {}).get("roles") or [])

    def role_enabled(role: str) -> bool:
        return not roles_filter or role in roles_filter

    inds = cfg.get("indicators") or {}

    def _get(k, default=None):
        return getattr(inds, k, default) if hasattr(inds, k) else inds.get(k, default)

    data_hash = compute_data_hash(df)
    saves = hits = 0

    def run_role(role: str, name: Optional[str], signal_col: str):
        nonlocal df, saves, hits
        if not name:
            return

        # Resolve function + params
        if role in ("c1", "c2"):
            try:
                full_name, func = _resolve_confirm_func(name, role=role)
            except Exception as e:
                if verbose:
                    print(f"❌ Confirm resolver failed for {role}/{name}: {e}")
                return
            params = (cfg.get("indicator_params") or {}).get(full_name, {}) or {}
        else:
            full_name, func = _resolve_indicator_func(role, name, verbose)
            if func is None:
                return
            params = (cfg.get("indicator_params") or {}).get(full_name, {}) or {}

        params_hash = compute_params_hash(params)
        parts_path, key = cache_key_parts(
            pair, timeframe, role, name, params_hash, data_hash, scope_key
        )

        # Try cache
        if cache_on:
            cached = load_from_cache(cache_dir, cache_fmt, parts_path, key)
            if cached is not None and not cached.empty:
                for col in cached.columns:
                    if col not in df.columns:
                        df[col] = cached[col]
                hits += 1
                if verbose:
                    print(f"⚡ Cache hit: {role}/{name}")
                return

        # Compute fresh
        before_cols = set(df.columns)
        df = _call_indicator(func, df, params, signal_col)

        # --- GUARANTEE signal_col exists (handles shared C1 funcs that ignore signal_col) ---
        if signal_col not in df.columns:
            if role == "c2" and "c1_signal" in df.columns:
                # Mirror c1 as a pragmatic fallback so downstream logic has c2_signal
                if verbose:
                    print(
                        "ℹ️  C2 fallback: copying c1_signal → c2_signal (indicator ignored signal_col)"
                    )
                df[signal_col] = (
                    pd.to_numeric(df["c1_signal"], errors="coerce")
                    .fillna(0)
                    .clip(-1, 1)
                    .astype(int)
                )
            else:
                # Create a neutral column (all zeros) to avoid KeyError
                if verbose:
                    print(f"ℹ️  Creating missing signal column '{signal_col}' filled with 0")
                df[signal_col] = 0

        # Normalize to int ∈ {-1,0,1}
        df[signal_col] = (
            pd.to_numeric(df[signal_col], errors="coerce").fillna(0).clip(-1, 1).astype(int)
        )

        created_cols = [
            c for c in (set(df.columns) - before_cols) if (c.endswith("_signal") or c == "baseline")
        ]

        if cache_on and created_cols:
            save_to_cache(cache_dir, cache_fmt, parts_path, key, df[list(created_cols)].copy())
            saves += 1
            if verbose:
                print(f"📝 Cache save: {role}/{name} -> {parts_path}")

    # Run roles (gated)
    run_role("c1", _get("c1"), "c1_signal")
    if _get("use_c2", False) and role_enabled("c2"):
        run_role("c2", _get("c2"), "c2_signal")
    if _get("use_baseline", False) and role_enabled("baseline"):
        run_role("baseline", _get("baseline"), "baseline_signal")
    if _get("use_volume", False) and role_enabled("volume"):
        run_role("volume", _get("volume"), "volume_signal")
    if _get("use_exit", False) and role_enabled("exit"):
        run_role("exit", _get("exit"), "exit_signal")

    if verbose:
        print(f"📦 cache stats → saves={saves} hits={hits}")
    return df
