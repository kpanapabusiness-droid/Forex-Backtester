# =============================================
# backtester.py — v1.9.8 (clean patched)
# =============================================
# - Entry ONLY from entry_signal (+1/-1)
# - Continuation logic lives in signal_logic.apply_signal_logic
# - TP1 partial, move to BE after TP1, TS = 1.5× ATR (ENTRY ATR)
# - Trailing activates after 2× ATR move from entry (close-based)
# - Intrabar priority: tp_first | sl_first | best | worst
# - Cache-aware indicators (roles: c1/c2/baseline/volume/exit)
# - DBCVIX risk filter (reduce/block) with CSV source
# - WFO entrypoint: run_backtest_walk_forward(...)
# - Writes: results/trades.csv, results/summary.txt, results/equity_curve.csv
# =============================================

from __future__ import annotations

__version__ = "1.9.8"

# stdlib
import importlib
import inspect
import math
import pkgutil
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# 3rd-party
import pandas as pd

from core.backtester_helpers import finalize_trade_row

# project utils
from core.utils import (
    calculate_atr,
    get_pip_size,
    pip_value_per_lot,
)

# caching
from indicators_cache import (
    cache_key_parts,
    compute_data_hash,
    compute_params_hash,
    load_from_cache,
    save_to_cache,
)

# optional helpers (present in your project)
from validators_config import load_and_validate_config
from validators_util import validate_contract

# signal logic
try:
    from core.signal_logic import apply_signal_logic  # required
except Exception as e:
    raise ImportError(
        "Could not import 'apply_signal_logic' from core.signal_logic.py. "
        "Ensure it defines: def apply_signal_logic(df: pd.DataFrame, config: dict) -> pd.DataFrame\n"
        f"Underlying error: {e}"
    )

# =============================================
# Config / constants
# =============================================

RESULTS_DIR_DEFAULT = Path("results")

TRADES_COLS: List[str] = [
    "pair",
    "entry_date",
    "entry_price",
    "direction",
    "direction_int",
    "atr_at_entry_price",
    "atr_at_entry_pips",
    "lots_total",
    "lots_half",
    "lots_runner",
    # Risk / filters
    "risk_pct_used",
    "dbcvix_val",
    "dbcvix_flag",
    # --- Entry levels (immutable for audits) ---
    "tp1_price",
    "sl_price",
    "tp1_at_entry_price",
    "sl_at_entry_price",
    # --- State & TS ---
    "tp1_hit",
    "breakeven_after_tp1",
    "ts_active",
    "ts_level",
    # --- Exit info ---
    "entry_idx",
    "exit_date",
    "exit_price",
    "exit_reason",
    "sl_at_exit_price",
    # --- Results ---
    "pnl",
    "win",
    "loss",
    "scratch",
    "spread_pips_used",
]

PathLikeT = Union[str, Path]

# =============================================
# Small helpers
# =============================================


def pip_size_for_pair(pair: str) -> float:
    """Delegate to utils.get_pip_size; safe float."""
    try:
        return float(get_pip_size(pair))
    except Exception:
        return 0.01 if str(pair).upper().endswith("JPY") else 0.0001


def _int_signal(val) -> int:
    """Coerce any scalar to {-1,0,1} safely."""
    try:
        v = float(val)
        if math.isnan(v):
            return 0
        v = int(v)
        return 1 if v > 0 else (-1 if v < 0 else 0)
    except Exception:
        return 0


def ensure_results_dir(path: Any, *_, **__) -> Any:
    """
    Backward-compatible helper that ignores extra positional/keyword args.
    Preserves prior return type:
      - if input was str/bytes, return str(path)
      - else return a Path
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p) if isinstance(path, (str, bytes)) else p


def intrabar_sequence(priority: str) -> List[str]:
    p = (priority or "tp_first").lower()
    if p == "sl_first":
        return ["sl", "tp"]
    if p == "best":
        return ["tp", "sl"]  # favorable
    if p == "worst":
        return ["sl", "tp"]  # unfavorable
    return ["tp", "sl"]  # default


def load_config(config_path: PathLikeT = "configs/config.yaml") -> dict:
    """Load + validate YAML config using validators_config."""
    path = Path(config_path)
    if not path.exists():
        # search upward a bit
        for root in (Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent):
            cand = root / "configs" / "config.yaml"
            if cand.exists():
                path = cand
                break
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return load_and_validate_config(str(path))


# =============================================
# DBCVIX — config + loader + resolver
# =============================================


def _get_nested(d, *path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_dbcvix_config(cfg: dict) -> dict:
    """
    Normalize dbcvix config from several shapes to:
      {enabled, mode, threshold, reduce_risk_to, source, csv_path}
    """
    candidates = [
        _get_nested(cfg, "filters", "dbcvix"),
        _get_nested(cfg, "risk_filters", "dbcvix"),
        _get_nested(cfg, "dbcvix"),
        _get_nested(cfg, "rules", "risk_filters", "dbcvix"),
        _get_nested(cfg, "risk", "dbcvix"),
    ]
    raw = next((c for c in candidates if isinstance(c, dict)), {}) or {}

    # unify keys
    csv_path = raw.get("csv_path") or raw.get("path") or raw.get("file") or raw.get("filepath")
    mode = raw.get("mode", "reduce")
    threshold = raw.get("threshold")
    reduce_to = raw.get("reduce_risk_to") or raw.get("target_risk") or raw.get("risk_to")
    source = raw.get("source") or ("csv" if csv_path else None)
    enabled = bool(raw.get("enabled", False))

    def _num(x):
        try:
            return float(x) if x is not None else None
        except Exception:
            return None

    return {
        "enabled": enabled,
        "mode": mode if mode in ("reduce", "block") else "reduce",
        "threshold": _num(threshold),
        "reduce_risk_to": _num(reduce_to),
        "source": source if source in (None, "csv", "synthetic") else "csv",
        "csv_path": csv_path,
    }


def load_dbcvix_series(cfg: dict) -> pd.Series | None:
    """
    Load DBCVIX time series from CSV: columns {date, value} (case-insensitive variants allowed).
    Returns pd.Series indexed by datetime or None.
    """
    f = resolve_dbcvix_config(cfg)
    if not (f["enabled"] and f["source"] == "csv" and f["csv_path"]):
        return None
    try:
        p = Path(f["csv_path"])
        df = pd.read_csv(p)
        date_col = next(
            (c for c in df.columns if c.lower() in ("date", "time", "timestamp", "datetime")), None
        )
        val_col = next(
            (c for c in df.columns if c.lower() in ("value", "dbcvix", "cvix", "regime")), None
        )
        if not date_col or not val_col:
            raise ValueError("DBCVIX CSV must contain date & value columns (case-insensitive).")
        s = pd.Series(
            pd.to_numeric(df[val_col], errors="coerce").astype(float).values,
            index=pd.to_datetime(df[date_col], errors="coerce"),
        ).dropna()
        return s.sort_index()
    except Exception as e:
        print(f"⚠️  DBCVIX CSV load failed: {e}")
        return None


def _last_val_asof(series: Optional[pd.Series], ts) -> Optional[float]:
    if series is None or getattr(series, "empty", True):
        return None
    try:
        sub = series.loc[: pd.to_datetime(ts)]
        if sub.empty:
            return None
        return float(sub.iloc[-1])
    except Exception:
        return None


def resolve_dbcvix_risk(dbcvix_series, trade_date, base_risk, fcfg):
    """Return (risk_pct_eff, dbcvix_flag, dbcvix_val) with risk in DECIMAL units."""
    try:
        if not fcfg or not fcfg.get("enabled"):
            return float(base_risk), False, None
        thr = fcfg.get("threshold")
        if thr is None:
            return float(base_risk), False, None
        mode = str(fcfg.get("mode", "reduce"))
        reduce_to = float(fcfg.get("reduce_risk_to", 0.01))  # decimal (1%)
        val = _last_val_asof(dbcvix_series, trade_date)
        if val is None:
            return float(base_risk), False, None
        if val > float(thr):
            if mode == "block":
                return 0.0, True, val
            return min(float(base_risk), reduce_to), True, val
        return float(base_risk), False, val
    except Exception:
        return float(base_risk), False, None


# =============================================
# Indicator discovery + cache application
# =============================================


def _iter_candidate_modules(role: str) -> List[str]:
    base = "indicators"
    mapping = {
        "c1": [f"{base}.confirmation_funcs", f"{base}.c1_funcs"],
        "c2": [f"{base}.confirmation_funcs", f"{base}.c2_funcs"],
        "baseline": [f"{base}.baseline_funcs"],
        "volume": [f"{base}.volume_funcs"],
        "exit": [f"{base}.exit_funcs", f"{base}.exit_rules"],
    }
    mods = list(mapping.get(role, []))
    try:
        pkg = importlib.import_module(base)
        if hasattr(pkg, "__path__"):
            for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
                if m.name.endswith(("_funcs", "_rules", "_indicators")) and m.name not in mods:
                    mods.append(m.name)
    except Exception:
        pass
    return mods


def _resolve_indicator_func(role: str, name: Optional[str], verbose: bool):
    if not name:
        return None, None
    wanted = str(name).lower()
    for modname in _iter_candidate_modules(role):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for cand in (name, f"{role}_{name}"):
            func = getattr(mod, cand, None)
            if callable(func):
                full = cand if cand.startswith(f"{role}_") else f"{role}_{name}"
                return full, func
        # fuzzy suffix (e.g., *_{name})
        for n, o in inspect.getmembers(mod, inspect.isfunction):
            low = n.lower()
            if low == wanted or low.endswith("_" + wanted):
                full = n if n.startswith(f"{role}_") else f"{role}_{n}"
                return full, o
    if verbose:
        print(f"⚠️  {role}/{name} not found in indicators/* modules")
    return None, None


def _call_indicator(func, frame: pd.DataFrame, params: dict, signal_col: str) -> pd.DataFrame:
    sig = inspect.signature(func)
    kwargs = {k: v for k, v in (params or {}).items() if k in sig.parameters}
    if "signal_col" in sig.parameters:
        kwargs["signal_col"] = signal_col
    return func(frame, **kwargs)


def apply_indicators_with_cache(df: pd.DataFrame, pair: str, cfg: dict) -> pd.DataFrame:
    """
    Apply indicators (c1/c2/baseline/volume/exit) with caching.
    C2 shares the confirmation pool with C1 (indicators.confirmation_funcs).
    """
    import os

    cache_cfg = cfg.get("cache") or {}
    # Honor FB_NO_CACHE env var or config cache.enabled setting
    cache_on = os.environ.get("FB_NO_CACHE") != "1" and cache_cfg.get("enabled", True)
    cache_dir = cache_cfg.get("dir", "cache")
    cache_fmt = cache_cfg.get("format", "parquet")
    scope_key = cache_cfg.get("scope_key")
    timeframe = cfg.get("timeframe", "D")
    verbose = (cfg.get("tracking") or {}).get("verbose_logs", False)

    inds = cfg.get("indicators") or {}

    def _get(k, default=None):
        return getattr(inds, k, default) if hasattr(inds, k) else inds.get(k, default)

    data_hash = compute_data_hash(df)
    saves = hits = 0

    def _params_for(full_func_name: str) -> dict:
        return (cfg.get("indicator_params") or {}).get(full_func_name, {}) or {}

    def _resolve_confirm_func(short_name: str, role: str = "c1"):
        mod = importlib.import_module("indicators.confirmation_funcs")
        candidates = [f"{role}_{short_name}", f"c1_{short_name}", short_name]
        for name in candidates:
            if hasattr(mod, name) and callable(getattr(mod, name)):
                return f"indicators.confirmation_funcs.{name}", getattr(mod, name)
        raise ImportError(f"No confirmation function found for {role}/{short_name}")

    def run_role(role: str, name: Optional[str], signal_col: str):
        nonlocal df, saves, hits
        if not name:
            return

        if role in ("c1", "c2"):
            try:
                full_name, func = _resolve_confirm_func(name, role=role)
            except Exception as e:
                if verbose:
                    print(f"❌ Confirm resolver failed for {role}/{name}: {e}")
                return
            params = _params_for(full_name)
        else:
            full_name, func = _resolve_indicator_func(role, name, verbose)
            if func is None:
                return
            params = _params_for(full_name)

        params_hash = compute_params_hash(params)
        parts_path, key = cache_key_parts(
            pair, timeframe, role, name, params_hash, data_hash, scope_key
        )

        # cache
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

        # compute fresh
        before_cols = set(df.columns)
        df = _call_indicator(func, df, params, signal_col)
        created_cols = [
            c for c in (set(df.columns) - before_cols) if c.endswith("_signal") or c == "baseline"
        ]

        if cache_on and created_cols:
            save_to_cache(cache_dir, cache_fmt, parts_path, key, df[list(created_cols)].copy())
            saves += 1
            if verbose:
                print(f"📝 Cache save: {role}/{name} -> {parts_path}")

    # ensure ATR first
    df = calculate_atr(df)

    run_role("c1", _get("c1"), "c1_signal")
    if _get("use_c2", False) and _get("c2"):
        run_role("c2", _get("c2"), "c2_signal")
    if _get("use_baseline", False) and _get("baseline"):
        run_role("baseline", _get("baseline"), "baseline_signal")
    if _get("use_volume", False) and _get("volume"):
        run_role("volume", _get("volume"), "volume_signal")
    if _get("use_exit", False) and _get("exit"):
        run_role("exit", _get("exit"), "exit_signal")

    if verbose:
        print(f"📦 cache stats → saves={saves} hits={hits}")

    # CACHE diagnostic for parity runs
    if not cache_on:
        print(
            f"[CACHE] disabled (FB_NO_CACHE={os.environ.get('FB_NO_CACHE', 'unset')} config_enabled={cache_cfg.get('enabled', True)})"
        )
    elif hits > 0 or saves > 0:
        print(f"[CACHE] used saves={saves} hits={hits}")

    return df


# =======
def _apply_trailing_stop_fill(
    row: dict, *, final_stop_price: float, is_long: bool, pair: str, cfg: dict
) -> dict:
    """
    Force the exit at the breached trailing stop (PnL-only slippage model) and stamp audit fields.
    """
    ps = pip_size_for_pair(pair)
    fills = (cfg or {}).get("fills") or {}
    sl = fills.get("slippage") or {}
    slip_pips = float(sl.get("pips", 0.0)) if sl.get("enabled", False) else 0.0
    slip_px = slip_pips * ps

    exit_px = final_stop_price - slip_px if is_long else final_stop_price + slip_px

    row["exit_reason"] = "trailing_stop"
    row["exit_price"] = float(exit_px)
    row["sl_at_exit_price"] = float(final_stop_price)  # <- audit cares about this
    row["ts_active"] = True
    row["ts_level"] = float(final_stop_price)
    row["slippage_pips"] = float(slip_pips)
    return row


def _finalize_and_append_trade(
    trades_list: list, trade_row: dict, *, current_sl: float | None
) -> None:
    """
    Persist the stop that was in effect at exit and append using finalize_trade_row(...).
    """
    reason = str(trade_row.get("exit_reason", "")).lower()
    if "breakeven" in reason:
        sl_at_exit = trade_row.get("entry_price")
    elif "trailing_stop" in reason or "trailing" in reason:
        sl_at_exit = trade_row.get("ts_level", current_sl)
    elif "stop" in reason:
        sl_at_exit = current_sl
    else:
        # keep whatever may have been stamped (e.g., for non-stop exits or trailing path above)
        sl_at_exit = trade_row.get("sl_at_exit_price", None)

    finalized = finalize_trade_row(dict(trade_row), current_stop_price_at_exit=sl_at_exit)
    trades_list.append({col: finalized.get(col) for col in TRADES_COLS})


# =============================================
# Spread model (PnL-only)
# =============================================


def resolve_spread_pips(pair: str, row: pd.Series, cfg: Dict[str, Any]) -> float:
    sp = cfg.get("spreads") or {}
    if not sp.get("enabled", False):
        return 0.0
    # per-bar override
    if "spread_pips" in row and pd.notna(row["spread_pips"]):
        try:
            return float(row["spread_pips"])
        except Exception:
            pass
    per_pair = sp.get("per_pair") or {}
    if pair in per_pair:
        try:
            return float(per_pair[pair])
        except Exception:
            pass
    mode = str(sp.get("mode", "constant")).lower()
    if mode == "atr_mult":
        atr_val = float(row.get("atr", 0.0))
        ps = pip_size_for_pair(pair)
        atr_pips = (atr_val / ps) if ps else 0.0
        k = float(sp.get("atr_mult", 0.0))
        return float(atr_pips * k)
    try:
        return float(sp.get("default_pips", 0.0))
    except Exception:
        return 0.0


# =============================================
# PnL calculation (account currency) using pip valuation per lot
# =============================================


def compute_trade_pnl_money(tr: Dict[str, Any], pair: str, pip_value_1lot: float) -> float:
    dir_int = int(tr["direction_int"])
    entry_mid = float(tr["entry_price"])
    exit_mid = float(tr["exit_price"])
    lots_half = float(tr["lots_half"])
    lots_runner = float(tr["lots_runner"])
    tp1_hit = bool(tr.get("tp1_hit", False))
    tp1_mid = float(tr.get("tp1_price") or entry_mid)
    sp_pips = float(tr.get("spread_pips_used", 0.0))

    ps = pip_size_for_pair(pair)
    sp_price = sp_pips * ps

    if dir_int > 0:  # long
        entry_fill = entry_mid + sp_price / 2.0
        tp1_fill = tp1_mid - sp_price / 2.0
        exit_fill = exit_mid - sp_price / 2.0
    else:  # short
        entry_fill = entry_mid - sp_price / 2.0
        tp1_fill = tp1_mid + sp_price / 2.0
        exit_fill = exit_mid + sp_price / 2.0

    def pips_between(px2, px1):
        return (dir_int * (px2 - px1)) / ps if ps else 0.0

    if tp1_hit:
        pips_half = pips_between(tp1_fill, entry_fill)
        pips_runner = pips_between(exit_fill, entry_fill)
        pnl = (pips_half * pip_value_1lot * lots_half) + (
            pips_runner * pip_value_1lot * lots_runner
        )
    else:
        pips_full = pips_between(exit_fill, entry_fill)
        pnl = pips_full * pip_value_1lot * (lots_half + lots_runner)

    return float(pnl)


# =============================================
# Simulation core
# =============================================


def simulate_pair_trades(
    rows: pd.DataFrame,
    pair: str,
    cfg: Dict[str, Any],
    equity_state: Dict[str, float],
    return_equity: bool = False,
    **overrides,
) -> List[Dict[str, Any]] | tuple[List[Dict[str, Any]], pd.DataFrame]:
    # --- Risk filter wiring (DBCVIX) ---
    dbcvix_series = overrides.get("dbcvix_series")
    dbcvix_cfg = (cfg.get("filters") or {}).get("dbcvix") or {}

    # --- Config shorthands
    entry_cfg = cfg.get("entry") or {}
    risk_cfg = cfg.get("risk") or {}
    exec_cfg = cfg.get("execution") or {}
    exit_cfg = cfg.get("exit") or {}

    SL_ATR_MULT = float(overrides.get("sl_atr_mult", entry_cfg.get("sl_atr", 1.5)))
    TP1_ATR_MULT = float(overrides.get("tp1_atr_mult", entry_cfg.get("tp1_atr", 1.0)))
    TRAIL_AFTER_ATR = float(overrides.get("trail_after_atr", entry_cfg.get("trail_after_atr", 2.0)))
    TS_ATR_MULT = float(overrides.get("ts_atr_mult", entry_cfg.get("ts_atr", 1.5)))
    intrabar_priority = str(
        overrides.get("intrabar_priority", exec_cfg.get("intrabar_priority", "tp_first"))
    )

    account_ccy = (risk_cfg.get("account_ccy") or "AUD").upper()
    base_risk_pct = float(risk_cfg.get("risk_per_trade", 0.02))
    fx_quotes = risk_cfg.get("fx_quotes") or {}

    ps = pip_size_for_pair(pair)

    # ----- small helpers -----
    def to_pips(price_move: float) -> float:
        return price_move / ps if ps else 0.0

    def hit_level(
        direction_int: int, high_px: float, low_px: float, level: float, kind: str
    ) -> bool:
        if level is None or not math.isfinite(level):
            return False
        if direction_int > 0:  # long
            if kind == "tp":
                return high_px >= level
            if kind == "sl":
                return low_px <= level
        else:  # short
            if kind == "tp":
                return low_px <= level
            if kind == "sl":
                return high_px >= level
        return False

    def signed_move_from_entry(direction_int: int, px_now: float, px_entry: float) -> float:
        return direction_int * (px_now - px_entry)

    def trail_level_from_close(direction_int: int, close_px: float, atr_entry: float) -> float:
        return (
            close_px - TS_ATR_MULT * atr_entry
            if direction_int > 0
            else close_px + TS_ATR_MULT * atr_entry
        )

    def better_stop(direction_int: int, a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None:
            return b
        if b is None:
            return a
        return max(a, b) if direction_int > 0 else min(a, b)

    def _cfg_slippage_pips(cfg: Dict[str, Any]) -> float:
        fills = (cfg or {}).get("fills") or {}
        sl = fills.get("slippage") or {}
        if not sl or not sl.get("enabled", False):
            return 0.0
        try:
            return float(sl.get("pips", 0.0))
        except Exception:
            return 0.0

    def _int_signal(val) -> int:
        try:
            v = float(val)
            if math.isnan(v):
                return 0
            v = int(v)
            return 1 if v > 0 else (-1 if v < 0 else 0)
        except Exception:
            return 0

    # ----- state/outputs -----
    trades: List[Dict[str, Any]] = []
    open_tr: Optional[Dict[str, Any]] = None

    realized_pnl_cum_local: float = 0.0
    equity_history: List[Dict[str, Any]] = []
    order = intrabar_sequence(intrabar_priority)

    # Ensure time‑sorted
    rows = rows.copy()
    if "date" in rows.columns:
        rows["date"] = pd.to_datetime(rows["date"])
        rows = rows.sort_values("date").reset_index(drop=True)
    else:
        rows = rows.sort_index().reset_index(drop=True)

    # ======================
    # main bar loop
    # ======================
    for i in range(len(rows)):
        r = rows.iloc[i]
        date_i = pd.to_datetime(r["date"]) if "date" in r else pd.to_datetime(r.name)
        _o_i, h_i, l_i, c_i = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])

        atr_raw = pd.to_numeric(r.get("atr"), errors="coerce")
        atr_i = float(0.0 if pd.isna(atr_raw) else atr_raw)

        entry_sig = _int_signal(r.get("entry_signal", 0))
        exit_sig = _int_signal(r.get("exit_signal", 0))

        # --------------------------
        # 1) manage open position
        # --------------------------
        if open_tr is not None:
            d = int(open_tr["direction_int"])
            entry_px = float(open_tr["entry_price"])
            atr_entry = float(open_tr["atr_at_entry_price"])
            tp1_px = float(open_tr["tp1_price"])
            sl_px = float(open_tr["sl_price"])
            tp1_done = bool(open_tr.get("tp1_hit", False))
            ts_active = bool(open_tr.get("ts_active", False))
            ts_level = open_tr.get("ts_level", None)
            be_price = entry_px if tp1_done else None

            # (a) trailing activation
            if not ts_active and math.isfinite(atr_entry) and atr_entry > 0:
                move_cl = signed_move_from_entry(d, c_i, entry_px)
                if move_cl >= TRAIL_AFTER_ATR * atr_entry:
                    ts_active = True
                    ts_level = trail_level_from_close(d, c_i, atr_entry)

            # (b) ratchet trailing
            if ts_active and math.isfinite(atr_entry) and atr_entry > 0:
                cand = trail_level_from_close(d, c_i, atr_entry)
                ts_level = better_stop(d, ts_level, cand)

            # (c) effective stop (SL → BE → TS)
            effective_stop = sl_px
            if tp1_done and be_price is not None:
                effective_stop = better_stop(d, effective_stop, be_price)
            if ts_active and ts_level is not None:
                effective_stop = better_stop(d, effective_stop, ts_level)

            # (d) intrabar checks
            closed_this_bar = False
            reason = None
            exit_px = None

            # TP1 vs SL priority when TP1 not yet done
            if not tp1_done:
                for ev in order:
                    if ev == "tp" and hit_level(d, h_i, l_i, tp1_px, "tp"):
                        tp1_done = True
                        be_price = entry_px
                        sl_px = be_price
                        open_tr["tp1_hit"] = True
                        open_tr["breakeven_after_tp1"] = True
                    elif ev == "sl" and hit_level(d, h_i, l_i, effective_stop, "sl"):
                        reason = "stoploss"
                        exit_px = effective_stop
                        closed_this_bar = True
                        break
                # recompute effective stop if TP1 just happened
                if not closed_this_bar and tp1_done:
                    effective_stop = sl_px
                    if ts_active and ts_level is not None:
                        effective_stop = better_stop(d, effective_stop, ts_level)

            # Golden Standard: System exits (C1 reversal) take priority over BE/TS
            if (not closed_this_bar) and exit_sig != 0:
                if exit_cfg.get("exit_on_exit_signal", False):
                    reason = "exit_indicator"
                elif exit_cfg.get("exit_on_c1_reversal", True):
                    reason = "c1_reversal"
                elif exit_cfg.get("exit_on_baseline_cross", False):
                    reason = "baseline_cross"
                else:
                    reason = "exit_indicator"
                exit_px = c_i
                closed_this_bar = True

            # BE/TS after TP1 or when TS active (only if no system exit)
            if (not closed_this_bar) and (tp1_done or ts_active):
                if hit_level(d, h_i, l_i, effective_stop, "sl"):
                    if (
                        ts_active
                        and ts_level is not None
                        and (
                            (d > 0 and effective_stop >= max(sl_px, ts_level, be_price or -1e18))
                            or (d < 0 and effective_stop <= min(sl_px, ts_level, be_price or 1e18))
                        )
                    ):
                        reason = "trailing_stop"
                    else:
                        reason = (
                            "breakeven_after_tp1"
                            if tp1_done and abs(effective_stop - entry_px) < 1e-12
                            else "stoploss"
                        )
                    exit_px = effective_stop
                    closed_this_bar = True

            # ---- finalize exit ----
            if closed_this_bar:
                current_effective_stop = (
                    float(effective_stop) if effective_stop is not None else None
                )

                if reason == "trailing_stop":
                    # ✅ Canonical TS fill + audit stamps (exit_price, ts_level, sl_at_exit_price, ts_active, slippage_pips)
                    open_tr = _apply_trailing_stop_fill(
                        open_tr,
                        final_stop_price=current_effective_stop,
                        is_long=(d > 0),
                        pair=pair,
                        cfg=cfg,
                    )
                    exit_px = float(open_tr["exit_price"])  # for clarity
                else:
                    # Non‑TS exits → use computed price
                    open_tr["exit_price"] = float(exit_px)

                # Common stamps
                open_tr["exit_date"] = date_i
                open_tr["exit_reason"] = str(reason)

                # W/L/S classification
                if bool(open_tr.get("tp1_hit", False)):
                    open_tr["win"], open_tr["loss"], open_tr["scratch"] = True, False, False
                else:
                    if reason == "stoploss":
                        open_tr["win"], open_tr["loss"], open_tr["scratch"] = False, True, False
                    else:
                        open_tr["win"], open_tr["loss"], open_tr["scratch"] = False, False, True

                # Cash PnL using spread model
                pip_val_1lot = float(pip_value_per_lot(pair, account_ccy, fx_quotes))
                pnl_money = compute_trade_pnl_money(open_tr, pair, pip_val_1lot)
                open_tr["pnl"] = float(pnl_money)

                equity_state["balance"] += float(pnl_money)
                realized_pnl_cum_local += float(pnl_money)

                # ✅ Persist the stop that was actually in force at exit (lets the audit pass)
                _finalize_and_append_trade(
                    trades_list=trades,
                    trade_row=open_tr,
                    current_sl=current_effective_stop,
                )
                open_tr = None

            else:
                # still open → update dynamic state only
                open_tr["current_sl"] = float(sl_px)
                open_tr["ts_active"] = bool(ts_active)
                open_tr["ts_level"] = None if ts_level is None else float(ts_level)
                open_tr["tp1_hit"] = bool(tp1_done)
                open_tr["breakeven_after_tp1"] = bool(open_tr.get("breakeven_after_tp1", tp1_done))

        # --------------------------
        # 2) equity snapshot per bar
        # --------------------------
        if return_equity:
            equity_history.append(
                {
                    "date": pd.to_datetime(date_i),
                    "pair": pair,
                    "pnl_realized_cum": float(realized_pnl_cum_local),
                    "equity": float(equity_state["balance"]),
                }
            )

        # --------------------------
        # 3) new entry if flat
        # --------------------------
        if open_tr is None and entry_sig != 0:
            direction = "long" if entry_sig > 0 else "short"
            d_int = 1 if entry_sig > 0 else -1
            entry_px = c_i

            atr_entry = atr_i
            if (not math.isfinite(atr_entry)) or atr_entry <= 0.0:
                continue

            atr_pips = to_pips(atr_entry)
            sl_dist_pips = SL_ATR_MULT * atr_pips
            TP1_ATR_MULT * atr_pips

            # DBCVIX effective risk
            trade_date = (
                pd.to_datetime(r["date"]) if "date" in r else pd.to_datetime(rows.loc[i, "date"])
            )
            risk_pct_eff, db_flag, db_val = resolve_dbcvix_risk(
                dbcvix_series=dbcvix_series,
                trade_date=trade_date,
                base_risk=base_risk_pct,
                fcfg=dbcvix_cfg,
            )
            if risk_pct_eff <= 0.0:
                continue  # block mode

            # position sizing
            risk_money = float(equity_state["balance"]) * float(risk_pct_eff)
            pip_val_1lot = float(pip_value_per_lot(pair, account_ccy, fx_quotes))
            lots_total = 0.0
            if sl_dist_pips > 0 and pip_val_1lot > 0:
                lots_total = risk_money / (sl_dist_pips * pip_val_1lot)
            if not math.isfinite(lots_total):
                continue
            if lots_total <= 0.0:
                lots_total = 0.01  # min lot safeguard

            lots_half = lots_total / 2.0
            lots_runner = lots_total - lots_half

            # levels (price units) using ENTRY ATR
            tp1_px = entry_px + d_int * (TP1_ATR_MULT * atr_entry)
            sl_px = entry_px - d_int * (SL_ATR_MULT * atr_entry)
            if not (math.isfinite(tp1_px) and math.isfinite(sl_px)):
                continue

            spread_pips_used = resolve_spread_pips(pair, r, cfg)

            open_tr = {
                "pair": pair,
                "entry_date": date_i,
                "entry_price": float(entry_px),
                "direction": direction,
                "direction_int": int(d_int),
                "atr_at_entry_price": float(atr_entry),
                "atr_at_entry_pips": float(atr_pips),
                "lots_total": float(lots_total),
                "lots_half": float(lots_half),
                "lots_runner": float(lots_runner),
                # DBCVIX audit fields
                "risk_pct_used": float(risk_pct_eff),
                "dbcvix_val": (float(db_val) if db_val is not None else None),
                "dbcvix_flag": bool(db_flag),
                # immutable entry levels
                "tp1_price": float(tp1_px),
                "sl_price": float(sl_px),
                "tp1_at_entry_price": float(tp1_px),
                "sl_at_entry_price": float(sl_px),
                # dynamic state
                "current_sl": float(sl_px),
                "ts_active": False,
                "ts_level": None,
                "tp1_hit": False,
                "breakeven_after_tp1": False,
                "entry_idx": int(i),
                "exit_date": None,
                "exit_price": None,
                "exit_reason": None,
                "pnl": 0.0,
                "win": False,
                "loss": False,
                "scratch": False,
                "spread_pips_used": float(spread_pips_used),
            }

    # --------------------------
    # return (optionally equity)
    # --------------------------
    if return_equity:
        eq_df = pd.DataFrame(equity_history, columns=["date", "pair", "pnl_realized_cum", "equity"])
        try:
            eq_df["date"] = pd.to_datetime(eq_df["date"])
        except Exception:
            pass
        return trades, eq_df

    return trades


# =============================================
# Runner
# =============================================


def run_backtest(
    config_path: PathLikeT | dict = "configs/config.yaml",
    results_dir: Optional[PathLike] = None,
):
    # load config
    cfg = config_path if isinstance(config_path, dict) else load_config(config_path)

    # results dir - check both output.results_dir and outputs.dir for compatibility
    configured_dir = (
        results_dir
        or (cfg.get("outputs") or {}).get("dir")  # New format (MT5 parity)
        or (cfg.get("output") or {}).get("results_dir")  # Legacy format
        or "results"
    )
    out_dir = ensure_results_dir(configured_dir)

    # Log results directory for diagnostics
    run_slug = Path(out_dir).name if Path(out_dir).name != "results" else "default"
    print(f"[RESULTS DIR] slug={run_slug} path={Path(out_dir).resolve()} created_by=engine")

    # DBCVIX (load once)
    dbcvix_series = load_dbcvix_series(cfg)
    if dbcvix_series is None:
        print("ℹ️  DBCVIX disabled or not loaded (series=None). Risk filter will not trigger.")
    else:
        print(
            f"ℹ️  DBCVIX loaded: {dbcvix_series.index.min().date()} → {dbcvix_series.index.max().date()} (n={len(dbcvix_series)})"
        )
    fcfg = resolve_dbcvix_config(cfg)
    print(
        "ℹ️  DBCVIX config:",
        {
            "enabled": fcfg.get("enabled"),
            "mode": fcfg.get("mode"),
            "threshold": fcfg.get("threshold"),
            "reduce_risk_to": fcfg.get("reduce_risk_to"),
            "source": fcfg.get("source"),
        },
    )

    summary_path = Path(out_dir) / "summary.txt"
    equity_path = Path(out_dir) / "equity_curve.csv"

    pairs = cfg.get("pairs") or (cfg.get("data") or {}).get("pairs") or []
    data_dir = cfg.get("data_dir") or (cfg.get("data") or {}).get("dir") or "data/daily"
    if not pairs:
        print("⚠️  No pairs configured.")
        return

    starting_balance = float(
        (cfg.get("risk") or {}).get("starting_balance", cfg.get("starting_balance", 10_000.0))
    )
    equity_state = {"balance": starting_balance}
    track_equity = bool((cfg.get("tracking") or {}).get("in_sim_equity", True))

    all_trades: List[Dict[str, Any]] = []
    equity_frames: List[pd.DataFrame] = []

    # load + process each pair
    for pair in pairs:
        try:
            # robust loader: your project has one; here we assume CSV <pair>.csv in data_dir
            # minimal version:
            path = None
            # try variants
            cand = [
                f"{pair}.csv",
                f"{pair.upper()}.csv",
                f"{pair.replace('/', '_')}.csv",
                f"{pair.replace('/', '_').upper()}.csv",
                f"{pair}_daily.csv",
                f"{pair.replace('/', '_')}_daily.csv",
            ]
            for pat in cand:
                hits = list(Path(data_dir).rglob(pat))
                if hits:
                    path = hits[0]
                    break
            if path is None:
                print(f"⚠️  Skipping {pair}: no CSV in {data_dir}")
                continue
            df = pd.read_csv(path)
            # normalize basic schema
            lc = {c.lower(): c for c in df.columns}
            for want in ["date", "open", "high", "low", "close"]:
                if want not in df.columns:
                    # try case-insensitive remap
                    if want in lc:
                        df = df.rename(columns={lc[want]: want})
            if "date" not in df.columns:
                df = df.rename(columns={df.columns[0]: "date"})
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = (
                df.dropna(subset=["date", "open", "high", "low", "close"])
                .sort_values("date")
                .reset_index(drop=True)
            )
            df["pair"] = pair

            # Apply date filtering if specified in config
            date_start = cfg.get("date_from") or (cfg.get("date_range") or {}).get("start")
            date_end = cfg.get("date_to") or (cfg.get("date_range") or {}).get("end")

            if date_start and date_end:
                from core.utils import slice_df_by_dates

                df, (first_ts, last_ts, rows_before, rows_after) = slice_df_by_dates(
                    df, date_start, date_end
                )

                if rows_after == 0:
                    raise ValueError(
                        f"Date slice produced empty dataset for {date_start}..{date_end}"
                    )

                print(
                    f"ℹ️  {pair}: Date filtered {rows_before} → {rows_after} rows ({first_ts.date()} to {last_ts.date()})"
                )

            base = calculate_atr(df.copy())
            base = apply_indicators_with_cache(base, pair, cfg)

            if (cfg.get("validation", {}) or {}).get("enabled", True):
                try:
                    validate_contract(
                        base,
                        config=cfg,
                        strict=(cfg.get("validation", {}) or {}).get("strict_contract", False),
                    )
                except Exception as _ve:
                    print(f"ℹ️  {pair}: validation skipped ({_ve})")

            # Enforce final slice immediately before trading (safety gate)
            date_start = cfg.get("date_from") or (cfg.get("date_range") or {}).get("start")
            date_end = cfg.get("date_to") or (cfg.get("date_range") or {}).get("end")

            if date_start and date_end:
                from core.utils import slice_df_by_dates

                base, (first_ts, last_ts, rows_before, rows_after) = slice_df_by_dates(
                    base, date_start, date_end
                )

                if rows_after == 0:
                    raise ValueError(
                        f"Final slice produced empty dataset for {date_start}..{date_end}"
                    )

                print(
                    f"[SLICE ENFORCE] rows_before={rows_before} rows_after={rows_after} first={first_ts.date()} last={last_ts.date()}"
                )

            # ENGINE INPUT diagnostic
            if len(base) > 0:
                first_date = base["date"].iloc[0]
                last_date = base["date"].iloc[-1]
                window_start = cfg.get("date_from") or (cfg.get("date_range") or {}).get(
                    "start", "N/A"
                )
                window_end = cfg.get("date_to") or (cfg.get("date_range") or {}).get("end", "N/A")
                print(
                    f"[ENGINE INPUT] first={pd.to_datetime(first_date).date()} last={pd.to_datetime(last_date).date()} rows={len(base)} window={window_start}..{window_end}"
                )

            signals_df = apply_signal_logic(base, cfg)

            # normalize entry/exit to {-1,0,1}
            for col in ["entry_signal", "exit_signal"]:
                if col in signals_df.columns:
                    signals_df[col] = (
                        pd.to_numeric(signals_df[col], errors="coerce")
                        .fillna(0)
                        .clip(-1, 1)
                        .astype(int)
                    )
                else:
                    signals_df[col] = 0

            # Add date range guard before trading simulation
            date_start = cfg.get("date_from") or (cfg.get("date_range") or {}).get("start")
            date_end = cfg.get("date_to") or (cfg.get("date_range") or {}).get("end")

            if date_start and date_end and len(signals_df) > 0:
                # Enforce date range on signals_df before trading simulation
                signal_dates = pd.to_datetime(signals_df["date"])
                start_ts = pd.to_datetime(date_start)
                end_ts = pd.to_datetime(date_end)
                mask = (signal_dates >= start_ts) & (signal_dates <= end_ts)

                if not mask.all():
                    print(
                        f"⚠️  WARNING: signals_df contains {(~mask).sum()} rows outside {date_start}..{date_end}, filtering before trading"
                    )
                    signals_df = signals_df[mask].copy().reset_index(drop=True)

            if track_equity:
                pair_trades, pair_eq = simulate_pair_trades(
                    rows=signals_df,
                    pair=pair,
                    cfg=cfg,
                    equity_state=equity_state,
                    return_equity=True,
                )
                equity_frames.append(pair_eq)
            else:
                pair_trades = simulate_pair_trades(
                    rows=signals_df,
                    pair=pair,
                    cfg=cfg,
                    equity_state=equity_state,
                    return_equity=False,
                )
            all_trades.extend(pair_trades)

        except Exception as e:
            print(f"❌ Error processing {pair}: {e}")

    trades_df = (
        pd.DataFrame(all_trades, columns=TRADES_COLS).copy()
        if all_trades
        else pd.DataFrame(columns=TRADES_COLS)
    )

    # equity curve (realized PnL only)
    if track_equity and equity_frames:
        eq = pd.concat(equity_frames, ignore_index=True)
        eq_wide = (
            eq.pivot_table(index="date", columns="pair", values="pnl_realized_cum", aggfunc="last")
            .sort_index()
            .ffill()
            .fillna(0.0)
        )
        eq_wide["pnl_realized_cum_total"] = eq_wide.sum(axis=1, numeric_only=True)
        equity_curve = eq_wide[["pnl_realized_cum_total"]].rename(
            columns={"pnl_realized_cum_total": "equity"}
        )
        equity_curve["equity"] = starting_balance + equity_curve["equity"]
        equity_curve["peak"] = equity_curve["equity"].cummax()
        equity_curve["drawdown"] = equity_curve["equity"] - equity_curve["peak"]
        equity_curve = equity_curve.reset_index()
    else:
        equity_curve = pd.DataFrame(columns=["date", "equity", "peak", "drawdown"])

    # enforce schema
    for c in TRADES_COLS:
        if c not in trades_df.columns:
            trades_df[c] = pd.NA
    trades_df = trades_df.reindex(columns=TRADES_COLS)

    # Filter output trades to date range (safety gate)
    date_start = cfg.get("date_from") or (cfg.get("date_range") or {}).get("start")
    date_end = cfg.get("date_to") or (cfg.get("date_range") or {}).get("end")

    if date_start and date_end and len(trades_df) > 0:
        start_ts = pd.to_datetime(date_start)
        end_ts = pd.to_datetime(date_end)

        # Filter trades by entry_date
        if "entry_date" in trades_df.columns:
            entry_dates = pd.to_datetime(trades_df["entry_date"])
            mask = (entry_dates >= start_ts) & (entry_dates <= end_ts)
            trades_df = trades_df[mask].copy()

        print(f"[TRADES FILTER] kept={len(trades_df)} start={date_start} end={date_end}")

    # OUT CHECK diagnostic
    if len(trades_df) > 0 and "entry_date" in trades_df.columns:
        entry_dates = pd.to_datetime(trades_df["entry_date"])
        min_date = entry_dates.min()
        max_date = entry_dates.max()
        print(f"[OUT CHECK] first={min_date.date()} last={max_date.date()} rows={len(trades_df)}")
    elif len(trades_df) == 0:
        print("[OUT CHECK] no trades generated")

    # Write trades CSV with comprehensive diagnostics
    write_trades_csv_with_diagnostics(trades_df, out_dir, cfg, run_slug)

    # Summary (best effort; use utils.summarize_results if available)
    try:
        from utils import summarize_results

        txt = summarize_results(
            trades_df.to_dict("records"),
            starting_balance=starting_balance,
            run_name=str(cfg.get("strategy_version", "default")),
            results_dir=str(out_dir),
            cfg=cfg,
        )
        if equity_curve is not None:
            txt = (txt or "") + f"\nequity_curve_rows: {int(len(equity_curve))}"
    except Exception:
        total = int(len(trades_df))
        wins = (
            int(
                pd.to_numeric(trades_df.get("win", 0), errors="coerce").fillna(0).astype(bool).sum()
            )
            if total
            else 0
        )
        losses = (
            int(
                pd.to_numeric(trades_df.get("loss", 0), errors="coerce")
                .fillna(0)
                .astype(bool)
                .sum()
            )
            if total
            else 0
        )
        scratches = (
            int(
                pd.to_numeric(trades_df.get("scratch", 0), errors="coerce")
                .fillna(0)
                .astype(bool)
                .sum()
            )
            if total
            else 0
        )
        ns = max(wins + losses, 0)
        roi_dollars = float(
            pd.to_numeric(trades_df.get("pnl", 0.0), errors="coerce").fillna(0.0).sum()
        )
        roi_pct = (roi_dollars / starting_balance * 100.0) if starting_balance else 0.0
        txt = (
            "📊 Backtest Summary\n"
            "-------------------\n"
            f"Total Trades : {total}\n"
            f"Wins         : {wins}\n"
            f"Losses       : {losses}\n"
            f"Scratches    : {scratches}\n"
            f"Win% (NS)    : {((wins / ns) * 100.0) if ns else 0.0:.2f}\n"
            f"ROI ($)      : {roi_dollars:.2f}\n"
            f"ROI (%)      : {roi_pct:.2f}\n"
            f"equity_curve_rows: {int(len(equity_curve))}\n"
        )
    Path(summary_path).write_text(txt or "", encoding="utf-8")

    if (cfg.get("tracking") or {}).get("in_sim_equity", True) and not equity_curve.empty:
        equity_curve.to_csv(equity_path, index=False)

    print(f"✅ Backtest complete. Results saved to '{out_dir}'")


# =============================================
# Walk-Forward (dict-friendly)
# =============================================


def _date_range_folds(start, end, train_years: int, test_years: int, step_years: int):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    cur = start
    one_day = pd.Timedelta(days=1)
    while True:
        is_start = cur
        is_end = is_start + pd.DateOffset(years=int(train_years)) - one_day
        oos_start = is_end + one_day
        oos_end = oos_start + pd.DateOffset(years=int(test_years)) - one_day
        if oos_end > end:
            break
        if not (is_end < oos_start):
            raise AssertionError(
                f"No-lookahead violation: train_end {is_end} !< oos_start {oos_start}"
            )
        yield (is_start, is_end, oos_start, oos_end)
        cur = is_start + pd.DateOffset(years=int(step_years))
        if cur >= end:
            break


def _slice_df_by_dates(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    m = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[m].copy().reset_index(drop=True)


def run_backtest_walk_forward(
    config_path: PathLikeT = "configs/config.yaml", results_dir: Optional[PathLike] = None
) -> None:
    cfg = load_config(config_path)

    wf_cfg = cfg.get("walk_forward") or {}
    wf_run_name = wf_cfg.get("run_name") or "wfo_default"
    out_dir = ensure_results_dir(
        Path("results") / wf_run_name if results_dir is None else results_dir
    )
    # Also ensure canonical default dir (used by analytics.monte_carlo)
    default_dir = ensure_results_dir(Path("results") / wf_run_name)
    # Write a pointer so analytics.monte_carlo can locate the actual folder
    try:
        ptr = Path(default_dir) / ".source_dir"
        ptr.write_text(str(Path(out_dir).resolve()), encoding="utf-8")
    except Exception:
        pass

    trades_path = Path(out_dir) / "trades.csv"
    summary_path = Path(out_dir) / "oos_summary.txt"
    equity_path = Path(out_dir) / "equity_curve.csv"
    folds_csv = Path(out_dir) / "wfo_folds.csv"

    # Tolerate missing walk_forward window by falling back to top-level date_range
    start_raw = wf_cfg.get("start") or ((cfg.get("date_range") or {}).get("start"))
    end_raw = wf_cfg.get("end") or ((cfg.get("date_range") or {}).get("end"))
    start = pd.to_datetime(start_raw) if start_raw else pd.NaT
    end = pd.to_datetime(end_raw) if end_raw else pd.NaT
    train_years = int(wf_cfg.get("train_years", 3))
    test_years = int(wf_cfg.get("test_years", 1))
    step_years = int(wf_cfg.get("step_years", 1))
    if pd.isna(start) or pd.isna(end):
        raise ValueError("walk_forward.start and walk_forward.end must be set (YYYY-MM-DD).")

    pairs = cfg.get("pairs") or (cfg.get("data") or {}).get("pairs") or []
    data_dir = cfg.get("data_dir") or (cfg.get("data") or {}).get("dir") or "data/daily"
    if not pairs:
        raise ValueError("No pairs configured for walk-forward.")

    starting_balance = float(
        (cfg.get("risk") or {}).get("starting_balance", cfg.get("starting_balance", 10_000.0))
    )
    track_equity = bool((cfg.get("tracking") or {}).get("in_sim_equity", True))

    # preload + basic normalize
    pair_data: Dict[str, pd.DataFrame] = {}
    for p in pairs:
        path = None
        pats = [
            f"{p}.csv",
            f"{p.upper()}.csv",
            f"{p.replace('/', '_')}.csv",
            f"{p.replace('/', '_').upper()}.csv",
            f"{p}_daily.csv",
            f"{p.replace('/', '_')}_daily.csv",
        ]
        for pat in pats:
            hits = list(Path(data_dir).rglob(pat))
            if hits:
                path = hits[0]
                break
        if path is None:
            print(f"⚠️  Skipping {p}: no CSV")
            continue
        df = pd.read_csv(path)
        lc = {c.lower(): c for c in df.columns}
        for want in ["date", "open", "high", "low", "close"]:
            if want not in df.columns and want in lc:
                df = df.rename(columns={lc[want]: want})
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = (
            df.dropna(subset=["date", "open", "high", "low", "close"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        df["pair"] = p
        pair_data[p] = df

    if not pair_data:
        raise RuntimeError("No data available for any pairs; cannot run WFO.")

    # DBCVIX once
    load_dbcvix_series(cfg)

    all_oos_trades: List[Dict[str, Any]] = []
    per_fold_rows: List[Dict[str, Any]] = []
    equity_state: Dict[str, Any] = {"balance": starting_balance}
    fold_equity_frames: List[pd.DataFrame] = []
    realized_so_far = 0.0

    fold_idx = 0
    for is_start, is_end, oos_start, oos_end in _date_range_folds(
        start, end, train_years, test_years, step_years
    ):
        fold_idx += 1
        print(f"—— Fold {fold_idx} ———————————————————————————————")
        print(
            f"Train: {is_start.date()} → {is_end.date()}  |  Test (OOS): {oos_start.date()} → {oos_end.date()}"
        )

        fold_trades: List[Dict[str, Any]] = []
        pair_eq_frames: List[pd.DataFrame] = []

        for pair, df in pair_data.items():
            try:
                base = calculate_atr(df.copy())
                base = apply_indicators_with_cache(base, pair, cfg)

                try:
                    if (cfg.get("validation", {}) or {}).get("enabled", True):
                        validate_contract(
                            base,
                            config=cfg,
                            strict=(cfg.get("validation", {}) or {}).get("strict_contract", False),
                        )
                except Exception as _ve:
                    print(f"ℹ️  {pair}: validation skipped/failed ({_ve})")

                base = apply_signal_logic(base, cfg)
                for col in ["entry_signal", "exit_signal"]:
                    if col in base.columns:
                        base[col] = (
                            pd.to_numeric(base[col], errors="coerce")
                            .fillna(0)
                            .clip(-1, 1)
                            .astype(int)
                        )
                    else:
                        base[col] = 0

                oos_rows = _slice_df_by_dates(base, oos_start, oos_end)
                if oos_rows.empty:
                    continue

                if track_equity:
                    trades, pair_eq = simulate_pair_trades(
                        oos_rows,
                        pair,
                        cfg,
                        equity_state,
                        return_equity=True,
                    )
                    if trades:
                        fold_trades.extend(trades)
                    if pair_eq is not None and not pair_eq.empty:
                        pair_eq = pair_eq[["date", "pair", "pnl_realized_cum"]].copy()
                        pair_eq["fold"] = fold_idx
                        pair_eq_frames.append(pair_eq)
                else:
                    trades = simulate_pair_trades(
                        oos_rows,
                        pair,
                        cfg,
                        equity_state,
                        return_equity=False,
                    )
                    if trades:
                        fold_trades.extend(trades)
            except Exception as e:
                print(f"❌ WFO fold {fold_idx} {pair}: {e}")

        # simple per-fold metrics (best effort)
        fold_df = pd.DataFrame(fold_trades)

        def _num_series(df, col, default=0.0):
            if df is None or df.empty:
                return pd.Series([], dtype=float)
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            return pd.Series([default] * len(df), index=df.index, dtype=float)

        def _bool_series(df, col):
            if df is None or df.empty:
                return pd.Series([], dtype=bool)
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).astype(bool)
            return pd.Series([False] * len(df), index=df.index, dtype=bool)

        pnl_sum = float(_num_series(fold_df, "pnl", 0.0).sum())
        total = int(len(fold_df))
        wins = int(_bool_series(fold_df, "win").sum()) if total else 0
        losses = int(_bool_series(fold_df, "loss").sum()) if total else 0
        scratches = int(_bool_series(fold_df, "scratch").sum()) if total else 0
        ns = max(wins + losses, 0)
        fold_roi_pct = float((pnl_sum / starting_balance) * 100.0) if starting_balance else 0.0

        per_fold_rows.append(
            {
                "fold": fold_idx,
                "is_start": is_start.date(),
                "is_end": is_end.date(),
                "oos_start": oos_start.date(),
                "oos_end": oos_end.date(),
                "oos_trades": total,
                "win_pct_ns": (wins / ns * 100.0) if ns else 0.0,
                "oos_roi_pct": fold_roi_pct,
            }
        )

        if total:
            all_oos_trades.extend(fold_trades)

        if track_equity and pair_eq_frames:
            eq = pd.concat(pair_eq_frames, ignore_index=True)
            eq_wide = (
                eq.pivot_table(
                    index="date", columns="pair", values="pnl_realized_cum", aggfunc="last"
                )
                .sort_index()
                .ffill()
                .fillna(0.0)
            )
            eq_wide["pnl_realized_cum_total"] = eq_wide.sum(axis=1, numeric_only=True)
            fold_equity = eq_wide[["pnl_realized_cum_total"]].rename(
                columns={"pnl_realized_cum_total": "equity"}
            )
            fold_equity["equity"] = (starting_balance + realized_so_far) + fold_equity["equity"]
            fold_equity = fold_equity.reset_index()
            fold_equity["fold"] = fold_idx
            fold_equity_frames.append(fold_equity)

            realized_so_far += pnl_sum

    # write OOS trades
    oos_df = pd.DataFrame(all_oos_trades)
    if oos_df.empty:
        for c in TRADES_COLS:
            if c not in oos_df.columns:
                oos_df[c] = pd.Series(dtype="object")
        if "exit_reason" not in oos_df.columns:
            oos_df["exit_reason"] = pd.Series(dtype="object")
        oos_df = oos_df[TRADES_COLS]
    else:
        for c in TRADES_COLS:
            if c not in oos_df.columns:
                oos_df[c] = pd.NA
        oos_df = oos_df[TRADES_COLS]
    oos_df.to_csv(trades_path, index=False)
    print(f"✅ Wrote OOS trades: {trades_path}")

    # folds CSV
    if per_fold_rows:
        pd.DataFrame(per_fold_rows).to_csv(folds_csv, index=False)
        print(f"✅ Wrote folds CSV: {folds_csv}")

    # OOS equity
    if track_equity and fold_equity_frames:
        equity_df = pd.concat(fold_equity_frames, ignore_index=True).sort_values("date")
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = equity_df["equity"] - equity_df["peak"]
        equity_df = equity_df[["date", "equity", "peak", "drawdown"]]
        equity_df.to_csv(equity_path, index=False)
        print(f"✅ Wrote OOS equity: {equity_path}")
    else:
        # fallback: step equity by exit_date
        try:
            dates = (
                pd.to_datetime(oos_df["exit_date"], errors="coerce")
                if "exit_date" in oos_df.columns
                else pd.Series(pd.NaT, index=oos_df.index)
            )
            equity_vals = (
                pd.to_numeric(oos_df.get("pnl", 0.0), errors="coerce").fillna(0.0).cumsum()
                + starting_balance
            )
            equity_df = pd.DataFrame({"date": dates, "equity": equity_vals}).dropna(subset=["date"])
            equity_df = equity_df.sort_values("date").reset_index(drop=True)
            equity_df["peak"] = equity_df["equity"].cummax()
            equity_df["drawdown"] = equity_df["equity"] - equity_df["peak"]
            if not equity_df.empty:
                equity_df.to_csv(equity_path, index=False)
                print(f"✅ Wrote OOS equity (fallback): {equity_path}")
        except Exception as e:
            print(f"ℹ️  Equity fallback failed: {e}")

    # OOS summary (best effort)
    try:
        from core.utils import summarize_results

        txt = summarize_results(results_dir=str(out_dir), config_path=str(config_path))
    except Exception:
        total = len(oos_df)
        wins = (
            int(pd.to_numeric(oos_df.get("win", 0), errors="coerce").fillna(0).astype(bool).sum())
            if total
            else 0
        )
        losses = (
            int(pd.to_numeric(oos_df.get("loss", 0), errors="coerce").fillna(0).astype(bool).sum())
            if total
            else 0
        )
        scratches = (
            int(
                pd.to_numeric(oos_df.get("scratch", 0), errors="coerce")
                .fillna(0)
                .astype(bool)
                .sum()
            )
            if total
            else 0
        )
        ns = max(wins + losses, 0)
        roi_dollars = float(
            pd.to_numeric(oos_df.get("pnl", 0.0), errors="coerce").fillna(0.0).sum()
        )
        roi_pct = (roi_dollars / starting_balance * 100.0) if starting_balance else 0.0
        txt = (
            f"📊 WFO OOS Summary\n"
            f"-------------------\n"
            f"Total Trades : {total}\n"
            f"Wins         : {wins}\n"
            f"Losses       : {losses}\n"
            f"Scratches    : {scratches}\n"
            f"Win% (non-scratch) : {(wins / ns * 100.0) if ns else 0.0:.2f}\n"
            f"Loss% (non-scratch): {(losses / ns * 100.0) if ns else 0.0:.2f}\n"
            f"ROI ($)      : {roi_dollars:.2f}\n"
            f"ROI (%)      : {roi_pct:.2f}\n"
        )
    Path(summary_path).write_text(txt or "", encoding="utf-8")
    print(f"✅ Walk-forward complete. OOS results saved to '{out_dir}'")

    # Mirror key artifacts into canonical results/<run_name> so analytics.monte_carlo can find them
    try:
        if str(out_dir) != str(default_dir):
            src_dir = Path(out_dir)
            dst_dir = Path(default_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)
            for fname in [
                "trades.csv",
                "equity_curve.csv",
                "oos_summary.txt",
                "wfo_folds.csv",
            ]:
                s = src_dir / fname
                if s.exists():
                    (dst_dir / fname).write_bytes(s.read_bytes())
    except Exception:
        pass


def load_pair_csv(pair: str, data_dir: str | None = None) -> pd.DataFrame:
    base = Path(data_dir) if data_dir else Path("data")
    name = pair.replace("/", "_")
    path = base / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_trades_csv_with_diagnostics(
    trades_df: pd.DataFrame, out_dir: str | Path, cfg: dict, run_slug: str = "default"
) -> bool:
    """
    Centralized trades CSV writer with comprehensive diagnostics.

    Decision tree:
    1. Schema validation → 2. Empty check → 3. Flag check → 4. Dir ensure → 5. Atomic write

    Returns:
        bool: True if written successfully, False if skipped or failed
    """
    import os
    import tempfile
    from pathlib import Path

    out_path = Path(out_dir).resolve()
    trades_path = out_path / "trades.csv"

    # Determine write flag from config hierarchy
    outputs_cfg = cfg.get("outputs", {})
    write_flag = outputs_cfg.get("write_trades_csv", True)  # Default True

    # Check for debug logging
    log_debug = os.environ.get("LOG_LEVEL") == "DEBUG"

    # Log decision inputs
    print(
        f"[WRITE TRADES] rows={len(trades_df)} path={trades_path} write_trades_csv={write_flag} dir_exists={out_path.exists()} slug={run_slug}"
    )

    # Empty check FIRST - but create empty CSV with headers for compatibility
    if len(trades_df) == 0:
        print("[WRITE TRADES SKIP] reason=empty")

        # Create empty CSV with standard headers for compatibility
        try:
            out_path.mkdir(parents=True, exist_ok=True)
            empty_df = pd.DataFrame(columns=TRADES_COLS)
            empty_df.to_csv(trades_path, index=False)
            print(f"[WRITE TRADES OK] wrote=0 path={trades_path} (empty file with headers)")
            return True
        except Exception as e:
            print(
                f"[WRITE TRADES ERROR] reason=empty_file_creation_failed exception={type(e).__name__}:{str(e)}"
            )
            return False

    # Schema validation (only for non-empty DataFrames)
    required_cols = ["pair", "entry_date", "direction_int"]  # Minimal required
    missing_cols = [col for col in required_cols if col not in trades_df.columns]
    if missing_cols:
        print(f"[WRITE TRADES SKIP] reason=schema_invalid missing_fields={missing_cols}")
        return False

    # Flag check
    if not write_flag:
        print("[WRITE TRADES SKIP] reason=flag_off")
        return False

    try:
        # Ensure directory exists
        out_path.mkdir(parents=True, exist_ok=True)

        # Prepare DataFrame for serialization
        df_clean = trades_df.copy()

        # Normalize datetime columns to ISO strings
        for col in df_clean.columns:
            if col.endswith("_date") or col.endswith("_time"):
                if col in df_clean.columns:
                    df_clean[col] = pd.to_datetime(df_clean[col]).dt.strftime("%Y-%m-%d %H:%M:%S")

        # Reset index to ensure clean CSV
        df_clean = df_clean.reset_index(drop=True)

        # Atomic write: temp file -> rename
        with tempfile.NamedTemporaryFile(
            mode="w", dir=out_path, prefix=".trades_tmp_", suffix=".csv", delete=False
        ) as tmp_file:
            df_clean.to_csv(tmp_file.name, index=False)
            temp_path = Path(tmp_file.name)

        # Atomic rename
        temp_path.rename(trades_path)

        print(f"[WRITE TRADES OK] wrote={len(trades_df)} path={trades_path}")

        # Debug: list directory contents
        if log_debug:
            files = list(out_path.glob("*"))
            print(f"[DEBUG] Results dir contents: {[f.name for f in files]}")

        return True

    except Exception as e:
        print(
            f"[WRITE TRADES ERROR] reason=serialization_failed exception={type(e).__name__}:{str(e)}"
        )
        # Clean up temp file if it exists
        try:
            if "temp_path" in locals() and temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        return False


def write_results(trades: list[dict], out_dir: str | Path) -> None:
    """Legacy writer - converts list to DataFrame and uses new writer"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if trades:
        trades_df = pd.DataFrame(trades)
        # Use new writer with minimal config
        cfg = {"outputs": {"write_trades_csv": True}}
        write_trades_csv_with_diagnostics(trades_df, out_dir, cfg, "legacy")
    else:
        print("[WRITE TRADES SKIP] reason=empty")

    # Still write summary for compatibility
    (out / "summary.txt").write_text(f"total_trades: {len(trades)}\n")
