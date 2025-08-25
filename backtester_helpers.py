# backtester_helpers.py
# ---------------------------------------------------------
# FX Backtester v1.9.8+
# Helpers for indicator wiring, signal hygiene, and
# audit-correct trade row finalization at exit.
# ---------------------------------------------------------
from __future__ import annotations

from typing import Callable, Dict, Tuple, List, Any, Optional, Iterable, Union
from functools import lru_cache
import importlib
from inspect import getmembers, isfunction

__all__ = [
    "apply_indicators",
    "coerce_entry_exit_signals",
    "finalize_trade_row",
    "build_and_finalize_trade_row",
]

# =========================================================
# Dynamic indicator loading (roles → modules)
# =========================================================

_ROLE_TO_MODULE = {
    "c1": "indicators.confirmation_funcs",
    "c2": "indicators.confirmation_funcs",
    "baseline": "indicators.baseline_funcs",
    "volume": "indicators.volume_funcs",
    "exit": "indicators.exit_funcs",
}

_SIGNAL_COL_BY_ROLE = {
    "c1": "c1_signal",
    "c2": "c2_signal",
    "baseline": "baseline_signal",
    "volume": "volume_signal",
    "exit": "exit_signal",
}

_ROLES = ("baseline", "c1", "c2", "volume", "exit")
_CONTAINER_KEYS = ("indicators", "pipeline", "steps", "specs", "list")

def _normalize_short_name(role: str, name: str) -> str:
    """Allow users to pass either 'ema' or 'baseline_ema' etc."""
    if not name:
        raise ImportError(f"No indicator short name provided for role '{role}'.")
    pref = f"{role}_"
    return name[len(pref):] if name.startswith(pref) else name

@lru_cache(maxsize=None)
def _resolve_module(role: str):
    mod_name = _ROLE_TO_MODULE.get(role)
    if not mod_name:
        raise ImportError(f"Unknown indicator role '{role}'. Known: {sorted(_ROLE_TO_MODULE)}")
    return importlib.import_module(mod_name)

@lru_cache(maxsize=None)
def _resolve_func(role: str, short_name: str) -> Tuple[Callable, str]:
    """Return (callable, fully_qualified_name) for role + short name."""
    short = _normalize_short_name(role, short_name)
    mod = _resolve_module(role)
    funcs = {name: fn for name, fn in getmembers(mod, isfunction)}
    target = f"{role}_{short}"
    if target not in funcs:
        available = [n for n in funcs if n.startswith(f"{role}_")]
        raise ImportError(
            f"Function '{target}' not found in module '{mod.__name__}'. "
            f"Available for role '{role}': {sorted(available)}"
        )
    return funcs[target], f"{mod.__name__}.{target}"

# ========================== Compact config detection =========================
def _is_compact_indicator_dict(d: Dict[str, Any]) -> bool:
    """True if dict looks like {'c1': '...', 'baseline': ..., 'use_exit': ...} (no 'role')."""
    if not isinstance(d, dict) or "role" in d:
        return False
    return any(k in d for k in _ROLES)

def _expand_compact_indicator_config(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Accept a compact dict like:
      {
        "c1": "schaff_trend_cycle",
        "use_baseline": True, "baseline": "ema",
        "use_c2": False,
        "use_volume": False,
        "use_exit": True, "exit": "twiggs_money_flow",
        # optional per-role params:
        "c1_params": {...}, "baseline_params": {...}, "exit_params": {...},
        # or nested dicts: "baseline": {"name":"ema","params":{...}}
        # optional order: ["baseline","c1","c2","volume","exit"]
      }
    Returns list of {"role":..., "name":..., "params":{...}} in the requested order.
    """
    roles = list(_ROLES)

    def _extract(role: str, val: Any) -> Optional[Dict[str, Any]]:
        if val is None or val is False:
            return None
        name = None
        params: Dict[str, Any] = {}
        if isinstance(val, dict):
            name = val.get("name") or val.get("short_name") or val.get("indicator") or val.get("func")
            if isinstance(val.get("params"), dict):
                params = dict(val["params"])
            elif isinstance(val.get("config"), dict):
                params = dict(val["config"])
        elif isinstance(val, str):
            name = val
            side_params = cfg.get(f"{role}_params") or cfg.get(f"{role}_config") or {}
            if isinstance(side_params, dict):
                params = dict(side_params)
        if not name:
            return None
        return {"role": role, "name": str(name), "params": params}

    include = {
        "baseline": bool(cfg.get("use_baseline", bool(cfg.get("baseline")))),
        "c1": True if ("c1" in cfg) else False,
        "c2": bool(cfg.get("use_c2", bool(cfg.get("c2")))),
        "volume": bool(cfg.get("use_volume", bool(cfg.get("volume")))),
        "exit": bool(cfg.get("use_exit", bool(cfg.get("exit")))),
    }

    enabled: Dict[str, Dict[str, Any]] = {}
    for role in roles:
        if not include.get(role, False):
            continue
        spec = _extract(role, cfg.get(role))
        if spec:
            enabled[role] = spec

    order = cfg.get("order")
    if isinstance(order, (list, tuple)):
        ordered_roles = [r for r in order if r in enabled]
        ordered_roles += [r for r in roles if r in enabled and r not in ordered_roles]
    else:
        ordered_roles = [r for r in roles if r in enabled]

    return [enabled[r] for r in ordered_roles]

# =============== Recursive container normalization/flattening ===============
def _is_spec_dict(d: Dict[str, Any]) -> bool:
    return isinstance(d, dict) and "role" in d and any(k in d for k in ("name", "short_name", "indicator"))

def _flatten_specs(obj: Any) -> List[Any]:
    """
    Recursively flatten any mixture of lists/tuples/dicts into a list of raw
    spec items. Compact dicts are expanded into proper spec dicts.
    Only recognized containers/specs are kept; unrelated scalars/param dicts are ignored.
    """
    out: List[Any] = []
    if obj is None:
        return out

    # Skip raw scalars/strings entirely (they are often params like 50, "meta", etc.)
    if isinstance(obj, (str, int, float, bool)):
        return out

    # Sequence
    if isinstance(obj, (list, tuple)):
        for item in obj:
            out.extend(_flatten_specs(item))
        return out

    # Dict
    if isinstance(obj, dict):
        # Compact indicator config → expand and return
        if _is_compact_indicator_dict(obj):
            out.extend(_expand_compact_indicator_config(obj))
            return out

        # Proper spec dict → keep as-is
        if _is_spec_dict(obj):
            out.append(obj)
            return out

        # Known container keys → recurse into those values only
        handled = False
        for key in _CONTAINER_KEYS:
            if key in obj:
                out.extend(_flatten_specs(obj[key]))
                handled = True
        if handled:
            return out

        # Otherwise: likely a parameter/meta dict → ignore (do NOT recurse)
        return out

    # Fallback: ignore unknown types
    return out

def _normalize_indicator_container(
    indicator_configs: Union[Iterable[Any], Dict[str, Any]]
) -> List[Any]:
    """
    Accept either:
      - iterable of specs [(role, name, params), {...}, ...]
      - dict with compact role keys ('c1','baseline','exit', plus 'use_*')
      - dict with list-like under one of: 'indicators','pipeline','steps','specs','list'
      - nested mixtures of the above
    Return a flat list of raw spec items (dict specs or tuples).
    """
    return _flatten_specs(indicator_configs)

# ------------------------ Flexible spec parsing -----------------------------
def _parse_indicator_spec(
    spec: Union[Tuple[Any, ...], Dict[str, Any]]
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Normalize to (role, short_name, params) from:
      - (role, short_name, params)
      - (role, short_name, params, anything_else...)  # extras ignored
      - {"role": "...", "name"/"short_name"/"indicator": "...", "params": {...}}
    """
    if isinstance(spec, dict):
        # By this point, compact dicts should have been expanded already.
        if _is_compact_indicator_dict(spec):
            raise ValueError(
                "Internal error: compact indicator dict reached _parse_indicator_spec."
            )
        role = str(spec.get("role") or "").strip().lower()
        name = spec.get("name") or spec.get("short_name") or spec.get("indicator")
        if not role or not name:
            raise ValueError(f"Invalid indicator spec dict (need role + name): {spec}")
        params = spec.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(f"'params' must be a dict in spec: {spec}")
        return role, str(name), params

    if isinstance(spec, (list, tuple)):
        if len(spec) < 3:
            raise ValueError(
                f"Indicator tuple must have at least 3 items (role, name, params); got: {spec}"
            )
        role = str(spec[0]).strip().lower()
        name = spec[1]
        params = spec[2] if isinstance(spec[2], dict) else {}
        return role, str(name), params

    raise TypeError(f"Unsupported indicator spec type: {type(spec).__name__} -> {spec!r}")

# -----------------------------------------------------------------------------

def apply_indicators(
    df,
    indicator_configs: Union[Iterable[Union[Tuple[Any, ...], Dict[str, Any]]], Dict[str, Any]]
):
    """
    Apply a list/iterable/dict of indicator specs to df in order.

    Each function must accept (df, *, signal_col=..., **kwargs) and return df.
    Baselines must also maintain/return df['baseline'] (numeric series).
    """
    queue = _normalize_indicator_container(indicator_configs)
    applied = []

    i = 0
    while i < len(queue):
        raw_spec = queue[i]
        i += 1

        # Skip no-ops / stray strings / None / stray scalars
        if raw_spec is None or isinstance(raw_spec, (str, int, float, bool)):
            continue

        # Defensively expand if a compact dict leaked in
        if isinstance(raw_spec, dict) and _is_compact_indicator_dict(raw_spec):
            expanded = _expand_compact_indicator_config(raw_spec)
            queue[i:i] = expanded
            continue

        role, short_name, params = _parse_indicator_spec(raw_spec)

        if role not in _SIGNAL_COL_BY_ROLE:
            raise ImportError(f"Unknown indicator role '{role}'. Known: {sorted(_SIGNAL_COL_BY_ROLE)}")

        sig_col = _SIGNAL_COL_BY_ROLE[role]
        fn, fq_name = _resolve_func(role, short_name)

        try:
            kwargs = dict(params or {})
            kwargs.setdefault("signal_col", sig_col)
            df = fn(df, **kwargs)
        except TypeError as e:
            raise TypeError(f"{fq_name} signature mismatch for role '{role}'.") from e

        if sig_col not in df.columns:
            raise ValueError(f"{fq_name} did not create/update required column '{sig_col}'.")

        applied.append((role, fq_name, params))

    df.attrs["applied_indicators"] = applied
    return df

# =========================================================
# Signal coercion / hygiene
# =========================================================
def _clip_to_set(series, valid=(-1, 0, 1)):
    return series.where(series.isin(valid), 0)

def coerce_entry_exit_signals(df):
    """Ensure canonical signal columns exist and are clean."""
    for col in ("c1_signal", "c2_signal", "baseline_signal", "volume_signal", "exit_signal"):
        if col not in df.columns:
            df[col] = 0
        df[col] = _clip_to_set(df[col].fillna(0))
    # baseline may be added by baseline module; no-op here
    return df

# =========================================================
# >>> Serialization hygiene: persist final stop at exit <<<
# =========================================================
def finalize_trade_row(
    row: Dict[str, Any],
    *,
    current_stop_price_at_exit: Optional[float] = None,
) -> Dict[str, Any]:
    """Persist the FINAL stop in effect at exit time for strict audits.

    Rules:
      - sl_at_exit_price := stop in effect at exit
      - Breakeven exits  -> sl_at_exit_price = entry_price
      - Trailing exits   -> ts_active=True, ts_level=final stop; sl_at_exit_price=final stop
      - Hard stops       -> sl_at_exit_price = stop at exit
    """
    exit_reason = str(row.get("exit_reason", "")).lower()
    entry_price = row.get("entry_price")

    # Initialize from provided final stop; otherwise keep prior (or entry SL)
    if current_stop_price_at_exit is not None:
        row["sl_at_exit_price"] = float(current_stop_price_at_exit)
    else:
        row.setdefault("sl_at_exit_price", row.get("sl_at_entry_price"))

    # Breakeven semantics
    if ("breakeven" in exit_reason) or bool(row.get("breakeven_after_tp1")):
        if entry_price is not None:
            row["sl_at_exit_price"] = float(entry_price)

    # Trailing stop semantics
    if "trailing" in exit_reason:
        if current_stop_price_at_exit is not None:
            row["ts_level"] = float(current_stop_price_at_exit)
        row["ts_active"] = True
        if current_stop_price_at_exit is not None:
            row["sl_at_exit_price"] = float(current_stop_price_at_exit)

    # Hard stop (non-TS)
    if ("stop" in exit_reason) and ("trailing" not in exit_reason):
        if current_stop_price_at_exit is not None:
            row["sl_at_exit_price"] = float(current_stop_price_at_exit)

    # Ensure immutable-at-entry fields exist (don’t mutate them here)
    row.setdefault("tp1_at_entry_price", row.get("tp1_price"))
    row.setdefault("sl_at_entry_price", row.get("sl_price"))

    return row

def build_and_finalize_trade_row(
    raw_row: Dict[str, Any],
    *,
    current_stop_price_at_exit: Optional[float] = None
) -> Dict[str, Any]:
    """Helper for callsites that want a copy before finalization."""
    row = dict(raw_row)  # shallow copy
    return finalize_trade_row(row, current_stop_price_at_exit=current_stop_price_at_exit)
