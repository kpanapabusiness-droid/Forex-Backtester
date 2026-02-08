# Phase B — shared helpers (config validation, discovery, default param grids).
# No WFO or leaderboard logic.

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


REQUIRED_KEYS_C1 = [
    "pairs",
    "date_range",
    "outputs",
    "spreads",
    "phaseB",
]
REQUIRED_KEYS_VOLUME = REQUIRED_KEYS_C1 + []
REQUIRED_KEYS_OVERFIT = REQUIRED_KEYS_C1 + []


def require_phaseB_config(cfg: dict, mode: str) -> None:
    """Fail fast if config missing required keys. No defaults."""
    dr = cfg.get("date_range") or {}
    out = cfg.get("outputs") or {}
    spreads = cfg.get("spreads") or {}
    pb = cfg.get("phaseB") or {}
    if not isinstance(cfg.get("pairs"), list) or len(cfg["pairs"]) == 0:
        raise ValueError("Config must have non-empty 'pairs' list")
    if not dr.get("start") or not dr.get("end"):
        raise ValueError("Config must have date_range.start and date_range.end")
    if not out.get("dir"):
        raise ValueError("Config must have outputs.dir")
    if spreads.get("enabled") is not True:
        raise ValueError("Config must have spreads.enabled: true")
    if not pb.get("run_name"):
        raise ValueError("Config must have phaseB.run_name")
    if not pb.get("mode"):
        raise ValueError("Config must have phaseB.mode")
    if mode == "volume" and not pb.get("c1_baseline"):
        raise ValueError("Config must have phaseB.c1_baseline for volume diagnostics")
    if mode == "controlled_overfit":
        folds = pb.get("diagnostic_fold_pairs") or []
        if len(folds) < 3:
            raise ValueError("phaseB.diagnostic_fold_pairs must have at least 3 fold pairs")


def discover_c1_indicators() -> List[str]:
    """Discover C1 names from indicators.confirmation_funcs (same pattern as Phase 4)."""
    from indicators import confirmation_funcs

    names = [
        name
        for name, obj in inspect.getmembers(confirmation_funcs, inspect.isfunction)
        if name.startswith("c1_")
    ]
    return sorted(set(names))


def discover_volume_indicators() -> List[str]:
    """Discover volume indicator short names (e.g. volatility_ratio). Same as batch_sweeper."""
    import importlib

    mod = importlib.import_module("indicators.volume_funcs")
    names = [
        name[len("volume_") :]
        for name, obj in inspect.getmembers(mod, inspect.isfunction)
        if name.startswith("volume_")
    ]
    return sorted(set(names))


def default_c1_param_grid(c1_name: str) -> List[Dict[str, Any]]:
    """Single place for small default param grid per C1. Override via YAML in config."""
    grids = {
        "c1_coral": [{"efficiency": 0.2}, {"efficiency": 0.35}, {"efficiency": 0.5}],
        "c1_disparity_index": [{"period": 14}, {"period": 21}, {"period": 28}],
        "c1_hlc_trend": [{"atr_period": 14}, {"atr_period": 21}],
        "c1_rsi": [{"period": 14}, {"period": 21}],
        "c1_lwpi": [{"length": 14}, {"length": 21}],
        "c1_twiggs_money_flow": [{"length": 15}, {"length": 21}],
        "c1_twiggs_money_flow_mq5": [{"period": 14}, {"period": 21}],
        "c1_fisher_transform": [{"period": 9}, {"period": 14}],
        "c1_waddah_attar_explosion": [
            {"sensitivity": 150, "fast_length": 20},
            {"sensitivity": 200, "fast_length": 20},
        ],
    }
    if c1_name in grids:
        return grids[c1_name]
    return [{}]


def default_volume_param_grid(vol_name: str) -> List[Dict[str, Any]]:
    """Default param grid for volume (threshold/length). Override via YAML."""
    if vol_name in ("volatility_ratio", "volume_volatility_ratio", "normalized"):
        return [
            {"length": 20, "smooth": 50, "threshold": 1.1},
            {"length": 20, "smooth": 50, "threshold": 1.2},
            {"length": 20, "smooth": 50, "threshold": 1.35},
        ]
    if vol_name in ("adx", "volume_adx", "trend_direction_force"):
        return [
            {"length": 14, "min_adx": 18},
            {"length": 14, "min_adx": 22},
            {"length": 14, "min_adx": 26},
        ]
    if vol_name == "waddah_attar_explosion":
        return [
            {"sensitivity": 150, "dead_zone": 20},
            {"sensitivity": 200, "dead_zone": 25},
        ]
    return [{}]


def merge_indicator_params(cfg: dict, func_name: str, params: dict) -> dict:
    """Return config with indicator_params set for backtester lookup (merge)."""
    from copy import deepcopy

    c = deepcopy(cfg)
    c.setdefault("indicator_params", {})[func_name] = dict(params)
    if func_name.startswith("c1_"):
        key = f"indicators.confirmation_funcs.{func_name}"
        c["indicator_params"][key] = dict(params)
    return c
