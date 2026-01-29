#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_c1_volume_sweep.py — Sweep over C1 indicators × all volume filters
---------------------------------------------------------------------------

Runs a backtest for every (pair, C1, volume) combination and writes results
into per-run folders whose names encode the combo:

    c1_<c1_name>__vol_<volume_name>__exit_<exit_name>__pair_<pair>

Usage:
    python scripts/run_c1_volume_sweep.py \
        --base-config configs/config.yaml \
        --results-root results/c1_volume_sweep
        --full-universe  # Use full C1 list from baseline aggregate
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from core.backtester import run_backtest  # noqa: E402,I001


# Results root for all C1 + exit + volume sweeps
RESULTS_ROOT = PROJECT_ROOT / "results" / "c1_w_exit_plus_vol_results"
RESULTS_ROOT_FULL = PROJECT_ROOT / "results" / "c1_w_exit_plus_vol_results_full"

# Top-performing C1s (YAML short names) - used only if --full-universe not specified
TOP_C1_NAMES: List[str] = [
    "disparity_index",
    "kalman_filter",
    "lwpi",
    "smoothed_momentum",
    "trend_akkam",
    "twiggs_money_flow",
]

# Standard NNFX-style exit
EXIT_NAME = "exit_twiggs_money_flow"
DEFAULT_EXIT_PARAMS: Dict[str, Any] = {}


def discover_pairs(data_dir: Path) -> List[str]:
    """Discover all pairs from data directory (e.g., data/daily)."""
    if not data_dir.exists():
        return []
    pairs: List[str] = []
    for csv_file in data_dir.glob("*.csv"):
        pairs.append(csv_file.stem)
    return sorted(set(pairs))


def discover_volume_indicators() -> List[str]:
    """Discover all volume indicator short names from indicators.volume_funcs."""
    try:
        mod = importlib.import_module("indicators.volume_funcs")
        names: List[str] = []
        for name, obj in vars(mod).items():
            if callable(obj) and name.startswith("volume_"):
                short = name[len("volume_") :]
                names.append(short)
        return sorted(set(names))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Error discovering volume indicators: {exc}")
        return []


def load_full_c1_universe(baseline_csv: Path) -> List[str]:
    """Load full C1 universe from baseline aggregate CSV."""
    if not baseline_csv.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {baseline_csv}")
    
    df = pd.read_csv(baseline_csv)
    if "c1" not in df.columns:
        raise ValueError(f"Baseline CSV missing 'c1' column. Available: {list(df.columns)}")
    
    c1_list = sorted(df["c1"].dropna().unique().tolist())
    return c1_list


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_run_config(
    base_config: Dict[str, Any],
    pair: str,
    c1_name: str,
    volume_name: str,
) -> Dict[str, Any]:
    """Return a config dict for a single (pair, C1, volume) run."""
    config: Dict[str, Any] = json.loads(json.dumps(base_config))

    config["pairs"] = [pair]
    indicators = config.get("indicators") or {}
    indicators["c1"] = c1_name
    indicators["use_c2"] = False
    indicators["use_baseline"] = indicators.get("use_baseline", False)
    indicators["use_exit"] = True
    indicators["use_volume"] = True
    indicators["volume"] = volume_name
    indicators["exit"] = EXIT_NAME
    config["indicators"] = indicators

    # Ensure exit behavior flags match NNFX-style Twiggs exit
    exit_cfg = config.get("exit") or {}
    exit_cfg.setdefault("use_trailing_stop", True)
    exit_cfg.setdefault("move_to_breakeven_after_atr", True)
    exit_cfg.setdefault("exit_on_c1_reversal", True)
    exit_cfg.setdefault("exit_on_baseline_cross", False)
    exit_cfg.setdefault("exit_on_exit_signal", True)
    config["exit"] = exit_cfg

    # Wire indicator_params for exit if provided (currently empty)
    if DEFAULT_EXIT_PARAMS:
        ind_params = config.get("indicator_params") or {}
        ind_params.setdefault(EXIT_NAME, {}).update(DEFAULT_EXIT_PARAMS)
        config["indicator_params"] = ind_params

    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep over C1 indicators × all volume filters",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/config.yaml",
        help="Base config YAML to clone per run",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/daily",
        help="Directory containing per-pair OHLCV CSVs",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="",
        help="Root directory for all (c1, volume, pair) runs (auto-set if --full-universe)",
    )
    parser.add_argument(
        "--volume-filter",
        type=str,
        default="",
        help=(
            "Comma-separated list of volume indicator short names to include; "
            "defaults to all discovered."
        ),
    )
    parser.add_argument(
        "--full-universe",
        action="store_true",
        help="Use full C1 universe from baseline aggregate (results/c1_exit_baseline_aggregated.csv)",
    )
    parser.add_argument(
        "--baseline-csv",
        type=str,
        default="results/c1_exit_baseline_aggregated.csv",
        help="Path to baseline aggregate CSV (used with --full-universe)",
    )
    args = parser.parse_args()

    base_config_path = PROJECT_ROOT / args.base_config
    data_dir = PROJECT_ROOT / args.data_dir

    if not base_config_path.exists():
        raise SystemExit(f"Base config not found: {base_config_path}")

    # Determine C1 list
    if args.full_universe:
        baseline_csv = PROJECT_ROOT / args.baseline_csv
        c1_names = load_full_c1_universe(baseline_csv)
        if args.results_root:
            results_root = Path(args.results_root)
        else:
            results_root = RESULTS_ROOT_FULL
        print(f"Using FULL C1 universe: {len(c1_names)} indicators from {baseline_csv}")
    else:
        c1_names = TOP_C1_NAMES
        if args.results_root:
            results_root = Path(args.results_root)
        else:
            results_root = RESULTS_ROOT
        print(f"Using TOP C1s: {len(c1_names)} indicators")

    base_config = load_config(base_config_path)
    pairs = discover_pairs(data_dir)
    volume_names = discover_volume_indicators()

    if args.volume_filter:
        wanted = {v.strip() for v in str(args.volume_filter).split(",") if v.strip()}
        volume_names = [v for v in volume_names if v in wanted]

    if args.volume_filter and not volume_names:
        raise SystemExit(
            f"No volume indicators matched filter {args.volume_filter!r}. "
            f"Available: {', '.join(discover_volume_indicators())}"
        )

    if not pairs:
        raise SystemExit(f"No pairs discovered in {data_dir}")
    if not volume_names:
        raise SystemExit("No volume indicators discovered in indicators.volume_funcs")

    results_root.mkdir(parents=True, exist_ok=True)

    print("Pairs:", ", ".join(pairs))
    print("C1s:", f"{len(c1_names)} indicators")
    print("Volumes:", ", ".join(volume_names))
    print(f"Total runs: {len(pairs)} × {len(c1_names)} × {len(volume_names)} = {len(pairs) * len(c1_names) * len(volume_names)}")

    start = datetime.now()
    total = 0

    for pair in pairs:
        for c1_name in c1_names:
            for vol_name in volume_names:
                config = create_run_config(base_config, pair, c1_name, vol_name)
                exit_name = EXIT_NAME
                run_slug = f"c1_{c1_name}__vol_{vol_name}__exit_{exit_name}__pair_{pair}"
                run_dir = results_root / run_slug
                run_dir.mkdir(parents=True, exist_ok=True)

                cfg_path = run_dir / "config.yaml"
                with cfg_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(config, f, sort_keys=False)

                print(f"pair={pair} c1={c1_name} volume={vol_name} exit={exit_name}")
                run_backtest(config_path=str(cfg_path), results_dir=str(run_dir))
                total += 1

    elapsed = datetime.now() - start
    print(f"\nDone. Runs: {total}, elapsed: {elapsed}")


if __name__ == "__main__":
    main()


