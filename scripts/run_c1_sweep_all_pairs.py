#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_c1_sweep_all_pairs.py — C1-only sweep over all pairs
----------------------------------------------------------
Runs a backtest for every (pair, C1) combination and aggregates results.

Usage:
    python scripts/run_c1_sweep_all_pairs.py --config configs/c1_only_sweep_2023_2025.yaml
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import itertools
import json
import multiprocessing as mp
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.backtester import run_backtest  # noqa: E402


def discover_c1_indicators() -> List[str]:
    """Discover all C1 indicator names from confirmation_funcs.py."""
    try:
        mod = importlib.import_module("indicators.confirmation_funcs")
        names = []
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if name.startswith("c1_") or name == "supertrend":
                # Remove 'c1_' prefix, keep 'supertrend' as-is
                base = name[3:] if name.startswith("c1_") else name
                names.append(base)
        return sorted(set(names))
    except Exception as e:
        print(f"Error discovering C1 indicators: {e}")
        return []


def discover_pairs(data_dir: Path) -> List[str]:
    """Discover all pairs from data/daily/ directory."""
    pairs = []
    if not data_dir.exists():
        return pairs
    for csv_file in data_dir.glob("*.csv"):
        pair_name = csv_file.stem  # e.g., "EUR_USD"
        pairs.append(pair_name)
    return sorted(pairs)


def load_base_config(config_path: Path) -> Dict[str, Any]:
    """Load base config and sweep settings."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_run_config(
    base_config: Dict[str, Any],
    pair: str,
    c1_name: str,
    sweep_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a config dict for a single (pair, C1) run."""
    # Deep copy base config
    config = json.loads(json.dumps(base_config))

    # Apply static overrides from sweep config
    static = sweep_config.get("static_overrides", {})
    for key, value in static.items():
        if isinstance(value, dict):
            config.setdefault(key, {}).update(value)
        else:
            config[key] = value

    # Set pair and C1
    config["pairs"] = [pair]
    config["indicators"] = config.get("indicators", {})
    config["indicators"]["c1"] = c1_name
    config["indicators"]["use_c2"] = False
    config["indicators"]["use_baseline"] = False
    config["indicators"]["use_volume"] = False
    config["indicators"]["use_exit"] = False

    return config


def worker_job(
    run_id: int,
    base_config: Dict[str, Any],
    pair: str,
    c1_name: str,
    sweep_config: Dict[str, Any],
    results_root: Path,
) -> Dict[str, Any]:
    """Run a single backtest and return metrics."""
    try:
        # Create run config
        config = create_run_config(base_config, pair, c1_name, sweep_config)

        # Create per-run results directory
        run_slug = f"{pair}__{c1_name}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = results_root / run_slug
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write temp config
        temp_config_path = run_dir / "config.yaml"
        with open(temp_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        # Run backtest
        run_backtest(config_path=str(temp_config_path), results_dir=str(run_dir))

        # Parse results
        from core.utils import summarize_results

        _, metrics = summarize_results(run_dir, starting_balance=10000.0)

        # Add metadata
        metrics["pair"] = pair
        metrics["c1_name"] = c1_name
        metrics["run_slug"] = run_slug
        metrics["run_dir"] = str(run_dir)

        return metrics

    except Exception as e:
        print(f"❌ Run failed: {pair} / {c1_name} -> {e}")
        traceback.print_exc()
        return {
            "pair": pair,
            "c1_name": c1_name,
            "run_slug": f"{pair}__{c1_name}__failed",
            "run_dir": "",
            "total_trades": 0,
            "roi_pct": 0.0,
            "win_rate_ns": 0.0,
            "max_dd_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run C1-only sweep over all pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/c1_only_sweep_2023_2025.yaml",
        help="Path to sweep config YAML",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/c1_sweep_2023_2025",
        help="Root directory for all run results",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto = CPU count - 1)",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/config.yaml",
        help="Base config file to merge with sweep config",
    )

    args = parser.parse_args()

    # Load configs
    sweep_config_path = PROJECT_ROOT / args.config
    base_config_path = PROJECT_ROOT / args.base_config

    if not sweep_config_path.exists():
        print(f"❌ Sweep config not found: {sweep_config_path}")
        sys.exit(1)

    if not base_config_path.exists():
        print(f"❌ Base config not found: {base_config_path}")
        sys.exit(1)

    sweep_config = load_base_config(sweep_config_path)
    base_config = load_base_config(base_config_path)

    # Discover C1 indicators and pairs
    print("🔍 Discovering C1 indicators...")
    c1_indicators = discover_c1_indicators()
    print(f"   Found {len(c1_indicators)} C1 indicators")

    data_dir = PROJECT_ROOT / sweep_config.get("static_overrides", {}).get("data_dir", "data/daily")
    print(f"🔍 Discovering pairs from {data_dir}...")
    pairs = discover_pairs(data_dir)
    print(f"   Found {len(pairs)} pairs")

    if not c1_indicators or not pairs:
        print("❌ No C1 indicators or pairs found!")
        sys.exit(1)

    # Create results root
    results_root = PROJECT_ROOT / args.results_root
    results_root.mkdir(parents=True, exist_ok=True)

    # Generate all (pair, C1) combinations
    combos = list(itertools.product(pairs, c1_indicators))
    print(f"\n📊 Total runs: {len(combos)} ({len(pairs)} pairs × {len(c1_indicators)} C1 indicators)")

    # Determine workers
    if args.workers:
        workers = args.workers
    else:
        workers = max(1, mp.cpu_count() - 1)
    print(f"⚙️  Using {workers} parallel workers\n")

    # Run all combinations
    started = datetime.now()
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for i, (pair, c1_name) in enumerate(combos):
            fut = executor.submit(
                worker_job, i, base_config, pair, c1_name, sweep_config, results_root
            )
            futures[fut] = (pair, c1_name)

        completed = 0
        for fut in as_completed(futures):
            pair, c1_name = futures[fut]
            try:
                metrics = fut.result()
                results.append(metrics)
                completed += 1
                if completed % 10 == 0:
                    print(f"   Progress: {completed}/{len(combos)} runs completed")
            except Exception as e:
                print(f"❌ Error processing {pair}/{c1_name}: {e}")

    # Save aggregated results
    if results:
        df = pd.DataFrame(results)
        output_csv = results_root / "aggregated_results.csv"
        df.to_csv(output_csv, index=False)
        print("\n✅ Sweep completed!")
        print(f"   Total runs: {len(results)}")
        print(f"   Results CSV: {output_csv}")
        print(f"   Results root: {results_root}")
        print(f"   Duration: {datetime.now() - started}")
    else:
        print("\n❌ No results collected!")


if __name__ == "__main__":
    main()

