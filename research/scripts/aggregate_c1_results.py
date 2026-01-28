#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_c1_results.py — Aggregate C1 sweep results into comparison CSV
----------------------------------------------------------------------------
Recursively walks results directory, extracts metrics from each run, and
creates a single comparison CSV table.

Usage:
    python scripts/aggregate_c1_results.py --results-root results/c1_sweep_2023_2025 --output results/c1_comparison_2023_2025.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.utils import summarize_results  # noqa: E402


def _parse_float(text: str) -> float:
    """Extract first float from text, return 0.0 if not found."""
    if not text:
        return 0.0
    try:
        match = re.search(r"-?\d+(?:\.\d+)?", str(text))
        return float(match.group(0)) if match else 0.0
    except Exception:
        return 0.0


def parse_summary_txt(summary_path: Path) -> Dict[str, Any]:
    """Parse summary.txt and extract key metrics."""
    metrics = {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "scratches": 0,
        "win_rate_pct": 0.0,
        "roi_pct": 0.0,
        "max_dd_pct": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "mar": 0.0,
        "expectancy": 0.0,
    }

    if not summary_path.exists():
        return metrics

    try:
        content = summary_path.read_text(encoding="utf-8", errors="ignore")
        lines = [line.strip() for line in content.splitlines() if line.strip()]

        for line in lines:
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            # Parse various metric formats
            if "total trades" in key or "trades" in key:
                metrics["total_trades"] = int(_parse_float(value))
            elif "wins" in key and "rate" not in key:
                metrics["wins"] = int(_parse_float(value))
            elif "losses" in key and "rate" not in key:
                metrics["losses"] = int(_parse_float(value))
            elif "scratches" in key and "rate" not in key:
                metrics["scratches"] = int(_parse_float(value))
            elif "win%" in key or ("win" in key and "rate" in key):
                metrics["win_rate_pct"] = _parse_float(value)
            elif "roi" in key and "%" in value:
                metrics["roi_pct"] = _parse_float(value)
            elif "max dd" in key or "maxdd" in key:
                metrics["max_dd_pct"] = _parse_float(value)
            elif "sharpe" in key:
                metrics["sharpe"] = _parse_float(value)
            elif "sortino" in key:
                metrics["sortino"] = _parse_float(value)
            elif "mar" in key and ":" in line:
                metrics["mar"] = _parse_float(value)
            elif "expectancy" in key:
                metrics["expectancy"] = _parse_float(value)

    except Exception as e:
        print(f"⚠️  Warning: Error parsing {summary_path}: {e}", file=sys.stderr)

    return metrics


def compute_metrics_from_trades(trades_path: Path) -> Dict[str, Any]:
    """Compute metrics from trades.csv as fallback."""
    metrics = {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "scratches": 0,
        "win_rate_pct": 0.0,
        "roi_pct": 0.0,
        "max_dd_pct": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "mar": 0.0,
        "expectancy": 0.0,
    }

    if not trades_path.exists():
        return metrics

    try:
        df = pd.read_csv(trades_path)
        if df.empty:
            return metrics

        metrics["total_trades"] = len(df)
        metrics["wins"] = int(df.get("win", pd.Series(dtype=bool)).sum() if "win" in df.columns else 0)
        metrics["losses"] = int(
            df.get("loss", pd.Series(dtype=bool)).sum() if "loss" in df.columns else 0
        )
        metrics["scratches"] = int(
            df.get("scratch", pd.Series(dtype=bool)).sum() if "scratch" in df.columns else 0
        )

        non_scratch = metrics["wins"] + metrics["losses"]
        if non_scratch > 0:
            metrics["win_rate_pct"] = (metrics["wins"] / non_scratch) * 100.0

        if "pnl" in df.columns:
            total_pnl = float(df["pnl"].sum())
            metrics["roi_pct"] = (total_pnl / 10000.0) * 100.0  # Assuming 10k starting balance

            # Compute max drawdown from cumulative PnL
            cum_pnl = df["pnl"].cumsum() + 10000.0
            peak = cum_pnl.cummax()
            dd = ((peak - cum_pnl) / peak.replace(0, 1)) * 100.0
            metrics["max_dd_pct"] = float(dd.max()) if not dd.empty else 0.0

            # Expectancy
            if non_scratch > 0:
                metrics["expectancy"] = total_pnl / non_scratch

    except Exception as e:
        print(f"⚠️  Warning: Error reading trades.csv {trades_path}: {e}", file=sys.stderr)

    return metrics


def extract_metadata_from_dir(run_dir: Path) -> Dict[str, Any]:
    """Extract pair, C1 name, and other metadata from directory structure or config."""
    metadata = {
        "pair": "",
        "c1_name": "",
        "timeframe": "",
        "from_date": "",
        "to_date": "",
        "run_id": run_dir.name,
        "output_dir": str(run_dir),
    }

    # Try to parse from directory name (format: PAIR__C1_NAME__TIMESTAMP)
    dir_name = run_dir.name
    if "__" in dir_name:
        parts = dir_name.split("__")
        if len(parts) >= 2:
            metadata["pair"] = parts[0]
            metadata["c1_name"] = parts[1]

    # Try to read from config.yaml if present
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if not metadata["pair"] and config.get("pairs"):
                metadata["pair"] = config["pairs"][0] if isinstance(config["pairs"], list) else ""

            if not metadata["c1_name"] and config.get("indicators", {}).get("c1"):
                metadata["c1_name"] = config["indicators"]["c1"]

            if config.get("timeframe"):
                metadata["timeframe"] = str(config["timeframe"])

            date_range = config.get("date_range", {})
            if date_range:
                metadata["from_date"] = str(date_range.get("start", ""))
                metadata["to_date"] = str(date_range.get("end", ""))

        except Exception:
            pass  # Ignore config parsing errors

    return metadata


def scan_results_directory(results_root: Path) -> List[Dict[str, Any]]:
    """Recursively scan results directory and extract metrics from each run."""
    rows = []

    if not results_root.exists():
        print(f"❌ Results root does not exist: {results_root}", file=sys.stderr)
        return rows

    # Find all directories that contain summary.txt or trades.csv
    for run_dir in results_root.rglob("*"):
        if not run_dir.is_dir():
            continue

        summary_path = run_dir / "summary.txt"
        trades_path = run_dir / "trades.csv"

        # Skip if neither file exists
        if not summary_path.exists() and not trades_path.exists():
            continue

        # Extract metadata
        metadata = extract_metadata_from_dir(run_dir)

        # PRIMARY: Use summarize_results() from core/utils.py as the source of truth
        # This ensures we always use the latest logic and handle pnl_dollars correctly
        metrics = {}
        try:
            # Try to get starting balance from config if available
            starting_balance = 10000.0  # Default
            config_path = run_dir / "config.yaml"
            if config_path.exists():
                try:
                    import yaml

                    with open(config_path, "r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)
                    risk_config = config.get("risk", {})
                    if risk_config and "starting_balance" in risk_config:
                        starting_balance = float(risk_config["starting_balance"])
                except Exception:
                    pass  # Use default if config parsing fails

            # Call summarize_results() - this is the authoritative source
            _, metrics_dict = summarize_results(run_dir, starting_balance=starting_balance)

            # Map summarize_results() keys to expected column names
            # If metrics_dict is empty (no trades.csv), use defaults (all zeros)
            metrics = {
                "total_trades": metrics_dict.get("total_trades", 0),
                "wins": metrics_dict.get("wins", 0),
                "losses": metrics_dict.get("losses", 0),
                "scratches": metrics_dict.get("scratches", 0),
                "win_rate_pct": metrics_dict.get("win_rate_ns", 0.0),  # Map win_rate_ns to win_rate_pct
                "roi_pct": metrics_dict.get("roi_pct", 0.0),
                "max_dd_pct": metrics_dict.get("max_dd_pct", 0.0),
                "sharpe": metrics_dict.get("sharpe", 0.0),
                "sortino": metrics_dict.get("sortino", 0.0),
                "mar": metrics_dict.get("mar", 0.0),
                "expectancy": metrics_dict.get("expectancy", 0.0),
            }

        except Exception as e:
            # FALLBACK: If summarize_results() fails, try parsing summary.txt
            # This should be rare and only for malformed runs
            print(
                f"⚠️  Warning: summarize_results() failed for {run_dir.name}: {e}",
                file=sys.stderr,
            )
            print("   Falling back to parsing summary.txt", file=sys.stderr)

            if summary_path.exists():
                metrics = parse_summary_txt(summary_path)
            elif trades_path.exists():
                # Last resort: use legacy compute_metrics_from_trades
                metrics = compute_metrics_from_trades(trades_path)
            else:
                # Skip this run if we can't get metrics
                print("   Skipping run: no valid data source", file=sys.stderr)
                continue

        # Combine metadata and metrics
        row = {**metadata, **metrics}
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate C1 sweep results into comparison CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-root",
        type=str,
        required=True,
        help="Root directory containing per-run result folders",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path",
    )

    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_path = Path(args.output)

    print(f"🔍 Scanning results directory: {results_root}")
    rows = scan_results_directory(results_root)

    if not rows:
        print("❌ No results found!", file=sys.stderr)
        sys.exit(1)

    print(f"   Found {len(rows)} runs")

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Ensure required columns exist
    required_cols = [
        "pair",
        "c1_name",
        "timeframe",
        "from_date",
        "to_date",
        "total_trades",
        "roi_pct",
        "win_rate_pct",
        "max_dd_pct",
        "sharpe",
        "sortino",
        "run_id",
        "output_dir",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    # Sort by pair, then by ROI descending
    df = df.sort_values(["pair", "roi_pct"], ascending=[True, False])

    # Reorder columns
    col_order = [
        "pair",
        "c1_name",
        "timeframe",
        "from_date",
        "to_date",
        "total_trades",
        "wins",
        "losses",
        "scratches",
        "win_rate_pct",
        "roi_pct",
        "max_dd_pct",
        "sharpe",
        "sortino",
        "mar",
        "expectancy",
        "run_id",
        "output_dir",
    ]

    # Only include columns that exist
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    # Write to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Aggregated {len(df)} runs")
    print(f"   Output: {output_path}")
    print("\n   Top 5 by ROI:")
    top5 = df.nlargest(5, "roi_pct")[["pair", "c1_name", "roi_pct", "win_rate_pct", "max_dd_pct"]]
    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()

