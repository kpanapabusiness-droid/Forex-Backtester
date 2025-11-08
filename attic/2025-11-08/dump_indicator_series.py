#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dump_indicator_series.py — Standalone indicator analysis for C1 sweep triage
---------------------------------------------------------------------------
Purpose: Analyze C1 indicators in isolation to verify they produce signals.

Usage:
    python scripts/dump_indicator_series.py --pair EURUSD --c1 c1_supertrend --from 2020-01-01 --to 2021-12-31

Features:
- Loads data and computes specified C1 indicator + baseline
- Writes indicator_summary.txt with signal counts and statistics
- Writes indicator_series.csv with time series data
- Prints console summary of signal distribution
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.utils import calculate_atr, load_pair_csv  # noqa: E402
from indicators_cache import apply_indicators_with_cache  # noqa: E402


def normalize_pair_format(pair: str) -> str:
    """Normalize pair format to match file naming convention."""
    pair = pair.upper().strip()
    if len(pair) == 6 and "_" not in pair:
        # Convert EURUSD to EUR_USD
        return f"{pair[:3]}_{pair[3:]}"
    return pair


def find_data_file(pair: str, timeframe: str, data_root: Path) -> Optional[Path]:
    """Find data file for given pair and timeframe."""

    # Map timeframe to directory
    timeframe_dirs = {"D": "daily", "H1": "hourly", "H4": "4h", "M15": "15m", "M30": "30m"}

    # Try exact timeframe first
    if timeframe in timeframe_dirs:
        data_dir = data_root / timeframe_dirs[timeframe]
        data_file = data_dir / f"{pair}.csv"
        if data_file.exists():
            return data_file

    # Fallback to daily data (most likely to exist)
    daily_dir = data_root / "daily"
    daily_file = daily_dir / f"{pair}.csv"
    if daily_file.exists():
        return daily_file

    # Try test data as last resort
    test_dir = data_root / "test"
    test_file = test_dir / f"{pair}.csv"
    if test_file.exists():
        return test_file

    return None


def create_indicator_config(
    pair: str, c1_indicator: str, timeframe: str = "H1", baseline: str = "baseline_ema"
) -> Dict[str, Any]:
    """Create minimal config for indicator computation."""

    # Map timeframe to data directory
    data_dir_map = {"D": "data/daily", "H1": "data/hourly", "H4": "data/4h", "M15": "data/15m"}
    data_dir = data_dir_map.get(timeframe, "data/daily")

    config = {
        "pairs": [pair],
        "timeframe": timeframe,
        "data_dir": data_dir,
        "indicators": {
            "c1": c1_indicator,
            "use_c2": False,
            "use_baseline": True,
            "baseline": baseline,
            "use_volume": False,
            "use_exit": False,
        },
        "cache": {
            "enabled": True,
            "dir": "cache",
            "format": "parquet",
        },
        "validation": {
            "enabled": False,  # Skip validation for speed
        },
    }

    return config


def analyze_indicator_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze indicator signals and return statistics."""
    stats = {
        "total_bars": len(df),
        "non_null_signals": 0,
        "signal_counts": {-1: 0, 0: 0, 1: 0},
        "first_nonzero_ts": None,
        "last_nonzero_ts": None,
        "baseline_available": False,
        "c1_column": None,
    }

    # Find C1 signal column
    c1_cols = [col for col in df.columns if "c1" in col.lower() and "signal" in col.lower()]
    if c1_cols:
        c1_col = c1_cols[0]
        stats["c1_column"] = c1_col

        # Analyze C1 signals
        c1_series = pd.to_numeric(df[c1_col], errors="coerce").fillna(0)
        stats["non_null_signals"] = (c1_series != 0).sum()

        # Count signal values
        for val in [-1, 0, 1]:
            stats["signal_counts"][val] = (c1_series == val).sum()

        # Find first and last non-zero signals
        nonzero_mask = c1_series != 0
        if nonzero_mask.any():
            first_idx = nonzero_mask.idxmax()
            last_idx = nonzero_mask[::-1].idxmax()

            if "date" in df.columns:
                stats["first_nonzero_ts"] = str(df.loc[first_idx, "date"])
                stats["last_nonzero_ts"] = str(df.loc[last_idx, "date"])
            else:
                stats["first_nonzero_ts"] = str(first_idx)
                stats["last_nonzero_ts"] = str(last_idx)

    # Check baseline availability
    if "baseline" in df.columns:
        stats["baseline_available"] = True

    return stats


def write_indicator_summary(stats: Dict[str, Any], output_file: Path) -> None:
    """Write indicator analysis summary to text file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_bars = stats["total_bars"]
    non_null = stats["non_null_signals"]
    counts = stats["signal_counts"]

    nonzero_pct = (non_null / total_bars * 100) if total_bars > 0 else 0

    summary = f"""Indicator Analysis Summary
========================

Data Overview:
- Total bars: {total_bars:,}
- Non-null signals: {non_null:,}
- Signal coverage: {nonzero_pct:.2f}%

Signal Distribution:
- Bearish (-1): {counts[-1]:,} ({counts[-1] / total_bars * 100:.2f}%)
- Neutral (0): {counts[0]:,} ({counts[0] / total_bars * 100:.2f}%)
- Bullish (+1): {counts[1]:,} ({counts[1] / total_bars * 100:.2f}%)

Time Range:
- First non-zero signal: {stats["first_nonzero_ts"]}
- Last non-zero signal: {stats["last_nonzero_ts"]}

Indicator Details:
- C1 column: {stats["c1_column"]}
- Baseline available: {stats["baseline_available"]}
"""

    with open(output_file, "w") as f:
        f.write(summary)


def write_indicator_series(df: pd.DataFrame, output_file: Path) -> None:
    """Write indicator time series to CSV."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Select columns for export
    export_cols = []

    # Add timestamp
    if "date" in df.columns:
        export_cols.append("date")

    # Add price data
    if "close" in df.columns:
        export_cols.append("close")

    # Add indicator columns
    for col in df.columns:
        if "c1" in col.lower() and "signal" in col.lower():
            export_cols.append(col)
        elif col == "baseline":
            export_cols.append(col)

    # Create export dataframe
    if export_cols:
        export_df = df[export_cols].copy()

        # Rename for clarity
        if "close" in export_df.columns:
            export_df = export_df.rename(columns={"close": "price"})

        # Add baseline trend if baseline is available
        if "baseline" in export_df.columns and "price" in export_df.columns:
            export_df["baseline_trend"] = export_df.apply(
                lambda row: "bullish"
                if row["price"] > row["baseline"]
                else "bearish"
                if row["price"] < row["baseline"]
                else "neutral",
                axis=1,
            )

        export_df.to_csv(output_file, index=False)
    else:
        # Create empty CSV with headers
        empty_df = pd.DataFrame(columns=["ts", "close", "c1_signal", "baseline_trend"])
        empty_df.to_csv(output_file, index=False)


def print_console_summary(stats: Dict[str, Any]) -> None:
    """Print one-line console summary."""
    counts = stats["signal_counts"]
    total = stats["total_bars"]
    nonzero = counts[-1] + counts[1]  # Exclude neutral signals

    nonzero_pct = (nonzero / total * 100) if total > 0 else 0

    print(
        f"signals: neg={counts[-1]} zero={counts[0]} pos={counts[1]}; nonzero% = {nonzero_pct:.1f}%"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Dump and analyze indicator series for debugging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/dump_indicator_series.py --pair EURUSD --c1 c1_supertrend --from 2020-01-01 --to 2021-12-31
  python scripts/dump_indicator_series.py --pair GBPUSD --c1 c1_twiggs_money_flow --from 2019-01-01 --to 2020-12-31 --timeframe D
        """,
    )

    parser.add_argument("--pair", required=True, help="Currency pair (e.g., EURUSD)")
    parser.add_argument("--c1", required=True, help="C1 indicator name (e.g., c1_supertrend)")
    parser.add_argument("--from", dest="start_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="end_date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", default="H1", help="Timeframe (default: H1)")
    parser.add_argument(
        "--output",
        default="results/indicator_dump",
        help="Output directory (default: results/indicator_dump)",
    )
    parser.add_argument(
        "--baseline", default="baseline_ema", help="Baseline indicator (default: baseline_ema)"
    )

    args = parser.parse_args()

    # Normalize pair format
    pair = normalize_pair_format(args.pair)

    print("🔍 Analyzing indicator series:")
    print(f"   Pair: {pair}")
    print(f"   C1: {args.c1}")
    print(f"   Period: {args.start_date} to {args.end_date}")
    print(f"   Timeframe: {args.timeframe}")
    print(f"   Output: {args.output}")

    # Find data file
    data_root = PROJECT_ROOT / "data"
    data_file = find_data_file(pair, args.timeframe, data_root)

    if data_file is None:
        print(f"❌ No data file found for {pair} in timeframe {args.timeframe}")
        sys.exit(1)

    print(f"   Data file: {data_file.relative_to(PROJECT_ROOT)}")

    try:
        # Load data
        print("\n⚙️  Loading data...")
        df = load_pair_csv(pair, data_file.parent)

        if df.empty:
            print("❌ Data file is empty")
            sys.exit(1)

        # Filter by date range if specified
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            if args.start_date:
                start_dt = pd.to_datetime(args.start_date)
                df = df[df["date"] >= start_dt]
            if args.end_date:
                end_dt = pd.to_datetime(args.end_date)
                df = df[df["date"] <= end_dt]

        if df.empty:
            print("❌ No data in specified date range")
            sys.exit(1)

        print(f"   Loaded {len(df):,} bars")

        # Create config and apply indicators
        print("⚙️  Computing indicators...")
        config = create_indicator_config(pair, args.c1, args.timeframe, args.baseline)

        # Apply indicators
        df = calculate_atr(df)
        df = apply_indicators_with_cache(df, pair, config)

        # Analyze signals
        print("📊 Analyzing signals...")
        stats = analyze_indicator_signals(df)

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write outputs
        summary_file = output_dir / "indicator_summary.txt"
        series_file = output_dir / "indicator_series.csv"

        write_indicator_summary(stats, summary_file)
        write_indicator_series(df, series_file)

        print("\n📁 Output files:")
        print(f"   ✅ {summary_file}")
        print(f"   ✅ {series_file}")

        # Print console summary
        print("\n📈 ", end="")
        print_console_summary(stats)

        print("✅ Indicator dump completed!")

    except Exception as e:
        print(f"❌ Error analyzing indicators: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
