#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_coverage_report.py — Data availability analysis for C1 sweep triage
------------------------------------------------------------------------
Purpose: Analyze data coverage across currency pairs to identify gaps
that might explain why C1 sweeps produce 0 trades.

Usage:
    python scripts/data_coverage_report.py --pairs "EURUSD,GBPUSD,AUDUSD" --timeframe H1 --from 2020-01-01 --to 2021-12-31

Features:
- Detects earliest and latest candle for each pair
- Counts total bars available
- Identifies data gaps or missing files
- Outputs CSV report with coverage statistics
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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


def analyze_data_coverage(
    data_file: Path, start_date: Optional[str] = None, end_date: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze data coverage for a single data file."""

    try:
        # Read CSV file
        df = pd.read_csv(data_file)

        if df.empty:
            return {
                "status": "empty",
                "earliest_ts": None,
                "latest_ts": None,
                "bar_count": 0,
                "filtered_bar_count": 0,
                "missing_ohlc_cols": [],
                "error": "File is empty",
            }

        # Detect timestamp column (common variations)
        timestamp_cols = ["timestamp", "date", "datetime", "time", "Date", "Timestamp"]
        ts_col = None

        for col in timestamp_cols:
            if col in df.columns:
                ts_col = col
                break

        if ts_col is None:
            # Try first column if no standard timestamp column found
            ts_col = df.columns[0]

        # Convert to datetime
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

        # Remove rows with invalid timestamps
        df = df.dropna(subset=[ts_col])

        if df.empty:
            return {
                "status": "no_valid_timestamps",
                "earliest_ts": None,
                "latest_ts": None,
                "bar_count": 0,
                "filtered_bar_count": 0,
                "missing_ohlc_cols": [],
                "error": "No valid timestamps found",
            }

        # Sort by timestamp
        df = df.sort_values(ts_col)

        # Get overall coverage
        earliest_ts = df[ts_col].min()
        latest_ts = df[ts_col].max()
        total_bars = len(df)

        # Filter by date range if specified
        filtered_df = df.copy()
        if start_date:
            start_dt = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df[ts_col] >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df[ts_col] <= end_dt]

        filtered_bar_count = len(filtered_df)

        # Check for required OHLC columns
        required_cols = ["open", "high", "low", "close"]
        missing_cols = []
        for col in required_cols:
            # Check common variations
            col_variations = [col, col.capitalize(), col.upper()]
            found = any(var in df.columns for var in col_variations)
            if not found:
                missing_cols.append(col)

        return {
            "status": "success",
            "earliest_ts": earliest_ts.strftime("%Y-%m-%d %H:%M:%S")
            if pd.notna(earliest_ts)
            else None,
            "latest_ts": latest_ts.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(latest_ts) else None,
            "bar_count": total_bars,
            "filtered_bar_count": filtered_bar_count,
            "missing_ohlc_cols": missing_cols,
            "error": None,
        }

    except Exception as e:
        return {
            "status": "error",
            "earliest_ts": None,
            "latest_ts": None,
            "bar_count": 0,
            "filtered_bar_count": 0,
            "missing_ohlc_cols": [],
            "error": str(e),
        }


def generate_coverage_report(
    pairs: List[str],
    timeframe: str,
    start_date: Optional[str],
    end_date: Optional[str],
    data_root: Path,
    output_file: Path,
) -> None:
    """Generate data coverage report for all pairs."""

    print(f"🔍 Analyzing data coverage for {len(pairs)} pairs...")
    print(f"   Timeframe: {timeframe}")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Data root: {data_root}")

    results = []

    for i, pair in enumerate(pairs, 1):
        normalized_pair = normalize_pair_format(pair)
        print(f"   [{i:2d}/{len(pairs)}] {normalized_pair}...", end=" ")

        # Find data file
        data_file = find_data_file(normalized_pair, timeframe, data_root)

        if data_file is None:
            print("❌ No data file found")
            results.append(
                {
                    "pair": normalized_pair,
                    "data_file": "NOT_FOUND",
                    "status": "missing_file",
                    "earliest_ts": None,
                    "latest_ts": None,
                    "bar_count": 0,
                    "filtered_bar_count": 0,
                    "missing_ohlc_cols": [],
                    "error": "Data file not found",
                }
            )
            continue

        # Analyze coverage
        coverage = analyze_data_coverage(data_file, start_date, end_date)

        result = {
            "pair": normalized_pair,
            "data_file": str(data_file.relative_to(PROJECT_ROOT)),
            **coverage,
        }
        results.append(result)

        if coverage["status"] == "success":
            print(
                f"✅ {coverage['filtered_bar_count']} bars ({coverage['earliest_ts']} to {coverage['latest_ts']})"
            )
        else:
            print(f"❌ {coverage['error']}")

    # Write CSV report
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pair",
        "data_file",
        "status",
        "earliest_ts",
        "latest_ts",
        "bar_count",
        "filtered_bar_count",
        "missing_ohlc_cols",
        "error",
    ]

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n📊 Coverage report saved to: {output_file}")

    # Print summary statistics
    total_pairs = len(results)
    successful = sum(1 for r in results if r["status"] == "success")
    missing_files = sum(1 for r in results if r["status"] == "missing_file")
    errors = sum(1 for r in results if r["status"] == "error")
    empty_files = sum(1 for r in results if r["status"] == "empty")

    print("\n📈 Summary:")
    print(f"   Total pairs: {total_pairs}")
    print(f"   ✅ Successful: {successful}")
    print(f"   📁 Missing files: {missing_files}")
    print(f"   ❌ Errors: {errors}")
    print(f"   📄 Empty files: {empty_files}")

    if successful > 0:
        successful_results = [r for r in results if r["status"] == "success"]
        total_bars = sum(r["filtered_bar_count"] for r in successful_results)
        avg_bars = total_bars / successful if successful > 0 else 0
        print(f"   📊 Average bars per pair: {avg_bars:.0f}")

        # Find pairs with low data
        low_data_pairs = [r for r in successful_results if r["filtered_bar_count"] < 100]
        if low_data_pairs:
            print(f"   ⚠️  Pairs with <100 bars: {len(low_data_pairs)}")
            for r in low_data_pairs[:5]:  # Show first 5
                print(f"      {r['pair']}: {r['filtered_bar_count']} bars")


def main():
    parser = argparse.ArgumentParser(
        description="Generate data coverage report for forex pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/data_coverage_report.py --pairs "EURUSD,GBPUSD,AUDUSD" --timeframe H1 --from 2020-01-01 --to 2021-12-31
  python scripts/data_coverage_report.py --pairs "EURUSD,GBPUSD" --timeframe D --output results/coverage.csv
        """,
    )

    parser.add_argument("--pairs", required=True, help="Comma-separated list of currency pairs")
    parser.add_argument("--timeframe", default="H1", help="Timeframe (default: H1)")
    parser.add_argument("--from", dest="start_date", help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--to", dest="end_date", help="End date filter (YYYY-MM-DD)")
    parser.add_argument("--output", default="results/data_coverage.csv", help="Output CSV file")
    parser.add_argument("--data-root", help="Data root directory (default: data/)")

    args = parser.parse_args()

    # Parse pairs
    pairs = [pair.strip() for pair in args.pairs.split(",")]
    pairs = [pair for pair in pairs if pair]  # Remove empty strings

    if not pairs:
        print("❌ No valid pairs specified")
        sys.exit(1)

    # Set data root
    data_root = Path(args.data_root) if args.data_root else PROJECT_ROOT / "data"
    if not data_root.exists():
        print(f"❌ Data root directory not found: {data_root}")
        sys.exit(1)

    # Generate report
    try:
        generate_coverage_report(
            pairs=pairs,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            data_root=data_root,
            output_file=Path(args.output),
        )
        print("✅ Coverage report generation completed!")

    except Exception as e:
        print(f"❌ Error generating coverage report: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
