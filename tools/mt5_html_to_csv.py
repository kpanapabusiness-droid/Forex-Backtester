#!/usr/bin/env python3
"""
MT5 HTML to CSV Converter

Converts an MT5 HTML backtest report to a trades CSV format.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def convert_mt5_html_to_csv(input_html: str, output_dir: str) -> None:
    """
    Convert MT5 HTML backtest report to CSV format.

    Args:
        input_html: Path to MT5 HTML report
        output_dir: Directory to write output CSV
    """
    input_path = Path(input_html)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input HTML file not found: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    # Read all tables from HTML
    try:
        tables = pd.read_html(str(input_path), flavor="lxml")
    except Exception as e:
        raise ValueError(f"Failed to parse HTML tables: {e}")

    if not tables:
        raise ValueError("No tables found in HTML file")

    # Find trades table - heuristic: contains "time" and "profit" columns
    trades_table = None
    for table in tables:
        columns_lower = [str(col).lower() for col in table.columns]
        if "time" in columns_lower and "profit" in columns_lower:
            trades_table = table
            break

    # Fallback: use largest table
    if trades_table is None:
        trades_table = max(tables, key=len)
        print("Warning: Could not identify trades table by column names, using largest table")

    # Normalize column names to minimal schema
    normalized_columns = {}
    for col in trades_table.columns:
        col_lower = str(col).lower()
        if "time" in col_lower:
            if "open" in col_lower or col_lower == "time":
                normalized_columns[col] = "time"
        elif "type" in col_lower:
            normalized_columns[col] = "type"
        elif "symbol" in col_lower:
            normalized_columns[col] = "symbol"
        elif "price" in col_lower:
            if "open" in col_lower:
                normalized_columns[col] = "price_open"
            elif "close" in col_lower:
                normalized_columns[col] = "price_close"
        elif "s/l" in col_lower or "sl" in col_lower:
            normalized_columns[col] = "sl"
        elif "t/p" in col_lower or "tp" in col_lower:
            normalized_columns[col] = "tp"
        elif "profit" in col_lower:
            normalized_columns[col] = "profit"
        elif "volume" in col_lower or "lots" in col_lower:
            normalized_columns[col] = "volume"
        elif "ticket" in col_lower or "order" in col_lower:
            normalized_columns[col] = "ticket"

    # Rename columns
    trades_df = trades_table.rename(columns=normalized_columns)

    # Ensure we have the minimal required columns
    required_cols = [
        "time",
        "type",
        "symbol",
        "price_open",
        "price_close",
        "sl",
        "tp",
        "profit",
        "volume",
        "ticket",
    ]

    for col in required_cols:
        if col not in trades_df.columns:
            trades_df[col] = None

    # Select only the normalized columns
    trades_df = trades_df[required_cols]

    # Write to CSV
    output_file = output_path / "eurusd_d1_2022_2024_trades.csv"
    trades_df.to_csv(output_file, index=False)

    print(f"Converted MT5 HTML report to CSV: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert MT5 HTML backtest report to CSV")
    parser.add_argument("input_html", help="Path to MT5 HTML report file")
    parser.add_argument("output_dir", help="Directory to write output CSV")

    args = parser.parse_args()

    try:
        convert_mt5_html_to_csv(args.input_html, args.output_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
