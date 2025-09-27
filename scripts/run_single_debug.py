#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_single_debug.py — Single backtest debug runner for C1 sweep triage
----------------------------------------------------------------------
Purpose: Run a single backtest with one C1 indicator over a small window
to debug why C1-only sweeps produce 0 trades or -99% ROI.

Usage:
    python scripts/run_single_debug.py --pair EURUSD --c1 supertrend --from 2020-01-01 --to 2021-12-31

Features:
- Minimal valid strategy with chosen C1, default baseline (ema200), default exit (atr_trailing)
- Spreads OFF, DBCVIX OFF, fixed-fraction risk sizing
- Outputs trades.csv, equity_curve.csv, summary.txt, merged_config.yaml
- Prints summary stats and config path to stdout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.backtester import run_backtest  # noqa: E402


def create_minimal_config(
    pair: str,
    c1_indicator: str,
    start_date: str,
    end_date: str,
    timeframe: str = "H1",
    baseline: str = "baseline_ema",
    exit_indicator: str = "exit_twiggs_money_flow",
) -> Dict[str, Any]:
    """Create a minimal valid config for single debug run."""

    # Map timeframe to data directory
    data_dir_map = {
        "D": "data/daily",
        "H1": "data/hourly",  # Assume hourly data exists or falls back to daily
        "H4": "data/4h",
        "M15": "data/15m",
    }

    # Use daily data as fallback since that's what we know exists
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
            "exit": exit_indicator,
        },
        "rules": {
            "one_candle_rule": False,
            "pullback_rule": False,
            "bridge_too_far_days": 7,
            "allow_baseline_as_catalyst": False,
        },
        "exit": {
            "use_trailing_stop": True,
            "move_to_breakeven_after_atr": True,
            "exit_on_c1_reversal": True,
            "exit_on_baseline_cross": False,
            "exit_on_exit_signal": False,
        },
        "continuation": {
            "allow_continuation": False,
            "skip_volume_check": False,
            "skip_pullback_check": False,
            "block_if_crossed_baseline_since_entry": False,
        },
        "spreads": {"enabled": False, "default_pips": 0.0, "mode": "fixed", "atr_mult": 0.0},
        "tracking": {
            "in_sim_equity": True,
            "track_win_loss_scratch": True,
            "track_roi": True,
            "track_drawdown": True,
            "verbose_logs": False,
        },
        "filters": {
            "dbcvix": {
                "enabled": False,
                "mode": "reduce",
                "threshold": 0.0,
                "reduce_risk_to": 1.0,
                "source": "synthetic",
                "csv_path": "data/external/dbcvix_synth.csv",
                "column": "cvix_synth",
            }
        },
        "cache": {
            "enabled": True,
            "dir": "cache",
            "format": "parquet",
            "scope_key": None,
            "roles": None,
        },
        "validation": {"enabled": True, "fail_fast": True, "strict_contract": False},
        "output": {"results_dir": "results"},
        "risk": {"starting_balance": 10000.0, "risk_per_trade_pct": 2.0},
        "date_range": {"start": start_date, "end": end_date},
    }

    return config


def print_trades_summary(trades_file: Path) -> None:
    """Print first/last 3 rows of trades and summary stats."""
    try:
        import pandas as pd

        df = pd.read_csv(trades_file)

        print("\n📊 Trades Summary:")
        print(f"   Total trades: {len(df)}")

        if len(df) > 0:
            print("\n🔝 First 3 trades:")
            print(df.head(3).to_string(index=False))

            if len(df) > 3:
                print("\n🔚 Last 3 trades:")
                print(df.tail(3).to_string(index=False))

            # Calculate basic stats
            if "pnl_dollars" in df.columns:
                total_pnl = df["pnl_dollars"].sum()
                print(f"\n💰 Total PnL: ${total_pnl:.2f}")

            if "outcome" in df.columns:
                outcome_counts = df["outcome"].value_counts()
                print("\n📈 Outcomes:")
                for outcome, count in outcome_counts.items():
                    print(f"   {outcome}: {count}")
        else:
            print("   ⚠️  No trades found!")

    except Exception as e:
        print(f"   ❌ Error reading trades: {e}")


def print_summary_stats(summary_file: Path) -> None:
    """Parse and print key stats from summary.txt."""
    try:
        with open(summary_file, "r") as f:
            content = f.read()

        # Extract key metrics using simple string parsing
        lines = content.split("\n")
        for line in lines:
            if any(
                keyword in line
                for keyword in ["ROI (%)", "MAR", "Max DD (%)", "Win%", "Total Trades"]
            ):
                print(f"   {line.strip()}")

    except Exception as e:
        print(f"   ❌ Error reading summary: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run single backtest for C1 sweep debugging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_single_debug.py --pair EURUSD --c1 supertrend --from 2020-01-01 --to 2021-12-31
  python scripts/run_single_debug.py --pair GBPUSD --c1 c1_twiggs_money_flow --from 2019-01-01 --to 2020-12-31 --timeframe D
        """,
    )

    parser.add_argument("--pair", required=True, help="Currency pair (e.g., EURUSD)")
    parser.add_argument("--c1", required=True, help="C1 indicator name (e.g., supertrend)")
    parser.add_argument("--from", dest="start_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="end_date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", default="H1", help="Timeframe (default: H1)")
    parser.add_argument(
        "--output-dir",
        default="results/single_debug",
        help="Output directory (default: results/single_debug)",
    )
    parser.add_argument(
        "--baseline", default="baseline_ema", help="Baseline indicator (default: baseline_ema)"
    )
    parser.add_argument(
        "--exit", default="atr_trailing", help="Exit indicator (default: atr_trailing)"
    )

    args = parser.parse_args()

    # Normalize pair format (add underscore if needed)
    pair = args.pair.upper()
    if len(pair) == 6 and "_" not in pair:
        pair = f"{pair[:3]}_{pair[3:]}"

    print("🚀 Running single debug backtest:")
    print(f"   Pair: {pair}")
    print(f"   C1: {args.c1}")
    print(f"   Period: {args.start_date} to {args.end_date}")
    print(f"   Timeframe: {args.timeframe}")
    print(f"   Output: {args.output_dir}")

    # Create minimal config
    config = create_minimal_config(
        pair=pair,
        c1_indicator=args.c1,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
        baseline=args.baseline,
        exit_indicator=args.exit,
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save merged config
    config_file = output_dir / "merged_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    try:
        # Run backtest
        print("\n⚙️  Running backtest...")

        # Run the backtest with results_dir parameter
        run_backtest(config, results_dir=str(output_dir))

        print("✅ Backtest completed!")

        # Check output files
        trades_file = output_dir / "trades.csv"
        summary_file = output_dir / "summary.txt"
        equity_file = output_dir / "equity_curve.csv"

        print("\n📁 Output files:")
        for file_path in [trades_file, summary_file, equity_file, config_file]:
            status = "✅" if file_path.exists() else "❌"
            print(f"   {status} {file_path}")

        # Print summary stats
        if summary_file.exists():
            print("\n📊 Summary Stats:")
            print_summary_stats(summary_file)

        # Print trades summary
        if trades_file.exists():
            print_trades_summary(trades_file)

        print(f"\n🔧 Config used: {config_file}")

    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
