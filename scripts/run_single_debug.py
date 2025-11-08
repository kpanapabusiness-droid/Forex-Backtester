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

import pandas as pd  # noqa: E402

from core.backtester import run_backtest  # noqa: E402
from core.utils import calculate_atr  # noqa: E402
from indicators_cache import apply_indicators_with_cache  # noqa: E402


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

        print("\nTrades Summary:")
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
                print("\nOutcomes:")
                for outcome, count in outcome_counts.items():
                    print(f"   {outcome}: {count}")
        else:
            print("   No trades found!")

    except Exception as e:
        print(f"   Error reading trades: {e}")


def generate_entry_signals(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Generate entry signals from C1 and baseline indicators following NNFX rules."""
    result = df.copy()

    # Initialize entry signal column
    result["entry_signal"] = 0

    # Get C1 signal column
    c1_col = "c1_signal"
    if c1_col not in result.columns:
        return result

    # Get baseline info
    use_baseline = cfg.get("indicators", {}).get("use_baseline", False)
    baseline_col = "baseline"  # The price series

    # Simple entry logic: C1 signal must align with baseline trend
    for i in range(1, len(result)):  # Start from 1 to avoid lookback issues
        c1_signal = result.loc[i, c1_col]

        if c1_signal == 0:
            continue

        # Check baseline alignment if enabled
        entry_allowed = True
        if use_baseline and baseline_col in result.columns:
            price = result.loc[i, "close"]
            baseline_value = result.loc[i, baseline_col]

            # Long: price above baseline, Short: price below baseline
            if c1_signal == 1 and price <= baseline_value:
                entry_allowed = False
            elif c1_signal == -1 and price >= baseline_value:
                entry_allowed = False

        if entry_allowed:
            result.loc[i, "entry_signal"] = c1_signal

    return result


def create_decision_log(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Create per-bar decision log for debugging."""
    decisions = []

    # Get relevant columns
    c1_col = "c1_signal"
    baseline_col = "baseline"
    entry_col = "entry_signal"

    use_baseline = cfg.get("indicators", {}).get("use_baseline", False)

    for i in range(len(df)):
        row = df.iloc[i]
        ts = row.get("date", row.name)

        c1_signal = row.get(c1_col, 0)
        baseline_trend = "neutral"
        entry_signal = row.get(entry_col, 0)
        entry_allowed = False
        reason = "no_c1"

        # Determine baseline trend
        if use_baseline and baseline_col in df.columns:
            price = row.get("close", 0)
            baseline_value = row.get(baseline_col, 0)
            baseline_trend = (
                "bullish"
                if price > baseline_value
                else "bearish"
                if price < baseline_value
                else "neutral"
            )

        # Determine entry decision
        if c1_signal != 0:
            if not use_baseline:
                entry_allowed = True
                reason = "c1_signal"
            elif baseline_col in df.columns:
                price = row.get("close", 0)
                baseline_value = row.get(baseline_col, 0)

                if c1_signal == 1 and price > baseline_value:
                    entry_allowed = True
                    reason = "c1_bullish_aligned"
                elif c1_signal == -1 and price < baseline_value:
                    entry_allowed = True
                    reason = "c1_bearish_aligned"
                else:
                    entry_allowed = False
                    reason = "wrong_trend"
            else:
                entry_allowed = False
                reason = "no_baseline_data"

        decisions.append(
            {
                "ts": ts,
                "c1_signal": c1_signal,
                "baseline_trend": baseline_trend,
                "entry_signal": entry_signal,
                "entry_allowed": entry_allowed,
                "reason": reason,
            }
        )

    return pd.DataFrame(decisions)


def export_indicator_series(df: pd.DataFrame, output_file: Path) -> None:
    """Export indicator series to CSV."""
    # Select relevant columns for export
    export_cols = ["date"] if "date" in df.columns else []

    # Add OHLC if available
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            export_cols.append(col)

    # Add indicator columns
    for col in df.columns:
        if "signal" in col.lower() or col == "baseline" or col == "atr":
            export_cols.append(col)

    # Create export dataframe
    export_df = df[export_cols].copy()

    # Rename columns for clarity
    if "close" in export_df.columns:
        export_df = export_df.rename(columns={"close": "price"})

    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_file, index=False)


def run_backtest_with_debug(
    config: Dict[str, Any],
    output_dir: Path,
    debug_decisions: bool = False,
    export_indicator: bool = False,
):
    """Run backtest with optional debugging features."""
    from core.utils import load_pair_csv

    # Get pair and data directory
    pairs = config.get("pairs", [])
    if not pairs:
        raise ValueError("No pairs configured")

    pair = pairs[0]  # Single pair for debug
    data_dir = config.get("data_dir", "data/daily")

    # Load data
    data_file = Path(data_dir) / f"{pair}.csv"
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        # Run normal backtest which will handle the error
        run_backtest(config, results_dir=str(output_dir))
        return

    # Load and process data
    df = load_pair_csv(pair, Path(data_dir))
    if df.empty:
        print(f"Empty data file: {data_file}")
        run_backtest(config, results_dir=str(output_dir))
        return

    # Apply indicators
    df = calculate_atr(df)
    df = apply_indicators_with_cache(df, pair, config)

    # Generate entry signals (this is the missing piece!)
    df = generate_entry_signals(df, config)

    # Export indicator series if requested
    if export_indicator:
        export_indicator_series(df, output_dir / "indicator_series.csv")

    # Create decision log if requested
    if debug_decisions:
        decision_df = create_decision_log(df, config)
        decision_df.to_csv(output_dir / "decisions.csv", index=False)

    # Run normal backtest with the processed data
    # We'll modify the config to point to a temporary processed file
    temp_data_file = output_dir / f"processed_{pair}.csv"
    temp_data_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(temp_data_file, index=False)

    # Update config to use processed data
    temp_config = config.copy()
    temp_config["data_dir"] = str(output_dir)
    temp_config["pairs"] = [f"processed_{pair}"]

    # Run backtest
    run_backtest(temp_config, results_dir=str(output_dir))

    # Clean up temp file
    if temp_data_file.exists():
        temp_data_file.unlink()


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
        print(f"   Error reading summary: {e}")


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
    parser.add_argument(
        "--debug-decisions",
        action="store_true",
        help="Enable per-bar decision logging to decisions.csv",
    )
    parser.add_argument(
        "--export-indicator",
        action="store_true",
        help="Export indicator series to indicator_series.csv",
    )

    args = parser.parse_args()

    # Normalize pair format (add underscore if needed)
    pair = args.pair.upper()
    if len(pair) == 6 and "_" not in pair:
        pair = f"{pair[:3]}_{pair[3:]}"

    print("Running single debug backtest:")
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
        print("\nRunning backtest...")

        # Run backtest with debug features if enabled
        if args.debug_decisions or args.export_indicator:
            run_backtest_with_debug(
                config,
                output_dir,
                debug_decisions=args.debug_decisions,
                export_indicator=args.export_indicator,
            )
        else:
            # Run the backtest with results_dir parameter
            run_backtest(config, results_dir=str(output_dir))

        print("Backtest completed!")

        # Check output files
        trades_file = output_dir / "trades.csv"
        summary_file = output_dir / "summary.txt"
        equity_file = output_dir / "equity_curve.csv"
        decisions_file = output_dir / "decisions.csv"
        indicator_file = output_dir / "indicator_series.csv"

        print("\nOutput files:")
        standard_files = [trades_file, summary_file, equity_file, config_file]
        for file_path in standard_files:
            status = "OK" if file_path.exists() else "MISSING"
            print(f"   {status} {file_path}")

        # Check debug files if enabled
        if args.debug_decisions:
            status = "OK" if decisions_file.exists() else "MISSING"
            print(f"   {status} {decisions_file}")

        if args.export_indicator:
            status = "OK" if indicator_file.exists() else "MISSING"
            print(f"   {status} {indicator_file}")

        # Print summary stats
        if summary_file.exists():
            print("\nSummary Stats:")
            print_summary_stats(summary_file)

        # Print trades summary
        if trades_file.exists():
            print_trades_summary(trades_file)

        print(f"\nConfig used: {config_file}")

    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
