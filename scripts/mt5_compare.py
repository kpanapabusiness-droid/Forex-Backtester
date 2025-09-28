#!/usr/bin/env python3
"""
MT5 Comparison Tool

Compares our backtest results with MT5 backtest results to validate parity.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_our_trades(results_dir: Path) -> pd.DataFrame:
    """Load our trades CSV and normalize to unified schema."""
    trades_file = results_dir / "trades.csv"
    if not trades_file.exists():
        raise FileNotFoundError(f"Our trades file not found: {trades_file}")

    df = pd.read_csv(trades_file)

    # Normalize to unified schema
    unified = pd.DataFrame()
    unified["open_time"] = pd.to_datetime(df["entry_date"])
    unified["close_time"] = pd.to_datetime(df["exit_date"])
    unified["symbol"] = df["pair"]
    unified["side"] = df["direction_int"]  # Use direction_int which should be +1/-1
    unified["open_price"] = df["entry_price"]
    unified["close_price"] = df["exit_price"]
    unified["sl"] = df.get("sl_at_entry_price", np.nan)
    unified["tp"] = df.get("tp1_at_entry_price", np.nan)
    unified["pnl_pips"] = df["pnl"] / 10  # Convert to pips (assuming 4-digit pairs)
    unified["pnl_currency"] = df["pnl"]
    unified["tag"] = df.get("exit_reason", "unknown")

    return unified


def load_mt5_trades(mt5_dir: Path) -> pd.DataFrame:
    """Load MT5 trades CSV and normalize to unified schema."""
    trades_file = mt5_dir / "eurusd_d1_2022_2024_trades.csv"
    report_file = mt5_dir / "report.html"

    if not trades_file.exists():
        if report_file.exists():
            raise FileNotFoundError(
                f"MT5 trades CSV not found: {trades_file}\n"
                f"Found HTML report: {report_file}\n"
                f"Please run: python tools/mt5_html_to_csv.py {report_file} {mt5_dir}"
            )
        else:
            raise FileNotFoundError(f"MT5 trades file not found: {trades_file}")

    df = pd.read_csv(trades_file)

    # Normalize to unified schema
    unified = pd.DataFrame()
    unified["open_time"] = pd.to_datetime(df["time"])
    # MT5 might not have close_time in the same format, use open_time as fallback
    unified["close_time"] = pd.to_datetime(df.get("close_time", df["time"]))
    unified["symbol"] = df["symbol"]

    # Convert MT5 type (buy/sell) to side (+1/-1)
    type_map = {"buy": 1, "sell": -1, "Buy": 1, "Sell": -1}
    unified["side"] = df["type"].map(type_map)

    unified["open_price"] = pd.to_numeric(df["price_open"], errors="coerce")
    unified["close_price"] = pd.to_numeric(df["price_close"], errors="coerce")
    unified["sl"] = pd.to_numeric(df["sl"], errors="coerce")
    unified["tp"] = pd.to_numeric(df["tp"], errors="coerce")

    # Calculate pnl_pips from price difference
    price_diff = unified["close_price"] - unified["open_price"]
    unified["pnl_pips"] = price_diff * unified["side"] * 10000  # Assuming 4-digit pairs

    unified["pnl_currency"] = pd.to_numeric(df["profit"], errors="coerce")
    unified["tag"] = "mt5"

    return unified


def match_trades(
    our_trades: pd.DataFrame, mt5_trades: pd.DataFrame, price_tol: float, time_tol_bars: int
) -> Tuple[List[Tuple], List[int], List[int]]:
    """
    Match trades between our results and MT5 results.

    Returns:
        matches: List of (our_idx, mt5_idx) tuples
        unmatched_ours: List of our trade indices that couldn't be matched
        unmatched_mt5: List of MT5 trade indices that couldn't be matched
    """
    matches = []
    matched_ours = set()
    matched_mt5 = set()

    # Convert time tolerance from bars to timedelta (assuming D1 = 1 day per bar)
    time_tol = pd.Timedelta(days=time_tol_bars)

    for our_idx, our_trade in our_trades.iterrows():
        best_match = None
        best_score = float("inf")

        for mt5_idx, mt5_trade in mt5_trades.iterrows():
            if mt5_idx in matched_mt5:
                continue

            # Check side match
            if our_trade["side"] != mt5_trade["side"]:
                continue

            # Check time proximity
            time_diff = abs(our_trade["open_time"] - mt5_trade["open_time"])
            if time_diff > time_tol:
                continue

            # Check price proximity
            price_diff = abs(our_trade["open_price"] - mt5_trade["open_price"])
            if price_diff > price_tol:
                continue

            # Calculate match score (lower is better)
            score = time_diff.total_seconds() + price_diff * 1000000
            if score < best_score:
                best_score = score
                best_match = mt5_idx

        if best_match is not None:
            matches.append((our_idx, best_match))
            matched_ours.add(our_idx)
            matched_mt5.add(best_match)

    unmatched_ours = [i for i in our_trades.index if i not in matched_ours]
    unmatched_mt5 = [i for i in mt5_trades.index if i not in matched_mt5]

    return matches, unmatched_ours, unmatched_mt5


def compare_totals(our_trades: pd.DataFrame, mt5_trades: pd.DataFrame, pnl_pct_tol: float) -> Dict:
    """Compare aggregate statistics between our trades and MT5 trades."""
    our_stats = {
        "total_trades": len(our_trades),
        "wins": len(our_trades[our_trades["pnl_currency"] > 0]),
        "losses": len(our_trades[our_trades["pnl_currency"] < 0]),
        "scratches": len(our_trades[our_trades["pnl_currency"] == 0]),
        "gross_pnl": our_trades["pnl_currency"].sum(),
        "net_pnl": our_trades["pnl_currency"].sum(),  # Assuming no commissions in parity test
    }

    mt5_stats = {
        "total_trades": len(mt5_trades),
        "wins": len(mt5_trades[mt5_trades["pnl_currency"] > 0]),
        "losses": len(mt5_trades[mt5_trades["pnl_currency"] < 0]),
        "scratches": len(mt5_trades[mt5_trades["pnl_currency"] == 0]),
        "gross_pnl": mt5_trades["pnl_currency"].sum(),
        "net_pnl": mt5_trades["pnl_currency"].sum(),
    }

    # Check tolerances
    results = {"our_stats": our_stats, "mt5_stats": mt5_stats, "passed": True, "errors": []}

    # Trade count must be identical
    if our_stats["total_trades"] != mt5_stats["total_trades"]:
        results["passed"] = False
        results["errors"].append(
            f"Trade count mismatch: ours={our_stats['total_trades']}, MT5={mt5_stats['total_trades']}"
        )

    # Check PnL within tolerance
    for key in ["gross_pnl", "net_pnl"]:
        our_val = our_stats[key]
        mt5_val = mt5_stats[key]
        if mt5_val != 0:
            pct_diff = abs(our_val - mt5_val) / abs(mt5_val)
            if pct_diff > pnl_pct_tol:
                results["passed"] = False
                results["errors"].append(
                    f"{key} difference too large: {pct_diff:.4f} > {pnl_pct_tol:.4f} "
                    f"(ours={our_val:.2f}, MT5={mt5_val:.2f})"
                )

    return results


def print_comparison_results(
    matches: List[Tuple], unmatched_ours: List[int], unmatched_mt5: List[int], totals_result: Dict
):
    """Print detailed comparison results."""
    print("\n📊 Trade Matching Results:")
    print(f"   Matched trades: {len(matches)}")
    print(f"   Unmatched ours: {len(unmatched_ours)}")
    print(f"   Unmatched MT5: {len(unmatched_mt5)}")

    print("\n📈 Aggregate Statistics:")
    our_stats = totals_result["our_stats"]
    mt5_stats = totals_result["mt5_stats"]

    print(f"   Total trades: ours={our_stats['total_trades']}, MT5={mt5_stats['total_trades']}")
    print(f"   Wins: ours={our_stats['wins']}, MT5={mt5_stats['wins']}")
    print(f"   Losses: ours={our_stats['losses']}, MT5={mt5_stats['losses']}")
    print(f"   Scratches: ours={our_stats['scratches']}, MT5={mt5_stats['scratches']}")
    print(f"   Gross PnL: ours={our_stats['gross_pnl']:.2f}, MT5={mt5_stats['gross_pnl']:.2f}")
    print(f"   Net PnL: ours={our_stats['net_pnl']:.2f}, MT5={mt5_stats['net_pnl']:.2f}")

    if totals_result["passed"]:
        print("\n✅ All comparisons passed!")
    else:
        print("\n❌ Comparison failed:")
        for error in totals_result["errors"]:
            print(f"   • {error}")


def main():
    parser = argparse.ArgumentParser(description="Compare our backtest results with MT5 results")
    parser.add_argument(
        "--ours",
        default="results/validation/mt5_parity_d1",
        help="Our results directory (default: results/validation/mt5_parity_d1)",
    )
    parser.add_argument(
        "--mt5",
        default="mt5/eurusd_d1_2022_2024",
        help="MT5 results directory (default: mt5/eurusd_d1_2022_2024)",
    )
    parser.add_argument(
        "--price-tol",
        type=float,
        default=0.00005,
        help="Price tolerance for matching trades (default: 0.00005)",
    )
    parser.add_argument(
        "--pnl-pct-tol", type=float, default=0.001, help="PnL percentage tolerance (default: 0.001)"
    )
    parser.add_argument(
        "--time-tol-bars",
        type=int,
        default=1,
        help="Time tolerance in bars for matching trades (default: 1)",
    )

    args = parser.parse_args()

    our_dir = Path(args.ours)
    mt5_dir = Path(args.mt5)

    print("🔍 MT5 Parity Comparison")
    print(f"   Our results: {our_dir}")
    print(f"   MT5 results: {mt5_dir}")
    print(f"   Price tolerance: {args.price_tol}")
    print(f"   PnL tolerance: {args.pnl_pct_tol}")
    print(f"   Time tolerance: {args.time_tol_bars} bars")

    try:
        # Load trades
        print("\n📂 Loading trade data...")
        our_trades = load_our_trades(our_dir)
        mt5_trades = load_mt5_trades(mt5_dir)

        print(f"   Our trades: {len(our_trades)}")
        print(f"   MT5 trades: {len(mt5_trades)}")

        # Match trades
        print("\n🔗 Matching trades...")
        matches, unmatched_ours, unmatched_mt5 = match_trades(
            our_trades, mt5_trades, args.price_tol, args.time_tol_bars
        )

        # Compare totals
        print("\n📊 Comparing totals...")
        totals_result = compare_totals(our_trades, mt5_trades, args.pnl_pct_tol)

        # Print results
        print_comparison_results(matches, unmatched_ours, unmatched_mt5, totals_result)

        # Exit with appropriate code
        if totals_result["passed"] and not unmatched_ours and not unmatched_mt5:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
