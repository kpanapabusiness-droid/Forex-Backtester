#!/usr/bin/env python3
"""
MT5 Parity Runner

Runs a single backtest using the MT5 parity configuration to generate
results for comparison with MT5 backtest reports.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from core.backtester import run_backtest  # noqa: E402


def main():
    """Run MT5 parity backtest with fixed configuration."""
    config_path = "configs/validation/mt5_parity_d1.yaml"

    print("🚀 Running MT5 parity backtest...")
    print(f"   Config: {config_path}")

    try:
        # Run backtest - results_dir will be taken from config outputs.dir
        run_backtest(config_path)
        print("✅ MT5 parity backtest completed!")

        # Verify output files
        results_dir = Path("results/validation/mt5_parity_d1")
        trades_file = results_dir / "trades.csv"
        equity_file = results_dir / "equity_curve.csv"

        print("\n📁 Output files:")
        for file_path in [trades_file, equity_file]:
            status = "✅" if file_path.exists() else "❌"
            print(f"   {status} {file_path}")

    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
