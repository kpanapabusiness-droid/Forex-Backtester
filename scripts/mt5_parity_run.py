#!/usr/bin/env python3
"""
MT5 Parity Runner

Runs a single backtest using the MT5 parity configuration to generate
results for comparison with MT5 backtest reports.
"""

import sys
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from core.backtester import run_backtest  # noqa: E402


def validate_mt5_parity_config(cfg):
    """Validate that config matches exact MT5 EA settings."""
    # Critical validations - must match MT5 EA exactly
    assert cfg["roles"]["c1"] == "sma_cross", f"Expected c1=sma_cross, got {cfg['roles']['c1']}"
    assert cfg["timeframe"] == "D1", f"Expected timeframe=D1, got {cfg['timeframe']}"
    assert cfg["symbol"] == "EURUSD", f"Expected symbol=EURUSD, got {cfg['symbol']}"
    assert cfg["c1"]["fast_period"] == 20, (
        f"Expected fast_period=20, got {cfg['c1']['fast_period']}"
    )
    assert cfg["c1"]["slow_period"] == 50, (
        f"Expected slow_period=50, got {cfg['c1']['slow_period']}"
    )
    assert cfg["spread"]["enabled"] is False, (
        f"Expected spread disabled, got {cfg['spread']['enabled']}"
    )
    assert cfg["commission"]["enabled"] is False, (
        f"Expected commission disabled, got {cfg['commission']['enabled']}"
    )
    assert cfg["slippage_pips"] in [0, 0.0], f"Expected slippage_pips=0, got {cfg['slippage_pips']}"

    # Engine validations for cross-only behavior
    engine = cfg.get("engine", {})
    assert engine.get("cross_only") is True, (
        f"Expected cross_only=true, got {engine.get('cross_only')}"
    )
    assert engine.get("reverse_on_signal") is True, (
        f"Expected reverse_on_signal=true, got {engine.get('reverse_on_signal')}"
    )
    assert engine.get("allow_pyramiding") is False, (
        f"Expected allow_pyramiding=false, got {engine.get('allow_pyramiding')}"
    )


def main():
    """Run MT5 parity backtest with fixed configuration."""
    config_path = "configs/validation/mt5_parity_d1.yaml"

    print("🚀 Running MT5 parity backtest...")
    print(f"   Config: {config_path}")

    try:
        # Load and validate config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Validate critical settings
        validate_mt5_parity_config(cfg)

        # Print effective config summary
        engine = cfg.get("engine", {})
        print(
            f"[MT5 PARITY] {cfg['symbol']} {cfg['timeframe']} {cfg['date_from']}→{cfg['date_to']} c1={cfg['roles']['c1']}({cfg['c1']['fast_period']},{cfg['c1']['slow_period']}) cross_only={int(engine.get('cross_only', False))} next_open_fills={int(engine.get('fill_on_next_bar_open', False))} reverse={int(engine.get('reverse_on_signal', False))} pyramiding={int(engine.get('allow_pyramiding', True))} costs=0"
        )

        # Run backtest - results_dir will be taken from config outputs.dir
        run_backtest(cfg)
        print("✅ MT5 parity backtest completed!")

        # Verify output files
        results_dir = Path(cfg["outputs"]["dir"])
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
