from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.backtester import run_backtest


def _run_with_config(tmp_path: Path, config_rel: str, slug: str) -> int:
    """
    Run the engine with a given config into an isolated results dir and
    return the number of trades produced.
    """
    results_dir = tmp_path / slug
    run_backtest(config_path=Path("configs") / config_rel, results_dir=str(results_dir))

    trades_path = results_dir / "trades.csv"
    assert trades_path.exists(), f"Expected trades.csv at {trades_path}"

    trades_df = pd.read_csv(trades_path)
    return len(trades_df)


def test_volume_pure_veto_trade_count(tmp_path: Path) -> None:
    """
    Volume is a pure veto:
    trades(volume_on) must never exceed trades(volume_off) on the same data/config.
    """
    trades_off = _run_with_config(tmp_path, "phase2_volume_gate_off.yaml", "phase2_off")
    trades_on = _run_with_config(tmp_path, "phase2_volume_gate_on.yaml", "phase2_on")

    assert trades_on <= trades_off, (
        f"Volume ON produced more trades than OFF: on={trades_on}, off={trades_off}"
    )

