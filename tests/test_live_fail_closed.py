"""Test fail-closed: missing market folder/files -> HOLD output."""

from pathlib import Path

import pytest

from live.run_daily import run_daily


def test_missing_market_folder_hold_output(tmp_path):
    out_dir = tmp_path / "live_out"
    market_dir = tmp_path / "nonexistent_market"
    market_dir.mkdir(parents=True)
    for f in market_dir.iterdir():
        f.unlink()
    market_dir.rmdir()
    positions_path = tmp_path / "positions.csv"
    positions_path.write_text("ticket,symbol,type,lots,open_time,open_price,sl,tp\n")
    history_path = tmp_path / "history.csv"
    history_path.write_text(
        "ticket,symbol,type,open_time,close_time,open_price,close_price,profit\n"
    )
    config_path = Path("configs/v1_system.yaml")
    if not config_path.exists():
        pytest.skip("configs/v1_system.yaml not found")
    run_daily(
        market_dir=tmp_path / "nonexistent_market",
        positions_path=positions_path,
        history_path=history_path,
        out_dir=out_dir,
        config_path=config_path,
    )
    summary_path = out_dir / "daily_summary.txt"
    assert summary_path.exists()
    text = summary_path.read_text()
    assert "HOLD" in text
    assert "no_market_data" in text or "timestamp" in text


def test_empty_market_folder_hold_output(tmp_path):
    market_dir = tmp_path / "market"
    market_dir.mkdir(parents=True)
    out_dir = tmp_path / "live_out"
    positions_path = tmp_path / "positions.csv"
    positions_path.write_text("ticket,symbol,type,lots,open_time,open_price,sl,tp\n")
    history_path = tmp_path / "history.csv"
    history_path.write_text(
        "ticket,symbol,type,open_time,close_time,open_price,close_price,profit\n"
    )
    config_path = Path("configs/v1_system.yaml")
    if not config_path.exists():
        pytest.skip("configs/v1_system.yaml not found")
    run_daily(
        market_dir=market_dir,
        positions_path=positions_path,
        history_path=history_path,
        out_dir=out_dir,
        config_path=config_path,
    )
    summary_path = out_dir / "daily_summary.txt"
    assert summary_path.exists()
    orders_path = out_dir / "orders.csv"
    assert orders_path.exists()
    actions_path = out_dir / "actions.csv"
    assert actions_path.exists()
