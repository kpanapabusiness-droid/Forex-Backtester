"""
Config must be authoritative: -c PATH loads exactly that file; no fallback.
Changing config (pairs, date_range, etc.) must change the run (different output).
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from core.backtester import load_config, run_backtest


def _minimal_phase1_base_config() -> dict:
    """Minimal Phase 1-style config dict used across tests (no repo file dependency)."""
    return {
        "pairs": ["EUR_USD"],
        "timeframe": "D",
        "indicators": {
            "c1": "fisher",
            "use_c2": False,
            "use_baseline": False,
            "use_volume": False,
            "use_exit": False,
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
        "tracking": {
            "track_win_loss_scratch": True,
            "track_roi": True,
            "track_drawdown": True,
            "in_sim_equity": True,
            "verbose_logs": False,
        },
        "spreads": {
            "enabled": True,
            "default_pips": 0.5,
            "per_pair": {},
            "mode": "fixed",
            "atr_mult": 0.0,
        },
        "risk": {
            "starting_balance": 10_000.0,
            "risk_per_trade": 0.005,
        },
        "entry": {
            "sl_atr": 1.5,
            "tp1_atr": 1.0,
            "trail_after_atr": 2.0,
            "ts_atr": 1.5,
        },
        "execution": {
            "intrabar_priority": "tp_first",
        },
        "date_range": {
            "start": "2019-01-01",
            "end": "2026-01-01",
        },
    }


def test_load_config_uses_given_path_no_fallback(tmp_path):
    """Given an explicit path, load_config must load exactly that file (no fallback)."""

    cfg_path_eur = tmp_path / "custom_eur.yaml"
    cfg_path_jpy = tmp_path / "custom_jpy.yaml"

    cfg_eur = _minimal_phase1_base_config()
    cfg_eur["pairs"] = ["EUR_USD"]
    with cfg_path_eur.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_eur, f, sort_keys=False)

    cfg_jpy = _minimal_phase1_base_config()
    cfg_jpy["pairs"] = ["USD_JPY"]
    with cfg_path_jpy.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_jpy, f, sort_keys=False)

    loaded_eur = load_config(str(cfg_path_eur))
    loaded_jpy = load_config(str(cfg_path_jpy))

    assert loaded_eur.get("pairs") == ["EUR_USD"]
    assert loaded_jpy.get("pairs") == ["USD_JPY"]
    # If load_config ignored the given path and fell back to a default,
    # these would be identical; asserting they differ guards against that.
    assert loaded_eur.get("pairs") != loaded_jpy.get("pairs")


def test_load_config_raises_when_path_missing():
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config("nonexistent_config_xyz.yaml")


def test_different_pairs_produce_different_output(tmp_path):
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "daily"
    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        pytest.skip("data/daily missing")

    base = _minimal_phase1_base_config()
    base["data_dir"] = str(data_dir)
    base["date_range"] = {"start": "2020-01-01", "end": "2020-03-31"}

    one_pair = dict(base)
    one_pair["pairs"] = ["EUR_USD"]
    three_pairs = dict(base)
    three_pairs["pairs"] = ["EUR_USD", "USD_JPY", "GBP_USD"]

    cfg_one = tmp_path / "one.yaml"
    cfg_three = tmp_path / "three.yaml"
    with open(cfg_one, "w", encoding="utf-8") as f:
        yaml.safe_dump(one_pair, f, sort_keys=False)
    with open(cfg_three, "w", encoding="utf-8") as f:
        yaml.safe_dump(three_pairs, f, sort_keys=False)

    out_one = tmp_path / "out_one"
    out_three = tmp_path / "out_three"
    out_one.mkdir()
    out_three.mkdir()

    run_backtest(config_path=str(cfg_one), results_dir=str(out_one))
    run_backtest(config_path=str(cfg_three), results_dir=str(out_three))

    t1 = out_one / "trades.csv"
    t3 = out_three / "trades.csv"
    assert t1.exists() and t3.exists()
    df1 = pd.read_csv(t1)
    df3 = pd.read_csv(t3)
    if len(df1) == 0 and len(df3) == 0:
        pytest.skip("No trades generated for either config; cannot compare outputs for pairs.")
    different = len(df1) != len(df3) or (df1.shape != df3.shape or not df1.equals(df3))
    assert different, "Different pairs must produce different trade count or output"


def test_cli_c_loads_that_file(tmp_path):
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "daily"
    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        pytest.skip("data/daily missing")

    cfg = _minimal_phase1_base_config()
    cfg["pairs"] = ["EUR_USD"]
    cfg["data_dir"] = str(data_dir)
    cfg["date_range"] = {"start": "2020-01-01", "end": "2020-02-28"}
    cfg_path = tmp_path / "cli_config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    result = subprocess.run(
        [sys.executable, "-m", "core.backtester", "-c", str(cfg_path), "--results-dir", str(out_dir)],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
    trades_path = out_dir / "trades.csv"
    assert trades_path.exists()
    df = pd.read_csv(trades_path)
    if "pair" in df.columns and len(df) > 0:
        assert set(df["pair"].unique()).issubset({"EUR_USD"}), "Loaded config pairs should match (only EUR_USD)"
    loaded = load_config(str(cfg_path))
    assert loaded.get("pairs") == ["EUR_USD"]
