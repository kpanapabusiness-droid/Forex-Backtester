from __future__ import annotations

import os
from pathlib import Path

import yaml

from core.backtester import run_backtest


def _write_minimal_config(path: Path, outputs_dir: Path) -> None:
    """
    Write a minimal backtest config that uses a single pair and points
    outputs.dir at the given directory.
    """
    cfg = f"""
strategy_version: "forex_backtester_v1.9.7"

pairs:
  - "EUR_USD"

timeframe: "D"

indicators:
  c1: "c1_fisher_transform"
  use_c2: false
  use_baseline: false
  use_volume: false
  use_exit: false

rules:
  one_candle_rule: false
  pullback_rule: false
  bridge_too_far_days: 7
  allow_baseline_as_catalyst: false

exit:
  use_trailing_stop: true
  move_to_breakeven_after_atr: true
  exit_on_c1_reversal: true
  exit_on_baseline_cross: false
  exit_on_exit_signal: false

tracking:
  in_sim_equity: false

risk:
  starting_balance: 10000.0
  risk_per_trade: 0.005

cache:
  enabled: false

outputs:
  dir: "{outputs_dir.as_posix()}"
  write_trades_csv: true

date_range:
  start: "2019-01-01"
  end: "2019-03-01"
"""
    path.write_text(cfg, encoding="utf-8")


def _load_yaml_config(path: Path) -> dict:
    """Load a YAML config file into a dict for run_backtest."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_yaml_outputs_dir_is_used(tmp_path: Path) -> None:
    """
    If YAML defines outputs.dir and CLI does NOT override, trades.csv
    must be written to outputs.dir and not to the default ./results.
    """
    cfg_path = tmp_path / "config.yaml"
    yaml_out = tmp_path / "out_yaml"
    _write_minimal_config(cfg_path, yaml_out)

    # Run in isolated CWD so any default 'results' folder lives under tmp_path.
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        cfg = _load_yaml_config(cfg_path)
        run_backtest(config_path=cfg)
    finally:
        os.chdir(old_cwd)

    trades_yaml = yaml_out / "trades.csv"
    assert trades_yaml.exists(), f"Expected trades.csv in YAML outputs.dir: {trades_yaml}"

    default_results = tmp_path / "results" / "trades.csv"
    assert not default_results.exists(), "Default ./results/trades.csv should not be created"


def test_cli_results_dir_overrides_yaml(tmp_path: Path) -> None:
    """
    When CLI/runner provides results_dir, it must win over YAML outputs.dir.
    """
    cfg_path = tmp_path / "config.yaml"
    yaml_out = tmp_path / "out_yaml"
    cli_out = tmp_path / "out_cli"
    _write_minimal_config(cfg_path, yaml_out)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        cfg = _load_yaml_config(cfg_path)
        run_backtest(config_path=cfg, results_dir=str(cli_out))
    finally:
        os.chdir(old_cwd)

    trades_cli = cli_out / "trades.csv"
    assert trades_cli.exists(), f"Expected trades.csv in CLI results_dir: {trades_cli}"

    trades_yaml = yaml_out / "trades.csv"
    assert not trades_yaml.exists(), "YAML outputs.dir must be ignored when CLI results_dir is set"

