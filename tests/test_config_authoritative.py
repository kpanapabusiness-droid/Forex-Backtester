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


def test_load_config_uses_given_path_no_fallback(tmp_path):
    root = Path(__file__).resolve().parent.parent
    base_cfg_path = root / "configs" / "phase4_sanity_onepair.yaml"
    if not base_cfg_path.exists():
        base_cfg_path = root / "configs" / "v1_system.yaml"
    with open(base_cfg_path, encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict["pairs"] = ["EUR_USD"]
    cfg_path = tmp_path / "custom.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)
    cfg = load_config(str(cfg_path))
    assert cfg.get("pairs") == ["EUR_USD"]


def test_load_config_raises_when_path_missing():
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config("nonexistent_config_xyz.yaml")


def test_different_pairs_produce_different_output(tmp_path):
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "daily"
    base_cfg_path = root / "configs" / "v1_system.yaml"
    if not base_cfg_path.exists() or not data_dir.exists() or not list(data_dir.glob("*.csv")):
        pytest.skip("v1_system or data/daily missing")

    with open(base_cfg_path, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    base["data_dir"] = str(data_dir)
    if "data" in base and isinstance(base["data"], dict):
        base["data"]["dir"] = str(data_dir)
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
    different = len(df1) != len(df3) or (df1.shape != df3.shape or not df1.equals(df3))
    assert different, "Different pairs must produce different trade count or output"


def test_cli_c_loads_that_file(tmp_path):
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "daily"
    base_cfg_path = root / "configs" / "phase4_sanity_onepair.yaml"
    if not base_cfg_path.exists():
        base_cfg_path = root / "configs" / "v1_system.yaml"
    if not base_cfg_path.exists() or not data_dir.exists() or not list(data_dir.glob("*.csv")):
        pytest.skip("config or data/daily missing")

    with open(base_cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
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
