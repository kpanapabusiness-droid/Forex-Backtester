"""
Phase 4 regression: C2 must change cache identity and act as a true gate.
- Changing C2 (or use_c2) must change the computed params hash / cache key.
- Enabling C2 on a run must not increase trade count and must produce different
  trades.csv content vs baseline (C2 is a confirmation filter).
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from indicators_cache import cache_key_parts, compute_params_hash, params_for_cache_hash


def test_c2_changes_params_hash_and_cache_key():
    base = {"pair": "EUR_USD", "timeframe": "D", "data_hash": "a" * 16, "scope_key": None}
    params_empty = {}
    h_c2_metro = compute_params_hash(params_for_cache_hash("c2", "metro_advanced", params_empty))
    h_c2_schaff = compute_params_hash(
        params_for_cache_hash("c2", "schaff_trend_cycle", params_empty)
    )
    h_c1_metro = compute_params_hash(params_for_cache_hash("c1", "metro_advanced", params_empty))
    assert h_c2_metro != h_c2_schaff
    assert h_c2_metro != h_c1_metro
    _, key_c2_metro = cache_key_parts(
        base["pair"], base["timeframe"], "c2", "metro_advanced", h_c2_metro, base["data_hash"], None
    )
    _, key_c2_schaff = cache_key_parts(
        base["pair"],
        base["timeframe"],
        "c2",
        "schaff_trend_cycle",
        h_c2_schaff,
        base["data_hash"],
        None,
    )
    assert key_c2_metro != key_c2_schaff


def test_c2_gate_produces_different_trades_than_baseline(tmp_path):
    root = Path(__file__).resolve().parent.parent
    base_cfg_path = root / "configs" / "v1_system.yaml"
    metro_cfg_path = root / "configs" / "v1_system_c2_metro_advanced.yaml"
    data_dir = root / "data" / "daily"
    if not base_cfg_path.exists() or not metro_cfg_path.exists():
        pytest.skip("v1_system configs not found")
    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        pytest.skip("data/daily not found or empty")

    from core import backtester as bt

    out_baseline = tmp_path / "baseline"
    out_baseline.mkdir()
    out_metro = tmp_path / "metro"
    out_metro.mkdir()

    with open(base_cfg_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    with open(metro_cfg_path, encoding="utf-8") as f:
        metro_cfg = yaml.safe_load(f)

    base_cfg["pairs"] = ["EUR_USD"]
    base_cfg["date_range"] = {"start": "2020-01-01", "end": "2020-03-31"}
    base_cfg["data_dir"] = str(data_dir)
    if "data" in base_cfg and isinstance(base_cfg["data"], dict):
        base_cfg["data"]["dir"] = str(data_dir)

    metro_cfg["pairs"] = ["EUR_USD"]
    metro_cfg["date_range"] = {"start": "2020-01-01", "end": "2020-03-31"}
    metro_cfg["data_dir"] = str(data_dir)
    if "data" in metro_cfg and isinstance(metro_cfg["data"], dict):
        metro_cfg["data"]["dir"] = str(data_dir)

    cfg_baseline_path = tmp_path / "cfg_baseline.yaml"
    cfg_metro_path = tmp_path / "cfg_metro.yaml"
    with open(cfg_baseline_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(base_cfg, f, sort_keys=False)
    with open(cfg_metro_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metro_cfg, f, sort_keys=False)

    bt.run_backtest(config_path=str(cfg_baseline_path), results_dir=str(out_baseline))
    bt.run_backtest(config_path=str(cfg_metro_path), results_dir=str(out_metro))

    trades_baseline = out_baseline / "trades.csv"
    trades_metro = out_metro / "trades.csv"
    assert trades_baseline.exists()
    assert trades_metro.exists()

    df_b = pd.read_csv(trades_baseline)
    df_m = pd.read_csv(trades_metro)
    assert len(df_m) <= len(df_b), "C2 must not increase trade count (gate only filters)"
    different = len(df_m) != len(df_b) or not df_b.equals(df_m)
    assert different, "C2 run must produce different trades.csv content vs baseline"
