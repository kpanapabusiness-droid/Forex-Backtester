# tests/test_risk_sizing.py
# risk_per_trade from config must drive position sizing: same signals, different risk => different ROI ($), same trade count.

from pathlib import Path

import pandas as pd
import pytest
import yaml

from validators_config import load_and_validate_config


def test_risk_per_trade_passed_through_validator():
    """Validated config must contain risk.risk_per_trade when YAML has risk_per_trade."""
    cfg_05 = load_and_validate_config("configs/phase6_spread_full.yaml")
    cfg_25 = load_and_validate_config("configs/phase6_1_spread_full.yaml")
    assert cfg_05["risk"]["risk_per_trade"] == 0.005
    assert cfg_25["risk"]["risk_per_trade"] == 0.0025
    assert cfg_05["risk"]["risk_per_trade"] != cfg_25["risk"]["risk_per_trade"]


def test_different_risk_per_trade_changes_roi_same_trade_count(tmp_path):
    """Same config except risk_per_trade: changing it must change ROI ($), not trade count."""
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "daily"
    base_cfg_path = root / "configs" / "phase6_spread_full.yaml"
    if not base_cfg_path.exists() or not data_dir.exists() or not list(data_dir.glob("*.csv")):
        pytest.skip("phase6_spread_full or data/daily missing")

    with open(base_cfg_path, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    base["output"] = base.get("output") or {}
    base["output"]["results_dir"] = "results"  # overwritten by results_dir arg

    cfg_high = dict(base)
    cfg_high["risk"] = dict(cfg_high.get("risk") or {})
    cfg_high["risk"]["risk_per_trade"] = 0.01
    cfg_high["risk"]["starting_balance"] = 10000.0
    cfg_low = dict(base)
    cfg_low["risk"] = dict(cfg_low.get("risk") or {})
    cfg_low["risk"]["risk_per_trade"] = 0.005
    cfg_low["risk"]["starting_balance"] = 10000.0

    path_high = tmp_path / "high_risk.yaml"
    path_low = tmp_path / "low_risk.yaml"
    with open(path_high, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_high, f, sort_keys=False)
    with open(path_low, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_low, f, sort_keys=False)

    out_high = tmp_path / "out_high"
    out_low = tmp_path / "out_low"
    out_high.mkdir()
    out_low.mkdir()

    from core.backtester import run_backtest

    run_backtest(config_path=str(path_high), results_dir=str(out_high))
    run_backtest(config_path=str(path_low), results_dir=str(out_low))

    t_high = pd.read_csv(out_high / "trades.csv")
    t_low = pd.read_csv(out_low / "trades.csv")
    assert len(t_high) == len(t_low), "Trade count must be unchanged when only risk_per_trade differs"
    assert len(t_high) > 0, "Need at least one trade"

    roi_high = t_high["pnl"].sum()
    roi_low = t_low["pnl"].sum()
    assert roi_high != roi_low, "ROI ($) must differ when risk_per_trade differs"
    # Roughly half risk => roughly half dollar pnl (same trade count)
    ratio = roi_low / roi_high
    assert 0.3 < ratio < 0.7, f"Expected ROI ratio ~0.5 (risk 0.005 vs 0.01); got {ratio}"
