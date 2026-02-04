# tests/test_risk_sizing.py
# risk_per_trade from config must drive position sizing: same signals, different risk => different ROI ($), same trade count.

from pathlib import Path

import pandas as pd
import pytest
import yaml

from validators_config import load_and_validate_config


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


def test_risk_per_trade_passed_through_validator(tmp_path):
    """Validated config must normalize risk.risk_per_trade and risk_per_trade_pct from YAML."""

    def write_cfg(path: Path, risk_per_trade: float) -> None:
        cfg = _minimal_phase1_base_config()
        cfg["risk"] = {
            "starting_balance": 10_000.0,
            "risk_per_trade": risk_per_trade,
        }
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    path_05 = tmp_path / "risk_05.yaml"
    path_25 = tmp_path / "risk_25.yaml"
    write_cfg(path_05, 0.005)
    write_cfg(path_25, 0.0025)

    cfg_05 = load_and_validate_config(str(path_05))
    cfg_25 = load_and_validate_config(str(path_25))

    assert cfg_05["risk"]["risk_per_trade"] == pytest.approx(0.005)
    assert cfg_25["risk"]["risk_per_trade"] == pytest.approx(0.0025)
    assert cfg_05["risk"]["risk_per_trade_pct"] == pytest.approx(0.5)
    assert cfg_25["risk"]["risk_per_trade_pct"] == pytest.approx(0.25)
    assert cfg_05["risk"]["risk_per_trade"] != cfg_25["risk"]["risk_per_trade"]


def test_different_risk_per_trade_changes_roi_same_trade_count(tmp_path):
    """Same config except risk_per_trade: changing it must change ROI ($), not trade count."""
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "daily"
    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        pytest.skip("data/daily missing")

    base = _minimal_phase1_base_config()
    base["data_dir"] = str(data_dir)
    base["output"] = {"results_dir": "results"}  # overwritten by results_dir arg

    cfg_high = dict(base)
    cfg_high["risk"] = {
        "risk_per_trade": 0.01,
        "starting_balance": 10_000.0,
    }
    cfg_low = dict(base)
    cfg_low["risk"] = {
        "risk_per_trade": 0.005,
        "starting_balance": 10_000.0,
    }

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
    if len(t_high) == 0:
        pytest.skip("No trades generated; cannot compare ROI across different risk_per_trade values.")

    roi_high = t_high["pnl"].sum()
    roi_low = t_low["pnl"].sum()
    assert roi_high != roi_low, "ROI ($) must differ when risk_per_trade differs"
    # Roughly half risk => roughly half dollar pnl (same trade count)
    ratio = roi_low / roi_high
    assert 0.3 < ratio < 0.7, f"Expected ROI ratio ~0.5 (risk 0.005 vs 0.01); got {ratio}"
