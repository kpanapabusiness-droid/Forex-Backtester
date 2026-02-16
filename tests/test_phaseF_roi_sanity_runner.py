# Phase F: CEB v3 ROI sanity runner — tests for scripts/phaseF_run_ceb_roi_sanity.py

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def _minimal_ceb_v3_config(tmp_path: Path, out_dir: Path) -> dict:
    """Minimal CEB v3 config for fast integration test (1 pair, short range)."""
    return {
        "strategy_version": "forex_backtester_v1.9.7",
        "timeframe": "D",
        "date_range": {"start": "2020-01-01", "end": "2020-06-30"},
        "pairs": ["EUR_USD"],
        "spreads": {"enabled": True, "default_pips": 0.5, "per_pair": {}, "mode": "fixed", "atr_mult": 0.0},
        "indicators": {
            "c1": "c1_compression_escape_ratio_state_machine",
            "use_c2": False,
            "use_baseline": False,
            "use_volume": False,
            "use_exit": False,
            "c2": None,
            "baseline": None,
            "volume": None,
            "exit": None,
        },
        "indicator_params": {
            "c1_compression_escape_ratio_state_machine": {
                "L_slow": 50,
                "r_atr": 0.85,
                "r_rng": 0.85,
                "r_box": 1.1,
                "M_enter": 4,
                "M_exit": 3,
                "K_max": 40,
                "cooldown_bars": 10,
            }
        },
        "rules": {
            "one_candle_rule": False,
            "pullback_rule": False,
            "bridge_too_far_days": 7,
            "allow_baseline_as_catalyst": False,
        },
        "entry": {"sl_atr": 3.0, "tp1_atr": 1.0, "trail_after_atr": 2.0, "ts_atr": 1.5},
        "execution": {"intrabar_priority": "tp_first"},
        "engine": {"allow_continuation": True, "duplicate_open_policy": "block"},
        "exit": {
            "use_trailing_stop": True,
            "move_to_breakeven_after_atr": True,
            "exit_on_c1_reversal": True,
            "exit_on_baseline_cross": False,
            "exit_on_exit_signal": False,
        },
        "risk": {"starting_balance": 10000.0, "risk_per_trade_pct": 2.0, "risk_per_trade": 0.02},
        "output": {"results_dir": str(out_dir)},
        "outputs": {"dir": str(out_dir), "write_trades_csv": True},
        "tracking": {
            "track_win_loss_scratch": True,
            "track_roi": True,
            "track_drawdown": True,
            "in_sim_equity": True,
            "verbose_logs": False,
        },
        "filters": {"dbcvix": {"enabled": False}},
    }


def test_phaseF_run_ceb_roi_sanity_produces_outputs(tmp_path: Path) -> None:
    """Run phaseF runner with minimal config (1 pair, 6 months); verify outputs exist."""
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "daily"
    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        pytest.skip("data/daily missing")

    out_dir = tmp_path / "phaseF_out"
    cfg = _minimal_ceb_v3_config(tmp_path, out_dir)
    cfg["data_dir"] = str(data_dir)
    cfg_path = tmp_path / "ceb_v3_mini.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env = {**os.environ, "PYTHONPATH": str(root)}
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "phaseF_run_ceb_roi_sanity.py"), "-c", str(cfg_path)],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")

    assert (out_dir / "trades.csv").exists(), "trades.csv must exist"
    assert (out_dir / "summary.txt").exists(), "summary.txt must exist"
    assert (out_dir / "equity_curve.csv").exists(), "equity_curve.csv must exist"
    assert "total_trades" in (result.stdout or ""), "Printed summary should include total_trades"
    assert "output_dir" in (result.stdout or ""), "Printed summary should include output_dir"
