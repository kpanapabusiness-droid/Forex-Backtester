# tests/test_wfo_v2_runner_smoke.py — WFO v2 runner smoke (1 fold IS+OOS, boundaries, cache off)
from pathlib import Path

import pandas as pd
import pytest
import yaml


def _minimal_ohlcv_csv(path: Path, start: str = "2020-01-01", days: int = 130) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(start, periods=days, freq="D")
    close = 1.1000 + (pd.Series(range(days)) * 0.0001).values
    high = close + 0.0010
    low = close - 0.0010
    open_ = close - 0.0005
    df = pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": 1000}
    )
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.to_csv(path, index=False)


def _write_wfo_v2_smoke_config(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _minimal_ohlcv_csv(data_dir / "EUR_USD.csv", start="2020-01-01", days=130)

    base = {
        "pairs": ["EUR_USD"],
        "timeframe": "D",
        "data_dir": str(data_dir.resolve()),
        "indicators": {
            "c1": "c1_is_calculation",
            "use_c2": False,
            "use_baseline": True,
            "baseline": "baseline_ema",
            "use_volume": False,
            "use_exit": True,
            "exit": "exit_twiggs_money_flow",
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
            "exit_on_exit_signal": True,
        },
        "engine": {"allow_continuation": True, "duplicate_open_policy": "block"},
        "continuation": {
            "allow_continuation": False,
            "skip_volume_check": False,
            "skip_pullback_check": False,
            "block_if_crossed_baseline_since_entry": False,
        },
        "tracking": {
            "in_sim_equity": True,
            "track_win_loss_scratch": True,
            "track_roi": True,
            "track_drawdown": True,
        },
        "filters": {
            "dbcvix": {
                "enabled": False,
                "mode": "reduce",
                "threshold": 0.0,
                "reduce_risk_to": 0.01,
                "source": "synthetic",
            }
        },
        "spreads": {"enabled": False, "default_pips": 1.0},
        "cache": {"enabled": False, "dir": "cache", "format": "parquet"},
        "validation": {"enabled": True, "fail_fast": True, "strict_contract": False},
        "output": {"results_dir": "results"},
        "risk": {"starting_balance": 10000.0, "risk_per_trade_pct": 2.0},
        "date_range": {"start": "2020-01-01", "end": "2020-04-30"},
    }
    base_path = tmp_path / "base.yaml"
    with base_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(base, f, sort_keys=False)

    wfo = {
        "base_config": "base.yaml",
        "data_scope": {"from_date": "2020-01-01", "to_date": "2020-04-30"},
        "fold_scheme": {"train_months": 3, "test_months": 1, "step_months": 1},
        "engine": {"cache_on": False, "spreads_on": False},
        "output_root": str((tmp_path / "results" / "wfo").resolve()),
    }
    wfo_path = tmp_path / "wfo_v2.yaml"
    with wfo_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(wfo, f, sort_keys=False)
    return wfo_path


def test_wfo_v2_runner_smoke_one_fold(tmp_path):
    from scripts.walk_forward import run_wfo_v2

    wfo_path = _write_wfo_v2_smoke_config(tmp_path)
    run_wfo_v2(wfo_path)
    run_dir = tmp_path / "results" / "wfo"
    assert run_dir.exists()
    run_ids = list(run_dir.iterdir())
    assert len(run_ids) >= 1
    run_id_dir = run_ids[0]
    fold_01 = run_id_dir / "fold_01"
    assert fold_01.exists()
    fold_dates_path = fold_01 / "fold_dates.json"
    assert fold_dates_path.exists()
    import json

    fold_dates = json.loads(fold_dates_path.read_text(encoding="utf-8"))
    assert fold_dates["train_start"] == "2020-01-01"
    assert fold_dates["train_end"] == "2020-03-31"
    assert fold_dates["test_start"] == "2020-04-01"
    assert fold_dates["test_end"] == "2020-04-30"
    in_sample = fold_01 / "in_sample"
    out_of_sample = fold_01 / "out_of_sample"
    assert in_sample.exists()
    assert out_of_sample.exists()
    assert (in_sample / "summary.txt").exists()
    assert (out_of_sample / "summary.txt").exists()


def test_wfo_v2_config_engine_cache_off():
    from scripts.walk_forward import _load_yaml

    root = Path(__file__).resolve().parents[1]
    wfo_path = root / "configs" / "wfo_v2.yaml"
    if not wfo_path.exists():
        pytest.skip("configs/wfo_v2.yaml not found")
    wfo = _load_yaml(wfo_path)
    engine = wfo.get("engine") or {}
    assert engine.get("cache_on") is False
