# tests/test_wfo_v2_invariants.py — WFO v2 invariants (no leakage, params unchanged, trade count stable)
import time
from pathlib import Path

import pandas as pd
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


def _write_wfo_v2_invariant_config(tmp_path: Path, with_sweep: bool = True) -> Path:
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

    if with_sweep:
        sweep = {
            "role_filters": ["c1"],
            "discover": {"c1": False, "c2": False, "baseline": False, "volume": False, "exit": False},
            "allowlist": {"c1": ["is_calculation", "supertrend"]},
            "blocklist": {"c1": []},
            "roles": {"c1": []},
            "default_params": {"c1": {}},
            "static_overrides": {"timeframe": "D", "spreads": {"enabled": False}},
            "parallel": {"workers": 1, "max_runs": None},
        }
        sweep_path = tmp_path / "sweep.yaml"
        with sweep_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(sweep, f, sort_keys=False)

    wfo = {
        "base_config": "base.yaml",
        "data_scope": {"from_date": "2020-01-01", "to_date": "2020-04-30"},
        "fold_scheme": {"train_months": 3, "test_months": 1, "step_months": 1},
        "engine": {"cache_on": False, "spreads_on": False},
        "output_root": str((tmp_path / "results" / "wfo").resolve()),
    }
    if with_sweep:
        wfo["sweep_config"] = "sweep.yaml"
        wfo["selection"] = {"metric": "roi_pct", "tie_break_order": ["max_dd_pct", "total_trades"]}

    wfo_path = tmp_path / "wfo_v2.yaml"
    with wfo_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(wfo, f, sort_keys=False)
    return wfo_path


def _get_run_dir(tmp_path: Path):
    run_dir = tmp_path / "results" / "wfo"
    if not run_dir.exists():
        return None
    run_ids = sorted(run_dir.iterdir(), key=lambda p: p.name)
    return run_ids[0] if run_ids else None


def _get_oos_trade_count(run_id_dir: Path) -> int:
    fold_01 = run_id_dir / "fold_01"
    trades_path = fold_01 / "out_of_sample" / "trades.csv"
    if not trades_path.exists():
        summary_path = fold_01 / "out_of_sample" / "summary.txt"
        if summary_path.exists():
            txt = summary_path.read_text(encoding="utf-8")
            for line in txt.splitlines():
                if "Total Trades" in line or "total_trades" in line.lower():
                    import re
                    m = re.search(r"\d+", line)
                    if m:
                        return int(m.group(0))
        return -1
    df = pd.read_csv(trades_path)
    return len(df)


def test_fold_boundaries_no_leakage():
    from analytics.wfo import generate_folds, validate_no_test_overlap

    folds = generate_folds("2020-01-01", "2021-12-31", 12, 3, 3)
    assert len(folds) == 4
    for f in folds:
        assert f.train_end < f.test_start
    validate_no_test_overlap(folds)


def test_params_unchanged_oos(tmp_path):
    from scripts.walk_forward import run_wfo_v2

    wfo_path = _write_wfo_v2_invariant_config(tmp_path, with_sweep=True)
    run_wfo_v2(wfo_path)
    run_id_dir = _get_run_dir(tmp_path)
    assert run_id_dir is not None
    fold_01 = run_id_dir / "fold_01"
    assert (fold_01 / "is_best_params.json").exists()
    assert (fold_01 / "params_hash.txt").exists()
    oos_hash_path = fold_01 / "out_of_sample" / "params_hash.txt"
    assert oos_hash_path.exists()
    fold_hash = (fold_01 / "params_hash.txt").read_text(encoding="utf-8").strip()
    oos_hash = oos_hash_path.read_text(encoding="utf-8").strip()
    assert fold_hash == oos_hash


def test_trade_count_stability(tmp_path):
    from scripts.walk_forward import run_wfo_v2

    wfo_path = _write_wfo_v2_invariant_config(tmp_path, with_sweep=True)
    run_wfo_v2(wfo_path)
    run_id_dir_1 = _get_run_dir(tmp_path)
    assert run_id_dir_1 is not None
    count1 = _get_oos_trade_count(run_id_dir_1)

    time.sleep(1.1)
    run_wfo_v2(wfo_path)
    run_ids = sorted((tmp_path / "results" / "wfo").iterdir(), key=lambda p: p.name)
    assert len(run_ids) >= 2
    run_id_dir_2 = run_ids[-1]
    count2 = _get_oos_trade_count(run_id_dir_2)
    assert count1 == count2
