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
            "allowlist": {"c1": ["c1_is_calculation", "c1_supertrend"]},
            "blocklist": {"c1": []},
            "roles": {
                "c1": [
                    {"name": "c1_is_calculation", "params": {}},
                    {"name": "c1_supertrend", "params": {}},
                ]
            },
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


def test_wfo_v2_sweep_selects_non_null_c1(tmp_path):
    """WFO v2 sweep must never select c1: null; at least one non-fallback candidate must run."""
    import json

    from scripts.walk_forward import run_wfo_v2

    wfo_path = _write_wfo_v2_invariant_config(tmp_path, with_sweep=True)
    run_wfo_v2(wfo_path)
    run_id_dir = _get_run_dir(tmp_path)
    assert run_id_dir is not None
    fold_01 = run_id_dir / "fold_01"
    is_best_path = fold_01 / "is_best_params.json"
    assert is_best_path.exists(), "fold_01/is_best_params.json must exist"
    best = json.loads(is_best_path.read_text(encoding="utf-8"))
    role_names = best.get("role_names") or {}
    c1 = role_names.get("c1")
    assert c1 is not None and c1 != "", "role_names.c1 must not be null or empty"
    in_sample = fold_01 / "in_sample"
    run_dirs = sorted(in_sample.iterdir()) if in_sample.exists() else []
    assert len(run_dirs) >= 2, "sweep must produce at least two candidate runs (no single null fallback)"


def test_wfo_v2_pinned_baseline_exit_in_is_selection(tmp_path):
    """Regression: WFO candidate configs include baseline and exit (not null) when base enables them."""
    import json

    from scripts.walk_forward import run_wfo_v2

    wfo_path = _write_wfo_v2_invariant_config(tmp_path, with_sweep=True)
    run_wfo_v2(wfo_path)
    run_id_dir = _get_run_dir(tmp_path)
    assert run_id_dir is not None
    assert (run_id_dir / "base_config_used.yaml").exists(), "WFO must write base_config_used.yaml"
    assert (run_id_dir / "wfo_run_meta.json").exists(), "WFO must write wfo_run_meta.json"
    meta = json.loads((run_id_dir / "wfo_run_meta.json").read_text(encoding="utf-8"))
    assert "base_config_path" in meta and "pinned_roles" in meta and "swept_roles" in meta and "folds" in meta
    assert "baseline" in meta["pinned_roles"] and "exit" in meta["pinned_roles"]
    assert meta["swept_roles"] == ["c1"]
    fold_01 = run_id_dir / "fold_01"
    is_best_path = fold_01 / "is_best_params.json"
    assert is_best_path.exists()
    best = json.loads(is_best_path.read_text(encoding="utf-8"))
    role_names = best.get("role_names") or {}
    assert role_names.get("baseline") is not None and role_names.get("baseline") != "", (
        "role_names.baseline must not be null when base has use_baseline=true"
    )
    assert role_names.get("exit") is not None and role_names.get("exit") != "", (
        "role_names.exit must not be null when base has use_exit=true"
    )
    is_selection_path = fold_01 / "is_selection.json"
    assert is_selection_path.exists()
    selection = json.loads(is_selection_path.read_text(encoding="utf-8"))
    for run_entry in selection.get("runs", []):
        rn = run_entry.get("role_names") or {}
        assert rn.get("baseline") is not None, "is_selection.runs[].role_names.baseline must not be null"
        assert rn.get("exit") is not None, "is_selection.runs[].role_names.exit must not be null"


def test_wfo_v2_refuses_null_baseline_when_use_baseline_true():
    """WFO base validation: use_baseline=true with baseline=null raises ValueError."""
    import pytest

    from scripts.walk_forward import _validate_base_indicators

    base_bad = {
        "indicators": {
            "use_baseline": True,
            "baseline": None,
            "use_exit": True,
            "exit": "exit_twiggs_money_flow",
        }
    }
    with pytest.raises(ValueError) as exc_info:
        _validate_base_indicators(base_bad)
    msg = str(exc_info.value)
    assert "use_baseline" in msg or "baseline" in msg


def test_wfo_phase5_config_folds_and_base_indicators(tmp_path):
    """Load configs/wfo_phase5.yaml; assert 4 explicit folds and base has baseline/exit (not null).
    Hermetic: override data dir to tmp_path so CI (no data/daily) passes without weakening validation.
    """
    from pathlib import Path

    import pytest

    from scripts.walk_forward import _get_folds_for_wfo, _load_yaml
    from validators_config import validate_config

    root = Path(__file__).resolve().parents[1]
    wfo_path = root / "configs" / "wfo_phase5.yaml"
    if not wfo_path.exists():
        pytest.skip("configs/wfo_phase5.yaml not found")
    wfo = _load_yaml(wfo_path)
    base_config_path = wfo.get("base_config") or "configs/v1_system.yaml"
    resolved_base = (wfo_path.parent / base_config_path).resolve()
    if not resolved_base.exists():
        resolved_base = (root / base_config_path).resolve()
    if not resolved_base.exists():
        pytest.skip(f"base config not found: {base_config_path}")
    with resolved_base.open("r", encoding="utf-8-sig") as f:
        base = yaml.safe_load(f) or {}
    hermetic_data_dir = tmp_path / "data"
    hermetic_data_dir.mkdir(parents=True, exist_ok=True)
    if "data" in base and isinstance(base["data"], dict) and "dir" in base["data"]:
        base["data"] = dict(base["data"])
        base["data"]["dir"] = str(hermetic_data_dir.resolve())
    base["data_dir"] = str(hermetic_data_dir.resolve())
    base = validate_config(base)
    folds = _get_folds_for_wfo(wfo, base)
    assert len(folds) == 4, "wfo_phase5.yaml must define exactly 4 folds"
    ind = base.get("indicators") or {}
    assert ind.get("baseline") is not None and str(ind.get("baseline")).strip(), (
        "Base config (v1_system) must have indicators.baseline set for Phase 5"
    )
    assert ind.get("exit") is not None and str(ind.get("exit")).strip(), (
        "Base config (v1_system) must have indicators.exit set for Phase 5"
    )
