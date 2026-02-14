"""Tests for Phase E-1.4 corridor sweeps."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))


def _params_hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, default=str)
    return __import__("hashlib").sha256(blob.encode()).hexdigest()[:12]


def expand_grid(grid: dict) -> list[dict]:
    import itertools
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, prod)) for prod in itertools.product(*values)]


def test_grid_expansion_deterministic_order() -> None:
    grid = {"a": [1, 2], "b": [10, 20]}
    combos = expand_grid(grid)
    assert len(combos) == 4
    assert combos[0] == {"a": 1, "b": 10}
    assert combos[1] == {"a": 1, "b": 20}
    assert combos[2] == {"a": 2, "b": 10}
    assert combos[3] == {"a": 2, "b": 20}
    combos2 = expand_grid(grid)
    assert combos == combos2


def test_params_hash_stable() -> None:
    p1 = {"q_atr": 0.25, "q_rng": 0.30}
    p2 = {"q_rng": 0.30, "q_atr": 0.25}
    assert _params_hash(p1) == _params_hash(p2)
    p3 = {"q_atr": 0.26, "q_rng": 0.30}
    assert _params_hash(p1) != _params_hash(p3)


def test_max_combos_cap() -> None:
    grid = {"a": [1, 2, 3], "b": [10, 20, 30]}
    combos = expand_grid(grid)
    assert len(combos) == 9
    capped = combos[:5]
    assert len(capped) == 5
    assert capped[0] == {"a": 1, "b": 10}
    assert capped[-1] == {"a": 2, "b": 20}


def test_aggregator_sorts_pass_first() -> None:
    import pandas as pd
    aggregated = [
        {"params_hash": "a1", "params_json": "{}", "PASS": False, "reject_reason": "x",
         "P_3R_before_2R_val": 0.5, "clustering_ratio": 0.1},
        {"params_hash": "b2", "params_json": "{}", "PASS": True, "reject_reason": "",
         "P_3R_before_2R_val": 0.4, "clustering_ratio": 0.2},
        {"params_hash": "c3", "params_json": "{}", "PASS": True, "reject_reason": "",
         "P_3R_before_2R_val": 0.6, "clustering_ratio": 0.05},
    ]
    df = pd.DataFrame(aggregated)
    pass_first = df["PASS"].fillna(False).astype(bool)
    pass_df = df[pass_first].sort_values(
        ["P_3R_before_2R_val", "clustering_ratio"],
        ascending=[False, True],
        na_position="last",
    )
    fail_df = df[~pass_first].sort_values(
        ["P_3R_before_2R_val", "clustering_ratio"],
        ascending=[False, True],
        na_position="last",
    )
    result = pd.concat([pass_df, fail_df], ignore_index=True)
    assert list(result["PASS"]) == [True, True, False]
    assert list(result["params_hash"]) == ["c3", "b2", "a1"]
    assert result.iloc[0]["P_3R_before_2R_val"] == 0.6
    assert result.iloc[1]["clustering_ratio"] == 0.2


def test_expand_grid_matches_script() -> None:
    from scripts.phaseE1_run_corridor_sweeps import expand_grid as script_expand
    grid = {"gamma": [0.6, 0.65], "alpha": [0.9, 1.0]}
    a = expand_grid(grid)
    b = script_expand(grid)
    assert a == b


def test_indicator_only_excludes_csv_signals(tmp_path: Path) -> None:
    """Phase E run with indicator_only produces leaderboard with indicator only, not csv:proto_signals."""
    import pandas as pd

    from scripts.phaseD6G_run_signal_geometry import _run

    csv_dir = tmp_path / "results" / "phaseD"
    csv_dir.mkdir(parents=True)
    proto_csv = csv_dir / "proto_signals.csv"
    pd.DataFrame({
        "pair": ["P1", "P1", "P2"],
        "date": ["2020-01-01", "2020-01-02", "2020-01-01"],
        "signal": [1, -1, 0],
    }).to_csv(proto_csv, index=False)

    data_dir = tmp_path / "data" / "daily"
    data_dir.mkdir(parents=True)
    ohlc = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "open": 1.0, "high": 1.01, "low": 0.99, "close": 1.0, "volume": 100,
    })
    ohlc.to_csv(data_dir / "P1.csv", index=False)
    ohlc.to_csv(data_dir / "P2.csv", index=False)

    clean_df = pd.DataFrame({
        "pair": ["P1", "P1", "P2"],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
        "valid_atr": [True, True, True],
        "valid_ref": [True, True, True],
        "valid_h40": [True, True, True],
        "clean_mfe_long_x1_h40": [1.5, 2.0, 0.5],
        "clean_mfe_short_x1_h40": [0.5, 1.0, 2.0],
    })
    clean_path = tmp_path / "clean.csv"
    clean_df.to_csv(clean_path, index=False)

    cfg = {
        "project_root": str(ROOT),
        "indicator_only": True,
        "signal_sources": {"modules": ["indicators.confirmation_funcs"], "csv_dirs": []},
        "signals": ["c1_coral"],
        "split_date": "2023-01-01",
        "mae_x": [1],
        "thresholds_y": [1.0],
        "date_range": {"start": "2019-01-01", "end": "2026-01-01"},
        "data_dir": str(tmp_path / "data" / "daily"),
        "primary_objective": "3R_before_2R",
    }
    out_dir = tmp_path / "out"
    _run(cfg, Path(clean_path), out_dir)

    lb_path = out_dir / "leaderboard_geometry_lock.csv"
    assert lb_path.exists()
    lb = pd.read_csv(lb_path)
    signal_names = lb["signal_name"].tolist()
    assert "indicators.confirmation_funcs:c1_coral" in signal_names
    assert not any(s.startswith("csv:") for s in signal_names)


def test_lsr_params_change_signal_count() -> None:
    """Changing LSR L_sw param produces different nonzero signal counts."""
    import numpy as np
    import pandas as pd

    from indicators.confirmation_funcs import c1_liquidity_sweep_rejection

    np.random.seed(42)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.Series(1.0 + np.cumsum(np.random.randn(n) * 0.01))
    high = close + np.abs(np.random.randn(n) * 0.005)
    low = close - np.abs(np.random.randn(n) * 0.005)
    open_ = close.shift(1).fillna(close.iloc[0])
    df = pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 100,
    })

    out_5 = c1_liquidity_sweep_rejection(df.copy(), L_sw=5, signal_col="c1_signal")
    out_50 = c1_liquidity_sweep_rejection(df.copy(), L_sw=50, signal_col="c1_signal")

    nz_5 = (out_5["c1_signal"] != 0).sum()
    nz_50 = (out_50["c1_signal"] != 0).sum()
    assert nz_5 != nz_50
