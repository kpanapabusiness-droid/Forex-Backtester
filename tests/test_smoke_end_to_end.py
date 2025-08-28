import os
from pathlib import Path
import json
import pandas as pd
import pytest

def _cfg_path():
    p = Path("config.yaml")
    assert p.exists()
    return str(p)

def test_smoke_backtest_outputs(tmp_path):
    import backtester as bt
    out_dir = tmp_path / "results_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    bt.run_backtest(config_path=_cfg_path(), results_dir=str(out_dir))
    trades = out_dir / "trades.csv"
    equity = out_dir / "equity_curve.csv"
    summary = out_dir / "summary.txt"
    assert trades.exists()
    assert equity.exists()
    assert summary.exists()
    df_tr = pd.read_csv(trades)
    assert len(df_tr) >= 0
    if "win" in df_tr.columns and "loss" in df_tr.columns and "scratch" in df_tr.columns:
        wins = int(df_tr["win"].fillna(0).sum())
        losses = int(df_tr["loss"].fillna(0).sum())
        scratches = int(df_tr["scratch"].fillna(0).sum())
        assert wins + losses + scratches == len(df_tr)
    df_eq = pd.read_csv(equity)
    assert {"date","equity"}.issubset(set(df_eq.columns))
    if "drawdown" in df_eq.columns:
        assert (df_eq["drawdown"].dropna() <= 0).all()

def test_smoke_walk_forward(tmp_path):
    import backtester as bt
    out_dir = tmp_path / "results_wfo"
    out_dir.mkdir(parents=True, exist_ok=True)
    bt.run_backtest_walk_forward(config_path=_cfg_path(), results_dir=str(out_dir))
    trades = out_dir / "trades.csv"
    equity = out_dir / "equity_curve.csv"
    assert trades.exists()
    assert equity.exists()
    df_tr = pd.read_csv(trades)
    assert len(df_tr) >= 0
    df_eq = pd.read_csv(equity)
    assert {"date","equity"}.issubset(set(df_eq.columns))

def test_smoke_monte_carlo(tmp_path):
    mc = pytest.importorskip("analytics.monte_carlo")
    import backtester as bt
    out_dir = tmp_path / "results_mc"
    out_dir.mkdir(parents=True, exist_ok=True)
    bt.run_backtest(config_path=_cfg_path(), results_dir=str(out_dir))
    trades = out_dir / "trades.csv"
    assert trades.exists()
    try:
        fn = getattr(mc, "run_monte_carlo")
        mc_out = out_dir / "mc"
        mc_out.mkdir(parents=True, exist_ok=True)
        fn(str(trades), str(mc_out))
        produced = list(mc_out.glob("*.csv")) + list(mc_out.glob("*.txt")) + list(mc_out.glob("*.json"))
        assert len(produced) > 0
    except AttributeError:
        pytest.skip("analytics.monte_carlo.run_monte_carlo not available")
