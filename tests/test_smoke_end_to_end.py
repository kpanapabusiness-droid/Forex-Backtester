import textwrap
from pathlib import Path

import pandas as pd
import pytest


def _write_test_cfg(tmp_path: Path):
    """Write a test config that uses test data directory."""
    # Get absolute path to test data directory
    test_data_dir = Path(__file__).parent.parent / "data" / "test"

    cfg = textwrap.dedent(f"""
    pairs: ["EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF"]
    timeframe: "D"
    data:
      dir: "{test_data_dir}"

    indicators:
      c1: "c1_twiggs_money_flow"
      use_c2: false
      use_baseline: true
      baseline: "baseline_ema"
      use_volume: false
      use_exit: false

    rules:
      one_candle_rule: false
      pullback_rule: false
      bridge_too_far_days: 7
      allow_baseline_as_catalyst: false

    entry:
      atr_multiple: 2.0

    exit:
      use_trailing_stop: true
      move_to_breakeven_after_atr: true
      exit_on_c1_reversal: true
      exit_on_baseline_cross: false
      exit_on_exit_signal: false

    spreads:
      enabled: false
      default_pips: 1.0

    tracking:
      in_sim_equity: true
      track_win_loss_scratch: true
      track_roi: true
      track_drawdown: true

    risk_filters:
      dbcvix:
        enabled: false
        mode: "reduce"
        threshold: 0.0
        reduce_risk_to: 0.01
        source: "synthetic"

    date_range:
      start: "2018-01-01"
      end: "2022-12-31"
    """).strip()

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    return str(cfg_path)


def test_smoke_backtest_outputs(tmp_path):
    import backtester as bt

    # Create test config that points to test data
    cfg_path = _write_test_cfg(tmp_path)

    out_dir = tmp_path / "results_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    bt.run_backtest(config_path=cfg_path, results_dir=str(out_dir))
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
    assert {"date", "equity"}.issubset(set(df_eq.columns))
    if "drawdown" in df_eq.columns:
        assert (df_eq["drawdown"].dropna() <= 0).all()


def test_smoke_walk_forward(tmp_path):
    import backtester as bt

    # Create test config that points to test data
    cfg_path = _write_test_cfg(tmp_path)

    out_dir = tmp_path / "results_wfo"
    out_dir.mkdir(parents=True, exist_ok=True)
    bt.run_backtest_walk_forward(config_path=cfg_path, results_dir=str(out_dir))
    trades = out_dir / "trades.csv"
    equity = out_dir / "equity_curve.csv"
    assert trades.exists()
    assert equity.exists()
    df_tr = pd.read_csv(trades)
    assert len(df_tr) >= 0
    df_eq = pd.read_csv(equity)
    assert {"date", "equity"}.issubset(set(df_eq.columns))


def test_smoke_monte_carlo(tmp_path):
    mc = pytest.importorskip("analytics.monte_carlo")
    import backtester as bt

    # Create test config that points to test data
    cfg_path = _write_test_cfg(tmp_path)

    out_dir = tmp_path / "results_mc"
    out_dir.mkdir(parents=True, exist_ok=True)
    bt.run_backtest(config_path=cfg_path, results_dir=str(out_dir))
    trades = out_dir / "trades.csv"
    assert trades.exists()
    try:
        fn = getattr(mc, "run_monte_carlo")
        mc_out = out_dir / "mc"
        mc_out.mkdir(parents=True, exist_ok=True)
        fn(str(trades), str(mc_out))
        produced = (
            list(mc_out.glob("*.csv")) + list(mc_out.glob("*.txt")) + list(mc_out.glob("*.json"))
        )
        assert len(produced) > 0
    except AttributeError:
        pytest.skip("analytics.monte_carlo.run_monte_carlo not available")
