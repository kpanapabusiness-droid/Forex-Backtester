# tests/test_resolver_and_pipeline_smoke.py
# v1.9.8 — resolver + WFO/MC pipeline smoke tests (fast, deterministic)

from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import importlib
import yaml


# ------------------------
# Helpers (synthetic data)
# ------------------------
def _synthetic_ohlcv(start="2017-01-02", periods=1200, freq="B", seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods, freq=freq)
    base = 1.20 + np.cumsum(rng.normal(0, 0.0005, size=len(dates)))
    high = base + np.abs(rng.normal(0.0004, 0.0002, size=len(dates)))
    low  = base - np.abs(rng.normal(0.0004, 0.0002, size=len(dates)))
    open_ = base + rng.normal(0.0, 0.0002, size=len(dates))
    close = base + rng.normal(0.0, 0.0002, size=len(dates))
    vol = rng.integers(100, 500, size=len(dates))
    df = pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
        "spread": 0.0001,
    })
    return df


def _write_cfg(tmp: Path, run_name="wfo_smoke_pytest"):
    cfg = textwrap.dedent(f"""
    run_name: "single_run_default"

    data:
      dir: "data/daily"
      pairs: ["EUR_USD"]

    data_dir: "data/daily"
    pairs: ["EUR_USD"]

    risk:
      starting_balance: 10000.0
      per_trade_risk_pct: 1.0

    tracking:
      in_sim_equity: true
      write_per_bar_equity: true
      verbose_logs: false

    validation:
      enabled: false
      strict_contract: false

    indicators:
      c1: "coral"       # exists (c1_coral)
      use_c2: true
      c2: "coral"       # resolved via shared pool to c1_coral
      use_baseline: false
      baseline: "ema"
      use_volume: false
      volume: "sma"
      use_exit: false
      exit: "atr_trailing"

    indicator_params: {{}}

    rules:
      one_candle_rule: false
      pullback_rule: false
      bridge_too_far_pips: 0
      baseline_catalyst_same_day: false

    continuation:
      enabled: true
      cooldown_bars: 0
      max_retries: 1
      require_c2_on_continuation: false

    exit:
      tp1_rr: 1.0
      partial_tp1: 0.5
      break_even_after_tp1: true
      trailing_stop_enabled: true
      trailing_atr_mult: 1.5
      trailing_from_entry_atr: true
      exit_on_c1_reverse: true
      exit_on_baseline_cross: true
      use_exit_indicator: false

    cache:
      enabled: true
      dir: "Cache"
      format: "parquet"
      scope_key: null

    walk_forward:
      enabled: true
      run_name: "{run_name}"
      seed: 42
      start: "2019-01-01"
      end:   "2023-12-31"
      train_years: 2
      test_years: 1
      step_years: 1
      mode: "dates"
      train_bars: 200
      test_bars: 50

    monte_carlo:
      enabled: true
      iterations: 300
      horizon: "oos"
      use_daily_returns: false   # per-trade shuffle (works even if no per-bar equity)
      save_samples: false
      seed: 42
    """).strip() + "\n"
    p = tmp / "config.yaml"
    p.write_text(cfg, encoding="utf-8")
    return p


# ------------------------------------------
# 1) Unit test: resolver uses shared pool
# ------------------------------------------
def test_confirm_resolver_shared_pool():
    import indicators.confirmation_funcs as cf
    import indicators_cache
    importlib.reload(indicators_cache)  # ensure latest patch loaded

    assert hasattr(cf, "c1_coral"), "Expected c1_coral to exist"

    fq, fn = indicators_cache._resolve_confirm_func("coral", role="c2")
    assert fq.endswith(".c1_coral"), f"Resolver should map c2/coral to c1_coral, got {fq}"
    assert callable(fn)


# ---------------------------------------------------
# 2) Pipeline smoke: WFO writes files + MC summary
# ---------------------------------------------------
def test_wfo_mc_pipeline_smoke(tmp_path, monkeypatch):
    # Imports
    import backtester
    import indicators_cache
    import utils as _utils
    import validators_util as _vutil
    import signal_logic as _slogic
    importlib.reload(backtester)
    importlib.reload(indicators_cache)

    # Synthetic data & config
    cfg_path = _write_cfg(tmp_path, run_name="wfo_smoke_pytest")
    df_syn = _synthetic_ohlcv(periods=1400, seed=11)

    # 1) Data loader
    def load_pair_csv_stub(pair, data_dir=None):
        return df_syn.copy()
    monkeypatch.setattr(backtester, "load_pair_csv", load_pair_csv_stub, raising=True)

    # 2) Lightweight ATR
    def calculate_atr_stub(df, period: int = 14):
        df = df.copy()
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(period, min_periods=1).mean()
        return df
    monkeypatch.setattr(_utils, "calculate_atr", calculate_atr_stub, raising=True)

    # 3) Validators no-op
    def validate_contract_stub(df, config=None, strict=False):
        return True
    monkeypatch.setattr(_vutil, "validate_contract", validate_contract_stub, raising=True)

    # 4) Signal logic (just ensure expected cols exist)
    def apply_signal_logic_stub(df, cfg):
        df = df.copy()
        df["entry_signal"] = df.get("entry_signal", 0)
        df["exit_signal"] = df.get("exit_signal", 0)
        return df
    monkeypatch.setattr(_slogic, "apply_signal_logic", apply_signal_logic_stub, raising=True)

    # 5) Let apply_indicators_with_cache run through indicators_cache,
    #    but short-circuit _call_indicator to avoid heavy indicator logic.
    #    It still exercises the shared resolver/caching path.
    def _call_indicator_stub(func, df, params, signal_col):
        df = df.copy()
        # create the signal column; alternate small +/- pulses so simulator has potential entries
        if signal_col not in df.columns:
            df[signal_col] = 0
        # deterministic tiny pattern
        idx = np.arange(len(df))
        df.loc[(idx % 97) == 0, signal_col] = 1
        df.loc[(idx % 223) == 0, signal_col] = -1
        # Add baseline if requested
        if signal_col == "baseline_signal" and "baseline" not in df.columns:
            df["baseline"] = df["close"].rolling(50, min_periods=1).mean()
        return df
    monkeypatch.setattr(indicators_cache, "_call_indicator", _call_indicator_stub, raising=True)

    # 6) Deterministic simulator that produces trades and per-bar realized equity
    def simulate_pair_trades_stub(rows, pair, cfg, equity_state, return_equity=True):
        rows = rows.sort_values("date").reset_index(drop=True)
        n = len(rows)
        k = max(6, n // 120)  # a few trades per fold
        idxs = np.linspace(5, n - 5, k, dtype=int)

        trades = []
        pnl_daily = np.zeros(n, dtype=float)
        for i, idx in enumerate(idxs):
            pnl = float((i % 3 - 1) * 20.0)  # -20, 0, +20 cycle
            win = int(pnl > 0)
            loss = int(pnl < 0)
            scratch = int(pnl == 0)
            exit_dt = pd.to_datetime(rows.loc[idx, "date"])
            trades.append({
                "pair": pair,
                "pnl": pnl,
                "win": win,
                "loss": loss,
                "scratch": scratch,
                "exit_date": exit_dt,
            })
            pnl_daily[idx:] += pnl / max(1, (n - idx))

        eq = None
        if return_equity:
            realized_cum = np.cumsum(pnl_daily)
            eq = pd.DataFrame({
                "date": rows["date"],
                "pair": pair,
                "pnl_realized_cum": realized_cum
            })
        return trades, eq

    monkeypatch.setattr(backtester, "simulate_pair_trades", simulate_pair_trades_stub, raising=True)

    # --- Run WFO ---
    out_dir = tmp_path / "results" / "wfo_smoke_pytest"
    out_dir.mkdir(parents=True, exist_ok=True)
    backtester.run_backtest_walk_forward(config_path=str(cfg_path), results_dir=str(out_dir))

    # Check artifacts
    folds_csv   = out_dir / "wfo_folds.csv"
    trades_csv  = out_dir / "trades.csv"
    equity_csv  = out_dir / "equity_curve.csv"
    summary_txt = out_dir / "oos_summary.txt"

    assert folds_csv.exists(), "wfo_folds.csv not written"
    assert trades_csv.exists(), "trades.csv not written"
    assert summary_txt.exists(), "oos_summary.txt not written"

    # Equity should exist because our simulator returns per-bar equity
    assert equity_csv.exists(), "equity_curve.csv not written"
    eq = pd.read_csv(equity_csv)
    assert not eq.empty and "equity" in eq.columns

    # --- Run Monte Carlo (per-trade mode for robustness) ---
    from backtester import load_config
    from analytics.monte_carlo import run_monte_carlo

    cfg = load_config(str(cfg_path))
    # keep horizon=oos; per-trade shuffling is set in cfg writer above
    summary = run_monte_carlo(cfg)
    mc_path = out_dir / "mc_summary.txt"
    assert mc_path.exists(), "mc_summary.txt not written"
    # Sanity keys
    assert "roi_pct_median" in summary and "maxdd_pct_median" in summary
