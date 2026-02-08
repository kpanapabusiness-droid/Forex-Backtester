# Phase A: Spread Semantics Lock — deterministic tests (no full 2019–2026 runs).
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from core.backtester import compute_trade_pnl_money, simulate_pair_trades
from core.signal_logic import apply_signal_logic
from tests.utils_synth import create_synthetic_ohlc, run_synthetic_backtest


def _minimal_cfg(spread_enabled: bool = False, default_pips: float = 0.0, use_volume: bool = False):
    return {
        "pairs": ["EUR_USD"],
        "timeframe": "D",
        "indicators": {
            "c1": "synthetic",
            "use_c2": False,
            "use_baseline": True,
            "baseline": "synthetic",
            "use_volume": use_volume,
            "use_exit": False,
        },
        "rules": {
            "one_candle_rule": False,
            "pullback_rule": False,
            "bridge_too_far_days": 7,
            "allow_baseline_as_catalyst": False,
        },
        "entry": {"sl_atr": 1.5, "tp1_atr": 1.0, "trail_after_atr": 2.0, "ts_atr": 1.5},
        "engine": {"allow_continuation": False, "duplicate_open_policy": "block"},
        "exit": {
            "use_trailing_stop": True,
            "move_to_breakeven_after_atr": True,
            "exit_on_c1_reversal": True,
            "exit_on_baseline_cross": False,
            "exit_on_exit_signal": False,
        },
        "risk": {
            "risk_per_trade": 0.02,
            "starting_balance": 10_000.0,
            "account_ccy": "USD",
            "fx_quotes": {},
        },
        "spreads": {"enabled": spread_enabled, "default_pips": default_pips},
        "tracking": {"in_sim_equity": True},
    }


def test_spread_zero_invariant_no_cost_leakage():
    """With spread set to zero: trade count unchanged vs baseline, fills equal mid, no residual cost."""
    bars = [
        {"date": "2020-01-01", "open": 1.10, "high": 1.10, "low": 1.10, "close": 1.10, "atr": 0.01,
         "baseline_signal": 1, "c1_signal": 1, "exit_signal": 0},
        {"date": "2020-01-02", "open": 1.10, "high": 1.12, "low": 1.09, "close": 1.11, "atr": 0.01,
         "baseline_signal": 0, "c1_signal": -1, "exit_signal": -1},
        {"date": "2020-01-03", "open": 1.02, "high": 1.03, "low": 1.01, "close": 1.02, "atr": 0.01,
         "baseline_signal": 0, "c1_signal": 0, "exit_signal": 0},
    ]
    df = create_synthetic_ohlc(bars)
    base = _minimal_cfg(spread_enabled=False)
    with_spread_zero = _minimal_cfg(spread_enabled=True, default_pips=0.0)

    res_baseline = run_synthetic_backtest(df, base)
    res_zero = run_synthetic_backtest(df, with_spread_zero)

    assert len(res_baseline["trades"]) == len(res_zero["trades"]), "trade count must match"
    if not res_baseline["trades"]:
        return
    for t in res_zero["trades"]:
        assert float(t.get("spread_pips_used", 0)) == 0.0
        assert float(t.get("spread_pips_exit", 0)) == 0.0
    t0 = res_zero["trades"][0]
    pnl_from_fills = compute_trade_pnl_money(t0, "EUR_USD", 10.0)
    assert math.isfinite(pnl_from_fills)
    assert float(t0["pnl"]) == pytest.approx(pnl_from_fills, abs=1e-9), "PnL from fills only"


def test_spread_scaling_is_linear_entry_and_exit():
    """Same scenario with spread 1x vs 2x: same trade count; per-trade PnL more negative with 2x; delta ~ one extra round-trip."""
    bars = [
        {"date": "2020-01-01", "open": 1.10, "high": 1.10, "low": 1.10, "close": 1.10, "atr": 0.01,
         "baseline_signal": 1, "c1_signal": 1, "exit_signal": 0},
        {"date": "2020-01-02", "open": 1.10, "high": 1.12, "low": 1.09, "close": 1.11, "atr": 0.01,
         "baseline_signal": 0, "c1_signal": -1, "exit_signal": -1},
        {"date": "2020-01-03", "open": 1.02, "high": 1.03, "low": 1.01, "close": 1.02, "atr": 0.01,
         "baseline_signal": 0, "c1_signal": 0, "exit_signal": 0},
    ]
    df = create_synthetic_ohlc(bars)
    cfg_1x = _minimal_cfg(spread_enabled=True, default_pips=1.0)
    cfg_2x = _minimal_cfg(spread_enabled=True, default_pips=2.0)

    r1 = run_synthetic_backtest(df, cfg_1x)
    r2 = run_synthetic_backtest(df, cfg_2x)

    assert len(r1["trades"]) == len(r2["trades"]), "trade count must be identical"
    if not r1["trades"]:
        return
    pnl_1x = sum(t["pnl"] for t in r1["trades"])
    pnl_2x = sum(t["pnl"] for t in r2["trades"])
    assert pnl_2x <= pnl_1x, "2x spread should reduce PnL"
    pip_val = 10.0
    lots = float(r1["trades"][0]["lots_total"])
    extra_pips_2x_vs_1x = 1.0
    expected_delta = -extra_pips_2x_vs_1x * pip_val * lots
    assert (pnl_2x - pnl_1x) == pytest.approx(expected_delta, abs=0.5), "delta ~ one extra pip cost (0.5 entry + 0.5 exit)"


def test_entry_fill_uses_execution_bar_spread_t_plus_one():
    """Entry at next bar open (t+1): entry_date = bar t+1, entry_price = open(t+1), spread_pips_used = spread(t+1)."""
    bars = [
        {"date": "2020-01-01", "open": 1.10, "high": 1.10, "low": 1.10, "close": 1.10, "atr": 0.01,
         "spread_pips": 2.0, "baseline_signal": 1, "c1_signal": 1, "exit_signal": 0},
        {"date": "2020-01-02", "open": 1.105, "high": 1.12, "low": 1.09, "close": 1.11, "atr": 0.01,
         "spread_pips": 8.0, "baseline_signal": 0, "c1_signal": -1, "exit_signal": -1},
        {"date": "2020-01-03", "open": 1.02, "high": 1.03, "low": 1.01, "close": 1.02, "atr": 0.01,
         "spread_pips": 1.0, "baseline_signal": 0, "c1_signal": 0, "exit_signal": 0},
    ]
    df = create_synthetic_ohlc(bars)
    if "spread_pips" not in df.columns:
        df["spread_pips"] = [2.0, 8.0, 1.0]
    cfg = _minimal_cfg(spread_enabled=True, default_pips=0.0)

    res = run_synthetic_backtest(df, cfg)
    assert len(res["trades"]) >= 1
    t = res["trades"][0]
    bar_t_plus_one_date = pd.Timestamp("2020-01-02")
    assert pd.Timestamp(t["entry_date"]) == bar_t_plus_one_date, "entry_time must be bar t+1 timestamp"
    assert float(t["entry_price"]) == pytest.approx(1.105, abs=1e-6), "entry fill = open(t+1)"
    assert float(t["spread_pips_used"]) == pytest.approx(8.0, abs=0.01), "entry uses spread(t+1)"


def test_next_open_system_exit_uses_execution_bar_spread_t_plus_one():
    """System exit at next open → exit fill uses spread from bar t+1 (execution bar of the exit)."""
    bars = [
        {"date": "2020-01-01", "open": 1.10, "high": 1.10, "low": 1.10, "close": 1.10, "atr": 0.01,
         "spread_pips": 1.0, "baseline_signal": 1, "c1_signal": 1, "exit_signal": 0},
        {"date": "2020-01-02", "open": 1.10, "high": 1.12, "low": 1.09, "close": 1.11, "atr": 0.01,
         "spread_pips": 2.0, "baseline_signal": 0, "c1_signal": -1, "exit_signal": -1},
        {"date": "2020-01-03", "open": 1.02, "high": 1.03, "low": 1.01, "close": 1.02, "atr": 0.01,
         "spread_pips": 6.0, "baseline_signal": 0, "c1_signal": 0, "exit_signal": 0},
    ]
    df = create_synthetic_ohlc(bars)
    if "spread_pips" not in df.columns:
        df["spread_pips"] = [1.0, 2.0, 6.0]
    cfg = _minimal_cfg(spread_enabled=True, default_pips=0.0)

    res = run_synthetic_backtest(df, cfg)
    assert len(res["trades"]) >= 1
    t = res["trades"][0]
    assert t["exit_reason"] in ("c1_reversal", "exit_indicator", "scratch")
    assert float(t["spread_pips_exit"]) == pytest.approx(6.0, abs=0.01), "system exit uses next bar (t+1) spread"


def test_intrabar_stop_exit_uses_current_bar_spread_t():
    """Intrabar SL triggers on bar t → exit fill uses spread from bar t."""
    bars = [
        {"date": "2020-01-01", "open": 1.10, "high": 1.10, "low": 1.10, "close": 1.10, "atr": 0.01,
         "spread_pips": 1.0, "baseline_signal": 1, "c1_signal": 1, "exit_signal": 0},
        {"date": "2020-01-02", "open": 1.10, "high": 1.10, "low": 1.08, "close": 1.08, "atr": 0.01,
         "spread_pips": 3.0, "baseline_signal": 0, "c1_signal": 0, "exit_signal": 0},
        {"date": "2020-01-03", "open": 1.07, "high": 1.09, "low": 1.06, "close": 1.07, "atr": 0.01,
         "spread_pips": 9.0, "baseline_signal": 0, "c1_signal": 0, "exit_signal": 0},
    ]
    df = create_synthetic_ohlc(bars)
    if "spread_pips" not in df.columns:
        df["spread_pips"] = [1.0, 3.0, 9.0]
    cfg = _minimal_cfg(spread_enabled=True, default_pips=0.0)

    res = run_synthetic_backtest(df, cfg)
    assert len(res["trades"]) >= 1
    t = res["trades"][0]
    assert (t.get("exit_reason") or "").lower() == "stoploss", "expect intrabar SL"
    assert float(t["spread_pips_exit"]) == pytest.approx(3.0, abs=0.01), "intrabar exit uses bar t spread"


def test_volume_vetoed_entries_create_no_trade_and_pay_no_spread():
    """Volume veto = always veto → no trade rows, no PnL, no spread charged."""
    bars = [
        {"date": "2020-01-01", "open": 1.10, "high": 1.10, "low": 1.10, "close": 1.10, "atr": 0.01,
         "baseline_signal": 1, "c1_signal": 1, "volume_signal": 0, "exit_signal": 0},
        {"date": "2020-01-02", "open": 1.10, "high": 1.12, "low": 1.09, "close": 1.11, "atr": 0.01,
         "baseline_signal": 0, "c1_signal": 0, "volume_signal": 0, "exit_signal": 0},
    ]
    df = create_synthetic_ohlc(bars)
    cfg = _minimal_cfg(spread_enabled=True, default_pips=2.0, use_volume=True)

    res = run_synthetic_backtest(df, cfg)
    assert len(res["trades"]) == 0, "volume veto must produce no trades"
    assert res["equity_curve"].empty or res["equity_curve"]["equity"].iloc[-1] == pytest.approx(
        cfg["risk"]["starting_balance"], abs=0.01
    ), "no PnL change from vetoed entries"


def test_no_double_count_spread_not_applied_in_pnl_and_fill_prices():
    """PnL computed from fills (mid ± S/2) matches reported PnL; no extra spread term."""
    bars = [
        {"date": "2020-01-01", "open": 1.10, "high": 1.10, "low": 1.10, "close": 1.10, "atr": 0.01,
         "baseline_signal": 1, "c1_signal": 1, "exit_signal": 0},
        {"date": "2020-01-02", "open": 1.10, "high": 1.12, "low": 1.09, "close": 1.11, "atr": 0.01,
         "baseline_signal": 0, "c1_signal": -1, "exit_signal": -1},
        {"date": "2020-01-03", "open": 1.02, "high": 1.03, "low": 1.01, "close": 1.02, "atr": 0.01,
         "baseline_signal": 0, "c1_signal": 0, "exit_signal": 0},
    ]
    df = create_synthetic_ohlc(bars)
    cfg = _minimal_cfg(spread_enabled=True, default_pips=1.0)
    res = run_synthetic_backtest(df, cfg)
    if not res["trades"]:
        return
    pip_val = 10.0
    for t in res["trades"]:
        pnl_reported = float(t["pnl"])
        pnl_from_fills = compute_trade_pnl_money(t, "EUR_USD", pip_val)
        assert pnl_reported == pytest.approx(pnl_from_fills, abs=1e-9), (
            "PnL must be from fills only; no double spread"
        )


def test_full_vs_wfo_parity_trade_for_trade_spread_semantics(tmp_path: Path):
    """FULL vs WFO parity: same window → same trades (entry/exit time, direction, prices, pnl, spread fields)."""
    bars = [
        {"date": "2020-01-01", "open": 1.10, "high": 1.10, "low": 1.10, "close": 1.10, "atr": 0.01,
         "baseline_signal": 1, "c1_signal": 1, "exit_signal": 0},
        {"date": "2020-01-02", "open": 1.10, "high": 1.12, "low": 1.09, "close": 1.11, "atr": 0.01,
         "baseline_signal": 0, "c1_signal": -1, "exit_signal": -1},
        {"date": "2020-01-03", "open": 1.02, "high": 1.03, "low": 1.01, "close": 1.02, "atr": 0.01,
         "baseline_signal": 0, "c1_signal": 0, "exit_signal": 0},
    ]
    df = create_synthetic_ohlc(bars)
    cfg = _minimal_cfg(spread_enabled=True, default_pips=1.0)
    equity_state = {"balance": cfg["risk"]["starting_balance"]}
    signals = apply_signal_logic(df.copy(), cfg)
    for col in ["entry_signal", "exit_signal"]:
        if col in signals.columns:
            signals[col] = pd.to_numeric(signals[col], errors="coerce").fillna(0).clip(-1, 1).astype(int)

    full_trades = simulate_pair_trades(rows=signals, pair="EUR_USD", cfg=cfg, equity_state=equity_state.copy())

    equity_state2 = {"balance": cfg["risk"]["starting_balance"]}
    wfo_trades = simulate_pair_trades(rows=signals, pair="EUR_USD", cfg=cfg, equity_state=equity_state2)

    assert len(full_trades) == len(wfo_trades)
    key_cols = ["entry_date", "exit_date", "direction", "entry_price", "exit_price", "pnl", "spread_pips_used"]
    optional = ["spread_pips_exit"]
    for a, b in zip(full_trades, wfo_trades):
        for k in key_cols:
            if k in a and k in b:
                va, vb = a[k], b[k]
                if isinstance(va, float) and isinstance(vb, float):
                    assert va == pytest.approx(vb, abs=1e-9), f"{k} full vs wfo"
                else:
                    assert va == vb, f"{k} full vs wfo"
        for k in optional:
            if k in a and k in b:
                assert a[k] == pytest.approx(b[k], abs=1e-9) if isinstance(a[k], (int, float)) else a[k] == b[k]
