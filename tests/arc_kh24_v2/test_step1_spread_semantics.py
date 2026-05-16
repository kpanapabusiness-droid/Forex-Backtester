"""Spread semantics — KH-24 v2.0 Step 1 plumbing.

Verifies that for every emitted trade:
  - entry_price (long) = open_mid(N+1) + S(N+1)/2
  - hard_sl exit_price = sl_at_entry_price - S(exit_bar)/2     (intrabar; current-bar spread)
  - bar_240 exit_price = open_mid(N+1+240) - S(N+1+240)/2      (execution-bar spread)
"""

from __future__ import annotations

import numpy as np

from scripts.arc_kh24_v2.step1._signal import SignalParams, evaluate_bare_signal
from scripts.arc_kh24_v2.step1._simulate import (
    POINTS_PER_PIP,
    ExecParams,
    _pip_size,
    simulate_pair,
)
from tests.arc_kh24_v2._synth import make_4h_with_signal, make_d1_for_4h


def _pair_run(pair: str = "EUR_USD"):
    df_4h = make_4h_with_signal(pair=pair)
    df_d1 = make_d1_for_4h(df_4h)
    sig_mask, atr_4h, _ = evaluate_bare_signal(df_4h, df_d1, SignalParams())
    trades, paths = simulate_pair(pair, df_4h, sig_mask, atr_4h, ExecParams())
    return df_4h, trades, paths


def _entry_spread_check(pair: str = "EUR_USD"):
    df_4h, trades, _ = _pair_run(pair)
    assert len(trades) >= 1
    pip = _pip_size(pair)
    for t in trades:
        entry_idx = int(np.flatnonzero(df_4h["date"].values == np.datetime64(t["entry_time"]))[0])
        sp_pips = float(df_4h["spread"].iat[entry_idx]) / POINTS_PER_PIP
        expected = float(df_4h["open"].iat[entry_idx]) + sp_pips * pip / 2.0
        assert abs(expected - t["entry_price"]) < 1e-12
        assert abs(sp_pips - t["spread_pips_used"]) < 1e-12


def test_entry_uses_execution_bar_spread_t_plus_one():
    _entry_spread_check()


def test_jpy_pair_pip_size_applied_correctly():
    df_4h = make_4h_with_signal(pair="USD_JPY", start_price=150.0, spread_points=20)
    df_d1 = make_d1_for_4h(df_4h)
    sig_mask, atr_4h, _ = evaluate_bare_signal(df_4h, df_d1, SignalParams())
    trades, _ = simulate_pair("USD_JPY", df_4h, sig_mask, atr_4h, ExecParams())
    if not trades:
        return
    pip = _pip_size("USD_JPY")
    assert pip == 0.01
    t = trades[0]
    entry_idx = int(np.flatnonzero(df_4h["date"].values == np.datetime64(t["entry_time"]))[0])
    sp_pips = float(df_4h["spread"].iat[entry_idx]) / POINTS_PER_PIP
    expected = float(df_4h["open"].iat[entry_idx]) + sp_pips * pip / 2.0
    assert abs(expected - t["entry_price"]) < 1e-12


def test_hard_sl_exit_uses_current_bar_spread_intrabar():
    df_4h, trades, _ = _pair_run()
    # Force a synthetic SL hit by pushing a low past SL on a near-future bar.
    if not trades:
        return
    t = trades[0]
    entry_idx = int(np.flatnonzero(df_4h["date"].values == np.datetime64(t["entry_time"]))[0])
    # Find the first SL trade if present.
    sl_trades = [tr for tr in trades if tr["exit_reason"] == "hard_sl"]
    if not sl_trades:
        # Synthesise one: lower a low to trigger SL.
        df_pert = df_4h.copy()
        sl_bar = entry_idx + 5
        df_pert.loc[sl_bar, "low"] = float(t["sl_at_entry_price"]) - 0.0001
        sig_mask, atr_4h, _ = evaluate_bare_signal(df_pert, make_d1_for_4h(df_pert), SignalParams())
        trades_pert, _ = simulate_pair("EUR_USD", df_pert, sig_mask, atr_4h, ExecParams())
        sl_trades = [tr for tr in trades_pert if tr["exit_reason"] == "hard_sl"]
        df_used = df_pert
    else:
        df_used = df_4h
    assert sl_trades, "expected an SL exit after fixture perturbation"
    pip = _pip_size("EUR_USD")
    for tr in sl_trades:
        ent_idx = int(np.flatnonzero(df_used["date"].values == np.datetime64(tr["entry_time"]))[0])
        exit_idx = ent_idx + int(tr["bars_held"])
        sp_pips = float(df_used["spread"].iat[exit_idx]) / POINTS_PER_PIP
        expected = float(tr["sl_at_entry_price"]) - sp_pips * pip / 2.0
        assert abs(expected - tr["exit_price"]) < 1e-12
        assert abs(sp_pips - tr["spread_pips_exit"]) < 1e-12


def test_bar_240_system_exit_uses_execution_bar_spread():
    df_4h, trades, _ = _pair_run()
    bar240_trades = [tr for tr in trades if tr["exit_reason"] == "bar_240_system_exit"]
    if not bar240_trades:
        return
    pip = _pip_size("EUR_USD")
    for tr in bar240_trades:
        ent_idx = int(np.flatnonzero(df_4h["date"].values == np.datetime64(tr["entry_time"]))[0])
        exit_idx = ent_idx + int(tr["bars_held"])  # bar_offset = HOLD_BARS
        sp_pips = float(df_4h["spread"].iat[exit_idx]) / POINTS_PER_PIP
        expected = float(df_4h["open"].iat[exit_idx]) - sp_pips * pip / 2.0
        assert abs(expected - tr["exit_price"]) < 1e-12
        assert abs(sp_pips - tr["spread_pips_exit"]) < 1e-12


def test_sl_distance_is_exactly_2x_atr_at_signal_bar():
    _, trades, _ = _pair_run()
    if not trades:
        return
    for t in trades:
        ratio = (t["entry_price"] - t["sl_at_entry_price"]) / t["signal_bar_atr_14"]
        assert abs(ratio - 2.0) < 1e-9
