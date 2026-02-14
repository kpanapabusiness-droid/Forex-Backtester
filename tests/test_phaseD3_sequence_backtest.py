"""
Phase D-3 Sequence backtest — synthetic tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_ohlc(n_bars: int = 30, pair: str = "EUR_USD") -> pd.DataFrame:
    """Build minimal OHLC with ATR-computable data."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=n_bars, freq="D")
    close = 1.1000 + np.cumsum(np.random.randn(n_bars) * 0.0005)
    high = close + np.abs(np.random.randn(n_bars) * 0.001)
    low = close - np.abs(np.random.randn(n_bars) * 0.001)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    })


def _synthetic_ohlc_tp1_then_return_to_entry(n_bars: int = 22, base: float = 1.1000, atr_val: float = 0.001) -> pd.DataFrame:
    """
    TP1 hits then price returns to entry. With BE: exit breakeven_after_tp1. Without BE: stay until SL.
    Entry bar 15 open. Bar 15: TP1. Bar 17: return to entry (BE hit if enabled). Bar 18: hit original SL.
    """
    dates = pd.date_range("2022-01-01", periods=n_bars, freq="D")
    open_ = np.zeros(n_bars)
    high = np.zeros(n_bars)
    low = np.zeros(n_bars)
    close = np.zeros(n_bars)
    for i in range(15):
        open_[i] = base + 0.0001 * max(0, i - 1)
        close[i] = base + 0.0001 * i
        high[i] = close[i] + 0.0005
        low[i] = close[i] - 0.0005
    open_[0] = close[0]
    entry_open = base + 0.0001 * 14
    tp1_px = entry_open + atr_val
    sl_px = entry_open - 1.5 * atr_val
    open_[15] = entry_open
    high[15] = tp1_px + 0.0002
    low[15] = sl_px + 0.0003
    close[15] = tp1_px
    open_[16] = close[15]
    high[16] = tp1_px
    low[16] = entry_open
    close[16] = entry_open + 0.0001
    open_[17] = close[16]
    high[17] = entry_open + 0.0002
    low[17] = entry_open - 0.00005
    close[17] = entry_open
    open_[18] = close[17]
    high[18] = entry_open
    low[18] = sl_px - 0.0001
    close[18] = sl_px
    for i in range(19, n_bars):
        open_[i] = close[i - 1]
        close[i] = base
        high[i] = base + 0.0002
        low[i] = base - 0.0002
    df = pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close})
    df["atr"] = atr_val
    return df


def _synthetic_ohlc_slow_upward(n_bars: int = 50, base: float = 1.1000) -> pd.DataFrame:
    """
    Uptrend to bar 38 then pullback. Long at bar 14: TP1 hits, BE exit at bar 41+.
    """
    dates = pd.date_range("2022-01-01", periods=n_bars, freq="D")
    t = np.arange(n_bars, dtype=float)
    up = 0.00012 * np.minimum(t, 38)
    down = 0.00025 * np.maximum(t - 38, 0)
    close = base + up - down
    high = close + 0.0005
    low = close - 0.0002
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close})


def _synthetic_signals(n_bars: int, entry_at: int, direction: str = "long") -> pd.DataFrame:
    """One entry at entry_at for the given direction."""
    dates = pd.date_range("2022-01-01", periods=n_bars, freq="D")
    rows = []
    for i, d in enumerate(dates):
        for dir_name in ("long", "short"):
            sig = 1 if (i == entry_at and dir_name == direction) else 0
            rows.append({
                "pair": "EUR_USD",
                "date": d,
                "direction": dir_name,
                "signal": sig,
                "signal_name": "test_signal",
            })
    return pd.DataFrame(rows)


def test_entry_at_next_open(tmp_path: Path) -> None:
    """Signal at bar t -> entry at bar t+1 open."""
    from core.backtester import simulate_pair_trades
    from core.utils import calculate_atr

    df = _synthetic_ohlc(30)
    df = calculate_atr(df)
    df["pair"] = "EUR_USD"
    df["entry_signal"] = 0
    df.loc[5, "entry_signal"] = 1
    df["exit_signal"] = 0

    cfg = {
        "entry": {"sl_atr": 1.5, "tp1_atr": 1.0, "trail_after_atr": 2.0, "ts_atr": 1.5},
        "exit": {"exit_on_c1_reversal": False, "time_stop_bars": 20},
        "risk": {"starting_balance": 10000.0, "risk_per_trade_pct": 0.25},
        "spreads": {"enabled": False},
        "filters": {"dbcvix": {"enabled": False}},
    }
    equity = {"balance": 10000.0}

    trades = simulate_pair_trades(df, "EUR_USD", cfg, equity, return_equity=False)

    if trades:
        t = trades[0]
        assert t["entry_date"] is not None
        entry_idx = int(t["entry_idx"])
        assert entry_idx == 5
        assert t["entry_price"] == float(df.iloc[6]["open"])


def test_time_stop_exit_at_n_bars(tmp_path: Path) -> None:
    """Position exits at next open after N bars (time stop)."""
    from core.backtester import simulate_pair_trades
    from core.utils import calculate_atr

    n_bars = 35
    df = _synthetic_ohlc(n_bars)
    df = calculate_atr(df)
    df["pair"] = "EUR_USD"
    df["entry_signal"] = 0
    df.loc[2, "entry_signal"] = 1
    df["exit_signal"] = 0

    cfg = {
        "entry": {"sl_atr": 1.5, "tp1_atr": 1.0, "trail_after_atr": 2.0, "ts_atr": 1.5},
        "exit": {"exit_on_c1_reversal": False, "time_stop_bars": 20},
        "risk": {"starting_balance": 10000.0, "risk_per_trade_pct": 0.25},
        "spreads": {"enabled": False},
        "filters": {"dbcvix": {"enabled": False}},
    }
    equity = {"balance": 10000.0}

    trades = simulate_pair_trades(df, "EUR_USD", cfg, equity, return_equity=False)

    if trades:
        t = trades[0]
        assert t["exit_reason"] == "time_stop"
        entry_idx = int(t["entry_idx"])
        assert entry_idx == 2
        exit_date = pd.to_datetime(t["exit_date"])
        entry_date = pd.to_datetime(t["entry_date"])
        bars_held = (exit_date - entry_date).days
        assert bars_held >= 19


def test_sl_uses_atr_mult(tmp_path: Path) -> None:
    """SL distance is ATR(t) * 1.5."""
    from core.backtester import simulate_pair_trades
    from core.utils import calculate_atr

    df = _synthetic_ohlc(30)
    df = calculate_atr(df)
    df["pair"] = "EUR_USD"
    df["entry_signal"] = 0
    df.loc[5, "entry_signal"] = 1
    df["exit_signal"] = 0

    cfg = {
        "entry": {"sl_atr": 1.5, "tp1_atr": 1.0, "trail_after_atr": 2.0, "ts_atr": 1.5},
        "exit": {"exit_on_c1_reversal": False, "time_stop_bars": 20},
        "risk": {"starting_balance": 10000.0, "risk_per_trade_pct": 0.25},
        "spreads": {"enabled": False},
        "filters": {"dbcvix": {"enabled": False}},
    }
    equity = {"balance": 10000.0}

    trades = simulate_pair_trades(df, "EUR_USD", cfg, equity, return_equity=False)

    if trades:
        t = trades[0]
        atr = float(df.iloc[5]["atr"])
        entry_px = float(t["entry_price"])
        sl_px = float(t["sl_at_entry_price"])
        expected_sl_dist = 1.5 * atr
        actual_sl_dist = abs(entry_px - sl_px)
        assert abs(actual_sl_dist - expected_sl_dist) < 1e-6


def test_determinism(tmp_path: Path) -> None:
    """Repeated run produces same outputs."""
    from core.backtester import simulate_pair_trades
    from core.utils import calculate_atr

    df = _synthetic_ohlc(30)
    df = calculate_atr(df)
    df["pair"] = "EUR_USD"
    df["entry_signal"] = 0
    df.loc[5, "entry_signal"] = 1
    df["exit_signal"] = 0

    cfg = {
        "entry": {"sl_atr": 1.5, "tp1_atr": 1.0, "trail_after_atr": 2.0, "ts_atr": 1.5},
        "exit": {"exit_on_c1_reversal": False, "time_stop_bars": 20},
        "risk": {"starting_balance": 10000.0, "risk_per_trade_pct": 0.25},
        "spreads": {"enabled": False},
        "filters": {"dbcvix": {"enabled": False}},
    }
    equity1 = {"balance": 10000.0}
    equity2 = {"balance": 10000.0}

    trades1 = simulate_pair_trades(df.copy(), "EUR_USD", cfg, equity1, return_equity=False)
    trades2 = simulate_pair_trades(df.copy(), "EUR_USD", cfg, equity2, return_equity=False)

    assert len(trades1) == len(trades2)
    if trades1:
        for k in ["entry_price", "exit_price", "pnl", "exit_reason"]:
            assert trades1[0].get(k) == trades2[0].get(k)


def test_no_time_stop_trade_remains_open_past_20_bars(tmp_path: Path) -> None:
    """Phase D-4: With time_stop disabled, trade stays open past 20 bars if SL not hit."""
    from core.backtester import simulate_pair_trades
    from core.utils import calculate_atr

    n_bars = 50
    df = _synthetic_ohlc_slow_upward(n_bars)
    df = calculate_atr(df)
    df["pair"] = "EUR_USD"
    df["entry_signal"] = 0
    df.loc[14, "entry_signal"] = 1
    df["exit_signal"] = 0

    cfg = {
        "entry": {"sl_atr": 1.5, "tp1_atr": 1.0, "trail_after_atr": 2.0, "ts_atr": 1.5},
        "exit": {"exit_on_c1_reversal": False, "time_stop_bars": None},
        "risk": {"starting_balance": 10000.0, "risk_per_trade_pct": 0.25},
        "spreads": {"enabled": False},
        "filters": {"dbcvix": {"enabled": False}},
    }
    equity = {"balance": 10000.0}

    trades = simulate_pair_trades(df, "EUR_USD", cfg, equity, return_equity=False)

    assert len(trades) >= 1, "Expected at least one trade to close"
    t = trades[0]
    entry_idx = int(t["entry_idx"])
    exit_date = pd.to_datetime(t["exit_date"])
    entry_date = pd.to_datetime(t["entry_date"])
    bars_held = (exit_date - entry_date).days
    assert bars_held > 20, (
        f"Phase D-4: Trade should remain open past 20 bars (time_stop disabled). "
        f"Got bars_held={bars_held}, entry_idx={entry_idx}"
    )
    assert t["exit_reason"] != "time_stop", "Time stop must be disabled for D-4"


def test_tp1_move_sl_to_be_false_keeps_runner(tmp_path: Path) -> None:
    """Phase D-4.1: With tp1_move_sl_to_be=false, runner does NOT exit at entry after TP1."""
    from core.backtester import simulate_pair_trades

    df = _synthetic_ohlc_tp1_then_return_to_entry()
    df["pair"] = "EUR_USD"
    df["entry_signal"] = 0
    df.loc[14, "entry_signal"] = 1
    df["exit_signal"] = 0

    cfg_be = {
        "entry": {"sl_atr": 1.5, "tp1_atr": 1.0, "trail_after_atr": 2.0, "ts_atr": 1.5},
        "exit": {"exit_on_c1_reversal": False, "time_stop_bars": None, "tp1_move_sl_to_be": True},
        "risk": {"starting_balance": 10000.0, "risk_per_trade_pct": 0.25},
        "spreads": {"enabled": False},
        "filters": {"dbcvix": {"enabled": False}},
    }
    cfg_no_be = {**cfg_be, "exit": {**cfg_be["exit"], "tp1_move_sl_to_be": False}}
    equity = {"balance": 10000.0}

    trades_be = simulate_pair_trades(df.copy(), "EUR_USD", cfg_be, dict(equity), return_equity=False)
    trades_no_be = simulate_pair_trades(df.copy(), "EUR_USD", cfg_no_be, dict(equity), return_equity=False)

    assert len(trades_be) >= 1 and len(trades_no_be) >= 1
    t_be = trades_be[0]
    t_no_be = trades_no_be[0]

    assert t_be["tp1_hit"], "TP1 should be hit in both runs"
    assert t_no_be["tp1_hit"], "TP1 should be hit in both runs"

    assert t_be["exit_reason"] == "breakeven_after_tp1", (
        f"With BE enabled, expected breakeven_after_tp1, got {t_be['exit_reason']}"
    )
    assert t_no_be["exit_reason"] != "breakeven_after_tp1", (
        f"With tp1_move_sl_to_be=false, exit_reason must not be breakeven_after_tp1, got {t_no_be['exit_reason']}"
    )
    assert not t_no_be.get("breakeven_after_tp1", True), (
        "With tp1_move_sl_to_be=false, breakeven_after_tp1 should be False"
    )


def test_validate_d4_rejects_time_stop(tmp_path: Path) -> None:
    """Phase D-4 validation raises when time_stop_bars > 0."""
    import pytest

    from scripts.phaseD3_run_sequence_backtest import _validate_d4_no_time_stop

    cfg_ok = {"exit": {"time_stop_bars": None}}
    _validate_d4_no_time_stop(cfg_ok)

    cfg_ok0 = {"exit": {"time_stop_bars": 0}}
    _validate_d4_no_time_stop(cfg_ok0)

    cfg_bad = {"exit": {"time_stop_bars": 20}}
    with pytest.raises(ValueError, match="Phase D-4 requires time_stop_bars disabled"):
        _validate_d4_no_time_stop(cfg_bad)


def test_require_config_uses_signal_from_config_when_cli_omit() -> None:
    """When --signal not passed, use external_signals.signal_name from config."""
    from scripts.phaseD3_run_sequence_backtest import _require_config

    raw = {
        "external_signals": {"path": "x.parquet", "signal_name": "seq_comp3_tratr90"},
    }
    cfg = _require_config(raw, None)
    assert cfg["external_signal_name"] == "seq_comp3_tratr90"
    cfg2 = _require_config(raw, "other_signal")
    assert cfg2["external_signal_name"] == "other_signal"


def test_validate_d41_requires_tp1_move_sl(tmp_path: Path) -> None:
    """Phase D-4.1: validation raises when tp1_move_sl_to_be is missing."""
    import pytest

    from scripts.phaseD3_run_sequence_backtest import _validate_d41_tp1_move_sl_required

    cfg_ok = {"exit": {"tp1_move_sl_to_be": False}}
    _validate_d41_tp1_move_sl_required(cfg_ok)

    cfg_missing = {"exit": {"time_stop_bars": None}}
    with pytest.raises(ValueError, match="tp1_move_sl_to_be"):
        _validate_d41_tp1_move_sl_required(cfg_missing)


def test_apply_breakout_filter_long_kept_short_blocked() -> None:
    """Phase D-4.2: long with breakout_up=1 kept, short with breakout_dn=0 blocked."""
    from scripts.phaseD3_run_sequence_backtest import _apply_breakout_filter

    signals = pd.DataFrame([
        {"pair": "EUR_USD", "date": pd.Timestamp("2022-01-10"), "direction": "long", "signal": 1, "signal_name": "seq_comp3_tratr90"},
        {"pair": "EUR_USD", "date": pd.Timestamp("2022-01-10"), "direction": "short", "signal": 1, "signal_name": "seq_comp3_tratr90"},
        {"pair": "EUR_USD", "date": pd.Timestamp("2022-01-15"), "direction": "short", "signal": 1, "signal_name": "seq_comp3_tratr90"},
    ])
    features = pd.DataFrame([
        {"pair": "EUR_USD", "date": pd.Timestamp("2022-01-10"), "breakout_up_20": 1.0, "breakout_dn_20": 0.0},
        {"pair": "EUR_USD", "date": pd.Timestamp("2022-01-15"), "breakout_up_20": 0.0, "breakout_dn_20": 0.0},
    ])
    out = _apply_breakout_filter(
        signals, features, breakout_up_col="breakout_up_20", breakout_dn_col="breakout_dn_20"
    )
    long_row = out[(out["direction"] == "long") & (out["date"] == pd.Timestamp("2022-01-10"))]
    short_10 = out[(out["direction"] == "short") & (out["date"] == pd.Timestamp("2022-01-10"))]
    short_15 = out[(out["direction"] == "short") & (out["date"] == pd.Timestamp("2022-01-15"))]
    assert int(long_row.iloc[0]["signal"]) == 1, "Long with breakout_up=1 must remain 1"
    assert int(short_10.iloc[0]["signal"]) == 0, "Short with breakout_dn=0 must become 0"
    assert int(short_15.iloc[0]["signal"]) == 0, "Short with breakout_dn=0 must become 0"


def test_entry_filter_validation_features_path_required(tmp_path: Path) -> None:
    """Phase D-4.2: require_breakout_confirm without features_path raises."""
    import pytest

    from scripts.phaseD3_run_sequence_backtest import _load_and_validate_entry_filter

    cfg = {"entry_filter": {"require_breakout_confirm": True}}  # no features_path
    signals = pd.DataFrame([{"pair": "X", "date": pd.Timestamp("2022-01-01")}])
    with pytest.raises(ValueError, match="features_path"):
        _load_and_validate_entry_filter(cfg, signals)


def test_apply_breakout_filter_nan_fail_closed() -> None:
    """Phase D-4.2: NaN breakout value -> signal becomes 0 (fail closed)."""
    from scripts.phaseD3_run_sequence_backtest import _apply_breakout_filter

    signals = pd.DataFrame([
        {"pair": "EUR_USD", "date": pd.Timestamp("2022-01-20"), "direction": "long", "signal": 1, "signal_name": "seq_comp3_tratr90"},
    ])
    features = pd.DataFrame([
        {"pair": "EUR_USD", "date": pd.Timestamp("2022-01-20"), "breakout_up_20": np.nan, "breakout_dn_20": 0.0},
    ])
    out = _apply_breakout_filter(
        signals, features, breakout_up_col="breakout_up_20", breakout_dn_col="breakout_dn_20"
    )
    assert int(out.iloc[0]["signal"]) == 0, "NaN breakout_up must fail closed (signal=0)"


def test_signals_to_entry_intent() -> None:
    from scripts.phaseD3_run_sequence_backtest import _signals_to_entry_intent

    sig = pd.DataFrame([
        {"pair": "X", "date": pd.Timestamp("2022-01-01"), "direction": "long", "signal": 1},
        {"pair": "X", "date": pd.Timestamp("2022-01-01"), "direction": "short", "signal": 0},
    ])
    out = _signals_to_entry_intent(sig)
    assert len(out) == 1
    assert out.iloc[0]["entry_signal"] == 1

    sig2 = pd.DataFrame([
        {"pair": "X", "date": pd.Timestamp("2022-01-02"), "direction": "short", "signal": 1},
    ])
    out2 = _signals_to_entry_intent(sig2)
    assert out2.iloc[0]["entry_signal"] == -1
