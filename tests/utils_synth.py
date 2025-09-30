"""
Synthetic data utilities for Golden Standard logic testing.
Creates deterministic OHLC scenarios to trigger specific trading rules.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def _deep_update(base: dict, patch: dict) -> None:
    """Deep update base dict with patch dict in-place."""
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def create_synthetic_ohlc(
    bars: List[Dict[str, Any]], pair: str = "EUR_USD", atr_value: float = 0.002
) -> pd.DataFrame:
    """
    Create synthetic OHLC DataFrame with indicator signals.

    Args:
        bars: List of dicts with keys: date, open, high, low, close, c1_signal, etc.
        pair: Currency pair name
        atr_value: Fixed ATR value for all bars

    Returns:
        DataFrame ready for backtesting with all required columns
    """
    df = pd.DataFrame(bars)

    # Ensure required columns exist
    required_cols = {
        "date": "2023-01-01",
        "open": 1.0000,
        "high": 1.0000,
        "low": 1.0000,
        "close": 1.0000,
        "volume": 1000,
        "atr": atr_value,
        "c1_signal": 0,
        "c2_signal": 0,
        "baseline": 1.0000,
        "baseline_signal": 0,
        "volume_signal": 1,
        "exit_signal": 0,
    }

    for col, default in required_cols.items():
        if col not in df.columns:
            df[col] = default

    # Generate sequential dates if not provided
    if "date" not in bars[0] or not bars[0]["date"]:
        base_date = pd.to_datetime("2023-01-01")
        df["date"] = [base_date + pd.Timedelta(days=i) for i in range(len(bars))]
    else:
        df["date"] = pd.to_datetime(df["date"])

    df["pair"] = pair

    # Ensure numeric types
    numeric_cols = ["open", "high", "low", "close", "volume", "atr", "baseline"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure signal columns are integers
    signal_cols = ["c1_signal", "c2_signal", "baseline_signal", "volume_signal", "exit_signal"]
    for col in signal_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df.sort_values("date").reset_index(drop=True)


def run_synthetic_backtest(
    df: pd.DataFrame, config_overrides: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Run backtest on synthetic data with minimal config.

    Args:
        df: Synthetic OHLC DataFrame
        config_overrides: Config modifications

    Returns:
        Dict with trades, equity_curve, and summary data
    """
    from core.backtester import simulate_pair_trades
    from core.signal_logic import apply_signal_logic

    # Minimal config for synthetic tests
    base_config = {
        "pairs": [df["pair"].iloc[0]],
        "timeframe": "D",
        "indicators": {
            "c1": "synthetic",  # Signals already in DF
            "use_c2": False,
            "use_baseline": True,
            "baseline": "synthetic",
            "use_volume": False,
            "use_exit": False,
        },
        "rules": {
            "one_candle_rule": False,
            "pullback_rule": False,
            "bridge_too_far_days": 7,
            "allow_baseline_as_catalyst": False,
        },
        "entry": {"sl_atr": 1.5, "tp1_atr": 1.0, "trail_after_atr": 2.0, "ts_atr": 1.5},
        "engine": {
            "allow_continuation": False,
            "duplicate_open_policy": "block",
        },
        "exit": {
            "use_trailing_stop": True,
            "move_to_breakeven_after_atr": True,
            "exit_on_c1_reversal": True,
            "exit_on_baseline_cross": False,
            "exit_on_exit_signal": False,
        },
        "risk": {
            "risk_per_trade": 0.02,
            "starting_balance": 10000.0,
            "account_ccy": "USD",
            "fx_quotes": {},
        },
        "spreads": {"enabled": False, "default_pips": 1.0},
        "tracking": {"in_sim_equity": True},
    }

    # Apply overrides (deep merge for nested configs)
    if config_overrides:
        _deep_update(base_config, config_overrides)

    # Apply signal logic (pass-through for synthetic data)
    signals_df = apply_signal_logic(df.copy(), base_config)

    # Run simulation
    equity_state = {"balance": base_config["risk"]["starting_balance"]}

    trades, equity_curve = simulate_pair_trades(
        rows=signals_df,
        pair=df["pair"].iloc[0],
        cfg=base_config,
        equity_state=equity_state,
        return_equity=True,
    )

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "config": base_config,
        "signals_df": signals_df,
    }
