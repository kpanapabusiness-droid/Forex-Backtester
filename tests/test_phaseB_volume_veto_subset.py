# Phase B volume comparisons must maintain: volume-on entries subset of volume-off.
from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.backtester import run_backtest


def _identity_set(trades: pd.DataFrame) -> set:
    if trades.empty:
        return set()
    cols = ["pair", "entry_date", "direction_int", "entry_price"]
    for c in cols:
        if c not in trades.columns:
            return set()
    return set(
        tuple(trades[c].iloc[i] for c in cols)
        for i in range(len(trades))
    )


def test_phaseB_volume_veto_subset_invariant(tmp_path: Path) -> None:
    """Volume-on entries must be a subset of volume-off (same C1 baseline). Use phase2-style configs."""
    off_cfg = {
        "pairs": ["EUR_USD"],
        "timeframe": "D",
        "indicators": {"c1": "c1_coral", "use_volume": False, "use_c2": False, "use_baseline": False, "use_exit": False},
        "date_range": {"start": "2019-01-01", "end": "2026-01-01"},
        "spreads": {"enabled": True, "default_pips": 0.5},
        "entry": {"sl_atr": 1.5, "tp1_atr": 1.0, "trail_after_atr": 2.0, "ts_atr": 1.5},
        "exit": {"use_trailing_stop": False, "exit_on_c1_reversal": True},
        "risk": {"starting_balance": 10000.0, "risk_per_trade": 0.005},
    }
    on_cfg = {**off_cfg, "indicators": {**off_cfg["indicators"], "use_volume": True, "volume": "volatility_ratio"}}
    off_dir = tmp_path / "off"
    on_dir = tmp_path / "on"
    off_dir.mkdir()
    on_dir.mkdir()
    run_backtest(off_cfg, results_dir=str(off_dir))
    run_backtest(on_cfg, results_dir=str(on_dir))
    toff = pd.read_csv(off_dir / "trades.csv") if (off_dir / "trades.csv").exists() else pd.DataFrame()
    ton = pd.read_csv(on_dir / "trades.csv") if (on_dir / "trades.csv").exists() else pd.DataFrame()
    off_ids = _identity_set(toff)
    on_ids = _identity_set(ton)
    assert len(ton) <= len(toff), "Volume ON must not create more trades than OFF"
    missing = on_ids - off_ids
    assert not missing, f"Volume ON must not add trades not in OFF: {len(missing)} extra"
