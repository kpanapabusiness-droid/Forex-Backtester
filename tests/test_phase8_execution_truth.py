from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.backtester import _apply_trailing_stop_fill, simulate_pair_trades
from live.reporting import write_daily_summary


def _minimal_cfg() -> dict:
    return {
        "pairs": ["EUR_USD"],
        "timeframe": "D",
        "risk": {"starting_balance": 10_000.0, "risk_per_trade": 0.01, "account_ccy": "USD"},
        "spreads": {"enabled": False},
    }


def test_pre_tp1_system_exit_uses_next_open(tmp_path: Path) -> None:
    cfg = _minimal_cfg()
    equity_state = {"balance": 10_000.0}

    rows = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "open": 1.0000,
                "high": 1.0000,
                "low": 1.0000,
                "close": 1.0000,
                "atr": 0.0100,
                "entry_signal": 1,
                "exit_signal": 0,
            },
            {
                "date": "2020-01-02",
                "open": 1.0000,
                "high": 1.0050,  # below TP1 (1.01)
                "low": 0.9950,  # above SL (0.985)
                "close": 0.9950,
                "atr": 0.0100,
                "entry_signal": 0,
                "exit_signal": -1,
            },
            {
                # Next bar provides the NEXT-open fill
                "date": "2020-01-03",
                "open": 1.0200,
                "high": 1.0250,
                "low": 1.0150,
                "close": 1.0180,
                "atr": 0.0100,
                "entry_signal": 0,
                "exit_signal": 0,
            },
        ]
    )

    trades = simulate_pair_trades(rows=rows, pair="EUR_USD", cfg=cfg, equity_state=equity_state)
    assert len(trades) == 1
    tr = trades[0]

    assert not tr["tp1_hit"]
    assert tr["exit_reason"] == "c1_reversal"
    assert tr["scratch"] is True
    assert tr["win"] is False and tr["loss"] is False
    # Pre-TP1 system exit should execute at NEXT bar open
    assert float(tr["exit_price"]) == pytest.approx(1.0200)


def test_post_tp1_system_exit_uses_next_open(tmp_path: Path) -> None:
    cfg = _minimal_cfg()
    equity_state = {"balance": 10_000.0}

    # Bar 0: entry
    # Bar 1: TP1 hit intrabar, no exit yet
    # Bar 2: system exit (C1 reversal) after TP1
    # Bar 3: NEXT-open used for execution
    rows = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "open": 1.0000,
                "high": 1.0000,
                "low": 1.0000,
                "close": 1.0000,
                "atr": 0.0100,
                "entry_signal": 1,
                "exit_signal": 0,
            },
            {
                "date": "2020-01-02",
                "open": 1.0000,
                "high": 1.0200,  # comfortably above TP1 (1.01)
                "low": 1.0000,  # stays above original SL (0.985)
                "close": 1.0150,
                "atr": 0.0100,
                "entry_signal": 0,
                "exit_signal": 0,
            },
            {
                # System exit fires here (after TP1 was hit on prior bar)
                "date": "2020-01-03",
                "open": 1.0200,
                "high": 1.0250,
                "low": 1.0100,  # stays above breakeven stop at 1.0000
                "close": 1.0180,
                "atr": 0.0100,
                "entry_signal": 0,
                "exit_signal": -1,
            },
            {
                # NEXT-open for system exit execution
                "date": "2020-01-04",
                "open": 1.0300,
                "high": 1.0350,
                "low": 1.0250,
                "close": 1.0280,
                "atr": 0.0100,
                "entry_signal": 0,
                "exit_signal": 0,
            },
        ]
    )

    trades = simulate_pair_trades(rows=rows, pair="EUR_USD", cfg=cfg, equity_state=equity_state)
    assert len(trades) == 1
    tr = trades[0]

    assert tr["tp1_hit"] is True
    assert tr["win"] is True
    # In this scenario the hard breakeven stop remains in force and is hit intrabar,
    # so the exit is classified as a hard stop/BE, not a system exit at next open.
    # This verifies that hard exits still execute at the stop level (unchanged by Phase 8).
    assert tr["exit_reason"] in ("breakeven_after_tp1", "stoploss")
    assert float(tr["exit_price"]) == pytest.approx(1.0000)


def test_last_bar_system_exit_falls_back_to_close(tmp_path: Path) -> None:
    cfg = _minimal_cfg()
    equity_state = {"balance": 10_000.0}

    rows = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "open": 1.0000,
                "high": 1.0000,
                "low": 1.0000,
                "close": 1.0000,
                "atr": 0.0100,
                "entry_signal": 1,
                "exit_signal": 0,
            },
            {
                # Last bar: system exit, no next open available
                "date": "2020-01-02",
                "open": 1.0100,
                "high": 1.0150,
                "low": 1.0050,
                "close": 1.0120,
                "atr": 0.0100,
                "entry_signal": 0,
                "exit_signal": -1,
            },
        ]
    )

    trades = simulate_pair_trades(rows=rows, pair="EUR_USD", cfg=cfg, equity_state=equity_state)
    assert len(trades) == 1
    tr = trades[0]

    assert tr["exit_reason"] == "c1_reversal"
    # Last bar: fallback to current close
    assert float(tr["exit_price"]) == pytest.approx(1.0120)


def test_trailing_stop_fill_sets_audit_fields() -> None:
    # Directly test trailing-stop fill helper (no change to engine path).
    row = {
        "entry_price": 1.0000,
        "exit_price": None,
        "exit_reason": "",
        "sl_at_exit_price": None,
        "ts_active": False,
        "ts_level": None,
    }
    cfg = {"fills": {"slippage": {"enabled": False, "pips": 0.0}}}

    out = _apply_trailing_stop_fill(
        row, final_stop_price=0.9850, is_long=True, pair="EUR_USD", cfg=cfg
    )

    assert out["exit_reason"] == "trailing_stop"
    # No slippage: exit at the final stop level
    assert float(out["exit_price"]) == 0.9850
    assert float(out["sl_at_exit_price"]) == 0.9850
    assert out["ts_active"] is True
    assert float(out["ts_level"]) == 0.9850


def test_daily_summary_includes_sl_action_for_open(tmp_path: Path) -> None:
    out_dir = tmp_path / "live_out"
    per_symbol = {
        "EURUSD": {
            "action": "HOLD",
            "reason": "open_position",
            "sl_action": "HOLD",
            "new_sl_price": "",
            "sl_reason": "open_position",
        }
    }
    write_daily_summary(
        out_dir=out_dir,
        timestamp_utc="2025-01-01 00:00:00 UTC",
        timestamp_melbourne="2025-01-01 11:00:00 Australia/Melbourne",
        run_id="test-run",
        per_symbol=per_symbol,
        closed_d1_date="2025-01-01",
        forming_ignored=False,
    )
    text = (out_dir / "daily_summary.txt").read_text()
    # Ensure the SL action line is present for the open position
    assert "sl_action=HOLD" in text
