"""
Phase D-6G: Tests for signal_events.csv dump.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.phaseD6G_run_signal_geometry import _write_signal_events


def test_signal_events_dump_exists_with_required_columns(tmp_path: Path) -> None:
    """signal_events.csv exists with pair, date, signal and only non-zero signals."""
    merged = pd.DataFrame({
        "pair": ["EUR_USD", "EUR_USD", "GBP_USD", "GBP_USD"],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
        "sig": [1, -1, 0, 1],
        "valid_atr": [True] * 4,
    })
    merged_by_signal = {"sig": merged}
    _write_signal_events(merged_by_signal, tmp_path)
    out_path = tmp_path / "signal_events.csv"
    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert "pair" in df.columns
    assert "date" in df.columns
    assert "signal" in df.columns
    assert set(df["signal"].astype(int).unique()).issubset({-1, 1})
    assert len(df) == 3
    assert df[df["signal"] == 0].empty


def test_signal_events_dump_empty_when_all_zero(tmp_path: Path) -> None:
    """No file written when all signals are zero."""
    merged = pd.DataFrame({
        "pair": ["EUR_USD"],
        "date": pd.to_datetime(["2020-01-01"]),
        "sig": [0],
        "valid_atr": [True],
    })
    merged_by_signal = {"sig": merged}
    _write_signal_events(merged_by_signal, tmp_path)
    assert not (tmp_path / "signal_events.csv").exists()


def test_signal_events_dump_empty_when_no_signals(tmp_path: Path) -> None:
    """No file written when merged_by_signal is empty."""
    _write_signal_events({}, tmp_path)
    assert not (tmp_path / "signal_events.csv").exists()
