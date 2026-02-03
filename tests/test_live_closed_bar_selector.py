"""Tests for closed-bar selection in live.run_daily."""


import pandas as pd

from live.run_daily import _select_closed_bar_index


def test_closed_bar_selector_uses_second_last_when_last_is_today():
    """If last row is 'today', use the previous (closed) bar."""
    today = pd.Timestamp.now("UTC").normalize()
    yesterday = today - pd.Timedelta(days=1)

    df = pd.DataFrame(
        {
            "date": [yesterday, today],
            "entry_signal": [0, 1],
            "exit_signal": [0, 0],
            "atr": [0.001, 0.001],
            "close": [1.0, 1.1],
        }
    )

    idx = _select_closed_bar_index(df, today=today)
    assert idx == 0


def test_closed_bar_selector_uses_last_when_no_today_row():
    """If last row is already the last closed bar, use the last row."""
    today = pd.Timestamp.now("UTC").normalize()
    yesterday = today - pd.Timedelta(days=1)
    two_days_ago = today - pd.Timedelta(days=2)

    df = pd.DataFrame(
        {
            "date": [two_days_ago, yesterday],
            "entry_signal": [0, 1],
            "exit_signal": [0, 0],
            "atr": [0.001, 0.001],
            "close": [1.0, 1.1],
        }
    )

    idx = _select_closed_bar_index(df, today=today)
    assert idx == 1


def test_closed_bar_selector_uses_second_last_when_last_equals_export_day():
    """Stale export: last bar date == export day (file mtime day) -> use second-to-last."""
    feb1 = pd.Timestamp("2025-02-01").normalize()
    feb2 = pd.Timestamp("2025-02-02").normalize()

    df = pd.DataFrame(
        {
            "date": [feb1, feb2],
            "entry_signal": [0, 1],
            "exit_signal": [0, 0],
            "atr": [0.001, 0.001],
            "close": [1.0, 1.1],
        }
    )

    idx = _select_closed_bar_index(df, export_day="2025-02-02")
    assert idx == 0


def test_closed_bar_selector_uses_last_when_last_after_export_day():
    """Fresh export: last bar date after export day -> use last row."""
    feb1 = pd.Timestamp("2025-02-01").normalize()
    feb2 = pd.Timestamp("2025-02-02").normalize()

    df = pd.DataFrame(
        {
            "date": [feb1, feb2],
            "entry_signal": [0, 1],
            "exit_signal": [0, 0],
            "atr": [0.001, 0.001],
            "close": [1.0, 1.1],
        }
    )

    idx = _select_closed_bar_index(df, export_day="2025-02-01")
    assert idx == 1

