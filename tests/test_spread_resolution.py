# tests/test_spread_resolution.py — resolve_spread_pips: data columns vs config fallbacks
import pandas as pd

from core.backtester import resolve_spread_pips


def test_spreads_disabled_returns_zero():
    cfg = {"spreads": {"enabled": False, "default_pips": 2.0}}
    row = pd.Series({"spread_pips": 3.0, "spread": 50.0})
    assert resolve_spread_pips("EUR_USD", row, cfg) == 0.0


def test_spread_pips_preferred_when_present():
    cfg = {"spreads": {"enabled": True, "default_pips": 1.0, "points_per_pip": 10.0}}
    row = pd.Series({"spread_pips": 2.5, "spread": 50.0})
    assert resolve_spread_pips("EUR_USD", row, cfg) == 2.5


def test_spread_column_converted_when_spread_pips_absent():
    cfg = {"spreads": {"enabled": True, "default_pips": 1.0, "points_per_pip": 10.0}}
    row = pd.Series({"spread": 50.0})
    assert resolve_spread_pips("EUR_USD", row, cfg) == 5.0


def test_spread_column_uses_points_per_pip_config():
    cfg = {"spreads": {"enabled": True, "default_pips": 1.0, "points_per_pip": 5.0}}
    row = pd.Series({"spread": 50.0})
    assert resolve_spread_pips("EUR_USD", row, cfg) == 10.0


def test_fallback_to_default_pips_when_no_data_columns():
    cfg = {"spreads": {"enabled": True, "default_pips": 1.5}}
    row = pd.Series({"open": 1.1, "close": 1.1})
    assert resolve_spread_pips("EUR_USD", row, cfg) == 1.5


def test_per_pair_override_when_no_bar_spread():
    cfg = {
        "spreads": {"enabled": True, "default_pips": 1.0, "per_pair": {"AUD_CAD": 3.0}},
    }
    row = pd.Series({"open": 1.0})
    assert resolve_spread_pips("AUD_CAD", row, cfg) == 3.0
    assert resolve_spread_pips("EUR_USD", row, cfg) == 1.0


def test_spread_pips_nan_ignored_then_spread_used():
    cfg = {"spreads": {"enabled": True, "default_pips": 1.0, "points_per_pip": 10.0}}
    row = pd.Series({"spread_pips": float("nan"), "spread": 60.0})
    assert resolve_spread_pips("EUR_USD", row, cfg) == 6.0
