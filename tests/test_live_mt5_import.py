"""Tests for live.mt5_import: MT5 space-delimited CSV format, symbol extraction from filenames."""


import pandas as pd
import pytest

from live.mt5_import import (
    canonical_df_for_engine,
    load_market_tsvs,
    parse_mt5_date,
    symbol_from_filename,
    symbol_to_config_pair,
)


def test_symbol_from_filename_audcad():
    assert symbol_from_filename("AUDCAD_Daily.csv") == "AUDCAD"


def test_symbol_from_filename_eurusd_range():
    assert symbol_from_filename("EURUSD_Daily_202401020000_202602020000.csv") == "EURUSD"


def test_symbol_from_filename_eur_usd():
    assert symbol_from_filename("EUR_USD.csv") == "EURUSD"


def test_symbol_from_filename_gbpjpy():
    assert symbol_from_filename("GBPJPY.csv") == "GBPJPY"


def test_symbol_to_config_pair():
    assert symbol_to_config_pair("EURUSD") == "EUR_USD"
    assert symbol_to_config_pair("AUDCAD") == "AUD_CAD"


def test_parse_mt5_date():
    s = pd.Series(["2024.01.02", "2024.01.03"])
    out = parse_mt5_date(s)
    assert len(out) == 2
    assert out.iloc[0].year == 2024 and out.iloc[0].month == 1 and out.iloc[0].day == 2
    assert out.iloc[1].year == 2024 and out.iloc[1].month == 1 and out.iloc[1].day == 3


def test_load_market_tsvs_parses_space_delimited(tmp_path):
    """MT5-exported market files are space-delimited (single-column appearance)."""
    market_dir = tmp_path / "market"
    market_dir.mkdir()
    csv_content = (
        "DATE OPEN HIGH LOW CLOSE TICKVOL VOL SPREAD\n"
        "2024.01.02 0.90216 0.90560 0.90037 0.90045 55645 0 41\n"
        "2024.01.03 0.90100 0.90400 0.89900 0.90300 60000 0 40\n"
    )
    (market_dir / "AUDCAD_Daily.csv").write_text(csv_content)
    result = load_market_tsvs(market_dir)
    assert "AUDCAD" in result
    df = result["AUDCAD"]
    assert list(df.columns) >= ["time", "open", "high", "low", "close", "spread_points"]
    assert len(df) == 2
    assert df["open"].iloc[0] == pytest.approx(0.90216)
    assert df["close"].iloc[1] == pytest.approx(0.90300)
    assert df["spread_points"].iloc[0] == 41
    assert pd.Timestamp(df["time"].iloc[0]).year == 2024


def test_canonical_df_for_engine():
    df = pd.DataFrame({
        "time": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "open": [1.0, 1.1],
        "high": [1.2, 1.3],
        "low": [0.9, 1.0],
        "close": [1.05, 1.15],
        "spread_points": [10, 11],
        "volume": [100, 200],
    })
    out = canonical_df_for_engine(df)
    assert "date" in out.columns
    assert "open" in out.columns
    assert "volume" in out.columns
    assert len(out) == 2
