"""Test MT5 data-fetcher schema + backoff semantics with a fake module."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from deployment.sidecar.boundary import CONVENTION_EET, CONVENTION_UTC
from deployment.sidecar.mt5_data_fetcher import (
    Mt5FetchError,
    _rates_to_df,
    fetch_d1_bars,
    fetch_h4_bars,
    with_mt5_initialize,
)


def _broker_epoch(y, m, d, h):
    """MT5 ``time`` for a broker wall-clock instant: the wall-clock numbers
    encoded as 'seconds since 1970 as if UTC'."""
    return int(datetime(y, m, d, h, tzinfo=timezone.utc).timestamp())


def test_fetch_h4_returns_canonical_dataframe(fake_mt5):
    df = fetch_h4_bars("EURUSD", count=50, mt5_module=fake_mt5)
    assert list(df.columns) == ["date", "open", "high", "low", "close"]
    assert df["date"].dtype == np.dtype("datetime64[ns]")
    assert df["date"].dt.tz is None
    assert df["open"].dtype == np.float64
    assert df["close"].dtype == np.float64
    assert len(df) == 50
    # Call signature was as expected.
    assert fake_mt5.copy_calls[-1] == ("EURUSD", fake_mt5.TIMEFRAME_H4, 0, 50)


def test_fetch_d1_returns_canonical_dataframe(fake_mt5):
    df = fetch_d1_bars("EURUSD", count=30, mt5_module=fake_mt5)
    assert list(df.columns) == ["date", "open", "high", "low", "close"]
    assert len(df) == 30
    assert fake_mt5.copy_calls[-1] == ("EURUSD", fake_mt5.TIMEFRAME_D1, 0, 30)


def test_fetch_raises_on_none_rates(fake_mt5, monkeypatch):
    monkeypatch.setattr(fake_mt5, "copy_rates_from_pos", lambda *a, **k: None)
    with pytest.raises(Mt5FetchError, match="returned None"):
        fetch_h4_bars("EURUSD", count=10, mt5_module=fake_mt5)


def test_fetch_raises_on_empty_rates(fake_mt5, monkeypatch):
    empty = np.array([], dtype=np.dtype([("time", "i8")]))
    monkeypatch.setattr(fake_mt5, "copy_rates_from_pos", lambda *a, **k: empty)
    with pytest.raises(Mt5FetchError, match="returned 0 bars"):
        fetch_h4_bars("EURUSD", count=10, mt5_module=fake_mt5)


def test_fetch_raises_on_missing_field(fake_mt5, monkeypatch):
    bad = np.array(
        [(1234, 1.0)], dtype=np.dtype([("time", "i8"), ("open", "f8")])
    )
    monkeypatch.setattr(fake_mt5, "copy_rates_from_pos", lambda *a, **k: bad)
    with pytest.raises(Mt5FetchError, match="missing fields"):
        fetch_h4_bars("EURUSD", count=10, mt5_module=fake_mt5)


def test_initialize_succeeds_first_try(fake_mt5):
    sleeps: list[float] = []
    ok = with_mt5_initialize(
        fake_mt5,
        initial_backoff_sec=1,
        max_backoff_sec=8,
        alert_after_failures=3,
        sleep_func=sleeps.append,
    )
    assert ok is True
    assert fake_mt5.initialize_calls == 1
    assert sleeps == []  # never slept


def test_initialize_backs_off_then_raises():
    from tests.sidecar.conftest import FakeMt5

    fake = FakeMt5(init_returns=False, last_error_value=(404, "no terminal"))
    sleeps: list[float] = []
    with pytest.raises(Mt5FetchError, match="initialize failed"):
        with_mt5_initialize(
            fake,
            initial_backoff_sec=1,
            max_backoff_sec=8,
            alert_after_failures=3,
            sleep_func=sleeps.append,
        )
    assert fake.initialize_calls == 3
    assert sleeps == [1, 2]  # backoff 1s then 2s before 3rd attempt


def test_h4_timestamps_are_utc_naive_and_anchored(fake_mt5):
    """The synthetic H4 panel in conftest is anchored on 00:00 UTC; the
    fetched DataFrame should preserve that anchor."""
    df = fetch_h4_bars("EURUSD", count=24, mt5_module=fake_mt5)
    first = df["date"].iloc[0]
    assert pd.Timestamp(first).tz is None
    assert pd.Timestamp(first).hour in (0, 4, 8, 12, 16, 20)
    assert pd.Timestamp(first).minute == 0


def _rates(epochs):
    return [
        {"time": e, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05} for e in epochs
    ]


def test_utc_convention_passes_broker_time_through_unchanged():
    """5ers: broker server runs UTC, so the epoch decodes directly to UTC."""
    rates = _rates([_broker_epoch(2013, 1, 14, 0)])
    df = _rates_to_df(rates, source="t", convention=CONVENTION_UTC)
    assert pd.Timestamp(df["date"].iloc[0]) == pd.Timestamp("2013-01-14 00:00:00")


def test_eet_convention_winter_normalizes_to_true_utc():
    """FundedNext winter (UTC+2): broker wall-clock Mon 00:00 EET-local must
    normalise to the true UTC instant Sun 22:00."""
    rates = _rates([_broker_epoch(2013, 1, 14, 0)])
    df = _rates_to_df(rates, source="t", convention=CONVENTION_EET)
    assert pd.Timestamp(df["date"].iloc[0]) == pd.Timestamp("2013-01-13 22:00:00")
    assert df["date"].dtype == np.dtype("datetime64[ns]")
    assert df["date"].dt.tz is None


def test_eet_convention_summer_normalizes_to_true_utc():
    """FundedNext summer (UTC+3): broker wall-clock 12:00 EEST-local → 09:00 UTC."""
    rates = _rates([_broker_epoch(2024, 7, 8, 12)])
    df = _rates_to_df(rates, source="t", convention=CONVENTION_EET)
    assert pd.Timestamp(df["date"].iloc[0]) == pd.Timestamp("2024-07-08 09:00:00")


def test_eet_convention_resolves_dst_per_bar():
    """A single fetch spanning the spring-forward boundary applies +2 before and
    +3 after, resolved by the tz db (broker wall-clock 12:00 each day)."""
    rates = _rates(
        [_broker_epoch(2019, 3, 29, 12), _broker_epoch(2019, 4, 1, 12)]
    )
    df = _rates_to_df(rates, source="t", convention=CONVENTION_EET)
    # 2019-03-29 winter: 12:00 EET → 10:00 UTC.
    assert pd.Timestamp(df["date"].iloc[0]) == pd.Timestamp("2019-03-29 10:00:00")
    # 2019-04-01 summer: 12:00 EEST → 09:00 UTC.
    assert pd.Timestamp(df["date"].iloc[1]) == pd.Timestamp("2019-04-01 09:00:00")
