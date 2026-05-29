"""Bar fetch via the MetaTrader5 Python library.

The library is Windows-only. Tests mock it via the ``mt5_module`` injection
point — ``fetch_h4_bars(..., mt5_module=fake)`` accepts any object with
``copy_rates_from_pos``, ``initialize``, ``shutdown``, ``last_error`` and
the ``TIMEFRAME_*`` constants.

Returned DataFrame schema (matches ``signals.lchar_dlr_long.compute_signal``
expectation):

  - ``date``  : ``datetime64[ns]``, UTC-naive (matches data/cache/utc/* parquet)
  - ``open``  : float64
  - ``high``  : float64
  - ``low``   : float64
  - ``close`` : float64

MT5's ``copy_rates_from_pos`` returns a structured ndarray with field
names ``time, open, high, low, close, tick_volume, spread, real_volume``.
We rename ``time`` → ``date``, cast unix epoch seconds to UTC-naive
datetime64[ns], keep only the OHLC columns.

Failure semantics (dispatch §1.7):
  - On any return value that isn't a populated ndarray, raises
    :class:`Mt5FetchError`. Caller decides whether to skip the cycle for
    that pair, retry, or alert.
  - The exponential-backoff reconnect lives in :func:`with_mt5_session`
    (used by the sidecar main loop).
"""

from __future__ import annotations

import importlib
import time
from typing import Any, Protocol

import pandas as pd

from deployment.sidecar.boundary import _EET_TZ, CONVENTION_UTC


class Mt5FetchError(RuntimeError):
    """Raised on MT5 fetch failure (no bars, connection drop, type error)."""


class Mt5Module(Protocol):
    """Minimal subset of the MetaTrader5 library the sidecar uses."""

    TIMEFRAME_H4: int
    TIMEFRAME_D1: int

    def initialize(self, *args: Any, **kwargs: Any) -> bool: ...
    def shutdown(self) -> None: ...
    def copy_rates_from_pos(
        self, symbol: str, timeframe: int, start_pos: int, count: int
    ) -> Any: ...
    def last_error(self) -> tuple[int, str]: ...


def import_mt5() -> Mt5Module:
    """Import the production MetaTrader5 module.

    Raises ImportError on non-Windows / no-library platforms; tests should
    inject a fake module via the ``mt5_module=`` parameter instead.
    """
    return importlib.import_module("MetaTrader5")  # type: ignore[return-value]


def _bar_time_to_utc_naive(
    epoch_series: pd.Series, convention: str
) -> pd.Series:
    """Map MT5 ``time`` epoch seconds to the canonical UTC-naive ``date`` column.

    MT5's ``copy_rates_from_pos`` reports ``time`` as the bar-open instant in
    the **broker server's wall clock**, encoded as "seconds since 1970 as if
    that wall clock were UTC".

      - ``"utc"`` (5ers): the server runs UTC, so the wall clock IS UTC and the
        epoch decodes directly to true UTC. Preserved verbatim.
      - ``"5ers_eet"`` (FundedNext): the server runs broker EET/EEST, so the
        decoded value is the Europe/Athens local wall clock. We localise it to
        Europe/Athens and convert to true UTC. DST (+2 winter / +3 summer) is
        resolved by the IANA tz db — byte-identical to the lab's EET aggregator,
        which also anchors on Europe/Athens. H4/D1 anchors avoid the ambiguous
        transition hour, so localisation is never ambiguous/non-existent.
    """
    naive_wall = pd.to_datetime(epoch_series, unit="s")  # broker wall clock, naive
    if convention == CONVENTION_UTC:
        return naive_wall.astype("datetime64[ns]")
    # EET: interpret the broker wall clock as Europe/Athens local → true UTC.
    return (
        naive_wall.dt.tz_localize(
            _EET_TZ, ambiguous=True, nonexistent="shift_forward"
        )
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
    )


def _rates_to_df(
    rates: Any, *, source: str, convention: str = CONVENTION_UTC
) -> pd.DataFrame:
    """Convert MT5 ``copy_rates_from_pos`` ndarray output to canonical DataFrame.

    Raises Mt5FetchError if ``rates`` is None, empty, or missing required
    fields. ``convention`` selects broker-wall-clock → UTC normalisation
    (see :func:`_bar_time_to_utc_naive`).
    """
    if rates is None:
        raise Mt5FetchError(f"{source}: copy_rates_from_pos returned None")
    if len(rates) == 0:
        raise Mt5FetchError(f"{source}: copy_rates_from_pos returned 0 bars")
    # numpy structured array → DataFrame (MT5 ships pandas-friendly dtypes).
    df = pd.DataFrame(rates)
    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise Mt5FetchError(f"{source}: missing fields {sorted(missing)} (got {list(df.columns)})")
    # Normalise the broker-server bar-open time to a UTC-naive datetime64[ns]
    # ``date`` column matching the lab cache parquet schema and convention.
    df = df.assign(
        date=_bar_time_to_utc_naive(df["time"], convention),
        open=df["open"].astype(float),
        high=df["high"].astype(float),
        low=df["low"].astype(float),
        close=df["close"].astype(float),
    )
    return df[["date", "open", "high", "low", "close"]].reset_index(drop=True)


def fetch_h4_bars(
    symbol: str,
    *,
    count: int,
    mt5_module: Mt5Module,
    convention: str = CONVENTION_UTC,
) -> pd.DataFrame:
    """Fetch the most recent ``count`` H4 bars for ``symbol``.

    ``mt5_module`` is the live ``MetaTrader5`` (or a test fake). The
    function does NOT call ``initialize``/``shutdown`` — that lifecycle
    is the caller's (see :func:`with_mt5_session`). ``convention`` selects
    broker-time normalisation.
    """
    tf_h4 = mt5_module.TIMEFRAME_H4
    rates = mt5_module.copy_rates_from_pos(symbol, tf_h4, 0, int(count))
    return _rates_to_df(rates, source=f"H4/{symbol}", convention=convention)


def fetch_d1_bars(
    symbol: str,
    *,
    count: int,
    mt5_module: Mt5Module,
    convention: str = CONVENTION_UTC,
) -> pd.DataFrame:
    """Fetch the most recent ``count`` D1 bars for ``symbol``."""
    tf_d1 = mt5_module.TIMEFRAME_D1
    rates = mt5_module.copy_rates_from_pos(symbol, tf_d1, 0, int(count))
    return _rates_to_df(rates, source=f"D1/{symbol}", convention=convention)


def with_mt5_initialize(
    mt5_module: Mt5Module,
    *,
    initial_backoff_sec: float = 1.0,
    max_backoff_sec: float = 60.0,
    alert_after_failures: int = 3,
    sleep_func: Any = time.sleep,
) -> bool:
    """Call ``mt5_module.initialize()`` with exponential backoff.

    Returns True on success. Raises Mt5FetchError if it never connects
    after ``alert_after_failures`` consecutive failures.

    ``sleep_func`` is injectable for tests.
    """
    backoff = float(initial_backoff_sec)
    failures = 0
    while True:
        ok = bool(mt5_module.initialize())
        if ok:
            return True
        failures += 1
        if failures >= alert_after_failures:
            err = mt5_module.last_error()
            raise Mt5FetchError(
                f"MT5 initialize failed {failures} times; last_error={err!r}"
            )
        sleep_func(backoff)
        backoff = min(backoff * 2.0, float(max_backoff_sec))


__all__ = (
    "Mt5FetchError",
    "Mt5Module",
    "fetch_d1_bars",
    "fetch_h4_bars",
    "import_mt5",
    "with_mt5_initialize",
)
