import numpy as np
import pandas as pd
import pytest


def has_parquet_engine() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        pass
    try:
        import fastparquet  # noqa: F401
        return True
    except Exception:
        return False


PARQUET_SKIP = pytest.mark.skipif(
    not has_parquet_engine(),
    reason="Parquet engine (pyarrow/fastparquet) not installed in CI clean env",
)


@pytest.fixture
def dummy_ohlcv():
    n = 120
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.Series(np.linspace(100, 110, n)) + np.random.normal(0, 0.2, n)
    high = close + 0.3
    low = close - 0.3
    open_ = close.shift(1).fillna(close.iloc[0])
    vol = pd.Series(1000, index=range(n))
    df = pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": vol}
    )
    return df
