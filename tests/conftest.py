import pandas as pd
import numpy as np
import datetime as dt
import pytest

@pytest.fixture
def dummy_ohlcv():
    n = 120
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.Series(np.linspace(100, 110, n)) + np.random.normal(0, 0.2, n)
    high  = close + 0.3
    low   = close - 0.3
    open_ = close.shift(1).fillna(close.iloc[0])
    vol   = pd.Series(1000, index=range(n))
    df = pd.DataFrame({"date":dates,"open":open_,"high":high,"low":low,"close":close,"volume":vol})
    return df
