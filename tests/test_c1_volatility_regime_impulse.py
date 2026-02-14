# tests/test_c1_volatility_regime_impulse.py — C1 VRSI indicator tests.

from __future__ import annotations

import pandas as pd
import pytest

from indicators.confirmation_funcs import c1_volatility_regime_impulse
from tests import _indicator_contracts as ic

SIGNAL_COL = "c1_signal"
VALID_SIGNALS = {-1, 0, 1}


@pytest.fixture(scope="module")
def test_df():
    """Load deterministic OHLCV for VRSI tests; skip if data/daily missing."""
    try:
        return ic.load_test_df()
    except FileNotFoundError:
        pytest.skip("data/daily missing or empty; need OHLCV CSV for VRSI tests")


def test_returns_dataframe_aligned(test_df):
    """VRSI must return a DataFrame with same index and length as input."""
    out = c1_volatility_regime_impulse(test_df.copy(), signal_col=SIGNAL_COL)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(test_df)
    assert out.index.equals(test_df.index)


def test_signal_values_in_valid_set(test_df):
    """VRSI must write c1_signal with values only in {-1, 0, 1}."""
    out = c1_volatility_regime_impulse(test_df.copy(), signal_col=SIGNAL_COL)
    assert SIGNAL_COL in out.columns
    warmup = 250
    if len(out) > warmup:
        s = out[SIGNAL_COL].iloc[warmup:]
        uniq = set(pd.to_numeric(s, errors="coerce").dropna().astype(int).unique())
        assert uniq <= VALID_SIGNALS, f"Signal values must be in {VALID_SIGNALS}; got {uniq}"


def test_no_lookahead(test_df):
    """VRSI must not use future bars (causality)."""
    if len(test_df) <= 20:
        pytest.skip("test_df too short for causality probe")
    ic.run_causality_probe(
        c1_volatility_regime_impulse,
        test_df,
        call_kwargs={"signal_col": SIGNAL_COL},
        signal_col=SIGNAL_COL,
        future_window=20,
    )


def test_determinism(test_df):
    """Same input -> identical output."""
    df1 = test_df.copy()
    df2 = test_df.copy()
    out1 = c1_volatility_regime_impulse(df1, signal_col=SIGNAL_COL)
    out2 = c1_volatility_regime_impulse(df2, signal_col=SIGNAL_COL)
    pd.testing.assert_series_equal(
        out1[SIGNAL_COL], out2[SIGNAL_COL], check_names=True, check_exact=False
    )


def test_parameter_elasticity(test_df):
    """Varying theta_rv or b_min changes output for at least one bar."""
    base_params = dict(
        signal_col=SIGNAL_COL,
        theta_rv=1.15,
        theta_atr=1.08,
        b_min=0.4,
        gamma=0.6,
        L_dc=15,
    )
    out_base = c1_volatility_regime_impulse(test_df.copy(), **base_params)
    params_rv = dict(base_params, theta_rv=1.4)
    params_b = dict(base_params, b_min=0.55)
    out_rv = c1_volatility_regime_impulse(test_df.copy(), **params_rv)
    out_b = c1_volatility_regime_impulse(test_df.copy(), **params_b)
    diff_rv = (out_base[SIGNAL_COL] != out_rv[SIGNAL_COL]).any()
    diff_b = (out_base[SIGNAL_COL] != out_b[SIGNAL_COL]).any()
    assert diff_rv or diff_b, "Parameter change must alter at least one bar"


def test_frequency_sanity(test_df):
    """With relaxed params, signal rate between 0.5% and 20% on fixture."""
    relaxed = dict(
        signal_col=SIGNAL_COL,
        theta_rv=1.15,
        theta_atr=1.08,
        b_min=0.4,
        gamma=0.6,
        L_dc=15,
    )
    out = c1_volatility_regime_impulse(test_df.copy(), **relaxed)
    s = pd.to_numeric(out[SIGNAL_COL], errors="coerce")
    non_zero = (s != 0) & s.notna()
    rate = non_zero.sum() / len(out)
    assert rate >= 0.005, f"Signal rate {rate:.4f} too low (not all zeros)"
    assert rate <= 0.20, f"Signal rate {rate:.4f} too high (not always-on)"
