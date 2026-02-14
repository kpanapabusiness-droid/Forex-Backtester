# tests/test_c1_channel_escape_gradient.py — C1 CEG indicator tests.

from __future__ import annotations

import pandas as pd
import pytest

from indicators.confirmation_funcs import c1_channel_escape_gradient
from tests import _indicator_contracts as ic

SIGNAL_COL = "c1_signal"
VALID_SIGNALS = {-1, 0, 1}


@pytest.fixture(scope="module")
def test_df():
    """Load deterministic OHLCV for CEG tests; skip if data/daily missing."""
    try:
        return ic.load_test_df()
    except FileNotFoundError:
        pytest.skip("data/daily missing or empty; need OHLCV CSV for CEG tests")


def test_returns_dataframe_aligned(test_df):
    """CEG must return a DataFrame with same index and length as input."""
    out = c1_channel_escape_gradient(test_df.copy(), signal_col=SIGNAL_COL)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(test_df)
    assert out.index.equals(test_df.index)


def test_signal_values_in_valid_set(test_df):
    """CEG must write c1_signal with values only in {-1, 0, 1}."""
    out = c1_channel_escape_gradient(test_df.copy(), signal_col=SIGNAL_COL)
    assert SIGNAL_COL in out.columns
    warmup = 250
    if len(out) > warmup:
        s = out[SIGNAL_COL].iloc[warmup:]
        uniq = set(pd.to_numeric(s, errors="coerce").dropna().astype(int).unique())
        assert uniq <= VALID_SIGNALS, f"Signal values must be in {VALID_SIGNALS}; got {uniq}"


def test_no_lookahead(test_df):
    """CEG must not use future bars. Modify last row, earlier signals unchanged."""
    if len(test_df) <= 20:
        pytest.skip("test_df too short for causality probe")
    ic.run_causality_probe(
        c1_channel_escape_gradient,
        test_df,
        call_kwargs={"signal_col": SIGNAL_COL},
        signal_col=SIGNAL_COL,
        future_window=20,
    )


def test_determinism(test_df):
    """Same input -> identical output."""
    df1 = test_df.copy()
    df2 = test_df.copy()
    out1 = c1_channel_escape_gradient(df1, signal_col=SIGNAL_COL)
    out2 = c1_channel_escape_gradient(df2, signal_col=SIGNAL_COL)
    pd.testing.assert_series_equal(
        out1[SIGNAL_COL], out2[SIGNAL_COL], check_names=True, check_exact=False
    )


def test_parameter_elasticity(test_df):
    """Varying q_va or gamma changes output for at least one bar."""
    base_params = dict(
        signal_col=SIGNAL_COL,
        q_va=0.35,
        gamma=0.6,
        q=0.75,
        L_va=50,
        g1=3,
        g2=10,
    )
    out_base = c1_channel_escape_gradient(test_df.copy(), **base_params)
    params_q = dict(base_params, q_va=0.5)
    params_g = dict(base_params, gamma=0.72)
    out_q = c1_channel_escape_gradient(test_df.copy(), **params_q)
    out_g = c1_channel_escape_gradient(test_df.copy(), **params_g)
    diff_q = (out_base[SIGNAL_COL] != out_q[SIGNAL_COL]).any()
    diff_g = (out_base[SIGNAL_COL] != out_g[SIGNAL_COL]).any()
    assert diff_q or diff_g, "Parameter change must alter at least one bar"


def test_frequency_sanity(test_df):
    """Signal rate between 0.5% and 25% on fixture."""
    relaxed = dict(
        signal_col=SIGNAL_COL,
        q_va=0.4,
        gamma=0.55,
        q=0.75,
        L_va=50,
        L_p=100,
        g1=3,
        g2=12,
    )
    out = c1_channel_escape_gradient(test_df.copy(), **relaxed)
    s = pd.to_numeric(out[SIGNAL_COL], errors="coerce")
    non_zero = (s != 0) & s.notna()
    rate = non_zero.sum() / len(out)
    assert rate >= 0.005, f"Signal rate {rate:.4f} too low (not all zeros)"
    assert rate <= 0.25, f"Signal rate {rate:.4f} too high (not always-on)"
