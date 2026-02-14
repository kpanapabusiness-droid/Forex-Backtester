# tests/test_c1_liquidity_sweep_rejection.py — C1 LSR indicator tests.

from __future__ import annotations

import pandas as pd
import pytest

from indicators.confirmation_funcs import c1_liquidity_sweep_rejection
from tests import _indicator_contracts as ic

SIGNAL_COL = "c1_signal"
VALID_SIGNALS = {-1, 0, 1}


@pytest.fixture(scope="module")
def test_df():
    """Load deterministic OHLCV for LSR tests; skip if data/daily missing."""
    try:
        return ic.load_test_df()
    except FileNotFoundError:
        pytest.skip("data/daily missing or empty; need OHLCV CSV for LSR tests")


def test_returns_dataframe_aligned(test_df):
    """LSR must return a DataFrame with same index and length as input."""
    out = c1_liquidity_sweep_rejection(test_df.copy(), signal_col=SIGNAL_COL)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(test_df)
    assert out.index.equals(test_df.index)


def test_signal_values_in_valid_set(test_df):
    """LSR must write c1_signal with values only in {-1, 0, 1}."""
    out = c1_liquidity_sweep_rejection(test_df.copy(), signal_col=SIGNAL_COL)
    assert SIGNAL_COL in out.columns
    warmup = 250
    if len(out) > warmup:
        s = out[SIGNAL_COL].iloc[warmup:]
        uniq = set(pd.to_numeric(s, errors="coerce").dropna().astype(int).unique())
        assert uniq <= VALID_SIGNALS, f"Signal values must be in {VALID_SIGNALS}; got {uniq}"


def test_no_lookahead(test_df):
    """LSR must not use future bars (causality). Modify last row, earlier signals unchanged."""
    if len(test_df) <= 20:
        pytest.skip("test_df too short for causality probe")
    ic.run_causality_probe(
        c1_liquidity_sweep_rejection,
        test_df,
        call_kwargs={"signal_col": SIGNAL_COL},
        signal_col=SIGNAL_COL,
        future_window=20,
    )


def test_determinism(test_df):
    """Same input -> identical output."""
    df1 = test_df.copy()
    df2 = test_df.copy()
    out1 = c1_liquidity_sweep_rejection(df1, signal_col=SIGNAL_COL)
    out2 = c1_liquidity_sweep_rejection(df2, signal_col=SIGNAL_COL)
    pd.testing.assert_series_equal(
        out1[SIGNAL_COL], out2[SIGNAL_COL], check_names=True, check_exact=False
    )


def test_parameter_elasticity(test_df):
    """Varying gamma or L_sw changes output for at least one bar."""
    base_params = dict(
        signal_col=SIGNAL_COL,
        gamma=0.6,
        L_sw=15,
        alpha=1.0,
        use_expansion_filter=True,
        use_compression_filter=False,
    )
    out_base = c1_liquidity_sweep_rejection(test_df.copy(), **base_params)
    params_g = dict(base_params, gamma=0.72)
    params_l = dict(base_params, L_sw=25)
    out_g = c1_liquidity_sweep_rejection(test_df.copy(), **params_g)
    out_l = c1_liquidity_sweep_rejection(test_df.copy(), **params_l)
    diff_g = (out_base[SIGNAL_COL] != out_g[SIGNAL_COL]).any()
    diff_l = (out_base[SIGNAL_COL] != out_l[SIGNAL_COL]).any()
    assert diff_g or diff_l, "Parameter change must alter at least one bar"


def test_frequency_sanity(test_df):
    """Signal rate between 0.5% and 25% on fixture."""
    relaxed = dict(
        signal_col=SIGNAL_COL,
        gamma=0.55,
        L_sw=15,
        alpha=1.0,
        use_expansion_filter=True,
        use_compression_filter=False,
    )
    out = c1_liquidity_sweep_rejection(test_df.copy(), **relaxed)
    s = pd.to_numeric(out[SIGNAL_COL], errors="coerce")
    non_zero = (s != 0) & s.notna()
    rate = non_zero.sum() / len(out)
    assert rate >= 0.005, f"Signal rate {rate:.4f} too low (not all zeros)"
    assert rate <= 0.25, f"Signal rate {rate:.4f} too high (not always-on)"
