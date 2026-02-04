# tests/test_contract_c1.py — Phase 3 C1 indicator contract tests (shared helper).

from __future__ import annotations

import pandas as pd
import pytest

from tests import _indicator_contracts as ic

WARMUP_BARS = 200
SIGNAL_COL = "c1_signal"
VALID_SIGNALS = {-1, 0, 1}
NAN_RATE_MAX = 0.005
NAN_COUNT_MAX = 3
NAN_STREAK_MAX = 3

_C1_LIST = ic.discover_c1_functions()
_C1_IDS = [name for name, _ in _C1_LIST]


@pytest.fixture(scope="module")
def test_df():
    """Load deterministic OHLCV for contract tests; skip if data/daily missing."""
    try:
        return ic.load_test_df()
    except FileNotFoundError:
        pytest.skip("data/daily missing or empty; need OHLCV CSV for C1 contract tests")


def _call_c1_with_signal_col(func, df, name: str):
    """Call C1 with df and signal_col='c1_signal'. Fail with clear message if signature rejects signal_col."""
    try:
        out = func(df.copy(), signal_col=SIGNAL_COL)
        return out
    except TypeError as e:
        if "signal_col" in str(e).lower() or "keyword" in str(e).lower():
            raise AssertionError(
                f"C1 contract violation: '{name}' must accept signal_col= (e.g. signal_col='c1_signal'). "
                f"Signature may not include signal_col. Error: {e}"
            ) from e
        raise
    except Exception as e:
        raise AssertionError(f"C1 '{name}' failed with: {e}") from e


@pytest.mark.parametrize("name,func", _C1_LIST, ids=_C1_IDS)
def test_c1_returns_dataframe_aligned(name, func, test_df):
    """C1 must return a DataFrame with same index and length as input."""
    out = _call_c1_with_signal_col(func, test_df, name)
    if not isinstance(out, type(test_df)) or not hasattr(out, "index"):
        raise AssertionError(
            f"C1 '{name}' must return a DataFrame; got {type(out).__name__}"
        )
    if len(out) != len(test_df):
        raise AssertionError(
            f"C1 '{name}': output length {len(out)} != input length {len(test_df)}"
        )
    if not out.index.equals(test_df.index):
        raise AssertionError(
            f"C1 '{name}': output index does not match input index (alignment contract)"
        )


@pytest.mark.parametrize("name,func", _C1_LIST, ids=_C1_IDS)
def test_c1_signal_values_after_warmup(name, func, test_df):
    """C1 must populate c1_signal with values in {-1, 0, 1} after warmup."""
    out = _call_c1_with_signal_col(func, test_df, name)
    if SIGNAL_COL not in out.columns:
        raise AssertionError(
            f"C1 '{name}' must write column '{SIGNAL_COL}'; columns: {list(out.columns)}"
        )
    series = out[SIGNAL_COL]
    uniq = ic.unique_non_nan_values_after_warmup(series, WARMUP_BARS)
    bad = set(uniq) - VALID_SIGNALS
    if bad:
        raise AssertionError(
            f"C1 '{name}': signal values after warmup must be in {VALID_SIGNALS}; "
            f"found {bad}. Unique (after warmup): {uniq.tolist()}"
        )


@pytest.mark.parametrize("name,func", _C1_LIST, ids=_C1_IDS)
def test_c1_nan_policy_after_warmup(name, func, test_df):
    """After warmup: nan_rate <= 0.5% OR <= 3 NaNs total; max_nan_streak <= 3."""
    out = _call_c1_with_signal_col(func, test_df, name)
    series = out[SIGNAL_COL]
    nan_rate = ic.nan_rate_after_warmup(series, WARMUP_BARS)
    nan_count = int(series.iloc[WARMUP_BARS:].isna().sum())
    streak = ic.max_nan_streak_after_warmup(series, WARMUP_BARS)
    rate_ok = nan_rate <= NAN_RATE_MAX
    count_ok = nan_count <= NAN_COUNT_MAX
    streak_ok = streak <= NAN_STREAK_MAX
    if not (rate_ok or count_ok):
        raise AssertionError(
            f"C1 '{name}': NaN policy violated after warmup={WARMUP_BARS}. "
            f"nan_rate={nan_rate:.4f} (max {NAN_RATE_MAX}) OR nan_count={nan_count} (max {NAN_COUNT_MAX}); "
            f"got nan_rate={nan_rate:.4f}, nan_count={nan_count}. Neither condition passed."
        )
    if not streak_ok:
        raise AssertionError(
            f"C1 '{name}': max_nan_streak after warmup={streak} (max {NAN_STREAK_MAX})"
        )


@pytest.mark.parametrize("name,func", _C1_LIST, ids=_C1_IDS)
def test_c1_determinism(name, func, test_df):
    """Same df copy -> identical signal series."""
    df1 = test_df.copy()
    df2 = test_df.copy()
    out1 = _call_c1_with_signal_col(func, df1, name)
    out2 = _call_c1_with_signal_col(func, df2, name)
    s1 = out1[SIGNAL_COL]
    s2 = out2[SIGNAL_COL]
    pd.testing.assert_series_equal(
        s1, s2, check_names=True, check_exact=False,
        obj=f"C1 '{name}' determinism",
    )


@pytest.mark.parametrize("name,func", _C1_LIST, ids=_C1_IDS)
def test_c1_causality_probe(name, func, test_df):
    """Mutating last 20 bars must not change earlier signals (no lookahead)."""
    if len(test_df) <= 20:
        pytest.skip("test_df too short for causality probe (need > 20 rows)")
    ic.run_causality_probe(
        func,
        test_df,
        call_kwargs={"signal_col": SIGNAL_COL},
        signal_col=SIGNAL_COL,
        future_window=20,
    )
