# tests/test_contract_volume.py — Volume indicator contract tests (hygiene, shared helper).
# Failures indicate contract violations (e.g. return ndarray instead of DataFrame) to be fixed in indicators.

from __future__ import annotations

import pandas as pd
import pytest

from tests import _indicator_contracts as ic

WARMUP_BARS = 200
SIGNAL_COL = "volume_signal"
VALID_SIGNALS = {0, 1}

_VOL_LIST = ic.discover_volume_functions()
_VOL_IDS = [name for name, _ in _VOL_LIST]


@pytest.fixture(scope="module")
def test_df():
    """Load deterministic OHLCV for contract tests; skip if data/daily missing."""
    try:
        return ic.load_test_df()
    except FileNotFoundError:
        pytest.skip("data/daily missing or empty; need OHLCV CSV for Volume contract tests")


def _call_volume_with_signal_col(func, df, name: str, signal_col: str = SIGNAL_COL):
    """Call volume indicator with df and signal_col. Fail with clear message if signature rejects signal_col."""
    try:
        out = func(df.copy(), signal_col=signal_col)
        return out
    except TypeError as e:
        if "signal_col" in str(e).lower() or "keyword" in str(e).lower():
            raise AssertionError(
                f"Volume contract violation: '{name}' must accept signal_col= (e.g. signal_col='volume_signal'). "
                f"Error: {e}"
            ) from e
        raise
    except Exception as e:
        raise AssertionError(f"Volume '{name}' failed with: {e}") from e


def _as_01_set(series: pd.Series) -> set:
    """Unique values after warmup, normalized to int 0/1 for comparison (allow int-like floats)."""
    uniq = ic.unique_non_nan_values_after_warmup(series, WARMUP_BARS)
    out = set()
    for v in uniq:
        try:
            out.add(int(round(float(v))))
        except (ValueError, TypeError):
            out.add(v)
    return out


@pytest.mark.parametrize("name,func", _VOL_LIST, ids=_VOL_IDS)
def test_volume_returns_dataframe_aligned(name, func, test_df):
    """Volume must return a DataFrame with same index and length as input."""
    out = _call_volume_with_signal_col(func, test_df, name)
    if not isinstance(out, pd.DataFrame):
        raise AssertionError(
            f"Volume '{name}' must return a DataFrame; got {type(out).__name__}"
        )
    if len(out) != len(test_df):
        raise AssertionError(
            f"Volume '{name}': output length {len(out)} != input length {len(test_df)}"
        )
    if not out.index.equals(test_df.index):
        raise AssertionError(
            f"Volume '{name}': output index does not match input index (alignment contract)"
        )


@pytest.mark.parametrize("name,func", _VOL_LIST, ids=_VOL_IDS)
def test_volume_signal_values_after_warmup(name, func, test_df):
    """Volume must populate signal_col with values in {0, 1} after warmup (allow int-like floats)."""
    out = _call_volume_with_signal_col(func, test_df, name)
    if SIGNAL_COL not in out.columns:
        raise AssertionError(
            f"Volume '{name}' must write column '{SIGNAL_COL}'; columns: {list(out.columns)}"
        )
    series = out[SIGNAL_COL]
    uniq_set = _as_01_set(series)
    raw_uniq = ic.unique_non_nan_values_after_warmup(series, WARMUP_BARS)
    bad = uniq_set - VALID_SIGNALS
    if bad:
        raise AssertionError(
            f"Volume '{name}': signal values after warmup must be in {VALID_SIGNALS}. "
            f"Unique (after warmup, raw): {raw_uniq.tolist()}; interpreted as {uniq_set}; invalid: {bad}. "
            f"Counts: {(series.iloc[WARMUP_BARS:].value_counts(dropna=False).to_dict())}"
        )


@pytest.mark.parametrize("name,func", _VOL_LIST, ids=_VOL_IDS)
def test_volume_nan_policy_after_warmup(name, func, test_df):
    """After warmup: no NaNs (nan_rate 0, max_nan_streak 0)."""
    out = _call_volume_with_signal_col(func, test_df, name)
    series = out[SIGNAL_COL]
    nan_rate = ic.nan_rate_after_warmup(series, WARMUP_BARS)
    nan_count = int(series.iloc[WARMUP_BARS:].isna().sum())
    streak = ic.max_nan_streak_after_warmup(series, WARMUP_BARS)
    if nan_rate != 0 or nan_count != 0:
        raise AssertionError(
            f"Volume '{name}': no NaNs allowed after warmup={WARMUP_BARS}. "
            f"nan_rate={nan_rate}, nan_count={nan_count}, total_after_warmup={len(series) - WARMUP_BARS}."
        )
    if streak != 0:
        raise AssertionError(
            f"Volume '{name}': max_nan_streak after warmup must be 0; got {streak}."
        )


@pytest.mark.parametrize("name,func", _VOL_LIST, ids=_VOL_IDS)
def test_volume_determinism(name, func, test_df):
    """Identical output on repeated runs."""
    df1 = test_df.copy()
    df2 = test_df.copy()
    out1 = _call_volume_with_signal_col(func, df1, name)
    out2 = _call_volume_with_signal_col(func, df2, name)
    s1 = out1[SIGNAL_COL]
    s2 = out2[SIGNAL_COL]
    pd.testing.assert_series_equal(
        s1, s2, check_names=True, check_exact=False,
        obj=f"Volume '{name}' determinism",
    )


@pytest.mark.parametrize("name,func", _VOL_LIST, ids=_VOL_IDS)
def test_volume_causality_probe(name, func, test_df):
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
