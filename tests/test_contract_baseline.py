# tests/test_contract_baseline.py — Baseline indicator contract tests (hygiene, shared helper).
# Failures indicate contract violations to be fixed in indicators.

from __future__ import annotations

import pandas as pd
import pytest

from tests import _indicator_contracts as ic

WARMUP_BARS = 200
SIGNAL_COL = "baseline_signal"
BASELINE_COL = "baseline"
VALID_SIGNALS = {-1, 0, 1}
NAN_RATE_MAX = 0.005
NAN_COUNT_MAX = 3
NAN_STREAK_MAX = 3

_BASELINE_LIST = ic.discover_baseline_functions()
_BASELINE_IDS = [name for name, _ in _BASELINE_LIST]


@pytest.fixture(scope="module")
def test_df():
    """Load deterministic OHLCV for contract tests; skip if data/daily missing."""
    try:
        return ic.load_test_df()
    except FileNotFoundError:
        pytest.skip("data/daily missing or empty; need OHLCV CSV for Baseline contract tests")


def _call_baseline_with_signal_col(func, df, name: str, signal_col: str = SIGNAL_COL):
    """
    Call baseline with df and signal_col if supported; else call in repo standard way.
    Enforce that baseline_signal (and baseline) exist in output; fail with clear message otherwise.
    """
    try:
        out = func(df.copy(), signal_col=signal_col)
    except TypeError as e:
        if "signal_col" in str(e).lower() or "keyword" in str(e).lower():
            try:
                out = func(df.copy())
            except Exception as e2:
                raise AssertionError(
                    f"Baseline contract: '{name}' must accept signal_col= or be callable as func(df). Error: {e2}"
                ) from e2
        else:
            raise
    except Exception as e:
        raise AssertionError(f"Baseline '{name}' failed with: {e}") from e
    if not isinstance(out, pd.DataFrame):
        raise AssertionError(
            f"Baseline '{name}' must return a DataFrame; got {type(out).__name__}"
        )
    if SIGNAL_COL not in out.columns:
        raise AssertionError(
            f"Baseline contract violation: '{name}' must populate '{SIGNAL_COL}'. "
            f"Columns: {list(out.columns)}"
        )
    if BASELINE_COL not in out.columns:
        raise AssertionError(
            f"Baseline contract violation: '{name}' must create numeric column '{BASELINE_COL}'. "
            f"Columns: {list(out.columns)}"
        )
    return out


@pytest.mark.parametrize("name,func", _BASELINE_LIST, ids=_BASELINE_IDS)
def test_baseline_returns_dataframe_aligned(name, func, test_df):
    """Baseline must return a DataFrame with same index and length as input."""
    out = _call_baseline_with_signal_col(func, test_df, name)
    if len(out) != len(test_df):
        raise AssertionError(
            f"Baseline '{name}': output length {len(out)} != input length {len(test_df)}"
        )
    if not out.index.equals(test_df.index):
        raise AssertionError(
            f"Baseline '{name}': output index does not match input index (alignment contract)"
        )


@pytest.mark.parametrize("name,func", _BASELINE_LIST, ids=_BASELINE_IDS)
def test_baseline_column_present_and_mostly_non_nan(name, func, test_df):
    """Baseline must create numeric column 'baseline' and it must be mostly non-NaN after warmup."""
    out = _call_baseline_with_signal_col(func, test_df, name)
    series = out[BASELINE_COL]
    if not pd.api.types.is_numeric_dtype(series):
        raise AssertionError(
            f"Baseline '{name}': column '{BASELINE_COL}' must be numeric; got dtype {series.dtype}"
        )
    nan_rate = ic.nan_rate_after_warmup(series, WARMUP_BARS)
    nan_count = int(series.iloc[WARMUP_BARS:].isna().sum())
    rate_ok = nan_rate <= NAN_RATE_MAX
    count_ok = nan_count <= NAN_COUNT_MAX
    if not (rate_ok or count_ok):
        raise AssertionError(
            f"Baseline '{name}': '{BASELINE_COL}' must be mostly non-NaN after warmup={WARMUP_BARS}. "
            f"nan_rate={nan_rate:.4f} (max {NAN_RATE_MAX}) OR nan_count={nan_count} (max {NAN_COUNT_MAX}); "
            f"got nan_rate={nan_rate:.4f}, nan_count={nan_count}."
        )


@pytest.mark.parametrize("name,func", _BASELINE_LIST, ids=_BASELINE_IDS)
def test_baseline_signal_values_after_warmup(name, func, test_df):
    """Baseline must populate signal_col with values in {-1, 0, 1} after warmup."""
    out = _call_baseline_with_signal_col(func, test_df, name)
    series = out[SIGNAL_COL]
    uniq = ic.unique_non_nan_values_after_warmup(series, WARMUP_BARS)
    uniq_set = set(int(round(float(v))) if isinstance(v, (int, float)) else v for v in uniq)
    bad = uniq_set - VALID_SIGNALS
    if bad:
        raise AssertionError(
            f"Baseline '{name}': signal values after warmup must be in {VALID_SIGNALS}. "
            f"Unique (after warmup): {uniq.tolist()}; invalid: {bad}. "
            f"Counts: {series.iloc[WARMUP_BARS:].value_counts(dropna=False).to_dict()}"
        )


@pytest.mark.parametrize("name,func", _BASELINE_LIST, ids=_BASELINE_IDS)
def test_baseline_nan_policy_after_warmup(name, func, test_df):
    """After warmup: nan_rate <= 0.5% OR <= 3 NaNs total; max_nan_streak <= 3."""
    out = _call_baseline_with_signal_col(func, test_df, name)
    series = out[SIGNAL_COL]
    nan_rate = ic.nan_rate_after_warmup(series, WARMUP_BARS)
    nan_count = int(series.iloc[WARMUP_BARS:].isna().sum())
    streak = ic.max_nan_streak_after_warmup(series, WARMUP_BARS)
    rate_ok = nan_rate <= NAN_RATE_MAX
    count_ok = nan_count <= NAN_COUNT_MAX
    streak_ok = streak <= NAN_STREAK_MAX
    if not (rate_ok or count_ok):
        raise AssertionError(
            f"Baseline '{name}': NaN policy violated after warmup={WARMUP_BARS}. "
            f"nan_rate={nan_rate:.4f} (max {NAN_RATE_MAX}) OR nan_count={nan_count} (max {NAN_COUNT_MAX}); "
            f"got nan_rate={nan_rate:.4f}, nan_count={nan_count}. Neither condition passed."
        )
    if not streak_ok:
        raise AssertionError(
            f"Baseline '{name}': max_nan_streak after warmup={streak} (max {NAN_STREAK_MAX})"
        )


@pytest.mark.parametrize("name,func", _BASELINE_LIST, ids=_BASELINE_IDS)
def test_baseline_determinism(name, func, test_df):
    """Identical output on repeated runs."""
    df1 = test_df.copy()
    df2 = test_df.copy()
    out1 = _call_baseline_with_signal_col(func, df1, name)
    out2 = _call_baseline_with_signal_col(func, df2, name)
    pd.testing.assert_series_equal(
        out1[SIGNAL_COL], out2[SIGNAL_COL],
        check_names=True, check_exact=False,
        obj=f"Baseline '{name}' determinism (signal)",
    )
    pd.testing.assert_series_equal(
        out1[BASELINE_COL], out2[BASELINE_COL],
        check_names=True, check_exact=False,
        obj=f"Baseline '{name}' determinism (baseline)",
    )


@pytest.mark.parametrize("name,func", _BASELINE_LIST, ids=_BASELINE_IDS)
def test_baseline_causality_probe(name, func, test_df):
    """Mutating last 20 bars must not change earlier baseline_signal values (no lookahead)."""
    if len(test_df) <= 20:
        pytest.skip("test_df too short for causality probe (need > 20 rows)")
    ic.run_causality_probe(
        func,
        test_df,
        call_kwargs={"signal_col": SIGNAL_COL},
        signal_col=SIGNAL_COL,
        future_window=20,
    )
