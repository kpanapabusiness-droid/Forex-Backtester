"""Tests for CSV schema normalization (legacy and MT5 compatibility)."""

import pandas as pd
import pytest

from core.utils import normalize_ohlcv_schema


@pytest.fixture
def legacy_csv_sample():
    """Legacy CSV schema sample."""
    return pd.DataFrame(
        {
            "date": ["10-01-02", "2010-01-03", "2010-01-04"],
            "open": [1.4312, 1.4303, 1.44123],
            "high": [1.43425, 1.4456, 1.44839],
            "low": [1.42926, 1.42576, 1.43468],
            "close": [1.43036, 1.44127, 1.4365],
            "volume": [1027, 42031, 45159],
        }
    )


@pytest.fixture
def mt5_csv_sample():
    """MT5 CSV schema sample."""
    return pd.DataFrame(
        {
            "time": ["2010-01-04", "2010-01-05", "2010-01-06"],
            "open": [1.43259, 1.4411100000000001, 1.43651],
            "high": [1.44543, 1.4483, 1.4433799999999999],
            "low": [1.42569, 1.4346700000000001, 1.42831],
            "close": [1.4411800000000001, 1.4365, 1.44086],
            "tick_volume": [44493, 47002, 45073],
            "spread": [10, 10, 9],
            "real_volume": [0, 0, 0],
        }
    )


def test_normalize_legacy_schema(legacy_csv_sample):
    """Test normalization of legacy schema produces correct columns and types."""
    result = normalize_ohlcv_schema(legacy_csv_sample)
    
    # Check exact column order and names
    expected_cols = ["date", "open", "high", "low", "close", "volume"]
    assert list(result.columns[:6]) == expected_cols, f"Expected {expected_cols}, got {list(result.columns[:6])}"
    
    # Check shapes match
    assert len(result) == len(legacy_csv_sample), "Row count should match"
    
    # Check dtypes are numeric for OHLCV
    assert pd.api.types.is_datetime64_any_dtype(result["date"]), "date should be datetime"
    assert pd.api.types.is_numeric_dtype(result["open"]), "open should be numeric"
    assert pd.api.types.is_numeric_dtype(result["high"]), "high should be numeric"
    assert pd.api.types.is_numeric_dtype(result["low"]), "low should be numeric"
    assert pd.api.types.is_numeric_dtype(result["close"]), "close should be numeric"
    assert pd.api.types.is_numeric_dtype(result["volume"]), "volume should be numeric"
    
    # Check values match (date parsing might change format but values should be equivalent)
    assert result["volume"].iloc[0] == 1027
    assert result["close"].iloc[1] == pytest.approx(1.44127)


def test_normalize_mt5_schema(mt5_csv_sample):
    """Test normalization of MT5 schema maps correctly."""
    result = normalize_ohlcv_schema(mt5_csv_sample)
    
    # Check exact column order and names (core columns first)
    expected_core_cols = ["date", "open", "high", "low", "close", "volume"]
    assert list(result.columns[:6]) == expected_core_cols
    
    # Check spread column is preserved as extra column if present
    assert "spread" in result.columns, "spread should be preserved as extra column"
    
    # Check volume equals tick_volume values
    assert (result["volume"] == mt5_csv_sample["tick_volume"]).all(), "volume should equal tick_volume"
    
    # Check shapes match
    assert len(result) == len(mt5_csv_sample), "Row count should match"
    
    # Check dtypes are numeric for OHLCV
    assert pd.api.types.is_datetime64_any_dtype(result["date"]), "date should be datetime"
    assert pd.api.types.is_numeric_dtype(result["open"]), "open should be numeric"
    assert pd.api.types.is_numeric_dtype(result["volume"]), "volume should be numeric"
    
    # Check real_volume is not in result (ignored)
    assert "real_volume" not in result.columns, "real_volume should be dropped"


def test_normalize_both_schemas_produce_same_downstream_format(legacy_csv_sample, mt5_csv_sample):
    """Test both schemas produce identical core columns for downstream compatibility."""
    legacy_result = normalize_ohlcv_schema(legacy_csv_sample)
    mt5_result = normalize_ohlcv_schema(mt5_csv_sample)
    
    # Core columns must be identical
    core_cols = ["date", "open", "high", "low", "close", "volume"]
    assert list(legacy_result[core_cols].columns) == list(mt5_result[core_cols].columns)
    
    # Column types should match
    for col in core_cols:
        assert isinstance(legacy_result[col].dtype, type(mt5_result[col].dtype)) or (
            pd.api.types.is_numeric_dtype(legacy_result[col])
            and pd.api.types.is_numeric_dtype(mt5_result[col])
        ), f"Column {col} types should match"


def test_normalize_handles_missing_volume():
    """Test normalizer handles legacy CSV without volume column."""
    df = pd.DataFrame(
        {
            "date": ["2010-01-01", "2010-01-02"],
            "open": [1.43, 1.44],
            "high": [1.45, 1.46],
            "low": [1.42, 1.43],
            "close": [1.44, 1.45],
        }
    )
    result = normalize_ohlcv_schema(df)
    
    assert "volume" in result.columns, "volume column should be created"
    assert result["volume"].dtype == float, "volume should be float dtype"
    assert (result["volume"] == 0.0).all() or result["volume"].isna().all(), "missing volume should be 0 or NaN"


def test_normalize_robust_date_parsing():
    """Test normalizer handles both YY-MM-DD and YYYY-MM-DD date formats."""
    df_yy = pd.DataFrame(
        {
            "date": ["10-01-02", "10-01-03"],
            "open": [1.43, 1.44],
            "high": [1.45, 1.46],
            "low": [1.42, 1.43],
            "close": [1.44, 1.45],
            "volume": [1000, 2000],
        }
    )
    df_yyyy = pd.DataFrame(
        {
            "date": ["2010-01-02", "2010-01-03"],
            "open": [1.43, 1.44],
            "high": [1.45, 1.46],
            "low": [1.42, 1.43],
            "close": [1.44, 1.45],
            "volume": [1000, 2000],
        }
    )
    
    result_yy = normalize_ohlcv_schema(df_yy)
    result_yyyy = normalize_ohlcv_schema(df_yyyy)
    
    assert pd.api.types.is_datetime64_any_dtype(result_yy["date"])
    assert pd.api.types.is_datetime64_any_dtype(result_yyyy["date"])
    assert not result_yy["date"].isna().any(), "YY-MM-DD dates should parse correctly"
    assert not result_yyyy["date"].isna().any(), "YYYY-MM-DD dates should parse correctly"


def test_normalize_raises_on_missing_required_columns():
    """Test normalizer raises ValueError when required columns are missing."""
    df = pd.DataFrame({"date": ["2010-01-01"], "open": [1.43]})  # missing high, low, close
    
    with pytest.raises(ValueError, match="Missing required columns"):
        normalize_ohlcv_schema(df)


def test_normalize_raises_on_invalid_numeric():
    """Test normalizer raises ValueError when numeric columns contain invalid data."""
    df = pd.DataFrame(
        {
            "date": ["2010-01-01", "2010-01-02"],
            "open": [1.43, "invalid"],
            "high": [1.45, 1.46],
            "low": [1.42, 1.43],
            "close": [1.44, 1.45],
            "volume": [1000, 2000],
        }
    )
    
    with pytest.raises(ValueError, match="non-numeric values"):
        normalize_ohlcv_schema(df)


def test_normalize_preserves_spread_extra_column(mt5_csv_sample):
    """Test that spread column from MT5 schema is preserved as extra."""
    result = normalize_ohlcv_schema(mt5_csv_sample)
    
    assert "spread" in result.columns, "spread should be preserved"
    assert (result["spread"] == mt5_csv_sample["spread"]).all(), "spread values should match"
    # spread should come after core columns
    core_cols = ["date", "open", "high", "low", "close", "volume"]
    assert result.columns.get_loc("spread") > len(core_cols) - 1, "spread should be after core columns"

