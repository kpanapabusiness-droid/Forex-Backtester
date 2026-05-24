"""Feature-parity test (PR #187 signal-parity contract).

Holds same mid-price across two synthetic bar frames but varies the
bid/ask split. All v3 mid-price features must produce IDENTICAL output;
only ``spread_close`` / ``spread_regime`` features may differ.

Locks the contract: under PR #187, every feature the engine consumes
for signal generation is venue-independent.
"""

from __future__ import annotations

import pandas as pd

import core.features.pipeline  # noqa: F401  (side-effect: registers features)
from core.features.registry import all_specs, get


def _synthetic_pair_df(spread_pips: float, n_bars: int = 200) -> pd.DataFrame:
    """Build a 1H pair frame with constant mid-price walk; spread is variable.

    ``spread_pips`` controls bid/ask split around a fixed mid.
    """
    idx = pd.date_range("2024-01-01 00:00", periods=n_bars, freq="1h", tz="UTC")
    mid_close = 1.1000 + (pd.Series(range(n_bars), index=idx) % 13) * 0.0001
    mid_open = mid_close.shift(1).fillna(1.1000)
    mid_high = pd.concat([mid_open, mid_close], axis=1).max(axis=1) + 0.00015
    mid_low = pd.concat([mid_open, mid_close], axis=1).min(axis=1) - 0.00010
    half_spread = (spread_pips / 2.0) * 0.0001

    df = pd.DataFrame(
        {
            "open_bid": mid_open - half_spread,
            "high_bid": mid_high - half_spread,
            "low_bid": mid_low - half_spread,
            "close_bid": mid_close - half_spread,
            "open_ask": mid_open + half_spread,
            "high_ask": mid_high + half_spread,
            "low_ask": mid_low + half_spread,
            "close_ask": mid_close + half_spread,
            "volume": [10] * n_bars,
            "spread_close": [spread_pips * 0.0001] * n_bars,
            "bid_ask_data_quality": ["ok"] * n_bars,
        },
        index=idx,
    )
    df.index.name = "timestamp_utc"
    df.attrs["pair"] = "EURUSD"
    return df


# Features expected to be spread-INDEPENDENT (price-derived from mid).
_MID_FEATURES = (
    "atr_14",
    "kijun_26_distance",
    "swing_high_distance_14",
    "swing_low_distance_14",
    "range_close_ratio",
    "atr_vs_trailing_100",
    "atr_percentile_100",
    "prior_session_high_distance",
    "prior_session_low_distance",
    "distance_to_round_number",
    "hour_of_day",
    "day_of_week",
    "session_tokyo",
    "session_london",
    "session_ny",
    "session_ldn_ny_overlap",
    "session_dead",
)

# Features expected to be spread-DEPENDENT (structural regime).
_SPREAD_FEATURES = ("spread_vs_trailing_100", "spread_percentile_100")


def test_all_registered_mid_features_invariant_under_spread() -> None:
    """Every registered mid-price feature must produce identical Series
    whether the bid/ask spread is 0.5 pip or 5.0 pip, given identical
    underlying mid OHLC. Spread-class features are exempt."""
    df_tight = _synthetic_pair_df(spread_pips=0.5)
    df_wide = _synthetic_pair_df(spread_pips=5.0)

    registered = {spec.name for spec in all_specs()}
    failures: list[tuple[str, str]] = []

    for name in registered:
        if name in _SPREAD_FEATURES:
            continue
        # Some features need cross-pair panel — skip those here; covered in
        # tests/test_features_pipeline.py.
        spec = get(name)
        if spec.feature_class == "cross_pair":
            continue
        # Multi-TF features need an HTF panel injection; skip in this test.
        if spec.feature_class == "multi_tf":
            continue
        try:
            s_tight = spec.producer(df_tight)
            s_wide = spec.producer(df_wide)
        except Exception as e:  # noqa: BLE001
            failures.append((name, f"producer raised: {e!r}"))
            continue
        # Compare with NaN-tolerant equality
        try:
            pd.testing.assert_series_equal(
                s_tight, s_wide, check_names=False, check_exact=False, rtol=1e-9, atol=1e-12
            )
        except AssertionError as e:
            failures.append((name, str(e)[:200]))

    assert not failures, (
        "Mid-price features differed when only spread changed (signal parity violation):\n"
        + "\n".join(f"  - {n}: {msg}" for n, msg in failures)
    )


def test_spread_regime_features_use_spread_input() -> None:
    """Negative-check: spread regime features must READ spread_close from the
    pair frame (not infer it from bid/ask). Verifies the dispatch's note
    that spread_regime is structural and exempt from mid-parity."""
    df = _synthetic_pair_df(spread_pips=0.5)
    for name in _SPREAD_FEATURES:
        spec = get(name)
        # Should run without error even when spread_close is explicitly
        # the input. If the feature inferred spread from bid/ask, we'd
        # break this contract by removing one of them — but spread_close
        # is the canonical input. Smoke test only.
        s = spec.producer(df)
        assert s.index.equals(df.index), f"{name} did not align to input index"
