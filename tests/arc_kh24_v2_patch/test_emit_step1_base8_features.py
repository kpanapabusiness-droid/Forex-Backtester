"""Unit tests for arc_kh24_v2 Step 1 base-8 feature emission.

Covers: formula correctness, no-lookahead, per-pair isolation, NaN handling,
RSI vs reference, determinism, byte-identical input preservation, row count.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.arc_kh24_v2_patch.emit_step1_base8_features import (  # noqa: E402
    build_pair_features,
    emit_sidecar,
    sha256_file,
)
from scripts.v2_0_diagnostic.entry_features import (  # noqa: E402
    _features_at_bar,
    _wilder_rsi,
)

TRADES_ALL = REPO_ROOT / "results" / "arc_kh24_v2" / "step1" / "trades_all.csv"
OHLCV_DIR = REPO_ROOT / "data" / "4hr"


def _synthetic_bars(n: int, seed: int = 0) -> pd.DataFrame:
    """Build n synthetic 4H bars with a deterministic price walk."""
    rng = np.random.default_rng(seed)
    base = 1.0 + np.cumsum(rng.normal(0, 0.001, size=n))
    open_ = base
    close = base + rng.normal(0, 0.001, size=n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.0005, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0005, size=n))
    times = pd.date_range("2020-01-01", periods=n, freq="4h")
    return pd.DataFrame({
        "time": times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    })


# -------------------------------------------------------------------------------------

def test_body_and_wick_ratios_match_formula():
    """Known O/H/L/C bar: body 0.5, upper wick 0.2, lower wick 0.3, all over range 1.0."""
    # bar at index 50 of a longer series so ATR/RSI warmup is satisfied (irrelevant here)
    n = 60
    df = _synthetic_bars(n)
    target = 50
    # Force a clean known bar at idx target: O=1.0, C=1.5, H=1.7, L=1.0 → body=0.5, upper=0.2, lower=0.0, range=0.7
    df.loc[target, ["open", "close", "high", "low"]] = [1.0, 1.5, 1.7, 1.0]
    feats = _features_at_bar(df)
    body_ratio = feats.loc[target, "body_to_range_ratio"]
    upper_ratio = feats.loc[target, "upper_wick_ratio"]
    lower_ratio = feats.loc[target, "lower_wick_ratio"]
    rng = 1.7 - 1.0
    assert body_ratio == pytest.approx(0.5 / rng)
    assert upper_ratio == pytest.approx((1.7 - 1.5) / rng)
    assert lower_ratio == pytest.approx((1.0 - 1.0) / rng)

    # Degenerate bar (high == low): all three ratios should be NaN (range guard) — script
    # rationalises NaN as "undefined ratio" which is the v2_0_diagnostic behaviour.
    df.loc[target, ["open", "close", "high", "low"]] = [1.2345, 1.2345, 1.2345, 1.2345]
    feats = _features_at_bar(df)
    assert pd.isna(feats.loc[target, "body_to_range_ratio"])


def test_no_lookahead_features_match_when_future_bars_truncated():
    """Compute features at bar T using full history vs history truncated at T;
    the values at T must be identical (no future-bar admission)."""
    df_full = _synthetic_bars(100, seed=7)
    target = 60  # well past Wilder warmup
    df_trunc = df_full.iloc[: target + 1].copy()
    feats_full = _features_at_bar(df_full).iloc[target]
    feats_trunc = _features_at_bar(df_trunc).iloc[target]
    for col in [
        "body_to_range_ratio", "upper_wick_ratio", "lower_wick_ratio",
        "range_to_atr_14", "ret_5bar_atr", "ret_20bar_atr",
        "pos_in_20bar_range", "rsi_14",
    ]:
        a, b = feats_full[col], feats_trunc[col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12), f"{col} diverges with future bars"


def test_per_pair_isolation(tmp_path):
    """Pair A and Pair B have totally different OHLCV; A's features must not change
    when B's data is present vs absent."""
    df_a = _synthetic_bars(80, seed=1)
    df_b = _synthetic_bars(80, seed=42)  # completely different series
    ohlcv_dir = tmp_path / "ohlcv"
    ohlcv_dir.mkdir()
    df_a.to_csv(ohlcv_dir / "PAIR_A.csv", index=False, columns=["time", "open", "high", "low", "close"])
    df_b.to_csv(ohlcv_dir / "PAIR_B.csv", index=False, columns=["time", "open", "high", "low", "close"])

    both = build_pair_features(ohlcv_dir, ["PAIR_A", "PAIR_B"])
    only_a = build_pair_features(ohlcv_dir, ["PAIR_A"])

    # Pair A's feature frame must be identical whether or not B is loaded
    pd.testing.assert_frame_equal(both["PAIR_A"], only_a["PAIR_A"])


def test_nan_handling_for_early_history():
    """At bar idx 5: rsi_14, ret_20bar_atr, pos_in_20bar_range, range_to_atr_14 NaN
    because Wilder warmup (period=14) or 20-bar window not yet satisfied.
    body / wick ratios should compute (single-bar metrics)."""
    df = _synthetic_bars(20, seed=3)
    feats = _features_at_bar(df).iloc[5]
    assert not pd.isna(feats["body_to_range_ratio"])
    assert not pd.isna(feats["upper_wick_ratio"])
    assert not pd.isna(feats["lower_wick_ratio"])
    assert pd.isna(feats["rsi_14"])
    assert pd.isna(feats["range_to_atr_14"])
    assert pd.isna(feats["ret_20bar_atr"])
    assert pd.isna(feats["pos_in_20bar_range"])


def test_rsi_matches_wilder_implementation():
    """Compute RSI two ways on the same synthetic series — _wilder_rsi vs manual EMA
    via the same formula. Tolerance 1e-9."""
    df = _synthetic_bars(50, seed=11)
    rsi_via_helper = _wilder_rsi(df, 14)
    # Manual: same formula, recomputed independently to catch silent drift in helper
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_manual = 100.0 - (100.0 / (1.0 + rs))
    pd.testing.assert_series_equal(
        rsi_via_helper, rsi_manual, check_names=False, atol=1e-9, rtol=1e-9
    )

    # Spot-check: mostly-up close with rare 1-bar dips → RSI high (> 80) but defined.
    # (Strictly monotone close gives avg_loss=0 → helper returns NaN by design; we want
    # to confirm the typical "strong uptrend" case yields a high but valid RSI.)
    rising = df.copy()
    base = np.linspace(1.0, 2.0, len(rising))  # step ≈ 0.020
    base[10::5] -= 0.05  # dips larger than step → at least some negative deltas
    rising["close"] = base
    rsi_rising = _wilder_rsi(rising, 14).iloc[-1]
    assert 60.0 < rsi_rising < 100.0, f"expected RSI in (60, 100), got {rsi_rising}"


def test_determinism_via_full_pipeline(tmp_path):
    """Run emit_sidecar twice on the actual trades_all.csv; sha256 must match."""
    if not TRADES_ALL.exists() or not OHLCV_DIR.is_dir():
        pytest.skip("Real arc data not available in this environment")
    out1 = tmp_path / "run1.csv"
    out2 = tmp_path / "run2.csv"
    emit_sidecar(TRADES_ALL, out1, OHLCV_DIR)
    emit_sidecar(TRADES_ALL, out2, OHLCV_DIR)
    assert sha256_file(out1) == sha256_file(out2)


def test_trades_all_byte_identical_after_run(tmp_path):
    """The script must not mutate the input trades_all.csv — sha256 unchanged
    after a full emission run."""
    if not TRADES_ALL.exists() or not OHLCV_DIR.is_dir():
        pytest.skip("Real arc data not available")
    # Make a copy we own so we don't risk touching the repo file even if test fails
    trades_copy = tmp_path / "trades_all.csv"
    shutil.copyfile(TRADES_ALL, trades_copy)
    before = sha256_file(trades_copy)
    emit_sidecar(trades_copy, tmp_path / "sidecar.csv", OHLCV_DIR)
    after = sha256_file(trades_copy)
    assert before == after, "emit_sidecar mutated trades_all.csv"


def test_row_count_matches_trades_all(tmp_path):
    """Sidecar row count = trades_all row count (no drops, no dupes)."""
    if not TRADES_ALL.exists() or not OHLCV_DIR.is_dir():
        pytest.skip("Real arc data not available")
    out_path = tmp_path / "sidecar.csv"
    out = emit_sidecar(TRADES_ALL, out_path, OHLCV_DIR)
    n_trades = pd.read_csv(TRADES_ALL).shape[0]
    assert len(out) == n_trades
    # And the file written should have n_trades + 1 lines (header + rows)
    with open(out_path) as f:
        lines = sum(1 for _ in f)
    assert lines == n_trades + 1


def test_trade_id_order_preserved(tmp_path):
    """Sidecar trade_id order must match trades_all order so positional joins work."""
    if not TRADES_ALL.exists() or not OHLCV_DIR.is_dir():
        pytest.skip("Real arc data not available")
    out = emit_sidecar(TRADES_ALL, tmp_path / "sidecar.csv", OHLCV_DIR)
    trades = pd.read_csv(TRADES_ALL, usecols=["trade_id"])
    assert list(out["trade_id"]) == list(trades["trade_id"])
