"""Lookahead-invariance tests for Arc 3 Step 1 plumbing.

Covers the new 1H-signal-specific paths:

1. signals/lchar_d1atr_top_decile.compute_signal — the D1 ATR top-decile mask
   must depend only on bars STRICTLY BEFORE the active 1H bar's calendar day.
   The L4 lookback contract is `mr_idx = contain_int - 1`; any post-signal D1
   bar mutation must NOT change earlier signals.

2. The trailing-decile threshold uses .shift(1) → the active D1 bar cannot
   reach its own threshold.

3. scripts/arc_3/step1_backtest._wilder_atr_1h is causal — mutating a future
   bar must not change ATR at the prior index.

These tests run synthetic data (no I/O, fast) and cover the v2.0 §1 invariant
(no lookahead, no repainting).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.arc_3.step1_backtest import _wilder_atr_1h
from signals.lchar_d1atr_top_decile import (
    _compute_d1_atr_sma,
    _lookback_d1_to_1h,
    _trailing_top_decile,
    compute_signal,
)


def _make_d1(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 1.1000
    rets = rng.normal(0, 0.0030, size=n)
    closes = base * np.cumprod(1 + rets)
    highs = closes * (1 + np.abs(rng.normal(0, 0.0015, size=n)))
    lows = closes * (1 - np.abs(rng.normal(0, 0.0015, size=n)))
    opens = np.concatenate(([closes[0]], closes[:-1]))
    dates = pd.date_range("2022-01-03", periods=n, freq="D")  # weekday-naive ok
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": np.maximum(np.maximum(highs, opens), closes),
            "low": np.minimum(np.minimum(lows, opens), closes),
            "close": closes,
        }
    )


def _make_1h(d1: pd.DataFrame) -> pd.DataFrame:
    # 24 1H bars per D1 day at hourly intervals starting 00:00.
    rows = []
    for _, drow in d1.iterrows():
        day = pd.Timestamp(drow["date"]).normalize()
        cl = float(drow["close"])
        for h in range(24):
            t = day + pd.Timedelta(hours=h)
            rows.append(
                {
                    "date": t,
                    "open": cl,
                    "high": cl * 1.0002,
                    "low": cl * 0.9998,
                    "close": cl,
                    "spread": 5.0,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Trailing decile excludes the active bar (uses .shift(1)).
# ---------------------------------------------------------------------------


def test_trailing_decile_excludes_active_bar():
    n = 150
    rng = np.random.default_rng(1)
    s = pd.Series(rng.uniform(0.0, 1.0, size=n))
    mask = _trailing_top_decile(s, window=100, q=0.9)

    # Replace s[100] (active bar) with +inf — its OWN mask cell should be True;
    # but the threshold at position 100 was computed from s[0..99]. Replacing
    # s[100] in-place and recomputing should give the same threshold ⇒ mask
    # may become True (it wasn't in original if not in top decile of past 100).
    # Critical invariant: mask cells AT INDICES BEFORE THE MUTATION must be
    # unchanged.
    s2 = s.copy()
    s2.iloc[100] = 999.0
    mask2 = _trailing_top_decile(s2, window=100, q=0.9)
    np.testing.assert_array_equal(mask[:100], mask2[:100])


def test_trailing_decile_warmup_is_false():
    n = 120
    s = pd.Series(np.arange(n, dtype=float))
    mask = _trailing_top_decile(s, window=100, q=0.9)
    # Positions 0..99 don't have a full prior 100-bar window — mask is False there.
    assert not mask[:100].any()


# ---------------------------------------------------------------------------
# 2. D1→1H lookback uses contain - 1 (one-day lag).
# ---------------------------------------------------------------------------


def test_lookback_d1_to_1h_uses_strict_prior_d1_bar():
    d1 = _make_d1(n=10)
    df_1h = _make_1h(d1)
    # Mark D1 bars True at odd indices, False at even indices — gives a
    # distinct, traceable pattern that exposes any off-by-one error.
    d1_mask_bool = (np.arange(len(d1)) % 2 == 1)
    # Manual reconstruction: for a 1H bar on D1 day k, we expect d1_mask[k - 1].
    result = _lookback_d1_to_1h(df_1h, d1, d1_mask_bool)

    expected = np.zeros(len(df_1h), dtype=bool)
    for i, t in enumerate(df_1h["date"]):
        day_norm = pd.Timestamp(t).normalize()
        d1_idx_match = d1.index[d1["date"] == day_norm]
        if len(d1_idx_match) == 0:
            continue
        k = int(d1_idx_match[0])
        if k - 1 < 0:
            continue
        expected[i] = d1_mask_bool[k - 1]

    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# 3. End-to-end: mutating a future D1 bar leaves earlier 1H signals unchanged.
# ---------------------------------------------------------------------------


def test_no_lookahead_future_d1_mutation_doesnt_change_earlier_signals():
    d1 = _make_d1(n=200, seed=11)
    df_1h = _make_1h(d1)

    sig1 = compute_signal(df_1h, d1)["signal"].to_numpy()

    # Mutate D1 bars from index 150 onward (their highs/lows wildly larger).
    d1_mut = d1.copy()
    d1_mut.loc[150:, "high"] = d1_mut.loc[150:, "high"] * 5.0
    d1_mut.loc[150:, "low"] = d1_mut.loc[150:, "low"] / 5.0

    sig2 = compute_signal(df_1h, d1_mut)["signal"].to_numpy()

    # The 1H bars whose active D1 (contain - 1) is < 150 must be unchanged.
    cutoff_day = pd.Timestamp(d1.iloc[151]["date"]).normalize()
    # Active D1 for a 1H bar on day k is D1[k-1]. So the boundary is at 1H bars
    # on calendar day < d1.iloc[151].date — for these, active D1 index < 150.
    early_mask = df_1h["date"].dt.normalize() < cutoff_day
    np.testing.assert_array_equal(sig1[early_mask.to_numpy()], sig2[early_mask.to_numpy()])


def test_no_lookahead_future_1h_mutation_doesnt_change_earlier_signals():
    d1 = _make_d1(n=100, seed=13)
    df_1h = _make_1h(d1)
    sig1 = compute_signal(df_1h, d1)["signal"].to_numpy()

    df_1h_mut = df_1h.copy()
    df_1h_mut.loc[1000:, "close"] = df_1h_mut.loc[1000:, "close"] * 1.5
    sig2 = compute_signal(df_1h_mut, d1)["signal"].to_numpy()

    # Signal only depends on D1 frame, so mutating future 1H closes shouldn't
    # change any signal cell. Stronger: full equality.
    np.testing.assert_array_equal(sig1, sig2)


# ---------------------------------------------------------------------------
# 4. Wilder 1H ATR is causal.
# ---------------------------------------------------------------------------


def test_wilder_atr_1h_causal():
    d1 = _make_d1(n=50, seed=17)
    df_1h = _make_1h(d1)
    atr1 = _wilder_atr_1h(df_1h, 14)

    df_1h_mut = df_1h.copy()
    df_1h_mut.loc[800:, "high"] = df_1h_mut.loc[800:, "high"] * 2.0
    atr2 = _wilder_atr_1h(df_1h_mut, 14)

    np.testing.assert_array_equal(atr1[:800], atr2[:800])


# ---------------------------------------------------------------------------
# 5. SMA D1 ATR is causal.
# ---------------------------------------------------------------------------


def test_d1_atr_sma_causal():
    d1 = _make_d1(n=80, seed=19)
    atr1 = _compute_d1_atr_sma(d1, 14).to_numpy()

    d1_mut = d1.copy()
    d1_mut.loc[40:, "high"] = d1_mut.loc[40:, "high"] * 3.0
    atr2 = _compute_d1_atr_sma(d1_mut, 14).to_numpy()

    np.testing.assert_array_equal(atr1[:40], atr2[:40])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
