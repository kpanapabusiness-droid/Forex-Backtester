"""Feature edge cases — KH-24 v2.0 Step 2.

Synthetic test trades exercising the §6 edge cases per the prompt:
  - All-negative path: monotonicity = 0, local_peaks = 0
  - Single-bar trade (bars_held = 1): meaningful values where defined
  - One peak only: pullback = 0
  - Trade that never enters profit: all four features = 0
"""

from __future__ import annotations

import pandas as pd

from scripts.arc_kh24_v2.step2._features import (
    FEATURE_COLUMNS,
    compute_features_for_all_trades,
    compute_features_for_trade,
)


def _path(rows: list[tuple[int, float, float]], trade_id: str = "T1") -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["bar_offset", "close_r", "mfe_so_far_r"]).assign(
        trade_id=trade_id
    )


def test_all_negative_path():
    # bars: offset 0..4, never in profit. mfe_so_far_r stays at the first bar's
    # value (non-decreasing). close_r consistently negative.
    p = _path(
        [(0, -0.1, -0.05), (1, -0.4, -0.05), (2, -0.5, -0.05), (3, -0.8, -0.05), (4, -1.0, -0.05)]
    )
    feats = compute_features_for_trade(p, bars_held=4)
    assert feats["monotonicity_ratio_in_profit"] == 0.0
    # local_peaks counts strict increases in mfe; flat -> zero
    assert feats["local_peaks_count"] == 0.0
    # never in profit -> max_mfe <= 0 -> ttp = 0
    assert feats["time_to_peak_mfe_relative"] == 0.0
    # < 2 peaks -> 0
    assert feats["pullback_magnitude_median"] == 0.0


def test_single_bar_trade():
    # bars_held = 0 means path has only bar 0. In our pipeline a bars_held=1
    # trade has 2 rows (offsets 0 and 1). Test both.
    p = _path([(0, 0.5, 0.6)])
    feats = compute_features_for_trade(p, bars_held=0)
    assert feats["local_peaks_count"] == 0.0  # only 1 row
    # max mfe > 0 -> ttp = 0 / max(0,1) = 0
    assert feats["time_to_peak_mfe_relative"] == 0.0

    p2 = _path([(0, 0.3, 0.4), (1, -0.2, 0.4)])
    feats2 = compute_features_for_trade(p2, bars_held=1)
    # local_peaks: diff(mfe) = 0 -> no strict increase
    assert feats2["local_peaks_count"] == 0.0
    # ttp: peak at offset 0, bars_held=1 -> 0/1 = 0
    assert feats2["time_to_peak_mfe_relative"] == 0.0


def test_one_peak_only():
    # Single new-peak bar (offset 1). No second peak -> pullback = 0.
    p = _path([(0, -0.1, 0.0), (1, 0.5, 0.5), (2, 0.4, 0.5), (3, 0.3, 0.5)])
    feats = compute_features_for_trade(p, bars_held=3)
    assert feats["pullback_magnitude_median"] == 0.0
    assert feats["local_peaks_count"] == 1.0
    # in-profit bars: offsets 1, 2, 3 with close_r values 0.5, 0.4, 0.3.
    # diffs >=0: -0.1 (no), -0.1 (no) -> 0/2 = 0.0
    assert feats["monotonicity_ratio_in_profit"] == 0.0
    # peak at offset 1, bars_held=3 -> 1/3
    assert abs(feats["time_to_peak_mfe_relative"] - 1 / 3) < 1e-12


def test_two_peaks_pullback():
    # Peaks at offsets 1 (mfe 0.5) and 3 (mfe 0.8). Between [1..2], min close_r = 0.2.
    # Dip = 0.5 - 0.2 = 0.3. Single dip -> median = 0.3.
    p = _path(
        [
            (0, -0.1, 0.0),
            (1, 0.5, 0.5),
            (2, 0.2, 0.5),
            (3, 0.8, 0.8),
            (4, 0.6, 0.8),
        ]
    )
    feats = compute_features_for_trade(p, bars_held=4)
    assert abs(feats["pullback_magnitude_median"] - 0.3) < 1e-12
    assert feats["local_peaks_count"] == 2.0


def test_never_in_profit_all_zero():
    p = _path([(0, -0.05, 0.0), (1, -0.3, 0.0), (2, -0.7, 0.0), (3, -1.0, 0.0)])
    feats = compute_features_for_trade(p, bars_held=3)
    for v in feats.values():
        assert v == 0.0


def test_outcome_blind_features_only_read_close_r_and_mfe():
    """The feature functions consume only close_r + mfe_so_far_r + bar_offset.

    Verify that swapping `final_r` or other outcome-only columns in the
    trade summary does not affect the computed features.
    """
    paths = pd.DataFrame(
        [
            {"trade_id": "T1", "bar_offset": 0, "close_r": 0.0, "mfe_so_far_r": 0.0},
            {"trade_id": "T1", "bar_offset": 1, "close_r": 0.3, "mfe_so_far_r": 0.4},
            {"trade_id": "T1", "bar_offset": 2, "close_r": 0.6, "mfe_so_far_r": 0.7},
        ]
    )
    trades_a = pd.DataFrame([{"trade_id": "T1", "bars_held": 2, "final_r": 0.5}])
    trades_b = pd.DataFrame([{"trade_id": "T1", "bars_held": 2, "final_r": -1.0}])
    a = compute_features_for_all_trades(paths, trades_a)
    b = compute_features_for_all_trades(paths, trades_b)
    pd.testing.assert_frame_equal(a, b)


def test_feature_columns_ordering():
    paths = pd.DataFrame(
        [
            {"trade_id": "T1", "bar_offset": 0, "close_r": 0.1, "mfe_so_far_r": 0.2},
            {"trade_id": "T1", "bar_offset": 1, "close_r": 0.4, "mfe_so_far_r": 0.5},
            {"trade_id": "T0", "bar_offset": 0, "close_r": 0.05, "mfe_so_far_r": 0.1},
            {"trade_id": "T0", "bar_offset": 1, "close_r": 0.2, "mfe_so_far_r": 0.3},
        ]
    )
    trades = pd.DataFrame([{"trade_id": "T0", "bars_held": 1}, {"trade_id": "T1", "bars_held": 1}])
    out = compute_features_for_all_trades(paths, trades)
    assert list(out.columns) == ["trade_id", *FEATURE_COLUMNS]
    # Output sorted by trade_id (lex).
    assert out["trade_id"].tolist() == ["T0", "T1"]
