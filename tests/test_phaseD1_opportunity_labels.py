from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _make_synthetic_ohlc(n: int) -> pd.DataFrame:
    """Helper: build a simple, monotonically dated D1 OHLC frame."""
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": np.zeros(n, dtype=float),
            "high": np.zeros(n, dtype=float),
            "low": np.zeros(n, dtype=float),
            "close": np.zeros(n, dtype=float),
            "volume": np.zeros(n, dtype=float),
        }
    )
    return df


def _minimal_phaseD1_cfg(atr_period: int = 14) -> dict:
    """Minimal config dict for calling the Phase D1 core."""
    return {
        "pairs": ["TEST_PAIR"],
        "timeframe": "D1",
        "date_range": {
            "start": "2019-01-01",
            "end": "2026-01-01",
        },
        "atr": {"period": atr_period},
        "outputs": {
            "dir": "results/phaseD/labels",
        },
    }


def test_phaseD1_reference_open_and_no_last_bar_rows() -> None:
    """ref_open must be open[t+1]; last bar (no t+1) produces no rows."""
    from scripts.phaseD1_generate_opportunity_labels import generate_labels_for_pair

    df = _make_synthetic_ohlc(5)
    # Make opens distinct so we can assert ref_open mapping
    df["open"] = np.arange(5, dtype=float) * 10.0

    cfg = _minimal_phaseD1_cfg(atr_period=1)
    out = generate_labels_for_pair(df, pair="EUR_USD", cfg=cfg)

    # One row per (date t, direction) except final bar
    unique_dates = sorted(out["date"].unique())
    assert len(unique_dates) == 4  # 5 bars -> 4 label dates
    assert unique_dates[0] == df["date"].iloc[0]
    assert unique_dates[-1] == df["date"].iloc[3]
    assert df["date"].iloc[-1] not in unique_dates

    # For each t < last, ref_date == date[t+1] and ref_open == open[t+1]
    for t in range(4):
        exp_date = df["date"].iloc[t]
        exp_ref_date = df["date"].iloc[t + 1]
        exp_ref_open = df["open"].iloc[t + 1]
        rows_t = out[out["date"] == exp_date]
        assert set(rows_t["direction"]) == {"long", "short"}
        for _dir in ("long", "short"):
            r = rows_t[rows_t["direction"] == _dir].iloc[0]
            assert r["ref_date"] == exp_ref_date
            assert r["ref_open"] == pytest.approx(exp_ref_open)


def test_phaseD1_horizons_shrink_near_end_of_data() -> None:
    """Window horizons must shrink cleanly near end-of-data, never going OOB."""
    from scripts.phaseD1_generate_opportunity_labels import generate_labels_for_pair

    df = _make_synthetic_ohlc(6)
    cfg = _minimal_phaseD1_cfg(atr_period=1)
    out = generate_labels_for_pair(df, pair="EUR_USD", cfg=cfg)

    # Map date -> horizon values (direction-independent)
    sample = out[out["direction"] == "long"].set_index("date")

    # For first bar (t=0 -> ref at index 1) we have 5 forward bars
    first_date = df["date"].iloc[0]
    r0 = sample.loc[first_date]
    assert r0["horizon_10"] == 5
    assert r0["horizon_20"] == 5
    assert r0["horizon_40"] == 5

    # For penultimate bar (t=4 -> ref at index 5) we have 1 forward bar
    penultimate_date = df["date"].iloc[4]
    r4 = sample.loc[penultimate_date]
    assert r4["horizon_10"] == 1
    assert r4["horizon_20"] == 1
    assert r4["horizon_40"] == 1


def test_phaseD1_long_short_direction_and_timing() -> None:
    """
    Long uses highs, short uses lows; timing metrics T1/T3/T6 count bars from ref.

    Construct a synthetic series where, for a chosen t, the long side hits:
      - 1R on ref bar (T1 = 0)
      - 3R on next bar (T3 = 1)
      - 6R on third bar (T6 = 2)
    with symmetric behaviour for shorts via lows.
    """
    from core.utils import calculate_atr
    from scripts.phaseD1_generate_opportunity_labels import generate_labels_for_pair

    df = _make_synthetic_ohlc(5)

    # Choose bar t = 0 as the ATR/label anchor. Force ATR(t) = 1.0 via TR = 1.0.
    df.loc[0, ["high", "low", "close"]] = [1.0, 0.0, 0.0]

    # Reference bar is t+1 (index 1); set ref_open = 0.0 for simplicity.
    df.loc[1:, "open"] = 0.0

    # Forward highs for long side relative to ref_open=0.0:
    #   offset 0 (ref bar): 1R  -> +1.5
    #   offset 1:           3R  -> +4.5
    #   offset 2:           6R  -> +9.0
    df.loc[1, ["high", "low", "close"]] = [1.5, 0.0, 0.0]
    df.loc[2, ["high", "low", "close"]] = [4.5, 0.0, 0.0]
    df.loc[3, ["high", "low", "close"]] = [9.0, 0.0, 0.0]

    # Mirror for short side via lows below ref_open=0.0
    df.loc[1, "low"] = -1.5
    df.loc[2, "low"] = -4.5
    df.loc[3, "low"] = -9.0

    # Compute ATR with period=1 so ATR(0) = TR(0) = 1.0 and 1R = 1.5.
    df = calculate_atr(df, period=1)

    cfg = _minimal_phaseD1_cfg(atr_period=1)
    out = generate_labels_for_pair(df, pair="EUR_USD", cfg=cfg)

    # Extract rows for date t0
    t0 = df["date"].iloc[0]
    rows_t0 = out[out["date"] == t0].set_index("direction")

    # Long side: highs drive MFE
    r_long = rows_t0.loc["long"]
    assert bool(r_long["zone_a_1r_10"]) is True
    assert bool(r_long["zone_b_3r_20"]) is True
    assert bool(r_long["zone_c_6r_40"]) is True
    assert r_long["t1"] == 0
    assert r_long["t3"] == 1
    assert r_long["t6"] == 2

    # Short side: lows drive MFE
    r_short = rows_t0.loc["short"]
    assert bool(r_short["zone_a_1r_10"]) is True
    assert bool(r_short["zone_b_3r_20"]) is True
    assert bool(r_short["zone_c_6r_40"]) is True
    assert r_short["t1"] == 0
    assert r_short["t3"] == 1
    assert r_short["t6"] == 2


def test_phaseD1_timing_never_hit_is_nan() -> None:
    """If thresholds are never reached within 40 bars, T1/T3/T6 must be NaN."""
    from core.utils import calculate_atr
    from scripts.phaseD1_generate_opportunity_labels import generate_labels_for_pair

    df = _make_synthetic_ohlc(50)

    # Small, flat moves so we never reach 1R even after 40 bars.
    df["open"] = 0.0
    df["high"] = 0.1
    df["low"] = -0.1
    df["close"] = 0.0

    df = calculate_atr(df, period=1)
    cfg = _minimal_phaseD1_cfg(atr_period=1)
    out = generate_labels_for_pair(df, pair="EUR_USD", cfg=cfg)

    # Pick an interior bar where full 40-bar lookahead is available.
    mid_date = df["date"].iloc[5]
    r_long = out[(out["date"] == mid_date) & (out["direction"] == "long")].iloc[0]
    r_short = out[(out["date"] == mid_date) & (out["direction"] == "short")].iloc[0]

    for r in (r_long, r_short):
        assert np.isnan(r["t1"])
        assert np.isnan(r["t3"])
        assert np.isnan(r["t6"])


def test_phaseD1_stop_ignorance_adverse_move_does_not_cancel_opportunity() -> None:
    """
    Large adverse moves before a later favourable move must not cancel zone labels.

    We construct:
      - first bar after ref: big adverse move (down for long, up for short),
      - second bar: strong favourable move that clearly meets 1R and 3R.
    """
    from core.utils import calculate_atr
    from scripts.phaseD1_generate_opportunity_labels import generate_labels_for_pair

    df = _make_synthetic_ohlc(4)

    # Anchor bar t=0 for ATR; ATR(0) = 1.0 -> 1R = 1.5
    df.loc[0, ["high", "low", "close"]] = [1.0, 0.0, 0.0]
    df.loc[1:, "open"] = 0.0

    # Adverse bar: big move against the eventual favourable direction
    df.loc[1, ["high", "low", "close"]] = [0.5, -3.0, -3.0]

    # Favourable bar: strong move that exceeds 3R
    df.loc[2, ["high", "low", "close"]] = [6.0, -6.0, 0.0]

    df = calculate_atr(df, period=1)
    cfg = _minimal_phaseD1_cfg(atr_period=1)
    out = generate_labels_for_pair(df, pair="EUR_USD", cfg=cfg)

    t0 = df["date"].iloc[0]
    rows_t0 = out[out["date"] == t0].set_index("direction")

    # Both directions should still see opportunity based on later favourable bar.
    r_long = rows_t0.loc["long"]
    r_short = rows_t0.loc["short"]
    assert bool(r_long["zone_a_1r_10"]) is True
    assert bool(r_short["zone_a_1r_10"]) is True


def test_phaseD1_global_rarity_ordering_zone_counts() -> None:
    """
    Globally: count(Zone A) >= count(Zone B) >= count(Zone C).

    This must hold for any deterministic input as thresholds tighten.
    """
    from core.utils import calculate_atr
    from scripts.phaseD1_generate_opportunity_labels import generate_labels_for_pair

    # Simple random-ish synthetic data across two pairs.
    rng = np.random.default_rng(seed=123)
    dfs = []
    for pair in ("EUR_USD", "GBP_USD"):
        df = _make_synthetic_ohlc(80)
        base = rng.normal(loc=0.0, scale=1.0, size=len(df))
        df["open"] = base.cumsum()
        df["close"] = df["open"] + rng.normal(loc=0.0, scale=0.5, size=len(df))
        spread = np.abs(rng.normal(loc=1.0, scale=0.2, size=len(df)))
        df["high"] = df[["open", "close"]].max(axis=1) + spread
        df["low"] = df[["open", "close"]].min(axis=1) - spread
        df = calculate_atr(df, period=14)
        dfs.append((pair, df))

    cfg = _minimal_phaseD1_cfg(atr_period=14)
    all_rows = []
    for pair, df in dfs:
        all_rows.append(generate_labels_for_pair(df, pair=pair, cfg=cfg))
    out = pd.concat(all_rows, ignore_index=True)

    a = int(out["zone_a_1r_10"].sum())
    b = int(out["zone_b_3r_20"].sum())
    c = int(out["zone_c_6r_40"].sum())
    assert a >= b >= c


def test_phaseD1_deterministic_outputs_for_same_inputs(tmp_path: Path) -> None:
    """
    Determinism: same inputs -> byte-identical outputs when written to disk.
    """
    import shutil

    from core.utils import calculate_atr
    from scripts.phaseD1_generate_opportunity_labels import (
        LABEL_VERSION,
        generate_labels_for_pair,
    )

    df = _make_synthetic_ohlc(40)
    df["open"] = np.linspace(0.0, 10.0, num=len(df))
    df["close"] = df["open"]
    df["high"] = df["open"] + 1.0
    df["low"] = df["open"] - 1.0
    df = calculate_atr(df, period=14)

    cfg = _minimal_phaseD1_cfg(atr_period=14)

    out1 = generate_labels_for_pair(df, pair="EUR_USD", cfg=cfg)
    out2 = generate_labels_for_pair(df, pair="EUR_USD", cfg=cfg)

    # In-memory equality
    pd.testing.assert_frame_equal(
        out1.reset_index(drop=True),
        out2.reset_index(drop=True),
        check_like=True,
    )

    # Deterministic sort order and CSV/Parquet bytes.
    out1_sorted = out1.sort_values(["pair", "date", "direction"]).reset_index(drop=True)
    out2_sorted = out2.sort_values(["pair", "date", "direction"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(out1_sorted, out2_sorted, check_like=False)

    # Write to two separate temp dirs and compare bytes
    out_dir1 = tmp_path / "run1"
    out_dir2 = tmp_path / "run2"
    out_dir1.mkdir(parents=True, exist_ok=True)
    out_dir2.mkdir(parents=True, exist_ok=True)

    csv1 = out_dir1 / "opportunity_labels.csv"
    csv2 = out_dir2 / "opportunity_labels.csv"
    pq1 = out_dir1 / "opportunity_labels.parquet"
    pq2 = out_dir2 / "opportunity_labels.parquet"

    out1_sorted.to_csv(csv1, index=False, float_format="%.8f")
    out2_sorted.to_csv(csv2, index=False, float_format="%.8f")
    assert csv1.read_bytes() == csv2.read_bytes()

    from tests.conftest import has_parquet_engine
    if has_parquet_engine():
        out1_sorted.to_parquet(pq1, index=False)
        out2_sorted.to_parquet(pq2, index=False)
        assert pq1.read_bytes() == pq2.read_bytes()

    # Label version column must be present and consistent
    assert (out1["label_version"] == LABEL_VERSION).all()
    assert (out2["label_version"] == LABEL_VERSION).all()

    # Clean up temp dirs (defensive; pytest will usually handle this)
    shutil.rmtree(out_dir1, ignore_errors=True)
    shutil.rmtree(out_dir2, ignore_errors=True)

