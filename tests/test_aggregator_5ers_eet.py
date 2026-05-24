"""Tests for 5ers EET boundary aggregation (PR #187 Sub-change C).

Covers:
  - EET boundary anchors in winter (UTC+2) and summer (UTC+3) wall-clock
  - DST autumn fall-back: no duplicate bars in the EET 02:00-03:00 wrap
  - DST spring-forward: no missing bars across the EET 03:00→04:00 jump
  - Cache namespace + key invalidation parity with the legacy UTC path
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from core.data.aggregator import (
    SUPPORTED_BOUNDARY_CONVENTIONS,
    SUPPORTED_TFS,
    aggregate,
    aggregate_m1_to_tf,
)
from core.data.cache_keys import read_meta
from core.data.histdata_loader import M1_COLUMNS, load_m1
from tests.fixtures.histdata_dst.build import build_dst_fixture
from tests.fixtures.histdata_mini.build import FixtureSpec, build_fixture


@pytest.fixture
def dst_root(tmp_path: Path) -> Path:
    return build_dst_fixture(tmp_path / "histdata")


@pytest.fixture
def m1_dst(dst_root: Path, tmp_path: Path) -> pd.DataFrame:
    return load_m1("EURUSD", histdata_root=dst_root, cache_root=tmp_path / "cache")


@pytest.fixture
def mini_root(tmp_path: Path) -> Path:
    return build_fixture(tmp_path / "histdata", FixtureSpec(minutes_per_month=1440))


# ── Convention plumbing ────────────────────────────────────────────────


def test_supported_conventions_include_utc_and_5ers_eet() -> None:
    assert "utc" in SUPPORTED_BOUNDARY_CONVENTIONS
    assert "5ers_eet" in SUPPORTED_BOUNDARY_CONVENTIONS


def test_unsupported_convention_raises(m1_dst: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unsupported boundary_convention"):
        aggregate_m1_to_tf(m1_dst, "H4", boundary_convention="bogus")


def test_utc_default_byte_identical_to_explicit_utc(mini_root: Path, tmp_path: Path) -> None:
    """Default convention is 'utc' and matches explicit-utc output exactly."""
    a = aggregate(
        "EURUSD",
        "H4",
        histdata_root=mini_root,
        cache_root=tmp_path / "cache_a",
        use_cache=False,
    )
    b = aggregate(
        "EURUSD",
        "H4",
        histdata_root=mini_root,
        cache_root=tmp_path / "cache_b",
        use_cache=False,
        boundary_convention="utc",
    )
    pd.testing.assert_frame_equal(a, b)


# ── EET wall-clock anchoring ───────────────────────────────────────────


def test_h4_anchors_to_eet_in_steady_state_winter(m1_dst: pd.DataFrame) -> None:
    """Steady-state winter (full day post-autumn-fallback): H4 bars at EET
    00/04/08/12/16/20 wall-clock → UTC 22/02/06/10/14/18."""
    out = aggregate_m1_to_tf(m1_dst, "H4", boundary_convention="5ers_eet")
    # 2024-10-29 is fully in EET (post-fallback steady state)
    winter_bars = out.loc["2024-10-29":"2024-10-29 23:59"]
    if len(winter_bars) == 0:
        pytest.skip("fixture window doesn't extend to 2024-10-29")
    expected_utc_hours = {22, 2, 6, 10, 14, 18}
    assert set(winter_bars.index.hour).issubset(expected_utc_hours), (
        f"Winter steady-state H4 bars must anchor to EET 00/04/.../20 "
        f"(UTC 22/02/.../18); got hours={sorted(set(winter_bars.index.hour))}"
    )


def test_h4_anchors_to_eet_in_steady_state_summer(m1_dst: pd.DataFrame) -> None:
    """Steady-state summer (full day post-spring-forward): H4 bars at EEST
    00/04/08/12/16/20 wall-clock → UTC 21/01/05/09/13/17."""
    out = aggregate_m1_to_tf(m1_dst, "H4", boundary_convention="5ers_eet")
    # 2024-04-02 is fully in EEST (post-spring-forward steady state)
    summer_bars = out.loc["2024-04-02":"2024-04-02 23:59"]
    if len(summer_bars) == 0:
        pytest.skip("fixture window doesn't extend to 2024-04-02")
    expected_utc_hours = {21, 1, 5, 9, 13, 17}
    assert set(summer_bars.index.hour).issubset(expected_utc_hours), (
        f"Summer steady-state H4 bars must anchor to EEST 00/04/.../20 "
        f"(UTC 21/01/.../17); got hours={sorted(set(summer_bars.index.hour))}"
    )


def test_d1_anchors_to_eet_midnight(m1_dst: pd.DataFrame) -> None:
    """D1 bars labelled at EET 00:00 → UTC 22:00 winter / UTC 21:00 summer."""
    out = aggregate_m1_to_tf(m1_dst, "D1", boundary_convention="5ers_eet")
    hours = set(out.index.hour)
    # All D1 bars should fall at UTC 21 or 22 depending on season
    assert hours.issubset({21, 22}), (
        f"D1 EET bars must label at UTC 21 (summer) or UTC 22 (winter); got {sorted(hours)}"
    )


def test_w1_anchors_to_eet_monday(m1_dst: pd.DataFrame) -> None:
    """W1 bars label at Monday 00:00 EET (= UTC 22:00 Sun winter / 21:00 Sun summer)."""
    out = aggregate_m1_to_tf(m1_dst, "W1", boundary_convention="5ers_eet")
    # Convert UTC label back to EET wall-clock to verify Monday 00:00 anchor.
    eet_index = out.index.tz_convert("Europe/Athens")
    assert all(d.weekday() == 0 for d in eet_index), (
        f"W1 EET bars must label at Monday in EET wall-clock; got "
        f"weekdays={[d.weekday() for d in eet_index]}"
    )
    assert all(d.hour == 0 for d in eet_index), (
        f"W1 EET bars must label at 00:00 EET; got hours={[d.hour for d in eet_index]}"
    )
    # UTC label hour is 21 or 22 depending on DST regime
    assert set(out.index.hour).issubset({21, 22}), (
        f"W1 EET bars must label at UTC 21/22; got {sorted(set(out.index.hour))}"
    )


# ── DST transition handling ────────────────────────────────────────────


def test_dst_autumn_no_duplicate_bars(m1_dst: pd.DataFrame) -> None:
    """Autumn fall-back: EET 02:00-03:00 wall-clock occurs twice on 2024-10-27.

    The aggregator's UTC-indexed output must remain monotonic with no
    duplicate timestamps despite the wall-clock ambiguity.
    """
    out = aggregate_m1_to_tf(m1_dst, "H4", boundary_convention="5ers_eet")
    fall_back_window = out.loc["2024-10-26":"2024-10-28"]
    assert fall_back_window.index.is_unique, "Duplicate UTC bar labels around DST fall-back"
    assert fall_back_window.index.is_monotonic_increasing, "Non-monotonic UTC index across DST"


def test_dst_total_volume_conserved(m1_dst: pd.DataFrame) -> None:
    """Across the entire DST-spanning fixture, total bar volume == total M1
    volume — no minute is dropped through spring-forward or fall-back."""
    out = aggregate_m1_to_tf(m1_dst, "H4", boundary_convention="5ers_eet")
    assert out["volume"].sum() == m1_dst["volume"].sum(), (
        "M1 volume not conserved across DST transitions (H4 5ers_eet)"
    )


def test_dst_d1_volume_conserved(m1_dst: pd.DataFrame) -> None:
    """D1 5ers_eet must also conserve volume across DST."""
    out = aggregate_m1_to_tf(m1_dst, "D1", boundary_convention="5ers_eet")
    assert out["volume"].sum() == m1_dst["volume"].sum(), (
        "M1 volume not conserved across DST transitions (D1 5ers_eet)"
    )


# ── Cache namespace separation ─────────────────────────────────────────


def test_cache_namespace_separates_conventions(mini_root: Path, tmp_path: Path) -> None:
    """UTC and 5ers_eet caches live under distinct directories."""
    cache_root = tmp_path / "cache"
    aggregate("EURUSD", "H4", histdata_root=mini_root, cache_root=cache_root)
    aggregate(
        "EURUSD",
        "H4",
        histdata_root=mini_root,
        cache_root=cache_root,
        boundary_convention="5ers_eet",
    )
    assert (cache_root / "H4" / "EURUSD.parquet").exists()
    assert (cache_root / "H4_5ers_eet" / "EURUSD.parquet").exists()


def test_cache_keys_differ_between_conventions(mini_root: Path, tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    aggregate("EURUSD", "H4", histdata_root=mini_root, cache_root=cache_root)
    aggregate(
        "EURUSD",
        "H4",
        histdata_root=mini_root,
        cache_root=cache_root,
        boundary_convention="5ers_eet",
    )
    utc_meta = read_meta(cache_root / "H4" / "EURUSD.parquet")
    eet_meta = read_meta(cache_root / "H4_5ers_eet" / "EURUSD.parquet")
    assert utc_meta is not None and eet_meta is not None
    assert utc_meta.cache_key != eet_meta.cache_key, (
        "Cache keys must differ between UTC and 5ers_eet conventions"
    )
    assert utc_meta.layer == "H4"
    assert eet_meta.layer == "H4_5ers_eet"


def test_cache_byte_identical_two_runs(m1_dst: pd.DataFrame, tmp_path: Path) -> None:
    """Two EET aggregations of the same M1 input produce byte-identical parquet."""
    a = aggregate_m1_to_tf(m1_dst, "H4", boundary_convention="5ers_eet")
    b = aggregate_m1_to_tf(m1_dst, "H4", boundary_convention="5ers_eet")
    pd.testing.assert_frame_equal(a, b)

    p_a = tmp_path / "a.parquet"
    p_b = tmp_path / "b.parquet"
    a.to_parquet(p_a, engine="pyarrow", compression="snappy", index=True)
    b.to_parquet(p_b, engine="pyarrow", compression="snappy", index=True)
    sha_a = hashlib.sha256(p_a.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(p_b.read_bytes()).hexdigest()
    assert sha_a == sha_b, "Two-run EET aggregation produced different parquet bytes"


# ── Schema parity ──────────────────────────────────────────────────────


@pytest.mark.parametrize("tf", SUPPORTED_TFS)
def test_eet_aggregate_preserves_canonical_schema(m1_dst: pd.DataFrame, tf: str) -> None:
    out = aggregate_m1_to_tf(m1_dst, tf, boundary_convention="5ers_eet")
    assert list(out.columns) == M1_COLUMNS
    assert out.index.tz is not None
    assert str(out.index.tz) == "UTC", "Output bar index must be tz_convert'd back to UTC"
