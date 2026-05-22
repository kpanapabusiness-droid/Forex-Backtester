"""Tests for core/wfo/folds.py — v3.0 + KH-24 anchor fold builders."""

from __future__ import annotations

from datetime import date

import pytest

from core.wfo.folds import (
    KH24_N_FOLDS,
    V3_HOLDOUT_START,
    V3_N_FOLDS,
    V3_TRAIN_END,
    V3_TRAIN_START,
    build_kh24_anchor_folds,
    build_v3_folds,
)

# ── v3.0 mode ─────────────────────────────────────────────────────────


def test_v3_produces_11_folds() -> None:
    s = build_v3_folds()
    assert s.n_folds == V3_N_FOLDS == 11
    assert s.name == "v3.0"


def test_v3_first_fold_has_empty_is() -> None:
    """Anchored expanding IS at 2010-01-01 → fold 1 has no prior data."""
    s = build_v3_folds()
    f1 = s.folds[0]
    assert f1.is_empty_is
    assert f1.is_days == 0
    assert f1.oos_start == V3_TRAIN_START
    assert f1.oos_end == date(2010, 12, 31)


def test_v3_each_fold_oos_is_one_year() -> None:
    s = build_v3_folds()
    for f in s.folds:
        # Each OOS year is exactly the calendar year; either 365 or 366 days.
        assert f.oos_days in (365, 366), f"fold {f.fold_id} has {f.oos_days} OOS days"
        assert f.oos_start.year == f.oos_end.year


def test_v3_is_expands_across_folds() -> None:
    """Fold k IS = [2010-01-01, year(2010+k-1) − 1 day]. Expanding monotonically."""
    s = build_v3_folds()
    prior_is_days = -1
    for f in s.folds:
        assert f.is_start == V3_TRAIN_START
        if f.fold_id == 1:
            assert f.is_empty_is
            continue
        assert f.is_days > prior_is_days
        prior_is_days = f.is_days


def test_v3_last_fold_oos_is_2020() -> None:
    s = build_v3_folds()
    assert s.folds[-1].oos_start == date(2020, 1, 1)
    assert s.folds[-1].oos_end == date(2020, 12, 31)


def test_v3_holdout_starts_2021() -> None:
    s = build_v3_folds()
    assert s.holdout is not None
    assert s.holdout.oos_start == V3_HOLDOUT_START
    # IS covers the full training window
    assert s.holdout.is_start == V3_TRAIN_START
    assert s.holdout.is_end == V3_TRAIN_END


def test_v3_holdout_end_defaults_to_last_complete_month() -> None:
    s = build_v3_folds()
    holdout_end = s.holdout.oos_end
    # holdout_end should be the last day of its month
    if holdout_end.month == 12:
        next_month = date(holdout_end.year + 1, 1, 1)
    else:
        next_month = date(holdout_end.year, holdout_end.month + 1, 1)
    assert (next_month - holdout_end).days == 1


def test_v3_no_overlap_between_search_folds() -> None:
    s = build_v3_folds()
    for i, fa in enumerate(s.folds):
        for fb in s.folds[i + 1 :]:
            assert fa.oos_end < fb.oos_start, (
                f"fold {fa.fold_id} OOS {fa.oos_end} overlaps fold {fb.fold_id} start {fb.oos_start}"
            )


def test_v3_holdout_strictly_after_all_search_folds() -> None:
    s = build_v3_folds()
    last_oos_end = max(f.oos_end for f in s.folds)
    assert s.holdout.oos_start > last_oos_end


def test_v3_rejects_mismatched_fold_count() -> None:
    with pytest.raises(ValueError, match="1-year folds require matching counts"):
        build_v3_folds(train_start=date(2010, 1, 1), train_end=date(2020, 12, 31), n_folds=5)


def test_v3_rejects_holdout_end_before_start() -> None:
    with pytest.raises(ValueError, match="holdout_end .* must be ≥ holdout_start"):
        build_v3_folds(holdout_end=date(2020, 12, 31))


# ── KH-24 anchor mode ─────────────────────────────────────────────────


def test_kh24_anchor_produces_7_folds() -> None:
    s = build_kh24_anchor_folds()
    assert s.n_folds == KH24_N_FOLDS == 7
    assert s.name == "kh24_anchor"


def test_kh24_anchor_has_no_holdout() -> None:
    s = build_kh24_anchor_folds()
    assert s.holdout is None


def test_kh24_anchor_first_fold_starts_2020_10_01() -> None:
    s = build_kh24_anchor_folds()
    assert s.folds[0].oos_start == date(2020, 10, 1)


def test_kh24_anchor_each_oos_is_9_months() -> None:
    s = build_kh24_anchor_folds()
    for f in s.folds:
        # 9 months is between 273 and 276 days depending on month lengths
        assert 265 <= f.oos_days <= 280, f"fold {f.fold_id} OOS days = {f.oos_days}"


def test_kh24_anchor_is_is_3_years() -> None:
    s = build_kh24_anchor_folds()
    for f in s.folds:
        # 3 years inclusive ≈ 1095..1097 days
        assert 1090 <= f.is_days <= 1100, f"fold {f.fold_id} IS days = {f.is_days}"


def test_kh24_anchor_oos_windows_contiguous() -> None:
    """Fold k OOS ends one day before fold k+1 OOS starts."""
    s = build_kh24_anchor_folds()
    for fa, fb in zip(s.folds, s.folds[1:]):
        gap = (fb.oos_start - fa.oos_end).days
        assert gap == 1, f"non-contiguous OOS between fold {fa.fold_id} and {fb.fold_id}: gap={gap}"


def test_kh24_anchor_spans_published_lineage() -> None:
    """Per ARC_HISTORY: Oct 2020 → Jan 2026, 7 folds × 9 months."""
    s = build_kh24_anchor_folds()
    first = s.folds[0]
    last = s.folds[-1]
    assert first.oos_start == date(2020, 10, 1)
    # 7 × 9 = 63 months from 2020-10-01 → 2025-12-31 (last day of month 63)
    assert last.oos_end == date(2025, 12, 31)
