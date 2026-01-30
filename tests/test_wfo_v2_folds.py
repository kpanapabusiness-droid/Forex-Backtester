# tests/test_wfo_v2_folds.py — WFO v2 fold generation and no-leakage (PR1)
from datetime import date

import pytest

from analytics.wfo import generate_folds, validate_no_test_overlap


def test_fold_count_and_boundaries():
    from_date = "2020-01-01"
    to_date = "2021-12-31"
    train_months = 12
    test_months = 3
    step_months = 3
    folds = generate_folds(from_date, to_date, train_months, test_months, step_months)
    assert len(folds) == 4

    expected = [
        (date(2020, 1, 1), date(2020, 12, 31), date(2021, 1, 1), date(2021, 3, 31)),
        (date(2020, 4, 1), date(2021, 3, 31), date(2021, 4, 1), date(2021, 6, 30)),
        (date(2020, 7, 1), date(2021, 6, 30), date(2021, 7, 1), date(2021, 9, 30)),
        (date(2020, 10, 1), date(2021, 9, 30), date(2021, 10, 1), date(2021, 12, 31)),
    ]
    for i, (f, (ts, te, qs, qe)) in enumerate(zip(folds, expected)):
        assert f.fold_id == i + 1
        assert f.train_start == ts and f.train_end == te
        assert f.test_start == qs and f.test_end == qe


def test_leakage_invariant_every_fold():
    folds = generate_folds("2020-01-01", "2021-12-31", 12, 3, 3)
    for f in folds:
        assert f.train_end < f.test_start, f"fold {f.fold_id}: train_end must be < test_start"


def test_test_window_overlap_validator_passes():
    folds = generate_folds("2020-01-01", "2021-12-31", 12, 3, 3)
    validate_no_test_overlap(folds)


def test_invalid_from_date_after_to_date_raises():
    with pytest.raises(ValueError, match="from_date must be strictly before to_date"):
        generate_folds("2021-12-31", "2020-01-01", 12, 3, 3)
    with pytest.raises(ValueError, match="from_date must be strictly before to_date"):
        generate_folds("2020-06-01", "2020-06-01", 12, 3, 3)


def test_invalid_non_positive_window_sizes_raise():
    with pytest.raises(ValueError, match="must be positive"):
        generate_folds("2020-01-01", "2021-12-31", 0, 3, 3)
    with pytest.raises(ValueError, match="must be positive"):
        generate_folds("2020-01-01", "2021-12-31", 12, 0, 3)
    with pytest.raises(ValueError, match="must be positive"):
        generate_folds("2020-01-01", "2021-12-31", 12, 3, 0)
    with pytest.raises(ValueError, match="must be positive"):
        generate_folds("2020-01-01", "2021-12-31", -1, 3, 3)
