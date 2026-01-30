# analytics/wfo.py — WFO v2 fold generation (rolling windows, no leakage)
from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence


@dataclass(frozen=True)
class Fold:
    """Single walk-forward fold with in-sample (train) and out-of-sample (test) bounds."""

    fold_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def _add_months(d: date, months: int) -> date:
    """Add months to d; clamp day to last day of month if needed. Deterministic."""
    y, m, day = d.year, d.month, d.day
    m += months
    while m > 12:
        m -= 12
        y += 1
    while m < 1:
        m += 12
        y -= 1
    _, last = monthrange(y, m)
    day = min(day, last)
    return date(y, m, day)


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value.strip())


def generate_folds(
    from_date: str | date,
    to_date: str | date,
    train_months: int,
    test_months: int,
    step_months: int,
) -> list[Fold]:
    """
    Generate rolling walk-forward folds with strict train/test separation.

    Folds are built so that for each fold:
    - train is [train_start, train_end] (inclusive)
    - test is [test_start, test_end] (inclusive)
    - train_end < test_start (no leakage).

    First fold: train starts at from_date; subsequent folds advance train_start
    by step_months. Same step advances the implied test window.

    Returns list of Fold (fold_id 1-based).
    """
    start = _parse_date(from_date)
    end = _parse_date(to_date)
    if start >= end:
        raise ValueError("from_date must be strictly before to_date")
    if train_months < 1 or test_months < 1 or step_months < 1:
        raise ValueError("train_months, test_months, and step_months must be positive")

    folds: list[Fold] = []
    fold_id = 1
    train_start = start

    while True:
        train_end = _add_months(train_start, train_months) - _days(1)
        test_start = train_end + _days(1)
        test_end = _add_months(test_start, test_months) - _days(1)

        if test_start > end or test_end > end:
            break

        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_id += 1
        train_start = _add_months(train_start, step_months)

    return folds


def _days(n: int) -> timedelta:
    return timedelta(days=n)


def validate_no_test_overlap(folds: Sequence[Fold], allow_overlap: bool = False) -> None:
    """
    Ensure test windows across folds do not overlap (each day in at most one test window).

    By default (allow_overlap=False) raises ValueError if any two test windows overlap.
    If allow_overlap=True, does nothing.
    """
    if allow_overlap:
        return
    for i, fa in enumerate(folds):
        for fb in folds[i + 1 :]:
            if _ranges_overlap(
                (fa.test_start, fa.test_end),
                (fb.test_start, fb.test_end),
            ):
                raise ValueError(
                    f"Test windows overlap: fold {fa.fold_id} [{fa.test_start}, {fa.test_end}] "
                    f"vs fold {fb.fold_id} [{fb.test_start}, {fb.test_end}]"
                )


def _ranges_overlap(a: tuple[date, date], b: tuple[date, date]) -> bool:
    (a0, a1), (b0, b1) = a, b
    return a0 <= b1 and b0 <= a1
