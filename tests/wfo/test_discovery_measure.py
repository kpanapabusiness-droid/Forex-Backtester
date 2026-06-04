"""Unit tests for the discovery measurement primitives
(``core/wfo/discovery_measure.py``).

Pure-function tests — no data corpus, no engine run — so they gate in CI
(unmarked = not ``research``). They lock the two genuinely-missing measurement
pieces (per-year OOS folds + the all-folds-positive discovery judge); a bug in
either is invisible to the gate (it IS the gate), so it is CI-pinned here.
"""

from __future__ import annotations

from datetime import date

import pytest

from core.wfo.discovery_measure import (
    DiscoveryVerdict,
    build_oos_year_folds,
    judge_all_folds_positive,
    run_config_over_folds,
)
from core.wfo.folds import Fold
from core.wfo.gates import FoldStats


def _fs(fold_id: int, roi: float, n_trades: int = 100) -> FoldStats:
    return FoldStats(
        fold_id=fold_id,
        n_trades=n_trades,
        roi_pct=roi,
        max_dd_pct=0.05,
        days_breaching_daily_5pct=0,
        roi_dd_ratio=(roi / 0.05) if roi else 0.0,
    )


# ---- build_oos_year_folds ------------------------------------------------


def test_oos_year_folds_one_per_year_2021_to_end():
    folds = build_oos_year_folds(start_year=2021, end=date(2026, 5, 31))
    assert [f.fold_id for f in folds] == [2021, 2022, 2023, 2024, 2025, 2026]
    # full prior years run Jan-1..Dec-31
    for f in folds[:-1]:
        assert f.oos_start == date(f.fold_id, 1, 1)
        assert f.oos_end == date(f.fold_id, 12, 31)
    # final (current, incomplete) year clamps to `end`
    assert folds[-1].oos_start == date(2026, 1, 1)
    assert folds[-1].oos_end == date(2026, 5, 31)


def test_oos_year_folds_fixed_is_window_and_no_lookahead():
    folds = build_oos_year_folds(start_year=2021, end=date(2024, 12, 31))
    for f in folds:
        # IS pinned to the v3 dev window 2010-2020
        assert f.is_start == date(2010, 1, 1)
        assert f.is_end == date(2020, 12, 31)
        # OOS strictly AFTER IS — lookahead-safe (train on past, measure future)
        assert f.oos_start > f.is_end


def test_oos_year_folds_custom_window():
    folds = build_oos_year_folds(start_year=2022, end=date(2023, 12, 31))
    assert [f.fold_id for f in folds] == [2022, 2023]


def test_oos_year_folds_rejects_start_after_end():
    with pytest.raises(ValueError):
        build_oos_year_folds(start_year=2027, end=date(2026, 5, 31))


def test_oos_year_folds_rejects_bad_is_window():
    with pytest.raises(ValueError):
        build_oos_year_folds(is_start=date(2021, 1, 1), is_end=date(2020, 1, 1))


# ---- judge_all_folds_positive --------------------------------------------


def test_judge_all_positive_passes():
    v = judge_all_folds_positive([_fs(1, 0.02), _fs(2, 0.05), _fs(3, 0.001)])
    assert isinstance(v, DiscoveryVerdict)
    assert v.all_folds_positive is True
    assert v.n_negative_folds == 0
    assert v.worst_fold_roi == pytest.approx(0.001)
    assert v.n_folds == 3


def test_judge_one_negative_fails():
    v = judge_all_folds_positive([_fs(1, 0.02), _fs(2, -0.03), _fs(3, 0.01)])
    assert v.all_folds_positive is False
    assert v.n_negative_folds == 1
    assert v.worst_fold_roi == pytest.approx(-0.03)


def test_judge_zero_fold_is_not_positive_but_not_negative():
    v = judge_all_folds_positive([_fs(1, 0.02), _fs(2, 0.0)])
    assert v.all_folds_positive is False  # 0.0 is not > 0
    assert v.n_negative_folds == 0  # ...but 0.0 is not < 0 either


def test_judge_min_trades_tracked():
    v = judge_all_folds_positive([_fs(1, 0.02, n_trades=80), _fs(2, 0.03, n_trades=42)])
    assert v.min_trades_per_fold == 42


def test_judge_empty_raises():
    with pytest.raises(ValueError):
        judge_all_folds_positive([])


# ---- run_config_over_folds -----------------------------------------------


def test_run_config_over_folds_calls_runner_per_fold():
    folds = build_oos_year_folds(start_year=2021, end=date(2023, 12, 31))
    calls = []

    def fake_runner(fold: Fold, config) -> FoldStats:
        calls.append((fold.fold_id, config))
        return _fs(fold.fold_id, 0.01)

    out = run_config_over_folds(fake_runner, folds, config="cfg")
    assert [fs.fold_id for fs in out] == [2021, 2022, 2023]
    assert calls == [(2021, "cfg"), (2022, "cfg"), (2023, "cfg")]
    # output feeds straight into the judge
    assert judge_all_folds_positive(out).all_folds_positive is True
