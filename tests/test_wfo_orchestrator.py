"""Tests for core/wfo/orchestrator.py — search loop + holdout one-shot."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.wfo.folds import Fold, build_v3_folds
from core.wfo.gates import FoldStats, Verdict
from core.wfo.orchestrator import run_holdout, run_search


@dataclass(frozen=True)
class FakeConfig:
    sl_mult: float
    exit_policy: str


def _make_runner(rec: dict, deploy_fold_ids: set[int] | None = None):
    """Return a fake fold_runner that records call order + invents stats."""
    rec.setdefault("calls", [])

    def runner(fold: Fold, config: FakeConfig) -> FoldStats:
        rec["calls"].append((fold.fold_id, config.sl_mult))
        # Deterministic synthetic stats — strong unless caller flags fold_id as weak.
        weak = deploy_fold_ids is not None and fold.fold_id not in deploy_fold_ids
        roi = 0.02 if weak else 0.07
        dd = 0.025
        return FoldStats(
            fold_id=fold.fold_id,
            n_trades=50,
            roi_pct=roi,
            max_dd_pct=dd,
            days_breaching_daily_5pct=0,
            roi_dd_ratio=roi / dd,
        )

    return runner


def test_run_search_skips_folds_below_min_is_days() -> None:
    """v3 fold 1 has empty IS → should be skipped at default min_is_days=365."""
    s = build_v3_folds()
    rec: dict = {}
    runner = _make_runner(rec)
    cfg = FakeConfig(sl_mult=2.0, exit_policy="sl_only")
    res = run_search(s, candidates=[("c1", cfg)], fold_runner=runner)

    fold_ids_called = [fid for fid, _ in rec["calls"]]
    assert 1 not in fold_ids_called  # empty-IS fold skipped
    assert set(fold_ids_called) == set(range(2, 12))
    assert res.candidates[0].fold_stats[0].fold_id == 2


def test_run_search_min_is_days_zero_evaluates_all() -> None:
    s = build_v3_folds()
    rec: dict = {}
    runner = _make_runner(rec)
    cfg = FakeConfig(sl_mult=2.0, exit_policy="sl_only")
    run_search(s, candidates=[("c1", cfg)], fold_runner=runner, min_is_days=0)
    assert {fid for fid, _ in rec["calls"]} == set(range(1, 12))


def test_run_search_does_not_touch_holdout() -> None:
    """Holdout fold MUST NOT appear in the search call log."""
    s = build_v3_folds()
    rec: dict = {}
    runner = _make_runner(rec)
    cfg = FakeConfig(sl_mult=2.0, exit_policy="sl_only")
    run_search(s, candidates=[("c1", cfg)], fold_runner=runner)
    # Holdout's fold_id is n_folds + 1 = 12
    fold_ids = [fid for fid, _ in rec["calls"]]
    assert s.holdout.fold_id not in fold_ids
    # Also: no OOS year > 2020 in any call
    holdout_year = s.holdout.oos_start.year
    assert holdout_year not in {f.oos_start.year for f in s.folds}  # sanity


def test_run_search_top_k_selection_ranks_by_worst_fold_ratio() -> None:
    s = build_v3_folds()
    rec: dict = {}
    # Candidate A is strong on every fold; candidate B is weak on fold 5
    runner_a = _make_runner({"calls": []})
    runner_b = _make_runner(rec, deploy_fold_ids={2, 3, 4, 6, 7, 8, 9, 10, 11})

    def routing_runner(fold, config):
        if config.exit_policy == "strong":
            return runner_a(fold, config)
        return runner_b(fold, config)

    res = run_search(
        s,
        candidates=[
            ("strong", FakeConfig(sl_mult=2.0, exit_policy="strong")),
            ("weak", FakeConfig(sl_mult=2.0, exit_policy="weak")),
        ],
        fold_runner=routing_runner,
        top_k=2,
    )
    assert res.n_candidates_evaluated == 2
    assert res.top_k[0].config_id == "strong"


def test_run_search_returns_structure_name() -> None:
    s = build_v3_folds()
    rec: dict = {}
    res = run_search(
        s,
        candidates=[("c1", FakeConfig(sl_mult=2.0, exit_policy="sl_only"))],
        fold_runner=_make_runner(rec),
    )
    assert res.structure_name == "v3.0"


def test_run_holdout_evaluates_each_top_k() -> None:
    s = build_v3_folds()
    rec: dict = {}
    runner = _make_runner(rec)
    cfg_a = FakeConfig(sl_mult=2.0, exit_policy="a")
    cfg_b = FakeConfig(sl_mult=2.5, exit_policy="b")
    search = run_search(s, candidates=[("a", cfg_a), ("b", cfg_b)], fold_runner=runner, top_k=2)

    # Clear records — holdout runner inspects its own calls
    rec_holdout: dict = {}
    holdout_runner = _make_runner(rec_holdout)
    holdout_results = run_holdout(s, search.top_k, holdout_runner)
    holdout_fold_ids = {fid for fid, _ in rec_holdout["calls"]}
    # Holdout has fold_id = 12 (after 11 search folds); only one fold per candidate
    assert holdout_fold_ids == {s.holdout.fold_id}
    assert len(holdout_results) == 2


def test_run_holdout_raises_when_no_holdout() -> None:
    from core.wfo.folds import build_kh24_anchor_folds

    s = build_kh24_anchor_folds()
    with pytest.raises(ValueError, match="no holdout window"):
        run_holdout(s, top_k=(), fold_runner=_make_runner({}))


def test_run_holdout_deployable_flag_only_when_both_passes() -> None:
    """Both search gate AND holdout gate must be PASS_DEPLOYABLE."""
    s = build_v3_folds()

    def search_strong_runner(fold, config):
        return FoldStats(
            fold_id=fold.fold_id,
            n_trades=50,
            roi_pct=0.07,
            max_dd_pct=0.025,
            days_breaching_daily_5pct=0,
            roi_dd_ratio=2.8,
        )

    def holdout_weak_runner(fold, config):
        # Holdout weak — single negative fold (ratio 0)
        return FoldStats(
            fold_id=fold.fold_id,
            n_trades=50,
            roi_pct=-0.01,
            max_dd_pct=0.05,
            days_breaching_daily_5pct=0,
            roi_dd_ratio=-0.2,
        )

    cfg = FakeConfig(sl_mult=2.0, exit_policy="sl_only")
    search = run_search(s, candidates=[("c1", cfg)], fold_runner=search_strong_runner)
    holdout = run_holdout(s, search.top_k, holdout_weak_runner)
    assert search.top_k[0].gate.verdict == Verdict.PASS_DEPLOYABLE
    assert holdout[0].holdout_gate.verdict == Verdict.FAIL
    assert holdout[0].deployable is False
