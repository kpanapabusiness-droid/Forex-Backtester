from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _write_fake_fold(run_dir: Path, fold_id: int, roi_pct: float, max_dd_pct: float, trades: int):
    fold_dir = run_dir / f"fold_{fold_id:02d}"
    oos_dir = fold_dir / "out_of_sample"
    oos_dir.mkdir(parents=True, exist_ok=True)
    dates = {
        "train_start": f"2020-01-0{fold_id}",
        "train_end": f"2020-03-0{fold_id}",
        "test_start": f"2020-04-0{fold_id}",
        "test_end": f"2020-04-1{fold_id}",
    }
    (fold_dir / "fold_dates.json").write_text(json.dumps(dates, indent=2), encoding="utf-8")
    (fold_dir / "params_hash.txt").write_text(f"hash_{fold_id}", encoding="utf-8")
    summary_lines = [
        "📊 Backtest Summary",
        "-------------------",
        f"Total Trades : {trades}",
        f"ROI (%)      : {roi_pct}",
        f"Max DD (%)   : {max_dd_pct}",
    ]
    (oos_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def test_wfo_aggregation_outputs_and_content(tmp_path: Path):
    from analytics.wfo_aggregate import aggregate_wfo_run

    run_dir = tmp_path / "wfo_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_fold(run_dir, fold_id=1, roi_pct=10.0, max_dd_pct=5.0, trades=10)
    _write_fake_fold(run_dir, fold_id=2, roi_pct=-5.0, max_dd_pct=12.0, trades=5)
    _write_fake_fold(run_dir, fold_id=3, roi_pct=0.0, max_dd_pct=8.0, trades=0)

    aggregate_wfo_run(run_dir)

    summary_csv = run_dir / "wfo_summary.csv"
    aggregate_json = run_dir / "wfo_aggregate.json"
    assert summary_csv.exists()
    assert aggregate_json.exists()

    df = pd.read_csv(summary_csv)
    assert len(df) == 3
    assert set(["fold_id", "train_start", "test_start", "roi_pct", "max_dd_pct", "total_trades"]).issubset(
        df.columns
    )

    agg = json.loads(aggregate_json.read_text(encoding="utf-8"))
    assert "worst_case_fold" in agg
    assert "median_fold" in agg
    assert "consistency_score" in agg
    assert agg["worst_case_fold"]["fold_id"] == 2
    assert agg["worst_case_fold"]["roi_pct"] == -5.0

    # Determinism: running again should produce identical aggregate
    aggregate_wfo_run(run_dir)
    agg2 = json.loads(aggregate_json.read_text(encoding="utf-8"))
    assert agg == agg2

