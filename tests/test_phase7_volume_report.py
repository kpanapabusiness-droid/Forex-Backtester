from __future__ import annotations

import json
from pathlib import Path

from analytics.phase7_volume_report import build_leaderboard


def _write_fold_summary(oos_dir: Path, trades: int, max_dd_pct: float, expectancy: float) -> None:
    oos_dir.mkdir(parents=True, exist_ok=True)
    summary_txt = "\n".join(
        [
            f"Total Trades: {trades}",
            f"Max Drawdown (%): {max_dd_pct}",
            f"Expectancy: {expectancy}",
            "",
        ]
    )
    (oos_dir / "summary.txt").write_text(summary_txt, encoding="utf-8")


def _scaffold_baseline_and_discovery(tmp_path: Path) -> Path:
    """
    Create a minimal Phase 7 results scaffold with:
      - results/phase7/wfo/baseline_off/<run_id>/fold_01/out_of_sample
      - results/phase7/phase7_discovery.json
    """
    root = tmp_path / "results" / "phase7"
    wfo_root = root / "wfo"
    baseline_dir = wfo_root / "baseline_off"
    run_dir = baseline_dir / "20260101_000000"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "wfo_run_meta.json").write_text("{}", encoding="utf-8")

    oos_dir = run_dir / "fold_01" / "out_of_sample"
    _write_fold_summary(oos_dir, trades=100, max_dd_pct=30.0, expectancy=0.30)

    discovery = {
        "all": [{"name": "vol_good"}, {"name": "vol_bad"}, {"name": "vol_skip"}],
        "runnable": [{"name": "vol_good"}, {"name": "vol_bad"}],
        "skipped": [{"name": "vol_skip", "reason": "stub/non-functional"}],
    }
    (root / "phase7_discovery.json").write_text(
        json.dumps(discovery, indent=2),
        encoding="utf-8",
    )
    return root


def test_phase7_report_creates_skip_for_missing_wfo(tmp_path: Path) -> None:
    """If a candidate has no WFO folder, the report must emit a SKIP row."""
    root = _scaffold_baseline_and_discovery(tmp_path)

    # Only create runs for vol_good, leave vol_bad and vol_skip missing
    wfo_root = root / "wfo"

    vol_good_dir = wfo_root / "vol_good"
    vg_run = vol_good_dir / "20260101_000001"
    vg_run.mkdir(parents=True, exist_ok=True)
    (vg_run / "wfo_run_meta.json").write_text("{}", encoding="utf-8")
    vg_oos = vg_run / "fold_01" / "out_of_sample"
    _write_fold_summary(vg_oos, trades=80, max_dd_pct=25.0, expectancy=0.30)

    df = build_leaderboard(root)

    # vol_skip is skipped by discovery manifest (stub/non-functional)
    row_skip = df.loc[df["volume_name"] == "vol_skip"].iloc[0]
    assert row_skip["decision"] == "SKIP"
    assert "stub" in str(row_skip["reason"])

    # vol_bad is runnable in discovery but has no WFO folder -> SKIP with no_wfo_run reason
    row_missing = df.loc[df["volume_name"] == "vol_bad"].iloc[0]
    assert row_missing["decision"] == "SKIP"
    assert "no_wfo_run" in str(row_missing["reason"])


def test_phase7_report_keep_and_discard_logic(tmp_path: Path) -> None:
    """
    Synthetic baseline/candidate folds should trigger KEEP/DISCARD decisions:
      - vol_keep improves DD by >=5% and expectancy degrades <=0.02R -> KEEP
      - vol_kill violates DD/expectancy criteria -> DISCARD
    """
    root = _scaffold_baseline_and_discovery(tmp_path)

    wfo_root = root / "wfo"

    # vol_good -> treat as KEEP candidate
    keep_dir = wfo_root / "vol_good"
    keep_run = keep_dir / "20260101_000001"
    keep_run.mkdir(parents=True, exist_ok=True)
    (keep_run / "wfo_run_meta.json").write_text("{}", encoding="utf-8")
    keep_oos = keep_run / "fold_01" / "out_of_sample"
    # Trades: less than baseline, DD improved (24 < 30*0.95=28.5), expectancy only slightly worse
    _write_fold_summary(keep_oos, trades=80, max_dd_pct=24.0, expectancy=0.28)

    # vol_bad -> treat as DISCARD candidate (worse DD and expectancy)
    bad_dir = wfo_root / "vol_bad"
    bad_run = bad_dir / "20260101_000002"
    bad_run.mkdir(parents=True, exist_ok=True)
    (bad_run / "wfo_run_meta.json").write_text("{}", encoding="utf-8")
    bad_oos = bad_run / "fold_01" / "out_of_sample"
    _write_fold_summary(bad_oos, trades=80, max_dd_pct=40.0, expectancy=0.20)

    df = build_leaderboard(root)

    row_keep = df.loc[df["volume_name"] == "vol_good"].iloc[0]
    assert row_keep["decision"] == "KEEP"

    row_discard = df.loc[df["volume_name"] == "vol_bad"].iloc[0]
    assert row_discard["decision"] == "DISCARD"

