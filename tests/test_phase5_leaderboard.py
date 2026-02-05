"""Phase 5 — Leaderboard schema, determinism, and rejection tagging."""
from pathlib import Path

import yaml

from analytics.phase5_leaderboard import (
    LEADERBOARD_COLUMNS,
    SCRATCH_RATE_MEAN_CEILING,
    build_leaderboard,
)


def _write_summary(path: Path, roi_pct: float, max_dd_pct: float, trades: int, scratches: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join([
        f"Total Trades: {trades}",
        f"Scratches: {scratches}",
        f"ROI (%): {roi_pct}",
        f"Max DD (%): {max_dd_pct}",
    ])
    path.write_text(text, encoding="utf-8")


def _write_base_config_used(run_dir: Path, c1_name: str, params: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "indicators": {"c1": c1_name},
        "indicator_params": {c1_name: params},
    }
    (run_dir / "base_config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )


def test_phase5_leaderboard_schema_columns_exact(tmp_path):
    phase5 = tmp_path / "phase5"
    c1_dir = phase5 / "c1_hlc_trend"
    run_dir = c1_dir / "run_001"
    _write_base_config_used(run_dir, "c1_hlc_trend", {"period": 14})
    for fold_idx in (1, 2):
        oos = run_dir / f"fold_{fold_idx:02d}" / "out_of_sample"
        oos.mkdir(parents=True, exist_ok=True)
        _write_summary(oos / "summary.txt", roi_pct=5.0, max_dd_pct=-3.0, trades=20, scratches=2)
    df = build_leaderboard(phase5)
    assert list(df.columns) == LEADERBOARD_COLUMNS, (
        f"Leaderboard columns must match exactly: expected {LEADERBOARD_COLUMNS}, got {list(df.columns)}"
    )


def test_phase5_leaderboard_determinism_identical_csv_bytes(tmp_path):
    phase5 = tmp_path / "phase5"
    c1_dir = phase5 / "c1_rsi"
    run_dir = c1_dir / "run_001"
    _write_base_config_used(run_dir, "c1_rsi", {"period": 7})
    for fold_idx in (1, 2):
        oos = run_dir / f"fold_{fold_idx:02d}" / "out_of_sample"
        oos.mkdir(parents=True, exist_ok=True)
        _write_summary(oos / "summary.txt", roi_pct=2.0, max_dd_pct=-4.0, trades=15, scratches=1)
    df1 = build_leaderboard(phase5)
    csv_path = tmp_path / "leaderboard.csv"
    df1.to_csv(csv_path, index=False)
    bytes1 = csv_path.read_bytes()
    df2 = build_leaderboard(phase5)
    df2.to_csv(csv_path, index=False)
    bytes2 = csv_path.read_bytes()
    assert bytes1 == bytes2, "Running leaderboard twice must produce identical CSV bytes"


def test_phase5_leaderboard_reject_zero_trades_in_fold(tmp_path):
    phase5 = tmp_path / "phase5"
    c1_dir = phase5 / "c1_forecast"
    run_dir = c1_dir / "run_zero"
    _write_base_config_used(run_dir, "c1_forecast", {"period": 10})
    (run_dir / "fold_01" / "out_of_sample").mkdir(parents=True, exist_ok=True)
    _write_summary(
        run_dir / "fold_01" / "out_of_sample" / "summary.txt",
        roi_pct=0.0, max_dd_pct=0.0, trades=10, scratches=0,
    )
    (run_dir / "fold_02" / "out_of_sample").mkdir(parents=True, exist_ok=True)
    _write_summary(
        run_dir / "fold_02" / "out_of_sample" / "summary.txt",
        roi_pct=0.0, max_dd_pct=0.0, trades=0, scratches=0,
    )
    df = build_leaderboard(phase5)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["status"] == "REJECT"
    assert row["reject_reason"] == "zero_trades_in_fold"


def test_phase5_leaderboard_reject_scratch_rate_above_ceiling(tmp_path):
    phase5 = tmp_path / "phase5"
    c1_dir = phase5 / "c1_lwpi"
    run_dir = c1_dir / "run_scratch"
    _write_base_config_used(run_dir, "c1_lwpi", {"period": 14})
    for fold_idx in (1, 2):
        oos = run_dir / f"fold_{fold_idx:02d}" / "out_of_sample"
        oos.mkdir(parents=True, exist_ok=True)
        trades = 20
        scratches = int(trades * (SCRATCH_RATE_MEAN_CEILING + 0.1))
        _write_summary(oos / "summary.txt", roi_pct=1.0, max_dd_pct=-2.0, trades=trades, scratches=scratches)
    df = build_leaderboard(phase5)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["status"] == "REJECT"
    assert row["reject_reason"] == "scratch_rate_above_ceiling"
    assert row["reject_reason"] != ""
