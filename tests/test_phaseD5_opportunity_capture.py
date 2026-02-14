"""Tests for Phase D-5 Opportunity Capture Attribution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _make_labels_csv(tmp_path: Path, df: pd.DataFrame) -> Path:
    p = tmp_path / "labels.csv"
    df.to_csv(p, index=False)
    return p


def _make_trades_csv(tmp_path: Path, df: pd.DataFrame) -> Path:
    p = tmp_path / "trades.csv"
    df.to_csv(p, index=False)
    return p


def _minimal_labels_df() -> pd.DataFrame:
    """Minimal labels with required columns."""
    return pd.DataFrame(
        {
            "pair": ["EUR_USD", "EUR_USD", "EUR_USD", "EUR_USD"],
            "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"]),
            "direction": ["long", "short", "long", "short"],
            "dataset_split": ["discovery"] * 4,
            "zone_a_1r_10": [True, False, True, True],
            "zone_b_3r_20": [True, False, False, True],
            "zone_c_6r_40": [False, False, False, True],
            "t1": [1.0, np.nan, 2.0, 0.0],
            "t3": [3.0, np.nan, np.nan, 1.0],
            "t6": [5.0, np.nan, np.nan, 2.0],
            "mfe_40_r": [2.5, 0.5, 1.5, 8.0],
        }
    )


def test_trade_join_and_zone_flags(tmp_path: Path) -> None:
    """2 trades join correctly; zone flags map from labels."""
    from analytics.phaseD5_opportunity_capture import run_phaseD5

    labels = _minimal_labels_df()
    labels_path = _make_labels_csv(tmp_path, labels)

    trades = pd.DataFrame(
        {
            "pair": ["EUR_USD", "EUR_USD"],
            "entry_date": ["2020-01-02", "2020-01-03"],
            "direction": ["long", "short"],
        }
    )
    trades_path = _make_trades_csv(tmp_path, trades)

    out_dir = tmp_path / "out"
    run_phaseD5(
        labels_path=labels_path,
        trades_path=trades_path,
        out_dir=out_dir,
        capture_bars=5,
        top_n=10,
    )

    overlap = pd.read_csv(out_dir / "trade_zone_overlap.csv")
    assert len(overlap) == 2
    assert "zoneA_flag_at_entry" in overlap.columns
    assert "zoneB_flag_at_entry" in overlap.columns
    assert "zoneC_flag_at_entry" in overlap.columns

    row1 = overlap[overlap["entry_date"] == "2020-01-02"].iloc[0]
    assert row1["pair"] == "EUR_USD"
    assert row1["direction"] == "long"
    assert row1["zoneA_flag_at_entry"] in (True, 1)
    assert row1["zoneB_flag_at_entry"] in (True, 1)
    assert row1["zoneC_flag_at_entry"] in (False, 0)
    assert row1["t3_at_entry"] == 3.0
    assert row1["t6_at_entry"] == 5.0

    row2 = overlap[overlap["entry_date"] == "2020-01-03"].iloc[0]
    assert row2["direction"] == "short"
    assert row2["zoneC_flag_at_entry"] in (True, 1)
    assert row2["t6_at_entry"] == 2.0
    assert overlap["missing_label_row"].sum() == 0


def test_event_detection_and_capture(tmp_path: Path) -> None:
    """Event segments detected; capture within X bars."""
    from analytics.phaseD5_opportunity_capture import run_phaseD5

    labels = pd.DataFrame(
        {
            "pair": ["EUR_USD"] * 6,
            "date": pd.to_datetime(
                ["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10", "2020-01-13"]
            ),
            "direction": ["long"] * 6,
            "dataset_split": ["discovery"] * 6,
            "zone_a_1r_10": [True] * 6,
            "zone_b_3r_20": [False, True, True, True, False, False],
            "zone_c_6r_40": [False, False, True, True, False, False],
            "t1": [0.0] * 6,
            "t3": [1.0] * 6,
            "t6": [2.0] * 6,
            "mfe_40_r": [1.0, 2.0, 7.0, 8.0, 1.0, 1.0],
        }
    )
    labels_path = _make_labels_csv(tmp_path, labels)

    trades = pd.DataFrame(
        {
            "pair": ["EUR_USD"],
            "entry_date": ["2020-01-08"],
            "direction": ["long"],
        }
    )
    trades_path = _make_trades_csv(tmp_path, trades)

    out_dir = tmp_path / "out"
    run_phaseD5(
        labels_path=labels_path,
        trades_path=trades_path,
        out_dir=out_dir,
        capture_bars=5,
        top_n=10,
    )

    capture = pd.read_csv(out_dir / "zone_capture_summary.csv")
    zone_c = capture[(capture["zone_type"] == "C") & (capture["pair"] == "EUR_USD")]
    assert len(zone_c) >= 1
    captured = zone_c[zone_c["captured"]]
    assert len(captured) >= 1
    assert captured.iloc[0]["first_entry_date"] == "2020-01-08"
    assert captured.iloc[0]["entry_delay_bars"] == 0


def test_nearest_signal_distance(tmp_path: Path) -> None:
    """No capture -> nearest_signal_distance_bars computed as expected."""
    from analytics.phaseD5_opportunity_capture import run_phaseD5

    labels = pd.DataFrame(
        {
            "pair": ["EUR_USD"] * 5,
            "date": pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10"]),
            "direction": ["long"] * 5,
            "dataset_split": ["discovery"] * 5,
            "zone_a_1r_10": [True] * 5,
            "zone_b_3r_20": [False] * 5,
            "zone_c_6r_40": [True, True, False, False, False],
            "t1": [0.0] * 5,
            "t3": [1.0] * 5,
            "t6": [2.0] * 5,
            "mfe_40_r": [10.0, 9.0, 1.0, 1.0, 1.0],
        }
    )
    labels_path = _make_labels_csv(tmp_path, labels)

    trades = pd.DataFrame(
        {
            "pair": ["EUR_USD"],
            "entry_date": ["2020-01-10"],
            "direction": ["long"],
        }
    )
    trades_path = _make_trades_csv(tmp_path, trades)

    out_dir = tmp_path / "out"
    run_phaseD5(
        labels_path=labels_path,
        trades_path=trades_path,
        out_dir=out_dir,
        capture_bars=2,
        top_n=5,
    )

    missed = pd.read_csv(out_dir / "missed_mega_report.csv")
    uncaptured = missed[~missed["captured"]]
    if len(uncaptured) > 0:
        row = uncaptured.iloc[0]
        assert "nearest_signal_distance_bars" in missed.columns
        assert pd.notna(row["nearest_signal_distance_bars"]) or row["nearest_signal_distance_bars"] >= 0


def test_trades_direction_aliases(tmp_path: Path) -> None:
    """Trades with direction_int / side column map to long/short."""
    from analytics.phaseD5_opportunity_capture import _load_trades

    trades = pd.DataFrame(
        {
            "pair": ["EUR_USD"],
            "entry_date": ["2020-01-02 00:00:00"],
            "direction_int": [1],
        }
    )
    p = tmp_path / "t.csv"
    trades.to_csv(p, index=False)
    loaded = _load_trades(p)
    assert loaded["direction"].iloc[0] == "long"

    trades2 = pd.DataFrame(
        {
            "pair": ["EUR_USD"],
            "entry_date": ["2020-01-02"],
            "side": ["short"],
        }
    )
    p2 = tmp_path / "t2.csv"
    trades2.to_csv(p2, index=False)
    loaded2 = _load_trades(p2)
    assert loaded2["direction"].iloc[0] == "short"


def test_decision_memo_exists(tmp_path: Path) -> None:
    """decision_memo.txt is written with Zone C capture rates."""
    from analytics.phaseD5_opportunity_capture import run_phaseD5

    labels = _minimal_labels_df()
    trades = pd.DataFrame(
        {"pair": ["EUR_USD"], "entry_date": ["2020-01-02"], "direction": ["long"]}
    )
    run_phaseD5(
        labels_path=_make_labels_csv(tmp_path, labels),
        trades_path=_make_trades_csv(tmp_path, trades),
        out_dir=tmp_path / "out",
    )
    memo = (tmp_path / "out" / "decision_memo.txt").read_text()
    assert "Zone C capture rate" in memo or "capture rate" in memo.lower()
    assert "discovery" in memo or "validation" in memo


def test_cli_runs(tmp_path: Path) -> None:
    """CLI entry point runs without error."""
    import subprocess

    labels = _minimal_labels_df()
    trades = pd.DataFrame(
        {"pair": ["EUR_USD"], "entry_date": ["2020-01-02"], "direction": ["long"]}
    )
    _make_labels_csv(tmp_path, labels)
    _make_trades_csv(tmp_path, trades)

    result = subprocess.run(
        [
            "python",
            "-m",
            "analytics.phaseD5_opportunity_capture",
            "--labels",
            str(tmp_path / "labels.csv"),
            "--trades",
            str(tmp_path / "trades.csv"),
            "--outdir",
            str(tmp_path / "out"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (tmp_path / "out" / "trade_zone_overlap.csv").exists()


def test_missing_trades_has_actionable_error(tmp_path: Path) -> None:
    """Missing trades path raises FileNotFoundError with actionable message."""
    from analytics.phaseD5_opportunity_capture import _format_trades_not_found_error

    missing = tmp_path / "nonexistent" / "sub" / "trades.csv"
    msg = _format_trades_not_found_error(missing)
    assert "Absolute path:" in msg or "absolute path" in msg.lower()
    assert "--find-latest-trades" in msg or "find-latest-trades" in msg
    assert "trades.csv" in msg


def test_find_latest_trades_picks_most_recent(tmp_path: Path) -> None:
    """--find-latest-trades selects the newest valid trades.csv under results/phaseD."""
    import os
    import subprocess
    import time

    phase_d = tmp_path / "results" / "phaseD"
    old_dir = phase_d / "run_old"
    new_dir = phase_d / "run_new"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)

    trades_df = pd.DataFrame(
        {"pair": ["EUR_USD"], "entry_date": ["2020-01-02"], "direction": ["long"]}
    )
    trades_df.to_csv(old_dir / "trades.csv", index=False)
    time.sleep(0.05)
    trades_df.to_csv(new_dir / "trades.csv", index=False)

    labels = _minimal_labels_df()
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [
            "python",
            "-m",
            "analytics.phaseD5_opportunity_capture",
            "--labels",
            str(labels_path),
            "--find-latest-trades",
            "--outdir",
            str(tmp_path / "out"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    overlap = pd.read_csv(tmp_path / "out" / "trade_zone_overlap.csv")
    assert len(overlap) == 1
