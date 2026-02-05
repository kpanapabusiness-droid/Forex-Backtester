import json
from pathlib import Path

import pandas as pd

from analytics.phase4_extract_wfo_folds import extract_wfo_v2_folds


def _write_summary(path: Path, *, roi_pct: float, max_dd_pct: float, trades: int, scratches: int) -> None:
    text = "\n".join(
        [
            f"Total Trades: {trades}",
            f"Scratches: {scratches}",
            f"ROI (%): {roi_pct}",
            f"Max DD (%): {max_dd_pct}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_phase4_extract_wfo_folds_creates_normalized_csv(tmp_path):
    phase4_root = tmp_path / "phase4"
    wfo_root = phase4_root / "wfo"

    # Build nested tree for two C1s:
    # phase4/wfo/wfo_c1_c1_a/20260101_000000/fold_01/out_of_sample/summary.txt
    # phase4/wfo/wfo_c1_c1_b/20260101_000000/fold_01/out_of_sample/summary.txt
    for c1_name, roi_pct, max_dd_pct, trades, scratches in [
        ("c1_a", 5.0, -4.0, 10, 1),
        ("c1_b", 3.0, -6.0, 15, 2),
    ]:
        c1_dir = wfo_root / f"wfo_c1_{c1_name}"
        run_dir = c1_dir / "20260101_000000"
        fold_dir = run_dir / "fold_01"
        oos_dir = fold_dir / "out_of_sample"
        oos_dir.mkdir(parents=True, exist_ok=True)

        fold_dates = {
            "train_start": "2020-01-01",
            "train_end": "2020-03-31",
            "test_start": "2020-04-01",
            "test_end": "2020-04-30",
        }
        (fold_dir / "fold_dates.json").write_text(json.dumps(fold_dates), encoding="utf-8")
        _write_summary(
            oos_dir / "summary.txt",
            roi_pct=roi_pct,
            max_dd_pct=max_dd_pct,
            trades=trades,
            scratches=scratches,
        )

    # Run extractor
    extract_wfo_v2_folds(wfo_root)

    # Assert normalized outputs exist and are correct
    for c1_name, roi_pct, max_dd_pct, trades, scratches in [
        ("c1_a", 5.0, -4.0, 10, 1),
        ("c1_b", 3.0, -6.0, 15, 2),
    ]:
        out_dir = phase4_root / f"wfo_c1_{c1_name}"
        out_csv = out_dir / "wfo_folds.csv"
        assert out_csv.exists()

        df = pd.read_csv(out_csv)

        # Exact columns and order
        assert list(df.columns) == [
            "fold_idx",
            "test_start",
            "test_end",
            "roi_pct",
            "max_dd_pct",
            "trades",
            "scratches",
        ]

        # Sorted by fold_idx ascending
        assert df["fold_idx"].tolist() == sorted(df["fold_idx"].tolist())

        # Values match inputs for the single fold
        assert len(df) == 1
        row = df.iloc[0]
        assert row["fold_idx"] == 1
        assert row["test_start"] == "2020-04-01"
        assert row["test_end"] == "2020-04-30"
        assert row["roi_pct"] == roi_pct
        assert row["max_dd_pct"] == max_dd_pct
        assert row["trades"] == trades
        assert row["scratches"] == scratches

