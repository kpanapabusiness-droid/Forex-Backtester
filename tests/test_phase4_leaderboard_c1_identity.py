import pandas as pd
import pytest

from analytics.leaderboard_c1_identity import build_leaderboard


def _write_wfo_folds(path, rows):
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_phase4_leaderboard_identity_happy_path(tmp_path):
    root = tmp_path / "phase4"

    # C1 A: two folds, worst fold ROI negative → REJECT
    _write_wfo_folds(
        root / "wfo_c1_a" / "wfo_folds.csv",
        [
            {
                "fold_idx": 1,
                "train_start": "2020-01-01",
                "train_end": "2020-03-31",
                "test_start": "2020-04-01",
                "test_end": "2020-04-30",
                "trades": 20,
                "wins": 10,
                "losses": 8,
                "scratches": 2,
                "win_pct_ns": 55.0,
                "loss_pct_ns": 45.0,
                "expectancy": 0.5,
                "roi_pct": 5.0,
                "max_dd_pct": -5.0,
            },
            {
                "fold_idx": 2,
                "train_start": "2020-05-01",
                "train_end": "2020-07-31",
                "test_start": "2020-08-01",
                "test_end": "2020-08-31",
                "trades": 15,
                "wins": 5,
                "losses": 8,
                "scratches": 2,
                "win_pct_ns": 38.0,
                "loss_pct_ns": 62.0,
                "expectancy": -0.3,
                "roi_pct": -2.0,
                "max_dd_pct": -10.0,
            },
        ],
    )

    # C1 B: two folds, all ROI positive → SURVIVOR
    _write_wfo_folds(
        root / "wfo_c1_b" / "wfo_folds.csv",
        [
            {
                "fold_idx": 1,
                "train_start": "2020-01-01",
                "train_end": "2020-03-31",
                "test_start": "2020-04-01",
                "test_end": "2020-04-30",
                "trades": 30,
                "wins": 18,
                "losses": 10,
                "scratches": 2,
                "win_pct_ns": 64.0,
                "loss_pct_ns": 36.0,
                "expectancy": 0.8,
                "roi_pct": 8.0,
                "max_dd_pct": -4.0,
            },
            {
                "fold_idx": 2,
                "train_start": "2020-05-01",
                "train_end": "2020-07-31",
                "test_start": "2020-08-01",
                "test_end": "2020-08-31",
                "trades": 25,
                "wins": 14,
                "losses": 9,
                "scratches": 2,
                "win_pct_ns": 61.0,
                "loss_pct_ns": 39.0,
                "expectancy": 0.6,
                "roi_pct": 3.0,
                "max_dd_pct": -6.0,
            },
        ],
    )

    df = build_leaderboard(root)

    # Exactly one row per C1
    assert set(df["c1_name"]) == {"a", "b"}
    assert len(df) == 2

    # Deterministic ordering: survivor (b) first, then reject (a)
    assert df.iloc[0]["c1_name"] == "b"
    assert df.iloc[0]["status"] == "SURVIVOR"
    assert df.iloc[1]["c1_name"] == "a"
    assert df.iloc[1]["status"] == "REJECT"
    assert df.iloc[1]["reject_reason"] == "worst_fold_negative_roi"

    # Worst-fold selection correct for C1 A (fold 2)
    row_a = df[df["c1_name"] == "a"].iloc[0]
    assert row_a["worst_fold_id"] == 2
    assert row_a["worst_fold_start"] == "2020-08-01"
    assert row_a["worst_fold_end"] == "2020-08-31"
    assert row_a["worst_fold_roi_pct"] == -2.0
    assert row_a["worst_fold_max_dd_pct"] == -10.0
    # scratches=2, trades=15 on worst fold
    assert row_a["worst_fold_scratch_pct"] == pytest.approx((2 / 15) * 100.0)

    # Aggregate trade counts and medians
    assert row_a["folds"] == 2
    assert row_a["total_trades"] == 35
    assert row_a["min_fold_trades"] == 15
    assert row_a["max_fold_trades"] == 20
    assert row_a["median_fold_roi_pct"] == pytest.approx(1.5)


def test_phase4_leaderboard_insufficient_data_reject(tmp_path):
    root = tmp_path / "phase4"

    # Single fold with very few trades → REJECT (insufficient_data)
    _write_wfo_folds(
        root / "wfo_c1_sparse" / "wfo_folds.csv",
        [
            {
                "fold_idx": 1,
                "train_start": "2020-01-01",
                "train_end": "2020-03-31",
                "test_start": "2020-04-01",
                "test_end": "2020-04-30",
                "trades": 5,
                "wins": 3,
                "losses": 2,
                "scratches": 0,
                "win_pct_ns": 60.0,
                "loss_pct_ns": 40.0,
                "expectancy": 0.4,
                "roi_pct": 4.0,
                "max_dd_pct": -2.0,
            }
        ],
    )

    df = build_leaderboard(root)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["c1_name"] == "sparse"
    assert row["status"] == "REJECT"
    assert row["reject_reason"] == "insufficient_data"

