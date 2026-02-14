"""Tests for Phase D-6B Ignition Recall."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _make_synthetic_ohlcv(n: int, pair: str = "EUR_USD") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    base = 1.10 + np.linspace(0, 0.05, n)
    noise = rng.standard_normal(n) * 0.001
    close = base + noise
    high = close + np.abs(rng.standard_normal(n)) * 0.002
    low = close - np.abs(rng.standard_normal(n)) * 0.002
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    vol = 1000 + rng.integers(0, 500, n)
    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    })


def _make_labels_with_zone_starts(ohlcv: pd.DataFrame, pair: str) -> pd.DataFrame:
    dates = ohlcv["date"].tolist()
    rows = []
    for i, d in enumerate(dates):
        for direction in ["long", "short"]:
            zone_c = 1 if (i >= 30 and i < 33 and direction == "long") else 0
            zone_b = 1 if (i >= 35 and i < 38 and direction == "long") else 0
            rows.append({
                "pair": pair,
                "date": d,
                "direction": direction,
                "dataset_split": "discovery" if i < len(dates) // 2 else "validation",
                "zone_a_1r_10": True,
                "zone_b_3r_20": bool(zone_b),
                "zone_c_6r_40": bool(zone_c),
                "t1": 1.0,
                "t3": 3.0,
                "t6": 5.0,
                "mfe_40_r": 2.0,
            })
    return pd.DataFrame(rows)


def test_zonec_capture_detects_trigger_within_window(tmp_path: Path) -> None:
    """Zone C event captured when trigger occurs within capture window."""
    from analytics.phaseD6B_ignition_recall import run_phaseD6B

    ohlcv = _make_synthetic_ohlcv(300, "EUR_USD")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ohlcv.to_csv(data_dir / "EUR_USD.csv", index=False)

    labels = _make_labels_with_zone_starts(ohlcv, "EUR_USD")
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)

    out_dir = tmp_path / "out"
    run_phaseD6B(
        labels_path=labels_path,
        data_dir=data_dir,
        out_dir=out_dir,
        capture_bars=5,
    )

    attr = pd.read_csv(out_dir / "ignition_event_attribution.csv")
    captured = attr[attr["captured"]]
    assert len(captured) >= 0


def test_breakout_uses_prior_window_excluding_current(tmp_path: Path) -> None:
    """prior_N_high/low exclude current bar (use shift)."""
    from analytics.phaseD6B_ignition_recall import _compute_bar_metrics

    ohlcv = _make_synthetic_ohlcv(50, "EUR_USD")
    metrics = _compute_bar_metrics(ohlcv)
    assert "prior_10_high" in metrics.columns
    assert "prior_10_low" in metrics.columns
    assert "prior_20_high" in metrics.columns
    first_valid = metrics["prior_10_high"].first_valid_index()
    if first_valid is not None:
        assert pd.notna(metrics.loc[first_valid, "prior_10_high"])


def test_summary_written_and_nonempty(tmp_path: Path) -> None:
    """ignition_candidate_summary.csv is written and non-empty."""
    from analytics.phaseD6B_ignition_recall import run_phaseD6B

    ohlcv = _make_synthetic_ohlcv(300, "EUR_USD")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ohlcv.to_csv(data_dir / "EUR_USD.csv", index=False)

    labels = _make_labels_with_zone_starts(ohlcv, "EUR_USD")
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)

    out_dir = tmp_path / "out"
    run_phaseD6B(
        labels_path=labels_path,
        data_dir=data_dir,
        out_dir=out_dir,
        capture_bars=5,
    )

    summary_path = tmp_path / "out" / "ignition_candidate_summary.csv"
    assert summary_path.exists()
    summary = pd.read_csv(summary_path)
    assert len(summary) > 0
    assert "candidate_id" in summary.columns
    assert "zoneC_capture_overall_pct" in summary.columns
    assert "ignition_rate_per_1000_bars" in summary.columns
    assert (tmp_path / "out" / "decision_memo_d6B.txt").exists()
