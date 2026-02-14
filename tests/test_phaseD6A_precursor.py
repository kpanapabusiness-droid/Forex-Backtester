"""Tests for Phase D-6A Ignition Precursor Analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _make_synthetic_ohlcv(n: int, pair: str = "EUR_USD") -> pd.DataFrame:
    """Create synthetic OHLCV with date, open, high, low, close, volume."""
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


def _make_synthetic_labels(ohlcv: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Create labels with a few Zone C starts (zone_c_6r_40=1, prev=0)."""
    dates = ohlcv["date"].tolist()
    rows = []
    for i, d in enumerate(dates):
        for direction in ["long", "short"]:
            zone_c = 1 if (i >= 25 and i < 28 and direction == "long") else 0
            rows.append({
                "pair": pair,
                "date": d,
                "direction": direction,
                "dataset_split": "discovery",
                "zone_a_1r_10": True,
                "zone_b_3r_20": i % 3 == 0,
                "zone_c_6r_40": bool(zone_c),
                "t1": 1.0,
                "t3": 3.0,
                "t6": 5.0,
                "mfe_40_r": 2.0,
            })
    return pd.DataFrame(rows)


def test_zone_c_starts_identified(tmp_path: Path) -> None:
    """Zone C start bars are correctly identified from labels."""
    from analytics.phaseD6A_ignition_precursor_analysis import (
        _identify_zone_c_starts,
        _load_labels,
    )

    ohlcv = _make_synthetic_ohlcv(100, "EUR_USD")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ohlcv.to_csv(data_dir / "EUR_USD.csv", index=False)

    labels = _make_synthetic_labels(ohlcv, "EUR_USD")
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)

    loaded = _load_labels(labels_path)
    starts = _identify_zone_c_starts(loaded)
    assert len(starts) >= 1
    assert "pair" in starts.columns
    assert "date" in starts.columns
    assert "direction" in starts.columns


def test_precursor_metrics_computed(tmp_path: Path) -> None:
    """Precursor metrics are computed and written to output CSVs."""
    from analytics.phaseD6A_ignition_precursor_analysis import run_phaseD6A

    ohlcv = _make_synthetic_ohlcv(120, "EUR_USD")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ohlcv.to_csv(data_dir / "EUR_USD.csv", index=False)

    labels = _make_synthetic_labels(ohlcv, "EUR_USD")
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)

    out_dir = tmp_path / "out"
    run_phaseD6A(
        labels_path=labels_path,
        data_dir=data_dir,
        out_dir=out_dir,
        min_bars=20,
    )

    zone_path = out_dir / "zoneC_precursor_metrics.csv"
    ctrl_path = out_dir / "control_precursor_metrics.csv"
    summary_path = out_dir / "precursor_comparison_summary.csv"

    assert zone_path.exists()
    assert ctrl_path.exists()
    assert summary_path.exists()

    zone_df = pd.read_csv(zone_path)
    ctrl_df = pd.read_csv(ctrl_path)
    summary_df = pd.read_csv(summary_path)

    assert "atr_14" in zone_df.columns or len(zone_df) == 0
    assert "atr_14" in ctrl_df.columns or len(ctrl_df) == 0
    assert "metric" in summary_df.columns or len(summary_df) == 0


def test_raises_if_no_ohlcv_found(tmp_path: Path) -> None:
    """Raises ValueError when data-dir has no OHLCV files (empty dir)."""
    import pytest

    from analytics.phaseD6A_ignition_precursor_analysis import run_phaseD6A

    labels = _make_synthetic_labels(_make_synthetic_ohlcv(120, "EUR_USD"), "EUR_USD")
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)

    data_dir = tmp_path / "empty_data"
    data_dir.mkdir()

    with pytest.raises(ValueError) as exc:
        run_phaseD6A(labels_path=labels_path, data_dir=data_dir, out_dir=tmp_path / "out")
    msg = str(exc.value)
    assert "zero pairs loaded" in msg or "data-dir" in msg.lower()
    assert "resolved" in msg or str(data_dir.resolve()) in msg


def test_raises_if_zonec_starts_exist_but_no_pairs_loaded(tmp_path: Path) -> None:
    """Raises when Zone C starts exist but data-dir has no matching pair CSVs."""
    import pytest

    from analytics.phaseD6A_ignition_precursor_analysis import run_phaseD6A

    ohlcv = _make_synthetic_ohlcv(120, "EUR_USD")
    labels = _make_synthetic_labels(ohlcv, "EUR_USD")
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)

    data_dir = tmp_path / "wrong_data"
    data_dir.mkdir()
    ohlcv.to_csv(data_dir / "USD_JPY.csv", index=False)

    with pytest.raises(ValueError) as exc:
        run_phaseD6A(labels_path=labels_path, data_dir=data_dir, out_dir=tmp_path / "out")
    msg = str(exc.value)
    assert "Zone C starts" in msg or "zero pairs" in msg.lower()
    assert "EUR_USD" in msg or "example expected" in msg.lower()


def test_cli_runs(tmp_path: Path) -> None:
    """CLI entry point runs without error."""
    import subprocess

    ohlcv = _make_synthetic_ohlcv(120, "EUR_USD")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ohlcv.to_csv(data_dir / "EUR_USD.csv", index=False)

    labels = _make_synthetic_labels(ohlcv, "EUR_USD")
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)

    result = subprocess.run(
        [
            "python",
            "-m",
            "analytics.phaseD6A_ignition_precursor_analysis",
            "--labels",
            str(labels_path),
            "--data-dir",
            str(data_dir),
            "--outdir",
            str(tmp_path / "out"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (tmp_path / "out" / "zoneC_precursor_metrics.csv").exists()
    assert (tmp_path / "out" / "precursor_comparison_summary.csv").exists()
