"""Tests for Phase D-6C Value of Capture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _make_synthetic_labels() -> pd.DataFrame:
    """Tiny synthetic labels with Zone C events."""
    dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06", "2020-01-07"]
    rows = []
    for i, d in enumerate(dates):
        for direction in ["long", "short"]:
            zone_c = 1 if (i == 2 and direction == "long") else 0
            split = "discovery" if i < 4 else "validation"
            t6 = 30.0 if i == 2 else 10.0
            mfe = 8.0 if i == 2 else 2.0
            rows.append({
                "pair": "EUR_USD",
                "date": d,
                "direction": direction,
                "dataset_split": split,
                "zone_c_6r_40": zone_c,
                "t6": t6,
                "mfe_40_r": mfe,
            })
    return pd.DataFrame(rows)


def _make_synthetic_attribution() -> pd.DataFrame:
    """Synthetic D6B event attribution. One Zone C event at 2020-01-03 long."""
    return pd.DataFrame([
        {
            "candidate_id": "cand_A",
            "params_json": '{"P": 20}',
            "pair": "EUR_USD",
            "direction": "long",
            "event_start_date": "2020-01-03",
            "dataset_split": "discovery",
            "captured": True,
            "first_trigger_date": "2020-01-03",
            "entry_delay_bars": 0,
            "t6_at_entry": 30.0,
        },
        {
            "candidate_id": "cand_B",
            "params_json": '{"P": 25}',
            "pair": "EUR_USD",
            "direction": "long",
            "event_start_date": "2020-01-03",
            "dataset_split": "discovery",
            "captured": True,
            "first_trigger_date": "2020-01-04",
            "entry_delay_bars": 1,
            "t6_at_entry": 28.0,
        },
        {
            "candidate_id": "cand_C",
            "params_json": '{"P": 30}',
            "pair": "EUR_USD",
            "direction": "long",
            "event_start_date": "2020-01-03",
            "dataset_split": "discovery",
            "captured": False,
            "first_trigger_date": None,
            "entry_delay_bars": None,
            "t6_at_entry": np.nan,
        },
    ])


def test_band_classification() -> None:
    """Band classification: pre, on, late, miss."""
    from analytics.phaseD6C_value_of_capture import _classify_band

    assert _classify_band(-2, pre_bars=3, on_bars=5, late_bars=15) == "pre"
    assert _classify_band(-1, pre_bars=3, on_bars=5, late_bars=15) == "pre"
    assert _classify_band(0, pre_bars=3, on_bars=5, late_bars=15) == "on"
    assert _classify_band(3, pre_bars=3, on_bars=5, late_bars=15) == "on"
    assert _classify_band(5, pre_bars=3, on_bars=5, late_bars=15) == "on"
    assert _classify_band(6, pre_bars=3, on_bars=5, late_bars=15) == "late"
    assert _classify_band(15, pre_bars=3, on_bars=5, late_bars=15) == "late"
    assert _classify_band(16, pre_bars=3, on_bars=5, late_bars=15) == "miss"
    assert _classify_band(None, pre_bars=3, on_bars=5, late_bars=15) == "miss"


def test_value_proxy_computed_correctly() -> None:
    """captured_value_proxy_r = event_mfe_40_r * clip(t6/40, 0, 1)."""
    from analytics.phaseD6C_value_of_capture import _compute_value_proxy

    rf, val = _compute_value_proxy(8.0, 30.0, "on")
    assert rf == pytest.approx(30.0 / 40.0)
    assert val == pytest.approx(8.0 * (30.0 / 40.0))

    rf, val = _compute_value_proxy(8.0, 50.0, "on")
    assert rf == 1.0
    assert val == 8.0

    rf, val = _compute_value_proxy(8.0, 0.0, "miss")
    assert val == 0.0


def test_run_phaseD6C_full_pipeline(tmp_path: Path) -> None:
    """Run full pipeline with synthetic labels and attribution."""
    from analytics.phaseD6C_value_of_capture import run_phaseD6C

    labels = _make_synthetic_labels()
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)

    d6b_dir = tmp_path / "d6b"
    d6b_dir.mkdir()
    attr = _make_synthetic_attribution()
    attr.to_csv(d6b_dir / "ignition_event_attribution.csv", index=False)

    out_dir = tmp_path / "out"
    run_phaseD6C(
        labels_path=labels_path,
        d6b_outdir=d6b_dir,
        out_dir=out_dir,
        pre_bars=3,
        on_bars=5,
        late_bars=15,
    )

    event_df = pd.read_csv(out_dir / "zoneC_event_value_by_candidate.csv")
    summary_df = pd.read_csv(out_dir / "candidate_value_summary.csv")
    memo = (out_dir / "decision_memo_d6C_value.txt").read_text()

    assert len(event_df) == 3
    assert "band" in event_df.columns
    assert "captured_value_proxy_r" in event_df.columns
    cand_a = event_df[event_df["candidate_id"] == "cand_A"]
    assert len(cand_a) == 1
    assert cand_a["band"].iloc[0] == "on"
    assert cand_a["entry_offset_bars"].iloc[0] == 0
    assert cand_a["captured_value_proxy_r"].iloc[0] == pytest.approx(8.0 * (30.0 / 40.0))

    cand_b = event_df[event_df["candidate_id"] == "cand_B"]
    assert cand_b["band"].iloc[0] == "on"
    assert cand_b["entry_offset_bars"].iloc[0] == 1
    cand_c = event_df[event_df["candidate_id"] == "cand_C"]
    assert cand_c["band"].iloc[0] == "miss"
    assert cand_c["captured_value_proxy_r"].iloc[0] == 0.0

    assert len(summary_df) == 3
    assert "zoneC_events_total" in summary_df.columns
    assert "capture_total_pct" in summary_df.columns
    assert "EV_proxy_per_event_r" in summary_df.columns
    assert "EV_proxy_per_event_discovery_r" in summary_df.columns
    assert "EV_proxy_per_event_validation_r" in summary_df.columns
    assert "median_entry_offset_bars_captured" in summary_df.columns

    assert "cand_A" in memo or "cand" in memo
    assert (out_dir / "decision_memo_d6C_value.txt").exists()


def test_summary_aggregates_correct() -> None:
    """Summary aggregates: capture pcts, EV, median offset."""
    from analytics.phaseD6C_value_of_capture import (
        _build_candidate_summary,
        _build_timelines,
        _identify_zone_c_events,
        _merge_events_attribution,
    )

    labels = _make_synthetic_labels()
    attr = _make_synthetic_attribution()

    events = _identify_zone_c_events(labels)
    timelines = _build_timelines(labels)
    event_value_df = _merge_events_attribution(
        events, attr, labels, timelines, pre_bars=3, on_bars=5, late_bars=15
    )
    summary = _build_candidate_summary(event_value_df)

    cand_a = summary[summary["candidate_id"] == "cand_A"].iloc[0]
    assert cand_a["zoneC_events_total"] == 1
    assert cand_a["capture_total_pct"] == 100.0
    assert cand_a["capture_on_pct"] == 100.0
    assert cand_a["EV_proxy_per_event_r"] == pytest.approx(8.0 * (30.0 / 40.0))
    assert cand_a["median_entry_offset_bars_captured"] == 0.0

    cand_c = summary[summary["candidate_id"] == "cand_C"].iloc[0]
    assert cand_c["capture_total_pct"] == 0.0
    assert cand_c["EV_proxy_per_event_r"] == 0.0


def test_deterministic_sorting(tmp_path: Path) -> None:
    """Outputs are deterministically sorted."""
    from analytics.phaseD6C_value_of_capture import run_phaseD6C

    labels = _make_synthetic_labels()
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)

    d6b_dir = tmp_path / "d6b"
    d6b_dir.mkdir()
    _make_synthetic_attribution().to_csv(d6b_dir / "ignition_event_attribution.csv", index=False)

    out_dir = tmp_path / "out"
    run_phaseD6C(labels_path=labels_path, d6b_outdir=d6b_dir, out_dir=out_dir)

    df1 = pd.read_csv(out_dir / "zoneC_event_value_by_candidate.csv")
    run_phaseD6C(labels_path=labels_path, d6b_outdir=d6b_dir, out_dir=out_dir)
    df2 = pd.read_csv(out_dir / "zoneC_event_value_by_candidate.csv")
    pd.testing.assert_frame_equal(df1, df2)


def test_labels_missing_required_raises(tmp_path: Path) -> None:
    """Missing required label columns raises ValueError."""
    from analytics.phaseD6C_value_of_capture import run_phaseD6C

    bad_labels = tmp_path / "bad_labels.csv"
    pd.DataFrame([{"pair": "X", "date": "2020-01-01"}]).to_csv(bad_labels, index=False)
    d6b_fake = tmp_path / "d6b"
    d6b_fake.mkdir()
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="missing required"):
        run_phaseD6C(labels_path=bad_labels, d6b_outdir=d6b_fake, out_dir=tmp_path / "out")


def test_d6b_attribution_not_found_raises(tmp_path: Path) -> None:
    """Missing D6B attribution raises FileNotFoundError."""
    from analytics.phaseD6C_value_of_capture import run_phaseD6C

    labels = _make_synthetic_labels()
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)
    d6b_empty = tmp_path / "d6b_empty"
    d6b_empty.mkdir()

    with pytest.raises(FileNotFoundError, match="ignition_event_attribution"):
        run_phaseD6C(
            labels_path=labels_path,
            d6b_outdir=d6b_empty,
            out_dir=tmp_path / "out",
        )
