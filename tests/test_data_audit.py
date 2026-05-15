"""Unit tests for the DAUDIT-01 data integrity audit."""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_data_integrity import (  # noqa: E402
    CHK_A2_REPEAT_BAR,
    CHK_LOAD_FAIL,
    CHK_O1_HIGH_VIOLATION,
    CHK_O3_HIGH_LT_LOW,
    CHK_O4_NON_POSITIVE_PRICE,
    CHK_O5_FLAT_BAR,
    CHK_P1_SPREAD_MISSING,
    CHK_P4_SPREAD_HIGH,
    CHK_S1_MISSING_COLS,
    CHK_S3_NAN_OHLC,
    CHK_T1_NON_MONOTONIC,
    CHK_T2_DUPLICATE_TS,
    CHK_T3_GRID_MISALIGN,
    CHK_T5_GAPS,
    CHK_X2_AGGREGATION_DIFF,
    SEV_CRITICAL,
    SEV_WARN,
    aggregate_to_target,
    audit_one_file,
    check_anomaly_returns,
    check_dtypes,
    check_duplicate_timestamps,
    check_flat_bars,
    check_grid_alignment,
    check_no_nan,
    check_ohlc_inequalities,
    check_positive_prices,
    check_repeat_bars,
    check_schema_raw_columns,
    check_spread,
    check_time_monotonic,
    cross_tf_compare,
    detect_gaps,
    discover_files,
    load_config,
    main,
    render_report,
    run_audit,
    write_outputs,
)

DEFAULT_CFG = {
    "schema": {
        "required_columns": [
            "time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume",
        ],
    },
    "anomaly": {
        "spread_pips_warn_threshold": 50.0,
        "return_atr_multiple": 20.0,
        "atr_window": 14,
        "identical_to_prior_bar_warn": True,
    },
    "cross_tf": {
        "random_sample_dates_per_pair": 5,
        "random_seed": 20260509,
        "ohlc_tolerance": 0.0,
        "volume_informational_only": True,
    },
    "output": {
        "results_dir": "results/data_audit_test",
        "report_filename": "DATA_AUDIT_REPORT.md",
        "summary_csv": "data_audit_summary.csv",
        "anomalies_subdir": "anomalies",
        "flat_bar_examples_max": 20,
    },
    "gate": {"critical_check_ids": []},
}


def _good_daily_df(days: int = 30, start: str = "2020-01-06") -> pd.DataFrame:
    """Build a normalized, well-formed daily df (Mon-Fri)."""
    dates = pd.bdate_range(start=start, periods=days, freq="C")
    n = len(dates)
    rng = np.random.default_rng(42)
    close = pd.Series(1.10 + np.cumsum(rng.normal(0, 0.001, n)))
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.001
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.001
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_.astype(float),
            "high": high.astype(float),
            "low": low.astype(float),
            "close": close.astype(float),
            "volume": pd.Series(1000.0, index=range(n)),
        }
    )


def _good_raw_csv_text(df: pd.DataFrame) -> str:
    """Render a normalized daily df as MT5-style raw CSV text."""
    lines = ["time,open,high,low,close,tick_volume,spread,real_volume"]
    for _, r in df.iterrows():
        lines.append(
            f"{r['date'].strftime('%Y-%m-%d')},{r['open']:.5f},{r['high']:.5f},"
            f"{r['low']:.5f},{r['close']:.5f},1000,10,0"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Schema, dtype, NaN checks
# --------------------------------------------------------------------------


def test_check_schema_raw_columns_passes_on_complete_set():
    cols = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    findings = check_schema_raw_columns(cols, cols, tf="daily", pair="EUR_USD")
    assert findings == []


def test_check_schema_raw_columns_flags_missing_real_volume():
    cols = ["time", "open", "high", "low", "close", "tick_volume", "spread"]
    required = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    findings = check_schema_raw_columns(cols, required, tf="daily", pair="EUR_USD")
    assert len(findings) == 1
    assert findings[0].check_id == CHK_S1_MISSING_COLS
    assert findings[0].severity == SEV_CRITICAL


def test_check_dtypes_passes_on_normalized_df():
    df = _good_daily_df()
    findings = check_dtypes(df, tf="daily", pair="EUR_USD")
    assert findings == []


def test_check_no_nan_flags_nan_close():
    df = _good_daily_df()
    df.loc[3, "close"] = np.nan
    findings = check_no_nan(df, tf="daily", pair="EUR_USD")
    assert any(f.check_id == CHK_S3_NAN_OHLC for f in findings)


# --------------------------------------------------------------------------
# Temporal checks
# --------------------------------------------------------------------------


def test_time_monotonic_flags_inversion():
    df = _good_daily_df()
    swap_a = df.loc[5, "date"]
    swap_b = df.loc[6, "date"]
    df.loc[5, "date"] = swap_b
    df.loc[6, "date"] = swap_a
    findings = check_time_monotonic(df, tf="daily", pair="EUR_USD")
    assert any(f.check_id == CHK_T1_NON_MONOTONIC for f in findings)


def test_duplicate_timestamps_flag_and_anomaly_rows():
    df = _good_daily_df()
    df.loc[5, "date"] = df.loc[4, "date"]  # introduce a duplicate
    findings, rows = check_duplicate_timestamps(df, tf="daily", pair="EUR_USD")
    assert any(f.check_id == CHK_T2_DUPLICATE_TS for f in findings)
    assert len(rows) >= 1
    assert rows[0]["pair"] == "EUR_USD"


def test_grid_alignment_daily_flags_intraday_timestamp():
    df = _good_daily_df()
    df.loc[2, "date"] = df.loc[2, "date"] + pd.Timedelta(hours=3)
    findings, _ = check_grid_alignment(df, grid="D", tf="daily", pair="EUR_USD")
    assert any(f.check_id == CHK_T3_GRID_MISALIGN for f in findings)


def test_grid_alignment_h4_passes_for_clean_4h_grid():
    rng = pd.date_range("2020-01-06", periods=24, freq="4h")
    df = pd.DataFrame(
        {
            "date": rng,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    )
    findings, _ = check_grid_alignment(df, grid="H4", tf="4hr", pair="EUR_USD")
    assert findings == []


def test_grid_alignment_h4_flags_off_grid_hour():
    rng = pd.date_range("2020-01-06 02:00:00", periods=12, freq="4h")
    df = pd.DataFrame(
        {
            "date": rng,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    )
    findings, _ = check_grid_alignment(df, grid="H4", tf="4hr", pair="EUR_USD")
    assert any(f.check_id == CHK_T3_GRID_MISALIGN for f in findings)


def test_detect_gaps_classifies_weekend_vs_suspicious():
    # Monday -> Tuesday (no gap), Tuesday -> next Monday (5-day weekend gap), Mon->Tue normal.
    dates = [
        pd.Timestamp("2020-01-06"),  # Mon
        pd.Timestamp("2020-01-07"),  # Tue
        pd.Timestamp("2020-01-13"),  # next Mon — 6-day jump (over weekend + 3 weekdays)
        pd.Timestamp("2020-01-14"),  # Tue
    ]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0,
            "volume": 1.0,
        }
    )
    findings, rows = detect_gaps(df, grid="D", tf="daily", pair="EUR_USD")
    assert findings and findings[0].check_id == CHK_T5_GAPS
    assert findings[0].severity == SEV_WARN
    classifications = [r["classification"] for r in rows]
    assert "suspicious" in classifications  # 6-day jump > 3 grid steps even with weekend


def test_detect_gaps_pure_weekend_is_not_suspicious():
    dates = [
        pd.Timestamp("2020-01-06"),  # Mon
        pd.Timestamp("2020-01-07"),  # Tue
        pd.Timestamp("2020-01-08"),  # Wed
        pd.Timestamp("2020-01-09"),  # Thu
        pd.Timestamp("2020-01-10"),  # Fri
        pd.Timestamp("2020-01-13"),  # Mon — pure weekend gap
    ]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0,
            "volume": 1.0,
        }
    )
    _, rows = detect_gaps(df, grid="D", tf="daily", pair="EUR_USD")
    assert rows
    assert rows[0]["classification"] == "weekend"


# --------------------------------------------------------------------------
# OHLC integrity
# --------------------------------------------------------------------------


def test_ohlc_inequalities_flag_high_below_open():
    df = _good_daily_df()
    df.loc[2, "high"] = df.loc[2, "open"] - 0.01
    findings, rows = check_ohlc_inequalities(df, tf="daily", pair="EUR_USD")
    assert any(f.check_id == CHK_O1_HIGH_VIOLATION for f in findings)
    assert rows[CHK_O1_HIGH_VIOLATION]


def test_ohlc_inequalities_flag_high_below_low():
    df = _good_daily_df()
    df.loc[2, "high"] = 0.5
    df.loc[2, "low"] = 0.6
    findings, _ = check_ohlc_inequalities(df, tf="daily", pair="EUR_USD")
    assert any(f.check_id == CHK_O3_HIGH_LT_LOW for f in findings)


def test_positive_prices_flags_zero_close():
    df = _good_daily_df()
    df.loc[2, "close"] = 0.0
    findings, rows = check_positive_prices(df, tf="daily", pair="EUR_USD")
    assert any(f.check_id == CHK_O4_NON_POSITIVE_PRICE for f in findings)
    assert rows


def test_flat_bar_warning_records_at_most_max_examples():
    df = _good_daily_df(days=10)
    for i in range(0, 5):
        df.loc[i, "open"] = df.loc[i, "high"] = df.loc[i, "low"] = df.loc[i, "close"] = 1.0
    findings, rows = check_flat_bars(df, max_examples=3, tf="daily", pair="EUR_USD")
    assert findings and findings[0].check_id == CHK_O5_FLAT_BAR
    assert findings[0].severity == SEV_WARN
    assert len(rows) == 3


# --------------------------------------------------------------------------
# Spread checks
# --------------------------------------------------------------------------


def test_spread_p4_flags_high_spread_bars():
    df = _good_daily_df(days=10)
    raw = pd.DataFrame(
        {
            "time": df["date"],
            "spread": [10] * 9 + [600],  # 60 pips on the last bar (>50 threshold)
        }
    )
    findings, rows, summary = check_spread(
        raw, threshold_pips=50.0, tf="daily", pair="EUR_USD"
    )
    assert any(f.check_id == CHK_P4_SPREAD_HIGH for f in findings)
    assert any(r["spread_pips"] == 60.0 for r in rows)
    assert summary["spread_max_pips"] == 60.0


def test_spread_missing_column_is_critical():
    raw = pd.DataFrame({"time": [pd.Timestamp("2020-01-01")], "open": [1.0]})
    findings, _, _ = check_spread(raw, threshold_pips=50.0, tf="daily", pair="EUR_USD")
    assert any(
        f.check_id == CHK_P1_SPREAD_MISSING and f.severity == SEV_CRITICAL for f in findings
    )


# --------------------------------------------------------------------------
# Anomaly screens
# --------------------------------------------------------------------------


def test_repeat_bars_flag_dead_feed_pattern():
    df = _good_daily_df(days=20)
    df.loc[5, ["open", "high", "low", "close"]] = df.loc[4, ["open", "high", "low", "close"]].values
    findings, rows = check_repeat_bars(df, tf="daily", pair="EUR_USD")
    assert any(f.check_id == CHK_A2_REPEAT_BAR for f in findings)
    assert rows


def test_anomaly_returns_no_spike_for_calm_data():
    df = _good_daily_df(days=60)
    findings, _ = check_anomaly_returns(
        df, atr_window=14, atr_multiple=20.0, tf="daily", pair="EUR_USD"
    )
    assert findings == []


# --------------------------------------------------------------------------
# Cross-timeframe aggregation
# --------------------------------------------------------------------------


def test_aggregate_h1_to_h4_first_open_last_close_max_high_min_low():
    rng = pd.date_range("2020-01-06 00:00:00", periods=8, freq="1h")
    h1 = pd.DataFrame(
        {
            "date": rng,
            "open": [1.0, 1.05, 1.02, 1.04, 1.10, 1.12, 1.11, 1.15],
            "high": [1.06, 1.10, 1.07, 1.05, 1.13, 1.15, 1.14, 1.16],
            "low": [0.99, 1.01, 1.00, 1.03, 1.09, 1.10, 1.09, 1.13],
            "close": [1.05, 1.02, 1.04, 1.05, 1.12, 1.11, 1.15, 1.14],
            "volume": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        }
    )
    h4 = aggregate_to_target(h1, target_grid="H4")
    assert len(h4) == 2
    assert h4.iloc[0]["open"] == 1.0
    assert h4.iloc[0]["high"] == 1.10
    assert h4.iloc[0]["low"] == 0.99
    assert h4.iloc[0]["close"] == 1.05
    assert h4.iloc[0]["volume"] == 46.0


def test_cross_tf_compare_zero_diff_when_high_is_aggregate_of_low():
    rng = pd.date_range("2020-01-06 00:00:00", periods=80, freq="1h")
    rng_seed = np.random.default_rng(0)
    closes = 1.10 + np.cumsum(rng_seed.normal(0, 0.001, len(rng)))
    opens = np.r_[closes[0], closes[:-1]]
    highs = np.maximum(opens, closes) + 0.001
    lows = np.minimum(opens, closes) - 0.001
    h1 = pd.DataFrame(
        {
            "date": rng,
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": np.ones(len(rng)),
        }
    )
    h4 = aggregate_to_target(h1, target_grid="H4")
    findings, rows = cross_tf_compare(
        h1, h4, "H4",
        sample_size=5, seed=42,
        pair="EUR_USD", low_tf="1hr", high_tf="4hr",
        tolerance=0.0,
    )
    assert findings == []
    assert rows
    assert all(r["open_diff"] == 0.0 for r in rows)


def test_cross_tf_compare_flags_aggregation_diff_when_high_diverges():
    rng = pd.date_range("2020-01-06 00:00:00", periods=80, freq="1h")
    h1 = pd.DataFrame(
        {
            "date": rng,
            "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05,
            "volume": 1.0,
        }
    )
    h4 = aggregate_to_target(h1, target_grid="H4").copy()
    h4["close"] = h4["close"] + 0.1  # tamper
    findings, _ = cross_tf_compare(
        h1, h4, "H4",
        sample_size=5, seed=42,
        pair="EUR_USD", low_tf="1hr", high_tf="4hr",
        tolerance=0.0,
    )
    assert any(f.check_id == CHK_X2_AGGREGATION_DIFF and f.severity == SEV_CRITICAL for f in findings)


# --------------------------------------------------------------------------
# audit_one_file integration
# --------------------------------------------------------------------------


def _write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_audit_one_file_pass_on_clean_csv(tmp_path):
    df = _good_daily_df(days=30)
    csv_path = tmp_path / "EUR_USD.csv"
    _write_csv(csv_path, _good_raw_csv_text(df))
    audit, norm = audit_one_file(csv_path, tf="daily", pair="EUR_USD", grid="D", cfg=DEFAULT_CFG)
    assert audit.status in {"PASS", "WARN"}  # weekend gaps may add WARN, never CRITICAL
    assert not any(f.severity == SEV_CRITICAL for f in audit.findings)
    assert norm is not None


def test_audit_one_file_load_fail_on_unreadable(tmp_path):
    bad = tmp_path / "BAD.csv"
    bad.write_bytes(b"\x00\x00\x00")  # not parseable as CSV → schema fail at minimum
    audit, _ = audit_one_file(bad, tf="daily", pair="BAD", grid="D", cfg=DEFAULT_CFG)
    assert any(
        f.check_id in (CHK_LOAD_FAIL, CHK_S1_MISSING_COLS) and f.severity == SEV_CRITICAL
        for f in audit.findings
    )


def test_audit_one_file_critical_on_ohlc_violation(tmp_path):
    df = _good_daily_df(days=10)
    df.loc[3, "high"] = df.loc[3, "open"] - 0.05  # violate O1
    csv_path = tmp_path / "EUR_USD.csv"
    _write_csv(csv_path, _good_raw_csv_text(df))
    audit, _ = audit_one_file(csv_path, tf="daily", pair="EUR_USD", grid="D", cfg=DEFAULT_CFG)
    assert audit.status == SEV_CRITICAL


def test_audit_one_file_critical_on_duplicate_timestamp(tmp_path):
    df = _good_daily_df(days=10)
    df.loc[5, "date"] = df.loc[4, "date"]
    csv_path = tmp_path / "EUR_USD.csv"
    _write_csv(csv_path, _good_raw_csv_text(df))
    audit, _ = audit_one_file(csv_path, tf="daily", pair="EUR_USD", grid="D", cfg=DEFAULT_CFG)
    # Duplicates also break monotonicity; both CRITICAL ids accepted.
    crit_ids = {f.check_id for f in audit.findings if f.severity == SEV_CRITICAL}
    assert CHK_T2_DUPLICATE_TS in crit_ids or CHK_T1_NON_MONOTONIC in crit_ids


def test_audit_one_file_critical_on_schema_missing_real_volume(tmp_path):
    df = _good_daily_df(days=10)
    raw_text_lines = ["time,open,high,low,close,tick_volume,spread"]  # drop real_volume
    for _, r in df.iterrows():
        raw_text_lines.append(
            f"{r['date'].strftime('%Y-%m-%d')},{r['open']:.5f},{r['high']:.5f},"
            f"{r['low']:.5f},{r['close']:.5f},1000,10"
        )
    csv_path = tmp_path / "EUR_USD.csv"
    _write_csv(csv_path, "\n".join(raw_text_lines) + "\n")
    audit, _ = audit_one_file(csv_path, tf="daily", pair="EUR_USD", grid="D", cfg=DEFAULT_CFG)
    assert any(
        f.check_id == CHK_S1_MISSING_COLS and f.severity == SEV_CRITICAL for f in audit.findings
    )


# --------------------------------------------------------------------------
# Discovery + end-to-end determinism
# --------------------------------------------------------------------------


def _build_mini_data_tree(root: Path) -> dict:
    """Build a tiny but valid data tree: 2 pairs across all 4 timeframes."""
    pairs = ["EUR_USD", "GBP_USD"]
    timeframes = {
        "daily": ("data/daily", "D"),
        "1hr": ("data/1hr", "H1"),
        "4hr": ("data/4hr", "H4"),
        "w1": ("data/w1", "W"),
    }

    # Build a single H1 dataset, then derive 4hr and daily by aggregation,
    # and synthesize a weekly file from daily.
    rng = pd.date_range("2020-01-06 00:00:00", periods=24 * 30, freq="1h")
    rng_seed = np.random.default_rng(0)
    closes = 1.10 + np.cumsum(rng_seed.normal(0, 0.0005, len(rng)))
    opens = np.r_[closes[0], closes[:-1]]
    highs = np.maximum(opens, closes) + 0.0005
    lows = np.minimum(opens, closes) - 0.0005
    h1 = pd.DataFrame(
        {
            "date": rng,
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": np.ones(len(rng)),
        }
    )
    h4 = aggregate_to_target(h1, "H4")
    daily = aggregate_to_target(h4, "D")
    weekly = aggregate_to_target(daily, "W")

    bundles = {"daily": daily, "1hr": h1, "4hr": h4, "w1": weekly}

    cfg = {
        "data_root": str(root / "data"),
        "timeframes": {
            tf: {"dir": str(root / d.split("/")[1]), "grid": g}
            for tf, (d, g) in timeframes.items()
        },
        # The discovery dir should be relative to root → write under <root>/<tf>.
    }
    # Update timeframes dirs to actual on-disk locations under tmp root.
    for tf in timeframes:
        cfg["timeframes"][tf]["dir"] = str(root / "data" / tf)

    for pair in pairs:
        for tf, df in bundles.items():
            tf_dir = root / "data" / tf
            tf_dir.mkdir(parents=True, exist_ok=True)
            csv_path = tf_dir / f"{pair}.csv"
            lines = ["time,open,high,low,close,tick_volume,spread,real_volume"]
            for _, r in df.iterrows():
                lines.append(
                    f"{r['date'].strftime('%Y-%m-%d %H:%M:%S')},"
                    f"{r['open']:.5f},{r['high']:.5f},{r['low']:.5f},{r['close']:.5f},"
                    "1000,10,0"
                )
            csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg["canonical_pairs"] = pairs
    cfg["schema"] = DEFAULT_CFG["schema"]
    cfg["anomaly"] = DEFAULT_CFG["anomaly"]
    cfg["cross_tf"] = DEFAULT_CFG["cross_tf"]
    out_dir = root / "results" / "data_audit_test"
    cfg["output"] = {
        "results_dir": str(out_dir),
        "report_filename": "DATA_AUDIT_REPORT.md",
        "summary_csv": "data_audit_summary.csv",
        "anomalies_subdir": "anomalies",
        "flat_bar_examples_max": 20,
    }
    cfg["gate"] = {"critical_check_ids": []}
    return cfg


def test_discovery_finds_canonical_pairs(tmp_path):
    cfg = _build_mini_data_tree(tmp_path)
    discovery = discover_files(cfg)
    assert set(discovery.keys()) == {"daily", "1hr", "4hr", "w1"}
    for tf in discovery:
        assert discovery[tf]["exists"]
        assert sorted(discovery[tf]["covered"]) == ["EUR_USD", "GBP_USD"]
        assert discovery[tf]["missing"] == []


def test_end_to_end_audit_passes_on_clean_tree(tmp_path):
    cfg = _build_mini_data_tree(tmp_path)
    payload = run_audit(cfg)
    report_path, verdict = write_outputs(payload, cfg)
    assert verdict == "PASS"
    assert report_path.exists()
    summary_csv = Path(cfg["output"]["results_dir"]) / cfg["output"]["summary_csv"]
    assert summary_csv.exists()
    rows = list(csv.DictReader(summary_csv.open(encoding="utf-8")))
    statuses = {r["status"] for r in rows}
    assert "CRITICAL" not in statuses


def test_end_to_end_audit_is_deterministic(tmp_path):
    cfg = _build_mini_data_tree(tmp_path)
    payload1 = run_audit(cfg)
    write_outputs(payload1, cfg)
    out_dir = Path(cfg["output"]["results_dir"])
    snap1 = {p.relative_to(out_dir): p.read_bytes() for p in out_dir.rglob("*") if p.is_file()}

    # Wipe and re-run — only the report header timestamp may differ.
    shutil.rmtree(out_dir)
    payload2 = run_audit(cfg)
    write_outputs(payload2, cfg)
    snap2 = {p.relative_to(out_dir): p.read_bytes() for p in out_dir.rglob("*") if p.is_file()}

    assert set(snap1.keys()) == set(snap2.keys())
    for rel in snap1:
        if rel.name.endswith(".csv"):
            assert snap1[rel] == snap2[rel], f"non-deterministic CSV: {rel}"
        else:
            # Markdown body should match after stripping the first line (timestamp header).
            t1 = snap1[rel].decode("utf-8").split("\n", 1)[1]
            t2 = snap2[rel].decode("utf-8").split("\n", 1)[1]
            assert t1 == t2, f"non-deterministic report body: {rel}"


def test_end_to_end_audit_fails_on_critical(tmp_path):
    cfg = _build_mini_data_tree(tmp_path)
    # Tamper one file: introduce a high<low bar.
    bad_path = Path(cfg["timeframes"]["daily"]["dir"]) / "EUR_USD.csv"
    text = bad_path.read_text(encoding="utf-8").splitlines()
    parts = text[3].split(",")  # 0=header, 1..=data
    # Force high < low: swap.
    parts[2], parts[3] = parts[3], parts[2]
    text[3] = ",".join(parts)
    bad_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    payload = run_audit(cfg)
    _, verdict = write_outputs(payload, cfg)
    assert verdict == "FAIL"


def test_render_report_contains_verdict_marker():
    text = render_report(
        payload={
            "discovery": {},
            "range_overlap_rows": [],
            "pair_missing_rows": [],
            "cross_rows": [],
        },
        audits=[],
        cross_findings=[],
        verdict="PASS",
    )
    assert "**PASS**" in text


def test_load_config_rejects_missing_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("data_root: data\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)


def test_main_returns_zero_on_pass(tmp_path):
    cfg = _build_mini_data_tree(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rc = main(["-c", str(cfg_path)])
    assert rc == 0
