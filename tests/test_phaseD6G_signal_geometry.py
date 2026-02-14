"""
Phase D-6G: Tests for signal geometry analysis.
"""
from __future__ import annotations

import pandas as pd
import pytest

from analytics.phaseD6G_signal_geometry import (
    _derive_metrics_from_stability,
    compute_clustering_ratio_simple,
    compute_leaderboard_geometry_lock,
    compute_leaderboard_signal_lift,
    compute_pooled_signal_lift,
    compute_pooled_signal_lift_stability,
)
from scripts.phaseD6G_run_signal_geometry import (
    _name_matches,
    _validate_signal_output,
    discover_signal_functions,
    normalize_csv_signal,
    resolve_signal_list,
    run_discovery_report,
)


def _make_toy_merged() -> pd.DataFrame:
    """Toy merged (clean + signal) with known geometry."""
    return pd.DataFrame({
        "pair": ["P1"] * 10 + ["P2"] * 6,
        "date": pd.to_datetime([
            "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05",
            "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10",
            "2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05", "2021-01-06",
        ]),
        "valid_atr": [True] * 16,
        "valid_ref": [True] * 16,
        "valid_h40": [True] * 16,
        "clean_mfe_long_x1_h40": [0.5, 2.0, 1.0, 3.0, 0.2, 4.0, 1.5, 0.0, 2.5, 1.0, 2.0, 0.5, 3.0, 1.0, 0.0, 2.0],
        "clean_mfe_short_x1_h40": [1.0, 0.5, 2.0, 0.0, 3.0, 1.0, 0.5, 2.5, 0.0, 1.5, 0.5, 2.0, 0.0, 1.0, 3.0, 0.5],
        "sig_coral": [1, -1, 0, 1, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, -1, 0],
    })


def test_conditional_rates_computed_correctly() -> None:
    """Verify conditional rates and lifts."""
    df = _make_toy_merged()
    out = compute_pooled_signal_lift(df, "sig_coral", (1,), (1.0, 2.0))
    assert not out.empty
    r = out[(out["direction"] == "long") & (out["x"] == 1) & (out["y"] == 2.0)].iloc[0]
    n_total = r["n_total"]
    n_signal = r["n_signal"]
    assert n_total > 0
    assert n_signal >= 0
    assert 0 <= r["baseline_rate"] <= 1
    assert 0 <= r["signal_rate"] <= 1
    assert r["lift"] == pytest.approx(r["signal_rate"] - r["baseline_rate"], rel=1e-5)


def test_per_pair_equals_pooled_when_one_pair() -> None:
    """When only one pair, per-signal logic matches pooled."""
    df = _make_toy_merged()
    single = df[df["pair"] == "P1"].copy()
    out = compute_pooled_signal_lift(single, "sig_coral", (1,), (1.0,))
    assert len(out) >= 1
    assert "lift" in out.columns
    assert "signal_rate" in out.columns


def test_stability_split_works() -> None:
    """Stability split yields discovery/validation rates."""
    df = _make_toy_merged()
    out = compute_pooled_signal_lift_stability(
        df, "sig_coral", (1,), (1.0,), discovery_end="2021-01-01"
    )
    assert not out.empty
    assert "n_total_discovery" in out.columns
    assert "n_total_validation" in out.columns
    assert "lift_discovery" in out.columns
    assert "lift_validation" in out.columns
    assert "delta_lift" in out.columns


def test_determinism_same_output_twice() -> None:
    """Output tables have stable ordering."""
    df = _make_toy_merged()
    a1 = compute_pooled_signal_lift(df, "sig_coral", (1, 2), (1.0, 2.0))
    a2 = compute_pooled_signal_lift(df, "sig_coral", (1, 2), (1.0, 2.0))
    pd.testing.assert_frame_equal(a1, a2)
    b1 = compute_pooled_signal_lift_stability(df, "sig_coral", (1,), (1.0,))
    b2 = compute_pooled_signal_lift_stability(df, "sig_coral", (1,), (1.0,))
    pd.testing.assert_frame_equal(b1, b2)


def test_leaderboard_ranking() -> None:
    """Leaderboard ranks by lift."""
    lift = pd.DataFrame([
        {"signal_name": "A", "direction": "long", "x": 1, "y": 2.0, "n_signal": 100, "lift": 0.05},
        {"signal_name": "A", "direction": "short", "x": 1, "y": 2.0, "n_signal": 80, "lift": 0.03},
        {"signal_name": "B", "direction": "long", "x": 1, "y": 2.0, "n_signal": 50, "lift": 0.02},
        {"signal_name": "B", "direction": "short", "x": 1, "y": 2.0, "n_signal": 60, "lift": 0.01},
    ])
    stab = pd.DataFrame([
        {"signal_name": "A", "direction": "long", "x": 1, "y": 2.0, "lift_validation": 0.04},
        {"signal_name": "A", "direction": "short", "x": 1, "y": 2.0, "lift_validation": 0.02},
        {"signal_name": "B", "direction": "long", "x": 1, "y": 2.0, "lift_validation": 0.03},
        {"signal_name": "B", "direction": "short", "x": 1, "y": 2.0, "lift_validation": 0.02},
    ])
    lb = compute_leaderboard_signal_lift(lift, stab, rank_by=(1, 2.0), min_n_signal=30)
    assert not lb.empty
    assert lb["rank"].iloc[0] == 1
    assert "signal_name" in lb.columns
    assert "lift_combined" in lb.columns
    assert "notes" in lb.columns


def test_discover_signal_functions() -> None:
    """Auto-discovery finds c1_* and regime functions."""
    found = discover_signal_functions(["indicators.confirmation_funcs"])
    assert len(found) >= 1
    assert all(":" in k for k in found)
    assert any("c1_coral" in k for k in found)
    c1_only = [k for k in found if k.endswith(":c1_coral")]
    assert len(c1_only) == 1
    assert c1_only[0] == "indicators.confirmation_funcs:c1_coral"


def test_resolve_signal_list_all() -> None:
    """signals=ALL returns discovered list."""
    discovered = ["indicators.confirmation_funcs:c1_coral", "indicators.confirmation_funcs:c1_supertrend"]
    out = resolve_signal_list("ALL", ["indicators.confirmation_funcs"], discovered)
    assert out == discovered


def test_resolve_signal_list_explicit() -> None:
    """Explicit list resolves bare names and module:name."""
    discovered = ["indicators.confirmation_funcs:c1_coral", "mod_b:other_func"]
    out = resolve_signal_list(["c1_coral", "mod_b:other_func"], [], discovered)
    assert "indicators.confirmation_funcs:c1_coral" in out
    assert "mod_b:other_func" in out


def test_validate_signal_output_valid() -> None:
    """Valid DataFrame with {-1,0,1} returns None."""
    df = pd.DataFrame({"c1_signal": [1, -1, 0, 1, 0]})
    assert _validate_signal_output(df, "x:y") is None


def test_validate_signal_output_invalid() -> None:
    """Invalid outputs produce deterministic reasons."""
    assert _validate_signal_output("not a df", "x:y") == "not_dataframe"
    assert _validate_signal_output(pd.DataFrame(), "x:y") == "empty_dataframe"
    df = pd.DataFrame({"other": [1, 2]})
    assert _validate_signal_output(df, "x:y") is not None
    df2 = pd.DataFrame({"c1_signal": [2, 3]})
    assert _validate_signal_output(df2, "x:y") == "signal_not_in_range_neg1_0_1"


def test_name_matches() -> None:
    """Discovery name filter is correct."""
    assert _name_matches("c1_coral") is True
    assert _name_matches("c1_regime_sm__binary") is True
    assert _name_matches("my_regime_func") is True
    assert _name_matches("_private") is False
    assert _name_matches("helper") is False


def test_binary_indicators_short_series_no_crash() -> None:
    """c1_regime_sm__binary, c1_vol_dir__binary, c1_persist_momo__binary work on 10 rows."""
    from core.utils import calculate_atr
    from indicators.confirmation_funcs import (
        c1_persist_momo__binary,
        c1_regime_sm__binary,
        c1_vol_dir__binary,
    )

    n = 10
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "open": 1.0, "high": 1.01, "low": 0.99, "close": 1.0,
    })
    df = calculate_atr(df, period=14)
    for func in (c1_regime_sm__binary, c1_vol_dir__binary, c1_persist_momo__binary):
        out = func(df.copy(), signal_col="c1_signal")
        assert "c1_signal" in out.columns
        s = pd.to_numeric(out["c1_signal"], errors="coerce").dropna()
        assert s.empty or set(s.astype(int).unique()).issubset({-1, 0, 1})


def test_skipped_signals_deterministic(tmp_path) -> None:
    """Invalid signal produces deterministic skipped_signals.csv."""
    from pathlib import Path

    import yaml

    from scripts.phaseD6G_run_signal_geometry import _load_yaml, _run

    cfg = {
        "signal_modules": ["indicators.confirmation_funcs"],
        "signals": ["math:sqrt"],
        "split_date": "2023-01-01",
        "mae_x": [1],
        "thresholds_y": [1],
        "date_range": {"start": "2019-01-01", "end": "2026-01-01"},
        "data": {"dir": "data/daily"},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    clean_df = pd.DataFrame({
        "pair": ["EUR_USD"],
        "date": pd.to_datetime(["2020-01-01"]),
        "valid_atr": [True],
        "valid_ref": [True],
        "valid_h40": [True],
        "clean_mfe_long_x1_h40": [1.0],
        "clean_mfe_short_x1_h40": [0.5],
    })
    clean_path = tmp_path / "clean.csv"
    clean_df.to_csv(clean_path, index=False)
    _run(_load_yaml(cfg_path), Path(clean_path), tmp_path / "out")
    skip_path = tmp_path / "out" / "skipped_signals.csv"
    assert skip_path.exists()
    skipped = pd.read_csv(skip_path)
    assert list(skipped.columns) == ["signal_key", "reason"]
    assert "math:sqrt" in skipped["signal_key"].values


def test_discovery_report_structure(tmp_path) -> None:
    """Discovery report has expected keys and deterministic structure."""
    report_path = tmp_path / "report.json"
    report = run_discovery_report(
        tmp_path,
        report_path,
        py_files_override=[],
        csv_paths_override=[],
    )
    assert "discovered_modules" in report
    assert "candidate_py_files" in report
    assert "discovered_signal_csv_files" in report
    assert "notes" in report
    assert isinstance(report["discovered_modules"], list)
    assert isinstance(report["candidate_py_files"], list)
    assert isinstance(report["discovered_signal_csv_files"], list)
    assert report_path.exists()
    import json
    loaded = json.loads(report_path.read_text())
    assert set(loaded.keys()) == set(report.keys())


def test_normalize_csv_signal_single_column() -> None:
    """Single signal column in {-1,0,+1} normalizes correctly."""
    df = pd.DataFrame({
        "pair": ["EUR_USD", "EUR_USD"],
        "date": ["2020-01-01", "2020-01-02"],
        "signal": [1, -1],
    })
    out = normalize_csv_signal(df)
    assert list(out.columns) == ["pair", "date", "signal"]
    assert out["signal"].tolist() == [1, -1]


def test_normalize_csv_signal_long_short() -> None:
    """long_signal/short_signal boolean columns normalize to {-1,0,+1}."""
    df = pd.DataFrame({
        "pair": ["EUR_USD", "EUR_USD", "EUR_USD"],
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "long_signal": [1, 0, 0],
        "short_signal": [0, 1, 0],
    })
    out = normalize_csv_signal(df)
    assert out["signal"].tolist() == [1, -1, 0]


def test_normalize_csv_signal_direction() -> None:
    """direction column (long/short) normalizes to 1/-1."""
    df = pd.DataFrame({
        "pair": ["EUR_USD", "EUR_USD", "EUR_USD"],
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "direction": ["long", "short", "0"],
    })
    out = normalize_csv_signal(df)
    assert out["signal"].tolist() == [1, -1, 0]


def test_runner_combined_module_csv(tmp_path) -> None:
    """Runner handles module signals + CSV signals without crashing."""
    from pathlib import Path

    import yaml

    from scripts.phaseD6G_run_signal_geometry import _load_yaml, _run

    csv_dir = tmp_path / "results" / "phaseD"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "proto_signals.csv"
    csv_df = pd.DataFrame({
        "pair": ["P1", "P1", "P2"],
        "date": ["2020-01-01", "2020-01-02", "2020-01-01"],
        "signal": [1, -1, 0],
    })
    csv_df.to_csv(csv_path, index=False)

    clean_df = pd.DataFrame({
        "pair": ["P1", "P1", "P2"],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
        "valid_atr": [True, True, True],
        "valid_ref": [True, True, True],
        "valid_h40": [True, True, True],
        "clean_mfe_long_x1_h40": [1.5, 2.0, 0.5],
        "clean_mfe_short_x1_h40": [0.5, 1.0, 2.0],
    })
    clean_path = tmp_path / "clean.csv"
    clean_df.to_csv(clean_path, index=False)

    cfg = {
        "project_root": str(tmp_path),
        "signal_sources": {
            "modules": ["indicators.confirmation_funcs"],
            "csv_dirs": ["results/phaseD"],
        },
        "signals": ["c1_coral"],
        "split_date": "2023-01-01",
        "mae_x": [1],
        "thresholds_y": [1.0],
        "date_range": {"start": "2019-01-01", "end": "2026-01-01"},
        "data": {"dir": "data/daily"},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    data_dir = tmp_path / "data" / "daily"
    data_dir.mkdir(parents=True)
    ohlc = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "open": 1.0, "high": 1.01, "low": 0.99, "close": 1.0, "volume": 100,
    })
    ohlc.to_csv(data_dir / "P1.csv", index=False)
    ohlc.to_csv(data_dir / "P2.csv", index=False)
    _run(_load_yaml(cfg_path), Path(clean_path), tmp_path / "out")
    assert (tmp_path / "out" / "discovery_report.json").exists()


def _make_stability_row(signal_name: str, direction: str, x: int, y: float, **kwargs) -> dict:
    defaults = {
        "n_total_discovery": 100,
        "n_signal_discovery": 50,
        "baseline_rate_discovery": 0.3,
        "signal_rate_discovery": 0.4,
        "n_total_validation": 100,
        "n_signal_validation": 50,
        "baseline_rate_validation": 0.3,
        "signal_rate_validation": 0.35,
    }
    defaults.update(kwargs)
    return {"signal_name": signal_name, "direction": direction, "x": x, "y": y, **defaults}


def test_derive_metrics_long_only_uses_long_rate() -> None:
    """P_3R_before_2R_disc equals long signal_rate when short n_signal==0."""
    stab = pd.DataFrame([
        _make_stability_row("sig_long", "long", 2, 3.0, n_signal_discovery=50, signal_rate_discovery=0.45),
        _make_stability_row("sig_long", "short", 2, 3.0, n_signal_discovery=0, signal_rate_discovery=0),
        _make_stability_row("sig_long", "long", 2, 4.0, n_signal_discovery=50, signal_rate_discovery=0.25),
        _make_stability_row("sig_long", "short", 2, 4.0, n_signal_discovery=0, signal_rate_discovery=0),
    ])
    m3 = _derive_metrics_from_stability(stab, "sig_long", 2, 3.0)
    assert m3["signal_rate_disc"] == 0.45
    assert not m3["insufficient_sample"]
    m4 = _derive_metrics_from_stability(stab, "sig_long", 2, 4.0)
    assert m4["signal_rate_disc"] == 0.25


def test_derive_metrics_p4r_long_only() -> None:
    """P_4R_before_2R from x=2,y=4 when short n_signal==0."""
    stab = pd.DataFrame([
        _make_stability_row("sig_a", "long", 2, 4.0, n_signal_discovery=60, signal_rate_discovery=0.5),
        _make_stability_row("sig_a", "short", 2, 4.0, n_signal_discovery=0, signal_rate_discovery=0),
    ])
    m = _derive_metrics_from_stability(stab, "sig_a", 2, 4.0)
    assert m["signal_rate_disc"] == 0.5


def test_p2r_before_1r_differs_across_signals() -> None:
    """P_2R_before_1R must differ when signal_rate differs (prevents constant baseline bug)."""
    rows = []
    for sig, rate_d in [("sig_A", 0.6), ("sig_B", 0.2)]:
        for x, y in [(1, 2.0), (2, 3.0), (2, 4.0)]:
            sr_d = rate_d if (x == 1 and y == 2.0) else 0.4
            rows.append(_make_stability_row(sig, "long", x, y, n_signal_discovery=40, signal_rate_discovery=sr_d, signal_rate_validation=sr_d * 0.9))
            rows.append(_make_stability_row(sig, "short", x, y, n_signal_discovery=0, signal_rate_discovery=0, signal_rate_validation=0))
    stab = pd.DataFrame(rows)
    n = 2000
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    lb = compute_leaderboard_geometry_lock(
        {
            "sig_A": pd.DataFrame({"pair": ["P1"] * (n // 2) + ["P2"] * (n // 2), "date": dates, "sig_A": [1] * 400 + [0] * (n - 400), "valid_atr": [True] * n, "valid_ref": [True] * n, "valid_h40": [True] * n}),
            "sig_B": pd.DataFrame({"pair": ["P1"] * (n // 2) + ["P2"] * (n // 2), "date": dates, "sig_B": [1] * 400 + [0] * (n - 400), "valid_atr": [True] * n, "valid_ref": [True] * n, "valid_h40": [True] * n}),
        },
        stability_df=stab,
        discovery_end="2022-12-31",
        date_start="2019-01-01",
        date_end="2026-01-01",
    )
    row_a = lb[lb["signal_name"] == "sig_A"].iloc[0]
    row_b = lb[lb["signal_name"] == "sig_B"].iloc[0]
    assert row_a["P_2R_before_1R_disc"] != row_b["P_2R_before_1R_disc"]
    assert row_a["P_2R_before_1R_disc"] == 0.6
    assert row_b["P_2R_before_1R_disc"] == 0.2


def test_leaderboard_geometry_lock_sorts_by_p_3r_before_2r() -> None:
    """Leaderboard sorts descending by P_3R_before_2R."""
    base = pd.DataFrame({
        "pair": ["P1"] * 100,
        "date": pd.date_range("2020-01-01", periods=100, freq="D"),
        "valid_atr": [True] * 100,
        "valid_ref": [True] * 100,
        "valid_h40": [True] * 100,
    })
    high_sig = base.copy()
    high_sig["sig_high"] = [1] * 40 + [0] * 60
    low_sig = base.copy()
    low_sig["sig_low"] = [1] * 10 + [0] * 90
    merged_by_signal = {"sig_high": high_sig, "sig_low": low_sig}
    stab_rows = [
        _make_stability_row("sig_high", "long", 2, 3.0, n_signal_discovery=35, signal_rate_discovery=0.7, n_signal_validation=30, signal_rate_validation=0.65),
        _make_stability_row("sig_high", "short", 2, 3.0, n_signal_discovery=5, signal_rate_discovery=0.4, n_signal_validation=5, signal_rate_validation=0.35),
        _make_stability_row("sig_high", "long", 2, 4.0, n_signal_discovery=35, signal_rate_discovery=0.5, n_signal_validation=30, signal_rate_validation=0.48),
        _make_stability_row("sig_high", "short", 2, 4.0, n_signal_discovery=5, signal_rate_discovery=0.3, n_signal_validation=5, signal_rate_validation=0.28),
        _make_stability_row("sig_high", "long", 1, 2.0, n_signal_discovery=35, signal_rate_discovery=0.6, n_signal_validation=30, signal_rate_validation=0.55),
        _make_stability_row("sig_high", "short", 1, 2.0, n_signal_discovery=5, signal_rate_discovery=0.35, n_signal_validation=5, signal_rate_validation=0.3),
        _make_stability_row("sig_low", "long", 2, 3.0, n_signal_discovery=8, signal_rate_discovery=0.25, n_signal_validation=7, signal_rate_validation=0.2),
        _make_stability_row("sig_low", "short", 2, 3.0, n_signal_discovery=2, signal_rate_discovery=0.1, n_signal_validation=2, signal_rate_validation=0.08),
        _make_stability_row("sig_low", "long", 2, 4.0, n_signal_discovery=8, signal_rate_discovery=0.15, n_signal_validation=7, signal_rate_validation=0.12),
        _make_stability_row("sig_low", "short", 2, 4.0, n_signal_discovery=2, signal_rate_discovery=0.05, n_signal_validation=2, signal_rate_validation=0.04),
        _make_stability_row("sig_low", "long", 1, 2.0, n_signal_discovery=8, signal_rate_discovery=0.3, n_signal_validation=7, signal_rate_validation=0.28),
        _make_stability_row("sig_low", "short", 1, 2.0, n_signal_discovery=2, signal_rate_discovery=0.15, n_signal_validation=2, signal_rate_validation=0.12),
    ]
    stab = pd.DataFrame(stab_rows)
    lb = compute_leaderboard_geometry_lock(
        merged_by_signal,
        stability_df=stab,
        discovery_end="2020-03-15",
        date_start="2020-01-01",
        date_end="2020-06-01",
    )
    assert not lb.empty
    assert "P_3R_before_2R_disc" in lb.columns
    assert lb["rank"].iloc[0] == 1
    sorted_vals = lb["P_3R_before_2R_disc"].tolist()
    assert sorted_vals == sorted(sorted_vals, reverse=True)
    row_high = lb[lb["signal_name"] == "sig_high"].iloc[0]
    expected_p3r = (35 * 0.7 + 5 * 0.4) / 40
    assert row_high["P_3R_before_2R_disc"] == pytest.approx(expected_p3r, rel=1e-5)


def test_frequency_floor_rejection() -> None:
    """Reject when annual_signals_per_pair < 24."""
    df = pd.DataFrame({
        "pair": ["P1"] * 20,
        "date": pd.date_range("2020-01-01", periods=20, freq="D"),
        "valid_atr": [True] * 20,
        "valid_ref": [True] * 20,
        "valid_h40": [True] * 20,
        "sig_sparse": [1] * 2 + [0] * 18,
    })
    merged_by_signal = {"sig_sparse": df}
    stab = pd.DataFrame([
        _make_stability_row("sig_sparse", "long", 2, 3.0, n_signal_discovery=2, signal_rate_discovery=0.5, n_signal_validation=1, signal_rate_validation=0.4),
        _make_stability_row("sig_sparse", "short", 2, 3.0, n_signal_discovery=0, signal_rate_discovery=0, n_signal_validation=0, signal_rate_validation=0),
        _make_stability_row("sig_sparse", "long", 2, 4.0, n_signal_discovery=2, signal_rate_discovery=0.3, n_signal_validation=1, signal_rate_validation=0.25),
        _make_stability_row("sig_sparse", "short", 2, 4.0, n_signal_discovery=0, signal_rate_discovery=0, n_signal_validation=0, signal_rate_validation=0),
        _make_stability_row("sig_sparse", "long", 1, 2.0, n_signal_discovery=2, signal_rate_discovery=0.5, n_signal_validation=1, signal_rate_validation=0.45),
        _make_stability_row("sig_sparse", "short", 1, 2.0, n_signal_discovery=0, signal_rate_discovery=0, n_signal_validation=0, signal_rate_validation=0),
    ])
    lb = compute_leaderboard_geometry_lock(
        merged_by_signal,
        stability_df=stab,
        discovery_end="2020-01-15",
        date_start="2020-01-01",
        date_end="2025-12-31",
    )
    assert not lb.empty
    row = lb[lb["signal_name"] == "sig_sparse"].iloc[0]
    assert row["PASS"] is False or "annual_signals" in str(row.get("reject_reason", ""))


def test_clustering_rejection() -> None:
    """Reject when clustering_ratio > 0.20."""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    sig_vals = [0] * 50
    sig_vals[0] = 1
    sig_vals[1] = 1
    sig_vals[2] = 1
    sig_vals[3] = 1
    df = pd.DataFrame({
        "pair": ["P1"] * 50,
        "date": dates,
        "valid_atr": [True] * 50,
        "valid_ref": [True] * 50,
        "valid_h40": [True] * 50,
        "clean_mfe_3r_before_mae_2r_long_h40": [True] * 25 + [False] * 25,
        "clean_mfe_3r_before_mae_2r_short_h40": [False] * 25 + [True] * 25,
        "sig_clustered": sig_vals,
    })
    clust = compute_clustering_ratio_simple(df, "sig_clustered", 3)
    assert clust > 0.20


def test_pass_when_metrics_above_threshold() -> None:
    """Geometry lock runs; PASS when frequency/clustering/lift all pass (mock)."""
    n = 800
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    sig_vals = [0] * n
    for i in range(0, n, 8):
        sig_vals[i] = 1 if i % 16 == 0 else -1
    df = pd.DataFrame({
        "pair": ["P1"] * (n // 2) + ["P2"] * (n // 2),
        "date": dates,
        "valid_atr": [True] * n,
        "valid_ref": [True] * n,
        "valid_h40": [True] * n,
        "sig_mock": sig_vals,
    })
    merged_by_signal = {"sig_mock": df}
    stab = pd.DataFrame([
        _make_stability_row("sig_mock", "long", 2, 3.0, n_signal_discovery=60, signal_rate_discovery=0.45, baseline_rate_discovery=0.30, n_signal_validation=50, signal_rate_validation=0.42, baseline_rate_validation=0.28),
        _make_stability_row("sig_mock", "short", 2, 3.0, n_signal_discovery=50, signal_rate_discovery=0.40, baseline_rate_discovery=0.28, n_signal_validation=45, signal_rate_validation=0.38, baseline_rate_validation=0.26),
        _make_stability_row("sig_mock", "long", 2, 4.0, n_signal_discovery=60, signal_rate_discovery=0.35, n_signal_validation=50, signal_rate_validation=0.33),
        _make_stability_row("sig_mock", "short", 2, 4.0, n_signal_discovery=50, signal_rate_discovery=0.30, n_signal_validation=45, signal_rate_validation=0.28),
        _make_stability_row("sig_mock", "long", 1, 2.0, n_signal_discovery=60, signal_rate_discovery=0.55, n_signal_validation=50, signal_rate_validation=0.52),
        _make_stability_row("sig_mock", "short", 1, 2.0, n_signal_discovery=50, signal_rate_discovery=0.50, n_signal_validation=45, signal_rate_validation=0.48),
    ])
    lb = compute_leaderboard_geometry_lock(
        merged_by_signal,
        stability_df=stab,
        discovery_end="2021-12-31",
        date_start="2020-01-01",
        date_end="2025-12-31",
    )
    assert not lb.empty
    assert "PASS" in lb.columns
    assert "reject_reason" in lb.columns


def test_p4r_secondary_protection_rejection() -> None:
    """REJECT when P_4R_before_2R drops >10% from discovery to validation."""
    n = 800
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    sig_vals = [0] * n
    for i in range(0, n, 3):
        sig_vals[i] = 1 if i % 6 == 0 else -1
    df = pd.DataFrame({
        "pair": ["P1"] * (n // 2) + ["P2"] * (n // 2),
        "date": dates,
        "valid_atr": [True] * n,
        "valid_ref": [True] * n,
        "valid_h40": [True] * n,
        "sig_test": sig_vals,
    })
    merged_by_signal = {"sig_test": df}
    stab = pd.DataFrame([
        _make_stability_row("sig_test", "long", 2, 3.0, n_signal_discovery=80, signal_rate_discovery=0.4, n_signal_validation=70, signal_rate_validation=0.38),
        _make_stability_row("sig_test", "short", 2, 3.0, n_signal_discovery=55, signal_rate_discovery=0.35, n_signal_validation=50, signal_rate_validation=0.32),
        _make_stability_row("sig_test", "long", 2, 4.0, n_signal_discovery=80, signal_rate_discovery=0.65, n_signal_validation=70, signal_rate_validation=0.05),
        _make_stability_row("sig_test", "short", 2, 4.0, n_signal_discovery=55, signal_rate_discovery=0.60, n_signal_validation=50, signal_rate_validation=0.04),
        _make_stability_row("sig_test", "long", 1, 2.0, n_signal_discovery=80, signal_rate_discovery=0.5, n_signal_validation=70, signal_rate_validation=0.48),
        _make_stability_row("sig_test", "short", 1, 2.0, n_signal_discovery=55, signal_rate_discovery=0.45, n_signal_validation=50, signal_rate_validation=0.43),
    ])
    lb = compute_leaderboard_geometry_lock(
        merged_by_signal,
        stability_df=stab,
        discovery_end="2021-06-01",
        date_start="2020-01-01",
        date_end="2025-12-31",
    )
    row = lb[lb["signal_name"] == "sig_test"].iloc[0]
    assert row["P_4R_before_2R_disc"] > 0.5
    assert row["P_4R_before_2R_val"] < 0.1
    drop = (row["P_4R_before_2R_disc"] - row["P_4R_before_2R_val"]) / max(row["P_4R_before_2R_disc"], 1e-12)
    assert drop > 0.10
    assert bool(row["PASS"]) is False
    assert "P_4R_before_2R_drop" in str(row.get("reject_reason", ""))
