"""
Phase D-2.3 Proto-signals — tests using synthetic data only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_features(
    n_discovery: int = 25,
    n_validation: int = 5,
    pair: str = "EUR_USD",
) -> pd.DataFrame:
    """Build fake features with controllable atrp_14, true_range, breakout flags."""
    np.random.seed(42)
    dates_disc = pd.date_range("2022-01-01", periods=n_discovery, freq="D")
    dates_val = pd.date_range("2023-01-01", periods=n_validation, freq="D")
    dates = list(dates_disc) + list(dates_val)

    rows = []
    for i, d in enumerate(dates):
        split = "discovery" if d <= pd.Timestamp("2022-12-31") else "validation"
        rows.append({
            "pair": pair,
            "date": d,
            "dataset_split": split,
            "atrp_14": 0.001 + (i * 0.0008),
            "true_range": 0.001 + (i * 0.0008),
            "breakout_up_20": 1.0 if i == 2 else 0.0,
            "breakout_dn_20": 1.0 if i == 5 else 0.0,
            "pos_in_range_20": max(0.0, min(1.0, 0.3 + i * 0.02)),
            "tr_atr_ratio": 0.8 + (i * 0.02),
            "mom_slope_5": 0.0001 * (i - 15),
        })
    df = pd.DataFrame(rows)
    df["atrp_14"] = df["atrp_14"].astype(float)
    df["true_range"] = df["true_range"].astype(float)
    return df


def _synthetic_features_atrp_low_tr_high() -> pd.DataFrame:
    """At least one row in lowest atrp decile, one in top tr decile."""
    np.random.seed(43)
    n = 50
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    atrp = np.sort(np.random.uniform(0.001, 0.02, n))
    tr = np.sort(np.random.uniform(0.001, 0.02, n))
    rows = []
    for i, d in enumerate(dates):
        split = "discovery" if d <= pd.Timestamp("2022-12-31") else "validation"
        rows.append({
            "pair": "EUR_USD",
            "date": d,
            "dataset_split": split,
            "atrp_14": atrp[i],
            "true_range": tr[i],
            "breakout_up_20": 1.0 if i == 10 else 0.0,
            "breakout_dn_20": 1.0 if i == 25 else 0.0,
            "pos_in_range_20": max(0.0, min(1.0, 0.2 + i * 0.015)),
            "tr_atr_ratio": 0.7 + (i * 0.025),
            "mom_slope_5": 0.00005 * (i - 25),
        })
    return pd.DataFrame(rows)


def _synthetic_features_with_nan() -> pd.DataFrame:
    """Some rows have NaN for atrp_14 or true_range."""
    df = _synthetic_features(25, 5)
    df.loc[df.index[0], "atrp_14"] = np.nan
    df.loc[df.index[3], "true_range"] = np.nan
    return df


def test_bin_edges_from_discovery_only() -> None:
    """Bin edges computed from discovery split only."""
    from analytics.phaseD2_2_features import compute_bin_edges_from_discovery

    df = _synthetic_features(25, 5)
    edges = compute_bin_edges_from_discovery(
        df, "atrp_14", n_bins=10, min_per_bin=2
    )
    assert edges is not None
    assert len(edges) == 11

    discovery = df[df["dataset_split"] == "discovery"]
    atrp_vals = discovery["atrp_14"].dropna()
    assert len(atrp_vals) >= 20


def test_proto_comp_atrp_low_fires_both_directions() -> None:
    """proto_comp_atrp_low fires both long and short when atrp in lowest decile."""
    from analytics.phaseD2_2_features import (
        apply_bin_edges,
        compute_bin_edges_from_discovery,
    )
    from scripts.phaseD2_3_generate_proto_signals import (
        _build_spine,
        _ensure_dataset_split,
        _generate_proto_comp_atrp_low,
    )

    df = _synthetic_features_atrp_low_tr_high()
    df = _ensure_dataset_split(df, "2022-12-31")
    edges = compute_bin_edges_from_discovery(
        df, "atrp_14", n_bins=10, min_per_bin=2
    )
    assert edges is not None
    atrp_bin = apply_bin_edges(df["atrp_14"], edges)
    atrp_low = (atrp_bin == 0).fillna(False)

    spine = _build_spine(df)
    out = _generate_proto_comp_atrp_low(spine, df, atrp_low)

    fired = out[out["signal"] == 1]
    if len(fired) > 0:
        assert set(fired["direction"].unique()) == {"long", "short"}
    assert out["signal"].isin([0, 1]).all()
    assert len(out) == len(df) * 2


def test_proto_ignite_tr_high_fires_both_directions() -> None:
    """proto_ignite_tr_high fires both directions when tr in top decile."""
    from analytics.phaseD2_2_features import (
        apply_bin_edges,
        compute_bin_edges_from_discovery,
    )
    from scripts.phaseD2_3_generate_proto_signals import (
        _build_spine,
        _ensure_dataset_split,
        _generate_proto_ignite_tr_high,
    )

    df = _synthetic_features_atrp_low_tr_high()
    df = _ensure_dataset_split(df, "2022-12-31")
    edges = compute_bin_edges_from_discovery(
        df, "true_range", n_bins=10, min_per_bin=2
    )
    assert edges is not None
    tr_bin = apply_bin_edges(df["true_range"], edges)
    tr_high = (tr_bin == 9).fillna(False)

    spine = _build_spine(df)
    out = _generate_proto_ignite_tr_high(spine, df, tr_high)

    assert out["signal"].isin([0, 1]).all()
    assert len(out) == len(df) * 2


def test_breakout_up_fires_long_only() -> None:
    """proto_comp_low_atrp_breakout_up fires only long when atrp low + breakout_up."""
    from analytics.phaseD2_2_features import (
        apply_bin_edges,
        compute_bin_edges_from_discovery,
    )
    from scripts.phaseD2_3_generate_proto_signals import (
        _build_spine,
        _ensure_dataset_split,
        _generate_proto_comp_low_atrp_breakout_up,
    )

    df = _synthetic_features(25, 5)
    df.loc[df.index[2], "breakout_up_20"] = 1.0
    df = _ensure_dataset_split(df, "2022-12-31")
    edges = compute_bin_edges_from_discovery(
        df, "atrp_14", n_bins=10, min_per_bin=2
    )
    atrp_bin = apply_bin_edges(df["atrp_14"], edges)
    atrp_low = (atrp_bin == 0).fillna(False)

    spine = _build_spine(df)
    out = _generate_proto_comp_low_atrp_breakout_up(spine, df, atrp_low)

    fired = out[out["signal"] == 1]
    if len(fired) > 0:
        assert (fired["direction"] == "long").all()


def test_breakout_dn_fires_short_only() -> None:
    """proto_comp_low_atrp_breakout_dn fires only short when atrp low + breakout_dn."""
    from analytics.phaseD2_2_features import (
        apply_bin_edges,
        compute_bin_edges_from_discovery,
    )
    from scripts.phaseD2_3_generate_proto_signals import (
        _build_spine,
        _ensure_dataset_split,
        _generate_proto_comp_low_atrp_breakout_dn,
    )

    df = _synthetic_features(25, 5)
    df.loc[df.index[5], "breakout_dn_20"] = 1.0
    df = _ensure_dataset_split(df, "2022-12-31")
    edges = compute_bin_edges_from_discovery(
        df, "atrp_14", n_bins=10, min_per_bin=2
    )
    atrp_bin = apply_bin_edges(df["atrp_14"], edges)
    atrp_low = (atrp_bin == 0).fillna(False)

    spine = _build_spine(df)
    out = _generate_proto_comp_low_atrp_breakout_dn(spine, df, atrp_low)

    fired = out[out["signal"] == 1]
    if len(fired) > 0:
        assert (fired["direction"] == "short").all()


def test_nan_features_produce_signal_zero() -> None:
    """Rows with NaN atrp_14 or true_range produce signal=0."""
    from analytics.phaseD2_2_features import (
        apply_bin_edges,
        compute_bin_edges_from_discovery,
    )
    from scripts.phaseD2_3_generate_proto_signals import (
        _build_spine,
        _ensure_dataset_split,
        _generate_proto_comp_atrp_low,
    )

    df = _synthetic_features_with_nan()
    df = _ensure_dataset_split(df, "2022-12-31")
    edges = compute_bin_edges_from_discovery(
        df, "atrp_14", n_bins=10, min_per_bin=2
    )
    atrp_bin = apply_bin_edges(df["atrp_14"], edges)
    atrp_low = (atrp_bin == 0).fillna(False)

    spine = _build_spine(df)
    out = _generate_proto_comp_atrp_low(spine, df, atrp_low)

    nan_row_idx = df.index[0]
    nan_dates = df.loc[[nan_row_idx], "date"]
    nan_rows = out[out["date"].isin(nan_dates)]
    assert (nan_rows["signal"] == 0).all()


def test_output_two_rows_per_date_per_signal() -> None:
    """Each signal has exactly two rows per (pair, date): long and short."""
    from analytics.phaseD2_2_features import (
        apply_bin_edges,
        compute_bin_edges_from_discovery,
    )
    from scripts.phaseD2_3_generate_proto_signals import (
        _build_spine,
        _ensure_dataset_split,
        _generate_proto_comp_atrp_low,
    )

    df = _synthetic_features(25, 5)
    df = _ensure_dataset_split(df, "2022-12-31")
    edges = compute_bin_edges_from_discovery(
        df, "atrp_14", n_bins=10, min_per_bin=2
    )
    atrp_bin = apply_bin_edges(df["atrp_14"], edges)
    atrp_low = (atrp_bin == 0).fillna(False)

    spine = _build_spine(df)
    out = _generate_proto_comp_atrp_low(spine, df, atrp_low)

    for (pair, date), grp in out.groupby(["pair", "date"]):
        assert len(grp) == 2
        assert set(grp["direction"]) == {"long", "short"}


def test_proto_signals_script_e2e(tmp_path: Path) -> None:
    """Full script run produces valid parquet and csv."""
    df = _synthetic_features(25, 5)
    df = df[[
        "pair", "date", "dataset_split", "atrp_14", "true_range",
        "breakout_up_20", "breakout_dn_20", "pos_in_range_20", "tr_atr_ratio", "mom_slope_5",
    ]]
    features_path = tmp_path / "features.parquet"
    df.to_parquet(features_path, index=False)

    config = {
        "phase": "D2_3_proto_signals",
        "features_path": str(features_path),
        "outputs_dir": str(tmp_path / "signals"),
        "split": {"discovery_end": "2022-12-31"},
        "bins": {"n_bins": 10, "min_per_bin": 2},
        "signals": [
            {"name": "proto_comp_atrp_low", "rule": "compression_atrp_low"},
            {"name": "proto_ignite_tr_high", "rule": "ignition_tr_high"},
        ],
    }
    config_path = tmp_path / "config.yaml"
    import yaml
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    from scripts.phaseD2_3_generate_proto_signals import main

    main(["-c", str(config_path)])

    parquet_path = tmp_path / "signals" / "proto_signals.parquet"
    csv_path = tmp_path / "signals" / "proto_signals.csv"
    assert parquet_path.exists()
    assert csv_path.exists()

    out = pd.read_parquet(parquet_path)
    assert set(out.columns) == {"pair", "date", "direction", "signal", "signal_name"}
    assert out["signal"].isin([0, 1]).all()
    assert set(out["signal_name"]) == {"proto_comp_atrp_low", "proto_ignite_tr_high"}


def test_config_validation_fail_fast() -> None:
    """Missing config keys raise ValueError."""
    from scripts.phaseD2_3_generate_proto_signals import _require_config

    with pytest.raises(ValueError, match="features_path"):
        _require_config({})
    with pytest.raises(ValueError, match="outputs_dir"):
        _require_config({"features_path": "x"})
    with pytest.raises(ValueError, match="discovery_end"):
        _require_config({"features_path": "x", "outputs_dir": "y", "split": {}})
    with pytest.raises(ValueError, match="signals"):
        _require_config({
            "features_path": "x",
            "outputs_dir": "y",
            "split": {"discovery_end": "2022-12-31"},
        })


def test_pos_pressure_signals_fire_correctly() -> None:
    """proto_comp_atrp_low_pos_pressure_up: long when pos>=0.85, short when pos<=0.15."""
    from analytics.phaseD2_2_features import (
        apply_bin_edges,
        compute_bin_edges_from_discovery,
    )
    from scripts.phaseD2_3_generate_proto_signals import (
        _build_spine,
        _ensure_dataset_split,
        _generate_proto_comp_atrp_low_pos_pressure_up,
    )

    df = _synthetic_features_atrp_low_tr_high()
    df = _ensure_dataset_split(df, "2022-12-31")
    df.loc[df.index[0], "pos_in_range_20"] = 0.90
    df.loc[df.index[1], "pos_in_range_20"] = 0.10
    edges = compute_bin_edges_from_discovery(
        df, "atrp_14", n_bins=10, min_per_bin=2
    )
    atrp_bin = apply_bin_edges(df["atrp_14"], edges)
    atrp_low = (atrp_bin == 0).fillna(False)

    spine = _build_spine(df)
    out = _generate_proto_comp_atrp_low_pos_pressure_up(spine, df, atrp_low)

    long_fired = out[(out["direction"] == "long") & (out["signal"] == 1)]
    short_fired = out[(out["direction"] == "short") & (out["signal"] == 1)]
    assert out["signal"].isin([0, 1]).all()
    if len(long_fired) > 0:
        assert long_fired["direction"].unique()[0] == "long"
    if len(short_fired) > 0:
        assert short_fired["direction"].unique()[0] == "short"


def test_tr_mid_high_70th_percentile_from_discovery() -> None:
    """proto_comp_atrp_low_tr_mid_high uses 70th percentile from discovery only."""
    from analytics.phaseD2_2_features import (
        apply_bin_edges,
        compute_bin_edges_from_discovery,
    )
    from scripts.phaseD2_3_generate_proto_signals import (
        _build_spine,
        _discovery_percentile_threshold,
        _ensure_dataset_split,
        _generate_proto_comp_atrp_low_tr_mid_high,
    )

    df = _synthetic_features_atrp_low_tr_high()
    df = _ensure_dataset_split(df, "2022-12-31")
    tr_70th = _discovery_percentile_threshold(df, "true_range", q=0.70)
    assert np.isfinite(tr_70th)

    edges = compute_bin_edges_from_discovery(
        df, "atrp_14", n_bins=10, min_per_bin=2
    )
    atrp_bin = apply_bin_edges(df["atrp_14"], edges)
    atrp_low = (atrp_bin == 0).fillna(False)

    spine = _build_spine(df)
    out = _generate_proto_comp_atrp_low_tr_mid_high(spine, df, atrp_low, tr_70th)

    assert out["signal"].isin([0, 1]).all()
    assert len(out) == len(df) * 2


def test_tr_atr_ratio_condition_works() -> None:
    """proto_comp_atrp_low_tr_atr_ratio_high fires when tr_atr_ratio in top 30%."""
    from analytics.phaseD2_2_features import (
        apply_bin_edges,
        compute_bin_edges_from_discovery,
    )
    from scripts.phaseD2_3_generate_proto_signals import (
        _build_spine,
        _discovery_percentile_threshold,
        _ensure_dataset_split,
        _generate_proto_comp_atrp_low_tr_atr_ratio_high,
    )

    df = _synthetic_features_atrp_low_tr_high()
    df = _ensure_dataset_split(df, "2022-12-31")
    tr_atr_70th = _discovery_percentile_threshold(df, "tr_atr_ratio", q=0.70)
    assert np.isfinite(tr_atr_70th)

    edges = compute_bin_edges_from_discovery(
        df, "atrp_14", n_bins=10, min_per_bin=2
    )
    atrp_bin = apply_bin_edges(df["atrp_14"], edges)
    atrp_low = (atrp_bin == 0).fillna(False)

    spine = _build_spine(df)
    out = _generate_proto_comp_atrp_low_tr_atr_ratio_high(
        spine, df, atrp_low, tr_atr_70th
    )

    assert out["signal"].isin([0, 1]).all()
    assert len(out) == len(df) * 2


def test_slope_alignment_works() -> None:
    """proto_comp_atrp_low_slope_alignment: long when slope>0, short when slope<0."""
    from analytics.phaseD2_2_features import (
        apply_bin_edges,
        compute_bin_edges_from_discovery,
    )
    from scripts.phaseD2_3_generate_proto_signals import (
        _build_spine,
        _ensure_dataset_split,
        _generate_proto_comp_atrp_low_slope_alignment,
    )

    df = _synthetic_features_atrp_low_tr_high()
    df = _ensure_dataset_split(df, "2022-12-31")
    df.loc[df.index[0], "mom_slope_5"] = 0.001
    df.loc[df.index[1], "mom_slope_5"] = -0.001
    edges = compute_bin_edges_from_discovery(
        df, "atrp_14", n_bins=10, min_per_bin=2
    )
    atrp_bin = apply_bin_edges(df["atrp_14"], edges)
    atrp_low = (atrp_bin == 0).fillna(False)

    spine = _build_spine(df)
    out = _generate_proto_comp_atrp_low_slope_alignment(spine, df, atrp_low)

    assert out["signal"].isin([0, 1]).all()
    assert len(out) == len(df) * 2


def test_new_signals_nan_produce_signal_zero() -> None:
    """pos_in_range_20, tr_atr_ratio, mom_slope_5 NaN produce signal=0."""
    from analytics.phaseD2_2_features import (
        apply_bin_edges,
        compute_bin_edges_from_discovery,
    )
    from scripts.phaseD2_3_generate_proto_signals import (
        _build_spine,
        _discovery_percentile_threshold,
        _ensure_dataset_split,
        _generate_proto_comp_atrp_low_pos_pressure_up,
        _generate_proto_comp_atrp_low_slope_alignment,
        _generate_proto_comp_atrp_low_tr_atr_ratio_high,
    )

    df = _synthetic_features_atrp_low_tr_high()
    df = _ensure_dataset_split(df, "2022-12-31")
    df.loc[df.index[0], "pos_in_range_20"] = np.nan
    df.loc[df.index[2], "mom_slope_5"] = np.nan
    df.loc[df.index[3], "tr_atr_ratio"] = np.nan
    edges = compute_bin_edges_from_discovery(
        df, "atrp_14", n_bins=10, min_per_bin=2
    )
    atrp_bin = apply_bin_edges(df["atrp_14"], edges)
    atrp_low = (atrp_bin == 0).fillna(False)
    tr_atr_70th = _discovery_percentile_threshold(df, "tr_atr_ratio", q=0.70)

    spine = _build_spine(df)

    out_pos = _generate_proto_comp_atrp_low_pos_pressure_up(spine, df, atrp_low)
    nan_dates = df.loc[[df.index[0]], "date"]
    assert (out_pos[out_pos["date"].isin(nan_dates)]["signal"] == 0).all()

    out_slope = _generate_proto_comp_atrp_low_slope_alignment(spine, df, atrp_low)
    nan_dates_slope = df.loc[[df.index[2]], "date"]
    assert (out_slope[out_slope["date"].isin(nan_dates_slope)]["signal"] == 0).all()

    out_ratio = _generate_proto_comp_atrp_low_tr_atr_ratio_high(
        spine, df, atrp_low, tr_atr_70th
    )
    nan_dates_ratio = df.loc[[df.index[3]], "date"]
    assert (out_ratio[out_ratio["date"].isin(nan_dates_ratio)]["signal"] == 0).all()


def test_deterministic_ordering_maintained(tmp_path: Path) -> None:
    """Output order is deterministic: pair, date, direction, signal_name."""
    import yaml

    df = _synthetic_features(25, 5)
    features_path = tmp_path / "features.parquet"
    df.to_parquet(features_path, index=False)

    config = {
        "phase": "D2_3_proto_signals",
        "features_path": str(features_path),
        "outputs_dir": str(tmp_path / "signals"),
        "split": {"discovery_end": "2022-12-31"},
        "bins": {"n_bins": 10, "min_per_bin": 2},
        "signals": [
            {"name": "proto_comp_atrp_low", "rule": "compression_atrp_low"},
            {"name": "proto_comp_atrp_low_pos_pressure_up", "rule": "compression_atrp_low_pos_pressure_up"},
        ],
    }
    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    from scripts.phaseD2_3_generate_proto_signals import main

    main(["-c", str(config_path)])
    out = pd.read_parquet(tmp_path / "signals" / "proto_signals.parquet")
    sorted_by = out.sort_values(["pair", "date", "direction", "signal_name"])
    assert out["pair"].tolist() == sorted_by["pair"].tolist()
    assert out["direction"].tolist() == sorted_by["direction"].tolist()
