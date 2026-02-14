"""
Phase D-2 Lift Harness — tests using synthetic data only.

Validates: join correctness, always_fire lift/coverage, oracle behavior,
random_fire determinism, split logic, minimum sample gating.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def has_parquet_engine() -> bool:
    """True if pyarrow or fastparquet available for pandas.read_parquet."""
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401
        return True
    except ImportError:
        pass
    return False


def _synthetic_labels(n: int = 500) -> pd.DataFrame:
    """Minimal labels with required columns for D-2."""
    np.random.seed(42)
    pairs = ["EUR_USD", "GBP_USD"]
    dates = pd.date_range("2020-01-01", periods=n // 4, freq="D")
    rows = []
    for pair in pairs:
        for d in dates:
            for direction in ("long", "short"):
                # ~10% zone A, ~5% zone B, ~2% zone C
                u = np.random.random()
                za = 1 if u < 0.10 else 0
                zb = 1 if u < 0.05 else 0
                zc = 1 if u < 0.02 else 0
                split = "discovery" if d <= pd.Timestamp("2022-12-31") else "validation"
                rows.append(
                    {
                        "pair": pair,
                        "date": d,
                        "direction": direction,
                        "dataset_split": split,
                        "zone_a_1r_10": za,
                        "zone_b_3r_20": zb,
                        "zone_c_6r_40": zc,
                    }
                )
    df = pd.DataFrame(rows)
    df["direction"] = pd.Categorical(df["direction"], categories=["long", "short"], ordered=True)
    return df.sort_values(["pair", "date", "direction"]).reset_index(drop=True)


def test_join_keys_correctness() -> None:
    """Join on (pair, date, direction) must preserve all label rows when signal covers them."""
    from scripts.phaseD2_run_lift_harness import generate_always_fire, join_signals_to_labels

    labels = _synthetic_labels(100)
    signals = generate_always_fire(labels)
    joined = join_signals_to_labels(
        labels, signals, discovery_end="2022-12-31"
    )
    assert len(joined) == len(labels)
    assert set(joined.columns) >= {
        "pair", "date", "direction", "signal", "signal_name",
        "zone_a_1r_10", "zone_b_3r_20", "zone_c_6r_40",
    }
    pd.testing.assert_frame_equal(
        joined[["pair", "date", "direction"]].sort_values(["pair", "date", "direction"]).reset_index(drop=True),
        labels[["pair", "date", "direction"]].sort_values(["pair", "date", "direction"]).reset_index(drop=True),
    )


def test_always_fire_lift_one_coverage_one() -> None:
    """always_fire: lift must be 1.0, coverage must be 1.0 for all zones."""
    from analytics.phaseD2_metrics import compute_coverage, compute_metrics
    from scripts.phaseD2_run_lift_harness import generate_always_fire, join_signals_to_labels

    labels = _synthetic_labels(500)
    signals = generate_always_fire(labels)
    joined = join_signals_to_labels(labels, signals, discovery_end="2022-12-31")

    metrics = compute_metrics(joined)
    cov = compute_coverage(joined)

    af_global = metrics["metrics_global"][metrics["metrics_global"]["signal_name"] == "always_fire"].iloc[0]
    assert af_global["lift_zone_a_1r_10"] == pytest.approx(1.0)
    assert af_global["lift_zone_b_3r_20"] == pytest.approx(1.0)
    assert af_global["lift_zone_c_6r_40"] == pytest.approx(1.0)
    assert af_global["fire_rate"] == pytest.approx(1.0)

    af_cov = cov[cov["signal_name"] == "always_fire"].iloc[0]
    assert af_cov["coverage_zone_a_1r_10"] == pytest.approx(1.0)
    assert af_cov["coverage_zone_b_3r_20"] == pytest.approx(1.0)
    assert af_cov["coverage_zone_c_6r_40"] == pytest.approx(1.0)


def test_oracle_behaviour() -> None:
    """oracle_zone_b: P(B|fire)=1, coverage_B=1, lift_B >> 1."""
    from analytics.phaseD2_metrics import compute_coverage, compute_metrics
    from scripts.phaseD2_run_lift_harness import generate_oracle_signal, join_signals_to_labels

    labels = _synthetic_labels(500)
    signals = generate_oracle_signal(labels, "zone_b_3r_20")
    joined = join_signals_to_labels(labels, signals, discovery_end="2022-12-31")

    metrics = compute_metrics(joined)
    cov = compute_coverage(joined)

    ora = metrics["metrics_global"][metrics["metrics_global"]["signal_name"] == "oracle_zone_b"].iloc[0]
    assert ora["p_zone_b_3r_20_given_fire"] == pytest.approx(1.0)
    assert ora["lift_zone_b_3r_20"] >= 10.0

    ora_cov = cov[cov["signal_name"] == "oracle_zone_b"].iloc[0]
    assert ora_cov["coverage_zone_b_3r_20"] == pytest.approx(1.0)


def test_random_fire_determinism() -> None:
    """Same seed => identical signal column."""
    from scripts.phaseD2_run_lift_harness import generate_random_fire

    labels = _synthetic_labels(300)
    s1 = generate_random_fire(labels, p=0.05, seed=12345)
    s2 = generate_random_fire(labels, p=0.05, seed=12345)
    pd.testing.assert_series_equal(s1["signal"], s2["signal"])

    s3 = generate_random_fire(labels, p=0.05, seed=99999)
    assert not s1["signal"].equals(s3["signal"])


def test_split_logic_discovery_vs_validation() -> None:
    """dataset_split: dates <= 2022-12-31 = discovery, else validation."""
    from scripts.phaseD2_run_lift_harness import join_signals_to_labels

    dates = list(pd.date_range("2022-12-01", periods=80, freq="D"))
    pairs = ["EUR_USD", "GBP_USD"]
    rows = []
    for pair in pairs:
        for d in dates:
            for direction in ("long", "short"):
                rows.append({
                    "pair": pair, "date": d, "direction": direction,
                    "zone_a_1r_10": 0, "zone_b_3r_20": 0, "zone_c_6r_40": 0,
                })
    labels = pd.DataFrame(rows)
    labels["dataset_split"] = labels["date"].apply(
        lambda d: "discovery" if d <= pd.Timestamp("2022-12-31") else "validation"
    )
    signals = labels[["pair", "date", "direction"]].copy()
    signals["signal"] = 1
    signals["signal_name"] = "always_fire"

    joined = join_signals_to_labels(labels, signals, discovery_end="2022-12-31")
    assert "discovery" in joined["dataset_split"].values
    assert "validation" in joined["dataset_split"].values
    assert (joined[joined["date"] <= "2022-12-31"]["dataset_split"] == "discovery").all()
    assert (joined[joined["date"] > "2022-12-31"]["dataset_split"] == "validation").all()


def test_minimum_sample_gating() -> None:
    """min_fire_ok = (F >= 200) globally; per pair F >= 30 for stable (flag)."""
    from analytics.phaseD2_metrics import compute_metrics
    from scripts.phaseD2_run_lift_harness import generate_always_fire, join_signals_to_labels

    labels = _synthetic_labels(500)
    signals = generate_always_fire(labels)
    joined = join_signals_to_labels(labels, signals, discovery_end="2022-12-31")

    metrics = compute_metrics(joined)
    af = metrics["metrics_global"][metrics["metrics_global"]["signal_name"] == "always_fire"].iloc[0]
    assert bool(af["min_fire_ok"]) is True
    assert af["fired"] == len(labels)

    pair_metrics = metrics["metrics_pair"]
    if not pair_metrics.empty:
        af_pair = pair_metrics[pair_metrics["signal_name"] == "always_fire"]
        assert "min_fire_ok_pair" in af_pair.columns
        assert af_pair["min_fire_ok_pair"].all()


def test_small_fire_not_ok() -> None:
    """When F < 200, min_fire_ok must be False."""
    from analytics.phaseD2_metrics import compute_metrics

    df = _synthetic_labels(50)
    df["signal"] = 1
    df["signal_name"] = "tiny_signal"

    metrics = compute_metrics(df, signal_col="signal", signal_name_col="signal_name")
    row = metrics["metrics_global"].iloc[0]
    assert row["fired"] < 200
    assert bool(row["min_fire_ok"]) is False


def test_metrics_missing_columns_raises() -> None:
    """compute_metrics must fail-fast on missing zone columns."""
    from analytics.phaseD2_metrics import compute_metrics

    df = pd.DataFrame({
        "pair": ["A"], "date": [pd.Timestamp("2020-01-01")], "direction": ["long"],
        "signal": [1], "signal_name": ["x"],
    })
    with pytest.raises(ValueError, match="Missing zone"):
        compute_metrics(df)


def test_external_signals_csv_loaded_and_joined(tmp_path: Path) -> None:
    """External signal CSV is loaded, joined, and multiple signal_names preserved."""
    from analytics.phaseD2_metrics import compute_metrics
    from scripts.phaseD2_run_lift_harness import (
        build_control_signals,
        join_signals_to_labels,
    )

    labels = _synthetic_labels(200)
    ext_path = tmp_path / "ext_signals.csv"
    rows = []
    for _, row in labels.iterrows():
        for sig_name in ["proto_a", "proto_b"]:
            rows.append({
                "pair": row["pair"],
                "date": row["date"],
                "direction": row["direction"],
                "signal": 1 if sig_name == "proto_a" else 0,
                "signal_name": sig_name,
            })
    ext_df = pd.DataFrame(rows)
    ext_df.to_csv(ext_path, index=False)

    cfg = {
        "always_fire": False,
        "random_fire_enabled": False,
        "oracle_zones": [],
        "external_signals": [{"path": str(ext_path)}],
    }
    signals = build_control_signals(labels, cfg)
    joined = join_signals_to_labels(labels, signals, discovery_end="2022-12-31")

    sig_names = set(joined["signal_name"].unique())
    assert "proto_a" in sig_names
    assert "proto_b" in sig_names

    metrics = compute_metrics(joined)
    global_df = metrics["metrics_global"]
    a_row = global_df[global_df["signal_name"] == "proto_a"].iloc[0]
    b_row = global_df[global_df["signal_name"] == "proto_b"].iloc[0]
    assert a_row["fired"] == len(labels)
    assert b_row["fired"] == 0


@pytest.mark.skipif(
    not has_parquet_engine(),
    reason="Parquet engine (pyarrow/fastparquet) not installed in CI clean env",
)
def test_external_signals_loaded_and_joined(tmp_path: Path) -> None:
    """External signal parquet is loaded, validated, and included in metrics."""
    from analytics.phaseD2_metrics import compute_metrics
    from scripts.phaseD2_run_lift_harness import (
        build_control_signals,
        join_signals_to_labels,
    )

    labels = _synthetic_labels(200)
    ext_path = tmp_path / "ext_signal.parquet"
    ext_df = pd.DataFrame({
        "pair": labels["pair"],
        "date": labels["date"],
        "direction": labels["direction"],
        "signal": (labels["pair"] == "EUR_USD").astype(int),
        "signal_name": "momentum_3bar",
    })
    ext_df.to_parquet(ext_path, index=False)

    cfg = {
        "always_fire": True,
        "random_fire_enabled": False,
        "oracle_zones": [],
        "external_signals": [{"path": str(ext_path), "name": "momentum_3bar"}],
    }
    signals = build_control_signals(labels, cfg)
    joined = join_signals_to_labels(
        labels, signals, discovery_end="2022-12-31"
    )

    sig_names = set(joined["signal_name"].unique())
    assert "always_fire" in sig_names
    assert "momentum_3bar" in sig_names

    metrics = compute_metrics(joined)
    global_df = metrics["metrics_global"]
    mom = global_df[global_df["signal_name"] == "momentum_3bar"].iloc[0]
    assert mom["fired"] > 0


@pytest.mark.skipif(
    not has_parquet_engine(),
    reason="Parquet engine (pyarrow/fastparquet) not installed in CI clean env",
)
def test_external_multi_signal_file_preserves_signal_names(tmp_path: Path) -> None:
    """External file with multiple signal_names: each appears separately in metrics."""
    from analytics.phaseD2_metrics import compute_metrics
    from scripts.phaseD2_run_lift_harness import (
        build_control_signals,
        join_signals_to_labels,
    )

    labels = _synthetic_labels(200)
    ext_path = tmp_path / "proto_signals.parquet"
    rows = []
    for _, row in labels.iterrows():
        for sig_name in ["proto_comp_atrp_low", "proto_ignite_tr_high"]:
            rows.append({
                "pair": row["pair"],
                "date": row["date"],
                "direction": row["direction"],
                "signal": 1 if sig_name == "proto_comp_atrp_low" else 0,
                "signal_name": sig_name,
            })
    ext_df = pd.DataFrame(rows)
    ext_df.to_parquet(ext_path, index=False)

    cfg = {
        "always_fire": False,
        "random_fire_enabled": False,
        "oracle_zones": [],
        "external_signals": [{"path": str(ext_path)}],
    }
    signals = build_control_signals(labels, cfg)
    joined = join_signals_to_labels(
        labels, signals, discovery_end="2022-12-31"
    )

    sig_names = set(joined["signal_name"].unique())
    assert "proto_comp_atrp_low" in sig_names
    assert "proto_ignite_tr_high" in sig_names

    metrics = compute_metrics(joined)
    global_df = metrics["metrics_global"]
    comp = global_df[global_df["signal_name"] == "proto_comp_atrp_low"].iloc[0]
    ign = global_df[global_df["signal_name"] == "proto_ignite_tr_high"].iloc[0]
    assert comp["fired"] == len(labels)
    assert ign["fired"] == 0
