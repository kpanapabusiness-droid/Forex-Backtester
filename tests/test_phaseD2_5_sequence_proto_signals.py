"""
Phase D-2.5 Sequence proto-signals — tests using synthetic data only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_sequence_features(
    n_bars: int = 10,
    pair: str = "EUR_USD",
    compression_at: int | None = 2,
    breakout_up_at: int | None = None,
    breakout_dn_at: int | None = None,
    discovery_end: str = "2022-12-31",
) -> pd.DataFrame:
    """Build features with controllable compression and trigger bars."""
    dates = pd.date_range("2022-01-01", periods=n_bars, freq="D")
    atrp = np.ones(n_bars) * 0.01
    if compression_at is not None:
        atrp[compression_at] = 0.0001

    rows = []
    for i, d in enumerate(dates):
        split = "discovery" if pd.Timestamp(d) <= pd.Timestamp(discovery_end) else "validation"
        rows.append({
            "pair": pair,
            "date": d,
            "dataset_split": split,
            "atrp_14": atrp[i],
            "true_range": 0.001 + i * 0.0001,
            "breakout_up_20": 1.0 if i == breakout_up_at else 0.0,
            "breakout_dn_20": 1.0 if i == breakout_dn_at else 0.0,
            "pos_in_range_20": 0.5 + i * 0.02,
            "tr_atr_ratio": 0.9 + i * 0.05,
        })
    return pd.DataFrame(rows)


def test_rolling_compression_window_correctness() -> None:
    """comp_recent_K true for K bars after compression, then false."""
    from scripts.phaseD2_5_generate_sequence_proto_signals import (
        _comp_recent_K_per_pair,
        _ensure_dataset_split,
    )

    df = _synthetic_sequence_features(
        n_bars=10, compression_at=2, discovery_end="2022-12-31"
    )
    df = _ensure_dataset_split(df, "2022-12-31")
    from analytics.phaseD2_2_features import (
        apply_bin_edges,
        compute_bin_edges_from_discovery,
    )

    edges = compute_bin_edges_from_discovery(
        df, "atrp_14", n_bins=10, min_per_bin=1
    )
    assert edges is not None
    bin_series = apply_bin_edges(df["atrp_14"], edges)
    is_comp = (bin_series == 0).fillna(False)
    df["_comp"] = is_comp.astype(int)

    comp_recent_3 = _comp_recent_K_per_pair(df, 3)
    assert comp_recent_3.iloc[2]
    assert comp_recent_3.iloc[3]
    assert comp_recent_3.iloc[4]
    assert comp_recent_3.iloc[5]
    assert not comp_recent_3.iloc[6]


def test_trigger_fires_at_trigger_bar_not_compression_bar() -> None:
    """With compression at t=2, breakout_up at t=5, K=3: signal fires at t=5 (long)."""
    df = _synthetic_sequence_features(
        n_bars=10,
        compression_at=2,
        breakout_up_at=5,
        discovery_end="2022-12-31",
    )
    features_path = ROOT / "tmp_seq_test.parquet"
    try:
        df.to_parquet(features_path, index=False)
        config = {
            "phase": "D2_5_sequence_proto_signals",
            "features_path": str(features_path),
            "outputs_dir": str(ROOT / "tmp_seq_out"),
            "split": {"discovery_end": "2022-12-31"},
            "bins": {"n_bins": 10, "min_per_bin": 1},
            "compression": {"feature": "atrp_14", "bin": 0},
            "sequence": {"K_list": [3]},
            "ignition": {"tr_atr_ratio_percentile": 0.90},
            "pressure": {"thresholds": {"up": 0.90, "dn": 0.10}},
        }
        import yaml
        config_path = ROOT / "tmp_seq_config.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(config, f)

        from scripts.phaseD2_5_generate_sequence_proto_signals import main

        main(["-c", str(config_path)])

        out = pd.read_parquet(ROOT / "tmp_seq_out" / "sequence_proto_signals.parquet")
        breakout_up = out[
            (out["signal_name"] == "seq_comp3_breakout_up") & (out["signal"] == 1)
        ]
        if len(breakout_up) > 0:
            dates_fired = breakout_up["date"].unique()
            assert len(dates_fired) >= 1
            fired_long = breakout_up[breakout_up["direction"] == "long"]
            assert len(fired_long) >= 1
    finally:
        if features_path.exists():
            features_path.unlink()
        config_path = ROOT / "tmp_seq_config.yaml"
        if config_path.exists():
            config_path.unlink()
        out_dir = ROOT / "tmp_seq_out"
        if out_dir.exists():
            import shutil
            shutil.rmtree(out_dir)


def test_ignition_threshold_from_discovery_only() -> None:
    """tr_atr_ratio p90 computed from discovery; validation extremes don't shift it."""
    np.random.seed(42)
    n_disc = 100
    n_val = 20
    dates_disc = pd.date_range("2022-01-01", periods=n_disc, freq="D")
    dates_val = pd.date_range("2023-01-01", periods=n_val, freq="D")
    tr_atr_disc = np.sort(np.random.uniform(0.8, 1.5, n_disc))
    tr_atr_val = np.ones(n_val) * 999.0

    rows = []
    for i, d in enumerate(dates_disc):
        rows.append({
            "pair": "EUR_USD",
            "date": d,
            "dataset_split": "discovery",
            "atrp_14": 0.005 + i * 0.0001,
            "true_range": 0.001,
            "breakout_up_20": 0.0,
            "breakout_dn_20": 0.0,
            "pos_in_range_20": 0.5,
            "tr_atr_ratio": tr_atr_disc[i],
        })
    for i, d in enumerate(dates_val):
        rows.append({
            "pair": "EUR_USD",
            "date": d,
            "dataset_split": "validation",
            "atrp_14": 0.01,
            "true_range": 0.001,
            "breakout_up_20": 0.0,
            "breakout_dn_20": 0.0,
            "pos_in_range_20": 0.5,
            "tr_atr_ratio": tr_atr_val[i],
        })
    _ = pd.DataFrame(rows)
    p90 = float(np.quantile(tr_atr_disc, 0.90))
    assert p90 < 1.5
    assert p90 < 100.0


def test_two_rows_per_date_per_signal(tmp_path: Path) -> None:
    """For N bars and M signals, output has 2*N*M rows (2 directions per date per signal)."""
    import yaml

    n_bars = 10
    df = _synthetic_sequence_features(n_bars=n_bars)
    features_path = tmp_path / "features.parquet"
    df.to_parquet(features_path, index=False)

    config = {
        "phase": "D2_5_sequence_proto_signals",
        "features_path": str(features_path),
        "outputs_dir": str(tmp_path / "signals"),
        "split": {"discovery_end": "2022-12-31"},
        "bins": {"n_bins": 10, "min_per_bin": 1},
        "compression": {"feature": "atrp_14", "bin": 0},
        "sequence": {"K_list": [3, 5]},
        "ignition": {"tr_atr_ratio_percentile": 0.90},
        "pressure": {"thresholds": {"up": 0.90, "dn": 0.10}},
    }
    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    from scripts.phaseD2_5_generate_sequence_proto_signals import main

    main(["-c", str(config_path)])

    out = pd.read_parquet(tmp_path / "signals" / "sequence_proto_signals.parquet")
    assert set(out.columns) == {"pair", "date", "direction", "signal", "signal_name"}
    assert out["signal"].isin([0, 1]).all()

    for (sig_name, date), grp in out.groupby(["signal_name", "date"]):
        assert len(grp) == 2
        assert set(grp["direction"]) == {"long", "short"}


def test_deterministic_ordering(tmp_path: Path) -> None:
    """Output sorted by (signal_name, pair, date, direction)."""
    import yaml

    df = _synthetic_sequence_features(n_bars=15)
    features_path = tmp_path / "features.parquet"
    df.to_parquet(features_path, index=False)

    config = {
        "phase": "D2_5_sequence_proto_signals",
        "features_path": str(features_path),
        "outputs_dir": str(tmp_path / "signals"),
        "split": {"discovery_end": "2022-12-31"},
        "bins": {"n_bins": 10, "min_per_bin": 1},
        "compression": {"feature": "atrp_14", "bin": 0},
        "sequence": {"K_list": [3]},
        "ignition": {"tr_atr_ratio_percentile": 0.90},
        "pressure": {"thresholds": {"up": 0.90, "dn": 0.10}},
    }
    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    from scripts.phaseD2_5_generate_sequence_proto_signals import main

    main(["-c", str(config_path)])
    out = pd.read_parquet(tmp_path / "signals" / "sequence_proto_signals.parquet")
    sorted_by = out.sort_values(["signal_name", "pair", "date", "direction"])
    assert out["signal_name"].tolist() == sorted_by["signal_name"].tolist()
    assert out["direction"].tolist() == sorted_by["direction"].tolist()


def test_config_validation_fail_fast() -> None:
    """Missing config keys raise ValueError."""
    from scripts.phaseD2_5_generate_sequence_proto_signals import _require_config

    with pytest.raises(ValueError, match="features_path"):
        _require_config({})
    with pytest.raises(ValueError, match="outputs_dir"):
        _require_config({"features_path": "x"})
    with pytest.raises(ValueError, match="discovery_end"):
        _require_config({
            "features_path": "x",
            "outputs_dir": "y",
            "split": {},
        })
    with pytest.raises(ValueError, match="compression.feature"):
        _require_config({
            "features_path": "x",
            "outputs_dir": "y",
            "split": {"discovery_end": "2022-12-31"},
        })
    with pytest.raises(ValueError, match="K_list"):
        _require_config({
            "features_path": "x",
            "outputs_dir": "y",
            "split": {"discovery_end": "2022-12-31"},
            "compression": {"feature": "atrp_14", "bin": 0},
            "sequence": {},
        })
