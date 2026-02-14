"""
Phase D-2.3 — Proto-signals from features (compression + ignition + breakout).

Generates feature-based proto-signals from D-2.2 discoveries.
No ROI, PnL, backtest, WFO, or ML. Config-driven, deterministic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.phaseD2_2_features import (  # noqa: E402
    apply_bin_edges,
    compute_bin_edges_from_discovery,
)

REQUIRED_FEATURE_COLS = frozenset(
    {
        "pair",
        "date",
        "atrp_14",
        "true_range",
        "breakout_up_20",
        "breakout_dn_20",
        "pos_in_range_20",
        "tr_atr_ratio",
        "mom_slope_5",
    }
)
DIRECTIONS = ("long", "short")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _require_config(raw: dict) -> dict:
    """Validate and normalize Phase D-2.3 config. Fail-fast on missing."""
    cfg = dict(raw or {})

    features_path = cfg.get("features_path")
    if not features_path:
        raise ValueError("Phase D2.3 config must set 'features_path'.")
    cfg["features_path"] = str(features_path)

    outputs_dir = cfg.get("outputs_dir")
    if not outputs_dir:
        raise ValueError("Phase D2.3 config must set 'outputs_dir'.")
    cfg["outputs_dir"] = str(outputs_dir)

    split_cfg = cfg.get("split") or {}
    discovery_end = split_cfg.get("discovery_end")
    if not discovery_end:
        raise ValueError("Phase D2.3 config must set split.discovery_end.")
    cfg["discovery_end"] = str(discovery_end)

    bins_cfg = cfg.get("bins") or {}
    cfg["n_bins"] = int(bins_cfg.get("n_bins", 10))
    cfg["min_per_bin"] = int(bins_cfg.get("min_per_bin", 200))

    signals = cfg.get("signals")
    if not signals or not isinstance(signals, list):
        raise ValueError("Phase D2.3 config must provide a non-empty 'signals' list.")
    cfg["signals"] = list(signals)

    return cfg


def _ensure_dataset_split(df: pd.DataFrame, discovery_end: str) -> pd.DataFrame:
    """Add dataset_split if missing."""
    if "dataset_split" in df.columns:
        return df
    cutoff = pd.Timestamp(discovery_end)
    df = df.copy()
    df["dataset_split"] = df["date"].apply(
        lambda d: "discovery" if pd.Timestamp(d) <= cutoff else "validation"
    )
    return df


def _build_spine(df: pd.DataFrame) -> pd.DataFrame:
    """Expand (pair, date) to (pair, date, direction) with two rows per date."""
    rows = []
    for _, row in df.iterrows():
        for direction in DIRECTIONS:
            rows.append({
                "pair": row["pair"],
                "date": row["date"],
                "direction": direction,
            })
    return pd.DataFrame(rows)


def _generate_proto_comp_atrp_low(
    spine: pd.DataFrame,
    features: pd.DataFrame,
    atrp_low: pd.Series,
) -> pd.DataFrame:
    """Both directions fire when atrp_14 in lowest decile."""
    out = spine.merge(
        features[["pair", "date"]].assign(atrp_low=atrp_low),
        on=["pair", "date"],
        how="left",
    )
    out["signal"] = out["atrp_low"].fillna(False).astype(int)
    out["signal_name"] = "proto_comp_atrp_low"
    return out[["pair", "date", "direction", "signal", "signal_name"]]


def _generate_proto_ignite_tr_high(
    spine: pd.DataFrame,
    features: pd.DataFrame,
    tr_high: pd.Series,
) -> pd.DataFrame:
    """Both directions fire when true_range in top decile."""
    out = spine.merge(
        features[["pair", "date"]].assign(tr_high=tr_high),
        on=["pair", "date"],
        how="left",
    )
    out["signal"] = out["tr_high"].fillna(False).astype(int)
    out["signal_name"] = "proto_ignite_tr_high"
    return out[["pair", "date", "direction", "signal", "signal_name"]]


def _generate_proto_comp_low_atrp_breakout_up(
    spine: pd.DataFrame,
    features: pd.DataFrame,
    atrp_low: pd.Series,
) -> pd.DataFrame:
    """Long fires when atrp low AND breakout_up_20; short signal=0."""
    out = spine.merge(
        features[["pair", "date", "breakout_up_20"]].assign(atrp_low=atrp_low),
        on=["pair", "date"],
        how="left",
    )
    breakout = (out["breakout_up_20"].fillna(0) == 1).to_numpy()
    atrp_ok = out["atrp_low"].fillna(False).to_numpy()
    long_mask = (out["direction"] == "long").to_numpy()
    out["signal"] = np.where(
        long_mask & atrp_ok & breakout, 1, 0
    ).astype(int)
    out["signal_name"] = "proto_comp_low_atrp_breakout_up"
    return out[["pair", "date", "direction", "signal", "signal_name"]]


def _generate_proto_comp_low_atrp_breakout_dn(
    spine: pd.DataFrame,
    features: pd.DataFrame,
    atrp_low: pd.Series,
) -> pd.DataFrame:
    """Short fires when atrp low AND breakout_dn_20; long signal=0."""
    out = spine.merge(
        features[["pair", "date", "breakout_dn_20"]].assign(atrp_low=atrp_low),
        on=["pair", "date"],
        how="left",
    )
    breakout = (out["breakout_dn_20"].fillna(0) == 1).to_numpy()
    atrp_ok = out["atrp_low"].fillna(False).to_numpy()
    short_mask = (out["direction"] == "short").to_numpy()
    out["signal"] = np.where(
        short_mask & atrp_ok & breakout, 1, 0
    ).astype(int)
    out["signal_name"] = "proto_comp_low_atrp_breakout_dn"
    return out[["pair", "date", "direction", "signal", "signal_name"]]


def _generate_proto_comp_low_atrp_ignite_high_tr(
    spine: pd.DataFrame,
    features: pd.DataFrame,
    atrp_low: pd.Series,
    tr_high: pd.Series,
) -> pd.DataFrame:
    """Both directions fire when atrp low AND true_range high."""
    out = spine.merge(
        features[["pair", "date"]].assign(
            atrp_low=atrp_low, tr_high=tr_high
        ),
        on=["pair", "date"],
        how="left",
    )
    out["signal"] = (
        out["atrp_low"].fillna(False) & out["tr_high"].fillna(False)
    ).astype(int)
    out["signal_name"] = "proto_comp_low_atrp_ignite_high_tr"
    return out[["pair", "date", "direction", "signal", "signal_name"]]


def _discovery_percentile_threshold(
    df: pd.DataFrame, feature: str, q: float = 0.70
) -> float:
    """Compute percentile threshold from discovery split only. Freeze for validation."""
    disc = df[df["dataset_split"] == "discovery"]
    vals = disc[feature].dropna()
    if len(vals) < 10:
        return np.nan
    return float(np.quantile(vals, q))


def _generate_proto_comp_atrp_low_pos_pressure_up(
    spine: pd.DataFrame,
    features: pd.DataFrame,
    atrp_low: pd.Series,
) -> pd.DataFrame:
    """Long when atrp_low & pos_in_range_20 >= 0.85; short when atrp_low & pos_in_range_20 <= 0.15."""
    out = spine.merge(
        features[["pair", "date", "pos_in_range_20"]].assign(atrp_low=atrp_low),
        on=["pair", "date"],
        how="left",
    )
    pos = out["pos_in_range_20"]
    atrp_ok = out["atrp_low"].fillna(False).to_numpy()
    long_mask = (out["direction"] == "long").to_numpy()
    long_cond = np.isfinite(pos) & (pos.to_numpy() >= 0.85) & atrp_ok & long_mask
    short_cond = np.isfinite(pos) & (pos.to_numpy() <= 0.15) & atrp_ok & ~long_mask
    out["signal"] = np.where(long_cond | short_cond, 1, 0).astype(int)
    out["signal_name"] = "proto_comp_atrp_low_pos_pressure_up"
    return out[["pair", "date", "direction", "signal", "signal_name"]]


def _generate_proto_comp_atrp_low_tr_mid_high(
    spine: pd.DataFrame,
    features: pd.DataFrame,
    atrp_low: pd.Series,
    tr_70th: float,
) -> pd.DataFrame:
    """Both directions fire when atrp_low & true_range >= 70th percentile (top 30%)."""
    out = spine.merge(
        features[["pair", "date", "true_range"]].assign(atrp_low=atrp_low),
        on=["pair", "date"],
        how="left",
    )
    tr = out["true_range"].to_numpy()
    atrp_ok = out["atrp_low"].fillna(False).to_numpy()
    tr_ok = (
        np.isfinite(tr) & (tr >= tr_70th)
        if np.isfinite(tr_70th)
        else np.zeros(len(out), dtype=bool)
    )
    out["signal"] = (atrp_ok & tr_ok).astype(int)
    out["signal_name"] = "proto_comp_atrp_low_tr_mid_high"
    return out[["pair", "date", "direction", "signal", "signal_name"]]


def _generate_proto_comp_atrp_low_tr_atr_ratio_high(
    spine: pd.DataFrame,
    features: pd.DataFrame,
    atrp_low: pd.Series,
    tr_atr_70th: float,
) -> pd.DataFrame:
    """Both directions fire when atrp_low & tr_atr_ratio >= 70th percentile (top 30%)."""
    out = spine.merge(
        features[["pair", "date", "tr_atr_ratio"]].assign(atrp_low=atrp_low),
        on=["pair", "date"],
        how="left",
    )
    ratio = out["tr_atr_ratio"].to_numpy()
    atrp_ok = out["atrp_low"].fillna(False).to_numpy()
    ratio_ok = (
        np.isfinite(ratio) & (ratio >= tr_atr_70th)
        if np.isfinite(tr_atr_70th)
        else np.zeros(len(out), dtype=bool)
    )
    out["signal"] = (atrp_ok & ratio_ok).astype(int)
    out["signal_name"] = "proto_comp_atrp_low_tr_atr_ratio_high"
    return out[["pair", "date", "direction", "signal", "signal_name"]]


def _generate_proto_comp_atrp_low_slope_alignment(
    spine: pd.DataFrame,
    features: pd.DataFrame,
    atrp_low: pd.Series,
) -> pd.DataFrame:
    """Long when atrp_low & mom_slope_5 > 0; short when atrp_low & mom_slope_5 < 0."""
    out = spine.merge(
        features[["pair", "date", "mom_slope_5"]].assign(atrp_low=atrp_low),
        on=["pair", "date"],
        how="left",
    )
    slope = out["mom_slope_5"]
    atrp_ok = out["atrp_low"].fillna(False).to_numpy()
    long_mask = (out["direction"] == "long").to_numpy()
    long_cond = np.isfinite(slope) & (slope.to_numpy() > 0) & atrp_ok & long_mask
    short_cond = np.isfinite(slope) & (slope.to_numpy() < 0) & atrp_ok & ~long_mask
    out["signal"] = np.where(long_cond | short_cond, 1, 0).astype(int)
    out["signal_name"] = "proto_comp_atrp_low_slope_alignment"
    return out[["pair", "date", "direction", "signal", "signal_name"]]


def _run_from_config(config_path: Path) -> None:
    raw = _load_yaml(config_path)
    cfg = _require_config(raw)

    features_path = Path(cfg["features_path"])
    if not features_path.is_absolute():
        features_path = ROOT / features_path
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"])
    df = _ensure_dataset_split(df, cfg["discovery_end"])

    missing = REQUIRED_FEATURE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Features missing required columns: {missing}")

    n_bins = cfg["n_bins"]
    min_per_bin = cfg["min_per_bin"]

    edges_atrp = compute_bin_edges_from_discovery(
        df, "atrp_14", n_bins=n_bins, min_per_bin=min_per_bin
    )
    edges_tr = compute_bin_edges_from_discovery(
        df, "true_range", n_bins=n_bins, min_per_bin=min_per_bin
    )

    atrp_bin = apply_bin_edges(df["atrp_14"], edges_atrp)
    tr_bin = apply_bin_edges(df["true_range"], edges_tr)

    atrp_low = (atrp_bin == 0).fillna(False)
    tr_high = (tr_bin == (n_bins - 1)).fillna(False)

    tr_70th = _discovery_percentile_threshold(df, "true_range", q=0.70)
    tr_atr_70th = _discovery_percentile_threshold(df, "tr_atr_ratio", q=0.70)

    spine = _build_spine(df)

    rule_handlers: dict[str, Any] = {
        "compression_atrp_low": lambda: _generate_proto_comp_atrp_low(spine, df, atrp_low),
        "ignition_tr_high": lambda: _generate_proto_ignite_tr_high(spine, df, tr_high),
        "compression_atrp_low_breakout_up": lambda: _generate_proto_comp_low_atrp_breakout_up(
            spine, df, atrp_low
        ),
        "compression_atrp_low_breakout_dn": lambda: _generate_proto_comp_low_atrp_breakout_dn(
            spine, df, atrp_low
        ),
        "compression_atrp_low_ignition_tr_high": lambda: _generate_proto_comp_low_atrp_ignite_high_tr(
            spine, df, atrp_low, tr_high
        ),
        "compression_atrp_low_pos_pressure_up": lambda: _generate_proto_comp_atrp_low_pos_pressure_up(
            spine, df, atrp_low
        ),
        "compression_atrp_low_tr_mid_high": lambda: _generate_proto_comp_atrp_low_tr_mid_high(
            spine, df, atrp_low, tr_70th
        ),
        "compression_atrp_low_tr_atr_ratio_high": lambda: _generate_proto_comp_atrp_low_tr_atr_ratio_high(
            spine, df, atrp_low, tr_atr_70th
        ),
        "compression_atrp_low_slope_alignment": lambda: _generate_proto_comp_atrp_low_slope_alignment(
            spine, df, atrp_low
        ),
    }

    frames: list[pd.DataFrame] = []
    for sig_cfg in cfg["signals"]:
        if not isinstance(sig_cfg, dict):
            continue
        name = sig_cfg.get("name")
        rule = sig_cfg.get("rule")
        if not name or not rule:
            raise ValueError(f"Signal config must set 'name' and 'rule'; got {sig_cfg}")
        if rule not in rule_handlers:
            raise ValueError(f"Unknown rule {rule!r}; known: {list(rule_handlers)}")
        frame = rule_handlers[rule]()
        frame["signal_name"] = name
        frames.append(frame)

    if not frames:
        raise ValueError("No signals generated.")

    combined = pd.concat(frames, ignore_index=True)
    combined["direction"] = pd.Categorical(
        combined["direction"], categories=list(DIRECTIONS), ordered=True
    )
    combined = combined.sort_values(
        ["pair", "date", "direction", "signal_name"]
    ).reset_index(drop=True)
    combined = combined[["pair", "date", "direction", "signal", "signal_name"]]

    out_dir = Path(cfg["outputs_dir"])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "proto_signals.parquet"
    csv_path = out_dir / "proto_signals.csv"
    combined.to_parquet(parquet_path, index=False)
    combined.to_csv(csv_path, index=False)

    print(f"Phase D2.3 proto-signals completed. Outputs: {parquet_path}, {csv_path}")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase D-2.3 — Proto-signals from features.",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to Phase D-2.3 config YAML.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    _run_from_config(config_path)


if __name__ == "__main__":
    main()
