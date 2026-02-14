"""
Phase D-2.5 — Sequence proto-signals: compression at t, trigger within K bars.

Fire at trigger bar when compression occurred within last K bars (including current).
No ROI, PnL, backtest, or ML. Config-driven, deterministic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

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
    }
)
DIRECTIONS = ("long", "short")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _require_config(raw: dict) -> dict:
    """Validate and normalize Phase D-2.5 config. Fail-fast on missing."""
    cfg = dict(raw or {})

    features_path = cfg.get("features_path")
    if not features_path:
        raise ValueError("Phase D2.5 config must set 'features_path'.")
    cfg["features_path"] = str(features_path)

    outputs_dir = cfg.get("outputs_dir")
    if not outputs_dir:
        raise ValueError("Phase D2.5 config must set 'outputs_dir'.")
    cfg["outputs_dir"] = str(outputs_dir)

    split_cfg = cfg.get("split") or {}
    discovery_end = split_cfg.get("discovery_end")
    if not discovery_end:
        raise ValueError("Phase D2.5 config must set split.discovery_end.")
    cfg["discovery_end"] = str(discovery_end)

    bins_cfg = cfg.get("bins") or {}
    cfg["n_bins"] = int(bins_cfg.get("n_bins", 10))
    cfg["min_per_bin"] = int(bins_cfg.get("min_per_bin", 200))

    comp_cfg = cfg.get("compression") or {}
    comp_feature = comp_cfg.get("feature")
    if not comp_feature:
        raise ValueError("Phase D2.5 config must set compression.feature.")
    cfg["compression_feature"] = str(comp_feature)
    cfg["compression_bin"] = int(comp_cfg.get("bin", 0))

    seq_cfg = cfg.get("sequence") or {}
    k_list = seq_cfg.get("K_list")
    if not k_list or not isinstance(k_list, list):
        raise ValueError("Phase D2.5 config must set sequence.K_list (list of int).")
    cfg["K_list"] = [int(k) for k in k_list]

    ign_cfg = cfg.get("ignition") or {}
    cfg["tr_atr_pct"] = float(ign_cfg.get("tr_atr_ratio_percentile", 0.90))

    press_cfg = cfg.get("pressure") or {}
    thresh = press_cfg.get("thresholds") or {}
    cfg["pressure_up"] = float(thresh.get("up", 0.90))
    cfg["pressure_dn"] = float(thresh.get("dn", 0.10))

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


def _comp_recent_K_per_pair(df: pd.DataFrame, K: int) -> pd.Series:
    """
    For each pair, rolling any(compression) over last K+1 bars (t, t-1, ..., t-K).
    comp_recent_K[t] = any(is_compression[t-i] for i=0..K).
    """
    window = K + 1
    rolled = df.groupby("pair", sort=False)["_comp"].apply(
        lambda s: s.rolling(window=window, min_periods=1).max()
    )
    return (rolled >= 1).fillna(False)


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
    df = df.sort_values(["pair", "date"]).reset_index(drop=True)
    df = _ensure_dataset_split(df, cfg["discovery_end"])

    missing = REQUIRED_FEATURE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Features missing required columns: {missing}")

    n_bins = cfg["n_bins"]
    min_per_bin = cfg["min_per_bin"]
    comp_feature = cfg["compression_feature"]
    comp_bin = cfg["compression_bin"]

    edges = compute_bin_edges_from_discovery(
        df, comp_feature, n_bins=n_bins, min_per_bin=min_per_bin
    )
    if edges is None:
        raise ValueError(
            f"Insufficient discovery data for {comp_feature} bin edges."
        )

    bin_series = apply_bin_edges(df[comp_feature], edges)
    is_compression = (bin_series == comp_bin).fillna(False)
    df["_comp"] = is_compression.astype(int)

    tratr_p90 = float(
        np.quantile(
            df.loc[df["dataset_split"] == "discovery", "tr_atr_ratio"].dropna(),
            cfg["tr_atr_pct"],
        )
    )

    pressure_up = cfg["pressure_up"]
    pressure_dn = cfg["pressure_dn"]

    frames: list[pd.DataFrame] = []

    for K in cfg["K_list"]:
        comp_recent = _comp_recent_K_per_pair(df, K)

        df["_comp_recent"] = comp_recent.values
        df["_breakout_up"] = (df["breakout_up_20"].fillna(0) == 1).to_numpy()
        df["_breakout_dn"] = (df["breakout_dn_20"].fillna(0) == 1).to_numpy()
        df["_tratr_ok"] = (
            np.isfinite(df["tr_atr_ratio"]) & (df["tr_atr_ratio"].to_numpy() >= tratr_p90)
        ).to_numpy()
        pos = df["pos_in_range_20"].to_numpy()
        df["_pressure_up"] = np.isfinite(pos) & (pos >= pressure_up)
        df["_pressure_dn"] = np.isfinite(pos) & (pos <= pressure_dn)

        comp_ok = df["_comp_recent"].to_numpy()

        sig_names = [
            f"seq_comp{K}_breakout_up",
            f"seq_comp{K}_breakout_dn",
            f"seq_comp{K}_tratr90",
            f"seq_comp{K}_pressure_up",
            f"seq_comp{K}_pressure_dn",
        ]

        for sig_name in sig_names:
            if "breakout_up" in sig_name:
                cond = comp_ok & df["_breakout_up"].to_numpy()
                long_sig = cond
                short_sig = np.zeros(len(df), dtype=bool)
            elif "breakout_dn" in sig_name:
                cond = comp_ok & df["_breakout_dn"].to_numpy()
                long_sig = np.zeros(len(df), dtype=bool)
                short_sig = cond
            elif "tratr90" in sig_name:
                cond = comp_ok & df["_tratr_ok"].to_numpy()
                long_sig = cond
                short_sig = cond
            elif "pressure_up" in sig_name:
                cond = comp_ok & df["_pressure_up"].to_numpy()
                long_sig = cond
                short_sig = np.zeros(len(df), dtype=bool)
            elif "pressure_dn" in sig_name:
                cond = comp_ok & df["_pressure_dn"].to_numpy()
                long_sig = np.zeros(len(df), dtype=bool)
                short_sig = cond
            else:
                long_sig = np.zeros(len(df), dtype=bool)
                short_sig = np.zeros(len(df), dtype=bool)

            spine = _build_spine(df)
            merged = spine.merge(
                df[["pair", "date"]].assign(
                    _long=long_sig.astype(int), _short=short_sig.astype(int)
                ),
                on=["pair", "date"],
                how="left",
            )
            merged["signal"] = np.where(
                merged["direction"] == "long",
                merged["_long"].fillna(0),
                merged["_short"].fillna(0),
            ).astype(int)
            merged["signal_name"] = sig_name
            merged = merged[["pair", "date", "direction", "signal", "signal_name"]]
            frames.append(merged)

    combined = pd.concat(frames, ignore_index=True)
    combined["direction"] = pd.Categorical(
        combined["direction"], categories=list(DIRECTIONS), ordered=True
    )
    combined = combined.sort_values(
        ["signal_name", "pair", "date", "direction"]
    ).reset_index(drop=True)

    out_dir = Path(cfg["outputs_dir"])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "sequence_proto_signals.parquet"
    csv_path = out_dir / "sequence_proto_signals.csv"
    combined.to_parquet(parquet_path, index=False)
    combined.to_csv(csv_path, index=False)

    print(
        f"Phase D2.5 sequence proto-signals completed. Outputs: {parquet_path}, {csv_path}"
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase D-2.5 — Sequence proto-signals (compression → trigger within K bars).",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to Phase D-2.5 sequence config YAML.",
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
