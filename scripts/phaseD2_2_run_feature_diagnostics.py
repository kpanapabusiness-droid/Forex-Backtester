"""
Phase D-2.2 — Opportunity Precursor Feature Diagnostics.

Discovers causal features at t that correlate with Zone B/C opportunity.
No ROI, PnL, backtest, WFO, or ML. Config-driven, deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.phaseD2_2_features import (  # noqa: E402
    FEATURE_NAMES,
    ZONE_B,
    ZONE_C,
    compute_bin_edges_from_discovery,
    compute_feature_rankings,
    compute_feature_stats,
    compute_features_for_pair,
)
from core.utils import load_pair_csv  # noqa: E402

REQUIRED_LABEL_COLS = frozenset(
    {"pair", "date", "direction", "zone_a_1r_10", "zone_b_3r_20", "zone_c_6r_40"}
)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _require_config(raw: dict) -> dict:
    """Validate and normalize Phase D-2.2 config. Fail-fast on missing."""
    cfg = dict(raw or {})

    labels_path = cfg.get("labels_path")
    if not labels_path:
        raise ValueError("Phase D2.2 config must set 'labels_path'.")
    cfg["labels_path"] = str(labels_path)

    data_dir = cfg.get("data_dir")
    if not data_dir:
        raise ValueError("Phase D2.2 config must set 'data_dir'.")
    cfg["data_dir"] = str(data_dir)

    outputs_dir = cfg.get("outputs_dir")
    if not outputs_dir:
        raise ValueError("Phase D2.2 config must set 'outputs_dir'.")
    cfg["outputs_dir"] = str(outputs_dir)

    date_range = cfg.get("date_range") or {}
    start = date_range.get("start")
    end = date_range.get("end")
    if not start or not end:
        raise ValueError("Phase D2.2 config must set date_range.start and date_range.end.")
    if str(start) != "2019-01-01" or str(end) != "2026-01-01":
        raise ValueError(
            "Phase D2.2 date window is locked to 2019-01-01 → 2026-01-01; "
            f"got start={start!r}, end={end!r}."
        )
    cfg["date_range"] = {"start": str(start), "end": str(end)}

    split_cfg = cfg.get("split") or {}
    discovery_end = split_cfg.get("discovery_end")
    if not discovery_end:
        raise ValueError("Phase D2.2 config must set split.discovery_end.")
    cfg["split"] = {"discovery_end": str(discovery_end)}

    atr_cfg = cfg.get("atr_period")
    if atr_cfg is None:
        atr_cfg = (cfg.get("atr") or {}).get("period")
    if atr_cfg is None:
        raise ValueError("Phase D2.2 config must set atr_period or atr.period.")
    cfg["atr_period"] = int(atr_cfg)

    bins_cfg = cfg.get("bins") or {}
    cfg["n_bins"] = int(bins_cfg.get("n_bins", 10))
    cfg["min_per_bin"] = int(bins_cfg.get("min_per_bin", 200))

    pairs = cfg.get("pairs") or []
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Phase D2.2 config must provide a non-empty 'pairs' list.")
    cfg["pairs"] = list(pairs)

    return cfg


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _config_hash(cfg: dict) -> str:
    canon = {
        k: v
        for k, v in sorted(cfg.items())
        if k not in ("labels_path", "outputs_dir")
    }
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode("utf-8")).hexdigest()


def _run_from_config(config_path: Path) -> None:
    raw = _load_yaml(config_path)
    cfg = _require_config(raw)

    labels_path = Path(cfg["labels_path"])
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels = pd.read_parquet(labels_path)
    labels["date"] = pd.to_datetime(labels["date"])
    missing = REQUIRED_LABEL_COLS - set(labels.columns)
    if missing:
        raise ValueError(f"Labels missing required columns: {missing}")

    labels["direction"] = pd.Categorical(
        labels["direction"], categories=["long", "short"], ordered=True
    )
    labels = labels.sort_values(["pair", "date", "direction"]).reset_index(drop=True)

    if "dataset_split" not in labels.columns:
        cutoff = pd.Timestamp(cfg["split"]["discovery_end"])
        labels["dataset_split"] = labels["date"].apply(
            lambda d: "discovery" if pd.Timestamp(d) <= cutoff else "validation"
        )

    data_dir = Path(cfg["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    all_features: list[pd.DataFrame] = []
    for pair in cfg["pairs"]:
        pair_cfg = {
            **cfg,
            "pair": pair,
        }
        df_ohlc = load_pair_csv(pair, data_dir=data_dir)
        feat_df = compute_features_for_pair(df_ohlc, pair_cfg)
        if not feat_df.empty:
            all_features.append(feat_df)

    if not all_features:
        raise ValueError("No feature rows computed (no OHLC data in window).")

    features_df = pd.concat(all_features, ignore_index=True)
    features_df = features_df.sort_values(["pair", "date"]).reset_index(drop=True)

    out_dir = Path(cfg["outputs_dir"])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    features_dir = out_dir / "features"
    reports_dir = out_dir / "reports"
    features_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    features_path = features_dir / "features.parquet"
    features_df.to_parquet(features_path, index=False)

    joined = pd.merge(
        labels,
        features_df.drop(columns=["dataset_split"], errors="ignore"),
        on=["pair", "date"],
        how="inner",
    )
    if "dataset_split" not in joined.columns:
        cutoff = pd.Timestamp(cfg["split"]["discovery_end"])
        joined["dataset_split"] = joined["date"].apply(
            lambda d: "discovery" if pd.Timestamp(d) <= cutoff else "validation"
        )
    else:
        joined["dataset_split"] = labels["dataset_split"].iloc[
            joined.index.map(lambda i: labels.index.get_loc(i) if i in labels.index else 0)
        ]
    joined = joined.sort_values(["pair", "date", "direction"]).reset_index(drop=True)

    for col in FEATURE_NAMES:
        if col not in joined.columns:
            joined[col] = np.nan

    if "dataset_split" not in joined.columns:
        cutoff = pd.Timestamp(cfg["split"]["discovery_end"])
        joined["dataset_split"] = joined["date"].apply(
            lambda d: "discovery" if pd.Timestamp(d) <= cutoff else "validation"
        )

    n_bins = cfg["n_bins"]
    min_per_bin = cfg["min_per_bin"]
    bin_edges: dict[str, np.ndarray | None] = {}
    for feat in FEATURE_NAMES:
        if feat in joined.columns:
            edges = compute_bin_edges_from_discovery(
                joined, feat, n_bins=n_bins, min_per_bin=min_per_bin
            )
            bin_edges[feat] = edges

    summary_global_rows: list[dict[str, Any]] = []
    for target in (ZONE_B, ZONE_C):
        for split in ("discovery", "validation", None):
            stats = compute_feature_stats(joined, target, split)
            for _, r in stats.iterrows():
                summary_global_rows.append(r)
    summary_global = pd.DataFrame(summary_global_rows)

    summary_split_rows: list[dict[str, Any]] = []
    for target in (ZONE_B, ZONE_C):
        for split in ("discovery", "validation"):
            stats = compute_feature_stats(joined, target, split)
            for _, r in stats.iterrows():
                summary_split_rows.append(r)
    summary_split = pd.DataFrame(summary_split_rows)

    summary_pair_rows: list[dict[str, Any]] = []
    for target in (ZONE_B, ZONE_C):
        for pair_val in joined["pair"].unique():
            sub = joined[joined["pair"] == pair_val]
            for split in ("discovery", "validation"):
                sub_split = sub[sub["dataset_split"] == split]
                if len(sub_split) < 50:
                    continue
                stats = compute_feature_stats(sub_split, target, None)
                for _, r in stats.iterrows():
                    r = dict(r)
                    r["pair"] = pair_val
                    r["dataset_split"] = split
                    summary_pair_rows.append(r)
    summary_pair = pd.DataFrame(summary_pair_rows) if summary_pair_rows else pd.DataFrame()

    summary_global.to_csv(reports_dir / "feature_summary_global.csv", index=False)
    summary_split.to_csv(reports_dir / "feature_summary_split.csv", index=False)
    if not summary_pair.empty:
        summary_pair.to_csv(reports_dir / "feature_summary_by_pair.csv", index=False)

    rankings_b = compute_feature_rankings(
        joined, bin_edges, ZONE_B, cfg["split"]
    )
    rankings_c = compute_feature_rankings(
        joined, bin_edges, ZONE_C, cfg["split"]
    )
    rankings_b = rankings_b.head(20)
    rankings_c = rankings_c.head(20)

    rankings_combined = pd.concat(
        [
            rankings_b.assign(target_zone=ZONE_B),
            rankings_c.assign(target_zone=ZONE_C),
        ],
        ignore_index=True,
    )
    rankings_combined.to_csv(reports_dir / "feature_rankings.csv", index=False)
    rankings_json = {
        "zone_b": rankings_b.to_dict(orient="records"),
        "zone_c": rankings_c.to_dict(orient="records"),
    }
    with (reports_dir / "feature_rankings.json").open("w", encoding="utf-8") as f:
        json.dump(rankings_json, f, indent=2)

    manifest = {
        "phase": "D2_2_feature_diagnostics",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "labels_path": str(labels_path),
        "labels_hash": _file_sha256(labels_path),
        "config_hash": _config_hash(cfg),
        "features_path": str(features_path),
        "n_features": len(FEATURE_NAMES),
        "n_rows_features": int(len(features_df)),
        "n_rows_joined": int(len(joined)),
        "determinism": {
            "date_range": cfg["date_range"],
            "split": cfg["split"],
            "atr_period": cfg["atr_period"],
            "n_bins": n_bins,
            "min_per_bin": min_per_bin,
        },
    }
    with (out_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Phase D2.2 feature diagnostics completed. Outputs in {out_dir}")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase D-2.2 — Opportunity precursor feature diagnostics.",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to Phase D-2.2 config YAML.",
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
