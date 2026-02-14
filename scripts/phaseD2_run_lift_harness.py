"""
Phase D-2 Lift Harness — strategy-agnostic signal vs opportunity truth measurement.

Measures: Does a signal fired at time t increase the probability of Zone A/B/C opportunity?
No ROI, PnL, backtest, or indicator evaluation. Config-driven and deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_LABEL_COLS = frozenset(
    {"pair", "date", "direction", "zone_a_1r_10", "zone_b_3r_20", "zone_c_6r_40"}
)
REQUIRED_SIGNAL_COLS = frozenset({"pair", "date", "direction", "signal", "signal_name"})
DISCOVERY_END_DEFAULT = "2022-12-31"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _require_config(raw: dict) -> dict:
    """Validate and normalize Phase D-2 config. Fail-fast on missing."""
    cfg = dict(raw or {})

    labels_path = cfg.get("labels_path")
    if not labels_path:
        raise ValueError("Phase D2 config must set 'labels_path'.")
    cfg["labels_path"] = str(labels_path)

    outputs_dir = cfg.get("outputs_dir")
    if not outputs_dir:
        raise ValueError("Phase D2 config must set 'outputs_dir'.")
    cfg["outputs_dir"] = str(outputs_dir)

    split_dates = cfg.get("split_dates") or {}
    discovery_end = split_dates.get("discovery_end")
    if not discovery_end:
        raise ValueError("Phase D2 config must set split_dates.discovery_end.")
    cfg["discovery_end"] = str(discovery_end)

    ctrl = cfg.get("control_signals") or {}
    if "always_fire" not in ctrl:
        raise ValueError("Phase D2 config must set control_signals.always_fire.")
    cfg["always_fire"] = bool(ctrl.get("always_fire"))

    rf = ctrl.get("random_fire") or {}
    cfg["random_fire_enabled"] = bool(rf.get("enabled", False))
    if cfg["random_fire_enabled"]:
        if "p" not in rf:
            raise ValueError("control_signals.random_fire must set 'p' when enabled.")
        if "seed" not in rf:
            raise ValueError("control_signals.random_fire must set 'seed' when enabled.")
        cfg["random_fire_p"] = float(rf["p"])
        cfg["random_fire_seed"] = int(rf["seed"])

    ora = ctrl.get("oracle") or {}
    cfg["oracle_enabled"] = bool(ora.get("enabled", False))
    if cfg["oracle_enabled"]:
        zones = ora.get("zones")
        if not zones or not isinstance(zones, list):
            raise ValueError("control_signals.oracle must set 'zones' list when enabled.")
        cfg["oracle_zones"] = [str(z) for z in zones]

    sig = cfg.get("signals") or {}
    ext = sig.get("external") or []
    if not isinstance(ext, list):
        ext = []
    cfg["external_signals"] = list(ext)

    return cfg


def _file_sha256(path: Path) -> str | None:
    """SHA256 of file bytes if exists."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _config_hash(cfg: dict) -> str:
    """Canonical JSON hash of effective config for determinism fingerprint."""
    canon = {k: v for k, v in sorted(cfg.items()) if k != "labels_path"}
    blob = json.dumps(canon, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _dataset_split_for_date(date_val: Any, discovery_end: str) -> str:
    """Discovery vs validation based on date <= discovery_end."""
    ts = pd.Timestamp(date_val)
    cutoff = pd.Timestamp(discovery_end)
    return "discovery" if ts <= cutoff else "validation"


def generate_always_fire(labels: pd.DataFrame) -> pd.DataFrame:
    """Control: signal=1 for all rows."""
    out = labels[["pair", "date", "direction"]].copy()
    out["signal"] = 1
    out["signal_name"] = "always_fire"
    return out


def generate_random_fire(
    labels: pd.DataFrame,
    p: float,
    seed: int,
) -> pd.DataFrame:
    """
    Control: fire randomly at probability p.
    Deterministic: stable hash of (pair, date, direction, seed) -> u in [0,1], fire iff u < p.
    """
    out = labels[["pair", "date", "direction"]].copy()
    key_cols = ["pair", "date", "direction"]
    keys = out[key_cols].astype(str).agg("|".join, axis=1)
    h = hashlib.sha256()
    h.update(f"{seed}".encode("utf-8"))
    u = keys.apply(
        lambda s: int(hashlib.sha256((s + str(seed)).encode("utf-8")).hexdigest(), 16)
        / (2**256)
    )
    out["signal"] = (u < p).astype(int)
    out["signal_name"] = f"random_fire_{int(p*100)}pct"
    return out


def _oracle_signal_name(zone_col: str) -> str:
    """Map zone column to oracle signal name: zone_b_3r_20 -> oracle_zone_b, etc."""
    if zone_col == "zone_b_3r_20":
        return "oracle_zone_b"
    if zone_col == "zone_c_6r_40":
        return "oracle_zone_c"
    if zone_col == "zone_a_1r_10":
        return "oracle_zone_a"
    return f"oracle_{zone_col}"


def generate_oracle_signal(
    labels: pd.DataFrame,
    zone_col: str,
) -> pd.DataFrame:
    """Control (test only): fire exactly when zone is true. Uses labels — never for live."""
    out = labels[["pair", "date", "direction", zone_col]].copy()
    out["signal"] = _to_bool_int(labels[zone_col])
    out["signal_name"] = _oracle_signal_name(zone_col)
    out = out.drop(columns=[zone_col])
    return out


def _to_bool_int(ser: pd.Series) -> pd.Series:
    return (ser.fillna(False).astype(bool)).astype(int)


def _load_external_signal(path: str | Path, name_override: str | None = None) -> pd.DataFrame:
    """Load external signal parquet. Validate schema. Override signal_name if config provides."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"External signal file not found: {p}")
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    missing = REQUIRED_SIGNAL_COLS - set(df.columns)
    if missing:
        raise ValueError(f"External signal {p} missing required columns: {missing}")
    df = df[list(REQUIRED_SIGNAL_COLS)].copy()
    df["signal"] = df["signal"].fillna(0).astype(int)
    if name_override is not None:
        df["signal_name"] = str(name_override)
    df["direction"] = pd.Categorical(
        df["direction"], categories=["long", "short"], ordered=True
    )
    return df.sort_values(["pair", "date", "direction"]).reset_index(drop=True)


def build_control_signals(labels: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Build and concatenate all enabled control signals plus external signals."""
    frames: list[pd.DataFrame] = []

    if cfg.get("always_fire"):
        frames.append(generate_always_fire(labels))

    if cfg.get("random_fire_enabled"):
        frames.append(
            generate_random_fire(
                labels,
                p=cfg["random_fire_p"],
                seed=cfg["random_fire_seed"],
            )
        )

    for zone in cfg.get("oracle_zones", []):
        if zone not in labels.columns:
            raise ValueError(f"Oracle zone {zone!r} not in labels columns.")
        frames.append(generate_oracle_signal(labels, zone))

    for ext_cfg in cfg.get("external_signals", []):
        if not isinstance(ext_cfg, dict):
            continue
        path = ext_cfg.get("path")
        name = ext_cfg.get("name")
        if not path:
            raise ValueError("signals.external entry must set 'path'.")
        ext_df = _load_external_signal(path, name_override=name)
        frames.append(ext_df)

    if not frames:
        raise ValueError("At least one control signal must be enabled.")
    return pd.concat(frames, ignore_index=True)


def join_signals_to_labels(
    labels: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    discovery_end: str,
) -> pd.DataFrame:
    """
    Join signals to labels on (pair, date, direction).
    Both inputs must be sorted by (pair, date, direction).
    """
    labels = labels.copy()
    if "dataset_split" not in labels.columns:
        labels["dataset_split"] = labels["date"].apply(
            lambda d: _dataset_split_for_date(d, discovery_end)
        )

    labels = labels.sort_values(["pair", "date", "direction"]).reset_index(drop=True)
    signals = signals.sort_values(["pair", "date", "direction"]).reset_index(drop=True)

    join_keys = ["pair", "date", "direction"]
    merged = pd.merge(
        labels,
        signals[join_keys + ["signal", "signal_name"]],
        on=join_keys,
        how="inner",
    )
    merged = merged.sort_values(["signal_name", "pair", "date", "direction"]).reset_index(
        drop=True
    )
    return merged


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

    signals = build_control_signals(labels, cfg)
    df_joined = join_signals_to_labels(
        labels, signals, discovery_end=cfg["discovery_end"]
    )

    out_dir = Path(cfg["outputs_dir"])
    joined_dir = out_dir / "joined"
    metrics_dir = out_dir / "metrics"
    joined_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    from analytics.phaseD2_metrics import compute_coverage, compute_metrics

    joined_path = joined_dir / "signals_x_labels.parquet"
    df_joined.to_parquet(joined_path, index=False)

    metrics = compute_metrics(df_joined)
    cov = compute_coverage(df_joined)

    metrics["metrics_global"].to_csv(
        metrics_dir / "metrics_by_signal_global.csv", index=False
    )
    if not metrics["metrics_split"].empty:
        metrics["metrics_split"].to_csv(
            metrics_dir / "metrics_by_signal_split.csv", index=False
        )
    if not metrics["metrics_pair"].empty:
        metrics["metrics_pair"].to_csv(
            metrics_dir / "metrics_by_signal_pair.csv", index=False
        )
    cov.to_csv(metrics_dir / "coverage_by_signal.csv", index=False)

    manifest = {
        "phase": "D2_lift_harness",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "labels_path": str(labels_path),
        "labels_hash": _file_sha256(labels_path),
        "config_hash": _config_hash(cfg),
        "determinism": {
            "discovery_end": cfg["discovery_end"],
            "control_signals": {
                "always_fire": cfg.get("always_fire"),
                "random_fire": {
                    "enabled": cfg.get("random_fire_enabled"),
                    "p": cfg.get("random_fire_p"),
                    "seed": cfg.get("random_fire_seed"),
                },
                "oracle": {
                    "enabled": cfg.get("oracle_enabled"),
                    "zones": cfg.get("oracle_zones", []),
                },
            },
        },
    }
    manifest_path = out_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Phase D2 lift harness completed. Outputs in {out_dir}")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase D-2 — Predictive lift harness (signals vs opportunity truth)."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to Phase D-2 config YAML.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    _run_from_config(config_path)


if __name__ == "__main__":
    main()
