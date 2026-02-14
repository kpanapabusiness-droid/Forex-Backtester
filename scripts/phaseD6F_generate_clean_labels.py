"""
Phase D-6F: Generate clean opportunity labels (drawdown-constrained).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.utils import load_pair_csv  # noqa: E402
from labels.clean_opportunity import compute_clean_labels_for_pair  # noqa: E402


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _require_config(raw: dict) -> dict:
    cfg = dict(raw or {})
    pairs = cfg.get("pairs") or []
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Config must provide non-empty pairs list.")
    date_range = cfg.get("date_range") or {}
    start = date_range.get("start", "2019-01-01")
    end = date_range.get("end", "2026-01-01")
    atr_cfg = cfg.get("atr") or {}
    atr_period = int(atr_cfg.get("period", 14))
    data_dir = cfg.get("data_dir") or (cfg.get("data") or {}).get("dir")
    if not data_dir:
        raise ValueError("Config must set data_dir or data.dir.")
    out_dir = (cfg.get("outputs") or {}).get("dir") or "results/phaseD/labels"
    return {
        "pairs": list(pairs),
        "date_range": {"start": str(start), "end": str(end)},
        "atr_period": atr_period,
        "data_dir": str(data_dir),
        "outputs_dir": str(out_dir),
    }


def _run_from_config(config_path: Path, out_path: Path | None = None) -> Path:
    raw = _load_yaml(config_path)
    cfg = _require_config(raw)
    data_dir = Path(cfg["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    out_dir = Path(cfg["outputs_dir"])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_path or out_dir / "opportunity_labels_clean.csv"

    all_labels = []
    for pair in cfg["pairs"]:
        df = load_pair_csv(pair, data_dir=data_dir)
        labels = compute_clean_labels_for_pair(
            df,
            pair=pair,
            date_start=cfg["date_range"]["start"],
            date_end=cfg["date_range"]["end"],
            atr_period=cfg["atr_period"],
        )
        if not labels.empty:
            all_labels.append(labels)

    if not all_labels:
        print("Phase D6F: no labels generated.")
        return out_file

    out = pd.concat(all_labels, ignore_index=True)
    out = out.sort_values(["pair", "date"]).reset_index(drop=True)
    out.to_csv(out_file, index=False, float_format="%.8f")
    print(f"Phase D6F clean labels written to: {out_file}")
    return out_file


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase D-6F — Generate clean opportunity labels (drawdown-constrained).",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to config YAML (e.g. configs/phaseD/phaseD6F_clean_labels.yaml).",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="Output path (default: results/phaseD/labels/opportunity_labels_clean.csv).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    out_path = Path(args.out) if args.out else None
    _run_from_config(config_path, out_path)


if __name__ == "__main__":
    main()
