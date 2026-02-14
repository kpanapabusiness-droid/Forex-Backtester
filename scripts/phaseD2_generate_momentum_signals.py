"""
Phase D-2.1 — 3-Bar Momentum Signal Generator.

Causal price-pattern signal: long if close[t]>close[t-1]>close[t-2];
short if close[t]<close[t-1]<close[t-2]. Outputs parquet+CSV for D-2 harness.
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

from core.utils import load_pair_csv, slice_df_by_dates  # noqa: E402

SIGNAL_SCHEMA = ("pair", "date", "direction", "signal", "signal_name")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _require_phaseD2_1_config(raw: dict) -> dict:
    """
    Validate and normalize Phase D-2.1 config.
    Enforces: timeframe D1/D, date window 2019-01-01 → 2026-01-01,
    non-empty pairs, data.dir, outputs.dir under results/phaseD2.
    """
    cfg = dict(raw or {})

    pairs = cfg.get("pairs") or []
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Phase D2.1 config must provide a non-empty 'pairs' list.")

    timeframe = str(cfg.get("timeframe") or "").upper()
    if timeframe not in {"D1", "D"}:
        raise ValueError(
            f"Phase D2.1 timeframe must be 'D1' or 'D'; got {timeframe!r}"
        )

    date_range = cfg.get("date_range") or {}
    start = date_range.get("start")
    end = date_range.get("end")
    if not start or not end:
        raise ValueError(
            "Phase D2.1 config must set date_range.start and date_range.end."
        )
    if str(start) != "2019-01-01" or str(end) != "2026-01-01":
        raise ValueError(
            "Phase D2.1 date window is locked to 2019-01-01 → 2026-01-01; "
            f"got start={start!r}, end={end!r}."
        )

    data_dir = (cfg.get("data") or {}).get("dir")
    if not data_dir:
        raise ValueError("Phase D2.1 config must set data.dir.")

    outputs = cfg.get("outputs") or {}
    out_dir = outputs.get("dir")
    if not out_dir:
        raise ValueError("Phase D2.1 config must set outputs.dir.")
    out_dir_str = str(out_dir)
    if "phaseD2" not in out_dir_str:
        raise ValueError("Phase D2.1 outputs.dir must live under results/phaseD2/*.")

    signal_name = cfg.get("signal_name")
    if not signal_name:
        raise ValueError("Phase D2.1 config must set signal_name.")

    return {
        "pairs": list(pairs),
        "timeframe": timeframe,
        "date_range": {"start": str(start), "end": str(end)},
        "data_dir": str(data_dir),
        "outputs_dir": out_dir_str,
        "signal_name": str(signal_name),
    }


def _momentum_3bar_signals(df: pd.DataFrame, pair: str, signal_name: str) -> pd.DataFrame:
    """
    Compute 3-bar momentum signals for one pair.
    Long: close[t]>close[t-1]>close[t-2]; Short: close[t]<close[t-1]<close[t-2].
    Warmup: first 2 bars emit signal=0 for both directions.
    """
    required_cols = {"date", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} for pair {pair}.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype="float64")
    dates = df["date"].tolist()
    n = len(df)

    rows: list[dict[str, Any]] = []
    for t in range(n):
        date_t = pd.to_datetime(dates[t])
        long_sig = 0
        short_sig = 0
        if t >= 2:
            c0, c1, c2 = close[t - 2], close[t - 1], close[t]
            if np.isfinite(c0) and np.isfinite(c1) and np.isfinite(c2):
                if c2 > c1 > c0:
                    long_sig = 1
                elif c2 < c1 < c0:
                    short_sig = 1

        rows.append({
            "pair": pair,
            "date": date_t,
            "direction": "long",
            "signal": long_sig,
            "signal_name": signal_name,
        })
        rows.append({
            "pair": pair,
            "date": date_t,
            "direction": "short",
            "signal": short_sig,
            "signal_name": signal_name,
        })

    out = pd.DataFrame(rows)
    out["direction"] = pd.Categorical(
        out["direction"], categories=["long", "short"], ordered=True
    )
    out = out[list(SIGNAL_SCHEMA)].sort_values(
        ["pair", "date", "direction"]
    ).reset_index(drop=True)
    return out


def _run_from_config(config_path: Path) -> None:
    raw = _load_yaml(config_path)
    cfg = _require_phaseD2_1_config(raw)

    data_dir = Path(cfg["data_dir"])
    outputs_dir = Path(cfg["outputs_dir"])
    date_start = cfg["date_range"]["start"]
    date_end = cfg["date_range"]["end"]
    signal_name = cfg["signal_name"]

    all_signals: list[pd.DataFrame] = []
    for pair in cfg["pairs"]:
        df = load_pair_csv(pair, data_dir=data_dir)
        df_slice, _ = slice_df_by_dates(df, date_start, date_end, inclusive="both")
        if df_slice.empty:
            raise ValueError(
                f"Pair {pair} has no data in date range {date_start} → {date_end}."
            )
        signals = _momentum_3bar_signals(df_slice, pair=pair, signal_name=signal_name)
        all_signals.append(signals)

    out = pd.concat(all_signals, ignore_index=True)
    out["direction"] = pd.Categorical(
        out["direction"], categories=["long", "short"], ordered=True
    )
    out = out[list(SIGNAL_SCHEMA)].sort_values(
        ["pair", "date", "direction"]
    ).reset_index(drop=True)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = outputs_dir / f"{signal_name}_signals.parquet"
    csv_path = outputs_dir / f"{signal_name}_signals.csv"
    out.to_parquet(parquet_path, index=False)
    out.to_csv(csv_path, index=False)

    print(f"Phase D2.1 momentum signals written to:\n  {parquet_path}\n  {csv_path}")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase D-2.1 — Generate 3-bar momentum signals for lift harness."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to Phase D-2.1 config YAML.",
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
