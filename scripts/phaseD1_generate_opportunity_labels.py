from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.utils import (  # noqa: E402
    calculate_atr,
    load_pair_csv,
    slice_df_by_dates,
)

LABEL_VERSION = "phaseD-1_v1"
MAX_LOOKAHEAD = 40

OPPORTUNITY_COLUMNS: List[str] = [
    # Identity
    "pair",
    "date",
    "direction",
    "dataset_split",
    # Reference & measurement
    "ref_date",
    "ref_open",
    "atr_t",
    "sl_dist",
    "r_dist",
    "tp1_dist",
    # Forward window bookkeeping
    "horizon_10",
    "horizon_20",
    "horizon_40",
    # MFE (R units)
    "mfe_10_r",
    "mfe_20_r",
    "mfe_40_r",
    # Zone flags
    "zone_a_1r_10",
    "zone_b_3r_20",
    "zone_c_6r_40",
    # Timing
    "t1",
    "t3",
    "t6",
    # Provenance / determinism
    "label_version",
    "data_fingerprint",
    "engine_atr_fingerprint",
]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _require_phaseD1_config(raw: dict) -> dict:
    """
    Validate and normalize Phase D-1 config.

    Rules:
      - Config-driven only (no silent defaults for core knobs).
      - Timeframe must be D1 (allow legacy 'D' as alias).
      - Date window locked to 2019-01-01 → 2026-01-01.
      - Pairs list must be explicit and non-empty.
      - ATR period must be provided explicitly.
      - Data directory must be provided via data_dir or data.dir.
      - Outputs.dir must be explicit and live under results/phaseD.
    """
    cfg = dict(raw or {})

    pairs = cfg.get("pairs") or []
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Phase D1 config must provide a non-empty 'pairs' list.")

    timeframe = str(cfg.get("timeframe") or "").upper()
    if timeframe not in {"D1", "D"}:
        raise ValueError(f"Phase D1 timeframe must be 'D1' (or legacy 'D'); got {timeframe!r}")

    date_range = cfg.get("date_range") or {}
    start = date_range.get("start")
    end = date_range.get("end")
    if not start or not end:
        raise ValueError("Phase D1 config must set date_range.start and date_range.end explicitly.")
    if str(start) != "2019-01-01" or str(end) != "2026-01-01":
        raise ValueError(
            "Phase D1 date window is locked to 2019-01-01 → 2026-01-01; "
            f"got start={start!r}, end={end!r}."
        )

    atr_cfg = cfg.get("atr") or {}
    if "period" not in atr_cfg:
        raise ValueError("Phase D1 config must set atr.period explicitly.")
    atr_period = int(atr_cfg["period"])
    if atr_period <= 0:
        raise ValueError("Phase D1 atr.period must be a positive integer.")

    data_dir = cfg.get("data_dir") or (cfg.get("data") or {}).get("dir")
    if not data_dir:
        raise ValueError("Phase D1 config must set either 'data_dir' or 'data.dir'.")

    outputs = cfg.get("outputs") or {}
    out_dir = outputs.get("dir")
    if not out_dir:
        raise ValueError("Phase D1 config must set outputs.dir (e.g. 'results/phaseD/labels').")
    out_dir_str = str(out_dir)
    if "phaseD" not in out_dir_str:
        raise ValueError("Phase D1 outputs.dir must live under results/phaseD/*.")

    return {
        "pairs": list(pairs),
        "timeframe": timeframe,
        "date_range": {"start": str(start), "end": str(end)},
        "atr_period": atr_period,
        "data_dir": str(data_dir),
        "outputs_dir": out_dir_str,
    }


def _compute_engine_atr_fingerprint() -> str:
    """Fingerprint engine ATR implementation + VERSION for provenance."""
    from core import utils  # local import to avoid cycles

    src = inspect.getsource(utils.calculate_atr)
    version_path = ROOT / "VERSION"
    version = version_path.read_text(encoding="utf-8").strip() if version_path.exists() else ""

    h = hashlib.sha256()
    h.update(src.encode("utf-8"))
    h.update(version.encode("utf-8"))
    return h.hexdigest()


ENGINE_ATR_FINGERPRINT = _compute_engine_atr_fingerprint()


def _compute_data_fingerprint(
    df: pd.DataFrame,
    *,
    pair: str,
    date_start: str,
    date_end: str,
) -> str:
    """Deterministic fingerprint of the sliced OHLCV data for one pair."""
    h = hashlib.sha256()

    meta = {
        "pair": pair,
        "rows": int(len(df)),
        "date_min": str(df["date"].min()) if not df.empty else None,
        "date_max": str(df["date"].max()) if not df.empty else None,
        "cfg_start": date_start,
        "cfg_end": date_end,
    }
    h.update(json.dumps(meta, sort_keys=True).encode("utf-8"))

    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            arr = df[col].to_numpy(dtype="float64", copy=False)
            h.update(arr.tobytes())

    return h.hexdigest()


def _dataset_split_for_date(ts: pd.Timestamp) -> str:
    """Discovery vs validation split based on date t."""
    cutoff = pd.Timestamp("2022-12-31")
    return "discovery" if ts <= cutoff else "validation"


def generate_labels_for_pair(df: pd.DataFrame, pair: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Core Phase D-1 labeling logic for a single pair.

    Contract:
      - Input df must contain at least: date, open, high, low, close.
      - Config must already be normalized via _require_phaseD1_config (or an equivalent).
      - Returns one row per (date t, direction) where t has a valid t+1.
      - Does NOT touch any indicator or strategy logic.
    """
    required_cols = {"date", "open", "high", "low", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns {missing} for pair {pair}.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Enforce locked date window on the data itself.
    date_start = cfg["date_range"]["start"]
    date_end = cfg["date_range"]["end"]
    df_slice, _meta = slice_df_by_dates(df, date_start, date_end, inclusive="both")
    if df_slice.empty or len(df_slice) < 2:
        # No tradable bars (need at least t and t+1)
        return pd.DataFrame(columns=OPPORTUNITY_COLUMNS)

    # Compute ATR(t) using engine helper; ATR evaluated at bar t only.
    atr_period = cfg.get("atr_period") or (cfg.get("atr") or {}).get("period")
    if atr_period is None:
        raise ValueError("Phase D1 config must provide ATR period (atr_period or atr.period).")
    atr_period = int(atr_period)
    df_slice = calculate_atr(df_slice, period=atr_period)

    dates = df_slice["date"].to_list()
    opens = pd.to_numeric(df_slice["open"], errors="coerce").to_numpy(dtype="float64", copy=False)
    highs = pd.to_numeric(df_slice["high"], errors="coerce").to_numpy(dtype="float64", copy=False)
    lows = pd.to_numeric(df_slice["low"], errors="coerce").to_numpy(dtype="float64", copy=False)
    atrs = pd.to_numeric(df_slice["atr"], errors="coerce").to_numpy(dtype="float64", copy=False)

    n = len(df_slice)
    data_fp = _compute_data_fingerprint(
        df_slice,
        pair=pair,
        date_start=date_start,
        date_end=date_end,
    )

    rows: List[Dict[str, Any]] = []

    for t in range(0, n - 1):
        date_t = pd.to_datetime(dates[t])
        ref_idx = t + 1
        ref_date = pd.to_datetime(dates[ref_idx])
        ref_open = float(opens[ref_idx])

        atr_t = float(atrs[t]) if np.isfinite(atrs[t]) else np.nan
        sl_dist = 1.5 * atr_t if np.isfinite(atr_t) else np.nan
        r_dist = sl_dist
        tp1_dist = 1.0 * atr_t if np.isfinite(atr_t) else np.nan

        # Available forward bars including ref bar, capped at MAX_LOOKAHEAD
        forward_bars = min(MAX_LOOKAHEAD, n - ref_idx)
        horizon_10 = min(10, forward_bars)
        horizon_20 = min(20, forward_bars)
        horizon_40 = min(40, forward_bars)

        # Precompute per-direction MFE and timing.
        for direction, sign in (("long", +1), ("short", -1)):
            best_10 = 0.0
            best_20 = 0.0
            best_40 = 0.0
            t1 = None
            t3 = None
            t6 = None

            # If ATR_t is non-positive or NaN, R-units are undefined; keep NaN metrics.
            if not np.isfinite(r_dist) or r_dist <= 0.0:
                mfe_10_r = np.nan
                mfe_20_r = np.nan
                mfe_40_r = np.nan
            else:
                for offset in range(forward_bars):
                    k = ref_idx + offset
                    if k >= n or offset >= MAX_LOOKAHEAD:
                        break

                    if direction == "long":
                        move = float(highs[k]) - ref_open
                    else:
                        move = ref_open - float(lows[k])

                    if not np.isfinite(move):
                        continue

                    r_val = move / r_dist
                    if not np.isfinite(r_val):
                        continue

                    if offset < 10:
                        best_10 = max(best_10, r_val)
                    if offset < 20:
                        best_20 = max(best_20, r_val)
                    if offset < 40:
                        best_40 = max(best_40, r_val)

                    if r_val >= 1.0 and t1 is None:
                        t1 = offset
                    if r_val >= 3.0 and t3 is None:
                        t3 = offset
                    if r_val >= 6.0 and t6 is None:
                        t6 = offset

                mfe_10_r = float(best_10) if horizon_10 > 0 else np.nan
                mfe_20_r = float(best_20) if horizon_20 > 0 else np.nan
                mfe_40_r = float(best_40) if horizon_40 > 0 else np.nan

            row: Dict[str, Any] = {
                "pair": pair,
                "date": date_t,
                "direction": direction,
                "dataset_split": _dataset_split_for_date(date_t),
                "ref_date": ref_date,
                "ref_open": ref_open,
                "atr_t": atr_t,
                "sl_dist": sl_dist,
                "r_dist": r_dist,
                "tp1_dist": tp1_dist,
                "horizon_10": int(horizon_10),
                "horizon_20": int(horizon_20),
                "horizon_40": int(horizon_40),
                "mfe_10_r": mfe_10_r,
                "mfe_20_r": mfe_20_r,
                "mfe_40_r": mfe_40_r,
                "zone_a_1r_10": bool(np.isfinite(mfe_10_r) and mfe_10_r >= 1.0),
                "zone_b_3r_20": bool(np.isfinite(mfe_20_r) and mfe_20_r >= 3.0),
                "zone_c_6r_40": bool(np.isfinite(mfe_40_r) and mfe_40_r >= 6.0),
                "t1": float(t1) if t1 is not None else np.nan,
                "t3": float(t3) if t3 is not None else np.nan,
                "t6": float(t6) if t6 is not None else np.nan,
                "label_version": LABEL_VERSION,
                "data_fingerprint": data_fp,
                "engine_atr_fingerprint": ENGINE_ATR_FINGERPRINT,
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=OPPORTUNITY_COLUMNS)

    out = pd.DataFrame(rows)

    # Ensure strict column ordering and deterministic sort.
    for col in OPPORTUNITY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[OPPORTUNITY_COLUMNS]

    # Deterministic sort: pair, date, direction (long then short).
    out["direction"] = pd.Categorical(out["direction"], categories=["long", "short"], ordered=True)
    out = out.sort_values(["pair", "date", "direction"]).reset_index(drop=True)

    return out


def _run_from_config(config_path: Path) -> None:
    raw_cfg = _load_yaml(config_path)
    cfg = _require_phaseD1_config(raw_cfg)

    data_dir = Path(cfg["data_dir"])
    outputs_dir = Path(cfg["outputs_dir"])
    labels_dir = outputs_dir
    labels_dir.mkdir(parents=True, exist_ok=True)

    all_labels: List[pd.DataFrame] = []
    for pair in cfg["pairs"]:
        df = load_pair_csv(pair, data_dir=data_dir)
        labels = generate_labels_for_pair(df, pair=pair, cfg=cfg)
        if not labels.empty:
            all_labels.append(labels)

    if not all_labels:
        print("Phase D1: no labels generated (no tradable bars in configured window).")
        return

    out = pd.concat(all_labels, ignore_index=True)

    # Final deterministic sort (redundant but cheap and explicit)
    out["direction"] = pd.Categorical(out["direction"], categories=["long", "short"], ordered=True)
    out = out.sort_values(["pair", "date", "direction"]).reset_index(drop=True)

    parquet_path = labels_dir / "opportunity_labels.parquet"
    csv_path = labels_dir / "opportunity_labels.csv"

    out.to_parquet(parquet_path, index=False)
    out.to_csv(csv_path, index=False, float_format="%.8f")

    print(f"Phase D1 labels written to:\n  {parquet_path}\n  {csv_path}")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase D-1 — Generate ground-truth directional opportunity labels.",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to Phase D-1 config YAML (e.g. configs/phaseD/phaseD1_opportunity_labels.yaml).",
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

