#!/usr/bin/env python3
"""
Normalize MT5 W1 OHLCV exports under data/raw/W1_export/ to CSV under data/w1/
matching the on-disk D1 schema: time, open, high, low, close, tick_volume, spread, real_volume.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from core.utils import normalize_ohlcv_schema  # noqa: E402, F401

RAW_DIR = REPO_ROOT / "data" / "raw" / "W1_export"
OUT_DIR = REPO_ROOT / "data" / "w1"

# 2010-01-01 <= date < 2026-01-01
DATE_START = pd.Timestamp("2010-01-01")
DATE_END = pd.Timestamp("2026-01-01")

RAW_NAME_RE = re.compile(r"^([A-Za-z]{3})([A-Za-z]{3})_W1\.csv$", re.IGNORECASE)

REQUIRED_RAW = ("time", "open", "high", "low", "close", "tick_volume", "spread")

# Integrity thresholds (post-write verification)
MIN_ROWS = 600
LATE_START_CUTOFF = pd.Timestamp("2019-02-01")
EARLY_END_CUTOFF = pd.Timestamp("2025-11-01")

# OHLC + tick volume (D1-style columns); spread/real_volume checked only if NaN breaks use
OHLC_TICK_COLS = ("open", "high", "low", "close", "tick_volume")


def _read_csv_mt5(path: Path) -> pd.DataFrame:
    """Read MT5 export; supports UTF-8 (with/without BOM) and UTF-16 LE/BE (common from MT5)."""
    head = path.read_bytes()[:4]
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        enc = "utf-16"
    else:
        enc = "utf-8-sig"
    return pd.read_csv(path, encoding=enc)


def raw_path_to_out_name(path: Path) -> str | None:
    m = RAW_NAME_RE.match(path.name)
    if not m:
        return None
    return f"{m.group(1).upper()}_{m.group(2).upper()}.csv"


def load_and_transform(raw_path: Path) -> pd.DataFrame:
    df = _read_csv_mt5(raw_path)
    missing = set(REQUIRED_RAW) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}; have {list(df.columns)}")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce", format="mixed")
    for col in ("open", "high", "low", "close", "tick_volume", "spread"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "real_volume" in df.columns:
        df["real_volume"] = pd.to_numeric(df["real_volume"], errors="coerce").fillna(0)
    else:
        df["real_volume"] = 0

    df = df.dropna(subset=["time", "open", "high", "low", "close", "tick_volume", "spread"])
    df = df[(df["time"] >= DATE_START) & (df["time"] < DATE_END)]
    df = df.sort_values("time", ascending=True).reset_index(drop=True)

    df["tick_volume"] = df["tick_volume"].round().astype("int64")
    df["spread"] = df["spread"].round().astype("int64")
    df["real_volume"] = df["real_volume"].round().astype("int64")

    out = df[list(REQUIRED_RAW) + ["real_volume"]].copy()
    out["time"] = out["time"].dt.strftime("%Y-%m-%d")
    return out


def verify_output_csv(path: Path) -> list[str]:
    issues: list[str] = []
    df = pd.read_csv(path, parse_dates=["time"])
    n = len(df)
    if n < MIN_ROWS:
        issues.append(f"row_count {n} < {MIN_ROWS} (likely missing history)")
    if n == 0:
        return issues
    mn = pd.to_datetime(df["time"].iloc[0])
    mx = pd.to_datetime(df["time"].iloc[-1])
    if mn > LATE_START_CUTOFF:
        issues.append(f"late_start min_date={mn} (after {LATE_START_CUTOFF.date()})")
    if mx < EARLY_END_CUTOFF:
        issues.append(f"early_end max_date={mx} (before {EARLY_END_CUTOFF.date()})")
    for c in OHLC_TICK_COLS:
        if df[c].isna().any():
            issues.append(f"NaN in column {c}")
    return issues


def main() -> int:
    if not RAW_DIR.is_dir():
        print(f"ERROR: raw directory not found: {RAW_DIR}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.glob("*.csv"))
    written = 0
    skipped = 0
    written_paths: list[Path] = []

    for raw in raw_files:
        out_name = raw_path_to_out_name(raw)
        if out_name is None:
            print(f"SKIP (name pattern): {raw.name}")
            skipped += 1
            continue

        out_path = OUT_DIR / out_name
        try:
            df = load_and_transform(raw)
        except Exception as e:
            print(f"SKIP {raw.name}: {e}")
            skipped += 1
            continue

        df.to_csv(out_path, index=False)
        written += 1
        written_paths.append(out_path)
        ts = pd.to_datetime(df["time"])
        mn, mx = ts.min(), ts.max()
        print(f"{out_name}: rows={len(df)}, range={mn} -> {mx}")

    print(f"\nDone: wrote {written} file(s), skipped {skipped}.")

    # Verification pass
    print("\n--- Integrity check ---")
    out_csvs = sorted(written_paths, key=lambda p: p.name)
    any_warn = False
    for p in out_csvs:
        issues = verify_output_csv(p)
        if issues:
            any_warn = True
            for msg in issues:
                print(f"WARNING: {p.name}: {msg}")
    if out_csvs and not any_warn:
        print("No issues flagged (row counts, date span, NaN checks).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
