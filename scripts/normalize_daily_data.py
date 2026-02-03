# ruff: noqa: I001
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Tuple

import pandas as pd


FTMO_COLS = {"DATE", "<DATE>"}


def _detect_ftmo_header(line: str) -> bool:
    """
    Return True if the header line looks like FTMO/MT4 tab-delimited schema.
    """
    if "\t" not in line:
        return False
    parts = [p.strip() for p in line.strip().split("\t")]
    if not parts:
        return False
    first = parts[0].strip()
    return first in FTMO_COLS or first.upper() == "DATE"


def _is_legacy_header(line: str) -> bool:
    """
    Return True if the header line already matches the legacy CSV schema.
    """
    return line.lower().startswith("time,")


def normalize_daily_file(path: Path) -> Tuple[bool, str]:
    """
    Normalize a single CSV file in-place to the legacy daily schema.

    Returns (converted: bool, reason: str).
    """
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            first_line = f.readline()
    except FileNotFoundError:
        return False, "missing"
    if not first_line:
        return False, "empty"

    if _is_legacy_header(first_line):
        return False, "already_legacy"
    if not _detect_ftmo_header(first_line):
        return False, "unknown_format"

    # Read FTMO-style tab-delimited file
    df = pd.read_csv(path, sep="\t")

    # Build a case-insensitive, angle-bracket-stripped mapping
    rename_map = {}
    for col in df.columns:
        key = col.strip().strip("<>").upper()
        if key == "DATE":
            rename_map[col] = "time"
        elif key == "OPEN":
            rename_map[col] = "open"
        elif key == "HIGH":
            rename_map[col] = "high"
        elif key == "LOW":
            rename_map[col] = "low"
        elif key == "CLOSE":
            rename_map[col] = "close"
        elif key in ("TICKVOL", "TICKVOLUME"):
            rename_map[col] = "tick_volume"
        elif key == "SPREAD":
            rename_map[col] = "spread"
        elif key in ("VOL", "VOLUME", "REAL_VOLUME"):
            rename_map[col] = "real_volume"

    df = df.rename(columns=rename_map)

    # Ensure all required columns exist
    required = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread",
        "real_volume",
    ]
    for col in required:
        if col not in df.columns:
            if col == "real_volume":
                df[col] = 0
            else:
                raise ValueError(f"Required column '{col}' missing in {path}")

    # Normalise date format: YYYY.MM.DD -> YYYY-MM-DD, output column name 'time'
    df["time"] = (
        pd.to_datetime(
            df["time"].astype(str).str.replace(".", "-", regex=False),
            errors="coerce",
        )
        .dt.strftime("%Y-%m-%d")
    )

    # Sort and drop duplicate times
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="first")

    # Reorder columns exactly
    df = df[required]

    # Atomic write
    dir_path = path.parent
    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=dir_path,
            prefix=f".{path.stem}_tmp_",
            suffix=".csv",
            delete=False,
            encoding="utf-8",
            newline="",
        ) as tmp:
            tmp_file = Path(tmp.name)
            # Use csv writer to ensure comma delimiter, no extra index
            writer = csv.writer(tmp)
            writer.writerow(required)
            for _, row in df.iterrows():
                writer.writerow([row[c] for c in required])

        # Replace original file (no explicit remove to avoid WinError 32 on Windows)
        try:
            os.replace(tmp_file, path)
        except PermissionError:
            # Best-effort: try to relax permissions on the target, then retry once.
            try:
                if path.exists():
                    os.chmod(path, 0o666)
            except OSError:
                pass
            try:
                os.replace(tmp_file, path)
            except PermissionError:
                # Give up on this file but clean up the temp file.
                return False, "permission_denied"
    finally:
        if tmp_file is not None and tmp_file.exists() and not path.exists():
            # Clean up temp on failure
            try:
                tmp_file.unlink()
            except OSError:
                pass

    return True, "converted"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "daily"
    if not data_dir.exists():
        print(f"[normalize_daily_data] data/daily not found at {data_dir}")
        return

    converted = 0
    skipped = 0
    for csv_path in sorted(data_dir.glob("*.csv")):
        # Skip our own temporary files (start with a dot and stem prefix we use)
        if csv_path.name.startswith("."):
            continue
        did_convert, reason = normalize_daily_file(csv_path)
        if did_convert:
            converted += 1
            print(f"[normalize] converted: {csv_path.name}")
        else:
            skipped += 1
            print(f"[normalize] skipped:   {csv_path.name} ({reason})")

    print(
        f"[normalize_daily_data] summary: converted={converted} skipped={skipped} "
        f"(total={converted + skipped})"
    )


if __name__ == "__main__":
    main()

