"""DAUDIT-01 — Data Integrity Audit.

Read-only audit of every OHLCV file under ``data/daily/``, ``data/w1/``,
``data/1hr/``, and ``data/4hr/``. Produces a deterministic, gate-decisional
report plus per-check anomaly CSVs.

Usage::

    python scripts/audit_data_integrity.py -c configs/data_audit.yaml

Exit codes
----------
0 : verdict is PASS (no CRITICAL findings)
1 : verdict is FAIL (>=1 CRITICAL finding)

Determinism
-----------
- Same inputs → byte-identical CSV outputs and a body-identical report.
- The single timestamp lives in the report header line only.
- Anomaly CSVs are always written with their header, even when empty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Make the repo root importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.utils import normalize_ohlcv_schema  # noqa: E402

# ---------------------------------------------------------------------------
# Severity / check-id constants
# ---------------------------------------------------------------------------

SEV_PASS = "PASS"
SEV_WARN = "WARN"
SEV_CRITICAL = "CRITICAL"

# Critical (gate-decisional) check ids
CHK_LOAD_FAIL = "LOAD_FAIL"
CHK_S1_MISSING_COLS = "S1_MISSING_COLS"
CHK_S2_DTYPE = "S2_DTYPE"
CHK_S3_NAN_OHLC = "S3_NAN_OHLC"
CHK_S3_NAN_TIME = "S3_NAN_TIME"
CHK_T1_NON_MONOTONIC = "T1_NON_MONOTONIC"
CHK_T2_DUPLICATE_TS = "T2_DUPLICATE_TS"
CHK_T3_GRID_MISALIGN = "T3_GRID_MISALIGN"
CHK_O1_HIGH_VIOLATION = "O1_HIGH_VIOLATION"
CHK_O2_LOW_VIOLATION = "O2_LOW_VIOLATION"
CHK_O3_HIGH_LT_LOW = "O3_HIGH_LT_LOW"
CHK_O4_NON_POSITIVE_PRICE = "O4_NON_POSITIVE_PRICE"
CHK_X2_AGGREGATION_DIFF = "X2_AGGREGATION_DIFF"

# Informational (WARN-only) check ids
CHK_T4_RANGE = "T4_RANGE"
CHK_T5_GAPS = "T5_GAPS"
CHK_O5_FLAT_BAR = "O5_FLAT_BAR"
CHK_P1_SPREAD_MISSING = "P1_SPREAD_MISSING"
CHK_P2_ZERO_SPREAD = "P2_ZERO_SPREAD"
CHK_P3_SPREAD_DIST = "P3_SPREAD_DIST"
CHK_P4_SPREAD_HIGH = "P4_SPREAD_HIGH"
CHK_A1_RETURN_SPIKE = "A1_RETURN_SPIKE"
CHK_A2_REPEAT_BAR = "A2_REPEAT_BAR"
CHK_X1_RANGE_OVERLAP = "X1_RANGE_OVERLAP"
CHK_X3_PAIR_MISSING = "X3_PAIR_MISSING"

ALL_CHECK_IDS: tuple[str, ...] = (
    CHK_LOAD_FAIL,
    CHK_S1_MISSING_COLS,
    CHK_S2_DTYPE,
    CHK_S3_NAN_OHLC,
    CHK_S3_NAN_TIME,
    CHK_T1_NON_MONOTONIC,
    CHK_T2_DUPLICATE_TS,
    CHK_T3_GRID_MISALIGN,
    CHK_T4_RANGE,
    CHK_T5_GAPS,
    CHK_O1_HIGH_VIOLATION,
    CHK_O2_LOW_VIOLATION,
    CHK_O3_HIGH_LT_LOW,
    CHK_O4_NON_POSITIVE_PRICE,
    CHK_O5_FLAT_BAR,
    CHK_P1_SPREAD_MISSING,
    CHK_P2_ZERO_SPREAD,
    CHK_P3_SPREAD_DIST,
    CHK_P4_SPREAD_HIGH,
    CHK_A1_RETURN_SPIKE,
    CHK_A2_REPEAT_BAR,
    CHK_X1_RANGE_OVERLAP,
    CHK_X2_AGGREGATION_DIFF,
    CHK_X3_PAIR_MISSING,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Finding:
    """A single audit observation. Sorted lexicographically for determinism."""

    timeframe: str
    pair: str
    check_id: str
    severity: str
    detail: str


@dataclass
class FileAudit:
    """Aggregated outcome for one (timeframe, pair) file."""

    timeframe: str
    pair: str
    path: Path
    findings: list[Finding] = field(default_factory=list)
    anomaly_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if any(f.severity == SEV_CRITICAL for f in self.findings):
            return SEV_CRITICAL
        if any(f.severity == SEV_WARN for f in self.findings):
            return SEV_WARN
        return SEV_PASS


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the audit YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    required_top = ["timeframes", "canonical_pairs", "schema", "output", "gate"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(f"data_audit.yaml is missing required keys: {missing}")
    return cfg


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_files(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Walk each timeframe dir and produce a deterministic discovery summary.

    Returns a dict keyed by timeframe label (sorted alphabetically).
    """
    canonical = sorted(set(cfg["canonical_pairs"]))
    out: dict[str, dict[str, Any]] = {}
    for tf in sorted(cfg["timeframes"].keys()):
        tf_meta = cfg["timeframes"][tf]
        tf_dir = Path(tf_meta["dir"])
        if not tf_dir.is_dir():
            out[tf] = {
                "dir": str(tf_dir),
                "exists": False,
                "files": [],
                "extensions": [],
                "footprint_bytes": 0,
                "naming_pattern": "",
                "covered": [],
                "missing": canonical,
                "extras": [],
            }
            continue
        files = sorted(p for p in tf_dir.iterdir() if p.is_file() and p.suffix == ".csv")
        extensions = sorted({p.suffix for p in tf_dir.iterdir() if p.is_file()})
        footprint = sum(p.stat().st_size for p in tf_dir.iterdir() if p.is_file())
        naming_pattern = _detect_naming_pattern([p.name for p in files])
        covered = sorted(p.stem for p in files if p.stem in canonical)
        missing = sorted(set(canonical) - set(covered))
        extras = sorted(p.stem for p in files if p.stem not in canonical)
        out[tf] = {
            "dir": str(tf_dir),
            "exists": True,
            "files": [str(p) for p in files],
            "extensions": extensions,
            "footprint_bytes": footprint,
            "naming_pattern": naming_pattern,
            "covered": covered,
            "missing": missing,
            "extras": extras,
        }
    return out


def _detect_naming_pattern(names: Iterable[str]) -> str:
    """Detect filename convention from a list of csv names. Conservative."""
    has_underscore = False
    has_no_underscore = False
    has_suffix = False
    for n in names:
        stem = Path(n).stem
        if "_" in stem:
            has_underscore = True
            parts = stem.split("_")
            if len(parts) >= 3:
                has_suffix = True
        else:
            has_no_underscore = True
    parts = []
    if has_underscore:
        parts.append("XXX_YYY[.csv]")
    if has_no_underscore:
        parts.append("XXXYYY[.csv]")
    if has_suffix:
        parts.append("with-tf-suffix")
    if not parts:
        return "(none)"
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Per-file checks
# ---------------------------------------------------------------------------


def check_schema_raw_columns(
    raw_columns: list[str],
    required: list[str],
    *,
    tf: str,
    pair: str,
) -> list[Finding]:
    """[S1] Required columns present in the raw on-disk file."""
    cols = list(raw_columns)
    missing = [c for c in required if c not in cols]
    if missing:
        return [
            Finding(
                timeframe=tf,
                pair=pair,
                check_id=CHK_S1_MISSING_COLS,
                severity=SEV_CRITICAL,
                detail=f"missing raw columns: {sorted(missing)}; have: {sorted(cols)}",
            )
        ]
    return []


def check_dtypes(df: pd.DataFrame, *, tf: str, pair: str) -> list[Finding]:
    """[S2] Dtypes: datetime64 for date, numeric OHLC/volume."""
    findings: list[Finding] = []
    if "date" not in df.columns:
        findings.append(
            Finding(tf, pair, CHK_S2_DTYPE, SEV_CRITICAL, "missing 'date' column post-normalize")
        )
        return findings
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        findings.append(
            Finding(
                tf,
                pair,
                CHK_S2_DTYPE,
                SEV_CRITICAL,
                f"date dtype is {df['date'].dtype}; expected datetime64",
            )
        )
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            findings.append(
                Finding(tf, pair, CHK_S2_DTYPE, SEV_CRITICAL, f"missing column post-normalize: {col}")
            )
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            findings.append(
                Finding(
                    tf,
                    pair,
                    CHK_S2_DTYPE,
                    SEV_CRITICAL,
                    f"{col} dtype is {df[col].dtype}; expected numeric",
                )
            )
    return findings


def check_no_nan(df: pd.DataFrame, *, tf: str, pair: str) -> list[Finding]:
    """[S3] No NaN in OHLC; no NaN in time."""
    findings: list[Finding] = []
    if "date" in df.columns:
        n_nan_time = int(df["date"].isna().sum())
        if n_nan_time > 0:
            findings.append(
                Finding(
                    tf,
                    pair,
                    CHK_S3_NAN_TIME,
                    SEV_CRITICAL,
                    f"{n_nan_time} NaN values in time column",
                )
            )
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            continue
        n_nan = int(df[col].isna().sum())
        if n_nan > 0:
            findings.append(
                Finding(
                    tf,
                    pair,
                    CHK_S3_NAN_OHLC,
                    SEV_CRITICAL,
                    f"{n_nan} NaN values in column {col}",
                )
            )
    return findings


def check_time_monotonic(df: pd.DataFrame, *, tf: str, pair: str) -> list[Finding]:
    """[T1] time is strictly monotonic increasing."""
    if "date" not in df.columns or df["date"].isna().any():
        return []
    diffs = df["date"].diff().dropna()
    bad = diffs[diffs <= pd.Timedelta(0)]
    if not bad.empty:
        first_idx = int(bad.index[0])
        return [
            Finding(
                tf,
                pair,
                CHK_T1_NON_MONOTONIC,
                SEV_CRITICAL,
                f"{len(bad)} non-increasing time deltas; first at row {first_idx}",
            )
        ]
    return []


def check_duplicate_timestamps(
    df: pd.DataFrame, *, tf: str, pair: str
) -> tuple[list[Finding], list[dict[str, Any]]]:
    """[T2] No duplicate timestamps. Returns findings and anomaly rows."""
    if "date" not in df.columns:
        return [], []
    dup_mask = df["date"].duplicated(keep=False) & df["date"].notna()
    if not dup_mask.any():
        return [], []
    rows: list[dict[str, Any]] = []
    for ts, group in df.loc[dup_mask].groupby("date"):
        rows.append(
            {
                "timeframe": tf,
                "pair": pair,
                "timestamp": _iso(ts),
                "occurrences": int(len(group)),
            }
        )
    rows.sort(key=lambda r: (r["timestamp"], r["pair"]))
    finding = Finding(
        tf,
        pair,
        CHK_T2_DUPLICATE_TS,
        SEV_CRITICAL,
        f"{int(dup_mask.sum())} duplicated rows across {len(rows)} timestamp(s)",
    )
    return [finding], rows


def check_grid_alignment(
    df: pd.DataFrame, *, grid: str, tf: str, pair: str
) -> tuple[list[Finding], list[dict[str, Any]]]:
    """[T3] Bar grid alignment per timeframe."""
    if "date" not in df.columns or df["date"].isna().all():
        return [], []
    t = df["date"]
    rows: list[dict[str, Any]] = []
    detail: str | None = None

    if grid == "D":
        bad_mask = (t.dt.hour != 0) | (t.dt.minute != 0) | (t.dt.second != 0)
        if bad_mask.any():
            detail = f"{int(bad_mask.sum())} daily bars not at 00:00"
    elif grid == "W":
        bad_time = (t.dt.hour != 0) | (t.dt.minute != 0) | (t.dt.second != 0)
        weekdays = sorted(set(int(x) for x in t.dt.weekday.unique() if pd.notna(x)))
        if bad_time.any():
            detail = f"{int(bad_time.sum())} weekly bars not at 00:00"
        if len(weekdays) > 1:
            detail = (detail + "; " if detail else "") + (
                f"weekly bars span multiple weekdays {weekdays} (expected all Sun=6 or all Mon=0)"
            )
        elif weekdays and weekdays[0] not in (0, 6):
            detail = (detail + "; " if detail else "") + (
                f"weekly bars start on weekday {weekdays[0]} (expected Sun=6 or Mon=0)"
            )
    elif grid == "H1":
        bad_mask = (t.dt.minute != 0) | (t.dt.second != 0)
        if bad_mask.any():
            detail = f"{int(bad_mask.sum())} 1H bars not minute=0,second=0"
    elif grid == "H4":
        bad_mask = (~t.dt.hour.isin([0, 4, 8, 12, 16, 20])) | (t.dt.minute != 0) | (t.dt.second != 0)
        if bad_mask.any():
            bad_hours = sorted(set(int(h) for h in t.loc[bad_mask].dt.hour.unique()))
            detail = (
                f"{int(bad_mask.sum())} 4H bars off-grid; offending hours present: {bad_hours}"
            )

    if detail is None:
        return [], []
    return [Finding(tf, pair, CHK_T3_GRID_MISALIGN, SEV_CRITICAL, detail)], rows


def detect_gaps(
    df: pd.DataFrame, *, grid: str, tf: str, pair: str
) -> tuple[list[Finding], list[dict[str, Any]]]:
    """[T5] Classify each gap > 1 grid step. Returns WARN findings + rows."""
    if "date" not in df.columns or len(df) < 2:
        return [], []
    step = _grid_step(grid)
    if step is None:
        return [], []
    t = df["date"].reset_index(drop=True)
    deltas = t.diff().dropna()
    rows: list[dict[str, Any]] = []
    suspicious_count = 0
    for i, d in deltas.items():
        if d <= step:
            continue
        n_steps = int(d / step)
        prev_ts = t.iloc[i - 1]
        cur_ts = t.iloc[i]
        klass = _classify_gap(prev_ts, cur_ts, grid, n_steps)
        if klass == "suspicious":
            suspicious_count += 1
        rows.append(
            {
                "timeframe": tf,
                "pair": pair,
                "from": _iso(prev_ts),
                "to": _iso(cur_ts),
                "gap_steps": n_steps,
                "classification": klass,
            }
        )
    rows.sort(key=lambda r: (r["from"], r["pair"]))
    findings: list[Finding] = []
    if rows:
        findings.append(
            Finding(
                tf,
                pair,
                CHK_T5_GAPS,
                SEV_WARN,
                f"{len(rows)} gap(s); suspicious={suspicious_count}",
            )
        )
    return findings, rows


def _grid_step(grid: str) -> pd.Timedelta | None:
    return {
        "D": pd.Timedelta(days=1),
        "W": pd.Timedelta(weeks=1),
        "H1": pd.Timedelta(hours=1),
        "H4": pd.Timedelta(hours=4),
    }.get(grid)


def _classify_gap(prev_ts: pd.Timestamp, cur_ts: pd.Timestamp, grid: str, n_steps: int) -> str:
    """Weekend / holiday-candidate / suspicious."""
    if grid == "W":
        return "holiday-candidate" if n_steps <= 2 else "suspicious"
    spans_weekend = _spans_weekend(prev_ts, cur_ts)
    if grid == "D":
        if spans_weekend and n_steps <= 3:
            return "weekend"
        if not spans_weekend and n_steps <= 2:
            return "holiday-candidate"
        return "suspicious"
    if grid == "H1":
        if spans_weekend and n_steps <= 60:
            return "weekend"
        if not spans_weekend and n_steps <= 24:
            return "holiday-candidate"
        return "suspicious"
    if grid == "H4":
        if spans_weekend and n_steps <= 18:
            return "weekend"
        if not spans_weekend and n_steps <= 8:
            return "holiday-candidate"
        return "suspicious"
    return "suspicious"


def _spans_weekend(prev_ts: pd.Timestamp, cur_ts: pd.Timestamp) -> bool:
    """True if the gap (prev_ts, cur_ts) overlaps a Saturday or Sunday."""
    one = pd.Timedelta(days=1)
    cursor = (prev_ts + one).normalize()
    end = cur_ts.normalize()
    while cursor < end:
        if cursor.weekday() >= 5:
            return True
        cursor += one
    return False


def check_ohlc_inequalities(
    df: pd.DataFrame, *, tf: str, pair: str
) -> tuple[list[Finding], dict[str, list[dict[str, Any]]]]:
    """[O1][O2][O3] OHLC inequalities, strict."""
    findings: list[Finding] = []
    rows: dict[str, list[dict[str, Any]]] = {
        CHK_O1_HIGH_VIOLATION: [],
        CHK_O2_LOW_VIOLATION: [],
        CHK_O3_HIGH_LT_LOW: [],
    }
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(df.columns):
        return findings, rows
    sub = df[["date", "open", "high", "low", "close"]].dropna()
    high_vio = sub.loc[sub["high"] < sub[["open", "close"]].max(axis=1)]
    low_vio = sub.loc[sub["low"] > sub[["open", "close"]].min(axis=1)]
    hl_vio = sub.loc[sub["high"] < sub["low"]]

    if len(high_vio):
        findings.append(
            Finding(
                tf, pair, CHK_O1_HIGH_VIOLATION, SEV_CRITICAL,
                f"{len(high_vio)} bar(s) where high < max(open, close)",
            )
        )
        rows[CHK_O1_HIGH_VIOLATION] = _ohlc_rows(high_vio, tf, pair)
    if len(low_vio):
        findings.append(
            Finding(
                tf, pair, CHK_O2_LOW_VIOLATION, SEV_CRITICAL,
                f"{len(low_vio)} bar(s) where low > min(open, close)",
            )
        )
        rows[CHK_O2_LOW_VIOLATION] = _ohlc_rows(low_vio, tf, pair)
    if len(hl_vio):
        findings.append(
            Finding(
                tf, pair, CHK_O3_HIGH_LT_LOW, SEV_CRITICAL,
                f"{len(hl_vio)} bar(s) where high < low",
            )
        )
        rows[CHK_O3_HIGH_LT_LOW] = _ohlc_rows(hl_vio, tf, pair)
    return findings, rows


def _ohlc_rows(sub: pd.DataFrame, tf: str, pair: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _, r in sub.iterrows():
        out.append(
            {
                "timeframe": tf,
                "pair": pair,
                "timestamp": _iso(r["date"]),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
            }
        )
    out.sort(key=lambda r: (r["timestamp"], r["pair"]))
    return out


def check_positive_prices(
    df: pd.DataFrame, *, tf: str, pair: str
) -> tuple[list[Finding], list[dict[str, Any]]]:
    """[O4] All OHLC > 0."""
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(df.columns):
        return [], []
    bad_mask = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) & df[
        ["open", "high", "low", "close"]
    ].notna().all(axis=1)
    if not bad_mask.any():
        return [], []
    sub = df.loc[bad_mask, ["date", "open", "high", "low", "close"]]
    rows = _ohlc_rows(sub, tf, pair)
    finding = Finding(
        tf, pair, CHK_O4_NON_POSITIVE_PRICE, SEV_CRITICAL,
        f"{int(bad_mask.sum())} bar(s) with non-positive OHLC",
    )
    return [finding], rows


def check_flat_bars(
    df: pd.DataFrame, *, max_examples: int, tf: str, pair: str
) -> tuple[list[Finding], list[dict[str, Any]]]:
    """[O5] open==high==low==close — informational."""
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(df.columns):
        return [], []
    sub = df[["date", "open", "high", "low", "close"]].dropna()
    flat = sub.loc[
        (sub["open"] == sub["high"])
        & (sub["high"] == sub["low"])
        & (sub["low"] == sub["close"])
    ]
    if flat.empty:
        return [], []
    rows = _ohlc_rows(flat.head(max_examples), tf, pair)
    finding = Finding(
        tf, pair, CHK_O5_FLAT_BAR, SEV_WARN,
        f"{len(flat)} flat bar(s); recorded first {len(rows)}",
    )
    return [finding], rows


def check_spread(
    raw_df: pd.DataFrame, *, threshold_pips: float, tf: str, pair: str
) -> tuple[list[Finding], list[dict[str, Any]], dict[str, Any]]:
    """[P1][P2][P3][P4] Spread sanity. Spread on disk is raw MT5 points (pips=points/10)."""
    findings: list[Finding] = []
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    if "spread" not in raw_df.columns:
        findings.append(
            Finding(tf, pair, CHK_P1_SPREAD_MISSING, SEV_CRITICAL, "spread column not present")
        )
        return findings, rows, summary

    spread_raw = pd.to_numeric(raw_df["spread"], errors="coerce")
    if not pd.api.types.is_numeric_dtype(spread_raw) or spread_raw.isna().all():
        findings.append(
            Finding(
                tf, pair, CHK_P1_SPREAD_MISSING, SEV_CRITICAL,
                "spread column not numeric or all NaN",
            )
        )
        return findings, rows, summary

    spread_pips = spread_raw / 10.0
    n_total = int(spread_pips.notna().sum())
    n_zero = int((spread_pips == 0).sum())
    pct_zero = (n_zero / n_total * 100.0) if n_total else 0.0
    summary["spread_zero_count"] = n_zero
    summary["spread_zero_pct"] = round(pct_zero, 4)

    if pct_zero > 0:
        findings.append(
            Finding(
                tf, pair, CHK_P2_ZERO_SPREAD, SEV_WARN,
                f"{n_zero} bar(s) with spread==0 ({pct_zero:.2f}%)",
            )
        )

    if n_total > 0:
        s = spread_pips.dropna()
        summary["spread_min_pips"] = round(float(s.min()), 4)
        summary["spread_p50_pips"] = round(float(s.median()), 4)
        summary["spread_p90_pips"] = round(float(s.quantile(0.90)), 4)
        summary["spread_p99_pips"] = round(float(s.quantile(0.99)), 4)
        summary["spread_max_pips"] = round(float(s.max()), 4)
        if summary["spread_max_pips"] > threshold_pips:
            findings.append(
                Finding(
                    tf, pair, CHK_P3_SPREAD_DIST, SEV_WARN,
                    f"max spread {summary['spread_max_pips']:.2f} pips > threshold {threshold_pips}",
                )
            )

    high_mask = spread_pips > threshold_pips
    if high_mask.any():
        sub = raw_df.loc[high_mask, ["time", "spread"]].copy()
        sub["spread_pips"] = spread_pips[high_mask].values
        for _, r in sub.iterrows():
            rows.append(
                {
                    "timeframe": tf,
                    "pair": pair,
                    "timestamp": _iso(pd.to_datetime(r["time"])),
                    "spread_raw": float(r["spread"]),
                    "spread_pips": float(r["spread_pips"]),
                }
            )
        rows.sort(key=lambda x: (x["timestamp"], x["pair"]))
        findings.append(
            Finding(
                tf, pair, CHK_P4_SPREAD_HIGH, SEV_WARN,
                f"{len(rows)} bar(s) with spread > {threshold_pips} pips",
            )
        )
    return findings, rows, summary


def check_anomaly_returns(
    df: pd.DataFrame,
    *,
    atr_window: int,
    atr_multiple: float,
    tf: str,
    pair: str,
) -> tuple[list[Finding], list[dict[str, Any]]]:
    """[A1] |bar return| > N × rolling ATR(window). Informational."""
    if not {"open", "high", "low", "close", "date"}.issubset(df.columns):
        return [], []
    if len(df) <= atr_window + 1:
        return [], []
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(atr_window, min_periods=atr_window).mean()
    bar_abs_ret = (close - prev_close).abs()
    threshold = atr * atr_multiple
    spike_mask = (bar_abs_ret > threshold) & threshold.notna()
    if not spike_mask.any():
        return [], []
    sub = df.loc[spike_mask, ["date", "open", "high", "low", "close"]].copy()
    sub["abs_return"] = bar_abs_ret[spike_mask].values
    sub["atr"] = atr[spike_mask].values
    rows: list[dict[str, Any]] = []
    for _, r in sub.iterrows():
        rows.append(
            {
                "timeframe": tf,
                "pair": pair,
                "timestamp": _iso(r["date"]),
                "open": float(r["open"]),
                "close": float(r["close"]),
                "abs_return": float(r["abs_return"]),
                "atr": float(r["atr"]),
            }
        )
    rows.sort(key=lambda r: (r["timestamp"], r["pair"]))
    finding = Finding(
        tf, pair, CHK_A1_RETURN_SPIKE, SEV_WARN,
        f"{len(rows)} bar(s) with |return| > {atr_multiple}*ATR({atr_window})",
    )
    return [finding], rows


def check_repeat_bars(
    df: pd.DataFrame, *, tf: str, pair: str
) -> tuple[list[Finding], list[dict[str, Any]]]:
    """[A2] Bars whose OHLC matches the prior bar exactly (potential dead feed)."""
    if not {"open", "high", "low", "close", "date"}.issubset(df.columns):
        return [], []
    cols = ["open", "high", "low", "close"]
    same = (df[cols].shift(1) == df[cols]).all(axis=1)
    if not same.any():
        return [], []
    sub = df.loc[same, ["date"] + cols]
    rows = _ohlc_rows(sub, tf, pair)
    finding = Finding(
        tf, pair, CHK_A2_REPEAT_BAR, SEV_WARN, f"{int(same.sum())} bar(s) identical to prior bar"
    )
    return [finding], rows


# ---------------------------------------------------------------------------
# Cross-timeframe consistency
# ---------------------------------------------------------------------------


def aggregate_to_target(low: pd.DataFrame, target_grid: str) -> pd.DataFrame:
    """Aggregate a lower-timeframe normalized DF to the next higher grid.

    Buckets are anchored consistently with the on-disk data:
      H1 → H4: floor to 4h boundary
      H4 → D : floor to day
      D  → W : floor to ISO week starting Sunday
    """
    if "date" not in low.columns:
        raise ValueError("aggregate_to_target requires a 'date' column")
    df = low[["date", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("date").reset_index(drop=True)
    if target_grid == "H4":
        df["bucket"] = df["date"].dt.floor("4h")
    elif target_grid == "D":
        df["bucket"] = df["date"].dt.floor("D")
    elif target_grid == "W":
        # Anchor the weekly bucket on Sunday so it matches the on-disk w1 layout.
        df["bucket"] = df["date"].apply(_floor_to_sunday)
    else:
        raise ValueError(f"unsupported target_grid: {target_grid}")
    grouped = df.groupby("bucket", sort=True)
    out = pd.DataFrame(
        {
            "date": grouped["bucket"].first(),
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
        }
    ).reset_index(drop=True)
    return out


def _floor_to_sunday(ts: pd.Timestamp) -> pd.Timestamp:
    ts = ts.normalize()
    # Python weekday(): Mon=0..Sun=6. Sunday-anchored week → subtract (weekday+1) % 7 days.
    return ts - pd.Timedelta(days=(ts.weekday() + 1) % 7)


def cross_tf_compare(
    low_df: pd.DataFrame,
    high_df: pd.DataFrame,
    target_grid: str,
    *,
    sample_size: int,
    seed: int,
    pair: str,
    low_tf: str,
    high_tf: str,
    tolerance: float,
) -> tuple[list[Finding], list[dict[str, Any]]]:
    """[X2] Sample N non-weekend buckets and check OHLC aggregation parity."""
    rows: list[dict[str, Any]] = []
    findings: list[Finding] = []
    if low_df.empty or high_df.empty:
        return findings, rows
    low_agg = aggregate_to_target(low_df, target_grid)
    high_norm = high_df[["date", "open", "high", "low", "close", "volume"]].copy()
    if target_grid == "W":
        high_norm["date"] = high_norm["date"].apply(_floor_to_sunday)
    elif target_grid == "D":
        high_norm["date"] = high_norm["date"].dt.floor("D")
    elif target_grid == "H4":
        high_norm["date"] = high_norm["date"].dt.floor("4h")
    merged = low_agg.merge(
        high_norm, on="date", how="inner", suffixes=("_agg", "_disk")
    )
    if merged.empty:
        return findings, rows
    merged = merged.loc[~merged["date"].dt.weekday.isin([5, 6])].reset_index(drop=True)
    if merged.empty:
        return findings, rows
    rng = np.random.default_rng(seed)
    sample_n = min(sample_size, len(merged))
    idx = rng.choice(len(merged), size=sample_n, replace=False)
    idx = np.sort(idx)
    sample = merged.iloc[idx]

    max_diff = {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0}
    for _, r in sample.iterrows():
        diffs = {
            "open": abs(float(r["open_agg"]) - float(r["open_disk"])),
            "high": abs(float(r["high_agg"]) - float(r["high_disk"])),
            "low": abs(float(r["low_agg"]) - float(r["low_disk"])),
            "close": abs(float(r["close_agg"]) - float(r["close_disk"])),
        }
        for k, v in diffs.items():
            if v > max_diff[k]:
                max_diff[k] = v
        rows.append(
            {
                "pair": pair,
                "low_tf": low_tf,
                "high_tf": high_tf,
                "bucket": _iso(r["date"]),
                "open_diff": diffs["open"],
                "high_diff": diffs["high"],
                "low_diff": diffs["low"],
                "close_diff": diffs["close"],
                "volume_diff": float(r["volume_agg"]) - float(r["volume_disk"]),
            }
        )
    rows.sort(key=lambda r: (r["pair"], r["low_tf"], r["high_tf"], r["bucket"]))
    worst = max(max_diff.values())
    if worst > tolerance:
        findings.append(
            Finding(
                high_tf,
                pair,
                CHK_X2_AGGREGATION_DIFF,
                SEV_CRITICAL,
                (
                    f"{low_tf}->{high_tf} aggregation max OHLC abs diff = {worst} > tol {tolerance}; "
                    f"per-col max={max_diff}"
                ),
            )
        )
    return findings, rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def audit_one_file(
    path: Path,
    *,
    tf: str,
    pair: str,
    grid: str,
    cfg: Mapping[str, Any],
) -> tuple[FileAudit, pd.DataFrame | None]:
    """Run all per-file checks. Returns the audit and the normalized df (or None)."""
    audit = FileAudit(timeframe=tf, pair=pair, path=path)
    audit.summary["pair"] = pair
    audit.summary["timeframe"] = tf
    audit.summary["file"] = str(path)
    audit.summary["bytes"] = path.stat().st_size if path.exists() else 0

    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 — surface any read failure as CRITICAL
        audit.findings.append(
            Finding(tf, pair, CHK_LOAD_FAIL, SEV_CRITICAL, f"failed to read csv: {exc!r}")
        )
        return audit, None

    audit.findings.extend(
        check_schema_raw_columns(
            list(raw.columns), cfg["schema"]["required_columns"], tf=tf, pair=pair
        )
    )

    try:
        norm = normalize_ohlcv_schema(raw)
    except Exception as exc:  # noqa: BLE001
        audit.findings.append(
            Finding(tf, pair, CHK_LOAD_FAIL, SEV_CRITICAL, f"normalize_ohlcv_schema raised: {exc!r}")
        )
        return audit, None

    audit.findings.extend(check_dtypes(norm, tf=tf, pair=pair))
    audit.findings.extend(check_no_nan(norm, tf=tf, pair=pair))
    audit.findings.extend(check_time_monotonic(norm, tf=tf, pair=pair))

    f, rows = check_duplicate_timestamps(norm, tf=tf, pair=pair)
    audit.findings.extend(f)
    if rows:
        audit.anomaly_rows.setdefault(CHK_T2_DUPLICATE_TS, []).extend(rows)

    f, _ = check_grid_alignment(norm, grid=grid, tf=tf, pair=pair)
    audit.findings.extend(f)

    f, gap_rows = detect_gaps(norm, grid=grid, tf=tf, pair=pair)
    audit.findings.extend(f)
    if gap_rows:
        audit.anomaly_rows.setdefault(CHK_T5_GAPS, []).extend(gap_rows)

    f, ohlc_rows = check_ohlc_inequalities(norm, tf=tf, pair=pair)
    audit.findings.extend(f)
    for k, v in ohlc_rows.items():
        if v:
            audit.anomaly_rows.setdefault(k, []).extend(v)

    f, pos_rows = check_positive_prices(norm, tf=tf, pair=pair)
    audit.findings.extend(f)
    if pos_rows:
        audit.anomaly_rows.setdefault(CHK_O4_NON_POSITIVE_PRICE, []).extend(pos_rows)

    f, flat_rows = check_flat_bars(
        norm, max_examples=int(cfg["output"].get("flat_bar_examples_max", 20)), tf=tf, pair=pair
    )
    audit.findings.extend(f)
    if flat_rows:
        audit.anomaly_rows.setdefault(CHK_O5_FLAT_BAR, []).extend(flat_rows)

    spread_findings, spread_rows, spread_summary = check_spread(
        raw,
        threshold_pips=float(cfg["anomaly"]["spread_pips_warn_threshold"]),
        tf=tf,
        pair=pair,
    )
    audit.findings.extend(spread_findings)
    if spread_rows:
        audit.anomaly_rows.setdefault(CHK_P4_SPREAD_HIGH, []).extend(spread_rows)
    audit.summary.update(spread_summary)

    f, ret_rows = check_anomaly_returns(
        norm,
        atr_window=int(cfg["anomaly"]["atr_window"]),
        atr_multiple=float(cfg["anomaly"]["return_atr_multiple"]),
        tf=tf,
        pair=pair,
    )
    audit.findings.extend(f)
    if ret_rows:
        audit.anomaly_rows.setdefault(CHK_A1_RETURN_SPIKE, []).extend(ret_rows)

    f, rep_rows = check_repeat_bars(norm, tf=tf, pair=pair)
    audit.findings.extend(f)
    if rep_rows:
        audit.anomaly_rows.setdefault(CHK_A2_REPEAT_BAR, []).extend(rep_rows)

    if "date" in norm.columns and norm["date"].notna().any():
        audit.summary["first_bar"] = _iso(norm["date"].iloc[0])
        audit.summary["last_bar"] = _iso(norm["date"].iloc[-1])
        audit.summary["bar_count"] = int(len(norm))
    else:
        audit.summary["first_bar"] = ""
        audit.summary["last_bar"] = ""
        audit.summary["bar_count"] = int(len(norm))
    return audit, norm


def run_audit(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Execute the full audit and produce the report payload (no I/O)."""
    discovery = discover_files(cfg)

    per_pair_norm: dict[str, dict[str, pd.DataFrame]] = {}
    audits: list[FileAudit] = []
    for tf in sorted(cfg["timeframes"].keys()):
        tf_meta = cfg["timeframes"][tf]
        grid = tf_meta["grid"]
        for path_str in sorted(discovery[tf]["files"]):
            path = Path(path_str)
            pair = path.stem
            audit, norm = audit_one_file(path, tf=tf, pair=pair, grid=grid, cfg=cfg)
            audits.append(audit)
            if norm is not None:
                per_pair_norm.setdefault(pair, {})[tf] = norm

    cross_findings: list[Finding] = []
    cross_rows: list[dict[str, Any]] = []
    range_overlap_rows: list[dict[str, Any]] = []
    pair_missing_rows: list[dict[str, Any]] = []
    sample_size = int(cfg["cross_tf"]["random_sample_dates_per_pair"])
    seed = int(cfg["cross_tf"]["random_seed"])
    tolerance = float(cfg["cross_tf"]["ohlc_tolerance"])

    for pair in sorted(per_pair_norm.keys()):
        tfs = per_pair_norm[pair]
        # X1: range overlap report
        for tf_name in sorted(tfs.keys()):
            df = tfs[tf_name]
            if not df.empty and df["date"].notna().any():
                range_overlap_rows.append(
                    {
                        "pair": pair,
                        "timeframe": tf_name,
                        "first_bar": _iso(df["date"].iloc[0]),
                        "last_bar": _iso(df["date"].iloc[-1]),
                        "bar_count": int(len(df)),
                    }
                )
        # X3: pair coverage gaps
        all_tfs = set(cfg["timeframes"].keys())
        present = set(tfs.keys())
        missing = sorted(all_tfs - present)
        if missing:
            pair_missing_rows.append({"pair": pair, "missing_timeframes": ",".join(missing)})

        # X2: deterministic per-pair seed (so each pair's draw is reproducible)
        pair_seed = (seed + int(hashlib.md5(pair.encode("utf-8")).hexdigest(), 16)) % (2**32)
        for low_tf, high_tf, target in (("1hr", "4hr", "H4"), ("4hr", "daily", "D"), ("daily", "w1", "W")):
            if low_tf in tfs and high_tf in tfs:
                f, rows = cross_tf_compare(
                    tfs[low_tf],
                    tfs[high_tf],
                    target,
                    sample_size=sample_size,
                    seed=pair_seed,
                    pair=pair,
                    low_tf=low_tf,
                    high_tf=high_tf,
                    tolerance=tolerance,
                )
                cross_findings.extend(f)
                cross_rows.extend(rows)

    range_overlap_rows.sort(key=lambda r: (r["pair"], r["timeframe"]))
    pair_missing_rows.sort(key=lambda r: r["pair"])
    cross_rows.sort(key=lambda r: (r["pair"], r["low_tf"], r["high_tf"], r["bucket"]))

    return {
        "discovery": discovery,
        "audits": audits,
        "cross_findings": cross_findings,
        "cross_rows": cross_rows,
        "range_overlap_rows": range_overlap_rows,
        "pair_missing_rows": pair_missing_rows,
    }


# ---------------------------------------------------------------------------
# Output / report
# ---------------------------------------------------------------------------


SUMMARY_COLUMNS: list[str] = [
    "timeframe",
    "pair",
    "status",
    "bar_count",
    "first_bar",
    "last_bar",
    "bytes",
    "spread_min_pips",
    "spread_p50_pips",
    "spread_p90_pips",
    "spread_p99_pips",
    "spread_max_pips",
    "spread_zero_count",
    "spread_zero_pct",
    "n_findings_critical",
    "n_findings_warn",
]


def write_outputs(payload: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[Path, str]:
    """Persist the report, summary CSV, and anomaly CSVs. Returns (report_path, verdict)."""
    out_root = Path(cfg["output"]["results_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    anomalies_dir = out_root / cfg["output"]["anomalies_subdir"]
    anomalies_dir.mkdir(parents=True, exist_ok=True)

    audits: list[FileAudit] = list(payload["audits"])
    cross_findings: list[Finding] = list(payload["cross_findings"])
    cross_rows: list[dict[str, Any]] = list(payload["cross_rows"])
    range_overlap_rows: list[dict[str, Any]] = list(payload["range_overlap_rows"])
    pair_missing_rows: list[dict[str, Any]] = list(payload["pair_missing_rows"])

    # --- Summary CSV -----------------------------------------------------
    summary_rows: list[dict[str, Any]] = []
    audits_sorted = sorted(audits, key=lambda a: (a.timeframe, a.pair))
    for a in audits_sorted:
        row = {col: "" for col in SUMMARY_COLUMNS}
        row.update(
            {
                "timeframe": a.timeframe,
                "pair": a.pair,
                "status": a.status,
                "bar_count": a.summary.get("bar_count", ""),
                "first_bar": a.summary.get("first_bar", ""),
                "last_bar": a.summary.get("last_bar", ""),
                "bytes": a.summary.get("bytes", ""),
                "spread_min_pips": a.summary.get("spread_min_pips", ""),
                "spread_p50_pips": a.summary.get("spread_p50_pips", ""),
                "spread_p90_pips": a.summary.get("spread_p90_pips", ""),
                "spread_p99_pips": a.summary.get("spread_p99_pips", ""),
                "spread_max_pips": a.summary.get("spread_max_pips", ""),
                "spread_zero_count": a.summary.get("spread_zero_count", ""),
                "spread_zero_pct": a.summary.get("spread_zero_pct", ""),
                "n_findings_critical": sum(
                    1 for f in a.findings if f.severity == SEV_CRITICAL
                ),
                "n_findings_warn": sum(1 for f in a.findings if f.severity == SEV_WARN),
            }
        )
        summary_rows.append(row)
    summary_csv = out_root / cfg["output"]["summary_csv"]
    _write_csv(summary_csv, SUMMARY_COLUMNS, summary_rows)

    # --- Anomaly CSVs (one per check_id; written even when empty) --------
    anomaly_buckets: dict[str, list[dict[str, Any]]] = {cid: [] for cid in ALL_CHECK_IDS}
    for a in audits_sorted:
        for cid, rows in a.anomaly_rows.items():
            anomaly_buckets.setdefault(cid, []).extend(rows)
    if cross_rows:
        anomaly_buckets[CHK_X2_AGGREGATION_DIFF].extend(cross_rows)
    if range_overlap_rows:
        anomaly_buckets[CHK_X1_RANGE_OVERLAP] = range_overlap_rows
    if pair_missing_rows:
        anomaly_buckets[CHK_X3_PAIR_MISSING] = pair_missing_rows

    for cid in sorted(anomaly_buckets.keys()):
        rows = anomaly_buckets[cid]
        cols = _columns_for_check(cid, rows)
        path = anomalies_dir / f"{cid}.csv"
        _write_csv(path, cols, rows)

    # --- Verdict ---------------------------------------------------------
    all_findings = [f for a in audits_sorted for f in a.findings] + cross_findings
    critical = [f for f in all_findings if f.severity == SEV_CRITICAL]
    verdict = "PASS" if not critical else "FAIL"

    # --- Report markdown -------------------------------------------------
    report_path = out_root / cfg["output"]["report_filename"]
    report_text = render_report(
        payload=payload, audits=audits_sorted, cross_findings=cross_findings, verdict=verdict
    )
    report_path.write_text(report_text, encoding="utf-8")
    return report_path, verdict


def render_report(
    *,
    payload: Mapping[str, Any],
    audits: list[FileAudit],
    cross_findings: list[Finding],
    verdict: str,
) -> str:
    discovery = payload["discovery"]
    range_overlap_rows = payload["range_overlap_rows"]
    pair_missing_rows = payload["pair_missing_rows"]
    cross_rows = payload["cross_rows"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append(f"# Data Audit Report — generated {now}")
    lines.append("")
    lines.append("## 1. Discovery")
    lines.append("")
    lines.append("| timeframe | dir | exists | files | extensions | naming | covered | missing | extras | bytes |")
    lines.append("|-----------|-----|--------|-------|------------|--------|---------|---------|--------|-------|")
    for tf in sorted(discovery.keys()):
        d = discovery[tf]
        lines.append(
            f"| {tf} | {d['dir']} | {d['exists']} | {len(d['files'])} | "
            f"{','.join(d['extensions']) or '(none)'} | {d['naming_pattern']} | "
            f"{len(d['covered'])} | {','.join(d['missing']) or '(none)'} | "
            f"{','.join(d['extras']) or '(none)'} | {d['footprint_bytes']} |"
        )
    lines.append("")

    lines.append("## 2. Per-timeframe summary")
    lines.append("")
    lines.append("| timeframe | PASS | WARN | CRITICAL |")
    lines.append("|-----------|------|------|----------|")
    by_tf: dict[str, dict[str, int]] = {}
    for a in audits:
        by_tf.setdefault(a.timeframe, {SEV_PASS: 0, SEV_WARN: 0, SEV_CRITICAL: 0})
        by_tf[a.timeframe][a.status] += 1
    for tf in sorted(by_tf.keys()):
        row = by_tf[tf]
        lines.append(f"| {tf} | {row[SEV_PASS]} | {row[SEV_WARN]} | {row[SEV_CRITICAL]} |")
    lines.append("")

    lines.append("## 3. Per-file table (sorted by timeframe, pair)")
    lines.append("")
    lines.append("| timeframe | pair | status | bar_count | first_bar | last_bar | crit | warn |")
    lines.append("|-----------|------|--------|-----------|-----------|----------|------|------|")
    for a in audits:
        lines.append(
            f"| {a.timeframe} | {a.pair} | {a.status} | "
            f"{a.summary.get('bar_count', '')} | "
            f"{a.summary.get('first_bar', '')} | "
            f"{a.summary.get('last_bar', '')} | "
            f"{sum(1 for f in a.findings if f.severity == SEV_CRITICAL)} | "
            f"{sum(1 for f in a.findings if f.severity == SEV_WARN)} |"
        )
    lines.append("")

    lines.append("## 4. CRITICAL findings (full)")
    lines.append("")
    crit = sorted(
        [f for a in audits for f in a.findings if f.severity == SEV_CRITICAL]
        + [f for f in cross_findings if f.severity == SEV_CRITICAL]
    )
    if not crit:
        lines.append("_None._")
    else:
        lines.append("| timeframe | pair | check_id | detail |")
        lines.append("|-----------|------|----------|--------|")
        for f in crit:
            lines.append(f"| {f.timeframe} | {f.pair} | {f.check_id} | {f.detail} |")
    lines.append("")

    lines.append("## 5. WARN findings (summary)")
    lines.append("")
    warn = sorted([f for a in audits for f in a.findings if f.severity == SEV_WARN] +
                  [f for f in cross_findings if f.severity == SEV_WARN])
    if not warn:
        lines.append("_None._")
    else:
        lines.append("| timeframe | pair | check_id | detail |")
        lines.append("|-----------|------|----------|--------|")
        for f in warn:
            lines.append(f"| {f.timeframe} | {f.pair} | {f.check_id} | {f.detail} |")
    lines.append("")

    lines.append("## 6. Cross-timeframe consistency")
    lines.append("")
    lines.append("### Range overlap (X1)")
    lines.append("")
    if range_overlap_rows:
        lines.append("| pair | timeframe | first_bar | last_bar | bar_count |")
        lines.append("|------|-----------|-----------|----------|-----------|")
        for r in range_overlap_rows:
            lines.append(
                f"| {r['pair']} | {r['timeframe']} | {r['first_bar']} | {r['last_bar']} | {r['bar_count']} |"
            )
    else:
        lines.append("_No timeframes loaded._")
    lines.append("")
    lines.append("### Aggregation parity (X2)")
    lines.append("")
    if cross_rows:
        worst = {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0}
        for r in cross_rows:
            for k in worst:
                v = abs(r[f"{k}_diff"])
                if v > worst[k]:
                    worst[k] = v
        lines.append(
            f"Sampled {len(cross_rows)} buckets across pairs; max abs OHLC diffs: "
            f"open={worst['open']}, high={worst['high']}, low={worst['low']}, close={worst['close']}."
        )
    else:
        lines.append("_No cross-TF samples generated._")
    lines.append("")
    lines.append("### Pair coverage gaps (X3)")
    lines.append("")
    if pair_missing_rows:
        lines.append("| pair | missing_timeframes |")
        lines.append("|------|--------------------|")
        for r in pair_missing_rows:
            lines.append(f"| {r['pair']} | {r['missing_timeframes']} |")
    else:
        lines.append("_All pairs present in all timeframes._")
    lines.append("")

    lines.append("## 7. Final verdict")
    lines.append("")
    lines.append(f"**{verdict}**")
    lines.append("")
    return "\n".join(lines)


def _columns_for_check(check_id: str, rows: list[dict[str, Any]]) -> list[str]:
    if rows:
        return list(rows[0].keys())
    schemas: dict[str, list[str]] = {
        CHK_T2_DUPLICATE_TS: ["timeframe", "pair", "timestamp", "occurrences"],
        CHK_T5_GAPS: ["timeframe", "pair", "from", "to", "gap_steps", "classification"],
        CHK_O1_HIGH_VIOLATION: ["timeframe", "pair", "timestamp", "open", "high", "low", "close"],
        CHK_O2_LOW_VIOLATION: ["timeframe", "pair", "timestamp", "open", "high", "low", "close"],
        CHK_O3_HIGH_LT_LOW: ["timeframe", "pair", "timestamp", "open", "high", "low", "close"],
        CHK_O4_NON_POSITIVE_PRICE: ["timeframe", "pair", "timestamp", "open", "high", "low", "close"],
        CHK_O5_FLAT_BAR: ["timeframe", "pair", "timestamp", "open", "high", "low", "close"],
        CHK_P4_SPREAD_HIGH: ["timeframe", "pair", "timestamp", "spread_raw", "spread_pips"],
        CHK_A1_RETURN_SPIKE: [
            "timeframe", "pair", "timestamp", "open", "close", "abs_return", "atr",
        ],
        CHK_A2_REPEAT_BAR: ["timeframe", "pair", "timestamp", "open", "high", "low", "close"],
        CHK_X1_RANGE_OVERLAP: ["pair", "timeframe", "first_bar", "last_bar", "bar_count"],
        CHK_X2_AGGREGATION_DIFF: [
            "pair", "low_tf", "high_tf", "bucket",
            "open_diff", "high_diff", "low_diff", "close_diff", "volume_diff",
        ],
        CHK_X3_PAIR_MISSING: ["pair", "missing_timeframes"],
    }
    return schemas.get(check_id, ["timeframe", "pair", "check_id", "detail"])


def _write_csv(path: Path, columns: list[str], rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use newline="" so csv writer controls line endings → identical bytes across runs.
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _iso(ts: Any) -> str:
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return ""
    if isinstance(ts, pd.Timestamp):
        return ts.isoformat()
    try:
        return pd.to_datetime(ts).isoformat()
    except Exception:
        return str(ts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DAUDIT-01 data integrity audit (read-only).")
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to configs/data_audit.yaml",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    payload = run_audit(cfg)
    report_path, verdict = write_outputs(payload, cfg)
    print(f"verdict: {verdict}")
    print(f"report : {report_path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
