"""03_build_distributions.py — Per-pair distribution + execution-bar CSVs.

Inputs:  data/external/dukascopy_processed/{PAIR}_hourly.csv  (Phase 2b output)
Outputs:
    results/spread_validation/per_pair_distributions.csv  (long)
    results/spread_validation/execution_bar_spreads.csv   (wide)

per_pair_distributions: unit = per-hour median spread; percentiles across
hours within each (pair, session).
execution_bar_spreads: percentiles of first5min_median_spread_pips for
tf in {1H (all hours), 4H (hours ∈ {0,4,8,12,16,20})}.
"""

from __future__ import annotations

import csv
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS: tuple[str, ...] = (
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
)

SESSIONS: tuple[str, ...] = ("london", "ny", "overlap", "off_hours", "weekend_edge")
PERCENTILES: tuple[int, ...] = (10, 25, 50, 75, 90, 95, 99)
EXEC_TF_LIST: tuple[str, ...] = ("1H", "4H")
HOURS_4H: set[int] = {0, 4, 8, 12, 16, 20}

CURRENT_FLOOR_PIPS: float = 0.1

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
INPUT_DIR: Path = REPO_ROOT / "data" / "external" / "dukascopy_processed"
OUTPUT_DIR: Path = REPO_ROOT / "results" / "spread_validation"


def fmt(x) -> str:
    if isinstance(x, float):
        if np.isnan(x):
            return ""
        return format(x, ".10g")
    return str(x)


def session_distribution(df: pd.DataFrame, pair: str, session: str, source: str) -> list[list[str]]:
    sub = df[df["session"] == session]
    n_hours = int(len(sub))
    n_ticks_total = int(sub["n_ticks"].sum()) if n_hours > 0 else 0
    if n_hours == 0:
        return [
            [pair, session, f"p{q}", "", str(n_ticks_total), str(n_hours), source, "", ""]
            for q in PERCENTILES
        ]
    hourly_medians = sub["p50_spread_pips"].dropna().to_numpy()
    if hourly_medians.size == 0:
        return [
            [pair, session, f"p{q}", "", str(n_ticks_total), str(n_hours), source, "", ""]
            for q in PERCENTILES
        ]
    pcts = np.percentile(hourly_medians, PERCENTILES, method="linear")
    window_start = sub["hour_utc"].min()
    window_end = sub["hour_utc"].max()
    rows = []
    for q, v in zip(PERCENTILES, pcts):
        rows.append([
            pair, session, f"p{q}",
            fmt(float(v)),
            str(n_ticks_total), str(n_hours),
            source,
            str(window_start), str(window_end),
        ])
    return rows


def execution_bar_summary(df: pd.DataFrame, pair: str, tf: str) -> list[str]:
    if tf == "1H":
        sub = df
    elif tf == "4H":
        sub = df[df["hour"].isin(HOURS_4H)]
    else:
        raise ValueError(f"unknown tf: {tf}")
    n_bars = int(len(sub))
    if n_bars == 0:
        return [
            pair, tf, str(n_bars),
            "", "", "", "",
            "", "", "", "",
            fmt(CURRENT_FLOOR_PIPS), "",
        ]

    first5 = sub["first5min_median_spread_pips"].dropna().to_numpy()
    fullh = sub["p50_spread_pips"].dropna().to_numpy()

    if first5.size > 0:
        p25_f, p50_f, p75_f, p95_f = np.percentile(first5, [25, 50, 75, 95], method="linear")
    else:
        p25_f = p50_f = p75_f = p95_f = float("nan")

    if fullh.size > 0:
        p25_h, p50_h, p75_h, p95_h = np.percentile(fullh, [25, 50, 75, 95], method="linear")
    else:
        p25_h = p50_h = p75_h = p95_h = float("nan")

    gap = (p50_f - CURRENT_FLOOR_PIPS) if not np.isnan(p50_f) else float("nan")

    return [
        pair, tf, str(n_bars),
        fmt(float(p25_f)), fmt(float(p50_f)), fmt(float(p75_f)), fmt(float(p95_f)),
        fmt(float(p25_h)), fmt(float(p50_h)), fmt(float(p75_h)), fmt(float(p95_h)),
        fmt(CURRENT_FLOOR_PIPS), fmt(float(gap)),
    ]


def main() -> int:
    if not INPUT_DIR.exists():
        print(f"BLOCKER: input dir not found: {INPUT_DIR}", flush=True)
        return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dist_rows: list[list[str]] = []
    exec_rows: list[list[str]] = []
    missing_pairs: list[str] = []
    t0 = time.time()

    for pair in PAIRS:
        path = INPUT_DIR / f"{pair}_hourly.csv"
        if not path.exists():
            missing_pairs.append(pair)
            continue
        df = pd.read_csv(path)
        if df.empty:
            missing_pairs.append(pair)
            continue
        source = f"histdata:{path.name}"
        for session in SESSIONS:
            dist_rows.extend(session_distribution(df, pair, session, source))
        for tf in EXEC_TF_LIST:
            exec_rows.append(execution_bar_summary(df, pair, tf))

    dist_path = OUTPUT_DIR / "per_pair_distributions.csv"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([
        "pair", "session", "percentile", "spread_pips",
        "n_ticks_in_session", "n_hours_in_session",
        "source", "window_start", "window_end",
    ])
    for r in dist_rows:
        w.writerow(r)
    dist_path.write_bytes(buf.getvalue().encode("utf-8"))

    exec_path = OUTPUT_DIR / "execution_bar_spreads.csv"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([
        "pair", "tf", "n_execution_bars",
        "p25_spread_pips_first5min", "p50_spread_pips_first5min",
        "p75_spread_pips_first5min", "p95_spread_pips_first5min",
        "p25_spread_pips_fullhour", "p50_spread_pips_fullhour",
        "p75_spread_pips_fullhour", "p95_spread_pips_fullhour",
        "current_floor_pips", "gap_p50_first5min_minus_floor_pips",
    ])
    for r in exec_rows:
        w.writerow(r)
    exec_path.write_bytes(buf.getvalue().encode("utf-8"))

    elapsed = time.time() - t0
    print(f"DONE  elapsed={elapsed:.1f}s", flush=True)
    print(f"  per_pair_distributions.csv  rows={len(dist_rows)}  -> {dist_path}", flush=True)
    print(f"  execution_bar_spreads.csv    rows={len(exec_rows)}  -> {exec_path}", flush=True)
    if missing_pairs:
        print(f"  MISSING pairs: {missing_pairs}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
