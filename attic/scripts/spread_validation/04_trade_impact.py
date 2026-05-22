"""04_trade_impact.py — Per-trade spread under-payment analysis.

For each of {KH-24, Arc 4, Arc 5} trade logs, compute the per-trade gap
between the modeled spread (backtester) and the real first-5-min median
spread (Dukascopy/HistData) at entry and exit execution bars.

KH-24: no floor; modeled exit = raw MT5 spread / 10 at exit_date bar.
L Arcs: 0.1 pip floor; modeled exit = spread_pips_exit in trade log.

1R = sl_distance_pips (KH-24: derived from sl_distance_atr × atr_abs / pip_size).
KH-24 risk = 1.0%; L arcs risk = 0.5%.
"""

from __future__ import annotations

import csv
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DUKAS_DIR: Path = REPO_ROOT / "data" / "external" / "dukascopy_processed"
OUTPUT_DIR: Path = REPO_ROOT / "results" / "spread_validation"

KH24_TRADES: Path = REPO_ROOT / "results" / "kh24" / "trades_all.csv"
ARC4_TRADES: Path = REPO_ROOT / "results" / "l_arc_4" / "step1" / "trades_all.csv"
ARC5_TRADES: Path = REPO_ROOT / "results" / "l_arc_5" / "step1" / "trades_all.csv"
WFO_KH24_YAML: Path = REPO_ROOT / "configs" / "wfo_kh24.yaml"

DATA_4HR: Path = REPO_ROOT / "data" / "4hr"
DATA_1HR: Path = REPO_ROOT / "data" / "1hr"

COVERAGE_START_ISO: str = "2024-01-01T00:00:00+00:00"
COVERAGE_END_EXCL_ISO: str = "2026-01-01T00:00:00+00:00"

KH24_FLOOR_PIPS: float = 0.0
ARC_FLOOR_PIPS: float = 0.1
KH24_RISK_PCT: float = 1.0
ARC_RISK_PCT: float = 0.5

POINTS_PER_PIP: float = 10.0


def pair_underscore_to_dukas(pair: str) -> str:
    return pair.replace("_", "")


def pip_size_in_price(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") or pair.endswith("JPY") else 0.0001


def fmt(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if np.isnan(x):
            return ""
        return format(x, ".10g")
    return str(x)


def floor_4h_open(ts: pd.Timestamp) -> pd.Timestamp:
    h = (ts.hour // 4) * 4
    return ts.replace(minute=0, second=0, microsecond=0, hour=h)


def floor_1h_open(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.replace(minute=0, second=0, microsecond=0)


def load_dukas(pair_underscore: str) -> pd.DataFrame | None:
    dukas_pair = pair_underscore_to_dukas(pair_underscore)
    path = DUKAS_DIR / f"{dukas_pair}_hourly.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True)
    df = df.set_index("hour_utc").sort_index()
    return df


def real_spread_at(dukas: pd.DataFrame, bar_open: pd.Timestamp) -> float | None:
    if dukas is None or bar_open not in dukas.index:
        return None
    val = dukas.at[bar_open, "first5min_median_spread_pips"]
    if pd.isna(val):
        return None
    return float(val)


def load_raw_4h(pair_underscore: str) -> pd.DataFrame | None:
    path = DATA_4HR / f"{pair_underscore}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    return df


def raw_spread_pips_at(raw_df: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    if raw_df is None or ts not in raw_df.index:
        return None
    s = raw_df.at[ts, "spread"]
    if pd.isna(s):
        return None
    return float(s) / POINTS_PER_PIP


def load_kh24_folds() -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    cfg = yaml.safe_load(WFO_KH24_YAML.read_text(encoding="utf-8"))
    folds_raw = cfg["wfo"]["folds"]
    folds: list[tuple[int, pd.Timestamp, pd.Timestamp]] = []
    for f in folds_raw:
        folds.append((
            int(f["fold"]),
            pd.to_datetime(f["oos_start"], utc=True),
            pd.to_datetime(f["oos_end"], utc=True),
        ))
    return folds


def assign_fold(ts: pd.Timestamp, folds: list[tuple[int, pd.Timestamp, pd.Timestamp]]) -> int | None:
    for fid, s, e in folds:
        if s <= ts < e:
            return fid
    return None


def evaluate_kh24() -> tuple[list[list[str]], list[list[str]]]:
    if not KH24_TRADES.exists():
        return [], []
    tr = pd.read_csv(KH24_TRADES)
    tr["entry_date"] = pd.to_datetime(tr["entry_date"], utc=True)
    tr["exit_date"] = pd.to_datetime(tr["exit_date"], utc=True)
    folds = load_kh24_folds()

    cov_start = pd.to_datetime(COVERAGE_START_ISO)
    cov_end = pd.to_datetime(COVERAGE_END_EXCL_ISO)

    dukas_cache: dict[str, pd.DataFrame | None] = {}
    raw_cache: dict[str, pd.DataFrame | None] = {}

    per_trade_rows: list[list[str]] = []
    for idx, row in tr.iterrows():
        pair = row["pair"]
        entry_ts = row["entry_date"]
        exit_ts = row["exit_date"]
        modeled_entry = float(row.get("spread_pips_used", float("nan")))

        if pair not in dukas_cache:
            dukas_cache[pair] = load_dukas(pair)
        if pair not in raw_cache:
            raw_cache[pair] = load_raw_4h(pair)

        dukas = dukas_cache[pair]
        raw4h = raw_cache[pair]

        entry_bar = floor_4h_open(entry_ts)
        exit_bar = floor_4h_open(exit_ts)

        in_window = (cov_start <= entry_ts < cov_end) and (cov_start <= exit_ts < cov_end)

        modeled_exit = raw_spread_pips_at(raw4h, exit_bar) if raw4h is not None else None
        floored_entry = modeled_entry
        floored_exit = modeled_exit if modeled_exit is not None else None
        entry_floor_act = (floored_entry == 0.0)
        exit_floor_act = (floored_exit == 0.0) if floored_exit is not None else False

        real_entry = real_spread_at(dukas, entry_bar) if (in_window and dukas is not None) else None
        real_exit = real_spread_at(dukas, exit_bar) if (in_window and dukas is not None) else None

        if not in_window:
            status = "OUTSIDE_WINDOW"
        elif real_entry is None and real_exit is None:
            status = "MISSING_BOTH"
        elif real_entry is None:
            status = "MISSING_REAL_ENTRY"
        elif real_exit is None:
            status = "MISSING_REAL_EXIT"
        else:
            status = "OK"

        entry_under = (real_entry - floored_entry) if (real_entry is not None) else float("nan")
        exit_under = (
            (real_exit - floored_exit)
            if (real_exit is not None and floored_exit is not None)
            else float("nan")
        )
        total_under = float("nan")
        if not np.isnan(entry_under) and not np.isnan(exit_under):
            total_under = entry_under + exit_under

        sl_atr = float(row.get("sl_distance_atr", float("nan")))
        atr_abs = float(row.get("atr_abs", float("nan")))
        psize = pip_size_in_price(pair)
        sl_pips = float("nan")
        if not np.isnan(sl_atr) and not np.isnan(atr_abs) and atr_abs > 0:
            sl_pips = sl_atr * atr_abs / psize

        under_R = float("nan")
        under_pct_equity = float("nan")
        if not np.isnan(total_under) and not np.isnan(sl_pips) and sl_pips > 0:
            under_R = total_under / sl_pips
            under_pct_equity = under_R * KH24_RISK_PCT
        elif not np.isnan(total_under) and (np.isnan(sl_pips) or sl_pips <= 0):
            status = "BAD_SL" if status == "OK" else status

        fold_id = assign_fold(entry_ts, folds)

        per_trade_rows.append([
            pair, "4H", str(int(idx)),
            str(fold_id) if fold_id is not None else "",
            entry_ts.isoformat(), exit_ts.isoformat(),
            fmt(modeled_entry),
            fmt(real_entry),
            fmt(entry_under),
            "1" if entry_floor_act else "0",
            fmt(modeled_exit),
            fmt(real_exit),
            fmt(exit_under),
            "1" if exit_floor_act else "0",
            fmt(total_under),
            fmt(sl_pips),
            fmt(under_R),
            fmt(under_pct_equity),
            status,
        ])

    summary_rows = summarise(per_trade_rows, "KH-24", risk_pct=KH24_RISK_PCT, by_fold=True)
    return per_trade_rows, summary_rows


def evaluate_arc(arc_name: str, trades_path: Path) -> tuple[list[list[str]], list[list[str]]]:
    if not trades_path.exists():
        return [], []
    tr = pd.read_csv(trades_path)
    tr["entry_time"] = pd.to_datetime(tr["entry_time"], utc=True)
    tr["exit_time"] = pd.to_datetime(tr["exit_time"], utc=True)

    cov_start = pd.to_datetime(COVERAGE_START_ISO)
    cov_end = pd.to_datetime(COVERAGE_END_EXCL_ISO)

    dukas_cache: dict[str, pd.DataFrame | None] = {}

    per_trade_rows: list[list[str]] = []
    for idx, row in tr.iterrows():
        pair = row["pair"]
        entry_ts = row["entry_time"]
        exit_ts = row["exit_time"]
        modeled_entry = float(row.get("spread_pips_used", float("nan")))
        modeled_exit = float(row.get("spread_pips_exit", float("nan")))

        if pair not in dukas_cache:
            dukas_cache[pair] = load_dukas(pair)
        dukas = dukas_cache[pair]

        entry_bar = floor_1h_open(entry_ts)
        exit_bar = floor_1h_open(exit_ts)

        in_window = (cov_start <= entry_ts < cov_end) and (cov_start <= exit_ts < cov_end)

        entry_floor_act = bool(modeled_entry == ARC_FLOOR_PIPS)
        exit_floor_act = bool(modeled_exit == ARC_FLOOR_PIPS)

        real_entry = real_spread_at(dukas, entry_bar) if (in_window and dukas is not None) else None
        real_exit = real_spread_at(dukas, exit_bar) if (in_window and dukas is not None) else None

        if not in_window:
            status = "OUTSIDE_WINDOW"
        elif real_entry is None and real_exit is None:
            status = "MISSING_BOTH"
        elif real_entry is None:
            status = "MISSING_REAL_ENTRY"
        elif real_exit is None:
            status = "MISSING_REAL_EXIT"
        else:
            status = "OK"

        entry_under = (real_entry - modeled_entry) if (real_entry is not None) else float("nan")
        exit_under = (real_exit - modeled_exit) if (real_exit is not None) else float("nan")
        total_under = float("nan")
        if not np.isnan(entry_under) and not np.isnan(exit_under):
            total_under = entry_under + exit_under

        sl_pips = float(row.get("sl_distance_pips", float("nan")))
        under_R = float("nan")
        under_pct_equity = float("nan")
        if not np.isnan(total_under) and not np.isnan(sl_pips) and sl_pips > 0:
            under_R = total_under / sl_pips
            under_pct_equity = under_R * ARC_RISK_PCT

        per_trade_rows.append([
            pair, "1H", str(row.get("trade_id", idx)),
            "",
            entry_ts.isoformat(), exit_ts.isoformat(),
            fmt(modeled_entry),
            fmt(real_entry),
            fmt(entry_under),
            "1" if entry_floor_act else "0",
            fmt(modeled_exit),
            fmt(real_exit),
            fmt(exit_under),
            "1" if exit_floor_act else "0",
            fmt(total_under),
            fmt(sl_pips),
            fmt(under_R),
            fmt(under_pct_equity),
            status,
        ])

    summary_rows = summarise(per_trade_rows, arc_name, risk_pct=ARC_RISK_PCT, by_fold=False)
    return per_trade_rows, summary_rows


def summarise(
    rows: list[list[str]], arc: str, risk_pct: float, by_fold: bool
) -> list[list[str]]:
    def _f(s: str) -> float:
        return float(s) if s else float("nan")

    if by_fold:
        keys = sorted({r[3] for r in rows if r[3] != ""} | {""})
    else:
        keys = [""]

    out: list[list[str]] = []
    for k in keys:
        if by_fold and k == "":
            subset = rows
            label = "ALL"
        elif by_fold:
            subset = [r for r in rows if r[3] == k]
            label = k
        else:
            subset = rows
            label = "ALL"

        n_total = len(subset)
        ok = [r for r in subset if r[18] == "OK"]
        n_eval = len(ok)

        if n_eval == 0:
            out.append([
                arc, label,
                str(n_total), str(n_eval),
                "", "",
                "", "", "", "",
                "", "",
            ])
            continue

        pct_e_floor = sum(1 for r in ok if r[9] == "1") / n_eval * 100.0
        pct_x_floor = sum(1 for r in ok if r[13] == "1") / n_eval * 100.0
        tot_under = np.array([_f(r[14]) for r in ok], dtype=float)
        under_R = np.array([_f(r[16]) for r in ok], dtype=float)
        under_pct = np.array([_f(r[17]) for r in ok], dtype=float)
        valid_under = tot_under[~np.isnan(tot_under)]
        if valid_under.size > 0:
            p25, p50, p75, p95 = np.percentile(valid_under, [25, 50, 75, 95], method="linear")
        else:
            p25 = p50 = p75 = p95 = float("nan")
        mean_R = np.nanmean(under_R) if under_R.size > 0 else float("nan")
        total_pct = np.nansum(under_pct) if under_pct.size > 0 else float("nan")

        out.append([
            arc, label,
            str(n_total), str(n_eval),
            fmt(float(pct_e_floor)),
            fmt(float(pct_x_floor)),
            fmt(float(p25)),
            fmt(float(p50)),
            fmt(float(p75)),
            fmt(float(p95)),
            fmt(float(mean_R)),
            fmt(float(total_pct)),
        ])
    return out


PER_TRADE_HEADER = [
    "pair", "tf", "trade_id", "fold",
    "entry_time_utc", "exit_time_utc",
    "modeled_entry_pips", "real_entry_pips", "entry_under_pips", "entry_floor_activated",
    "modeled_exit_pips", "real_exit_pips", "exit_under_pips", "exit_floor_activated",
    "total_under_pips", "sl_distance_pips", "under_R", "under_pct_equity", "status",
]

SUMMARY_HEADER = [
    "arc", "fold",
    "n_trades_total", "n_trades_evaluated",
    "pct_entry_floor_activated", "pct_exit_floor_activated",
    "p25_under_pips", "p50_under_pips", "p75_under_pips", "p95_under_pips",
    "mean_under_R", "total_under_pct_equity",
]


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(r)
    path.write_bytes(buf.getvalue().encode("utf-8"))


def main() -> int:
    t0 = time.time()
    print("Evaluating KH-24 …", flush=True)
    kh_rows, kh_sum = evaluate_kh24()
    print(f"  {len(kh_rows)} trade rows", flush=True)

    print("Evaluating Arc 4 …", flush=True)
    a4_rows, a4_sum = evaluate_arc("L_ARC_4", ARC4_TRADES)
    print(f"  {len(a4_rows)} trade rows", flush=True)

    print("Evaluating Arc 5 …", flush=True)
    a5_rows, a5_sum = evaluate_arc("L_ARC_5", ARC5_TRADES)
    print(f"  {len(a5_rows)} trade rows", flush=True)

    write_csv(OUTPUT_DIR / "per_trade_impact_kh24.csv", PER_TRADE_HEADER, kh_rows)
    write_csv(OUTPUT_DIR / "per_trade_impact_arc4.csv", PER_TRADE_HEADER, a4_rows)
    write_csv(OUTPUT_DIR / "per_trade_impact_arc5.csv", PER_TRADE_HEADER, a5_rows)
    write_csv(OUTPUT_DIR / "per_arc_summary.csv", SUMMARY_HEADER, kh_sum + a4_sum + a5_sum)

    elapsed = time.time() - t0
    print(f"DONE  elapsed={elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
