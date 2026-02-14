"""
Phase D-5: Opportunity Capture Attribution (Analytics Only).

Purpose: Attribute executed trades to opportunity labels and measure zone capture rates.
This module reads labels + trades CSVs and outputs summary CSVs. It does NOT modify
entry logic, exits, filters, or run any optimization/WFO.

HARD CONSTRAINTS (locked):
  - No changes to engine, signal_logic, or indicators.
  - All outputs are CSVs + one decision memo text file.
  - Trade-to-label matching: join on (pair, date == entry_date, direction).
  - Zone events: consecutive True segments in zone flag series; capture rule uses
    first X label-dates from event start.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Label columns required for Phase D-5
LABEL_REQUIRED = [
    "pair",
    "date",
    "direction",
    "dataset_split",
    "zone_a_1r_10",
    "zone_b_3r_20",
    "zone_c_6r_40",
    "t1",
    "t3",
    "t6",
    "mfe_40_r",
]

ENTRY_DATE_ALIASES = [
    "entry_date",
    "entry_time",
    "entry_datetime",
    "open_time",
    "date_open",
    "opened_at",
]

DIRECTION_ALIASES = [
    "direction",
    "side",
    "trade_direction",
    "direction_int",
    "is_long",
]


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        lc = cand.lower()
        if lc in cols_lower:
            return cols_lower[lc]
    return None


def _trades_csv_is_valid(path: Path) -> bool:
    """Check trades.csv has required columns and at least 1 row."""
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
        if df.empty or len(df) < 1:
            return False
        if not _resolve_column(df, ["pair"]):
            return False
        if not _resolve_column(df, ENTRY_DATE_ALIASES):
            return False
        if not any(_resolve_column(df, [a]) for a in DIRECTION_ALIASES):
            return False
        return True
    except Exception:
        return False


def _nearest_existing_folder(path: Path) -> Path | None:
    """Return the first existing parent when walking from path upward."""
    p = Path(path).resolve()
    p = p.parent if p.suffix else p  # Start from directory that would contain file
    while True:
        if p.exists():
            return p
        parent = p.parent
        if parent == p:
            return None
        p = parent


def _find_latest_trades_under(root: Path) -> Path | None:
    """Search root recursively for trades.csv; return path with most recent mtime if valid."""
    root = Path(root).resolve()
    if not root.exists():
        return None
    candidates: list[Path] = []
    for p in root.rglob("trades.csv"):
        if p.is_file() and _trades_csv_is_valid(p):
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.stat().st_mtime)


def _format_trades_not_found_error(path: Path) -> str:
    """Build actionable FileNotFoundError message for missing trades."""
    abs_path = Path(path).resolve()
    nearest = _nearest_existing_folder(abs_path)
    parts = [
        f"Trades file not found. Absolute path: {abs_path}",
        "",
    ]
    if nearest:
        parts.append(f"Nearest existing folder in path chain: {nearest}")
        parts.append("")
    parts.append("Next steps:")
    parts.append("  1. Run a backtest to generate trades.csv:")
    parts.append(
        "     python -m core.backtester -c config.yaml --results-dir <output_dir>"
    )
    parts.append(
        "     (Or use scripts/phaseD3_run_sequence_backtest.py for Phase D sequence backtests.)"
    )
    parts.append("  2. Or use --find-latest-trades to auto-select the most recent trades.csv under results/phaseD")
    parts.append("  3. Or point --trades to an existing trades.csv path")
    return "\n".join(parts)


def _load_labels(path: Path) -> pd.DataFrame:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {path}\n"
            "Run scripts/phaseD1_generate_opportunity_labels.py to produce opportunity_labels.csv."
        )
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    missing = [c for c in LABEL_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Labels missing required columns: {missing}. Found: {list(df.columns)}"
        )
    return df


def _load_trades(path: Path) -> pd.DataFrame:
    """Load trades and normalize to pair, entry_date (YYYY-MM-DD), direction {long,short}."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(_format_trades_not_found_error(path))
    df = pd.read_csv(path)
    if df.empty:
        return df

    pair_col = _resolve_column(df, ["pair"])
    if not pair_col:
        raise ValueError(
            f"Cannot infer 'pair' column. Found columns: {list(df.columns)}"
        )

    entry_col = _resolve_column(df, ENTRY_DATE_ALIASES)
    if not entry_col:
        raise ValueError(
            f"Cannot infer entry date column. Tried: {ENTRY_DATE_ALIASES}. "
            f"Found: {list(df.columns)}"
        )

    dir_col = None
    for alias in DIRECTION_ALIASES:
        c = _resolve_column(df, [alias])
        if c:
            dir_col = c
            break
    if not dir_col:
        raise ValueError(
            f"Cannot infer direction column. Tried: {DIRECTION_ALIASES}. "
            f"Found: {list(df.columns)}"
        )

    out = df[[pair_col, entry_col, dir_col]].copy()
    out = out.rename(columns={pair_col: "pair", entry_col: "entry_date_raw", dir_col: "dir_raw"})
    out["pair"] = out["pair"].astype(str).str.strip()
    out["entry_date"] = pd.to_datetime(out["entry_date_raw"]).dt.normalize()
    out["entry_date"] = out["entry_date"].dt.strftime("%Y-%m-%d")

    raw_dir = out["dir_raw"]
    direction = []
    for v in raw_dir:
        s = str(v).strip().lower() if pd.notna(v) else ""
        try:
            n = int(float(v)) if pd.notna(v) else None
        except (ValueError, TypeError):
            n = None
        if s in ("long", "1", "+1", "bull") or n == 1:
            direction.append("long")
        elif s in ("short", "-1", "bear") or n == -1:
            direction.append("short")
        elif pd.notna(v) and str(v).lower() in ("true", "yes"):
            direction.append("long")
        elif pd.notna(v) and str(v).lower() in ("false", "no"):
            direction.append("short")
        else:
            direction.append(s if s in ("long", "short") else "long")
    out["direction"] = direction
    out = out[["pair", "entry_date", "direction"]]
    return out


def _to_bool(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(bool)


def _build_trade_zone_overlap(
    labels: pd.DataFrame, trades: pd.DataFrame, missing_threshold: float = 0.01
) -> tuple[pd.DataFrame, int]:
    """Join trades to labels. Return (overlap_df, missing_count)."""
    labels["date_str"] = pd.to_datetime(labels["date"]).dt.strftime("%Y-%m-%d")
    merged = trades.merge(
        labels,
        left_on=["pair", "entry_date", "direction"],
        right_on=["pair", "date_str", "direction"],
        how="left",
        suffixes=("", "_label"),
    )
    missing = merged["zone_a_1r_10"].isna()
    missing_count = int(missing.sum())
    total = len(merged)
    if total > 0 and missing_count / total > missing_threshold:
        raise RuntimeError(
            f"Missing label rate {missing_count}/{total} = {missing_count/total:.2%} "
            f"exceeds threshold {missing_threshold:.1%}. Check data alignment."
        )

    out_cols = [
        "pair",
        "entry_date",
        "direction",
        "dataset_split",
        "zoneA_flag_at_entry",
        "zoneB_flag_at_entry",
        "zoneC_flag_at_entry",
        "t3_at_entry",
        "t6_at_entry",
        "mfe_40_r_at_entry",
        "missing_label_row",
    ]
    result = pd.DataFrame()
    result["pair"] = merged["pair"]
    result["entry_date"] = merged["entry_date"]
    result["direction"] = merged["direction"]
    result["dataset_split"] = merged["dataset_split"]
    result["zoneA_flag_at_entry"] = _to_bool(merged["zone_a_1r_10"])
    result["zoneB_flag_at_entry"] = _to_bool(merged["zone_b_3r_20"])
    result["zoneC_flag_at_entry"] = _to_bool(merged["zone_c_6r_40"])
    result["t3_at_entry"] = merged["t3"]
    result["t6_at_entry"] = merged["t6"]
    result["mfe_40_r_at_entry"] = merged["mfe_40_r"]
    result["missing_label_row"] = (merged["zone_a_1r_10"].isna()).astype(int)
    return result[out_cols], missing_count


def _find_zone_events(
    labels: pd.DataFrame, zone_type: str
) -> pd.DataFrame:
    """Find consecutive True segments per (pair, direction) for zone_type B or C."""
    zone_col = "zone_b_3r_20" if zone_type == "B" else "zone_c_6r_40"
    labels = labels.copy()
    labels["date_str"] = pd.to_datetime(labels["date"]).dt.strftime("%Y-%m-%d")
    labels = labels.sort_values(["pair", "direction", "date"]).reset_index(drop=True)
    labels["_zone"] = _to_bool(labels[zone_col])
    labels["_run"] = (labels["_zone"] != labels["_zone"].shift(1)).cumsum()

    events = []
    for (pair, direction, run_id), grp in labels.groupby(["pair", "direction", "_run"]):
        if not grp["_zone"].iloc[0]:
            continue
        start_date = grp["date"].min()
        end_date = grp["date"].max()
        duration_bars = len(grp)
        dataset_split = grp["dataset_split"].mode().iloc[0] if len(grp["dataset_split"].mode()) else "unknown"
        mfe_at_start = grp[grp["date"] == start_date]["mfe_40_r"].iloc[0] if zone_type == "C" else np.nan
        events.append({
            "pair": pair,
            "direction": direction,
            "zone_type": zone_type,
            "dataset_split": dataset_split,
            "start_date": start_date,
            "end_date": end_date,
            "duration_bars": duration_bars,
            "mfe_40_r_at_start": mfe_at_start if zone_type == "C" else np.nan,
            "_dates": grp["date_str"].tolist(),
        })
    return events


def _is_event_captured(
    event: dict, trades: pd.DataFrame, capture_bars: int
) -> tuple[bool, str | None, int | None, float | None, float | None]:
    """Return (captured, first_entry_date, entry_delay_bars, t3, t6)."""
    pair = event["pair"]
    direction = event["direction"]
    dates = event["_dates"][:capture_bars]
    subset = trades[(trades["pair"] == pair) & (trades["direction"] == direction)]
    for i, d in enumerate(dates):
        hits = subset[subset["entry_date"] == d]
        if not hits.empty:
            return True, d, i, None, None
    return False, None, None, None, None


def _build_zone_capture_summary(
    labels: pd.DataFrame, trades: pd.DataFrame, capture_bars: int
) -> pd.DataFrame:
    """Build zone_capture_summary.csv with event capture info."""
    labels_with_str = labels.copy()
    labels_with_str["date_str"] = pd.to_datetime(labels_with_str["date"]).dt.strftime("%Y-%m-%d")

    rows = []
    for zone_type in ("B", "C"):
        events = _find_zone_events(labels_with_str, zone_type)
        for ev in events:
            captured, first_entry_date, delay_bars, _, _ = _is_event_captured(
                ev, trades, capture_bars
            )
            t3_at_entry = np.nan
            t6_at_entry = np.nan
            if captured and first_entry_date:
                match = labels_with_str[
                    (labels_with_str["pair"] == ev["pair"])
                    & (labels_with_str["direction"] == ev["direction"])
                    & (labels_with_str["date_str"] == first_entry_date)
                ]
                if not match.empty:
                    t3_at_entry = match["t3"].iloc[0]
                    t6_at_entry = match["t6"].iloc[0]

            rows.append({
                "pair": ev["pair"],
                "direction": ev["direction"],
                "zone_type": zone_type,
                "dataset_split": ev["dataset_split"],
                "start_date": ev["start_date"],
                "end_date": ev["end_date"],
                "duration_bars": ev["duration_bars"],
                "captured": captured,
                "first_entry_date": first_entry_date,
                "entry_delay_bars": delay_bars,
                "t3_at_entry": t3_at_entry,
                "t6_at_entry": t6_at_entry,
            })
    return pd.DataFrame(rows)


def _nearest_signal_distance(
    event: dict, trades: pd.DataFrame, label_dates: list[str]
) -> tuple[int, str | None, bool]:
    """Return (nearest_signal_distance_bars, nearest_signal_date, nearest_signal_is_after)."""
    pair = event["pair"]
    direction = event["direction"]
    subset = trades[(trades["pair"] == pair) & (trades["direction"] == direction)]
    if subset.empty:
        return -1, None, False

    start_date = event["_dates"][0]
    try:
        start_idx = label_dates.index(start_date)
    except ValueError:
        return -1, None, False

    entry_dates = subset["entry_date"].unique().tolist()
    min_dist = -1
    nearest_date = None
    is_after = False
    for ed in entry_dates:
        try:
            entry_idx = label_dates.index(ed)
        except ValueError:
            continue
        dist = abs(entry_idx - start_idx)
        if min_dist < 0 or dist < min_dist:
            min_dist = dist
            nearest_date = ed
            is_after = entry_idx > start_idx
    return min_dist, nearest_date, is_after


def _build_missed_mega_report(
    labels: pd.DataFrame, trades: pd.DataFrame, capture_bars: int, top_n: int
) -> pd.DataFrame:
    """Build missed_mega_report.csv for top Zone C events per pair."""
    labels_temp = labels.copy()
    labels_temp["date_str"] = pd.to_datetime(labels_temp["date"]).dt.strftime("%Y-%m-%d")
    events = _find_zone_events(labels_temp, "C")
    per_pair_dates: dict[tuple[str, str], list[str]] = {}
    for (pair, direction), grp in labels_temp.groupby(["pair", "direction"]):
        per_pair_dates[(pair, direction)] = grp.sort_values("date")["date_str"].tolist()

    by_pair: dict[str, list] = {}
    for ev in events:
        p = ev["pair"]
        if p not in by_pair:
            by_pair[p] = []
        by_pair[p].append(ev)

    rows = []
    for pair, evs in by_pair.items():
        evs_sorted = sorted(evs, key=lambda e: (e.get("mfe_40_r_at_start", 0) or 0), reverse=True)
        for ev in evs_sorted[:top_n]:
            captured, first_entry_date, delay_bars, _, t6 = _is_event_captured(
                ev, trades, capture_bars
            )
            t6_val = None
            if captured and first_entry_date:
                label_lookup = labels[
                    (labels["pair"] == ev["pair"])
                    & (labels["direction"] == ev["direction"])
                ]
                label_lookup = label_lookup[label_lookup["date"].dt.strftime("%Y-%m-%d") == first_entry_date]
                if not label_lookup.empty:
                    t6_val = label_lookup["t6"].iloc[0]

            if captured:
                rows.append({
                    "pair": pair,
                    "direction": ev["direction"],
                    "zone_type": "C",
                    "start_date": ev["start_date"],
                    "event_magnitude_R": ev.get("mfe_40_r_at_start", np.nan),
                    "captured": True,
                    "first_entry_date": first_entry_date,
                    "entry_delay_bars": delay_bars,
                    "t6_at_entry": t6_val,
                    "nearest_signal_distance_bars": np.nan,
                    "nearest_signal_date": None,
                    "nearest_signal_is_after": np.nan,
                })
            else:
                dates_list = per_pair_dates.get((pair, ev["direction"]), [])
                dist, nearest_date, is_after = _nearest_signal_distance(ev, trades, dates_list)
                rows.append({
                    "pair": pair,
                    "direction": ev["direction"],
                    "zone_type": "C",
                    "start_date": ev["start_date"],
                    "event_magnitude_R": ev.get("mfe_40_r_at_start", np.nan),
                    "captured": False,
                    "first_entry_date": None,
                    "entry_delay_bars": np.nan,
                    "t6_at_entry": np.nan,
                    "nearest_signal_distance_bars": dist if dist >= 0 else np.nan,
                    "nearest_signal_date": nearest_date,
                    "nearest_signal_is_after": 1 if is_after else 0,
                })
    return pd.DataFrame(rows)


def _build_discovery_vs_validation(
    overlap: pd.DataFrame, capture_summary: pd.DataFrame
) -> pd.DataFrame:
    """Build discovery_vs_validation_comparison.csv."""
    valid = overlap["missing_label_row"] == 0
    ov = overlap[valid]

    metrics = []
    for split in ("discovery", "validation"):
        m = ov[ov["dataset_split"] == split]
        m_b = capture_summary[(capture_summary["dataset_split"] == split) & (capture_summary["zone_type"] == "B")]
        m_c = capture_summary[(capture_summary["dataset_split"] == split) & (capture_summary["zone_type"] == "C")]

        trade_count = len(m)
        in_zoneB = _to_bool(m["zoneB_flag_at_entry"]).sum()
        in_zoneC = _to_bool(m["zoneC_flag_at_entry"]).sum()
        pct_B = in_zoneB / trade_count if trade_count else 0
        pct_C = in_zoneC / trade_count if trade_count else 0
        m_zoneC = m[_to_bool(m["zoneC_flag_at_entry"])]
        avg_t3 = m["t3_at_entry"].mean() if trade_count and "t3_at_entry" in m.columns else np.nan
        avg_t6 = m["t6_at_entry"].mean() if trade_count and "t6_at_entry" in m.columns else np.nan
        avg_t3_zoneC = m_zoneC["t3_at_entry"].mean() if len(m_zoneC) else np.nan
        avg_t6_zoneC = m_zoneC["t6_at_entry"].mean() if len(m_zoneC) else np.nan

        b_captured = m_b[m_b["captured"]]
        c_captured = m_c[m_c["captured"]]
        b_count = len(m_b)
        c_count = len(m_c)
        b_pct = len(b_captured) / b_count if b_count else 0
        c_pct = len(c_captured) / c_count if c_count else 0
        b_delay = b_captured["entry_delay_bars"].mean() if len(b_captured) else np.nan
        c_delay = c_captured["entry_delay_bars"].mean() if len(c_captured) else np.nan
        b_t6 = b_captured["t6_at_entry"].mean() if len(b_captured) else np.nan
        c_t6 = c_captured["t6_at_entry"].mean() if len(c_captured) else np.nan

        metrics.append({
            "dataset_split": split,
            "trade_count": trade_count,
            "pct_trades_in_zoneB": pct_B,
            "pct_trades_in_zoneC": pct_C,
            "avg_t3_at_entry": avg_t3,
            "avg_t6_at_entry": avg_t6,
            "avg_t3_at_entry_zoneC": avg_t3_zoneC,
            "avg_t6_at_entry_zoneC": avg_t6_zoneC,
            "zoneB_event_count": b_count,
            "zoneB_pct_events_captured": b_pct,
            "zoneB_avg_entry_delay_bars": b_delay,
            "zoneB_avg_t6_at_entry_captured": b_t6,
            "zoneC_event_count": c_count,
            "zoneC_pct_events_captured": c_pct,
            "zoneC_avg_entry_delay_bars": c_delay,
            "zoneC_avg_t6_at_entry_captured": c_t6,
        })
    return pd.DataFrame(metrics)


def _write_decision_memo(
    out_dir: Path, capture_summary: pd.DataFrame, overlap: pd.DataFrame
) -> None:
    """Write decision_memo.txt with Zone C interpretation."""
    c_events = capture_summary[capture_summary["zone_type"] == "C"]
    c_captured = c_events[c_events["captured"]]
    c_t6 = c_captured["t6_at_entry"].dropna()

    lines = []
    for split in ("discovery", "validation"):
        m = c_events[c_events["dataset_split"] == split]
        capt = m[m["captured"]]
        total = len(m)
        rate = len(capt) / total if total else 0
        lines.append(f"Zone C capture rate ({split}): {rate:.1%}")

        if rate >= 0.60:
            lines.append("  => Entry captures most large trends; geometry/tail capture is bottleneck.")
        elif rate <= 0.30:
            lines.append("  => Entry misses large trends; structure must change.")
        else:
            lines.append("  => Hybrid redesign required.")

    median_t6 = c_t6.median() if len(c_t6) else np.nan
    lines.append("")
    if pd.notna(median_t6) and median_t6 <= 2:
        lines.append("Timing lag note: median t6_at_entry for captured Zone C events is <= 2 bars — entry lag problem.")
    else:
        lines.append("Timing lag note: median t6_at_entry for captured Zone C events > 2 bars — no entry lag problem.")

    (out_dir / "decision_memo.txt").write_text("\n".join(lines), encoding="utf-8")


def run_phaseD5(
    labels_path: Path | str,
    trades_path: Path | str,
    out_dir: Path | str,
    *,
    capture_bars: int = 5,
    top_n: int = 10,
) -> dict[str, Path]:
    """
    Main Phase D-5 entry point. Produces 4 CSVs + decision_memo.txt.
    """
    labels_path = Path(labels_path)
    trades_path = Path(trades_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = _load_labels(labels_path)
    trades = _load_trades(trades_path)

    overlap, missing_count = _build_trade_zone_overlap(labels, trades)
    overlap_path = out_dir / "trade_zone_overlap.csv"
    overlap.to_csv(overlap_path, index=False)

    capture_summary = _build_zone_capture_summary(labels, trades, capture_bars)
    capture_path = out_dir / "zone_capture_summary.csv"
    capture_summary.to_csv(capture_path, index=False)

    missed_mega = _build_missed_mega_report(labels, trades, capture_bars, top_n)
    missed_path = out_dir / "missed_mega_report.csv"
    missed_mega.to_csv(missed_path, index=False)

    discovery_validation = _build_discovery_vs_validation(overlap, capture_summary)
    dv_path = out_dir / "discovery_vs_validation_comparison.csv"
    discovery_validation.to_csv(dv_path, index=False)

    _write_decision_memo(out_dir, capture_summary, overlap)

    return {
        "trade_zone_overlap": overlap_path,
        "zone_capture_summary": capture_path,
        "missed_mega_report": missed_path,
        "discovery_vs_validation_comparison": dv_path,
        "decision_memo": out_dir / "decision_memo.txt",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase D-5 Opportunity Capture Attribution (analytics only)"
    )
    parser.add_argument("--labels", required=True, help="Path to opportunity_labels.csv or .parquet")
    parser.add_argument("--trades", help="Path to trades.csv (optional if --find-latest-trades)")
    parser.add_argument(
        "--find-latest-trades",
        action="store_true",
        help="Search results/phaseD for most recent trades.csv and use it",
    )
    parser.add_argument(
        "--outdir",
        default="results/phaseD/d5_capture_attribution",
        help="Output directory (default: results/phaseD/d5_capture_attribution)",
    )
    parser.add_argument("--capture-bars", type=int, default=5, help="Capture window in bars (default: 5)")
    parser.add_argument("--top-n", type=int, default=10, help="Top N mega events per pair (default: 10)")
    args = parser.parse_args()

    trades_path: str | Path
    if args.find_latest_trades:
        root = Path("results/phaseD").resolve()
        found = _find_latest_trades_under(root)
        if found is None:
            raise FileNotFoundError(
                f"No valid trades.csv found under {root}. "
                "Run a backtest first: python -m core.backtester -c config.yaml --results-dir <dir>"
            )
        trades_path = found
    elif args.trades:
        trades_path = Path(args.trades)
        if not trades_path.exists():
            raise FileNotFoundError(_format_trades_not_found_error(trades_path))
    else:
        raise ValueError("Provide --trades <path> or --find-latest-trades")

    run_phaseD5(
        labels_path=args.labels,
        trades_path=trades_path,
        out_dir=args.outdir,
        capture_bars=args.capture_bars,
        top_n=args.top_n,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
