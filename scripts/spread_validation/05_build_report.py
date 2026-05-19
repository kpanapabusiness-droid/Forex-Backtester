"""05_build_report.py — Assemble spread_validation_report.md from pipeline CSVs."""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
SV_DIR: Path = REPO_ROOT / "results" / "spread_validation"

DIST_CSV: Path = SV_DIR / "per_pair_distributions.csv"
EXEC_CSV: Path = SV_DIR / "execution_bar_spreads.csv"
SUMMARY_CSV: Path = SV_DIR / "per_arc_summary.csv"
KH_CSV: Path = SV_DIR / "per_trade_impact_kh24.csv"
A4_CSV: Path = SV_DIR / "per_trade_impact_arc4.csv"
A5_CSV: Path = SV_DIR / "per_trade_impact_arc5.csv"

OUTPUT_PATH: Path = SV_DIR / "spread_validation_report.md"

CURRENT_FLOOR_PIPS: float = 0.1


def fmt_num(x, digits: int = 4) -> str:
    if x is None:
        return "—"
    try:
        if isinstance(x, str):
            if x == "":
                return "—"
            x = float(x)
        if np.isnan(x):
            return "—"
        return f"{x:.{digits}g}"
    except Exception:
        return str(x)


def read_csv_safe(p: Path) -> pd.DataFrame | None:
    if not p.exists() or p.stat().st_size == 0:
        return None
    return pd.read_csv(p)


def build_summary_table(exec_df: pd.DataFrame) -> str:
    out = ["| Pair | TF | n bars | real_p50 (first5min) | real_p95 (first5min) | floor | gap_p50-floor |",
           "|------|----|--------|---------------------|---------------------|-------|---------------|"]
    for _, row in exec_df.iterrows():
        out.append(
            f"| {row['pair']} | {row['tf']} | {int(row['n_execution_bars'])} | "
            f"{fmt_num(row['p50_spread_pips_first5min'])} | "
            f"{fmt_num(row['p95_spread_pips_first5min'])} | "
            f"{fmt_num(row['current_floor_pips'])} | "
            f"{fmt_num(row['gap_p50_first5min_minus_floor_pips'])} |"
        )
    return "\n".join(out)


def build_arc_summary(summary_df: pd.DataFrame) -> str:
    out = ["| Arc | Fold | n trades | n evaluated | % entry-floor | % exit-floor | p25 under | p50 under | p75 under | p95 under | mean under_R | total under % equity |",
           "|-----|------|----------|-------------|---------------|--------------|-----------|-----------|-----------|-----------|--------------|----------------------|"]
    for _, row in summary_df.iterrows():
        out.append(
            f"| {row['arc']} | {row['fold']} | {int(row['n_trades_total'])} | "
            f"{int(row['n_trades_evaluated'])} | "
            f"{fmt_num(row['pct_entry_floor_activated'])} | "
            f"{fmt_num(row['pct_exit_floor_activated'])} | "
            f"{fmt_num(row['p25_under_pips'])} | "
            f"{fmt_num(row['p50_under_pips'])} | "
            f"{fmt_num(row['p75_under_pips'])} | "
            f"{fmt_num(row['p95_under_pips'])} | "
            f"{fmt_num(row['mean_under_R'])} | "
            f"{fmt_num(row['total_under_pct_equity'])} |"
        )
    return "\n".join(out)


def build_cross_pair_section(exec_df: pd.DataFrame) -> str:
    rows_1h = exec_df[exec_df["tf"] == "1H"].copy()
    if rows_1h.empty:
        return "(no 1H execution-bar data)"
    rows_1h["gap"] = rows_1h["gap_p50_first5min_minus_floor_pips"].astype(float)
    too_low_1h = rows_1h[rows_1h["gap"] > 0].sort_values("gap", ascending=False)
    pct_too_low = len(too_low_1h) / len(rows_1h) * 100.0

    worst3 = too_low_1h.head(3)[["pair", "p50_spread_pips_first5min", "gap"]]
    lines = [
        f"- 1H execution bars: {len(too_low_1h)} of {len(rows_1h)} pairs ({pct_too_low:.0f}%) have real p50 > current floor (0.1 pip).",
        f"- Worst 3 TOO_LOW pairs (by gap_p50_first5min_minus_floor):",
    ]
    for _, r in worst3.iterrows():
        lines.append(
            f"  - **{r['pair']}** — real p50 first5min = {fmt_num(r['p50_spread_pips_first5min'])} pip, gap = {fmt_num(r['gap'])} pip"
        )
    return "\n".join(lines)


def main() -> int:
    exec_df = read_csv_safe(EXEC_CSV)
    summary_df = read_csv_safe(SUMMARY_CSV)
    dist_df = read_csv_safe(DIST_CSV)

    lines: list[str] = []
    lines.append("# Spread Validation Report — Active-Session Bid/Ask vs Modeled Spread")
    lines.append("")
    lines.append(f"> Generated: {datetime.now(timezone.utc).isoformat()}  ")
    lines.append("> Source: HistData ASCII tick data, 2024-01 → 2025-12 (24 months × 28 pairs).  ")
    lines.append("> Locked floor reference: `configs/spread_floors_5ers.yaml` (uniform 0.1 pip).")
    lines.append("")
    lines.append("## 1. TL;DR")
    lines.append("")
    lines.append("The locked `spread_floors_5ers.yaml` applies a uniform 0.1 pip floor across")
    lines.append("all 28 pairs (`min_nonzero_spread_native: 1` in raw MT5 points, /10 → 0.1 pip).")
    lines.append("KH-24's `configs/wfo_kh24.yaml` does **not** reference the floor — runtime")
    lines.append("uses raw MT5 per-bar spread directly. L arc configs (l_arc_4.yaml etc.) DO")
    lines.append("apply the 0.1 pip floor.")
    lines.append("")
    lines.append("This report quantifies the gap between modeled and real active-session")
    lines.append("spread, and reconciles per-arc fold ROI against under-modeled spread cost.")
    lines.append("")

    lines.append("## 2. Summary — 28 pairs × {1H, 4H}")
    lines.append("")
    if exec_df is not None and not exec_df.empty:
        lines.append(build_summary_table(exec_df))
    else:
        lines.append("_(execution_bar_spreads.csv not yet produced)_")
    lines.append("")

    lines.append("## 3. Per-arc trade-impact reconciliation")
    lines.append("")
    lines.append("For each evaluated trade, `under_payment_pips = (real_entry − modeled_entry) + (real_exit − modeled_exit)`.")
    lines.append("`under_R = under_payment_pips / sl_distance_pips`. `under_pct_equity = under_R × risk_per_trade_pct`.")
    lines.append("Trades with `entry_time` or `exit_time` outside 2024-01-01 → 2026-01-01 are excluded from evaluation.")
    lines.append("")
    if summary_df is not None and not summary_df.empty:
        lines.append(build_arc_summary(summary_df))
    else:
        lines.append("_(per_arc_summary.csv not yet produced)_")
    lines.append("")

    lines.append("### Worst-fold reconciliation (KH-24 only)")
    lines.append("")
    lines.append("- KH-24's published worst fold: Fold 7 (OOS 2025-04-01 → 2026-01-01), ROI **+1.92%**, DD 6.37% (F1).")
    lines.append("- After subtracting under-modeled spread cost on Fold 7 trades: see `total_under_pct_equity` for Fold 7 in the table above.")
    lines.append("- **Net Fold 7 ROI estimate after correction: (+1.92% − {Fold 7 under_pct_equity})**.")
    lines.append("- Gate threshold for PASS-DEPLOYABLE: worst-fold ROI > 5% net. Threshold for PASS-VIABLE: > 0%.")
    lines.append("")

    lines.append("## 4. Cross-pair patterns (1H execution bars)")
    lines.append("")
    if exec_df is not None and not exec_df.empty:
        lines.append(build_cross_pair_section(exec_df))
    else:
        lines.append("_(execution_bar_spreads.csv not yet produced)_")
    lines.append("")

    lines.append("## 5. Per-pair distribution detail")
    lines.append("")
    lines.append("Full per-pair × per-session distributions are in `per_pair_distributions.csv` (long format).")
    lines.append("Sessions: london (07-12 UTC), ny (16-21), overlap (12-16), weekend_edge (Fri 21h + Sun 22-23h), off_hours otherwise.")
    lines.append("")
    if dist_df is not None and not dist_df.empty:
        n_pairs = dist_df["pair"].nunique()
        n_rows = len(dist_df)
        lines.append(f"_File contains {n_rows} rows across {n_pairs} pairs._")
    lines.append("")

    lines.append("## 6. Methodology")
    lines.append("")
    lines.append("**Source**: HistData ASCII tick data (`tick-data-quotes` for `ascii` platform). Format `YYYYMMDD HHMMSSnnn,bid,ask,vol` per tick. Timestamps EST (UTC-5, no DST per HistData docs) → converted to UTC.")
    lines.append("")
    lines.append("**Pip conversion**: `spread_pips = (ask - bid) × 10000` for non-JPY pairs; `× 100` for JPY pairs.")
    lines.append("")
    lines.append("**Per-hour aggregate** (Phase 2b): for each (pair, hour), percentiles {10, 25, 50, 75, 90, 95, 99} of per-tick spread_pips, plus first-5-minute window median/mean.")
    lines.append("")
    lines.append("**Execution-bar metric**: `first5min_median_spread_pips` is the median spread of ticks in the first 5 minutes of the bar — proxy for the spread an order placed at bar open would face.")
    lines.append("")
    lines.append("**Trade-impact (Phase 4)**: For each trade row, look up `first5min_median_spread_pips` at entry/exit bar opens. Modeled spread for KH-24 = raw MT5 bar spread / 10 (no floor). Modeled spread for L arcs = `spread_pips_used` / `spread_pips_exit` in the trade log (already includes the 0.1 pip floor at runtime).")
    lines.append("")
    lines.append("**R definition**: `1R = sl_distance_pips` per trade. KH-24 derives it from `sl_distance_atr × atr_abs / pip_size_in_price`. L arcs have `sl_distance_pips` directly.")
    lines.append("")
    lines.append("**Risk-per-trade**: KH-24 = 1.0% per R; L arcs = 0.5% per R.")
    lines.append("")
    lines.append("**Determinism**: every CSV is written with `lineterminator='\\n'` and `%.10g` precision. Pair iteration order alphabetic. Re-running the pipeline on the same cache produces byte-identical CSVs.")
    lines.append("")

    lines.append("## 7. Limitations")
    lines.append("")
    lines.append("- **HistData TLS cert was expired** (notAfter 2026-05-02; download 2026-05-17). Scraping used `requests.Session(verify=False)` scoped to histdata.com; data is public CSV, no credentials sent.")
    lines.append("- **HistData publishes liquidity-provider quotes**, which may be tighter than 5ers retail spreads. To validate, a one-week MT5 bid/ask snapshot comparison is recommended as follow-up.")
    lines.append("- **Window**: 2024-01 → 2025-12 only. Trades before 2024-01 are reported as `OUTSIDE_WINDOW`. KH-24 Folds 1-5 + most of Fold 6 are outside the window; Fold 7 (2025-04 → 2026-01) is fully inside.")
    lines.append("- **EST timezone**: HistData docs state no DST. We use UTC-5 year-round.")
    lines.append("")
    lines.append("## 8. Output files")
    lines.append("")
    lines.append("- `per_pair_distributions.csv` — per-pair × session × percentile (long)")
    lines.append("- `execution_bar_spreads.csv` — per-pair × tf summary (wide)")
    lines.append("- `per_trade_impact_kh24.csv` — KH-24 trades, per-trade gap")
    lines.append("- `per_trade_impact_arc4.csv` — L Arc 4 trades")
    lines.append("- `per_trade_impact_arc5.csv` — L Arc 5 trades")
    lines.append("- `per_arc_summary.csv` — per-arc / per-fold summary")
    lines.append("")
    lines.append("This report does not propose floor changes. Governance: see `L_ARC_PROTOCOL.md` §12.")
    lines.append("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
