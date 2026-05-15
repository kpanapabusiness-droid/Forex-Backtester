"""L4 magnitude distribution descriptives.

Reads results/l6/characterisation/signals_features.csv and writes a single
markdown report at results/l6/characterisation/magnitude_distribution.md.

No RNG, no new pipeline, no locked artefact modification. Pure pandas
describe + groupby.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("results/l6/characterisation/signals_features.csv")
DST = Path("results/l6/characterisation/magnitude_distribution.md")

R_QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
WIN_QUANTILES = [0.50, 0.75, 0.90, 0.95, 0.99]
LOSS_QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50]
EXC_QUANTILES = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
HORIZON_QUANTILES = [0.50, 0.75, 0.90, 0.95, 0.99]
HORIZONS = [1, 6, 24, 72, 120, 240]


def fmt(x, dp: int = 4) -> str:
    if pd.isna(x):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{float(x):.{dp}f}"


def r_distribution_row(s: pd.Series) -> dict:
    s = s.dropna()
    return {
        "count": int(s.shape[0]),
        "mean": s.mean(),
        "std": s.std(),
        "skew": s.skew(),
        "kurtosis": s.kurtosis(),
        "min": s.min(),
        "p1": s.quantile(0.01),
        "p5": s.quantile(0.05),
        "p25": s.quantile(0.25),
        "p50": s.quantile(0.50),
        "p75": s.quantile(0.75),
        "p95": s.quantile(0.95),
        "p99": s.quantile(0.99),
        "max": s.max(),
    }


def section_1(df: pd.DataFrame, lines: list[str]) -> None:
    lines.append("## 1. Net R / Gross R distribution\n")
    cols = [
        "count",
        "mean",
        "std",
        "skew",
        "kurtosis",
        "min",
        "p1",
        "p5",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
        "max",
    ]
    header = "| stat | " + " | ".join(cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for label, col in [("net_r", "net_r"), ("gross_r", "gross_r")]:
        row = r_distribution_row(df[col])
        cells = [label] + [fmt(row[c], 0 if c == "count" else 4) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")


def section_2(df: pd.DataFrame, lines: list[str]) -> None:
    lines.append("## 2. Conditional means\n")
    n_total = len(df)

    lines.append("### 2a. By exit_reason (sl, time)\n")
    lines.append("| group | count | pct_pop | mean_net_r | mean_gross_r |")
    lines.append("|---|---|---|---|---|")
    for er in ["sl", "time"]:
        sub = df[df["exit_reason"] == er]
        cnt = len(sub)
        pct = cnt / n_total if n_total else float("nan")
        mn_net = sub["net_r"].mean() if cnt else float("nan")
        mn_gross = sub["gross_r"].mean() if cnt else float("nan")
        lines.append(f"| {er} | {fmt(cnt, 0)} | {fmt(pct, 4)} | {fmt(mn_net)} | {fmt(mn_gross)} |")
    # also show "other" exit reasons if any
    other = df[~df["exit_reason"].isin(["sl", "time"])]
    if len(other):
        cnt = len(other)
        pct = cnt / n_total
        lines.append(
            f"| other | {fmt(cnt, 0)} | {fmt(pct, 4)} | "
            f"{fmt(other['net_r'].mean())} | {fmt(other['gross_r'].mean())} |"
        )
    lines.append("")

    lines.append("### 2b. Winners (net_r > 0) vs Losers (net_r <= 0)\n")
    win = df[df["net_r"] > 0]
    lose = df[df["net_r"] <= 0]
    lines.append("| group | count | pct_pop | mean_net_r | mean_gross_r |")
    lines.append("|---|---|---|---|---|")
    for label, sub in [("winners", win), ("losers", lose)]:
        cnt = len(sub)
        pct = cnt / n_total if n_total else float("nan")
        mn_net = sub["net_r"].mean() if cnt else float("nan")
        mn_gross = sub["gross_r"].mean() if cnt else float("nan")
        lines.append(
            f"| {label} | {fmt(cnt, 0)} | {fmt(pct, 4)} | {fmt(mn_net)} | {fmt(mn_gross)} |"
        )
    lines.append("")

    lines.append("### 2c. Winners-only net_r quantiles\n")
    lines.append("| p50 | p75 | p90 | p95 | p99 | max |")
    lines.append("|---|---|---|---|---|---|")
    s = win["net_r"]
    qs = [s.quantile(q) for q in WIN_QUANTILES] + [s.max() if len(s) else float("nan")]
    lines.append("| " + " | ".join(fmt(v) for v in qs) + " |")
    lines.append("")

    lines.append("### 2d. Losers-only net_r quantiles\n")
    lines.append("| p1 | p5 | p10 | p25 | p50 |")
    lines.append("|---|---|---|---|---|")
    s = lose["net_r"]
    qs = [s.quantile(q) for q in LOSS_QUANTILES]
    lines.append("| " + " | ".join(fmt(v) for v in qs) + " |")
    lines.append("")


def section_3(df: pd.DataFrame, lines: list[str]) -> None:
    lines.append("## 3. Held-bar MFE / MAE by exit_reason (ATR units)\n")
    qs = EXC_QUANTILES
    cols = ["p25", "p50", "p75", "p90", "p95", "p99", "max"]
    for col in ["mfe_held_atr", "mae_held_atr"]:
        lines.append(f"### {col}\n")
        header = "| exit_reason | count | " + " | ".join(cols) + " |"
        lines.append(header)
        lines.append("|---" * (len(cols) + 2) + "|")
        for er, sub in df.groupby("exit_reason"):
            s = sub[col].dropna()
            row_vals = [s.quantile(q) for q in qs] + [s.max() if len(s) else float("nan")]
            lines.append(
                f"| {er} | {fmt(len(sub), 0)} | " + " | ".join(fmt(v) for v in row_vals) + " |"
            )
        lines.append("")


def section_4(df: pd.DataFrame, lines: list[str]) -> None:
    lines.append("## 4. Wasted favourable excursion (time-exit only)\n")
    lines.append(
        "Definition: `max(0, mfe_held_atr - max(0, fwd_logret_h1 / atr_1h_at_n))` "
        "for trades with `exit_reason == 'time'`.\n"
    )
    sub = df[df["exit_reason"] == "time"].copy()
    captured = (sub["fwd_logret_h1"] / sub["atr_1h_at_n"]).clip(lower=0)
    wasted = (sub["mfe_held_atr"] - captured).clip(lower=0)
    wasted = wasted.dropna()
    lines.append("| count | mean | p50 | p75 | p95 | max |")
    lines.append("|---|---|---|---|---|---|")
    if len(wasted):
        lines.append(
            f"| {fmt(len(wasted), 0)} | {fmt(wasted.mean())} | "
            f"{fmt(wasted.quantile(0.5))} | {fmt(wasted.quantile(0.75))} | "
            f"{fmt(wasted.quantile(0.95))} | {fmt(wasted.max())} |"
        )
    else:
        lines.append("| 0 | NA | NA | NA | NA | NA |")
    lines.append("")


def section_5(df: pd.DataFrame, lines: list[str]) -> None:
    lines.append("## 5. Forward horizon distribution in ATR units\n")
    lines.append(
        "Computed as `fwd_logret_h{H} / atr_1h_at_n`. Quantiles p50/p75/p90/p95/p99/max.\n"
    )

    qs = HORIZON_QUANTILES

    lines.append("### 5a. Pooled (all signals)\n")
    cols = ["p50", "p75", "p90", "p95", "p99", "max"]
    lines.append("| horizon_h | count | " + " | ".join(cols) + " |")
    lines.append("|---" * (len(cols) + 2) + "|")
    for H in HORIZONS:
        col = f"fwd_logret_h{H}"
        if col not in df.columns:
            continue
        s = (df[col] / df["atr_1h_at_n"]).dropna()
        row_vals = [s.quantile(q) for q in qs] + [s.max() if len(s) else float("nan")]
        lines.append(f"| {H} | {fmt(len(s), 0)} | " + " | ".join(fmt(v) for v in row_vals) + " |")
    lines.append("")

    lines.append("### 5b. Per structural_pattern\n")
    for H in HORIZONS:
        col = f"fwd_logret_h{H}"
        if col not in df.columns:
            continue
        lines.append(f"#### horizon_h = {H}\n")
        lines.append("| structural_pattern | count | " + " | ".join(cols) + " |")
        lines.append("|---" * (len(cols) + 2) + "|")
        tmp = df.copy()
        tmp["_v"] = tmp[col] / tmp["atr_1h_at_n"]
        for sp, sub in tmp.groupby("structural_pattern", dropna=False):
            s = sub["_v"].dropna()
            if not len(s):
                continue
            row_vals = [s.quantile(q) for q in qs] + [s.max()]
            label = "NaN" if pd.isna(sp) else str(sp)
            lines.append(
                f"| {label} | {fmt(len(s), 0)} | " + " | ".join(fmt(v) for v in row_vals) + " |"
            )
        lines.append("")


def section_6(df: pd.DataFrame, lines: list[str]) -> None:
    lines.append("## 6. SL hit rate by structural_pattern, session, mtf_alignment\n")
    df = df.copy()
    df["_sl"] = (df["exit_reason"] == "sl").astype(int)

    for col in ["structural_pattern", "session", "mtf_alignment"]:
        lines.append(f"### 6.{col}\n")
        lines.append(f"| {col} | count | sl_count | sl_hit_rate |")
        lines.append("|---|---|---|---|")
        if col not in df.columns:
            lines.append("| (column missing) | NA | NA | NA |")
            lines.append("")
            continue
        grp = df.groupby(col, dropna=False)
        for key, sub in grp:
            cnt = len(sub)
            sl_cnt = int(sub["_sl"].sum())
            rate = sl_cnt / cnt if cnt else float("nan")
            label = "NaN" if pd.isna(key) else str(key)
            lines.append(f"| {label} | {fmt(cnt, 0)} | {fmt(sl_cnt, 0)} | {fmt(rate)} |")
        lines.append("")


def main() -> None:
    df = pd.read_csv(SRC)

    lines: list[str] = []
    lines.append("# L4 magnitude distribution\n")
    lines.append(f"Source: `{SRC.as_posix()}`  ")
    lines.append(f"Rows: {len(df):,}  ")
    lines.append("Generator: `scripts/lchar/magnitude_distribution.py` (deterministic, no RNG)\n")

    section_1(df, lines)
    section_2(df, lines)
    section_3(df, lines)
    section_4(df, lines)
    section_5(df, lines)
    section_6(df, lines)

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text("\n".join(lines), encoding="utf-8")

    sha = hashlib.sha256(DST.read_bytes()).hexdigest()
    print(f"wrote {DST.as_posix()}")
    print(f"sha256 {sha}")


if __name__ == "__main__":
    main()
