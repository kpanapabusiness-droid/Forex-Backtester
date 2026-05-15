"""Generate aggregated reports from signals_features.csv.

Reports produced (all in `results/l6/characterisation/`):
- regime_breakdown.csv
- forward_horizon_curves.csv
- pair_breakdown.csv
- feature_lag_audit.txt
- characterisation_report.md (narrative)

Disposition: descriptive only. No PASS/FAIL, no recommendations, no system
suggestions. Pure aggregations of the per-signal feature matrix.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lchar.compute_spread_floors import compute_body_sha256  # noqa: E402

ALL_PAIRS: Tuple[str, ...] = (
    "AUD_CAD",
    "AUD_CHF",
    "AUD_JPY",
    "AUD_NZD",
    "AUD_USD",
    "CAD_CHF",
    "CAD_JPY",
    "CHF_JPY",
    "EUR_AUD",
    "EUR_CAD",
    "EUR_CHF",
    "EUR_GBP",
    "EUR_JPY",
    "EUR_NZD",
    "EUR_USD",
    "GBP_AUD",
    "GBP_CAD",
    "GBP_CHF",
    "GBP_JPY",
    "GBP_NZD",
    "GBP_USD",
    "NZD_CAD",
    "NZD_CHF",
    "NZD_JPY",
    "NZD_USD",
    "USD_CAD",
    "USD_CHF",
    "USD_JPY",
)


def _safe_mean(s: pd.Series) -> float:
    return float(s.mean(skipna=True)) if s.notna().any() else math.nan


def _safe_quantile(s: pd.Series, q: float) -> float:
    return float(s.quantile(q)) if s.notna().any() else math.nan


def _win_rate(s_net_r: pd.Series) -> float:
    s = s_net_r.dropna()
    if len(s) == 0:
        return math.nan
    return float((s > 0).mean()) * 100.0


def _format_table(
    headers: List[str], rows: List[List[Any]], widths: Optional[List[int]] = None
) -> str:
    """Render a markdown table with simple deterministic formatting."""
    if not widths:
        widths = [
            max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
            for i, h in enumerate(headers)
        ]
    out_lines = []
    out_lines.append(
        "| "
        + " | ".join(
            f"{h:>{widths[i]}}" if i > 0 else f"{h:<{widths[i]}}" for i, h in enumerate(headers)
        )
        + " |"
    )
    out_lines.append("|" + "|".join(("-" * (widths[i] + 2)) for i in range(len(headers))) + "|")
    for r in rows:
        out_lines.append(
            "| "
            + " | ".join(
                f"{r[i]!s:>{widths[i]}}" if i > 0 else f"{r[i]!s:<{widths[i]}}"
                for i in range(len(headers))
            )
            + " |"
        )
    return "\n".join(out_lines)


def _fmt_num(x: Any, prec: int = 4) -> str:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(f) or not math.isfinite(f):
        return "n/a"
    return f"{f:.{prec}f}"


def _fmt_int(x: Any) -> str:
    try:
        return str(int(x))
    except (TypeError, ValueError):
        return "n/a"


def _decile_label(idx: int) -> str:
    return f"D{idx + 1}"


def _bin_to_deciles(s: pd.Series) -> pd.Series:
    """Bin a numeric series into 10 deciles (D1 lowest, D10 highest). NaN-aware."""
    notna = s.dropna()
    if len(notna) < 20:
        return pd.Series(["n/a"] * len(s), index=s.index)
    try:
        labels = [_decile_label(i) for i in range(10)]
        binned = pd.qcut(notna, q=10, labels=labels, duplicates="drop")
    except Exception:
        return pd.Series(["n/a"] * len(s), index=s.index)
    out = pd.Series(["n/a"] * len(s), index=s.index, dtype=object)
    out.loc[notna.index] = binned.astype(str).values
    return out


def _label_outcome_table(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Per-label aggregate: trade count, mean net_r, win %, mean gross_r,
    mean spread_cost_r, MFE quantiles 25/50/75/95, MAE quantiles 25/50/75/95.
    """
    rows = []
    if label_col not in df.columns:
        return pd.DataFrame(
            columns=[
                "label",
                "n",
                "mean_net_r",
                "win_pct",
                "mean_gross_r",
                "mean_spread_cost_r",
                "mfe_q25",
                "mfe_q50",
                "mfe_q75",
                "mfe_q95",
                "mae_q25",
                "mae_q50",
                "mae_q75",
                "mae_q95",
            ]
        )
    for label_val, sub in df.groupby(label_col, dropna=False, sort=True):
        rows.append(
            {
                "label": str(label_val) if pd.notna(label_val) else "n/a",
                "n": int(len(sub)),
                "mean_net_r": _safe_mean(sub["net_r"]),
                "win_pct": _win_rate(sub["net_r"]),
                "mean_gross_r": _safe_mean(sub["gross_r"]),
                "mean_spread_cost_r": _safe_mean(sub["spread_cost_r"]),
                "mfe_q25": _safe_quantile(sub["mfe_held_atr"], 0.25),
                "mfe_q50": _safe_quantile(sub["mfe_held_atr"], 0.50),
                "mfe_q75": _safe_quantile(sub["mfe_held_atr"], 0.75),
                "mfe_q95": _safe_quantile(sub["mfe_held_atr"], 0.95),
                "mae_q25": _safe_quantile(sub["mae_held_atr"], 0.25),
                "mae_q50": _safe_quantile(sub["mae_held_atr"], 0.50),
                "mae_q75": _safe_quantile(sub["mae_held_atr"], 0.75),
                "mae_q95": _safe_quantile(sub["mae_held_atr"], 0.95),
            }
        )
    return pd.DataFrame(rows)


def _decile_outcome_table(df: pd.DataFrame, feat_col: str) -> pd.DataFrame:
    """Per-decile aggregate for a continuous feature: trade count, mean net_r,
    win %, mean MFE quantile per decile (use q50 as point estimate)."""
    rows = []
    if feat_col not in df.columns:
        return pd.DataFrame(columns=["decile", "n", "mean_net_r", "win_pct", "mean_mfe_held_atr"])
    deciles = _bin_to_deciles(df[feat_col])
    for label_val in [_decile_label(i) for i in range(10)] + ["n/a"]:
        mask = deciles == label_val
        sub = df[mask]
        if len(sub) == 0:
            continue
        rows.append(
            {
                "decile": label_val,
                "n": int(len(sub)),
                "mean_net_r": _safe_mean(sub["net_r"]),
                "win_pct": _win_rate(sub["net_r"]),
                "mean_mfe_held_atr": _safe_mean(sub["mfe_held_atr"]),
            }
        )
    return pd.DataFrame(rows)


def _regime_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sp, mtf), sub in df.groupby(
        ["structural_pattern", "mtf_alignment"], dropna=False, sort=True
    ):
        rows.append(
            {
                "structural_pattern": str(sp),
                "mtf_alignment": str(mtf),
                "n": int(len(sub)),
                "mean_net_r": _safe_mean(sub["net_r"]),
                "mean_gross_r": _safe_mean(sub["gross_r"]),
                "win_pct": _win_rate(sub["net_r"]),
                "mfe_q50": _safe_quantile(sub["mfe_held_atr"], 0.50),
                "mfe_q75": _safe_quantile(sub["mfe_held_atr"], 0.75),
                "mfe_q95": _safe_quantile(sub["mfe_held_atr"], 0.95),
                "mae_q50": _safe_quantile(sub["mae_held_atr"], 0.50),
                "mae_q75": _safe_quantile(sub["mae_held_atr"], 0.75),
                "mae_q95": _safe_quantile(sub["mae_held_atr"], 0.95),
            }
        )
    return pd.DataFrame(rows)


def _forward_horizon_curves(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    rows = []

    # Pooled
    for H in horizons:
        col = f"fwd_logret_h{H}"
        mfe = f"fwd_mfe_h{H}_atr"
        mae = f"fwd_mae_h{H}_atr"
        s = df[col]
        rows.append(
            {
                "structural_pattern": "pooled",
                "horizon_h": H,
                "n_trades": int(s.notna().sum()),
                "mean_fwd_logret": _safe_mean(s),
                "mean_mfe_atr": _safe_mean(df[mfe]),
                "mean_mae_atr": _safe_mean(df[mae]),
                "p25_fwd_logret": _safe_quantile(s, 0.25),
                "p50_fwd_logret": _safe_quantile(s, 0.50),
                "p75_fwd_logret": _safe_quantile(s, 0.75),
                "p95_fwd_logret": _safe_quantile(s, 0.95),
            }
        )

    for sp, sub in df.groupby("structural_pattern", dropna=False, sort=True):
        for H in horizons:
            col = f"fwd_logret_h{H}"
            mfe = f"fwd_mfe_h{H}_atr"
            mae = f"fwd_mae_h{H}_atr"
            s = sub[col]
            rows.append(
                {
                    "structural_pattern": str(sp),
                    "horizon_h": H,
                    "n_trades": int(s.notna().sum()),
                    "mean_fwd_logret": _safe_mean(s),
                    "mean_mfe_atr": _safe_mean(sub[mfe]),
                    "mean_mae_atr": _safe_mean(sub[mae]),
                    "p25_fwd_logret": _safe_quantile(s, 0.25),
                    "p50_fwd_logret": _safe_quantile(s, 0.50),
                    "p75_fwd_logret": _safe_quantile(s, 0.75),
                    "p95_fwd_logret": _safe_quantile(s, 0.95),
                }
            )
    return pd.DataFrame(rows)


def _pair_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair, sub in df.groupby("pair", sort=True):
        # NaN rates for volume features
        for_nan = {}
        for col in (
            "volume_1h_at_n",
            "volume_4h_at_lag1",
            "volume_d1_at_lag1",
            "volume_w1_at_lag1",
        ):
            if col in sub.columns:
                for_nan[col + "_nan_rate"] = float(sub[col].isna().mean())
        rows.append(
            {
                "pair": pair,
                "n": int(len(sub)),
                "mean_net_r": _safe_mean(sub["net_r"]),
                "mean_gross_r": _safe_mean(sub["gross_r"]),
                "mean_spread_cost_r": _safe_mean(sub["spread_cost_r"]),
                "win_pct": _win_rate(sub["net_r"]),
                "mfe_q50": _safe_quantile(sub["mfe_held_atr"], 0.50),
                "mfe_q75": _safe_quantile(sub["mfe_held_atr"], 0.75),
                "mfe_q95": _safe_quantile(sub["mfe_held_atr"], 0.95),
                "mae_q50": _safe_quantile(sub["mae_held_atr"], 0.50),
                "mae_q75": _safe_quantile(sub["mae_held_atr"], 0.75),
                "mae_q95": _safe_quantile(sub["mae_held_atr"], 0.95),
                **for_nan,
            }
        )
    return pd.DataFrame(rows)


def _volume_nan_audit(df: pd.DataFrame) -> pd.DataFrame:
    """NaN-rate per (pair × tf) for volume features. Also adds per-session NaN."""
    rows = []
    vol_cols = {
        "1H": "volume_1h_at_n",
        "4H": "volume_4h_at_lag1",
        "D1": "volume_d1_at_lag1",
        "W1": "volume_w1_at_lag1",
    }
    for pair, sub in df.groupby("pair", sort=True):
        for tf, col in vol_cols.items():
            n = len(sub)
            n_nan = int(sub[col].isna().sum())
            rows.append(
                {
                    "pair": pair,
                    "tf": tf,
                    "n_total": n,
                    "n_nan": n_nan,
                    "nan_rate_pct": (100.0 * n_nan / n) if n > 0 else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _classification_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grp_cols = ["structural_pattern", "mtf_alignment", "cluster_label"]
    for vals, sub in df.groupby(grp_cols, dropna=False, sort=True):
        rows.append(
            {
                **{c: str(v) for c, v in zip(grp_cols, vals)},
                "n": int(len(sub)),
                "pct_of_total": 100.0 * len(sub) / len(df) if len(df) > 0 else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _per_fold_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """For each Arc-1 fold (1-7), enumerate over-representation of structural
    patterns, MTF alignments, and continuous-feature deciles vs the pooled
    baseline. Reports relative frequency.

    "Over-representation" defined as (fold_freq / pooled_freq), with a tolerance
    requirement: pooled_freq >= 5% to avoid noise from rare patterns.
    """
    rows = []
    fold_ids = sorted(df["fold_id"].astype(str).unique())
    for cat_col in ("structural_pattern", "mtf_alignment", "session"):
        if cat_col not in df.columns:
            continue
        pooled_freq = df[cat_col].value_counts(normalize=True, dropna=False)
        for fid in fold_ids:
            sub = df[df["fold_id"].astype(str) == str(fid)]
            if len(sub) == 0:
                continue
            fold_freq = sub[cat_col].value_counts(normalize=True, dropna=False)
            for label, pct in fold_freq.items():
                pooled_pct = pooled_freq.get(label, 0.0)
                if pooled_pct < 0.02:  # don't report sub-2% labels
                    continue
                ratio = pct / pooled_pct if pooled_pct > 0 else math.nan
                rows.append(
                    {
                        "fold_id": str(fid),
                        "category": cat_col,
                        "label": str(label),
                        "fold_pct": 100.0 * pct,
                        "pooled_pct": 100.0 * pooled_pct,
                        "lift_ratio": ratio,
                        "fold_n": int((sub[cat_col].astype(str) == str(label)).sum()),
                    }
                )
    return pd.DataFrame(rows)


def _edge_vs_cost(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}

    # Pooled
    pooled = pd.DataFrame(
        [
            {
                "scope": "pooled",
                "n": int(len(df)),
                "mean_gross_r": _safe_mean(df["gross_r"]),
                "mean_spread_cost_r": _safe_mean(df["spread_cost_r"]),
                "mean_net_r": _safe_mean(df["net_r"]),
            }
        ]
    )

    # Per structural_pattern
    sp_rows = []
    for sp, sub in df.groupby("structural_pattern", dropna=False, sort=True):
        sp_rows.append(
            {
                "scope": f"structural_pattern={sp}",
                "n": int(len(sub)),
                "mean_gross_r": _safe_mean(sub["gross_r"]),
                "mean_spread_cost_r": _safe_mean(sub["spread_cost_r"]),
                "mean_net_r": _safe_mean(sub["net_r"]),
            }
        )

    # Floored vs not-floored at entry
    floor_rows = []
    if "spread_floored_at_signal" in df.columns:
        for floor_val in [True, False]:
            sub = df[
                df["spread_floored_at_signal"].astype(str).str.lower() == str(floor_val).lower()
            ]
            floor_rows.append(
                {
                    "scope": f"floored_at_entry={floor_val}",
                    "n": int(len(sub)),
                    "mean_gross_r": _safe_mean(sub["gross_r"]),
                    "mean_spread_cost_r": _safe_mean(sub["spread_cost_r"]),
                    "mean_net_r": _safe_mean(sub["net_r"]),
                }
            )

    out["pooled"] = pooled
    out["per_structural_pattern"] = pd.DataFrame(sp_rows)
    out["floored_split"] = pd.DataFrame(floor_rows)
    return out


def _findings_bullets(df: pd.DataFrame) -> List[str]:
    """Produce a deterministic bullet list of empirical observations.

    NO recommendations. NO "this would make a good filter." Pure descriptive
    statements: "X label has mean net_r of Y vs pooled Z, n=W".
    """
    bullets: List[str] = []
    pooled_net = _safe_mean(df["net_r"])
    pooled_n = int(len(df))
    bullets.append(
        f"Pooled mean net_r = {_fmt_num(pooled_net, 4)} R over n={pooled_n} taken trades."
    )

    # Per structural_pattern
    for sp, sub in df.groupby("structural_pattern", dropna=False, sort=True):
        if len(sub) < 50:
            continue
        bullets.append(
            f"structural_pattern={sp}: mean net_r = {_fmt_num(_safe_mean(sub['net_r']), 4)} R, "
            f"win % = {_fmt_num(_win_rate(sub['net_r']), 1)}, "
            f"n={len(sub)} (pooled {_fmt_num(100.0 * len(sub) / pooled_n, 1)}%)."
        )

    # Per fold disposition
    for dispo, sub in df.groupby("arc1_fold_disposition", dropna=False, sort=True):
        if len(sub) < 50:
            continue
        bullets.append(
            f"arc1_fold_disposition={dispo}: mean net_r = {_fmt_num(_safe_mean(sub['net_r']), 4)} R, "
            f"win % = {_fmt_num(_win_rate(sub['net_r']), 1)}, "
            f"n={len(sub)}."
        )

    # Per session
    for sess, sub in df.groupby("session", dropna=False, sort=True):
        if len(sub) < 50:
            continue
        bullets.append(
            f"session={sess}: mean net_r = {_fmt_num(_safe_mean(sub['net_r']), 4)} R, "
            f"win % = {_fmt_num(_win_rate(sub['net_r']), 1)}, "
            f"n={len(sub)}."
        )

    # Spread-floor split
    if "spread_floored_at_signal" in df.columns:
        for floor_val in [True, False]:
            sub = df[
                df["spread_floored_at_signal"].astype(str).str.lower() == str(floor_val).lower()
            ]
            if len(sub) < 50:
                continue
            bullets.append(
                f"spread_floored_at_signal={floor_val}: mean gross_r = {_fmt_num(_safe_mean(sub['gross_r']), 4)} R, "
                f"mean spread_cost_r = {_fmt_num(_safe_mean(sub['spread_cost_r']), 4)} R, "
                f"mean net_r = {_fmt_num(_safe_mean(sub['net_r']), 4)} R, n={len(sub)}."
            )

    # Forward horizon mean returns (pooled)
    for H in (1, 6, 24, 72, 120, 240):
        col = f"fwd_logret_h{H}"
        if col not in df.columns:
            continue
        bullets.append(
            f"forward h={H}h: pooled mean fwd_logret = {_fmt_num(_safe_mean(df[col]), 6)} (signed log return; positive = up move). "
            f"n with non-NaN = {int(df[col].notna().sum())}."
        )

    return bullets


def _arc1_fold_window_count(
    arc1_signals_log_csv: Path, signal_start: pd.Timestamp, signal_end: pd.Timestamp
) -> int:
    """Count Arc-1 signals_log.csv rows within the characterisation window."""
    if not arc1_signals_log_csv.exists():
        return -1
    df = pd.read_csv(arc1_signals_log_csv)
    df["signal_bar_ts"] = pd.to_datetime(df["signal_bar_ts"])
    in_window = df[(df["signal_bar_ts"] >= signal_start) & (df["signal_bar_ts"] <= signal_end)]
    return int(len(in_window))


def _generate_lag_audit(df: pd.DataFrame, sample_n: int, seed: int) -> str:
    """Deterministic 100-row sample listing signal_bar_ts, ts_4h_used, ts_d1_used, ts_w1_used and assertions."""
    rng = np.random.default_rng(seed)
    n = len(df)
    sample_n = min(sample_n, n)
    indices = sorted(rng.choice(n, size=sample_n, replace=False).tolist())
    lines: List[str] = []
    lines.append(f"L4 characterisation lag-1 audit — sample of {sample_n} signals (seed={seed})")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Lag-1 rules:")
    lines.append("  4H: ts_4h_used.close_time (= time + 4h) <= signal_bar_ts")
    lines.append("  D1: ts_d1_used.date < signal_bar_ts.date  (strict prior calendar day)")
    lines.append("  W1: ts_w1_used < weekstart(signal_bar_ts)  (strict prior week)")
    lines.append("")
    lines.append(
        "Header: pair signal_bar_ts ts_4h_used ts_d1_used ts_w1_used  4H_PASS  D1_PASS  W1_PASS"
    )
    lines.append("-" * 90)
    fail_count = 0
    for i in indices:
        row = df.iloc[i]
        t_n = pd.Timestamp(row["signal_bar_ts"])
        ts_4h = row.get("ts_4h_used")
        ts_d1 = row.get("ts_d1_used")
        ts_w1 = row.get("ts_w1_used")
        ts_4h_pd = pd.Timestamp(ts_4h) if pd.notna(ts_4h) else None
        ts_d1_pd = pd.Timestamp(ts_d1) if pd.notna(ts_d1) else None
        ts_w1_pd = pd.Timestamp(ts_w1) if pd.notna(ts_w1) else None
        h4_pass = ts_4h_pd is None or (ts_4h_pd + pd.Timedelta(hours=4)) <= t_n
        d1_pass = ts_d1_pd is None or (ts_d1_pd.date() < t_n.date())
        # Compute weekstart
        days_since_sun = (t_n.dayofweek + 1) % 7
        weekstart = t_n.normalize() - pd.Timedelta(days=int(days_since_sun))
        w1_pass = ts_w1_pd is None or (ts_w1_pd < weekstart)
        if not (h4_pass and d1_pass and w1_pass):
            fail_count += 1
        lines.append(
            f"{row['pair']:>8}  {t_n}  {ts_4h_pd}  {ts_d1_pd}  {ts_w1_pd}  "
            f"{'PASS' if h4_pass else 'FAIL'}  {'PASS' if d1_pass else 'FAIL'}  {'PASS' if w1_pass else 'FAIL'}"
        )
    lines.append("")
    lines.append(f"Total assertion failures across the {sample_n}-row sample: {fail_count}")
    lines.append("Pooled pipeline-runtime assertion failures (must be zero): SEE pipeline run log")
    return "\n".join(lines)


def _author_report(
    df: pd.DataFrame,
    horizons: List[int],
    signal_start: pd.Timestamp,
    signal_end: pd.Timestamp,
    arc1_overlap_count: int,
    input_sha256s: Dict[str, str],
    git_commit: str,
    timestamp_str: str,
    deterministic_rerun_confirmed: bool,
    rerun_run1_sha: str,
    rerun_run2_sha: str,
) -> str:
    pooled_n = int(len(df))
    _safe_mean(df["net_r"])
    _safe_mean(df["gross_r"])
    _safe_mean(df["spread_cost_r"])
    _win_rate(df["net_r"])

    # 1. Header
    lines: List[str] = []
    lines.append("# L4 Univariate-Extreme — Descriptive Characterisation Report")
    lines.append("")
    lines.append(
        "**Disposition:** descriptive only — no PASS/FAIL, no gate, no filter derivation. "
    )
    lines.append(
        "This document is research output, not a tradable system. Per L6.0 §9 / §11 / §13 "
    )
    lines.append(
        "any pattern observed below is a candidate hypothesis for a fresh arc with a pre-committed "
    )
    lines.append("filter, never a derived gate within this exercise.")
    lines.append("")
    lines.append("## 1. Header")
    lines.append("")
    lines.append(f"- **Run timestamp:** {timestamp_str}")
    lines.append(f"- **Git commit:** `{git_commit}`")
    lines.append(
        f"- **Signal window:** {signal_start.isoformat()} → {signal_end.isoformat()} (1H bars, inclusive)"
    )
    lines.append("- **Pairs:** all 28 (per L0/L6.0/Arc 1 universe)")
    lines.append(
        f"- **Deterministic-rerun confirmation:** {'PASS — byte-identical' if deterministic_rerun_confirmed else 'NOT YET CONFIRMED'}"
    )
    lines.append(f"  - Run #1 signals_features.csv sha256: `{rerun_run1_sha}`")
    lines.append(f"  - Run #2 signals_features.csv sha256: `{rerun_run2_sha}`")
    lines.append("")
    lines.append("**Input file sha256s:**")
    lines.append("")
    for k, v in input_sha256s.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")

    # 2. Population summary
    lines.append("## 2. Population summary")
    lines.append("")
    lines.append(f"- **Total signals in window:** {pooled_n:,}")
    lines.append(f"- **Arc 1 signals_log overlap-window count:** {arc1_overlap_count:,}")
    if arc1_overlap_count > 0:
        diff_pct = abs(pooled_n - arc1_overlap_count) / arc1_overlap_count * 100.0
        lines.append(
            f"- **Bar-set divergence vs Arc 1 over the same window:** {diff_pct:.4f}% "
            f"(target ≤ 0.5% per spec — {'WITHIN' if diff_pct <= 0.5 else 'OUTSIDE'} band)"
        )
    else:
        lines.append("- **Bar-set divergence vs Arc 1:** n/a (Arc 1 signals_log unavailable)")
    lines.append("")

    # Per-pair counts
    lines.append("### 2.1 Per-pair signal counts")
    lines.append("")
    lines.append("| pair | n |")
    lines.append("|------|---:|")
    for pair, sub in df.groupby("pair", sort=True):
        lines.append(f"| {pair} | {len(sub):,} |")
    lines.append("")

    # Per-fold counts
    lines.append("### 2.2 Per-Arc-1-fold signal counts")
    lines.append("")
    lines.append("| fold_id | disposition | n |")
    lines.append("|---:|---|---:|")
    for fid, sub in df.groupby("fold_id", sort=True):
        dispos = sub["arc1_fold_disposition"].iloc[0] if len(sub) > 0 else ""
        lines.append(f"| {fid} | {dispos} | {len(sub):,} |")
    lines.append("")

    # Per-session counts
    lines.append("### 2.3 Per-session signal counts")
    lines.append("")
    lines.append("| session | n | pct |")
    lines.append("|---|---:|---:|")
    for sess, sub in df.groupby("session", sort=True):
        lines.append(f"| {sess} | {len(sub):,} | {100.0 * len(sub) / pooled_n:.2f}% |")
    lines.append("")

    # 3. Volume null-handling audit
    lines.append("## 3. Volume null-handling audit")
    lines.append("")
    lines.append("Per-pair × per-TF NaN rates for volume features (volume == 0 in the source data ")
    lines.append(
        "is treated as NaN; ratios and z-scores propagate NaN). High NaN rates flag pair × "
    )
    lines.append("session combinations where tick volume is unreliable.")
    lines.append("")
    nan_audit = _volume_nan_audit(df)
    pivot = nan_audit.pivot_table(index="pair", columns="tf", values="nan_rate_pct", aggfunc="mean")
    lines.append(
        "### 3.1 NaN rate per (pair, TF) — pct of signals with NaN volume at the relevant lag-1 bar"
    )
    lines.append("")
    lines.append("| pair | 1H | 4H | D1 | W1 |")
    lines.append("|------|---:|---:|---:|---:|")
    for pair in sorted(df["pair"].unique()):
        row = pivot.loc[pair] if pair in pivot.index else None
        h1 = _fmt_num(row.get("1H", math.nan), 2) if row is not None else "n/a"
        h4 = _fmt_num(row.get("4H", math.nan), 2) if row is not None else "n/a"
        d1 = _fmt_num(row.get("D1", math.nan), 2) if row is not None else "n/a"
        w1 = _fmt_num(row.get("W1", math.nan), 2) if row is not None else "n/a"
        lines.append(f"| {pair} | {h1} | {h4} | {d1} | {w1} |")
    lines.append("")

    # 4. Classification breakdown
    lines.append("## 4. Classification breakdown")
    lines.append("")
    lines.append("Joint counts by structural_pattern × mtf_alignment × cluster_label.")
    lines.append("")
    classn = _classification_breakdown(df)
    lines.append("| structural_pattern | mtf_alignment | cluster_label | n | pct |")
    lines.append("|---|---|---|---:|---:|")
    for _, r in classn.iterrows():
        lines.append(
            f"| {r['structural_pattern']} | {r['mtf_alignment']} | {r['cluster_label']} | "
            f"{int(r['n']):,} | {_fmt_num(r['pct_of_total'], 2)}% |"
        )
    lines.append("")

    # 5. Conditional outcome tables per label
    lines.append("## 5. Conditional outcome tables")
    lines.append("")
    lines.append("Per-label aggregation: trade count (n), mean net_r, win % (positive net_r),")
    lines.append("mean gross_r (mid-mid), mean spread_cost_r (gross − net), MFE/MAE quantiles.")
    lines.append("")
    for label_col in (
        "structural_pattern",
        "mtf_alignment",
        "d1_trend_label",
        "h4_trend_label",
        "h1_trend_label",
        "pre_momentum_label",
        "cluster_label",
        "session",
    ):
        if label_col not in df.columns:
            continue
        lines.append(f"### 5.{label_col}")
        lines.append("")
        tbl = _label_outcome_table(df, label_col)
        if tbl.empty:
            lines.append("_n/a_")
            lines.append("")
            continue
        lines.append(
            "| label | n | mean_net_r | win_pct | mean_gross_r | mean_spread_cost_r | mfe_q25 | mfe_q50 | mfe_q75 | mfe_q95 | mae_q25 | mae_q50 | mae_q75 | mae_q95 |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in tbl.iterrows():
            lines.append(
                f"| {r['label']} | {int(r['n']):,} | "
                f"{_fmt_num(r['mean_net_r'], 4)} | {_fmt_num(r['win_pct'], 1)} | "
                f"{_fmt_num(r['mean_gross_r'], 4)} | {_fmt_num(r['mean_spread_cost_r'], 4)} | "
                f"{_fmt_num(r['mfe_q25'], 3)} | {_fmt_num(r['mfe_q50'], 3)} | {_fmt_num(r['mfe_q75'], 3)} | {_fmt_num(r['mfe_q95'], 3)} | "
                f"{_fmt_num(r['mae_q25'], 3)} | {_fmt_num(r['mae_q50'], 3)} | {_fmt_num(r['mae_q75'], 3)} | {_fmt_num(r['mae_q95'], 3)} |"
            )
        lines.append("")

    # 6. Conditional outcome by continuous-feature decile
    lines.append("## 6. Conditional outcome by continuous-feature decile")
    lines.append("")
    lines.append("Each continuous feature is binned into deciles (D1=lowest, D10=highest). ")
    lines.append(
        "Per decile: trade count, mean net_r, win %, mean MFE during held bar (in ATR units)."
    )
    lines.append("")
    decile_features = (
        "atr_1h_regime",
        "atr_d1_regime",
        "cum_logret_d1_5",
        "dist_to_kijun_d1_atr",
        "dist_to_ema50_d1_atr",
        "realized_vol_24h",
        "volume_d1_ratio",
        "concurrent_signals_within_3h",
        "usd_basket_3h",
        "eur_basket_3h",
        "jpy_basket_3h",
        "gbp_basket_3h",
    )
    for feat in decile_features:
        if feat not in df.columns:
            continue
        lines.append(f"### 6.{feat}")
        lines.append("")
        tbl = _decile_outcome_table(df, feat)
        if tbl.empty:
            lines.append("_n/a_")
            lines.append("")
            continue
        lines.append("| decile | n | mean_net_r | win_pct | mean_mfe_held_atr |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in tbl.iterrows():
            lines.append(
                f"| {r['decile']} | {int(r['n']):,} | {_fmt_num(r['mean_net_r'], 4)} | "
                f"{_fmt_num(r['win_pct'], 1)} | {_fmt_num(r['mean_mfe_held_atr'], 3)} |"
            )
        lines.append("")

    # 7. Forward-horizon mean curves
    lines.append("## 7. Forward-horizon mean curves")
    lines.append("")
    lines.append(
        "Forward outcomes are measured from entry (bar N+1 open) to bar N+1+H. They are NOT "
    )
    lines.append(
        "what Arc 1 traded (Arc 1 holds for 1 bar). These curves are descriptive only — they "
    )
    lines.append(
        "answer 'what would the trade have looked like at longer horizons' without modifying "
    )
    lines.append("Arc 1's verbatim execution.")
    lines.append("")
    fhc = _forward_horizon_curves(df, horizons)
    lines.append("### 7.1 Pooled and per-structural_pattern curves")
    lines.append("")
    lines.append(
        "| structural_pattern | horizon_h | n_trades | mean_fwd_logret | mean_mfe_atr | mean_mae_atr | p25 | p50 | p75 | p95 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in fhc.iterrows():
        lines.append(
            f"| {r['structural_pattern']} | {int(r['horizon_h'])} | {int(r['n_trades']):,} | "
            f"{_fmt_num(r['mean_fwd_logret'], 6)} | {_fmt_num(r['mean_mfe_atr'], 3)} | {_fmt_num(r['mean_mae_atr'], 3)} | "
            f"{_fmt_num(r['p25_fwd_logret'], 6)} | {_fmt_num(r['p50_fwd_logret'], 6)} | "
            f"{_fmt_num(r['p75_fwd_logret'], 6)} | {_fmt_num(r['p95_fwd_logret'], 6)} |"
        )
    lines.append("")

    # 8. Per-fold-disposition breakdown
    lines.append("## 8. Per-fold-disposition breakdown")
    lines.append("")
    lines.append("For each Arc-1 fold, list categories where the fold's frequency differs ")
    lines.append(
        "substantially from pooled. lift_ratio = fold_pct / pooled_pct. Reported only when "
    )
    lines.append("pooled_pct ≥ 2% to avoid noise.")
    lines.append("")
    pf = _per_fold_breakdown(df)
    if pf.empty:
        lines.append("_n/a_")
        lines.append("")
    else:
        lines.append("| fold_id | category | label | fold_pct | pooled_pct | lift_ratio | fold_n |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        # Sort by fold_id then |lift_ratio - 1| descending
        pf["abs_lift"] = (pf["lift_ratio"] - 1.0).abs()
        pf_sorted = pf.sort_values(["fold_id", "abs_lift"], ascending=[True, False])
        for _, r in pf_sorted.iterrows():
            lines.append(
                f"| {r['fold_id']} | {r['category']} | {r['label']} | "
                f"{_fmt_num(r['fold_pct'], 2)}% | {_fmt_num(r['pooled_pct'], 2)}% | "
                f"{_fmt_num(r['lift_ratio'], 3)} | {int(r['fold_n']):,} |"
            )
        lines.append("")

    # 9. Edge-vs-cost decomposition
    lines.append("## 9. Edge-vs-cost decomposition")
    lines.append("")
    edge = _edge_vs_cost(df)
    lines.append("### 9.1 Pooled, per structural_pattern, and per spread-floor split")
    lines.append("")
    lines.append("| scope | n | mean_gross_r | mean_spread_cost_r | mean_net_r |")
    lines.append("|---|---:|---:|---:|---:|")
    for tbl_name in ("pooled", "per_structural_pattern", "floored_split"):
        tbl = edge[tbl_name]
        for _, r in tbl.iterrows():
            lines.append(
                f"| {r['scope']} | {int(r['n']):,} | "
                f"{_fmt_num(r['mean_gross_r'], 4)} | {_fmt_num(r['mean_spread_cost_r'], 4)} | "
                f"{_fmt_num(r['mean_net_r'], 4)} |"
            )
    lines.append("")

    # 10. Findings summary
    lines.append("## 10. Findings summary")
    lines.append("")
    lines.append("**Empirical observations only.** No recommendations. No filter suggestions. ")
    lines.append("Anything that sounds system-relevant is logged; downstream action requires a ")
    lines.append("fresh arc with a pre-committed filter per L6.0 §9 / §11.")
    lines.append("")
    bullets = _findings_bullets(df)
    for b in bullets:
        lines.append(f"- {b}")
    lines.append("")

    # Counts of cells computed (for top-level findings count)
    n_label_cells = sum(
        1
        for c in (
            "structural_pattern",
            "mtf_alignment",
            "d1_trend_label",
            "h4_trend_label",
            "h1_trend_label",
            "pre_momentum_label",
            "cluster_label",
            "session",
        )
        for v in df[c].dropna().unique()
        if c in df.columns
    )
    n_decile_cells = 0
    for f in decile_features:
        if f in df.columns:
            n_decile_cells += 10
    n_pattern_horizon_cells = (df["structural_pattern"].nunique() + 1) * len(horizons)
    lines.append("### 10.1 Cells computed")
    lines.append("")
    lines.append(f"- Label cells (sec 5): {n_label_cells}")
    lines.append(f"- Decile cells (sec 6): {n_decile_cells}")
    lines.append(f"- Pattern × horizon cells (sec 7): {n_pattern_horizon_cells}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*End of L4 characterisation report. No system has been derived. No filter has been "
    )
    lines.append(
        "pre-committed. No gate has been evaluated. The output is information, not a tradeable "
    )
    lines.append("hypothesis. — per task brief / L6.0 §9 / §11 / §13.*")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-csv", default="results/l6/characterisation/signals_features.csv"
    )
    parser.add_argument("--config", default="configs/l4_characterisation.yaml")
    parser.add_argument("--rerun-run1-sha", default="")
    parser.add_argument("--rerun-run2-sha", default="")
    parser.add_argument("--deterministic-confirmed", action="store_true")
    args = parser.parse_args()

    features_csv = (REPO_ROOT / args.features_csv).resolve()
    if not features_csv.exists():
        print(f"FATAL: features csv not found at {features_csv}", file=sys.stderr)
        return 2

    import yaml as _yaml

    config_path = (REPO_ROOT / args.config).resolve()
    raw = _yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out_dir = (REPO_ROOT / raw["characterisation"]["output_dir"]).resolve()
    horizons: List[int] = [int(h) for h in raw["characterisation"]["forward_horizons"]]
    signal_start = pd.Timestamp(raw["window"]["signal_start"])
    signal_end = pd.Timestamp(raw["window"]["signal_end"])
    sample_n = int(raw["characterisation"]["lag_audit_sample_n"])
    sample_seed = int(raw["characterisation"]["lag_audit_sample_seed"])

    print(f"Reading {features_csv} ...")
    df = pd.read_csv(features_csv, low_memory=False)
    print(f"  {len(df):,} rows")
    df["signal_bar_ts"] = pd.to_datetime(df["signal_bar_ts"])
    if "ts_4h_used" in df.columns:
        df["ts_4h_used"] = pd.to_datetime(df["ts_4h_used"], errors="coerce")
        df["ts_d1_used"] = pd.to_datetime(df["ts_d1_used"], errors="coerce")
        df["ts_w1_used"] = pd.to_datetime(df["ts_w1_used"], errors="coerce")

    # ----- regime_breakdown.csv -----
    regime_csv = out_dir / "regime_breakdown.csv"
    _regime_breakdown(df).to_csv(regime_csv, index=False, lineterminator="\n", float_format="%.6f")
    print(f"Wrote {regime_csv}")

    # ----- forward_horizon_curves.csv -----
    fhc_csv = out_dir / "forward_horizon_curves.csv"
    _forward_horizon_curves(df, horizons).to_csv(
        fhc_csv, index=False, lineterminator="\n", float_format="%.6f"
    )
    print(f"Wrote {fhc_csv}")

    # ----- pair_breakdown.csv -----
    pair_csv = out_dir / "pair_breakdown.csv"
    _pair_breakdown(df).to_csv(pair_csv, index=False, lineterminator="\n", float_format="%.6f")
    print(f"Wrote {pair_csv}")

    # ----- feature_lag_audit.txt -----
    lag_audit_txt = out_dir / "feature_lag_audit.txt"
    lag_audit_txt.write_text(
        _generate_lag_audit(df, sample_n, sample_seed) + "\n", encoding="utf-8"
    )
    print(f"Wrote {lag_audit_txt}")

    # ----- characterisation_report.md -----
    # Compute Arc 1 overlap-window count
    arc1_signals_log = REPO_ROOT / "results" / "l6" / "arc1" / "signals_log.csv"
    arc1_overlap = _arc1_fold_window_count(arc1_signals_log, signal_start, signal_end)

    spread_floor_path = REPO_ROOT / "configs" / "spread_floors_5ers.yaml"
    arc1_trades_path = REPO_ROOT / "results" / "l6" / "arc1" / "trades_all.csv"
    l4_module_path = REPO_ROOT / "core" / "signals" / "l4_univariate_extreme.py"

    input_sha256s = {
        "configs/l4_characterisation.yaml": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "configs/spread_floors_5ers.yaml (body)": compute_body_sha256(spread_floor_path),
        "core/signals/l4_univariate_extreme.py": hashlib.sha256(
            l4_module_path.read_bytes()
        ).hexdigest(),
        "results/l6/arc1/trades_all.csv": hashlib.sha256(arc1_trades_path.read_bytes()).hexdigest()
        if arc1_trades_path.exists()
        else "(missing)",
        "results/l6/arc1/signals_log.csv": hashlib.sha256(arc1_signals_log.read_bytes()).hexdigest()
        if arc1_signals_log.exists()
        else "(missing)",
    }

    # Get git commit
    import subprocess

    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        git_commit = "(unavailable)"

    timestamp_str = _dt.datetime.now().isoformat(timespec="seconds")

    report = _author_report(
        df=df,
        horizons=horizons,
        signal_start=signal_start,
        signal_end=signal_end,
        arc1_overlap_count=arc1_overlap,
        input_sha256s=input_sha256s,
        git_commit=git_commit,
        timestamp_str=timestamp_str,
        deterministic_rerun_confirmed=bool(args.deterministic_confirmed),
        rerun_run1_sha=args.rerun_run1_sha or "(not yet recorded)",
        rerun_run2_sha=args.rerun_run2_sha or "(not yet recorded)",
    )
    report_path = out_dir / "characterisation_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
