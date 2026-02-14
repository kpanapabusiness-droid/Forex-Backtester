"""
Phase D-6A: Ignition Precursor Analysis (Analytics Only).

Identifies Zone C start bars from opportunity labels, computes precursor metrics
using prior bars from OHLCV data, and compares to a control sample.

HARD CONSTRAINTS: No entry/exit changes, no WFO, no optimization.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.utils import calculate_atr, load_pair_csv  # noqa: E402


def _filename_variants_for_pair(pair: str) -> list[str]:
    """Return filename patterns tried for a pair (matches load_pair_csv)."""
    return [
        f"{pair}.csv",
        *([f"{pair.replace('/', '_')}.csv", f"{pair.replace('/', '')}.csv"] if "/" in pair else []),
        *([f"{pair.replace('_', '')}.csv", f"{pair.replace('_', '/')}.csv"] if "_" in pair else []),
    ]

RANDOM_SEED = 42
ATR_PERIOD = 14
ROLLING_PERCENTILE_WINDOW = 252
MIN_BARS = 20


def _load_labels(path: Path) -> pd.DataFrame:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col in ["pair", "date", "direction", "zone_c_6r_40"]:
        if col not in df.columns:
            raise ValueError(f"Labels missing required column: {col}. Found: {list(df.columns)}")
    return df


def _to_bool(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(bool)


def _identify_zone_c_starts(labels: pd.DataFrame) -> pd.DataFrame:
    """Return rows where zone_c_6r_40==1 and previous bar (same pair,direction) had zone_c_6r_40==0."""
    labels = labels.copy()
    labels["zone_c"] = _to_bool(labels["zone_c_6r_40"])
    labels = labels.sort_values(["pair", "direction", "date"]).reset_index(drop=True)

    def _starts_in_group(grp: pd.DataFrame) -> pd.DataFrame:
        zone = grp["zone_c"].values
        prev_zone = np.roll(zone, 1)
        prev_zone[0] = False  # First bar has no previous
        is_start = zone & ~prev_zone
        return grp.loc[is_start].copy()

    starts = labels.groupby(["pair", "direction"], group_keys=True).apply(_starts_in_group)
    if isinstance(starts.index, pd.MultiIndex):
        starts = starts.reset_index(level=["pair", "direction"])
    out_cols = [c for c in ["pair", "date", "direction"] if c in starts.columns]
    return starts[out_cols].reset_index(drop=True)


def _compute_precursor_metrics(df: pd.DataFrame, bar_idx: int) -> dict:
    """
    Compute precursor metrics at bar_idx using OHLCV up to that bar.
    bar_idx is 0-based; requires sufficient prior bars for rolling computations.
    """
    n = len(df)
    if bar_idx < 0 or bar_idx >= n:
        return {}
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    rng = high - low

    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    else:
        vol = pd.Series(0.0, index=df.index)

    ohlc = df[["high", "low", "close"]].copy()
    atr_14 = calculate_atr(ohlc, period=ATR_PERIOD)["atr"]
    atr_val = atr_14.iloc[bar_idx] if bar_idx < len(atr_14) else np.nan

    if bar_idx >= ROLLING_PERCENTILE_WINDOW - 1:
        window = atr_14.iloc[bar_idx - ROLLING_PERCENTILE_WINDOW + 1 : bar_idx + 1]
        atr_pct = (window < atr_val).sum() / len(window) * 100.0 if len(window) else np.nan
    else:
        atr_pct = np.nan

    atr5 = atr_14.rolling(5, min_periods=5).mean().iloc[bar_idx] if bar_idx >= 4 else np.nan
    atr50 = atr_14.rolling(50, min_periods=50).mean().iloc[bar_idx] if bar_idx >= 49 else np.nan
    atr_ratio = atr5 / atr50 if pd.notna(atr5) and pd.notna(atr50) and atr50 > 0 else np.nan

    avg_range_5 = rng.iloc[max(0, bar_idx - 4) : bar_idx + 1].mean() if bar_idx >= 0 else np.nan
    avg_range_10 = rng.iloc[max(0, bar_idx - 9) : bar_idx + 1].mean() if bar_idx >= 0 else np.nan

    if bar_idx >= ROLLING_PERCENTILE_WINDOW - 1:
        rng_window = rng.iloc[bar_idx - ROLLING_PERCENTILE_WINDOW + 1 : bar_idx + 1]
        rng_pct = (rng_window < rng.iloc[bar_idx]).sum() / len(rng_window) * 100.0
    else:
        rng_pct = np.nan

    if bar_idx >= ROLLING_PERCENTILE_WINDOW - 1:
        vol_window = vol.iloc[bar_idx - ROLLING_PERCENTILE_WINDOW + 1 : bar_idx + 1]
        vol_pct = (vol_window < vol.iloc[bar_idx]).sum() / len(vol_window) * 100.0
    else:
        vol_pct = np.nan

    vol_avg_5 = vol.iloc[max(0, bar_idx - 4) : bar_idx + 1].mean() if bar_idx >= 0 else 0.0
    if bar_idx >= ROLLING_PERCENTILE_WINDOW - 1:
        vol_avgs = np.array([
            vol.iloc[max(0, i - 4) : i + 1].mean()
            for i in range(bar_idx - ROLLING_PERCENTILE_WINDOW + 1, bar_idx + 1)
        ])
        vol_avg_5_pct = (
            (vol_avgs < vol_avg_5).sum() / len(vol_avgs) * 100.0 if len(vol_avgs) else np.nan
        )
    else:
        vol_avg_5_pct = np.nan

    vol_curr = vol.iloc[bar_idx] if bar_idx < len(vol) else 0.0
    vol_change_ratio = vol_curr / vol_avg_5 if vol_avg_5 > 0 else np.nan

    if bar_idx >= 6:
        last7 = rng.iloc[bar_idx - 6 : bar_idx + 1]
        nr7 = 1 if rng.iloc[bar_idx] <= last7.min() else 0
    else:
        nr7 = 0

    if bar_idx >= 19:
        h20 = high.iloc[bar_idx - 19 : bar_idx].max()
        l20 = low.iloc[bar_idx - 19 : bar_idx].min()
        close_val = close.iloc[bar_idx]
        break_flag = 1 if (close_val > h20 or close_val < l20) else 0
    else:
        break_flag = 0

    return {
        "atr_14": atr_val,
        "atr_percentile_252": atr_pct,
        "atr_ratio_5_50": atr_ratio,
        "avg_range_5": avg_range_5,
        "avg_range_10": avg_range_10,
        "range_percentile_252": rng_pct,
        "volume_percentile_252": vol_pct,
        "volume_avg5_percentile_252": vol_avg_5_pct,
        "volume_change_ratio": vol_change_ratio,
        "nr7_flag": nr7,
        "break_20bar_hl_flag": break_flag,
    }


def _get_bar_idx_for_date(df: pd.DataFrame, target_date: pd.Timestamp) -> int | None:
    """Return 0-based iloc position of row with date == target_date, or None."""
    dates = pd.to_datetime(df["date"]).dt.normalize()
    target = pd.Timestamp(target_date).normalize()
    pos = np.where(dates.values == target)[0]
    return int(pos[0]) if len(pos) > 0 else None


def _list_ohlcv_files_and_pairs(data_dir: Path, pairs: list[str]) -> tuple[list[Path], dict[str, list[str]]]:
    """Return (found_csv_paths, {pair: [tried_patterns]}) for diagnostics."""
    data_dir = Path(data_dir).resolve()
    found: list[Path] = []
    tried: dict[str, list[str]] = {}
    for pair in pairs:
        patterns = _filename_variants_for_pair(pair)
        tried[pair] = [str(data_dir / p) for p in patterns]
        for p in patterns:
            fp = data_dir / p
            if fp.exists() and fp.is_file():
                found.append(fp)
                break
    return found, tried


def _build_zone_c_metrics(
    labels_path: Path, data_dir: Path, min_bars: int = MIN_BARS
) -> tuple[pd.DataFrame, dict]:
    """
    Return (zone_df, stats).
    stats: n_starts, n_file_not_found, n_no_date_match, n_insufficient_history, n_produced
    """
    labels = _load_labels(labels_path)
    starts = _identify_zone_c_starts(labels)

    if starts.empty:
        total_labels = len(labels)
        zone_c_count = _to_bool(labels["zone_c_6r_40"]).sum()
        raise ValueError(
            f"Zero Zone C starts found in labels. "
            f"Total label rows={total_labels}, zone_c_6r_40=True count={zone_c_count}. "
            f"Check labels path and zone definition: {Path(labels_path).resolve()}"
        )

    data_dir = Path(data_dir).resolve()
    pairs_in_starts = starts["pair"].unique().tolist()
    found_files, _tried = _list_ohlcv_files_and_pairs(data_dir, pairs_in_starts)
    pairs_loaded = set()
    for pair in pairs_in_starts:
        try:
            load_pair_csv(pair, data_dir)
            pairs_loaded.add(pair)
        except FileNotFoundError:
            pass
    if len(pairs_loaded) == 0:
        file_count = len(set(found_files))
        example = pairs_in_starts[0] if pairs_in_starts else "EUR_USD"
        all_patterns = [
            str(data_dir / p)
            for pair in pairs_in_starts[:3]
            for p in _filename_variants_for_pair(pair)
        ]
        patterns = list(dict.fromkeys(all_patterns))[:6]
        raise ValueError(
            f"Zone C starts exist ({len(starts)} rows) but zero pairs loaded from data-dir. "
            f"resolved data-dir: {data_dir} | "
            f"file patterns tried (sample): {patterns[:6]} | "
            f"CSV files found in dir: {file_count} | "
            f"example expected: {data_dir / f'{example}.csv'}"
        )

    stats = {"n_starts": len(starts), "n_file_not_found": 0, "n_no_date_match": 0, "n_insufficient_history": 0, "n_produced": 0}
    rows = []
    for _, row in starts.iterrows():
        pair = row["pair"]
        date = row["date"]
        direction = row["direction"]
        try:
            df = load_pair_csv(pair, data_dir)
        except FileNotFoundError:
            stats["n_file_not_found"] += 1
            continue
        df = df.sort_values("date").reset_index(drop=True)
        idx = _get_bar_idx_for_date(df, date)
        if idx is None:
            stats["n_no_date_match"] += 1
            continue
        if idx < min_bars:
            stats["n_insufficient_history"] += 1
            continue
        metrics = _compute_precursor_metrics(df, idx)
        if not metrics:
            stats["n_insufficient_history"] += 1
            continue
        stats["n_produced"] += 1
        out_row = {"pair": pair, "date": date, "direction": direction, **metrics}
        rows.append(out_row)

    zone_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    if stats["n_produced"] == 0 and stats["n_starts"] > 0:
        raise ValueError(
            f"Zone C starts exist ({stats['n_starts']}) but zero precursor rows produced. "
            f"skipped file_not_found={stats['n_file_not_found']}, "
            f"no_date_match={stats['n_no_date_match']}, "
            f"insufficient_history={stats['n_insufficient_history']}. "
            f"Ensure OHLCV has matching dates and >= {min_bars} bars before each Zone C start."
        )

    return zone_df, stats


def _get_eligible_control_bars(
    labels: pd.DataFrame, data_dir: Path, min_bars: int = MIN_BARS
) -> pd.DataFrame:
    """Get (pair, date) that are NOT Zone C starts, with at least min_bars prior for metrics."""
    starts = _identify_zone_c_starts(labels)
    start_keys = set(zip(starts["pair"], starts["date"].dt.strftime("%Y-%m-%d"), starts["direction"]))

    pairs = labels["pair"].unique().tolist()
    data_dir = Path(data_dir).resolve()

    eligible = []
    for pair in pairs:
        try:
            df = load_pair_csv(pair, data_dir)
        except FileNotFoundError:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        for direction in ["long", "short"]:
            for i in range(min_bars, len(df)):
                date_str = df.iloc[i]["date_str"]
                if (pair, date_str, direction) in start_keys:
                    continue
                eligible.append({"pair": pair, "bar_idx": i, "date": df.iloc[i]["date"], "direction": direction})
    return pd.DataFrame(eligible)


def _build_control_metrics(
    labels_path: Path,
    data_dir: Path,
    target_count: int,
    seed: int = RANDOM_SEED,
    min_bars: int = MIN_BARS,
) -> pd.DataFrame:
    labels = _load_labels(labels_path)
    starts = _identify_zone_c_starts(labels)
    count_per_pair = starts.groupby("pair").size()
    if count_per_pair.empty:
        return pd.DataFrame()

    eligible = _get_eligible_control_bars(labels, data_dir, min_bars=min_bars)
    if eligible.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    data_dir = Path(data_dir).resolve()
    rows = []

    for pair, n_target in count_per_pair.items():
        pool = eligible[eligible["pair"] == pair]
        if len(pool) < n_target:
            indices = pool.index.tolist()
        else:
            indices = rng.choice(pool.index, size=int(n_target), replace=False).tolist()
        for idx in indices:
            r = pool.loc[idx]
            try:
                df = load_pair_csv(r["pair"], data_dir)
            except FileNotFoundError:
                continue
            df = df.sort_values("date").reset_index(drop=True)
            bar_idx = int(r["bar_idx"])
            metrics = _compute_precursor_metrics(df, bar_idx)
            if not metrics:
                continue
            out_row = {"pair": r["pair"], "date": r["date"], "direction": r["direction"], **metrics}
            rows.append(out_row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_comparison_summary(zone_df: pd.DataFrame, ctrl_df: pd.DataFrame) -> pd.DataFrame:
    if zone_df.empty:
        return _diagnostic_summary_row()
    metric_cols = [c for c in zone_df.columns if c not in ("pair", "date", "direction")]
    rows = []
    for col in metric_cols:
        if col not in ctrl_df.columns or ctrl_df.empty:
            continue
        z = pd.to_numeric(zone_df[col], errors="coerce").dropna()
        c = pd.to_numeric(ctrl_df[col], errors="coerce").dropna()
        if len(z) == 0 or len(c) == 0:
            continue
        diff = z.mean() - c.mean()
        med_diff = z.median() - c.median()
        z_p25, z_p50, z_p75 = z.quantile([0.25, 0.5, 0.75]).values
        c_p25, c_p50, c_p75 = c.quantile([0.25, 0.5, 0.75]).values
        rows.append(
            {
                "metric": col,
                "mean_diff_zoneC_minus_control": diff,
                "median_diff_zoneC_minus_control": med_diff,
                "zoneC_p25": z_p25,
                "zoneC_p50": z_p50,
                "zoneC_p75": z_p75,
                "control_p25": c_p25,
                "control_p50": c_p50,
                "control_p75": c_p75,
            }
        )

    summary = pd.DataFrame(rows)

    extreme_rows = []
    if not ctrl_df.empty:
        for col in metric_cols:
            if col not in ctrl_df.columns:
                continue
            z = pd.to_numeric(zone_df[col], errors="coerce").dropna()
            c = pd.to_numeric(ctrl_df[col], errors="coerce").dropna()
            if len(z) == 0:
                continue
            threshold_high = c.quantile(0.9) if len(c) > 0 else np.nan
            threshold_low = c.quantile(0.1) if len(c) > 0 else np.nan
            pct_above_90 = (z >= threshold_high).mean() * 100 if pd.notna(threshold_high) else np.nan
            pct_below_10 = (z <= threshold_low).mean() * 100 if pd.notna(threshold_low) else np.nan
            extreme_rows.append(
                {
                    "metric": col,
                    "pct_zoneC_above_control_90pct": pct_above_90,
                    "pct_zoneC_below_control_10pct": pct_below_10,
                }
            )
    extreme_df = pd.DataFrame(extreme_rows)
    if not extreme_df.empty:
        summary = summary.merge(extreme_df, on="metric", how="left")

    if summary.empty:
        return _diagnostic_summary_row()
    return summary


def _diagnostic_summary_row() -> pd.DataFrame:
    """Return non-empty summary with one diagnostic row (when no metric comparison possible)."""
    return pd.DataFrame([{
        "metric": "diagnostic",
        "mean_diff_zoneC_minus_control": np.nan,
        "median_diff_zoneC_minus_control": np.nan,
        "zoneC_p25": np.nan,
        "zoneC_p50": np.nan,
        "zoneC_p75": np.nan,
        "control_p25": np.nan,
        "control_p50": np.nan,
        "control_p75": np.nan,
        "pct_zoneC_above_control_90pct": np.nan,
        "pct_zoneC_below_control_10pct": np.nan,
    }])


def run_phaseD6A(
    labels_path: Path | str,
    data_dir: Path | str,
    out_dir: Path | str,
    min_bars: int = MIN_BARS,
) -> dict[str, Path]:
    """Main entry point. Writes zoneC_precursor_metrics.csv, control_precursor_metrics.csv, precursor_comparison_summary.csv."""
    labels_path = Path(labels_path).resolve()
    data_dir = Path(data_dir).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zone_df, _stats = _build_zone_c_metrics(labels_path, data_dir, min_bars=min_bars)
    ctrl_df = _build_control_metrics(
        labels_path, data_dir, target_count=len(zone_df), min_bars=min_bars
    )
    summary_df = _build_comparison_summary(zone_df, ctrl_df)
    if summary_df.empty:
        summary_df = _diagnostic_summary_row()

    zone_path = out_dir / "zoneC_precursor_metrics.csv"
    ctrl_path = out_dir / "control_precursor_metrics.csv"
    summary_path = out_dir / "precursor_comparison_summary.csv"

    zone_df.to_csv(zone_path, index=False)
    ctrl_df.to_csv(ctrl_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    return {
        "zoneC_precursor_metrics": zone_path,
        "control_precursor_metrics": ctrl_path,
        "precursor_comparison_summary": summary_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase D-6A Ignition Precursor Analysis (analytics only)"
    )
    parser.add_argument("--labels", required=True, help="Path to opportunity_labels.csv or .parquet")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing OHLCV CSVs (e.g. data/daily)",
    )
    parser.add_argument(
        "--outdir",
        default="results/phaseD/d6a_precursor",
        help="Output directory",
    )
    args = parser.parse_args()

    run_phaseD6A(
        labels_path=args.labels,
        data_dir=args.data_dir,
        out_dir=args.outdir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
