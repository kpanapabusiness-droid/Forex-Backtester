"""v2.0 archetype diagnostic — Step 1: path-shape feature engineering.

Four outcome-blind path-shape features per trade. All defined in terms
of the in-trade held window (bar_offset in [0, bars_held]):

  monotonicity_ratio_in_profit
      Among bars where close_r > 0, fraction where close_r >= the close_r
      at the previous in-profit bar (advance includes ties).
      Edge: zero in-profit bars -> 0.

  local_peaks_count
      Count of bars in [1, bars_held] where mfe_so_far_r strictly
      increased from the previous bar (i.e. new running peak).
      Edge: bars_held = 0 -> 0.

  pullback_magnitude_median
      For each pair of consecutive local peaks, take (mfe_so_far_r at
      earlier peak) - min(mfe_so_far_r over bars between peaks); median
      across pairs. Edge: <2 local peaks -> 0.

  time_to_peak_mfe_relative
      time_to_peak_mfe / max(bars_held, 1), capped at 1.0.
      Edge: trade never in profit -> 0.

Loader normalisation reused from `scripts/v1_3_calibration/load_paths.py`
(R = SL distance = 2 x ATR; Arc 1 close_r derived from cum_logret).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v1_3_calibration.load_paths import load_paths


PERCENTILES = (5, 25, 50, 75, 95)
FEATURE_COLS = (
    "monotonicity_ratio_in_profit",
    "local_peaks_count",
    "pullback_magnitude_median",
    "time_to_peak_mfe_relative",
)


def _scoped_to_held(paths: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Restrict per-bar frame to bar_offset in [0, bars_held] per trade."""
    m = meta[["trade_id", "bars_held"]].copy()
    p = paths.merge(m, on="trade_id", how="inner", validate="many_to_one")
    p = p[(p["bar_offset"] >= 0) & (p["bar_offset"] <= p["bars_held"])]
    return p.sort_values(["trade_id", "bar_offset"])


def _monotonicity_in_profit(p: pd.DataFrame) -> pd.Series:
    """For bars with close_r > 0, fraction where close_r >= prior in-profit close_r."""
    # Filter to in-profit rows, then compute per-trade advance-fraction.
    ip = p[p["close_r"] > 0][["trade_id", "bar_offset", "close_r"]].copy()
    ip = ip.sort_values(["trade_id", "bar_offset"])
    ip["close_prev"] = ip.groupby("trade_id", sort=False)["close_r"].shift(1)
    ip["advanced"]   = ip["close_r"] >= ip["close_prev"]
    # First in-profit bar has no prior in-profit comparator -> not counted as advance,
    # not counted as decline either. We sum advances and divide by COUNT OF COMPARED
    # PAIRS (in-profit bars with a prior in-profit bar). If no comparator pairs -> 0.
    has_prev = ip["close_prev"].notna()
    num = (ip["advanced"] & has_prev).groupby(ip["trade_id"]).sum()
    den = has_prev.groupby(ip["trade_id"]).sum()
    ratio = (num / den.replace(0, np.nan)).fillna(0.0)
    return ratio.astype("float32")


def _local_peaks(p: pd.DataFrame) -> pd.Series:
    """Bars in [1, bars_held] where mfe_so_far_r strictly increased from prior bar."""
    p = p[["trade_id", "bar_offset", "mfe_so_far_r"]].copy()
    p["mfe_prev"] = p.groupby("trade_id", sort=False)["mfe_so_far_r"].shift(1)
    is_peak = (p["mfe_so_far_r"] > p["mfe_prev"]).fillna(False)
    return is_peak.groupby(p["trade_id"]).sum().astype("int32")


def _pullback_median(p: pd.DataFrame) -> pd.Series:
    """Median pullback between consecutive local peaks.

    Implementation note: the spec text says "mfe_so_far_r at earlier peak -
    min(mfe_so_far_r over bars between)" — but mfe_so_far_r is the per-bar
    RUNNING maximum and therefore non-decreasing, so that literal expression
    is identically 0 across all trades. The intended metric, per logical
    reading + downstream "Random walk / Stepwise climber" archetype labels,
    is the magnitude of close_r retrace between peaks: (mfe_so_far_r at
    earlier peak) - (min close_r over bars strictly between consecutive
    peak bars). This is a deviation from the literal spec wording and is
    flagged in the report.
    """
    p = p[["trade_id", "bar_offset", "close_r", "mfe_so_far_r"]].copy()
    p["mfe_prev"] = p.groupby("trade_id", sort=False)["mfe_so_far_r"].shift(1)
    p["is_peak"]  = (p["mfe_so_far_r"] > p["mfe_prev"]).fillna(False)

    results: dict[str, float] = {}
    for tid, sub in p.groupby("trade_id", sort=False):
        peaks = sub[sub["is_peak"]][["bar_offset", "mfe_so_far_r"]].to_numpy()
        if len(peaks) < 2:
            results[tid] = 0.0
            continue
        bar_off = sub["bar_offset"].to_numpy()
        close_r = sub["close_r"].to_numpy()
        gaps = []
        for i in range(len(peaks) - 1):
            lo = peaks[i, 0]
            hi = peaks[i + 1, 0]
            if hi - lo <= 1:
                gaps.append(0.0)
                continue
            mask = (bar_off > lo) & (bar_off < hi)
            if not mask.any():
                gaps.append(0.0)
                continue
            trough = float(np.nanmin(close_r[mask]))
            gaps.append(float(peaks[i, 1]) - trough)
        results[tid] = float(np.median(gaps))
    s = pd.Series(results, name="pullback_magnitude_median", dtype="float32")
    s.index.name = "trade_id"
    return s


def _time_to_peak_relative(p: pd.DataFrame, meta: pd.DataFrame) -> pd.Series:
    """time_to_peak_mfe / max(bars_held, 1), capped at 1.0; 0 if never in profit."""
    # Time to peak = bar_offset of the first bar achieving the trade's max mfe.
    # Per spec: "Trade never in profit -> 0". Use mfe_so_far_r maximum: if max
    # is <= 0, the trade was never in profit.
    idx = p.groupby("trade_id", sort=False)["mfe_so_far_r"].idxmax()
    peak = p.loc[idx, ["trade_id", "bar_offset", "mfe_so_far_r"]].set_index("trade_id")
    bars_held = meta.set_index("trade_id")["bars_held"].astype("float64")
    bars_held_safe = bars_held.where(bars_held > 0, 1.0)
    rel = peak["bar_offset"].astype("float64") / bars_held_safe
    rel = rel.clip(upper=1.0)
    # Never-in-profit (max mfe <= 0): set 0.
    rel = rel.where(peak["mfe_so_far_r"] > 0, 0.0)
    return rel.astype("float32")


def compute_path_features(
    paths: pd.DataFrame, meta: pd.DataFrame
) -> pd.DataFrame:
    """Compute the four path-shape features per trade.

    Returns one row per trade with columns trade_id, pair, plus the four
    feature columns. Row order matches meta order.
    """
    held = _scoped_to_held(paths, meta)

    mono = _monotonicity_in_profit(held)
    peaks = _local_peaks(held)
    pullback = _pullback_median(held)
    ttp = _time_to_peak_relative(held, meta)

    out = meta[["trade_id", "pair"]].copy()
    out["monotonicity_ratio_in_profit"] = (
        mono.reindex(meta["trade_id"]).fillna(0.0).astype("float32").to_numpy()
    )
    out["local_peaks_count"] = (
        peaks.reindex(meta["trade_id"]).fillna(0).astype("int32").to_numpy()
    )
    out["pullback_magnitude_median"] = (
        pullback.reindex(meta["trade_id"]).fillna(0.0).astype("float32").to_numpy()
    )
    out["time_to_peak_mfe_relative"] = (
        ttp.reindex(meta["trade_id"]).fillna(0.0).astype("float32").to_numpy()
    )
    return out


def summarise_distributions(features: pd.DataFrame) -> pd.DataFrame:
    """Per-feature percentile + mean/std/CV summary."""
    rows = []
    for col in FEATURE_COLS:
        s = features[col].astype("float64")
        mean = float(s.mean())
        std = float(s.std())
        cv = (std / abs(mean)) if mean != 0.0 else float("nan")
        row = {
            "feature": col,
            "p5":  float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "mean": mean,
            "std": std,
            "cv":  cv,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def degenerate_flags(features: pd.DataFrame) -> pd.DataFrame:
    """Flag any feature with >80% of trades at a single value."""
    rows = []
    n = len(features)
    for col in FEATURE_COLS:
        vc = features[col].value_counts()
        if len(vc) == 0:
            rows.append({"feature": col, "is_degenerate": True, "top_value": float("nan"), "top_share": 1.0})
            continue
        top_value = float(vc.index[0])
        top_share = float(vc.iloc[0] / n)
        rows.append({
            "feature": col,
            "is_degenerate": bool(top_share > 0.80),
            "top_value": top_value,
            "top_share": top_share,
        })
    return pd.DataFrame(rows)


def run_for_dataset(name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wrapper: load + compute features + distribution summary + degeneracy flags."""
    paths, meta = load_paths(name)
    feats = compute_path_features(paths, meta)
    dist = summarise_distributions(feats)
    degen = degenerate_flags(feats)
    return feats, dist, degen
