"""Step 4 (Angle C): path-so-far features at observation point t.

For each trade, observe at t bars after entry the following features
(strict no-lookahead — only bars with bar_offset <= t used):

  close_r_at_t            close_r at bar_offset = t
  mfe_so_far_r_at_t       max(mfe_so_far_r) over [0, t]
  mae_so_far_r_at_t       min(mae_so_far_r) over [0, t]
  bars_in_profit_at_t     count of bars in [0, t] with close_r > 0
  local_peaks_so_far_at_t local peaks count over [0, t]
  monotonicity_so_far_at_t monotonicity ratio over [0, t] in-profit bars
  velocity_first_t        mfe_so_far_r_at_t / t

Trades with bars_held < t are EXCLUDED at observation point t (their
features at t are not observable). Exclusion counts are reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_path_so_far_features(
    paths: pd.DataFrame,
    meta: pd.DataFrame,
    t: int,
) -> tuple[pd.DataFrame, int]:
    """Compute features at observation point t.

    Returns:
      features  DataFrame[trade_id, 7 feature cols] — one row per ELIGIBLE trade
      excluded  count of trades excluded (bars_held < t)
    """
    bars_held = meta.set_index("trade_id")["bars_held"]
    eligible_ids = bars_held[bars_held >= t].index
    excluded = int(len(bars_held) - len(eligible_ids))

    # Scope to bars in [0, t] for eligible trades only.
    p = paths[paths["trade_id"].isin(eligible_ids)][
        ["trade_id", "bar_offset", "close_r", "mfe_so_far_r", "mae_so_far_r"]
    ]
    p = p[(p["bar_offset"] >= 0) & (p["bar_offset"] <= t)].copy()
    p = p.sort_values(["trade_id", "bar_offset"])

    g = p.groupby("trade_id", sort=False)
    out = pd.DataFrame(index=g.size().index)
    out.index.name = "trade_id"

    out["mfe_so_far_r_at_t"] = g["mfe_so_far_r"].max().astype("float64")
    out["mae_so_far_r_at_t"] = g["mae_so_far_r"].min().astype("float64")
    out["bars_in_profit_at_t"] = g["close_r"].apply(lambda s: int((s > 0).sum()))

    # close_r at bar_offset = t.
    at_t = p[p["bar_offset"] == t].set_index("trade_id")["close_r"]
    out["close_r_at_t"] = at_t.reindex(out.index).astype("float64")
    # Fallback: trades that don't have an exact bar_offset = t row -> use last
    # available within [0, t]. (Should not happen given the eligibility filter,
    # but Arc 1 etc could be sparse.)
    if out["close_r_at_t"].isna().any():
        tail = g["close_r"].last().astype("float64")
        out["close_r_at_t"] = out["close_r_at_t"].fillna(tail)

    # local peaks over [0, t]
    p["mfe_prev"] = g["mfe_so_far_r"].shift(1)
    is_peak = (p["mfe_so_far_r"] > p["mfe_prev"]).fillna(False)
    peaks = is_peak.groupby(p["trade_id"]).sum().astype("int64")
    out["local_peaks_so_far_at_t"] = peaks.reindex(out.index).fillna(0).astype("int64")

    # monotonicity over [0, t] in-profit bars
    ip = p[p["close_r"] > 0][["trade_id", "bar_offset", "close_r"]].copy()
    ip = ip.sort_values(["trade_id", "bar_offset"])
    ip["close_prev"] = ip.groupby("trade_id", sort=False)["close_r"].shift(1)
    ip["advanced"] = ip["close_r"] >= ip["close_prev"]
    has_prev = ip["close_prev"].notna()
    num = (ip["advanced"] & has_prev).groupby(ip["trade_id"]).sum()
    den = has_prev.groupby(ip["trade_id"]).sum()
    mono = (num / den.replace(0, np.nan)).fillna(0.0)
    out["monotonicity_so_far_at_t"] = mono.reindex(out.index).fillna(0.0).astype("float64")

    out["velocity_first_t"] = out["mfe_so_far_r_at_t"] / max(t, 1)

    out = out.reset_index()
    return out, excluded
