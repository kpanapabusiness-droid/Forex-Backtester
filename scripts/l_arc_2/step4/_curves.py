"""Cluster-0 time-exit curve (pre-eval gate for candidate 2).

For cluster_id == 0 trades only (sentinel -2 excluded), for each h in
{120, 144, 168, 192, 216, 240}, run simulate_time_exit_h and report
mean_net_r, n_trades_active_at_h, n_trades_sl_before_h, mean_capture_ratio.

capture_ratio_i = net_r_i / (fwd_mfe_h_at_h_i / 2.0)
(MFE in R-units; SL = 2 ATR ⇒ MFE_ATR / 2.0 = MFE in R).

Gate rule for cand 2: drop if mean_net_r(h=240) <= mean_net_r(h=120) - 0.05 R.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _common as C
from . import _simulator as S

CURVE_HORIZONS = (120, 144, 168, 192, 216, 240)

FWD_MFE_AVAILABLE = (24, 48, 72, 120, 240, 360, 480)


def _fwd_mfe_col_for(h: int) -> str:
    """Pick the fwd_mfe column at horizon closest to h that is <= h."""
    # signals_features.csv supplies these exact horizons. Use h if present,
    # otherwise nearest <= h, otherwise nearest >= h.
    if h in FWD_MFE_AVAILABLE:
        return f"fwd_mfe_h{h}_atr"
    leq = [x for x in FWD_MFE_AVAILABLE if x <= h]
    if leq:
        return f"fwd_mfe_h{max(leq)}_atr"
    return f"fwd_mfe_h{min(FWD_MFE_AVAILABLE)}_atr"


def compute_cluster_0_time_exit_curve(
    signals: pd.DataFrame,
    paths: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-h summary for cluster_id == 0 trades (sentinel -2 excluded)."""
    merged = signals.merge(clusters, on="trade_id", how="left")
    sub = merged[merged[C.CLUSTER_COL_INTERNAL] == 0].copy()
    tids = sub["trade_id"].values

    rows = []
    for h in CURVE_HORIZONS:
        sim = S.simulate_time_exit_h(
            horizon=h, trade_ids=tids, signals=sub, paths_long=paths,
        )
        # Active at h = not SL-hit before h.
        # sim returns exit_reason 'sl_hit' for trades that SL'd at any bar <=h.
        # "Active at h" we define as not SL'd at any earlier bar — i.e. exit_reason != sl_hit
        n_sl_before_h = int((sim["exit_reason"] == "sl_hit").sum())
        n_active_at_h = int(len(sim) - n_sl_before_h)

        mean_net_r = float(np.nanmean(sim["net_r"].values))

        # Capture ratio
        mfe_col = _fwd_mfe_col_for(h)
        sim2 = sim.merge(sub[["trade_id", mfe_col]], on="trade_id", how="left")
        denom = sim2[mfe_col].values / 2.0
        with np.errstate(invalid="ignore", divide="ignore"):
            cap = np.where(denom > 0, sim2["net_r"].values / denom, np.nan)
        mean_cap = float(np.nanmean(cap))

        rows.append({
            "h_bars": h,
            "mean_net_r": mean_net_r,
            "n_trades_active_at_h": n_active_at_h,
            "n_trades_sl_before_h": n_sl_before_h,
            "mean_capture_ratio": mean_cap,
        })

    return pd.DataFrame(rows)


def cluster_0_curve_gate_pass(curve_df: pd.DataFrame,
                              tolerance_r: float = C.CLUSTER0_CURVE_TOLERANCE_R) -> bool:
    """Gate rule: PASS iff mean_net_r(h=240) > mean_net_r(h=120) - tolerance."""
    r_120 = float(curve_df.loc[curve_df["h_bars"] == 120, "mean_net_r"].iloc[0])
    r_240 = float(curve_df.loc[curve_df["h_bars"] == 240, "mean_net_r"].iloc[0])
    return r_240 > (r_120 - tolerance_r)
