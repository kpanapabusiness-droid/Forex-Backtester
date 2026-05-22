"""Per-candidate action evaluators.

Each function returns a DataFrame with cols:
    trade_id, fold, pair, fire_bar, action_bar, exit_bar, exit_reason,
    net_r, gross_r, spread_cost_r, mfe_at_exit, mae_at_exit

Filter candidates (4, 5-8, 9, 10):
    Apply filter predicate; retained trades use verbatim signals_features.net_r.

Exit-only (11): simulate_time_exit_h(h=240) on all trades.

Cluster-cond exit (1): predicted C1 -> close_at_bar_t, predicted C0 -> verbatim h=120.
Cluster-cond exit h=240 (2): predicted C1 -> close_at_bar_t, predicted C0 -> simulate h=240.
Delayed entry (3): predicted C0 -> simulate_delayed_entry(t), predicted C1 -> skip (not in retained).

Trades with bars_held < t or sentinel -2 cluster are excluded from cluster-conditional
candidates' eval scope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import _simulator as S

JPY_PAIRS = {
    "AUD_JPY",
    "CAD_JPY",
    "CHF_JPY",
    "EUR_JPY",
    "GBP_JPY",
    "NZD_JPY",
    "USD_JPY",
}


# =============================================================================
# Filter predicates
# =============================================================================


def make_filter_predicate(slug: str, signals_full: pd.DataFrame):
    """Return a callable(signals_df) -> bool mask. Thresholds computed on the
    full pool (signals_full) for stability across subset evaluations.
    """
    if slug == "filter_jpy_pairs":
        return lambda df: df["pair"].isin(JPY_PAIRS).values

    if slug.startswith("filter_basket_") and slug.endswith("_above_p50"):
        ccy = slug.replace("filter_basket_", "").replace("_above_p50", "").upper()
        col = f"currency_basket_3h_{ccy}"
        p50 = float(np.nanpercentile(signals_full[col].values, 50))
        return lambda df, _col=col, _p50=p50: df[_col].values >= _p50

    if slug == "filter_atr_at_signal_above_p50":
        col = "atr_at_signal_1h"
        p50 = float(np.nanpercentile(signals_full[col].values, 50))
        return lambda df, _col=col, _p50=p50: df[_col].values >= _p50

    if slug == "filter_concurrent_signals_above_p75":
        col = "concurrent_signals_same_bar"
        p75 = float(np.nanpercentile(signals_full[col].values, 75))
        return lambda df, _col=col, _p75=p75: df[_col].values >= _p75

    raise ValueError(f"unknown filter slug: {slug}")


# =============================================================================
# Filter action: keep verbatim outcomes for matched rows
# =============================================================================


def run_filter_action(slug: str, signals: pd.DataFrame) -> pd.DataFrame:
    pred = make_filter_predicate(slug, signals)
    mask = pred(signals)
    kept = signals.loc[mask].copy()

    out = pd.DataFrame(
        {
            "trade_id": kept["trade_id"].values,
            "fold": kept["fold_id"].values,
            "pair": kept["pair"].values,
            "fire_bar": kept["signal_bar_ts"].values,
            "action_bar": kept["signal_bar_ts"].values,  # no action; same as fire
            "exit_bar": kept["exit_bar_ts"].values,
            "exit_reason": kept["exit_reason"].values,
            "net_r": kept["net_r"].values,
            "gross_r": kept["gross_r"].values,
            "spread_cost_r": kept["spread_cost_R"].values,
            "mfe_at_exit": kept["mfe_held_atr"].values,
            "mae_at_exit": kept["mae_held_atr"].values,
        }
    )
    return out.sort_values("trade_id").reset_index(drop=True)


# =============================================================================
# Exit-only h=240 (cand 11)
# =============================================================================


def run_exit_only_h240(signals: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    tids = signals["trade_id"].values
    sim = S.simulate_time_exit_h(
        horizon=240,
        trade_ids=tids,
        signals=signals,
        paths_long=paths,
    )
    meta = signals.set_index("trade_id").loc[tids, ["fold_id", "pair", "signal_bar_ts"]]
    out = pd.DataFrame(
        {
            "trade_id": tids,
            "fold": meta["fold_id"].values,
            "pair": meta["pair"].values,
            "fire_bar": meta["signal_bar_ts"].values,
            "action_bar": sim["action_bar"].values,
            "exit_bar": sim["action_bar"].values,
            "exit_reason": sim["exit_reason"].values,
            "net_r": sim["net_r"].values,
            "gross_r": sim["gross_r"].values,
            "spread_cost_r": sim["spread_cost_r"].values,
            "mfe_at_exit": sim["mfe_at_exit"].values,
            "mae_at_exit": sim["mae_at_exit"].values,
        }
    )
    return out.sort_values("trade_id").reset_index(drop=True)


# =============================================================================
# Cluster-conditional exit (cand 1: C1->close@t, C0->verbatim h=120)
# =============================================================================


def run_exit_cluster_cond(
    signals_with_clu: pd.DataFrame,
    paths: pd.DataFrame,
    predictions: pd.DataFrame,
    t: int,
) -> pd.DataFrame:
    """
    predictions: from fit_predict_cluster — has cols trade_id, predicted_cluster, fold.
    Only F6/F7 predictions are used (validate folds). Active-at-t valid-cluster only.
    For predicted C1 -> close at bar t. For predicted C0 -> verbatim h=120 (from signals).
    """
    pred = predictions.set_index("trade_id")
    sig_idx = signals_with_clu.set_index("trade_id")

    # Predicted C1 trades -> simulate close at bar t
    c1_tids = pred[pred["predicted_cluster"] == 1].index.values
    c0_tids = pred[pred["predicted_cluster"] == 0].index.values

    c1_rows = pd.DataFrame()
    if len(c1_tids) > 0:
        sim_c1 = S.simulate_close_at_bar_t(
            t=t,
            trade_ids=c1_tids,
            signals=signals_with_clu,
            paths_long=paths,
        )
        meta = sig_idx.loc[c1_tids, ["fold_id", "pair", "signal_bar_ts"]]
        c1_rows = pd.DataFrame(
            {
                "trade_id": c1_tids,
                "fold": meta["fold_id"].values,
                "pair": meta["pair"].values,
                "fire_bar": meta["signal_bar_ts"].values,
                "action_bar": sim_c1["action_bar"].values,
                "exit_bar": sim_c1["action_bar"].values,
                "exit_reason": sim_c1["exit_reason"].values,
                "net_r": sim_c1["net_r"].values,
                "gross_r": sim_c1["gross_r"].values,
                "spread_cost_r": sim_c1["spread_cost_r"].values,
                "mfe_at_exit": sim_c1["mfe_at_exit"].values,
                "mae_at_exit": sim_c1["mae_at_exit"].values,
            }
        )

    # Predicted C0 trades -> verbatim h=120 from signals_features
    c0_rows = pd.DataFrame()
    if len(c0_tids) > 0:
        meta = sig_idx.loc[c0_tids]
        c0_rows = pd.DataFrame(
            {
                "trade_id": c0_tids,
                "fold": meta["fold_id"].values,
                "pair": meta["pair"].values,
                "fire_bar": meta["signal_bar_ts"].values,
                "action_bar": meta["signal_bar_ts"].values,
                "exit_bar": meta["exit_bar_ts"].values,
                "exit_reason": meta["exit_reason"].values,
                "net_r": meta["net_r"].values,
                "gross_r": meta["gross_r"].values,
                "spread_cost_r": meta["spread_cost_R"].values,
                "mfe_at_exit": meta["mfe_held_atr"].values,
                "mae_at_exit": meta["mae_held_atr"].values,
            }
        )

    out = pd.concat([c1_rows, c0_rows], axis=0, ignore_index=True)
    return out.sort_values("trade_id").reset_index(drop=True)


# =============================================================================
# Cluster-conditional exit h=240 (cand 2: C1->close@t, C0->simulate h=240)
# =============================================================================


def run_exit_cluster_cond_h240(
    signals_with_clu: pd.DataFrame,
    paths: pd.DataFrame,
    predictions: pd.DataFrame,
    t: int,
) -> pd.DataFrame:
    pred = predictions.set_index("trade_id")
    sig_idx = signals_with_clu.set_index("trade_id")

    c1_tids = pred[pred["predicted_cluster"] == 1].index.values
    c0_tids = pred[pred["predicted_cluster"] == 0].index.values

    c1_rows = pd.DataFrame()
    if len(c1_tids) > 0:
        sim_c1 = S.simulate_close_at_bar_t(
            t=t,
            trade_ids=c1_tids,
            signals=signals_with_clu,
            paths_long=paths,
        )
        meta = sig_idx.loc[c1_tids, ["fold_id", "pair", "signal_bar_ts"]]
        c1_rows = pd.DataFrame(
            {
                "trade_id": c1_tids,
                "fold": meta["fold_id"].values,
                "pair": meta["pair"].values,
                "fire_bar": meta["signal_bar_ts"].values,
                "action_bar": sim_c1["action_bar"].values,
                "exit_bar": sim_c1["action_bar"].values,
                "exit_reason": sim_c1["exit_reason"].values,
                "net_r": sim_c1["net_r"].values,
                "gross_r": sim_c1["gross_r"].values,
                "spread_cost_r": sim_c1["spread_cost_r"].values,
                "mfe_at_exit": sim_c1["mfe_at_exit"].values,
                "mae_at_exit": sim_c1["mae_at_exit"].values,
            }
        )

    c0_rows = pd.DataFrame()
    if len(c0_tids) > 0:
        sim_c0 = S.simulate_time_exit_h(
            horizon=240,
            trade_ids=c0_tids,
            signals=signals_with_clu,
            paths_long=paths,
        )
        meta = sig_idx.loc[c0_tids, ["fold_id", "pair", "signal_bar_ts"]]
        c0_rows = pd.DataFrame(
            {
                "trade_id": c0_tids,
                "fold": meta["fold_id"].values,
                "pair": meta["pair"].values,
                "fire_bar": meta["signal_bar_ts"].values,
                "action_bar": sim_c0["action_bar"].values,
                "exit_bar": sim_c0["action_bar"].values,
                "exit_reason": sim_c0["exit_reason"].values,
                "net_r": sim_c0["net_r"].values,
                "gross_r": sim_c0["gross_r"].values,
                "spread_cost_r": sim_c0["spread_cost_r"].values,
                "mfe_at_exit": sim_c0["mfe_at_exit"].values,
                "mae_at_exit": sim_c0["mae_at_exit"].values,
            }
        )

    out = pd.concat([c1_rows, c0_rows], axis=0, ignore_index=True)
    return out.sort_values("trade_id").reset_index(drop=True)


# =============================================================================
# Delayed entry (cand 3: C0->simulate_delayed_entry(t), C1->skip)
# =============================================================================


def run_delayed_entry(
    signals_with_clu: pd.DataFrame,
    paths: pd.DataFrame,
    held_ctx_t: pd.DataFrame,
    predictions: pd.DataFrame,
    t: int,
) -> pd.DataFrame:
    pred = predictions.set_index("trade_id")
    sig_idx = signals_with_clu.set_index("trade_id")

    c0_tids = pred[pred["predicted_cluster"] == 0].index.values
    # C1 trades skipped — not in output

    if len(c0_tids) == 0:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "fold",
                "pair",
                "fire_bar",
                "action_bar",
                "exit_bar",
                "exit_reason",
                "net_r",
                "gross_r",
                "spread_cost_r",
                "mfe_at_exit",
                "mae_at_exit",
            ]
        )

    sim = S.simulate_delayed_entry(
        t=t,
        trade_ids=c0_tids,
        signals=signals_with_clu,
        paths_long=paths,
        held_ctx_t=held_ctx_t,
    )
    meta = sig_idx.loc[c0_tids, ["fold_id", "pair", "signal_bar_ts"]]

    out = pd.DataFrame(
        {
            "trade_id": c0_tids,
            "fold": meta["fold_id"].values,
            "pair": meta["pair"].values,
            "fire_bar": meta["signal_bar_ts"].values,
            "action_bar": sim["action_bar"].values,
            "exit_bar": sim["action_bar"].values,
            "exit_reason": sim["exit_reason"].values,
            "net_r": sim["net_r"].values,
            "gross_r": sim["gross_r"].values,
            "spread_cost_r": sim["spread_cost_r"].values,
            "mfe_at_exit": sim["mfe_at_exit"].values,
            "mae_at_exit": sim["mae_at_exit"].values,
        }
    )
    return out.sort_values("trade_id").reset_index(drop=True)
