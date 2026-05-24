"""Synthetic fixtures for Step 6 tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.step_6.inputs import Step6Inputs


def build_clean_inputs(
    arc_root: Path,
    *,
    arc_name: str = "synth_arc",
    n_trades: int = 100,
    features: tuple[str, ...] = ("atr_pct", "d1_slope_sign", "kijun_dist"),
    configs_evaluated: int = 80,
    pair_set: tuple[str, ...] = ("EURUSD", "GBPUSD", "USDJPY"),
) -> Step6Inputs:
    """Build a Step6Inputs bundle with a clean synthetic pool + lineage."""
    arc_root.mkdir(parents=True, exist_ok=True)

    # Synthetic trades: timestamped, multi-pair, with R outcomes
    import numpy as np
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", periods=n_trades, freq="D", tz="UTC")
    trades = pd.DataFrame({
        "trade_id": range(1, n_trades + 1),
        "pair": [pair_set[i % len(pair_set)] for i in range(n_trades)],
        "signal_time": dates,
        "entry_time": dates + pd.Timedelta(hours=4),
        "final_r": rng.normal(0.5, 1.5, n_trades),
        "spread_pips": rng.uniform(1.0, 2.5, n_trades),
        "sl_distance_atr": rng.uniform(1.5, 2.5, n_trades),
    })

    # Synthetic feature matrix
    fm = pd.DataFrame({
        "trade_id": trades["trade_id"],
        **{f: rng.normal(0, 1, n_trades) for f in features},
    })

    # Synthetic lineage — all clean
    lineage = pd.DataFrame({
        "name": list(features),
        "feature_class": ["test"] * len(features),
        "causal_lineage": ["clean"] * len(features),
        "needs_panel": [False] * len(features),
        "description": [""] * len(features),
    })

    return Step6Inputs(
        arc_name=arc_name,
        arc_root=arc_root,
        best_candidate_config_id="A1::cfg_synth",
        best_candidate_architecture="A1",
        best_candidate_features=features,
        pool_trades=trades,
        feature_matrix=fm,
        feature_lineage=lineage,
        primary_tf="H4",
        pair_set=pair_set,
        r_safe_pct=0.005,
        sizing_convention="reset_floor",
        configs_evaluated_step5=configs_evaluated,
    )


def build_leaky_inputs(arc_root: Path) -> Step6Inputs:
    """Inputs with a forward-path feature in entry decisions → §6.1 critical fail."""
    inp = build_clean_inputs(arc_root)
    # Replace one feature with a path-shape leak
    leaky_features = tuple(
        f if f != "atr_pct" else "mfe_p50_r" for f in inp.best_candidate_features
    )
    # Update lineage table to include the new name
    new_lineage = inp.feature_lineage.copy()
    new_lineage.loc[new_lineage["name"] == "atr_pct", "name"] = "mfe_p50_r"
    new_fm = inp.feature_matrix.rename(columns={"atr_pct": "mfe_p50_r"})

    return Step6Inputs(
        arc_name=inp.arc_name,
        arc_root=inp.arc_root,
        best_candidate_config_id=inp.best_candidate_config_id,
        best_candidate_architecture=inp.best_candidate_architecture,
        best_candidate_features=leaky_features,
        pool_trades=inp.pool_trades,
        feature_matrix=new_fm,
        feature_lineage=new_lineage,
        primary_tf=inp.primary_tf,
        pair_set=inp.pair_set,
        r_safe_pct=inp.r_safe_pct,
        sizing_convention=inp.sizing_convention,
        configs_evaluated_step5=inp.configs_evaluated_step5,
    )


__all__ = ("build_clean_inputs", "build_leaky_inputs")
