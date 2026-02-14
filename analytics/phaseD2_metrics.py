"""
Phase D-2 Lift Harness — pure metrics computation.

Computes alignment metrics between signals (fire at t) and truth labels (Zone A/B/C).
No ROI, PnL, or backtest logic. Deterministic, side-effect free.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

ZONE_COLS = ("zone_a_1r_10", "zone_b_3r_20", "zone_c_6r_40")
MIN_FIRE_GLOBAL = 200
MIN_FIRE_PAIR = 30


def _to_bool_int(ser: pd.Series) -> pd.Series:
    """Convert zone columns to 0/1 for safe arithmetic."""
    return (ser.fillna(False).astype(bool)).astype(int)


def compute_metrics(
    df_joined: pd.DataFrame,
    *,
    signal_col: str = "signal",
    signal_name_col: str = "signal_name",
) -> dict[str, pd.DataFrame]:
    """
    Compute lift metrics from joined signals+labels dataframe.

    Returns dict with:
      - metrics_global: one row per signal_name
      - metrics_split: one row per (signal_name, dataset_split)
      - metrics_pair: one row per (signal_name, pair)
    """
    if signal_name_col not in df_joined.columns:
        raise ValueError(f"Missing column {signal_name_col!r}")
    if signal_col not in df_joined.columns:
        raise ValueError(f"Missing column {signal_col!r}")
    for z in ZONE_COLS:
        if z not in df_joined.columns:
            raise ValueError(f"Missing zone column {z!r}")

    # Ensure numeric
    fire = df_joined[signal_col].fillna(0).astype(int)
    fire = (fire != 0).astype(int)

    results: dict[str, list[dict[str, Any]]] = {
        "global": [],
        "split": [],
        "pair": [],
    }

    for name, grp in df_joined.groupby(signal_name_col, sort=True):
        grp_fire = grp[signal_col].fillna(0).astype(int)
        grp_fire_bool = (grp_fire != 0).astype(int)
        n = len(grp)
        f = int(grp_fire_bool.sum())

        # Global
        row_global = _metric_row(
            name, grp, grp_fire_bool, n, f, split=None, pair=None
        )
        results["global"].append(row_global)

        # By split
        if "dataset_split" in grp.columns:
            for split_val, sgrp in grp.groupby("dataset_split", sort=True):
                sf = int(sgrp[signal_col].fillna(0).astype(int).ne(0).sum())
                row_split = _metric_row(name, sgrp, sgrp[signal_col].fillna(0).astype(int).ne(0).astype(int), len(sgrp), sf, split=split_val, pair=None)
                results["split"].append(row_split)

        # By pair
        if "pair" in grp.columns:
            for pair_val, pgrp in grp.groupby("pair", sort=True):
                pf = int(pgrp[signal_col].fillna(0).astype(int).ne(0).sum())
                row_pair = _metric_row(name, pgrp, pgrp[signal_col].fillna(0).astype(int).ne(0).astype(int), len(pgrp), pf, split=None, pair=pair_val)
                results["pair"].append(row_pair)

    return {
        "metrics_global": pd.DataFrame(results["global"]),
        "metrics_split": pd.DataFrame(results["split"]) if results["split"] else pd.DataFrame(),
        "metrics_pair": pd.DataFrame(results["pair"]) if results["pair"] else pd.DataFrame(),
    }


def _metric_row(
    signal_name: str,
    grp: pd.DataFrame,
    fire: pd.Series,
    n: int,
    f: int,
    *,
    split: str | None = None,
    pair: str | None = None,
) -> dict[str, Any]:
    """Build one metric row for global/split/pair."""
    row: dict[str, Any] = {"signal_name": signal_name}
    if split is not None:
        row["dataset_split"] = split
    if pair is not None:
        row["pair"] = pair

    row["n"] = n
    row["fired"] = f
    row["fire_rate"] = f / n if n > 0 else 0.0
    row["min_fire_ok"] = f >= MIN_FIRE_GLOBAL
    if pair is not None:
        row["min_fire_ok_pair"] = f >= MIN_FIRE_PAIR

    for zone in ZONE_COLS:
        a = _to_bool_int(grp[zone])
        base_rate = a.mean()
        hit_rate = a[fire.astype(bool)].mean() if fire.sum() > 0 else 0.0
        lift = hit_rate / base_rate if base_rate > 0 else 0.0
        row[f"p_{zone}"] = base_rate
        row[f"p_{zone}_given_fire"] = hit_rate
        row[f"lift_{zone}"] = lift

    return row


def compute_coverage(
    df_joined: pd.DataFrame,
    *,
    signal_col: str = "signal",
    signal_name_col: str = "signal_name",
) -> pd.DataFrame:
    """
    Compute coverage (recall-style): fraction of opportunities caught by signal.

    For each zone: coverage = count(fire==1 AND zone==1) / count(zone==1)
    """
    if signal_name_col not in df_joined.columns or signal_col not in df_joined.columns:
        raise ValueError(f"Need {signal_col} and {signal_name_col}")

    rows: list[dict[str, Any]] = []
    for name, grp in df_joined.groupby(signal_name_col, sort=True):
        grp_fire = grp[signal_col].fillna(0).astype(int).ne(0)
        row: dict[str, Any] = {"signal_name": name}
        for zone in ZONE_COLS:
            a = _to_bool_int(grp[zone])
            total_opp = int(a.sum())
            caught = int((grp_fire & a.astype(bool)).sum())
            row[f"coverage_{zone}"] = caught / total_opp if total_opp > 0 else 0.0
            row[f"caught_{zone}"] = caught
            row[f"total_{zone}"] = total_opp
        rows.append(row)

    return pd.DataFrame(rows)
