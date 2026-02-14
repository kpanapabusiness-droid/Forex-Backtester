"""
Phase D-6F.1: Opportunity geometry analysis from clean labels.

Computes: How often does favorable excursion reach Y before adverse reaches X?
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PHASE_D_DISCOVERY_END = "2022-12-31"
X_VALUES = (1, 2, 3)
Y_VALUES = (0.67, 1.0, 2.0, 3.0, 6.0)
DIRECTIONS = ("long", "short")


def _to_bool(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(bool)


def _valid_mask(df: pd.DataFrame) -> pd.Series:
    valid_atr = _to_bool(df["valid_atr"]) if "valid_atr" in df.columns else pd.Series(True, index=df.index)
    valid_ref = _to_bool(df["valid_ref"]) if "valid_ref" in df.columns else pd.Series(True, index=df.index)
    valid_h40 = _to_bool(df["valid_h40"]) if "valid_h40" in df.columns else pd.Series(True, index=df.index)
    return valid_atr & valid_ref & valid_h40


def _get_mfe_col(direction: str, x: int) -> str:
    return f"clean_mfe_{direction}_x{x}_h40"


def compute_pooled_prob_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled probability matrix: direction, x, y, n_valid, n_hit, rate."""
    mask = _valid_mask(df)
    sub = df[mask].copy()
    if sub.empty:
        return pd.DataFrame(columns=["direction", "x", "y", "n_valid", "n_hit", "rate"])

    rows = []
    for direction in DIRECTIONS:
        for x in X_VALUES:
            col = _get_mfe_col(direction, x)
            if col not in sub.columns:
                continue
            vals = sub[col].dropna()
            vals = vals[np.isfinite(vals)]
            n_valid = len(vals)
            if n_valid == 0:
                continue
            for y in Y_VALUES:
                n_hit = int((vals >= y).sum())
                rate = round(n_hit / n_valid, 6)
                rows.append({
                    "direction": direction,
                    "x": x,
                    "y": float(y),
                    "n_valid": n_valid,
                    "n_hit": n_hit,
                    "rate": rate,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["direction", "x", "y"]).reset_index(drop=True)
    return out


def compute_per_pair_prob_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pair probability matrix: pair, direction, x, y, n_valid, n_hit, rate."""
    mask = _valid_mask(df)
    sub = df[mask].copy()
    if sub.empty or "pair" not in sub.columns:
        return pd.DataFrame(columns=["pair", "direction", "x", "y", "n_valid", "n_hit", "rate"])

    rows = []
    for (pair, grp), direction, x, y in [
        ((p, g), d, xi, yi)
        for (p, g) in sub.groupby("pair", sort=True)
        for d in DIRECTIONS
        for xi in X_VALUES
        for yi in Y_VALUES
    ]:
        col = _get_mfe_col(direction, x)
        if col not in grp.columns:
            continue
        vals = grp[col].dropna()
        vals = vals[np.isfinite(vals)]
        n_valid = len(vals)
        if n_valid == 0:
            continue
        n_hit = int((vals >= y).sum())
        rate = round(n_hit / n_valid, 6)
        rows.append({
            "pair": pair,
            "direction": direction,
            "x": x,
            "y": float(y),
            "n_valid": n_valid,
            "n_hit": n_hit,
            "rate": rate,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["pair", "direction", "x", "y"]).reset_index(drop=True)
    return out


def compute_pooled_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled quantiles: direction, x, n, mean, std, p50, p75, p90, p95, p99, max."""
    mask = _valid_mask(df)
    sub = df[mask].copy()
    if sub.empty:
        return pd.DataFrame(columns=["direction", "x", "n", "mean", "std", "p50", "p75", "p90", "p95", "p99", "max"])

    rows = []
    for direction in DIRECTIONS:
        for x in X_VALUES:
            col = _get_mfe_col(direction, x)
            if col not in sub.columns:
                continue
            vals = sub[col].dropna()
            vals = vals[np.isfinite(vals)]
            n = len(vals)
            if n == 0:
                continue
            rows.append({
                "direction": direction,
                "x": x,
                "n": n,
                "mean": round(float(vals.mean()), 6),
                "std": round(float(vals.std()) if n > 1 else 0.0, 6),
                "p50": round(float(vals.quantile(0.50)), 6),
                "p75": round(float(vals.quantile(0.75)), 6),
                "p90": round(float(vals.quantile(0.90)), 6),
                "p95": round(float(vals.quantile(0.95)), 6),
                "p99": round(float(vals.quantile(0.99)), 6),
                "max": round(float(vals.max()), 6),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["direction", "x"]).reset_index(drop=True)
    return out


def compute_per_pair_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pair quantiles: pair, direction, x, same stat columns."""
    mask = _valid_mask(df)
    sub = df[mask].copy()
    if sub.empty or "pair" not in sub.columns:
        return pd.DataFrame(columns=["pair", "direction", "x", "n", "mean", "std", "p50", "p75", "p90", "p95", "p99", "max"])

    rows = []
    for (pair, grp) in sub.groupby("pair", sort=True):
        for direction in DIRECTIONS:
            for x in X_VALUES:
                col = _get_mfe_col(direction, x)
                if col not in grp.columns:
                    continue
                vals = grp[col].dropna()
                vals = vals[np.isfinite(vals)]
                n = len(vals)
                if n == 0:
                    continue
                rows.append({
                    "pair": pair,
                    "direction": direction,
                    "x": x,
                    "n": n,
                    "mean": round(float(vals.mean()), 6),
                    "std": round(float(vals.std()) if n > 1 else 0.0, 6),
                    "p50": round(float(vals.quantile(0.50)), 6),
                    "p75": round(float(vals.quantile(0.75)), 6),
                    "p90": round(float(vals.quantile(0.90)), 6),
                    "p95": round(float(vals.quantile(0.95)), 6),
                    "p99": round(float(vals.quantile(0.99)), 6),
                    "max": round(float(vals.max()), 6),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["pair", "direction", "x"]).reset_index(drop=True)
    return out


def _add_split(df: pd.DataFrame, discovery_end: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(discovery_end)
    df = df.copy()
    df["_split"] = np.where(df["date"] <= cutoff, "discovery", "validation")
    return df


def compute_pooled_stability_prob_matrix(
    df: pd.DataFrame, discovery_end: str = PHASE_D_DISCOVERY_END
) -> pd.DataFrame:
    """Pooled stability: direction, x, y, n_valid_discovery, rate_discovery, n_valid_validation, rate_validation, delta, ratio."""
    mask = _valid_mask(df)
    sub = _add_split(df[mask].copy(), discovery_end)
    if sub.empty:
        return pd.DataFrame(columns=[
            "direction", "x", "y",
            "n_valid_discovery", "rate_discovery",
            "n_valid_validation", "rate_validation",
            "delta", "ratio",
        ])

    rows = []
    for direction in DIRECTIONS:
        for x in X_VALUES:
            col = _get_mfe_col(direction, x)
            if col not in sub.columns:
                continue
            for y in Y_VALUES:
                disc = sub[sub["_split"] == "discovery"][col].dropna()
                disc = disc[np.isfinite(disc)]
                val = sub[sub["_split"] == "validation"][col].dropna()
                val = val[np.isfinite(val)]
                n_disc = len(disc)
                n_val = len(val)
                rate_disc = (disc >= y).mean() if n_disc else 0.0
                rate_val = (val >= y).mean() if n_val else 0.0
                rate_disc = round(float(rate_disc), 6)
                rate_val = round(float(rate_val), 6)
                delta = round(rate_val - rate_disc, 6)
                denom = max(rate_disc, 1e-12)
                ratio = round(rate_val / denom, 6)
                rows.append({
                    "direction": direction,
                    "x": x,
                    "y": float(y),
                    "n_valid_discovery": n_disc,
                    "rate_discovery": rate_disc,
                    "n_valid_validation": n_val,
                    "rate_validation": rate_val,
                    "delta": delta,
                    "ratio": ratio,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["direction", "x", "y"]).reset_index(drop=True)
    return out


def compute_per_pair_stability_prob_matrix(
    df: pd.DataFrame, discovery_end: str = PHASE_D_DISCOVERY_END
) -> pd.DataFrame:
    """Per-pair stability: pair, direction, x, y, same stability columns."""
    mask = _valid_mask(df)
    sub = _add_split(df[mask].copy(), discovery_end)
    if sub.empty or "pair" not in sub.columns:
        return pd.DataFrame(columns=[
            "pair", "direction", "x", "y",
            "n_valid_discovery", "rate_discovery",
            "n_valid_validation", "rate_validation",
            "delta", "ratio",
        ])

    rows = []
    for (pair, grp) in sub.groupby("pair", sort=True):
        for direction in DIRECTIONS:
            for x in X_VALUES:
                col = _get_mfe_col(direction, x)
                if col not in grp.columns:
                    continue
                for y in Y_VALUES:
                    disc = grp[grp["_split"] == "discovery"][col].dropna()
                    disc = disc[np.isfinite(disc)]
                    val = grp[grp["_split"] == "validation"][col].dropna()
                    val = val[np.isfinite(val)]
                    n_disc = len(disc)
                    n_val = len(val)
                    rate_disc = (disc >= y).mean() if n_disc else 0.0
                    rate_val = (val >= y).mean() if n_val else 0.0
                    rate_disc = round(float(rate_disc), 6)
                    rate_val = round(float(rate_val), 6)
                    delta = round(rate_val - rate_disc, 6)
                    denom = max(rate_disc, 1e-12)
                    ratio = round(rate_val / denom, 6)
                    rows.append({
                        "pair": pair,
                        "direction": direction,
                        "x": x,
                        "y": float(y),
                        "n_valid_discovery": n_disc,
                        "rate_discovery": rate_disc,
                        "n_valid_validation": n_val,
                        "rate_validation": rate_val,
                        "delta": delta,
                        "ratio": ratio,
                    })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["pair", "direction", "x", "y"]).reset_index(drop=True)
    return out


def run_geometry_analysis(
    df: pd.DataFrame,
    out_dir: str | Path,
    discovery_end: str = PHASE_D_DISCOVERY_END,
) -> dict[str, Path]:
    """Compute all tables and write CSVs. Returns paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Any] = {}

    prob = compute_pooled_prob_matrix(df)
    if not prob.empty:
        p = out_dir / "pooled_prob_matrix.csv"
        prob.to_csv(p, index=False, float_format="%.6f")
        paths["pooled_prob_matrix"] = p

    pp = compute_per_pair_prob_matrix(df)
    if not pp.empty:
        p = out_dir / "per_pair_prob_matrix.csv"
        pp.to_csv(p, index=False, float_format="%.6f")
        paths["per_pair_prob_matrix"] = p

    q = compute_pooled_quantiles(df)
    if not q.empty:
        p = out_dir / "pooled_quantiles.csv"
        q.to_csv(p, index=False, float_format="%.6f")
        paths["pooled_quantiles"] = p

    pq = compute_per_pair_quantiles(df)
    if not pq.empty:
        p = out_dir / "per_pair_quantiles.csv"
        pq.to_csv(p, index=False, float_format="%.6f")
        paths["per_pair_quantiles"] = p

    stab = compute_pooled_stability_prob_matrix(df, discovery_end)
    if not stab.empty:
        p = out_dir / "pooled_stability_prob_matrix.csv"
        stab.to_csv(p, index=False, float_format="%.6f")
        paths["pooled_stability_prob_matrix"] = p

    pstab = compute_per_pair_stability_prob_matrix(df, discovery_end)
    if not pstab.empty:
        p = out_dir / "per_pair_stability_prob_matrix.csv"
        pstab.to_csv(p, index=False, float_format="%.6f")
        paths["per_pair_stability_prob_matrix"] = p

    return paths
