"""v2.0 archetype diagnostic — Step 3: per-archetype characterisation.

For each (dataset, K, archetype): identity + path-shape centroid + forward
geometry distribution (fwd_mfe_h240 + final_r) + shape tag + mass-in-band.

Reuses normalised per-bar frame; computes forward-geometry directly.

fwd_mfe_h240   max(mfe_so_far_r) over bar_offset in [0, 240]
final_r        close_r at the last held bar (bar_offset == bars_held); falls
               back to last available bar within forward window if exact
               match not present (Arc 1 / Arc 2 forward window may truncate)
frac_reach_1R  fraction with max(mfe_so_far_r over [0, 240]) >= 1.0
frac_reach_2R  same at 2.0
frac_wrong_way fraction that touch -1.5R (mae_so_far_r <= -1.5) before any
               first +1R; uses first-cross bar offsets per the v1.3 axis 3
               "wrong way" definition (capped at 240, 240-vs-240 -> false)
pct_peak_and_collapse  max_mfe_h240 >= 1.0 AND final_r < 0
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.v1_3_calibration.metrics import shape_tag

CAP = 240
SHAPE_DIST_PERCENTILES = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99)


def _per_trade_geometry(paths: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """One row per trade: max_mfe_h240, final_r, first_1R, first_neg_1_5R, ..."""
    h240 = paths[(paths["bar_offset"] >= 0) & (paths["bar_offset"] <= CAP)]
    g = h240.groupby("trade_id", sort=False)
    out = pd.DataFrame(index=g.size().index)
    out.index.name = "trade_id"

    out["max_mfe_h240"] = g["mfe_so_far_r"].max().astype("float64")
    out["max_mae_h240"] = g["mae_so_far_r"].min().astype("float64")

    def _first_above(thresh: float) -> pd.Series:
        sub = h240.loc[h240["mfe_so_far_r"] >= thresh, ["trade_id", "bar_offset"]]
        return sub.groupby("trade_id", sort=False)["bar_offset"].min()

    def _first_below(thresh: float) -> pd.Series:
        sub = h240.loc[h240["mae_so_far_r"] <= thresh, ["trade_id", "bar_offset"]]
        return sub.groupby("trade_id", sort=False)["bar_offset"].min()

    out["first_1R"] = _first_above(1.0)
    out["first_2R"] = _first_above(2.0)
    out["first_1_5R"] = _first_above(1.5)
    out["first_neg_1_5R"] = _first_below(-1.5)

    # final_r = close_r at the last held bar.
    m = meta[["trade_id", "bars_held"]].set_index("trade_id")
    p = paths.merge(m, left_on="trade_id", right_index=True, how="left", validate="many_to_one")
    exit_rows = p[p["bar_offset"] == p["bars_held"]].set_index("trade_id")
    final = exit_rows["close_r"].astype("float64")
    # Fallback for trades without exact bars_held match: last available bar.
    missing = set(out.index) - set(final.index)
    if missing:
        tail = (
            paths[paths["trade_id"].isin(missing)]
            .groupby("trade_id", sort=False).tail(1).set_index("trade_id")["close_r"]
            .astype("float64")
        )
        final = pd.concat([final, tail], axis=0)
    out["final_r"] = final.reindex(out.index)

    return out.reset_index()


def _mass_in_band(s: pd.Series) -> dict[str, float]:
    s = s.dropna()
    n = len(s)
    if n == 0:
        return {f"band_{k}": float("nan") for k in (
            "0_to_0_5R", "0_5_to_1R", "1_to_2R", "2_to_5R", "above_5R"
        )}
    return {
        "band_0_to_0_5R":  float(((s >= 0) & (s < 0.5)).sum() / n),
        "band_0_5_to_1R":  float(((s >= 0.5) & (s < 1.0)).sum() / n),
        "band_1_to_2R":    float(((s >= 1.0) & (s < 2.0)).sum() / n),
        "band_2_to_5R":    float(((s >= 2.0) & (s < 5.0)).sum() / n),
        "band_above_5R":   float((s >= 5.0).sum() / n),
    }


def _shape_tag_for_subset(mfe: pd.Series) -> str:
    sub = pd.DataFrame({"max_mfe_h240": mfe})
    return shape_tag(sub)["tag"]


def characterise(
    paths: pd.DataFrame,
    meta: pd.DataFrame,
    features: pd.DataFrame,
    assignments: pd.DataFrame,
    centroids: pd.DataFrame,
    k: int,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Per-archetype summary CSV + per-archetype side distributions.

    Returns:
      summary       one row per archetype
      side_dists    {archetype_id: DataFrame of fwd_mfe_h240/final_r percentiles}
    """
    geom = _per_trade_geometry(paths, meta)
    # Merge per-trade features + assignments + geometry.
    df = features.merge(assignments, on="trade_id", how="left", validate="one_to_one")
    df = df.merge(geom, on="trade_id", how="left", validate="one_to_one")

    n_total = len(df)
    rows = []
    side_dists: dict[int, pd.DataFrame] = {}
    for arch in range(k):
        sub = df[df["archetype_id"] == arch]
        size = len(sub)
        size_frac = size / n_total if n_total else 0.0
        cen = centroids[centroids["archetype_id"] == arch].iloc[0]

        mfe = sub["max_mfe_h240"]
        fin = sub["final_r"]

        # Race-style wrong way: -1.5R touched before +1.5R within 240. NaN -> CAP.
        first_p = sub["first_1_5R"].fillna(CAP).clip(upper=CAP)
        first_n = sub["first_neg_1_5R"].fillna(CAP).clip(upper=CAP)
        wrong_active = (first_p < CAP) | (first_n < CAP)
        wrong = ((first_n < first_p) & wrong_active).mean()

        # Peak-and-collapse: max_mfe_h240 >= 1 AND final_r < 0
        peak_collapse = ((mfe >= 1.0) & (fin < 0)).mean()

        row = {
            "archetype_id": arch,
            "size_count":   int(size),
            "size_fraction_of_pool": float(size_frac),
            "monotonicity_centroid":       float(cen["monotonicity_centroid"]),
            "local_peaks_centroid":        float(cen["local_peaks_centroid"]),
            "pullback_magnitude_centroid": float(cen["pullback_magnitude_centroid"]),
            "time_to_peak_relative_centroid": float(cen["time_to_peak_relative_centroid"]),
            "fwd_mfe_h240_p5":  float(mfe.quantile(0.05)),
            "fwd_mfe_h240_p25": float(mfe.quantile(0.25)),
            "fwd_mfe_h240_p50": float(mfe.quantile(0.50)),
            "fwd_mfe_h240_p75": float(mfe.quantile(0.75)),
            "fwd_mfe_h240_p95": float(mfe.quantile(0.95)),
            "final_r_p5":  float(fin.quantile(0.05)),
            "final_r_p25": float(fin.quantile(0.25)),
            "final_r_p50": float(fin.quantile(0.50)),
            "final_r_p75": float(fin.quantile(0.75)),
            "final_r_p95": float(fin.quantile(0.95)),
            "final_r_mean":   float(fin.mean()),
            "final_r_t_stat": _t_stat_vs_zero(fin),
            "frac_reach_1R":  float((mfe >= 1.0).mean()),
            "frac_reach_2R":  float((mfe >= 2.0).mean()),
            "frac_wrong_way": float(wrong),
            "pct_peak_and_collapse": float(peak_collapse),
            "shape_tag":  _shape_tag_for_subset(mfe),
        }
        row.update(_mass_in_band(mfe))
        rows.append(row)

        # Side per-archetype distribution.
        dist = pd.DataFrame({
            "percentile":     list(SHAPE_DIST_PERCENTILES),
            "fwd_mfe_h240_R": [float(mfe.quantile(p / 100)) for p in SHAPE_DIST_PERCENTILES],
            "final_r":        [float(fin.quantile(p / 100)) for p in SHAPE_DIST_PERCENTILES],
        })
        side_dists[arch] = dist

    summary = pd.DataFrame(rows)
    return summary, side_dists


def _t_stat_vs_zero(s: pd.Series) -> float:
    s = s.dropna().astype("float64")
    n = len(s)
    if n < 2:
        return float("nan")
    mean = s.mean()
    std = s.std(ddof=1)
    if std == 0.0:
        return float("nan")
    return float(mean / (std / np.sqrt(n)))
