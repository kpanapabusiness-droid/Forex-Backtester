"""Arc 3 — Step 3 capturability characterisation.

L_ARC_PROTOCOL v2.0 §7 + §2. Per-archetype forward-geometry summary and
six-floor capturability gate.

Archetype groups (from Step 2, K=7):
  Early-peak hold        clusters 0 + 3   n=1028   full evaluation
  Stepwise climber       clusters 2 + 4   n=707    full evaluation
  Unassigned cluster 1   cluster  1       n=493    full evaluation (P&C near-miss)
  Unassigned cluster 6   cluster  6       n=230    diagnostic-only (size<10%)
  Unassigned cluster 5   cluster  5       n=110    diagnostic-only (size<10%)

§2 floors (conjunctive):
  monotonicity_centroid >= 0.55
  local_peaks_centroid within §11 archetype ceiling
  fwd_mfe_h240_p50 >= 1.5R
  frac_reach_1R >= 0.70 AND frac_wrong_way <= 0.30
  shape_tag in {tight_unimodal, heavy_right_tail}
  size_fraction_of_pool >= 0.10

Determinism: pure data-processing, no RNG. Verified by sha256 spot check.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FEATURE_COLS: Tuple[str, ...] = (
    "monotonicity_ratio_in_profit",
    "local_peaks_count",
    "pullback_magnitude_median",
    "time_to_peak_mfe_relative",
)

# §11 local_peaks ceiling per archetype (per dispatch). Tuple (low, high, kind):
#   kind="range"     → low <= peaks <= high
#   kind="ceiling"   → peaks <= high
#   kind="floor"     → peaks >= low (Random walk; lower-bound, retained for completeness)
#   kind="any"       → no peaks gate
LOCAL_PEAKS_RULE: Dict[str, Tuple[Optional[float], Optional[float], str]] = {
    "Monotone ascent": (None, 4.0, "ceiling"),
    "Stepwise climber": (5.0, 30.0, "range"),
    "Early-peak hold": (None, None, "any"),
    "Peak-and-collapse": (None, None, "any"),
    "V-shape recovery": (None, None, "any"),
    "Random walk": (8.0, None, "floor"),
    # Cluster 1: dispatch says treat as Peak-and-collapse for the local_peaks
    # ceiling (its closest §11 match per Step 2 Open-07 audit).
    "unassigned (cluster 1 / Peak-and-collapse near-miss)": (None, None, "any"),
    # Diagnostic-only groups; ceiling not gated.
    "unassigned (cluster 5 / diagnostic)": (None, None, "any"),
    "unassigned (cluster 6 / diagnostic)": (None, None, "any"),
}

# Gate constants (locked per v2.0 §2).
GATE_MONOTONICITY: float = 0.55
GATE_MFE_P50: float = 1.5
GATE_REACH_1R: float = 0.70
GATE_WRONG_WAY: float = 0.30
GATE_SIZE_FRACTION: float = 0.10

# Mass-in-band edges on final_r.
FINAL_R_BANDS: Tuple[Tuple[float, float, str], ...] = (
    (-float("inf"), -1.0, "le_-1R"),
    (-1.0, -0.5, "-1R_to_-0.5R"),
    (-0.5, 0.0, "-0.5R_to_0R"),
    (0.0, 0.5, "0R_to_0.5R"),
    (0.5, 1.0, "0.5R_to_1R"),
    (1.0, 2.0, "1R_to_2R"),
    (2.0, 5.0, "2R_to_5R"),
    (5.0, float("inf"), "gt_5R"),
)


# ---------------------------------------------------------------------------
# Archetype group definitions (from Step 2 K=7 + dispatch).
# ---------------------------------------------------------------------------


@dataclass
class _ArchetypeGroup:
    label: str
    cluster_ids: List[int]
    treatment: str  # "full_evaluation" or "diagnostic_only"


ARCHETYPE_GROUPS: Tuple[_ArchetypeGroup, ...] = (
    _ArchetypeGroup("Early-peak hold", [0, 3], "full_evaluation"),
    _ArchetypeGroup("Stepwise climber", [2, 4], "full_evaluation"),
    _ArchetypeGroup(
        "unassigned (cluster 1 / Peak-and-collapse near-miss)", [1], "full_evaluation"
    ),
    _ArchetypeGroup("unassigned (cluster 6 / diagnostic)", [6], "diagnostic_only"),
    _ArchetypeGroup("unassigned (cluster 5 / diagnostic)", [5], "diagnostic_only"),
)


# ---------------------------------------------------------------------------
# Stats helpers.
# ---------------------------------------------------------------------------


def _pct(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _t_stat(arr: np.ndarray) -> float:
    if arr.size < 2:
        return float("nan")
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd == 0:
        return float("nan")
    return mu / (sd / np.sqrt(arr.size))


def _mass_in_band(arr: np.ndarray) -> Dict[str, float]:
    n = arr.size
    out = {}
    for lo, hi, label in FINAL_R_BANDS:
        mask = (arr >= lo) & (arr < hi) if hi != float("inf") else (arr >= lo)
        out[label] = float(mask.sum() / n) if n > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# Shape tag classification (v2.0 §7).
# ---------------------------------------------------------------------------


def _classify_shape_tag(final_r: np.ndarray) -> Tuple[str, Dict[str, Any]]:
    """Return (shape_tag, decision_log).

    Document-order check: no_magnitude → tight_unimodal → heavy_right_tail
    → bimodal → scattered → unclassified.
    """
    log: Dict[str, Any] = {}
    if final_r.size < 3:
        log["reason"] = "insufficient_sample"
        return "unclassified", log

    p25, p50, p75, p95 = (
        _pct(final_r, 25),
        _pct(final_r, 50),
        _pct(final_r, 75),
        _pct(final_r, 95),
    )
    iqr = p75 - p25
    skew = float(sps.skew(final_r, bias=False))
    std = float(final_r.std(ddof=1))
    log.update(
        {
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p95": p95,
            "iqr": iqr,
            "skew": skew,
            "std": std,
        }
    )

    # 1. no_magnitude: p75 <= 0.5R.
    if p75 <= 0.5:
        log["no_magnitude_p75_le_0.5R"] = True
        return "no_magnitude", log

    # 2. tight_unimodal: IQR <= 1.5R AND |skew| <= 1.0.
    if iqr <= 1.5 and abs(skew) <= 1.0:
        log["tight_unimodal_iqr_le_1.5_skew_abs_le_1.0"] = True
        return "tight_unimodal", log

    # 3. heavy_right_tail: p95/p75 >= 2.5 AND skew >= 1.0. Guard against
    # p75 ≤ 0 (would already have triggered no_magnitude). If p75 > 0:
    if p75 > 0:
        ratio_95_75 = p95 / p75
        log["p95_over_p75"] = float(ratio_95_75)
        if ratio_95_75 >= 2.5 and skew >= 1.0:
            log["heavy_right_tail_ratio_ge_2.5_skew_ge_1.0"] = True
            return "heavy_right_tail", log

    # 4. bimodal: Hartigan dip test p < 0.05, OR KDE valley >= 0.5R between
    # local maxima. scipy doesn't have Hartigan's dip directly; use KDE +
    # local-extrema valley test only (consistent with §7's "OR" clause).
    try:
        kde = sps.gaussian_kde(final_r)
        x_grid = np.linspace(final_r.min(), final_r.max(), 512)
        dens = kde(x_grid)
        # Find local maxima (peaks of the density).
        peaks: List[int] = []
        for i in range(1, len(dens) - 1):
            if dens[i] > dens[i - 1] and dens[i] > dens[i + 1]:
                peaks.append(i)
        log["kde_peak_count"] = len(peaks)
        if len(peaks) >= 2:
            # Sort peaks by density descending; take top two distinct ones.
            peaks_sorted = sorted(peaks, key=lambda i: -dens[i])[:2]
            i1, i2 = sorted(peaks_sorted)
            valley_dens = float(np.min(dens[i1:i2 + 1]))
            mode_dens_min = float(min(dens[i1], dens[i2]))
            # "Visible two-mode structure with valley >= 0.5R between modes"
            mode_separation = abs(float(x_grid[i2]) - float(x_grid[i1]))
            valley_depth_pct = (
                (mode_dens_min - valley_dens) / mode_dens_min if mode_dens_min > 0 else 0.0
            )
            log["mode_separation_R"] = mode_separation
            log["valley_depth_pct"] = float(valley_depth_pct)
            if mode_separation >= 0.5 and valley_depth_pct >= 0.10:
                # Valley at least 10% below the shorter mode — required to
                # distinguish a genuine bimodal pattern from a wide unimodal
                # one with a wobble. 0.5R separation alone isn't enough.
                log["bimodal_decision"] = "fire"
                return "bimodal", log
            else:
                log["bimodal_decision"] = "separation_or_valley_insufficient"
    except Exception as e:
        log["kde_error"] = str(e)

    # 5. scattered: std >= 2.5R.
    if std >= 2.5:
        log["scattered_std_ge_2.5"] = True
        return "scattered", log

    log["fall_through"] = True
    return "unclassified", log


# ---------------------------------------------------------------------------
# Per-archetype summary.
# ---------------------------------------------------------------------------


@dataclass
class _ArchetypeSummary:
    label: str
    cluster_ids: List[int]
    trade_ids: np.ndarray
    treatment: str

    # Identity.
    size_count: int = 0
    size_fraction_of_pool: float = 0.0
    centroid_monotonicity: float = 0.0
    centroid_local_peaks: float = 0.0
    centroid_pullback: float = 0.0
    centroid_time_to_peak_rel: float = 0.0

    # Forward geometry.
    final_r_mean: float = 0.0
    final_r_t_stat: float = 0.0
    final_r_p5: float = 0.0
    final_r_p25: float = 0.0
    final_r_p50: float = 0.0
    final_r_p75: float = 0.0
    final_r_p95: float = 0.0
    mfe_h240_p5: float = 0.0
    mfe_h240_p25: float = 0.0
    mfe_h240_p50: float = 0.0
    mfe_h240_p75: float = 0.0
    mfe_h240_p95: float = 0.0

    # Frequencies.
    frac_reach_1R: float = 0.0
    frac_reach_2R: float = 0.0
    frac_wrong_way: float = 0.0
    pct_peak_and_collapse: float = 0.0

    # Distribution shape.
    shape_tag: str = ""
    shape_tag_log: Dict[str, Any] = field(default_factory=dict)
    mass_in_band: Dict[str, float] = field(default_factory=dict)

    # §2 gate.
    gate_monotonicity: str = "FAIL"
    gate_local_peaks: str = "FAIL"
    gate_mfe_p50: str = "FAIL"
    gate_direction: str = "FAIL"
    gate_shape_tag: str = "FAIL"
    gate_size: str = "FAIL"
    capturability_verdict: str = "FAIL"
    gate_failure_reasons: List[str] = field(default_factory=list)


def _build_summary(
    group: _ArchetypeGroup,
    trades_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    features_df: pd.DataFrame,
    pool_size: int,
) -> _ArchetypeSummary:
    # Trade IDs in this group.
    trade_ids = clusters_df.loc[
        clusters_df["cluster_label"].isin(group.cluster_ids), "trade_id"
    ].to_numpy()
    sub_trades = trades_df[trades_df["trade_id"].isin(trade_ids)]
    sub_features = features_df[features_df["trade_id"].isin(trade_ids)]

    summ = _ArchetypeSummary(
        label=group.label,
        cluster_ids=list(group.cluster_ids),
        trade_ids=np.array(sorted(trade_ids.tolist()), dtype=int),
        treatment=group.treatment,
    )
    summ.size_count = int(len(sub_trades))
    summ.size_fraction_of_pool = float(summ.size_count / pool_size) if pool_size > 0 else 0.0

    # Aggregated centroid (re-computed mean over the group's trades).
    if not sub_features.empty:
        summ.centroid_monotonicity = float(sub_features["monotonicity_ratio_in_profit"].mean())
        summ.centroid_local_peaks = float(sub_features["local_peaks_count"].mean())
        summ.centroid_pullback = float(sub_features["pullback_magnitude_median"].mean())
        summ.centroid_time_to_peak_rel = float(sub_features["time_to_peak_mfe_relative"].mean())

    final_r = sub_trades["final_r"].to_numpy(dtype=float)
    mfe_r = sub_trades["mfe_r"].to_numpy(dtype=float)

    summ.final_r_mean = float(final_r.mean()) if final_r.size > 0 else 0.0
    summ.final_r_t_stat = _t_stat(final_r)
    summ.final_r_p5 = _pct(final_r, 5)
    summ.final_r_p25 = _pct(final_r, 25)
    summ.final_r_p50 = _pct(final_r, 50)
    summ.final_r_p75 = _pct(final_r, 75)
    summ.final_r_p95 = _pct(final_r, 95)
    summ.mfe_h240_p5 = _pct(mfe_r, 5)
    summ.mfe_h240_p25 = _pct(mfe_r, 25)
    summ.mfe_h240_p50 = _pct(mfe_r, 50)
    summ.mfe_h240_p75 = _pct(mfe_r, 75)
    summ.mfe_h240_p95 = _pct(mfe_r, 95)

    summ.frac_reach_1R = float((mfe_r >= 1.0).sum() / mfe_r.size) if mfe_r.size > 0 else 0.0
    summ.frac_reach_2R = float((mfe_r >= 2.0).sum() / mfe_r.size) if mfe_r.size > 0 else 0.0
    summ.frac_wrong_way = float((final_r <= -0.5).sum() / final_r.size) if final_r.size > 0 else 0.0

    pc_mask = (mfe_r >= 1.0) & (final_r <= 0.4 * mfe_r)
    summ.pct_peak_and_collapse = float(pc_mask.sum() / final_r.size) if final_r.size > 0 else 0.0

    summ.shape_tag, summ.shape_tag_log = _classify_shape_tag(final_r)
    summ.mass_in_band = _mass_in_band(final_r)

    # §2 gate evaluation.
    reasons: List[str] = []
    summ.gate_monotonicity = (
        "PASS" if summ.centroid_monotonicity >= GATE_MONOTONICITY else "FAIL"
    )
    if summ.gate_monotonicity == "FAIL":
        reasons.append(
            f"monotonicity_centroid {summ.centroid_monotonicity:.3f} < {GATE_MONOTONICITY}"
        )

    lp_lo, lp_hi, kind = LOCAL_PEAKS_RULE[group.label]
    peaks_pass = True
    if kind == "ceiling":
        peaks_pass = summ.centroid_local_peaks <= (lp_hi if lp_hi is not None else float("inf"))
    elif kind == "range":
        peaks_pass = (lp_lo or 0.0) <= summ.centroid_local_peaks <= (
            lp_hi if lp_hi is not None else float("inf")
        )
    elif kind == "floor":
        peaks_pass = summ.centroid_local_peaks >= (lp_lo if lp_lo is not None else 0.0)
    elif kind == "any":
        peaks_pass = True
    summ.gate_local_peaks = "PASS" if peaks_pass else "FAIL"
    if not peaks_pass:
        rule_text = {
            "ceiling": f"<= {lp_hi}",
            "range": f"in [{lp_lo}, {lp_hi}]",
            "floor": f">= {lp_lo}",
        }.get(kind, "(unknown)")
        reasons.append(
            f"local_peaks_centroid {summ.centroid_local_peaks:.2f} not {rule_text} (§11 {group.label})"
        )

    summ.gate_mfe_p50 = "PASS" if summ.mfe_h240_p50 >= GATE_MFE_P50 else "FAIL"
    if summ.gate_mfe_p50 == "FAIL":
        reasons.append(f"mfe_h240_p50 {summ.mfe_h240_p50:.3f}R < {GATE_MFE_P50}R")

    dir_pass = (summ.frac_reach_1R >= GATE_REACH_1R) and (summ.frac_wrong_way <= GATE_WRONG_WAY)
    summ.gate_direction = "PASS" if dir_pass else "FAIL"
    if not dir_pass:
        if summ.frac_reach_1R < GATE_REACH_1R:
            reasons.append(f"frac_reach_1R {summ.frac_reach_1R:.3f} < {GATE_REACH_1R}")
        if summ.frac_wrong_way > GATE_WRONG_WAY:
            reasons.append(f"frac_wrong_way {summ.frac_wrong_way:.3f} > {GATE_WRONG_WAY}")

    summ.gate_shape_tag = (
        "PASS" if summ.shape_tag in {"tight_unimodal", "heavy_right_tail"} else "FAIL"
    )
    if summ.gate_shape_tag == "FAIL":
        reasons.append(f"shape_tag={summ.shape_tag} not in {{tight_unimodal, heavy_right_tail}}")

    summ.gate_size = "PASS" if summ.size_fraction_of_pool >= GATE_SIZE_FRACTION else "FAIL"
    if summ.gate_size == "FAIL":
        reasons.append(
            f"size_fraction {summ.size_fraction_of_pool:.3f} < {GATE_SIZE_FRACTION}"
        )

    all_pass = all(
        g == "PASS"
        for g in (
            summ.gate_monotonicity,
            summ.gate_local_peaks,
            summ.gate_mfe_p50,
            summ.gate_direction,
            summ.gate_shape_tag,
            summ.gate_size,
        )
    )
    summ.capturability_verdict = "PASS" if all_pass else "FAIL"
    summ.gate_failure_reasons = reasons
    return summ


# ---------------------------------------------------------------------------
# Output writers.
# ---------------------------------------------------------------------------


def _safe_label_slug(label: str) -> str:
    return (
        label.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )


def _write_archetype_summaries(
    summaries: List[_ArchetypeSummary], path: Path, float_fmt: str
) -> None:
    rows = []
    for s in summaries:
        rows.append(
            {
                "archetype_label": s.label,
                "cluster_ids": "+".join(str(c) for c in s.cluster_ids),
                "size_count": s.size_count,
                "size_fraction_of_pool": s.size_fraction_of_pool,
                "centroid_monotonicity": s.centroid_monotonicity,
                "centroid_local_peaks": s.centroid_local_peaks,
                "centroid_pullback": s.centroid_pullback,
                "centroid_time_to_peak_rel": s.centroid_time_to_peak_rel,
                "final_r_mean": s.final_r_mean,
                "final_r_t_stat": s.final_r_t_stat,
                "final_r_p5": s.final_r_p5,
                "final_r_p25": s.final_r_p25,
                "final_r_p50": s.final_r_p50,
                "final_r_p75": s.final_r_p75,
                "final_r_p95": s.final_r_p95,
                "mfe_h240_p5": s.mfe_h240_p5,
                "mfe_h240_p25": s.mfe_h240_p25,
                "mfe_h240_p50": s.mfe_h240_p50,
                "mfe_h240_p75": s.mfe_h240_p75,
                "mfe_h240_p95": s.mfe_h240_p95,
                "frac_reach_1R": s.frac_reach_1R,
                "frac_reach_2R": s.frac_reach_2R,
                "frac_wrong_way": s.frac_wrong_way,
                "pct_peak_and_collapse": s.pct_peak_and_collapse,
                "shape_tag": s.shape_tag,
                "gate_monotonicity": s.gate_monotonicity,
                "gate_local_peaks": s.gate_local_peaks,
                "gate_mfe_p50": s.gate_mfe_p50,
                "gate_direction": s.gate_direction,
                "gate_shape_tag": s.gate_shape_tag,
                "gate_size": s.gate_size,
                "capturability_verdict": s.capturability_verdict,
                "treatment": s.treatment,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, float_format=float_fmt, na_rep="", lineterminator="\n")


def _write_per_archetype_distribution(
    summary: _ArchetypeSummary,
    trades_df: pd.DataFrame,
    out_dir: Path,
    float_fmt: str,
) -> Path:
    sub = trades_df[trades_df["trade_id"].isin(summary.trade_ids)][
        [
            "trade_id",
            "pair",
            "final_r",
            "mfe_r",
            "mae_r",
            "bars_held",
            "exit_reason",
            "time_to_peak_mfe",
        ]
    ].sort_values("trade_id")
    path = out_dir / f"{_safe_label_slug(summary.label)}_distribution.csv"
    sub.to_csv(path, index=False, float_format=float_fmt, na_rep="", lineterminator="\n")
    return path


def _write_pass_list(summaries: List[_ArchetypeSummary], path: Path) -> None:
    rows = [
        {"archetype_label": s.label}
        for s in summaries
        if s.capturability_verdict == "PASS" and s.treatment == "full_evaluation"
    ]
    pd.DataFrame(rows, columns=["archetype_label"]).to_csv(
        path, index=False, lineterminator="\n"
    )


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------


def run(
    trades_csv: Path,
    paths_csv: Path,
    clusters_csv: Path,
    features_csv: Path,
    out_dir: Path,
    *,
    float_fmt: str = "%.10g",
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_df = pd.read_csv(trades_csv)
    clusters_df = pd.read_csv(clusters_csv)
    features_df = pd.read_csv(features_csv)

    pool_size = int(len(trades_df))

    summaries: List[_ArchetypeSummary] = []
    per_archetype_paths: Dict[str, str] = {}
    for group in ARCHETYPE_GROUPS:
        summ = _build_summary(group, trades_df, clusters_df, features_df, pool_size)
        summaries.append(summ)
        dist_path = _write_per_archetype_distribution(summ, trades_df, out_dir, float_fmt)
        per_archetype_paths[summ.label] = str(dist_path.relative_to(_REPO_ROOT))

    _write_archetype_summaries(summaries, out_dir / "archetype_summaries.csv", float_fmt)
    _write_pass_list(summaries, out_dir / "capturability_pass_list.csv")

    # Diagnostics — full per-archetype breakdown.
    full_eval_passing = [
        s.label
        for s in summaries
        if s.treatment == "full_evaluation" and s.capturability_verdict == "PASS"
    ]
    step3_pass = len(full_eval_passing) >= 1

    diag: Dict[str, Any] = {
        "input_trades_csv": str(trades_csv.relative_to(_REPO_ROOT)),
        "input_paths_csv": str(paths_csv.relative_to(_REPO_ROOT)),
        "input_clusters_csv": str(clusters_csv.relative_to(_REPO_ROOT)),
        "input_features_csv": str(features_csv.relative_to(_REPO_ROOT)),
        "pool_size": pool_size,
        "archetypes": [
            {
                "label": s.label,
                "cluster_ids": s.cluster_ids,
                "treatment": s.treatment,
                "size_count": s.size_count,
                "size_fraction_of_pool": s.size_fraction_of_pool,
                "centroid": {
                    "monotonicity": s.centroid_monotonicity,
                    "local_peaks": s.centroid_local_peaks,
                    "pullback": s.centroid_pullback,
                    "time_to_peak_rel": s.centroid_time_to_peak_rel,
                },
                "final_r": {
                    "mean": s.final_r_mean,
                    "t_stat": s.final_r_t_stat,
                    "p5": s.final_r_p5,
                    "p25": s.final_r_p25,
                    "p50": s.final_r_p50,
                    "p75": s.final_r_p75,
                    "p95": s.final_r_p95,
                },
                "mfe_h240": {
                    "p5": s.mfe_h240_p5,
                    "p25": s.mfe_h240_p25,
                    "p50": s.mfe_h240_p50,
                    "p75": s.mfe_h240_p75,
                    "p95": s.mfe_h240_p95,
                },
                "frequencies": {
                    "frac_reach_1R": s.frac_reach_1R,
                    "frac_reach_2R": s.frac_reach_2R,
                    "frac_wrong_way": s.frac_wrong_way,
                    "pct_peak_and_collapse": s.pct_peak_and_collapse,
                },
                "mass_in_band": s.mass_in_band,
                "shape_tag": s.shape_tag,
                "shape_tag_log": s.shape_tag_log,
                "gates": {
                    "monotonicity": s.gate_monotonicity,
                    "local_peaks": s.gate_local_peaks,
                    "mfe_p50": s.gate_mfe_p50,
                    "direction": s.gate_direction,
                    "shape_tag": s.gate_shape_tag,
                    "size": s.gate_size,
                    "verdict": s.capturability_verdict,
                    "failure_reasons": s.gate_failure_reasons,
                },
                "distribution_csv": per_archetype_paths[s.label],
                "out_of_scope_for_step_4": s.treatment == "diagnostic_only",
            }
            for s in summaries
        ],
        "step3_verdict": "PASS" if step3_pass else "FAIL",
        "full_evaluation_passing_archetypes": full_eval_passing,
        "open_07_notes": _open_07_notes(summaries),
        "sha256": {
            "archetype_summaries_csv": _sha256_file(out_dir / "archetype_summaries.csv"),
            "capturability_pass_list_csv": _sha256_file(out_dir / "capturability_pass_list.csv"),
        },
        "env": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "run_timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "step3_diagnostics.json").write_text(
        json.dumps(diag, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return diag


def _open_07_notes(summaries: List[_ArchetypeSummary]) -> List[Dict[str, Any]]:
    """Open-07 cross-arc note: flag the unassigned cluster 1 group's §2
    outcome — if it passes, it's evidence the §11 P&C threshold may need
    relaxation; if it fails, record why.
    """
    notes: List[Dict[str, Any]] = []
    for s in summaries:
        if "cluster 1" in s.label:
            notes.append(
                {
                    "subject": s.label,
                    "context": "Step 2 Open-07 audit flagged cluster 1 as Peak-and-collapse near-miss (time_to_peak_rel 0.385 > 0.30 ceiling by 0.085; pct_peak_and_collapse 0.663 already passes the >=0.50 leg).",
                    "verdict": s.capturability_verdict,
                    "implication": (
                        "If PASS, this is evidence the §11 Peak-and-collapse time_to_peak_rel <= 0.30 ceiling is over-strict — a candidate for cross-arc revision."
                        if s.capturability_verdict == "PASS"
                        else "FAIL: failure modes below show whether §11 threshold relaxation alone would have helped or whether forward geometry blocks it independently."
                    ),
                    "failure_reasons": s.gate_failure_reasons,
                }
            )
    return notes


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arc 3 Step 3 capturability characterisation (v2.0 §7).")
    p.add_argument(
        "--trades-csv",
        type=Path,
        default=_REPO_ROOT / "results/l_arc_3/step1_plumbing/trades_all.csv",
    )
    p.add_argument(
        "--paths-csv",
        type=Path,
        default=_REPO_ROOT / "results/l_arc_3/step1_plumbing/trades_paths.csv",
    )
    p.add_argument(
        "--clusters-csv",
        type=Path,
        default=_REPO_ROOT / "results/l_arc_3/step2_clustering/clusters_K7.csv",
    )
    p.add_argument(
        "--features-csv",
        type=Path,
        default=_REPO_ROOT / "results/l_arc_3/step2_clustering/path_features.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "results/l_arc_3/step3_capturability",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    diag = run(
        args.trades_csv,
        args.paths_csv,
        args.clusters_csv,
        args.features_csv,
        args.out_dir,
    )
    print(
        f"[arc_3 step 3] verdict={diag['step3_verdict']} "
        f"passing={diag['full_evaluation_passing_archetypes']}"
    )
    return 0 if diag["step3_verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
