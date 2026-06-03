"""Arc 3D — Step 3 capturability with dynamic archetype grouping.

Variant of scripts/arc_3/step3_capturability.py for the Arc 3D diagnostic
tail. Two structural differences:

  1. Archetype groups are derived dynamically from step2's
     archetype_assignments.csv (cluster_label -> archetype_label) rather
     than hardcoded to Arc 3's K=7 cluster IDs. This lets the script run
     on B/C cells where K and cluster labels differ.

  2. Two evaluation modes:
       --mode aggregated      §6 same-archetype aggregation rule (matches
                              Arc 3 step3 baseline behaviour). Unassigned
                              clusters remain separate (one row per
                              unassigned cluster).
       --mode un_aggregated   each cluster is its own group; no §6
                              aggregation.

§2 floor numbers and §11 local_peaks ceiling rules are UNCHANGED. Per-
criterion PASS/FAIL is exposed for every group (not just the conjunctive
verdict).

Spread treatment, archetype labelling logic, and §2 floor constants are
read-only imports from the Arc 3 implementation — there are no protocol
changes here, only different grouping.

Usage:
    py scripts/arc_3d/step3_3d.py \\
        --trades-csv  <abs path to step1/trades_all.csv> \\
        --paths-csv   <abs path to step1/trades_paths.csv> \\
        --clusters-csv <abs path to step2/clusters_K<chosen>.csv> \\
        --features-csv <abs path to step2/path_features.csv> \\
        --assignments-csv <abs path to step2/archetype_assignments.csv> \\
        --mode {aggregated, un_aggregated} \\
        --out-dir <abs path to step3/<mode>/>
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

# §11 local_peaks ceiling per archetype (per L_ARC_PROTOCOL v2.0 §11).
# Tuple (low, high, kind):
#   kind="range"     -> low <= peaks <= high
#   kind="ceiling"   -> peaks <= high
#   kind="floor"     -> peaks >= low
#   kind="any"       -> no peaks gate
LOCAL_PEAKS_RULE: Dict[str, Tuple[Optional[float], Optional[float], str]] = {
    "Monotone ascent": (None, 4.0, "ceiling"),
    "Stepwise climber": (5.0, 30.0, "range"),
    "Early-peak hold": (None, None, "any"),
    "Peak-and-collapse": (None, None, "any"),
    "V-shape recovery": (None, None, "any"),
    "Random walk": (8.0, None, "floor"),
    "unassigned": (None, None, "any"),
}

# Gate constants (locked per v2.0 §2 — NEVER modified by Arc 3D).
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
# Group definitions (built dynamically).
# ---------------------------------------------------------------------------


@dataclass
class _Group:
    """A group is either one cluster (un_aggregated mode) or several clusters
    sharing the same §11 archetype label (aggregated mode)."""

    label: str                   # display label (e.g., "Stepwise climber" or "cluster_2 (Stepwise climber)")
    archetype_label: str         # canonical §11 archetype name (drives LOCAL_PEAKS_RULE)
    cluster_ids: List[int]


def _build_groups(assignments: pd.DataFrame, mode: str) -> List[_Group]:
    """Build the list of groups according to the requested mode.

    `assignments` is step2's archetype_assignments.csv with columns:
        cluster_label, archetype_label, ... (plus centroid + boundary_reason)
    Rows where archetype_label starts with "unassigned" are kept as one
    group per cluster (their centroids are structurally distinct).
    """
    assignments = assignments.sort_values("cluster_label").reset_index(drop=True)
    groups: List[_Group] = []

    if mode == "un_aggregated":
        for _, row in assignments.iterrows():
            cl = int(row["cluster_label"])
            arc_raw = str(row["archetype_label"]).strip()
            # Normalise unassigned label to bare "unassigned" for the
            # LOCAL_PEAKS_RULE lookup; preserve display detail in the label.
            if arc_raw.startswith("unassigned"):
                canonical = "unassigned"
                disp = f"cluster_{cl} (unassigned)"
            else:
                canonical = arc_raw
                disp = f"cluster_{cl} ({arc_raw})"
            groups.append(_Group(label=disp, archetype_label=canonical, cluster_ids=[cl]))
        return groups

    # aggregated mode
    # Bucket clusters by canonical archetype. Unassigned clusters each get
    # their own bucket (no §6 aggregation for unassigned).
    arc_buckets: Dict[str, List[int]] = {}
    unassigned_buckets: List[Tuple[int, str]] = []
    for _, row in assignments.iterrows():
        cl = int(row["cluster_label"])
        arc_raw = str(row["archetype_label"]).strip()
        if arc_raw.startswith("unassigned"):
            unassigned_buckets.append((cl, arc_raw))
        else:
            arc_buckets.setdefault(arc_raw, []).append(cl)

    # Stable order: §11 row order from LOCAL_PEAKS_RULE keys, with any
    # other arcs (shouldn't happen) appended at the end alphabetically.
    rule_order = [
        a for a in LOCAL_PEAKS_RULE.keys() if a not in {"unassigned"}
    ]
    seen = set()
    for arc in rule_order:
        if arc in arc_buckets:
            cluster_ids = sorted(arc_buckets[arc])
            groups.append(
                _Group(label=arc, archetype_label=arc, cluster_ids=cluster_ids)
            )
            seen.add(arc)
    for arc in sorted(arc_buckets.keys()):
        if arc not in seen:
            cluster_ids = sorted(arc_buckets[arc])
            groups.append(
                _Group(label=arc, archetype_label=arc, cluster_ids=cluster_ids)
            )

    # Append unassigned clusters in cluster_label order.
    for cl, _arc_raw in sorted(unassigned_buckets):
        groups.append(
            _Group(label=f"unassigned (cluster {cl})", archetype_label="unassigned", cluster_ids=[cl])
        )
    return groups


# ---------------------------------------------------------------------------
# Stats helpers (mirror scripts/arc_3/step3_capturability.py — read-only
# library use; no behavioural change).
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
    out: Dict[str, float] = {}
    for lo, hi, label in FINAL_R_BANDS:
        mask = (arr >= lo) & (arr < hi) if hi != float("inf") else (arr >= lo)
        out[label] = float(mask.sum() / n) if n > 0 else 0.0
    return out


def _classify_shape_tag(final_r: np.ndarray) -> Tuple[str, Dict[str, Any]]:
    """Return (shape_tag, decision_log). Mirrors Arc 3 step3."""
    log: Dict[str, Any] = {}
    if final_r.size < 3:
        log["reason"] = "insufficient_sample"
        return "unclassified", log

    p25 = _pct(final_r, 25)
    p50 = _pct(final_r, 50)
    p75 = _pct(final_r, 75)
    p95 = _pct(final_r, 95)
    iqr = p75 - p25
    skew = float(sps.skew(final_r, bias=False))
    std = float(final_r.std(ddof=1))
    log.update({"p25": p25, "p50": p50, "p75": p75, "p95": p95,
                "iqr": iqr, "skew": skew, "std": std})

    if p75 <= 0.5:
        log["no_magnitude_p75_le_0.5R"] = True
        return "no_magnitude", log

    if iqr <= 1.5 and abs(skew) <= 1.0:
        log["tight_unimodal_iqr_le_1.5_skew_abs_le_1.0"] = True
        return "tight_unimodal", log

    if p75 > 0:
        ratio_95_75 = p95 / p75
        log["p95_over_p75"] = float(ratio_95_75)
        if ratio_95_75 >= 2.5 and skew >= 1.0:
            log["heavy_right_tail_ratio_ge_2.5_skew_ge_1.0"] = True
            return "heavy_right_tail", log

    try:
        kde = sps.gaussian_kde(final_r)
        x_grid = np.linspace(final_r.min(), final_r.max(), 512)
        dens = kde(x_grid)
        peaks: List[int] = []
        for i in range(1, len(dens) - 1):
            if dens[i] > dens[i - 1] and dens[i] > dens[i + 1]:
                peaks.append(i)
        log["kde_peak_count"] = len(peaks)
        if len(peaks) >= 2:
            peaks_sorted = sorted(peaks, key=lambda i: -dens[i])[:2]
            i1, i2 = sorted(peaks_sorted)
            valley_dens = float(np.min(dens[i1:i2 + 1]))
            mode_dens_min = float(min(dens[i1], dens[i2]))
            mode_separation = abs(float(x_grid[i2]) - float(x_grid[i1]))
            valley_depth_pct = (
                (mode_dens_min - valley_dens) / mode_dens_min if mode_dens_min > 0 else 0.0
            )
            log["mode_separation_R"] = mode_separation
            log["valley_depth_pct"] = float(valley_depth_pct)
            if mode_separation >= 0.5 and valley_depth_pct >= 0.10:
                log["bimodal_decision"] = "fire"
                return "bimodal", log
            else:
                log["bimodal_decision"] = "separation_or_valley_insufficient"
    except Exception as e:
        log["kde_error"] = str(e)

    if std >= 2.5:
        log["scattered_std_ge_2.5"] = True
        return "scattered", log

    log["fall_through"] = True
    return "unclassified", log


# ---------------------------------------------------------------------------
# Per-group summary.
# ---------------------------------------------------------------------------


@dataclass
class _GroupSummary:
    label: str
    archetype_label: str
    cluster_ids: List[int]

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

    # §2 per-criterion verdicts.
    gate_monotonicity: str = "FAIL"
    gate_local_peaks: str = "FAIL"
    gate_mfe_p50: str = "FAIL"
    gate_reach_1R: str = "FAIL"
    gate_wrong_way: str = "FAIL"
    gate_shape_tag: str = "FAIL"
    gate_size: str = "FAIL"
    capturability_verdict: str = "FAIL"
    gate_failure_reasons: List[str] = field(default_factory=list)
    gates_passed_count: int = 0
    gates_total: int = 6  # mono, local_peaks, mfe_p50, direction (reach_1R AND wrong_way), shape_tag, size


def _build_summary(
    group: _Group,
    trades_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    features_df: pd.DataFrame,
    pool_size: int,
) -> _GroupSummary:
    trade_ids = clusters_df.loc[
        clusters_df["cluster_label"].isin(group.cluster_ids), "trade_id"
    ].to_numpy()
    sub_trades = trades_df[trades_df["trade_id"].isin(trade_ids)]
    sub_features = features_df[features_df["trade_id"].isin(trade_ids)]

    summ = _GroupSummary(
        label=group.label,
        archetype_label=group.archetype_label,
        cluster_ids=list(group.cluster_ids),
    )
    summ.size_count = int(len(sub_trades))
    summ.size_fraction_of_pool = (
        float(summ.size_count / pool_size) if pool_size > 0 else 0.0
    )

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
    summ.frac_wrong_way = (
        float((final_r <= -0.5).sum() / final_r.size) if final_r.size > 0 else 0.0
    )

    pc_mask = (mfe_r >= 1.0) & (final_r <= 0.4 * mfe_r)
    summ.pct_peak_and_collapse = (
        float(pc_mask.sum() / final_r.size) if final_r.size > 0 else 0.0
    )

    summ.shape_tag, summ.shape_tag_log = _classify_shape_tag(final_r)
    summ.mass_in_band = _mass_in_band(final_r)

    # §2 per-criterion evaluation.
    reasons: List[str] = []

    summ.gate_monotonicity = (
        "PASS" if summ.centroid_monotonicity >= GATE_MONOTONICITY else "FAIL"
    )
    if summ.gate_monotonicity == "FAIL":
        reasons.append(
            f"monotonicity_centroid {summ.centroid_monotonicity:.3f} < {GATE_MONOTONICITY}"
        )

    lp_rule = LOCAL_PEAKS_RULE.get(
        group.archetype_label, (None, None, "any")
    )
    lp_lo, lp_hi, kind = lp_rule
    peaks = summ.centroid_local_peaks
    if kind == "ceiling":
        peaks_pass = peaks <= (lp_hi if lp_hi is not None else float("inf"))
        rule_text = f"<= {lp_hi}"
    elif kind == "range":
        peaks_pass = (lp_lo or 0.0) <= peaks <= (
            lp_hi if lp_hi is not None else float("inf")
        )
        rule_text = f"in [{lp_lo}, {lp_hi}]"
    elif kind == "floor":
        peaks_pass = peaks >= (lp_lo if lp_lo is not None else 0.0)
        rule_text = f">= {lp_lo}"
    else:  # "any"
        peaks_pass = True
        rule_text = "any"
    summ.gate_local_peaks = "PASS" if peaks_pass else "FAIL"
    if not peaks_pass:
        reasons.append(
            f"local_peaks_centroid {peaks:.2f} not {rule_text} (§11 {group.archetype_label})"
        )

    summ.gate_mfe_p50 = "PASS" if summ.mfe_h240_p50 >= GATE_MFE_P50 else "FAIL"
    if summ.gate_mfe_p50 == "FAIL":
        reasons.append(f"mfe_h240_p50 {summ.mfe_h240_p50:.3f}R < {GATE_MFE_P50}R")

    summ.gate_reach_1R = "PASS" if summ.frac_reach_1R >= GATE_REACH_1R else "FAIL"
    if summ.gate_reach_1R == "FAIL":
        reasons.append(f"frac_reach_1R {summ.frac_reach_1R:.3f} < {GATE_REACH_1R}")

    summ.gate_wrong_way = "PASS" if summ.frac_wrong_way <= GATE_WRONG_WAY else "FAIL"
    if summ.gate_wrong_way == "FAIL":
        reasons.append(f"frac_wrong_way {summ.frac_wrong_way:.3f} > {GATE_WRONG_WAY}")

    summ.gate_shape_tag = (
        "PASS" if summ.shape_tag in {"tight_unimodal", "heavy_right_tail"} else "FAIL"
    )
    if summ.gate_shape_tag == "FAIL":
        reasons.append(
            f"shape_tag={summ.shape_tag} not in {{tight_unimodal, heavy_right_tail}}"
        )

    summ.gate_size = (
        "PASS" if summ.size_fraction_of_pool >= GATE_SIZE_FRACTION else "FAIL"
    )
    if summ.gate_size == "FAIL":
        reasons.append(
            f"size_fraction {summ.size_fraction_of_pool:.3f} < {GATE_SIZE_FRACTION}"
        )

    # Direction is a conjunction of reach_1R AND wrong_way for the
    # six-of-six count (matching Arc 3 step3 baseline's accounting).
    dir_pass = (summ.gate_reach_1R == "PASS") and (summ.gate_wrong_way == "PASS")
    summ.capturability_verdict = "PASS" if (
        summ.gate_monotonicity == "PASS"
        and summ.gate_local_peaks == "PASS"
        and summ.gate_mfe_p50 == "PASS"
        and dir_pass
        and summ.gate_shape_tag == "PASS"
        and summ.gate_size == "PASS"
    ) else "FAIL"
    summ.gates_passed_count = sum(
        1 for v in (
            summ.gate_monotonicity,
            summ.gate_local_peaks,
            summ.gate_mfe_p50,
            "PASS" if dir_pass else "FAIL",
            summ.gate_shape_tag,
            summ.gate_size,
        ) if v == "PASS"
    )
    summ.gate_failure_reasons = reasons
    return summ


# ---------------------------------------------------------------------------
# IO.
# ---------------------------------------------------------------------------


def _safe_slug(label: str) -> str:
    return (
        label.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )


def _write_archetype_summaries(
    summaries: List[_GroupSummary], path: Path, float_fmt: str
) -> None:
    rows = []
    for s in summaries:
        rows.append(
            {
                "group_label": s.label,
                "archetype_label": s.archetype_label,
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
                "gate_reach_1R": s.gate_reach_1R,
                "gate_wrong_way": s.gate_wrong_way,
                "gate_shape_tag": s.gate_shape_tag,
                "gate_size": s.gate_size,
                "gates_passed_count": s.gates_passed_count,
                "capturability_verdict": s.capturability_verdict,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, float_format=float_fmt, na_rep="", lineterminator="\n")


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
    assignments_csv: Path,
    out_dir: Path,
    mode: str,
    *,
    float_fmt: str = "%.10g",
) -> Dict[str, Any]:
    assert mode in {"aggregated", "un_aggregated"}, mode
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_df = pd.read_csv(trades_csv)
    clusters_df = pd.read_csv(clusters_csv)
    features_df = pd.read_csv(features_csv)
    assignments_df = pd.read_csv(assignments_csv)

    pool_size = int(len(trades_df))

    groups = _build_groups(assignments_df, mode)
    summaries: List[_GroupSummary] = []
    for g in groups:
        summ = _build_summary(g, trades_df, clusters_df, features_df, pool_size)
        summaries.append(summ)

    summaries_path = out_dir / "archetype_summaries.csv"
    _write_archetype_summaries(summaries, summaries_path, float_fmt)

    # Diagnostics file.
    diag: Dict[str, Any] = {
        "mode": mode,
        "input_trades_csv": str(trades_csv),
        "input_paths_csv": str(paths_csv),
        "input_clusters_csv": str(clusters_csv),
        "input_features_csv": str(features_csv),
        "input_assignments_csv": str(assignments_csv),
        "pool_size": pool_size,
        "n_groups": len(summaries),
        "groups": [
            {
                "label": s.label,
                "archetype_label": s.archetype_label,
                "cluster_ids": s.cluster_ids,
                "size_count": s.size_count,
                "size_fraction": s.size_fraction_of_pool,
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
                    "reach_1R": s.gate_reach_1R,
                    "wrong_way": s.gate_wrong_way,
                    "shape_tag": s.gate_shape_tag,
                    "size": s.gate_size,
                    "gates_passed_count": s.gates_passed_count,
                    "verdict": s.capturability_verdict,
                    "failure_reasons": s.gate_failure_reasons,
                },
            }
            for s in summaries
        ],
        "sha256": {
            "archetype_summaries_csv": _sha256_file(summaries_path),
        },
        "env": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "run_timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "step3_3d_diagnostics.json").write_text(
        json.dumps(diag, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return diag


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Arc 3D Step 3 capturability with dynamic archetype grouping."
    )
    p.add_argument("--trades-csv", type=Path, required=True)
    p.add_argument("--paths-csv", type=Path, required=True)
    p.add_argument("--clusters-csv", type=Path, required=True)
    p.add_argument("--features-csv", type=Path, required=True)
    p.add_argument("--assignments-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mode", choices=["aggregated", "un_aggregated"], required=True)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    diag = run(
        args.trades_csv,
        args.paths_csv,
        args.clusters_csv,
        args.features_csv,
        args.assignments_csv,
        args.out_dir,
        args.mode,
    )
    n_pass = sum(
        1 for g in diag["groups"] if g["gates"]["verdict"] == "PASS"
    )
    n_5of6 = sum(
        1 for g in diag["groups"] if g["gates"]["gates_passed_count"] >= 5
    )
    print(
        f"[arc_3d step 3] mode={args.mode} groups={diag['n_groups']} "
        f"verdict_pass={n_pass} ge_5of6={n_5of6}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
