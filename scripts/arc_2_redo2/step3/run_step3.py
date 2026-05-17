"""Arc 2 redo2 — Step 3 capturability characterisation driver (v2.1.1).

Per cluster (4 from Step 2): run SL sweep over {0.5, 1.0, 1.5, 2.0, 3.0, 4.0}
× ATR(14), compute §2 floors at each candidate, run shape_tag with
bimodal_separated test, select SL maximising capturability composite (among
those passing all §2 floors), route per §11.

Outputs go to results/l_arc_2_redo2/step3/.

Usage:
  py scripts/arc_2_redo2/step3/run_step3.py -c configs/arc_2_redo2/step3.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.arc_2_redo2.step3.sl_sweep import (  # noqa: E402
    aggregate_per_cluster,
    sweep_all_trades,
)
from scripts.arc_2_redo2.step3.shape_tag_v2_1_1 import (  # noqa: E402
    ShapeTagDecision,
    detect_shape_tag_v2_1_1,
)


# ============================================================
# §2 floor evaluation per (cluster, SL)
# ============================================================


@dataclass
class FloorEval:
    name: str
    threshold: str
    value: str
    passed: bool


@dataclass
class ClusterSLRow:
    cluster_id: int
    X: float
    n: int
    size_fraction: float
    mono_pp_centroid: float
    frac_reach_1R: float
    frac_reach_2R: float
    frac_wrong_way_pp: float
    fwd_mfe_p50: float
    fwd_mfe_p50_atr: float
    final_r_mean: float
    final_r_t_stat: float
    mass_gt_5R: float
    composite: float
    shape_tag: str
    shape_tag_detail: str
    dip_stat: Optional[float]
    dip_p: Optional[float]
    mode_loc_1: Optional[float]
    mode_loc_2: Optional[float]
    mode_mass_1: Optional[float]
    mode_mass_2: Optional[float]
    mode_separation: Optional[float]
    bimodal_sep_admits: bool
    local_peaks_centroid: float
    floors: List[FloorEval] = field(default_factory=list)
    floors_passed_count: int = 0
    floors_failed_list: List[str] = field(default_factory=list)
    overall_pass: bool = False


def _local_peaks_floor_passes(
    centroid: float, archetype_label: str, ceiling_by_arch: dict
) -> Tuple[bool, str]:
    """For Step 2 'Stepwise climber' label, ceiling = 30. For 'Monotone ascent', 4.
    Others (unclassified, etc.): no ceiling — automatic pass."""
    # Use the base archetype label (Step 2 assignment) for ceiling lookup.
    # Step 2 may label as "Stepwise climber" or "Early-peak hold OR ..." etc.
    base = archetype_label.split(" OR ")[0].split(" (")[0].strip()
    ceiling = ceiling_by_arch.get(base, None)
    if ceiling is None:
        return True, "no ceiling for archetype '" + base + "'"
    if centroid <= ceiling:
        return True, f"{centroid:.2f} <= {ceiling}"
    return False, f"{centroid:.2f} > {ceiling}"


def evaluate_floors(
    row: dict,
    centroid_local_peaks: float,
    archetype_label: str,
    shape_tag: str,
    cfg: dict,
) -> Tuple[List[FloorEval], int, List[str], bool]:
    cap = cfg["capturability"]
    mono_min = float(cap["mono_pre_peak_min"])
    p50_min = float(cap["fwd_mfe_p50_min_r"])
    r1_min = float(cap["frac_reach_1r_min"])
    fw_max = float(cap["frac_wrong_way_pre_peak_max"])
    admit_set = set(cap["shape_tag_admit"])
    size_min = float(cap["size_fraction_min"])
    ceiling_by_arch = cap["local_peaks_ceiling_by_archetype"]

    floors: List[FloorEval] = []
    floors.append(
        FloorEval(
            "mono_pre_peak",
            f">= {mono_min}",
            f"{row['mono_pp_centroid']:.4f}",
            bool(row["mono_pp_centroid"] >= mono_min),
        )
    )
    floors.append(
        FloorEval(
            "frac_reach_1R",
            f">= {r1_min}",
            f"{row['frac_reach_1R']:.4f}",
            bool(row["frac_reach_1R"] >= r1_min),
        )
    )
    floors.append(
        FloorEval(
            "frac_wrong_way_pp",
            f"<= {fw_max}",
            f"{row['frac_wrong_way_pp']:.4f}",
            bool(row["frac_wrong_way_pp"] <= fw_max),
        )
    )
    floors.append(
        FloorEval(
            "fwd_mfe_p50",
            f">= {p50_min}",
            f"{row['fwd_mfe_p50']:.4f}",
            bool(row["fwd_mfe_p50"] >= p50_min),
        )
    )
    floors.append(
        FloorEval(
            "shape_tag",
            f"in {sorted(admit_set)}",
            shape_tag,
            shape_tag in admit_set,
        )
    )
    lp_passed, lp_msg = _local_peaks_floor_passes(
        centroid_local_peaks, archetype_label, ceiling_by_arch
    )
    floors.append(
        FloorEval(
            "local_peaks_ceiling",
            f"per §11 ({archetype_label})",
            lp_msg,
            lp_passed,
        )
    )
    floors.append(
        FloorEval(
            "size_fraction",
            f">= {size_min}",
            f"{row['size_fraction']:.4f}",
            bool(row["size_fraction"] >= size_min),
        )
    )

    passed = sum(1 for f in floors if f.passed)
    failed = [f.name for f in floors if not f.passed]
    overall = (passed == len(floors))
    return floors, passed, failed, overall


# ============================================================
# Composite selection with v2.1.1 tiebreakers
# ============================================================


def select_sl_for_cluster(
    cluster_rows: List[ClusterSLRow], tiebreaker_tol: float
) -> Optional[ClusterSLRow]:
    """Among cluster_rows with overall_pass==True, pick max composite.
    Tiebreakers: within `tiebreaker_tol` composite → larger peak_mfe in ATR;
    further tie → smaller SL.
    """
    passing = [r for r in cluster_rows if r.overall_pass]
    if not passing:
        return None
    max_c = max(r.composite for r in passing)
    candidates = [r for r in passing if (max_c - r.composite) <= tiebreaker_tol]
    # Tiebreaker 1: larger fwd_mfe_p50_atr
    candidates.sort(key=lambda r: (-r.fwd_mfe_p50_atr, r.X))
    return candidates[0]


# ============================================================
# pct_peak_and_collapse (full-window, prior arc_2_redo definition)
# ============================================================


def compute_pct_peak_and_collapse(
    paths_df: pd.DataFrame, trades_df: pd.DataFrame, peak_floor_r: float, collapse_ratio: float
) -> pd.DataFrame:
    """For each trade, compute:
      fwd_mfe_h240 = max(close_r) over full path (baseline R)
      close_r_at_240 = close_r at bar_offset=240 (baseline R)
      peaked = fwd_mfe_h240 >= peak_floor_r
      collapsed = peaked AND (close_r_at_240 <= collapse_ratio * fwd_mfe_h240)
    Returns DataFrame with trade_id, fwd_mfe_h240, close_r_at_240, peaked, collapsed.
    """
    paths = paths_df[["trade_id", "bar_offset", "close_r"]].copy()
    agg = paths.groupby("trade_id")["close_r"].max().rename("fwd_mfe_h240").reset_index()
    at_240 = (
        paths[paths["bar_offset"] == 240]
        .set_index("trade_id")["close_r"]
        .rename("close_r_at_240")
        .reset_index()
    )
    out = agg.merge(at_240, on="trade_id", how="left")
    out["peaked"] = (out["fwd_mfe_h240"] >= peak_floor_r).astype(int)
    out["collapsed"] = (
        (out["peaked"] == 1) & (out["close_r_at_240"] <= collapse_ratio * out["fwd_mfe_h240"])
    ).astype(int)
    return out


# ============================================================
# §11 disambiguation
# ============================================================


def disambiguate_early_vs_collapse(pct_pc: float, cfg: dict) -> str:
    d = cfg["disambiguation"]
    if pct_pc < float(d["early_peak_hold_max"]):
        return "Early-peak hold"
    if pct_pc >= float(d["peak_and_collapse_min"]):
        return "Peak-and-collapse"
    return "empirical-defer"


# ============================================================
# CSV writers
# ============================================================


def _fmt_g(x) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
        if not math.isfinite(xf):
            return ""
    except Exception:
        return ""
    return f"{xf:.10g}"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_per_trade_sl_csv(out_path: Path, per_trade_df: pd.DataFrame) -> None:
    cols = [
        "trade_id", "X", "truncated_at_bar", "sl_triggered",
        "peak_mfe_bar_X", "peak_mfe_X", "peak_mfe_atr",
        "final_r_X", "reached_1R_X", "wrong_way_pre_peak_X", "mono_pre_peak_X",
    ]
    df = per_trade_df[cols].sort_values(["trade_id", "X"], kind="mergesort").reset_index(drop=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for _, r in df.iterrows():
            w.writerow([
                int(r["trade_id"]),
                _fmt_g(r["X"]),
                int(r["truncated_at_bar"]),
                int(bool(r["sl_triggered"])),
                int(r["peak_mfe_bar_X"]),
                _fmt_g(r["peak_mfe_X"]),
                _fmt_g(r["peak_mfe_atr"]),
                _fmt_g(r["final_r_X"]),
                int(bool(r["reached_1R_X"])),
                int(bool(r["wrong_way_pre_peak_X"])),
                _fmt_g(r["mono_pre_peak_X"]),
            ])


def write_cluster_sl_sweep_csv(out_path: Path, sweep_rows: List[ClusterSLRow]) -> None:
    cols = [
        "cluster_id", "X", "n", "size_fraction",
        "mono_pp_centroid", "frac_reach_1R", "frac_wrong_way_pp",
        "fwd_mfe_p50", "fwd_mfe_p50_atr", "frac_reach_2R", "mass_gt_5R",
        "final_r_mean", "final_r_t_stat",
        "shape_tag", "shape_tag_detail",
        "dip_stat", "dip_p", "mode_loc_1", "mode_loc_2", "mode_mass_1", "mode_mass_2",
        "mode_separation", "bimodal_sep_admits",
        "composite", "floors_passed_count", "floors_failed", "overall_pass",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in sweep_rows:
            w.writerow([
                int(r.cluster_id), _fmt_g(r.X), int(r.n), _fmt_g(r.size_fraction),
                _fmt_g(r.mono_pp_centroid), _fmt_g(r.frac_reach_1R), _fmt_g(r.frac_wrong_way_pp),
                _fmt_g(r.fwd_mfe_p50), _fmt_g(r.fwd_mfe_p50_atr),
                _fmt_g(r.frac_reach_2R), _fmt_g(r.mass_gt_5R),
                _fmt_g(r.final_r_mean), _fmt_g(r.final_r_t_stat),
                r.shape_tag, r.shape_tag_detail.replace(",", ";"),
                _fmt_g(r.dip_stat), _fmt_g(r.dip_p),
                _fmt_g(r.mode_loc_1), _fmt_g(r.mode_loc_2),
                _fmt_g(r.mode_mass_1), _fmt_g(r.mode_mass_2),
                _fmt_g(r.mode_separation), int(r.bimodal_sep_admits),
                _fmt_g(r.composite), int(r.floors_passed_count),
                ";".join(r.floors_failed_list), int(r.overall_pass),
            ])


def write_per_cluster_sl_detail_csv(out_path: Path, rows: List[ClusterSLRow], cluster_id: int) -> None:
    cluster_rows = [r for r in rows if r.cluster_id == cluster_id]
    cluster_rows.sort(key=lambda r: r.X)
    cols = [
        "X", "n", "mono_pp_centroid", "frac_reach_1R", "frac_wrong_way_pp",
        "fwd_mfe_p50_R", "fwd_mfe_p50_atr",
        "shape_tag", "bimodal_sep_admits",
        "dip_stat", "dip_p", "mode_loc_1", "mode_loc_2",
        "mode_mass_1", "mode_mass_2", "mode_separation",
        "composite", "floors_failed", "overall_pass",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in cluster_rows:
            w.writerow([
                _fmt_g(r.X), int(r.n), _fmt_g(r.mono_pp_centroid),
                _fmt_g(r.frac_reach_1R), _fmt_g(r.frac_wrong_way_pp),
                _fmt_g(r.fwd_mfe_p50), _fmt_g(r.fwd_mfe_p50_atr),
                r.shape_tag, int(r.bimodal_sep_admits),
                _fmt_g(r.dip_stat), _fmt_g(r.dip_p),
                _fmt_g(r.mode_loc_1), _fmt_g(r.mode_loc_2),
                _fmt_g(r.mode_mass_1), _fmt_g(r.mode_mass_2),
                _fmt_g(r.mode_separation),
                _fmt_g(r.composite),
                ";".join(r.floors_failed_list), int(r.overall_pass),
            ])


def write_per_cluster_distribution_csv(
    out_path: Path, per_trade_df: pd.DataFrame, cluster_id: int,
    clusters_df: pd.DataFrame, selected_X: Optional[float],
) -> None:
    """Distribution at the selected SL (or X=2.0 baseline if none selected)."""
    X = selected_X if selected_X is not None else 2.0
    sub = per_trade_df.merge(clusters_df, on="trade_id")
    sub = sub[(sub["cluster_id"] == cluster_id) & (sub["X"] == X)]
    cols = [
        "trade_id", "selected_SL", "final_r_X", "peak_mfe_X", "peak_mfe_atr",
        "reached_1R_X", "wrong_way_pre_peak_X", "mono_pre_peak_X",
        "sl_triggered", "truncated_at_bar", "peak_mfe_bar_X",
    ]
    sub = sub.sort_values("trade_id", kind="mergesort").reset_index(drop=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for _, r in sub.iterrows():
            w.writerow([
                int(r["trade_id"]), _fmt_g(X), _fmt_g(r["final_r_X"]),
                _fmt_g(r["peak_mfe_X"]), _fmt_g(r["peak_mfe_atr"]),
                int(bool(r["reached_1R_X"])), int(bool(r["wrong_way_pre_peak_X"])),
                _fmt_g(r["mono_pre_peak_X"]),
                int(bool(r["sl_triggered"])), int(r["truncated_at_bar"]),
                int(r["peak_mfe_bar_X"]),
            ])


def write_archetype_summaries_csv(
    out_path: Path,
    selected_per_cluster: Dict[int, Optional[ClusterSLRow]],
    cluster_meta: Dict[int, dict],
    pct_pc_per_cluster: Dict[int, float],
) -> None:
    cols = [
        "cluster_id", "archetype_label_step2", "archetype_label_final",
        "selected_SL", "n", "size_fraction",
        "mono_pre_peak_centroid", "local_peaks_centroid",
        "frac_reach_1R", "frac_reach_2R", "frac_wrong_way_pp",
        "fwd_mfe_p50_R", "fwd_mfe_p50_atr", "mass_gt_5R",
        "final_r_mean", "final_r_t_stat",
        "shape_tag", "bimodal_sep_admits", "composite",
        "pct_peak_and_collapse_full_window", "capturability_pass",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for cid in sorted(selected_per_cluster.keys()):
            sel = selected_per_cluster[cid]
            meta = cluster_meta[cid]
            pct_pc = pct_pc_per_cluster.get(cid, float("nan"))
            if sel is None:
                w.writerow([
                    int(cid), meta.get("label_step2", ""), meta.get("label_final", ""),
                    "", "", _fmt_g(meta.get("size_fraction")),
                    _fmt_g(meta.get("mono_centroid")), _fmt_g(meta.get("local_peaks_centroid")),
                    "", "", "", "", "", "", "", "",
                    "", "", "",
                    _fmt_g(pct_pc),
                    "FAIL",
                ])
            else:
                w.writerow([
                    int(cid), meta.get("label_step2", ""), meta.get("label_final", ""),
                    _fmt_g(sel.X), int(sel.n), _fmt_g(sel.size_fraction),
                    _fmt_g(sel.mono_pp_centroid), _fmt_g(sel.local_peaks_centroid),
                    _fmt_g(sel.frac_reach_1R), _fmt_g(sel.frac_reach_2R),
                    _fmt_g(sel.frac_wrong_way_pp),
                    _fmt_g(sel.fwd_mfe_p50), _fmt_g(sel.fwd_mfe_p50_atr),
                    _fmt_g(sel.mass_gt_5R),
                    _fmt_g(sel.final_r_mean), _fmt_g(sel.final_r_t_stat),
                    sel.shape_tag, int(sel.bimodal_sep_admits), _fmt_g(sel.composite),
                    _fmt_g(pct_pc),
                    "PASS",
                ])


def write_capturability_pass_list_csv(
    out_path: Path,
    selected_per_cluster: Dict[int, Optional[ClusterSLRow]],
    cluster_meta: Dict[int, dict],
) -> None:
    cols = ["cluster_id", "archetype_label_final", "selected_SL_X_atr", "composite", "verdict"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for cid in sorted(selected_per_cluster.keys()):
            sel = selected_per_cluster[cid]
            meta = cluster_meta[cid]
            if sel is None:
                w.writerow([int(cid), meta.get("label_final", ""), "", "", "FAIL"])
            else:
                w.writerow([
                    int(cid), meta.get("label_final", ""),
                    _fmt_g(sel.X), _fmt_g(sel.composite), "PASS",
                ])


def write_cluster_routing_csv(
    out_path: Path,
    selected_per_cluster: Dict[int, Optional[ClusterSLRow]],
    cluster_meta: Dict[int, dict],
) -> None:
    cols = [
        "cluster_id", "archetype_label_final", "base_archetype_match",
        "bimodal_sep_admits", "routing",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for cid in sorted(selected_per_cluster.keys()):
            sel = selected_per_cluster[cid]
            meta = cluster_meta[cid]
            label = meta.get("label_final", "")
            base_match = label not in ("unclassified", "empirical-defer", "")
            if sel is None:
                routing = "DIES (no SL passed §2)"
            else:
                if sel.bimodal_sep_admits and base_match:
                    routing = f"DUAL: base ({label}) + row 7 split-exit advance to Step 4; ship best at Step 6"
                elif sel.bimodal_sep_admits and not base_match:
                    routing = "row 7 split-exit only"
                elif base_match:
                    routing = f"base archetype ({label}) only"
                else:
                    routing = "PASS §2 but no §11 base; defer to chat"
            w.writerow([
                int(cid), label, int(bool(base_match)),
                int(bool(sel.bimodal_sep_admits)) if sel else 0, routing,
            ])


def write_shape_tag_diagnostics_csv(
    out_path: Path, sweep_rows: List[ClusterSLRow]
) -> None:
    cols = [
        "cluster_id", "X", "shape_tag", "bimodal_sep_admits",
        "dip_stat", "dip_p",
        "mode_loc_1", "mode_loc_2", "mode_mass_1", "mode_mass_2", "mode_separation",
        "detail",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in sorted(sweep_rows, key=lambda x: (x.cluster_id, x.X)):
            w.writerow([
                int(r.cluster_id), _fmt_g(r.X), r.shape_tag, int(r.bimodal_sep_admits),
                _fmt_g(r.dip_stat), _fmt_g(r.dip_p),
                _fmt_g(r.mode_loc_1), _fmt_g(r.mode_loc_2),
                _fmt_g(r.mode_mass_1), _fmt_g(r.mode_mass_2),
                _fmt_g(r.mode_separation),
                r.shape_tag_detail.replace(",", ";"),
            ])


def write_pct_pc_definition(out_path: Path, cfg: dict) -> None:
    pc = cfg["peak_and_collapse"]
    txt = (
        "pct_peak_and_collapse — Arc 2 redo2 Step 3 (v2.1.1) definition\n"
        "================================================================\n\n"
        "Source: inherited verbatim from prior arc_2_redo step3 config.\n"
        "Reference: scripts/arc_2_redo/step3_capturability.py:1140-1144.\n\n"
        "Per-trade boolean:\n"
        f"  fwd_mfe_h240 = max(close_r) over full path (baseline R units)\n"
        f"  close_r_at_240 = close_r at bar_offset = 240 (baseline R units)\n"
        f"  peaked     = (fwd_mfe_h240 >= {pc['peak_floor_r']:.2f} R)\n"
        f"  collapsed  = peaked AND (close_r_at_240 <= {pc['collapse_ratio']:.2f} * fwd_mfe_h240)\n\n"
        "Cluster aggregate: mean(collapsed) across trades.\n\n"
        "v2.1.1 note: the fork's is_held=0 forward bars now carry REAL-MARKET\n"
        "R-fields (not frozen at final_r as in the prior arc_2_redo). The\n"
        "conceptual definition is unchanged; the values are now genuine\n"
        "post-exit price observation over the full 240-bar window rather\n"
        "than the frozen-at-final-R substitute.\n\n"
        "§11 disambiguation thresholds (cfg disambiguation block):\n"
        f"  pct_pc <  {cfg['disambiguation']['early_peak_hold_max']} → Early-peak hold\n"
        f"  pct_pc >= {cfg['disambiguation']['peak_and_collapse_min']} → Peak-and-collapse\n"
        "  in between → empirical-defer (flag for chat, do NOT auto-resolve)\n"
    )
    out_path.write_text(txt, encoding="utf-8")


def write_step3_summary_md(
    out_path: Path,
    cfg: dict,
    sweep_rows: List[ClusterSLRow],
    selected_per_cluster: Dict[int, Optional[ClusterSLRow]],
    cluster_meta: Dict[int, dict],
    pct_pc_per_cluster: Dict[int, float],
    cid2_disambiguation: str,
    determinism_gate: str,
    csv_hashes_run1: Dict[str, str],
    csv_hashes_run2: Optional[Dict[str, str]],
) -> None:
    lines: List[str] = []
    lines.append("# Arc 2 redo2 — Step 3 capturability characterisation summary")
    lines.append("")
    lines.append("Protocol: `L_ARC_PROTOCOL.md` v2.1.1 §§2, 7, 11")
    lines.append("")
    n_pass = sum(1 for s in selected_per_cluster.values() if s is not None)
    disp = "PASS" if n_pass > 0 else "FAIL (arc dies — 0 clusters pass §2)"
    lines.append(f"**Step 3 disposition: {disp}**")
    lines.append(f"Clusters surviving §2 capturability gate: **{n_pass} of {len(selected_per_cluster)}**.")
    lines.append("")

    # Per-cluster verdict table at selected SL
    lines.append("## Per-cluster verdict at selected SL")
    lines.append("")
    lines.append(
        "| cid | label_final | sel_SL | mono_pp | reach_1R | fw_pp | fwd_mfe_p50_R | "
        "shape_tag | bimodal_sep | size_frac | composite | verdict |"
    )
    lines.append("|---:|---|---|---|---|---|---|---|:---:|---|---|:---:|")
    for cid in sorted(selected_per_cluster.keys()):
        sel = selected_per_cluster[cid]
        meta = cluster_meta[cid]
        label = meta.get("label_final", "")
        if sel is None:
            # Find the best partial-pass row for diagnostic context
            cluster_rows = [r for r in sweep_rows if r.cluster_id == cid]
            if cluster_rows:
                best = max(
                    cluster_rows,
                    key=lambda r: (r.floors_passed_count, r.composite),
                )
                lines.append(
                    f"| {cid} | {label} | (none) | "
                    f"{best.mono_pp_centroid:.4f} | {best.frac_reach_1R:.4f} | "
                    f"{best.frac_wrong_way_pp:.4f} | {best.fwd_mfe_p50:.4f} | "
                    f"{best.shape_tag} | {'YES' if best.bimodal_sep_admits else 'NO'} | "
                    f"{best.size_fraction:.4f} | {best.composite:.4f} | **FAIL** |"
                )
            else:
                lines.append(
                    f"| {cid} | {label} | (none) | – | – | – | – | – | NO | – | – | **FAIL** |"
                )
        else:
            lines.append(
                f"| {cid} | {label} | {sel.X:g}×ATR | "
                f"{sel.mono_pp_centroid:.4f} | {sel.frac_reach_1R:.4f} | "
                f"{sel.frac_wrong_way_pp:.4f} | {sel.fwd_mfe_p50:.4f} | "
                f"{sel.shape_tag} | {'YES' if sel.bimodal_sep_admits else 'NO'} | "
                f"{sel.size_fraction:.4f} | {sel.composite:.4f} | **PASS** |"
            )
    lines.append("")

    # Per-cluster floor breakdown at best-of-failing or selected
    lines.append("## Per-cluster §2 floor breakdown")
    lines.append("")
    for cid in sorted(selected_per_cluster.keys()):
        sel = selected_per_cluster[cid]
        cluster_rows = [r for r in sweep_rows if r.cluster_id == cid]
        if sel is not None:
            target = sel
            target_label = f"selected SL = {sel.X:g}×ATR"
        else:
            target = max(
                cluster_rows,
                key=lambda r: (r.floors_passed_count, r.composite),
            ) if cluster_rows else None
            target_label = (
                f"best partial-pass SL = {target.X:g}×ATR ({target.floors_passed_count}/7 floors)"
                if target
                else "no sweep data"
            )
        lines.append(f"### cid {cid} — {cluster_meta[cid].get('label_final', '')} — {target_label}")
        lines.append("")
        if target is None:
            lines.append("(no sweep data)")
            lines.append("")
            continue
        lines.append("| Floor | Threshold | Value | Result |")
        lines.append("|---|---|---|:---:|")
        for fl in target.floors:
            lines.append(
                f"| {fl.name} | {fl.threshold} | {fl.value} | {'PASS' if fl.passed else 'FAIL'} |"
            )
        lines.append("")

    # SL sweep table per cluster (composite focus)
    lines.append("## SL sweep — composite + floors-failed per cluster")
    lines.append("")
    for cid in sorted(selected_per_cluster.keys()):
        lines.append(f"### cid {cid}")
        lines.append("")
        cluster_rows = sorted([r for r in sweep_rows if r.cluster_id == cid], key=lambda r: r.X)
        lines.append("| X (×ATR) | mono_pp | reach_1R | fw_pp | fwd_mfe_p50_R | shape_tag | composite | floors_failed | overall |")
        lines.append("|---|---|---|---|---|---|---|---|:---:|")
        for r in cluster_rows:
            lines.append(
                f"| {r.X:g} | {r.mono_pp_centroid:.4f} | {r.frac_reach_1R:.4f} | "
                f"{r.frac_wrong_way_pp:.4f} | {r.fwd_mfe_p50:.4f} | {r.shape_tag} | "
                f"{r.composite:.4f} | {','.join(r.floors_failed_list) or '(none)'} | "
                f"{'PASS' if r.overall_pass else 'FAIL'} |"
            )
        lines.append("")

    # cid 2 disambiguation
    lines.append("## cid 2 — §11 disambiguation (Early-peak hold vs Peak-and-collapse)")
    lines.append("")
    pct = pct_pc_per_cluster.get(2, float("nan"))
    lines.append(f"- `pct_peak_and_collapse` (full-window, prior definition): **{pct:.4f}**")
    lines.append(f"- Disambiguation branch: **{cid2_disambiguation}**")
    lines.append(
        f"- Thresholds: < {cfg['disambiguation']['early_peak_hold_max']} → Early-peak hold; "
        f"≥ {cfg['disambiguation']['peak_and_collapse_min']} → Peak-and-collapse; "
        "in between → empirical-defer"
    )
    lines.append("")

    # cid 1 informational pct_pc
    if 1 in pct_pc_per_cluster:
        lines.append("## cid 1 — Stepwise climber pct_pc (informational; not gated)")
        lines.append("")
        lines.append(f"- `pct_peak_and_collapse` (full-window): {pct_pc_per_cluster[1]:.4f}")
        lines.append("- Used as §11 row 1 exit-policy information; not a §2 gate input.")
        lines.append("")

    # Determinism
    lines.append("## Determinism")
    lines.append("")
    lines.append(f"**Gate: {determinism_gate}**")
    lines.append("")
    lines.append("Run-1 CSV hashes:")
    for name, h in sorted(csv_hashes_run1.items()):
        lines.append(f"- `{name}`: `{h}`")
    if csv_hashes_run2 is not None:
        lines.append("")
        lines.append("Run-2 CSV hashes:")
        for name, h in sorted(csv_hashes_run2.items()):
            match = "MATCH" if csv_hashes_run1.get(name) == h else "MISMATCH"
            lines.append(f"- `{name}`: `{h}` ({match})")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Driver
# ============================================================


def _run_once(cfg: dict) -> Tuple[Dict[str, str], dict]:
    in_cfg = cfg["input"]
    out_cfg = cfg["output"]

    step1_dir = (_REPO_ROOT / in_cfg["step1_dir"]).resolve()
    step2_dir = (_REPO_ROOT / in_cfg["step2_dir"]).resolve()
    out_dir = (_REPO_ROOT / out_cfg["results_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[step3] loading trades_all.csv from {step1_dir}", file=sys.stderr)
    trades = pd.read_csv(step1_dir / in_cfg["trades_csv"])
    print(f"[step3] loading clusters_K4.csv + centroids_K4.csv from {step2_dir}", file=sys.stderr)
    clusters = pd.read_csv(step2_dir / in_cfg["clusters_csv"])
    centroids = pd.read_csv(step2_dir / in_cfg["centroids_csv"])
    arche = pd.read_csv(step2_dir / in_cfg["archetypes_csv"])

    print(f"[step3] loading trades_paths.csv (large) from {step1_dir}", file=sys.stderr)
    paths = pd.read_csv(
        step1_dir / in_cfg["paths_csv"],
        usecols=["trade_id", "bar_offset", "low", "close", "close_r", "is_held"],
    )
    print(f"[step3] paths rows: {len(paths):,}", file=sys.stderr)

    candidates = list(map(float, cfg["sl_sweep"]["candidates_atr"]))
    print(f"[step3] SL sweep candidates: {candidates}", file=sys.stderr)

    print("[step3] running per-trade SL sweep", file=sys.stderr)
    per_trade_df = sweep_all_trades(paths, trades, candidates)
    print(f"[step3] per-trade rows: {len(per_trade_df):,}", file=sys.stderr)

    # Pool size for size_fraction.
    pool_size = int(len(trades))

    # Cluster-level aggregates per SL
    cluster_agg = aggregate_per_cluster(per_trade_df, clusters)
    cluster_agg["size_fraction"] = cluster_agg["n"] / pool_size

    # Cluster metadata.
    cluster_meta: Dict[int, dict] = {}
    for _, ar in arche.iterrows():
        cid = int(ar["cluster_id"])
        cent_row = centroids[centroids["cluster_id"] == cid].iloc[0]
        cluster_meta[cid] = {
            "label_step2": str(ar["archetype_label"]),
            "label_final": str(ar["archetype_label"]),
            "mono_centroid": float(cent_row["monotonicity_ratio_in_profit"]),
            "local_peaks_centroid": float(cent_row["local_peaks_count"]),
            "pullback_centroid": float(cent_row["pullback_magnitude_median"]),
            "ttp_centroid": float(cent_row["time_to_peak_mfe_relative"]),
            "size_fraction": float(cent_row["fraction"]),
        }

    # pct_peak_and_collapse per cluster (full-window).
    print("[step3] computing pct_peak_and_collapse (full-window)", file=sys.stderr)
    pc_cfg = cfg["peak_and_collapse"]
    pc_per_trade = compute_pct_peak_and_collapse(
        paths, trades, float(pc_cfg["peak_floor_r"]), float(pc_cfg["collapse_ratio"])
    )
    pc_per_trade = pc_per_trade.merge(clusters, on="trade_id", how="left")
    pct_pc_per_cluster: Dict[int, float] = {}
    for cid, sub in pc_per_trade.groupby("cluster_id"):
        pct_pc_per_cluster[int(cid)] = float(sub["collapsed"].mean())

    # cid 2 disambiguation
    pct_pc_cid2 = pct_pc_per_cluster.get(2, float("nan"))
    cid2_branch = disambiguate_early_vs_collapse(pct_pc_cid2, cfg)
    # Update cid 2 label_final
    if 2 in cluster_meta:
        cluster_meta[2]["label_final"] = cid2_branch

    # Build per-(cluster, SL) ClusterSLRow with shape_tag + floors + composite
    print("[step3] shape_tag detection per (cluster, SL)", file=sys.stderr)
    sweep_rows: List[ClusterSLRow] = []
    for _, row in cluster_agg.iterrows():
        cid = int(row["cluster_id"])
        X = float(row["X"])
        # peak_mfe_X per-trade for shape_tag detection
        cluster_sl_trades = per_trade_df[per_trade_df["X"] == X].merge(clusters, on="trade_id")
        peak_arr = cluster_sl_trades[cluster_sl_trades["cluster_id"] == cid]["peak_mfe_X"].to_numpy(dtype=float)
        st = detect_shape_tag_v2_1_1(peak_arr, cfg)

        mode_loc_1 = st.mode_locs[0] if len(st.mode_locs) >= 1 else None
        mode_loc_2 = st.mode_locs[1] if len(st.mode_locs) == 2 else None
        mode_mass_1 = st.mode_masses[0] if len(st.mode_masses) >= 1 else None
        mode_mass_2 = st.mode_masses[1] if len(st.mode_masses) == 2 else None

        clr = ClusterSLRow(
            cluster_id=cid, X=X, n=int(row["n"]),
            size_fraction=float(row["size_fraction"]),
            mono_pp_centroid=float(row["mono_pre_peak_centroid"]),
            frac_reach_1R=float(row["frac_reach_1R"]),
            frac_reach_2R=float(row["frac_reach_2R"]),
            frac_wrong_way_pp=float(row["frac_wrong_way_pre_peak"]),
            fwd_mfe_p50=float(row["fwd_mfe_p50"]),
            fwd_mfe_p50_atr=float(row["fwd_mfe_p50_atr"]),
            final_r_mean=float(row["final_r_mean"]),
            final_r_t_stat=float(row["final_r_t_stat"]),
            mass_gt_5R=float(row["mass_gt_5R"]),
            composite=float(row["composite"]),
            shape_tag=st.tag,
            shape_tag_detail=st.detail,
            dip_stat=st.dip_stat, dip_p=st.dip_p,
            mode_loc_1=mode_loc_1, mode_loc_2=mode_loc_2,
            mode_mass_1=mode_mass_1, mode_mass_2=mode_mass_2,
            mode_separation=st.mode_separation,
            bimodal_sep_admits=st.bimodal_sep_admits,
            local_peaks_centroid=cluster_meta[cid]["local_peaks_centroid"],
        )
        floors, passed_cnt, failed_list, overall = evaluate_floors(
            {
                "mono_pp_centroid": clr.mono_pp_centroid,
                "frac_reach_1R": clr.frac_reach_1R,
                "frac_wrong_way_pp": clr.frac_wrong_way_pp,
                "fwd_mfe_p50": clr.fwd_mfe_p50,
                "size_fraction": clr.size_fraction,
            },
            centroid_local_peaks=clr.local_peaks_centroid,
            archetype_label=cluster_meta[cid]["label_step2"],
            shape_tag=clr.shape_tag,
            cfg=cfg,
        )
        clr.floors = floors
        clr.floors_passed_count = passed_cnt
        clr.floors_failed_list = failed_list
        clr.overall_pass = overall
        sweep_rows.append(clr)

    # Select SL per cluster
    print("[step3] selecting SL per cluster", file=sys.stderr)
    tie_tol = float(cfg["composite"]["tiebreaker_tolerance"])
    selected_per_cluster: Dict[int, Optional[ClusterSLRow]] = {}
    for cid in sorted(cluster_meta.keys()):
        cluster_rows = [r for r in sweep_rows if r.cluster_id == cid]
        sel = select_sl_for_cluster(cluster_rows, tie_tol)
        selected_per_cluster[cid] = sel

    # Write outputs
    print("[step3] writing CSVs", file=sys.stderr)
    csv_hashes: Dict[str, str] = {}

    p_per_trade = out_dir / out_cfg["per_trade_sl_csv"]
    write_per_trade_sl_csv(p_per_trade, per_trade_df)
    csv_hashes[p_per_trade.name] = _file_sha256(p_per_trade)

    p_sweep = out_dir / out_cfg["cluster_sl_sweep_csv"]
    write_cluster_sl_sweep_csv(p_sweep, sweep_rows)
    csv_hashes[p_sweep.name] = _file_sha256(p_sweep)

    p_arch = out_dir / out_cfg["archetype_summaries_csv"]
    write_archetype_summaries_csv(p_arch, selected_per_cluster, cluster_meta, pct_pc_per_cluster)
    csv_hashes[p_arch.name] = _file_sha256(p_arch)

    p_pass = out_dir / out_cfg["capturability_pass_list_csv"]
    write_capturability_pass_list_csv(p_pass, selected_per_cluster, cluster_meta)
    csv_hashes[p_pass.name] = _file_sha256(p_pass)

    p_routing = out_dir / out_cfg["cluster_routing_csv"]
    write_cluster_routing_csv(p_routing, selected_per_cluster, cluster_meta)
    csv_hashes[p_routing.name] = _file_sha256(p_routing)

    p_shape = out_dir / out_cfg["shape_tag_diagnostics_csv"]
    write_shape_tag_diagnostics_csv(p_shape, sweep_rows)
    csv_hashes[p_shape.name] = _file_sha256(p_shape)

    for cid in sorted(cluster_meta.keys()):
        p_dist = out_dir / f"{out_cfg['per_cluster_dist_prefix']}{cid}_distribution.csv"
        sel = selected_per_cluster.get(cid)
        write_per_cluster_distribution_csv(
            p_dist, per_trade_df, cid, clusters,
            selected_X=sel.X if sel else None,
        )
        csv_hashes[p_dist.name] = _file_sha256(p_dist)

        p_detail = out_dir / f"{out_cfg['per_cluster_sl_detail_prefix']}{cid}_sl_sweep_detail.csv"
        write_per_cluster_sl_detail_csv(p_detail, sweep_rows, cid)
        csv_hashes[p_detail.name] = _file_sha256(p_detail)

    p_pcdef = out_dir / out_cfg["pct_peak_and_collapse_def_txt"]
    write_pct_pc_definition(p_pcdef, cfg)

    ctx = {
        "sweep_rows": sweep_rows,
        "selected_per_cluster": selected_per_cluster,
        "cluster_meta": cluster_meta,
        "pct_pc_per_cluster": pct_pc_per_cluster,
        "cid2_disambiguation": cid2_branch,
        "out_dir": out_dir,
        "trades": trades,
        "clusters": clusters,
        "per_trade_df": per_trade_df,
    }
    return csv_hashes, ctx


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Arc 2 redo2 Step 3 — capturability characterisation (v2.1.1).")
    ap.add_argument("-c", "--config", required=True, type=Path)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    print("[step3] === RUN 1 ===", file=sys.stderr)
    hashes1, ctx = _run_once(cfg)

    hashes2: Optional[Dict[str, str]] = None
    if bool(cfg["output"].get("determinism_check", True)):
        print("[step3] === RUN 2 (determinism) ===", file=sys.stderr)
        hashes2, _ = _run_once(cfg)

    if hashes2 is not None:
        det_gate = "PASS" if all(hashes1[k] == hashes2.get(k) for k in hashes1) else "FAIL"
    else:
        det_gate = "N/A"

    out_dir = ctx["out_dir"]
    write_step3_summary_md(
        out_dir / cfg["output"]["summary_md"],
        cfg,
        ctx["sweep_rows"],
        ctx["selected_per_cluster"],
        ctx["cluster_meta"],
        ctx["pct_pc_per_cluster"],
        ctx["cid2_disambiguation"],
        det_gate,
        hashes1,
        hashes2,
    )

    n_pass = sum(1 for s in ctx["selected_per_cluster"].values() if s is not None)
    print(
        f"[step3] DONE. clusters_passing_§2={n_pass}/{len(ctx['selected_per_cluster'])}, "
        f"det={det_gate}, cid2_disambiguation={ctx['cid2_disambiguation']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
