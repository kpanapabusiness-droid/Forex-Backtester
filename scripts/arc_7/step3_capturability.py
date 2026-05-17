"""Arc 7 — Step 3 capturability characterisation.

L_ARC_PROTOCOL.md v2.1.2 §§2, 7, 11, 17. SL sweep across
{0.5, 1.0, 1.5, 2.0, 3.0, 4.0} × ATR(14)_4H at entry; for each candidate, scan
the full §15a trades_paths.csv (held + forward-observation rows) for adverse
excursion to re-impose the candidate SL, recompute peak_mfe_bar within the
truncated path, derive pre-peak metrics (mono, frac_wrong_way_pre_peak per
§17 Def C), apply §2 floors conjunctively, select the SL maximising the
capturability composite per §7 tiebreaker rules.

Per-cluster AND per-aggregate evaluation per v2.1 routing (Open-14 closure):
each tentative-label group is evaluated for each constituent cluster
individually AND for the aggregate.

bimodal_separated test (§7): Hartigan dip p < 0.05 + min-mode-mass ≥ 0.20 +
mode separation ≥ 1R on the fwd_mfe distribution at the selected SL. Routes
to §11 row 7 in parallel with the base archetype label when it fires.

Tentative archetype disambiguation:
  - tentative_Early-peak hold OR Peak-and-collapse → pct_peak_and_collapse
    < 0.30 → Early-peak hold; ≥ 0.50 → Peak-and-collapse; else flag and
    defer to Step 4 internal validation.
  - tentative_V-shape recovery → MAE-before-peak ≥ 5 bars AND peak in
    [0.4, 0.8] of (truncated) trade.

R-rescaling math: trades_paths.csv R-fields are denominated in the original
Step 1 SL (2.0 × ATR at entry). For candidate SL X × ATR, scale factor
``s = 2.0 / X`` converts original-R to new-R. Adverse-excursion threshold
in original-R units is ``low_r ≤ -1/s = -X/2.0`` (long); 1R reach is
``mfe_so_far_r ≥ X/2.0``; new close_r = original_close_r × s.

Determinism: pure data processing, no RNG except Hartigan dip's internal
permutation tests (deterministic given fixed seed inside diptest). Verified
by two consecutive runs producing byte-identical outputs.

Usage:
    py scripts/arc_7/step3_capturability.py -c configs/arc_7/step3.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ============================================================
# Per-trade SL evaluation
# ============================================================

@dataclass
class _TradeEval:
    trade_id: int
    sl_atr_mult: float
    sl_hit: bool
    truncated_at_bar: int        # bar_offset of last bar in truncated path
    peak_mfe_bar: int            # bar_offset of max mfe within truncated path
    fwd_mfe_new_r: float         # max mfe in new R units (within truncated path)
    final_r_new: float           # in new R units
    mono_pre_peak: float
    wrong_way_pre_peak: bool
    bars_pre_peak: int           # peak_mfe_bar + 1 (count of bars 0..peak inclusive)
    peak_mfe_atr_units: float    # for tiebreaker 1 (physical capture)


def _eval_trade_at_sl(
    path: pd.DataFrame, sl_atr_mult: float, original_sl_atr_mult: float
) -> _TradeEval:
    """Evaluate a single trade's path under a candidate SL.

    ``path`` is expected to be ordered by bar_offset ascending and contain the
    full forward window (held + forward-observation rows). All R fields use the
    original SL (``original_sl_atr_mult`` × ATR) as denominator.
    """
    scale = original_sl_atr_mult / sl_atr_mult           # original_R → new_R
    sl_threshold_orig_r = -1.0 / scale                   # = -sl_atr_mult / original_sl_atr_mult

    bar_offsets = path["bar_offset"].to_numpy(dtype=int)
    low_r = path["low_r"].to_numpy(dtype=float)
    close_r = path["close_r"].to_numpy(dtype=float)
    mfe_so_far = path["mfe_so_far_r"].to_numpy(dtype=float)
    mae_so_far = path["mae_so_far_r"].to_numpy(dtype=float)
    n_bars = bar_offsets.size

    # Find first adverse-excursion hit (long: low_r <= threshold).
    sl_hit_idx: int = -1
    for i in range(n_bars):
        if low_r[i] <= sl_threshold_orig_r:
            sl_hit_idx = i
            break

    if sl_hit_idx >= 0:
        # Truncate at SL hit bar.
        trunc_end = sl_hit_idx + 1
        sl_hit = True
        final_r_new = -1.0
        truncated_bar = int(bar_offsets[sl_hit_idx])
    else:
        # Survives to end of recorded window.
        trunc_end = n_bars
        sl_hit = False
        last_close_orig = float(close_r[n_bars - 1])
        final_r_new = last_close_orig * scale
        truncated_bar = int(bar_offsets[n_bars - 1])

    mfe_trunc = mfe_so_far[:trunc_end]
    close_r_trunc = close_r[:trunc_end]
    mae_trunc = mae_so_far[:trunc_end]
    bar_offsets_trunc = bar_offsets[:trunc_end]

    # Recompute peak_mfe_bar within truncated path. Use argmax (first occurrence
    # of the max for ties — deterministic).
    if mfe_trunc.size == 0:
        peak_idx = 0
        peak_mfe_new = 0.0
        peak_mfe_atr = 0.0
        peak_bar = 0
    else:
        peak_idx = int(np.argmax(mfe_trunc))
        peak_mfe_new = float(mfe_trunc[peak_idx] * scale)
        # peak_mfe in ATR units: original_R × original_sl_atr_mult → ATR units
        peak_mfe_atr = float(mfe_trunc[peak_idx] * original_sl_atr_mult)
        peak_bar = int(bar_offsets_trunc[peak_idx])

    # Pre-peak metrics: bars 0..peak_idx inclusive.
    pre_close_orig = close_r_trunc[: peak_idx + 1]
    pre_mae_orig = mae_trunc[: peak_idx + 1]
    # close_r in new R units for monotonicity_in_profit.
    pre_close_new = pre_close_orig * scale
    in_profit = pre_close_new[pre_close_new > 0]
    if in_profit.size >= 2:
        mono_pp = float(np.mean(in_profit[1:] >= in_profit[:-1]))
    else:
        mono_pp = 0.0

    # frac_wrong_way_pre_peak (per-trade): MAE ≤ −1R in new units on or before
    # peak_mfe_bar. mae_so_far_r is running min of low_r, so mae at peak_idx is
    # the running min over bars 0..peak_idx (inclusive).
    mae_at_peak_orig = float(pre_mae_orig[-1]) if pre_mae_orig.size > 0 else 0.0
    mae_at_peak_new = mae_at_peak_orig * scale
    wrong_way_pp = bool(mae_at_peak_new <= -1.0)

    return _TradeEval(
        trade_id=int(path["trade_id"].iloc[0]),
        sl_atr_mult=float(sl_atr_mult),
        sl_hit=sl_hit,
        truncated_at_bar=truncated_bar,
        peak_mfe_bar=peak_bar,
        fwd_mfe_new_r=peak_mfe_new,
        final_r_new=float(final_r_new),
        mono_pre_peak=mono_pp,
        wrong_way_pre_peak=wrong_way_pp,
        bars_pre_peak=int(peak_idx + 1),
        peak_mfe_atr_units=peak_mfe_atr,
    )


# ============================================================
# Unit (cluster or aggregate) evaluation
# ============================================================

@dataclass
class _UnitEvalAtSL:
    unit_id: str
    sl_atr_mult: float
    n: int
    size_fraction: float
    mono_pre_peak: float          # mean of per-trade
    frac_wrong_way_pre_peak: float
    fwd_mfe_p50: float
    fwd_mfe_p5: float
    fwd_mfe_p10: float
    fwd_mfe_p25: float
    fwd_mfe_p75: float
    fwd_mfe_p90: float
    fwd_mfe_p95: float
    fwd_mfe_p99: float
    frac_reach_1R: float
    frac_reach_2R: float
    final_r_mean: float
    final_r_p5: float
    final_r_p25: float
    final_r_p50: float
    final_r_p75: float
    final_r_p95: float
    shape_tag: str
    bimodal_separated: bool
    hartigan_dip_stat: float
    hartigan_dip_pvalue: float
    bimodal_mode_separation_r: float
    bimodal_min_mode_mass: float
    bimodal_left_mode: float
    bimodal_right_mode: float
    bimodal_left_mass: float
    bimodal_right_mass: float
    pct_peak_and_collapse: float
    peak_mfe_p50_atr_units: float
    composite: float
    gate_size: bool
    gate_mono: bool
    gate_mfe_p50: bool
    gate_reach_1R: bool
    gate_wrong_way: bool
    gate_shape: bool
    gate_local_peaks: bool
    gate_pass: bool
    kill_reasons: List[str] = field(default_factory=list)


def _classify_shape_tag(
    fwd_mfe: np.ndarray, dip_stat: float, dip_p: float, cfg_bimodal: dict
) -> Tuple[str, bool, float, float, float, float, float, float]:
    """Return shape_tag plus bimodal-test detail.

    Order: bimodal_separated → tight_unimodal → heavy_right_tail → scattered →
    unclassified. The bimodal_separated test combines Hartigan dip with KDE
    mode finding for min-mode-mass and separation checks.
    """
    n = fwd_mfe.size
    if n < 4:
        return "unclassified", False, dip_stat, dip_p, 0.0, 0.0, 0.0, 0.0

    bimodal_pass = False
    sep = 0.0
    min_mass = 0.0
    left_mode = 0.0
    right_mode = 0.0
    left_mass = 0.0
    right_mass = 0.0

    if dip_p < cfg_bimodal["hartigan_dip_p_max"]:
        # Run KDE mode finding.
        try:
            from scipy import stats as sps
            kde = sps.gaussian_kde(fwd_mfe)
            x_grid = np.linspace(float(fwd_mfe.min()), float(fwd_mfe.max()), int(cfg_bimodal["kde_grid_points"]))
            dens = kde(x_grid)
            peaks: List[int] = []
            for i in range(1, len(dens) - 1):
                if dens[i] > dens[i - 1] and dens[i] > dens[i + 1]:
                    peaks.append(i)
            if len(peaks) >= 2:
                peaks_sorted = sorted(peaks, key=lambda i: -dens[i])[:2]
                i1, i2 = sorted(peaks_sorted)
                left_mode = float(x_grid[i1])
                right_mode = float(x_grid[i2])
                sep = right_mode - left_mode
                # Mass: split sample at the valley between the two modes.
                valley_i = i1 + int(np.argmin(dens[i1:i2 + 1]))
                cut = float(x_grid[valley_i])
                left_mass = float(np.mean(fwd_mfe <= cut))
                right_mass = float(np.mean(fwd_mfe > cut))
                min_mass = min(left_mass, right_mass)
                if (
                    sep >= cfg_bimodal["mode_separation_r_min"]
                    and min_mass >= cfg_bimodal["min_mode_mass"]
                ):
                    bimodal_pass = True
        except Exception:
            pass

    if bimodal_pass:
        return "bimodal_separated", True, dip_stat, dip_p, sep, min_mass, left_mode, right_mode

    # Fall through to v2.0-style classifier on the fwd_mfe distribution.
    from scipy import stats as sps
    p25 = float(np.percentile(fwd_mfe, 25))
    p75 = float(np.percentile(fwd_mfe, 75))
    p95 = float(np.percentile(fwd_mfe, 95))
    iqr = p75 - p25
    try:
        skew = float(sps.skew(fwd_mfe, bias=False))
    except Exception:
        skew = 0.0
    std = float(np.std(fwd_mfe, ddof=1)) if fwd_mfe.size >= 2 else 0.0

    if iqr <= 1.5 and abs(skew) <= 1.0:
        tag = "tight_unimodal"
    elif p75 > 0 and (p95 / p75) >= 2.5 and skew >= 1.0:
        tag = "heavy_right_tail"
    elif std >= 2.5:
        tag = "scattered"
    else:
        tag = "unclassified"
    return tag, False, dip_stat, dip_p, sep, min_mass, left_mode, right_mode


def _eval_unit(
    unit_id: str,
    paths_for_trades: Dict[int, pd.DataFrame],
    trade_ids: List[int],
    cfg: dict,
    pool_size: int,
    local_peaks_centroid: float,
    local_peaks_rule: str,
) -> Tuple[Dict[float, _UnitEvalAtSL], Dict[float, List[_TradeEval]]]:
    """Sweep candidate SLs for a unit (cluster or aggregate)."""
    import diptest
    from scipy import stats as sps  # noqa: F401 — used inside _classify_shape_tag

    original_sl = float(cfg["original_sl_atr_mult"])
    candidates = list(cfg["sl_sweep"]["candidates_atr_mult"])
    floors = cfg["floors"]
    bimodal_cfg = cfg["bimodal_test"]
    pc_cfg = cfg["peak_and_collapse"]

    per_sl_eval: Dict[float, _UnitEvalAtSL] = {}
    per_sl_trades: Dict[float, List[_TradeEval]] = {}

    n_trades = len(trade_ids)
    size_fraction = n_trades / pool_size if pool_size > 0 else 0.0

    for sl in candidates:
        trade_evals: List[_TradeEval] = []
        for tid in trade_ids:
            path = paths_for_trades[tid]
            te = _eval_trade_at_sl(path, sl, original_sl)
            trade_evals.append(te)
        per_sl_trades[sl] = trade_evals

        if not trade_evals:
            continue

        fwd_mfe = np.array([t.fwd_mfe_new_r for t in trade_evals], dtype=float)
        final_r = np.array([t.final_r_new for t in trade_evals], dtype=float)
        mono_pp = np.array([t.mono_pre_peak for t in trade_evals], dtype=float)
        wrong_way = np.array([t.wrong_way_pre_peak for t in trade_evals], dtype=bool)
        peak_atr = np.array([t.peak_mfe_atr_units for t in trade_evals], dtype=float)

        mono_centroid = float(mono_pp.mean())
        frac_wrong = float(wrong_way.mean())
        fwd_p5 = float(np.percentile(fwd_mfe, 5))
        fwd_p10 = float(np.percentile(fwd_mfe, 10))
        fwd_p25 = float(np.percentile(fwd_mfe, 25))
        fwd_p50 = float(np.percentile(fwd_mfe, 50))
        fwd_p75 = float(np.percentile(fwd_mfe, 75))
        fwd_p90 = float(np.percentile(fwd_mfe, 90))
        fwd_p95 = float(np.percentile(fwd_mfe, 95))
        fwd_p99 = float(np.percentile(fwd_mfe, 99))
        reach_1R = float(np.mean(fwd_mfe >= 1.0))
        reach_2R = float(np.mean(fwd_mfe >= 2.0))
        fr_mean = float(final_r.mean())
        fr_p5 = float(np.percentile(final_r, 5))
        fr_p25 = float(np.percentile(final_r, 25))
        fr_p50 = float(np.percentile(final_r, 50))
        fr_p75 = float(np.percentile(final_r, 75))
        fr_p95 = float(np.percentile(final_r, 95))
        peak_p50_atr = float(np.percentile(peak_atr, 50))

        # Hartigan dip test on fwd_mfe (deterministic given fixed array).
        try:
            dip_stat, dip_p = diptest.diptest(fwd_mfe)
        except Exception:
            dip_stat, dip_p = 0.0, 1.0

        shape_tag, bimodal_pass, dip_stat, dip_p, sep, min_mass, left_mode, right_mode = _classify_shape_tag(
            fwd_mfe, float(dip_stat), float(dip_p), bimodal_cfg
        )
        left_mass = 0.0
        right_mass = 0.0
        if bimodal_pass and sep > 0:
            # Recompute masses using the same valley point used in classification.
            mid = (left_mode + right_mode) / 2.0
            left_mass = float(np.mean(fwd_mfe <= mid))
            right_mass = float(np.mean(fwd_mfe > mid))
            min_mass = min(left_mass, right_mass)

        # pct_peak_and_collapse: trades that reached threshold MFE but collapsed.
        pc_mfe_thresh = float(pc_cfg["mfe_threshold_r"])
        pc_ratio = float(pc_cfg["final_r_ratio_threshold"])
        pc_mask = (fwd_mfe >= pc_mfe_thresh) & (final_r <= pc_ratio * fwd_mfe)
        pct_pc = float(pc_mask.mean()) if fwd_mfe.size > 0 else 0.0

        # §2 gates.
        gate_size = size_fraction >= float(floors["size_fraction_min"])
        gate_mono = mono_centroid >= float(floors["mono_pre_peak_min"])
        gate_mfe_p50 = fwd_p50 >= float(floors["fwd_mfe_p50_min_r"])
        gate_reach_1R = reach_1R >= float(floors["frac_reach_1R_min"])
        gate_wrong_way = frac_wrong <= float(floors["frac_wrong_way_pre_peak_max"])
        gate_shape = shape_tag != "scattered"

        # local_peaks rule per archetype. For tentative groups we use "any".
        if local_peaks_rule == "any":
            gate_local_peaks = True
        elif local_peaks_rule == "monotone_ceiling_4":
            gate_local_peaks = local_peaks_centroid <= 4.0
        elif local_peaks_rule == "stepwise_5_to_50":
            gate_local_peaks = 5.0 <= local_peaks_centroid <= 50.0
        elif local_peaks_rule == "random_walk_floor_8":
            gate_local_peaks = local_peaks_centroid >= 8.0
        else:
            gate_local_peaks = True

        gate_pass = all(
            [gate_size, gate_mono, gate_mfe_p50, gate_reach_1R, gate_wrong_way, gate_shape, gate_local_peaks]
        )

        kill_reasons: List[str] = []
        if not gate_size:
            kill_reasons.append(f"size_fraction {size_fraction:.4f} < {floors['size_fraction_min']}")
        if not gate_mono:
            kill_reasons.append(f"mono_pre_peak {mono_centroid:.4f} < {floors['mono_pre_peak_min']}")
        if not gate_mfe_p50:
            kill_reasons.append(f"fwd_mfe_p50 {fwd_p50:.4f}R < {floors['fwd_mfe_p50_min_r']}R")
        if not gate_reach_1R:
            kill_reasons.append(f"frac_reach_1R {reach_1R:.4f} < {floors['frac_reach_1R_min']}")
        if not gate_wrong_way:
            kill_reasons.append(f"frac_wrong_way_pre_peak {frac_wrong:.4f} > {floors['frac_wrong_way_pre_peak_max']}")
        if not gate_shape:
            kill_reasons.append("shape_tag=scattered")
        if not gate_local_peaks:
            kill_reasons.append(f"local_peaks_centroid {local_peaks_centroid:.2f} fails {local_peaks_rule}")

        composite = (
            (mono_centroid - float(floors["mono_pre_peak_min"]))
            + (reach_1R - float(floors["frac_reach_1R_min"]))
            + (float(floors["frac_wrong_way_pre_peak_max"]) - frac_wrong)
        )

        per_sl_eval[sl] = _UnitEvalAtSL(
            unit_id=unit_id,
            sl_atr_mult=float(sl),
            n=n_trades,
            size_fraction=size_fraction,
            mono_pre_peak=mono_centroid,
            frac_wrong_way_pre_peak=frac_wrong,
            fwd_mfe_p50=fwd_p50,
            fwd_mfe_p5=fwd_p5,
            fwd_mfe_p10=fwd_p10,
            fwd_mfe_p25=fwd_p25,
            fwd_mfe_p75=fwd_p75,
            fwd_mfe_p90=fwd_p90,
            fwd_mfe_p95=fwd_p95,
            fwd_mfe_p99=fwd_p99,
            frac_reach_1R=reach_1R,
            frac_reach_2R=reach_2R,
            final_r_mean=fr_mean,
            final_r_p5=fr_p5,
            final_r_p25=fr_p25,
            final_r_p50=fr_p50,
            final_r_p75=fr_p75,
            final_r_p95=fr_p95,
            shape_tag=shape_tag,
            bimodal_separated=bimodal_pass,
            hartigan_dip_stat=float(dip_stat),
            hartigan_dip_pvalue=float(dip_p),
            bimodal_mode_separation_r=float(sep),
            bimodal_min_mode_mass=float(min_mass),
            bimodal_left_mode=float(left_mode),
            bimodal_right_mode=float(right_mode),
            bimodal_left_mass=float(left_mass),
            bimodal_right_mass=float(right_mass),
            pct_peak_and_collapse=pct_pc,
            peak_mfe_p50_atr_units=peak_p50_atr,
            composite=float(composite),
            gate_size=gate_size,
            gate_mono=gate_mono,
            gate_mfe_p50=gate_mfe_p50,
            gate_reach_1R=gate_reach_1R,
            gate_wrong_way=gate_wrong_way,
            gate_shape=gate_shape,
            gate_local_peaks=gate_local_peaks,
            gate_pass=gate_pass,
            kill_reasons=kill_reasons,
        )

    return per_sl_eval, per_sl_trades


def _select_sl(per_sl: Dict[float, _UnitEvalAtSL], tie_tolerance: float) -> Optional[float]:
    passing = [(sl, ev) for sl, ev in per_sl.items() if ev.gate_pass]
    if not passing:
        return None
    max_comp = max(ev.composite for _, ev in passing)
    tied = [(sl, ev) for sl, ev in passing if (max_comp - ev.composite) <= tie_tolerance]
    # Tiebreaker 1: larger peak_mfe in ATR units.
    tied.sort(key=lambda x: (-x[1].peak_mfe_p50_atr_units, x[0]))
    if len(tied) > 1 and abs(tied[0][1].peak_mfe_p50_atr_units - tied[1][1].peak_mfe_p50_atr_units) < 1e-9:
        # Tiebreaker 2: smaller SL.
        tied.sort(key=lambda x: x[0])
    return float(tied[0][0])


# ============================================================
# Archetype disambiguation (Step 3 resolves tentative labels)
# ============================================================


def _disambiguate_archetype(tentative_label: str, ev: _UnitEvalAtSL, trade_evals: List[_TradeEval]) -> Tuple[str, str]:
    """Return (final_archetype_label, notes)."""
    if "Early-peak hold OR Peak-and-collapse" in tentative_label:
        pct = ev.pct_peak_and_collapse
        if pct < 0.30:
            return "Early-peak hold", f"pct_peak_and_collapse {pct:.4f} < 0.30"
        if pct >= 0.50:
            return "Peak-and-collapse", f"pct_peak_and_collapse {pct:.4f} >= 0.50"
        return (
            "Early-peak hold OR Peak-and-collapse (Step 4 disambiguation)",
            f"pct_peak_and_collapse {pct:.4f} in [0.30, 0.50) — defer to Step 4 per-fold internal validation",
        )
    if "V-shape recovery" in tentative_label:
        # MAE-before-peak >= 5 bars AND peak in [0.4, 0.8] of trade.
        # peak position fraction within truncated trade.
        peak_frac = np.array(
            [
                (t.peak_mfe_bar / max(t.truncated_at_bar, 1)) if t.truncated_at_bar > 0 else 0.0
                for t in trade_evals
            ],
            dtype=float,
        )
        # MAE-before-peak >= 5 bars per-trade (treat as positional: peak_mfe_bar >= 5).
        peak_bars = np.array([t.peak_mfe_bar for t in trade_evals], dtype=float)
        mae_5plus = float(np.mean(peak_bars >= 5.0))
        peak_in_band = float(np.mean((peak_frac >= 0.4) & (peak_frac <= 0.8)))
        # V-shape confirmation: at least 50% of trades meet each condition.
        if mae_5plus >= 0.5 and peak_in_band >= 0.5:
            return "V-shape recovery", f"peak_bars>=5 frac={mae_5plus:.3f}; peak_pos in [0.4,0.8] frac={peak_in_band:.3f}"
        return (
            "V-shape recovery (forward-geometry weak)",
            f"V-shape conditions: peak_bars>=5 frac={mae_5plus:.3f} (need >=0.5); peak_pos in [0.4,0.8] frac={peak_in_band:.3f} (need >=0.5)",
        )
    return tentative_label, "no disambiguation rule for label"


# ============================================================
# Output writers
# ============================================================


def _fmt_g(x: Any) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
        if not math.isfinite(xf):
            return ""
    except Exception:
        return str(x)
    return f"{xf:.10g}"


def _slug(label: str) -> str:
    return (
        label.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("+", "and")
    )


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_archetype_summaries(
    out_path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    cols = [
        "unit_id", "type", "constituent_clusters", "n", "size_fraction",
        "tentative_label", "final_archetype_label", "selected_SL_atr_mult", "selected_R_atr",
        "composite_at_selected_sl", "mono_pre_peak", "frac_reach_1R", "frac_reach_2R",
        "frac_wrong_way_pre_peak", "fwd_mfe_h240_p50", "shape_tag", "bimodal_separated_pass",
        "pct_peak_and_collapse", "gate_status", "kill_criteria",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r.get("unit_id", ""),
                r.get("type", ""),
                r.get("constituent_clusters", ""),
                str(int(r.get("n", 0))),
                _fmt_g(r.get("size_fraction", 0.0)),
                r.get("tentative_label", ""),
                r.get("final_archetype_label", ""),
                _fmt_g(r.get("selected_SL_atr_mult", "")),
                _fmt_g(r.get("selected_R_atr", "")),
                _fmt_g(r.get("composite", "")),
                _fmt_g(r.get("mono_pre_peak", "")),
                _fmt_g(r.get("frac_reach_1R", "")),
                _fmt_g(r.get("frac_reach_2R", "")),
                _fmt_g(r.get("frac_wrong_way_pre_peak", "")),
                _fmt_g(r.get("fwd_mfe_h240_p50", "")),
                r.get("shape_tag", ""),
                "1" if r.get("bimodal_separated", False) else "0",
                _fmt_g(r.get("pct_peak_and_collapse", "")),
                r.get("gate_status", "FAIL"),
                ";".join(r.get("kill_reasons", [])),
            ])


def write_unit_sl_sweep(
    out_path: Path,
    unit_id: str,
    per_sl: Dict[float, _UnitEvalAtSL],
    selected_sl: Optional[float],
    original_sl: float,
) -> None:
    cols = [
        "sl_atr_mult", "R_atr", "size_fraction", "mono_pre_peak",
        "fwd_mfe_p5", "fwd_mfe_p25", "fwd_mfe_p50", "fwd_mfe_p75", "fwd_mfe_p95",
        "frac_reach_1R", "frac_reach_2R", "frac_wrong_way_pre_peak",
        "shape_tag", "bimodal_separated", "hartigan_dip_stat", "hartigan_dip_pvalue",
        "pct_peak_and_collapse", "peak_mfe_p50_atr_units",
        "gate_size", "gate_mono", "gate_mfe_p50", "gate_reach_1R", "gate_wrong_way",
        "gate_shape", "gate_local_peaks", "gate_pass", "composite", "selected",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for sl in sorted(per_sl.keys()):
            ev = per_sl[sl]
            sel = (selected_sl is not None and abs(sl - selected_sl) < 1e-9)
            w.writerow([
                _fmt_g(sl), _fmt_g(sl),
                _fmt_g(ev.size_fraction), _fmt_g(ev.mono_pre_peak),
                _fmt_g(ev.fwd_mfe_p5), _fmt_g(ev.fwd_mfe_p25), _fmt_g(ev.fwd_mfe_p50),
                _fmt_g(ev.fwd_mfe_p75), _fmt_g(ev.fwd_mfe_p95),
                _fmt_g(ev.frac_reach_1R), _fmt_g(ev.frac_reach_2R),
                _fmt_g(ev.frac_wrong_way_pre_peak),
                ev.shape_tag, "1" if ev.bimodal_separated else "0",
                _fmt_g(ev.hartigan_dip_stat), _fmt_g(ev.hartigan_dip_pvalue),
                _fmt_g(ev.pct_peak_and_collapse), _fmt_g(ev.peak_mfe_p50_atr_units),
                "1" if ev.gate_size else "0", "1" if ev.gate_mono else "0",
                "1" if ev.gate_mfe_p50 else "0", "1" if ev.gate_reach_1R else "0",
                "1" if ev.gate_wrong_way else "0", "1" if ev.gate_shape else "0",
                "1" if ev.gate_local_peaks else "0", "1" if ev.gate_pass else "0",
                _fmt_g(ev.composite), "1" if sel else "0",
            ])


def write_unit_distribution(
    out_path: Path,
    unit_id: str,
    ev: _UnitEvalAtSL,
    trade_evals: List[_TradeEval],
) -> None:
    fwd_mfe = np.array([t.fwd_mfe_new_r for t in trade_evals], dtype=float)
    final_r = np.array([t.final_r_new for t in trade_evals], dtype=float)
    # Mass-in-band on final_r.
    bands = [
        ("le_0R", -float("inf"), 0.0),
        ("0R_to_0.5R", 0.0, 0.5),
        ("0.5R_to_1R", 0.5, 1.0),
        ("1R_to_2R", 1.0, 2.0),
        ("2R_to_5R", 2.0, 5.0),
        ("gt_5R", 5.0, float("inf")),
    ]
    band_rows = []
    for name, lo, hi in bands:
        mask = (final_r >= lo) & (final_r < hi) if hi != float("inf") else (final_r >= lo)
        band_rows.append((name, lo, hi, int(mask.sum()), float(mask.mean())))

    pct_set = [5, 10, 25, 50, 75, 90, 95, 99]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["metric", "value"])
        w.writerow(["unit_id", unit_id])
        w.writerow(["n", int(ev.n)])
        w.writerow(["sl_atr_mult", _fmt_g(ev.sl_atr_mult)])
        for q in pct_set:
            w.writerow([f"fwd_mfe_p{q}", _fmt_g(float(np.percentile(fwd_mfe, q)))])
        for q in pct_set:
            w.writerow([f"final_r_p{q}", _fmt_g(float(np.percentile(final_r, q)))])
        w.writerow(["fwd_mfe_mean", _fmt_g(float(fwd_mfe.mean()))])
        w.writerow(["final_r_mean", _fmt_g(float(final_r.mean()))])
        w.writerow(["hartigan_dip_stat", _fmt_g(ev.hartigan_dip_stat)])
        w.writerow(["hartigan_dip_pvalue", _fmt_g(ev.hartigan_dip_pvalue)])
        w.writerow(["shape_tag", ev.shape_tag])
        w.writerow(["bimodal_separated", "1" if ev.bimodal_separated else "0"])
        w.writerow(["bimodal_mode_separation_r", _fmt_g(ev.bimodal_mode_separation_r)])
        w.writerow(["bimodal_min_mode_mass", _fmt_g(ev.bimodal_min_mode_mass)])
        w.writerow(["bimodal_left_mode_r", _fmt_g(ev.bimodal_left_mode)])
        w.writerow(["bimodal_right_mode_r", _fmt_g(ev.bimodal_right_mode)])
        w.writerow(["bimodal_left_mass", _fmt_g(ev.bimodal_left_mass)])
        w.writerow(["bimodal_right_mass", _fmt_g(ev.bimodal_right_mass)])
        w.writerow(["pct_peak_and_collapse", _fmt_g(ev.pct_peak_and_collapse)])
        for name, lo, hi, count, frac in band_rows:
            w.writerow([f"mass_final_r_{name}_count", str(count)])
            w.writerow([f"mass_final_r_{name}_frac", _fmt_g(frac)])


def write_cluster_routing(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = ["cluster_id", "tentative_label", "individual_pass", "aggregate_pass",
            "disposition", "final_archetype_label", "notes"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([
                str(int(r["cluster_id"])),
                r["tentative_label"],
                "1" if r["individual_pass"] else "0",
                "1" if r["aggregate_pass"] else "0",
                r["disposition"],
                r["final_archetype_label"],
                r["notes"],
            ])


def write_pass_list(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = [
        "unit_id", "type", "final_archetype_label", "selected_SL_atr_mult",
        "selected_R_atr", "n", "candidate_pipelines",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r["unit_id"], r["type"], r["final_archetype_label"],
                _fmt_g(r["selected_SL_atr_mult"]), _fmt_g(r["selected_R_atr"]),
                str(int(r["n"])), r["candidate_pipelines"],
            ])


def write_histograms(unit_id: str, ev: _UnitEvalAtSL, trade_evals: List[_TradeEval], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fwd = np.array([t.fwd_mfe_new_r for t in trade_evals], dtype=float)
    fin = np.array([t.final_r_new for t in trade_evals], dtype=float)
    for arr, name, color in [(fwd, "fwd_mfe", "#4c72b0"), (fin, "final_r", "#dd8452")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(arr, bins=40, color=color, edgecolor="black", alpha=0.85)
        ax.axvline(0.0, color="black", linestyle="-", linewidth=0.6)
        if name == "fwd_mfe":
            ax.axvline(1.0, color="red", linestyle="--", linewidth=0.8, label="1R floor reach")
            ax.axvline(1.5, color="green", linestyle="--", linewidth=0.8, label="p50 floor 1.5R")
            ax.legend()
        ax.set_xlabel(f"{name} (R)")
        ax.set_ylabel("trade count")
        ax.set_title(
            f"Arc 7 Step 3 — {unit_id} {name} @ SL={ev.sl_atr_mult}×ATR "
            f"(n={ev.n}, shape={ev.shape_tag})"
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"archetype_{_slug(unit_id)}_{name}_histogram.png", dpi=110,
                    metadata={"Software": ""})
        plt.close(fig)


# ============================================================
# Driver
# ============================================================


def _build_paths_index(paths_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Return {trade_id: sorted-by-bar_offset path DataFrame}."""
    out: Dict[int, pd.DataFrame] = {}
    paths_sorted = paths_df.sort_values(["trade_id", "bar_offset"], kind="mergesort")
    for tid, g in paths_sorted.groupby("trade_id", sort=True):
        out[int(tid)] = g.reset_index(drop=True)
    return out


def _run_once(cfg: dict) -> Tuple[Dict[str, str], dict]:
    in_cfg = cfg["input"]
    out_cfg = cfg["output"]

    step1_dir = _REPO_ROOT / in_cfg["step1_dir"]
    step2_dir = _REPO_ROOT / in_cfg["step2_dir"]
    out_dir = _REPO_ROOT / out_cfg["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_df = pd.read_csv(step1_dir / in_cfg["trades_csv"])
    paths_df = pd.read_csv(step1_dir / in_cfg["paths_csv"])
    clusters_df = pd.read_csv(step2_dir / in_cfg["clusters_csv"])
    features_df = pd.read_csv(step2_dir / in_cfg["features_csv"])

    pool_size = int(len(trades_df))

    paths_index = _build_paths_index(paths_df)

    # cluster_id → (trade_ids, centroid_local_peaks).
    cluster_trades: Dict[int, List[int]] = {}
    for cid, g in clusters_df.groupby("cluster_id", sort=True):
        cluster_trades[int(cid)] = sorted(g["trade_id"].astype(int).tolist())

    cluster_peaks_centroid: Dict[int, float] = {}
    for cid in cluster_trades:
        tids = cluster_trades[cid]
        sub = features_df[features_df["trade_id"].isin(tids)]
        cluster_peaks_centroid[int(cid)] = float(sub["local_peaks_count"].mean())

    # Build units: per-cluster + per-aggregate per group.
    units: List[Tuple[str, str, List[int], List[int], float, str, str]] = []
    # tuple: (unit_id, type, cluster_ids, trade_ids, local_peaks_centroid, local_peaks_rule, tentative_label)
    for group in cfg["archetype_groups"]:
        tlab = group["tentative_label"]
        rule = group["local_peaks_rule"]
        cids = list(group["cluster_ids"])
        # Per-cluster units.
        for cid in cids:
            tids = cluster_trades[int(cid)]
            unit_id = f"c{cid}"
            units.append((unit_id, "cluster", [int(cid)], tids, cluster_peaks_centroid[int(cid)], rule, tlab))
        # Aggregate (if >= 2 clusters in group).
        if len(cids) >= 2:
            tids = sorted(sum((cluster_trades[int(cid)] for cid in cids), []))
            unit_id = "agg_" + "_".join(f"c{cid}" for cid in cids)
            sub = features_df[features_df["trade_id"].isin(tids)]
            agg_peaks = float(sub["local_peaks_count"].mean())
            units.append((unit_id, "aggregate", [int(c) for c in cids], tids, agg_peaks, rule, tlab))

    # Evaluate each unit.
    unit_evals: Dict[str, Tuple[Dict[float, _UnitEvalAtSL], Dict[float, List[_TradeEval]]]] = {}
    for unit_id, unit_type, cids, tids, lp_centroid, rule, tlab in units:
        print(f"[arc_7 step3] evaluating {unit_id} (n={len(tids)}, peaks_centroid={lp_centroid:.2f})", file=sys.stderr)
        per_sl, per_sl_trades = _eval_unit(
            unit_id, paths_index, tids, cfg, pool_size, lp_centroid, rule
        )
        unit_evals[unit_id] = (per_sl, per_sl_trades)

    # Select SL per unit, write per-unit SL sweep CSV.
    tie_tol = float(cfg["composite"]["tie_tolerance_absolute"])
    selected_sls: Dict[str, Optional[float]] = {}
    original_sl = float(cfg["original_sl_atr_mult"])
    for unit_id, unit_type, cids, tids, lp_centroid, rule, tlab in units:
        per_sl, per_sl_trades = unit_evals[unit_id]
        selected_sl = _select_sl(per_sl, tie_tol)
        selected_sls[unit_id] = selected_sl
        write_unit_sl_sweep(
            out_dir / f"archetype_{_slug(unit_id)}_sl_sweep.csv",
            unit_id, per_sl, selected_sl, original_sl,
        )

    # Write per-unit distribution (at selected SL, or at SL=2.0 fallback per dispatch).
    summary_rows: List[Dict[str, Any]] = []
    pass_rows: List[Dict[str, Any]] = []
    routing_rows: List[Dict[str, Any]] = []
    csv_hashes: Dict[str, str] = {}

    for unit_id, unit_type, cids, tids, lp_centroid, rule, tlab in units:
        per_sl, per_sl_trades = unit_evals[unit_id]
        selected_sl = selected_sls[unit_id]
        # Use selected SL if any passes; otherwise use SL=2.0 for distribution-only reporting.
        ref_sl = selected_sl if selected_sl is not None else 2.0
        if ref_sl not in per_sl:
            ref_sl = sorted(per_sl.keys())[0]
        ev_ref = per_sl[ref_sl]
        trade_evals = per_sl_trades[ref_sl]

        write_unit_distribution(
            out_dir / f"archetype_{_slug(unit_id)}_distribution.csv",
            unit_id, ev_ref, trade_evals,
        )
        write_histograms(unit_id, ev_ref, trade_evals, out_dir)

        # Disambiguate label (if PASS, use selected; if FAIL, still disambiguate at ref for reporting).
        final_label, dis_notes = _disambiguate_archetype(tlab, ev_ref, trade_evals)

        summary_rows.append({
            "unit_id": unit_id,
            "type": unit_type,
            "constituent_clusters": "+".join(f"c{c}" for c in cids),
            "n": ev_ref.n,
            "size_fraction": ev_ref.size_fraction,
            "tentative_label": tlab,
            "final_archetype_label": final_label,
            "selected_SL_atr_mult": ev_ref.sl_atr_mult if selected_sl is not None else "",
            "selected_R_atr": ev_ref.sl_atr_mult if selected_sl is not None else "",
            "composite": ev_ref.composite if selected_sl is not None else "",
            "mono_pre_peak": ev_ref.mono_pre_peak,
            "frac_reach_1R": ev_ref.frac_reach_1R,
            "frac_reach_2R": ev_ref.frac_reach_2R,
            "frac_wrong_way_pre_peak": ev_ref.frac_wrong_way_pre_peak,
            "fwd_mfe_h240_p50": ev_ref.fwd_mfe_p50,
            "shape_tag": ev_ref.shape_tag,
            "bimodal_separated": ev_ref.bimodal_separated,
            "pct_peak_and_collapse": ev_ref.pct_peak_and_collapse,
            "gate_status": "PASS" if selected_sl is not None else "FAIL",
            "kill_reasons": [] if selected_sl is not None else ev_ref.kill_reasons,
        })

        if selected_sl is not None:
            pass_rows.append({
                "unit_id": unit_id,
                "type": unit_type,
                "final_archetype_label": final_label,
                "selected_SL_atr_mult": ev_ref.sl_atr_mult,
                "selected_R_atr": ev_ref.sl_atr_mult,
                "n": ev_ref.n,
                "candidate_pipelines": "E_and_D1",   # initial — full extractability at Step 4
            })

    # Routing per cluster.
    for group in cfg["archetype_groups"]:
        tlab = group["tentative_label"]
        cids = list(group["cluster_ids"])
        agg_id = "agg_" + "_".join(f"c{c}" for c in cids) if len(cids) >= 2 else None
        agg_pass = selected_sls.get(agg_id) is not None if agg_id else False
        for cid in cids:
            unit_id = f"c{cid}"
            indiv_pass = selected_sls.get(unit_id) is not None
            if indiv_pass and agg_pass:
                disposition = "proceeds_both"
            elif indiv_pass and not agg_pass:
                disposition = "proceeds_as_individual"
            elif not indiv_pass and agg_pass:
                disposition = "proceeds_as_aggregate"
            else:
                disposition = "dies"
            # Use individual eval for final label if individual passes; else aggregate; else ref.
            ref_unit = unit_id if indiv_pass else (agg_id if agg_pass else unit_id)
            per_sl, per_sl_trades = unit_evals[ref_unit]
            sel = selected_sls.get(ref_unit)
            ref_sl = sel if sel is not None else 2.0
            if ref_sl not in per_sl:
                ref_sl = sorted(per_sl.keys())[0]
            final_label, dis_notes = _disambiguate_archetype(tlab, per_sl[ref_sl], per_sl_trades[ref_sl])
            routing_rows.append({
                "cluster_id": cid,
                "tentative_label": tlab,
                "individual_pass": indiv_pass,
                "aggregate_pass": agg_pass,
                "disposition": disposition,
                "final_archetype_label": final_label,
                "notes": dis_notes,
            })

    # Write top-level outputs.
    summaries_path = out_dir / "archetype_summaries.csv"
    write_archetype_summaries(summaries_path, summary_rows)
    routing_path = out_dir / "cluster_routing.csv"
    write_cluster_routing(routing_path, routing_rows)
    pass_path = out_dir / "capturability_pass_list.csv"
    write_pass_list(pass_path, pass_rows)

    # Hash all outputs.
    for unit_id, _, _, _, _, _, _ in units:
        csv_hashes[f"archetype_{_slug(unit_id)}_sl_sweep.csv"] = _file_sha256(
            out_dir / f"archetype_{_slug(unit_id)}_sl_sweep.csv"
        )
        csv_hashes[f"archetype_{_slug(unit_id)}_distribution.csv"] = _file_sha256(
            out_dir / f"archetype_{_slug(unit_id)}_distribution.csv"
        )
    csv_hashes["archetype_summaries.csv"] = _file_sha256(summaries_path)
    csv_hashes["cluster_routing.csv"] = _file_sha256(routing_path)
    csv_hashes["capturability_pass_list.csv"] = _file_sha256(pass_path)

    ctx = {
        "summary_rows": summary_rows,
        "pass_rows": pass_rows,
        "routing_rows": routing_rows,
        "unit_evals": unit_evals,
        "selected_sls": selected_sls,
        "units": units,
        "out_dir": out_dir,
        "pool_size": pool_size,
    }
    return csv_hashes, ctx


# ============================================================
# Summary doc
# ============================================================


def write_summary(
    out_path: Path,
    cfg: dict,
    ctx: dict,
    hashes_run1: Dict[str, str],
    hashes_run2: Optional[Dict[str, str]],
    det_gate: str,
) -> None:
    pass_rows = ctx["pass_rows"]
    summary_rows = ctx["summary_rows"]
    routing_rows = ctx["routing_rows"]
    units = ctx["units"]
    unit_evals = ctx["unit_evals"]
    selected_sls = ctx["selected_sls"]

    lines: List[str] = []
    lines.append("# Arc 7 — Step 3 capturability characterisation summary")
    lines.append("")
    lines.append("Protocol: `L_ARC_PROTOCOL.md` v2.1.2 §§2, 7, 11, 17")
    lines.append("")
    if pass_rows:
        lines.append("## Verdict")
        lines.append(
            f"**PASS** — {len(pass_rows)} unit(s) pass §2 floors conjunctively; proceed to Step 4."
        )
    else:
        lines.append("## Verdict")
        lines.append("**FAIL — CLEAN-NULL at Step 3.** Zero archetypes pass §2 across all candidate SLs. Arc closes per §7 arc-level gate.")
    lines.append("")

    # Surviving units table
    lines.append("## Surviving units (passing §2 at selected SL)")
    lines.append("")
    if not pass_rows:
        lines.append("_None — arc dies CLEAN-NULL._")
    else:
        lines.append("| unit | type | n | size_frac | archetype | sl | R(atr) | composite | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag |")
        lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in summary_rows:
            if r["gate_status"] != "PASS":
                continue
            lines.append(
                f"| {r['unit_id']} | {r['type']} | {r['n']} | {r['size_fraction']:.4f} "
                f"| {r['final_archetype_label']} | {r['selected_SL_atr_mult']} | {r['selected_R_atr']} "
                f"| {r['composite']:.4f} | {r['mono_pre_peak']:.4f} | {r['frac_reach_1R']:.4f} "
                f"| {r['frac_wrong_way_pre_peak']:.4f} | {r['fwd_mfe_h240_p50']:.4f} | {r['shape_tag']} |"
            )
    lines.append("")

    # SL sweep summary per unit
    lines.append("## SL sweep summary (per cluster + aggregate)")
    lines.append("")
    for unit_id, unit_type, cids, tids, lp_centroid, rule, tlab in units:
        per_sl, _ = unit_evals[unit_id]
        sel = selected_sls[unit_id]
        n = len(tids)
        lines.append(f"### {unit_id} ({unit_type}, n={n}, peaks_centroid={lp_centroid:.2f}, tentative: {tlab})")
        lines.append("")
        lines.append("| SL | mono_pp | reach_1R | wrong_way_pp | fwd_mfe_p50 | shape_tag | size_frac | floors_passed | composite | selected |")
        lines.append("|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|")
        for sl in sorted(per_sl.keys()):
            ev = per_sl[sl]
            floors_passed = sum([ev.gate_size, ev.gate_mono, ev.gate_mfe_p50, ev.gate_reach_1R,
                                 ev.gate_wrong_way, ev.gate_shape, ev.gate_local_peaks])
            sel_mark = "**SEL**" if (sel is not None and abs(sl - sel) < 1e-9) else "—"
            lines.append(
                f"| {sl} | {ev.mono_pre_peak:.4f} | {ev.frac_reach_1R:.4f} | {ev.frac_wrong_way_pre_peak:.4f} "
                f"| {ev.fwd_mfe_p50:.4f} | {ev.shape_tag} | {ev.size_fraction:.4f} | {floors_passed}/7 "
                f"| {ev.composite:.4f} | {sel_mark} |"
            )
        lines.append("")

    # Tentative label disambiguation
    lines.append("## Tentative label disambiguation")
    lines.append("")
    for r in summary_rows:
        lines.append(
            f"- **{r['unit_id']}** — tentative `{r['tentative_label']}` → final `{r['final_archetype_label']}` (pct_peak_and_collapse={r['pct_peak_and_collapse']:.4f})"
        )
    lines.append("")

    # bimodal_separated test per unit (at selected SL or SL=2.0 fallback)
    lines.append("## bimodal_separated test (at selected SL, or SL=2.0 if no SL passed)")
    lines.append("")
    lines.append("| unit | sl_ref | dip stat | p-value | min mode mass | mode separation (R) | result |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|")
    for r in summary_rows:
        unit_id = r["unit_id"]
        per_sl, _ = unit_evals[unit_id]
        sel = selected_sls[unit_id]
        ref_sl = sel if sel is not None else 2.0
        if ref_sl not in per_sl:
            ref_sl = sorted(per_sl.keys())[0]
        ev = per_sl[ref_sl]
        res = "bimodal_separated" if ev.bimodal_separated else "no"
        lines.append(
            f"| {unit_id} | {ref_sl} | {ev.hartigan_dip_stat:.4f} | {ev.hartigan_dip_pvalue:.4g} "
            f"| {ev.bimodal_min_mode_mass:.4f} | {ev.bimodal_mode_separation_r:.4f} | {res} |"
        )
    lines.append("")

    # Routing
    lines.append("## Per-cluster / per-aggregate routing")
    lines.append("")
    lines.append("| cluster | tentative | individual passes | aggregate passes | disposition | final archetype |")
    lines.append("|---:|---|:---:|:---:|---|---|")
    for r in routing_rows:
        ip = "yes" if r["individual_pass"] else "no"
        ap = "yes" if r["aggregate_pass"] else "no"
        lines.append(
            f"| c{r['cluster_id']} | {r['tentative_label']} | {ip} | {ap} | {r['disposition']} | {r['final_archetype_label']} |"
        )
    lines.append("")

    # Distribution detail (per surviving unit) — already in CSVs; reference here
    lines.append("## Distribution detail")
    lines.append("")
    if pass_rows:
        for r in pass_rows:
            unit_id = r["unit_id"]
            lines.append(f"- See `archetype_{_slug(unit_id)}_distribution.csv` for full percentiles, mass-in-band, and bimodal mode info.")
            lines.append(f"- Histograms: `archetype_{_slug(unit_id)}_fwd_mfe_histogram.png`, `archetype_{_slug(unit_id)}_final_r_histogram.png`")
    else:
        lines.append("No surviving units — see per-unit `_sl_sweep.csv` files for full SL sweep evidence.")
    lines.append("")

    # Kill reasons
    lines.append("## Kill reasons (per non-surviving unit)")
    lines.append("")
    failing = [r for r in summary_rows if r["gate_status"] != "PASS"]
    if not failing:
        lines.append("None — all units survive.")
    else:
        lines.append("| unit | n | size_frac | best-composite SL | best mono_pp | best reach_1R | best wrong_way_pp | best fwd_mfe_p50 | failing floors at best SL |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in failing:
            unit_id = r["unit_id"]
            per_sl, _ = unit_evals[unit_id]
            # Best by composite among all SLs.
            best_sl = max(per_sl.keys(), key=lambda s: per_sl[s].composite)
            ev = per_sl[best_sl]
            failing_floors = []
            if not ev.gate_size:
                failing_floors.append("size")
            if not ev.gate_mono:
                failing_floors.append(f"mono({ev.mono_pre_peak:.3f}<0.55)")
            if not ev.gate_mfe_p50:
                failing_floors.append(f"mfe_p50({ev.fwd_mfe_p50:.3f}R<1.5R)")
            if not ev.gate_reach_1R:
                failing_floors.append(f"reach_1R({ev.frac_reach_1R:.3f}<0.70)")
            if not ev.gate_wrong_way:
                failing_floors.append(f"wrong_way({ev.frac_wrong_way_pre_peak:.3f}>0.30)")
            if not ev.gate_shape:
                failing_floors.append(f"shape={ev.shape_tag}")
            if not ev.gate_local_peaks:
                failing_floors.append("local_peaks")
            lines.append(
                f"| {unit_id} | {ev.n} | {ev.size_fraction:.4f} | {best_sl} | {ev.mono_pre_peak:.4f} "
                f"| {ev.frac_reach_1R:.4f} | {ev.frac_wrong_way_pre_peak:.4f} | {ev.fwd_mfe_p50:.4f} "
                f"| {', '.join(failing_floors)} |"
            )
    lines.append("")

    # Determinism
    lines.append("## Determinism")
    lines.append("")
    lines.append(f"**Gate: {det_gate}**")
    lines.append("")
    if hashes_run2 is not None:
        lines.append("| File | run 1 sha256 | run 2 sha256 | match |")
        lines.append("|---|---|---|:---:|")
        for name in sorted(hashes_run1.keys()):
            h1 = hashes_run1[name]
            h2 = hashes_run2.get(name, "(missing)")
            match = "YES" if h1 == h2 else "NO"
            lines.append(f"| `{name}` | `{h1[:16]}…` | `{h2[:16]}…` | {match} |")
    lines.append("")

    # Files
    lines.append("## Files")
    lines.append("")
    out_dir = ctx["out_dir"].relative_to(_REPO_ROOT)
    for name in sorted(hashes_run1.keys()):
        lines.append(f"- `{out_dir}/{name}`")
    lines.append(f"- `{out_dir}/STEP3_SUMMARY.md`")
    lines.append(f"- `{out_dir}/archetype_*_fwd_mfe_histogram.png`")
    lines.append(f"- `{out_dir}/archetype_*_final_r_histogram.png`")
    lines.append("- `configs/arc_7/step3.yaml`")
    lines.append("- `scripts/arc_7/step3_capturability.py`")
    lines.append("")

    # Commit
    lines.append("## Step 3 commit")
    lines.append("hash: _pending_")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Main
# ============================================================


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Arc 7 Step 3 — capturability characterisation.")
    ap.add_argument("-c", "--config", type=Path, default=Path("configs/arc_7/step3.yaml"))
    args = ap.parse_args(argv)
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (_REPO_ROOT / cfg_path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    print("[arc_7 step3] === RUN 1 ===", file=sys.stderr)
    hashes1, ctx = _run_once(cfg)
    hashes2 = None
    if bool(cfg["output"].get("determinism_check", True)):
        print("[arc_7 step3] === RUN 2 (determinism) ===", file=sys.stderr)
        hashes2, _ = _run_once(cfg)
    det_gate = "PASS" if hashes2 and all(hashes1[k] == hashes2.get(k) for k in hashes1) else (
        "PASS" if hashes2 is None else "FAIL"
    )
    if hashes2 is None:
        det_gate = "N/A"

    summary_path = ctx["out_dir"] / cfg["output"]["summary_md"]
    write_summary(summary_path, cfg, ctx, hashes1, hashes2, det_gate)

    n_pass = len(ctx["pass_rows"])
    print(f"[arc_7 step3] DONE. units_passing={n_pass}, det={det_gate}", file=sys.stderr)
    return 0 if n_pass > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
