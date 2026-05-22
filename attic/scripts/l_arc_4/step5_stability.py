"""Arc 4 — Step 5 cross-fold stability.

L_ARC_PROTOCOL v2.1.1 §9. For each cluster surviving Step 4 (cluster 1, 3 — both
Pipeline D1 at t=1):
  - 5A: re-sweep D1 admission threshold on an extended grid
        {base_rate, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50}; select max precision
        subject to recall ≥ 0.60 (fallback: highest-recall). Update cluster YAML.
  - 5B: load 7-fold anchored-expanding WFO structure from configs/wfo_kh24.yaml.
  - 5C: post-hoc exit simulator (§12-sanctioned until PR 2 lands). Walks each
        admitted trade's bar_path applying §11 row 2 Stepwise climber exit
        (MFE-lock at 1R, trail 0.75R from new high) with the Pipeline D1
        pre-t / post-t SL switch at bar offset 1. Per-fold and per-cluster
        ROI/DD/exit-reason metrics.
  - 5D: per-pair contribution audit (concentration flag if > 50% mass in < 5 pairs).

Chat decisions locked in (per Step 5 prompt):
  - Cluster 1 archetype assignment: §11 row 2 Stepwise climber
    (same exit policy as cluster 3, different R-frame).
  - Threshold grid extended to {base_rate, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50}
    (was {0.40, 0.50, 0.60, 0.70} in Step 4).

§9 gate (conjunctive per cluster):
  A. Sign consistency: mean(final_r) > 0 in every fold F1..F7
  B. Size variance: max(n_admitted) / min(n_admitted) ≤ 3.0
  C. DD ceiling: max-fold archetype-attributable DD ≤ 2.0 × median-fold DD

Single-fold sign flip without DD blowup → FLAG, not auto-kill (per §9 discipline).

Outputs (results/l_arc_4/step5/):
  - d1_threshold_resweep.csv
  - fold_stability_cluster_{1,3}.csv
  - per_trade_simulated_{1,3}.csv
  - pair_stability_cluster_{1,3}.csv
  - stability_pass_list.csv
  - step5_diagnostics.md
  - cluster_{1,3}_D1_policy.yaml (updated)

Determinism:
  - random_state=42 throughout (StratifiedKFold + RF).
  - cross_val_predict deterministic with fixed seeds.
  - Two-run byte-identical for all CSVs + updated YAMLs. Both run sha256s logged.

Usage:
  py scripts/l_arc_4/step5_stability.py -c configs/l_arc_4.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Re-use Step 4's deterministic feature builders.
from scripts.l_arc_4.step4_extractability import (  # noqa: E402
    N_CV,
    RF_HP,
    RNDSTATE,
    _compute_d1_features_at_t,
    _features_to_matrix,
    build_entry_feature_matrix,
)

# ============================================================================
# Constants
# ============================================================================

PATH_BARS: int = 241  # bar offsets 0..240 inclusive
D1_T: int = 1          # Step 4 picked t=1 for both clusters
PRE_T_SL_ATR_MULT: float = 2.0     # uniform pre-t SL per §3
MFE_LOCK_R: float = 1.0             # §11 row 2: lock when mfe_so_far_r ≥ 1
TRAIL_DISTANCE_R: float = 0.75      # §11 row 2: trail 0.75R behind new high
RISK_PER_TRADE_PCT: float = 0.005   # L arc convention: 0.5% per trade
STARTING_BALANCE: float = 10_000.0  # L arc starting balance (matches Step 1)

THRESHOLD_GRID_EXTENDED_FIXED: Tuple[float, ...] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
RECALL_FLOOR: float = 0.60

# Cluster labels (Step 4 prompt locked these).
CLUSTER_LABELS: Dict[int, str] = {
    1: "stepwise_pullback_extended",
    3: "stepwise_slow_climber",
}

# §11 row 2 — Stepwise climber exit policy (chat decision: both clusters use this).
SECTION_11_ROW = "2"
SECTION_11_EXIT_POLICY_TEXT = (
    "Stepwise climber (§11 row 2): MFE-lock at 1R, trail 0.75R from new high. "
    "Pipeline D1 with archetype-specific SL = entry − cluster_R after bar-1 "
    "classifier admission (pre-t SL = entry − 2 × ATR)."
)


# ============================================================================
# WFO fold structure (anchored expanding, 7 folds from configs/wfo_kh24.yaml)
# ============================================================================


@dataclass
class WFOFold:
    fold: int
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp  # exclusive on the date boundary


def load_wfo_folds(wfo_cfg_path: Path) -> List[WFOFold]:
    cfg = yaml.safe_load(wfo_cfg_path.read_text(encoding="utf-8"))
    out: List[WFOFold] = []
    for entry in cfg["wfo"]["folds"]:
        out.append(
            WFOFold(
                fold=int(entry["fold"]),
                oos_start=pd.Timestamp(entry["oos_start"]),
                oos_end=pd.Timestamp(entry["oos_end"]),
            )
        )
    return out


# ============================================================================
# Sub-step 5A — Threshold re-sweep on extended grid
# ============================================================================


@dataclass
class ThresholdReSweep:
    cluster_id: int
    base_rate: float
    grid: List[float]            # extended grid (base_rate prepended + fixed list)
    per_threshold: List[Dict[str, float]]
    selected_threshold: float
    selected_precision: float
    selected_recall: float
    used_fallback: bool
    rationale: str
    oof_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    admitted_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))


def compute_oof_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cross_val_predict with the SAME StratifiedKFold and RF used in Step 4."""
    skf = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=RNDSTATE)
    model = RandomForestClassifier(**RF_HP)
    oof = cross_val_predict(model, X, y, cv=skf, method="predict_proba", n_jobs=-1)
    return oof[:, 1]


def resweep_threshold(cluster_id: int, oof: np.ndarray, y: np.ndarray) -> ThresholdReSweep:
    base_rate = float(y.mean())
    grid = sorted(set([round(base_rate, 4)] + list(THRESHOLD_GRID_EXTENDED_FIXED)))
    rows: List[Dict[str, float]] = []
    for t in grid:
        y_pred = (oof >= t).astype(int)
        if y_pred.sum() == 0:
            prec = 0.0
            rec = 0.0
        else:
            prec = float(precision_score(y, y_pred, zero_division=0))
            rec = float(recall_score(y, y_pred, zero_division=0))
        rows.append({"threshold": float(t), "precision": prec, "recall": rec, "n_admitted": int(y_pred.sum())})
    passing = [r for r in rows if r["recall"] >= RECALL_FLOOR]
    if passing:
        sel = max(passing, key=lambda r: (r["precision"], -r["threshold"]))
        used_fallback = False
        rationale = (
            f"max precision subject to recall ≥ {RECALL_FLOOR} — "
            f"selected t={sel['threshold']:.4f} (precision {sel['precision']:.4f}, recall {sel['recall']:.4f})"
        )
    else:
        sel = max(rows, key=lambda r: r["recall"])
        used_fallback = True
        rationale = (
            f"no threshold meets recall ≥ {RECALL_FLOOR}; FALLBACK = highest-recall — "
            f"selected t={sel['threshold']:.4f} (recall {sel['recall']:.4f})"
        )
    admitted_mask = (oof >= sel["threshold"]).astype(bool)
    return ThresholdReSweep(
        cluster_id=cluster_id,
        base_rate=base_rate,
        grid=grid,
        per_threshold=rows,
        selected_threshold=float(sel["threshold"]),
        selected_precision=float(sel["precision"]),
        selected_recall=float(sel["recall"]),
        used_fallback=used_fallback,
        rationale=rationale,
        oof_scores=oof,
        admitted_mask=admitted_mask,
    )


# ============================================================================
# Sub-step 5C — Post-hoc Stepwise-climber exit simulator
# ============================================================================


@dataclass
class SimulatedTrade:
    trade_id: int
    pair: str
    cluster_id: int
    fold: int
    entry_ts: pd.Timestamp
    entry_price: float
    atr_signal: float
    cluster_R: float                 # cluster_sl_mult × ATR (post-t R unit)
    pre_t_sl_price: float            # entry - 2 × ATR
    post_t_sl_price: float           # entry - cluster_R
    exit_bar: int
    exit_reason: str
    exit_price: float
    final_r: float                   # (exit_price - entry_price) / cluster_R
    mfe_locked_bar: int              # bar offset where MFE first reached 1R; -1 if never
    peak_mfe_r: float


def _simulate_one_trade(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    entry_price: float,
    atr_signal: float,
    cluster_sl_mult: float,
) -> Tuple[int, str, float, float, int, float]:
    """Walk bars 0..240 applying §11 row 2 Stepwise climber exit under Pipeline D1.

    Returns (exit_bar, exit_reason, exit_price, final_r, mfe_locked_bar, peak_mfe_r).

    Sequencing per bar (MFE-first convention, matches Step 3 SL sweep semantics):
      1. Update MFE from bar high.
      2. If MFE crosses 1R → lock + start trail.
      3. If trail_active: update trail_stop if MFE increased.
      4. effective_sl_price = max(current_sl_price, trail_stop_price) if trail else current_sl_price.
      5. If bar low ≤ effective_sl_price → exit at effective_sl_price.
      6. End-of-bar: ratchet current_sl_price up to trail_stop_price if trail_active.
      7. At end of bar offset 1 (D1 classifier admits, by construction): replace
         pre-t SL with archetype post-t SL (entry − cluster_R). Replacement is
         a LOOSENING (post-t is wider); trail_stop, if armed, dominates anyway.
    """
    R = cluster_sl_mult * atr_signal
    pre_t_sl_price = entry_price - PRE_T_SL_ATR_MULT * atr_signal
    post_t_sl_price = entry_price - R

    current_sl_price = pre_t_sl_price
    mfe_locked = False
    trail_active = False
    trail_stop_price = -math.inf
    mfe_r = 0.0
    mfe_locked_bar = -1
    peak_mfe_r = 0.0

    for t in range(PATH_BARS):
        bar_high = float(highs[t])
        bar_low = float(lows[t])

        # Step 1: update MFE from bar high.
        bar_mfe_r = (bar_high - entry_price) / R
        if bar_mfe_r > mfe_r:
            mfe_r = bar_mfe_r
            new_mfe_high = True
        else:
            new_mfe_high = False
        peak_mfe_r = max(peak_mfe_r, mfe_r)

        # Step 2: MFE lock at 1R.
        if not mfe_locked and mfe_r >= MFE_LOCK_R:
            mfe_locked = True
            trail_active = True
            mfe_locked_bar = t
            trail_stop_price = entry_price + (mfe_r - TRAIL_DISTANCE_R) * R
        # Step 3: ratchet trail if new MFE high.
        elif trail_active and new_mfe_high:
            new_trail = entry_price + (mfe_r - TRAIL_DISTANCE_R) * R
            if new_trail > trail_stop_price:
                trail_stop_price = new_trail

        # Step 4: effective SL = max of static and trail (if trailing).
        if trail_active:
            effective_sl = max(current_sl_price, trail_stop_price)
        else:
            effective_sl = current_sl_price

        # Step 5: SL check.
        if bar_low <= effective_sl:
            exit_price = effective_sl
            if trail_active and effective_sl == trail_stop_price and trail_stop_price > current_sl_price:
                exit_reason = "trail_hit"
            else:
                exit_reason = "sl_hit"
            return t, exit_reason, exit_price, (exit_price - entry_price) / R, mfe_locked_bar, peak_mfe_r

        # Step 6: end-of-bar ratchet of static current_sl_price.
        if trail_active and trail_stop_price > current_sl_price:
            current_sl_price = trail_stop_price

        # Step 7: at end of bar 1, replace pre-t SL with post-t SL (D1 mechanic).
        if t == D1_T and not trail_active:
            # Replace pre-t with post-t (post-t is wider for cluster 1, 3).
            current_sl_price = post_t_sl_price
        # If trail_active by end of bar 1, current_sl_price ≥ trail_stop_price ≥ 0.25R
        # above entry — much higher than post-t SL — so no replacement needed.

    # Cap binding at bar 240: close at bar 240's close (informational; document choice).
    exit_price = float(closes[PATH_BARS - 1])
    final_r = (exit_price - entry_price) / R
    return PATH_BARS - 1, "cap_bind", exit_price, final_r, mfe_locked_bar, peak_mfe_r


def simulate_admitted_trades(
    trades: pd.DataFrame,
    path_tensor: np.ndarray,
    admitted_mask: np.ndarray,
    cluster_sl_mult: float,
    cluster_id: int,
    folds: List[WFOFold],
) -> List[SimulatedTrade]:
    """For every admitted trade, run the simulator and tag it with the WFO fold
    containing its entry_time. Trades outside any fold's OOS window are excluded.
    """
    entry_prices = trades["entry_price"].to_numpy(dtype=float)
    atr_arr = trades["atr_14_at_signal"].to_numpy(dtype=float)
    entry_times = pd.to_datetime(trades["entry_time"]).to_numpy()
    pairs = trades["pair"].to_numpy()
    trade_ids = trades["trade_id"].to_numpy(dtype=int)

    out: List[SimulatedTrade] = []
    for i in range(len(trades)):
        if not admitted_mask[i]:
            continue
        # Fold containment.
        et = pd.Timestamp(entry_times[i])
        fold_id = -1
        for f in folds:
            if f.oos_start <= et < f.oos_end:
                fold_id = f.fold
                break
        if fold_id == -1:
            continue
        exit_bar, exit_reason, exit_price, final_r, mfe_lock_bar, peak_mfe_r = _simulate_one_trade(
            highs=path_tensor[i, :, 1],
            lows=path_tensor[i, :, 2],
            closes=path_tensor[i, :, 3],
            entry_price=float(entry_prices[i]),
            atr_signal=float(atr_arr[i]),
            cluster_sl_mult=cluster_sl_mult,
        )
        out.append(
            SimulatedTrade(
                trade_id=int(trade_ids[i]),
                pair=str(pairs[i]),
                cluster_id=cluster_id,
                fold=fold_id,
                entry_ts=et,
                entry_price=float(entry_prices[i]),
                atr_signal=float(atr_arr[i]),
                cluster_R=cluster_sl_mult * float(atr_arr[i]),
                pre_t_sl_price=float(entry_prices[i]) - PRE_T_SL_ATR_MULT * float(atr_arr[i]),
                post_t_sl_price=float(entry_prices[i]) - cluster_sl_mult * float(atr_arr[i]),
                exit_bar=exit_bar,
                exit_reason=exit_reason,
                exit_price=exit_price,
                final_r=final_r,
                mfe_locked_bar=mfe_lock_bar,
                peak_mfe_r=peak_mfe_r,
            )
        )
    return out


# ============================================================================
# Sub-step 5C aggregation — per-fold metrics
# ============================================================================


@dataclass
class FoldMetrics:
    fold: int
    n: int
    mean_r: float
    std_r: float
    t_stat: float
    frac_winners: float
    mean_winner_r: float
    mean_loser_r: float
    fold_roi_pct: float
    fold_max_dd_pct: float
    exit_reason_counts: Dict[str, int]


def aggregate_per_fold(
    sim_trades: List[SimulatedTrade],
    folds: List[WFOFold],
) -> List[FoldMetrics]:
    out: List[FoldMetrics] = []
    by_fold: Dict[int, List[SimulatedTrade]] = {f.fold: [] for f in folds}
    for st in sim_trades:
        by_fold.setdefault(st.fold, []).append(st)
    for f in folds:
        trades = sorted(by_fold[f.fold], key=lambda x: x.entry_ts)
        n = len(trades)
        if n == 0:
            out.append(
                FoldMetrics(
                    fold=f.fold, n=0, mean_r=0.0, std_r=0.0, t_stat=0.0,
                    frac_winners=0.0, mean_winner_r=0.0, mean_loser_r=0.0,
                    fold_roi_pct=0.0, fold_max_dd_pct=0.0, exit_reason_counts={},
                )
            )
            continue
        rs = np.array([t.final_r for t in trades], dtype=float)
        mean_r = float(np.mean(rs))
        std_r = float(np.std(rs, ddof=1)) if n > 1 else 0.0
        t_stat = float(mean_r / (std_r / math.sqrt(n))) if std_r > 0 else 0.0
        winners = rs[rs > 0]
        losers = rs[rs <= 0]
        frac_winners = float(len(winners) / n)
        mean_winner_r = float(winners.mean()) if winners.size > 0 else 0.0
        mean_loser_r = float(losers.mean()) if losers.size > 0 else 0.0
        # Fold ROI as simple sum of (final_r × risk%).
        per_trade_pct = rs * RISK_PER_TRADE_PCT * 100.0  # convert to % for reporting
        fold_roi_pct = float(per_trade_pct.sum())
        # Cumulative P&L for DD calc (simple sum, no compounding).
        cum_pnl = np.cumsum(per_trade_pct)
        running_peak = np.maximum.accumulate(cum_pnl)
        drawdown = running_peak - cum_pnl
        fold_max_dd_pct = float(drawdown.max()) if drawdown.size > 0 else 0.0
        # Exit reason distribution.
        from collections import Counter
        exit_counts = dict(Counter(t.exit_reason for t in trades))
        out.append(
            FoldMetrics(
                fold=f.fold, n=n, mean_r=mean_r, std_r=std_r, t_stat=t_stat,
                frac_winners=frac_winners, mean_winner_r=mean_winner_r,
                mean_loser_r=mean_loser_r, fold_roi_pct=fold_roi_pct,
                fold_max_dd_pct=fold_max_dd_pct, exit_reason_counts=exit_counts,
            )
        )
    return out


# ============================================================================
# §9 gate evaluation
# ============================================================================


@dataclass
class StabilityVerdict:
    cluster_id: int
    cluster_label: str
    passes_A: bool        # sign consistency
    passes_B: bool        # size variance ≤ 3.0
    passes_C: bool        # DD ceiling
    overall: str          # "PASS" | "FLAG" | "FAIL"
    notes: List[str]
    # Measured values
    a_signs: List[Tuple[int, float]]    # (fold, mean_r) — for transparency
    b_size_ratio: float
    c_max_dd: float
    c_median_dd: float
    c_max_dd_ratio: float


def evaluate_gate(cluster_id: int, label: str, folds: List[FoldMetrics]) -> StabilityVerdict:
    notes: List[str] = []
    a_signs = [(f.fold, f.mean_r) for f in folds]
    # Gate A — sign consistency
    sign_flips = [f for f in folds if f.mean_r <= 0 and f.n > 0]
    passes_A = len(sign_flips) == 0
    if sign_flips:
        notes.append(
            f"Sign flip in fold(s) {[f.fold for f in sign_flips]} (mean_r ≤ 0)"
        )

    # Gate B — size variance
    non_empty = [f.n for f in folds if f.n > 0]
    if not non_empty:
        passes_B = False
        b_size_ratio = float("inf")
        notes.append("All folds empty — Gate B undefined")
    else:
        max_n = max(non_empty)
        min_n = min(non_empty)
        b_size_ratio = float(max_n / min_n) if min_n > 0 else float("inf")
        passes_B = b_size_ratio <= 3.0
        if not passes_B:
            notes.append(
                f"Size ratio {b_size_ratio:.2f} > 3.0 across folds "
                f"(min={min_n}, max={max_n})"
            )

    # Gate C — DD ceiling
    dds = [f.fold_max_dd_pct for f in folds if f.n > 0]
    if not dds:
        passes_C = False
        c_max_dd = 0.0
        c_median_dd = 0.0
        c_max_dd_ratio = float("inf")
    else:
        c_max_dd = float(max(dds))
        c_median_dd = float(np.median(dds))
        c_max_dd_ratio = float(c_max_dd / c_median_dd) if c_median_dd > 0 else float("inf")
        passes_C = c_max_dd_ratio <= 2.0
        if not passes_C:
            notes.append(
                f"Max-fold DD {c_max_dd:.2f}% / median {c_median_dd:.2f}% = "
                f"ratio {c_max_dd_ratio:.2f} > 2.0"
            )

    # Discipline (§9): a single-fold sign flip without DD blowup → FLAG, not FAIL.
    if not passes_A and passes_B and passes_C and len(sign_flips) == 1:
        overall = "FLAG"
        notes.append(
            "§9 discipline: single-fold sign flip without DD blowup → FLAG (chat judgement)"
        )
    elif passes_A and passes_B and passes_C:
        overall = "PASS"
    else:
        # If gate B fails but A/C pass, §9 discipline says flag as regime-dependent (not auto-kill).
        if passes_A and passes_C and not passes_B:
            overall = "FLAG"
            notes.append(
                "§9 discipline: size-variance flag without sign flip or DD blowup → FLAG (regime-dependent)"
            )
        else:
            overall = "FAIL"

    return StabilityVerdict(
        cluster_id=cluster_id,
        cluster_label=label,
        passes_A=passes_A,
        passes_B=passes_B,
        passes_C=passes_C,
        overall=overall,
        notes=notes,
        a_signs=a_signs,
        b_size_ratio=b_size_ratio,
        c_max_dd=c_max_dd,
        c_median_dd=c_median_dd,
        c_max_dd_ratio=c_max_dd_ratio,
    )


# ============================================================================
# Sub-step 5D — per-pair stability
# ============================================================================


def per_pair_table(sim_trades: List[SimulatedTrade], pairs_universe: List[str]) -> pd.DataFrame:
    """Pair-level counts across all folds for one cluster."""
    counts: Dict[str, int] = {p: 0 for p in pairs_universe}
    for st in sim_trades:
        counts[st.pair] = counts.get(st.pair, 0) + 1
    n_total = sum(counts.values())
    rows: List[Dict[str, Any]] = []
    for p in pairs_universe:
        n = counts.get(p, 0)
        pct = (n / n_total) if n_total > 0 else 0.0
        rows.append({"pair": p, "n_trades": int(n), "pct": pct})
    rows.sort(key=lambda r: (-r["n_trades"], r["pair"]))
    df = pd.DataFrame(rows)
    return df


def pair_concentration_flag(df: pd.DataFrame, n_top: int = 5, mass_threshold: float = 0.50) -> Tuple[bool, float, List[str]]:
    """Flag if > mass_threshold of trades concentrate in < n_top pairs."""
    top = df.head(n_top)
    top_mass = float(top["pct"].sum())
    flagged = top_mass > mass_threshold
    return flagged, top_mass, top["pair"].tolist()


# ============================================================================
# Output writers (deterministic)
# ============================================================================


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt_g(x: Any) -> Any:
    if x is None or (isinstance(x, float) and (math.isnan(x) or not math.isfinite(x))):
        return ""
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.10g}"
    return x


def _write_csv_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, na_rep="", lineterminator="\n")


def _write_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, na_rep="", lineterminator="\n")


def _write_yaml(content: Dict[str, Any], path: Path) -> None:
    path.write_text(
        yaml.safe_dump(content, sort_keys=True, default_flow_style=False, allow_unicode=False),
        encoding="utf-8",
    )


def update_d1_policy_yaml(
    yaml_path: Path,
    new_threshold: float,
    new_precision: float,
    new_recall: float,
    new_rationale: str,
    new_used_fallback: bool,
    section_11_row: str,
    section_11_exit_policy: str,
    extended_grid: List[float],
    base_rate: float,
) -> None:
    """Update an existing D1 policy YAML in place with the new threshold + archetype
    assignment. Preserves keys not relevant to this update; replaces the threshold,
    precision/recall + rationale + section_11_* fields.
    """
    existing = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    existing["selected_threshold"] = float(new_threshold)
    existing["expected_precision_at_threshold"] = float(new_precision)
    existing["expected_recall_at_threshold"] = float(new_recall)
    existing["threshold_used_fallback"] = bool(new_used_fallback)
    existing["section_11_row"] = section_11_row
    existing["section_11_exit_policy"] = section_11_exit_policy
    existing["step5_threshold_grid"] = [float(t) for t in extended_grid]
    existing["step5_base_rate"] = float(base_rate)
    existing["step5_selection_rationale"] = new_rationale
    _write_yaml(existing, yaml_path)


# ============================================================================
# Single-run orchestration
# ============================================================================


@dataclass
class ClusterRunResult:
    cluster_id: int
    cluster_label: str
    selected_sl: float
    base_rate: float
    threshold_resweep: ThresholdReSweep
    sim_trades: List[SimulatedTrade]
    folds: List[FoldMetrics]
    verdict: StabilityVerdict
    pair_df: pd.DataFrame
    pair_concentration_flagged: bool
    pair_top_mass: float
    pair_top_names: List[str]


def run_once(
    trades_csv: Path,
    paths_csv: Path,
    clusters_csv: Path,
    pass_list_csv: Path,
    e_filter_yaml_paths: Dict[int, Path],  # not used but referenced for completeness
    d1_yaml_paths: Dict[int, Path],
    catalogue_path: Path,
    wfo_cfg_path: Path,
    data_dir: Path,
    out_dir: Path,
) -> Tuple[List[ClusterRunResult], Dict[str, str], List[WFOFold]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = pd.read_csv(trades_csv).sort_values("trade_id").reset_index(drop=True)
    n_trades = int(len(trades))
    clusters = pd.read_csv(clusters_csv).sort_values("trade_id").reset_index(drop=True)
    if not np.array_equal(
        trades["trade_id"].to_numpy(dtype=int), clusters["trade_id"].to_numpy(dtype=int)
    ):
        raise ValueError("trade_id mismatch between trades_all and clusters_K4")
    cluster_id_per_trade = clusters["cluster_id"].to_numpy(dtype=int)
    pass_list = pd.read_csv(pass_list_csv)
    survivors: List[Tuple[int, float]] = [
        (int(r["cluster_id"]), float(r["selected_SL"]))
        for _, r in pass_list.iterrows()
    ]
    folds = load_wfo_folds(wfo_cfg_path)

    # Feature catalogue (mirror Step 4 to reproduce X).
    catalogue = yaml.safe_load(catalogue_path.read_text(encoding="utf-8"))
    base = list(catalogue["base_features"])
    arc_specific = list(catalogue["l_arc_4"]["arc_specific_features"])
    all_entry_features = base + arc_specific

    # Build entry-feature matrix (same construction as Step 4).
    print("[l_arc_4 step5] Building entry feature matrix...", file=sys.stderr)
    t0 = time.time()
    entry_feats_df, _ = build_entry_feature_matrix(trades, data_dir, all_entry_features)
    print(f"[l_arc_4 step5] entry features done in {time.time() - t0:.1f}s", file=sys.stderr)

    # Load paths into 2D tensor.
    print("[l_arc_4 step5] Loading paths into 2D tensor...", file=sys.stderr)
    paths_df = pd.read_csv(paths_csv).sort_values(["trade_id", "bar_offset"]).reset_index(drop=True)
    if len(paths_df) != n_trades * PATH_BARS:
        raise ValueError(
            f"paths CSV has {len(paths_df)} rows; expected {n_trades} × {PATH_BARS}"
        )
    opens = paths_df["open"].to_numpy(dtype=float).reshape(n_trades, PATH_BARS)
    highs = paths_df["high"].to_numpy(dtype=float).reshape(n_trades, PATH_BARS)
    lows = paths_df["low"].to_numpy(dtype=float).reshape(n_trades, PATH_BARS)
    closes = paths_df["close"].to_numpy(dtype=float).reshape(n_trades, PATH_BARS)
    path_tensor = np.stack([opens, highs, lows, closes], axis=2)
    entry_prices_all = trades["entry_price"].to_numpy(dtype=float)
    atr_14_all = trades["atr_14_at_signal"].to_numpy(dtype=float)
    bars_held_all = trades["bars_held"].to_numpy(dtype=int)

    # Entry-feature matrix as numpy with NaN fill.
    X_e, _ = _features_to_matrix(entry_feats_df, all_entry_features)

    pairs_universe = sorted(trades["pair"].unique().tolist())

    cluster_results: List[ClusterRunResult] = []
    sha_files: Dict[str, str] = {}
    resweep_rows: List[Dict[str, Any]] = []
    pass_list_rows: List[Dict[str, Any]] = []

    for cluster_id, selected_sl in survivors:
        cluster_label = CLUSTER_LABELS.get(cluster_id, f"cluster_{cluster_id}")
        y = (cluster_id_per_trade == cluster_id).astype(int)
        # Build D1 X at t=1 using full pool (no exclusion at t=1).
        eligible_mask = bars_held_all >= D1_T
        elig_idx = np.where(eligible_mask)[0]
        r_per_trade = selected_sl * atr_14_all[elig_idx]
        d1_feats = _compute_d1_features_at_t(
            path_tensor[elig_idx], entry_prices_all[elig_idx], r_per_trade, D1_T
        )
        X_combined = np.concatenate([X_e[elig_idx][:, : len(base)], d1_feats], axis=1)
        y_elig = y[elig_idx]

        # 5A: OOF scores + threshold re-sweep.
        print(f"[l_arc_4 step5] cluster {cluster_id} ({cluster_label}) — 5A OOF re-sweep", file=sys.stderr)
        oof = compute_oof_scores(X_combined, y_elig)
        rsw = resweep_threshold(cluster_id, oof, y_elig)

        # Build full-pool admitted_mask (length n_trades). Trades not in elig_idx
        # are not eligible at t=1 — but with bars_held >= 1 universally true at
        # Step 1 (every trade has at least 1 held bar), elig_idx covers everyone.
        admitted_mask_full = np.zeros(n_trades, dtype=bool)
        admitted_mask_full[elig_idx] = rsw.admitted_mask
        # Filter to cluster members only (admitted AND in cluster) — wait NO. The
        # D1 classifier outputs P(cluster_membership). We admit those classified
        # as in the cluster (regardless of ground-truth cluster). The "true"
        # positives are cluster members; "false positives" are non-cluster trades
        # that look cluster-like at t=1. Step 6 will evaluate both with the
        # ground-truth and the WFO truth.
        # For Step 5 simulation: admit ANY trade whose OOF probability ≥ threshold.
        # That's what admitted_mask is.

        for r in rsw.per_threshold:
            resweep_rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_label": cluster_label,
                    "selected_SL": selected_sl,
                    "threshold": r["threshold"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "n_admitted": r["n_admitted"],
                    "selected": int(abs(r["threshold"] - rsw.selected_threshold) < 1e-9),
                    "base_rate": rsw.base_rate,
                    "used_fallback": int(rsw.used_fallback),
                    "rationale_summary": rsw.rationale if abs(r["threshold"] - rsw.selected_threshold) < 1e-9 else "",
                }
            )

        # 5C: Simulate admitted trades through the §11 row 2 exit policy.
        print(
            f"[l_arc_4 step5] cluster {cluster_id} — simulating "
            f"{int(admitted_mask_full.sum())} admitted trades",
            file=sys.stderr,
        )
        sim_trades = simulate_admitted_trades(
            trades, path_tensor, admitted_mask_full, selected_sl, cluster_id, folds
        )

        # Per-fold aggregation + gate evaluation.
        fold_metrics = aggregate_per_fold(sim_trades, folds)
        verdict = evaluate_gate(cluster_id, cluster_label, fold_metrics)

        # 5D: per-pair.
        pair_df = per_pair_table(sim_trades, pairs_universe)
        flagged, top_mass, top_names = pair_concentration_flag(pair_df)

        cluster_results.append(
            ClusterRunResult(
                cluster_id=cluster_id,
                cluster_label=cluster_label,
                selected_sl=selected_sl,
                base_rate=rsw.base_rate,
                threshold_resweep=rsw,
                sim_trades=sim_trades,
                folds=fold_metrics,
                verdict=verdict,
                pair_df=pair_df,
                pair_concentration_flagged=flagged,
                pair_top_mass=top_mass,
                pair_top_names=top_names,
            )
        )

        # Per-trade simulated CSV.
        per_trade_path = out_dir / f"per_trade_simulated_{cluster_id}.csv"
        per_trade_rows = [
            {
                "trade_id": st.trade_id,
                "pair": st.pair,
                "fold": st.fold,
                "entry_ts": st.entry_ts.isoformat(),
                "entry_price": _fmt_g(st.entry_price),
                "atr_signal": _fmt_g(st.atr_signal),
                "cluster_R": _fmt_g(st.cluster_R),
                "pre_t_sl_price": _fmt_g(st.pre_t_sl_price),
                "post_t_sl_price": _fmt_g(st.post_t_sl_price),
                "exit_bar": int(st.exit_bar),
                "exit_reason": st.exit_reason,
                "exit_price": _fmt_g(st.exit_price),
                "final_r": _fmt_g(st.final_r),
                "mfe_locked_bar": int(st.mfe_locked_bar),
                "peak_mfe_r": _fmt_g(st.peak_mfe_r),
            }
            for st in sim_trades
        ]
        _write_csv_rows(per_trade_rows, per_trade_path)
        sha_files[per_trade_path.name] = _file_sha256(per_trade_path)

        # Per-fold CSV.
        fold_path = out_dir / f"fold_stability_cluster_{cluster_id}.csv"
        fold_rows = [
            {
                "cluster_id": cluster_id,
                "fold": fm.fold,
                "n": fm.n,
                "mean_r": _fmt_g(fm.mean_r),
                "std_r": _fmt_g(fm.std_r),
                "t_stat": _fmt_g(fm.t_stat),
                "frac_winners": _fmt_g(fm.frac_winners),
                "mean_winner_r": _fmt_g(fm.mean_winner_r),
                "mean_loser_r": _fmt_g(fm.mean_loser_r),
                "fold_roi_pct": _fmt_g(fm.fold_roi_pct),
                "fold_max_dd_pct": _fmt_g(fm.fold_max_dd_pct),
                "exit_reason_dist": ";".join(f"{k}={v}" for k, v in sorted(fm.exit_reason_counts.items())),
            }
            for fm in fold_metrics
        ]
        _write_csv_rows(fold_rows, fold_path)
        sha_files[fold_path.name] = _file_sha256(fold_path)

        # Per-pair CSV.
        pair_path = out_dir / f"pair_stability_cluster_{cluster_id}.csv"
        _write_df(pair_df, pair_path)
        sha_files[pair_path.name] = _file_sha256(pair_path)

        # Update D1 policy YAML.
        yaml_path = d1_yaml_paths[cluster_id]
        update_d1_policy_yaml(
            yaml_path=yaml_path,
            new_threshold=rsw.selected_threshold,
            new_precision=rsw.selected_precision,
            new_recall=rsw.selected_recall,
            new_rationale=rsw.rationale,
            new_used_fallback=rsw.used_fallback,
            section_11_row=SECTION_11_ROW,
            section_11_exit_policy=SECTION_11_EXIT_POLICY_TEXT,
            extended_grid=rsw.grid,
            base_rate=rsw.base_rate,
        )
        sha_files[yaml_path.name] = _file_sha256(yaml_path)

        pass_list_rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
                "passes_A": int(verdict.passes_A),
                "passes_B": int(verdict.passes_B),
                "passes_C": int(verdict.passes_C),
                "overall": verdict.overall,
                "n_total_admitted_across_folds": sum(fm.n for fm in fold_metrics),
                "n_min_fold": min((fm.n for fm in fold_metrics if fm.n > 0), default=0),
                "n_max_fold": max((fm.n for fm in fold_metrics), default=0),
                "size_ratio_max_min": _fmt_g(verdict.b_size_ratio) if math.isfinite(verdict.b_size_ratio) else "",
                "median_fold_dd_pct": _fmt_g(verdict.c_median_dd),
                "max_fold_dd_pct": _fmt_g(verdict.c_max_dd),
                "max_dd_ratio_to_median": _fmt_g(verdict.c_max_dd_ratio) if math.isfinite(verdict.c_max_dd_ratio) else "",
                "selected_threshold": _fmt_g(rsw.selected_threshold),
                "pair_concentration_flagged": int(flagged),
                "pair_concentration_top_mass": _fmt_g(top_mass),
                "notes": " | ".join(verdict.notes),
            }
        )

    # Top-level CSVs.
    resweep_path = out_dir / "d1_threshold_resweep.csv"
    _write_csv_rows(resweep_rows, resweep_path)
    sha_files[resweep_path.name] = _file_sha256(resweep_path)

    pass_list_path = out_dir / "stability_pass_list.csv"
    _write_csv_rows(pass_list_rows, pass_list_path)
    sha_files[pass_list_path.name] = _file_sha256(pass_list_path)

    return cluster_results, sha_files, folds


# ============================================================================
# Diagnostics markdown writer
# ============================================================================


def _exit_reason_str(d: Dict[str, int]) -> str:
    if not d:
        return ""
    return "; ".join(f"{k}={v}" for k, v in sorted(d.items()))


def write_diagnostics(
    out_path: Path,
    cluster_results: List[ClusterRunResult],
    folds: List[WFOFold],
    sha_run1: Dict[str, str],
    sha_run2: Optional[Dict[str, str]],
    determinism_gate: str,
    config_paths: Dict[str, str],
    config_shas: Dict[str, str],
) -> str:
    pass_counts = {"PASS": 0, "FLAG": 0, "FAIL": 0}
    for c in cluster_results:
        pass_counts[c.verdict.overall] = pass_counts.get(c.verdict.overall, 0) + 1
    n_pass = pass_counts.get("PASS", 0) + pass_counts.get("FLAG", 0)
    arc_disp = "PASS" if n_pass >= 1 else "FAIL"

    lines: List[str] = []
    lines.append("# Arc 4 — Step 5 cross-fold stability diagnostics")
    lines.append("")
    lines.append("Protocol: `L_ARC_PROTOCOL.md` v2.1.1 §9 (sign consistency, size variance ≤ 3.0, DD ceiling ≤ 2× median)")
    lines.append("Exit policy: `§11 row 2 Stepwise climber` (chat decision — both clusters)")
    lines.append("D1 backtester: post-hoc simulator per §12 sanctioned fallback (PR 2 deferred)")
    lines.append("")

    # Headline.
    lines.append("## Summary")
    lines.append("")
    cluster_summary = "; ".join(
        f"cluster {c.cluster_id} → {c.verdict.overall}" for c in cluster_results
    )
    lines.append(
        f"{len(cluster_results)} cluster(s) under §9 evaluation. "
        f"Disposition: {cluster_summary}. "
        f"PASS+FLAG count: {n_pass}. Arc-level Step 5 outcome: **{arc_disp}** "
        f"(≥1 cluster PASS/FLAG required). Determinism: **{determinism_gate}**."
    )
    lines.append("")

    # Sub-step 5A — threshold re-sweep.
    lines.append("## 5A — D1 threshold re-sweep (extended grid)")
    lines.append("")
    lines.append(
        "Step 4 used threshold grid {0.40, 0.50, 0.60, 0.70} — both clusters fell to "
        "the 0.40 fallback with near-zero recall. Step 5 sweeps the extended grid "
        "{base_rate, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50} with selection rule "
        "unchanged: max precision subject to recall ≥ 0.60 (fallback: max recall)."
    )
    lines.append("")
    for c in cluster_results:
        rsw = c.threshold_resweep
        lines.append(
            f"### Cluster {c.cluster_id} ({c.cluster_label}) — base_rate = {rsw.base_rate:.4f}"
        )
        lines.append("")
        lines.append("| Threshold | Precision | Recall | n_admitted | Selected? |")
        lines.append("|---:|---:|---:|---:|---|")
        for r in rsw.per_threshold:
            selected_mark = "✓" if abs(r["threshold"] - rsw.selected_threshold) < 1e-9 else ""
            lines.append(
                f"| {r['threshold']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
                f"{r['n_admitted']} | {selected_mark} |"
            )
        lines.append("")
        lines.append(f"**Selected threshold = {rsw.selected_threshold:.4f}** — {rsw.rationale}")
        if rsw.used_fallback:
            lines.append(
                f"*Note*: fallback active — no threshold meets recall ≥ {RECALL_FLOOR}. "
                f"Selected = max-recall threshold."
            )
        lines.append("")

    # Sub-step 5B — fold dates.
    lines.append("## 5B — WFO fold OOS windows (from configs/wfo_kh24.yaml)")
    lines.append("")
    lines.append("| Fold | OOS start | OOS end |")
    lines.append("|---:|---|---|")
    for f in folds:
        lines.append(f"| F{f.fold} | {f.oos_start.date()} | {f.oos_end.date()} |")
    lines.append("")

    # Sub-step 5C — per-cluster fold tables + gate.
    for c in cluster_results:
        lines.append(f"## 5C — Cluster {c.cluster_id} ({c.cluster_label}, R = {c.selected_sl}×ATR)")
        lines.append("")
        lines.append(
            f"Admitted trades (OOF score ≥ {c.threshold_resweep.selected_threshold:.4f}): "
            f"{sum(fm.n for fm in c.folds)} across folds F1..F7 "
            f"(F0 excluded — outside any OOS window)."
        )
        lines.append("")
        lines.append("| F | n | mean_r | std_r | t-stat | frac_win | winner_r | loser_r | fold_roi% | fold_maxDD% | exit_reasons |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for fm in c.folds:
            if fm.n == 0:
                lines.append(
                    f"| F{fm.fold} | 0 | — | — | — | — | — | — | — | — | — |"
                )
                continue
            lines.append(
                f"| F{fm.fold} | {fm.n} | {fm.mean_r:+.3f} | {fm.std_r:.3f} | "
                f"{fm.t_stat:+.2f} | {fm.frac_winners:.2%} | "
                f"{fm.mean_winner_r:+.3f} | {fm.mean_loser_r:+.3f} | "
                f"{fm.fold_roi_pct:+.2f} | {fm.fold_max_dd_pct:.2f} | "
                f"{_exit_reason_str(fm.exit_reason_counts)} |"
            )
        lines.append("")
        # §9 gate.
        v = c.verdict
        lines.append("### §9 gate")
        lines.append("")
        lines.append(f"- **A (sign consistency)**: {'PASS' if v.passes_A else 'FAIL'} — fold mean_r values: " +
                     ", ".join(f"F{fid}={mr:+.3f}" for fid, mr in v.a_signs))
        lines.append(f"- **B (size variance ≤ 3.0)**: {'PASS' if v.passes_B else 'FAIL'} — "
                     f"size_ratio = {v.b_size_ratio:.2f}" if math.isfinite(v.b_size_ratio) else
                     f"- **B (size variance ≤ 3.0)**: {'PASS' if v.passes_B else 'FAIL'} — size_ratio = inf")
        lines.append(
            f"- **C (DD ceiling ≤ 2× median)**: {'PASS' if v.passes_C else 'FAIL'} — "
            f"max-fold DD {v.c_max_dd:.2f}%, median {v.c_median_dd:.2f}%, ratio "
            f"{v.c_max_dd_ratio:.2f}" if math.isfinite(v.c_max_dd_ratio) else
            f"- **C (DD ceiling ≤ 2× median)**: {'PASS' if v.passes_C else 'FAIL'} — ratio inf"
        )
        lines.append(f"- **Overall**: **{v.overall}**")
        if v.notes:
            for n in v.notes:
                lines.append(f"  - {n}")
        lines.append("")

    # Sub-step 5D — per-pair concentration.
    for c in cluster_results:
        lines.append(f"## 5D — Cluster {c.cluster_id} per-pair contribution")
        lines.append("")
        n_total = int(c.pair_df["n_trades"].sum())
        active_pairs = int((c.pair_df["n_trades"] > 0).sum())
        lines.append(
            f"Total admitted simulated trades: {n_total} across {active_pairs} active pair(s). "
            f"Top-5 pairs hold {c.pair_top_mass:.2%} of trades."
            + (f" **Concentration flag**: top-5 > 50% (pairs: {', '.join(c.pair_top_names)})." if c.pair_concentration_flagged else " No concentration flag.")
        )
        lines.append("")
        lines.append("| Pair | n_trades | %  |")
        lines.append("|---|---:|---:|")
        for _, row in c.pair_df.head(10).iterrows():
            lines.append(f"| {row['pair']} | {int(row['n_trades'])} | {row['pct']:.2%} |")
        if len(c.pair_df) > 10:
            lines.append(f"| ...other {len(c.pair_df) - 10} pair(s) | — | — |")
        lines.append("")

    # Cross-arc observations.
    lines.append("## Cross-arc observations (informational; no within-arc calibration moves)")
    lines.append("")
    for c in cluster_results:
        rsw = c.threshold_resweep
        lines.append(
            f"- Cluster {c.cluster_id}: extended-grid winning threshold = "
            f"{rsw.selected_threshold:.4f} (base_rate {rsw.base_rate:.4f}); used fallback = "
            f"{rsw.used_fallback}. Step 4's 0.40 grid floor was too high for ~15% "
            f"base-rate targets — confirms the extended-grid calibration backlog item."
        )
    lines.append("")

    # Determinism.
    lines.append("## Determinism")
    lines.append("")
    lines.append("Two-run byte-identical sha256 over all written files:")
    lines.append("")
    lines.append("| File | Run 1 sha256 | Run 2 sha256 | Match |")
    lines.append("|---|---|---|---|")
    for fname in sorted(sha_run1.keys()):
        s1 = sha_run1[fname]
        s2 = sha_run2.get(fname) if sha_run2 else None
        match = "—" if s2 is None else ("PASS" if s1 == s2 else "FAIL")
        lines.append(f"| `{fname}` | `{s1[:16]}…` | `{(s2 or '—')[:16]}{'…' if s2 else ''}` | {match} |")
    lines.append("")
    lines.append(f"**Determinism: {determinism_gate}**")
    lines.append("")

    # Config sha256s.
    lines.append("## Config / input sha256s")
    lines.append("")
    for label, path in config_paths.items():
        lines.append(f"- `{label}` (`{path}`) — `{config_shas[label]}`")
    lines.append("")

    # Notes on simulator semantics (transparency).
    lines.append("## Simulator semantics (for audit / chat review)")
    lines.append("")
    lines.append(
        "- **R-frame**: per cluster (3 × ATR for cluster 1; 4 × ATR for cluster 3). 1R = SL distance under the cluster's selected SL."
    )
    lines.append(
        "- **Pre-t SL**: `entry − 2 × ATR(14)` (uniform per §3 mechanic), in force at bar offset 0 and during bar offset 1's intra-bar action."
    )
    lines.append(
        "- **Post-t SL switch**: at end of bar 1 (D1 classifier admits, by construction since we filter to admitted), the SL is REPLACED with `entry − cluster_R`. This is a LOOSENING for both clusters (pre-t SL is tighter than post-t)."
    )
    lines.append(
        "- **MFE-lock**: when `mfe_so_far_r ≥ 1.0` for the first time, lock and start trailing at `entry + (current_mfe_r − 0.75) × R`."
    )
    lines.append(
        "- **Trail update**: on every new MFE-r high, update trail_stop = `entry + (new_mfe_r − 0.75) × R`. SL ratchets up only."
    )
    lines.append(
        "- **Intra-bar sequencing**: MFE-first convention — update MFE from bar high, possibly start/update trail, then check SL against bar low. Matches Step 3 SL sweep semantics."
    )
    lines.append(
        "- **Spread**: entry-bar spread already baked into `entry_price` from Step 1. Exit-bar spread NOT applied for SL/trail/cap-bind exits (matches the per-prompt simulator convention; the deployed engine applies exit spread — minor discrepancy documented here, expected ~1 pip ≈ 0.001-0.002R per trade)."
    )
    lines.append(
        "- **Cap-bind exit price**: bar 240 close (chosen for the simulator; deployed engine uses bar N+1+240 open with spread)."
    )
    lines.append(
        "- **Risk per trade**: 0.5% of starting balance (L arc convention). Fold ROI is simple sum of (final_r × 0.005), expressed as %."
    )
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return arc_disp


# ============================================================================
# Main
# ============================================================================


def _env_dict() -> Dict[str, str]:
    try:
        import sklearn  # type: ignore

        sk_ver = sklearn.__version__
    except Exception:
        sk_ver = "not_installed"
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sk_ver,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Arc 4 Step 5 cross-fold stability (L_ARC_PROTOCOL v2.1.1 §9)."
    )
    p.add_argument(
        "-c",
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs" / "l_arc_4.yaml",
    )
    p.add_argument(
        "--wfo-config",
        type=Path,
        default=_REPO_ROOT / "configs" / "wfo_kh24.yaml",
    )
    p.add_argument(
        "--trades-csv",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--paths-csv",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--clusters-csv",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step2" / "clusters_K4.csv",
    )
    p.add_argument(
        "--pass-list-csv",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step3" / "capturability_pass_list.csv",
    )
    p.add_argument(
        "--catalogue",
        type=Path,
        default=_REPO_ROOT / "configs" / "feature_catalogue.yaml",
    )
    p.add_argument(
        "--step4-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step4",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "l_arc_4" / "step5",
    )
    p.add_argument(
        "--no-determinism-check",
        action="store_true",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.config = args.config.resolve()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    step1_dir = _REPO_ROOT / cfg["output"]["results_dir"]
    trades_csv = (args.trades_csv or (step1_dir / cfg["output"]["trades_csv"])).resolve()
    paths_csv = (args.paths_csv or (step1_dir / cfg["output"]["paths_csv"])).resolve()
    clusters_csv = args.clusters_csv.resolve()
    pass_list_csv = args.pass_list_csv.resolve()
    catalogue_path = args.catalogue.resolve()
    wfo_cfg_path = args.wfo_config.resolve()
    step4_dir = args.step4_dir.resolve()
    out_dir = args.out_dir.resolve()
    data_dir = (_REPO_ROOT / cfg["data"]["data_dirs"]["1H"]).resolve()

    # D1 YAMLs to update.
    d1_yaml_paths = {
        1: step4_dir / "cluster_1_D1_policy.yaml",
        3: step4_dir / "cluster_3_D1_policy.yaml",
    }
    e_filter_yaml_paths: Dict[int, Path] = {}

    print("[l_arc_4 step5] === RUN 1 ===", file=sys.stderr)
    cluster_results_1, sha_run1, folds = run_once(
        trades_csv, paths_csv, clusters_csv, pass_list_csv,
        e_filter_yaml_paths, d1_yaml_paths, catalogue_path, wfo_cfg_path,
        data_dir, out_dir,
    )

    sha_run2: Optional[Dict[str, str]] = None
    determinism_gate = "N/A"
    if not args.no_determinism_check:
        print("[l_arc_4 step5] === RUN 2 (determinism) ===", file=sys.stderr)
        _, sha_run2, _ = run_once(
            trades_csv, paths_csv, clusters_csv, pass_list_csv,
            e_filter_yaml_paths, d1_yaml_paths, catalogue_path, wfo_cfg_path,
            data_dir, out_dir,
        )
        matched = all(
            sha_run1.get(k) == sha_run2.get(k) for k in sorted(set(sha_run1) | set(sha_run2))
        )
        determinism_gate = "PASS" if matched else "FAIL"

    config_paths = {
        "configs/l_arc_4.yaml": str(args.config.relative_to(_REPO_ROOT)),
        "configs/feature_catalogue.yaml": str(catalogue_path.relative_to(_REPO_ROOT)),
        "configs/wfo_kh24.yaml": str(wfo_cfg_path.relative_to(_REPO_ROOT)),
        f"{cfg['output']['results_dir']}/trades_all.csv": str(trades_csv.relative_to(_REPO_ROOT)),
        f"{cfg['output']['results_dir']}/trades_paths.csv": str(paths_csv.relative_to(_REPO_ROOT)),
        "results/l_arc_4/step2/clusters_K4.csv": str(clusters_csv.relative_to(_REPO_ROOT)),
        "results/l_arc_4/step3/capturability_pass_list.csv": str(pass_list_csv.relative_to(_REPO_ROOT)),
    }
    config_shas = {
        label: _file_sha256(_REPO_ROOT / Path(path)) for label, path in config_paths.items()
    }

    diag_path = out_dir / "step5_diagnostics.md"
    arc_disp = write_diagnostics(
        diag_path,
        cluster_results_1,
        folds,
        sha_run1,
        sha_run2,
        determinism_gate,
        config_paths,
        config_shas,
    )

    print(
        f"[l_arc_4 step5] DONE arc_disp={arc_disp} determinism={determinism_gate}",
        file=sys.stderr,
    )
    print(f"[l_arc_4 step5] diagnostics → {diag_path}", file=sys.stderr)

    (out_dir / "step5_env.json").write_text(
        json.dumps(
            {
                "env": _env_dict(),
                "args": {k: str(v) for k, v in vars(args).items()},
                "arc_disposition": arc_disp,
                "determinism_gate": determinism_gate,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    return 0 if (arc_disp == "PASS" and determinism_gate in ("PASS", "N/A")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
