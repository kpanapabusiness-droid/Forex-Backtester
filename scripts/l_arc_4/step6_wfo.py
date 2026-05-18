"""Arc 4 — Step 6 (WFO truth + §10 ship rule) under 2026-05-17 per-pair p50 spread floors.

Per L_ARC_PROTOCOL.md v2.1.1 §10. Final gate before ship/no-ship decision.

Adapted from scripts/arc_5/step6_wfo_truth.py. Key Arc 4 differences:
  - ADMITTED trade outcome comes from §11 row 2 simulator (per_trade_simulated_refit_1.csv).
    Arc 5 used trades_all.final_r (SL=2×ATR baseline + time exit). Arc 4 cluster 1's
    production exit is §11 row 2 (MFE-lock 1R, trail 0.75R) at SL=3×ATR.
  - REJECTED path is bar-2 close-at-market — same mechanic as Arc 5.
  - Early-exit (bars_held ≤ 2 in step1 sim): trade SL-hit before classifier verdict;
    use step1 trades_all.final_r (already includes both spreads per SPREAD_SEMANTICS_LOCK).
  - **S/2 EXIT SPREAD CORRECTION on admitted final_r** — this is the genuinely new piece.
    Step 5C simulator's final_r omitted the S/2 exit cost (known prior bug per step5b_spread).
    Phase 6 corrects: subtract (S_exit_pips/2 × pip_size) / cluster_R_price from admitted final_r.
  - Continuous F2→F7 equity AND per-fold equity (Arc 5 only had per-fold).
  - Risk grid: 0.20% protocol-primary + 0.15% + 0.10% sensitivity (per user spec).
  - 5ers reality check: 5% daily DD threshold + 10% all-time DD threshold.

Per §10 spec: COMPOUNDED equity (not additive cum-sum). Per-trade update:
    eq_after = eq_before × (1 + r × risk_pct)
Compounded ROI = (eq_final − 1) × 100. Compounded DD = max((peak − eq) / peak) × 100.

Inputs (all read-only):
  - results/l_arc_4_rerun/step1/trades_all.csv  (10,893 trades, new-floor spreads)
  - results/l_arc_4_rerun/step1/trades_paths.csv  (per-bar OHLC)
  - results/l_arc_4_rerun/step2/clusters_K4.csv  (cluster labels)
  - results/l_arc_4_rerun/step5c/per_trade_simulated_refit_1.csv  (per-fold-refit admitted set)
  - results/l_arc_4_rerun/step5c/fold_definitions.csv
  - results/l_arc_4_rerun/step4/cluster_1_D1_policy.yaml (locked policy)
  - configs/spread_floors_5ers.yaml (NEW p50 floor, sha 8da7644b252…)
  - data/1hr/<pair>.csv (raw spread for S/2 exit correction)

Outputs (results/l_arc_4_rerun/step6/):
  - per_fold_metrics.csv
  - per_fold_equity_F<n>.csv (per-bar continuous equity within a fold)
  - full_data_metrics.csv
  - full_data_equity.csv (continuous F2→F7)
  - per_trade_audit_full_sequence.csv
  - daily_dd_events.csv
  - risk_sweep.csv
  - ship_decision.yaml
  - STEP_6_VERDICT.md (separate post-processing step writes this)
  - equity_curve.png

Determinism: deterministic; two-run byte-identical for all CSVs.

Usage:
  py scripts/l_arc_4/step6_wfo.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ============================================================
# Paths + constants
# ============================================================

RERUN_DIR = _REPO_ROOT / "results" / "l_arc_4_rerun"
STEP1_DIR = RERUN_DIR / "step1"
STEP2_DIR = RERUN_DIR / "step2"
STEP4_DIR = RERUN_DIR / "step4"
STEP5C_DIR = RERUN_DIR / "step5c"
OUT_DIR = RERUN_DIR / "step6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = _REPO_ROOT / "data" / "1hr"
SPREAD_FLOOR_PATH = _REPO_ROOT / "configs" / "spread_floors_5ers.yaml"

# Risk grid: per user spec — 0.20% protocol-primary + 0.15% + 0.10% sensitivity
RISK_GRID_PCT: List[float] = [0.20, 0.15, 0.10]
PRIMARY_RISK_PCT: float = 0.20

# Cluster 1 archetype-R from step4 policy YAML
CLUSTER_R_FRAME_ATR_MULT: float = 3.0  # cluster 1 archetype SL
PRE_T_SL_ATR_MULT: float = 2.0  # Pipeline D1 uniform pre-t SL

# §10 thresholds (per L_ARC_PROTOCOL.md §10)
DEPLOY_WORST_ROI_ANN_PCT = 5.0
DEPLOY_MEAN_ROI_ANN_PCT = 8.0
DEPLOY_FULL_DATA_ROI_PCT = 5.0  # annualised
DEPLOY_WORST_FOLD_DD_PCT = 8.0
DEPLOY_FULL_DATA_DD_PCT = 10.0
DEPLOY_TRADES_PER_FOLD = 15

VIABLE_WORST_ROI_PCT = 0.0  # per-fold OOS ROI > 0 (period, not annualised — matches Arc 5)
VIABLE_MEAN_ROI_ANN_PCT = 3.0
VIABLE_FULL_DATA_ROI_PCT = 3.0
VIABLE_WORST_FOLD_DD_PCT = 8.0
VIABLE_FULL_DATA_DD_PCT = 10.0
VIABLE_TRADES_PER_FOLD = 5

# 5ers prop firm reality check (informational, not gating per §10)
FERS_DAILY_DD_LIMIT_PCT = 5.0
FERS_ALL_TIME_DD_LIMIT_PCT = 10.0

# ============================================================
# Helpers
# ============================================================


def _pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_spread_floor() -> Dict[str, float]:
    """Return {pair: floor_pips} from the locked p50 floor file."""
    data = yaml.safe_load(SPREAD_FLOOR_PATH.read_text(encoding="utf-8"))
    floors_native = data["floors"]
    return {pair: float(stats["min_nonzero_spread_native"]) / 10.0 for pair, stats in floors_native.items()}


def _load_1h_spread_lookup(pair: str) -> pd.Series:
    """Return pd.Series indexed by Timestamp, values = raw spread in native points."""
    df = pd.read_csv(DATA_DIR / f"{pair}.csv", parse_dates=["time"])
    return df.set_index("time")["spread"].astype(float)


# ============================================================
# Annualisation factor — based on actual OOS days
# ============================================================


def _annualisation_factor(days: int) -> float:
    return 365.0 / max(days, 1)


# ============================================================
# Compounded metrics
# ============================================================


def _compounded_equity(r_chrono: np.ndarray, risk_pct: float) -> np.ndarray:
    """Compute per-step compounded equity, starting at 1.0. Returns array of length n+1."""
    if len(r_chrono) == 0:
        return np.array([1.0])
    factors = 1.0 + r_chrono * (risk_pct / 100.0)
    eq = np.concatenate([[1.0], np.cumprod(factors)])
    return eq


def _compounded_metrics_from_eq(eq: np.ndarray, oos_days: int) -> Dict[str, float]:
    """Return ROI % (period), ROI ann %, max DD %, terminal equity from an equity curve."""
    if len(eq) <= 1:
        return {"roi_pct": 0.0, "roi_ann_pct": 0.0, "max_dd_pct": 0.0, "terminal_equity": 1.0}
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    max_dd_pct = float(np.max(dd) * 100.0)
    roi_pct = float((eq[-1] - 1.0) * 100.0)
    af = _annualisation_factor(oos_days)
    # Annualise via simple linear scaling (matches Arc 5)
    roi_ann_pct = roi_pct * af
    return {"roi_pct": roi_pct, "roi_ann_pct": roi_ann_pct, "max_dd_pct": max_dd_pct, "terminal_equity": float(eq[-1])}


# ============================================================
# Per-trade R routing: admit (with S/2 exit correction) / reject / early-exit
# ============================================================


@dataclass
class TradeRecord:
    trade_id: int
    pair: str
    fold: int
    entry_time: pd.Timestamp
    entry_price: float
    atr_signal: float
    cluster_R_price: float
    bars_held_step1: int
    cluster_id: int  # true cluster (informational)
    is_admit: bool
    is_early_exit: bool
    final_r_raw: float            # original from §11 simulator (admit) or step1 (early) or close-at-market (reject)
    s_exit_pips: float            # bar-of-exit spread (for admit) or bar-2 spread (for reject)
    s_exit_correction_r: float    # subtracted from raw to get net final_r (admit only; 0 for others)
    final_r: float                # post-correction R outcome
    exit_bar: int                 # bar offset of exit (for admit: from simulator; for reject: 2; for early: from step1)


def _floored_spread_pips(raw_native: float, floor_pips: float, pair: str) -> float:
    """Apply spread floor. If raw is 0 → floor. Else use raw / 10."""
    raw_pips = raw_native / 10.0
    if raw_pips <= 0 or pd.isna(raw_native):
        return floor_pips
    return max(raw_pips, floor_pips)


def _lookup_exit_spread_pips(pair: str, exit_ts: pd.Timestamp, spread_lookup: pd.Series, floor_pips: float) -> float:
    """Look up raw spread at exit timestamp, apply floor."""
    if exit_ts not in spread_lookup.index:
        # No data for this hour — use floor as conservative
        return floor_pips
    raw = float(spread_lookup.loc[exit_ts])
    return _floored_spread_pips(raw, floor_pips, pair)


# ============================================================
# Build trade records
# ============================================================


def build_trade_records(
    trades_all: pd.DataFrame,
    paths_2d: Dict[str, np.ndarray],
    clusters: pd.DataFrame,
    refit_admitted: pd.DataFrame,
    fold_defs: pd.DataFrame,
    spread_floors: Dict[str, float],
    spread_lookups: Dict[str, pd.Series],
) -> List[TradeRecord]:
    """For every signal in trades_all, route to admit/reject/early-exit and produce final_r."""
    admit_ids = set(refit_admitted["trade_id"].tolist())
    admit_by_id = refit_admitted.set_index("trade_id")
    cluster_by_id = clusters.set_index("trade_id")["cluster_id"].to_dict()

    # Build fold lookup: each row in fold_defs gives oos_start, oos_end
    folds = [(int(r.fold), pd.Timestamp(r.oos_start), pd.Timestamp(r.oos_end)) for _, r in fold_defs.iterrows()]

    records: List[TradeRecord] = []
    for _, t in trades_all.iterrows():
        tid = int(t["trade_id"])
        pair = str(t["pair"])
        entry_time = pd.Timestamp(t["entry_time"])
        # Skip trades outside F2-F7 OOS (F1 is structurally excluded; pre-F1 is impossible)
        fold = -1
        for f, s, e in folds:
            if s <= entry_time < e:
                fold = f
                break
        if fold not in (2, 3, 4, 5, 6, 7):
            continue

        atr_signal = float(t["atr_14_at_signal"])
        cluster_R_price = CLUSTER_R_FRAME_ATR_MULT * atr_signal
        entry_price = float(t["entry_price"])
        bars_held_step1 = int(t["bars_held"])
        cluster_id = int(cluster_by_id.get(tid, -1))
        floor_pips = spread_floors[pair]
        spread_lookup = spread_lookups[pair]

        is_admit = tid in admit_ids
        is_early_exit = bars_held_step1 <= 2 and not is_admit
        # NOTE: an "early-exit" trade is one the classifier never got to evaluate because
        # the step1 SL=2×ATR fired before bar 2. Under cluster 1's wider SL=3×ATR, the
        # step1 final_r is an OVER-statement of cost (the 3×ATR trade might have survived).
        # However Pipeline D1's pre-t SL is 2×ATR (uniform), so the early-exit trade is
        # genuinely closed under SL=2×ATR before classifier verdict. step1 final_r is correct here.

        if is_admit:
            row = admit_by_id.loc[tid]
            final_r_raw = float(row["final_r"])
            exit_bar = int(row["exit_bar"])
            # Compute exit timestamp = entry_time + exit_bar hours
            exit_ts = entry_time + pd.Timedelta(hours=exit_bar)
            s_exit_pips = _lookup_exit_spread_pips(pair, exit_ts, spread_lookup, floor_pips)
            # S/2 exit correction: half_spread_price = (S_exit_pips / 2) × pip_size
            half_spread_price = (s_exit_pips / 2.0) * _pip_size(pair)
            s_exit_correction_r = half_spread_price / cluster_R_price
            final_r = final_r_raw - s_exit_correction_r
        elif is_early_exit:
            # Step1 final_r is in step1's R-frame (2×ATR). For cluster 1 deployment, position
            # is sized to cluster_R_price = 3×ATR. So convert R-frame: cluster_1_r = step1_r × 2/3.
            # Step1 final_r already incorporates entry + exit spread per SPREAD_SEMANTICS_LOCK.
            final_r_raw = float(t["final_r"])
            # R-frame conversion: PnL_price = step1_r × (2×ATR); cluster_1_r = PnL_price / (3×ATR) = step1_r × 2/3
            final_r = final_r_raw * (PRE_T_SL_ATR_MULT / CLUSTER_R_FRAME_ATR_MULT)
            s_exit_pips = float(t["spread_pips_exit"])
            s_exit_correction_r = 0.0
            exit_bar = bars_held_step1
        else:
            # Reject path: placeholder; filled below by fill_rejected_records()
            final_r_raw = float("nan")
            final_r = float("nan")
            s_exit_pips = 0.0
            s_exit_correction_r = 0.0
            exit_bar = 2

        records.append(TradeRecord(
            trade_id=tid,
            pair=pair,
            fold=fold,
            entry_time=entry_time,
            entry_price=entry_price,
            atr_signal=atr_signal,
            cluster_R_price=cluster_R_price,
            bars_held_step1=bars_held_step1,
            cluster_id=cluster_id,
            is_admit=is_admit,
            is_early_exit=is_early_exit,
            final_r_raw=final_r_raw,
            s_exit_pips=s_exit_pips,
            s_exit_correction_r=s_exit_correction_r,
            final_r=final_r,
            exit_bar=exit_bar,
        ))

    return records


def fill_rejected_records(records: List[TradeRecord], paths_df: pd.DataFrame,
                          trades_all: pd.DataFrame, spread_floors: Dict[str, float],
                          spread_lookups: Dict[str, pd.Series]) -> None:
    """Fill in rejected-trade close-at-market R using bar 2 open from paths_df.

    bar 2 open: bar_offset=2, column 'open'.
    Reject PnL formula: (bar2_open − entry_price − (S_exit/2 × pip_size)) / cluster_R_price
    Use bar 2 spread (timestamp = entry_time + 2 hours) with floor applied.
    """
    paths_b2 = paths_df[paths_df["bar_offset"] == 2][["trade_id", "open", "timestamp"]].set_index("trade_id")
    trades_by_id = trades_all.set_index("trade_id")
    for r in records:
        if r.is_admit or r.is_early_exit:
            continue
        if r.trade_id not in paths_b2.index:
            r.final_r = float("nan")
            continue
        bar2_open = float(paths_b2.loc[r.trade_id, "open"])
        bar2_ts = pd.Timestamp(paths_b2.loc[r.trade_id, "timestamp"])
        s_exit = _lookup_exit_spread_pips(r.pair, bar2_ts, spread_lookups[r.pair], spread_floors[r.pair])
        half_spread_price = (s_exit / 2.0) * _pip_size(r.pair)
        pnl_price = bar2_open - r.entry_price - half_spread_price
        r.final_r_raw = pnl_price / r.cluster_R_price
        r.final_r = r.final_r_raw  # rejected R already has S/2 deducted in formula
        r.s_exit_pips = s_exit
        r.s_exit_correction_r = 0.0  # already applied above
        r.exit_bar = 2


# ============================================================
# Daily DD breakdown from per-trade chrono r
# ============================================================


def compute_daily_dd_events(per_trade_audit: pd.DataFrame, risk_pct: float) -> pd.DataFrame:
    """For each calendar day with at least one trade, compute the day's max DD from
    the day's start equity. Returns DataFrame keyed by date with intraday-DD %.

    NOTE: This is an approximation. True hourly MTM with overlapping open positions
    would require bar-by-bar reconstruction. Here we treat closed-trade PnL ordered
    by exit time as the equity-update stream.
    """
    audit = per_trade_audit.copy()
    audit["exit_time"] = pd.to_datetime(audit["exit_time"])
    audit = audit.sort_values("exit_time")
    # Compute equity update per trade
    factors = 1.0 + audit["final_r"].to_numpy() * (risk_pct / 100.0)
    eq = np.concatenate([[1.0], np.cumprod(factors)])
    audit["eq_after"] = eq[1:]
    audit["eq_before"] = eq[:-1]
    audit["date"] = audit["exit_time"].dt.date

    rows = []
    for date, g in audit.groupby("date", sort=True):
        # Day's start equity: eq_before of first trade on this day
        start_eq = float(g["eq_before"].iloc[0])
        # Intraday running equity for trades closing this day
        eq_during = g["eq_after"].to_numpy()
        # Intraday min equity is the minimum of day's eq_during AND start (in case day opens at peak)
        intraday_min = min(start_eq, eq_during.min())
        # Daily DD = (start_eq - intraday_min) / start_eq * 100
        daily_dd_pct = (start_eq - intraday_min) / start_eq * 100.0
        end_eq = float(eq_during[-1])
        rows.append({
            "date": str(date),
            "n_trades_closed": int(len(g)),
            "start_eq": start_eq,
            "end_eq": end_eq,
            "intraday_min_eq": float(intraday_min),
            "daily_dd_pct": daily_dd_pct,
            "exceeds_5pct": int(daily_dd_pct > FERS_DAILY_DD_LIMIT_PCT),
        })
    return pd.DataFrame(rows)


# ============================================================
# §10 gate evaluation
# ============================================================


def evaluate_deploy(per_fold: List[Dict], full_metrics: Dict, n_admit_per_fold: List[int],
                    worst_fold_dd: float) -> Tuple[bool, List[str]]:
    reasons = []
    worst_roi_ann = min(f["fold_roi_ann_pct"] for f in per_fold)
    mean_roi_ann = float(np.mean([f["fold_roi_ann_pct"] for f in per_fold]))
    min_trades = min(n_admit_per_fold)
    if worst_roi_ann < DEPLOY_WORST_ROI_ANN_PCT:
        reasons.append(f"worst-fold ann ROI {worst_roi_ann:.2f}% < {DEPLOY_WORST_ROI_ANN_PCT}%")
    if mean_roi_ann < DEPLOY_MEAN_ROI_ANN_PCT:
        reasons.append(f"mean-fold ann ROI {mean_roi_ann:.2f}% < {DEPLOY_MEAN_ROI_ANN_PCT}%")
    if worst_fold_dd > DEPLOY_WORST_FOLD_DD_PCT:
        reasons.append(f"worst-fold DD {worst_fold_dd:.2f}% > {DEPLOY_WORST_FOLD_DD_PCT}%")
    if any(f["fold_roi_pct"] <= 0 for f in per_fold):
        reasons.append("one or more folds have non-positive ROI")
    if min_trades < DEPLOY_TRADES_PER_FOLD:
        reasons.append(f"min trades/fold {min_trades} < {DEPLOY_TRADES_PER_FOLD}")
    if full_metrics["roi_ann_pct"] < DEPLOY_FULL_DATA_ROI_PCT:
        reasons.append(f"full-data ann ROI {full_metrics['roi_ann_pct']:.2f}% < {DEPLOY_FULL_DATA_ROI_PCT}%")
    if full_metrics["max_dd_pct"] > DEPLOY_FULL_DATA_DD_PCT:
        reasons.append(f"full-data DD {full_metrics['max_dd_pct']:.2f}% > {DEPLOY_FULL_DATA_DD_PCT}%")
    return (len(reasons) == 0), reasons


def evaluate_viable(per_fold: List[Dict], full_metrics: Dict, n_admit_per_fold: List[int],
                    worst_fold_dd: float) -> Tuple[bool, List[str]]:
    reasons = []
    worst_roi_pct = min(f["fold_roi_pct"] for f in per_fold)
    mean_roi_ann = float(np.mean([f["fold_roi_ann_pct"] for f in per_fold]))
    min_trades = min(n_admit_per_fold)
    if worst_roi_pct <= VIABLE_WORST_ROI_PCT:
        reasons.append(f"worst-fold ROI {worst_roi_pct:.2f}% <= {VIABLE_WORST_ROI_PCT}%")
    if mean_roi_ann < VIABLE_MEAN_ROI_ANN_PCT:
        reasons.append(f"mean-fold ann ROI {mean_roi_ann:.2f}% < {VIABLE_MEAN_ROI_ANN_PCT}%")
    if worst_fold_dd > VIABLE_WORST_FOLD_DD_PCT:
        reasons.append(f"worst-fold DD {worst_fold_dd:.2f}% > {VIABLE_WORST_FOLD_DD_PCT}%")
    if any(f["fold_roi_pct"] <= 0 for f in per_fold):
        reasons.append("one or more folds have non-positive ROI")
    if min_trades < VIABLE_TRADES_PER_FOLD:
        reasons.append(f"min trades/fold {min_trades} < {VIABLE_TRADES_PER_FOLD}")
    if full_metrics["roi_ann_pct"] < VIABLE_FULL_DATA_ROI_PCT:
        reasons.append(f"full-data ann ROI {full_metrics['roi_ann_pct']:.2f}% < {VIABLE_FULL_DATA_ROI_PCT}%")
    if full_metrics["max_dd_pct"] > VIABLE_FULL_DATA_DD_PCT:
        reasons.append(f"full-data DD {full_metrics['max_dd_pct']:.2f}% > {VIABLE_FULL_DATA_DD_PCT}%")
    return (len(reasons) == 0), reasons


# ============================================================
# Main
# ============================================================


def main() -> int:
    t0 = time.time()
    print("[step6] loading inputs...", file=sys.stderr)

    trades_all = pd.read_csv(STEP1_DIR / "trades_all.csv", parse_dates=["signal_time", "entry_time"])
    trades_all = trades_all.sort_values("trade_id").reset_index(drop=True)
    paths = pd.read_csv(STEP1_DIR / "trades_paths.csv", parse_dates=["timestamp"])
    clusters = pd.read_csv(STEP2_DIR / "clusters_K4.csv")
    refit_admitted = pd.read_csv(STEP5C_DIR / "per_trade_simulated_refit_1.csv", parse_dates=["entry_ts"])
    fold_defs = pd.read_csv(STEP5C_DIR / "fold_definitions.csv")

    # Cluster 1 classifier admissions only (per_trade_simulated_refit_1 has admissions from
    # cluster 1 classifier scoring ALL trades, not just true cluster 1 — this IS the deployable
    # framing: live classifier doesn't know true cluster label).
    print(f"  trades_all: {len(trades_all):,}", file=sys.stderr)
    print(f"  refit-admitted (cluster 1 classifier output): {len(refit_admitted):,}", file=sys.stderr)

    # Verify Pipeline D1 policy
    pol_path = STEP4_DIR / "cluster_1_D1_policy.yaml"
    pol = yaml.safe_load(pol_path.read_text(encoding="utf-8"))
    assert pol["pipeline"] == "D1"
    assert pol["selected_t"] == 1
    assert abs(pol["pre_t_sl_atr_mult"] - PRE_T_SL_ATR_MULT) < 1e-9
    assert abs(pol["post_t_sl_atr_mult"] - CLUSTER_R_FRAME_ATR_MULT) < 1e-9
    print(f"  policy: pipeline={pol['pipeline']}, t={pol['selected_t']}, "
          f"pre-t SL={pol['pre_t_sl_atr_mult']}×ATR, post-t SL={pol['post_t_sl_atr_mult']}×ATR", file=sys.stderr)

    # Spread floors + per-pair raw spread lookups
    print("[step6] loading spread floors + raw 1H spread lookups...", file=sys.stderr)
    spread_floors = _load_spread_floor()
    pairs_in_pool = sorted(trades_all["pair"].unique())
    spread_lookups = {p: _load_1h_spread_lookup(p) for p in pairs_in_pool}
    print(f"  {len(pairs_in_pool)} pairs loaded; floor range: "
          f"{min(spread_floors.values()):.1f}-{max(spread_floors.values()):.1f} pips", file=sys.stderr)

    # Build per-trade records
    print("[step6] building trade records (admit/reject/early-exit routing + S/2 exit correction)...", file=sys.stderr)
    records: List[TradeRecord] = build_trade_records(
        trades_all, {}, clusters, refit_admitted, fold_defs, spread_floors, spread_lookups
    )
    print(f"  {len(records):,} records covering F2-F7", file=sys.stderr)
    n_admit = sum(1 for r in records if r.is_admit)
    n_reject = sum(1 for r in records if (not r.is_admit) and (not r.is_early_exit))
    n_early = sum(1 for r in records if r.is_early_exit)
    print(f"  admit={n_admit}, reject={n_reject}, early-exit={n_early}", file=sys.stderr)

    # Fill rejected records' bar-2 close-at-market R
    print("[step6] computing rejected-trade bar-2 close-at-market R...", file=sys.stderr)
    fill_rejected_records(records, paths, trades_all, spread_floors, spread_lookups)
    # Sanity check on rejected
    rej_rs = [r.final_r for r in records if (not r.is_admit) and (not r.is_early_exit) and not np.isnan(r.final_r)]
    if rej_rs:
        print(f"  rejected mean R={np.mean(rej_rs):+.4f}, p50={np.median(rej_rs):+.4f}, std={np.std(rej_rs):.4f}", file=sys.stderr)

    # Build per-trade audit DataFrame
    audit_df = pd.DataFrame([{
        "trade_id": r.trade_id,
        "pair": r.pair,
        "fold": r.fold,
        "entry_time": r.entry_time,
        "exit_time": r.entry_time + pd.Timedelta(hours=r.exit_bar),
        "entry_price": r.entry_price,
        "atr_signal": r.atr_signal,
        "cluster_R_price": r.cluster_R_price,
        "cluster_id_true": r.cluster_id,
        "is_admit": int(r.is_admit),
        "is_early_exit": int(r.is_early_exit),
        "bars_held_step1": r.bars_held_step1,
        "exit_bar": r.exit_bar,
        "s_exit_pips": r.s_exit_pips,
        "s_exit_correction_r": r.s_exit_correction_r,
        "final_r_raw": r.final_r_raw,
        "final_r": r.final_r,
    } for r in records])
    # Drop rows where final_r is NaN (shouldn't happen but defensive)
    audit_df = audit_df[audit_df["final_r"].notna()].copy()
    audit_df = audit_df.sort_values(["fold", "entry_time", "trade_id"]).reset_index(drop=True)
    print(f"[step6] audit DataFrame: {len(audit_df):,} rows", file=sys.stderr)
    audit_df.to_csv(OUT_DIR / "per_trade_audit_full_sequence.csv", index=False)

    # ===== Per-fold + continuous metrics across risk grid =====
    # OOS days per fold (constant across runs)
    fold_oos_days = {int(r.fold): int((pd.Timestamp(r.oos_end) - pd.Timestamp(r.oos_start)).total_seconds() / 86400)
                     for _, r in fold_defs.iterrows()}

    # Build chrono r per fold
    folds_to_evaluate = [2, 3, 4, 5, 6, 7]
    fold_chrono_r: Dict[int, np.ndarray] = {}
    fold_chrono_audit: Dict[int, pd.DataFrame] = {}
    fold_admit_count: Dict[int, int] = {}
    for f in folds_to_evaluate:
        sub = audit_df[audit_df["fold"] == f].sort_values("entry_time").reset_index(drop=True)
        fold_chrono_r[f] = sub["final_r"].to_numpy(dtype=float)
        fold_chrono_audit[f] = sub
        fold_admit_count[f] = int(sub["is_admit"].sum())

    print(f"[step6] per-fold trade counts: {[len(fold_chrono_r[f]) for f in folds_to_evaluate]}", file=sys.stderr)
    print(f"[step6] per-fold admit counts: {[fold_admit_count[f] for f in folds_to_evaluate]}", file=sys.stderr)

    # Continuous sequence (F2 → F7 unbroken, ordered by entry_time)
    continuous_audit = audit_df.sort_values("entry_time").reset_index(drop=True)
    continuous_r = continuous_audit["final_r"].to_numpy(dtype=float)
    # Continuous days from F2 OOS start to F7 OOS end
    F2_start = pd.Timestamp(fold_defs.loc[fold_defs["fold"] == 2, "oos_start"].iloc[0])
    F7_end = pd.Timestamp(fold_defs.loc[fold_defs["fold"] == 7, "oos_end"].iloc[0])
    continuous_days = int((F7_end - F2_start).total_seconds() / 86400)
    print(f"[step6] continuous F2→F7 window: {F2_start.date()} → {F7_end.date()}, {continuous_days} days, {len(continuous_r):,} trades", file=sys.stderr)

    # Risk sweep
    sweep_rows = []
    per_fold_at_risk: Dict[int, List[Dict]] = {}
    full_data_at_risk: Dict[int, Dict] = {}
    daily_events_at_risk: Dict[int, pd.DataFrame] = {}
    equity_curve_at_risk: Dict[int, pd.DataFrame] = {}

    for risk_pct in RISK_GRID_PCT:
        r_bps = int(round(risk_pct * 100))
        print(f"[step6] === risk {risk_pct:.2f}% ({r_bps} bps) ===", file=sys.stderr)
        per_fold: List[Dict] = []
        for f in folds_to_evaluate:
            eq = _compounded_equity(fold_chrono_r[f], risk_pct)
            m = _compounded_metrics_from_eq(eq, fold_oos_days[f])
            per_fold.append({
                "fold": f,
                "oos_days": fold_oos_days[f],
                "n_trades": len(fold_chrono_r[f]),
                "n_admit": fold_admit_count[f],
                "fold_roi_pct": m["roi_pct"],
                "fold_roi_ann_pct": m["roi_ann_pct"],
                "fold_max_dd_pct": m["max_dd_pct"],
                "fold_terminal_equity": m["terminal_equity"],
            })
            # Save per-fold equity curve (continuous WITHIN fold)
            fold_curve = pd.DataFrame({
                "step": np.arange(len(eq)),
                "equity": eq,
            })
            fold_curve.to_csv(OUT_DIR / f"per_fold_equity_F{f}_at_{r_bps}bps.csv", index=False)

        per_fold_at_risk[r_bps] = per_fold

        # Full-data continuous equity
        eq_full = _compounded_equity(continuous_r, risk_pct)
        full_m = _compounded_metrics_from_eq(eq_full, continuous_days)
        full_data_at_risk[r_bps] = full_m

        # Attach equity to continuous audit
        cont_eq_df = continuous_audit.copy()
        cont_eq_df["step"] = np.arange(1, len(cont_eq_df) + 1)
        cont_eq_df["equity_after"] = eq_full[1:]
        cont_eq_df["risk_pct"] = risk_pct
        cont_eq_df.to_csv(OUT_DIR / f"full_data_equity_at_{r_bps}bps.csv", index=False)
        equity_curve_at_risk[r_bps] = cont_eq_df

        # Daily DD breakdown
        daily = compute_daily_dd_events(cont_eq_df, risk_pct)
        daily.to_csv(OUT_DIR / f"daily_dd_at_{r_bps}bps.csv", index=False)
        daily_events_at_risk[r_bps] = daily

        # §10 gate eval
        n_admit_per_fold = [fold_admit_count[f] for f in folds_to_evaluate]
        worst_fold_dd = max(f["fold_max_dd_pct"] for f in per_fold)
        ok_deploy, deploy_reasons = evaluate_deploy(per_fold, full_m, n_admit_per_fold, worst_fold_dd)
        ok_viable, viable_reasons = evaluate_viable(per_fold, full_m, n_admit_per_fold, worst_fold_dd)
        if ok_deploy:
            tier = "DEPLOYABLE"
        elif ok_viable:
            tier = "VIABLE"
        else:
            tier = "FAIL"

        # 5ers reality check
        max_daily_dd = float(daily["daily_dd_pct"].max())
        days_over_5pct = int(daily["exceeds_5pct"].sum())
        max_alltime_dd = full_m["max_dd_pct"]

        sweep_rows.append({
            "risk_pct": risk_pct,
            "risk_bps": r_bps,
            "n_admit_total": int(sum(n_admit_per_fold)),
            "n_admit_min_per_fold": int(min(n_admit_per_fold)),
            "worst_fold_roi_ann_pct": min(f["fold_roi_ann_pct"] for f in per_fold),
            "mean_fold_roi_ann_pct": float(np.mean([f["fold_roi_ann_pct"] for f in per_fold])),
            "worst_fold_max_dd_pct": worst_fold_dd,
            "full_data_roi_pct": full_m["roi_pct"],
            "full_data_roi_ann_pct": full_m["roi_ann_pct"],
            "full_data_max_dd_pct": full_m["max_dd_pct"],
            "max_daily_dd_pct": max_daily_dd,
            "days_over_5pct_daily_dd": days_over_5pct,
            "passes_deployable": int(ok_deploy),
            "passes_viable": int(ok_viable),
            "tier": tier,
            "deploy_fail_reasons": " | ".join(deploy_reasons) if deploy_reasons else "",
            "viable_fail_reasons": " | ".join(viable_reasons) if viable_reasons else "",
        })

        # Print headline
        print(f"  risk={risk_pct}%: worst-fold ann ROI={min(f['fold_roi_ann_pct'] for f in per_fold):.2f}%, "
              f"worst-fold DD={worst_fold_dd:.2f}%, full-data ROI={full_m['roi_pct']:.2f}% "
              f"(ann {full_m['roi_ann_pct']:.2f}%), full DD={full_m['max_dd_pct']:.2f}%, "
              f"max daily DD={max_daily_dd:.2f}%, tier={tier}", file=sys.stderr)

    # Write outputs
    pd.DataFrame(sweep_rows).to_csv(OUT_DIR / "risk_sweep.csv", index=False)
    # Per-fold metrics at primary risk
    primary_bps = int(round(PRIMARY_RISK_PCT * 100))
    pd.DataFrame(per_fold_at_risk[primary_bps]).to_csv(OUT_DIR / "per_fold_metrics.csv", index=False)
    # Full-data metrics at primary risk
    full_primary = full_data_at_risk[primary_bps]
    pd.DataFrame([{
        "risk_pct": PRIMARY_RISK_PCT,
        "roi_pct": full_primary["roi_pct"],
        "roi_ann_pct": full_primary["roi_ann_pct"],
        "max_dd_pct": full_primary["max_dd_pct"],
        "terminal_equity": full_primary["terminal_equity"],
        "n_trades": int(len(continuous_r)),
        "n_admit_total": int(sum(fold_admit_count[f] for f in folds_to_evaluate)),
        "continuous_days": continuous_days,
        "max_daily_dd_pct": float(daily_events_at_risk[primary_bps]["daily_dd_pct"].max()),
        "days_over_5pct": int(daily_events_at_risk[primary_bps]["exceeds_5pct"].sum()),
    }]).to_csv(OUT_DIR / "full_data_metrics.csv", index=False)

    # Daily DD events (primary risk only)
    daily_events_at_risk[primary_bps].to_csv(OUT_DIR / "daily_dd_events.csv", index=False)

    # Ship decision (primary risk)
    primary_row = next(r for r in sweep_rows if r["risk_bps"] == primary_bps)
    ship_decision = {
        "primary_risk_pct": PRIMARY_RISK_PCT,
        "verdict": primary_row["tier"],
        "verdict_5ers_compatible": bool(
            primary_row["tier"] in ("DEPLOYABLE", "VIABLE")
            and primary_row["max_daily_dd_pct"] <= FERS_DAILY_DD_LIMIT_PCT
            and primary_row["full_data_max_dd_pct"] <= FERS_ALL_TIME_DD_LIMIT_PCT
        ),
        "worst_fold_roi_ann_pct": primary_row["worst_fold_roi_ann_pct"],
        "mean_fold_roi_ann_pct": primary_row["mean_fold_roi_ann_pct"],
        "worst_fold_max_dd_pct": primary_row["worst_fold_max_dd_pct"],
        "full_data_roi_ann_pct": primary_row["full_data_roi_ann_pct"],
        "full_data_max_dd_pct": primary_row["full_data_max_dd_pct"],
        "max_daily_dd_pct": primary_row["max_daily_dd_pct"],
        "days_over_5pct_daily_dd": primary_row["days_over_5pct_daily_dd"],
        "deploy_fail_reasons": primary_row["deploy_fail_reasons"],
        "viable_fail_reasons": primary_row["viable_fail_reasons"],
        "sensitivity_risk_pcts": [r for r in RISK_GRID_PCT if r != PRIMARY_RISK_PCT],
    }
    (OUT_DIR / "ship_decision.yaml").write_text(yaml.safe_dump(ship_decision, sort_keys=False), encoding="utf-8")

    # Metadata
    meta = {
        "produced_at_iso": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": time.time() - t0,
        "n_trades_total": int(len(audit_df)),
        "n_admit_total": int(sum(fold_admit_count[f] for f in folds_to_evaluate)),
        "n_reject_total": n_reject,
        "n_early_exit_total": n_early,
        "fold_breakdown": {str(f): {"n_trades": len(fold_chrono_r[f]), "n_admit": fold_admit_count[f]} for f in folds_to_evaluate},
        "input_shas": {
            "trades_all.csv": _file_sha256(STEP1_DIR / "trades_all.csv"),
            "per_trade_simulated_refit_1.csv": _file_sha256(STEP5C_DIR / "per_trade_simulated_refit_1.csv"),
            "spread_floors_5ers.yaml": _file_sha256(SPREAD_FLOOR_PATH),
            "cluster_1_D1_policy.yaml": _file_sha256(pol_path),
        },
        "primary_risk_pct": PRIMARY_RISK_PCT,
        "risk_grid": RISK_GRID_PCT,
        "constants": {
            "PRE_T_SL_ATR_MULT": PRE_T_SL_ATR_MULT,
            "CLUSTER_R_FRAME_ATR_MULT": CLUSTER_R_FRAME_ATR_MULT,
            "DEPLOY_WORST_ROI_ANN_PCT": DEPLOY_WORST_ROI_ANN_PCT,
            "DEPLOY_WORST_FOLD_DD_PCT": DEPLOY_WORST_FOLD_DD_PCT,
            "DEPLOY_FULL_DATA_ROI_PCT": DEPLOY_FULL_DATA_ROI_PCT,
            "DEPLOY_FULL_DATA_DD_PCT": DEPLOY_FULL_DATA_DD_PCT,
            "VIABLE_WORST_FOLD_DD_PCT": VIABLE_WORST_FOLD_DD_PCT,
            "FERS_DAILY_DD_LIMIT_PCT": FERS_DAILY_DD_LIMIT_PCT,
            "FERS_ALL_TIME_DD_LIMIT_PCT": FERS_ALL_TIME_DD_LIMIT_PCT,
        },
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(f"[step6] DONE in {time.time()-t0:.1f}s; verdict at primary risk {PRIMARY_RISK_PCT}% = {primary_row['tier']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
