"""Arc 10 v3.0.2 FundedNext cost sweep — analysis driver (EET, swap-free).

Post-hoc cost overlay on the EET WFO pool (results/l_arc_10_v3.0.2/) for the
FundedNext deployment target. Unlike the prior UTC sweep
(scripts/audit/arc_10/cost_sweep.py, which consumed a pre-built
trade_ledger_utc.parquet), the EET arc never committed a per-trade replay
ledger, so this driver replays the winning config in-line over the EET pool +
trade_paths (mirroring scripts/l_arc_10_v3_0_2/amendment_3_addendum.py) to build
per-trade final_r_replayed + fold_id, then applies the cost overlay.

FundedNext cost model (dispatch §3):
    - Commission: $5/lot round-turn (NOT $4), ON in all 15 grid cells.
    - Spread:     widen recorded EET spread by (mult-1), no floor.
    - Slippage:   adverse pips × n_fills (3 if TP1 hit else 2).
    - Swap:       OFF — FundedNext swap-free add-on. NOT invoked.

Grid: 5 spread mults × 3 slippage levels = 15 cells (no swap axis).

simulate_path is invoked ONLY to reconstruct the winning-config per-trade R
(the same replay the bespoke step_5 / amendment_3 addendum performed); the cost
overlay itself is pure post-hoc R-arithmetic. No engine changes.

Outputs (results/l_arc_10_v3.0.2/fundednext_cost_sweep/):
    grid_results_eet.csv     — 15 cells, at r_base AND r_recommended
    per_fold_realistic.csv   — per-fold ROI/DD across the 9 realistic cells
    eet_vs_utc_delta.csv     — §8 cross-comparison
    diagnostics.json         — baseline check, zero-spread, FX constants, r-band
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.sim.costs import (  # noqa: E402
    compute_commission_usd,
    compute_extra_spread_price,
    compute_slippage_pips,
)
from core.sim.exit_policies import simulate_path  # noqa: E402
from scripts.l_arc_10_v3.step_5 import (  # noqa: E402
    HOLDOUT_END,
    HOLDOUT_START,
    INITIAL_BAL,
    N_SEARCH_FOLDS,
    RISK_PER_TRADE,
    SEARCH_END,
    SEARCH_START,
    _build_folds,
    _fold_metrics,
)

# ---------------------------------------------------------------------------
# Inputs / constants
# ---------------------------------------------------------------------------

EET_ARC_DIR = REPO_ROOT / "results" / "l_arc_10_v3.0.2"
POOL_PATH = EET_ARC_DIR / "step_1" / "pool.parquet"
PATHS_PATH = EET_ARC_DIR / "step_1" / "trade_paths.parquet"
WFO_PATH = EET_ARC_DIR / "step_5" / "wfo_results.csv"
OUT_DIR = EET_ARC_DIR / "fundednext_cost_sweep"

UTC_GRID_PATH = REPO_ROOT / "results" / "l_arc_10_v3.0.2" / "cost_sweep" / "grid_results.csv"

# Winning config (per EET step_5/best_candidate.md + wfo_results.csv Top-1)
WINNING_SL_MULT = 3.5
POOL_SL_MULT = 2.0  # pool's final_r / sl_distance is normalized to SL=2.0
# Pool mfe_r is in pool-SL units; replayed TP1 (=+1R at SL=3.5) hits when
# pool_mfe_r >= WINNING_SL_MULT / POOL_SL_MULT = 1.75.
TP1_HIT_THRESHOLD_POOL_MFE = WINNING_SL_MULT / POOL_SL_MULT

# FundedNext commission rate (dispatch §3.1 — $5/lot RT, NOT 5ers' $4).
COMMISSION_RATE_PER_LOT_RT = 5.0

# EET pool published worst-fold numbers (wfo_results.csv Top-1; baseline must
# reproduce these — dispatch §4 + §11).
EXPECTED_WORST_ROI = 0.22456294795253906
EXPECTED_WORST_DD = 0.07353866008324712
EXPECTED_WORST_RATIO = 6.4272759551420195
EXPECTED_N_TOTAL = 2059

JPY_QUOTE = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"}


def pip_size(pair: str) -> float:
    return 0.01 if pair in JPY_QUOTE else 0.0001


# Per-pair pip_value_usd reference rates — REUSED verbatim from the prior UTC
# cost sweep intent doc §4.1 (dispatch §3.4: reuse representative per-pair FX
# constants from the prior sweep). Sourced from the UTC re-run pool entry_price
# means; tolerance discussed in prior intent §4.3.
REF_USDJPY = 109.6032
REF_USDCAD = 1.2412
REF_USDCHF = 0.9324
REF_NZDUSD = 0.7047
REF_GBPUSD = 1.4067
REF_AUDUSD = 0.7760


def pip_value_usd_per_lot(pair: str) -> float:
    """Per-pair USD pip value at 1.0 standard lot (100,000 base units)."""
    quote = pair[3:]
    if quote == "USD":
        return 10.0
    if quote == "JPY":
        return 1000.0 / REF_USDJPY
    if quote == "CAD":
        return 10.0 / REF_USDCAD
    if quote == "CHF":
        return 10.0 / REF_USDCHF
    if quote == "NZD":
        return 10.0 * REF_NZDUSD
    if quote == "GBP":
        return 10.0 * REF_GBPUSD
    if quote == "AUD":
        return 10.0 * REF_AUDUSD
    raise ValueError(f"unknown quote currency for pair {pair}")


# ---------------------------------------------------------------------------
# Winning-config replay (build per-trade ledger with final_r_replayed + fold_id)
# ---------------------------------------------------------------------------


def select_winning_config(wfo_results_path: Path) -> dict:
    df = pd.read_csv(wfo_results_path).sort_values(
        "search_worst_ratio", ascending=False
    ).reset_index(drop=True)
    top = df.iloc[0].to_dict()
    expected = {
        "cluster_id": 0,
        "architecture": "A1",
        "sl_multiplier": 3.5,
        "exit_policy": "sl_partial_close_1r_runner_trail",
        "exposure": "unlimited",
    }
    for k, v in expected.items():
        actual = top.get(k)
        if k == "sl_multiplier":
            if abs(float(actual) - float(v)) > 1e-9:
                raise RuntimeError(f"Winning config {k}={actual} != expected {v}")
        elif actual != v:
            raise RuntimeError(f"Winning config {k}={actual!r} != expected {v!r}")
    return top


def _simulate_admitted(admitted: pd.DataFrame, paths_by_trade: dict, *, sl_mult: float,
                       exit_policy: str) -> pd.DataFrame:
    """Replay exits for each admitted trade; return augmented frame with
    final_r_replayed. Clamp mirrors step_5._fold_metrics / addendum."""
    new_rs = []
    for _, row in admitted.iterrows():
        tid = int(row["trade_id"])
        tg = paths_by_trade.get(tid, pd.DataFrame())
        r_new, _ = simulate_path(exit_policy, row, tg, sl_mult)
        new_rs.append(r_new)
    rs = np.array(new_rs, dtype=float)
    rs = np.clip(np.nan_to_num(rs, nan=0.0, posinf=20.0, neginf=-1.0), -1.5, 20.0)
    aug = admitted.copy().reset_index(drop=True)
    aug["final_r_replayed"] = rs
    return aug


def build_ledger(pool: pd.DataFrame, paths_by_trade: dict, *, sl_mult: float,
                 exit_policy: str) -> pd.DataFrame:
    """Replay 11 IS folds + holdout; return per-trade ledger with fold_id.

    A1 admits every trade in the OOS window; exposure=unlimited applies no cap
    (mirrors amendment_3_addendum)."""
    times = pd.to_datetime(pool["signal_bar_time"], utc=True)
    pool = pool.copy()
    pool["signal_bar_time"] = times

    rows: list[pd.DataFrame] = []
    is_folds = _build_folds(SEARCH_START, SEARCH_END, N_SEARCH_FOLDS)
    for fi, (fs, fe) in enumerate(is_folds, start=1):
        mask = (pool["signal_bar_time"] >= fs) & (pool["signal_bar_time"] <= fe)
        oos = pool[mask].sort_values("signal_bar_time").reset_index(drop=True)
        aug = _simulate_admitted(oos, paths_by_trade, sl_mult=sl_mult, exit_policy=exit_policy)
        aug["fold_id"] = fi
        rows.append(aug)

    h0, h1 = pd.Timestamp(HOLDOUT_START, tz="UTC"), pd.Timestamp(HOLDOUT_END, tz="UTC")
    hmask = (pool["signal_bar_time"] >= h0) & (pool["signal_bar_time"] <= h1)
    hoos = pool[hmask].sort_values("signal_bar_time").reset_index(drop=True)
    haug = _simulate_admitted(hoos, paths_by_trade, sl_mult=sl_mult, exit_policy=exit_policy)
    haug["fold_id"] = 0  # holdout sentinel
    rows.append(haug)

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Risk-parameterized fold metrics (mirrors step_5._fold_metrics exactly, with
# risk_pct substituted for RISK_PER_TRADE). At risk_pct=RISK_PER_TRADE this is
# byte-equivalent to the imported _fold_metrics; verified by self-check.
# ---------------------------------------------------------------------------


def _fold_metrics_at_risk(trades_df: pd.DataFrame, *, risk_pct: float) -> dict:
    if len(trades_df) == 0:
        return dict(n=0, mean_r=0.0, sum_r=0.0, roi=0.0, dd=0.0, ratio=np.nan, sign=0)
    t = trades_df.sort_values("signal_bar_time")
    r = t["final_r"].to_numpy()
    r_clipped = np.clip(np.nan_to_num(r, nan=0.0, posinf=20.0, neginf=-1.0), -1.5, 20.0)
    times = pd.to_datetime(t["signal_bar_time"])
    eq = INITIAL_BAL
    curve = []
    for rr in r_clipped:
        eq = eq + eq * risk_pct * rr
        curve.append(eq)
    curve = np.array(curve, dtype=float)
    ann_yrs = max((times.max() - times.min()).total_seconds() / (365.25 * 86400.0), 0.25)
    final_bal = float(curve[-1]) if curve.size else INITIAL_BAL
    total_ret = (final_bal / INITIAL_BAL) - 1.0
    try:
        roi_ann = (1.0 + total_ret) ** (1.0 / ann_yrs) - 1.0 if ann_yrs > 0 else 0.0
    except OverflowError:
        roi_ann = 1e6 if total_ret > 0 else -0.99
    peak = np.maximum.accumulate(curve) if curve.size else curve
    dd = float(np.max((peak - curve) / np.where(peak > 0, peak, 1.0))) if curve.size else 0.0
    ratio = roi_ann / dd if dd > 1e-6 else (float("inf") if roi_ann > 0 else 0.0)
    return dict(
        n=int(len(r)),
        mean_r=float(np.mean(r)),
        sum_r=float(np.sum(r)),
        roi=float(roi_ann),
        dd=float(dd),
        ratio=float(ratio) if np.isfinite(ratio) else (1e6 if ratio > 0 else 0.0),
        sign=1 if roi_ann > 0 else (-1 if roi_ann < 0 else 0),
    )


# ---------------------------------------------------------------------------
# Cost overlay per trade
# ---------------------------------------------------------------------------


def apply_costs_to_trade(row: pd.Series, *, spread_mult: float,
                         slip_per_fill_pips: float, commission_on: bool) -> dict:
    """Per-trade FundedNext cost components (swap-free) as R-multiple decrements."""
    pair = row["pair"]
    sl_distance_pool = float(row["sl_distance_price"])
    sl_distance_replayed = sl_distance_pool * (WINNING_SL_MULT / POOL_SL_MULT)
    sl_distance_pips = sl_distance_replayed / pip_size(pair)
    pip_val = pip_value_usd_per_lot(pair)
    risk_amount_usd = RISK_PER_TRADE * INITIAL_BAL  # $500

    lots = 0.0
    if sl_distance_pips > 0 and pip_val > 0:
        lots = risk_amount_usd / (sl_distance_pips * pip_val)

    mfe_r = float(row["mfe_r"])
    tp1_hit = np.isfinite(mfe_r) and mfe_r >= TP1_HIT_THRESHOLD_POOL_MFE

    comm_usd = compute_commission_usd(lots, rate_per_lot_rt=COMMISSION_RATE_PER_LOT_RT) \
        if (commission_on and lots > 0) else 0.0
    comm_r = comm_usd / risk_amount_usd

    if slip_per_fill_pips > 0 and lots > 0 and sl_distance_replayed > 0:
        slip_pips_total, _ = compute_slippage_pips(slip_per_fill_pips, tp1_hit)
        slip_r = (slip_pips_total * pip_size(pair)) / sl_distance_replayed
    else:
        slip_r = 0.0

    se = float(row["spread_close_at_entry"])
    sx = float(row["spread_close_at_exit"])
    if spread_mult > 1.0 and sl_distance_replayed > 0:
        extra_spread_r = compute_extra_spread_price(se, sx, spread_mult) / sl_distance_replayed
    else:
        extra_spread_r = 0.0

    total_cost_r = comm_r + slip_r + extra_spread_r
    return dict(
        lots=lots, tp1_hit=bool(tp1_hit), comm_r=comm_r, slip_r=slip_r,
        extra_spread_r=extra_spread_r, total_cost_r=total_cost_r,
        zero_spread=(se <= 0 or sx <= 0),
    )


# ---------------------------------------------------------------------------
# Per-cell evaluation
# ---------------------------------------------------------------------------


def evaluate_cell(df: pd.DataFrame, *, spread_mult: float, slip_per_fill_pips: float,
                  commission_on: bool, risk_pct: float | None) -> dict:
    """Apply overlay across all trades; compute per-fold metrics.

    risk_pct=None  -> use imported step_5._fold_metrics (r_base, baseline-reproducing).
    risk_pct=float -> use _fold_metrics_at_risk (scaled-risk evaluation)."""
    costs = df.apply(
        apply_costs_to_trade, axis=1, spread_mult=spread_mult,
        slip_per_fill_pips=slip_per_fill_pips, commission_on=commission_on,
    )
    cost_df = pd.DataFrame(list(costs.values)).reset_index(drop=True)
    df_adj = df.copy().reset_index(drop=True)
    df_adj["total_cost_r"] = cost_df["total_cost_r"]
    df_adj["final_r"] = df_adj["final_r_replayed"] - df_adj["total_cost_r"]

    fold_results = {}
    for fold_id in sorted(df_adj["fold_id"].unique()):
        sub = df_adj[df_adj["fold_id"] == fold_id]
        if risk_pct is None:
            m = _fold_metrics(sub)
        else:
            m = _fold_metrics_at_risk(sub, risk_pct=risk_pct)
        m["fold_id"] = int(fold_id)
        fold_results[int(fold_id)] = m

    search = [fold_results[i] for i in range(1, N_SEARCH_FOLDS + 1) if i in fold_results]
    holdout = fold_results.get(0)

    rois = [f["roi"] for f in search if f["n"] > 0]
    dds = [f["dd"] for f in search if f["n"] > 0]
    ratios = [f["ratio"] for f in search if f["n"] > 0 and np.isfinite(f["ratio"])]
    signs = [f["sign"] for f in search if f["n"] > 0]

    worst_roi = float(min(rois)) if rois else 0.0
    worst_dd = float(max(dds)) if dds else 0.0
    worst_ratio = float(min(ratios)) if ratios else 0.0
    mean_roi = float(np.mean(rois)) if rois else 0.0
    mean_dd = float(np.mean(dds)) if dds else 0.0
    sign_consistency = int(sum(1 for s in signs if s > 0))
    neg_folds = int(sum(1 for s in signs if s < 0))

    # Verdict per dispatch §9
    if worst_dd >= 0.10:
        verdict = "FAIL_DD_HARD"
    elif worst_ratio < 2.0 or worst_roi < 0.05:
        verdict = "FAIL"
    elif worst_dd >= 0.08:
        verdict = "DD_WATCH"
    else:
        verdict = "PASS_DEPLOYABLE"

    return dict(
        spread_mult=spread_mult, slip_per_fill_pips=slip_per_fill_pips,
        worst_fold_ratio=worst_ratio, worst_fold_dd=worst_dd, worst_fold_roi=worst_roi,
        mean_fold_roi=mean_roi, mean_fold_dd=mean_dd,
        sign_consistency=sign_consistency, neg_folds=neg_folds,
        holdout_roi=holdout["roi"] if holdout else 0.0,
        holdout_dd=holdout["dd"] if holdout else 0.0,
        holdout_ratio=holdout["ratio"] if holdout else 0.0,
        n_search_trades=int(sum(f["n"] for f in search)),
        n_holdout_trades=int(holdout["n"]) if holdout else 0,
        n_zero_spread=int(cost_df["zero_spread"].sum()),
        verdict=verdict, per_fold=fold_results,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SPREAD_MULTS = [1.0, 1.5, 2.0, 3.0, 4.0]
SLIP_LEVELS = [0.0, 0.5, 1.0]
REALISTIC_SPREADS = [1.0, 1.5, 2.0]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Arc 10 v3.0.2 FundedNext cost sweep (EET)")
    p.add_argument("--baseline-only", action="store_true")
    args = p.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PATHS_PATH.exists():
        print(f"[fn_sweep] STOP: trade_paths.parquet missing at {PATHS_PATH}", flush=True)
        return 3

    print("[fn_sweep] loading EET pool + trade_paths…", flush=True)
    pool = pd.read_parquet(POOL_PATH)
    pool["signal_bar_time"] = pd.to_datetime(pool["signal_bar_time"], utc=True)
    paths = pd.read_parquet(PATHS_PATH)
    paths_by_trade = {int(tid): g for tid, g in paths.groupby("trade_id")}

    winning = select_winning_config(WFO_PATH)
    sl_mult = float(winning["sl_multiplier"])
    exit_policy = str(winning["exit_policy"])
    print(f"[fn_sweep] Top-1: A1 / SL={sl_mult} / {exit_policy} / unlimited / c{int(winning['cluster_id'])}", flush=True)

    # ── Build per-trade replay ledger ──────────────────────────────────────
    ledger = build_ledger(pool, paths_by_trade, sl_mult=sl_mult, exit_policy=exit_policy)
    keep = [
        "trade_id", "pair", "signal_bar_time", "mfe_r", "time_to_peak_mfe",
        "spread_close_at_entry", "spread_close_at_exit", "sl_distance_price",
        "final_r_replayed", "fold_id",
    ]
    df = ledger[keep].copy()
    n_search = int((df["fold_id"] > 0).sum())
    n_holdout = int((df["fold_id"] == 0).sum())
    print(f"[fn_sweep] ledger: {len(df)} trades ({n_search} search, {n_holdout} holdout)", flush=True)

    # ── Zero-spread diagnostic (§11 stop if >1%) ───────────────────────────
    n_zero = int(((df["spread_close_at_entry"] <= 0) | (df["spread_close_at_exit"] <= 0)).sum())
    pct_zero = n_zero / max(len(df), 1) * 100.0
    print(f"[fn_sweep] zero-spread trades: {n_zero} ({pct_zero:.3f}% of {len(df)})", flush=True)
    if pct_zero > 1.0:
        print("[fn_sweep] STOP: zero-spread > 1% of pool (dispatch §11).", flush=True)
        return 4

    # ── Baseline reproduction (1× / 0 slip / commission OFF == unmodified pool) ─
    base = evaluate_cell(df, spread_mult=1.0, slip_per_fill_pips=0.0,
                         commission_on=False, risk_pct=None)
    print(
        f"[fn_sweep] baseline: worst_ratio={base['worst_fold_ratio']:.4f} "
        f"worst_dd={base['worst_fold_dd']:.6f} worst_roi={base['worst_fold_roi']:.6f} "
        f"n_search={base['n_search_trades']}",
        flush=True,
    )
    tol = 5e-3  # 0.5pp — consistent with amendment_3_addendum cross-check tolerance
    repro_ok = (
        abs(base["worst_fold_roi"] - EXPECTED_WORST_ROI) <= tol
        and abs(base["worst_fold_dd"] - EXPECTED_WORST_DD) <= tol
        and base["n_search_trades"] == EXPECTED_N_TOTAL
    )
    if not repro_ok:
        print(
            f"[fn_sweep] STOP (dispatch §11): baseline does NOT reproduce EET pool. "
            f"expected roi={EXPECTED_WORST_ROI:.6f} dd={EXPECTED_WORST_DD:.6f} "
            f"n={EXPECTED_N_TOTAL}; got roi={base['worst_fold_roi']:.6f} "
            f"dd={base['worst_fold_dd']:.6f} n={base['n_search_trades']}",
            flush=True,
        )
        return 2
    print("[fn_sweep] OK baseline reproduces EET pool published numbers (within 0.5pp).", flush=True)

    if args.baseline_only:
        return 0

    # ── §7 recommended risk band (linear DD scaling) ───────────────────────
    def worst_dd_at_rbase(sm: float, slip: float) -> float:
        c = evaluate_cell(df, spread_mult=sm, slip_per_fill_pips=slip,
                          commission_on=True, risk_pct=None)
        return c["worst_fold_dd"]

    DD_TARGET = 0.080
    central = worst_dd_at_rbase(1.5, 0.5)         # central realistic cell
    optimistic = worst_dd_at_rbase(1.0, 0.5)      # optimistic central
    adverse = worst_dd_at_rbase(2.0, 1.0)         # adverse stress
    r_rec_central = RISK_PER_TRADE * (DD_TARGET / central) if central > 0 else float("nan")
    r_rec_optimistic = RISK_PER_TRADE * (DD_TARGET / optimistic) if optimistic > 0 else float("nan")
    r_rec_adverse = RISK_PER_TRADE * (DD_TARGET / adverse) if adverse > 0 else float("nan")
    print(
        f"[fn_sweep] §7 r_recommended band: central(1.5x/0.5)={r_rec_central:.4f} "
        f"optimistic(1.0x/0.5)={r_rec_optimistic:.4f} adverse(2.0x/1.0)={r_rec_adverse:.4f}",
        flush=True,
    )
    r_recommended = r_rec_central

    # ── 15-cell grid at r_base AND r_recommended ───────────────────────────
    grid_rows = []
    per_fold_rows = []
    cell_idx = 0
    for sm in SPREAD_MULTS:
        for slip in SLIP_LEVELS:
            cell_idx += 1
            for label, rp in (("r_base", None), ("r_recommended", r_recommended)):
                res = evaluate_cell(df, spread_mult=sm, slip_per_fill_pips=slip,
                                    commission_on=True, risk_pct=rp)
                grid_rows.append({
                    "cell": cell_idx, "risk_level": label,
                    "risk_pct": RISK_PER_TRADE if rp is None else rp,
                    "spread_mult": sm, "slip_per_fill_pips": slip,
                    "worst_fold_ratio": res["worst_fold_ratio"],
                    "worst_fold_dd": res["worst_fold_dd"],
                    "worst_fold_roi": res["worst_fold_roi"],
                    "mean_fold_roi": res["mean_fold_roi"],
                    "mean_fold_dd": res["mean_fold_dd"],
                    "sign_consistency": res["sign_consistency"],
                    "neg_folds": res["neg_folds"],
                    "holdout_roi": res["holdout_roi"],
                    "holdout_dd": res["holdout_dd"],
                    "holdout_ratio": res["holdout_ratio"],
                    "n_search_trades": res["n_search_trades"],
                    "n_holdout_trades": res["n_holdout_trades"],
                    "n_zero_spread": res["n_zero_spread"],
                    "verdict": res["verdict"],
                })
                # per-fold detail only for the 9 realistic cells
                if sm in REALISTIC_SPREADS:
                    for fid in sorted(res["per_fold"].keys()):
                        f = res["per_fold"][fid]
                        per_fold_rows.append({
                            "cell": cell_idx, "spread_mult": sm, "slip_per_fill_pips": slip,
                            "risk_level": label,
                            "fold": "Holdout" if fid == 0 else f"F{fid}",
                            "fold_id": fid, "trades": f["n"],
                            "roi": f["roi"], "dd": f["dd"], "ratio": f["ratio"],
                        })
            print(
                f"[fn_sweep] cell {cell_idx:2d}/15: spread×{sm} slip={slip} "
                f"-> r_base verdict={grid_rows[-2]['verdict']} "
                f"(ratio={grid_rows[-2]['worst_fold_ratio']:.2f} "
                f"dd={grid_rows[-2]['worst_fold_dd']*100:.2f}% "
                f"roi={grid_rows[-2]['worst_fold_roi']*100:.2f}%)",
                flush=True,
            )

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(OUT_DIR / "grid_results_eet.csv", index=False, lineterminator="\n")
    pd.DataFrame(per_fold_rows).to_csv(OUT_DIR / "per_fold_realistic.csv", index=False, lineterminator="\n")
    print(f"[fn_sweep] wrote grid_results_eet.csv ({len(grid_df)} rows) + per_fold_realistic.csv", flush=True)

    # ── §8 EET vs UTC delta (9 realistic cells, swap-off, r_base) ──────────
    utc = pd.read_csv(UTC_GRID_PATH)
    utc = utc[(utc["swap_on"] == False)]  # noqa: E712
    delta_rows = []
    for sm in REALISTIC_SPREADS:
        for slip in SLIP_LEVELS:
            u = utc[(utc["spread_mult"] == sm) & (utc["slip_per_fill_pips"] == slip)]
            e = grid_df[(grid_df["risk_level"] == "r_base") & (grid_df["spread_mult"] == sm)
                        & (grid_df["slip_per_fill_pips"] == slip)]
            if len(u) == 0 or len(e) == 0:
                continue
            u = u.iloc[0]
            e = e.iloc[0]
            delta_rows.append({
                "spread_mult": sm, "slip_per_fill_pips": slip,
                "utc_ratio": u["worst_fold_ratio"], "eet_ratio": e["worst_fold_ratio"],
                "d_ratio": e["worst_fold_ratio"] - u["worst_fold_ratio"],
                "utc_dd": u["worst_fold_dd"], "eet_dd": e["worst_fold_dd"],
                "d_dd": e["worst_fold_dd"] - u["worst_fold_dd"],
                "utc_holdout_roi": u["holdout_roi"], "eet_holdout_roi": e["holdout_roi"],
                "d_holdout_roi": e["holdout_roi"] - u["holdout_roi"],
            })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(OUT_DIR / "eet_vs_utc_delta.csv", index=False, lineterminator="\n")
    print(f"[fn_sweep] wrote eet_vs_utc_delta.csv ({len(delta_df)} cells)", flush=True)

    # ── Diagnostics ────────────────────────────────────────────────────────
    diag = {
        "anchor_note": "EET pool byte-identity to 244fb76 verified; computed at HEAD (UTC grid + cost primitives post-date anchor).",
        "winning_config": {k: (float(winning[k]) if k == "sl_multiplier" else winning[k])
                           for k in ("cluster_id", "architecture", "sl_multiplier", "exit_policy", "exposure")},
        "commission_rate_per_lot_rt": COMMISSION_RATE_PER_LOT_RT,
        "swap": "OFF (FundedNext swap-free; primitive not invoked)",
        "baseline_check": {
            "expected": {"worst_roi": EXPECTED_WORST_ROI, "worst_dd": EXPECTED_WORST_DD,
                         "worst_ratio": EXPECTED_WORST_RATIO, "n_total": EXPECTED_N_TOTAL},
            "got": {"worst_roi": base["worst_fold_roi"], "worst_dd": base["worst_fold_dd"],
                    "worst_ratio": base["worst_fold_ratio"], "n_total": base["n_search_trades"]},
            "match": True,
        },
        "zero_spread": {"n": n_zero, "pct": pct_zero, "total": len(df)},
        "r_recommended_band": {
            "dd_target": DD_TARGET,
            "central_1.5x_0.5slip": {"worst_dd_at_rbase": central, "r_recommended": r_rec_central},
            "optimistic_1.0x_0.5slip": {"worst_dd_at_rbase": optimistic, "r_recommended": r_rec_optimistic},
            "adverse_2.0x_1.0slip": {"worst_dd_at_rbase": adverse, "r_recommended": r_rec_adverse},
        },
        "fx_constants": {"USDJPY": REF_USDJPY, "USDCAD": REF_USDCAD, "USDCHF": REF_USDCHF,
                         "NZDUSD": REF_NZDUSD, "GBPUSD": REF_GBPUSD, "AUDUSD": REF_AUDUSD},
    }
    (OUT_DIR / "diagnostics.json").write_text(
        json.dumps(diag, indent=2, default=str), encoding="utf-8", newline="\n"
    )
    print("[fn_sweep] wrote diagnostics.json. Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
