"""Arc 10 v3.0.2 cost-realism sweep — analysis driver (Option B, post-hoc overlay).

Per the revised dispatch (2026-05-27, REVISED Option B): consumes the UTC re-run
trade-ledger and pool (results/l_arc_10_v3_0_2_utc_rerun/), applies canonical
cost primitives (core.sim.costs.*) as per-trade R-multiple adjustments, and
sweeps a 5 × 2 × 3 grid of spread-mult × swap × slippage cells. Commission is
ON in every cell. simulate_path is NOT invoked — the trade ledger already
replays the winning A1 + SL=3.5 + sl_partial_close_1r_runner_trail config.

Per-trade cost computation:
    lots          = risk_amount_usd / (sl_distance_pips × pip_value_usd_per_lot)
    swap_r        = compute_swap_usd(...) / risk_amount_usd      [if swap=on]
    commission_r  = compute_commission_usd(lots) / risk_amount_usd
    slippage_r    = (slip_per_fill × n_fills) × pip_size / sl_distance_price
    extra_spread_r = compute_extra_spread_price(...) / sl_distance_price
    adjusted_r    = final_r_replayed − (swap_r + commission_r + slippage_r + extra_spread_r)

Per-fold aggregation: mirrors step_5._fold_metrics by importing it directly so
baseline cell (mult=1, swap=off, slip=0) reproduces by construction.

Outputs:
    results/l_arc_10_v3.0.2/cost_sweep/grid_results.csv
    results/l_arc_10_v3.0.2/cost_sweep/cost_sweep.log (via stdout redirection)
    Plus diagnostic prints for §5.2 (2400 EET entry assertion) + zero-spread count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.sim.costs import (  # noqa: E402
    compute_commission_usd,
    compute_extra_spread_price,
    compute_slippage_pips,
    compute_swap_usd,
)
from scripts.l_arc_10_v3.step_5 import (  # noqa: E402
    INITIAL_BAL,
    RISK_PER_TRADE,
    _fold_metrics,
)

# ---------------------------------------------------------------------------
# Inputs / constants
# ---------------------------------------------------------------------------

ARC_RESULTS = REPO_ROOT / "results" / "l_arc_10_v3_0_2_utc_rerun"
TRADE_LEDGER_PATH = ARC_RESULTS / "trade_ledger_utc.parquet"
POOL_PATH = ARC_RESULTS / "step_1" / "pool.parquet"
SWAPS_YAML = REPO_ROOT / "configs" / "swaps_5ers.yaml"

COST_SWEEP_OUT = REPO_ROOT / "results" / "l_arc_10_v3.0.2" / "cost_sweep"

# Winning config (per UTC re-run step_5/best_candidate.md)
WINNING_SL_MULT = 3.5
POOL_SL_MULT = 2.0  # pool's final_r is normalized to SL=2.0 sl_only

# Pool's mfe_r is in pool-SL units (2.0); TP1 (=+1R under winning SL=3.5) hits when
# pool_mfe_r ≥ WINNING_SL_MULT / POOL_SL_MULT
TP1_HIT_THRESHOLD_POOL_MFE = WINNING_SL_MULT / POOL_SL_MULT  # = 1.75

# Per-pair pip size (0.01 for JPY-quote, 0.0001 otherwise)
JPY_QUOTE = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"}


def pip_size(pair: str) -> float:
    return 0.01 if pair in JPY_QUOTE else 0.0001


# Per-pair pip_value_usd reference rates (sourced from UTC re-run pool entry_price means;
# documented in cost_sweep_intent.md §4.1)
REF_USDJPY = 109.6032
REF_USDCAD = 1.2412
REF_USDCHF = 0.9324
REF_NZDUSD = 0.7047
REF_GBPUSD = 1.4067
REF_AUDUSD = 0.7760


def pip_value_usd_per_lot(pair: str) -> float:
    """Per-pair USD pip value at 1.0 standard lot (100,000 base units).

    Uses representative FX constants from intent doc §4. Single per-pair value
    over 2010-2026; tolerance discussed in intent §4.3.
    """
    quote = pair[3:]
    base = pair[:3]
    if quote == "USD":
        # USD-quoted: $10/lot for non-JPY, but JPY-quote is captured above.
        return 10.0
    if quote == "JPY":
        return 1000.0 / REF_USDJPY  # ≈ 9.124
    if quote == "CAD":
        return 10.0 / REF_USDCAD  # ≈ 8.057
    if quote == "CHF":
        return 10.0 / REF_USDCHF  # ≈ 10.725
    if quote == "NZD":
        # 10 / USDNZD = 10 × NZDUSD
        return 10.0 * REF_NZDUSD  # ≈ 7.047
    if quote == "GBP":
        return 10.0 * REF_GBPUSD  # ≈ 14.067
    if quote == "AUD":
        return 10.0 * REF_AUDUSD  # ≈ 7.760
    raise ValueError(f"unknown quote currency for pair {pair} (base={base} quote={quote})")


# ---------------------------------------------------------------------------
# Load static inputs
# ---------------------------------------------------------------------------


def load_swaps_yaml() -> dict:
    with open(SWAPS_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_trade_ledger() -> pd.DataFrame:
    df = pd.read_parquet(TRADE_LEDGER_PATH)
    # Normalize timestamps to UTC tz-aware
    for col in ["signal_bar_time", "entry_time", "exit_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def load_pool_extras() -> pd.DataFrame:
    """Pool columns NOT in the trade ledger: mfe_r, time_to_peak_mfe,
    spread_close_at_entry, spread_close_at_exit, sl_distance_price.
    """
    pool = pd.read_parquet(POOL_PATH)
    cols = [
        "trade_id",
        "mfe_r",
        "time_to_peak_mfe",
        "spread_close_at_entry",
        "spread_close_at_exit",
        "sl_distance_price",
    ]
    return pool[cols].copy()


# ---------------------------------------------------------------------------
# Cost overlay per trade
# ---------------------------------------------------------------------------


def compute_lot_size(row: pd.Series) -> float:
    """Lot size at entry to achieve fixed risk (RISK_PER_TRADE × INITIAL_BAL)
    given the SL distance.

    lots = risk_amount_usd / (sl_distance_pips × pip_value_usd_per_lot)

    Pool's sl_distance_price is at SL=2.0 ATR (pool's policy). Replayed trades use
    SL=3.5 ATR; sl_distance is 3.5/2.0 × pool's value.
    """
    pair = row["pair"]
    sl_distance_price_pool = float(row["sl_distance_price"])
    sl_distance_price_replayed = sl_distance_price_pool * (WINNING_SL_MULT / POOL_SL_MULT)
    sl_distance_pips = sl_distance_price_replayed / pip_size(pair)
    pip_val = pip_value_usd_per_lot(pair)
    risk_amount_usd = RISK_PER_TRADE * INITIAL_BAL  # 0.005 × 100,000 = $500
    if sl_distance_pips <= 0 or pip_val <= 0:
        return 0.0
    return risk_amount_usd / (sl_distance_pips * pip_val)


def derive_tp1_hit_time(row: pd.Series) -> tuple[bool, pd.Timestamp | None]:
    """Returns (tp1_hit, tp1_time_utc).

    Per dispatch §3.1 req 3 + intent §5.3: TP1 hit iff pool mfe_r ≥ 1.75 (since
    pool MFE is in SL=2.0 units; replayed TP1 = +1R = +3.5 ATR = 1.75 pool-R).
    Timing: conservative upper bound from time_to_peak_mfe (peak ≥ TP1 crossing).
    """
    mfe_r = float(row["mfe_r"])
    if not np.isfinite(mfe_r) or mfe_r < TP1_HIT_THRESHOLD_POOL_MFE:
        return False, None
    bars_to_peak = float(row.get("time_to_peak_mfe", 0))
    if not np.isfinite(bars_to_peak) or bars_to_peak < 0:
        bars_to_peak = 0
    # H4 timeframe: bar = 4 hours
    tp1_time = row["entry_time"] + pd.Timedelta(hours=4 * bars_to_peak)
    # Floor to exit_time if derivation overshoots (defensive)
    if tp1_time > row["exit_time"]:
        tp1_time = row["exit_time"]
    return True, tp1_time


def apply_costs_to_trade(
    row: pd.Series,
    *,
    spread_mult: float,
    swap_on: bool,
    slip_per_fill_pips: float,
    swaps_yaml: dict,
    commission_on: bool = True,
) -> dict:
    """Compute the cost components for a single trade → returns dict of R-multiple
    adjustments and supporting metadata."""
    pair = row["pair"]
    lots = compute_lot_size(row)
    risk_amount_usd = RISK_PER_TRADE * INITIAL_BAL
    sl_distance_price_pool = float(row["sl_distance_price"])
    sl_distance_price_replayed = sl_distance_price_pool * (WINNING_SL_MULT / POOL_SL_MULT)

    tp1_hit, tp1_time = derive_tp1_hit_time(row)

    # Commission — always ON in the 30-cell grid (dispatch §6); OFF in baseline
    # reproduction assertion (dispatch §4: "unmodified pool").
    if commission_on:
        comm_usd = compute_commission_usd(lots) if lots > 0 else 0.0
    else:
        comm_usd = 0.0
    comm_r = comm_usd / risk_amount_usd

    # Slippage
    if slip_per_fill_pips > 0 and lots > 0:
        slip_pips_total, _n_fills = compute_slippage_pips(slip_per_fill_pips, tp1_hit)
        slip_r = (slip_pips_total * pip_size(pair)) / sl_distance_price_replayed
    else:
        slip_r = 0.0

    # Swap. The 5ers swap_long_points are negative for typical long-pays-interest
    # pairs (all 28 in the config). compute_swap_usd returns the signed cashflow:
    # negative = position loses USD overnight. To convert to a positive *cost* for
    # the `total_cost_r = swap_r + comm_r + ...` accumulator (where `adjusted_r
    # = final_r - total_cost_r`), negate the signed cashflow.
    swap_usd_signed = 0.0
    swap_detail = {}
    if swap_on and lots > 0:
        pair_yaml = swaps_yaml["pairs"].get(pair)
        if pair_yaml is None:
            swap_usd_signed = 0.0
        else:
            swap_long_points = float(pair_yaml["swap_long_points"])
            pip_val = pip_value_usd_per_lot(pair)
            swap_usd_signed, swap_detail = compute_swap_usd(
                entry_time_utc=row["entry_time"].to_pydatetime(),
                exit_time_utc=row["exit_time"].to_pydatetime(),
                tp1_hit_time_utc=tp1_time.to_pydatetime() if tp1_time is not None else None,
                swap_long_points=swap_long_points,
                pip_value_usd=pip_val,
                lots_original=lots,
            )
    # Cost = -cashflow. For typical long-pays pairs: swap_usd_signed < 0 →
    # swap_r > 0 (positive cost subtracted from final_r). For exotic positive-swap
    # pairs (none in the 28-pair set today): swap_usd_signed > 0 → swap_r < 0
    # (negative cost = credit, ADDED to final_r when subtracted). Correct in both cases.
    swap_r = (-swap_usd_signed) / risk_amount_usd if risk_amount_usd > 0 else 0.0

    # Spread multiplier
    se = float(row["spread_close_at_entry"])
    sx = float(row["spread_close_at_exit"])
    if spread_mult > 1.0 and sl_distance_price_replayed > 0:
        extra_spread_price = compute_extra_spread_price(se, sx, spread_mult)
        # Adverse: long entry pays ask (higher), exits at bid (lower). Cost is in price units.
        # Convert to R: extra_spread_r = extra_spread_price / sl_distance_replayed.
        extra_spread_r = extra_spread_price / sl_distance_price_replayed
    else:
        extra_spread_r = 0.0

    total_cost_r = swap_r + comm_r + slip_r + extra_spread_r
    # All four R-decrements are now positive costs (when long pays); subtract
    # from final_r downstream. Signed swap_usd kept for diagnostic.
    return dict(
        lots=lots,
        tp1_hit=tp1_hit,
        comm_r=comm_r,
        slip_r=slip_r,
        swap_r=swap_r,
        swap_usd_signed=swap_usd_signed,
        extra_spread_r=extra_spread_r,
        total_cost_r=total_cost_r,
        swap_n_rollovers=swap_detail.get("n_rollovers", 0),
        swap_n_friday=swap_detail.get("n_friday_rollovers", 0),
        zero_spread=(se <= 0 or sx <= 0),
    )


# ---------------------------------------------------------------------------
# Per-cell evaluation
# ---------------------------------------------------------------------------


def evaluate_cell(
    df: pd.DataFrame,
    *,
    spread_mult: float,
    swap_on: bool,
    slip_per_fill_pips: float,
    swaps_yaml: dict,
    commission_on: bool = True,
) -> dict:
    """Apply cost overlay across all trades; compute per-fold metrics."""
    costs = df.apply(
        apply_costs_to_trade,
        axis=1,
        spread_mult=spread_mult,
        swap_on=swap_on,
        slip_per_fill_pips=slip_per_fill_pips,
        swaps_yaml=swaps_yaml,
        commission_on=commission_on,
    )
    cost_df = pd.DataFrame(list(costs.values))
    df_adj = df.copy().reset_index(drop=True)
    cost_df = cost_df.reset_index(drop=True)
    df_adj["total_cost_r"] = cost_df["total_cost_r"]
    df_adj["adjusted_final_r"] = df_adj["final_r_replayed"] - df_adj["total_cost_r"]
    # _fold_metrics expects column "final_r" + sort by "signal_bar_time"
    df_adj["final_r"] = df_adj["adjusted_final_r"]

    # Per-fold metrics
    fold_results = {}
    for fold_id in sorted(df_adj["fold_id"].unique()):
        sub = df_adj[df_adj["fold_id"] == fold_id]
        m = _fold_metrics(sub)
        m["fold_id"] = int(fold_id)
        fold_results[int(fold_id)] = m

    # Search folds = 1..11, holdout = 0
    search_folds = [fold_results[i] for i in range(1, 12) if i in fold_results]
    holdout_fold = fold_results.get(0)

    rois = [f["roi"] for f in search_folds if f["n"] > 0]
    dds = [f["dd"] for f in search_folds if f["n"] > 0]
    ratios = [
        f["ratio"]
        for f in search_folds
        if f["n"] > 0 and np.isfinite(f["ratio"])
    ]
    signs = [f["sign"] for f in search_folds if f["n"] > 0]

    worst_roi = float(min(rois)) if rois else 0.0
    worst_dd = float(max(dds)) if dds else 0.0
    worst_ratio = float(min(ratios)) if ratios else 0.0
    mean_roi = float(np.mean(rois)) if rois else 0.0
    mean_dd = float(np.mean(dds)) if dds else 0.0
    sign_consistency = int(sum(1 for s in signs if s > 0))
    neg_folds = int(sum(1 for s in signs if s < 0))

    # Verdict
    if worst_dd >= 0.10:
        verdict = "FAIL_DD_HARD"
    elif worst_ratio < 2.0 or worst_roi < 0.05:
        verdict = "FAIL"
    elif worst_dd >= 0.08:
        verdict = "DD_WATCH"
    else:
        verdict = "PASS_DEPLOYABLE"

    # Zero-spread count
    n_zero_spread = int(cost_df["zero_spread"].sum())

    return dict(
        spread_mult=spread_mult,
        swap_on=swap_on,
        slip_per_fill_pips=slip_per_fill_pips,
        worst_fold_ratio=worst_ratio,
        worst_fold_dd=worst_dd,
        worst_fold_roi=worst_roi,
        mean_fold_roi=mean_roi,
        mean_fold_dd=mean_dd,
        sign_consistency=sign_consistency,
        neg_folds=neg_folds,
        holdout_roi=holdout_fold["roi"] if holdout_fold else 0.0,
        holdout_dd=holdout_fold["dd"] if holdout_fold else 0.0,
        holdout_ratio=holdout_fold["ratio"] if holdout_fold else 0.0,
        n_search_trades=int(sum(f["n"] for f in search_folds)),
        n_holdout_trades=int(holdout_fold["n"]) if holdout_fold else 0,
        n_zero_spread=n_zero_spread,
        verdict=verdict,
        # Per-fold detail for record-keeping
        per_fold=fold_results,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def diag_entry_time_assertion(df: pd.DataFrame) -> dict:
    """§5.2 — count entries with timestamp == 2200 UTC and == 2100 UTC.
    UTC H4 grid = 00/04/08/12/16/20 → both should be 0.
    """
    et = df["entry_time"]
    n_2200 = int(((et.dt.hour == 22) & (et.dt.minute == 0)).sum())
    n_2100 = int(((et.dt.hour == 21) & (et.dt.minute == 0)).sum())
    return dict(n_2200_utc=n_2200, n_2100_utc=n_2100)


def diag_zero_spread_count(df: pd.DataFrame) -> dict:
    """Count and report trades with zero spread on either entry or exit side.
    Expects df with spread_close_at_entry / spread_close_at_exit already merged in.
    """
    n_zero_entry = int((df["spread_close_at_entry"] <= 0).sum())
    n_zero_exit = int((df["spread_close_at_exit"] <= 0).sum())
    n_zero_either = int(
        ((df["spread_close_at_entry"] <= 0) | (df["spread_close_at_exit"] <= 0)).sum()
    )
    return dict(
        n_zero_entry=n_zero_entry,
        n_zero_exit=n_zero_exit,
        n_zero_either=n_zero_either,
        total=len(df),
        pct_zero_either=n_zero_either / max(len(df), 1) * 100.0,
    )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Arc 10 v3.0.2 cost-realism sweep")
    p.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the baseline cell (1×/swap-off/0-slip) and assert reproduction.",
    )
    args = p.parse_args(argv)

    COST_SWEEP_OUT.mkdir(parents=True, exist_ok=True)

    print("[cost_sweep] loading inputs…", flush=True)
    ledger = load_trade_ledger()
    pool_extras = load_pool_extras()
    df = ledger.merge(pool_extras, on="trade_id", how="left")
    swaps_yaml = load_swaps_yaml()

    print(f"[cost_sweep] {len(df)} trades loaded "
          f"({(df['fold_id'] > 0).sum()} search, {(df['fold_id'] == 0).sum()} holdout)", flush=True)

    # ── Diagnostics §5.2 ────────────────────────────────────────────────────
    entry_diag = diag_entry_time_assertion(df)
    print(
        f"[cost_sweep] §5.2 entry-time assertion: "
        f"n_at_2200_UTC={entry_diag['n_2200_utc']}, "
        f"n_at_2100_UTC={entry_diag['n_2100_utc']}",
        flush=True,
    )
    if entry_diag["n_2200_utc"] != 0 or entry_diag["n_2100_utc"] != 0:
        print("[cost_sweep] ⚠ §5.2 STOP CONDITION TRIGGERED — bar misalignment!", flush=True)
        return 1

    # ── Zero-spread count ───────────────────────────────────────────────────
    zs = diag_zero_spread_count(df)
    print(
        f"[cost_sweep] zero-spread trades: n_zero_either={zs['n_zero_either']} "
        f"({zs['pct_zero_either']:.2f}% of {zs['total']})",
        flush=True,
    )
    if zs["pct_zero_either"] > 1.0:
        print("[cost_sweep] ⚠ zero-spread > 1% — surface in report", flush=True)

    # ── Baseline reproduction assertion (no costs at all per dispatch §4) ──
    print("[cost_sweep] running baseline assertion (no costs — must reproduce UTC re-run)…", flush=True)
    baseline = evaluate_cell(
        df,
        spread_mult=1.0,
        swap_on=False,
        slip_per_fill_pips=0.0,
        swaps_yaml=swaps_yaml,
        commission_on=False,
    )
    print(
        f"[cost_sweep] baseline: worst_ratio={baseline['worst_fold_ratio']:.4f} "
        f"worst_roi={baseline['worst_fold_roi']:.6f} "
        f"worst_dd={baseline['worst_fold_dd']:.6f} "
        f"holdout_roi={baseline['holdout_roi']:.4f} "
        f"verdict={baseline['verdict']}",
        flush=True,
    )

    # Baseline assertion per dispatch §4 + §8
    EXPECTED_WORST_RATIO = 5.4185045
    EXPECTED_WORST_DD = 0.0922413
    EXPECTED_WORST_ROI = 0.2649054
    tol = 1e-4
    if (
        abs(baseline["worst_fold_ratio"] - EXPECTED_WORST_RATIO) > tol
        or abs(baseline["worst_fold_dd"] - EXPECTED_WORST_DD) > tol
        or abs(baseline["worst_fold_roi"] - EXPECTED_WORST_ROI) > tol
    ):
        print(
            f"[cost_sweep] BASELINE MISMATCH -- expected "
            f"({EXPECTED_WORST_RATIO}, {EXPECTED_WORST_DD}, {EXPECTED_WORST_ROI}), "
            f"got ({baseline['worst_fold_ratio']}, {baseline['worst_fold_dd']}, "
            f"{baseline['worst_fold_roi']}). STOP per dispatch section 8.",
            flush=True,
        )
        return 2
    print("[cost_sweep] OK baseline reproduces UTC re-run published numbers (within 1e-4).", flush=True)

    if args.baseline_only:
        return 0

    # ── Grid: 5 × 2 × 3 = 30 cells ─────────────────────────────────────────
    spread_mults = [1.0, 1.5, 2.0, 3.0, 4.0]
    swap_options = [False, True]  # swap=off, swap=on
    slip_options = [0.0, 0.5, 1.0]

    rows = []
    cell_idx = 0
    for sm in spread_mults:
        for swap_on in swap_options:
            for slip in slip_options:
                cell_idx += 1
                print(
                    f"[cost_sweep] cell {cell_idx}/30: "
                    f"spread×{sm}, swap={'ON' if swap_on else 'OFF'}, slip={slip}pip/fill",
                    flush=True,
                )
                res = evaluate_cell(
                    df,
                    spread_mult=sm,
                    swap_on=swap_on,
                    slip_per_fill_pips=slip,
                    swaps_yaml=swaps_yaml,
                )
                row = {
                    "cell": cell_idx,
                    "spread_mult": sm,
                    "swap_on": swap_on,
                    "slip_per_fill_pips": slip,
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
                }
                print(
                    f"   -> worst_ratio={row['worst_fold_ratio']:.3f} "
                    f"worst_roi={row['worst_fold_roi']*100:.2f}% "
                    f"worst_dd={row['worst_fold_dd']*100:.2f}% "
                    f"holdout_roi={row['holdout_roi']*100:.2f}% "
                    f"verdict={row['verdict']}",
                    flush=True,
                )
                rows.append(row)

    df_out = pd.DataFrame(rows)
    out_csv = COST_SWEEP_OUT / "grid_results.csv"
    df_out.to_csv(out_csv, index=False, lineterminator="\n")
    print(f"\n[cost_sweep] wrote {out_csv} ({len(df_out)} cells)", flush=True)

    # Summary
    diag_out = {
        "entry_time_assertion": entry_diag,
        "zero_spread": zs,
        "baseline_check": {
            "expected": {"worst_ratio": EXPECTED_WORST_RATIO, "worst_dd": EXPECTED_WORST_DD, "worst_roi": EXPECTED_WORST_ROI},
            "got": {
                "worst_ratio": baseline["worst_fold_ratio"],
                "worst_dd": baseline["worst_fold_dd"],
                "worst_roi": baseline["worst_fold_roi"],
            },
            "match": True,
        },
        "fx_constants": {
            "USDJPY": REF_USDJPY,
            "USDCAD": REF_USDCAD,
            "USDCHF": REF_USDCHF,
            "NZDUSD": REF_NZDUSD,
            "GBPUSD": REF_GBPUSD,
            "AUDUSD": REF_AUDUSD,
        },
    }
    (COST_SWEEP_OUT / "diagnostics.json").write_text(
        json.dumps(diag_out, indent=2, default=str), encoding="utf-8", newline="\n"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
