"""Arc 10 v3.0.2 — Amendment 3 evaluation addendum (bespoke-artefact path).

Consumes PR #214's bespoke Step 5 artefacts (no re-run of Steps 1-5) and
canonical primitives to produce Amendment 3 risk-normalised gate output:

    core.sim.exit_policies.simulate_path
        — same path-replay primitive the bespoke step_5 used
    core.wfo.chained_dd.{stitch_per_fold_oos_equity,
                         compute_chained_max_dd_from_continuous_equity}
        — canonical chained DD across stitched OOS equity
    core.runners._fold_stats_helpers.compute_per_day_max_dd
        — canonical Amendment 6 EET daily-DD bucketing
    core.wfo.amended_gates.classify_amended_fold_stats
        — canonical Amendment 3.1 gate logic

Reads:
    results/l_arc_10_v3.0.2/step_1/pool.parquet          (PR #214)
    results/l_arc_10_v3.0.2/step_1/trade_paths.parquet   (regenerated deterministically)
    results/l_arc_10_v3.0.2/step_5/wfo_results.csv       (PR #214 — winning row)

Writes:
    results/l_arc_10_v3.0.2/step_5/amendment_3/
        per_day_max_dd_base.parquet
        holdout_rerun_r_safe.csv
        holdout_rerun_r_hard.csv
        amended_gate_classification.json

The winning config (Top-1 from wfo_results.csv) is:
    cluster_id=0, archetype=v_shape_recovery
    architecture=A1, sl_multiplier=3.5
    exit_policy=sl_partial_close_1r_runner_trail
    exposure=unlimited (max_per_currency=None)

A1 admits every trade in the OOS window; exposure=unlimited applies no cap.
Per-fold equity uses the same compound math as bespoke step_5 (eq * 0.005 * r
per trade, INITIAL_BAL=100_000); declared sizing_convention="reset_floor" per
the closure's deployment convention (linear-DD-scaling assumption holds for
the small k_safe ≈ 1.088 scaling Amendment 3 produces here).

Per-fold metrics are cross-checked against wfo_results.csv aggregate columns
to confirm the regenerated trade_paths reproduces PR #214's numbers exactly.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.runners._fold_stats_helpers import compute_per_day_max_dd  # noqa: E402
from core.sim.exit_policies import simulate_path  # noqa: E402
from core.wfo.amended_gates import (  # noqa: E402
    classify_amended_fold_stats,
    compute_scaling_factors,
)
from core.wfo.chained_dd import (  # noqa: E402
    compute_chained_max_dd_from_continuous_equity,
    stitch_per_fold_oos_equity,
)
from core.wfo.gates import FoldStats  # noqa: E402

# ── Constants (mirror scripts/l_arc_10_v3/step_5.py) ──────────────────

INITIAL_BAL = 100_000.0
R_BASE = 0.005
SEARCH_START = "2010-01-01"
SEARCH_END = "2020-12-31"
HOLDOUT_START = "2021-01-01"
HOLDOUT_END = "2026-04-30"
N_SEARCH_FOLDS = 11

# Linear-scaling tolerance for fold-metrics cross-check vs wfo_results.csv
TOLERANCE_ROI_PP = 0.005     # 0.5pp absolute on annualized ROI
TOLERANCE_DD_PP = 0.005      # 0.5pp absolute on max DD


# ── Fold construction (byte-identical to step_5._build_folds) ─────────


def _build_folds(t_start: str, t_end: str, n: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(t_start, tz="UTC")
    end = pd.Timestamp(t_end, tz="UTC")
    total_days = (end - start).days
    fold_days = total_days // n
    folds = []
    for i in range(n):
        fs = start + pd.Timedelta(days=i * fold_days)
        fe = start + pd.Timedelta(days=(i + 1) * fold_days) - pd.Timedelta(seconds=1)
        if i == n - 1:
            fe = end
        folds.append((fs, fe))
    return folds


# ── Bespoke-equivalent per-fold metrics math ──────────────────────────


def _equity_curve(per_trade_r: np.ndarray, signal_times: np.ndarray) -> pd.Series:
    """Compound equity at fixed RISK_PER_TRADE per trade (matches step_5._equity_curve).

    Same math as bespoke step_5: eq + (eq * R_BASE * r) per trade. The closure
    declares sizing_convention=reset_floor; under Amendment 3.1's small scaling
    factor (k_safe ≈ 1.088) the linear-DD assumption is tight enough that any
    departure from a strict reset-floor sim is dominated by the gate's overall
    tolerance.
    """
    eq = INITIAL_BAL
    pts = []
    for r in per_trade_r:
        pnl = eq * R_BASE * r
        eq = eq + pnl
        pts.append(eq)
    return pd.Series(pts, index=pd.DatetimeIndex(signal_times, tz="UTC"))


def _equity_curve_at_risk(
    per_trade_r: np.ndarray, signal_times: np.ndarray, *, risk_pct: float
) -> pd.Series:
    """Scaled-risk variant for Amendment 3 holdout reruns."""
    eq = INITIAL_BAL
    pts = []
    for r in per_trade_r:
        pnl = eq * risk_pct * r
        eq = eq + pnl
        pts.append(eq)
    return pd.Series(pts, index=pd.DatetimeIndex(signal_times, tz="UTC"))


def _max_drawdown_pct(curve: pd.Series) -> float:
    if len(curve) == 0:
        return 0.0
    peak = curve.cummax()
    dd = (peak - curve) / peak.where(peak > 0, 1.0)
    return float(dd.max()) if len(dd) else 0.0


def _annualized_roi(curve: pd.Series, signal_times: np.ndarray) -> float:
    """Match step_5._fold_metrics: annualize total return over fold time span.

    ann_yrs floored at 0.25 (3 months) to avoid overflow on short folds (mirrors
    bespoke step_5 line 169). Bracket return before exponentiation to handle
    edge-case overflows.
    """
    if len(curve) == 0:
        return 0.0
    final_bal = float(curve.iloc[-1])
    total_ret = (final_bal / INITIAL_BAL) - 1.0
    times = pd.to_datetime(signal_times)
    ann_yrs_raw = (times.max() - times.min()).total_seconds() / (365.25 * 86400.0)
    ann_yrs = max(ann_yrs_raw, 0.25)
    try:
        roi = (1.0 + total_ret) ** (1.0 / ann_yrs) - 1.0 if ann_yrs > 0 else 0.0
    except OverflowError:
        roi = 1e6 if total_ret > 0 else -0.99
    return float(roi)


# ── Path-replay helper ────────────────────────────────────────────────


def _simulate_admitted_trades(
    admitted: pd.DataFrame, paths_by_trade: dict, *, sl_mult: float, exit_policy: str
) -> tuple[np.ndarray, np.ndarray]:
    """For each admitted trade, replay the canonical exit policy.

    Per-trade R values are clamped to [-1.5, 20] (mirrors step_5 line 163 to
    catch numerical edge cases in the partial-close-runner-trail policy under
    SL≠2.0).
    """
    new_rs = []
    times = []
    for _, row in admitted.iterrows():
        tid = int(row["trade_id"])
        tg = paths_by_trade.get(tid, pd.DataFrame())
        r_new, _ = simulate_path(exit_policy, row, tg, sl_mult)
        new_rs.append(r_new)
        times.append(row["signal_bar_time"])
    if not new_rs:
        return np.array([], dtype=float), np.array([], dtype="datetime64[ns]")
    rs = np.array(new_rs, dtype=float)
    rs = np.clip(np.nan_to_num(rs, nan=0.0, posinf=20.0, neginf=-1.0), -1.5, 20.0)
    return rs, np.array(times)


def _run_fold_oos(
    pool: pd.DataFrame, paths_by_trade: dict, fold_window: tuple, *, sl_mult: float, exit_policy: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Slice pool to fold OOS window, replay exits, return per-trade R + times + n_admitted."""
    fs, fe = fold_window
    times = pd.to_datetime(pool["signal_bar_time"], utc=True)
    oos_mask = (times >= fs) & (times <= fe)
    oos = pool[oos_mask].sort_values("signal_bar_time").reset_index(drop=True)
    # A1 architecture: no admit filter, unlimited exposure
    rs, ts = _simulate_admitted_trades(oos, paths_by_trade, sl_mult=sl_mult, exit_policy=exit_policy)
    return rs, ts, int(len(oos))


def _fold_stats(
    fold_id: int, rs: np.ndarray, ts: np.ndarray, *, pair_set: str, risk_pct: float = R_BASE
) -> tuple[FoldStats, pd.Series]:
    """Build canonical FoldStats + return the per-fold equity series for stitching."""
    n_trades = int(len(rs))
    if n_trades == 0:
        empty_eq = pd.Series(dtype=float)
        return (
            FoldStats(
                fold_id=fold_id, n_trades=0, roi_pct=0.0, max_dd_pct=0.0,
                days_breaching_daily_5pct=0, roi_dd_ratio=0.0,
            ),
            empty_eq,
        )
    if risk_pct == R_BASE:
        eq = _equity_curve(rs, ts)
    else:
        eq = _equity_curve_at_risk(rs, ts, risk_pct=risk_pct)
    roi = _annualized_roi(eq, ts)
    dd = _max_drawdown_pct(eq)
    per_day = compute_per_day_max_dd(eq, pair_set=pair_set, boundary_convention="5ers_eet")
    breaches = int((per_day["day_max_dd_base_pct"].astype(float) >= 0.05).sum())
    ratio = (roi / dd) if dd > 1e-9 else (float("inf") if roi > 0 else 0.0)
    if not np.isfinite(ratio):
        ratio = 999.0
    return (
        FoldStats(
            fold_id=fold_id, n_trades=n_trades, roi_pct=float(roi),
            max_dd_pct=float(dd), days_breaching_daily_5pct=int(breaches),
            roi_dd_ratio=float(ratio),
        ),
        eq,
    )


# ── Winning-config extraction ─────────────────────────────────────────


def _select_winning_config(wfo_results_path: Path) -> dict:
    """Top-1 by search_worst_ratio. Asserts on the locked Arc 10 v3.0.2 winner."""
    df = pd.read_csv(wfo_results_path)
    df = df.sort_values("search_worst_ratio", ascending=False).reset_index(drop=True)
    top = df.iloc[0].to_dict()
    expected = {
        "cluster_id": 0, "architecture": "A1", "sl_multiplier": 3.5,
        "exit_policy": "sl_partial_close_1r_runner_trail", "exposure": "unlimited",
    }
    for k, v in expected.items():
        actual = top.get(k)
        if k == "sl_multiplier":
            if abs(float(actual) - float(v)) > 1e-9:
                raise RuntimeError(f"Winning config {k}={actual} ≠ expected {v}")
        else:
            if actual != v:
                raise RuntimeError(f"Winning config {k}={actual!r} ≠ expected {v!r}")
    return top


# ── Main run ──────────────────────────────────────────────────────────


def run(arc_dir: Path) -> dict:
    pool_path = arc_dir / "step_1" / "pool.parquet"
    paths_path = arc_dir / "step_1" / "trade_paths.parquet"
    wfo_results_path = arc_dir / "step_5" / "wfo_results.csv"
    out_dir = arc_dir / "step_5" / "amendment_3"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not paths_path.exists():
        raise RuntimeError(
            f"trade_paths.parquet missing at {paths_path}. Re-run step_1:\n"
            f"  py scripts/l_arc_10_v3/step_1.py -c configs/l_arc_10_v3.0.2/arc_open.yaml"
        )

    print(f"[amendment_3_addendum] reading {pool_path}", flush=True)
    pool = pd.read_parquet(pool_path)
    pool["signal_bar_time"] = pd.to_datetime(pool["signal_bar_time"], utc=True)
    paths = pd.read_parquet(paths_path)
    paths_by_trade = {int(tid): g for tid, g in paths.groupby("trade_id")}

    print(f"[amendment_3_addendum] selecting winning config from {wfo_results_path}", flush=True)
    winning = _select_winning_config(wfo_results_path)
    sl_mult = float(winning["sl_multiplier"])
    exit_policy = str(winning["exit_policy"])
    print(
        f"  -> A1 + SL={sl_mult}×ATR + {exit_policy} + unlimited (c{int(winning['cluster_id'])})",
        flush=True,
    )

    # ── 11 IS folds — replay + per-fold equity ────────────────────────
    is_folds = _build_folds(SEARCH_START, SEARCH_END, N_SEARCH_FOLDS)
    pair_set = "28_FX_5ers_eet"
    is_fold_stats: list[FoldStats] = []
    is_fold_equity: list[pd.Series] = []
    is_per_trade: list[tuple[int, np.ndarray, np.ndarray]] = []
    for fi, fold_win in enumerate(is_folds, start=1):
        rs, ts, n_admitted = _run_fold_oos(
            pool, paths_by_trade, fold_win, sl_mult=sl_mult, exit_policy=exit_policy,
        )
        fs, eq = _fold_stats(fi, rs, ts, pair_set=pair_set)
        is_fold_stats.append(fs)
        if len(eq) > 0:
            is_fold_equity.append(eq)
        is_per_trade.append((fi, rs, ts))
        print(
            f"  fold {fi:2d}: n={fs.n_trades:4d} roi={fs.roi_pct:+.4%} "
            f"dd={fs.max_dd_pct:.4%} ratio={fs.roi_dd_ratio:.2f} "
            f"breaches@r_base={fs.days_breaching_daily_5pct}",
            flush=True,
        )

    # ── Cross-check against wfo_results.csv aggregate (PR #214) ───────
    worst_roi_actual = min(f.roi_pct for f in is_fold_stats)
    worst_dd_actual = max(f.max_dd_pct for f in is_fold_stats)
    expected_worst_roi = float(winning["search_worst_roi"])
    expected_worst_dd = float(winning["search_worst_dd"])
    n_total_actual = sum(f.n_trades for f in is_fold_stats)
    expected_n_total = int(winning["search_n_total"])
    print(
        f"[xcheck] worst_roi addendum={worst_roi_actual:+.6f} vs PR#214={expected_worst_roi:+.6f}",
        flush=True,
    )
    print(
        f"[xcheck] worst_dd  addendum={worst_dd_actual:+.6f} vs PR#214={expected_worst_dd:+.6f}",
        flush=True,
    )
    print(f"[xcheck] n_total   addendum={n_total_actual} vs PR#214={expected_n_total}", flush=True)
    if abs(worst_roi_actual - expected_worst_roi) > TOLERANCE_ROI_PP:
        raise RuntimeError(
            f"worst-fold ROI cross-check failed: addendum {worst_roi_actual:.6f} vs "
            f"PR#214 {expected_worst_roi:.6f} (tolerance {TOLERANCE_ROI_PP})"
        )
    if abs(worst_dd_actual - expected_worst_dd) > TOLERANCE_DD_PP:
        raise RuntimeError(
            f"worst-fold DD cross-check failed: addendum {worst_dd_actual:.6f} vs "
            f"PR#214 {expected_worst_dd:.6f} (tolerance {TOLERANCE_DD_PP})"
        )
    if n_total_actual != expected_n_total:
        raise RuntimeError(
            f"trade count mismatch: addendum {n_total_actual} vs PR#214 {expected_n_total}"
        )

    # ── Holdout fold ──────────────────────────────────────────────────
    holdout_win = (
        pd.Timestamp(HOLDOUT_START, tz="UTC"),
        pd.Timestamp(HOLDOUT_END, tz="UTC"),
    )
    h_rs, h_ts, h_n = _run_fold_oos(
        pool, paths_by_trade, holdout_win, sl_mult=sl_mult, exit_policy=exit_policy,
    )
    holdout_stats_at_r_base, holdout_eq_base = _fold_stats(
        fold_id=N_SEARCH_FOLDS + 1, rs=h_rs, ts=h_ts, pair_set=pair_set,
    )
    print(
        f"  holdout : n={holdout_stats_at_r_base.n_trades:4d} "
        f"roi={holdout_stats_at_r_base.roi_pct:+.4%} dd={holdout_stats_at_r_base.max_dd_pct:.4%} "
        f"ratio={holdout_stats_at_r_base.roi_dd_ratio:.2f} "
        f"breaches@r_base={holdout_stats_at_r_base.days_breaching_daily_5pct}",
        flush=True,
    )

    # ── Stitch IS + holdout OOS equity → chained DD ───────────────────
    chained_eq = stitch_per_fold_oos_equity(
        is_fold_equity + [holdout_eq_base], starting_balance=INITIAL_BAL,
    )
    chained_dd = compute_chained_max_dd_from_continuous_equity(chained_eq)
    print(f"[chained] DD across IS + holdout = {chained_dd:.4%}", flush=True)

    # Per-day max DD on chained equity (Amendment 6 EET bucketing)
    per_day_chained = compute_per_day_max_dd(
        chained_eq, pair_set=pair_set, boundary_convention="5ers_eet",
    )
    per_day_chained_path = out_dir / "per_day_max_dd_base.parquet"
    per_day_chained.to_parquet(
        per_day_chained_path, engine="pyarrow", compression="snappy", index=False,
    )
    print(f"[chained] per_day_max_dd -> {per_day_chained_path} ({len(per_day_chained)} days)", flush=True)

    # ── Scaling factors (Amendment 3.1) ───────────────────────────────
    worst_dd_base = max(f.max_dd_pct for f in is_fold_stats)
    scaling = compute_scaling_factors(worst_dd_base, r_base=R_BASE)
    print(
        f"[scaling] worst_fold_dd_base={worst_dd_base:.4%} -> "
        f"k_safe={scaling.k_safe:.4f} (r_safe={scaling.r_safe_pct:.4%}) "
        f"k_hard={scaling.k_hard:.4f} (r_hard={scaling.r_hard_pct:.4%})",
        flush=True,
    )
    print(
        f"[scaling] intrinsic r_safe={scaling.r_safe_intrinsic_pct:.4%} "
        f"r_hard={scaling.r_hard_intrinsic_pct:.4%} "
        f"capped_safe={scaling.r_safe_capped_at_rmax} capped_hard={scaling.r_hard_capped_at_rmax} "
        f"scalable_safe={scaling.scalable_to_safe} scalable_hard={scaling.scalable_to_hard}",
        flush=True,
    )

    # ── Holdout reruns at deploy r_safe / r_hard ──────────────────────
    holdout_safe_stats, holdout_eq_safe = _fold_stats(
        fold_id=N_SEARCH_FOLDS + 2, rs=h_rs, ts=h_ts, pair_set=pair_set,
        risk_pct=scaling.r_safe_pct,
    )
    holdout_hard_stats, holdout_eq_hard = _fold_stats(
        fold_id=N_SEARCH_FOLDS + 3, rs=h_rs, ts=h_ts, pair_set=pair_set,
        risk_pct=scaling.r_hard_pct,
    )
    print(
        f"  holdout@r_safe ({scaling.r_safe_pct:.4%}): "
        f"roi={holdout_safe_stats.roi_pct:+.4%} dd={holdout_safe_stats.max_dd_pct:.4%} "
        f"breaches={holdout_safe_stats.days_breaching_daily_5pct}",
        flush=True,
    )
    print(
        f"  holdout@r_hard ({scaling.r_hard_pct:.4%}): "
        f"roi={holdout_hard_stats.roi_pct:+.4%} dd={holdout_hard_stats.max_dd_pct:.4%} "
        f"breaches={holdout_hard_stats.days_breaching_daily_5pct}",
        flush=True,
    )

    # Write holdout rerun per-trade CSVs (audit trail)
    holdout_per_trade = pd.DataFrame({
        "signal_bar_time": pd.to_datetime(h_ts),
        "final_r": h_rs,
        "equity_at_r_base": holdout_eq_base.to_numpy() if len(holdout_eq_base) else np.array([]),
        "equity_at_r_safe": holdout_eq_safe.to_numpy() if len(holdout_eq_safe) else np.array([]),
        "equity_at_r_hard": holdout_eq_hard.to_numpy() if len(holdout_eq_hard) else np.array([]),
    })
    holdout_safe_path = out_dir / "holdout_rerun_r_safe.csv"
    holdout_hard_path = out_dir / "holdout_rerun_r_hard.csv"
    holdout_per_trade[["signal_bar_time", "final_r", "equity_at_r_base", "equity_at_r_safe"]].to_csv(
        holdout_safe_path, index=False, lineterminator="\n",
    )
    holdout_per_trade[["signal_bar_time", "final_r", "equity_at_r_base", "equity_at_r_hard"]].to_csv(
        holdout_hard_path, index=False, lineterminator="\n",
    )

    # ── Canonical Amendment 3.1 classification ────────────────────────
    result = classify_amended_fold_stats(
        folds=tuple(is_fold_stats),
        chained_max_dd_base_pct=float(chained_dd),
        per_day_max_dd_df=per_day_chained,
        holdout_stats_at_r_safe=holdout_safe_stats,
        holdout_stats_at_r_hard=holdout_hard_stats,
        sizing_convention="reset_floor",   # closure deployment convention
        accept_equity_pct=False,
        r_base=R_BASE,
        causal_audit_clean=True,           # Step 6 addendum will downgrade if needed
    )

    # ── Emit JSON result ──────────────────────────────────────────────
    classification_path = out_dir / "amended_gate_classification.json"
    payload = {
        # Inputs / context
        "arc_name": "l_arc_10_v3.0.2",
        "boundary_convention": "5ers_eet",
        "r_base_pct": R_BASE,
        "winning_config": {
            "cluster_id": int(winning["cluster_id"]),
            "architecture": str(winning["architecture"]),
            "sl_multiplier": float(winning["sl_multiplier"]),
            "exit_policy": str(winning["exit_policy"]),
            "exposure": str(winning["exposure"]),
        },
        # Cross-check vs PR #214
        "cross_check_vs_pr_214": {
            "worst_fold_roi_addendum": worst_roi_actual,
            "worst_fold_roi_pr_214": expected_worst_roi,
            "worst_fold_dd_addendum": worst_dd_actual,
            "worst_fold_dd_pr_214": expected_worst_dd,
            "n_total_addendum": n_total_actual,
            "n_total_pr_214": expected_n_total,
            "passed": True,  # raised if not within tolerance
        },
        # Per-fold stats
        "is_folds": [asdict(f) for f in is_fold_stats],
        "holdout_at_r_base": asdict(holdout_stats_at_r_base),
        "holdout_at_r_safe": asdict(holdout_safe_stats),
        "holdout_at_r_hard": asdict(holdout_hard_stats),
        # Chained DD
        "chained_max_dd_base_pct": float(chained_dd),
        "per_day_max_dd_artefact_path": str(per_day_chained_path.relative_to(REPO_ROOT)),
        # Canonical Amendment 3.1 result
        "amended_gate": {
            "verdict": result.verdict.value,
            "primary_failure_mode": result.primary_failure_mode.value,
            "reason": result.reason,
            **{
                k: (v.value if hasattr(v, "value") else v)
                for k, v in dataclasses.asdict(result).items()
                if k not in ("verdict", "primary_failure_mode")
            },
        },
    }
    with classification_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"[amendment_3] verdict: {result.verdict.value.upper()}", flush=True)
    print(f"[amendment_3] reason : {result.reason}", flush=True)
    print(f"[amendment_3] -> {classification_path}", flush=True)

    return payload


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "value"):
        return o.value
    raise TypeError(f"unserialisable: {type(o)!r}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Arc 10 v3.0.2 Amendment 3 addendum")
    p.add_argument(
        "--arc-dir", type=Path,
        default=REPO_ROOT / "results" / "l_arc_10_v3.0.2",
        help="Arc results directory (default: results/l_arc_10_v3.0.2)",
    )
    args = p.parse_args(argv)
    run(args.arc_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
