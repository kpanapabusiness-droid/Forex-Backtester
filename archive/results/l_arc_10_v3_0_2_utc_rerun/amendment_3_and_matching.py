"""Arc 10 v3.0.2 UTC rerun — Amendment 3 evaluation + trade-matching artefacts.

Mirrors scripts/l_arc_10_v3_0_2/amendment_3_addendum.py but consumes the UTC
rerun's Step 1 + Step 5 outputs under boundary_convention="utc". Also emits:

  - results/l_arc_10_v3_0_2_utc_rerun/trade_ledger_utc.parquet
        Admitted Top-1 trades (IS + holdout) with replayed final_r at SL=3.5
        + sl_partial_close_1r_runner_trail.

  - results/l_arc_10_v3_0_2_utc_rerun/trade_matching.csv
        Per-trade EET-vs-UTC diff on (pair, signal_bar_time ±1 H4 bar).
        Quantifies bar-shift reorganisation between conventions.

  - results/l_arc_10_v3_0_2_utc_rerun/amended_gate_classification_utc.json
        Canonical Amendment 3 verdict under UTC rerun numbers.

Per-fold path simulation logic is byte-equivalent to the v3.0.2 EET addendum:
calls core.sim.exit_policies.simulate_path with identical inputs from the
UTC rerun's pool + trade_paths.

Self-contained; lives under results/ so it doesn't pollute scripts/.
"""

from __future__ import annotations

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

INITIAL_BAL = 100_000.0
R_BASE = 0.005
SEARCH_START = "2010-01-01"
SEARCH_END = "2020-12-31"
HOLDOUT_START = "2021-01-01"
HOLDOUT_END = "2026-04-30"
N_SEARCH_FOLDS = 11
BOUNDARY_CONVENTION = "utc"
PAIR_SET = "28_FX_utc"

UTC_ARC_DIR = REPO_ROOT / "results" / "l_arc_10_v3_0_2_utc_rerun"
EET_ARC_DIR = REPO_ROOT / "results" / "l_arc_10_v3.0.2"


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


# ── Per-fold metrics (byte-equivalent to step_5._fold_metrics + v3.0.2 EET addendum) ──


def _equity_curve_at_risk(rs: np.ndarray, ts: np.ndarray, *, risk_pct: float = R_BASE) -> pd.Series:
    eq = INITIAL_BAL
    pts = []
    for r in rs:
        eq = eq + eq * risk_pct * r
        pts.append(eq)
    return pd.Series(pts, index=pd.DatetimeIndex(ts, tz="UTC"))


def _max_drawdown_pct(curve: pd.Series) -> float:
    if len(curve) == 0:
        return 0.0
    peak = curve.cummax()
    dd = (peak - curve) / peak.where(peak > 0, 1.0)
    return float(dd.max())


def _annualized_roi(curve: pd.Series, ts: np.ndarray) -> float:
    if len(curve) == 0:
        return 0.0
    total_ret = float(curve.iloc[-1] / INITIAL_BAL) - 1.0
    times = pd.to_datetime(ts)
    ann_yrs = max((times.max() - times.min()).total_seconds() / (365.25 * 86400.0), 0.25)
    try:
        return (1.0 + total_ret) ** (1.0 / ann_yrs) - 1.0
    except OverflowError:
        return 1e6 if total_ret > 0 else -0.99


def _simulate_admitted(
    admitted: pd.DataFrame, paths_by_trade: dict, *, sl_mult: float, exit_policy: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Replay exits; return per-trade R + times + the augmented per-trade frame."""
    new_rs = []
    bars_held = []
    times = []
    for _, row in admitted.iterrows():
        tid = int(row["trade_id"])
        tg = paths_by_trade.get(tid, pd.DataFrame())
        r_new, h_new = simulate_path(exit_policy, row, tg, sl_mult)
        new_rs.append(r_new)
        bars_held.append(h_new)
        times.append(row["signal_bar_time"])
    rs = np.array(new_rs, dtype=float)
    rs = np.clip(np.nan_to_num(rs, nan=0.0, posinf=20.0, neginf=-1.0), -1.5, 20.0)
    ts = np.array(times)
    augmented = admitted.copy()
    augmented["final_r_replayed"] = rs
    augmented["bars_held_replayed"] = bars_held
    return rs, ts, augmented


def _fold_stats(
    fold_id: int, rs: np.ndarray, ts: np.ndarray, *, risk_pct: float = R_BASE
) -> tuple[FoldStats, pd.Series]:
    if len(rs) == 0:
        return FoldStats(fold_id, 0, 0.0, 0.0, 0, 0.0), pd.Series(dtype=float)
    eq = _equity_curve_at_risk(rs, ts, risk_pct=risk_pct)
    roi = _annualized_roi(eq, ts)
    dd = _max_drawdown_pct(eq)
    per_day = compute_per_day_max_dd(eq, pair_set=PAIR_SET, boundary_convention=BOUNDARY_CONVENTION)
    breaches = int((per_day["day_max_dd_base_pct"].astype(float) >= 0.05).sum())
    ratio = (roi / dd) if dd > 1e-9 else (999.0 if roi > 0 else 0.0)
    if not np.isfinite(ratio):
        ratio = 999.0
    return FoldStats(fold_id, int(len(rs)), float(roi), float(dd), breaches, float(ratio)), eq


# ── Top-1 selection ───────────────────────────────────────────────────


def _select_winning_config(wfo_results_csv: Path) -> dict:
    df = pd.read_csv(wfo_results_csv).sort_values("search_worst_ratio", ascending=False).reset_index(drop=True)
    top = df.iloc[0].to_dict()
    expected = {
        "architecture": "A1",
        "sl_multiplier": 3.5,
        "exit_policy": "sl_partial_close_1r_runner_trail",
        "exposure": "unlimited",
        "archetype": "v_shape_recovery",
    }
    actual = {k: top.get(k) for k in expected}
    matches = all(
        (abs(float(actual[k]) - float(expected[k])) < 1e-9) if k == "sl_multiplier"
        else (actual[k] == expected[k])
        for k in expected
    )
    return {"top_1": top, "expected": expected, "matches_v3_0_2_winner": matches}


# ── Trade-matching ────────────────────────────────────────────────────


def _build_trade_matching(eet_pool: pd.DataFrame, utc_ledger: pd.DataFrame, *, h4_seconds: int = 4 * 3600) -> pd.DataFrame:
    """For each EET trade, find nearest UTC trade on (pair, signal_bar_time ±1 H4 bar).

    match_kind ∈ {exact, near_1bar, none}. Reports per-trade R deltas where both
    sides have replayed final_r (i.e. for the holdout window if available).
    """
    rows = []
    utc_by_pair: dict[str, pd.DataFrame] = {p: g.sort_values("signal_bar_time").reset_index(drop=True) for p, g in utc_ledger.groupby("pair")}
    matched_utc_ids: set = set()
    tol = pd.Timedelta(seconds=h4_seconds + 60)  # ±1 H4 bar with 1-min slop

    for _, e in eet_pool.iterrows():
        pair = e["pair"]
        e_t = pd.Timestamp(e["signal_bar_time"])
        g = utc_by_pair.get(pair)
        if g is None or len(g) == 0:
            rows.append({
                "eet_trade_id": int(e["trade_id"]),
                "eet_pair": pair,
                "eet_signal_bar_time": e_t,
                "eet_final_r": float(e.get("final_r", np.nan)),
                "match_kind": "none",
                "utc_trade_id": None,
                "utc_signal_bar_time": None,
                "utc_final_r": None,
                "bar_offset_h4": None,
                "r_delta": None,
            })
            continue
        u_times = pd.to_datetime(g["signal_bar_time"])
        diffs = (u_times - e_t).abs()
        i = int(diffs.idxmin())
        delta = diffs.iloc[i]
        if delta == pd.Timedelta(0):
            kind = "exact"
            offset = 0
        elif delta <= tol:
            kind = "near_1bar"
            offset = int(np.round((u_times.iloc[i] - e_t).total_seconds() / (4 * 3600)))
        else:
            kind = "none"
            offset = None
            i = None
        if i is not None:
            u_row = g.iloc[i]
            matched_utc_ids.add(int(u_row["trade_id"]))
            u_final = float(u_row["final_r_replayed"]) if "final_r_replayed" in u_row else float(u_row.get("final_r", np.nan))
            rows.append({
                "eet_trade_id": int(e["trade_id"]),
                "eet_pair": pair,
                "eet_signal_bar_time": e_t,
                "eet_final_r": float(e.get("final_r", np.nan)),
                "match_kind": kind,
                "utc_trade_id": int(u_row["trade_id"]),
                "utc_signal_bar_time": pd.Timestamp(u_row["signal_bar_time"]),
                "utc_final_r": u_final,
                "bar_offset_h4": offset,
                "r_delta": (u_final - float(e.get("final_r", np.nan))) if pd.notna(e.get("final_r", np.nan)) else None,
            })
        else:
            rows.append({
                "eet_trade_id": int(e["trade_id"]),
                "eet_pair": pair,
                "eet_signal_bar_time": e_t,
                "eet_final_r": float(e.get("final_r", np.nan)),
                "match_kind": "none",
                "utc_trade_id": None,
                "utc_signal_bar_time": None,
                "utc_final_r": None,
                "bar_offset_h4": None,
                "r_delta": None,
            })

    # Append UTC trades unmatched to any EET trade
    for _, u in utc_ledger.iterrows():
        if int(u["trade_id"]) in matched_utc_ids:
            continue
        rows.append({
            "eet_trade_id": None,
            "eet_pair": None,
            "eet_signal_bar_time": None,
            "eet_final_r": None,
            "match_kind": "utc_only",
            "utc_trade_id": int(u["trade_id"]),
            "utc_signal_bar_time": pd.Timestamp(u["signal_bar_time"]),
            "utc_final_r": float(u.get("final_r_replayed", u.get("final_r", np.nan))),
            "bar_offset_h4": None,
            "r_delta": None,
        })

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    pool_path = UTC_ARC_DIR / "step_1" / "pool.parquet"
    paths_path = UTC_ARC_DIR / "step_1" / "trade_paths.parquet"
    wfo_path = UTC_ARC_DIR / "step_5" / "wfo_results.csv"
    out_dir = UTC_ARC_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = pd.read_parquet(pool_path)
    pool["signal_bar_time"] = pd.to_datetime(pool["signal_bar_time"], utc=True)
    paths = pd.read_parquet(paths_path)
    paths_by_trade = {int(tid): g for tid, g in paths.groupby("trade_id")}
    print(f"[utc_rerun] pool={len(pool)} trades; paths={len(paths)} rows", flush=True)

    selection = _select_winning_config(wfo_path)
    top = selection["top_1"]
    sl_mult = float(top["sl_multiplier"])
    exit_policy = str(top["exit_policy"])
    print(
        f"[utc_rerun] Top-1: {top['architecture']} / SL={sl_mult} / {exit_policy} / "
        f"{top['exposure']} / cluster c{int(top['cluster_id'])} ({top['archetype']})",
        flush=True,
    )
    print(f"[utc_rerun] matches v3.0.2 winning architecture/exit/sl: {selection['matches_v3_0_2_winner']}", flush=True)

    # ── Replay 11 IS folds ────────────────────────────────────────────
    is_folds = _build_folds(SEARCH_START, SEARCH_END, N_SEARCH_FOLDS)
    is_fold_stats: list[FoldStats] = []
    is_fold_equity: list[pd.Series] = []
    ledger_rows: list[pd.DataFrame] = []

    for fi, (fs_, fe) in enumerate(is_folds, start=1):
        oos_mask = (pool["signal_bar_time"] >= fs_) & (pool["signal_bar_time"] <= fe)
        oos = pool[oos_mask].sort_values("signal_bar_time").reset_index(drop=True)
        rs, ts, aug = _simulate_admitted(oos, paths_by_trade, sl_mult=sl_mult, exit_policy=exit_policy)
        aug["fold_id"] = fi
        ledger_rows.append(aug)
        fs, eq = _fold_stats(fi, rs, ts)
        is_fold_stats.append(fs)
        if len(eq) > 0:
            is_fold_equity.append(eq)
        print(
            f"  fold {fi:2d}: n={fs.n_trades:4d} roi={fs.roi_pct:+.4%} "
            f"dd={fs.max_dd_pct:.4%} ratio={fs.roi_dd_ratio:.2f} "
            f"breaches@r_base={fs.days_breaching_daily_5pct}",
            flush=True,
        )

    # ── Cross-check against wfo_results.csv ───────────────────────────
    worst_roi_actual = min(f.roi_pct for f in is_fold_stats)
    worst_dd_actual = max(f.max_dd_pct for f in is_fold_stats)
    n_total_actual = sum(f.n_trades for f in is_fold_stats)
    expected_worst_roi = float(top["search_worst_roi"])
    expected_worst_dd = float(top["search_worst_dd"])
    expected_n_total = int(top["search_n_total"])
    xcheck_passed = (
        abs(worst_roi_actual - expected_worst_roi) <= 0.005
        and abs(worst_dd_actual - expected_worst_dd) <= 0.005
        and n_total_actual == expected_n_total
    )
    print(
        f"[xcheck] worst_roi addendum={worst_roi_actual:.6f} vs step_5={expected_worst_roi:.6f}  "
        f"worst_dd addendum={worst_dd_actual:.6f} vs step_5={expected_worst_dd:.6f}  "
        f"n_total addendum={n_total_actual} vs step_5={expected_n_total}  passed={xcheck_passed}",
        flush=True,
    )

    # ── Holdout fold ──────────────────────────────────────────────────
    h_win = (pd.Timestamp(HOLDOUT_START, tz="UTC"), pd.Timestamp(HOLDOUT_END, tz="UTC"))
    h_mask = (pool["signal_bar_time"] >= h_win[0]) & (pool["signal_bar_time"] <= h_win[1])
    h_oos = pool[h_mask].sort_values("signal_bar_time").reset_index(drop=True)
    h_rs, h_ts, h_aug = _simulate_admitted(h_oos, paths_by_trade, sl_mult=sl_mult, exit_policy=exit_policy)
    h_aug["fold_id"] = 0  # holdout sentinel
    ledger_rows.append(h_aug)
    holdout_at_r_base, holdout_eq_base = _fold_stats(N_SEARCH_FOLDS + 1, h_rs, h_ts)
    print(
        f"  holdout : n={holdout_at_r_base.n_trades:4d} roi={holdout_at_r_base.roi_pct:+.4%} "
        f"dd={holdout_at_r_base.max_dd_pct:.4%} ratio={holdout_at_r_base.roi_dd_ratio:.2f} "
        f"breaches@r_base={holdout_at_r_base.days_breaching_daily_5pct}",
        flush=True,
    )

    # ── Write trade ledger ────────────────────────────────────────────
    ledger = pd.concat(ledger_rows, ignore_index=True)
    ledger_keep = ledger[[
        "trade_id", "pair", "signal_bar_time", "entry_time", "exit_time",
        "entry_price", "sl_at_entry_price", "exit_price", "exit_reason",
        "bars_held", "final_r", "final_r_replayed", "bars_held_replayed", "fold_id",
    ]].copy()
    ledger_keep.to_parquet(
        out_dir / "trade_ledger_utc.parquet", engine="pyarrow", compression="snappy", index=False,
    )
    print(f"[ledger] {len(ledger_keep)} trades -> trade_ledger_utc.parquet", flush=True)

    # ── Chained DD ────────────────────────────────────────────────────
    chained_eq = stitch_per_fold_oos_equity(is_fold_equity + [holdout_eq_base], starting_balance=INITIAL_BAL)
    chained_dd = compute_chained_max_dd_from_continuous_equity(chained_eq)
    per_day_chained = compute_per_day_max_dd(chained_eq, pair_set=PAIR_SET, boundary_convention=BOUNDARY_CONVENTION)
    per_day_chained_path = out_dir / "per_day_max_dd_base_utc.parquet"
    per_day_chained.to_parquet(per_day_chained_path, engine="pyarrow", compression="snappy", index=False)
    print(f"[chained] DD across IS + holdout = {chained_dd:.4%}", flush=True)

    # ── Amendment 3 scaling ───────────────────────────────────────────
    worst_dd_base = max(f.max_dd_pct for f in is_fold_stats)
    scaling = compute_scaling_factors(worst_dd_base, r_base=R_BASE)
    print(
        f"[scaling] worst_fold_dd_base={worst_dd_base:.4%} -> "
        f"k_safe={scaling.k_safe:.4f} (r_safe={scaling.r_safe_pct:.4%}) "
        f"k_hard={scaling.k_hard:.4f} (r_hard={scaling.r_hard_pct:.4%})",
        flush=True,
    )

    # Holdout reruns at scaled risk
    holdout_safe, holdout_eq_safe = _fold_stats(
        N_SEARCH_FOLDS + 2, h_rs, h_ts, risk_pct=scaling.r_safe_pct,
    )
    holdout_hard, holdout_eq_hard = _fold_stats(
        N_SEARCH_FOLDS + 3, h_rs, h_ts, risk_pct=scaling.r_hard_pct,
    )

    # Canonical Amendment 3.1 classification
    result = classify_amended_fold_stats(
        folds=tuple(is_fold_stats),
        chained_max_dd_base_pct=float(chained_dd),
        per_day_max_dd_df=per_day_chained,
        holdout_stats_at_r_safe=holdout_safe,
        holdout_stats_at_r_hard=holdout_hard,
        sizing_convention="reset_floor",
        accept_equity_pct=False,
        r_base=R_BASE,
        causal_audit_clean=True,
    )

    payload = {
        "arc_name": "l_arc_10_v3.0.2_utc_rerun",
        "boundary_convention": BOUNDARY_CONVENTION,
        "r_base_pct": R_BASE,
        "winning_config_top_1_utc": {
            "cluster_id": int(top["cluster_id"]),
            "architecture": str(top["architecture"]),
            "sl_multiplier": float(top["sl_multiplier"]),
            "exit_policy": str(top["exit_policy"]),
            "exposure": str(top["exposure"]),
            "archetype": str(top["archetype"]),
        },
        "matches_v3_0_2_winner": selection["matches_v3_0_2_winner"],
        "cross_check_vs_step_5": {
            "worst_fold_roi_addendum": worst_roi_actual,
            "worst_fold_roi_step_5": expected_worst_roi,
            "worst_fold_dd_addendum": worst_dd_actual,
            "worst_fold_dd_step_5": expected_worst_dd,
            "n_total_addendum": n_total_actual,
            "n_total_step_5": expected_n_total,
            "passed": bool(xcheck_passed),
        },
        "is_folds": [asdict(f) for f in is_fold_stats],
        "holdout_at_r_base": asdict(holdout_at_r_base),
        "holdout_at_r_safe": asdict(holdout_safe),
        "holdout_at_r_hard": asdict(holdout_hard),
        "chained_max_dd_base_pct": float(chained_dd),
        "scaling": {
            "k_safe": scaling.k_safe,
            "k_hard": scaling.k_hard,
            "r_safe_pct": scaling.r_safe_pct,
            "r_hard_pct": scaling.r_hard_pct,
            "r_safe_intrinsic_pct": scaling.r_safe_intrinsic_pct,
            "r_hard_intrinsic_pct": scaling.r_hard_intrinsic_pct,
            "r_safe_capped_at_rmax": bool(scaling.r_safe_capped_at_rmax),
            "r_hard_capped_at_rmax": bool(scaling.r_hard_capped_at_rmax),
            "scalable_to_safe": bool(scaling.scalable_to_safe),
            "scalable_to_hard": bool(scaling.scalable_to_hard),
        },
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
    classification_path = out_dir / "amended_gate_classification_utc.json"
    with classification_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"[amendment_3] verdict: {result.verdict.value.upper()}", flush=True)
    print(f"[amendment_3] reason : {result.reason}", flush=True)

    # ── Trade matching ────────────────────────────────────────────────
    eet_pool = pd.read_parquet(EET_ARC_DIR / "step_1" / "pool.parquet")
    eet_pool["signal_bar_time"] = pd.to_datetime(eet_pool["signal_bar_time"], utc=True)
    matching = _build_trade_matching(eet_pool, ledger_keep)
    matching.to_csv(out_dir / "trade_matching.csv", index=False, lineterminator="\n")

    # Match summary
    eet_total = int((matching["eet_trade_id"].notna()).sum())
    exact = int((matching["match_kind"] == "exact").sum())
    near = int((matching["match_kind"] == "near_1bar").sum())
    none = int((matching["match_kind"] == "none").sum())
    utc_only = int((matching["match_kind"] == "utc_only").sum())
    print(
        f"[matching] EET trades={eet_total} (exact={exact} near={near} none={none}); UTC-only={utc_only}",
        flush=True,
    )

    # Per-pair summary
    per_pair = matching[matching["eet_trade_id"].notna()].groupby("eet_pair")["match_kind"].value_counts().unstack(fill_value=0)
    print("[matching] per-pair breakdown:", flush=True)
    print(per_pair.to_string(), flush=True)

    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
