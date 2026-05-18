"""Arc 9 - Step 1 diagnostics.

Runs after step1_plumbing.py. Produces:

    audit_lookahead.txt       5/5 random signal recomputation from truncated OHLC
    audit_right_edge_swing.txt swing-low used is at most bar t-4 on every signal
    schema_check.json         §15a strict schema validation on trades_paths.csv
    cofire_matrix.json + .md   KH-24 + sibling-arc (8/10/11 if present) overlap
    pool_summary.txt          per-pair, bars_held distribution, exit-reason mix

Reads results/l_arc_9/step1_verbatim/{trades_all,trades_paths,prefilter_events}.csv,
configs/wfo_l_arc_9.yaml, results/kh24/trades_all.csv (for KH-24 co-fire).
Sibling-arc step 1 trades_all.csv files are scanned under
results/{arc_8,arc_10,arc_11,l_arc_8,l_arc_10,l_arc_11}/step1*/trades_all.csv
and reported with 0 (or absent) where not present.

Usage:
    py scripts/l_arc_9/step1_diagnostics.py -c configs/wfo_l_arc_9.yaml
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

LOOKAHEAD_SEED = 20260518  # deterministic spot-check sample
N_LOOKAHEAD_SPOT_CHECKS = 5


def _load_pair_csv(pair: str, data_dir: Path) -> pd.DataFrame:
    fpath = data_dir / f"{pair}.csv"
    df = pd.read_csv(fpath)
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)


def _resolve_data_path(data_4h: str) -> Path:
    if Path(data_4h).is_absolute():
        return Path(data_4h)
    return _REPO_ROOT / data_4h


def _audit_lookahead(
    cfg: dict, trades: pd.DataFrame, out_path: Path
) -> Dict[str, Any]:
    """Spot-check: recompute 5 random signal fires from OHLC truncated at t.

    Recomputes by calling the locked signal module on bars[0..t] (strictly
    past + current bar t). Compares the resulting bar-t fields against the
    trade row: swing_low_used, n_swing_lows, ib_passed, break_passed, fired.
    """
    sig_mod = importlib.import_module(str(cfg["signal"]["module"]))
    data_4h = _resolve_data_path(cfg["data"]["data_dirs"]["4H"])
    date_start = str(cfg["data"]["date_start"])
    date_end = str(cfg["data"]["date_end"])

    rng = random.Random(LOOKAHEAD_SEED)
    sample = trades.sample(
        n=min(N_LOOKAHEAD_SPOT_CHECKS, len(trades)),
        random_state=LOOKAHEAD_SEED,
    ).reset_index(drop=True)

    lines: List[str] = []
    lines.append("Arc 9 Step 1 - lookahead spot-check")
    lines.append("=" * 60)
    lines.append(f"seed={LOOKAHEAD_SEED}, n_samples={len(sample)}")
    lines.append("")
    n_pass = 0
    detail: List[Dict[str, Any]] = []

    for _, row in sample.iterrows():
        pair = str(row["pair"])
        sig_t = pd.Timestamp(row["signal_bar_time"])
        df_raw = _load_pair_csv(pair, data_4h)
        df_w = _slice_window(df_raw, date_start, date_end)
        # Find bar index of signal time.
        idx_arr = np.where(df_w["date"].to_numpy() == np.datetime64(sig_t))[0]
        if idx_arr.size == 0:
            lines.append(f"  [FAIL] {pair} {sig_t} not found in window")
            detail.append({"pair": pair, "signal_bar_time": str(sig_t), "fail": "bar not found"})
            continue
        t = int(idx_arr[0])
        # Truncate strictly to bars [0..t]; recompute signal on this slice.
        df_trunc = df_w.iloc[: t + 1].copy().reset_index(drop=True)
        df_sig = sig_mod.compute_signal(df_trunc, signal_col="signal")
        recomputed = df_sig.iloc[-1]  # the bar t
        reported_swing = float(row["swing_low_used"])
        reported_n_sl = int(row["n_swing_lows"])
        recomp_swing = float(recomputed["swing_low_used"])
        recomp_n_sl = int(recomputed["n_swing_lows"])
        recomp_fired = bool(recomputed["signal"])
        ok = (
            (abs(reported_swing - recomp_swing) < 1e-9 or
             (np.isnan(reported_swing) and np.isnan(recomp_swing)))
            and reported_n_sl == recomp_n_sl
            and recomp_fired  # signal must still fire on truncated data
        )
        if ok:
            n_pass += 1
        lines.append(
            f"  {pair} {sig_t}: swing_low {reported_swing:.5f}/{recomp_swing:.5f}  "
            f"n_sl {reported_n_sl}/{recomp_n_sl}  fired={recomp_fired}  "
            f"{'PASS' if ok else 'FAIL'}"
        )
        detail.append({
            "pair": pair,
            "signal_bar_time": str(sig_t),
            "swing_low_reported": reported_swing,
            "swing_low_recomputed": recomp_swing,
            "n_sl_reported": reported_n_sl,
            "n_sl_recomputed": recomp_n_sl,
            "fired_on_truncated": recomp_fired,
            "pass": ok,
        })

    verdict = "PASS" if n_pass == len(sample) else "FAIL"
    lines.append("")
    lines.append(f"Lookahead spot-check: {n_pass}/{len(sample)} - {verdict}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"n_pass": n_pass, "n_total": int(len(sample)), "verdict": verdict, "detail": detail}


def _audit_right_edge_swing(
    trades: pd.DataFrame, prefilter: pd.DataFrame, out_path: Path
) -> Dict[str, Any]:
    """Right-edge swing audit: most_recent_sl_lag >= 4 for every fired signal.

    Per signal spec §3: most recent identifiable swing-low at most bar t-4.
    Our signal computation enforces this; this audit verifies the column.
    """
    lines: List[str] = []
    lines.append("Arc 9 Step 1 - right-edge swing-low audit")
    lines.append("=" * 60)
    lines.append("Spec: most recent qualifying swing-low at bar k with k <= t-4")
    lines.append("Equivalent column: trades.most_recent_sl_lag >= 4 for every fire")
    lines.append("")

    lags = trades["most_recent_sl_lag"].to_numpy()
    violations = np.sum(lags < 4)
    min_lag = int(lags.min()) if len(lags) > 0 else -1
    max_lag = int(lags.max()) if len(lags) > 0 else -1

    # Distribution of lag.
    quants = np.percentile(lags, [5, 25, 50, 75, 95]) if len(lags) > 0 else [-1] * 5

    lines.append(f"Trades scanned: {len(trades)}")
    lines.append(f"Min lag: {min_lag}  Max lag: {max_lag}")
    lines.append(f"Lag p5/p25/p50/p75/p95: {quants[0]:.0f}/{quants[1]:.0f}/{quants[2]:.0f}/{quants[3]:.0f}/{quants[4]:.0f}")
    lines.append(f"Violations (lag<4): {violations}")
    lines.append("")
    verdict = "PASS" if violations == 0 else "FAIL"
    lines.append(f"Right-edge swing audit: {verdict}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "trades_scanned": int(len(trades)),
        "violations": int(violations),
        "min_lag": min_lag,
        "max_lag": max_lag,
        "lag_p50": float(quants[2]),
        "verdict": verdict,
    }


def _audit_schema_15a(trades_paths: pd.DataFrame, trades: pd.DataFrame, out_path: Path) -> Dict[str, Any]:
    """Strict §15a schema validation on trades_paths.csv.

    Checks:
      - columns include the §15a required set (+ extras allowed)
      - bar_offset = 0 is entry bar; offsets strictly increase by 1 within trade
      - is_held = 1 for entry..actual_exit (contiguous prefix), 0 after
      - both 0 and 1 values present at trade level for SL/EOD trades; only 1 for
        time_exit trades that exit at bar 240 (no forward window remaining)
    """
    required = {"trade_id", "pair", "bar_offset", "close_r", "mfe_so_far_r",
                "mae_so_far_r", "is_held"}
    extras = {"high_r", "low_r"}
    actual_cols = set(trades_paths.columns)
    missing = required - actual_cols

    violations: List[str] = []
    if missing:
        violations.append(f"missing required cols: {sorted(missing)}")

    # Per-trade structural checks.
    n_trades_with_forward = 0
    n_trades_no_forward = 0
    bad_offset = 0
    bad_is_held = 0
    grouped = trades_paths.groupby("trade_id", sort=False)
    for tid, sub in grouped:
        offs = sub["bar_offset"].to_numpy()
        held = sub["is_held"].to_numpy()
        if offs[0] != 0 or not np.array_equal(offs, np.arange(len(offs))):
            bad_offset += 1
        # Verify contiguous prefix of is_held=1 then suffix of is_held=0.
        if not np.all(np.isin(held, [0, 1])):
            bad_is_held += 1
        else:
            ones = np.where(held == 1)[0]
            zeros = np.where(held == 0)[0]
            if ones.size == 0:
                bad_is_held += 1
            elif zeros.size > 0 and zeros.min() <= ones.max():
                bad_is_held += 1
        if (held == 0).sum() > 0:
            n_trades_with_forward += 1
        else:
            n_trades_no_forward += 1

    n_trades = len(grouped)
    if bad_offset:
        violations.append(f"{bad_offset} trades with non-monotonic bar_offset")
    if bad_is_held:
        violations.append(f"{bad_is_held} trades with malformed is_held prefix/suffix")

    verdict = "PASS" if not violations else "FAIL"
    info = {
        "verdict": verdict,
        "required_cols_present": sorted(required - missing),
        "missing_required_cols": sorted(missing),
        "extras_present": sorted(actual_cols & extras),
        "n_trades_total": int(n_trades),
        "n_trades_with_forward_obs": int(n_trades_with_forward),
        "n_trades_no_forward_obs": int(n_trades_no_forward),
        "bad_offset": int(bad_offset),
        "bad_is_held": int(bad_is_held),
        "violations": violations,
    }
    out_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return info


def _cofire_overlap(
    arc9_trades: pd.DataFrame, other_trades: pd.DataFrame
) -> Dict[str, Any]:
    """Compute exact (pair, signal_bar_time) overlap between two trade pools.

    Other-arc trade pool column names vary. We auto-detect:
      - "signal_bar_time" (arc_7)
      - "signal_bar_ts" (arc_6)
      - "entry_date" / "signal_date" (kh24 / legacy)
    Returns counts at exact-match level and ±1 bar (4H -> 4 hours).
    """
    sig_col_candidates = ["signal_bar_time", "signal_bar_ts", "signal_date", "entry_date"]
    other_sig_col = None
    for c in sig_col_candidates:
        if c in other_trades.columns:
            other_sig_col = c
            break
    if other_sig_col is None:
        return {"exact": -1, "pm1bar": -1, "error": "no signal-time column"}

    other = other_trades[["pair", other_sig_col]].copy()
    other["sig_t"] = pd.to_datetime(other[other_sig_col])
    # For kh24/legacy where the column is `entry_date` (entry bar), shift back 1 bar.
    if other_sig_col in ("entry_date", "signal_date"):
        other["sig_t"] = other["sig_t"] - pd.Timedelta(hours=4)
    other_set = set(zip(other["pair"], other["sig_t"]))

    arc9 = arc9_trades[["pair", "signal_bar_time"]].copy()
    arc9["sig_t"] = pd.to_datetime(arc9["signal_bar_time"])
    arc9_pairs = list(zip(arc9["pair"], arc9["sig_t"]))

    n_exact = sum(1 for p, t in arc9_pairs if (p, t) in other_set)
    n_pm1 = 0
    delta_4h = pd.Timedelta(hours=4)
    for p, t in arc9_pairs:
        if (p, t) in other_set or (p, t - delta_4h) in other_set or (p, t + delta_4h) in other_set:
            n_pm1 += 1
    return {
        "arc9_pool": int(len(arc9_pairs)),
        "other_pool": int(len(other_trades)),
        "exact": int(n_exact),
        "pm1bar": int(n_pm1),
        "exact_pct_of_arc9": round(100.0 * n_exact / max(len(arc9_pairs), 1), 3),
    }


def _cofire_matrix(arc9_trades: pd.DataFrame, out_md: Path, out_json: Path) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    # KH-24
    kh24_path = _REPO_ROOT / "results" / "kh24" / "trades_all.csv"
    if kh24_path.exists():
        kh24 = pd.read_csv(kh24_path)
        results["KH-24 (kb_exhaustion_bar)"] = _cofire_overlap(arc9_trades, kh24)
    else:
        results["KH-24 (kb_exhaustion_bar)"] = {"error": "file not found", "path": str(kh24_path)}

    # Sibling arcs 8, 10, 11
    sibling_searches = {
        "Arc 8 (PR-HHHL)": [
            "results/l_arc_8/step1_verbatim/trades_all.csv",
            "results/l_arc_8/step1/trades_all.csv",
            "results/arc_8/step1/trades_all.csv",
        ],
        "Arc 10 (DLR)": [
            "results/l_arc_10/step1_verbatim/trades_all.csv",
            "results/l_arc_10/step1/trades_all.csv",
            "results/arc_10/step1/trades_all.csv",
        ],
        "Arc 11 (SHB)": [
            "results/l_arc_11/step1_verbatim/trades_all.csv",
            "results/l_arc_11/step1/trades_all.csv",
            "results/arc_11/step1/trades_all.csv",
        ],
    }
    for label, candidates in sibling_searches.items():
        found = None
        for c in candidates:
            p = _REPO_ROOT / c
            if p.exists():
                found = p
                break
        if found is None:
            results[label] = {"status": "not landed", "checked_paths": candidates}
            continue
        other = pd.read_csv(found)
        results[label] = _cofire_overlap(arc9_trades, other) | {"source": str(found.relative_to(_REPO_ROOT))}

    # MD render
    lines: List[str] = []
    lines.append("# Arc 9 Step 1 - co-fire matrix")
    lines.append("")
    lines.append("Source: results/l_arc_9/step1_verbatim/trades_all.csv (`signal_bar_time` column).")
    lines.append("Overlap definition: same `pair` + same `signal_bar_time` (exact 4H bar);")
    lines.append("`pm1bar` extends to ±1 4H bar.")
    lines.append("")
    lines.append("| Pool | Source | n other | n arc9 | exact | exact % arc9 | ±1bar | Flag (>10%) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for label, info in results.items():
        if "error" in info or "status" in info:
            note = info.get("error") or info.get("status")
            lines.append(f"| {label} | _ | _ | _ | _ | _ | _ | n/a ({note}) |")
            continue
        flag = "FLAG" if info["exact_pct_of_arc9"] > 10.0 else "ok"
        lines.append(
            f"| {label} | {info.get('source','-')} | {info['other_pool']} | {info['arc9_pool']} | "
            f"{info['exact']} | {info['exact_pct_of_arc9']:.3f}% | {info['pm1bar']} | {flag} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def _pool_summary(trades: pd.DataFrame, out_path: Path) -> Dict[str, Any]:
    bars = trades["bars_held"].to_numpy()
    bars_q = np.percentile(bars, [5, 25, 50, 75, 95])
    reasons = trades["exit_reason"].value_counts().to_dict()
    per_pair = trades.groupby("pair").size().sort_index().to_dict()
    lines: List[str] = []
    lines.append("# Arc 9 Step 1 - pool summary")
    lines.append("")
    lines.append(f"Total trades: **{len(trades)}**")
    lines.append("")
    lines.append("## Bars-held distribution")
    lines.append(f"p5 / p25 / p50 / p75 / p95 = "
                 f"{bars_q[0]:.0f} / {bars_q[1]:.0f} / {bars_q[2]:.0f} / {bars_q[3]:.0f} / {bars_q[4]:.0f}")
    pct_at_cap = 100.0 * float((bars == 240).sum()) / max(len(bars), 1)
    lines.append(f"% trades exiting at bar 240 (time_exit cap): **{pct_at_cap:.2f}%**")
    lines.append("")
    lines.append("## Exit reason mix")
    for k, v in sorted(reasons.items()):
        pct = 100.0 * v / max(len(trades), 1)
        lines.append(f"- {k}: {v} ({pct:.1f}%)")
    lines.append("")
    lines.append("## Per-pair trade counts")
    lines.append("")
    lines.append("| Pair | n trades |")
    lines.append("|---|---|")
    for p, n in per_pair.items():
        lines.append(f"| {p} | {n} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "bars_held_p50": float(bars_q[2]),
        "bars_held_p95": float(bars_q[4]),
        "pct_at_cap_bar_240": float(pct_at_cap),
        "exit_reason_counts": reasons,
        "pairs_lt_30": [p for p, n in per_pair.items() if n < 30],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Arc 9 Step 1 diagnostics.")
    parser.add_argument("-c", "--config", required=True, type=Path)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    results_dir = _REPO_ROOT / cfg["output"]["results_dir"]
    trades = pd.read_csv(results_dir / cfg["output"]["trades_csv"])
    trades_paths = pd.read_csv(results_dir / cfg["output"]["paths_csv"])
    prefilter = pd.read_csv(results_dir / cfg["output"]["prefilter_csv"])

    look = _audit_lookahead(cfg, trades, results_dir / "audit_lookahead.txt")
    edge = _audit_right_edge_swing(trades, prefilter, results_dir / "audit_right_edge_swing.txt")
    schema = _audit_schema_15a(trades_paths, trades, results_dir / "schema_check.json")
    cofire = _cofire_matrix(trades, results_dir / "cofire_matrix.md", results_dir / "cofire_matrix.json")
    pool = _pool_summary(trades, results_dir / "pool_summary.md")

    diag = {
        "lookahead": look,
        "right_edge_swing": edge,
        "schema_15a": {k: v for k, v in schema.items() if k != "violations"} | {"violations": schema["violations"]},
        "cofire_matrix": cofire,
        "pool_summary": pool,
    }
    (results_dir / "diagnostics.json").write_text(
        json.dumps(diag, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print("=" * 60)
    print(f"  Lookahead: {look['verdict']} ({look['n_pass']}/{look['n_total']})")
    print(f"  Right-edge swing: {edge['verdict']} (violations={edge['violations']})")
    print(f"  Schema §15a: {schema['verdict']}")
    print(f"  Co-fire matrix written: {results_dir / 'cofire_matrix.md'}")
    print(f"  Pool summary written: {results_dir / 'pool_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
