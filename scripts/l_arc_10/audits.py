"""Arc 10 Step 1 audits.

Per L_ARC_PROTOCOL v2.3 §5 + v2.2 §7 (live-execution equivalence) + the
Arc 10 dispatch + signal spec D1-lag verification.

Audits emitted (CSV / TXT per dispatch outputs spec):
  audit_lookahead.txt     5/5 spot-check that signal at bar t is unchanged
                          when bars > t are masked. (Standard.)
  audit_determinism.txt   Two-run byte-identical check on trades_all.csv,
                          trades_paths.csv, prefilter_events.csv.
  audit_d1_lag.txt        Arc 10 mandatory:
                            (a) right-edge swing audit — every signal's L_1
                                must be at most D1[d_t - 4].
                            (b) NaN-perturbation test — for 3 synthetic test
                                cases (3 actual signal-fire events), NaN-out
                                D1[d_t] row and confirm the signal still fires
                                identically. If any signal disappears, lag is
                                broken (engine-touching; do not patch).
  cofire_matrix.csv       KH-24 only (arcs 8/9/11 step1 not landed yet).
                          Overlap = signals firing within +/- 1 4H bar of each
                          other on the same pair.

Usage:
    py scripts/l_arc_10/audits.py -c configs/wfo_l_arc_10.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.l_arc_10 import step1_plumbing as s1  # noqa: E402

# ---------------------------------------------------------------------------
# Determinism audit — two-run sha256 comparison.
# ---------------------------------------------------------------------------


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_determinism(cfg: dict, config_path: Path, out_path: Path) -> Tuple[bool, str]:
    """Run Step 1 twice into a tmp results dir, compare sha256s of all CSV outputs."""
    import shutil
    import tempfile

    tmp_root = Path(tempfile.mkdtemp(prefix="arc10_step1_det_"))
    try:
        results = {}
        for run_label in ("a", "b"):
            run_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            run_results_dir = tmp_root / run_label
            run_cfg["output"]["results_dir"] = str(run_results_dir)
            run_results_dir.mkdir(parents=True, exist_ok=True)
            s1.run(run_cfg, config_path, write_manifest=False)
            results[run_label] = {
                "trades_all": _sha256(run_results_dir / run_cfg["output"]["trades_csv"]),
                "trades_paths": _sha256(run_results_dir / run_cfg["output"]["paths_csv"]),
                "prefilter": _sha256(run_results_dir / run_cfg["output"]["prefilter_csv"]),
            }
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    ok = results["a"] == results["b"]
    lines = ["# Arc 10 Step 1 — determinism audit", ""]
    lines.append(f"Result: {'PASS' if ok else 'FAIL'} (byte-identical two-run)")
    lines.append("")
    for key in ("trades_all", "trades_paths", "prefilter"):
        lines.append(f"{key}:")
        lines.append(f"  run a sha256: {results['a'][key]}")
        lines.append(f"  run b sha256: {results['b'][key]}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ok, "PASS" if ok else "FAIL"


# ---------------------------------------------------------------------------
# Lookahead audit — 5/5 spot-check.
# ---------------------------------------------------------------------------


def _spot_check_pair_bar(
    sig_mod, df_4h: pd.DataFrame, df_d1: pd.DataFrame, t_idx: int
) -> Tuple[bool, bool]:
    """Return (sig_full, sig_truncated) for the bar at position t_idx."""
    full_out = sig_mod.compute_signal(df_4h, df_d1, signal_col="signal")
    full_fire = bool(full_out["signal"].iloc[t_idx])

    # Truncate 4H bars after t_idx (keep <=t_idx); D1 truncated to bars whose
    # date is <= the 4H bar's date (i.e., remove any D1 row not seen yet).
    trunc_4h = df_4h.iloc[: t_idx + 1].copy()
    bar_date = pd.Timestamp(df_4h["date"].iloc[t_idx])
    trunc_d1 = df_d1[df_d1["date"] <= bar_date].reset_index(drop=True)
    trunc_out = sig_mod.compute_signal(trunc_4h, trunc_d1, signal_col="signal")
    trunc_fire = bool(trunc_out["signal"].iloc[t_idx]) if t_idx < len(trunc_out) else False
    return full_fire, trunc_fire


def audit_lookahead(cfg: dict, config_path: Path, out_path: Path) -> Tuple[bool, str]:
    """5/5 spot check: pick 5 signal-fire events, re-run signal on data
    truncated to t (no future bars on 4H, no future D1 rows). Same result
    required."""
    sig_mod_name = str(cfg["signal"]["module"])
    sig_mod = importlib.import_module(sig_mod_name)

    results_dir = _REPO_ROOT / cfg["output"]["results_dir"]
    trades_csv = results_dir / cfg["output"]["trades_csv"]
    if not trades_csv.exists():
        raise FileNotFoundError(f"trades_all.csv not found at {trades_csv}; run Step 1 first")
    trades = pd.read_csv(trades_csv, parse_dates=["signal_bar_time", "entry_time"])
    if len(trades) < 5:
        raise RuntimeError(f"need >=5 trades to spot-check; have {len(trades)}")

    sample = trades.sort_values("signal_bar_time").iloc[
        np.linspace(0, len(trades) - 1, 5, dtype=int)
    ]

    data_dirs = cfg["data"]["data_dirs"]
    dir_4h = s1._resolve_data_path(data_dirs["4H"])
    dir_d1 = s1._resolve_data_path(data_dirs["D1"])
    date_start = str(cfg["data"]["date_start"])
    date_end = str(cfg["data"]["date_end"])

    lines = ["# Arc 10 Step 1 — lookahead spot-check (5/5 audit)", ""]
    all_pass = True
    for _, trow in sample.iterrows():
        pair = trow["pair"]
        sig_time = pd.Timestamp(trow["signal_bar_time"])
        df_4h = s1._slice_window(s1._load_pair_csv(pair, dir_4h), date_start, date_end)
        df_d1 = s1._slice_d1_for_4h_window(s1._load_pair_csv(pair, dir_d1), date_start, date_end)
        match = df_4h.index[df_4h["date"] == sig_time]
        if len(match) == 0:
            lines.append(f"FAIL: {pair} @ {sig_time}: signal bar not found in sliced 4H")
            all_pass = False
            continue
        t_idx = int(match[0])
        full_fire, trunc_fire = _spot_check_pair_bar(sig_mod, df_4h, df_d1, t_idx)
        pass_ = full_fire and trunc_fire
        all_pass = all_pass and pass_
        lines.append(
            f"{'PASS' if pass_ else 'FAIL'}: {pair} @ {sig_time} "
            f"full={full_fire} truncated={trunc_fire}"
        )
    lines.append("")
    lines.append(f"Overall: {'PASS (5/5)' if all_pass else 'FAIL'}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return all_pass, "PASS" if all_pass else "FAIL"


# ---------------------------------------------------------------------------
# D1 lag audit — right-edge swing + NaN perturbation.
# ---------------------------------------------------------------------------


def audit_d1_lag(cfg: dict, config_path: Path, out_path: Path) -> Tuple[bool, str]:
    """Three checks in one audit:

    (a) Right-edge swing audit: every signal fired must have L_1 originate
        from a D1 bar whose index d_l1 satisfies d_l1 <= d_t - 4. This is
        enforced by construction in the signal module; we verify on the
        actual outputs.
    (b) L_1 age sanity: L_1_age_d1_bars must be in [4, 20].
    (c) NaN-perturbation test: for 3 actual signal-fire events, NaN out the
        D1 row at d_t and confirm the same signal still fires (since the
        signal references only D1 rows d <= d_t - 4, structurally).
    """
    sig_mod_name = str(cfg["signal"]["module"])
    sig_mod = importlib.import_module(sig_mod_name)

    results_dir = _REPO_ROOT / cfg["output"]["results_dir"]
    trades_csv = results_dir / cfg["output"]["trades_csv"]
    trades = pd.read_csv(trades_csv, parse_dates=["signal_bar_time", "entry_time"])

    data_dirs = cfg["data"]["data_dirs"]
    dir_4h = s1._resolve_data_path(data_dirs["4H"])
    dir_d1 = s1._resolve_data_path(data_dirs["D1"])
    date_start = str(cfg["data"]["date_start"])
    date_end = str(cfg["data"]["date_end"])

    lines = ["# Arc 10 Step 1 — D1 lag audit", ""]

    # (a) Right-edge swing audit.
    right_edge_offset = int(cfg["signal"]["d1_right_edge_offset"])
    l1_age_min = int((trades["L1_age_d1_bars"]).min())
    l1_age_max = int((trades["L1_age_d1_bars"]).max())
    right_edge_pass = l1_age_min >= right_edge_offset
    lines.append("## (a) Right-edge swing audit")
    lines.append(
        f"L_1_age_d1_bars min/max across {len(trades)} signals: {l1_age_min} / {l1_age_max}"
    )
    lines.append(f"Required: L_1_age >= {right_edge_offset} (right_edge_offset)")
    lines.append(f"Result: {'PASS' if right_edge_pass else 'FAIL'}")
    lines.append("")

    # (b) L_1 freshness sanity.
    freshness_max = int(cfg["signal"]["d1_l1_freshness_max_bars"])
    freshness_pass = l1_age_max <= freshness_max
    lines.append("## (b) L_1 freshness sanity")
    lines.append(f"L_1_age_d1_bars max: {l1_age_max} | freshness_max: {freshness_max}")
    lines.append(f"Result: {'PASS' if freshness_pass else 'FAIL'}")
    lines.append("")

    # (c) NaN-perturbation test on 3 signal-fire events.
    n_cases = int(cfg["d1_lag_audits"]["nan_perturbation_test_cases"])
    if len(trades) < n_cases:
        lines.append(f"## (c) NaN-perturbation test — INSUFFICIENT TRADES ({len(trades)})")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return False, "FAIL"

    # Choose n_cases trades spaced evenly across the trade pool for diversity.
    pick_idxs = np.linspace(0, len(trades) - 1, n_cases, dtype=int)
    sample = trades.sort_values("signal_bar_time").iloc[pick_idxs]

    lines.append(f"## (c) NaN-perturbation test ({n_cases} synthetic cases)")
    lines.append("Procedure: for each signal-fire event, NaN out the D1 row at d_t")
    lines.append("(the D1 bar containing 4H bar t). If the signal still fires identically,")
    lines.append("the lag is structurally correct (D1[d_t] is unread).")
    lines.append("")

    nan_pass = True
    for _, trow in sample.iterrows():
        pair = trow["pair"]
        sig_time = pd.Timestamp(trow["signal_bar_time"])
        df_4h = s1._slice_window(s1._load_pair_csv(pair, dir_4h), date_start, date_end)
        df_d1_full = s1._load_pair_csv(pair, dir_d1)
        df_d1 = s1._slice_d1_for_4h_window(df_d1_full, date_start, date_end)

        # Baseline: confirm the signal fires.
        baseline = sig_mod.compute_signal(df_4h, df_d1, signal_col="signal")
        match = df_4h.index[df_4h["date"] == sig_time]
        if len(match) == 0:
            lines.append(f"  FAIL: {pair} @ {sig_time}: bar not found")
            nan_pass = False
            continue
        t_idx = int(match[0])
        baseline_fire = bool(baseline["signal"].iloc[t_idx])
        if not baseline_fire:
            lines.append(
                f"  FAIL: {pair} @ {sig_time}: baseline signal did not fire (data drift?)"
            )
            nan_pass = False
            continue

        # Perturbation: NaN out the D1 row at d_t.
        bar_date = sig_time.normalize()
        # The D1 row whose date matches bar_date — that is d_t.
        d_t_mask = pd.to_datetime(df_d1["date"]).dt.normalize() == bar_date
        if not d_t_mask.any():
            lines.append(
                f"  WARN: {pair} @ {sig_time}: no D1 row at d_t={bar_date.date()} "
                f"(weekend/holiday). Try d_t = most recent D1 row <= bar_date."
            )
            d_t_mask = (
                pd.to_datetime(df_d1["date"]).dt.normalize() <= bar_date
            ) & ~(pd.to_datetime(df_d1["date"]).dt.normalize() < bar_date - pd.Timedelta(days=3))
            # Take the last such row.
            cand = np.where(d_t_mask.to_numpy())[0]
            if len(cand) == 0:
                lines.append(f"  FAIL: {pair} @ {sig_time}: no d_t row to NaN")
                nan_pass = False
                continue
            d_t_pos = int(cand[-1])
        else:
            d_t_pos = int(np.where(d_t_mask.to_numpy())[0][-1])

        df_d1_perturbed = df_d1.copy()
        for col in ("open", "high", "low", "close"):
            if col in df_d1_perturbed.columns:
                df_d1_perturbed.iloc[d_t_pos, df_d1_perturbed.columns.get_loc(col)] = float("nan")

        perturbed = sig_mod.compute_signal(df_4h, df_d1_perturbed, signal_col="signal")
        perturbed_fire = bool(perturbed["signal"].iloc[t_idx])
        ok = baseline_fire == perturbed_fire
        nan_pass = nan_pass and ok
        lines.append(
            f"  {'PASS' if ok else 'FAIL'}: {pair} @ {sig_time} "
            f"baseline={baseline_fire} perturbed_with_d_t_NaN={perturbed_fire} "
            f"d_t_idx_in_df_d1={d_t_pos} d_t_date={pd.Timestamp(df_d1['date'].iloc[d_t_pos]).date()}"
        )

    lines.append("")
    lines.append(f"NaN-perturbation overall: {'PASS' if nan_pass else 'FAIL'}")

    overall_pass = right_edge_pass and freshness_pass and nan_pass
    lines.append("")
    lines.append(f"=== Overall D1 lag audit: {'PASS' if overall_pass else 'FAIL'} ===")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return overall_pass, "PASS" if overall_pass else "FAIL"


# ---------------------------------------------------------------------------
# Co-fire matrix (KH-24 only at present; arcs 8/9/11 step1 not landed).
# ---------------------------------------------------------------------------


def cofire_matrix(cfg: dict, config_path: Path, out_path: Path) -> Tuple[bool, str]:
    """Compute co-fire % between Arc 10 signals and reference trades_all.csv
    sources. Co-fire = same pair, signal bar within +/- 1 4H bar."""
    results_dir = _REPO_ROOT / cfg["output"]["results_dir"]
    trades_csv = results_dir / cfg["output"]["trades_csv"]
    arc10 = pd.read_csv(trades_csv, parse_dates=["signal_bar_time"])

    cofire_block = cfg.get("cofire", {}) or {}
    refs = cofire_block.get("references", []) or []

    rows: List[Dict[str, Any]] = []
    for ref in refs:
        name = str(ref["name"])
        kind = str(ref["kind"])
        path = _REPO_ROOT / str(ref["path"])
        max_overlap_pct = float(ref.get("expected_overlap_max_pct", 10))
        if not path.exists():
            rows.append(
                {
                    "reference": name,
                    "n_arc10": len(arc10),
                    "n_reference": 0,
                    "overlap_count": 0,
                    "overlap_pct": 0.0,
                    "expected_overlap_max_pct": max_overlap_pct,
                    "flag": "REF_MISSING",
                    "note": str(ref.get("note", "")),
                }
            )
            continue
        if kind == "trades_all_csv":
            ref_df = pd.read_csv(path)
            time_col = "entry_date" if "entry_date" in ref_df.columns else (
                "signal_bar_time" if "signal_bar_time" in ref_df.columns else None
            )
            if time_col is None:
                rows.append(
                    {
                        "reference": name,
                        "n_arc10": len(arc10),
                        "n_reference": len(ref_df),
                        "overlap_count": 0,
                        "overlap_pct": 0.0,
                        "expected_overlap_max_pct": max_overlap_pct,
                        "flag": "NO_TIME_COLUMN",
                        "note": str(ref.get("note", "")),
                    }
                )
                continue
            ref_df[time_col] = pd.to_datetime(ref_df[time_col])
            # Build per-pair set of 4H-bar-aligned reference times (floor to 4H).
            ref_times_by_pair: Dict[str, set] = {}
            for _, r in ref_df.iterrows():
                p = str(r["pair"])
                t = pd.Timestamp(r[time_col]).floor("4h")
                ref_times_by_pair.setdefault(p, set()).add(t)
            overlap = 0
            for _, r in arc10.iterrows():
                p = str(r["pair"])
                t = pd.Timestamp(r["signal_bar_time"]).floor("4h")
                candidates = {t - pd.Timedelta(hours=4), t, t + pd.Timedelta(hours=4)}
                if any(c in ref_times_by_pair.get(p, set()) for c in candidates):
                    overlap += 1
            pct = 100.0 * overlap / len(arc10) if len(arc10) > 0 else 0.0
            flag = "OK" if pct <= max_overlap_pct else "OVER_EXPECTED"
            rows.append(
                {
                    "reference": name,
                    "n_arc10": len(arc10),
                    "n_reference": len(ref_df),
                    "overlap_count": overlap,
                    "overlap_pct": round(pct, 3),
                    "expected_overlap_max_pct": max_overlap_pct,
                    "flag": flag,
                    "note": str(ref.get("note", "")),
                }
            )
        else:
            rows.append(
                {
                    "reference": name,
                    "n_arc10": len(arc10),
                    "n_reference": 0,
                    "overlap_count": 0,
                    "overlap_pct": 0.0,
                    "expected_overlap_max_pct": max_overlap_pct,
                    "flag": "UNSUPPORTED_KIND",
                    "note": str(ref.get("note", "")),
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, lineterminator="\n")
    flags = set(out_df["flag"].tolist()) if not out_df.empty else set()
    any_over = "OVER_EXPECTED" in flags
    summary = "FLAG_OVER_EXPECTED" if any_over else "OK"
    return (not any_over), summary


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arc 10 Step 1 audits.")
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument(
        "--skip-determinism",
        action="store_true",
        help="Skip the determinism audit (which re-runs Step 1 twice).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    results_dir = _REPO_ROOT / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    statuses: Dict[str, str] = {}

    if not args.skip_determinism:
        ok, status = audit_determinism(cfg, args.config, results_dir / "audit_determinism.txt")
        statuses["determinism"] = status
        print(f"[arc_10 audit] determinism: {status}")
    else:
        statuses["determinism"] = "SKIPPED"

    ok, status = audit_lookahead(cfg, args.config, results_dir / "audit_lookahead.txt")
    statuses["lookahead"] = status
    print(f"[arc_10 audit] lookahead: {status}")

    ok, status = audit_d1_lag(cfg, args.config, results_dir / "audit_d1_lag.txt")
    statuses["d1_lag"] = status
    print(f"[arc_10 audit] d1_lag: {status}")

    ok, status = cofire_matrix(cfg, args.config, results_dir / "cofire_matrix.csv")
    statuses["cofire"] = status
    print(f"[arc_10 audit] cofire: {status}")

    any_fail = any(s == "FAIL" for s in statuses.values())
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
