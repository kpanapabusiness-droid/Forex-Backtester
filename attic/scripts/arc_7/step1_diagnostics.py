"""Arc 7 — Step 1 diagnostics + gate verification.

Reads the outputs of scripts/arc_7/step1_backtest.py and computes the seven
diagnostics required for STEP1_SUMMARY.md (per ARC_7_LIVE Step 1 dispatch),
plus the lookahead spot-check (5 random fires recomputed from OHLC truncated
at the signal bar) and the §15a schema check.

Outputs (under results/l_arc_7/step1/):
  diagnostics.json          all computed numbers, ready for the summary doc
  reclaim_strength_hist.png histogram of pre-filter reclaim ratios
  schema_check.json         §15a column / is_held invariant results
  lookahead_check.json      5-fire reproducibility receipts

Usage:
    py scripts/arc_7/step1_diagnostics.py -c configs/wfo_l_arc_7.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Required §15a columns (strict).
SCHEMA_REQUIRED_COLS = [
    "trade_id",
    "pair",
    "bar_offset",
    "close_r",
    "mfe_so_far_r",
    "mae_so_far_r",
    "is_held",
]


def _load_pair_4h(pair: str, data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / f"{pair}.csv")
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Diagnostic 1: pool size, per-pair counts.
# ---------------------------------------------------------------------------
def diag_pool_size(trades: pd.DataFrame, pairs: List[str]) -> Dict[str, Any]:
    per_pair = trades.groupby("pair").size().to_dict()
    pp_full = {p: int(per_pair.get(p, 0)) for p in pairs}
    flagged = sorted([p for p, n in pp_full.items() if n < 30])
    return {
        "total_trades": int(len(trades)),
        "per_pair": pp_full,
        "pairs_lt_30": flagged,
    }


# ---------------------------------------------------------------------------
# Diagnostic 2: bars_held distribution + cap-binding rate.
# ---------------------------------------------------------------------------
def diag_bars_held(trades: pd.DataFrame, hold_bars: int) -> Dict[str, Any]:
    bh = trades["bars_held"].to_numpy()
    pct_at_cap = float((trades["exit_reason"] == "time_exit").mean()) * 100.0
    return {
        "p5": float(np.percentile(bh, 5)),
        "p25": float(np.percentile(bh, 25)),
        "p50": float(np.percentile(bh, 50)),
        "p75": float(np.percentile(bh, 75)),
        "p95": float(np.percentile(bh, 95)),
        "p95_value": float(np.percentile(bh, 95)),
        "pct_at_cap_240": pct_at_cap,
        "auto_extend_flag": pct_at_cap > 20.0,
        "exit_reason_counts": trades["exit_reason"].value_counts().to_dict(),
    }


# ---------------------------------------------------------------------------
# Diagnostic 3: pre-filter reclaim strength distribution + Hartigan dip.
# ---------------------------------------------------------------------------
def diag_reclaim_strength(
    prefilter: pd.DataFrame, out_png: Path
) -> Dict[str, Any]:
    r = prefilter["reclaim_ratio"].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 4:
        return {"n": int(r.size), "error": "too few pre-filter events for distribution"}

    pct = {
        "p5": float(np.percentile(r, 5)),
        "p25": float(np.percentile(r, 25)),
        "p50": float(np.percentile(r, 50)),
        "p75": float(np.percentile(r, 75)),
        "p95": float(np.percentile(r, 95)),
        "min": float(r.min()),
        "max": float(r.max()),
        "n": int(r.size),
    }
    mass_below_0p5 = float((r < 0.5).mean())

    # Hartigan dip test (requires `diptest`).
    try:
        import diptest  # type: ignore
        dip, pval = diptest.diptest(r)
        dip_result = {"statistic": float(dip), "p_value": float(pval)}
    except Exception as exc:
        dip_result = {"statistic": None, "p_value": None, "error": str(exc)}

    # Histogram (matplotlib if available; skip silently if not).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(r, bins=50, color="steelblue", edgecolor="black", alpha=0.85)
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1.2, label="filter cut (0.5)")
        ax.set_xlabel("(close - swing_low) / (swing_low - low)")
        ax.set_ylabel("count")
        ax.set_title(f"Arc 7 pre-filter reclaim ratio (n={r.size})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        hist_path: Any = str(out_png.relative_to(_REPO_ROOT))
    except Exception as exc:
        hist_path = {"error": str(exc)}

    return {
        "percentiles": pct,
        "mass_below_0p5": mass_below_0p5,
        "hartigan_dip": dip_result,
        "histogram_path": hist_path,
    }


# ---------------------------------------------------------------------------
# Diagnostic 4: KH-24 co-fire.
# ---------------------------------------------------------------------------
def diag_kh24_cofire(
    trades: pd.DataFrame, kh24_path: Path, bar_seconds: int = 14400
) -> Dict[str, Any]:
    if not kh24_path.exists():
        return {"present": False, "note": "results/kh24/trades_all.csv missing"}

    kh = pd.read_csv(kh24_path)
    # KH-24's entry_date = signal_bar + 1*4h (next-open). Recover its signal bar.
    kh["entry_date"] = pd.to_datetime(kh["entry_date"], errors="coerce")
    kh["kh_signal_bar"] = kh["entry_date"] - pd.Timedelta(seconds=bar_seconds)
    kh_set = {(p, t) for p, t in zip(kh["pair"].astype(str), kh["kh_signal_bar"])}

    trades = trades.copy()
    trades["signal_bar_time"] = pd.Timedelta(0) + pd.to_datetime(trades["signal_bar_time"])
    bar_td = pd.Timedelta(seconds=bar_seconds)
    same_bar = 0
    pm_one = 0
    for _, r in trades.iterrows():
        p = str(r["pair"])
        t = r["signal_bar_time"]
        if (p, t) in kh_set:
            same_bar += 1
            pm_one += 1
            continue
        if (p, t - bar_td) in kh_set or (p, t + bar_td) in kh_set:
            pm_one += 1
    n = len(trades)
    return {
        "present": True,
        "same_bar": int(same_bar),
        "pm_one_bar": int(pm_one),
        "pct_of_arc7_pool_pm_one": float(pm_one / n * 100.0) if n else 0.0,
        "flag_gt_10pct": (pm_one / n * 100.0) > 10.0 if n else False,
    }


# ---------------------------------------------------------------------------
# Diagnostic 5: Arc 6 overlap (disjoint expected).
# ---------------------------------------------------------------------------
def diag_arc6_overlap(trades: pd.DataFrame, arc6_paths: List[Path]) -> Dict[str, Any]:
    """Check (pair, signal_bar) overlap against the first existing Arc 6 file.

    Per dispatch, Arc 6 results may live at either ``results/l_arc_6/step1/`` or
    ``results/arc_6/step1/``; the latter is the path used by commit b178ea4.
    Arc 6 uses ``signal_bar_ts`` (not ``signal_bar_time``).
    """
    arc6_path: Path | None = None
    for p in arc6_paths:
        if p.exists():
            arc6_path = p
            break
    if arc6_path is None:
        return {
            "file_present": False,
            "searched": [str(p) for p in arc6_paths],
            "note": "Arc 6 step 1 trades_all.csv missing",
        }
    a6 = pd.read_csv(arc6_path)
    sig_col = (
        "signal_bar_ts"
        if "signal_bar_ts" in a6.columns
        else "signal_bar_time"
        if "signal_bar_time" in a6.columns
        else "signal_time"
    )
    a6["_sig"] = pd.to_datetime(a6[sig_col], errors="coerce")
    a6_set = {(p, t) for p, t in zip(a6["pair"].astype(str), a6["_sig"])}
    trades = trades.copy()
    trades["signal_bar_time"] = pd.to_datetime(trades["signal_bar_time"])
    overlaps = [
        (str(r["pair"]), r["signal_bar_time"])
        for _, r in trades.iterrows()
        if (str(r["pair"]), r["signal_bar_time"]) in a6_set
    ]
    return {
        "file_present": True,
        "arc6_source": str(arc6_path),
        "arc6_signal_col": sig_col,
        "arc6_pool_size": int(len(a6)),
        "overlap_count": int(len(overlaps)),
        "examples": [{"pair": p, "bar": str(t)} for p, t in overlaps[:10]],
    }


# ---------------------------------------------------------------------------
# Diagnostic 6: spread cost reality.
# ---------------------------------------------------------------------------
def diag_spread_reality(trades: pd.DataFrame) -> Dict[str, Any]:
    s_used = trades["spread_pips_used"].to_numpy(dtype=float)
    s_exit = trades["spread_pips_exit"].to_numpy(dtype=float)
    def _pct(a):
        return {
            "p5": float(np.percentile(a, 5)),
            "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)),
        }
    weekend_mask = s_exit > 50.0
    weekend_count = int(weekend_mask.sum())
    worst = trades[weekend_mask].nlargest(3, "spread_pips_exit")
    worst_examples = []
    if len(worst):
        for _, r in worst.iterrows():
            worst_examples.append(
                {
                    "pair": str(r["pair"]),
                    "exit_time": str(r["exit_time"]),
                    "spread_pips_exit": float(r["spread_pips_exit"]),
                }
            )
    return {
        "spread_pips_used": _pct(s_used),
        "spread_pips_exit": _pct(s_exit),
        "weekend_tail_count_exit_gt_50": weekend_count,
        "worst_examples": worst_examples,
    }


# ---------------------------------------------------------------------------
# §15a schema check.
# ---------------------------------------------------------------------------
def schema_check(paths: pd.DataFrame, hold_window: int) -> Dict[str, Any]:
    cols = list(paths.columns)
    missing = [c for c in SCHEMA_REQUIRED_COLS if c not in cols]
    # is_held invariants per trade.
    violations: List[Dict[str, Any]] = []
    grouped = paths.groupby("trade_id", sort=True)
    n_trades = 0
    n_with_forward = 0
    for tid, g in grouped:
        n_trades += 1
        g = g.sort_values("bar_offset").reset_index(drop=True)
        # bar_offset 0 must exist.
        if int(g["bar_offset"].iloc[0]) != 0:
            violations.append({"trade_id": int(tid), "issue": "missing bar_offset=0"})
            continue
        # bar_offset strictly increasing by 1.
        if not (np.diff(g["bar_offset"].to_numpy()) == 1).all():
            violations.append({"trade_id": int(tid), "issue": "bar_offset not contiguous"})
            continue
        ih = g["is_held"].to_numpy().astype(int)
        # is_held=1 must be a contiguous prefix; is_held=0 must be a contiguous suffix.
        if ih.max() not in (0, 1) or ih.min() not in (0, 1):
            violations.append({"trade_id": int(tid), "issue": "is_held not in {0,1}"})
            continue
        if 0 in ih:
            n_with_forward += 1
            first_zero = int(np.argmax(ih == 0))
            if not (ih[:first_zero] == 1).all() or not (ih[first_zero:] == 0).all():
                violations.append(
                    {"trade_id": int(tid), "issue": "is_held not prefix-1 / suffix-0"}
                )
                continue
        else:
            if not (ih == 1).all():
                violations.append({"trade_id": int(tid), "issue": "is_held not all 1"})
                continue
    return {
        "required_columns_present": len(missing) == 0,
        "missing_columns": missing,
        "trades_checked": n_trades,
        "trades_with_forward_obs_rows": n_with_forward,
        "violations": violations[:10],
        "violation_count": len(violations),
        "pass": len(missing) == 0 and len(violations) == 0,
    }


# ---------------------------------------------------------------------------
# Lookahead spot-check.
# ---------------------------------------------------------------------------
def lookahead_spotcheck(
    cfg: dict, trades: pd.DataFrame, n_check: int = 5, seed: int = 42
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    sample = trades.sample(n=min(n_check, len(trades)), random_state=int(rng.integers(0, 2**31 - 1)))
    data_dir = _REPO_ROOT / cfg["data"]["data_dirs"]["4H"]
    sig_mod_name = str(cfg["signal"]["module"])
    sig_mod = importlib.import_module(sig_mod_name)

    results: List[Dict[str, Any]] = []
    all_ok = True
    for _, row in sample.iterrows():
        pair = str(row["pair"])
        sig_t = pd.Timestamp(row["signal_bar_time"])
        df = _load_pair_4h(pair, data_dir)
        if sig_t not in set(df["date"]):
            results.append(
                {"pair": pair, "signal_bar_time": str(sig_t), "ok": False, "reason": "bar missing in source"}
            )
            all_ok = False
            continue
        sig_idx = int(df.index[df["date"] == sig_t][0])
        # Truncate OHLC to bars <= sig_idx inclusive (strict no future).
        truncated = df.iloc[: sig_idx + 1].copy().reset_index(drop=True)
        out = sig_mod.compute_signal(truncated)
        fired_at_sig = bool(out["signal"].iloc[sig_idx])
        # Compare reported swing_low_used and reclaim ratio.
        sw_recomputed = float(out["swing_low_N"].iloc[sig_idx])
        rec_recomputed = float(out["reclaim_ratio"].iloc[sig_idx])
        sw_reported = float(row["swing_low_used"])
        rec_reported = float(row["reclaim_strength_ratio"])
        ok = (
            fired_at_sig
            and abs(sw_recomputed - sw_reported) < 1e-9
            and abs(rec_recomputed - rec_reported) < 1e-9
        )
        if not ok:
            all_ok = False
        results.append(
            {
                "pair": pair,
                "signal_bar_time": str(sig_t),
                "fired_at_sig_under_truncation": fired_at_sig,
                "swing_low_recomputed": sw_recomputed,
                "swing_low_reported": sw_reported,
                "reclaim_ratio_recomputed": rec_recomputed,
                "reclaim_ratio_reported": rec_reported,
                "ok": ok,
            }
        )
    return {"n_checked": len(results), "all_pass": all_ok, "results": results}


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cfg_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    results_dir = _REPO_ROOT / cfg["output"]["results_dir"]
    trades_path = results_dir / cfg["output"]["trades_csv"]
    paths_path = results_dir / cfg["output"]["paths_csv"]
    prefilter_path = results_dir / cfg["output"]["prefilter_csv"]

    trades = pd.read_csv(trades_path)
    paths = pd.read_csv(paths_path)
    prefilter = pd.read_csv(prefilter_path)

    pairs = sorted(list(cfg["pairs"]))
    hold_bars = int(cfg["exit"]["time_exit"]["bars_after_entry"])

    diag1 = diag_pool_size(trades, pairs)
    diag2 = diag_bars_held(trades, hold_bars)
    diag3 = diag_reclaim_strength(prefilter, results_dir / "reclaim_strength_hist.png")
    diag4 = diag_kh24_cofire(trades, _REPO_ROOT / "results" / "kh24" / "trades_all.csv")
    diag5 = diag_arc6_overlap(
        trades,
        [
            _REPO_ROOT / "results" / "l_arc_6" / "step1" / "trades_all.csv",
            _REPO_ROOT / "results" / "arc_6" / "step1" / "trades_all.csv",
        ],
    )
    diag6 = diag_spread_reality(trades)

    schema = schema_check(paths, hold_bars)
    look = lookahead_spotcheck(cfg, trades)

    out: Dict[str, Any] = {
        "diag_1_pool": diag1,
        "diag_2_bars_held": diag2,
        "diag_3_reclaim_strength": diag3,
        "diag_4_kh24_cofire": diag4,
        "diag_5_arc6_overlap": diag5,
        "diag_6_spread_reality": diag6,
        "schema_check": schema,
        "lookahead_spotcheck": look,
        "input_sha256": {
            "trades_all_csv": _sha(trades_path),
            "trades_paths_csv": _sha(paths_path),
            "prefilter_events_csv": _sha(prefilter_path),
        },
    }
    (results_dir / "diagnostics.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (results_dir / "schema_check.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (results_dir / "lookahead_check.json").write_text(
        json.dumps(look, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return out


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arc 7 Step 1 diagnostics.")
    p.add_argument("-c", "--config", required=True, type=Path)
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    out = run(args.config)
    print(f"[arc_7 diag] pool={out['diag_1_pool']['total_trades']}")
    print(f"[arc_7 diag] bars_held p95={out['diag_2_bars_held']['p95']:.1f}")
    print(f"[arc_7 diag] pct_at_cap={out['diag_2_bars_held']['pct_at_cap_240']:.2f}%")
    pf = out["diag_3_reclaim_strength"]
    if "percentiles" in pf:
        print(
            f"[arc_7 diag] reclaim pre-filter n={pf['percentiles']['n']} "
            f"p50={pf['percentiles']['p50']:.3f} mass<0.5={pf['mass_below_0p5']:.3f}"
        )
    sc = out["schema_check"]
    print(f"[arc_7 diag] schema PASS={sc['pass']} violations={sc['violation_count']}")
    la = out["lookahead_spotcheck"]
    print(f"[arc_7 diag] lookahead {sum(r['ok'] for r in la['results'])}/{la['n_checked']} reproduce")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
