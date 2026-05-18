"""Arc 9 Step 5 WFO with LightGBM Pipeline E filter (load-bearing experiment).

Held-open experiment under v2.3. The fourth Step 5 reference point in Arc 9:
  raw baseline          full pool, no filter, default exit       FAIL all gates
  oracle                cluster 0 only, §11 exit                  PASS-DEPLOYABLE
  calibration recovery  classifier probability remap diagnostic   OUTCOME_B
  THIS                  full pool, LightGBM admission, §11 exit   PENDING

Two threshold candidates run side-by-side:
  A) thr=0.40 (locked v2.2 §3 grid best per Pipeline E retry extended sweep
              -- recall 0.34, precision 0.47)
  B) thr=0.05 (recall-floor operating point per extended sweep -- recall 0.60,
              precision 0.34; hypothetical v2.x amendment "Option A" point)

Classifier provenance:
  Pipeline E retry trained LightGBM in 5-fold TimeSeriesSplit and reported
  mean CV AUC 0.7508. No full-data joblib was persisted (same pattern as
  Step 4: artefact-on-PASS only). The dispatch says: "Reload existing
  artefact or deterministic reproduction only."
  This script:
   (i)  reproduces the TSS-CV AUC 0.7508 byte-for-byte (parity check; fails
        loudly if mismatch)
   (ii) trains a new LGBM PER KH-24 FOLD using anchored-expanding windows
        (training data = all trades with entry_time < fold OOS_start, same
        hyperparams + seed + feature pipeline). This is honest WFO discipline.
   Fold 1 of KH-24 (OOS 2020-10-01 -> 2021-07-01) coincides with the Arc 9
   data start (2020-10-01), so no trades exist before fold 1 -- no training
   data -> 0 admits for fold 1 under both candidates. Documented in report.

Exit policy:
  cluster_0_individual is labelled "Unclassified" in §2 archetype assignment
  (boundary cluster: 3 of 4 §11 Stepwise climber criteria match, pullback
  0.580 fails the 0.5 ceiling by 0.08). Step 5 oracle dispatch routed it to
  §11 Stepwise climber Pipeline E exit (MFE-lock at 1R, trail 0.75R) and the
  dispatcher accepted that routing (PASS-DEPLOYABLE oracle result). This
  experiment uses the same exit policy via direct re-use of
  scripts.l_arc_9.experiments.step5_validation._resimulate_trade.

Live-execution semantics:
  - Signal at bar t close -> entry at bar t+1 open (long, +S/2 fill)
  - Real per-bar MT5 spread; floor only when raw spread = 0
  - Intrabar SL trigger on mid (low <= SL), fill on bid
  - D1 features at classifier inference one-day lagged via merge_asof backward
    (same pattern as training; reused from pipeline_e_retry)
  - 4H features at bar t close (no peek into t+1)
  - Risk: 0.5% per trade compounded from $10k starting balance

Outputs in results/l_arc_9/experiments/step5_lgbm_pipeline_e/:
  reproduced_tss_cv_aucs.csv           parity check vs pipeline_e_retry
  parity_check.json                    PASS/FAIL on AUC reproduction
  candidate_A_thr0.40/                 subfolder per candidate
    per_fold_metrics.csv
    full_data_metrics.json
    admitted_trades.csv
    resim_trades.csv
  candidate_B_thr0.05/                 subfolder per candidate
    per_fold_metrics.csv
    full_data_metrics.json
    admitted_trades.csv
    resim_trades.csv
  comparison.csv                       side-by-side w/ 3 prior reference points
  determinism_check.json
  STEP5_LGBM_E_RESULT.md               report
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse training-time feature pipeline + WFO/exit machinery.
from scripts.l_arc_9.experiments.pipeline_e_retry import (  # noqa: E402
    EXPANDED_28, BASELINE_16, D1_8, SESSION_4, LGBM_KW, SEED,
    _attach_d1_features, _attach_session_features,
    FORBIDDEN_LEAK_FEATURES,
)
from scripts.l_arc_9.experiments.step5_validation import (  # noqa: E402
    STARTING_BALANCE, RISK_PCT, PASS_DEPLOYABLE, PASS_VIABLE,
    _resimulate_trade, _compute_fold_metrics, _full_data_equity,
    evaluate_gates,
)
from core.spread_floor import (  # noqa: E402
    STATE_CFG_KEY,
    load_spread_floor,
)

# Candidate thresholds per dispatch.
CANDIDATES = [
    ("A_thr0.40", 0.40, "locked v2.2 §3 grid best (Pipeline E retry extended sweep)"),
    ("B_thr0.05", 0.05, "recall-floor operating point (extended sweep recall ~0.60)"),
]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_pair_csv(pair: str, data_dir: Path) -> pd.DataFrame:
    fpath = data_dir / f"{pair}.csv"
    df = pd.read_csv(fpath)
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df[(df["date"] >= s) & (df["date"] <= e)].reset_index(drop=True)


def _build_feature_matrix(out_dir: Path) -> pd.DataFrame:
    """Build the 28-feature matrix identical to pipeline_e_retry.

    Reads:
        results/l_arc_9/step4_extractability/entry_features.csv     16 baseline
        results/l_arc_9/step1_verbatim/trades_all.csv               atr14_at_signal, entry_time
        results/l_arc_9/step2_clustering/clusters_K3.csv            labels
        data/daily/<pair>.csv                                       D1 lag features
    """
    entry = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "step4_extractability" / "entry_features.csv")
    forbidden = set(entry.columns) & FORBIDDEN_LEAK_FEATURES
    if forbidden:
        raise RuntimeError(f"path-shape features leaked into entry features: {forbidden}")
    for c in BASELINE_16:
        if c not in entry.columns:
            raise RuntimeError(f"missing baseline feature: {c}")

    clusters = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "step2_clustering" / "clusters_K3.csv")
    cid0 = set(clusters[clusters["cluster_id"] == 0]["trade_id"].astype(int))

    trades_all = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "step1_verbatim" / "trades_all.csv")
    atr_4h_by_tid: Dict[int, float] = dict(zip(
        trades_all["trade_id"].astype(int),
        trades_all["atr14_at_signal"].astype(float),
    ))
    entry_time_by_tid: Dict[int, str] = dict(zip(
        trades_all["trade_id"].astype(int),
        trades_all["entry_time"].astype(str),
    ))

    data_d1_dir = Path("C:/Users/panap/Documents/Forex-Backtester/data/daily")
    if not data_d1_dir.exists():
        raise FileNotFoundError(f"D1 data dir not found: {data_d1_dir}")

    df = _attach_d1_features(entry, data_d1_dir, atr_4h_by_tid)
    df = _attach_session_features(df)
    df["entry_time"] = df["trade_id"].astype(int).map(entry_time_by_tid)
    df["y"] = df["trade_id"].astype(int).apply(lambda x: 1 if int(x) in cid0 else 0)

    df_clean = df.dropna(subset=EXPANDED_28).reset_index(drop=True)
    df_clean["entry_time"] = pd.to_datetime(df_clean["entry_time"])
    df_clean = df_clean.sort_values(["entry_time", "pair"], kind="mergesort").reset_index(drop=True)
    return df_clean


def _parity_check_tss_cv(
    df_clean: pd.DataFrame, expected_mean_auc: float = 0.7508, tolerance: float = 1e-4,
) -> Dict[str, Any]:
    """Reproduce pipeline_e_retry's TimeSeriesSplit(5) LGBM expanded AUC."""
    X = df_clean[EXPANDED_28].to_numpy(dtype=float)
    y = df_clean["y"].to_numpy(dtype=int)
    tscv = TimeSeriesSplit(n_splits=5)
    fold_aucs: List[float] = []
    for tr_idx, te_idx in tscv.split(X):
        mdl = lgb.LGBMClassifier(**LGBM_KW)
        mdl.fit(X[tr_idx], y[tr_idx])
        p = mdl.predict_proba(X[te_idx])[:, 1]
        fold_aucs.append(float(roc_auc_score(y[te_idx], p)))
    mean_auc = float(np.mean(fold_aucs))
    parity = abs(mean_auc - expected_mean_auc) < tolerance
    return {
        "reproduced_mean_auc": mean_auc,
        "expected_mean_auc": expected_mean_auc,
        "tolerance": tolerance,
        "parity": bool(parity),
        "per_fold_aucs": [round(a, 6) for a in fold_aucs],
    }


def _wfo_for_threshold(
    df_clean: pd.DataFrame, threshold: float, kh24_folds: List[Tuple[int, pd.Timestamp, pd.Timestamp]],
    cfg_arc: dict, spread_state, pair_cache: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """For each KH-24 OOS fold:
      1. Train LGBM on trades with entry_time < fold OOS_start (anchored
         expanding). If empty, all trades in fold rejected.
      2. Score trades in OOS window; admit if prob >= threshold.
      3. Re-simulate each admitted trade with §11 Stepwise exit
         (via step5_validation._resimulate_trade).
      4. Compute per-fold metrics (compounded 0.5% risk equity).
    """
    feat_cols = EXPANDED_28
    X = df_clean[feat_cols].to_numpy(dtype=float)
    y = df_clean["y"].to_numpy(dtype=int)
    entry_time = pd.to_datetime(df_clean["entry_time"]).to_numpy()

    # Bring in raw trade fields needed for re-simulation (signal_bar_time,
    # entry_time, pair, atr14_at_signal). atr is in trades_all.
    trades_all = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "step1_verbatim" / "trades_all.csv")
    trades_all["signal_bar_time"] = pd.to_datetime(trades_all["signal_bar_time"])
    trades_all["entry_time"] = pd.to_datetime(trades_all["entry_time"])
    tidx_to_resim_row = trades_all.set_index("trade_id")

    all_admitted_rows: List[Dict[str, Any]] = []
    all_resim_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []

    for (fold_id, s, e) in kh24_folds:
        train_mask = entry_time < np.datetime64(s)
        oos_mask = (entry_time >= np.datetime64(s)) & (entry_time < np.datetime64(e))
        n_train = int(train_mask.sum())
        n_oos = int(oos_mask.sum())
        n_train_pos = int(y[train_mask].sum())
        # WFO discipline: train on prior data only. Fold 1 has empty training.
        if n_train == 0 or n_train_pos < 10:
            # No usable training data -> reject all OOS trades.
            fold_rows.append({
                "fold": fold_id,
                "oos_start": s.strftime("%Y-%m-%d"),
                "oos_end": e.strftime("%Y-%m-%d"),
                "n_oos_signals": n_oos,
                "n_train": n_train,
                "n_train_pos": n_train_pos,
                "n_admitted": 0,
                "n_admitted_cluster0_true_pos": 0,
                "admit_rate": 0.0,
                "n_trades": 0,
                "final_r_mean": float("nan"),
                "final_r_sign_positive": 0,
                "fold_roi_pct": 0.0,
                "annualised_roi_pct": 0.0,
                "max_dd_pct": 0.0,
                "ending_equity": STARTING_BALANCE,
                "note": "no training data (anchored expanding under KH-24 fold dates; Arc 9 data starts at fold 1 OOS_start)",
            })
            continue

        mdl = lgb.LGBMClassifier(**LGBM_KW)
        mdl.fit(X[train_mask], y[train_mask])
        prob_oos = mdl.predict_proba(X[oos_mask])[:, 1]
        admit = prob_oos >= threshold
        admit_idx_in_clean = np.where(oos_mask)[0][admit]
        admit_trade_ids = df_clean["trade_id"].iloc[admit_idx_in_clean].astype(int).tolist()
        admit_true_pos = int(y[admit_idx_in_clean].sum())
        n_admitted = int(admit.sum())

        # Re-simulate each admitted trade with §11 Stepwise exit.
        resim_list = []
        for tid in admit_trade_ids:
            row = tidx_to_resim_row.loc[tid].copy()
            row["trade_id"] = tid
            row["pair"] = str(row["pair"])
            df_pair = pair_cache[row["pair"]]
            r = _resimulate_trade(row, df_pair, cfg_arc, spread_state)
            if r is None:
                continue
            resim_list.append({
                "trade_id": r.trade_id, "pair": r.pair,
                "signal_bar_time": r.signal_bar_time,
                "entry_time": r.entry_time,
                "exit_time": r.exit_time, "exit_reason": r.exit_reason,
                "bars_held": r.bars_held,
                "mfe_lock_active_at_exit": int(r.mfe_lock_active_at_exit),
                "final_r": r.final_r, "mfe_r": r.mfe_r, "mae_r": r.mae_r,
                "spread_pips_used": r.spread_pips_used,
                "spread_pips_exit": r.spread_pips_exit,
                "fold": fold_id,
                "classifier_prob": float(prob_oos[np.where(admit)[0][admit_trade_ids.index(tid)]]),
            })
            all_admitted_rows.append({"trade_id": tid, "fold": fold_id, "prob": float(prob_oos[admit][len(resim_list) - 1])})
        all_resim_rows.extend(resim_list)
        resim_df = pd.DataFrame(resim_list)

        if len(resim_df) == 0:
            fmetrics_subset = {
                "n_trades": 0, "final_r_mean": float("nan"),
                "final_r_sign_positive": 0, "fold_roi_pct": 0.0,
                "annualised_roi_pct": 0.0, "max_dd_pct": 0.0,
                "ending_equity": STARTING_BALANCE,
            }
        else:
            resim_df["entry_time"] = pd.to_datetime(resim_df["entry_time"])
            fmetrics_subset = _compute_fold_metrics(resim_df, s, e)

        fold_rows.append({
            "fold": fold_id,
            "oos_start": s.strftime("%Y-%m-%d"),
            "oos_end": e.strftime("%Y-%m-%d"),
            "n_oos_signals": n_oos,
            "n_train": n_train,
            "n_train_pos": n_train_pos,
            "n_admitted": n_admitted,
            "n_admitted_cluster0_true_pos": admit_true_pos,
            "admit_rate": float(n_admitted / max(n_oos, 1)),
            **fmetrics_subset,
            "note": "",
        })

    fold_df = pd.DataFrame(fold_rows)
    admitted_df = pd.DataFrame(all_admitted_rows)
    resim_df = pd.DataFrame(all_resim_rows)
    if len(resim_df) > 0:
        resim_df = resim_df.sort_values(["entry_time", "pair"], kind="mergesort").reset_index(drop=True)
        in_window = resim_df[
            (resim_df["entry_time"] >= kh24_folds[0][1]) &
            (resim_df["entry_time"] < kh24_folds[-1][2])
        ]
        full_m = _full_data_equity(in_window, [(s, e) for _, s, e in kh24_folds])
    else:
        full_m = {
            "n_trades": 0,
            "full_data_roi_pct": 0.0,
            "full_data_annualised_roi_pct": 0.0,
            "full_data_max_dd_pct": 0.0,
            "ending_equity": STARTING_BALANCE,
        }
    return fold_df, admitted_df, resim_df, full_m


def run(out_dir: Path, kh24_cfg_path: Path, arc_cfg_path: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_kh24 = yaml.safe_load(kh24_cfg_path.read_text(encoding="utf-8"))
    cfg_arc = yaml.safe_load(arc_cfg_path.read_text(encoding="utf-8"))

    folds = cfg_kh24["wfo"]["folds"]
    kh24_folds: List[Tuple[int, pd.Timestamp, pd.Timestamp]] = [
        (int(f["fold"]), pd.Timestamp(f["oos_start"]), pd.Timestamp(f["oos_end"]))
        for f in folds
    ]

    # 1) Build feature matrix.
    print("[lgbm-e] building 28-feature matrix...")
    df_clean = _build_feature_matrix(out_dir)
    df_clean.to_csv(out_dir / "feature_matrix.csv", index=False,
                    float_format="%.10g", lineterminator="\n")
    print(f"[lgbm-e] feature matrix: n={len(df_clean)}, n_pos={int(df_clean['y'].sum())}")

    # 2) Parity check vs Pipeline E retry TSS-CV AUC 0.7508.
    print("[lgbm-e] parity check vs Pipeline E retry TSS-CV AUC 0.7508...")
    parity = _parity_check_tss_cv(df_clean)
    (out_dir / "parity_check.json").write_text(
        json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame({"fold": [1, 2, 3, 4, 5], "auc": parity["per_fold_aucs"]}).to_csv(
        out_dir / "reproduced_tss_cv_aucs.csv", index=False,
        float_format="%.10g", lineterminator="\n",
    )
    print(f"  reproduced TSS-CV mean AUC: {parity['reproduced_mean_auc']:.6f} "
          f"(expected {parity['expected_mean_auc']}, "
          f"parity {'PASS' if parity['parity'] else 'FAIL'})")
    if not parity["parity"]:
        raise RuntimeError(
            f"Parity check FAILED: reproduced AUC {parity['reproduced_mean_auc']} "
            f"differs from Pipeline E retry's 0.7508 by > {parity['tolerance']}. "
            "Aborting -- the classifier we'd evaluate isn't the same one scored at 0.7508."
        )

    # 3) Set up shared resources for re-simulation.
    pairs = sorted(df_clean["pair"].unique().tolist())
    data_4h_path = cfg_arc["data"]["data_dirs"]["4H"]
    data_dir_4h = Path(data_4h_path) if Path(data_4h_path).is_absolute() else _REPO_ROOT / data_4h_path
    pair_cache: Dict[str, pd.DataFrame] = {}
    for pair in pairs:
        df_raw = _load_pair_csv(pair, data_dir_4h)
        pair_cache[pair] = _slice_window(
            df_raw,
            str(cfg_arc["data"]["date_start"]),
            str(cfg_arc["data"]["date_end"]),
        )
    spread_state = load_spread_floor(cfg_arc)
    cfg_arc[STATE_CFG_KEY] = spread_state
    cfg_arc.setdefault("spreads", {})
    cfg_arc["spreads"].setdefault("points_per_pip", float(spread_state.points_per_pip))

    # 4) Run both candidates.
    candidate_results: Dict[str, Dict[str, Any]] = {}
    for cand_name, thr, note in CANDIDATES:
        print(f"[lgbm-e] running candidate {cand_name} (thr={thr}) ...")
        cand_dir = out_dir / f"candidate_{cand_name}"
        cand_dir.mkdir(exist_ok=True)
        fold_df, admitted_df, resim_df, full_m = _wfo_for_threshold(
            df_clean, thr, kh24_folds, cfg_arc, spread_state, pair_cache,
        )
        fold_df.to_csv(cand_dir / "per_fold_metrics.csv", index=False,
                       float_format="%.10g", lineterminator="\n")
        admitted_df.to_csv(cand_dir / "admitted_trades.csv", index=False,
                           float_format="%.10g", lineterminator="\n")
        resim_df.to_csv(cand_dir / "resim_trades.csv", index=False,
                        float_format="%.10g", lineterminator="\n")
        (cand_dir / "full_data_metrics.json").write_text(
            json.dumps(full_m, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Build a gate-evaluation compatible fold_df.
        gates = evaluate_gates(fold_df, full_m)
        candidate_results[cand_name] = {
            "threshold": thr, "note": note,
            "fold_df": fold_df, "admitted_df": admitted_df,
            "resim_df": resim_df, "full_m": full_m, "gates": gates,
        }
        print(f"  fold n_admitted: {fold_df['n_admitted'].tolist()}")
        print(f"  worst-fold ann ROI {gates['summary']['worst_fold_ann_roi_pct']:+.2f}%, "
              f"mean ann ROI {gates['summary']['mean_fold_ann_roi_pct']:+.2f}%, "
              f"worst DD {gates['summary']['worst_fold_max_dd_pct']:.2f}%")

    # 5) Comparison to prior reference points.
    raw_fold = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_raw_baseline" / "per_fold_metrics.csv")
    raw_full = json.loads((_REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_raw_baseline" / "full_data_metrics.json").read_text(encoding="utf-8"))
    raw_gates = evaluate_gates(raw_fold, raw_full)
    oracle_fold = pd.read_csv(_REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_validation" / "per_fold_metrics.csv")
    oracle_full = json.loads((_REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_validation" / "full_data_metrics.json").read_text(encoding="utf-8"))
    oracle_gates = evaluate_gates(oracle_fold, oracle_full)

    def _summary_dict(g, fold_df, full_m):
        s = g["summary"]
        return {
            "worst_fold_ann_roi_pct": s["worst_fold_ann_roi_pct"],
            "mean_fold_ann_roi_pct": s["mean_fold_ann_roi_pct"],
            "worst_fold_max_dd_pct": s["worst_fold_max_dd_pct"],
            "full_data_ann_roi_pct": s["full_data_ann_roi_pct"],
            "full_data_max_dd_pct": s["full_data_max_dd_pct"],
            "total_trades_admitted": int(fold_df["n_trades"].sum()),
            "pass_deployable": int(g["pass_deployable"]),
            "pass_viable": int(g["pass_viable"]),
        }

    comp_rows: List[Dict[str, Any]] = []
    for name, g, fdf, fm in [
        ("raw_baseline", raw_gates, raw_fold, raw_full),
        ("oracle_cluster0", oracle_gates, oracle_fold, oracle_full),
    ]:
        comp_rows.append({"reference_point": name, **_summary_dict(g, fdf, fm)})
    for cand_name, info in candidate_results.items():
        comp_rows.append({"reference_point": f"lgbm_e_{cand_name}", **_summary_dict(info["gates"], info["fold_df"], info["full_m"])})
    pd.DataFrame(comp_rows).to_csv(out_dir / "comparison.csv", index=False,
                                    float_format="%.10g", lineterminator="\n")

    summary: Dict[str, Any] = {
        "parity": parity,
        "candidates": {},
    }
    for cand_name, info in candidate_results.items():
        summary["candidates"][cand_name] = {
            "threshold": info["threshold"], "note": info["note"],
            "gates_summary": info["gates"]["summary"],
            "pass_deployable": bool(info["gates"]["pass_deployable"]),
            "pass_viable": bool(info["gates"]["pass_viable"]),
            "total_trades_admitted": int(info["fold_df"]["n_trades"].sum()),
            "fold_n_admitted": info["fold_df"]["n_admitted"].tolist(),
        }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Arc 9 Step 5 WFO with LightGBM Pipeline E filter.")
    parser.add_argument("--out-dir", type=Path,
                        default=_REPO_ROOT / "results" / "l_arc_9" / "experiments" / "step5_lgbm_pipeline_e")
    parser.add_argument("--kh24-cfg", type=Path, default=_REPO_ROOT / "configs" / "wfo_kh24.yaml")
    parser.add_argument("--arc-cfg", type=Path, default=_REPO_ROOT / "configs" / "wfo_l_arc_9.yaml")
    parser.add_argument("--verify-determinism", action="store_true")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = run(args.out_dir, args.kh24_cfg, args.arc_cfg)
    print(f"[lgbm-e] DONE.")
    for cn, info in summary["candidates"].items():
        head = "PASS-DEPLOYABLE" if info["pass_deployable"] else "PASS-VIABLE" if info["pass_viable"] else "FAIL"
        s = info["gates_summary"]
        print(f"  candidate {cn}: {head} | worst-fold ann ROI {s['worst_fold_ann_roi_pct']:+.2f}% | "
              f"mean ann ROI {s['mean_fold_ann_roi_pct']:+.2f}% | worst DD {s['worst_fold_max_dd_pct']:.2f}%")

    if args.verify_determinism:
        scratch = args.out_dir / "_determinism_scratch"
        scratch.mkdir(exist_ok=True)
        run(scratch, args.kh24_cfg, args.arc_cfg)
        sha1s: Dict[str, str] = {}
        sha2s: Dict[str, str] = {}
        for cn, _, _ in CANDIDATES:
            p1 = args.out_dir / f"candidate_{cn}" / "per_fold_metrics.csv"
            p2 = scratch / f"candidate_{cn}" / "per_fold_metrics.csv"
            sha1s[f"candidate_{cn}_per_fold"] = _sha256_file(p1)
            sha2s[f"candidate_{cn}_per_fold"] = _sha256_file(p2)
        det = {
            "run1": sha1s, "run2": sha2s,
            "byte_identical": bool(all(sha1s[k] == sha2s[k] for k in sha1s)),
        }
        (args.out_dir / "determinism_check.json").write_text(
            json.dumps(det, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # cleanup scratch
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)
        print(f"[lgbm-e] determinism: {'PASS' if det['byte_identical'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
