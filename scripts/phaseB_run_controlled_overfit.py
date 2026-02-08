# Phase B — Controlled overfit diagnostic. Fit window param search, check window eval.
# NO WFO selection, NO ranking. Stability proxy: drawdown + trade sanity + scratch penalty.
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phaseB_common import (  # noqa: E402
    default_c1_param_grid,
    discover_c1_indicators,
    merge_indicator_params,
    require_phaseB_config,
)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _run_backtest(cfg: dict, results_dir: Path) -> None:
    from core.backtester import run_backtest

    c = deepcopy(cfg)
    c["outputs"] = {"dir": str(results_dir)}
    c.setdefault("output", {})["results_dir"] = str(results_dir)
    run_backtest(c, results_dir=str(results_dir))


def _parse_summary(results_dir: Path) -> dict:
    path = results_dir / "summary.txt"
    if not path.exists():
        return {"total_trades": 0, "roi_pct": 0.0, "max_dd_pct": 0.0, "scratches": 0}
    text = path.read_text(encoding="utf-8")
    out = {"total_trades": 0, "roi_pct": 0.0, "max_dd_pct": 0.0, "scratches": 0}
    for line in text.splitlines():
        if "Total Trades" in line or "Total trades" in line:
            try:
                out["total_trades"] = int(line.split(":")[-1].strip())
            except Exception:
                pass
        if "ROI (%)" in line:
            try:
                out["roi_pct"] = float(line.split(":")[-1].strip().replace("%", ""))
            except Exception:
                pass
        if "Max DD" in line or "max_dd" in line.lower():
            try:
                s = line.split(":")[-1].strip().replace("%", "")
                out["max_dd_pct"] = -abs(float(s))
            except Exception:
                pass
        if "Scratch" in line and ":" in line:
            try:
                out["scratches"] = int(line.split(":")[-1].strip())
            except Exception:
                pass
    return out


def _scratch_cluster_penalty(trades_path: Path, window: int = 30, min_scratch_in_window: int = 3) -> int:
    if not trades_path.exists():
        return 0
    df = pd.read_csv(trades_path)
    if "scratch" not in df.columns or len(df) < min_scratch_in_window:
        return 0
    scratch = pd.to_numeric(df["scratch"], errors="coerce").fillna(0).astype(bool)
    count = 0
    for i in range(len(df) - window + 1):
        if scratch.iloc[i : i + window].sum() >= min_scratch_in_window:
            count += 1
    return count


def _stability_proxy(
    results_dir: Path,
    w_dd: float = 0.01,
    w_scratch: float = 1.0,
    min_trades: int = 5,
    max_trades: int = 10000,
) -> float:
    s = _parse_summary(results_dir)
    trades_path = results_dir / "trades.csv"
    penalty = _scratch_cluster_penalty(trades_path)
    trade_sanity = 1.0 if min_trades <= s["total_trades"] <= max_trades else 0.0
    return -w_dd * abs(s["max_dd_pct"]) - w_scratch * penalty + 0.1 * trade_sanity


def _run_controlled_overfit(cfg: dict, base_out: Path) -> None:
    pb = cfg.get("phaseB") or {}
    fold_pairs = pb.get("diagnostic_fold_pairs") or []
    if len(fold_pairs) < 3:
        raise ValueError("phaseB.diagnostic_fold_pairs must have at least 3 pairs")
    c1_list = discover_c1_indicators()
    param_grids = pb.get("param_grids", {}).get("c1")
    overfit_base = base_out / "controlled_overfit"
    overfit_base.mkdir(parents=True, exist_ok=True)
    for c1_name in c1_list:
        param_list = param_grids.get(c1_name) if param_grids else None
        if not param_list:
            param_list = default_c1_param_grid(c1_name)
        ind_out = overfit_base / c1_name
        ind_out.mkdir(parents=True, exist_ok=True)
        pair_rows = []
        for fp_idx, fp in enumerate(fold_pairs):
            fit_w = fp.get("fit_window") or {}
            check_w = fp.get("check_window") or {}
            fit_start = fit_w.get("start") or "2019-01-01"
            fit_end = fit_w.get("end") or "2021-12-31"
            check_start = check_w.get("start") or "2022-01-01"
            check_end = check_w.get("end") or "2023-06-30"
            best_params = {}
            best_score = -1e9
            best_run_dir = None
            fit_cfg = deepcopy(cfg)
            fit_cfg["date_range"] = {"start": fit_start, "end": fit_end}
            for idx, params in enumerate(param_list):
                run_dir = ind_out / f"fold_{fp_idx}_fit_run_{idx}"
                run_dir.mkdir(parents=True, exist_ok=True)
                c = merge_indicator_params(fit_cfg, c1_name, params)
                c["indicators"] = (c.get("indicators") or {}).copy()
                c["indicators"]["c1"] = c1_name
                _run_backtest(c, run_dir)
                score = _stability_proxy(run_dir)
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_run_dir = run_dir
            check_cfg = deepcopy(cfg)
            check_cfg["date_range"] = {"start": check_start, "end": check_end}
            check_dir = ind_out / f"fold_{fp_idx}_check"
            check_dir.mkdir(parents=True, exist_ok=True)
            c_check = merge_indicator_params(check_cfg, c1_name, best_params)
            c_check["indicators"] = (c_check.get("indicators") or {}).copy()
            c_check["indicators"]["c1"] = c1_name
            _run_backtest(c_check, check_dir)
            fit_s = _parse_summary(best_run_dir) if best_run_dir and best_run_dir.exists() else {}
            check_s = _parse_summary(check_dir)
            pair_rows.append({
                "fold_idx": fp_idx,
                "fit_start": fit_start,
                "fit_end": fit_end,
                "check_start": check_start,
                "check_end": check_end,
                "best_params": json.dumps(best_params),
                "fit_trades": fit_s.get("total_trades", 0),
                "check_trades": check_s.get("total_trades", 0),
                "check_roi_pct": check_s.get("roi_pct", 0),
                "check_max_dd_pct": check_s.get("max_dd_pct", 0),
            })
        if pair_rows:
            pd.DataFrame(pair_rows).to_csv(ind_out / "overfit_pairs.csv", index=False)
            summary = pd.DataFrame(pair_rows)
            summary.to_csv(ind_out / "overfit_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B — Controlled overfit diagnostic (no WFO).")
    parser.add_argument("--config", required=True, help="Path to phaseB_controlled_overfit config.")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = _load_yaml(config_path)
    require_phaseB_config(cfg, "controlled_overfit")
    base_out = Path((cfg.get("outputs") or {}).get("dir", "results/phaseB/controlled_overfit"))
    base_out.mkdir(parents=True, exist_ok=True)
    _run_controlled_overfit(cfg, base_out)


if __name__ == "__main__":
    main()
