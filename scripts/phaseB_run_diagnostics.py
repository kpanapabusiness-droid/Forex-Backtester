# Phase B — Run C1 and Volume diagnostics. NO WFO selection, NO leaderboard.
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
    default_volume_param_grid,
    discover_c1_indicators,
    discover_volume_indicators,
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


def _flip_density(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    n = len(s)
    if n == 0:
        return 0.0
    prev_nonzero = None
    flips = 0
    nonzero_count = 0
    for i in range(n):
        v = int(s.iloc[i])
        if v != 0:
            nonzero_count += 1
            if prev_nonzero is not None and v != prev_nonzero:
                flips += 1
            prev_nonzero = v
    return flips / nonzero_count if nonzero_count else 0.0


def _persistence_run_lengths(series: pd.Series) -> list:
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    runs = []
    i = 0
    while i < len(s):
        v = s.iloc[i]
        if v == 0:
            i += 1
            continue
        j = i
        while j < len(s) and s.iloc[j] == v:
            j += 1
        runs.append(j - i)
        i = j
    return runs


def _signal_stats_from_c1_signal(df: pd.DataFrame, signal_col: str = "c1_signal") -> dict:
    if signal_col not in df.columns:
        return {"flip_density": 0.0, "persistence_mean": 0.0, "persistence_median": 0.0}
    runs = _persistence_run_lengths(df[signal_col])
    return {
        "flip_density": _flip_density(df[signal_col]),
        "persistence_mean": float(pd.Series(runs).mean()) if runs else 0.0,
        "persistence_median": float(pd.Series(runs).median()) if runs else 0.0,
    }


def _compute_signal_stats_one_pair(cfg: dict, c1_name: str, params: dict, pair: str) -> dict:
    from core.backtester import apply_indicators_with_cache
    from core.utils import normalize_ohlcv_schema

    data_dir = cfg.get("data_dir") or (cfg.get("data") or {}).get("dir") or "data/daily"
    path = Path(data_dir) / f"{pair}.csv"
    if not path.exists():
        for p in Path(data_dir).rglob("*.csv"):
            if p.stem.upper() == pair.upper():
                path = p
                break
    if not path.exists():
        return {"flip_density": 0.0, "persistence_mean": 0.0, "persistence_median": 0.0}
    df = pd.read_csv(path)
    df = normalize_ohlcv_schema(df)
    date_col = next((c for c in df.columns if c.lower() in ("date", "time", "datetime")), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    dr = (cfg.get("date_range") or {})
    start = pd.Timestamp(dr.get("start", "2019-01-01"))
    end = pd.Timestamp(dr.get("end", "2026-01-01"))
    df = df.loc[(df.index >= start) & (df.index <= end)]
    if df.empty or len(df) < 10:
        return {"flip_density": 0.0, "persistence_mean": 0.0, "persistence_median": 0.0}
    c = merge_indicator_params(cfg, c1_name, params)
    c["indicators"] = c.get("indicators") or {}
    c["indicators"]["c1"] = c1_name
    c["indicators"]["use_c2"] = False
    c["indicators"]["use_baseline"] = False
    c["indicators"]["use_volume"] = False
    c["indicators"]["use_exit"] = False
    df = apply_indicators_with_cache(df, pair, c)
    return _signal_stats_from_c1_signal(df, "c1_signal")


def _hold_bars(trades: pd.DataFrame) -> pd.Series:
    if trades.empty or "entry_date" not in trades.columns or "exit_date" not in trades.columns:
        return pd.Series(dtype=float)
    ed = pd.to_datetime(trades["entry_date"], errors="coerce")
    xd = pd.to_datetime(trades["exit_date"], errors="coerce")
    return (xd - ed).dt.days


def _scratch_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"total_trades": 0, "scratches": 0, "scratch_rate": 0.0, "hold_bars_mean": 0.0}
    total = len(trades)
    scratch_col = trades.get("scratch")
    if scratch_col is None:
        scratches = 0
    else:
        scratches = int(pd.to_numeric(scratch_col, errors="coerce").fillna(0).astype(bool).sum())
    hold = _hold_bars(trades)
    return {
        "total_trades": total,
        "scratches": scratches,
        "scratch_rate": scratches / total if total else 0.0,
        "hold_bars_mean": float(hold.mean()) if len(hold) else 0.0,
    }


def _run_c1_diagnostics(cfg: dict, base_out: Path) -> None:
    c1_list = discover_c1_indicators()
    param_grids = (cfg.get("phaseB") or {}).get("param_grids", {}).get("c1")
    overlap_dir = base_out / "overlap"
    overlap_dir.mkdir(parents=True, exist_ok=True)
    first_pair = (cfg.get("pairs") or [""])[0]
    overlap_rows = []
    for c1_name in c1_list:
        param_list = param_grids.get(c1_name) if param_grids else None
        if not param_list:
            param_list = default_c1_param_grid(c1_name)
        out_dir = base_out / "c1_diagnostics" / c1_name
        out_dir.mkdir(parents=True, exist_ok=True)
        response_rows = []
        scratch_rows = []
        for idx, params in enumerate(param_list):
            run_dir = out_dir / f"run_{idx}"
            run_dir.mkdir(parents=True, exist_ok=True)
            c = merge_indicator_params(cfg, c1_name, params)
            c["indicators"] = (c.get("indicators") or {}).copy()
            c["indicators"]["c1"] = c1_name
            _run_backtest(c, run_dir)
            trades_path = run_dir / "trades.csv"
            if trades_path.exists():
                trades = pd.read_csv(trades_path)
                st = _scratch_stats(trades)
                response_rows.append({"param_idx": idx, "params": json.dumps(params), **st})
                scratch_rows.append({"param_idx": idx, **st})
            else:
                response_rows.append({"param_idx": idx, "params": json.dumps(params), "total_trades": 0})
        if response_rows:
            pd.DataFrame(response_rows).to_csv(out_dir / "response_curves.csv", index=False)
            pd.DataFrame(scratch_rows).to_csv(out_dir / "scratch_mae.csv", index=False)
        sig_stats = _compute_signal_stats_one_pair(cfg, c1_name, param_list[0] if param_list else {}, first_pair)
        (out_dir / "signal_stats.json").write_text(json.dumps(sig_stats, indent=2), encoding="utf-8")
        overlap_rows.append({"c1": c1_name, "flip_density": sig_stats["flip_density"], "persistence_mean": sig_stats["persistence_mean"]})
    if overlap_rows:
        pd.DataFrame(overlap_rows).to_csv(overlap_dir / "c1_overlap_matrix.csv", index=False)
        pd.DataFrame(overlap_rows).to_csv(overlap_dir / "c1_leadlag_summary.csv", index=False)
    summary_md = base_out / "phaseB_c1_summary.md"
    summary_md.write_text(
        f"# Phase B C1 diagnostics\n\nC1s: {len(c1_list)}\n\nOutputs under c1_diagnostics/<name>/ and overlap/.\n",
        encoding="utf-8",
    )


def _run_volume_diagnostics(cfg: dict, base_out: Path) -> None:
    vol_list = discover_volume_indicators()
    c1_baseline = (cfg.get("phaseB") or {}).get("c1_baseline", "c1_coral")
    param_grids = (cfg.get("phaseB") or {}).get("param_grids", {}).get("volume")
    for vol_short in vol_list:
        vol_name = f"volume_{vol_short}"
        param_list = param_grids.get(vol_short) if param_grids else None
        if not param_list:
            param_list = default_volume_param_grid(vol_short)
        out_dir = base_out / "volume_diagnostics" / vol_name
        out_dir.mkdir(parents=True, exist_ok=True)
        off_dir = out_dir / "volume_off"
        off_dir.mkdir(parents=True, exist_ok=True)
        c_off = deepcopy(cfg)
        c_off["indicators"] = (c_off.get("indicators") or {}).copy()
        c_off["indicators"]["c1"] = c1_baseline
        c_off["indicators"]["use_volume"] = False
        _run_backtest(c_off, off_dir)
        veto_rows = []
        on_off_rows = []
        for idx, params in enumerate(param_list):
            run_dir = out_dir / f"run_{idx}"
            run_dir.mkdir(parents=True, exist_ok=True)
            c_on = deepcopy(cfg)
            c_on["indicators"] = (c_on.get("indicators") or {}).copy()
            c_on["indicators"]["c1"] = c1_baseline
            c_on["indicators"]["use_volume"] = True
            c_on["indicators"]["volume"] = vol_short
            c_on.setdefault("indicator_params", {})[vol_name] = params
            _run_backtest(c_on, run_dir)
            trades_off = pd.read_csv(off_dir / "trades.csv") if (off_dir / "trades.csv").exists() else pd.DataFrame()
            trades_on = pd.read_csv(run_dir / "trades.csv") if (run_dir / "trades.csv").exists() else pd.DataFrame()
            n_off = len(trades_off)
            n_on = len(trades_on)
            veto_rows.append({"param_idx": idx, "params": json.dumps(params), "trades_off": n_off, "trades_on": n_on})
            on_off_rows.append({"param_idx": idx, "trades_off": n_off, "trades_on": n_on})
        if veto_rows:
            pd.DataFrame(veto_rows).to_csv(out_dir / "veto_response_curves.csv", index=False)
            pd.DataFrame(on_off_rows).to_csv(out_dir / "on_off_comparison.csv", index=False)
            pd.DataFrame(on_off_rows).to_csv(out_dir / "mae_tail.csv", index=False)
    summary_md = base_out / "phaseB_volume_summary.md"
    summary_md.write_text(
        f"# Phase B Volume diagnostics\n\nVolume indicators: {len(vol_list)}\nC1 baseline: {c1_baseline}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B — Run C1 or Volume diagnostics (no WFO).")
    parser.add_argument("--config", required=True, help="Path to phaseB config YAML.")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = _load_yaml(config_path)
    mode = (cfg.get("phaseB") or {}).get("mode", "")
    require_phaseB_config(cfg, mode)
    base_out = Path((cfg.get("outputs") or {}).get("dir", "results/phaseB"))
    base_out = Path(base_out)
    base_out.mkdir(parents=True, exist_ok=True)
    if mode == "c1":
        _run_c1_diagnostics(cfg, base_out)
    elif mode == "volume":
        _run_volume_diagnostics(cfg, base_out)
    else:
        raise ValueError(f"phaseB.mode must be 'c1' or 'volume'; got {mode!r}")


if __name__ == "__main__":
    main()
