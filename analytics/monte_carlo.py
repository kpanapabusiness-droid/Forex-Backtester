# /Users/keanupanapa/Notebooks/Forex_Backtester/analytics/monte_carlo.py
# v1.9.8 — Monte Carlo & Trade Shuffling (daily bootstrap or trade permutation)

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------- Metrics helpers -------------------------
def _max_drawdown_pct(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak - 1.0) * 100.0
    return float(dd.min())


def _sharpe_sortino(daily_returns: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    """
    Risk-free assumed 0% for backtest context. Annualization: 252 days.
    """
    if daily_returns.size == 0:
        return (float("nan"), float("nan"))
    mu = float(np.nanmean(daily_returns))
    sigma = float(np.nanstd(daily_returns, ddof=1))
    downside = daily_returns[daily_returns < 0]
    downside_std = float(np.nanstd(downside, ddof=1)) if downside.size > 0 else 0.0

    sharpe_daily = mu / (sigma + eps) if sigma > 0 else float("inf") if mu > 0 else 0.0
    sortino_daily = (
        mu / (downside_std + eps) if downside_std > 0 else float("inf") if mu > 0 else 0.0
    )

    sharpe_ann = sharpe_daily * math.sqrt(252.0)
    sortino_ann = sortino_daily * math.sqrt(252.0)
    return (float(sharpe_ann), float(sortino_ann))


def _mar(cagr: float, max_dd_pct: float) -> float:
    if max_dd_pct is None or max_dd_pct == 0:
        return float("inf")
    return float(cagr / (abs(max_dd_pct) / 100.0))


def _cagr_from_equity(equity: pd.Series) -> float:
    """
    CAGR computed from first→last equity over N trading days, annualized (252).
    """
    if equity.empty or equity.iloc[0] <= 0:
        return float("nan")
    total_return = float(equity.iloc[-1] / equity.iloc[0])
    n_days = len(equity)
    if n_days <= 1:
        return 0.0
    return float(total_return ** (252.0 / n_days) - 1.0)


def _equity_from_returns(returns: np.ndarray, start_equity: float = 1.0) -> np.ndarray:
    path = np.cumprod(1.0 + returns, dtype=float)
    return start_equity * path


# ------------------------- Config helpers -------------------------
def _results_dir_from_cfg(cfg: dict) -> Path:
    """
    Resolve which results/<run_name>/ to read from:
      - If horizon == 'oos' and WFO enabled → use walk_forward.run_name
      - Else use top-level run_name (fallback to WFO name if present; else 'results')
    """
    mc = cfg.get("monte_carlo", {}) or {}
    horizon = (mc.get("horizon") or "oos").lower()

    wf = cfg.get("walk_forward", {}) or {}
    wf_enabled = bool(wf.get("enabled"))
    wf_run = wf.get("run_name") or "wfo_default"

    top_run = cfg.get("run_name")

    if horizon == "oos" and wf_enabled:
        run_name = wf_run
    else:
        run_name = top_run or (wf_run if wf_enabled else "results")

    return Path("results") / (run_name or "results")


def _starting_balance_from_cfg(cfg: dict) -> float:
    return float(
        (cfg.get("risk") or {}).get("starting_balance", cfg.get("starting_balance", 10_000.0))
    )


# ------------------------- Public API -------------------------
def run_monte_carlo(
    cfg: dict,
    equity_df: Optional[pd.DataFrame] = None,
    trades_df: Optional[pd.DataFrame] = None,
    results_dir_override: Optional[Path | str] = None,  # <-- NEW: force output folder
) -> Dict[str, Any]:
    """
    Monte Carlo on equity (daily) or per-trade series.

    Config block:
      monte_carlo:
        enabled: true
        iterations: 2000
        horizon: "oos"           # "all" or "oos" (which results/<run_name> to read)
        use_daily_returns: true  # daily bootstrap; false => per-trade reshuffle
        save_samples: false
        seed: 42

    Outputs (under results/<resolved_run_name>/ or results_dir_override):
      - mc_summary.txt                          (canonical)
      - mc_summary_daily.txt / _trades.txt      (mode-specific convenience)
      - mc_samples.csv                          (canonical; optional)
      - mc_samples_daily.csv / _trades.csv      (mode-specific; optional)

    Returns a dict with quantiles for ROI%, MaxDD%, MAR and (if daily) Sharpe/Sortino medians.
    """

    # --- local helper to avoid NameError if not defined elsewhere ---
    def _starting_balance_from_cfg_local(c: dict) -> Optional[float]:
        risk = (c or {}).get("risk") or {}
        if "starting_balance" in risk:
            try:
                return float(risk["starting_balance"])
            except Exception:
                pass
        if "starting_balance" in (c or {}):
            try:
                return float(c["starting_balance"])
            except Exception:
                pass
        return None

    # Robust results dir resolution (prefer explicit override; else prefer WFO run_name if present)
    def _results_dir_from_cfg_robust(c: dict) -> Path:
        wf = (c or {}).get("walk_forward") or {}
        # handle dict or pydantic-ish model
        wf_run = wf.get("run_name") if isinstance(wf, dict) else getattr(wf, "run_name", None)
        top_run = (c or {}).get("run_name", None)
        run_name = wf_run or top_run or "results"
        return Path("results") / str(run_name)

    mc = cfg.get("monte_carlo", {}) or {}
    if not mc.get("enabled", False):
        print("ℹ️  monte_carlo.enabled is False — skipping MC.")
        return {}

    iters = int(mc.get("iterations", 2000))
    use_daily = bool(mc.get("use_daily_returns", True))
    save_samples = bool(mc.get("save_samples", False))

    seed = mc.get("seed")
    rng = np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()

    # Resolve output folder
    results_dir = (
        Path(results_dir_override) if results_dir_override else _results_dir_from_cfg_robust(cfg)
    )
    # Backward-compatible: if a pointer exists, use the actual directory it references
    try:
        ptr = results_dir / ".source_dir"
        if ptr.exists():
            target = Path(ptr.read_text(encoding="utf-8").strip())
            if target.exists():
                results_dir = target
    except Exception:
        pass
    results_dir.mkdir(parents=True, exist_ok=True)

    # Fallback: discover a pointer under results/*/.source_dir if selected dir has no artifacts
    try:
        eq_exists = (results_dir / "equity_curve.csv").exists()
        tr_exists = (results_dir / "trades.csv").exists()
        if not (eq_exists or tr_exists):
            root = Path("results")
            for ptr in root.glob("*/.source_dir"):
                try:
                    target = Path(ptr.read_text(encoding="utf-8").strip())
                except Exception:
                    continue
                if target.exists() and (
                    (target / "equity_curve.csv").exists() or (target / "trades.csv").exists()
                ):
                    results_dir = target
                    break
    except Exception:
        pass

    # Load sources if not supplied
    if equity_df is None:
        eq_path = results_dir / "equity_curve.csv"
        if eq_path.exists():
            try:
                equity_df = pd.read_csv(eq_path)
            except Exception as e:
                print(f"ℹ️  Failed to read equity_curve.csv: {e}")
    if trades_df is None:
        t_path = results_dir / "trades.csv"
        if t_path.exists():
            try:
                trades_df = pd.read_csv(t_path)
            except Exception as e:
                print(f"ℹ️  Failed to read trades.csv: {e}")

    # ----------------- Build base series -----------------
    series: Optional[np.ndarray] = None
    series_type: Optional[str] = None

    def _clean_series(x: np.ndarray) -> np.ndarray:
        x = x.astype(float)
        x = x[np.isfinite(x)]
        return x

    # Prefer daily equity if requested and available
    if use_daily and equity_df is not None and "equity" in equity_df.columns:
        eq_vals = pd.to_numeric(equity_df["equity"], errors="coerce").dropna().values
        if len(eq_vals) >= 3 and (eq_vals > 0).all():
            daily = (eq_vals[1:] / eq_vals[:-1]) - 1.0
            daily = _clean_series(daily)
            if daily.size >= 5:
                series = daily
                series_type = "daily"
            else:
                print("ℹ️  Daily series too short after cleaning; falling back to per-trade mode.")
        else:
            print(
                "ℹ️  Equity curve insufficient or contains non-positive values; falling back to per-trade mode."
            )

    # Fall back to per-trade mode if needed
    if series is None and trades_df is not None and "pnl" in trades_df.columns:
        if "return_pct" in trades_df.columns:
            returns = pd.to_numeric(trades_df["return_pct"], errors="coerce").values / 100.0
        else:
            pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").fillna(0.0).values
            denom = _starting_balance_from_cfg_local(cfg)
            if not (denom and denom > 0):
                # Conservative fallback scaling if starting_balance not set
                denom = max(1.0, float(np.nanmean(np.abs(pnl)) or 0.0) * 10.0)
            returns = pnl / denom
        returns = _clean_series(returns)
        if returns.size >= 5:
            series = returns
            series_type = "per_trade"

    if series is None:
        raise RuntimeError(
            "Unable to build Monte Carlo series: need a usable equity_curve.csv (for daily) "
            "or trades.csv with pnl/return_pct (for per-trade)."
        )

    n = int(series.size)
    print(f"ℹ️  MC series mode: {series_type}  | length: {n}  | results_dir: {results_dir}")

    # ----------------- Simulate -----------------
    roi_list: List[float] = []
    maxdd_list: List[float] = []
    mar_list: List[float] = []
    sharpe_list: List[float] = []
    sortino_list: List[float] = []

    start_equity = 1.0  # normalized; ROI% is relative to this base

    for _ in range(iters):
        if series_type == "daily":
            # i.i.d. bootstrap with replacement
            sample = rng.choice(series, size=n, replace=True)
            eq_path = _equity_from_returns(sample, start_equity=start_equity)
            roi = (eq_path[-1] - start_equity) * 100.0
            maxdd = _max_drawdown_pct(pd.Series(eq_path))
            cagr = _cagr_from_equity(pd.Series(eq_path))
            mar = _mar(cagr, maxdd)
            sh, so = _sharpe_sortino(sample)

            roi_list.append(float(roi))
            maxdd_list.append(float(maxdd))
            mar_list.append(float(mar))
            sharpe_list.append(float(sh))
            sortino_list.append(float(so))

        else:
            # per-trade permutation (reshuffle sequence without replacement)
            sample = rng.permutation(series)
            eq_path = _equity_from_returns(sample, start_equity=start_equity)
            roi = (eq_path[-1] - start_equity) * 100.0
            maxdd = _max_drawdown_pct(pd.Series(eq_path))
            cagr = _cagr_from_equity(pd.Series(eq_path))
            mar = _mar(cagr, maxdd)

            roi_list.append(float(roi))
            maxdd_list.append(float(maxdd))
            mar_list.append(float(mar))

    # ----------------- Aggregate & write -----------------
    def q(a: List[float], p: float) -> float:
        return float(np.nanpercentile(a, p)) if len(a) else float("nan")

    summary: Dict[str, Any] = {
        "mode": series_type,
        "iterations": iters,
        "roi_pct_median": q(roi_list, 50),
        "roi_pct_p5": q(roi_list, 5),
        "roi_pct_p95": q(roi_list, 95),
        "maxdd_pct_median": q(maxdd_list, 50),
        "maxdd_pct_p5": q(maxdd_list, 5),
        "maxdd_pct_p95": q(maxdd_list, 95),
        "mar_median": q(mar_list, 50),
    }
    if series_type == "daily":
        summary.update(
            {
                "sharpe_median": q(sharpe_list, 50),
                "sortino_median": q(sortino_list, 50),
            }
        )

    # Mode suffix for duplicates
    mode_suffix = "daily" if series_type == "daily" else "trades"

    # Save optional samples (canonical + mode-specific)
    if save_samples:
        samples_df = pd.DataFrame(
            {
                "roi_pct": roi_list,
                "maxdd_pct": maxdd_list,
                "mar": mar_list,
                **(
                    {"sharpe": sharpe_list, "sortino": sortino_list}
                    if series_type == "daily"
                    else {}
                ),
            }
        )
        samples_path = results_dir / "mc_samples.csv"
        samples_df.to_csv(samples_path, index=False)
        print(f"✅ Wrote MC samples: {samples_path}")
        # mode-specific copy
        try:
            samples_mode = results_dir / f"mc_samples_{mode_suffix}.csv"
            samples_df.to_csv(samples_mode, index=False)
            print(f"✅ Wrote MC samples (mode): {samples_mode}")
        except Exception:
            pass

    # Summary txt (canonical)
    def fmt(x: Any) -> str:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "NA"
        return f"{x:.4f}"

    sum_path = results_dir / "mc_summary.txt"
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("🎲 Monte Carlo Summary (v1.9.8)\n")
        f.write("--------------------------------\n")
        f.write(f"Mode         : {summary['mode']}\n")
        f.write(f"Iterations   : {summary['iterations']}\n")
        f.write(f"ROI%  (P50)  : {fmt(summary['roi_pct_median'])}\n")
        f.write(f"ROI%   P5/P95: {fmt(summary['roi_pct_p5'])} / {fmt(summary['roi_pct_p95'])}\n")
        f.write(f"MaxDD% (P50) : {fmt(summary['maxdd_pct_median'])}\n")
        f.write(
            f"MaxDD% P5/P95: {fmt(summary['maxdd_pct_p5'])} / {fmt(summary['maxdd_pct_p95'])}\n"
        )
        f.write(f"MAR   (P50)  : {fmt(summary['mar_median'])}\n")
        if "sharpe_median" in summary:
            f.write(f"Sharpe(P50)  : {fmt(summary['sharpe_median'])}\n")
            f.write(f"Sortino(P50) : {fmt(summary['sortino_median'])}\n")
    print(f"✅ Wrote MC summary: {sum_path}")

    # Mode-specific copy so daily/trades don’t overwrite each other
    try:
        sum_mode = results_dir / f"mc_summary_{mode_suffix}.txt"
        sum_mode.write_text(sum_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"✅ Wrote MC summary (mode): {sum_mode}")
    except Exception:
        pass

    return summary


def run_monte_carlo_modes(
    cfg: dict | None = None,
    results_dir: str | os.PathLike | None = None,
    equity_csv: str | os.PathLike | None = None,
    trades_csv: str | os.PathLike | None = None,
    iterations: int | None = None,
    horizon: str | int | None = None,
    rng_seed: int | None = None,
    progress: bool = True,
) -> dict:
    """
    Convenience helper to run Monte Carlo in multiple modes (per-trade and/or daily)
    based on cfg["monte_carlo"]["modes"]. Falls back to a single mode if not provided.
    Returns a dict of mode->summary dicts.
    """
    mc_cfg = (cfg or {}).get("monte_carlo", {}) if cfg else {}
    # Resolve inputs
    _results_dir = results_dir or _results_dir_from_cfg(cfg or {})
    _equity_csv = equity_csv or os.path.join(_results_dir, "equity_curve.csv")
    _trades_csv = trades_csv or os.path.join(_results_dir, "trades.csv")

    modes = mc_cfg.get("modes")
    if not modes:
        # Infer from use_daily_returns or default to per-trade
        modes = ["daily"] if mc_cfg.get("use_daily_returns") else ["trades"]

    out = {}
    for mode in modes:
        use_daily = mode.lower() in ["daily", "per_day", "equity", "equity_daily"]
        res = run_monte_carlo(
            cfg=cfg,
            results_dir=_results_dir,
            equity_csv=_equity_csv,
            trades_csv=_trades_csv,
            iterations=iterations or mc_cfg.get("iterations", 300),
            horizon=horizon or mc_cfg.get("horizon", "oos"),
            use_daily_returns=use_daily,
            save_mode_specific=mc_cfg.get("save_mode_specific", True),
            save_main_as_latest=mc_cfg.get("save_main_as_latest", True),
            rng_seed=rng_seed or mc_cfg.get("rng_seed"),
            progress=progress,
        )
        out["daily" if use_daily else "trades"] = res
    return out
